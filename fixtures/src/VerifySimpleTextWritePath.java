import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.simpletext.SimpleTextCodec;
import org.apache.lucene.document.BinaryDocValuesField;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.FieldType;
import org.apache.lucene.document.IntPoint;
import org.apache.lucene.document.KnnByteVectorField;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.NumericDocValuesField;
import org.apache.lucene.document.SortedDocValuesField;
import org.apache.lucene.document.SortedNumericDocValuesField;
import org.apache.lucene.document.SortedSetDocValuesField;
import org.apache.lucene.document.StoredField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.BinaryDocValues;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.CheckIndex;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FilterMergePolicy;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.MergePolicy;
import org.apache.lucene.index.MergeTrigger;
import org.apache.lucene.index.NumericDocValues;
import org.apache.lucene.index.PointValues;
import org.apache.lucene.index.PostingsEnum;
import org.apache.lucene.index.SegmentCommitInfo;
import org.apache.lucene.index.SegmentInfos;
import org.apache.lucene.index.SortedDocValues;
import org.apache.lucene.index.SortedNumericDocValues;
import org.apache.lucene.index.SortedSetDocValues;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.Terms;
import org.apache.lucene.index.TermsEnum;
import org.apache.lucene.index.TieredMergePolicy;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.BytesRef;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

/**
 * Verifies {@code crates/lucene-search/examples/write_simpletext_index_fixture.rs}: this port's
 * IndexWriter publishing through {@code SimpleTextDirectory} wrote {@code <dir>/flushed} (two
 * segments, the second compound, with deletions and two doc-values update rounds) and {@code
 * <dir>/merged} (the same force-merged into one), and {@code <dir>/appended} and {@code
 * <dir>/appended-merged}: an index written in two sessions, the second a writer opened on the
 * first's SimpleText index (another update round on its segments, a third segment, deletions,
 * and a force merge for the second).
 *
 * <p>Lucene writes the same documents, deletions and updates with {@link SimpleTextCodec}
 * itself. Every segment of the port's index must name the SimpleText codec, carry Lucene's
 * deletion, field-infos and doc-values generations and file names, and hold Lucene's bytes in
 * every file but the {@code .si} (whose diagnostics name the writer's JVM) -- compound entries
 * unpacked, and each updated field's generation file compared with that field's. CheckIndex
 * must pass on the port's index, and Lucene must read every stored field, term vector, posting,
 * norm, doc value, point and vector of it exactly as of its own.
 */
public class VerifySimpleTextWritePath {
  static final int PER_SEGMENT = 60;

  /** {@code write_simpletext_index_fixture::doc}. */
  static Document doc(int i) {
    Document d = new Document();
    d.add(new StringField("id", "d" + i, Field.Store.YES));
    FieldType body = new FieldType(TextField.TYPE_STORED);
    body.setStoreTermVectors(true);
    body.setStoreTermVectorPositions(true);
    body.setStoreTermVectorOffsets(true);
    body.freeze();
    d.add(new Field("body", "w" + (i % 5) + " x" + (i % 7) + " common t" + i + " w" + (i % 3), body));
    d.add(new StoredField("n", i));
    if (i % 4 != 1) {
      d.add(new IntPoint("pt", i, -i * 3));
      d.add(new NumericDocValuesField("num", i * 7L - 50));
      d.add(new SortedDocValuesField("sorted", new BytesRef("s" + (i % 6))));
      byte[] bin = new byte[i % 5];
      Arrays.fill(bin, (byte) i);
      d.add(new BinaryDocValuesField("bin", new BytesRef(bin)));
    }
    d.add(new SortedNumericDocValuesField("snum", i % 9));
    d.add(new SortedNumericDocValuesField("snum", -i));
    d.add(new SortedSetDocValuesField("sset", new BytesRef("a" + (i % 4))));
    d.add(new SortedSetDocValuesField("sset", new BytesRef("b" + (i % 3))));
    if (i % 3 != 2) {
      d.add(
          new KnnFloatVectorField(
              "vf", new float[] {i * 0.5f, 1f, -i, 0.25f}, VectorSimilarityFunction.EUCLIDEAN));
      d.add(
          new KnnByteVectorField(
              "vb",
              new byte[] {(byte) i, (byte) (255 - i), 7},
              VectorSimilarityFunction.EUCLIDEAN));
    }
    return d;
  }

  /**
   * Flushed segments compound as the config says, merged ones never: what the port's writer does
   * with its default merge policy for these sizes.
   */
  static final class CompoundAsConfigured extends FilterMergePolicy {
    CompoundAsConfigured() {
      super(new TieredMergePolicy());
    }

    @Override
    public boolean useCompoundFile(
        SegmentInfos infos, SegmentCommitInfo segment, MergeContext context) {
      return "flush".equals(segment.info.getDiagnostics().get("source"));
    }

    @Override
    public MergeSpecification findMerges(
        MergeTrigger trigger, SegmentInfos infos, MergeContext context) {
      return null;
    }
  }

  static IndexWriterConfig config() {
    IndexWriterConfig config = new IndexWriterConfig();
    config.setCodec(new SimpleTextCodec());
    config.setMaxBufferedDocs(IndexWriterConfig.DISABLE_AUTO_FLUSH);
    config.setRAMBufferSizeMB(256);
    config.setMergePolicy(new CompoundAsConfigured());
    return config;
  }

  /** {@code write_simpletext_index_fixture::deleted}. */
  static Term[] deleted() {
    List<Term> deleted = new ArrayList<>();
    for (int i = 0; i < 2 * PER_SEGMENT; i++) {
      if (i % 11 == 4) {
        deleted.add(new Term("id", "d" + i));
      }
    }
    return deleted.toArray(new Term[0]);
  }

  /** {@code write_simpletext_index_fixture::write_first}. */
  static void writeFirst(Path path) throws Exception {
    try (Directory dir = FSDirectory.open(path);
        IndexWriter w = new IndexWriter(dir, config())) {
      for (int seg = 0; seg < 2; seg++) {
        w.getConfig().setUseCompoundFile(seg == 1);
        for (int i = seg * PER_SEGMENT; i < (seg + 1) * PER_SEGMENT; i++) {
          w.addDocument(doc(i));
        }
        w.commit();
      }
      w.deleteDocuments(deleted());
      w.commit();
      w.updateNumericDocValue(new Term("id", "d3"), "num", 1000);
      w.updateNumericDocValue(new Term("id", "d71"), "num", -1000);
      w.commit();
    }
  }

  /** {@code write_simpletext_index_fixture::write_second}: a writer opened on an existing index. */
  static void writeSecond(Path path, boolean merge) throws Exception {
    try (Directory dir = FSDirectory.open(path);
        IndexWriter w = new IndexWriter(dir, config())) {
      w.updateNumericDocValue(new Term("id", "d3"), "num", 2000);
      w.updateBinaryDocValue(new Term("id", "d75"), "bin", new BytesRef(new byte[] {9, 9}));
      w.commit();
      w.getConfig().setUseCompoundFile(false);
      for (int i = 2 * PER_SEGMENT; i < 2 * PER_SEGMENT + 30; i++) {
        w.addDocument(doc(i));
      }
      w.commit();
      w.deleteDocuments(new Term("id", "d5"), new Term("id", "d130"));
      w.updateNumericDocValue(new Term("id", "d125"), "num", 7);
      w.commit();
      if (merge) {
        w.forceMerge(1);
        w.commit();
      }
    }
  }

  /** {@code write_simpletext_index_fixture::write}. */
  static void write(Path path, boolean merge) throws Exception {
    try (Directory dir = FSDirectory.open(path)) {
      try (IndexWriter w = new IndexWriter(dir, config())) {
        for (int seg = 0; seg < 2; seg++) {
          w.getConfig().setUseCompoundFile(seg == 1);
          for (int i = seg * PER_SEGMENT; i < (seg + 1) * PER_SEGMENT; i++) {
            w.addDocument(doc(i));
          }
          w.commit();
        }
        w.deleteDocuments(deleted());
        w.commit();
        w.updateNumericDocValue(new Term("id", "d3"), "num", 1000);
        w.updateNumericDocValue(new Term("id", "d71"), "num", -1000);
        w.commit();
        w.updateNumericDocValue(new Term("id", "d3"), "num", 2000);
        w.updateBinaryDocValue(new Term("id", "d7"), "bin", new BytesRef(new byte[] {9, 9}));
        w.commit();
        if (merge) {
          w.forceMerge(1);
          w.commit();
        }
      }
    }
  }

  public static void main(String[] args) throws Exception {
    Path root = Path.of(args[0]);
    for (String name : new String[] {"flushed", "merged", "appended", "appended-merged"}) {
      Path rust = root.resolve(name);
      Path java = Files.createTempDirectory("simpletext-write-path");
      if (name.startsWith("appended")) {
        writeFirst(java);
        writeSecond(java, name.endsWith("merged"));
      } else {
        write(java, name.equals("merged"));
      }
      try (Directory r = FSDirectory.open(rust);
          Directory j = FSDirectory.open(java)) {
        compare(rust, r, j);
        CheckIndex.Status status;
        try (CheckIndex check = new CheckIndex(r)) {
          check.setLevel(CheckIndex.Level.MIN_LEVEL_FOR_SLOW_CHECKS);
          ByteArrayOutputStream log = new ByteArrayOutputStream();
          check.setInfoStream(new PrintStream(log, true, StandardCharsets.UTF_8));
          status = check.checkIndex();
          if (!status.clean) {
            throw new AssertionError(rust + ": CheckIndex failed\n" + log);
          }
        }
        String mine = dump(r);
        String lucene = dump(j);
        if (!mine.equals(lucene)) {
          String[] a = mine.split("\n");
          String[] b = lucene.split("\n");
          for (int k = 0; k < Math.min(a.length, b.length); k++) {
            if (!a[k].equals(b[k])) {
              throw new AssertionError(rust + ": read back\n  " + a[k] + "\nLucene's\n  " + b[k]);
            }
          }
          throw new AssertionError(rust + ": " + a.length + " dump lines, Lucene's " + b.length);
        }
      }
    }
    System.out.println("VerifySimpleTextWritePath: ok");
  }

  static byte[] read(Directory dir, String file) throws Exception {
    try (IndexInput in = dir.openInput(file, IOContext.READONCE)) {
      byte[] all = new byte[(int) in.length()];
      in.readBytes(all, 0, all.length);
      return all;
    }
  }

  /**
   * Every file of the segment but its {@code .si}, compound entries unpacked, by name without the
   * segment's; doc-values generation files are left to {@link #compare}.
   */
  static Map<String, byte[]> contents(Directory dir, SegmentCommitInfo sci) throws Exception {
    Map<String, byte[]> out = new TreeMap<>();
    Codec codec = sci.info.getCodec();
    String prefix = sci.info.name;
    List<String> dvFiles = new ArrayList<>();
    for (var files : sci.getDocValuesUpdatesFiles().values()) {
      dvFiles.addAll(files);
    }
    for (String file : sci.files()) {
      if (file.endsWith(".si") || dvFiles.contains(file)) {
        continue;
      }
      if (file.endsWith(".scf")) {
        try (Directory cfs = codec.compoundFormat().getCompoundReader(dir, sci.info)) {
          for (String entry : cfs.listAll()) {
            out.put("[cfs]" + entry.substring(prefix.length()), read(cfs, entry));
          }
        }
      } else {
        out.put(file.substring(prefix.length()), read(dir, file));
      }
    }
    return out;
  }

  static void compare(Path path, Directory r, Directory j) throws Exception {
    SegmentInfos rs = SegmentInfos.readLatestCommit(r);
    SegmentInfos js = SegmentInfos.readLatestCommit(j);
    if (rs.size() != js.size()) {
      throw new AssertionError(path + ": segments " + rs.size() + ", Lucene's " + js.size());
    }
    List<String> differ = new ArrayList<>();
    for (int s = 0; s < rs.size(); s++) {
      SegmentCommitInfo rsci = rs.info(s);
      SegmentCommitInfo jsci = js.info(s);
      String what =
          rsci.info.getCodec().getName()
              + " " + rsci.info.name
              + " maxDoc=" + rsci.info.maxDoc()
              + " compound=" + rsci.info.getUseCompoundFile()
              + " del=" + rsci.getDelCount() + "@" + rsci.getDelGen()
              + " fi=" + rsci.getFieldInfosGen()
              + " dv=" + rsci.getDocValuesGen()
              + " files=" + new java.util.TreeSet<>(rsci.files());
      String expected =
          "SimpleText"
              + " " + jsci.info.name
              + " maxDoc=" + jsci.info.maxDoc()
              + " compound=" + jsci.info.getUseCompoundFile()
              + " del=" + jsci.getDelCount() + "@" + jsci.getDelGen()
              + " fi=" + jsci.getFieldInfosGen()
              + " dv=" + jsci.getDocValuesGen()
              + " files=" + new java.util.TreeSet<>(jsci.files());
      if (!what.equals(expected)) {
        throw new AssertionError(path + ": segment " + s + "\n  " + what + "\nLucene's\n  " + expected);
      }
      Map<String, byte[]> mine = contents(r, rsci);
      Map<String, byte[]> lucene = contents(j, jsci);
      if (!mine.keySet().equals(lucene.keySet())) {
        throw new AssertionError(path + ": files " + mine.keySet() + ", Lucene's " + lucene.keySet());
      }
      for (String file : mine.keySet()) {
        if (!Arrays.equals(mine.get(file), lucene.get(file))) {
          differ.add(rsci.info.name + file);
        }
      }
      // Each updated field's generation file: the fields of one update round take generations
      // in HashMap order in Lucene, so compare the file each field's own generation names.
      try (DirectoryReader rr = DirectoryReader.open(r);
          DirectoryReader jr = DirectoryReader.open(j)) {
        LeafReader rl = rr.leaves().get(s).reader();
        LeafReader jl = jr.leaves().get(s).reader();
        for (FieldInfo fi : rl.getFieldInfos()) {
          long gen = fi.getDocValuesGen();
          FieldInfo jfi = jl.getFieldInfos().fieldInfo(fi.name);
          if ((gen == -1) != (jfi.getDocValuesGen() == -1)) {
            throw new AssertionError(path + ": " + fi.name + " dvGen " + gen);
          }
          if (gen == -1) {
            continue;
          }
          String rf = rsci.info.name + "_" + Long.toString(gen, Character.MAX_RADIX) + ".dat";
          String jf =
              jsci.info.name + "_" + Long.toString(jfi.getDocValuesGen(), Character.MAX_RADIX) + ".dat";
          if (!Arrays.equals(read(r, rf), read(j, jf))) {
            differ.add(rf + " (" + fi.name + ")");
          }
        }
      }
    }
    if (!differ.isEmpty()) {
      throw new AssertionError(path + ": differ from Lucene's bytes (" + js.getSegmentsFileName() + " in " + j + "): " + differ);
    }
  }

  /** Everything Lucene reads of an index, as text. */
  static String dump(Directory dir) throws Exception {
    StringBuilder out = new StringBuilder();
    try (DirectoryReader reader = DirectoryReader.open(dir)) {
      for (LeafReaderContext ctx : reader.leaves()) {
        LeafReader leaf = ctx.reader();
        int maxDoc = leaf.maxDoc();
        Bits live = leaf.getLiveDocs();
        out.append("segment ").append(maxDoc).append('\n');
        for (int doc = 0; doc < maxDoc; doc++) {
          out.append("live ").append(doc).append(' ').append(live == null || live.get(doc)).append('\n');
          out.append("stored ").append(leaf.storedFields().document(doc)).append('\n');
          var tv = leaf.termVectors().get(doc);
          if (tv != null) {
            for (String field : tv) {
              out.append("tv ").append(doc).append(' ').append(field);
              terms(out, tv.terms(field), true);
            }
          }
        }
        for (FieldInfo fi : leaf.getFieldInfos()) {
          String f = fi.name;
          out.append("field ").append(f).append(' ').append(fi.getIndexOptions()).append(' ')
              .append(fi.getDocValuesType()).append('\n');
          Terms terms = leaf.terms(f);
          if (terms != null) {
            out.append(" stats ").append(terms.size()).append(' ').append(terms.getSumDocFreq())
                .append(' ').append(terms.getSumTotalTermFreq()).append(' ')
                .append(terms.getDocCount());
            terms(out, terms, fi.getIndexOptions().compareTo(
                org.apache.lucene.index.IndexOptions.DOCS_AND_FREQS_AND_POSITIONS) >= 0);
          }
          NumericDocValues norms = leaf.getNormValues(f);
          if (norms != null) {
            while (norms.nextDoc() != DocIdSetIterator.NO_MORE_DOCS) {
              out.append(" norm ").append(norms.docID()).append('=').append(norms.longValue());
            }
            out.append('\n');
          }
          switch (fi.getDocValuesType()) {
            case NUMERIC -> {
              NumericDocValues v = leaf.getNumericDocValues(f);
              while (v.nextDoc() != DocIdSetIterator.NO_MORE_DOCS) {
                out.append(' ').append(v.docID()).append('=').append(v.longValue());
              }
            }
            case BINARY -> {
              BinaryDocValues v = leaf.getBinaryDocValues(f);
              while (v.nextDoc() != DocIdSetIterator.NO_MORE_DOCS) {
                out.append(' ').append(v.docID()).append('=').append(v.binaryValue());
              }
            }
            case SORTED -> {
              SortedDocValues v = leaf.getSortedDocValues(f);
              while (v.nextDoc() != DocIdSetIterator.NO_MORE_DOCS) {
                out.append(' ').append(v.docID()).append('=').append(v.ordValue()).append(':')
                    .append(v.lookupOrd(v.ordValue()));
              }
            }
            case SORTED_NUMERIC -> {
              SortedNumericDocValues v = leaf.getSortedNumericDocValues(f);
              while (v.nextDoc() != DocIdSetIterator.NO_MORE_DOCS) {
                out.append(' ').append(v.docID()).append('=');
                for (int k = 0; k < v.docValueCount(); k++) {
                  out.append(v.nextValue()).append(',');
                }
              }
            }
            case SORTED_SET -> {
              SortedSetDocValues v = leaf.getSortedSetDocValues(f);
              while (v.nextDoc() != DocIdSetIterator.NO_MORE_DOCS) {
                out.append(' ').append(v.docID()).append('=');
                for (int k = 0; k < v.docValueCount(); k++) {
                  long ord = v.nextOrd();
                  out.append(ord).append(':').append(v.lookupOrd(ord)).append(',');
                }
              }
            }
            default -> {}
          }
          out.append('\n');
          PointValues points = leaf.getPointValues(f);
          if (points != null) {
            List<String> all = new ArrayList<>();
            points.intersect(
                new PointValues.IntersectVisitor() {
                  @Override
                  public void visit(int docID) {}

                  @Override
                  public void visit(int docID, byte[] packedValue) {
                    all.add(docID + ":" + Arrays.toString(packedValue));
                  }

                  @Override
                  public PointValues.Relation compare(byte[] min, byte[] max) {
                    return PointValues.Relation.CELL_CROSSES_QUERY;
                  }
                });
            all.sort(null);
            out.append(" points ").append(points.size()).append(' ').append(points.getDocCount())
                .append(' ').append(Arrays.toString(points.getMinPackedValue())).append(' ')
                .append(Arrays.toString(points.getMaxPackedValue())).append(' ').append(all)
                .append('\n');
          }
          if (fi.getVectorDimension() > 0) {
            switch (fi.getVectorEncoding()) {
              case FLOAT32 -> {
                FloatVectorValues v = leaf.getFloatVectorValues(f);
                KnnVectorValues.DocIndexIterator it = v.iterator();
                while (it.nextDoc() != DocIdSetIterator.NO_MORE_DOCS) {
                  out.append(' ').append(it.docID()).append('=')
                      .append(Arrays.toString(v.vectorValue(it.index())));
                }
              }
              case BYTE -> {
                ByteVectorValues v = leaf.getByteVectorValues(f);
                KnnVectorValues.DocIndexIterator it = v.iterator();
                while (it.nextDoc() != DocIdSetIterator.NO_MORE_DOCS) {
                  out.append(' ').append(it.docID()).append('=')
                      .append(Arrays.toString(v.vectorValue(it.index())));
                }
              }
            }
            out.append('\n');
          }
        }
      }
    }
    return out.toString();
  }

  static void terms(StringBuilder out, Terms terms, boolean positions) throws Exception {
    TermsEnum te = terms.iterator();
    for (BytesRef term = te.next(); term != null; term = te.next()) {
      out.append(" term ").append(term);
      PostingsEnum pe = te.postings(null, PostingsEnum.ALL);
      while (pe.nextDoc() != DocIdSetIterator.NO_MORE_DOCS) {
        out.append(' ').append(pe.docID()).append('x').append(pe.freq());
        if (positions) {
          for (int k = 0; k < pe.freq(); k++) {
            out.append('[').append(pe.nextPosition()).append(' ').append(pe.startOffset())
                .append(' ').append(pe.endOffset()).append(' ').append(pe.getPayload()).append(']');
          }
        }
      }
    }
    out.append('\n');
  }
}
