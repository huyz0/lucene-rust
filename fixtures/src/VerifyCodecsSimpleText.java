import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.CompoundDirectory;
import org.apache.lucene.codecs.StoredFieldsReader;
import org.apache.lucene.codecs.StoredFieldsWriter;
import org.apache.lucene.codecs.DocValuesConsumer;
import org.apache.lucene.codecs.DocValuesProducer;
import org.apache.lucene.codecs.NormsConsumer;
import org.apache.lucene.codecs.PointsReader;
import org.apache.lucene.codecs.PointsWriter;
import org.apache.lucene.index.PointValues;
import org.apache.lucene.codecs.TermVectorsReader;
import org.apache.lucene.codecs.TermVectorsWriter;
import org.apache.lucene.index.BinaryDocValues;
import org.apache.lucene.index.CheckIndex;
import org.apache.lucene.index.DocValuesSkipper;
import org.apache.lucene.index.Fields;
import org.apache.lucene.index.SortedDocValues;
import org.apache.lucene.index.SortedNumericDocValues;
import org.apache.lucene.index.SortedSetDocValues;
import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexFileNames;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.SegmentCommitInfo;
import org.apache.lucene.index.SegmentInfos;
import org.apache.lucene.index.SegmentReader;
import org.apache.lucene.index.SortFieldProvider;
import org.apache.lucene.index.StoredFieldVisitor;
import org.apache.lucene.index.StoredFields;
import org.apache.lucene.codecs.KnnFieldVectorsWriter;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.search.AcceptDocs;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TopKnnCollector;
import org.apache.lucene.util.FixedBitSet;
import org.apache.lucene.search.Sort;
import org.apache.lucene.search.SortField;
import org.apache.lucene.store.ByteBuffersDataOutput;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.StringHelper;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.TreeMap;
import java.util.TreeSet;
import org.apache.lucene.codecs.FieldsConsumer;
import org.apache.lucene.codecs.FieldsProducer;
import org.apache.lucene.codecs.NormsProducer;
import org.apache.lucene.codecs.PostingsFormat;
import org.apache.lucene.codecs.lucene104.Lucene104Codec;
import org.apache.lucene.codecs.simpletext.SimpleTextCodec;
import org.apache.lucene.index.DocValuesSkipIndexType;
import org.apache.lucene.index.DocValuesType;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.NumericDocValues;
import org.apache.lucene.index.PostingsEnum;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.Terms;
import org.apache.lucene.index.TermsEnum;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.util.BytesRef;
import org.apache.lucene.util.InfoStream;
import org.apache.lucene.util.Version;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HexFormat;
import java.util.List;
import java.util.Map;

/**
 * Reverse-direction check for lucene-codecs' {@code simpletext} postings: {@code
 * write_simpletext_fixture} (Rust) writes {@code _0_SimpleText_0.pst} (docs-only fields with the
 * empty term and with binary terms holding {@code \n}, {@code \} and {@code 0xff}; a freqs field
 * and a positions/offsets/payloads field, both with norms), and real Lucene opens it through its
 * own {@code SimpleTextCodec().postingsFormat()} with a hand-built {@link SegmentInfo}/{@link
 * FieldInfos}, then:
 *
 * <ul>
 *   <li>(at open) checks the {@code checksum} line, and runs {@code checkIntegrity};
 *   <li>walks every term with {@code next()}, requiring the exact term list, {@code
 *       docFreq}/{@code totalTermFreq}, and every document's freq, positions, offsets and payloads
 *       through {@code postings(ALL)} to equal what Rust wrote, and {@code seekExact}s each term
 *       on a fresh enum;
 *   <li>re-writes the file with its own {@code SimpleTextFieldsWriter} from its own reader and the
 *       norms Rust used, as {@code _0_SimpleText_1.pst}, and requires the two files to be equal
 *       byte for byte -- which pins every skip list, its levels and its competitive impacts.
 * </ul>
 *
 * <p>Then {@code codec/}: a Rust-written SimpleText index with no segments file -- {@code _0}
 * (index sort, diagnostics, attributes, every stored type, a {@code .liv}, every doc-values type
 * with skip indexes, norms, postings and term vectors) and the compound {@code _1} (blocks and
 * their parent field) -- plus a {@code _2.inf} holding every kind of field. Each {@code .si},
 * {@code .inf}, {@code .liv}, {@code .scf}, {@code .fld}, {@code .dat}, {@code .len}, {@code .pst},
 * {@code .tvc}, {@code .dim}/{@code .dii} (points; one field built through temp files) and
 * {@code .vec}/{@code .gri} (vectors: every value, then searches with and without accept bits,
 * hits, scores and visited counts) is read by SimpleTextCodec's own format and must match Rust's manifest (doc
 * values, dictionaries and skipper intervals; norms; every vector occurrence; each points field's
 * stats and its points as a multiset), then re-written by Java's writer (points from Rust's
 * input, in Rust's order): byte for byte equal, except the {@code .si}, whose hash-ordered maps only compare
 * as line sets. Java then commits a {@code segments_N} over the Rust segments, opens them through
 * {@link DirectoryReader} (max/num docs, live docs, every stored value, points stats and a full
 * intersect, vector counts and the live-docs searches) and runs {@link
 * CheckIndex} at its slowest level: index-sort order, vectors against postings, impacts against
 * norms.
 *
 * <p>What it cannot catch: anything that binds the {@code _0_SimpleText_0.pst} into a segment
 * ({@code PerFieldPostingsFormat} attributes), since this port's IndexWriter does not route a field
 * to SimpleText; nor an {@code .si} whose maps are right but in an order Java would not write; nor
 * a doc-values file whose fields are in an order other than field-number order (Java's re-write
 * adds them in that order, where a flush adds them in hash order).
 */
public class VerifyCodecsSimpleText {
  static final String[] NAMES = {"id", "bin", "tag", "body"};

  public static void main(String[] args) throws Exception {
    Path dir = Path.of(args[0]);
    Map<String, String> manifest = new HashMap<>();
    for (String line : Files.readAllLines(dir.resolve("manifest.properties"))) {
      int eq = line.indexOf('=');
      if (eq > 0) {
        manifest.put(line.substring(0, eq), line.substring(eq + 1));
      }
    }
    byte[] id = HexFormat.of().parseHex(manifest.get("id_hex"));
    int maxDoc = Integer.parseInt(manifest.get("max_doc"));

    IndexOptions[] options = {
      IndexOptions.DOCS,
      IndexOptions.DOCS,
      IndexOptions.DOCS_AND_FREQS,
      IndexOptions.DOCS_AND_FREQS_AND_POSITIONS_AND_OFFSETS
    };
    FieldInfo[] infos = new FieldInfo[NAMES.length];
    for (int i = 0; i < NAMES.length; i++) {
      infos[i] =
          new FieldInfo(
              NAMES[i],
              i,
              false,
              i < 2,
              i == 3,
              options[i],
              DocValuesType.NONE,
              DocValuesSkipIndexType.NONE,
              -1,
              new HashMap<>(),
              0,
              0,
              0,
              0,
              VectorEncoding.FLOAT32,
              VectorSimilarityFunction.EUCLIDEAN,
              false,
              false);
    }
    FieldInfos fis = new FieldInfos(infos);
    Map<String, long[]> norms = new HashMap<>();
    for (String line : Files.readAllLines(dir.resolve("norms.tsv"))) {
      String[] e = line.split("\t");
      long[] values = new long[e.length - 1];
      for (int i = 1; i < e.length; i++) values[i - 1] = Long.parseLong(e[i]);
      norms.put(e[0], values);
    }
    Map<String, List<String>> byField = new HashMap<>();
    for (String line : Files.readAllLines(dir.resolve("terms.tsv"))) {
      byField.computeIfAbsent(line.substring(0, line.indexOf('\t')), k -> new ArrayList<>())
          .add(line);
    }

    int failures = 0;
    PostingsFormat format = new SimpleTextCodec().postingsFormat();
    try (Directory directory = FSDirectory.open(dir)) {
      SegmentInfo si =
          new SegmentInfo(
              directory,
              Version.LATEST,
              Version.LATEST,
              "_0",
              maxDoc,
              false,
              false,
              new Lucene104Codec(),
              Collections.emptyMap(),
              id,
              new HashMap<>(),
              null);
      SegmentReadState read =
          new SegmentReadState(directory, si, fis, IOContext.DEFAULT, "SimpleText_0");
      try (FieldsProducer fields = format.fieldsProducer(read)) {
        fields.checkIntegrity();
        long termCount = 0;
        for (String field : List.of("bin", "body", "id", "tag")) {
          List<String> lines = byField.get(field);
          failures += verifyField(field, fields.terms(field), lines);
          termCount += lines.size();
        }
        System.out.println("SimpleText_0: " + termCount + " terms, ok");

        // Idempotent, so a failing run can be re-run to show its output.
        Files.deleteIfExists(dir.resolve("_0_SimpleText_1.pst"));
        SegmentWriteState write =
            new SegmentWriteState(
                InfoStream.NO_OUTPUT, directory, si, fis, null, IOContext.DEFAULT, "SimpleText_1");
        try (FieldsConsumer consumer = format.fieldsConsumer(write)) {
          consumer.write(fields, new ArrayNorms(norms));
        }
      }
    }
    byte[] rust = Files.readAllBytes(dir.resolve("_0_SimpleText_0.pst"));
    byte[] java = Files.readAllBytes(dir.resolve("_0_SimpleText_1.pst"));
    if (!Arrays.equals(rust, java)) {
      System.out.println("re-written .pst differs: Rust " + rust.length + " bytes, Java "
          + java.length + ", first difference at " + Arrays.mismatch(rust, java));
      failures++;
    }
    failures += verifyCodec(dir.resolve("codec"));
    if (failures > 0) {
      System.out.println(failures + " failure(s)");
      System.exit(1);
    }
    System.out.println("SimpleText postings and segment formats verified against real Lucene. PASS");
  }

  /** Norms out of {@code norms.tsv}, every document holding one. */
  static final class ArrayNorms extends NormsProducer {
    final Map<String, long[]> norms;

    ArrayNorms(Map<String, long[]> norms) {
      this.norms = norms;
    }

    @Override
    public NumericDocValues getNorms(FieldInfo field) {
      long[] values = norms.get(field.name);
      if (values == null) throw new IllegalStateException("no norms for " + field.name);
      return new NumericDocValues() {
        int doc = -1;

        @Override
        public long longValue() {
          return values[doc];
        }

        @Override
        public boolean advanceExact(int target) {
          doc = target;
          return target < values.length;
        }

        @Override
        public int docID() {
          return doc;
        }

        @Override
        public int nextDoc() {
          return advance(doc + 1);
        }

        @Override
        public int advance(int target) {
          doc = target < values.length ? target : DocIdSetIterator.NO_MORE_DOCS;
          return doc;
        }

        @Override
        public long cost() {
          return values.length;
        }
      };
    }

    @Override
    public void checkIntegrity() {}

    @Override
    public void close() {}
  }

  static String hex(BytesRef b) {
    return HexFormat.of().formatHex(b.bytes, b.offset, b.offset + b.length);
  }

  /** {@code write_simpletext_fixture}'s line for the enum's current term. */
  static String termLine(String field, Terms terms, TermsEnum te) throws Exception {
    StringBuilder sb = new StringBuilder(field);
    sb.append('\t').append(hex(te.term())).append('\t').append(te.docFreq()).append('\t')
        .append(te.totalTermFreq());
    PostingsEnum pe = te.postings(null, PostingsEnum.ALL);
    for (int doc = pe.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS; doc = pe.nextDoc()) {
      int freq = pe.freq();
      sb.append('\t').append(doc).append(':').append(freq);
      if (terms.hasPositions()) {
        StringBuilder pos = new StringBuilder();
        StringBuilder offs = new StringBuilder();
        StringBuilder pays = new StringBuilder();
        for (int k = 0; k < freq; k++) {
          pos.append(k == 0 ? "" : ".").append(pe.nextPosition());
          offs.append(k == 0 ? "" : ".").append(pe.startOffset()).append('-')
              .append(pe.endOffset());
          BytesRef pl = pe.getPayload();
          pays.append(k == 0 ? "" : ".").append(pl == null || pl.length == 0 ? "-" : hex(pl));
        }
        sb.append(':').append(pos).append(':').append(offs).append(':').append(pays);
      }
    }
    return sb.toString();
  }

  static int verifyField(String field, Terms terms, List<String> lines) throws Exception {
    int failures = 0;
    TermsEnum walk = terms.iterator();
    for (String line : lines) {
      String[] e = line.split("\t", 5);
      BytesRef want = new BytesRef(HexFormat.of().parseHex(e[1]));
      BytesRef got = walk.next();
      if (got == null || !got.equals(want)) {
        System.out.println(field + " next() " + got + " expected " + want);
        return failures + 1;
      }
      String gotLine = termLine(field, terms, walk);
      if (!gotLine.equals(line)) {
        System.out.println(field + " " + e[1] + "\n  got  " + gotLine + "\n  want " + line);
        failures++;
      }
      if (!terms.iterator().seekExact(want)) {
        System.out.println(field + " seekExact missed " + e[1]);
        failures++;
      }
    }
    if (walk.next() != null) {
      System.out.println("extra term in " + field);
      failures++;
    }
    return failures;
  }

  // ---- the per-segment formats: .si, .inf, .liv, .scf, .fld ------------------------------

  static final HexFormat HEX = HexFormat.of();
  static final IOContext CTX = IOContext.DEFAULT;

  static String hexs(String s) {
    return HEX.formatHex(s.getBytes(StandardCharsets.UTF_8));
  }

  /** Lines that must equal a Rust-written manifest. */
  static int expect(Path manifest, List<String> got) throws Exception {
    List<String> want = Files.readAllLines(manifest);
    if (want.equals(got)) return 0;
    for (int i = 0; i < Math.max(want.size(), got.size()); i++) {
      String w = i < want.size() ? want.get(i) : "<none>";
      String g = i < got.size() ? got.get(i) : "<none>";
      if (!w.equals(g)) {
        System.out.println(manifest.getFileName() + " line " + i + "\n  got  " + g + "\n  want " + w);
        break;
      }
    }
    return 1;
  }

  /** Two files that must be equal byte for byte. */
  static int sameBytes(Directory a, String an, Directory b, String bn) throws Exception {
    byte[] x = readAll(a, an);
    byte[] y = readAll(b, bn);
    if (Arrays.equals(x, y)) return 0;
    System.out.println(an + " (Rust " + x.length + " bytes) vs Java's re-write " + bn + " ("
        + y.length + " bytes): first difference at " + Arrays.mismatch(x, y));
    return 1;
  }

  static byte[] readAll(Directory d, String name) throws Exception {
    try (IndexInput in = d.openInput(name, IOContext.READONCE)) {
      byte[] b = new byte[Math.toIntExact(in.length())];
      in.readBytes(b, 0, b.length);
      return b;
    }
  }

  static List<String> sortedLines(byte[] b) {
    List<String> lines = new ArrayList<>(List.of(new String(b, StandardCharsets.UTF_8).split("\n")));
    lines.removeIf(l -> l.startsWith("checksum "));
    Collections.sort(lines);
    return lines;
  }

  static void sortedMap(List<String> lines, String tag, Map<String, String> map) {
    for (Map.Entry<String, String> e : new TreeMap<>(map).entrySet()) {
      lines.add(tag + "\t" + hexs(e.getKey()) + "\t" + hexs(e.getValue()));
    }
  }

  /** The .si as {@code GenCodecsSimpleTextCodec}'s {@code si.SEG.tsv} records it. */
  static List<String> segmentInfoLines(SegmentInfo si) throws Exception {
    List<String> lines = new ArrayList<>();
    lines.add("version\t" + si.getVersion());
    lines.add("minVersion\t" + si.getMinVersion());
    lines.add("docCount\t" + si.maxDoc());
    lines.add("compound\t" + si.getUseCompoundFile());
    lines.add("hasBlocks\t" + si.getHasBlocks());
    sortedMap(lines, "diag", si.getDiagnostics());
    sortedMap(lines, "att", si.getAttributes());
    for (String f : new TreeSet<>(si.files())) lines.add("file\t" + f);
    lines.add("id\t" + HEX.formatHex(si.getId()));
    Sort sort = si.getIndexSort();
    if (sort != null) {
      for (SortField sf : sort.getSort()) {
        ByteBuffersDataOutput b = new ByteBuffersDataOutput();
        SortFieldProvider.write(sf, b);
        lines.add("sort\t" + sf.getIndexSorter().getProviderName() + "\t" + hexs(sf.toString())
            + "\t" + HEX.formatHex(b.toArrayCopy()));
      }
    }
    return lines;
  }

  /** Every property of every FieldInfo, as {@code fields.FILE.tsv} records it. */
  static List<String> fieldInfoLines(FieldInfos fis) {
    List<String> lines = new ArrayList<>();
    for (FieldInfo fi : fis) {
      StringBuilder atts = new StringBuilder();
      for (Map.Entry<String, String> e : new TreeMap<>(fi.attributes()).entrySet()) {
        if (atts.length() > 0) atts.append(',');
        atts.append(hexs(e.getKey())).append('=').append(hexs(e.getValue()));
      }
      lines.add(String.join("\t", hexs(fi.name), Integer.toString(fi.number),
          fi.getIndexOptions().toString(), Boolean.toString(fi.hasTermVectors()),
          Boolean.toString(fi.hasPayloads()), Boolean.toString(fi.omitsNorms()),
          fi.getDocValuesType().toString(), fi.docValuesSkipIndexType().toString(),
          Long.toString(fi.getDocValuesGen()), atts.toString(),
          Integer.toString(fi.getPointDimensionCount()),
          Integer.toString(fi.getPointIndexDimensionCount()),
          Integer.toString(fi.getPointNumBytes()), Integer.toString(fi.getVectorDimension()),
          fi.getVectorEncoding().name(), fi.getVectorSimilarityFunction().name(),
          Boolean.toString(fi.isSoftDeletesField()), Boolean.toString(fi.isParentField())));
    }
    return lines;
  }

  /** Every stored value as a {@code stored.SEG.tsv} line; optionally re-written by Java. */
  static final class Dump extends StoredFieldVisitor {
    final List<String> lines;
    final StoredFieldsWriter copy;

    Dump(List<String> lines, StoredFieldsWriter copy) {
      this.lines = lines;
      this.copy = copy;
    }

    @Override
    public Status needsField(FieldInfo fieldInfo) {
      return Status.YES;
    }

    void add(FieldInfo fi, String type, String value) {
      lines.add(fi.number + "\t" + type + "\t" + value);
    }

    @Override
    public void stringField(FieldInfo fi, String value) throws IOException {
      add(fi, "string", hexs(value));
      if (copy != null) copy.writeField(fi, value);
    }

    @Override
    public void binaryField(FieldInfo fi, byte[] value) throws IOException {
      add(fi, "binary", HEX.formatHex(value));
      if (copy != null) copy.writeField(fi, new BytesRef(value));
    }

    @Override
    public void intField(FieldInfo fi, int value) throws IOException {
      add(fi, "int", Integer.toString(value));
      if (copy != null) copy.writeField(fi, value);
    }

    @Override
    public void longField(FieldInfo fi, long value) throws IOException {
      add(fi, "long", Long.toString(value));
      if (copy != null) copy.writeField(fi, value);
    }

    @Override
    public void floatField(FieldInfo fi, float value) throws IOException {
      add(fi, "float",
          String.format("%08x", Float.floatToRawIntBits(value)) + "\t" + Float.toString(value));
      if (copy != null) copy.writeField(fi, value);
    }

    @Override
    public void doubleField(FieldInfo fi, double value) throws IOException {
      add(fi, "double", String.format("%016x", Double.doubleToRawLongBits(value)) + "\t"
          + Double.toString(value));
      if (copy != null) copy.writeField(fi, value);
    }
  }

  /** A doc-values field as {@code GenCodecsSimpleTextCodec} dumps it: values, dictionary, skipper. */
  static void dvLines(List<String> lines, DocValuesProducer p, FieldInfo fi) throws IOException {
    final int end = DocIdSetIterator.NO_MORE_DOCS;
    lines.add("field\t" + hexs(fi.name) + "\t" + fi.getDocValuesType());
    switch (fi.getDocValuesType()) {
      case NUMERIC -> {
        NumericDocValues v = p.getNumeric(fi);
        for (int d = v.nextDoc(); d != end; d = v.nextDoc()) {
          lines.add("d\t" + d + "\t" + v.longValue());
        }
      }
      case BINARY -> {
        BinaryDocValues v = p.getBinary(fi);
        for (int d = v.nextDoc(); d != end; d = v.nextDoc()) {
          lines.add("d\t" + d + "\t" + hex(v.binaryValue()));
        }
      }
      case SORTED -> {
        SortedDocValues v = p.getSorted(fi);
        for (int d = v.nextDoc(); d != end; d = v.nextDoc()) {
          lines.add("d\t" + d + "\t" + v.ordValue() + "\t" + hex(v.lookupOrd(v.ordValue())));
        }
        for (int o = 0; o < v.getValueCount(); o++) {
          lines.add("ord\t" + o + "\t" + hex(v.lookupOrd(o)));
        }
      }
      case SORTED_NUMERIC -> {
        SortedNumericDocValues v = p.getSortedNumeric(fi);
        for (int d = v.nextDoc(); d != end; d = v.nextDoc()) {
          StringBuilder sb = new StringBuilder();
          for (int k = 0; k < v.docValueCount(); k++) sb.append(k == 0 ? "" : ",").append(v.nextValue());
          lines.add("d\t" + d + "\t" + sb);
        }
      }
      case SORTED_SET -> {
        SortedSetDocValues v = p.getSortedSet(fi);
        for (int d = v.nextDoc(); d != end; d = v.nextDoc()) {
          StringBuilder sb = new StringBuilder();
          for (int k = 0; k < v.docValueCount(); k++) sb.append(k == 0 ? "" : ",").append(v.nextOrd());
          lines.add("d\t" + d + "\t" + sb);
        }
        for (long o = 0; o < v.getValueCount(); o++) {
          lines.add("ord\t" + o + "\t" + hex(v.lookupOrd(o)));
        }
      }
      default -> throw new AssertionError(fi.getDocValuesType());
    }
    if (fi.docValuesSkipIndexType() != DocValuesSkipIndexType.NONE) {
      DocValuesSkipper s = p.getSkipper(fi);
      lines.add("skipall\t" + s.minValue() + "\t" + s.maxValue() + "\t" + s.docCount() + "\t"
          + s.maxValueCount());
      for (int t = 0; ; ) {
        s.advance(t);
        if (s.minDocID(0) == end) break;
        lines.add("skip\t" + s.minDocID(0) + "\t" + s.maxDocID(0) + "\t" + s.minValue(0) + "\t"
            + s.maxValue(0) + "\t" + s.docCount(0));
        t = s.maxDocID(0) + 1;
      }
    }
  }

  /** Rust's {@code .dat}: read against its manifest, re-written by Java's consumer in field order. */
  static int verifyDocValues(Path root, Directory core, Directory out, SegmentInfo si,
      FieldInfos fis, String name) throws Exception {
    Codec codec = new SimpleTextCodec();
    List<String> lines = new ArrayList<>();
    try (DocValuesProducer p =
            codec.docValuesFormat().fieldsProducer(new SegmentReadState(core, si, fis, CTX));
        DocValuesConsumer c = codec.docValuesFormat().fieldsConsumer(
            new SegmentWriteState(InfoStream.NO_OUTPUT, out, si, fis, null, CTX))) {
      p.checkIntegrity();
      for (FieldInfo fi : fis) {
        switch (fi.getDocValuesType()) {
          case NONE -> {
            continue;
          }
          case NUMERIC -> c.addNumericField(fi, p);
          case BINARY -> c.addBinaryField(fi, p);
          case SORTED -> c.addSortedField(fi, p);
          case SORTED_NUMERIC -> c.addSortedNumericField(fi, p);
          case SORTED_SET -> c.addSortedSetField(fi, p);
        }
        dvLines(lines, p, fi);
      }
    }
    return expect(root.resolve("dv." + name + ".dat.tsv"), lines)
        + sameBytes(core, name + ".dat", out, name + ".dat");
  }

  /**
   * Rust's {@code .len}, {@code .pst} and {@code .tvc}: norms and vectors read against their
   * manifests; each file re-written by Java's writer (the postings with the norms Rust wrote).
   */
  static int verifyIndexed(Path root, Directory core, Directory out, SegmentInfo si,
      FieldInfos fis, String name) throws Exception {
    Codec codec = new SimpleTextCodec();
    int failures = 0;
    final int end = DocIdSetIterator.NO_MORE_DOCS;
    SegmentReadState read = new SegmentReadState(core, si, fis, CTX);
    SegmentWriteState write = new SegmentWriteState(InfoStream.NO_OUTPUT, out, si, fis, null, CTX);
    try (NormsProducer norms = codec.normsFormat().normsProducer(read)) {
      norms.checkIntegrity();
      List<String> lines = new ArrayList<>();
      try (NormsConsumer c = codec.normsFormat().normsConsumer(write)) {
        for (FieldInfo fi : fis) {
          if (!fi.hasNorms()) continue;
          lines.add("field\t" + hexs(fi.name));
          NumericDocValues v = norms.getNorms(fi);
          for (int d = v.nextDoc(); d != end; d = v.nextDoc()) {
            lines.add("d\t" + d + "\t" + v.longValue());
          }
          c.addNormsField(fi, norms);
        }
      }
      failures += expect(root.resolve("norms." + name + ".tsv"), lines);
      failures += sameBytes(core, name + ".len", out, name + ".len");
      try (FieldsProducer fields = codec.postingsFormat().fieldsProducer(read)) {
        fields.checkIntegrity();
        try (FieldsConsumer c = codec.postingsFormat().fieldsConsumer(write)) {
          c.write(fields, norms);
        }
      }
      failures += sameBytes(core, name + ".pst", out, name + ".pst");
    }
    List<String> lines = new ArrayList<>();
    try (TermVectorsReader tv = codec.termVectorsFormat().vectorsReader(core, si, fis, CTX);
        TermVectorsWriter w = codec.termVectorsFormat().vectorsWriter(out, si, CTX)) {
      tv.checkIntegrity();
      for (int d = 0; d < si.maxDoc(); d++) {
        lines.add("doc\t" + d);
        Fields f = tv.get(d);
        if (f == null) {
          lines.add("none");
          w.startDocument(0);
          w.finishDocument();
          continue;
        }
        w.startDocument(f.size());
        for (String field : f) {
          Terms terms = f.terms(field);
          boolean pos = terms.hasPositions();
          boolean off = terms.hasOffsets();
          boolean pay = terms.hasPayloads();
          lines.add("field\t" + hexs(field) + "\t" + pos + "\t" + off + "\t" + pay + "\t"
              + terms.size());
          w.startField(fis.fieldInfo(field), (int) terms.size(), pos, off, pay);
          TermsEnum te = terms.iterator();
          for (BytesRef t = te.next(); t != null; t = te.next()) {
            int freq = (int) te.totalTermFreq();
            lines.add("term\t" + hex(t) + "\t" + freq);
            w.startTerm(t, freq);
            if (pos || off) {
              PostingsEnum pe = te.postings(null, PostingsEnum.ALL);
              pe.nextDoc();
              for (int k = 0; k < freq; k++) {
                int position = pe.nextPosition();
                BytesRef payload = pe.getPayload();
                lines.add("pos\t" + position + "\t" + pe.startOffset() + "\t" + pe.endOffset()
                    + "\t" + (payload == null ? "-" : hex(payload)));
                w.addPosition(position, pe.startOffset(), pe.endOffset(), payload);
              }
            }
            w.finishTerm();
          }
          w.finishField();
        }
        w.finishDocument();
      }
      w.finish(si.maxDoc());
    }
    failures += expect(root.resolve("tv." + name + ".tsv"), lines);
    failures += sameBytes(core, name + ".tvc", out, name + ".tvc");
    return failures;
  }

  /** Each segment/field's points stats line, as Rust's manifest has it. */
  static final Map<String, String> POINT_HEADS = new HashMap<>();

  /** Per segment/field: the vector count, then the {@code live} searches' lines. */
  static final Map<String, List<String>> KNN_LIVE = new HashMap<>();

  static String floatsHex(float[] v) {
    StringBuilder sb = new StringBuilder();
    for (float f : v) sb.append(String.format("%08x", Float.floatToRawIntBits(f)));
    return sb.toString();
  }

  static float[] parseFloats(String hex) {
    float[] v = new float[hex.length() / 8];
    for (int i = 0; i < v.length; i++) {
      v[i] = Float.intBitsToFloat(Integer.parseUnsignedInt(hex.substring(8 * i, 8 * i + 8), 16));
    }
    return v;
  }

  /** One search's {@code hit}/{@code visited} lines. */
  static void hitLines(List<String> lines, TopDocs td) {
    for (ScoreDoc sd : td.scoreDocs) {
      lines.add("hit\t" + sd.doc + "\t" + String.format("%08x", Float.floatToRawIntBits(sd.score)));
    }
    lines.add("visited\t" + td.totalHits.value() + "\t" + td.totalHits.relation());
  }

  /**
   * knn.SEG.tsv: Java's SimpleTextKnnVectorsReader must read every vector Rust wrote and its
   * searches must hit the same docs with the same score bits and visit as many; its writer,
   * handed the same vectors, must write the same .vec and .gri.
   */
  static int verifyKnn(Path root, Directory core, Directory out, SegmentInfo si, FieldInfos fis,
      String name) throws Exception {
    Codec codec = new SimpleTextCodec();
    int maxDoc = si.maxDoc();
    FixedBitSet live = new FixedBitSet(maxDoc);
    Path liveTsv = root.resolve("live." + name + ".tsv");
    String bitsLine = Files.exists(liveTsv) ? Files.readAllLines(liveTsv).get(1) : null;
    FixedBitSet odd = new FixedBitSet(maxDoc);
    for (int d = 0; d < maxDoc; d++) {
      if (bitsLine == null || bitsLine.charAt(d) == '1') live.set(d);
      if (d % 2 == 1) odd.set(d);
    }
    List<String> want = Files.readAllLines(root.resolve("knn." + name + ".tsv"));
    List<String> got = new ArrayList<>();
    Map<String, List<Object>> vectors = new java.util.LinkedHashMap<>();
    try (KnnVectorsReader kr =
        codec.knnVectorsFormat().fieldsReader(new SegmentReadState(core, si, fis, CTX))) {
      kr.checkIntegrity();
      String field = null;
      boolean bytes = false;
      List<String> liveLines = null;
      for (int i = 0; i < want.size(); i++) {
        String[] r = want.get(i).split("\t");
        if (r[0].equals("field")) {
          field = new String(HEX.parseHex(r[1]), StandardCharsets.UTF_8);
          FieldInfo fi = fis.fieldInfo(field);
          bytes = fi.getVectorEncoding() == VectorEncoding.BYTE;
          KnnVectorValues values =
              bytes ? kr.getByteVectorValues(field) : kr.getFloatVectorValues(field);
          got.add(String.join("\t", "field", r[1], fi.getVectorEncoding().name(),
              fi.getVectorSimilarityFunction().name(), Integer.toString(values.dimension()),
              Integer.toString(values.size())));
          List<Object> vs = new ArrayList<>();
          vectors.put(field, vs);
          for (int ord = 0; ord < values.size(); ord++) {
            int doc = values.ordToDoc(ord);
            String v;
            if (bytes) {
              byte[] b = ((ByteVectorValues) values).vectorValue(ord).clone();
              v = HEX.formatHex(b);
              vs.add(new Object[] {doc, b});
            } else {
              float[] f = ((FloatVectorValues) values).vectorValue(ord).clone();
              v = floatsHex(f);
              vs.add(new Object[] {doc, f});
            }
            got.add("v\t" + ord + "\t" + doc + "\t" + v);
          }
          liveLines = new ArrayList<>();
          liveLines.add(Integer.toString(values.size()));
          KNN_LIVE.put(name + "/" + field, liveLines);
        } else if (r[0].equals("q")) {
          got.add(want.get(i));
          int k = Integer.parseInt(r[1]);
          int limit = Integer.parseInt(r[2]);
          Bits bits = r[3].equals("none") ? null : r[3].equals("live") ? live : odd;
          TopKnnCollector c = new TopKnnCollector(k, limit);
          AcceptDocs accept = AcceptDocs.fromLiveDocs(bits, maxDoc);
          if (bytes) {
            kr.search(field, HEX.parseHex(r[4]), c, accept);
          } else {
            kr.search(field, parseFloats(r[4]), c, accept);
          }
          int from = got.size();
          hitLines(got, c.topDocs());
          if (r[3].equals("live")) {
            liveLines.add(want.get(i));
            liveLines.addAll(got.subList(from, got.size()));
          }
        }
      }
    }
    int failures = expect(root.resolve("knn." + name + ".tsv"), got);

    try (KnnVectorsWriter w = codec.knnVectorsFormat().fieldsWriter(
        new SegmentWriteState(InfoStream.NO_OUTPUT, out, si, fis, null, CTX))) {
      for (Map.Entry<String, List<Object>> e : vectors.entrySet()) {
        KnnFieldVectorsWriter<?> fw = w.addField(fis.fieldInfo(e.getKey()));
        for (Object o : e.getValue()) {
          Object[] dv = (Object[]) o;
          if (dv[1] instanceof byte[] b) {
            @SuppressWarnings("unchecked")
            KnnFieldVectorsWriter<byte[]> bw = (KnnFieldVectorsWriter<byte[]>) fw;
            bw.addValue((Integer) dv[0], b);
          } else {
            @SuppressWarnings("unchecked")
            KnnFieldVectorsWriter<float[]> f = (KnnFieldVectorsWriter<float[]>) fw;
            f.addValue((Integer) dv[0], (float[]) dv[1]);
          }
        }
      }
      w.flush(maxDoc, null);
      w.finish();
    }
    failures += sameBytes(core, name + ".vec", out, name + ".vec");
    failures += sameBytes(core, name + ".gri", out, name + ".gri");
    System.out.println(name + ": " + vectors.size() + " vector fields read, searched, re-written");
    return failures;
  }

  /** Through the DirectoryReader: each vector field's count and its {@code live} searches. */
  static int knnThroughReader(LeafReader leaf, String name) throws Exception {
    int failures = 0;
    for (FieldInfo fi : leaf.getFieldInfos()) {
      if (fi.getVectorDimension() == 0) continue;
      List<String> want = KNN_LIVE.get(name + "/" + fi.name);
      if (want == null) {
        System.out.println(name + "/" + fi.name + ": a vector field Java never read");
        failures++;
        continue;
      }
      boolean bytes = fi.getVectorEncoding() == VectorEncoding.BYTE;
      KnnVectorValues values =
          bytes ? leaf.getByteVectorValues(fi.name) : leaf.getFloatVectorValues(fi.name);
      List<String> got = new ArrayList<>();
      got.add(Integer.toString(values.size()));
      for (String q : want) {
        if (!q.startsWith("q\t")) continue;
        String[] r = q.split("\t");
        got.add(q);
        TopKnnCollector c = new TopKnnCollector(Integer.parseInt(r[1]), Integer.parseInt(r[2]));
        AcceptDocs accept = AcceptDocs.fromLiveDocs(leaf.getLiveDocs(), leaf.maxDoc());
        if (bytes) {
          leaf.searchNearestVectors(fi.name, HEX.parseHex(r[4]), c, accept);
        } else {
          leaf.searchNearestVectors(fi.name, parseFloats(r[4]), c, accept);
        }
        hitLines(got, c.topDocs());
      }
      if (!got.equals(want)) {
        System.out.println(name + "/" + fi.name + ": DirectoryReader's vectors differ: " + got);
        failures++;
      }
    }
    return failures;
  }

  static String pointsHead(String field, PointValues v) throws IOException {
    return String.join("\t", "field", hexs(field), Integer.toString(v.getNumDimensions()),
        Integer.toString(v.getNumIndexDimensions()), Integer.toString(v.getBytesPerDimension()),
        Long.toString(v.size()), Integer.toString(v.getDocCount()),
        HEX.formatHex(v.getMinPackedValue()), HEX.formatHex(v.getMaxPackedValue()));
  }

  /**
   * Rust's {@code .dim}/{@code .dii}: each field's stats against {@code points.SEG.tsv} and its
   * points against the manifest's as a multiset; then Java's writer is handed the manifest's
   * points in Rust's order and must write both files byte for byte.
   */
  static int verifyPoints(Path root, Directory core, Directory out, SegmentInfo si,
      FieldInfos fis, String name) throws Exception {
    Codec codec = new SimpleTextCodec();
    Map<String, List<String>> manifest = new TreeMap<>();
    Map<String, String> heads = new TreeMap<>();
    String current = null;
    for (String line : Files.readAllLines(root.resolve("points." + name + ".tsv"))) {
      String[] r = line.split("\t");
      if (r[0].equals("field")) {
        current = new String(HEX.parseHex(r[1]), StandardCharsets.UTF_8);
        heads.put(current, line);
        manifest.put(current, new ArrayList<>());
      } else {
        manifest.get(current).add(r[1] + "\t" + r[2]);
      }
    }
    int failures = 0;
    Map<String, PointValues> replay = new HashMap<>();
    try (PointsReader pr =
        codec.pointsFormat().fieldsReader(new SegmentReadState(core, si, fis, CTX))) {
      pr.checkIntegrity();
      for (FieldInfo fi : fis) {
        if (fi.getPointDimensionCount() == 0) continue;
        PointValues v = pr.getValues(fi.name);
        String head = pointsHead(fi.name, v);
        POINT_HEADS.put(name + "/" + fi.name, head);
        if (!head.equals(heads.get(fi.name))) {
          System.out.println(name + "/" + fi.name + ": got " + head + "\n  want " + heads.get(fi.name));
          failures++;
        }
        List<String> got = new ArrayList<>();
        v.getPointTree().visitDocValues(
            new PointValues.IntersectVisitor() {
              @Override
              public void visit(int doc) {
                throw new IllegalStateException();
              }

              @Override
              public void visit(int doc, byte[] packed) {
                got.add(doc + "\t" + HEX.formatHex(packed));
              }

              @Override
              public PointValues.Relation compare(byte[] min, byte[] max) {
                return PointValues.Relation.CELL_CROSSES_QUERY;
              }
            });
        List<String> want = new ArrayList<>(manifest.get(fi.name));
        Collections.sort(got);
        Collections.sort(want);
        if (!got.equals(want)) {
          System.out.println(name + "/" + fi.name + ": Java reads other points than Rust wrote");
          failures++;
        }
        replay.put(fi.name, new Replay(v, manifest.get(fi.name)));
      }
    }
    try (PointsWriter w = codec.pointsFormat().fieldsWriter(
        new SegmentWriteState(InfoStream.NO_OUTPUT, out, si, fis, null, CTX))) {
      PointsReader in = new PointsReader() {
        @Override
        public void checkIntegrity() {}

        @Override
        public PointValues getValues(String field) {
          return replay.get(field);
        }

        @Override
        public void close() {}
      };
      for (FieldInfo fi : fis) {
        if (fi.getPointDimensionCount() != 0) w.writeField(fi, in);
      }
      w.finish();
    }
    return failures + sameBytes(core, name + ".dim", out, name + ".dim")
        + sameBytes(core, name + ".dii", out, name + ".dii");
  }

  /** A field's points as one leaf, visited in the manifest's order. */
  static final class Replay extends PointValues implements PointValues.PointTree {
    final PointValues like;
    final int[] docs;
    final byte[][] packed;

    Replay(PointValues like, List<String> points) {
      this.like = like;
      docs = new int[points.size()];
      packed = new byte[points.size()][];
      for (int i = 0; i < docs.length; i++) {
        String[] r = points.get(i).split("\t");
        docs[i] = Integer.parseInt(r[0]);
        packed[i] = HEX.parseHex(r[1]);
      }
    }

    @Override
    public PointTree getPointTree() {
      return this;
    }

    @Override
    public Replay clone() {
      return this;
    }

    @Override
    public boolean moveToChild() {
      return false;
    }

    @Override
    public boolean moveToSibling() {
      return false;
    }

    @Override
    public boolean moveToParent() {
      return false;
    }

    @Override
    public byte[] getMinPackedValue() {
      try {
        return like.getMinPackedValue();
      } catch (IOException e) {
        throw new java.io.UncheckedIOException(e);
      }
    }

    @Override
    public byte[] getMaxPackedValue() {
      try {
        return like.getMaxPackedValue();
      } catch (IOException e) {
        throw new java.io.UncheckedIOException(e);
      }
    }

    @Override
    public void visitDocIDs(PointValues.IntersectVisitor visitor) throws IOException {
      for (int d : docs) visitor.visit(d);
    }

    @Override
    public void visitDocValues(PointValues.IntersectVisitor visitor) throws IOException {
      for (int i = 0; i < docs.length; i++) visitor.visit(docs[i], packed[i]);
    }

    @Override
    public int getNumDimensions() throws IOException {
      return like.getNumDimensions();
    }

    @Override
    public int getNumIndexDimensions() throws IOException {
      return like.getNumIndexDimensions();
    }

    @Override
    public int getBytesPerDimension() throws IOException {
      return like.getBytesPerDimension();
    }

    @Override
    public long size() {
      return docs.length;
    }

    @Override
    public int getDocCount() {
      try {
        return like.getDocCount();
      } catch (Exception e) {
        throw new IllegalStateException(e);
      }
    }
  }

  static String bitString(Bits live) {
    StringBuilder sb = new StringBuilder();
    for (int d = 0; d < live.length(); d++) sb.append(live.get(d) ? '1' : '0');
    return sb.toString();
  }

  static void clean(Path p) throws IOException {
    if (Files.exists(p)) {
      try (var walk = Files.walk(p)) {
        walk.sorted((a, b) -> b.compareTo(a)).forEach(f -> f.toFile().delete());
      }
    }
    Files.createDirectories(p);
  }

  /**
   * {@code codec/}: Rust's {@code .si}/{@code .inf}/{@code .liv}/{@code .scf}/{@code .fld}, read
   * by SimpleTextCodec's own formats and re-written by its own writers, then bound into a commit
   * Java writes and opened through {@link DirectoryReader}.
   */
  static int verifyCodec(Path root) throws Exception {
    Codec codec = new SimpleTextCodec();
    int failures = 0;
    Path javaDir = root.resolve("java");
    clean(javaDir);
    List<String> segRows = Files.readAllLines(root.resolve("segments.tsv"));
    SegmentInfos sis = new SegmentInfos(Version.LATEST.major);
    Map<String, int[]> expected = new TreeMap<>();
    try (Directory dir = FSDirectory.open(root.resolve("index"));
        Directory out = FSDirectory.open(javaDir)) {
      for (String f : dir.listAll()) {
        if (f.startsWith("segments") || f.startsWith("pending_segments")) dir.deleteFile(f);
      }
      for (String row : segRows) {
        String[] r = row.split("\t");
        String name = r[0];
        int maxDoc = Integer.parseInt(r[1]);
        int delCount = Integer.parseInt(r[2]);
        long delGen = Long.parseLong(r[3]);
        byte[] id = HEX.parseHex(r[4]);
        expected.put(name, new int[] {maxDoc, delCount});

        SegmentInfo si = codec.segmentInfoFormat().read(dir, name, id, CTX);
        failures += expect(root.resolve("si." + name + ".tsv"), segmentInfoLines(si));
        si.setCodec(codec);
        codec.segmentInfoFormat().write(out, si, CTX);
        // Diagnostics, attributes and files are hash maps/sets in Java: same lines, any order.
        if (!sortedLines(readAll(dir, name + ".si")).equals(sortedLines(readAll(out, name + ".si")))) {
          System.out.println(name + ".si: Java's re-write has different lines");
          failures++;
        }

        CompoundDirectory cfs =
            si.getUseCompoundFile() ? codec.compoundFormat().getCompoundReader(dir, si) : null;
        Directory core = cfs == null ? dir : cfs;
        if (cfs != null) {
          List<String> lines = new ArrayList<>();
          Path subPath = root.resolve("java-cfs-" + name);
          clean(subPath);
          try (Directory sub = FSDirectory.open(subPath)) {
            for (String f : cfs.listAll()) {
              lines.add(f + "\t" + cfs.fileLength(f));
              sub.copyFrom(cfs, f, f, CTX);
            }
            failures += expect(root.resolve("cfs." + name + ".tsv"), lines);
            SegmentInfo w = new SegmentInfo(sub, si.getVersion(), si.getMinVersion(), name, maxDoc,
                false, false, codec, Map.of(), id, Map.of(), null);
            w.setFiles(Arrays.asList(cfs.listAll()));
            codec.compoundFormat().write(sub, w, CTX);
            failures += sameBytes(dir, name + ".scf", sub, name + ".scf");
          }
        }

        FieldInfos fis = codec.fieldInfosFormat().read(core, si, "", CTX);
        failures += expect(root.resolve("fields." + name + ".inf.tsv"), fieldInfoLines(fis));
        codec.fieldInfosFormat().write(out, si, "", fis, CTX);
        failures += sameBytes(core, name + ".inf", out, name + ".inf");

        List<String> stored = new ArrayList<>();
        try (StoredFieldsReader reader = codec.storedFieldsFormat().fieldsReader(core, si, fis, CTX);
            StoredFieldsWriter writer = codec.storedFieldsFormat().fieldsWriter(out, si, CTX)) {
          reader.checkIntegrity();
          for (int d = 0; d < maxDoc; d++) {
            stored.add("doc\t" + d);
            writer.startDocument();
            reader.document(d, new Dump(stored, writer));
            writer.finishDocument();
          }
          writer.finish(maxDoc);
        }
        failures += expect(root.resolve("stored." + name + ".tsv"), stored);
        failures += sameBytes(core, name + ".fld", out, name + ".fld");
        failures += verifyDocValues(root, core, out, si, fis, name);
        if (fis.hasNorms()) failures += verifyIndexed(root, core, out, si, fis, name);
        if (fis.hasPointValues()) failures += verifyPoints(root, core, out, si, fis, name);
        if (fis.hasVectorValues()) failures += verifyKnn(root, core, out, si, fis, name);

        SegmentCommitInfo sci =
            new SegmentCommitInfo(si, delCount, 0, delGen, -1, -1, StringHelper.randomId());
        if (delGen != -1) {
          Bits live = codec.liveDocsFormat().readLiveDocs(dir, sci, CTX);
          failures += expect(root.resolve("live." + name + ".tsv"),
              List.of(Integer.toString(live.length()), bitString(live)));
          codec.liveDocsFormat().writeLiveDocs(live, out, sci, 0, CTX);
          failures += sameBytes(dir, IndexFileNames.fileNameFromGeneration(name, "liv", delGen),
              out, IndexFileNames.fileNameFromGeneration(name, "liv", sci.getNextDelGen()));
        }
        if (cfs != null) cfs.close();
        sis.add(sci);
      }
      sis.counter = segRows.size();
      sis.commit(dir);

      // The whole index, opened the way a searcher would.
      try (DirectoryReader reader = DirectoryReader.open(dir)) {
        if (reader.leaves().size() != segRows.size()) {
          System.out.println("DirectoryReader: " + reader.leaves().size() + " leaves");
          failures++;
        }
        for (LeafReaderContext ctx : reader.leaves()) {
          SegmentReader sr = (SegmentReader) ctx.reader();
          String name = sr.getSegmentName();
          int[] e = expected.get(name);
          if (sr.maxDoc() != e[0] || sr.numDocs() != e[0] - e[1]) {
            System.out.println(name + ": maxDoc " + sr.maxDoc() + " numDocs " + sr.numDocs());
            failures++;
          }
          Bits live = sr.getLiveDocs();
          if (e[1] > 0) {
            List<String> want = Files.readAllLines(root.resolve("live." + name + ".tsv"));
            if (live == null || !bitString(live).equals(want.get(1))) {
              System.out.println(name + ": DirectoryReader's live docs differ");
              failures++;
            }
          } else if (live != null) {
            System.out.println(name + ": unexpected live docs");
            failures++;
          }
          List<String> stored = new ArrayList<>();
          StoredFields sf = sr.storedFields();
          for (int d = 0; d < sr.maxDoc(); d++) {
            stored.add("doc\t" + d);
            sf.document(d, new Dump(stored, null));
          }
          failures += expect(root.resolve("stored." + name + ".tsv"), stored);
          failures += knnThroughReader(sr, name);
          for (FieldInfo fi : sr.getFieldInfos()) {
            if (fi.getPointDimensionCount() == 0) continue;
            PointValues pv = sr.getPointValues(fi.name);
            long[] seen = new long[1];
            pv.intersect(
                new PointValues.IntersectVisitor() {
                  @Override
                  public void visit(int doc) {
                    seen[0]++;
                  }

                  @Override
                  public void visit(int doc, byte[] packed) {
                    seen[0]++;
                  }

                  @Override
                  public PointValues.Relation compare(byte[] min, byte[] max) {
                    return PointValues.Relation.CELL_CROSSES_QUERY;
                  }
                });
            String head = pointsHead(fi.name, pv);
            if (!head.equals(POINT_HEADS.get(name + "/" + fi.name)) || seen[0] != pv.size()) {
              System.out.println(name + "/" + fi.name + ": DirectoryReader's points differ: " + head);
              failures++;
            }
          }
        }
        System.out.println("codec: " + reader.leaves().size() + " segments, " + reader.maxDoc()
            + " docs (" + reader.numDocs() + " live) opened through DirectoryReader");
      }
      ByteArrayOutputStream log = new ByteArrayOutputStream();
      try (CheckIndex ci = new CheckIndex(dir)) {
        ci.setLevel(CheckIndex.Level.MIN_LEVEL_FOR_SLOW_CHECKS);
        ci.setInfoStream(new PrintStream(log, true, StandardCharsets.UTF_8));
        CheckIndex.Status status = ci.checkIndex();
        if (!status.clean) {
          System.out.println(log.toString(StandardCharsets.UTF_8));
          System.out.println("codec: CheckIndex found problems");
          failures++;
        } else {
          System.out.println("codec: CheckIndex clean (slow checks)");
        }
      }
    }

    // A field-infos file holding every kind of field, outside the index.
    try (Directory d2 = FSDirectory.open(root.resolve("infos"));
        Directory o2 = FSDirectory.open(javaDir)) {
      SegmentInfo si2 = new SegmentInfo(d2, Version.LATEST, Version.LATEST, "_2", 1, false, false,
          codec, Map.of(), new byte[16], Map.of(), null);
      FieldInfos fis = codec.fieldInfosFormat().read(d2, si2, "", CTX);
      failures += expect(root.resolve("fields._2.inf.tsv"), fieldInfoLines(fis));
      codec.fieldInfosFormat().write(o2, si2, "", fis, CTX);
      failures += sameBytes(d2, "_2.inf", o2, "_2.inf");
      System.out.println("codec: _2.inf, " + fis.size() + " fields of every kind");
    }
    return failures;
  }
}
