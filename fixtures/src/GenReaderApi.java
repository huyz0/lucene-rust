import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.Tokenizer;
import org.apache.lucene.analysis.core.WhitespaceTokenizer;
import org.apache.lucene.analysis.miscellaneous.PerFieldAnalyzerWrapper;
import org.apache.lucene.analysis.payloads.DelimitedPayloadTokenFilter;
import org.apache.lucene.analysis.payloads.IdentityEncoder;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
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
import org.apache.lucene.index.CodecReader;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.DocValuesType;
import org.apache.lucene.index.ExitableDirectoryReader;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.Fields;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.Impacts;
import org.apache.lucene.index.ImpactsEnum;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.MultiDocValues;
import org.apache.lucene.index.MultiReader;
import org.apache.lucene.index.MultiTerms;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.NumericDocValues;
import org.apache.lucene.index.ParallelCompositeReader;
import org.apache.lucene.index.ParallelLeafReader;
import org.apache.lucene.index.PointValues;
import org.apache.lucene.index.PostingsEnum;
import org.apache.lucene.index.QueryTimeout;
import org.apache.lucene.index.ReaderApiAccess;
import org.apache.lucene.index.SlowCodecReaderWrapper;
import org.apache.lucene.index.SortedDocValues;
import org.apache.lucene.index.SortedNumericDocValues;
import org.apache.lucene.index.SortedSetDocValues;
import org.apache.lucene.index.SortingCodecReader;
import org.apache.lucene.index.StoredFieldVisitor;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.Terms;
import org.apache.lucene.index.TermsEnum;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.Sort;
import org.apache.lucene.search.SortField;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.BytesRef;
import org.apache.lucene.util.Version;
import org.apache.lucene.util.automaton.CompiledAutomaton;
import org.apache.lucene.util.automaton.Operations;
import org.apache.lucene.util.automaton.RegExp;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

/**
 * {@code reader_api/}: ground truth for {@code crates/lucene-search/src/reader/} -- Lucene's
 * reader API over segments and over the views built from them.
 *
 * <p>Four indexes: {@code multi} (three segments, every kind of field: positions, offsets,
 * payloads, the five doc-values types, points, float and byte vectors, term vectors, stored
 * fields; a deletion in two segments), {@code par_a}/{@code par_b} (two indexes holding the same
 * documents with different fields, for the parallel readers), and {@code exitable} (one segment
 * of 2500 documents, for {@code ExitableDirectoryReader}'s check sampling, plus a single-valued
 * sparse sorted-set field and a text field on a third of the documents, whose norms are sparse).
 *
 * <p>Every leaf-shaped reader is written out by one {@link #dump} -- the {@code
 * SegmentReader}s, {@code SlowCompositeCodecReaderWrapper}, two {@code SortingCodecReader}s,
 * {@code ParallelCompositeReader}'s leaves, a {@code ParallelLeafReader} with separate
 * stored-fields readers and {@code SlowCodecReaderWrapper} -- and the Rust test writes the same
 * lines from its port of each, so a key names one fact about one view. Composite-level keys
 * cover {@code MultiDocValues}, {@code MultiTerms} (positions, offsets, payloads, {@code
 * intersect}, {@code impacts}) and {@code MultiReader}.
 */
public class GenReaderApi {
  static final String[] BODIES = {
    "the quick brown fox", "quick quick fox jumps", "lazy dog sleeps", "the fox and the dog",
    "quiet quilt quota", "brown bear fox", "zebra zone", "fox fox fox quick"
  };
  static final String[] KWS = {"kilo", "alpha", "mike", "bravo", "alpha", "zulu", "echo"};
  static final String[][] TAGS = {{"red", "blue"}, {"green"}, {}, {"blue", "amber", "red"}};

  static Analyzer analyzer() {
    Analyzer payloads =
        new Analyzer() {
          @Override
          protected TokenStreamComponents createComponents(String fieldName) {
            Tokenizer t = new WhitespaceTokenizer();
            return new TokenStreamComponents(
                t, new DelimitedPayloadTokenFilter(t, '|', new IdentityEncoder()));
          }
        };
    return new PerFieldAnalyzerWrapper(new StandardAnalyzer(), Map.of("pay", payloads));
  }

  static Document multiDoc(int i) {
    Document d = new Document();
    d.add(new StringField("id", "d" + i, Field.Store.YES));
    FieldType body = new FieldType(TextField.TYPE_NOT_STORED);
    body.setIndexOptions(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS_AND_OFFSETS);
    body.setStoreTermVectors(true);
    body.setStoreTermVectorPositions(true);
    body.setStoreTermVectorOffsets(true);
    body.freeze();
    d.add(new Field("body", BODIES[i % BODIES.length], body));
    FieldType pay = new FieldType(TextField.TYPE_NOT_STORED);
    pay.setStoreTermVectors(true);
    pay.setStoreTermVectorPositions(true);
    pay.setStoreTermVectorPayloads(true);
    pay.freeze();
    d.add(new Field("pay", "a|" + i + " b|x" + (i % 3) + " a|z", pay));
    if (i % 7 != 3) d.add(new NumericDocValuesField("rank", (i * 7) % 5));
    String kw = KWS[i % KWS.length];
    d.add(new StringField("kw", kw, Field.Store.NO));
    d.add(new SortedDocValuesField("kw", new BytesRef(kw)));
    for (String tag : TAGS[i % TAGS.length]) d.add(new SortedSetDocValuesField("tags", new BytesRef(tag)));
    if (i % 4 != 2) {
      d.add(new SortedNumericDocValuesField("nums", i * 3L - 10));
      if (i % 2 == 0) d.add(new SortedNumericDocValuesField("nums", -i));
    }
    if (i % 2 == 0) d.add(new BinaryDocValuesField("bin", new BytesRef(new byte[] {(byte) i, (byte) (255 - i), 7})));
    d.add(new IntPoint("pt", (i * 3) % 11, i - 5));
    if (i % 3 != 1) d.add(new KnnFloatVectorField("vec", new float[] {i, 1.5f * i, -i}));
    if (i % 2 == 0) d.add(new KnnByteVectorField("bvec", new byte[] {(byte) i, (byte) -i}));
    d.add(new StoredField("sint", i));
    d.add(new StoredField("sdouble", i * 1.5));
    d.add(new StoredField("sbin", new byte[] {(byte) i, 1, 2}));
    return d;
  }

  static IndexWriterConfig cfg() {
    IndexWriterConfig c = new IndexWriterConfig(analyzer());
    c.setMergePolicy(NoMergePolicy.INSTANCE);
    c.setUseCompoundFile(false);
    return c;
  }

  static Path fresh(Path dir) throws Exception {
    if (Files.exists(dir)) {
      try (var files = Files.list(dir)) {
        for (Path f : (Iterable<Path>) files::iterator) Files.delete(f);
      }
    }
    Files.createDirectories(dir);
    return dir;
  }

  public static void main(String[] args) throws Exception {
    Path out = Path.of(args[0]).resolve("reader_api");
    Files.createDirectories(out);
    StringBuilder sb = new StringBuilder();
    sb.append("# Generated by fixtures/src/GenReaderApi.java against Lucene ")
        .append(Version.LATEST)
        .append('\n');

    // --- multi: three segments, deletions in the first and the last ---------------------------
    try (Directory d = FSDirectory.open(fresh(out.resolve("multi")))) {
      try (IndexWriter w = new IndexWriter(d, cfg())) {
        int i = 0;
        for (int size : new int[] {6, 5, 4}) {
          for (int k = 0; k < size; k++) w.addDocument(multiDoc(i++));
          w.commit();
        }
        w.deleteDocuments(new Term("id", "d4"), new Term("id", "d12"));
        w.commit();
      }
      try (DirectoryReader r = DirectoryReader.open(d)) {
        List<LeafReaderContext> leaves = r.leaves();
        List<CodecReader> codecs = new ArrayList<>();
        for (LeafReaderContext l : leaves) {
          dump(l.reader(), "seg" + l.ord, sb);
          codecs.add((CodecReader) l.reader());
        }
        dump(ReaderApiAccess.slowComposite(codecs), "slow", sb);
        dump(SortingCodecReader.wrap(codecs.get(0), new Sort(new SortField("rank", SortField.Type.LONG))), "sorted", sb);
        dump(
            SortingCodecReader.wrap(codecs.get(1), new Sort(new SortField("kw", SortField.Type.STRING, true))),
            "sorted_kw",
            sb);
        composite(r, "multi", sb);
      }
    }

    // --- par_a / par_b: the same documents, different fields ----------------------------------
    try (Directory a = FSDirectory.open(fresh(out.resolve("par_a")));
        Directory b = FSDirectory.open(fresh(out.resolve("par_b")))) {
      try (IndexWriter wa = new IndexWriter(a, cfg());
          IndexWriter wb = new IndexWriter(b, cfg())) {
        int i = 0;
        for (int size : new int[] {3, 2}) {
          for (int k = 0; k < size; k++, i++) {
            Document da = new Document();
            da.add(new StringField("id", "p" + i, Field.Store.YES));
            da.add(new TextField("title", "title " + BODIES[i], Field.Store.NO));
            da.add(new NumericDocValuesField("na", 100 - i));
            wa.addDocument(da);
            Document db = new Document();
            FieldType tv = new FieldType(TextField.TYPE_STORED);
            tv.setStoreTermVectors(true);
            tv.setStoreTermVectorPositions(true);
            tv.freeze();
            db.add(new Field("body2", BODIES[(i + 3) % BODIES.length], tv));
            db.add(new StoredField("extra", i * 11));
            db.add(new SortedDocValuesField("nb", new BytesRef("v" + (i % 2))));
            // A field par_a also has: the parallel reader reads it from par_a.
            db.add(new StringField("id", "shadow" + i, Field.Store.YES));
            wb.addDocument(db);
          }
          wa.commit();
          wb.commit();
        }
        wa.deleteDocuments(new Term("id", "p1"));
        wa.commit();
      }
      try (DirectoryReader ra = DirectoryReader.open(a);
          DirectoryReader rb = DirectoryReader.open(b)) {
        ParallelCompositeReader pc = new ParallelCompositeReader(false, new DirectoryReader[] {ra, rb}, new DirectoryReader[] {ra, rb});
        for (LeafReaderContext l : pc.leaves()) dump(l.reader(), "par" + l.ord, sb);
        line(sb, "par.max_doc", pc.maxDoc());
        line(sb, "par.num_docs", pc.numDocs());
        LeafReader a0 = ra.leaves().get(0).reader();
        LeafReader b0 = rb.leaves().get(0).reader();
        ParallelLeafReader stored = new ParallelLeafReader(false, new LeafReader[] {a0, b0}, new LeafReader[] {b0});
        dump(stored, "parstored", sb);
        dump(SlowCodecReaderWrapper.wrap(new ParallelLeafReader(false, new LeafReader[] {b0, a0}, new LeafReader[] {b0, a0})), "slowpar", sb);

        try (Directory dm = FSDirectory.open(out.resolve("multi"));
            DirectoryReader rm = DirectoryReader.open(dm)) {
          MultiReader mr = new MultiReader(new IndexReader[] {rm, ra}, false);
          StringBuilder ls = new StringBuilder();
          for (LeafReaderContext l : mr.leaves()) {
            if (ls.length() > 0) ls.append(',');
            ls.append(l.docBase).append(':').append(l.reader().maxDoc());
          }
          line(sb, "mr.leaves", ls);
          line(sb, "mr.max_doc", mr.maxDoc());
          line(sb, "mr.num_docs", mr.numDocs());
          line(sb, "mr.doc_freq.body.fox", mr.docFreq(new Term("body", "fox")));
          line(sb, "mr.doc_freq.id.p2", mr.docFreq(new Term("id", "p2")));
          line(sb, "mr.sum_doc_freq.id", mr.getSumDocFreq("id"));
          line(sb, "mr.doc_count.id", mr.getDocCount("id"));
          line(sb, "mr.sum_ttf.body", mr.getSumTotalTermFreq("body"));
          line(sb, "mr.ttf.body.fox", mr.totalTermFreq(new Term("body", "fox")));
          line(sb, "mr.fields", fieldList(FieldInfos.getMergedFieldInfos(mr)));
          for (int doc : new int[] {0, 14, 15, 19}) {
            line(sb, "mr.stored." + doc, stored(mr.storedFields(), doc, null));
          }
        }
      }
    }

    // --- exitable: where a counting timeout stops each enumeration ----------------------------
    try (Directory d = FSDirectory.open(fresh(out.resolve("exitable")))) {
      IndexWriterConfig c = cfg();
      c.setMaxBufferedDocs(10000);
      c.setRAMBufferSizeMB(256);
      try (IndexWriter w = new IndexWriter(d, c)) {
        for (int i = 0; i < 2500; i++) {
          Document doc = new Document();
          String t = String.format("t%02d", i % 40);
          doc.add(new StringField("t", t, Field.Store.NO));
          doc.add(new NumericDocValuesField("n", i));
          doc.add(new SortedDocValuesField("s", new BytesRef(t)));
          doc.add(new BinaryDocValuesField("b", new BytesRef(t)));
          doc.add(new SortedNumericDocValuesField("sn", i));
          doc.add(new SortedSetDocValuesField("ss", new BytesRef(t)));
          doc.add(new SortedSetDocValuesField("ss", new BytesRef("x" + t)));
          // Single-valued and sparse: the SORTED_SET shape stored as SORTED
          // ordinals, and a text field whose norms are sparse.
          if (i % 2 == 0) doc.add(new SortedSetDocValuesField("ss1", new BytesRef(t)));
          if (i % 3 == 0) doc.add(new TextField("txt", t + " words here", Field.Store.NO));
          w.addDocument(doc);
        }
        w.commit();
      }
      try (DirectoryReader base = DirectoryReader.open(d)) {
        for (int exitAt = 1; exitAt <= 4; exitAt++) {
          String p = "exit." + exitAt + ".";
          line(sb, p + "terms", exitTerms(base, exitAt));
          line(sb, p + "numeric_next", exitDv(base, exitAt, "numeric_next"));
          line(sb, p + "sorted_exact", exitDv(base, exitAt, "sorted_exact"));
          line(sb, p + "binary_advance", exitDv(base, exitAt, "binary_advance"));
          line(sb, p + "sorted_numeric_advance", exitDv(base, exitAt, "sorted_numeric_advance"));
          line(sb, p + "sorted_set_next", exitDv(base, exitAt, "sorted_set_next"));
        }
      }
    }

    Files.writeString(out.resolve("manifest.properties"), sb.toString(), StandardCharsets.UTF_8);
  }

  /** Exits on the {@code exitAt}-th {@code shouldExit} call. */
  static final class Countdown implements QueryTimeout {
    final int exitAt;
    int calls;

    Countdown(int exitAt) {
      this.exitAt = exitAt;
    }

    @Override
    public boolean shouldExit() {
      return ++calls >= exitAt;
    }
  }

  /** How many terms {@code next()} returned before the exit, or {@code end:N}. */
  static String exitTerms(DirectoryReader base, int exitAt) throws Exception {
    DirectoryReader r = ExitableDirectoryReader.wrap(base, new Countdown(exitAt));
    Terms terms = r.leaves().get(0).reader().terms("t");
    int n = 0;
    try {
      TermsEnum te = terms.iterator();
      while (te.next() != null) n++;
      return "end:" + n;
    } catch (ExitableDirectoryReader.ExitingReaderException e) {
      return "exit:" + n;
    }
  }

  /** The last document an iteration reached before the exit, or {@code end:N}. */
  static String exitDv(DirectoryReader base, int exitAt, String how) throws Exception {
    DirectoryReader r = ExitableDirectoryReader.wrap(base, new Countdown(exitAt));
    LeafReader leaf = r.leaves().get(0).reader();
    int reached = -1;
    int steps = 0;
    try {
      switch (how) {
        case "numeric_next": {
          NumericDocValues v = leaf.getNumericDocValues("n");
          for (int doc = v.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS; doc = v.nextDoc()) {
            reached = doc;
            steps++;
          }
          break;
        }
        case "sorted_exact": {
          SortedDocValues v = leaf.getSortedDocValues("s");
          for (int doc = 0; doc < leaf.maxDoc(); doc += 3) {
            v.advanceExact(doc);
            reached = doc;
            steps++;
          }
          break;
        }
        case "binary_advance": {
          BinaryDocValues v = leaf.getBinaryDocValues("b");
          for (int target = 0; target < leaf.maxDoc(); target += 450) {
            int doc = v.advance(target);
            reached = doc;
            steps++;
          }
          break;
        }
        case "sorted_numeric_advance": {
          SortedNumericDocValues v = leaf.getSortedNumericDocValues("sn");
          for (int target = 0; target < leaf.maxDoc(); target += 450) {
            int doc = v.advance(target);
            reached = doc;
            steps++;
          }
          break;
        }
        case "sorted_set_next": {
          SortedSetDocValues v = leaf.getSortedSetDocValues("ss");
          for (int doc = v.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS; doc = v.nextDoc()) {
            reached = doc;
            steps++;
          }
          break;
        }
        default:
          throw new IllegalArgumentException(how);
      }
      return "end:" + steps + ":" + reached;
    } catch (ExitableDirectoryReader.ExitingReaderException e) {
      return "exit:" + steps + ":" + reached;
    }
  }

  // --- the leaf dump ----------------------------------------------------------------------------

  static void line(StringBuilder sb, String key, Object value) {
    sb.append(key).append('=').append(value).append('\n');
  }

  static String hex(byte[] b, int off, int len) {
    StringBuilder s = new StringBuilder();
    for (int i = off; i < off + len; i++) s.append(String.format("%02x", b[i] & 0xff));
    return s.toString();
  }

  static String hex(BytesRef b) {
    return b == null ? "" : hex(b.bytes, b.offset, b.length);
  }

  static String fieldList(FieldInfos infos) {
    StringBuilder s = new StringBuilder();
    for (FieldInfo fi : infos) {
      if (s.length() > 0) s.append(',');
      s.append(fi.name).append(':').append(fi.number);
    }
    return s.toString();
  }

  static String postings(PostingsEnum pe, boolean positions) throws Exception {
    StringBuilder s = new StringBuilder();
    for (int doc = pe.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS; doc = pe.nextDoc()) {
      if (s.length() > 0) s.append(' ');
      s.append(doc).append('/').append(pe.freq());
      if (positions) {
        s.append('[');
        for (int k = 0; k < pe.freq(); k++) {
          if (k > 0) s.append(' ');
          s.append(pe.nextPosition())
              .append('@')
              .append(pe.startOffset())
              .append('-')
              .append(pe.endOffset())
              .append('#')
              .append(hex(pe.getPayload()));
        }
        s.append(']');
      }
    }
    return s.toString();
  }

  static String termsDump(Terms t, boolean withPostings) throws Exception {
    StringBuilder s = new StringBuilder();
    TermsEnum te = t.iterator();
    boolean pos = t.hasPositions();
    for (BytesRef term = te.next(); term != null; term = te.next()) {
      if (s.length() > 0) s.append(';');
      s.append(term.utf8ToString()).append(':').append(te.docFreq()).append(':').append(te.totalTermFreq());
      if (withPostings) {
        s.append(':').append(postings(te.postings(null, pos ? PostingsEnum.ALL : PostingsEnum.FREQS), pos));
      }
    }
    return s.toString();
  }

  static String termsStats(Terms t) throws Exception {
    return t.size() + "|" + t.getSumTotalTermFreq() + "|" + t.getSumDocFreq() + "|" + t.getDocCount()
        + "|" + (t.getMin() == null ? "" : t.getMin().utf8ToString())
        + "|" + (t.getMax() == null ? "" : t.getMax().utf8ToString())
        + "|" + (t.hasFreqs() ? 1 : 0) + (t.hasPositions() ? 1 : 0) + (t.hasOffsets() ? 1 : 0)
        + (t.hasPayloads() ? 1 : 0);
  }

  static String numeric(NumericDocValues v) throws Exception {
    StringBuilder s = new StringBuilder();
    for (int doc = v.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS; doc = v.nextDoc()) {
      if (s.length() > 0) s.append(',');
      s.append(doc).append(':').append(v.longValue());
    }
    return s.toString();
  }

  static String binary(BinaryDocValues v) throws Exception {
    StringBuilder s = new StringBuilder();
    for (int doc = v.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS; doc = v.nextDoc()) {
      if (s.length() > 0) s.append(',');
      s.append(doc).append(':').append(hex(v.binaryValue()));
    }
    return s.toString();
  }

  static String sorted(SortedDocValues v) throws Exception {
    StringBuilder s = new StringBuilder();
    for (int doc = v.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS; doc = v.nextDoc()) {
      if (s.length() > 0) s.append(',');
      s.append(doc).append(':').append(v.ordValue());
    }
    s.append('|').append(v.getValueCount()).append(':');
    for (int o = 0; o < v.getValueCount(); o++) {
      if (o > 0) s.append('/');
      s.append(v.lookupOrd(o).utf8ToString());
    }
    return s.toString();
  }

  static String sortedNumeric(SortedNumericDocValues v) throws Exception {
    StringBuilder s = new StringBuilder();
    for (int doc = v.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS; doc = v.nextDoc()) {
      if (s.length() > 0) s.append(',');
      s.append(doc).append(":[");
      for (int k = 0; k < v.docValueCount(); k++) {
        if (k > 0) s.append('|');
        s.append(v.nextValue());
      }
      s.append(']');
    }
    return s.toString();
  }

  static String sortedSet(SortedSetDocValues v) throws Exception {
    StringBuilder s = new StringBuilder();
    for (int doc = v.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS; doc = v.nextDoc()) {
      if (s.length() > 0) s.append(',');
      s.append(doc).append(":[");
      for (int k = 0; k < v.docValueCount(); k++) {
        if (k > 0) s.append('|');
        s.append(v.nextOrd());
      }
      s.append(']');
    }
    s.append('|').append(v.getValueCount()).append(':');
    for (long o = 0; o < v.getValueCount(); o++) {
      if (o > 0) s.append('/');
      s.append(v.lookupOrd(o).utf8ToString());
    }
    return s.toString();
  }

  static String stored(org.apache.lucene.index.StoredFields sf, int doc, FieldInfos unused) throws Exception {
    List<String> parts = new ArrayList<>();
    sf.document(
        doc,
        new StoredFieldVisitor() {
          @Override
          public Status needsField(FieldInfo fieldInfo) {
            return Status.YES;
          }

          @Override
          public void stringField(FieldInfo fi, String value) {
            parts.add(fi.name + "=s:" + value);
          }

          @Override
          public void binaryField(FieldInfo fi, byte[] value) {
            parts.add(fi.name + "=b:" + hex(value, 0, value.length));
          }

          @Override
          public void intField(FieldInfo fi, int value) {
            parts.add(fi.name + "=i:" + value);
          }

          @Override
          public void longField(FieldInfo fi, long value) {
            parts.add(fi.name + "=l:" + value);
          }

          @Override
          public void floatField(FieldInfo fi, float value) {
            parts.add(fi.name + "=f:" + Integer.toHexString(Float.floatToIntBits(value)));
          }

          @Override
          public void doubleField(FieldInfo fi, double value) {
            parts.add(fi.name + "=d:" + Long.toHexString(Double.doubleToLongBits(value)));
          }
        });
    return String.join(";", parts);
  }

  static String termVectors(Fields f) throws Exception {
    if (f == null) return "null";
    TreeMap<String, String> byName = new TreeMap<>();
    for (String name : f) {
      Terms t = f.terms(name);
      if (t == null) continue;
      StringBuilder s = new StringBuilder();
      TermsEnum te = t.iterator();
      for (BytesRef term = te.next(); term != null; term = te.next()) {
        if (s.length() > 0) s.append(' ');
        PostingsEnum pe = te.postings(null, PostingsEnum.ALL);
        pe.nextDoc();
        s.append(term.utf8ToString()).append('/').append(pe.freq()).append('[');
        for (int k = 0; k < pe.freq(); k++) {
          if (k > 0) s.append(' ');
          int p = pe.nextPosition();
          s.append(t.hasPositions() ? p : -1)
              .append('@')
              .append(t.hasOffsets() ? pe.startOffset() : -1)
              .append('-')
              .append(t.hasOffsets() ? pe.endOffset() : -1)
              .append('#')
              .append(t.hasPayloads() ? hex(pe.getPayload()) : "");
        }
        s.append(']');
      }
      byName.put(name, name + "{" + s + "}");
    }
    return String.join(";", byName.values());
  }

  /** Every point of {@code pv}: {@code doc:hex}, sorted. */
  static String points(PointValues pv) throws Exception {
    List<String> seen = new ArrayList<>();
    pv.intersect(
        new PointValues.IntersectVisitor() {
          @Override
          public void visit(int docID) {
            throw new IllegalStateException("inside cells are not asked for");
          }

          @Override
          public void visit(int docID, byte[] packedValue) {
            seen.add(docID + ":" + hex(packedValue, 0, packedValue.length));
          }

          @Override
          public PointValues.Relation compare(byte[] min, byte[] max) {
            return PointValues.Relation.CELL_CROSSES_QUERY;
          }
        });
    java.util.Collections.sort(seen);
    return pv.getNumDimensions() + "|" + pv.getNumIndexDimensions() + "|" + pv.getBytesPerDimension()
        + "|" + pv.size() + "|" + pv.getDocCount() + "|" + hex(pv.getMinPackedValue(), 0, pv.getMinPackedValue().length)
        + "|" + hex(pv.getMaxPackedValue(), 0, pv.getMaxPackedValue().length) + "|" + String.join(",", seen);
  }

  static void dump(LeafReader r, String p, StringBuilder sb) throws Exception {
    line(sb, p + ".max_doc", r.maxDoc());
    line(sb, p + ".num_docs", r.numDocs());
    Bits live = r.getLiveDocs();
    StringBuilder del = new StringBuilder();
    if (live == null) {
      del.append("none");
    } else {
      for (int d = 0; d < r.maxDoc(); d++) {
        if (!live.get(d)) {
          if (del.length() > 0) del.append(',');
          del.append(d);
        }
      }
    }
    line(sb, p + ".deleted", del);
    line(sb, p + ".fields", fieldList(r.getFieldInfos()));
    line(sb, p + ".sorted", r.getMetaData().sort() != null);
    TreeMap<String, FieldInfo> byName = new TreeMap<>();
    for (FieldInfo fi : r.getFieldInfos()) byName.put(fi.name, fi);
    for (FieldInfo fi : byName.values()) {
      String f = fi.name;
      if (fi.getIndexOptions() != IndexOptions.NONE) {
        Terms t = r.terms(f);
        if (t != null) {
          line(sb, p + ".terms." + f, termsStats(t));
          line(sb, p + ".postings." + f, termsDump(t, true));
        }
      }
      if (fi.hasNorms()) {
        NumericDocValues n = r.getNormValues(f);
        if (n != null) line(sb, p + ".norms." + f, numeric(n));
      }
      DocValuesType dv = fi.getDocValuesType();
      if (dv == DocValuesType.NUMERIC) line(sb, p + ".dv." + f, numeric(r.getNumericDocValues(f)));
      if (dv == DocValuesType.BINARY) line(sb, p + ".dv." + f, binary(r.getBinaryDocValues(f)));
      if (dv == DocValuesType.SORTED) line(sb, p + ".dv." + f, sorted(r.getSortedDocValues(f)));
      if (dv == DocValuesType.SORTED_NUMERIC) line(sb, p + ".dv." + f, sortedNumeric(r.getSortedNumericDocValues(f)));
      if (dv == DocValuesType.SORTED_SET) line(sb, p + ".dv." + f, sortedSet(r.getSortedSetDocValues(f)));
      if (fi.getPointDimensionCount() > 0) {
        PointValues pv = r.getPointValues(f);
        if (pv != null) line(sb, p + ".points." + f, points(pv));
      }
      if (fi.getVectorDimension() > 0) {
        StringBuilder s = new StringBuilder();
        if (fi.getVectorEncoding() == org.apache.lucene.index.VectorEncoding.FLOAT32) {
          FloatVectorValues v = r.getFloatVectorValues(f);
          KnnVectorValues.DocIndexIterator it = v.iterator();
          for (int doc = it.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS; doc = it.nextDoc()) {
            if (s.length() > 0) s.append(',');
            s.append(doc).append(':');
            float[] x = v.vectorValue(it.index());
            for (int k = 0; k < x.length; k++) {
              if (k > 0) s.append('/');
              s.append(Integer.toHexString(Float.floatToIntBits(x[k])));
            }
          }
        } else {
          ByteVectorValues v = r.getByteVectorValues(f);
          KnnVectorValues.DocIndexIterator it = v.iterator();
          for (int doc = it.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS; doc = it.nextDoc()) {
            if (s.length() > 0) s.append(',');
            byte[] x = v.vectorValue(it.index());
            s.append(doc).append(':').append(hex(x, 0, x.length));
          }
        }
        line(sb, p + ".vectors." + f, s);
      }
    }
    for (int doc = 0; doc < r.maxDoc(); doc++) {
      line(sb, p + ".stored." + doc, stored(r.storedFields(), doc, null));
      line(sb, p + ".tv." + doc, termVectors(r.termVectors().get(doc)));
    }
  }

  /** Composite-level views: merged field infos, MultiDocValues, MultiTerms. */
  static void composite(IndexReader r, String p, StringBuilder sb) throws Exception {
    line(sb, p + ".fields", fieldList(FieldInfos.getMergedFieldInfos(r)));
    line(sb, p + ".mdv.rank", numeric(MultiDocValues.getNumericValues(r, "rank")));
    line(sb, p + ".mdv.bin", binary(MultiDocValues.getBinaryValues(r, "bin")));
    line(sb, p + ".mdv.kw", sorted(MultiDocValues.getSortedValues(r, "kw")));
    line(sb, p + ".mdv.nums", sortedNumeric(MultiDocValues.getSortedNumericValues(r, "nums")));
    line(sb, p + ".mdv.tags", sortedSet(MultiDocValues.getSortedSetValues(r, "tags")));
    line(sb, p + ".mnorms.body", numeric(MultiDocValues.getNormValues(r, "body")));
    for (String f : new String[] {"body", "pay", "kw"}) {
      Terms t = MultiTerms.getTerms(r, f);
      line(sb, p + ".mterms." + f, termsStats(t));
      line(sb, p + ".mpostings." + f, termsDump(t, true));
    }
    Terms body = MultiTerms.getTerms(r, "body");
    CompiledAutomaton ca =
        new CompiledAutomaton(
            Operations.determinize(new RegExp("f.*|qu[a-z]*|b.*").toAutomaton(), Operations.DEFAULT_DETERMINIZE_WORK_LIMIT));
    line(sb, p + ".intersect.body", termsDump(new TermsWrapper(body, ca, null), false));
    line(sb, p + ".intersect_from_fox.body", termsDump(new TermsWrapper(body, ca, new BytesRef("fox")), false));
    TermsEnum te = body.iterator();
    te.seekExact(new BytesRef("fox"));
    ImpactsEnum ie = te.impacts(PostingsEnum.FREQS);
    ie.advanceShallow(0);
    Impacts im = ie.getImpacts();
    line(sb, p + ".impacts.body.fox",
        im.numLevels() + "|" + im.getDocIdUpTo(0) + "|" + im.getImpacts(0).size + "|" + im.getImpacts(0).freqs[0] + "|" + im.getImpacts(0).norms[0]
            + "|" + postings(ie, false));
  }

  /** Terms whose {@code iterator()} is {@code intersect(compiled, start)}, for {@link #termsDump}. */
  static final class TermsWrapper extends org.apache.lucene.index.FilterLeafReader.FilterTerms {
    final CompiledAutomaton ca;
    final BytesRef start;

    TermsWrapper(Terms in, CompiledAutomaton ca, BytesRef start) {
      super(in);
      this.ca = ca;
      this.start = start;
    }

    @Override
    public TermsEnum iterator() throws java.io.IOException {
      return in.intersect(ca, start);
    }
  }
}
