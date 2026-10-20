import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.HexFormat;
import java.util.Random;
import org.apache.lucene.codecs.FieldsConsumer;
import org.apache.lucene.codecs.FieldsProducer;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.PostingsFormat;
import org.apache.lucene.codecs.PostingsReaderBase;
import org.apache.lucene.codecs.PostingsWriterBase;
import org.apache.lucene.codecs.bitvectors.HnswBitVectorsFormat;
import org.apache.lucene.codecs.blockterms.BlockTermsReader;
import org.apache.lucene.codecs.blockterms.BlockTermsWriter;
import org.apache.lucene.codecs.blockterms.FixedGapTermsIndexReader;
import org.apache.lucene.codecs.blockterms.FixedGapTermsIndexWriter;
import org.apache.lucene.codecs.blocktreeords.BlockTreeOrdsPostingsFormat;
import org.apache.lucene.codecs.bloom.BloomFilteringPostingsFormat;
import org.apache.lucene.codecs.lucene104.Lucene104Codec;
import org.apache.lucene.codecs.lucene104.Lucene104PostingsFormat;
import org.apache.lucene.codecs.lucene104.Lucene104PostingsReader;
import org.apache.lucene.codecs.lucene104.Lucene104PostingsWriter;
import org.apache.lucene.codecs.memory.DirectPostingsFormat;
import org.apache.lucene.codecs.memory.FSTPostingsFormat;
import org.apache.lucene.codecs.simpletext.SimpleTextCodec;
import org.apache.lucene.codecs.uniformsplit.UniformSplitPostingsFormat;
import org.apache.lucene.codecs.uniformsplit.sharedterms.STUniformSplitPostingsFormat;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.FieldType;
import org.apache.lucene.document.KnnByteVectorField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.CodecReader;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.PostingsEnum;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.index.SegmentInfos;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.Terms;
import org.apache.lucene.index.TermsEnum;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.AcceptDocs;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TopKnnCollector;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.util.BytesRef;

/**
 * Java side of the lucene-codecs benchmark pair (M14.g); the Rust side is {@code
 * benchmarks/rust-runner/src/micro_codecs.rs}, with the same case names. {@code build <dir>}
 * writes each codec's index once, in its own subdirectory; {@code run <dir>} times every case.
 *
 * <p>{@code bloom/}: 200 000 documents, one segment, whose unique {@code id} is routed to {@code
 * BloomFilteringPostingsFormat} over the default {@code Lucene104PostingsFormat}, and {@code
 * targets.tsv}: 200 000 lookups, a quarter of them present -- the primary-key lookup the format
 * exists for, where most misses are answered by the filter without touching the terms index.
 *
 * <p>{@code memory/}: the same 200 000 ids twice, {@code fid} routed to {@code FSTPostingsFormat}
 * and {@code did} to {@code DirectPostingsFormat}, and the same kind of {@code targets.tsv}:
 * {@code cx_fst_seek} and {@code cx_direct_seek} are {@code seekExact} on each.
 *
 * <p>{@code bitvectors/}: 100 000 documents, each with a 32-byte (256-bit) vector {@code bv} routed
 * to {@code HnswBitVectorsFormat}, and {@code queries.tsv}: 2 000 random queries; {@code
 * cx_bitvec_search} is a top-10 {@code searchNearestVectors} per query.
 *
 * <p>{@code ords/}: the same 200 000 ids, {@code oid} routed to {@code BlockTreeOrdsPostingsFormat},
 * the same kind of {@code targets.tsv} (half present) and {@code ords.tsv}, one random ord per
 * target: {@code cx_ords_seek} is, per target, {@code seekExact(term)} (and its {@code ord()})
 * then {@code seekExact(ord)} (and its term).
 *
 * <p>{@code usplit/}: the same 200 000 ids, {@code uid} routed to {@code
 * UniformSplitPostingsFormat} (default 32/3 block lines, no encoder), and the same kind of {@code
 * targets.tsv} (half present): {@code cx_usplit_seek} is {@code seekExact} on each.
 *
 * <p>{@code stsplit/}: the same 200 000 ids in {@code sa}, every third one also in {@code sb},
 * both routed to one {@code STUniformSplitPostingsFormat} (default 32/3 block lines, one shared
 * dictionary), and the same kind of {@code targets.tsv} (half present in {@code sa}): {@code
 * cx_stsplit_seek} is {@code seekExact} on {@code sb} for each, most hits landing on a line only
 * {@code sa} has.
 *
 * <p>{@code bterms/}: the same 200 000 ids, {@code bid} routed to {@link BlockTermsBenchFormat}
 * ({@code BlockTermsWriter} over the default {@code FixedGapTermsIndexWriter}, interval 32, and
 * {@code Lucene104} postings; Lucene ships no postings format over this package), and the same
 * kind of {@code targets.tsv} (half present): {@code cx_bterms_seek} is {@code seekExact} on each,
 * through a {@code BlockTermsReader} opened by hand over the segment's files.
 *
 * <p>{@code stext/}: 50 000 documents of 1-8 skewed words ({@code w0}..{@code w2999}, freqs) in
 * {@code st}, routed to {@code SimpleTextCodec}'s postings format, and 4 000 {@code targets.tsv}
 * words (about half present): {@code cx_stext_seek} is {@code seekExact} on each, then a hit's
 * postings walked to the end with {@code postings(FREQS)}.
 */
public final class CodecsMicro {
  static long warmupMs = Long.getLong("warmupMs", 1500);
  static long measureMs = Long.getLong("measureMs", 2000);
  static long sink;
  static final int DOCS = 200_000;

  interface Op {
    long run() throws IOException;
  }

  static void measure(String name, Op op) throws IOException {
    loop(op, warmupMs);
    long start = System.nanoTime();
    long units = loop(op, measureMs);
    long elapsed = System.nanoTime() - start;
    System.out.printf("%s\t%.3f\t%d%n", name, (double) elapsed / units, units);
    System.out.flush();
  }

  static long loop(Op op, long budgetMs) throws IOException {
    long budgetNs = budgetMs * 1_000_000L;
    long units = 0;
    long start = System.nanoTime();
    do {
      units += op.run();
    } while (System.nanoTime() - start < budgetNs);
    if (sink == 0xDEADBEEFL) System.err.print("");
    return units;
  }

  static final class Fnv {
    long h = 0xcbf29ce484222325L;

    void add(long x) {
      h ^= x;
      h *= 0x100000001b3L;
    }
  }

  static boolean selected(String name) {
    String only = System.getenv("MICRO_CASE");
    return only == null || only.equals(name);
  }

  /** Document {@code i}'s id: a scrambled base-36 number. */
  static String idOf(long i) {
    return Long.toString((i * 0x9E3779B97F4A7C15L) >>> 24, 36);
  }

  static IndexWriterConfig config(String field, PostingsFormat format) {
    return config(Map.of(field, format));
  }

  static IndexWriterConfig config(Map<String, PostingsFormat> formats) {
    IndexWriterConfig cfg = new IndexWriterConfig();
    cfg.setRAMBufferSizeMB(512);
    cfg.setMaxBufferedDocs(IndexWriterConfig.DISABLE_AUTO_FLUSH);
    cfg.setMergePolicy(NoMergePolicy.INSTANCE);
    cfg.setUseCompoundFile(false);
    cfg.setOpenMode(IndexWriterConfig.OpenMode.CREATE);
    cfg.setCodec(
        new Lucene104Codec() {
          @Override
          public PostingsFormat getPostingsFormatForField(String f) {
            PostingsFormat format = formats.get(f);
            return format != null ? format : super.getPostingsFormatForField(f);
          }
        });
    return cfg;
  }

  static void buildBloom(Path dir) throws IOException {
    Random r = new Random(0x10_5_2026_1020L);
    PostingsFormat bloom = new BloomFilteringPostingsFormat(new Lucene104PostingsFormat());
    try (Directory d = FSDirectory.open(dir);
        IndexWriter w = new IndexWriter(d, config("id", bloom))) {
      for (int i = 0; i < DOCS; i++) {
        Document doc = new Document();
        doc.add(new StringField("id", idOf(i), Field.Store.NO));
        w.addDocument(doc);
      }
      w.commit();
    }
    List<String> targets = new ArrayList<>();
    for (int k = 0; k < 200_000; k++) {
      // A quarter present (ids below DOCS), the rest absent.
      targets.add(idOf(r.nextInt(4) == 0 ? r.nextInt(DOCS) : DOCS + r.nextInt(10 * DOCS)));
    }
    Files.write(dir.resolve("targets.tsv"), targets);
    Files.writeString(dir.resolve("built"), "");
  }

  static void buildMemory(Path dir) throws IOException {
    Random r = new Random(0x14_7_2026_1020L);
    Map<String, PostingsFormat> formats =
        Map.of("fid", new FSTPostingsFormat(), "did", new DirectPostingsFormat());
    try (Directory d = FSDirectory.open(dir);
        IndexWriter w = new IndexWriter(d, config(formats))) {
      for (int i = 0; i < DOCS; i++) {
        Document doc = new Document();
        doc.add(new StringField("fid", idOf(i), Field.Store.NO));
        doc.add(new StringField("did", idOf(i), Field.Store.NO));
        w.addDocument(doc);
      }
      w.commit();
    }
    List<String> targets = new ArrayList<>();
    for (int k = 0; k < 200_000; k++) {
      // Half present, half absent.
      targets.add(idOf(r.nextInt(2) == 0 ? r.nextInt(DOCS) : DOCS + r.nextInt(10 * DOCS)));
    }
    Files.write(dir.resolve("targets.tsv"), targets);
    Files.writeString(dir.resolve("built"), "");
  }

  static void buildOrds(Path dir) throws IOException {
    Random r = new Random(0x0d_5_2026_1020L);
    try (Directory d = FSDirectory.open(dir);
        IndexWriter w = new IndexWriter(d, config("oid", new BlockTreeOrdsPostingsFormat()))) {
      for (int i = 0; i < DOCS; i++) {
        Document doc = new Document();
        doc.add(new StringField("oid", idOf(i), Field.Store.NO));
        w.addDocument(doc);
      }
      w.commit();
    }
    List<String> targets = new ArrayList<>();
    List<String> ords = new ArrayList<>();
    for (int k = 0; k < 200_000; k++) {
      // Half present, half absent; every ord below the term count.
      targets.add(idOf(r.nextInt(2) == 0 ? r.nextInt(DOCS) : DOCS + r.nextInt(10 * DOCS)));
      ords.add(Integer.toString(r.nextInt(DOCS)));
    }
    Files.write(dir.resolve("targets.tsv"), targets);
    Files.write(dir.resolve("ords.tsv"), ords);
    Files.writeString(dir.resolve("built"), "");
  }

  static void buildBitVectors(Path dir) throws IOException {
    Random r = new Random(0x14_8_2026_1020L);
    KnnVectorsFormat format = new HnswBitVectorsFormat();
    IndexWriterConfig cfg = config(Map.of());
    cfg.setCodec(
        new Lucene104Codec() {
          @Override
          public KnnVectorsFormat getKnnVectorsFormatForField(String f) {
            return format;
          }
        });
    try (Directory d = FSDirectory.open(dir);
        IndexWriter w = new IndexWriter(d, cfg)) {
      for (int i = 0; i < 100_000; i++) {
        byte[] v = new byte[32];
        r.nextBytes(v);
        Document doc = new Document();
        doc.add(new KnnByteVectorField("bv", v, VectorSimilarityFunction.EUCLIDEAN));
        w.addDocument(doc);
      }
      w.commit();
    }
    List<String> queries = new ArrayList<>();
    for (int k = 0; k < 2_000; k++) {
      byte[] q = new byte[32];
      r.nextBytes(q);
      queries.add(HexFormat.of().formatHex(q));
    }
    Files.write(dir.resolve("queries.tsv"), queries);
    Files.writeString(dir.resolve("built"), "");
  }

  /** {@code cx_bitvec_search}: a top-10 graph search per query. */
  static void runBitVectors(Path dir) throws IOException {
    if (!selected("cx_bitvec_search")) return;
    List<byte[]> queries = new ArrayList<>();
    for (String line : Files.readAllLines(dir.resolve("queries.tsv"))) {
      queries.add(HexFormat.of().parseHex(line));
    }
    try (Directory d = FSDirectory.open(dir);
        DirectoryReader reader = DirectoryReader.open(d)) {
      LeafReader leaf = reader.leaves().get(0).reader();
      AcceptDocs all = AcceptDocs.fromLiveDocs(null, leaf.maxDoc());
      // The per-field vectors reader, as the Rust side calls its reader.
      KnnVectorsReader vectors = ((CodecReader) leaf).getVectorReader();
      Op op =
          () -> {
            Fnv g = new Fnv();
            for (byte[] q : queries) {
              TopKnnCollector c = new TopKnnCollector(10, Integer.MAX_VALUE);
        vectors.search("bv", q, c, all);
        TopDocs top = c.topDocs();
              for (ScoreDoc sd : top.scoreDocs) {
                g.add(sd.doc);
                g.add(Float.floatToIntBits(sd.score));
              }
            }
            sink += g.h;
            return queries.size();
          };
      Fnv f = new Fnv();
      for (byte[] q : queries) {
        TopKnnCollector c = new TopKnnCollector(10, Integer.MAX_VALUE);
        vectors.search("bv", q, c, all);
        TopDocs top = c.topDocs();
        for (ScoreDoc sd : top.scoreDocs) {
          f.add(sd.doc);
          f.add(Float.floatToIntBits(sd.score));
        }
      }
      System.out.printf("#check	cx_bitvec_search	%016x	%d%n", f.h, queries.size());
      measure("cx_bitvec_search", op);
    }
  }

  /** Builds {@code usplit/}: {@code cx_usplit_seek}'s ids under UniformSplit. */
  static void buildUniformSplit(Path dir) throws IOException {
    Random r = new Random(0x15_5_2026_1020L);
    try (Directory d = FSDirectory.open(dir);
        IndexWriter w = new IndexWriter(d, config("uid", new UniformSplitPostingsFormat()))) {
      for (int i = 0; i < DOCS; i++) {
        Document doc = new Document();
        doc.add(new StringField("uid", idOf(i), Field.Store.NO));
        w.addDocument(doc);
      }
      w.commit();
    }
    List<String> targets = new ArrayList<>();
    for (int k = 0; k < 200_000; k++) {
      // Half present, half absent.
      targets.add(idOf(r.nextInt(2) == 0 ? r.nextInt(DOCS) : DOCS + r.nextInt(10 * DOCS)));
    }
    Files.write(dir.resolve("targets.tsv"), targets);
    Files.writeString(dir.resolve("built"), "");
  }

  /**
   * {@code blockterms}' writer over the default fixed-gap index (interval 32) and {@code Lucene104}
   * postings: Lucene ships no postings format over the package, and none is registered by SPI, so
   * {@link #runBlockTerms} opens the reader by hand rather than through {@code DirectoryReader}.
   */
  static final class BlockTermsBenchFormat extends PostingsFormat {
    BlockTermsBenchFormat() {
      super("BlockTermsBench");
    }

    @Override
    public FieldsConsumer fieldsConsumer(SegmentWriteState state) throws IOException {
      PostingsWriterBase docs = new Lucene104PostingsWriter(state);
      return new BlockTermsWriter(new FixedGapTermsIndexWriter(state), state, docs);
    }

    @Override
    public FieldsProducer fieldsProducer(SegmentReadState state) throws IOException {
      PostingsReaderBase docs = new Lucene104PostingsReader(state);
      return new BlockTermsReader(new FixedGapTermsIndexReader(state), docs, state);
    }
  }

  /** Builds {@code bterms/}: {@code cx_bterms_seek}'s ids under {@link BlockTermsBenchFormat}. */
  static void buildBlockTerms(Path dir) throws IOException {
    Random r = new Random(0x15_5_2026_1022L);
    try (Directory d = FSDirectory.open(dir);
        IndexWriter w = new IndexWriter(d, config("bid", new BlockTermsBenchFormat()))) {
      for (int i = 0; i < DOCS; i++) {
        Document doc = new Document();
        doc.add(new StringField("bid", idOf(i), Field.Store.NO));
        w.addDocument(doc);
      }
      w.commit();
    }
    List<String> targets = new ArrayList<>();
    for (int k = 0; k < 200_000; k++) {
      // Half present, half absent.
      targets.add(idOf(r.nextInt(2) == 0 ? r.nextInt(DOCS) : DOCS + r.nextInt(10 * DOCS)));
    }
    Files.write(dir.resolve("targets.tsv"), targets);
    Files.writeString(dir.resolve("built"), "");
  }

  static final int ST_DOCS = 50_000;

  /**
   * {@code SimpleTextCodec}'s postings format under a name of its own: the format itself is
   * package-private and not registered by SPI, so {@link #runSimpleText} opens it by hand.
   */
  static final class SimpleTextBenchFormat extends PostingsFormat {
    final PostingsFormat delegate = new SimpleTextCodec().postingsFormat();

    SimpleTextBenchFormat() {
      super("SimpleTextBench");
    }

    @Override
    public FieldsConsumer fieldsConsumer(SegmentWriteState state) throws IOException {
      return delegate.fieldsConsumer(state);
    }

    @Override
    public FieldsProducer fieldsProducer(SegmentReadState state) throws IOException {
      return delegate.fieldsProducer(state);
    }
  }

  /** Builds {@code stext/}: {@code cx_stext_seek}'s skewed words under SimpleText. */
  static void buildSimpleText(Path dir) throws IOException {
    Random r = new Random(0x16_5_2026_1020L);
    FieldType freqs = new FieldType(TextField.TYPE_NOT_STORED);
    freqs.setIndexOptions(IndexOptions.DOCS_AND_FREQS);
    freqs.freeze();
    try (Directory d = FSDirectory.open(dir);
        IndexWriter w = new IndexWriter(d, config("st", new SimpleTextBenchFormat()))) {
      for (int i = 0; i < ST_DOCS; i++) {
        StringBuilder sb = new StringBuilder();
        for (int j = 0, n = 1 + r.nextInt(8); j < n; j++) {
          sb.append('w').append(r.nextInt(1 + r.nextInt(3000))).append(' ');
        }
        Document doc = new Document();
        doc.add(new Field("st", sb.toString(), freqs));
        w.addDocument(doc);
      }
      w.commit();
    }
    List<String> targets = new ArrayList<>();
    for (int k = 0; k < 4_000; k++) {
      // Mostly present (some words never drawn), and half absent.
      targets.add("w" + (r.nextInt(2) == 0 ? r.nextInt(3000) : 3000 + r.nextInt(100_000)));
    }
    Files.write(dir.resolve("targets.tsv"), targets);
    Files.writeString(dir.resolve("built"), "");
  }

  /** Builds {@code stsplit/}: {@code cx_stsplit_seek}'s ids under SharedTermsUniformSplit. */
  static void buildSharedTerms(Path dir) throws IOException {
    Random r = new Random(0x15_5_2026_1021L);
    PostingsFormat shared = new STUniformSplitPostingsFormat();
    try (Directory d = FSDirectory.open(dir);
        IndexWriter w = new IndexWriter(d, config(Map.of("sa", shared, "sb", shared)))) {
      for (int i = 0; i < DOCS; i++) {
        Document doc = new Document();
        doc.add(new StringField("sa", idOf(i), Field.Store.NO));
        if (i % 3 == 0) {
          doc.add(new StringField("sb", idOf(i), Field.Store.NO));
        }
        w.addDocument(doc);
      }
      w.commit();
    }
    List<String> targets = new ArrayList<>();
    for (int k = 0; k < 200_000; k++) {
      // Half present (in sa), half absent.
      targets.add(idOf(r.nextInt(2) == 0 ? r.nextInt(DOCS) : DOCS + r.nextInt(10 * DOCS)));
    }
    Files.write(dir.resolve("targets.tsv"), targets);
    Files.writeString(dir.resolve("built"), "");
  }

  /** {@code cx_bloom_seek}: {@code seekExact} on one filtered enum, per target. */
  static void runBloom(Path dir) throws IOException {
    runSeek(dir, "cx_bloom_seek", "id");
  }

  /** {@code cx_fst_seek}/{@code cx_direct_seek}: {@code seekExact} on FST50 and Direct. */
  static void runMemory(Path dir) throws IOException {
    runSeek(dir, "cx_fst_seek", "fid");
    runSeek(dir, "cx_direct_seek", "did");
  }

  /**
   * {@code cx_ords_seek}: per target, {@code seekExact(term)} hashing its {@code ord()} (-1 when
   * absent), then {@code seekExact(ord)} hashing the term's length and last byte, on one enum.
   */
  static void runOrds(Path dir) throws IOException {
    if (!selected("cx_ords_seek")) return;
    List<BytesRef> ids = new ArrayList<>();
    for (String line : Files.readAllLines(dir.resolve("targets.tsv"))) {
      ids.add(new BytesRef(line));
    }
    List<String> ordLines = Files.readAllLines(dir.resolve("ords.tsv"));
    long[] ords = new long[ordLines.size()];
    for (int i = 0; i < ords.length; i++) ords[i] = Long.parseLong(ordLines.get(i));
    try (Directory d = FSDirectory.open(dir);
        DirectoryReader reader = DirectoryReader.open(d)) {
      Terms terms = reader.leaves().get(0).reader().terms("oid");
      Op op =
          () -> {
            Fnv g = new Fnv();
            ordsPass(terms, ids, ords, g);
            sink += g.h;
            return ids.size();
          };
      Fnv f = new Fnv();
      ordsPass(terms, ids, ords, f);
      System.out.printf("#check	%s	%016x	%d%n", "cx_ords_seek", f.h, ids.size());
      measure("cx_ords_seek", op);
    }
  }

  static void ordsPass(Terms terms, List<BytesRef> ids, long[] ords, Fnv f) throws IOException {
    TermsEnum te = terms.iterator();
    for (int i = 0; i < ords.length; i++) {
      f.add(te.seekExact(ids.get(i)) ? te.ord() : -1);
      te.seekExact(ords[i]);
      BytesRef t = te.term();
      f.add(((long) t.length << 8) | (t.bytes[t.offset + t.length - 1] & 0xff));
    }
  }

  static void runSeek(Path dir, String name, String field) throws IOException {
    if (!selected(name)) return;
    try (Directory d = FSDirectory.open(dir);
        DirectoryReader reader = DirectoryReader.open(d)) {
      timeSeek(dir, name, reader.leaves().get(0).reader().terms(field));
    }
  }

  /**
   * {@code cx_bterms_seek}: {@code seekExact} on {@code bid}, through a {@code BlockTermsReader}
   * opened over the one segment's {@code BlockTermsBench_0} files.
   */
  static void runBlockTerms(Path dir) throws IOException {
    if (!selected("cx_bterms_seek")) return;
    try (Directory d = FSDirectory.open(dir)) {
      SegmentInfo si = SegmentInfos.readLatestCommit(d).info(0).info;
      FieldInfos fis = si.getCodec().fieldInfosFormat().read(d, si, "", IOContext.DEFAULT);
      SegmentReadState state =
          new SegmentReadState(d, si, fis, IOContext.DEFAULT, "BlockTermsBench_0");
      try (FieldsProducer fields = new BlockTermsBenchFormat().fieldsProducer(state)) {
        timeSeek(dir, "cx_bterms_seek", fields.terms("bid"));
      }
    }
  }

  /** One {@code cx_stext_seek} pass on one enum: seekExact, then every posting of a hit. */
  static void stextPass(Terms terms, List<BytesRef> ids, Fnv f) throws IOException {
    TermsEnum te = terms.iterator();
    for (BytesRef id : ids) {
      if (!te.seekExact(id)) {
        f.add(-1);
        continue;
      }
      f.add(te.docFreq());
      PostingsEnum pe = te.postings(null, PostingsEnum.FREQS);
      long s = 0;
      for (int doc = pe.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS; doc = pe.nextDoc()) {
        s += doc * 31L + pe.freq();
      }
      f.add(s);
    }
  }

  /**
   * {@code cx_stext_seek}: per target, {@code seekExact} on {@code st}, then a hit's postings
   * (docs and freqs) walked to the end, through the one segment's {@code SimpleTextBench_0} file.
   */
  static void runSimpleText(Path dir) throws IOException {
    if (!selected("cx_stext_seek")) return;
    List<BytesRef> ids = new ArrayList<>();
    for (String line : Files.readAllLines(dir.resolve("targets.tsv"))) {
      ids.add(new BytesRef(line));
    }
    try (Directory d = FSDirectory.open(dir)) {
      SegmentInfo si = SegmentInfos.readLatestCommit(d).info(0).info;
      FieldInfos fis = si.getCodec().fieldInfosFormat().read(d, si, "", IOContext.DEFAULT);
      SegmentReadState state =
          new SegmentReadState(d, si, fis, IOContext.DEFAULT, "SimpleTextBench_0");
      try (FieldsProducer fields = new SimpleTextBenchFormat().fieldsProducer(state)) {
        Terms terms = fields.terms("st");
        Fnv f = new Fnv();
        stextPass(terms, ids, f);
        System.out.printf("#check\t%s\t%016x\t%d%n", "cx_stext_seek", f.h, ids.size());
        measure(
            "cx_stext_seek",
            () -> {
              Fnv g = new Fnv();
              stextPass(terms, ids, g);
              sink += g.h;
              return ids.size();
            });
      }
    }
  }

  /** Times {@code seekExact} (hashing {@code docFreq}, -1 when absent) on each of the targets. */
  static void timeSeek(Path dir, String name, Terms terms) throws IOException {
    List<BytesRef> ids = new ArrayList<>();
    for (String line : Files.readAllLines(dir.resolve("targets.tsv"))) {
      ids.add(new BytesRef(line));
    }
    Op op =
        () -> {
          TermsEnum te = terms.iterator();
          Fnv g = new Fnv();
          for (BytesRef id : ids) {
            g.add(te.seekExact(id) ? te.docFreq() : -1);
          }
          sink += g.h;
          return ids.size();
        };
    Fnv f = new Fnv();
    TermsEnum te = terms.iterator();
    for (BytesRef id : ids) {
      f.add(te.seekExact(id) ? te.docFreq() : -1);
    }
    System.out.printf("#check\t%s\t%016x\t%d%n", name, f.h, ids.size());
    measure(name, op);
  }

  public static void main(String[] args) throws IOException {
    Path dir = Path.of(args[1]);
    if (args[0].equals("build")) {
      if (!Files.exists(dir.resolve("bloom").resolve("built"))) {
        Files.createDirectories(dir.resolve("bloom"));
        buildBloom(dir.resolve("bloom"));
      }
      if (!Files.exists(dir.resolve("memory").resolve("built"))) {
        Files.createDirectories(dir.resolve("memory"));
        buildMemory(dir.resolve("memory"));
      }
      if (!Files.exists(dir.resolve("bitvectors").resolve("built"))) {
        Files.createDirectories(dir.resolve("bitvectors"));
        buildBitVectors(dir.resolve("bitvectors"));
      }
      if (!Files.exists(dir.resolve("ords").resolve("built"))) {
        Files.createDirectories(dir.resolve("ords"));
        buildOrds(dir.resolve("ords"));
      }
      if (!Files.exists(dir.resolve("usplit").resolve("built"))) {
        Files.createDirectories(dir.resolve("usplit"));
        buildUniformSplit(dir.resolve("usplit"));
      }
      if (!Files.exists(dir.resolve("stsplit").resolve("built"))) {
        Files.createDirectories(dir.resolve("stsplit"));
        buildSharedTerms(dir.resolve("stsplit"));
      }
      if (!Files.exists(dir.resolve("bterms").resolve("built"))) {
        Files.createDirectories(dir.resolve("bterms"));
        buildBlockTerms(dir.resolve("bterms"));
      }
      if (!Files.exists(dir.resolve("stext").resolve("built"))) {
        Files.createDirectories(dir.resolve("stext"));
        buildSimpleText(dir.resolve("stext"));
      }
    } else {
      runBloom(dir.resolve("bloom"));
      runMemory(dir.resolve("memory"));
      runBitVectors(dir.resolve("bitvectors"));
      runOrds(dir.resolve("ords"));
      runSeek(dir.resolve("usplit"), "cx_usplit_seek", "uid");
      runSeek(dir.resolve("stsplit"), "cx_stsplit_seek", "sb");
      runBlockTerms(dir.resolve("bterms"));
      runSimpleText(dir.resolve("stext"));
    }
  }
}
