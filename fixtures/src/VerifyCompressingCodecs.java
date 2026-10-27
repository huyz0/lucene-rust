import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.codecs.StoredFieldsReader;
import org.apache.lucene.codecs.TermVectorsReader;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.DocumentStoredFieldVisitor;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.FieldType;
import org.apache.lucene.document.StoredField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.CheckIndex;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.Fields;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.PostingsEnum;
import org.apache.lucene.index.SegmentCommitInfo;
import org.apache.lucene.index.SegmentInfos;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.Terms;
import org.apache.lucene.index.TermsEnum;
import org.apache.lucene.index.TieredMergePolicy;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.tests.codecs.compressing.CompressingCodec;
import org.apache.lucene.tests.codecs.compressing.DeflateWithPresetCompressingCodec;
import org.apache.lucene.tests.codecs.compressing.FastCompressingCodec;
import org.apache.lucene.tests.codecs.compressing.FastDecompressionCompressingCodec;
import org.apache.lucene.tests.codecs.compressing.HighCompressionCompressingCodec;
import org.apache.lucene.tests.codecs.compressing.LZ4WithPresetCompressingCodec;
import org.apache.lucene.tests.codecs.compressing.dummy.DummyCompressingCodec;
import org.apache.lucene.util.Bits;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.BitSet;
import java.util.List;
import java.util.TreeSet;
import java.util.function.Supplier;

/**
 * Verifies {@code crates/lucene-test-framework/examples/write_compressing_codec_fixture.rs}: for
 * each case, {@code <dir>/<case>/flushed} (two segments, every 17th document from {@code d3}
 * deleted) and {@code <dir>/<case>/merged} (the same force-merged into one) were written by this
 * port's IndexWriter with that {@code CompressingCodec} installed.
 *
 * <p>Lucene writes the same documents with the same codec. The codec's toString() must be
 * Lucene's, every segment must record the codec's name and hold the same files (stored fields and
 * term vectors under the codec's segment suffix), and its stored-fields files must be Lucene's
 * byte for byte between header and checksum (the same compression, chunks and index, flushed and
 * merged). Lucene then reads every stored field and every term vector of the port's index: through
 * DirectoryReader and CheckIndex when the codec's SPI instance can (no segment suffix), else
 * through the codec's own formats.
 */
public class VerifyCompressingCodecs {
  static final int PER_SEGMENT = 400;
  static final int SEGMENTS = 2;
  static final int TOTAL = PER_SEGMENT * SEGMENTS;
  static final int BIG_EVERY = 50;

  static final class Case {
    final String name;
    final Supplier<CompressingCodec> codec;
    /** The stored-fields and term-vectors files' segment suffix. */
    final String suffix;

    Case(String name, Supplier<CompressingCodec> codec, String suffix) {
      this.name = name;
      this.codec = codec;
      this.suffix = suffix;
    }
  }

  /** {@code write_compressing_codec_fixture::CASES}. */
  static final List<Case> CASES =
      List.of(
          new Case("fast-default", FastCompressingCodec::new, ""),
          new Case(
              "fast-decompression",
              () -> new FastDecompressionCompressingCodec(100, 7, true, 5),
              "FastDecompressionCompressingStoredFields"),
          new Case(
              "high-compression",
              () -> new HighCompressionCompressingCodec(250, 64, false, 8),
              ""),
          new Case(
              "dummy",
              () -> new DummyCompressingCodec(100, 10, true, 2),
              "DummyCompressingStoredFields"),
          new Case(
              "deflate-preset",
              () -> new DeflateWithPresetCompressingCodec(5000, 512, false, 10),
              ""),
          new Case("deflate-preset-default", DeflateWithPresetCompressingCodec::new, ""),
          // `LZ4WithPresetCompressingCodec` takes the Deflate codec's suffix.
          new Case(
              "lz4-preset",
              () -> new LZ4WithPresetCompressingCodec(90, 3, true, 22),
              "DeflateWithPresetCompressingStoredFields"),
          new Case("fast-small", () -> new FastCompressingCodec(64, 1000, false, 3), ""));

  public static void main(String[] args) throws Exception {
    Path root = Path.of(args[0]);
    for (Case c : CASES) {
      for (boolean merge : new boolean[] {false, true}) {
        Path rust = root.resolve(c.name).resolve(merge ? "merged" : "flushed");
        Path java = Files.createTempDirectory("compressing-codec-java");
        CompressingCodec codec = c.codec.get();
        write(java, codec, merge);
        String mine = Files.readString(rust.resolve("codec.txt"));
        if (!mine.equals(codec.toString())) {
          throw new AssertionError(rust + ": codec " + mine + ", Lucene's " + codec);
        }
        compare(rust, java, merge ? 1 : SEGMENTS, codec, c.suffix);
        if (c.suffix.isEmpty()) {
          verify(rust);
        } else {
          verifyFormats(rust, codec);
        }
      }
    }
    System.out.println("VerifyCompressingCodecs: ok");
  }

  static boolean deleted(int i) {
    return i % 17 == 3;
  }

  /** {@code write_compressing_codec_fixture::body}. */
  static String body(int i) {
    return "w" + (i % 5) + " x" + (i % 7) + " common t" + i;
  }

  /** {@code write_compressing_codec_fixture::big}. */
  static String big(int i) {
    return i % BIG_EVERY == 0 ? ("z" + i + " ").repeat(140) : null;
  }

  static final FieldType BODY = new FieldType(TextField.TYPE_STORED);

  static {
    BODY.setStoreTermVectors(true);
    BODY.setStoreTermVectorPositions(true);
    BODY.setStoreTermVectorOffsets(true);
    BODY.freeze();
  }

  /** {@code write_compressing_codec_fixture::doc}. */
  static Document doc(int i) {
    Document d = new Document();
    d.add(new StringField("id", "d" + i, Field.Store.YES));
    d.add(new Field("body", body(i), BODY));
    d.add(new StoredField("n", i));
    if (big(i) != null) {
      d.add(new StoredField("big", big(i)));
    }
    return d;
  }

  static void write(Path path, Codec codec, boolean merge) throws Exception {
    try (Directory dir = FSDirectory.open(path)) {
      IndexWriterConfig config = new IndexWriterConfig();
      config.setCodec(codec);
      config.setUseCompoundFile(false);
      config.setMaxBufferedDocs(IndexWriterConfig.DISABLE_AUTO_FLUSH);
      config.setRAMBufferSizeMB(256);
      TieredMergePolicy tiered = new TieredMergePolicy();
      tiered.setNoCFSRatio(0);
      config.setMergePolicy(merge ? tiered : NoMergePolicy.INSTANCE);
      try (IndexWriter w = new IndexWriter(dir, config)) {
        for (int seg = 0; seg < SEGMENTS; seg++) {
          for (int i = seg * PER_SEGMENT; i < (seg + 1) * PER_SEGMENT; i++) {
            w.addDocument(doc(i));
          }
          w.commit();
        }
        for (int i = 0; i < TOTAL; i++) {
          if (deleted(i)) {
            w.deleteDocuments(new Term("id", "d" + i));
          }
        }
        w.commit();
        if (merge) {
          w.forceMerge(1);
          w.commit();
        }
      }
    }
  }

  /** The segment's own files, without its name. */
  static TreeSet<String> files(SegmentCommitInfo sci) throws Exception {
    TreeSet<String> out = new TreeSet<>();
    for (String file : sci.info.files()) {
      out.add(file.substring(sci.info.name.length()));
    }
    return out;
  }

  static byte[] body(Directory dir, String file, String codec, String suffix) throws Exception {
    try (IndexInput in = dir.openInput(file, IOContext.READONCE)) {
      byte[] all = new byte[(int) in.length()];
      in.readBytes(all, 0, all.length);
      int header = CodecUtil.indexHeaderLength(codec, suffix);
      return Arrays.copyOfRange(all, header, all.length - 8);
    }
  }

  static String suffixed(String segment, String suffix, String ext) {
    return segment + (suffix.isEmpty() ? "" : "_" + suffix) + "." + ext;
  }

  static void compare(Path rust, Path java, int segments, Codec codec, String suffix)
      throws Exception {
    String name = codec.getName();
    String[] exts = {"fdt", "fdx", "fdm"};
    String[] codecs = {name, "Lucene90FieldsIndexIdx", "Lucene90FieldsIndexMeta"};
    try (Directory r = FSDirectory.open(rust);
        Directory j = FSDirectory.open(java)) {
      SegmentInfos rs = SegmentInfos.readLatestCommit(r);
      SegmentInfos js = SegmentInfos.readLatestCommit(j);
      if (rs.size() != segments || js.size() != segments) {
        throw new AssertionError(rust + ": segments " + rs.size() + " / " + js.size());
      }
      for (int s = 0; s < segments; s++) {
        SegmentCommitInfo rsci = rs.info(s);
        SegmentCommitInfo jsci = js.info(s);
        if (!name.equals(rsci.info.getCodec().getName())) {
          throw new AssertionError(rust + ": codec " + rsci.info.getCodec().getName());
        }
        if (rsci.getDelCount() != jsci.getDelCount()) {
          throw new AssertionError(rust + ": deletes " + rsci.getDelCount());
        }
        if (!files(rsci).equals(files(jsci))) {
          throw new AssertionError(
              rust + ": segment " + s + " files " + files(rsci) + ", Lucene's " + files(jsci));
        }
        if (!rsci.info.getAttributes().equals(jsci.info.getAttributes())) {
          throw new AssertionError(
              rust + ": attributes " + rsci.info.getAttributes() + ", Lucene's "
                  + jsci.info.getAttributes());
        }
        for (int k = 0; k < exts.length; k++) {
          byte[] mine = body(r, suffixed(rsci.info.name, suffix, exts[k]), codecs[k], suffix);
          byte[] lucene = body(j, suffixed(jsci.info.name, suffix, exts[k]), codecs[k], suffix);
          if (!Arrays.equals(mine, lucene)) {
            throw new AssertionError(
                rust + ": segment " + s + " ." + exts[k] + " differs from Lucene's ("
                    + mine.length + " / " + lucene.length + " bytes)");
          }
        }
      }
    }
  }

  /** The document a stored id names, after checking its other stored fields. */
  static int checkStored(Path path, Document stored) {
    String id = stored.get("id");
    int i = Integer.parseInt(id.substring(1));
    if (!body(i).equals(stored.get("body"))) {
      throw new AssertionError(path + ": body of " + id);
    }
    if (stored.getField("n").numericValue().intValue() != i) {
      throw new AssertionError(path + ": n of " + id);
    }
    String big = big(i);
    if (big == null ? stored.get("big") != null : !big.equals(stored.get("big"))) {
      throw new AssertionError(path + ": big of " + id);
    }
    return i;
  }

  /** {@code body(i)}'s four tokens, positions and offsets. */
  static void checkVectors(Path path, int i, Fields fields) throws Exception {
    Terms terms = fields == null ? null : fields.terms("body");
    if (terms == null) {
      throw new AssertionError(path + ": no term vector for d" + i);
    }
    String[] tokens = body(i).split(" ");
    TreeSet<String> sorted = new TreeSet<>(Arrays.asList(tokens));
    TermsEnum te = terms.iterator();
    int seen = 0;
    for (String expected : sorted) {
      if (te.next() == null || !te.term().utf8ToString().equals(expected)) {
        throw new AssertionError(path + ": term vector terms of d" + i);
      }
      PostingsEnum pe = te.postings(null, PostingsEnum.ALL);
      pe.nextDoc();
      int position = Arrays.asList(tokens).indexOf(expected);
      int start = body(i).indexOf(expected + (position == tokens.length - 1 ? "" : " "));
      if (pe.freq() != 1
          || pe.nextPosition() != position
          || pe.startOffset() != start
          || pe.endOffset() != start + expected.length()) {
        throw new AssertionError(path + ": term vector of " + expected + " in d" + i);
      }
      seen++;
    }
    if (te.next() != null || seen != tokens.length) {
      throw new AssertionError(path + ": extra term vector terms in d" + i);
    }
  }

  static void checkCount(Path path, BitSet live) {
    for (int i = 0; i < TOTAL; i++) {
      if (live.get(i) == deleted(i)) {
        throw new AssertionError(path + ": d" + i + " live " + live.get(i));
      }
    }
  }

  /** Through DirectoryReader, CheckIndex and the codec's SPI instance. */
  static void verify(Path path) throws Exception {
    try (Directory dir = FSDirectory.open(path)) {
      try (CheckIndex check = new CheckIndex(dir)) {
        if (!check.checkIndex().clean) {
          throw new AssertionError(path + ": CheckIndex failed");
        }
      }
      BitSet live = new BitSet();
      try (DirectoryReader reader = DirectoryReader.open(dir)) {
        for (LeafReaderContext ctx : reader.leaves()) {
          LeafReader leaf = ctx.reader();
          Bits liveDocs = leaf.getLiveDocs();
          for (int doc = 0; doc < leaf.maxDoc(); doc++) {
            int i = checkStored(path, leaf.storedFields().document(doc));
            checkVectors(path, i, leaf.termVectors().get(doc));
            if (liveDocs == null || liveDocs.get(doc)) {
              live.set(i);
            }
          }
        }
      }
      checkCount(path, live);
    }
  }

  /**
   * Through the codec's own stored-fields and term-vectors formats: the SPI instance has no
   * segment suffix, so it cannot open these files.
   */
  static void verifyFormats(Path path, CompressingCodec codec) throws Exception {
    try (Directory dir = FSDirectory.open(path)) {
      int count = 0;
      for (SegmentCommitInfo sci : SegmentInfos.readLatestCommit(dir)) {
        FieldInfos infos = codec.fieldInfosFormat().read(dir, sci.info, "", IOContext.DEFAULT);
        try (StoredFieldsReader stored =
                codec.storedFieldsFormat().fieldsReader(dir, sci.info, infos, IOContext.DEFAULT);
            TermVectorsReader vectors =
                codec.termVectorsFormat().vectorsReader(dir, sci.info, infos, IOContext.DEFAULT)) {
          stored.checkIntegrity();
          vectors.checkIntegrity();
          for (int doc = 0; doc < sci.info.maxDoc(); doc++) {
            DocumentStoredFieldVisitor visitor = new DocumentStoredFieldVisitor();
            stored.document(doc, visitor);
            int i = checkStored(path, visitor.getDocument());
            checkVectors(path, i, vectors.get(doc));
            count++;
          }
        }
      }
      // A merge drops the deleted documents; flushed segments keep them.
      int expected = TOTAL;
      if (path.getFileName().toString().equals("merged")) {
        for (int i = 0; i < TOTAL; i++) {
          if (deleted(i)) expected--;
        }
      }
      if (count != expected) {
        throw new AssertionError(path + ": " + count + " documents, expected " + expected);
      }
    }
  }
}
