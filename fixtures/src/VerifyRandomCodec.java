import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.document.BinaryDocValuesField;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.IntPoint;
import org.apache.lucene.document.NumericDocValuesField;
import org.apache.lucene.document.SortedDocValuesField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.BinaryDocValues;
import org.apache.lucene.index.CheckIndex;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.NumericDocValues;
import org.apache.lucene.index.SegmentCommitInfo;
import org.apache.lucene.index.SegmentInfos;
import org.apache.lucene.index.SortedDocValues;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.TieredMergePolicy;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.tests.index.RandomCodec;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.BytesRef;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Random;
import java.util.TreeSet;
import java.util.function.IntPredicate;

/**
 * Verifies {@code crates/lucene-test-framework/examples/write_random_codec_fixture.rs}: for
 * each seed, {@code <dir>/<seed>/flushed} (two segments, every 17th document from {@code d3}
 * deleted) and {@code <dir>/<seed>/merged} (the same force-merged into one) were written by
 * this port's IndexWriter with {@code new RandomCodec(new Random(seed))} installed.
 *
 * <p>Lucene writes the same documents with the same codec itself. The codec's toString() must
 * be Lucene's (the same fields in the same formats), every segment must record
 * {@code Asserting} and hold the same files (each field in the same per-field format
 * instance), its points files must be Lucene's byte for byte between header and checksum
 * (the same leaf size and the same random split dimensions, flushed and merged), and the
 * port's index must pass CheckIndex and answer every term, doc-value and point-range
 * question as the documents say.
 */
public class VerifyRandomCodec {
  static final long[] SEEDS = {3, 5, 14, 20, 56, 58};
  static final int PER_SEGMENT = 1500;
  static final int SEGMENTS = 2;
  static final int TOTAL = PER_SEGMENT * SEGMENTS;
  static final String[] POINTS_CODECS = {
    "Lucene90PointsFormatMeta", "Lucene90PointsFormatIndex", "Lucene90PointsFormatData"
  };
  static final String[] POINTS_EXTENSIONS = {"kdm", "kdi", "kdd"};

  public static void main(String[] args) throws Exception {
    Path root = Path.of(args[0]);
    for (long seed : SEEDS) {
      for (boolean merge : new boolean[] {false, true}) {
        Path rust = root.resolve(Long.toString(seed)).resolve(merge ? "merged" : "flushed");
        Path java = Files.createTempDirectory("random-codec-java");
        String codec = write(java, seed, merge);
        String mine = Files.readString(rust.resolve("codec.txt"));
        if (!mine.equals(codec)) {
          throw new AssertionError(rust + ": codec " + mine + ", Lucene's " + codec);
        }
        compare(rust, java, merge ? 1 : SEGMENTS);
        verify(rust);
      }
    }
    System.out.println("VerifyRandomCodec: ok");
  }

  static boolean deleted(int i) {
    return i % 17 == 3;
  }

  /** {@code write_random_codec_fixture::point}. */
  static int[] point(int i) {
    return new int[] {i % 97, (i * 7) % 101 - 50, i / 3};
  }

  /** {@code write_random_codec_fixture::doc}. */
  static Document doc(int i) {
    Document d = new Document();
    d.add(new StringField("id", "d" + i, Field.Store.YES));
    d.add(new TextField("body", "w" + (i % 5) + " x" + (i % 7) + " common", Field.Store.NO));
    d.add(new StringField("kind", "k" + (i % 11), Field.Store.NO));
    d.add(new StringField("code", "c" + (i % 3), Field.Store.NO));
    d.add(new NumericDocValuesField("n", i));
    d.add(new SortedDocValuesField("s", new BytesRef("s" + (i % 13))));
    d.add(new BinaryDocValuesField("b", new BytesRef("b" + i)));
    d.add(new IntPoint("p", point(i)));
    return d;
  }

  /** {@code write_random_codec_fixture::write}, by Lucene: the codec's toString() after. */
  static String write(Path path, long seed, boolean merge) throws Exception {
    RandomCodec codec = new RandomCodec(new Random(seed));
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
    return codec.toString();
  }

  /** The segment's own files, without its name. */
  static TreeSet<String> files(SegmentCommitInfo sci) throws Exception {
    TreeSet<String> out = new TreeSet<>();
    for (String file : sci.info.files()) {
      out.add(file.substring(sci.info.name.length()));
    }
    return out;
  }

  static byte[] body(Directory dir, String file, String codec) throws Exception {
    try (IndexInput in = dir.openInput(file, IOContext.READONCE)) {
      byte[] all = new byte[(int) in.length()];
      in.readBytes(all, 0, all.length);
      int header = CodecUtil.indexHeaderLength(codec, "");
      return Arrays.copyOfRange(all, header, all.length - 8);
    }
  }

  static void compare(Path rust, Path java, int segments) throws Exception {
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
        if (!"Asserting".equals(rsci.info.getCodec().getName())) {
          throw new AssertionError(rust + ": codec " + rsci.info.getCodec().getName());
        }
        if (rsci.getDelCount() != jsci.getDelCount()) {
          throw new AssertionError(rust + ": deletes " + rsci.getDelCount());
        }
        if (!files(rsci).equals(files(jsci))) {
          throw new AssertionError(
              rust + ": segment " + s + " files " + files(rsci) + ", Lucene's " + files(jsci));
        }
        for (int k = 0; k < 3; k++) {
          String ext = POINTS_EXTENSIONS[k];
          byte[] mine = body(r, rsci.info.name + "." + ext, POINTS_CODECS[k]);
          byte[] lucene = body(j, jsci.info.name + "." + ext, POINTS_CODECS[k]);
          if (!Arrays.equals(mine, lucene)) {
            throw new AssertionError(
                rust + ": segment " + s + " ." + ext + " differs from Lucene's (" + mine.length
                    + " / " + lucene.length + " bytes)");
          }
        }
      }
    }
  }

  static int count(IntPredicate p) {
    int n = 0;
    for (int i = 0; i < TOTAL; i++) {
      if (!deleted(i) && p.test(i)) n++;
    }
    return n;
  }

  static void verify(Path path) throws Exception {
    try (Directory dir = FSDirectory.open(path)) {
      try (CheckIndex check = new CheckIndex(dir)) {
        if (!check.checkIndex().clean) {
          throw new AssertionError(path + ": CheckIndex failed");
        }
      }
      try (DirectoryReader reader = DirectoryReader.open(dir)) {
        if (reader.numDocs() != count(i -> true)) {
          throw new AssertionError(path + ": numDocs " + reader.numDocs());
        }
        IndexSearcher searcher = new IndexSearcher(reader);
        for (int k = 0; k < 7; k++) {
          final int kk = k;
          check(path, "body:x" + k, searcher.count(new TermQuery(new Term("body", "x" + k))),
              count(i -> i % 7 == kk));
        }
        for (int k = 0; k < 11; k++) {
          final int kk = k;
          check(path, "kind:k" + k, searcher.count(new TermQuery(new Term("kind", "k" + k))),
              count(i -> i % 11 == kk));
        }
        for (int k = 0; k < 3; k++) {
          final int kk = k;
          check(path, "code:c" + k, searcher.count(new TermQuery(new Term("code", "c" + k))),
              count(i -> i % 3 == kk));
        }
        int[][] boxes = {
          {0, -50, 0, 96, 50, 1000}, {10, -10, 100, 40, 20, 600}, {50, 0, 0, 60, 50, 999},
        };
        for (int[] box : boxes) {
          int[] lo = {box[0], box[1], box[2]};
          int[] hi = {box[3], box[4], box[5]};
          check(path, "p:" + Arrays.toString(box),
              searcher.count(IntPoint.newRangeQuery("p", lo, hi)),
              count(i -> {
                int[] p = point(i);
                for (int d = 0; d < 3; d++) {
                  if (p[d] < lo[d] || p[d] > hi[d]) return false;
                }
                return true;
              }));
        }
        for (LeafReaderContext ctx : reader.leaves()) {
          LeafReader leaf = ctx.reader();
          Bits live = leaf.getLiveDocs();
          NumericDocValues n = leaf.getNumericDocValues("n");
          SortedDocValues s = leaf.getSortedDocValues("s");
          BinaryDocValues b = leaf.getBinaryDocValues("b");
          for (int doc = 0; doc < leaf.maxDoc(); doc++) {
            if (live != null && !live.get(doc)) continue;
            String id = leaf.storedFields().document(doc).get("id");
            int i = Integer.parseInt(id.substring(1));
            if (!n.advanceExact(doc) || n.longValue() != i) {
              throw new AssertionError(path + ": n of " + id);
            }
            if (!s.advanceExact(doc) || !s.lookupOrd(s.ordValue()).utf8ToString().equals("s" + (i % 13))) {
              throw new AssertionError(path + ": s of " + id);
            }
            if (!b.advanceExact(doc) || !b.binaryValue().utf8ToString().equals("b" + i)) {
              throw new AssertionError(path + ": b of " + id);
            }
          }
        }
      }
    }
  }

  static void check(Path path, String what, int got, int expected) {
    if (got != expected) {
      throw new AssertionError(path + ": " + what + " count " + got + ", expected " + expected);
    }
  }
}
