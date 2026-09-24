import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.LongPoint;
import org.apache.lucene.document.NumericDocValuesField;
import org.apache.lucene.document.SortedDocValuesField;
import org.apache.lucene.document.StoredField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.CheckIndex;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.NumericDocValues;
import org.apache.lucene.index.PointValues;
import org.apache.lucene.index.SortedDocValues;
import org.apache.lucene.index.StoredFields;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.PhraseQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.BytesRef;
import org.apache.lucene.util.SmallFloat;

/**
 * The Java half of M4's interoperability matrix (T4.6): indexes written partly by real Lucene and
 * partly by this port, in either order, and read back by both.
 *
 * <p>Document {@code i} is the same in both engines -- {@code crates/lucene-search/examples/interop.rs}
 * has the same table:
 *
 * <table>
 *   <tr><th>field</th><th>Java</th><th>value</th></tr>
 *   <tr><td>{@code id}</td><td>stored</td><td>{@code "doc" + i}</td></tr>
 *   <tr><td>{@code body}</td><td>{@code TextField}, stored: positions and norms</td>
 *       <td>{@link #body}</td></tr>
 *   <tr><td>{@code score}</td><td>stored + {@code NumericDocValuesField}</td>
 *       <td>{@code 3i - 1000}</td></tr>
 *   <tr><td>{@code pt}</td><td>stored + {@code LongPoint}</td><td>{@code 7i - 500}</td></tr>
 *   <tr><td>{@code cat}</td><td>stored + {@code SortedDocValuesField}</td>
 *       <td>{@code "c" + (i % 10)}</td></tr>
 * </table>
 *
 * <p>Usage:
 *
 * <ul>
 *   <li>{@code write <dir> <from> <to> <docsPerSegment>}: add documents {@code [from, to)} with a
 *       default {@code IndexWriterConfig} -- so compound files, as Lucene flushes them -- appending
 *       when an index is already there, and commit. {@code NoMergePolicy}, so the segments stay for
 *       the other engine to merge.
 *   <li>{@code verify <dir> <numDocs> <maxSegments> [w<n>]}: open with {@code DirectoryReader}
 *       and require exactly documents {@code 0 .. numDocs} -- less every document whose body has
 *       word {@code w<n>}, when given, which the other engine deleted -- each with every field above
 *       (stored, doc values, points, postings with positions and norms), then {@code CheckIndex}.
 *   <li>{@code delete <dir> <word>}: delete every document whose body has {@code word}.
 * </ul>
 */
public class InteropIndex {
  private static int failures = 0;

  /** {@code w} when every document whose body has word {@code w<n>} was deleted, else -1. */
  private static int deletedWord = -1;

  static boolean live(int i) {
    return deletedWord < 0 || i % 20 != deletedWord;
  }

  private static void fail(String message) {
    if (++failures <= 30) {
      System.out.println("MISMATCH " + message);
    }
  }

  /** Must match {@code interop.rs}: a word of 20, a word of 97, and "shared" once to three times. */
  static String body(int i) {
    StringBuilder b = new StringBuilder();
    b.append("w").append(i % 20).append(' ');
    for (int r = 0; r <= i % 3; r++) {
      b.append("shared ");
    }
    b.append("v").append(i % 97);
    return b.toString();
  }

  static Document document(int i) {
    Document d = new Document();
    d.add(new StoredField("id", "doc" + i));
    d.add(new TextField("body", body(i), Field.Store.YES));
    long score = 3L * i - 1000;
    d.add(new StoredField("score", score));
    d.add(new NumericDocValuesField("score", score));
    long pt = 7L * i - 500;
    d.add(new StoredField("pt", pt));
    d.add(new LongPoint("pt", pt));
    String cat = "c" + (i % 10);
    d.add(new StoredField("cat", cat));
    d.add(new SortedDocValuesField("cat", new BytesRef(cat)));
    return d;
  }

  static void write(Path path, int from, int to, int docsPerSegment) throws IOException {
    IndexWriterConfig cfg = new IndexWriterConfig();
    cfg.setOpenMode(IndexWriterConfig.OpenMode.CREATE_OR_APPEND);
    cfg.setMergePolicy(NoMergePolicy.INSTANCE);
    cfg.setMaxBufferedDocs(docsPerSegment);
    cfg.setRAMBufferSizeMB(IndexWriterConfig.DISABLE_AUTO_FLUSH);
    try (Directory dir = FSDirectory.open(path);
        IndexWriter w = new IndexWriter(dir, cfg)) {
      for (int i = from; i < to; i++) {
        w.addDocument(document(i));
      }
      w.commit();
    }
  }

  static void verify(Path path, int numDocs, int maxSegments) throws IOException {
    try (Directory dir = FSDirectory.open(path);
        DirectoryReader reader = DirectoryReader.open(dir)) {
      int liveDocs = 0;
      for (int i = 0; i < numDocs; i++) {
        if (live(i)) liveDocs++;
      }
      if (reader.numDocs() != liveDocs) {
        fail("numDocs=" + reader.numDocs() + ", want " + liveDocs);
      }
      if (reader.leaves().size() > maxSegments) {
        fail(reader.leaves().size() + " segments, want at most " + maxSegments);
      }
      boolean[] seen = new boolean[numDocs];
      for (LeafReaderContext ctx : reader.leaves()) {
        verifyLeaf(ctx.reader(), seen);
      }
      for (int i = 0; i < numDocs; i++) {
        if (seen[i] != live(i)) {
          fail("doc" + i + (live(i) ? " is missing" : " was deleted but is still live"));
        }
      }

      IndexSearcher searcher = new IndexSearcher(reader);
      searcher.setQueryCache(null);
      for (int w = 0; w < 20; w++) {
        final int word = w;
        checkCount(searcher, new TermQuery(new Term("body", "w" + w)), numDocs, i -> i % 20 == word);
      }
      checkCount(searcher, new TermQuery(new Term("body", "shared")), numDocs, i -> true);
      // Positions: "shared shared" is adjacent only where the word repeats.
      checkCount(
          searcher, new PhraseQuery("body", "shared", "shared"), numDocs, i -> i % 3 != 0);
      checkCount(searcher, new PhraseQuery("body", "w3", "shared"), numDocs, i -> i % 20 == 3);
      checkCount(
          searcher, LongPoint.newRangeQuery("pt", -100, 9000), numDocs,
          i -> 7L * i - 500 >= -100 && 7L * i - 500 <= 9000);
    }
    try (Directory dir = FSDirectory.open(path);
        CheckIndex checker = new CheckIndex(dir)) {
      ByteArrayOutputStream captured = new ByteArrayOutputStream();
      checker.setInfoStream(new PrintStream(captured, true, StandardCharsets.UTF_8));
      checker.setLevel(CheckIndex.Level.MIN_LEVEL_FOR_SLOW_CHECKS);
      if (!checker.checkIndex().clean) {
        fail("CheckIndex reported the index unclean");
        System.out.println(captured.toString(StandardCharsets.UTF_8));
      }
    }
  }

  static void verifyLeaf(LeafReader leaf, boolean[] seen) throws IOException {
    StoredFields stored = leaf.storedFields();
    NumericDocValues score = leaf.getNumericDocValues("score");
    SortedDocValues cat = leaf.getSortedDocValues("cat");
    // BM25's norm: the body's length, `SmallFloat.intToByte4`-encoded. Term
    // and phrase counts do not depend on norms, so a merge that attached one
    // document's norm to another would pass everything else here.
    NumericDocValues norms = leaf.getNormValues("body");
    Map<Integer, List<Long>> points = new HashMap<>();
    PointValues values = leaf.getPointValues("pt");
    if (values != null) {
      values.intersect(
          new PointValues.IntersectVisitor() {
            @Override
            public void visit(int docID) {
              throw new IllegalStateException("every cell crosses the query");
            }

            @Override
            public void visit(int docID, byte[] packedValue) {
              points.computeIfAbsent(docID, d -> new ArrayList<>())
                  .add(LongPoint.decodeDimension(packedValue, 0));
            }

            @Override
            public PointValues.Relation compare(byte[] min, byte[] max) {
              return PointValues.Relation.CELL_CROSSES_QUERY;
            }
          });
    }
    for (int doc = 0; doc < leaf.maxDoc(); doc++) {
      if (leaf.getLiveDocs() != null && !leaf.getLiveDocs().get(doc)) {
        continue;
      }
      Document d = stored.document(doc);
      String id = d.get("id");
      int i = Integer.parseInt(id.substring(3));
      if (i < 0 || i >= seen.length || seen[i]) {
        fail(id + " is out of range or seen twice");
        continue;
      }
      seen[i] = true;
      check(id + " body", d.get("body"), body(i));
      check(id + " stored score", d.getField("score").numericValue().longValue(), 3L * i - 1000);
      check(id + " stored pt", d.getField("pt").numericValue().longValue(), 7L * i - 500);
      check(id + " stored cat", d.get("cat"), "c" + (i % 10));
      if (score == null || !score.advanceExact(doc)) {
        fail(id + ": no score doc value");
      } else {
        check(id + " score doc value", score.longValue(), 3L * i - 1000);
      }
      if (cat == null || !cat.advanceExact(doc)) {
        fail(id + ": no cat doc value");
      } else {
        check(id + " cat doc value", cat.lookupOrd(cat.ordValue()).utf8ToString(), "c" + (i % 10));
      }
      if (norms == null || !norms.advanceExact(doc)) {
        fail(id + ": no body norm");
      } else {
        long want = SmallFloat.intToByte4(2 + i % 3 + 1);
        check(id + " body norm", norms.longValue(), want);
      }
      List<Long> have = points.getOrDefault(doc, Collections.emptyList());
      check(id + " points", have, List.of(7L * i - 500));
    }
  }

  static void check(String what, Object got, Object want) {
    if (!want.equals(got)) {
      fail(what + ": got " + got + ", want " + want);
    }
  }

  interface DocPredicate {
    boolean test(int i);
  }

  static void checkCount(IndexSearcher searcher, Query q, int numDocs, DocPredicate matches)
      throws IOException {
    int want = 0;
    for (int i = 0; i < numDocs; i++) {
      if (live(i) && matches.test(i)) want++;
    }
    int got = searcher.count(q);
    if (got != want) {
      fail(q + ": count " + got + ", want " + want);
    }
  }

  public static void main(String[] args) throws IOException {
    Path path = Path.of(args[1]);
    switch (args[0]) {
      case "write":
        write(path, Integer.parseInt(args[2]), Integer.parseInt(args[3]), Integer.parseInt(args[4]));
        System.out.println("InteropIndex: Lucene wrote documents " + args[2] + ".." + args[3]);
        break;
      case "verify":
        if (args.length > 4) {
          deletedWord = Integer.parseInt(args[4].substring(1));
        }
        verify(path, Integer.parseInt(args[2]), Integer.parseInt(args[3]));
        if (failures > 0) {
          System.out.println(failures + " check(s) failed");
          System.exit(1);
        }
        System.out.println("InteropIndex: real Lucene reads all " + args[2] + " documents. PASS");
        break;
      case "delete":
        try (Directory dir = FSDirectory.open(path);
            IndexWriter w =
                new IndexWriter(
                    dir,
                    new IndexWriterConfig()
                        .setOpenMode(IndexWriterConfig.OpenMode.APPEND)
                        .setMergePolicy(NoMergePolicy.INSTANCE))) {
          w.deleteDocuments(new Term("body", args[2]));
          w.commit();
        }
        System.out.println("InteropIndex: Lucene deleted body:" + args[2]);
        break;
      default:
        throw new IllegalArgumentException(args[0]);
    }
  }
}
