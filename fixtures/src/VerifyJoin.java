import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.NumericDocValuesField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.CheckIndex;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.StoredFields;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.Sort;
import org.apache.lucene.search.SortField;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.join.BitSetProducer;
import org.apache.lucene.search.join.CheckJoinIndex;
import org.apache.lucene.search.join.QueryBitSetProducer;
import org.apache.lucene.search.join.ScoreMode;
import org.apache.lucene.search.join.ToChildBlockJoinQuery;
import org.apache.lucene.search.join.ToParentBlockJoinQuery;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Reverse-direction verifier (Rust writes, Java reads) for document blocks
 * through merges, with and without an index sort: {@code
 * write_block_join_fixture.rs} writes {@code <dir>/unsorted} and {@code
 * <dir>/sorted}, each a seeded stream of block adds, updates, deletes,
 * commits and merges with the parent field {@code _parent}, and lists the
 * live blocks in {@code blocks.tsv}.
 *
 * <p>For each index: {@link CheckIndex} is clean; every segment names {@code
 * _parent} as its parent field (and the sorted one records the sort);
 * {@link CheckJoinIndex} passes; every listed block's children come back
 * through {@link ToChildBlockJoinQuery} from its parent, in order, and its
 * parent through {@link ToParentBlockJoinQuery} from its children; and no
 * other parent is live. Then Lucene's own {@code IndexWriter}, configured
 * with the same parent field and sort, appends blocks and force-merges the
 * index to one segment, and the checks run again -- a Rust-written block
 * index is one Lucene can keep writing.
 *
 * <p>Usage: {@code java VerifyJoin <dir>}. Exits nonzero with a diagnosis on
 * any mismatch.
 */
public class VerifyJoin {
  private static int failures = 0;

  private static void fail(String message) {
    System.out.println("MISMATCH " + message);
    failures++;
  }

  public static void main(String[] args) throws IOException {
    Path root = Path.of(args[0]);
    for (String variant : new String[] {"unsorted", "sorted"}) {
      verify(root.resolve(variant), variant.equals("sorted"));
    }
    if (failures > 0) {
      System.out.println(failures + " failure(s)");
      System.exit(1);
    }
    System.out.println("OK");
  }

  private static void verify(Path variant, boolean sorted) throws IOException {
    List<long[]> blocks = new ArrayList<>();
    for (String line : Files.readAllLines(variant.resolve("blocks.tsv"), StandardCharsets.UTF_8)) {
      String[] p = line.split("\t");
      blocks.add(
          new long[] {
            Long.parseLong(p[0]), Long.parseLong(p[1]), Long.parseLong(p[2]), Long.parseLong(p[3])
          });
    }
    Sort sort = new Sort(new SortField("rank", SortField.Type.LONG, false));
    BitSetProducer parents =
        new QueryBitSetProducer(new TermQuery(new Term("type", "parent")));
    try (Directory dir = FSDirectory.open(variant.resolve("index"))) {
      checkIndex(dir, variant + " (as Rust wrote it)");
      try (DirectoryReader reader = DirectoryReader.open(dir)) {
        if (reader.leaves().size() < 2) {
          fail(variant + ": expected several segments, got " + reader.leaves().size());
        }
        boolean anyBlocks = false;
        for (LeafReaderContext leaf : reader.leaves()) {
          anyBlocks |= leaf.reader().getMetaData().hasBlocks();
          String parentField = leaf.reader().getFieldInfos().getParentField();
          if ("_parent".equals(parentField) == false) {
            fail(variant + ": segment " + leaf.reader() + " has parent field " + parentField);
          }
          Sort segmentSort = leaf.reader().getMetaData().sort();
          if (sorted && sort.equals(segmentSort) == false) {
            fail(variant + ": segment " + leaf.reader() + " is sorted by " + segmentSort);
          }
          if (!sorted && segmentSort != null) {
            fail(variant + ": unsorted segment " + leaf.reader() + " has sort " + segmentSort);
          }
        }
        if (!anyBlocks) {
          fail(variant + ": no segment has blocks");
        }
        checkBlocks(reader, parents, blocks, variant + " (as Rust wrote it)");
      }

      // Lucene keeps writing the index: more blocks, then one segment.
      IndexWriterConfig iwc = new IndexWriterConfig().setParentField("_parent");
      if (sorted) {
        iwc.setIndexSort(sort);
      }
      try (IndexWriter w = new IndexWriter(dir, iwc)) {
        for (int i = 0; i < 25; i++) {
          long b = 1_000_000 + i;
          int children = i % 4;
          long rank = (i * 7) % 40;
          w.addDocuments(block(b, 0, children, rank));
          blocks.add(new long[] {b, 0, children, rank});
        }
        w.forceMerge(1);
      }
      checkIndex(dir, variant + " (after Lucene's merge)");
      try (DirectoryReader reader = DirectoryReader.open(dir)) {
        if (reader.leaves().size() != 1) {
          fail(variant + ": forceMerge(1) left " + reader.leaves().size() + " segments");
        }
        checkBlocks(reader, parents, blocks, variant + " (after Lucene's merge)");
      }
    }
  }

  private static List<Document> block(long b, long v, int children, long rank) {
    List<Document> docs = new ArrayList<>();
    for (int j = 0; j < children; j++) {
      Document d = new Document();
      d.add(new StringField("id", "c" + b + "." + v + "." + j, Field.Store.YES));
      d.add(new StringField("block", Long.toString(b), Field.Store.NO));
      d.add(new StringField("type", "child", Field.Store.NO));
      docs.add(d);
    }
    Document p = new Document();
    p.add(new StringField("id", "p" + b + "." + v, Field.Store.YES));
    p.add(new StringField("block", Long.toString(b), Field.Store.NO));
    p.add(new StringField("type", "parent", Field.Store.NO));
    p.add(new NumericDocValuesField("rank", rank));
    docs.add(p);
    return docs;
  }

  private static void checkIndex(Directory dir, String what) throws IOException {
    ByteArrayOutputStream log = new ByteArrayOutputStream();
    try (CheckIndex checker = new CheckIndex(dir)) {
      checker.setInfoStream(new PrintStream(log, true, StandardCharsets.UTF_8));
      CheckIndex.Status status = checker.checkIndex();
      if (!status.clean) {
        fail(what + ": CheckIndex is not clean:\n" + log.toString(StandardCharsets.UTF_8));
      }
    }
  }

  private static List<String> ids(IndexSearcher searcher, Query q) throws IOException {
    TopDocs top = searcher.search(q, 10_000);
    StoredFields stored = searcher.storedFields();
    List<String> out = new ArrayList<>();
    for (ScoreDoc sd : top.scoreDocs) {
      out.add(stored.document(sd.doc).get("id"));
    }
    Collections.sort(out);
    return out;
  }

  private static void checkBlocks(
      DirectoryReader reader, BitSetProducer parents, List<long[]> blocks, String what)
      throws IOException {
    try {
      CheckJoinIndex.check(reader, parents);
    } catch (IllegalStateException e) {
      fail(what + ": CheckJoinIndex: " + e.getMessage());
      return;
    }
    IndexSearcher searcher = new IndexSearcher(reader);
    int liveParents = searcher.count(new TermQuery(new Term("type", "parent")));
    if (liveParents != blocks.size()) {
      fail(what + ": " + liveParents + " live parents, expected " + blocks.size());
    }
    for (long[] blk : blocks) {
      long b = blk[0], v = blk[1], children = blk[2];
      String parentId = "p" + b + "." + v;
      List<String> want = new ArrayList<>();
      for (int j = 0; j < children; j++) {
        want.add("c" + b + "." + v + "." + j);
      }
      Collections.sort(want);
      List<String> got =
          ids(searcher, new ToChildBlockJoinQuery(new TermQuery(new Term("id", parentId)), parents));
      if (!got.equals(want)) {
        fail(what + ": children of " + parentId + " are " + got + ", expected " + want);
      }
      BooleanQuery childrenOfBlock =
          new BooleanQuery.Builder()
              .add(new TermQuery(new Term("type", "child")), BooleanClause.Occur.FILTER)
              .add(new TermQuery(new Term("block", Long.toString(b))), BooleanClause.Occur.FILTER)
              .build();
      List<String> parent =
          ids(searcher, new ToParentBlockJoinQuery(childrenOfBlock, parents, ScoreMode.Avg));
      List<String> wantParent = children > 0 ? List.of(parentId) : List.of();
      if (!parent.equals(wantParent)) {
        fail(what + ": parent of block " + b + " is " + parent + ", expected " + wantParent);
      }
    }
  }
}
