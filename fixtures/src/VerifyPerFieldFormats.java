import org.apache.lucene.index.CheckIndex;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.SegmentInfos;
import org.apache.lucene.index.StoredFields;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.PhraseQuery;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.Bits;

import java.nio.file.Path;
import java.util.function.IntPredicate;

/**
 * Verifies {@code crates/lucene-index/examples/write_per_field_formats_fixture.rs}:
 * every segment of {@code <dir>/flushed} (three, a delete by {@code tag:t3}
 * applied) and {@code <dir>/merged} (one) records {@code body} under
 * {@code PerFieldPostingsFormat} suffix 0 and {@code key}/{@code tag} under
 * suffix 1, Lucene's {@code PerFieldPostingsFormat.FieldsReader} finds every
 * field's terms in its own group's files, the delete removed exactly the
 * {@code t3} documents, and {@code CheckIndex} is clean.
 */
public class VerifyPerFieldFormats {
  static final int TOTAL = 1200;

  public static void main(String[] args) throws Exception {
    verify(Path.of(args[0]).resolve("flushed"), 3);
    verify(Path.of(args[0]).resolve("merged"), 1);
    System.out.println("VerifyPerFieldFormats: ok");
  }

  static void expectSuffix(LeafReader leaf, String field, String suffix) {
    FieldInfo fi = leaf.getFieldInfos().fieldInfo(field);
    if (!"Lucene104".equals(fi.getAttribute("PerFieldPostingsFormat.format"))
        || !suffix.equals(fi.getAttribute("PerFieldPostingsFormat.suffix"))) {
      throw new AssertionError(field + " attributes " + fi.attributes());
    }
  }

  /** Live documents (every one but {@code i % 7 == 3}) matching {@code p}. */
  static int count(IntPredicate p) {
    int n = 0;
    for (int i = 0; i < TOTAL; i++) {
      if (i % 7 != 3 && p.test(i)) n++;
    }
    return n;
  }

  static void verify(Path path, int segments) throws Exception {
    try (Directory dir = FSDirectory.open(path)) {
      SegmentInfos sis = SegmentInfos.readLatestCommit(dir);
      if (sis.size() != segments) {
        throw new AssertionError(path + ": expected " + segments + " segments, got " + sis.size());
      }
      try (DirectoryReader reader = DirectoryReader.open(dir)) {
        if (reader.numDocs() != count(i -> true)) {
          throw new AssertionError("numDocs " + reader.numDocs());
        }
        IndexSearcher searcher = new IndexSearcher(reader);
        for (int t = 0; t < 7; t++) {
          final int tt = t;
          int got = searcher.count(new TermQuery(new Term("tag", "t" + t)));
          if (got != count(i -> i % 7 == tt)) {
            throw new AssertionError("tag:t" + t + " count " + got);
          }
        }
        for (int k = 0; k < 50; k++) {
          final int kk = k;
          int got = searcher.count(new TermQuery(new Term("key", "k" + k)));
          if (got != count(i -> i % 50 == kk)) {
            throw new AssertionError("key:k" + k + " count " + got);
          }
        }
        if (searcher.count(new PhraseQuery("body", "shared", "a5")) != count(i -> i % 13 == 5)) {
          throw new AssertionError("phrase count");
        }
        for (LeafReaderContext ctx : reader.leaves()) {
          LeafReader leaf = ctx.reader();
          expectSuffix(leaf, "body", "0");
          expectSuffix(leaf, "key", "1");
          expectSuffix(leaf, "tag", "1");
          if (leaf.getFieldInfos().fieldInfo("id").getAttribute("PerFieldPostingsFormat.format")
              != null) {
            throw new AssertionError("a stored-only field names a postings format");
          }
          StoredFields stored = leaf.storedFields();
          Bits live = leaf.getLiveDocs();
          for (int d = 0; d < leaf.maxDoc(); d++) {
            if (live != null && !live.get(d)) continue;
            int i = Integer.parseInt(stored.document(d).get("id").substring(3));
            if (i % 7 == 3) {
              throw new AssertionError("doc" + i + " survived the delete");
            }
          }
        }
      }
      try (CheckIndex checker = new CheckIndex(dir)) {
        CheckIndex.Status status = checker.checkIndex();
        if (!status.clean) {
          throw new AssertionError(path + ": CheckIndex is not clean");
        }
      }
    }
  }
}
