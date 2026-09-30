import org.apache.lucene.index.CheckIndex;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.NumericDocValues;
import org.apache.lucene.index.StoredFields;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;

import java.nio.file.Path;

/**
 * Verifies {@code crates/lucene-search/examples/write_merge_reorder_fixture.rs} (the documents,
 * deletes and merge hooks of {@code GenMergeReorder}): one merged segment holding exactly the
 * documents whose number is not a multiple of 13 and whose {@code rank} is not a multiple of 11,
 * ordered by {@code rank} descending, each document's {@code rank}
 * doc value the one {@code GenMergeReorder.rank} gives its number, and a clean {@code CheckIndex}.
 */
public class VerifyMergeReorder {
  /** {@code GenMergeReorder.rank}. */
  static long rank(int i) {
    long x = (i + 1) * 2654435761L;
    return ((x ^ (x >>> 13)) % 100000) % 1000;
  }

  public static void main(String[] args) throws Exception {
    try (Directory dir = FSDirectory.open(Path.of(args[0]))) {
      try (DirectoryReader reader = DirectoryReader.open(dir)) {
        if (reader.leaves().size() != 1) {
          throw new AssertionError("one merged segment expected");
        }
        LeafReader leaf = reader.leaves().get(0).reader();
        StoredFields stored = leaf.storedFields();
        NumericDocValues rank = leaf.getNumericDocValues("rank");
        long lastRank = Long.MAX_VALUE;

        int expected = 0;
        for (int i = 0; i < 3 * 60; i++) {
          if (i % 13 != 0 && rank(i) % 11 != 0) {
            expected++;
          }
        }
        if (leaf.maxDoc() != expected || leaf.numDocs() != expected) {
          throw new AssertionError("expected " + expected + " documents, got " + leaf.maxDoc());
        }
        for (int d = 0; d < leaf.maxDoc(); d++) {
          int n = stored.document(d).getField("n").numericValue().intValue();
          if (rank.advance(d) != d || rank.longValue() != rank(n)) {
            throw new AssertionError("doc " + d + ": rank of d" + n);
          }
          long r = rank.longValue();
          if (n % 13 == 0 || r % 11 == 0) {
            throw new AssertionError("d" + n + " should not have been carried over");
          }
          if (r > lastRank) {
            throw new AssertionError("doc " + d + " (d" + n + ", rank " + r + ") out of order");
          }
          lastRank = r;

        }
      }
      try (CheckIndex check = new CheckIndex(dir)) {
        if (!check.checkIndex().clean) {
          throw new AssertionError("CheckIndex failed");
        }
      }
    }
    System.out.println("VerifyMergeReorder: ok");
  }
}
