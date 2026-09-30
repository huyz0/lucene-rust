import org.apache.lucene.index.CheckIndex;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.NumericDocValues;
import org.apache.lucene.index.SegmentCommitInfo;
import org.apache.lucene.index.SegmentInfos;
import org.apache.lucene.index.StoredFields;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.PhraseQuery;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;

import java.nio.file.Path;

/**
 * Verifies {@code crates/lucene-index/examples/write_compound_segment_fixture.rs}:
 * {@code <dir>/flushed} must hold three compound segments and {@code
 * <dir>/merged} one, every one of them opened through Lucene's own {@code
 * Lucene90CompoundFormat} reader -- stored ids, the postings (a term in every
 * document, a phrase), norms-backed scoring, and the numeric doc values of
 * every document -- and clean under {@code CheckIndex}.
 */
public class VerifyCompoundSegments {
  static final int PER_SEGMENT = 400;
  static final int SEGMENTS = 3;

  public static void main(String[] args) throws Exception {
    verify(Path.of(args[0]).resolve("flushed"), SEGMENTS);
    verify(Path.of(args[0]).resolve("merged"), 1);
    System.out.println("VerifyCompoundSegments: ok");
  }

  static void verify(Path path, int segments) throws Exception {
    try (Directory dir = FSDirectory.open(path)) {
      SegmentInfos sis = SegmentInfos.readLatestCommit(dir);
      if (sis.size() != segments) {
        throw new AssertionError(path + ": expected " + segments + " segments, got " + sis.size());
      }
      for (SegmentCommitInfo sci : sis) {
        if (!sci.info.getUseCompoundFile()) {
          throw new AssertionError(sci.info.name + " is not compound");
        }
        for (String f : sci.files()) {
          if (!f.endsWith(".cfs") && !f.endsWith(".cfe") && !f.endsWith(".si")) {
            throw new AssertionError(sci.info.name + " lists a loose file " + f);
          }
        }
      }
      int total = PER_SEGMENT * SEGMENTS;
      try (DirectoryReader reader = DirectoryReader.open(dir)) {
        if (reader.numDocs() != total) {
          throw new AssertionError("numDocs " + reader.numDocs());
        }
        IndexSearcher searcher = new IndexSearcher(reader);
        if (searcher.count(new TermQuery(new Term("body", "shared"))) != total) {
          throw new AssertionError("shared does not match every document");
        }
        int w5 = 0;
        for (int i = 0; i < total; i++) {
          if (i % 13 == 5) w5++;
        }
        if (searcher.count(new TermQuery(new Term("body", "a5"))) != w5) {
          throw new AssertionError("a5 count");
        }
        if (searcher.count(new PhraseQuery("body", "shared", "a5")) != w5) {
          throw new AssertionError("phrase count");
        }
        if (searcher.search(new TermQuery(new Term("body", "b7")), 10).scoreDocs.length == 0) {
          throw new AssertionError("no scored hits");
        }
        // Every id once, each with its own doc value. (A merge may order its
        // sources by size, as TieredMergePolicy's forced merges do, so the
        // merged segment need not keep insertion order.)
        boolean[] seen = new boolean[total];
        for (LeafReaderContext ctx : reader.leaves()) {
          LeafReader leaf = ctx.reader();
          StoredFields stored = leaf.storedFields();
          NumericDocValues num = leaf.getNumericDocValues("num");
          for (int d = 0; d < leaf.maxDoc(); d++) {
            int i = Integer.parseInt(stored.document(d).get("id").substring(3));
            if (seen[i]) {
              throw new AssertionError(path + ": doc " + d + " of a segment has id doc" + i);
            }
            seen[i] = true;
            if (num.advance(d) != d || num.longValue() != i * 7L - 500) {
              throw new AssertionError("doc " + i + " doc value");
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
