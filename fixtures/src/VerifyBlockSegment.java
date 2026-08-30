import org.apache.lucene.index.CheckIndex;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.PostingsEnum;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;

/**
 * Reverse-direction verifier (Rust writes, Java reads) for a segment built out
 * of <b>document blocks</b> -- runs of documents added by a single
 * {@code IndexWriter.addDocuments} call, which Lucene records by setting
 * {@code SegmentInfo.hasBlocks}.
 *
 * <p>Two things are checked that nothing else in this suite can see:
 *
 * <ol>
 *   <li>{@code LeafMetaData.hasBlocks()} is <b>true</b>. This is one byte in
 *       the {@code .si}, and a segment that carries blocks but reports
 *       {@code false} reads back perfectly while silently making every
 *       parent/child join query against it invalid. Nothing errors; the flag is
 *       the only evidence.
 *   <li>The blocks are actually contiguous: every parent document's doc ID is a
 *       multiple of the block size, in ascending order, which is the property
 *       {@code hasBlocks} promises a join query it may rely on.
 *   <li>The {@code .liv} written by a <b>buffered delete</b> is real and
 *       consistent: {@code numDocs} is {@code maxDoc - NUM_DELETED}, and the
 *       deleted block's term matches nothing through a searcher. Unlike this
 *       port's doc-values-update overlay, {@code .liv} is genuine Lucene
 *       format, so this is the one place real Lucene can check the
 *       buffered-delete path's output -- including that the {@code delCount}
 *       recorded in {@code segments_N} agrees with the bitset, which
 *       {@link CheckIndex} validates.
 * </ol>
 *
 * <p>{@link CheckIndex} at full level runs on top, to confirm setting the flag
 * did not break anything else about the segment.
 *
 * <p>Usage: {@code java VerifyBlockSegment <index-dir>}. Exits nonzero with a
 * diagnosis on any mismatch.
 */
public class VerifyBlockSegment {
  /** Must match `write_block_segment_fixture.rs`. */
  private static final int NUM_BLOCKS = 300;

  private static final int BLOCK_SIZE = 4;

  /** One extra block, added last and deleted by term. */
  private static final int NUM_DELETED = BLOCK_SIZE;

  private static final int NUM_DOCS = (NUM_BLOCKS + 1) * BLOCK_SIZE;
  private static final int NUM_LIVE = NUM_DOCS - NUM_DELETED;

  public static void main(String[] args) throws IOException {
    Path path = Path.of(args[0]);
    int failures = 0;

    try (Directory dir = FSDirectory.open(path);
        DirectoryReader reader = DirectoryReader.open(dir)) {
      if (reader.maxDoc() != NUM_DOCS || reader.numDocs() != NUM_LIVE) {
        System.out.println(
            "MISMATCH doc count: maxDoc="
                + reader.maxDoc()
                + " numDocs="
                + reader.numDocs()
                + " expected maxDoc="
                + NUM_DOCS
                + " numDocs="
                + NUM_LIVE);
        failures++;
      }

      // The buffered delete: the tombstone block's term must match nothing
      // through a searcher, which consults live docs -- while the postings
      // themselves still carry the deleted doc IDs.
      IndexSearcher searcher = new IndexSearcher(reader);
      int stillMatching = searcher.count(new TermQuery(new Term("body", "tombstone")));
      if (stillMatching != 0) {
        System.out.println(
            "MISMATCH body:tombstone still matches " + stillMatching + " live docs after delete");
        failures++;
      }

      for (LeafReaderContext ctx : reader.leaves()) {
        if (!ctx.reader().getMetaData().hasBlocks()) {
          System.out.println(
              "MISMATCH hasBlocks: segment reports false, but every document was added"
                  + " through addDocuments()");
          failures++;
        }
      }

      // Contiguity: `parent shared` matches exactly the first document of each
      // block, so the parent doc IDs must be 0, BLOCK_SIZE, 2*BLOCK_SIZE, ...
      LeafReaderContext leaf = reader.leaves().get(0);
      PostingsEnum parents =
          leaf.reader().postings(new Term("body", "parent"), PostingsEnum.NONE);
      if (parents == null) {
        System.out.println("MISSING postings for body:parent");
        failures++;
      } else {
        int expected = 0;
        int seen = 0;
        int docID;
        while ((docID = parents.nextDoc()) != DocIdSetIterator.NO_MORE_DOCS) {
          if (docID != expected) {
            System.out.println(
                "MISMATCH block start: parent #" + seen + " is doc " + docID + ", expected "
                    + expected);
            failures++;
            break;
          }
          expected += BLOCK_SIZE;
          seen++;
        }
        if (seen != NUM_BLOCKS) {
          System.out.println("MISMATCH parent count: " + seen + " expected " + NUM_BLOCKS);
          failures++;
        }
      }
    }

    try (Directory dir = FSDirectory.open(path);
        CheckIndex checker = new CheckIndex(dir)) {
      ByteArrayOutputStream captured = new ByteArrayOutputStream();
      checker.setInfoStream(new PrintStream(captured, true, StandardCharsets.UTF_8), false);
      checker.setLevel(CheckIndex.Level.MIN_LEVEL_FOR_SLOW_CHECKS);
      CheckIndex.Status status = checker.checkIndex();
      if (!status.clean) {
        System.out.println("CheckIndex reported problems:");
        System.out.println(captured.toString(StandardCharsets.UTF_8));
        failures++;
      }
    }

    if (failures == 0) {
      System.out.println(
          "VerifyBlockSegment: ok ("
              + NUM_DOCS
              + " docs in blocks of "
              + BLOCK_SIZE
              + ", "
              + NUM_DELETED
              + " deleted, hasBlocks set, CheckIndex clean)");
    }
    System.exit(failures == 0 ? 0 : 1);
  }
}
