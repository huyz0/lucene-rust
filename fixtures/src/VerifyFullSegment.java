import org.apache.lucene.index.CheckIndex;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.MultiDocValues;
import org.apache.lucene.index.MultiTerms;
import org.apache.lucene.index.NumericDocValues;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.Terms;
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
 * Reverse-direction verifier (Rust writes, Java reads) for a whole index, not
 * a single codec file: opens the directory
 * {@code write_full_segment_fixture} produced with a real
 * {@link DirectoryReader} and runs {@link CheckIndex} over it at full level.
 *
 * <p>Every other verifier here hands Lucene one codec file with a hand-built
 * {@code SegmentInfo}/{@code FieldInfos}, which scopes each check to one
 * format and, by construction, cannot see anything that binds those files into
 * a segment: per-field format routing and the {@code .fnm} attributes
 * recording it, the {@code .psm} metadata file, cross-file lengths declared in
 * {@code .tmd}, or a {@code .fnm} promising norms the segment lacks. All four
 * were broken simultaneously while all thirteen single-format checks passed.
 *
 * <p>The trap this closes is that Lucene does not always *fail* on such a
 * segment. With no postings format registered against a field it reports the
 * field as having no terms and raises nothing at all, so this verifier asserts
 * on the term and document counts rather than merely on opening cleanly.
 *
 * <p>Usage: {@code java VerifyFullSegment <index-dir>}. Exits nonzero with a
 * diagnosis on any mismatch.
 */
public class VerifyFullSegment {
  /** Must match `write_full_segment_fixture.rs`. */
  private static final int NUM_DOCS = 2500;

  public static void main(String[] args) throws IOException {
    Path path = Path.of(args[0]);
    int failures = 0;

    try (Directory dir = FSDirectory.open(path);
        DirectoryReader reader = DirectoryReader.open(dir)) {
      if (reader.maxDoc() != NUM_DOCS || reader.numDocs() != NUM_DOCS) {
        System.out.println(
            "MISMATCH doc count: maxDoc=" + reader.maxDoc() + " numDocs=" + reader.numDocs()
                + " expected " + NUM_DOCS);
        failures++;
      }

      // The silent-failure case: a field with no registered postings format
      // reads back as absent, not as an error.
      Terms terms = MultiTerms.getTerms(reader, "body");
      if (terms == null) {
        System.out.println(
            "MISMATCH field \"body\" has no terms at all -- typically a missing "
                + "PerFieldPostingsFormat.format/.suffix attribute in .fnm");
        failures++;
      } else {
        if (terms.size() <= 0) {
          System.out.println("MISMATCH field \"body\" reports " + terms.size() + " terms");
          failures++;
        }
        // "shared" is in every document by construction.
        // count(), not search(): a TopDocs total is a *lower bound* once
        // dynamic pruning kicks in past the hit threshold, so asserting on it
        // would flag correct pruning as a mismatch.
        IndexSearcher searcher = new IndexSearcher(reader);
        int matched = searcher.count(new TermQuery(new Term("body", "shared")));
        if (matched != NUM_DOCS) {
          System.out.println("MISMATCH body:shared matched " + matched + ", expected " + NUM_DOCS);
          failures++;
        }
      }

      // Doc values: the other per-field format, so the other half of the
      // file-naming and .fnm-attribute contract the postings check above
      // exercises. A field with no format registered reads back as absent
      // rather than as an error, same silent shape as postings.
      NumericDocValues dv = MultiDocValues.getNumericValues(reader, "score");
      if (dv == null) {
        System.out.println(
            "MISMATCH field \"score\" has no numeric doc values -- typically a missing "
                + "PerFieldDocValuesFormat.format/.suffix attribute in .fnm");
        failures++;
      } else {
        int seen = 0;
        for (int doc = dv.nextDoc(); doc != NumericDocValues.NO_MORE_DOCS; doc = dv.nextDoc()) {
          long expected = (long) doc * 3 - 1000;
          if (dv.longValue() != expected) {
            System.out.println(
                "MISMATCH score for doc " + doc + ": " + dv.longValue() + " != " + expected);
            failures++;
            break;
          }
          seen++;
        }
        if (seen != NUM_DOCS) {
          System.out.println("MISMATCH score doc values covered " + seen + " of " + NUM_DOCS);
          failures++;
        }
      }

      // Stored fields must survive alongside the postings.
      String id = reader.storedFields().document(NUM_DOCS - 1).get("id");
      if (!("doc" + (NUM_DOCS - 1)).equals(id)) {
        System.out.println("MISMATCH last document's id=" + id);
        failures++;
      }
    }

    // Full-level CheckIndex last: it is the broadest check but the least
    // specific about what went wrong, so the targeted assertions run first.
    try (Directory dir = FSDirectory.open(path);
        CheckIndex checker = new CheckIndex(dir)) {
      ByteArrayOutputStream captured = new ByteArrayOutputStream();
      checker.setInfoStream(new PrintStream(captured, true, StandardCharsets.UTF_8));
      checker.setLevel(CheckIndex.Level.MIN_LEVEL_FOR_SLOW_CHECKS);
      CheckIndex.Status status = checker.checkIndex();
      if (!status.clean) {
        System.out.println("MISMATCH CheckIndex reported the index unclean:");
        System.out.println(captured.toString(StandardCharsets.UTF_8));
        failures++;
      }
    }

    if (failures > 0) {
      System.out.println(failures + " check(s) failed");
      System.exit(1);
    }
    System.out.println("Full segment verified against real Lucene. PASS");
  }
}
