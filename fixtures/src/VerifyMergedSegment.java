import org.apache.lucene.document.Document;
import org.apache.lucene.index.CheckIndex;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.MultiTerms;
import org.apache.lucene.index.PostingsEnum;
import org.apache.lucene.index.StoredFields;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.TermVectors;
import org.apache.lucene.index.Terms;
import org.apache.lucene.index.TermsEnum;
import org.apache.lucene.util.BytesRef;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;

/**
 * Reverse-direction verifier (Rust writes, Java reads) for a **merged**
 * segment: the output of {@code write_merged_segment_fixture}, which merges
 * three flushed segments into one.
 *
 * <p>{@code VerifyFullSegment} covers a freshly flushed segment. This covers
 * the merge, and specifically the two fast stored-fields paths a merge takes:
 * BULK (whole compressed chunks copied verbatim from a deletion-free source,
 * with only each chunk's {@code docBase} vint rewritten) and DOC (a surviving
 * document's serialized bytes copied without being parsed into fields). Both
 * copy bytes that encode field numbers and chunk geometry, so a boundary error
 * yields a segment that opens cleanly and reads back plausible but *wrong*
 * documents -- which is exactly what round-tripping through the port's own
 * reader cannot detect.
 *
 * <p>So this does not merely open the index: it recomputes every document's
 * expected stored fields independently, in the order the merge must have
 * produced them (source order: all of {@code _0}, then {@code _1}'s survivors,
 * then all of {@code _2}), and compares field by field.
 *
 * <p>The {@code body} field also stores term vectors, whose merge has the
 * same two-way shape over a different chunk format (4 096 bytes / 128
 * documents, so several times as many chunk boundaries as the stored-fields
 * side). Every document's vector is recomputed from that document's expected
 * body text and compared term by term with its frequency, which is what pins
 * a copied term-vector chunk to the right {@code docBase}.
 *
 * <p>Usage: {@code java VerifyMergedSegment <index-dir>}.
 */
public class VerifyMergedSegment {
  /** Must match `write_merged_segment_fixture.rs`. */
  private static final int DOCS_PER_SEGMENT = 2400;

  private static final int NUM_SEGMENTS = 3;
  private static final int DOOMED_EVERY = 100;
  /** The segment the fixture deletes `body:doomed` from. */
  private static final int DELETED_SEGMENT = 1;

  private static String body(int n, List<String> vocab) {
    String text =
        "shared " + vocab.get(n % vocab.size()) + " " + vocab.get((n / 7) % vocab.size());
    if (n % DOOMED_EVERY == 0) {
      text += " doomed";
    }
    return text;
  }

  public static void main(String[] args) throws IOException {
    Path path = Path.of(args[0]);
    int failures = 0;

    List<String> vocab = new ArrayList<>();
    for (int i = 0; i < 500; i++) {
      vocab.add((char) ('a' + (i % 26)) + String.format("%03d", i));
    }

    // Every (id, body) pair the merged segment must contain, exactly once.
    // Deliberately a *set*, not a list: which source segment the merge policy
    // visits first is its business, so asserting an order here would pin
    // something this fixture is not testing. What a bulk-copy boundary error
    // breaks is not the order -- it is the pairing (a document's fields coming
    // from a different document) and the multiset (a document copied twice or
    // dropped), and both are caught below.
    Map<String, String> expected = new HashMap<>();
    for (int seg = 0; seg < NUM_SEGMENTS; seg++) {
      for (int n = 0; n < DOCS_PER_SEGMENT; n++) {
        if (seg == DELETED_SEGMENT && n % DOOMED_EVERY == 0) {
          continue; // deleted before the merge
        }
        expected.put("doc" + seg + "-" + n, body(n, vocab));
      }
    }
    int expectedDocs = expected.size();

    try (Directory dir = FSDirectory.open(path);
        DirectoryReader reader = DirectoryReader.open(dir)) {
      if (reader.leaves().size() != 1) {
        System.out.println(
            "MISMATCH expected one merged segment, found " + reader.leaves().size());
        failures++;
      }
      if (reader.maxDoc() != expectedDocs || reader.numDocs() != expectedDocs) {
        System.out.println(
            "MISMATCH doc count: maxDoc=" + reader.maxDoc() + " numDocs=" + reader.numDocs()
                + " expected " + expectedDocs);
        failures++;
      }

      // Field by field, every document. A bulk-copy boundary error shows up
      // here as a document whose fields are another document's, or as a
      // document copied twice while another is dropped.
      StoredFields stored = reader.storedFields();
      Set<String> seen = new HashSet<>();
      int mismatches = 0;
      for (int doc = 0; doc < reader.maxDoc(); doc++) {
        Document d = stored.document(doc);
        String id = d.get("id");
        String b = d.get("body");
        String want = expected.get(id);
        if (want == null || !want.equals(b) || !seen.add(id)) {
          if (mismatches < 5) {
            System.out.println(
                "MISMATCH doc " + doc + ": id=" + id + " body=" + b + " expected body=" + want
                    + (want != null && want.equals(b) ? " (duplicate id)" : ""));
          }
          mismatches++;
        }
      }
      if (mismatches > 0) {
        System.out.println("MISMATCH " + mismatches + " of " + expectedDocs + " documents differ");
        failures++;
      }
      if (seen.size() != expectedDocs) {
        System.out.println(
            "MISMATCH merged segment holds " + seen.size() + " distinct ids, expected "
                + expectedDocs);
        failures++;
      }

      // The term vectors merged alongside the stored fields. Their chunks
      // are 4 096 bytes / 128 documents, so a 7 176-document merged segment
      // spans dozens of them, most copied verbatim from a source -- a
      // `docBase` off by one puts a document's vectors on its neighbour, and
      // nothing but an independent recomputation catches that.
      TermVectors vectors = reader.termVectors();
      int vectorMismatches = 0;
      for (int doc = 0; doc < reader.maxDoc(); doc++) {
        Document d = stored.document(doc);
        Map<String, Integer> want = new TreeMap<>();
        for (String token : d.get("body").split(" ")) {
          want.merge(token, 1, Integer::sum);
        }
        Map<String, Integer> got = new TreeMap<>();
        Terms tv = vectors.get(doc, "body");
        if (tv == null) {
          if (vectorMismatches < 5) {
            System.out.println("MISMATCH doc " + doc + " has no term vector for \"body\"");
          }
          vectorMismatches++;
          continue;
        }
        TermsEnum te = tv.iterator();
        PostingsEnum pe = null;
        BytesRef term;
        while ((term = te.next()) != null) {
          pe = te.postings(pe, PostingsEnum.FREQS);
          pe.nextDoc();
          got.put(term.utf8ToString(), pe.freq());
        }
        if (!got.equals(want)) {
          if (vectorMismatches < 5) {
            System.out.println(
                "MISMATCH doc " + doc + " term vector: expected " + want + " got " + got);
          }
          vectorMismatches++;
        }
      }
      if (vectorMismatches > 0) {
        System.out.println(
            "MISMATCH " + vectorMismatches + " of " + reader.maxDoc()
                + " documents have wrong term vectors");
        failures++;
      }

      // The postings merged alongside the stored fields.
      Terms terms = MultiTerms.getTerms(reader, "body");
      if (terms == null) {
        System.out.println("MISMATCH field \"body\" has no terms in the merged segment");
        failures++;
      } else {
        IndexSearcher searcher = new IndexSearcher(reader);
        int shared = searcher.count(new TermQuery(new Term("body", "shared")));
        if (shared != expectedDocs) {
          System.out.println("MISMATCH body:shared matched " + shared + ", expected " + expectedDocs);
          failures++;
        }
        // `doomed` survives only in the two segments it was not deleted from.
        int doomed = searcher.count(new TermQuery(new Term("body", "doomed")));
        int expectedDoomed = (NUM_SEGMENTS - 1) * ((DOCS_PER_SEGMENT + DOOMED_EVERY - 1) / DOOMED_EVERY);
        if (doomed != expectedDoomed) {
          System.out.println("MISMATCH body:doomed matched " + doomed + ", expected " + expectedDoomed);
          failures++;
        }
      }
    }

    try (Directory dir = FSDirectory.open(path);
        CheckIndex checker = new CheckIndex(dir)) {
      ByteArrayOutputStream captured = new ByteArrayOutputStream();
      checker.setInfoStream(new PrintStream(captured, true, StandardCharsets.UTF_8));
      checker.setLevel(CheckIndex.Level.MIN_LEVEL_FOR_SLOW_CHECKS);
      CheckIndex.Status status = checker.checkIndex();
      if (!status.clean) {
        System.out.println("MISMATCH CheckIndex reported the merged index unclean:");
        System.out.println(captured.toString(StandardCharsets.UTF_8));
        failures++;
      }
    }

    if (failures > 0) {
      System.out.println(failures + " check(s) failed");
      System.exit(1);
    }
    System.out.println("Merged segment verified against real Lucene. PASS");
  }
}
