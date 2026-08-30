import org.apache.lucene.index.CheckIndex;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.NumericDocValues;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.Terms;
import org.apache.lucene.index.TermsEnum;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.Sort;
import org.apache.lucene.search.SortField;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.BytesRef;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

/**
 * Reverse-direction verifier (Rust writes, Java reads) for an
 * <b>index-sorted</b> segment written by this port's {@code IndexWriter} with
 * {@code set_index_sort}.
 *
 * <p>An index sort is unlike every other property these verifiers check,
 * because getting it wrong produces a segment in which <i>every file is
 * valid</i>. The checksums pass, every doc id is in range, every term
 * dictionary decodes -- only the association between the files is wrong, and
 * only if you know what order the documents were supposed to be in. Three
 * independent things are therefore asserted here:
 *
 * <ol>
 *   <li>{@code LeafMetaData.sort()} round-trips the exact two-tier
 *       {@code Sort} the Rust writer configured, tier for tier, including
 *       each tier's reverse flag and missing value. That is the {@code .si}'s
 *       {@code SortFieldProvider} block read by real Lucene.
 *   <li>the documents come back in <b>that</b> order: Java re-derives the
 *       expected permutation from the fixture's own generator functions,
 *       using its own comparator, and compares it against the stored
 *       {@code id} of every doc id in turn. The sort is deliberately the
 *       awkward one -- a reversed first tier, so its <i>missing</i>
 *       documents belong at the front, not the back.
 *   <li>every other format still describes the same document at the same doc
 *       id: the two doc-values columns, the unique postings term, the norm
 *       (whose value is a function of the document's field length), and the
 *       vector (whose first component is the document's ordinal). A
 *       permutation applied to some formats and not others is invisible to
 *       {@code CheckIndex} and shows up only here.
 * </ol>
 *
 * <p>Then {@code CheckIndex} at {@code MIN_LEVEL_FOR_SLOW_CHECKS}, which runs
 * Lucene's own {@code testSort} -- it rebuilds the sort's comparators from the
 * {@code .si} and walks adjacent doc ids asserting the order. That is the
 * check that catches a comparator whose missing-value semantics disagree with
 * the sort it wrote.
 *
 * <p>The same class verifies both an index-sorted segment produced by a single
 * <b>flush</b> and one produced by a <b>merge</b> of several internally-sorted
 * flushes, deliberately: the claim under test is that the two are
 * indistinguishable. The optional second argument is the fixture's deletion
 * rule (every {@code deletedEvery}-th document was deleted before the merge),
 * which is the only thing that differs -- the expected permutation is then
 * derived over the survivors.
 *
 * <p>Usage: {@code java VerifySortedSegment <index-dir> [deletedEvery]}.
 */
public class VerifySortedSegment {
  /** Must match `write_sorted_segment_fixture.rs`. */
  private static final int NUM_DOCS = 2_000;

  private static final int MISSING_EVERY = 37;
  private static final int DIM = 8;

  private static boolean hasRank(int i) {
    return i % MISSING_EVERY != 0;
  }

  private static long rankOf(int i) {
    return ((long) i * 7919) % 50 - 20;
  }

  private static long tieOf(int i) {
    return ((long) i * 104_729) % NUM_DOCS;
  }

  private static float[] vectorOf(int i) {
    float[] v = new float[DIM];
    v[0] = i;
    for (int k = 1; k < DIM; k++) {
      v[k] = (((long) i * 31 + (long) k * 17) % 1000) / 1000.0f;
    }
    return v;
  }

  /** The field length the Rust side gave document {@code i}: "shared u<i>" + (i%5) x " pad". */
  private static int lengthOf(int i) {
    return 2 + (i % 5);
  }

  public static void main(String[] args) throws IOException {
    Path path = Path.of(args[0]);
    // 0 (or absent) means the fixture deleted nothing; otherwise every
    // deletedEvery-th document was deleted before the segments were merged,
    // and only the survivors are expected in the index.
    int deletedEvery = args.length > 1 ? Integer.parseInt(args[1]) : 0;
    int failures = 0;

    // The order Lucene's own semantics say the documents must be in: rank
    // descending with missing == Long.MAX_VALUE (so missing docs come FIRST
    // once reversed), then tie ascending.
    List<Integer> expected = new ArrayList<>();
    for (int i = 0; i < NUM_DOCS; i++) {
      if (deletedEvery > 0 && i % deletedEvery == 0) {
        continue;
      }
      expected.add(i);
    }
    expected.sort(
        Comparator.<Integer>comparingLong(i -> hasRank(i) ? rankOf(i) : Long.MAX_VALUE)
            .reversed()
            .thenComparingLong(VerifySortedSegment::tieOf));

    try (Directory dir = FSDirectory.open(path);
        DirectoryReader reader = DirectoryReader.open(dir)) {
      if (reader.leaves().size() != 1) {
        System.out.println("MISMATCH expected one segment, got " + reader.leaves().size());
        System.exit(1);
      }
      LeafReaderContext ctx = reader.leaves().get(0);
      LeafReader leaf = ctx.reader();

      int liveDocs = expected.size();
      if (leaf.maxDoc() != liveDocs) {
        System.out.println("MISMATCH maxDoc=" + leaf.maxDoc() + " expected " + liveDocs);
        failures++;
      }
      if (leaf.numDeletedDocs() != 0) {
        System.out.println(
            "MISMATCH the merge must drop deleted documents, but "
                + leaf.numDeletedDocs()
                + " are still marked deleted");
        failures++;
      }

      // (1) The sort itself.
      Sort sort = leaf.getMetaData().sort();
      if (sort == null) {
        System.out.println(
            "MISMATCH LeafMetaData.sort() is null -- the .si records no index sort");
        failures++;
      } else {
        SortField[] tiers = sort.getSort();
        if (tiers.length != 2) {
          System.out.println("MISMATCH sort has " + tiers.length + " tiers, expected 2");
          failures++;
        } else {
          failures += checkTier(tiers[0], "rank", true, Long.MAX_VALUE);
          failures += checkTier(tiers[1], "tie", false, Long.MIN_VALUE);
        }
      }

      // (2) The physical order.
      int mismatched = 0;
      for (int d = 0; d < leaf.maxDoc() && mismatched < 5; d++) {
        String got = leaf.storedFields().document(d).get("id");
        String want = "doc" + expected.get(d);
        if (!want.equals(got)) {
          System.out.println("MISMATCH docID=" + d + " holds " + got + ", expected " + want);
          mismatched++;
          failures++;
        }
      }

      // (3) Every other format, per doc id.
      NumericDocValues ranks = leaf.getNumericDocValues("rank");
      NumericDocValues ties = leaf.getNumericDocValues("tie");
      if (ranks == null || ties == null) {
        System.out.println("MISMATCH a doc-values column is missing (rank/tie)");
        failures++;
      } else {
        failures += checkColumn(ranks, expected, true);
        failures += checkColumn(ties, expected, false);
      }

      NumericDocValues norms = leaf.getNormValues("body");
      if (norms == null) {
        System.out.println("MISMATCH field \"body\" has no norms");
        failures++;
      } else {
        int bad = 0;
        for (int d = 0; d < leaf.maxDoc() && bad < 5; d++) {
          if (norms.advanceExact(d) == false) {
            System.out.println("MISMATCH no norm for docID=" + d);
            bad++;
            failures++;
            continue;
          }
          // The stored norm byte is compared directly against the length,
          // which is exact only because SmallFloat.intToByte4 is the identity
          // below 8 and every length here is 2..6.
          long want = lengthOf(expected.get(d));
          if (norms.longValue() != want) {
            System.out.println(
                "MISMATCH norm for docID=" + d + " is " + norms.longValue() + ", expected " + want);
            bad++;
            failures++;
          }
        }
      }

      Terms terms = leaf.terms("body");
      if (terms == null) {
        System.out.println("MISMATCH field \"body\" has no terms");
        failures++;
      } else {
        TermsEnum te = terms.iterator();
        int bad = 0;
        // Every 97th document, so the check is cheap but spread across the
        // whole term dictionary and every postings block.
        for (int d = 0; d < leaf.maxDoc() && bad < 5; d += 97) {
          int original = expected.get(d);
          BytesRef term = new BytesRef("u" + original);
          if (te.seekExact(term) == false) {
            System.out.println("MISMATCH term " + term.utf8ToString() + " is not in the dictionary");
            bad++;
            failures++;
            continue;
          }
          var postings = te.postings(null);
          int got = postings.nextDoc();
          if (got != d || postings.nextDoc() != DocIdSetIterator.NO_MORE_DOCS) {
            System.out.println(
                "MISMATCH term " + term.utf8ToString() + " lands on docID=" + got
                    + ", expected exactly " + d);
            bad++;
            failures++;
          }
        }
      }

      FloatVectorValues vectors = leaf.getFloatVectorValues("v");
      if (vectors == null) {
        System.out.println("MISMATCH field \"v\" has no float vectors");
        failures++;
      } else {
        var iter = vectors.iterator();
        int seen = 0;
        int bad = 0;
        for (int d = iter.nextDoc(); d != DocIdSetIterator.NO_MORE_DOCS; d = iter.nextDoc()) {
          seen++;
          if (bad >= 5) {
            continue;
          }
          float[] got = vectors.vectorValue(iter.index());
          float[] want = vectorOf(expected.get(d));
          for (int k = 0; k < DIM; k++) {
            if (got[k] != want[k]) {
              System.out.println(
                  "MISMATCH vector on docID=" + d + " component " + k + ": " + got[k]
                      + " != " + want[k] + " (the vector belongs to another document)");
              bad++;
              failures++;
              break;
            }
          }
        }
        if (seen != liveDocs) {
          System.out.println("MISMATCH vector values covered " + seen + " of " + liveDocs);
          failures++;
        }
        // And a real KNN query over the Rust-built graph, whose ordinals live
        // in the sorted space: the nearest vector to document `probe`'s own
        // vector is that document itself.
        IndexSearcher searcher = new IndexSearcher(reader);
        int probe = Math.min(733, liveDocs - 1);
        TopDocs top =
            searcher.search(new KnnFloatVectorQuery("v", vectorOf(expected.get(probe)), 1), 1);
        if (top.scoreDocs.length != 1 || top.scoreDocs[0].doc != probe) {
          System.out.println(
              "MISMATCH KnnFloatVectorQuery for docID=" + probe + " returned "
                  + (top.scoreDocs.length == 0 ? "nothing" : String.valueOf(top.scoreDocs[0].doc)));
          failures++;
        }
      }

      // A term query still resolves, i.e. the whole postings list is intact.
      IndexSearcher searcher = new IndexSearcher(reader);
      int matched = searcher.count(new org.apache.lucene.search.TermQuery(new Term("body", "shared")));
      if (matched != liveDocs) {
        System.out.println("MISMATCH body:shared matched " + matched + ", expected " + liveDocs);
        failures++;
      }
      // A deleted document's unique term must be gone from the merged
      // dictionary entirely, not merely unreachable: the postings merge
      // rebuilds the term dictionary from the surviving documents.
      if (deletedEvery > 0) {
        Terms bodyTerms = leaf.terms("body");
        TermsEnum te = bodyTerms.iterator();
        int bad = 0;
        for (int i = 0; i < NUM_DOCS && bad < 5; i += deletedEvery) {
          if (te.seekExact(new BytesRef("u" + i))) {
            System.out.println("MISMATCH deleted document " + i + "'s term survived the merge");
            bad++;
            failures++;
          }
        }
      }
    }

    // CheckIndex last: it runs Lucene's own testSort, which rebuilds the
    // comparators from the .si and walks adjacent doc ids.
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
    System.out.println("Sorted segment verified against real Lucene. PASS");
  }

  private static int checkTier(SortField tier, String field, boolean reverse, long missing) {
    int failures = 0;
    if (!field.equals(tier.getField())) {
      System.out.println("MISMATCH sort tier field=" + tier.getField() + ", expected " + field);
      failures++;
    }
    if (tier.getType() != SortField.Type.LONG) {
      System.out.println("MISMATCH sort tier " + field + " type=" + tier.getType());
      failures++;
    }
    if (tier.getReverse() != reverse) {
      System.out.println("MISMATCH sort tier " + field + " reverse=" + tier.getReverse());
      failures++;
    }
    if (!Long.valueOf(missing).equals(tier.getMissingValue())) {
      System.out.println(
          "MISMATCH sort tier " + field + " missingValue=" + tier.getMissingValue()
              + ", expected " + missing);
      failures++;
    }
    return failures;
  }

  /** Each doc id's column value must be the one belonging to the document that landed there. */
  private static int checkColumn(NumericDocValues values, List<Integer> expected, boolean isRank)
      throws IOException {
    int failures = 0;
    int bad = 0;
    for (int d = 0; d < expected.size() && bad < 5; d++) {
      int original = expected.get(d);
      boolean present = isRank ? hasRank(original) : true;
      boolean got = values.advanceExact(d);
      if (got != present) {
        System.out.println(
            "MISMATCH " + (isRank ? "rank" : "tie") + " presence at docID=" + d + ": " + got);
        bad++;
        failures++;
        continue;
      }
      if (present) {
        long want = isRank ? rankOf(original) : tieOf(original);
        if (values.longValue() != want) {
          System.out.println(
              "MISMATCH " + (isRank ? "rank" : "tie") + " at docID=" + d + ": "
                  + values.longValue() + " != " + want);
          bad++;
          failures++;
        }
      }
    }
    return failures;
  }
}
