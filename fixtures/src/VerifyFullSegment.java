import org.apache.lucene.index.CheckIndex;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FreqAndNormBuffer;
import org.apache.lucene.index.Impacts;
import org.apache.lucene.index.ImpactsEnum;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.PostingsEnum;
import org.apache.lucene.index.TermsEnum;
import org.apache.lucene.index.MultiDocValues;
import org.apache.lucene.index.MultiTerms;
import org.apache.lucene.index.NumericDocValues;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.Terms;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.SmallFloat;

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

  /**
   * `title`'s token count for document {@code i}, or {@code -1} when the
   * document does not carry the field at all. Must match
   * {@code write_full_segment_fixture.rs}'s {@code title_for}.
   */
  private static int titleLength(int i) {
    int n = i % 7;
    if (n == 0) {
      return -1; // no `title` field on this document at all: no norm
    }
    if (n == 1) {
      return 0; // present but empty: an explicit zero norm
    }
    return n;
  }

  /**
   * {@code body}'s token count: {@code shared} repeated {@code 1 + (i % 4)}
   * times plus two vocabulary terms. Must match
   * {@code write_full_segment_fixture.rs}. The repetition is what makes
   * {@code body:shared}'s per-block competitive impacts a *frontier* of
   * several {@code (freq, norm)} pairs -- see that file's comment.
   */
  private static int bodyLength(int i) {
    return 1 + (i % 4) + 2;
  }

  /**
   * Walks {@code body:shared}'s postings through real Lucene's own
   * {@link ImpactsEnum} and requires that at least one level-0 block carries
   * **more than one** competitive {@code (freq, norm)} pair.
   *
   * <p>Every rule impacts must obey is checked by {@code CheckIndex}
   * ({@code checkImpacts}); what that cannot tell anyone is whether the
   * segment under test contains a multi-impact block at all. Until item 18
   * this port wrote exactly one {@code (maxFreq, 1)} pair per block, so every
   * one of those rules was vacuously satisfied.
   */
  private static int checkImpactsHaveSeveralEntries(DirectoryReader reader) throws IOException {
    int mostImpactsSeen = 0;
    for (LeafReaderContext leaf : reader.leaves()) {
      Terms terms = leaf.reader().terms("body");
      if (terms == null) {
        continue;
      }
      TermsEnum te = terms.iterator();
      if (!te.seekExact(new org.apache.lucene.util.BytesRef("shared"))) {
        continue;
      }
      ImpactsEnum impactsEnum = te.impacts(PostingsEnum.FREQS);
      for (int doc = impactsEnum.nextDoc();
          doc != PostingsEnum.NO_MORE_DOCS;
          doc = impactsEnum.nextDoc()) {
        impactsEnum.advanceShallow(doc);
        Impacts impacts = impactsEnum.getImpacts();
        for (int level = 0; level < impacts.numLevels(); ++level) {
          FreqAndNormBuffer buf = impacts.getImpacts(level);
          mostImpactsSeen = Math.max(mostImpactsSeen, buf.size);
        }
      }
    }
    if (mostImpactsSeen < 2) {
      System.out.println(
          "MISMATCH body:shared's richest impacts list had " + mostImpactsSeen
              + " entry/entries -- the fixture no longer exercises multi-impact blocks, "
              + "so CheckIndex's impact rules are being checked against nothing");
      return 1;
    }
    return 0;
  }

  /**
   * Reads {@code field}'s norm for every document through real Lucene and
   * compares it to {@code SmallFloat.intToByte4(length)} -- the value
   * {@code BM25Similarity.computeNorm} would have stored.
   *
   * <p>The absent-value case is checked too: {@code NumericDocValues} skips a
   * document with no norm, which is what {@code NormValuesWriter}'s
   * {@code DocsWithFieldSet} produces for a document that does not carry the
   * field, and is distinguishable from the explicit {@code 0} a
   * present-but-empty field gets.
   */
  private static int checkNorms(DirectoryReader reader, String field) throws IOException {
    NumericDocValues norms = MultiDocValues.getNormValues(reader, field);
    if (norms == null) {
      System.out.println(
          "MISMATCH field \"" + field + "\" has no norms -- typically an .fnm claiming "
              + "omitNorms, or a missing .nvm/.nvd");
      return 1;
    }
    boolean[] seen = new boolean[NUM_DOCS];
    for (int doc = norms.nextDoc(); doc != NumericDocValues.NO_MORE_DOCS; doc = norms.nextDoc()) {
      int length = field.equals("body") ? bodyLength(doc) : titleLength(doc);
      if (length < 0) {
        System.out.println(
            "MISMATCH " + field + " doc " + doc + " has a norm but does not carry the field");
        return 1;
      }
      long expected = SmallFloat.intToByte4(length);
      if (norms.longValue() != expected) {
        System.out.println(
            "MISMATCH " + field + " norm for doc " + doc + ": " + norms.longValue()
                + " != " + expected + " (length " + length + ")");
        return 1;
      }
      seen[doc] = true;
    }
    for (int doc = 0; doc < NUM_DOCS; doc++) {
      boolean expectNorm = field.equals("body") || titleLength(doc) >= 0;
      if (seen[doc] != expectNorm) {
        System.out.println(
            "MISMATCH " + field + " doc " + doc + (expectNorm ? " is missing its norm"
                : " has a norm it should not have"));
        return 1;
      }
    }
    return 0;
  }

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

      // Norms, for a field NOTHING opted into them. Real Lucene writes a
      // norm for every indexed field whose `omitNorms` is false; this port
      // required an explicit per-field opt-in until c35 and forced
      // `omit_norms: true` into the `.fnm` for every other indexed field, so
      // a caller who indexed a text field and searched it got BM25 against a
      // constant length. Compared value by value rather than merely asserting
      // the column exists: a column of the wrong lengths reads back perfectly.
      failures += checkNorms(reader, "title");
      // And the field that *was* configured, whose length varies per document
      // (`shared` repeated 1..4 times plus two vocabulary terms).
      failures += checkNorms(reader, "body");
      // The impacts this port now writes, read back through real Lucene's own
      // `ImpactsEnum`. `CheckIndex` below validates their *rules* (ordering,
      // non-empty, non-zero first norm, freq <= block max); this asserts the
      // shape is actually exercised, so a future change that collapsed the
      // frontier back to one pair would be visible rather than silently
      // untested.
      failures += checkImpactsHaveSeveralEntries(reader);

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
