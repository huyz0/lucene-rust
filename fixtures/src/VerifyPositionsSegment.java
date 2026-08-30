import org.apache.lucene.index.CheckIndex;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.PostingsEnum;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.Terms;
import org.apache.lucene.index.TermsEnum;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.PhraseQuery;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.BytesRef;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HexFormat;
import java.util.List;
import java.util.Properties;

/**
 * Reverse-direction verifier (Rust writes, Java reads) for a segment whose
 * postings carry <b>positions, offsets and payloads</b>: the {@code .pos} and
 * {@code .pay} files, which before batch c23 no Java reader had ever seen this
 * port produce.
 *
 * <p>{@code VerifyFullSegment} already opens a whole Rust-written index, but
 * its one indexed field is {@code DOCS_AND_FREQS}, so that index contains no
 * {@code .pos} at all. Batch c20 built the level-0/level-1 {@code .pos}/{@code
 * .pay} skip records on both sides and recorded in its own carry-over list
 * that the only evidence for them was two of this port's readers agreeing with
 * each other -- the evidence shape that let b4's FST framing bug and b11's
 * invented {@code .si} sort encoding both round-trip perfectly while being
 * wrong.
 *
 * <p>What this checks, in order of specificity:
 *
 * <ol>
 *   <li>The {@code .fnm} describes the four {@link IndexOptions} rungs and the
 *       payload bit the way the fixture declared them. Lucene frames {@code
 *       .pay} off {@code FieldInfo.hasPayloads()}, so a segment whose
 *       {@code .fnm} and postings disagree here decodes plausible garbage
 *       rather than failing.
 *   <li>Term statistics, and explicit <b>non-degeneracy</b>: the dense term
 *       must reach past two whole {@code LEVEL1_NUM_DOCS} spans and past a
 *       full {@code .pos} block, or the fixture would exercise none of the
 *       machinery it exists for. (c20's Tier-2 review found exactly this class
 *       of silent degeneracy in its own fixture.)
 *   <li>Occurrence-by-occurrence comparison against the manifest for 51
 *       sampled documents chosen around the format's block boundaries, with a
 *       <b>fresh {@link PostingsEnum} per sample</b> so every one is reached
 *       through {@code advance(doc)} and the skip records rather than by
 *       sequential iteration.
 *   <li>The payload rule recomputed independently in Java, so a payload that
 *       survives byte-identical but attached to the wrong occurrence fails.
 *   <li>A real {@link PhraseQuery}: positions that decode individually but
 *       land at the wrong absolute values still match every term query and no
 *       phrase.
 *   <li>{@link CheckIndex} at {@code MIN_LEVEL_FOR_SLOW_CHECKS}, which walks
 *       every term's positions and offsets for ordering and bounds.
 * </ol>
 *
 * <p>Usage: {@code java VerifyPositionsSegment <index-dir>}. Exits nonzero
 * with a diagnosis on any mismatch.
 */
public class VerifyPositionsSegment {
  /** {@code Lucene104PostingsFormat.BLOCK_SIZE}. */
  private static final int BLOCK_SIZE = 256;

  /** {@code Lucene104PostingsFormat.LEVEL1_NUM_DOCS}. */
  private static final int LEVEL1_NUM_DOCS = 32 * BLOCK_SIZE;

  private static int failures = 0;

  private static void fail(String message) {
    System.out.println("MISMATCH " + message);
    failures++;
  }

  public static void main(String[] args) throws IOException {
    Path path = Path.of(args[0]);
    Properties manifest = new Properties();
    try (var in = Files.newBufferedReader(path.resolve("positions-manifest.properties"))) {
      manifest.load(in);
    }
    int numDocs = Integer.parseInt(manifest.getProperty("num_docs"));

    try (Directory dir = FSDirectory.open(path);
        DirectoryReader reader = DirectoryReader.open(dir)) {
      if (reader.leaves().size() != 1) {
        fail("expected one segment, got " + reader.leaves().size());
      }
      LeafReaderContext leaf = reader.leaves().get(0);
      LeafReader leafReader = leaf.reader();
      if (leafReader.maxDoc() != numDocs) {
        fail("maxDoc=" + leafReader.maxDoc() + " expected " + numDocs);
      }

      checkFieldInfos(leafReader.getFieldInfos());
      checkTermStats(leafReader, manifest);
      checkSamples(leafReader, manifest);
      checkDocsOnlyField(leafReader);
      checkTermVectors(leafReader, manifest);

      // Positions are the only thing a phrase query can be right about.
      IndexSearcher searcher = new IndexSearcher(reader);
      int expectedPhrase = Integer.parseInt(manifest.getProperty("phrase_doc_count"));
      int gotPhrase =
          searcher.count(
              new PhraseQuery.Builder()
                  .add(new Term("body", "alpha"), 0)
                  .add(new Term("body", "beta"), 1)
                  .build());
      if (gotPhrase != expectedPhrase) {
        fail("PhraseQuery(body:\"alpha beta\") matched " + gotPhrase + ", expected "
            + expectedPhrase);
      }
      // The same two terms in the other order must match nothing: that is what
      // separates "positions decode" from "positions are correct".
      int reversed =
          searcher.count(
              new PhraseQuery.Builder()
                  .add(new Term("body", "beta"), 0)
                  .add(new Term("body", "alpha"), 1)
                  .build());
      if (reversed != 0) {
        fail("PhraseQuery(body:\"beta alpha\") matched " + reversed + ", expected 0");
      }
      int dense = searcher.count(new TermQuery(new Term("body", "dense")));
      if (dense != numDocs) {
        fail("body:dense matched " + dense + ", expected " + numDocs);
      }
    }

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
    System.out.println("Positions/offsets/payloads segment verified against real Lucene. PASS");
  }

  /**
   * The `.fnm` half of the contract. Lucene opens `.pay` when {@code
   * hasPayloads() || hasOffsets()} and frames every block's payload-length run
   * off {@code hasPayloads()} alone, so these bits are not documentation: they
   * decide how the bytes are read.
   */
  private static void checkFieldInfos(FieldInfos infos) {
    expectField(infos, "body", IndexOptions.DOCS_AND_FREQS_AND_POSITIONS_AND_OFFSETS, true);
    expectField(infos, "title", IndexOptions.DOCS_AND_FREQS_AND_POSITIONS, false);
    expectField(infos, "tag", IndexOptions.DOCS, false);
    expectField(infos, "count", IndexOptions.DOCS_AND_FREQS, false);
    // Offsets without payloads, and payloads without offsets: Lucene creates
    // `.pay` for either, but writes a different record for each, so a writer
    // that framed one as the other passes on `body` (which has both) and fails
    // here.
    expectField(infos, "head", IndexOptions.DOCS_AND_FREQS_AND_POSITIONS_AND_OFFSETS, false);
    expectField(infos, "notes", IndexOptions.DOCS_AND_FREQS_AND_POSITIONS, true);
  }

  private static void expectField(
      FieldInfos infos, String name, IndexOptions options, boolean payloads) {
    FieldInfo info = infos.fieldInfo(name);
    if (info == null) {
      fail("field \"" + name + "\" absent from .fnm");
      return;
    }
    if (info.getIndexOptions() != options) {
      fail("field \"" + name + "\" indexOptions=" + info.getIndexOptions() + " expected " + options);
    }
    if (info.hasPayloads() != payloads) {
      fail("field \"" + name + "\" hasPayloads=" + info.hasPayloads() + " expected " + payloads);
    }
  }

  private static void checkTermStats(LeafReader reader, Properties manifest) throws IOException {
    for (String key : manifest.stringPropertyNames()) {
      if (!key.startsWith("stat.") || !key.endsWith(".doc_freq")) {
        continue;
      }
      String rest = key.substring("stat.".length(), key.length() - ".doc_freq".length());
      int dot = rest.indexOf('.');
      String field = rest.substring(0, dot);
      String term = rest.substring(dot + 1);
      long wantDocFreq = Long.parseLong(manifest.getProperty(key));
      long wantTtf =
          Long.parseLong(manifest.getProperty("stat." + field + "." + term + ".total_term_freq"));

      Terms terms = reader.terms(field);
      if (terms == null) {
        fail("field \"" + field + "\" has no terms at all");
        continue;
      }
      TermsEnum te = terms.iterator();
      if (!te.seekExact(new BytesRef(term))) {
        fail("field \"" + field + "\" is missing term \"" + term + "\"");
        continue;
      }
      if (te.docFreq() != wantDocFreq) {
        fail(field + ":" + term + " docFreq=" + te.docFreq() + " expected " + wantDocFreq);
      }
      if (te.totalTermFreq() != wantTtf) {
        fail(field + ":" + term + " totalTermFreq=" + te.totalTermFreq() + " expected " + wantTtf);
      }
    }

    // Non-degeneracy: without these the samples below could all pass against a
    // fixture that never reaches a single skip record.
    long denseDocFreq = Long.parseLong(manifest.getProperty("stat.body.dense.doc_freq"));
    long denseTtf = Long.parseLong(manifest.getProperty("stat.body.dense.total_term_freq"));
    if (denseDocFreq < 2L * LEVEL1_NUM_DOCS) {
      fail("body:dense docFreq " + denseDocFreq + " does not span two level-1 spans ("
          + (2 * LEVEL1_NUM_DOCS) + "), so no level-1 skip record is exercised twice");
    }
    if (denseTtf < 2L * BLOCK_SIZE) {
      fail("body:dense totalTermFreq " + denseTtf + " never fills two .pos blocks");
    }
    if (denseTtf % BLOCK_SIZE == 0) {
      fail("body:dense totalTermFreq " + denseTtf + " is a whole number of .pos blocks, so the "
          + "vint tail is never written");
    }
  }

  private static void checkSamples(LeafReader reader, Properties manifest) throws IOException {
    HexFormat hex = HexFormat.of();
    int sampleCount = Integer.parseInt(manifest.getProperty("sample_count"));
    int crossing = Integer.parseInt(manifest.getProperty("dense_block_crossing_doc"));
    boolean sawCrossing = false;
    int payloadLengthsSeen = 0;
    boolean[] payloadLengthSeen = new boolean[8];

    for (int i = 0; i < sampleCount; i++) {
      int doc = Integer.parseInt(manifest.getProperty("sample." + i + ".doc"));
      if (doc == crossing) {
        sawCrossing = true;
      }
      int entryCount = Integer.parseInt(manifest.getProperty("sample." + i + ".entry_count"));
      for (int j = 0; j < entryCount; j++) {
        String[] parts = manifest.getProperty("sample." + i + "." + j).split("\\|", -1);
        String field = parts[0];
        String term = parts[1];
        List<int[]> wantPositions = new ArrayList<>();
        List<byte[]> wantPayloads = new ArrayList<>();
        for (String occ : parts[2].split(",", -1)) {
          String[] f = occ.split(":", -1);
          wantPositions.add(
              new int[] {Integer.parseInt(f[0]), Integer.parseInt(f[1]), Integer.parseInt(f[2])});
          wantPayloads.add(hex.parseHex(f[3]));
        }

        // A fresh enum per sampled document, so every sample is reached
        // through advance() and the skip records rather than by walking there.
        Terms terms = reader.terms(field);
        TermsEnum te = terms.iterator();
        if (!te.seekExact(new BytesRef(term))) {
          fail("field \"" + field + "\" is missing term \"" + term + "\"");
          continue;
        }
        PostingsEnum postings = te.postings(null, PostingsEnum.ALL);
        if (postings.advance(doc) != doc) {
          fail(field + ":" + term + " advance(" + doc + ") landed on " + postings.docID());
          continue;
        }
        if (postings.freq() != wantPositions.size()) {
          fail(field + ":" + term + " doc " + doc + " freq=" + postings.freq() + " expected "
              + wantPositions.size());
          continue;
        }
        for (int k = 0; k < wantPositions.size(); k++) {
          int[] want = wantPositions.get(k);
          int position = postings.nextPosition();
          if (position != want[0]) {
            fail(field + ":" + term + " doc " + doc + " occurrence " + k + " position=" + position
                + " expected " + want[0]);
          }
          if (postings.startOffset() != want[1] || postings.endOffset() != want[2]) {
            fail(field + ":" + term + " doc " + doc + " occurrence " + k + " offsets="
                + postings.startOffset() + ".." + postings.endOffset() + " expected " + want[1]
                + ".." + want[2]);
          }
          BytesRef payload = postings.getPayload();
          byte[] got = payload == null ? new byte[0] : copy(payload);
          byte[] want3 = wantPayloads.get(k);
          if (!java.util.Arrays.equals(got, want3)) {
            fail(field + ":" + term + " doc " + doc + " occurrence " + k + " payload="
                + hex.formatHex(got) + " expected " + hex.formatHex(want3));
          }
          if ("body".equals(field) || "notes".equals(field)) {
            // Recomputed here rather than read from the manifest: the manifest
            // and the index are both produced by the Rust side, so an
            // independent derivation is what rules out a shared mistake.
            byte[] rule =
                "body".equals(field) ? payloadRule(doc, want[0]) : notesPayloadRule(doc, want[0]);
            if (!java.util.Arrays.equals(got, rule)) {
              fail(field + ":" + term + " doc " + doc + " position " + want[0] + " payload="
                  + hex.formatHex(got) + " but the fixture's rule says " + hex.formatHex(rule));
            }
            if ("body".equals(field)
                && got.length < payloadLengthSeen.length
                && !payloadLengthSeen[got.length]) {
              payloadLengthSeen[got.length] = true;
              payloadLengthsSeen++;
            }
          }
        }
        if (postings.nextPosition() >= 0 && postings.freq() > wantPositions.size()) {
          fail(field + ":" + term + " doc " + doc + " has more positions than freq claims");
        }
      }
    }

    if (!sawCrossing) {
      fail("the sampled set does not include document " + crossing
          + ", whose occurrences straddle a .pos block boundary");
    }
    if (payloadLengthsSeen < 3) {
      fail("only " + payloadLengthsSeen + " distinct payload lengths observed; a uniform payload "
          + "run cannot distinguish a correct payload-length stream from a constant one");
    }
  }

  /** Must match {@code notes_payload_for} in `write_positions_segment_fixture.rs`. */
  private static byte[] notesPayloadRule(int doc, int position) {
    int len = (doc % 3) * 37;
    byte[] out = new byte[len];
    for (int i = 0; i < len; i++) {
      out[i] = (byte) ((doc * 3 + position * 5 + i) & 0xFF);
    }
    return out;
  }

  /** Must match {@code payload_for} in `write_positions_segment_fixture.rs`. */
  private static byte[] payloadRule(int doc, int position) {
    int len = (doc * 7 + position) % 5;
    byte[] out = new byte[len];
    for (int i = 0; i < len; i++) {
      out[i] = (byte) ((doc + position + i) & 0xFF);
    }
    return out;
  }

  /**
   * A {@code DOCS} field must report freq 1 for every document (Lucene's
   * documented behaviour when frequencies are not indexed) and must not carry
   * positions -- a writer that emitted a `.pos` region for it anyway would
   * otherwise go unnoticed, since nothing would read it.
   */
  private static void checkDocsOnlyField(LeafReader reader) throws IOException {
    Terms terms = reader.terms("tag");
    if (terms == null) {
      fail("field \"tag\" has no terms at all");
      return;
    }
    if (terms.hasFreqs() || terms.hasPositions() || terms.hasOffsets() || terms.hasPayloads()) {
      fail("field \"tag\" reports freqs=" + terms.hasFreqs() + " positions=" + terms.hasPositions()
          + " offsets=" + terms.hasOffsets() + " payloads=" + terms.hasPayloads()
          + ", expected all false for IndexOptions.DOCS");
    }
    TermsEnum te = terms.iterator();
    if (!te.seekExact(new BytesRef("always"))) {
      fail("field \"tag\" is missing term \"always\"");
      return;
    }
    PostingsEnum postings = te.postings(null, PostingsEnum.NONE);
    int seen = 0;
    for (int doc = postings.nextDoc(); doc != PostingsEnum.NO_MORE_DOCS; doc = postings.nextDoc()) {
      if (postings.freq() != 1) {
        fail("tag:always doc " + doc + " freq=" + postings.freq() + ", expected 1 for a DOCS field");
        break;
      }
      seen++;
    }
    if (seen != reader.maxDoc()) {
      fail("tag:always covered " + seen + " documents, expected " + reader.maxDoc());
    }
  }

  /**
   * The stored term vector for {@code body} must carry the same three axes its
   * postings do. {@code CheckIndex.testTermVectors} already cross-checks a
   * vector against the postings occurrence by occurrence -- but only for the
   * axes the vector actually declares, so a vector that silently dropped
   * offsets and payloads would make that check vacuous rather than fail it.
   * This asserts the axes are there, and spot-checks one document's whole
   * vector against the manifest.
   */
  private static void checkTermVectors(LeafReader reader, Properties manifest) throws IOException {
    int sampleCount = Integer.parseInt(manifest.getProperty("sample_count"));
    HexFormat hex = HexFormat.of();
    for (int i = 0; i < sampleCount; i++) {
      int doc = Integer.parseInt(manifest.getProperty("sample." + i + ".doc"));
      Terms vector = reader.termVectors().get(doc, "body");
      if (vector == null) {
        fail("document " + doc + " has no term vector for \"body\"");
        return;
      }
      if (!vector.hasPositions() || !vector.hasOffsets() || !vector.hasPayloads()) {
        fail("document " + doc + " body term vector has positions=" + vector.hasPositions()
            + " offsets=" + vector.hasOffsets() + " payloads=" + vector.hasPayloads()
            + ", expected all true (the field indexes all three)");
        return;
      }
      int entryCount = Integer.parseInt(manifest.getProperty("sample." + i + ".entry_count"));
      for (int j = 0; j < entryCount; j++) {
        String[] parts = manifest.getProperty("sample." + i + "." + j).split("\\|", -1);
        if (!"body".equals(parts[0])) {
          continue;
        }
        TermsEnum te = vector.iterator();
        if (!te.seekExact(new BytesRef(parts[1]))) {
          fail("document " + doc + " body term vector is missing term \"" + parts[1] + "\"");
          continue;
        }
        PostingsEnum postings = te.postings(null, PostingsEnum.ALL);
        if (postings.nextDoc() != 0) {
          fail("document " + doc + " body term vector postings did not start at 0");
          continue;
        }
        String[] occurrences = parts[2].split(",", -1);
        if (postings.freq() != occurrences.length) {
          fail("document " + doc + " body term vector \"" + parts[1] + "\" freq="
              + postings.freq() + " expected " + occurrences.length);
          continue;
        }
        for (String occ : occurrences) {
          String[] f = occ.split(":", -1);
          int position = postings.nextPosition();
          if (position != Integer.parseInt(f[0])
              || postings.startOffset() != Integer.parseInt(f[1])
              || postings.endOffset() != Integer.parseInt(f[2])) {
            fail("document " + doc + " body term vector \"" + parts[1] + "\" occurrence "
                + position + "/" + postings.startOffset() + ".." + postings.endOffset()
                + " expected " + occ);
          }
          BytesRef payload = postings.getPayload();
          byte[] got = payload == null ? new byte[0] : copy(payload);
          if (!hex.formatHex(got).equals(f[3])) {
            fail("document " + doc + " body term vector \"" + parts[1] + "\" payload="
                + hex.formatHex(got) + " expected " + f[3]);
          }
        }
      }
    }
  }

  private static byte[] copy(BytesRef ref) {
    byte[] out = new byte[ref.length];
    System.arraycopy(ref.bytes, ref.offset, out, 0, ref.length);
    return out;
  }
}
