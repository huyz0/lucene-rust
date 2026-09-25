import org.apache.lucene.codecs.lucene103.blocktree.Stats;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.NumericDocValuesField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.index.CheckIndex;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.PostingsEnum;
import org.apache.lucene.index.StoredFields;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.Terms;
import org.apache.lucene.index.TermsEnum;
import org.apache.lucene.search.BooleanClause.Occur;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.PhraseQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.Sort;
import org.apache.lucene.search.SortField;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.FieldDoc;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.BytesRef;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * M3's end-to-end proof (task T3.4): real Lucene opens a whole index this
 * port's {@code IndexWriter} wrote, finds it clean, and searches it to the
 * same results this port's own searcher produced.
 *
 * <p>{@code write_verify_index_fixture} (in {@code lucene-search}) writes the
 * index -- 120 000 Zipfian documents in several segments, with deletes --
 * then runs a fixed query set over it with this port's searcher and leaves
 * two files beside it: the query set ({@code verify-queries.tsv}) and its own
 * top 50 per query ({@code verify-rust-results.tsv}). This program:
 *
 * <ol>
 *   <li>opens the index with {@link DirectoryReader} and asserts its shape --
 *       document and deletion counts, several segments, the index options and
 *       payload flag of each field, and at least one term far past
 *       {@code BLOCK_SIZE} -- so a fixture that silently degenerated would
 *       fail here rather than pass on an index that tests nothing;
 *   <li>reads positions, offsets and payloads for a sample of occurrences and
 *       checks each against the stored text and the payload function the
 *       fixture used, which is a function of term and position only;
 *   <li>runs full-level {@link CheckIndex};
 *   <li>runs every query through {@link IndexSearcher} and requires the top 50
 *       to be <b>the same documents in the same order</b> as this port's, with
 *       every score within {@link #SCORE_TOLERANCE} (and every sort value
 *       equal, for a doc-values sort).
 * </ol>
 *
 * <p>The tolerance is the milestone's: {@code PLAN.md} requires the same
 * {@code f32} operations in the same order, not bit-identical results across
 * compilers. The program also reports how many scores were bit-identical,
 * because a drift that stays under the tolerance is still worth seeing.
 *
 * <p>Usage: {@code java VerifyIndex <index-dir>}. Exits nonzero with a
 * diagnosis on any mismatch.
 */
public class VerifyIndex {
  /** Must match `write_verify_index_fixture.rs`. */
  private static final int NUM_DOCS = 120_000;
  private static final int FIRST_COMMIT_AT = 70_000;
  private static final int DELETE_EVERY = 97;
  private static final int TOP_N = 50;
  private static final double SCORE_TOLERANCE = 1e-5;
  /** `BLOCK_SIZE` in `Lucene104PostingsFormat`. */
  private static final int BLOCK_SIZE = 256;
  /** The milestone asks for at least this many queries. */
  private static final int MIN_QUERIES = 50;

  private static int failures = 0;

  private static void fail(String msg) {
    System.out.println("MISMATCH " + msg);
    failures++;
  }

  public static void main(String[] args) throws IOException {
    Path path = Path.of(args[0]);
    try (Directory dir = FSDirectory.open(path);
        DirectoryReader reader = DirectoryReader.open(dir)) {
      checkShape(reader);
      checkOccurrences(reader);
      checkQueries(reader, path);
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
        fail("CheckIndex reported the index unclean:\n" + captured.toString(StandardCharsets.UTF_8));
      } else {
        System.out.println(
            "CheckIndex: clean, " + status.numSegments + " segments, " + status.totLoseDocCount
                + " docs lost");
      }
    }

    if (failures > 0) {
      System.out.println(failures + " check(s) failed");
      System.exit(1);
    }
    System.out.println("Rust-written index verified against real Lucene. PASS");
  }

  // ---------------------------------------------------------------- shape

  private static void checkShape(DirectoryReader reader) throws IOException {
    // Docs 0, 97, 194, ... below FIRST_COMMIT_AT are deleted.
    int deleted = (FIRST_COMMIT_AT - 1) / DELETE_EVERY + 1;
    if (reader.maxDoc() != NUM_DOCS) {
      fail("maxDoc=" + reader.maxDoc() + " expected " + NUM_DOCS);
    }
    if (reader.numDocs() != NUM_DOCS - deleted) {
      fail("numDocs=" + reader.numDocs() + " expected " + (NUM_DOCS - deleted));
    }
    if (reader.leaves().size() < 3) {
      fail("only " + reader.leaves().size() + " segment(s); the fixture must be multi-segment");
    }
    int withDeletes = 0;
    for (LeafReaderContext ctx : reader.leaves()) {
      if (ctx.reader().hasDeletions()) {
        withDeletes++;
      }
      FieldInfos infos = ctx.reader().getFieldInfos();
      expectField(infos, "body", IndexOptions.DOCS_AND_FREQS_AND_POSITIONS_AND_OFFSETS, true, true);
      expectField(infos, "title", IndexOptions.DOCS_AND_FREQS_AND_POSITIONS, false, true);
      expectField(infos, "tag", IndexOptions.DOCS, false, false);
      expectField(infos, "id", IndexOptions.DOCS, false, false);
    }
    if (withDeletes < 2) {
      fail("deletes reached " + withDeletes + " segment(s); expected several");
    }

    // The bit-packed block path has to be exercised, in more than one segment.
    int bigSegments = 0;
    int maxDf = 0;
    for (LeafReaderContext ctx : reader.leaves()) {
      Terms terms = ctx.reader().terms("body");
      TermsEnum te = terms.iterator();
      if (te.seekExact(new BytesRef("w0"))) {
        maxDf = Math.max(maxDf, te.docFreq());
        if (te.docFreq() > 8 * BLOCK_SIZE) {
          bigSegments++;
        }
      }
    }
    if (bigSegments < 2) {
      fail("body:w0 exceeds 8*BLOCK_SIZE docs in only " + bigSegments + " segment(s), max " + maxDf);
    }

    // The term dictionary must have the shapes the blocktree writer claims:
    // more than one block, floor blocks, inner blocks (a block whose entries
    // include sub-blocks), and blocks at prefix lengths past the root, which
    // is what a multi-level `.tip` trie addresses. Lucene's own Stats walks
    // every block of the Rust-written `.tim` to count these.
    int floor = 0;
    int inner = 0;
    int deepest = 0;
    for (LeafReaderContext ctx : reader.leaves()) {
      Stats st = (Stats) ctx.reader().terms("body").getStats();
      if (st.totalBlockCount < 2) {
        fail(ctx + " body has " + st.totalBlockCount + " block(s)");
      }
      floor += st.floorBlockCount;
      inner += st.mixedBlockCount + st.subBlocksOnlyBlockCount;
      for (int len = 0; len < st.blockCountByPrefixLen.length; len++) {
        if (st.blockCountByPrefixLen[len] > 0) {
          deepest = Math.max(deepest, len);
        }
      }
    }
    if (floor == 0 || inner == 0 || deepest < 2) {
      fail("blocktree shapes missing: floor=" + floor + " inner=" + inner + " deepest prefix=" + deepest);
    }
    System.out.println(
        "blocktree(body): " + floor + " floor blocks, " + inner + " inner blocks, blocks down to prefix length "
            + deepest);

    // Block splitting is a function of the term sequence alone, so real
    // Lucene's own writer, handed the same terms, must cut the same blocks:
    // same count, same floor runs, same leaf/inner mix, same count at every
    // prefix length. (Byte sizes are left out -- this port writes suffixes
    // uncompressed where Java may pick LZ4 or LOWERCASE_ASCII.)
    for (LeafReaderContext ctx : reader.leaves()) {
      Stats rust = (Stats) ctx.reader().terms("body").getStats();
      Stats java = javaStatsForSameTerms(ctx.reader().terms("body"));
      String r = blockShape(rust);
      String j = blockShape(java);
      if (!r.equals(j)) {
        fail(ctx + " block structure differs from Lucene's own writer:\n  rust " + r + "\n  java " + j);
      }
    }

    // Stored fields line up with the global document order.
    StoredFields stored = reader.storedFields();
    for (int doc : new int[] {1, 12_345, FIRST_COMMIT_AT, NUM_DOCS - 1}) {
      String id = stored.document(doc).get("id");
      if (!("doc" + doc).equals(id)) {
        fail("doc " + doc + " has stored id " + id);
      }
    }
    System.out.println(
        "shape: " + reader.leaves().size() + " segments, " + reader.numDocs() + " live docs, "
            + withDeletes + " segments with deletes, max docFreq(body:w0) per segment " + maxDf);
  }

  private static String blockShape(Stats st) {
    return "terms=" + st.totalTermCount + " blocks=" + st.totalBlockCount + " nonFloor="
        + st.nonFloorBlockCount + " floor=" + st.floorBlockCount + " floorSub="
        + st.floorSubBlockCount + " mixed=" + st.mixedBlockCount + " termsOnly="
        + st.termsOnlyBlockCount + " subBlocksOnly=" + st.subBlocksOnlyBlockCount
        + " byPrefixLen=" + java.util.Arrays.toString(st.blockCountByPrefixLen);
  }

  /** Lucene's own term dictionary over exactly the terms of {@code terms}. */
  private static Stats javaStatsForSameTerms(Terms terms) throws IOException {
    try (Directory dir = new ByteBuffersDirectory()) {
      IndexWriterConfig config = new IndexWriterConfig().setRAMBufferSizeMB(256);
      try (IndexWriter w = new IndexWriter(dir, config)) {
        TermsEnum te = terms.iterator();
        for (BytesRef t = te.next(); t != null; t = te.next()) {
          Document doc = new Document();
          doc.add(new StringField("body", BytesRef.deepCopyOf(t), Field.Store.NO));
          w.addDocument(doc);
        }
        w.forceMerge(1);
      }
      try (DirectoryReader r = DirectoryReader.open(dir)) {
        return (Stats) r.leaves().get(0).reader().terms("body").getStats();
      }
    }
  }

  private static void expectField(
      FieldInfos infos, String name, IndexOptions options, boolean payloads, boolean norms) {
    FieldInfo fi = infos.fieldInfo(name);
    if (fi == null) {
      fail("field " + name + " missing");
      return;
    }
    if (fi.getIndexOptions() != options) {
      fail(name + " index options " + fi.getIndexOptions() + " expected " + options);
    }
    if (fi.hasPayloads() != payloads) {
      fail(name + " hasPayloads=" + fi.hasPayloads());
    }
    if (fi.hasNorms() != norms) {
      fail(name + " hasNorms=" + fi.hasNorms());
    }
  }

  // ---------------------------------------------------------- occurrences

  /** The fixture's `payload_for`, bit for bit. */
  static byte[] payloadFor(String term, int position) {
    int len = (position + term.length()) % 4;
    if (len == 0) {
      return null;
    }
    int seed = position;
    for (byte b : term.getBytes(StandardCharsets.UTF_8)) {
      seed = seed * 31 + (b & 0xFF);
    }
    byte[] out = new byte[len];
    for (int i = 0; i < len; i++) {
      out[i] = (byte) (seed >>> (8 * i));
    }
    return out;
  }

  /**
   * For three terms at very different frequencies, walk every live posting in
   * every segment and check each occurrence: its offsets must slice the term
   * out of the stored text, its position must be the token's index in that
   * text, and its payload must be the fixture's function of term and position.
   */
  private static void checkOccurrences(DirectoryReader reader) throws IOException {
    long checked = 0;
    for (String term : new String[] {"w0", "quick", "w900"}) {
      BytesRef bytes = new BytesRef(term);
      for (LeafReaderContext ctx : reader.leaves()) {
        TermsEnum te = ctx.reader().terms("body").iterator();
        if (!te.seekExact(bytes)) {
          continue;
        }
        StoredFields stored = ctx.reader().storedFields();
        PostingsEnum pe = te.postings(null, PostingsEnum.ALL);
        int seen = 0;
        for (int doc = pe.nextDoc(); doc != PostingsEnum.NO_MORE_DOCS; doc = pe.nextDoc()) {
          // Every document for the rare terms; a stride for w0, which is in
          // tens of thousands -- still thousands of occurrences per segment.
          if (!term.equals("w0") || seen++ % 7 == 0) {
            checked += checkDoc(stored.document(doc).get("body"), term, pe, ctx.docBase + doc);
          }
        }
      }
    }
    if (checked < 10_000) {
      fail("only " + checked + " occurrences checked");
    }
    System.out.println("occurrences: " + checked + " positions/offsets/payloads checked");
  }

  private static int checkDoc(String text, String term, PostingsEnum pe, int globalDoc)
      throws IOException {
    String[] tokens = text.split(" ");
    int[] starts = new int[tokens.length];
    for (int i = 0, at = 0; i < tokens.length; i++) {
      starts[i] = at;
      at += tokens[i].length() + 1;
    }
    int freq = pe.freq();
    int expectedFreq = 0;
    for (String t : tokens) {
      if (t.equals(term)) {
        expectedFreq++;
      }
    }
    if (freq != expectedFreq) {
      fail("doc " + globalDoc + " " + term + " freq " + freq + " expected " + expectedFreq);
      return 0;
    }
    for (int i = 0; i < freq; i++) {
      int pos = pe.nextPosition();
      if (pos < 0 || pos >= tokens.length || !tokens[pos].equals(term)) {
        fail("doc " + globalDoc + " " + term + " position " + pos + " is not that term");
        return i;
      }
      if (pe.startOffset() != starts[pos] || pe.endOffset() != starts[pos] + term.length()) {
        fail("doc " + globalDoc + " " + term + "@" + pos + " offsets " + pe.startOffset() + "-"
            + pe.endOffset() + " expected " + starts[pos] + "-" + (starts[pos] + term.length()));
      }
      byte[] want = payloadFor(term, pos);
      BytesRef got = pe.getPayload();
      boolean same =
          want == null
              ? got == null || got.length == 0
              : got != null && new BytesRef(want).bytesEquals(got);
      if (!same) {
        fail("doc " + globalDoc + " " + term + "@" + pos + " payload " + got);
      }
    }
    return freq;
  }

  // -------------------------------------------------------------- queries

  private static Query buildScored(String kind, String field, List<String> a) {
    switch (kind) {
      case "term":
        return new TermQuery(new Term(field, a.get(0)));
      case "and":
      case "or":
        {
          BooleanQuery.Builder b = new BooleanQuery.Builder();
          Occur occur = kind.equals("and") ? Occur.MUST : Occur.SHOULD;
          for (String t : a) {
            b.add(new TermQuery(new Term(field, t)), occur);
          }
          return b.build();
        }
      case "not":
      case "mixed":
        {
          BooleanQuery.Builder b = new BooleanQuery.Builder();
          b.add(new TermQuery(new Term(field, a.get(0))), Occur.MUST);
          Occur rest = kind.equals("not") ? Occur.MUST_NOT : Occur.SHOULD;
          for (String t : a.subList(1, a.size())) {
            b.add(new TermQuery(new Term(field, t)), rest);
          }
          return b.build();
        }
      case "msm":
        {
          BooleanQuery.Builder b = new BooleanQuery.Builder();
          b.setMinimumNumberShouldMatch(Integer.parseInt(a.get(0)));
          for (String t : a.subList(1, a.size())) {
            b.add(new TermQuery(new Term(field, t)), Occur.SHOULD);
          }
          return b.build();
        }
      case "filter":
        return new BooleanQuery.Builder()
            .add(new TermQuery(new Term(field, a.get(0))), Occur.MUST)
            .add(new TermQuery(new Term(a.get(1), a.get(2))), Occur.FILTER)
            .build();
      case "phrase":
        return new PhraseQuery(
            Integer.parseInt(a.get(0)), field, a.subList(1, a.size()).toArray(new String[0]));
      default:
        throw new IllegalArgumentException("unknown query kind " + kind);
    }
  }

  private static void checkQueries(DirectoryReader reader, Path dir) throws IOException {
    Map<String, String[]> rust = new HashMap<>();
    for (String line : Files.readAllLines(dir.resolve("verify-rust-results.tsv"))) {
      String[] f = line.split("\t", -1);
      rust.put(f[0], f);
    }
    IndexSearcher searcher = new IndexSearcher(reader);
    int queries = 0;
    int nonEmpty = 0;
    int kinds = 0;
    long scores = 0;
    long bitExact = 0;
    double maxDiff = 0;
    Map<String, Integer> byKind = new HashMap<>();
    for (String line : Files.readAllLines(dir.resolve("verify-queries.tsv"))) {
      if (line.isBlank() || line.startsWith("#")) {
        continue;
      }
      String[] f = line.split("\t");
      String id = f[0];
      String kind = f[1];
      String field = f[2];
      List<String> a = List.of(f).subList(3, f.length);
      queries++;
      byKind.merge(kind, 1, Integer::sum);

      int[] docs;
      String[] values;
      if (kind.equals("dvrange")) {
        long min = Long.parseLong(a.get(0));
        long max = Long.parseLong(a.get(1));
        boolean reverse = a.get(2).equals("desc");
        Query q = NumericDocValuesField.newSlowRangeQuery(field, min, max);
        TopDocs td = searcher.search(q, TOP_N, new Sort(new SortField(field, SortField.Type.LONG, reverse)));
        docs = new int[td.scoreDocs.length];
        values = new String[td.scoreDocs.length];
        for (int i = 0; i < docs.length; i++) {
          docs[i] = td.scoreDocs[i].doc;
          values[i] = String.valueOf(((FieldDoc) td.scoreDocs[i]).fields[0]);
        }
      } else {
        TopDocs td = searcher.search(buildScored(kind, field, a), TOP_N);
        docs = new int[td.scoreDocs.length];
        values = new String[td.scoreDocs.length];
        for (int i = 0; i < docs.length; i++) {
          docs[i] = td.scoreDocs[i].doc;
          values[i] = Float.toString(td.scoreDocs[i].score);
        }
      }
      if (docs.length > 0) {
        nonEmpty++;
      }

      String[] r = rust.get(id);
      if (r == null) {
        fail(id + " has no Rust result");
        continue;
      }
      int rustHits = Integer.parseInt(r[1]);
      String[] pairs = rustHits == 0 ? new String[0] : r[2].split(",");
      if (pairs.length != docs.length) {
        fail(id + " (" + line + "): Java " + docs.length + " hits, Rust " + pairs.length);
        continue;
      }
      for (int i = 0; i < docs.length; i++) {
        String[] p = pairs[i].split(":");
        int rustDoc = Integer.parseInt(p[0]);
        if (rustDoc != docs[i]) {
          fail(id + " (" + line + ") rank " + i + ": Java doc " + docs[i] + " (" + values[i]
              + "), Rust doc " + rustDoc + " (" + p[1] + ")");
          break;
        }
        if (kind.equals("dvrange")) {
          if (!p[1].equals(values[i])) {
            fail(id + " rank " + i + " doc " + rustDoc + ": Java value " + values[i]
                + ", Rust " + p[1]);
          }
        } else {
          float js = Float.parseFloat(values[i]);
          float rs = Float.parseFloat(p[1]);
          double diff = Math.abs((double) js - (double) rs);
          scores++;
          if (Float.floatToIntBits(js) == Float.floatToIntBits(rs)) {
            bitExact++;
          }
          maxDiff = Math.max(maxDiff, diff);
          if (diff > SCORE_TOLERANCE) {
            fail(id + " rank " + i + " doc " + rustDoc + ": Java score " + js + ", Rust " + rs);
          }
        }
      }
    }
    if (queries < MIN_QUERIES) {
      fail("only " + queries + " queries; the milestone requires at least " + MIN_QUERIES);
    }
    for (String k : new String[] {"term", "and", "or", "phrase", "dvrange"}) {
      if (!byKind.containsKey(k)) {
        fail("query set has no " + k + " query");
      }
    }
    if (nonEmpty < queries - 5) {
      fail("only " + nonEmpty + " of " + queries + " queries matched anything");
    }
    System.out.printf(
        "queries: %d run (%s), %d non-empty; %d scores compared, %d bit-identical, max |diff| %.3g%n",
        queries, byKind, nonEmpty, scores, bitExact, maxDiff);
  }
}
