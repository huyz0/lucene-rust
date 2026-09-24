import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.apache.lucene.document.NumericDocValuesField;
import org.apache.lucene.index.CheckIndex;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.DocValuesType;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.MultiTerms;
import org.apache.lucene.index.PostingsEnum;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.Terms;
import org.apache.lucene.index.TermsEnum;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.FieldDoc;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.PhraseQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.Sort;
import org.apache.lucene.search.SortField;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.BytesRef;

/**
 * M3's end-to-end write-path proof (T3.4), Java half: real Lucene 10.5.0 opens a whole index this
 * port wrote through its {@code IndexWriter} -- 120 000 documents, several segments, text with
 * positions, offsets, payloads and norms, a keyword field and a numeric doc-values field -- and must
 * find exactly what the generator put there and answer queries exactly as this port's searcher
 * does. The Rust half, which writes the index and the expectations, is {@code
 * crates/lucene-search/examples/write_verify_index.rs}.
 *
 * <ol>
 *   <li>The index opens with {@link DirectoryReader}, has more than one segment, and its field
 *       infos say what the generator asked for.
 *   <li><b>Every term's postings</b> (T3.1): each {@code body}, {@code keyword} and {@code id} term in the
 *       index, walked through {@link MultiTerms} with global document ids, must match {@code
 *       postings.tsv} -- docFreq, totalTermFreq and an FNV-1a hash over every document, freq,
 *       position, both offsets and payload. The expectations are computed from the generated text,
 *       not from this port's writer or reader, so a misreading of the format the two share cannot
 *       cancel out here. A missing or extra term fails too.
 *   <li><b>The same answers</b>: each query in {@code queries.tsv} runs through {@link
 *       IndexSearcher} with its default BM25, and its top 50 must be the documents {@code
 *       rust-results.tsv} lists, in the same order, each score within 1e-5 (range queries: the same
 *       documents and sort values).
 *   <li>{@link CheckIndex} at {@code MIN_LEVEL_FOR_SLOW_CHECKS} reports the index clean.
 * </ol>
 *
 * <p>Usage: {@code java VerifyIndex <out-dir>}, where {@code <out-dir>} is what the Rust example
 * wrote. Exits nonzero with a diagnosis on any mismatch.
 */
public class VerifyIndex {
  /** Must match {@code write_verify_index.rs}. */
  private static final int NUM_DOCS = 120_000;

  private static final int TOP_N = 50;
  private static final double SCORE_TOLERANCE = 1e-5;

  private static int failures = 0;
  private static double maxScoreDiff = 0;

  private static void fail(String message) {
    failures++;
    if (failures <= 40) {
      System.out.println("MISMATCH " + message);
    }
  }

  public static void main(String[] args) throws IOException {
    Path out = Path.of(args[0]);
    Path indexPath = out.resolve("index");

    try (Directory dir = FSDirectory.open(indexPath);
        DirectoryReader reader = DirectoryReader.open(dir)) {
      checkShape(reader);
      int terms = checkPostings(reader, out.resolve("postings.tsv"));
      int queries = checkQueries(reader, out.resolve("queries.tsv"), out.resolve("rust-results.tsv"));
      System.out.println(
          "segments="
              + reader.leaves().size()
              + " terms checked="
              + terms
              + " queries checked="
              + queries
              + " max score difference="
              + maxScoreDiff);
    }

    try (Directory dir = FSDirectory.open(indexPath);
        CheckIndex checker = new CheckIndex(dir)) {
      ByteArrayOutputStream captured = new ByteArrayOutputStream();
      checker.setInfoStream(new PrintStream(captured, true, StandardCharsets.UTF_8));
      checker.setLevel(CheckIndex.Level.MIN_LEVEL_FOR_SLOW_CHECKS);
      CheckIndex.Status status = checker.checkIndex();
      if (!status.clean) {
        fail("CheckIndex reported the index unclean:");
        System.out.println(captured.toString(StandardCharsets.UTF_8));
      }
    }

    if (failures > 0) {
      System.out.println(failures + " check(s) failed");
      System.exit(1);
    }
    System.out.println("Rust-written index verified against real Lucene. PASS");
  }

  private static void checkShape(DirectoryReader reader) {
    if (reader.maxDoc() != NUM_DOCS || reader.numDocs() != NUM_DOCS) {
      fail("maxDoc=" + reader.maxDoc() + " numDocs=" + reader.numDocs() + ", want " + NUM_DOCS);
    }
    if (reader.leaves().size() < 2) {
      fail("expected several segments, got " + reader.leaves().size());
    }
    FieldInfos infos = FieldInfos.getMergedFieldInfos(reader);
    FieldInfo body = infos.fieldInfo("body");
    if (body == null
        || body.getIndexOptions() != IndexOptions.DOCS_AND_FREQS_AND_POSITIONS_AND_OFFSETS
        || !body.hasPayloads()
        || !body.hasNorms()) {
      fail("body field info: " + describe(body));
    }
    FieldInfo keyword = infos.fieldInfo("keyword");
    if (keyword == null || keyword.getIndexOptions() != IndexOptions.DOCS) {
      fail("keyword field info: " + describe(keyword));
    }
    FieldInfo id = infos.fieldInfo("id");
    if (id == null || id.getIndexOptions() != IndexOptions.DOCS) {
      fail("id field info: " + describe(id));
    }
    FieldInfo num = infos.fieldInfo("num");
    if (num == null || num.getDocValuesType() != DocValuesType.NUMERIC) {
      fail("num field info: " + describe(num));
    }
  }

  private static String describe(FieldInfo fi) {
    return fi == null
        ? "absent"
        : fi.getIndexOptions()
            + " payloads="
            + fi.hasPayloads()
            + " norms="
            + fi.hasNorms()
            + " dv="
            + fi.getDocValuesType();
  }

  /** FNV-1a 64 over little-endian ints and raw bytes; the Rust side has the same function. */
  private static final class Fnv {
    long h = 0xcbf29ce484222325L;

    void bytes(byte[] b, int off, int len) {
      for (int i = off; i < off + len; i++) {
        h ^= (b[i] & 0xFF);
        h *= 0x100000001b3L;
      }
    }

    void i32(int v) {
      bytes(
          new byte[] {(byte) v, (byte) (v >>> 8), (byte) (v >>> 16), (byte) (v >>> 24)}, 0, 4);
    }
  }

  private static int checkPostings(DirectoryReader reader, Path manifest) throws IOException {
    Map<String, String> expected = new HashMap<>();
    for (String line : Files.readAllLines(manifest)) {
      int tab = line.indexOf('\t', line.indexOf('\t') + 1);
      expected.put(line.substring(0, tab), line.substring(tab + 1));
    }
    int checked = 0;
    for (String field : new String[] {"body", "keyword", "id"}) {
      boolean positions = field.equals("body");
      Terms terms = MultiTerms.getTerms(reader, field);
      if (terms == null) {
        fail("field " + field + " has no terms");
        continue;
      }
      TermsEnum te = terms.iterator();
      PostingsEnum pe = null;
      BytesRef term;
      while ((term = te.next()) != null) {
        String key = field + "\t" + term.utf8ToString();
        Fnv h = new Fnv();
        pe = te.postings(pe, positions ? PostingsEnum.ALL : PostingsEnum.NONE);
        int docFreq = 0;
        long ttf = 0;
        for (int doc = pe.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS; doc = pe.nextDoc()) {
          docFreq++;
          h.i32(doc);
          if (positions) {
            int freq = pe.freq();
            ttf += freq;
            h.i32(freq);
            for (int i = 0; i < freq; i++) {
              h.i32(pe.nextPosition());
              h.i32(pe.startOffset());
              h.i32(pe.endOffset());
              BytesRef payload = pe.getPayload();
              int len = payload == null ? 0 : payload.length;
              h.i32(len);
              if (len > 0) {
                h.bytes(payload.bytes, payload.offset, payload.length);
              }
            }
          }
        }
        String got =
            docFreq + "\t" + (positions ? Long.toString(ttf) : "-1") + "\t" + String.format("%016x", h.h);
        String want = expected.remove(key);
        if (want == null) {
          fail("unexpected term " + key.replace('\t', ':'));
        } else if (!want.equals(got)) {
          fail(key.replace('\t', ':') + " postings: got " + got + ", want " + want);
        } else if (te.docFreq() != docFreq
            || (positions && te.totalTermFreq() != ttf)) {
          fail(key.replace('\t', ':') + " term stats " + te.docFreq() + "/" + te.totalTermFreq()
              + " disagree with its postings " + docFreq + "/" + ttf);
        }
        checked++;
      }
    }
    for (String missing : expected.keySet()) {
      fail("missing term " + missing.replace('\t', ':'));
    }
    return checked;
  }

  private static Query build(String kind, String field, List<String> args) {
    switch (kind) {
      case "term":
        return new TermQuery(new Term(field, args.get(0)));
      case "and":
      case "or":
        {
          BooleanQuery.Builder b = new BooleanQuery.Builder();
          BooleanClause.Occur occur =
              kind.equals("and") ? BooleanClause.Occur.MUST : BooleanClause.Occur.SHOULD;
          for (String t : args) {
            b.add(new TermQuery(new Term(field, t)), occur);
          }
          return b.build();
        }
      case "and_kw":
        return new BooleanQuery.Builder()
            .add(new TermQuery(new Term("body", args.get(0))), BooleanClause.Occur.MUST)
            .add(new TermQuery(new Term("keyword", args.get(1))), BooleanClause.Occur.MUST)
            .build();
      case "phrase":
        return new PhraseQuery("body", args.toArray(new String[0]));
      case "dv_range":
        return NumericDocValuesField.newSlowRangeQuery(
            field, Long.parseLong(args.get(0)), Long.parseLong(args.get(1)));
      default:
        throw new IllegalArgumentException("unknown query kind " + kind);
    }
  }

  private static int checkQueries(DirectoryReader reader, Path queryFile, Path rustFile)
      throws IOException {
    Map<String, String> rust = new HashMap<>();
    for (String line : Files.readAllLines(rustFile)) {
      int tab = line.indexOf('\t');
      rust.put(line.substring(0, tab), line.substring(tab + 1));
    }
    IndexSearcher searcher = new IndexSearcher(reader);
    searcher.setQueryCache(null);
    int checked = 0;
    for (String line : Files.readAllLines(queryFile)) {
      String[] f = line.split("\t");
      String id = f[0];
      String kind = f[1];
      List<String> args = new ArrayList<>(List.of(f).subList(3, f.length));
      Query q = build(kind, f[2], args);
      boolean sorted = kind.equals("dv_range");
      TopDocs top =
          sorted
              ? searcher.search(q, TOP_N, new Sort(new SortField(f[2], SortField.Type.LONG)))
              : searcher.search(q, TOP_N);
      String rustLine = rust.getOrDefault(id, "");
      String[] rustHits = rustLine.isEmpty() ? new String[0] : rustLine.split(",");
      ScoreDoc[] hits = top.scoreDocs;
      if (hits.length != rustHits.length) {
        fail(id + " (" + q + "): Lucene returned " + hits.length + " hits, Rust " + rustHits.length);
      }
      for (int i = 0; i < Math.min(hits.length, rustHits.length); i++) {
        String[] dv = rustHits[i].split(":");
        int rustDoc = Integer.parseInt(dv[0]);
        boolean ok;
        String javaValue;
        if (sorted) {
          long value = (Long) ((FieldDoc) hits[i]).fields[0];
          javaValue = Long.toString(value);
          ok = hits[i].doc == rustDoc && value == Long.parseLong(dv[1]);
        } else {
          javaValue = Float.toString(hits[i].score);
          double diff = Math.abs(hits[i].score - Double.parseDouble(dv[1]));
          maxScoreDiff = Math.max(maxScoreDiff, diff);
          ok = hits[i].doc == rustDoc && diff <= SCORE_TOLERANCE;
        }
        if (!ok) {
          fail(
              id
                  + " ("
                  + q
                  + ") rank "
                  + i
                  + ": Lucene doc "
                  + hits[i].doc
                  + " "
                  + javaValue
                  + ", Rust "
                  + rustHits[i]);
          break;
        }
      }
      checked++;
    }
    return checked;
  }
}
