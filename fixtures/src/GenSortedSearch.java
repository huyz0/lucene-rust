import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.DoublePoint;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.FloatPoint;
import org.apache.lucene.document.IntPoint;
import org.apache.lucene.document.LongPoint;
import org.apache.lucene.document.NumericDocValuesField;
import org.apache.lucene.document.SortedNumericDocValuesField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.SegmentInfos;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.FieldDoc;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.MatchAllDocsQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.Sort;
import org.apache.lucene.search.SortField;
import org.apache.lucene.search.SortedNumericSelector;
import org.apache.lucene.search.SortedNumericSortField;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TopFieldCollectorManager;
import org.apache.lucene.search.TotalHits;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.NumericUtils;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Random;
import java.util.stream.Stream;

/**
 * Sorted searches -- {@code TopFieldCollector} -- by numeric doc-values fields, the score and the
 * document, with {@code searchAfter} paging, recorded from Lucene for
 * crates/lucene-search/tests/sorted_search_fixtures.rs.
 *
 * <p>The corpus: three segments of 20,000 documents, the first and last with deletions. Each has a
 * Zipf-like {@code body} (as GenMixedBooleanScoring's), and numeric fields indexed both as doc
 * values and as points, so {@code NumericComparator} skips with the points:
 *
 * <pre>
 *   l   long, SortedNumeric + LongPoint, 10% of documents without a value, many ties
 *   i   int, SortedNumeric + IntPoint, 0..999 (heavy ties)
 *   d   double, SortedNumeric (sortable long) + DoublePoint, negative and -0.0 included
 *   f   float, SortedNumeric (sortable int) + FloatPoint
 *   m   long, SortedNumeric + LongPoint, 0-3 values per document (MIN and MAX selectors)
 *   s   long, NUMERIC (single-valued column) + LongPoint
 *   np  long, SortedNumeric without points (no skipping)
 * </pre>
 *
 * <p>Every run is a query (the S-expression grammar of GenMixedBooleanScoring, plus {@code
 * (all)}), a sort, a page size and a total-hits threshold, and optionally the {@code after} hit.
 * Each sort key is written for the Rust side as {@code field:type:selector:reverse:missing}, the
 * missing value already as the comparable long {@code NumericComparator} uses, and each hit as
 * {@code doc:v1:v2...} in the same encoding (a score as its float bits).
 */
public class GenSortedSearch {
  static final int DOCS_PER_SEGMENT = 20_000;
  static final int SEGMENTS = 3;
  static final int VOCAB = 60;

  static final String[] QUERIES = {
    "(all)",
    "(t w0)",
    "(t w30)",
    "(b 0 (+ (t w1)) (- (t w2)))",
    "(b 0 (? (t w3)) (? (t w4)) (? (t w5)))",
    "(b 0 (# (t w0)) (# (r 5000 45000)))",
    // Two-phase: the lazily advanced scoring tree must land on a phrase's matches.
    "(b 0 (+ (t w0)) (? (p w1 w2)))",
  };

  /** Sorts: comma-separated keys, {@code score}, {@code doc} or {@code FIELD:TYPE:SEL:ORDER:MISSING}. */
  static final String[] SORTS = {
    "l:long:min:asc:last",
    "l:long:min:desc:last",
    "l:long:min:asc:first",
    "i:int:min:asc:last",
    "i:int:min:desc:last,l:long:min:asc:last",
    "d:double:min:asc:last",
    "d:double:min:desc:first",
    "f:float:min:desc:last",
    "m:long:min:asc:last",
    "m:long:max:desc:last",
    "s:long:min:desc:last",
    "np:long:min:asc:last",
    "doc",
    "i:int:min:asc:last,score",
    "score,i:int:min:desc:last",
    "l:long:min:asc:none",
    "score!,l:long:min:asc:last",
    "i:int:min:asc:last,score!",
  };

  public static void main(String[] args) throws IOException {
    Path out = Path.of(args[0]).resolve("sorted_search_index");
    if (Files.exists(out)) {
      try (Stream<Path> s = Files.walk(out)) {
        s.sorted(Comparator.reverseOrder()).forEach(p -> p.toFile().delete());
      }
    }
    Files.createDirectories(out);

    StringBuilder m = new StringBuilder();
    try (Directory dir = FSDirectory.open(out)) {
      IndexWriterConfig cfg = new IndexWriterConfig(new StandardAnalyzer());
      cfg.setUseCompoundFile(false);
      cfg.setMergePolicy(NoMergePolicy.INSTANCE);
      cfg.setRAMBufferSizeMB(256);
      cfg.setMaxBufferedDocs(IndexWriterConfig.DISABLE_AUTO_FLUSH);
      Random random = new Random(20260926L);
      try (IndexWriter w = new IndexWriter(dir, cfg)) {
        for (int seg = 0; seg < SEGMENTS; seg++) {
          for (int i = 0; i < DOCS_PER_SEGMENT; i++) {
            int id = seg * DOCS_PER_SEGMENT + i;
            Document doc = new Document();
            doc.add(new StringField("id", Integer.toString(id), Field.Store.NO));
            doc.add(new TextField("body", GenMixedBooleanScoring.body(random), Field.Store.NO));
            doc.add(new LongPoint("r", id));
            if (random.nextInt(10) != 0) {
              long l = random.nextInt(5000) - 2500L;
              doc.add(new SortedNumericDocValuesField("l", l));
              doc.add(new LongPoint("l", l));
            }
            int iv = random.nextInt(1000);
            doc.add(new SortedNumericDocValuesField("i", iv));
            doc.add(new IntPoint("i", iv));
            double d = switch (random.nextInt(20)) {
              case 0 -> -0.0;
              case 1 -> 0.0;
              default -> (random.nextDouble() - 0.3) * 1e6;
            };
            doc.add(new SortedNumericDocValuesField("d", NumericUtils.doubleToSortableLong(d)));
            doc.add(new DoublePoint("d", d));
            float f = (random.nextFloat() - 0.5f) * 1000f;
            doc.add(new SortedNumericDocValuesField("f", NumericUtils.floatToSortableInt(f)));
            doc.add(new FloatPoint("f", f));
            int values = random.nextInt(4);
            for (int k = 0; k < values; k++) {
              long v = random.nextInt(100_000);
              doc.add(new SortedNumericDocValuesField("m", v));
              doc.add(new LongPoint("m", v));
            }
            long s = (long) random.nextInt(1_000_000) * 1_000_003L;
            doc.add(new NumericDocValuesField("s", s));
            doc.add(new LongPoint("s", s));
            doc.add(new SortedNumericDocValuesField("np", random.nextInt(3000)));
            w.addDocument(doc);
          }
          w.commit();
        }
        for (int id = 0; id < DOCS_PER_SEGMENT; id += 29) {
          w.deleteDocuments(new Term("id", Integer.toString(id)));
          w.deleteDocuments(new Term("id", Integer.toString(2 * DOCS_PER_SEGMENT + id + 3)));
        }
        w.commit();
      }

      SegmentInfos sis = SegmentInfos.readLatestCommit(dir);
      if (sis.size() != SEGMENTS) {
        throw new AssertionError("expected " + SEGMENTS + " segments, got " + sis.size());
      }
      int run = 0;
      try (DirectoryReader reader = DirectoryReader.open(dir)) {
        IndexSearcher searcher = new IndexSearcher(reader);
        searcher.setQueryCache(null);
        for (String qs : QUERIES) {
          Query q = parse(new GenMixedBooleanScoring.Tokens(qs));
          for (String ss : SORTS) {
            Sort sort = sort(ss);
            for (int topN : new int[] {10, 50}) {
              for (int threshold : new int[] {100, Integer.MAX_VALUE}) {
                TopDocs page1 =
                    searcher.search(q, new TopFieldCollectorManager(sort, topN, null, threshold));
                run = record(m, run, qs, sort, topN, threshold, null, page1);
                if (page1.scoreDocs.length == 0) {
                  continue;
                }
                FieldDoc last = (FieldDoc) page1.scoreDocs[page1.scoreDocs.length - 1];
                TopDocs page2 =
                    searcher.search(q, new TopFieldCollectorManager(sort, topN, last, threshold));
                run = record(m, run, qs, sort, topN, threshold, last, page2);
                // OpenSearch's search_after: the values alone, the document past every id.
                FieldDoc byValues = new FieldDoc(Integer.MAX_VALUE, Float.NaN, last.fields);
                TopDocs page2v =
                    searcher.search(
                        q, new TopFieldCollectorManager(sort, topN, byValues, threshold));
                run = record(m, run, qs, sort, topN, threshold, byValues, page2v);
              }
            }
          }
        }
      }
      m.insert(0, "run_count=" + run + "\n");
    }
    Files.writeString(out.resolve("manifest.properties"), m.toString());
    System.out.println("wrote " + out);
  }

  static int record(
      StringBuilder m, int run, String q, Sort sort, int topN, int threshold, FieldDoc after, TopDocs td) {
    String k = "run." + run;
    m.append(k).append(".query=").append(q).append('\n');
    m.append(k).append(".sort=").append(spec(sort)).append('\n');
    m.append(k).append(".top_n=").append(topN).append('\n');
    m.append(k).append(".threshold=").append(threshold == Integer.MAX_VALUE ? "max" : threshold).append('\n');
    if (after != null) {
      m.append(k).append(".after=").append(hit(sort, after)).append('\n');
    }
    StringBuilder hits = new StringBuilder();
    for (ScoreDoc sd : td.scoreDocs) {
      if (hits.length() > 0) {
        hits.append(',');
      }
      hits.append(hit(sort, (FieldDoc) sd));
    }
    m.append(k).append(".hits=").append(hits).append('\n');
    m.append(k).append(".total=").append(td.totalHits.value()).append('\n');
    m.append(k).append(".relation=")
        .append(td.totalHits.relation() == TotalHits.Relation.EQUAL_TO ? "eq" : "gte")
        .append('\n');
    return run + 1;
  }

  static String hit(Sort sort, FieldDoc fd) {
    StringBuilder b = new StringBuilder().append(fd.doc);
    for (int i = 0; i < fd.fields.length; i++) {
      b.append(':').append(comparable(sort.getSort()[i], fd.fields[i]));
    }
    return b.toString();
  }

  static long comparable(SortField f, Object v) {
    return switch (type(f)) {
      case SCORE -> Float.floatToIntBits((Float) v);
      case DOC -> (Integer) v;
      case LONG -> (Long) v;
      case INT -> (Integer) v;
      case DOUBLE -> NumericUtils.doubleToSortableLong((Double) v);
      case FLOAT -> NumericUtils.floatToSortableInt((Float) v);
      default -> throw new IllegalArgumentException(f.toString());
    };
  }

  static SortField.Type type(SortField f) {
    return f instanceof SortedNumericSortField sn ? sn.getNumericType() : f.getType();
  }

  static String spec(Sort sort) {
    StringBuilder b = new StringBuilder();
    for (SortField f : sort.getSort()) {
      if (b.length() > 0) {
        b.append(',');
      }
      String type = type(f).name().toLowerCase();
      String sel = f instanceof SortedNumericSortField sn ? sn.getSelector().name().toLowerCase() : "min";
      long missing = f.getMissingValue() == null ? 0 : comparable(f, f.getMissingValue());
      b.append(f.getField() == null ? "" : f.getField())
          .append(':').append(type)
          .append(':').append(sel)
          .append(':').append(f.getReverse())
          .append(':').append(missing);
    }
    return b.toString();
  }

  static Sort sort(String s) {
    List<SortField> fields = new ArrayList<>();
    for (String k : s.split(",")) {
      if (k.equals("score")) {
        fields.add(SortField.FIELD_SCORE);
        continue;
      }
      if (k.equals("score!")) {
        fields.add(new SortField(null, SortField.Type.SCORE, true));
        continue;
      }
      if (k.equals("doc")) {
        fields.add(SortField.FIELD_DOC);
        continue;
      }
      String[] p = k.split(":");
      SortField.Type type = SortField.Type.valueOf(p[1].toUpperCase());
      SortedNumericSelector.Type sel =
          p[2].equals("max") ? SortedNumericSelector.Type.MAX : SortedNumericSelector.Type.MIN;
      boolean reverse = p[3].equals("desc");
      SortedNumericSortField f = new SortedNumericSortField(p[0], type, reverse, sel);
      if (!p[4].equals("none")) {
        // OpenSearch's `missing: _last`/`_first`: past every value in the sort's direction.
        boolean high = p[4].equals("last") != reverse;
        f.setMissingValue(
            switch (type) {
              case LONG -> high ? Long.MAX_VALUE : Long.MIN_VALUE;
              case INT -> high ? Integer.MAX_VALUE : Integer.MIN_VALUE;
              case DOUBLE -> high ? Double.POSITIVE_INFINITY : Double.NEGATIVE_INFINITY;
              case FLOAT -> high ? Float.POSITIVE_INFINITY : Float.NEGATIVE_INFINITY;
              default -> throw new IllegalArgumentException(k);
            });
      }
      fields.add(f);
    }
    return new Sort(fields.toArray(new SortField[0]));
  }

  static Query parse(GenMixedBooleanScoring.Tokens t) {
    if (t.toks.get(t.at + 1).equals("all")) {
      t.expect("(");
      t.expect("all");
      t.expect(")");
      return new MatchAllDocsQuery();
    }
    if (t.toks.get(t.at + 1).equals("b")) {
      t.expect("(");
      t.expect("b");
      BooleanQuery.Builder b = new BooleanQuery.Builder();
      b.setMinimumNumberShouldMatch(Integer.parseInt(t.next()));
      while (t.peek().equals("(")) {
        t.expect("(");
        BooleanClause.Occur occur =
            switch (t.next()) {
              case "+" -> BooleanClause.Occur.MUST;
              case "#" -> BooleanClause.Occur.FILTER;
              case "?" -> BooleanClause.Occur.SHOULD;
              case "-" -> BooleanClause.Occur.MUST_NOT;
              default -> throw new IllegalArgumentException("bad occur");
            };
        b.add(parse(t), occur);
        t.expect(")");
      }
      t.expect(")");
      return b.build();
    }
    if (t.toks.get(t.at + 1).equals("r")) {
      t.expect("(");
      t.expect("r");
      Query q = LongPoint.newRangeQuery("r", Long.parseLong(t.next()), Long.parseLong(t.next()));
      t.expect(")");
      return q;
    }
    return GenMixedBooleanScoring.parse(t);
  }
}
