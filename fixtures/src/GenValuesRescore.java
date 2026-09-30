import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.DoubleDocValuesField;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.FloatDocValuesField;
import org.apache.lucene.document.IntPoint;
import org.apache.lucene.document.KnnByteVectorField;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.LateInteractionField;
import org.apache.lucene.document.LongPoint;
import org.apache.lucene.document.NumericDocValuesField;
import org.apache.lucene.document.SortedDocValuesField;
import org.apache.lucene.document.SortedNumericDocValuesField;
import org.apache.lucene.document.SortedSetDocValuesField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.SegmentCommitInfo;
import org.apache.lucene.index.SegmentInfos;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.DoubleValues;
import org.apache.lucene.search.DoubleValuesSource;
import org.apache.lucene.search.DoubleValuesSourceRescorer;
import org.apache.lucene.search.FieldDoc;
import org.apache.lucene.search.FullPrecisionFloatVectorSimilarityValuesSource;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.LateInteractionFloatValuesSource;
import org.apache.lucene.search.LateInteractionRescorer;
import org.apache.lucene.search.LongValues;
import org.apache.lucene.search.LongValuesSource;
import org.apache.lucene.search.NumericFieldStats;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.QueryRescorer;
import org.apache.lucene.search.RescoreTopNQuery;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.Sort;
import org.apache.lucene.search.SortField;
import org.apache.lucene.search.SortRescorer;
import org.apache.lucene.search.SortedNumericSelector;
import org.apache.lucene.search.SortedNumericSortField;
import org.apache.lucene.search.SortedSetSelector;
import org.apache.lucene.search.SortedSetSortField;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.BytesRef;
import org.apache.lucene.util.NumericUtils;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Comparator;
import java.util.List;
import java.util.Random;
import java.util.stream.Stream;

/**
 * The values-source API and the rescorers, recorded from Lucene for
 * crates/lucene-search/tests/rescore_fixtures.rs.
 *
 * <p>Two segments (the first with deletions) with numeric, float, double, sorted, sorted-set and
 * sorted-numeric doc values (sparse), points and a doc-values skip index, float and byte vectors
 * and a {@link LateInteractionField}. Recorded:
 *
 * <ul>
 *   <li>{@code vs.NAME} / {@code ls.NAME}: every document's value (deleted ones too) under each
 *       double / long values source, as {@code doc:bits} (a double's raw bits, a long as is);
 *   <li>{@code stats.FIELD}: {@code NumericFieldStats.getStats};
 *   <li>{@code fp.N}: a first-pass top 40, then {@code qr.*}, {@code dvr.*}, {@code late.*},
 *       {@code sort.*} and {@code rtn.*}: QueryRescorer, DoubleValuesSourceRescorer,
 *       LateInteractionRescorer, SortRescorer and RescoreTopNQuery results over it, as {@code
 *       doc:scoreBits[:sortValue...]}.
 * </ul>
 */
public class GenValuesRescore {
  static final int[] SEGMENTS = {300, 250};
  static final String[] TERMS = {"apple", "b", "banana", "cherry", "déjà", "zeta", "ab"};

  static final String[] FIRST_PASS = {
    "(t w0)", "(b 0 (? (t w1)) (? (t w2)))", "(all)",
  };
  static final String[] SECOND_PASS = {"(t w3)", "(p w0 w1)", "(b 0 (+ (t w2)) (- (t w5)))"};
  static final String[] QUERY_SOURCES = {
    "(t w1)", "(b 0 (? (t w2)) (? (p w1 w3)))", "(boost 2.5 (t w5))",
  };

  static final float[] QV = {0.5f, -1.25f, 2f, 0.75f};
  static final float[] QEU = {1f, 0.5f, -0.5f};
  static final byte[] QB = {3, -7, 12, 1};
  static final float[][] QMV = {{0.5f, 1f, -0.25f}, {1.5f, -0.5f, 0.25f}};

  public static void main(String[] args) throws IOException {
    Path out = Path.of(args[0]).resolve("values_rescore_index");
    if (Files.exists(out)) {
      try (Stream<Path> s = Files.walk(out)) {
        s.sorted(Comparator.reverseOrder()).forEach(p -> p.toFile().delete());
      }
    }
    Files.createDirectories(out);
    StringBuilder m = new StringBuilder();
    Random random = new Random(20260930L);
    try (Directory dir = FSDirectory.open(out)) {
      IndexWriterConfig cfg = new IndexWriterConfig(new StandardAnalyzer());
      cfg.setUseCompoundFile(false);
      cfg.setMergePolicy(NoMergePolicy.INSTANCE);
      cfg.setRAMBufferSizeMB(256);
      cfg.setMaxBufferedDocs(IndexWriterConfig.DISABLE_AUTO_FLUSH);
      int id = 0;
      try (IndexWriter w = new IndexWriter(dir, cfg)) {
        for (int size : SEGMENTS) {
          for (int i = 0; i < size; i++, id++) {
            w.addDocument(doc(random, id));
          }
          w.commit();
        }
        for (int d = 0; d < SEGMENTS[0]; d += 13) {
          w.deleteDocuments(new Term("id", Integer.toString(d)));
        }
        w.commit();
      }

      SegmentInfos sis = SegmentInfos.readLatestCommit(dir);
      if (sis.size() != SEGMENTS.length) {
        throw new AssertionError("expected " + SEGMENTS.length + " segments, got " + sis.size());
      }
      int docBase = 0;
      for (int s = 0; s < sis.size(); s++) {
        SegmentCommitInfo sci = sis.info(s);
        String p = "s" + s + ".";
        m.append(p).append("segment_name=").append(sci.info.name).append('\n');
        m.append(p).append("doc_base=").append(docBase).append('\n');
        for (String f : sci.info.files()) {
          for (String ext : new String[] {"vec", "vemf", "vem", "vex"}) {
            if (f.endsWith("." + ext)) {
              m.append(p).append(ext).append("_file=").append(f).append('\n');
              String suffix = f.substring(sci.info.name.length() + 1, f.length() - ext.length() - 1);
              m.append(p).append("vector_suffix=").append(suffix).append('\n');
            }
          }
        }
        docBase += sci.info.maxDoc();
      }

      try (DirectoryReader reader = DirectoryReader.open(dir)) {
        IndexSearcher searcher = new IndexSearcher(reader);
        searcher.setQueryCache(null);

        // Values sources over every document.
        String[] dnames = {
          "long", "int", "float", "double", "const", "query0", "query1", "query2", "fvec", "feu",
          "bvec", "full", "full_mip", "late_cos", "late_eu", "from_long",
        };
        DoubleValuesSource[] dsources = {
          DoubleValuesSource.fromLongField("n"),
          DoubleValuesSource.fromIntField("i"),
          DoubleValuesSource.fromFloatField("f"),
          DoubleValuesSource.fromDoubleField("d"),
          DoubleValuesSource.constant(2.5),
          DoubleValuesSource.fromQuery(query(QUERY_SOURCES[0])),
          DoubleValuesSource.fromQuery(query(QUERY_SOURCES[1])),
          DoubleValuesSource.fromQuery(query(QUERY_SOURCES[2])),
          null,
          null,
          null,
          new FullPrecisionFloatVectorSimilarityValuesSource(QV, "vec"),
          new FullPrecisionFloatVectorSimilarityValuesSource(
              QEU, "veu", VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT),
          new LateInteractionFloatValuesSource("li", QMV),
          new LateInteractionFloatValuesSource("li", QMV, VectorSimilarityFunction.EUCLIDEAN),
          LongValuesSource.fromLongField("n").toDoubleValuesSource(),
        };
        for (int k = 0; k < QUERY_SOURCES.length; k++) {
          m.append("query_source.").append(k).append('=').append(QUERY_SOURCES[k]).append('\n');
        }
        for (int s = 0; s < dnames.length; s++) {
          StringBuilder b = new StringBuilder();
          for (LeafReaderContext ctx : reader.leaves()) {
            DoubleValues v;
            switch (dnames[s]) {
              case "fvec" -> v = DoubleValuesSource.similarityToQueryVector(ctx, QV, "vec");
              case "feu" -> v = DoubleValuesSource.similarityToQueryVector(ctx, QEU, "veu");
              case "bvec" -> v = DoubleValuesSource.similarityToQueryVector(ctx, QB, "bvec");
              default -> v = dsources[s].rewrite(searcher).getValues(ctx, null);
            }
            for (int doc = 0; doc < ctx.reader().maxDoc(); doc++) {
              if (v.advanceExact(doc)) {
                if (b.length() > 0) b.append(',');
                b.append(ctx.docBase + doc)
                    .append(':')
                    .append(Double.doubleToRawLongBits(v.doubleValue()));
              }
            }
          }
          m.append("vs.").append(dnames[s]).append('=').append(b).append('\n');
        }
        String[] lnames = {"long", "int", "const", "cast", "sortable"};
        LongValuesSource[] lsources = {
          LongValuesSource.fromLongField("n"),
          LongValuesSource.fromIntField("i"),
          LongValuesSource.constant(-7),
          DoubleValuesSource.fromDoubleField("d").toLongValuesSource(),
          DoubleValuesSource.fromFloatField("f").toSortableLongDoubleValuesSource(),
        };
        for (int s = 0; s < lnames.length; s++) {
          StringBuilder b = new StringBuilder();
          for (LeafReaderContext ctx : reader.leaves()) {
            LongValues v = lsources[s].getValues(ctx, null);
            for (int doc = 0; doc < ctx.reader().maxDoc(); doc++) {
              if (v.advanceExact(doc)) {
                if (b.length() > 0) b.append(',');
                b.append(ctx.docBase + doc).append(':').append(v.longValue());
              }
            }
          }
          m.append("ls.").append(lnames[s]).append('=').append(b).append('\n');
        }

        // NumericFieldStats.
        for (String f : new String[] {"r", "ip", "rs", "n", "missing"}) {
          NumericFieldStats.Stats st = NumericFieldStats.getStats(reader, f);
          m.append("stats.").append(f).append('=');
          if (st == null) {
            m.append("null");
          } else {
            m.append(st.min()).append(':').append(st.max()).append(':').append(st.docCount());
          }
          m.append('\n');
        }

        // Sorting by a values source (`getSortField`).
        String[] vsorts = {
          "float", "double_rev", "long", "long_rev", "scores", "late", "float_scores", "query",
          "query_rev", "full", "query_long",
        };
        int vsort = 0;
        for (String qs : new String[] {"(t w0)", "(all)"}) {
          for (String vs : vsorts) {
            SortField[] fields =
                switch (vs) {
                  case "float" -> new SortField[] {DoubleValuesSource.fromFloatField("f").getSortField(false)};
                  case "double_rev" -> new SortField[] {DoubleValuesSource.fromDoubleField("d").getSortField(true, -1.5)};
                  case "long" -> new SortField[] {LongValuesSource.fromLongField("n").getSortField(false)};
                  case "long_rev" -> new SortField[] {LongValuesSource.fromLongField("n").getSortField(true, 42)};
                  case "scores" -> new SortField[] {DoubleValuesSource.SCORES.getSortField(true)};
                  case "late" -> new SortField[] {new LateInteractionFloatValuesSource("li", QMV).getSortField(true)};
                  case "query" -> new SortField[] {
                    DoubleValuesSource.fromQuery(query(QUERY_SOURCES[1])).getSortField(false)
                  };
                  case "query_rev" -> new SortField[] {
                    DoubleValuesSource.fromQuery(query(QUERY_SOURCES[0])).getSortField(true),
                    DoubleValuesSource.SCORES.getSortField(true)
                  };
                  case "full" -> new SortField[] {
                    new FullPrecisionFloatVectorSimilarityValuesSource(
                            QEU, "veu", VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT)
                        .getSortField(true)
                  };
                  case "query_long" -> new SortField[] {
                    DoubleValuesSource.fromQuery(query(QUERY_SOURCES[2]))
                        .toLongValuesSource()
                        .getSortField(true)
                  };
                  default -> new SortField[] {
                    DoubleValuesSource.fromFloatField("f").getSortField(true),
                    DoubleValuesSource.SCORES.getSortField(false)
                  };
                };
            org.apache.lucene.search.TopFieldDocs td =
                searcher.search(query(qs), 15, new Sort(fields));
            String k = "vsort." + vsort++;
            m.append(k).append(".query=").append(qs).append('\n');
            m.append(k).append(".sort=").append(vs).append('\n');
            StringBuilder b = new StringBuilder();
            for (ScoreDoc sd : td.scoreDocs) {
              if (b.length() > 0) b.append(',');
              b.append(sd.doc);
              for (Object v : ((FieldDoc) sd).fields) {
                b.append(':');
                if (v instanceof Double dv) {
                  b.append('d').append(Double.doubleToRawLongBits(dv));
                } else {
                  b.append('l').append(v);
                }
              }
            }
            m.append(k).append(".hits=").append(b).append('\n');
            m.append(k).append(".total=").append(td.totalHits.value()).append('\n');
          }
        }
        m.append("vsort_count=").append(vsort).append('\n');

        // Rescorers.
        int qr = 0, dvr = 0, late = 0, sort = 0, rtn = 0, rtnb = 0;
        for (int f = 0; f < FIRST_PASS.length; f++) {
          TopDocs first = searcher.search(query(FIRST_PASS[f]), 40);
          m.append("fp.").append(f).append(".query=").append(FIRST_PASS[f]).append('\n');
          m.append("fp.").append(f).append(".hits=").append(hits(first, null)).append('\n');
          m.append("fp.").append(f).append(".total=").append(first.totalHits.value()).append('\n');

          for (String second : SECOND_PASS) {
            for (double weight : new double[] {1.0, 2.5}) {
              for (int topN : new int[] {10, 40}) {
                TopDocs r =
                    QueryRescorer.rescore(searcher, copy(first), query(second), weight, topN);
                String k = "qr." + qr++;
                m.append(k).append(".fp=").append(f).append('\n');
                m.append(k).append(".query=").append(second).append('\n');
                m.append(k).append(".weight=").append(weight).append('\n');
                m.append(k).append(".top_n=").append(topN).append('\n');
                m.append(k).append(".hits=").append(hits(r, null)).append('\n');
              }
            }
          }

          for (String src : new String[] {"float", "fvec", "query0"}) {
            DoubleValuesSource source =
                switch (src) {
                  case "float" -> DoubleValuesSource.fromFloatField("f");
                  case "fvec" -> new FullPrecisionFloatVectorSimilarityValuesSource(QV, "vec");
                  default -> DoubleValuesSource.fromQuery(query(QUERY_SOURCES[0]));
                };
            DoubleValuesSourceRescorer rescorer =
                new DoubleValuesSourceRescorer(source) {
                  @Override
                  protected float combine(float firstPassScore, boolean present, double value) {
                    return present ? (float) (firstPassScore + 0.5 * value) : firstPassScore * 0.5f;
                  }
                };
            for (int topN : new int[] {7, 40}) {
              TopDocs r = rescorer.rescore(searcher, copy(first), topN);
              String k = "dvr." + dvr++;
              m.append(k).append(".fp=").append(f).append('\n');
              m.append(k).append(".source=").append(src).append('\n');
              m.append(k).append(".top_n=").append(topN).append('\n');
              m.append(k).append(".hits=").append(hits(r, null)).append('\n');
            }
          }

          for (String kind : new String[] {"plain", "fallback"}) {
            LateInteractionRescorer rescorer =
                kind.equals("plain")
                    ? LateInteractionRescorer.create("li", QMV)
                    : LateInteractionRescorer.withFallbackToFirstPassScore(
                        "li", QMV, VectorSimilarityFunction.DOT_PRODUCT);
            TopDocs r = rescorer.rescore(searcher, copy(first), 15);
            String k = "late." + late++;
            m.append(k).append(".fp=").append(f).append('\n');
            m.append(k).append(".kind=").append(kind).append('\n');
            m.append(k).append(".hits=").append(hits(r, null)).append('\n');
          }

          String[][] sorts = {
            {"n:long:false"},
            {"s:string_first:false"},
            {"s:string_last:true", "score:score:false"},
            {"ss:middle_min:false"},
            {"ss:middle_max:true"},
            {"ss:min:false", "n:long:true"},
            {"ss:max:false"},
            {"sn:max_long:true", "score:score:false"},
            {"sd:double:false"},
            {"score:score:false", "doc:doc:false"},
          };
          for (String[] spec : sorts) {
            for (int topN : new int[] {10, 40}) {
              TopDocs r = new SortRescorer(sort(spec)).rescore(searcher, copy(first), topN);
              String k = "sort." + sort++;
              m.append(k).append(".fp=").append(f).append('\n');
              m.append(k).append(".sort=").append(String.join(",", spec)).append('\n');
              m.append(k).append(".top_n=").append(topN).append('\n');
              m.append(k).append(".hits=").append(hits(r, spec)).append('\n');
            }
          }

          for (String src : new String[] {"float", "full", "late"}) {
            for (int n : new int[] {5, 30}) {
              Query q =
                  switch (src) {
                    case "float" ->
                        new RescoreTopNQuery(
                            query(FIRST_PASS[f]), DoubleValuesSource.fromFloatField("f"), n);
                    case "full" ->
                        RescoreTopNQuery.createFullPrecisionRescorerQuery(
                            query(FIRST_PASS[f]), QV, "vec", n);
                    default ->
                        RescoreTopNQuery.createLateInteractionQuery(
                            query(FIRST_PASS[f]), n, "li", QMV, VectorSimilarityFunction.COSINE);
                  };
              TopDocs r = searcher.search(q, 20);
              String k = "rtn." + rtn++;
              m.append(k).append(".fp=").append(f).append('\n');
              m.append(k).append(".source=").append(src).append('\n');
              m.append(k).append(".n=").append(n).append('\n');
              m.append(k).append(".hits=").append(hits(r, null)).append('\n');
              m.append(k).append(".total=").append(r.totalHits.value()).append('\n');
            }
          }

          // RescoreTopNQuery as a clause: required, next to an optional term.
          for (int n : new int[] {5, 30}) {
            Query q =
                new BooleanQuery.Builder()
                    .add(
                        new RescoreTopNQuery(
                            query(FIRST_PASS[f]), DoubleValuesSource.fromFloatField("f"), n),
                        BooleanClause.Occur.MUST)
                    .add(query("(t w1)"), BooleanClause.Occur.SHOULD)
                    .build();
            TopDocs r = searcher.search(q, 20);
            String k = "rtnb." + rtnb++;
            m.append(k).append(".fp=").append(f).append('\n');
            m.append(k).append(".n=").append(n).append('\n');
            m.append(k).append(".hits=").append(hits(r, null)).append('\n');
            m.append(k).append(".total=").append(r.totalHits.value()).append('\n');
          }
        }
        m.append("rtnb_count=").append(rtnb).append('\n');
        m.append("qr_count=").append(qr).append('\n');
        m.append("dvr_count=").append(dvr).append('\n');
        m.append("late_count=").append(late).append('\n');
        m.append("sort_count=").append(sort).append('\n');
        m.append("rtn_count=").append(rtn).append('\n');
      }
    }
    Files.writeString(out.resolve("manifest.properties"), m.toString(), StandardCharsets.UTF_8);
  }

  static Document doc(Random r, int id) {
    Document d = new Document();
    d.add(new StringField("id", Integer.toString(id), Field.Store.NO));
    d.add(new TextField("body", GenMixedBooleanScoring.body(r), Field.Store.NO));
    if (id % 5 != 0) {
      long v = r.nextInt(8) == 0 ? (r.nextBoolean() ? Long.MAX_VALUE : Long.MIN_VALUE) : r.nextInt(2001) - 1000;
      d.add(new NumericDocValuesField("n", v));
    }
    d.add(new NumericDocValuesField("i", r.nextInt()));
    if (id % 4 != 1) {
      float[] fs = {0.5f, -0.0f, 0f, 3.25f, -17.5f, Float.MIN_VALUE, 1e30f};
      d.add(new FloatDocValuesField("f", r.nextInt(3) == 0 ? fs[r.nextInt(fs.length)] : r.nextFloat() * 10 - 5));
    }
    if (id % 3 != 2) {
      double[] ds = {-0.0, 0.0, 1e300, -2.5, Double.MIN_VALUE};
      d.add(new DoubleDocValuesField("d", r.nextInt(3) == 0 ? ds[r.nextInt(ds.length)] : r.nextGaussian() * 1e6));
    }
    int sn = r.nextInt(4);
    for (int k = 0; k < sn; k++) {
      d.add(new SortedNumericDocValuesField("sn", r.nextInt(21) - 10));
    }
    if (id % 6 != 3) {
      d.add(new SortedNumericDocValuesField("sd", NumericUtils.doubleToSortableLong(r.nextGaussian())));
    }
    if (id % 5 != 4) {
      d.add(new SortedDocValuesField("s", new BytesRef(TERMS[r.nextInt(TERMS.length)])));
    }
    int ss = r.nextInt(5);
    for (int k = 0; k < ss; k++) {
      d.add(new SortedSetDocValuesField("ss", new BytesRef(TERMS[r.nextInt(TERMS.length)])));
    }
    if (id % 10 != 7) {
      d.add(new LongPoint("r", (long) r.nextInt(100_000) - 50_000));
    }
    d.add(new IntPoint("ip", r.nextInt(1000) - 500));
    if (id % 3 != 0) {
      d.add(NumericDocValuesField.indexedField("rs", r.nextInt(5000) - 2500));
    }
    if (id % 7 != 3) {
      float[] v = new float[4];
      for (int k = 0; k < 4; k++) v[k] = r.nextFloat() * 2 - 1;
      d.add(new KnnFloatVectorField("vec", v, VectorSimilarityFunction.COSINE));
    }
    if (id % 3 != 1) {
      float[] v = new float[3];
      for (int k = 0; k < 3; k++) v[k] = r.nextFloat() * 4 - 2;
      d.add(new KnnFloatVectorField("veu", v, VectorSimilarityFunction.EUCLIDEAN));
    }
    if (id % 4 != 2) {
      byte[] v = new byte[4];
      for (int k = 0; k < 4; k++) v[k] = (byte) (r.nextInt(256) - 128);
      d.add(new KnnByteVectorField("bvec", v, VectorSimilarityFunction.DOT_PRODUCT));
    }
    if (id % 10 < 7) {
      int tokens = 1 + r.nextInt(4);
      float[][] mv = new float[tokens][3];
      for (float[] t : mv) for (int k = 0; k < 3; k++) t[k] = r.nextFloat() * 2 - 1;
      d.add(new LateInteractionField("li", mv));
    }
    return d;
  }

  static Query query(String s) {
    return GenSortedSearch.parse(new GenMixedBooleanScoring.Tokens(s));
  }

  static TopDocs copy(TopDocs td) {
    ScoreDoc[] sds = new ScoreDoc[td.scoreDocs.length];
    for (int i = 0; i < sds.length; i++) {
      sds[i] = new ScoreDoc(td.scoreDocs[i].doc, td.scoreDocs[i].score);
    }
    return new TopDocs(td.totalHits, sds);
  }

  static Sort sort(String[] spec) {
    SortField[] fields = new SortField[spec.length];
    for (int i = 0; i < spec.length; i++) {
      String[] p = spec[i].split(":");
      boolean reverse = Boolean.parseBoolean(p[2]);
      fields[i] =
          switch (p[1]) {
            case "long" -> new SortField(p[0], SortField.Type.LONG, reverse);
            case "string_first" -> {
              SortField sf = new SortField(p[0], SortField.Type.STRING, reverse);
              sf.setMissingValue(SortField.STRING_FIRST);
              yield sf;
            }
            case "string_last" -> {
              SortField sf = new SortField(p[0], SortField.Type.STRING, reverse);
              sf.setMissingValue(SortField.STRING_LAST);
              yield sf;
            }
            case "min" -> new SortedSetSortField(p[0], reverse, SortedSetSelector.Type.MIN);
            case "max" -> new SortedSetSortField(p[0], reverse, SortedSetSelector.Type.MAX);
            case "middle_min" -> new SortedSetSortField(p[0], reverse, SortedSetSelector.Type.MIDDLE_MIN);
            case "middle_max" -> new SortedSetSortField(p[0], reverse, SortedSetSelector.Type.MIDDLE_MAX);
            case "max_long" ->
                new SortedNumericSortField(
                    p[0], SortField.Type.LONG, reverse, SortedNumericSelector.Type.MAX);
            case "double" -> new SortedNumericSortField(p[0], SortField.Type.DOUBLE, reverse);
            case "score" -> new SortField(null, SortField.Type.SCORE, reverse);
            case "doc" -> new SortField(null, SortField.Type.DOC, reverse);
            default -> throw new AssertionError(p[1]);
          };
    }
    return new Sort(fields);
  }

  static String hits(TopDocs td, String[] spec) {
    StringBuilder b = new StringBuilder();
    for (ScoreDoc sd : td.scoreDocs) {
      if (b.length() > 0) b.append(',');
      b.append(sd.doc).append(':').append(Float.floatToRawIntBits(sd.score));
      if (spec != null) {
        FieldDoc fd = (FieldDoc) sd;
        for (int k = 0; k < spec.length; k++) {
          Object v = fd.fields[k];
          b.append(':');
          if (v == null) {
            b.append('-');
          } else if (v instanceof BytesRef br) {
            b.append('x');
            for (int i = 0; i < br.length; i++) {
              b.append(String.format("%02x", br.bytes[br.offset + i] & 0xff));
            }
          } else if (v instanceof Double dv) {
            b.append(Double.doubleToRawLongBits(dv));
          } else if (v instanceof Float fv) {
            b.append(Float.floatToRawIntBits(fv));
          } else {
            b.append(v);
          }
        }
      }
    }
    return b.toString();
  }
}
