import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.InetAddressPoint;
import org.apache.lucene.document.IntPoint;
import org.apache.lucene.document.KnnByteVectorField;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.LongPoint;
import org.apache.lucene.document.NumericDocValuesField;
import org.apache.lucene.document.SortedNumericDocValuesField;
import org.apache.lucene.document.SortedSetDocValuesField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.AutomatonQuery;
import org.apache.lucene.search.BayesianScoreEstimator;
import org.apache.lucene.search.BayesianScoreQuery;
import org.apache.lucene.search.BlendedTermQuery;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.BoostQuery;
import org.apache.lucene.search.ByteVectorSimilarityQuery;
import org.apache.lucene.search.CombinedFieldQuery;
import org.apache.lucene.search.ConstantScoreQuery;
import org.apache.lucene.search.DisjunctionMaxQuery;
import org.apache.lucene.search.FloatVectorSimilarityQuery;
import org.apache.lucene.search.FuzzyQuery;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.IndexSortSortedNumericDocValuesRangeQuery;
import org.apache.lucene.search.IndriAndQuery;
import org.apache.lucene.search.KnnByteVectorQuery;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.LogOddsFusionQuery;
import org.apache.lucene.search.MatchAllDocsQuery;
import org.apache.lucene.search.MultiPhraseQuery;
import org.apache.lucene.search.MultiTermQuery;
import org.apache.lucene.search.NGramPhraseQuery;
import org.apache.lucene.search.PatienceKnnVectorQuery;
import org.apache.lucene.search.PhraseQuery;
import org.apache.lucene.search.PrefixQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.RegexpQuery;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.SeededKnnVectorQuery;
import org.apache.lucene.search.Sort;
import org.apache.lucene.search.SortedNumericSortField;
import org.apache.lucene.search.SortField;
import org.apache.lucene.search.SynonymQuery;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.search.TermRangeQuery;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.WildcardQuery;
import org.apache.lucene.search.knn.KnnSearchStrategy;
import org.apache.lucene.search.similarities.AfterEffectB;
import org.apache.lucene.search.similarities.BM25Similarity;
import org.apache.lucene.search.similarities.BasicModelIn;
import org.apache.lucene.search.similarities.ClassicSimilarity;
import org.apache.lucene.search.similarities.DFRSimilarity;
import org.apache.lucene.search.similarities.NormalizationH2;
import org.apache.lucene.search.similarities.Similarity;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.BytesRef;
import org.apache.lucene.util.automaton.Operations;
import org.apache.lucene.util.automaton.RegExp;

import java.io.IOException;
import java.net.InetAddress;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.stream.Stream;

/**
 * M7's query and scoring half, for crates/lucene-search/tests/m7_query_fixtures.rs: every query
 * Lucene 10.5.0 adds on top of the term/phrase/boolean core -- synonyms, BM25F, n-gram phrases,
 * explicit phrase positions, the multi-term rewrite methods, blended terms, the Indri and fusion
 * queries, numeric doc-values and index-sort ranges, multi-dimensional and 4/16-byte points,
 * vector similarity thresholds and KNN as a query clause -- searched over one index.
 *
 * <p>{@code m7_queries_index}: three segments (NoMergePolicy, a commit after each batch), two with
 * deleted documents, index-sorted by {@code snum}. Fields: {@code body}/{@code title} text,
 * {@code gram} (letter bigrams, one token each), {@code tag} (keywords, also SORTED_SET doc
 * values), {@code num} (LongPoint + NUMERIC doc values, on most documents), {@code snum} (the
 * index sort, SORTED_NUMERIC), {@code p2} (a 2-D IntPoint), {@code i1} (a 1-D IntPoint),
 * {@code ip} (a 16-byte InetAddressPoint), {@code vec} (4-d float vectors, EUCLIDEAN) and
 * {@code bvec} (4-d byte vectors, EUCLIDEAN).
 *
 * <p>{@code searches.tsv}: {@code sim  query  hits}, the query in the prefix syntax {@link
 * #parse} reads (the Rust test reads it too), the top 20 as {@code doc:scoreBitsHex} with global
 * doc ids. {@code estimator.tsv}: {@code BayesianScoreEstimator.estimate} over {@code body} for a
 * few settings, as float bits.
 */
public class GenM7Queries {
  static final int TOP = 20;

  static Map<String, Similarity> sims() {
    Map<String, Similarity> m = new LinkedHashMap<>();
    m.put("bm25", new BM25Similarity());
    m.put("classic", new ClassicSimilarity());
    m.put("dfr", new DFRSimilarity(new BasicModelIn(), new AfterEffectB(), new NormalizationH2()));
    return m;
  }

  /** Queries whose scores come from a similarity: run under every similarity in {@link #sims}. */
  static final String[] SCORED = {
    // SynonymQuery
    "S body 2 t1 1 t2 1",
    "S body 2 t0 1 t5 0.5",
    "S body 3 t3 0.25 t4 1 t9 0.75",
    "S body 1 t6 0.5",
    "S body 2 t7 1 zz 1",
    "S title 3 t0 1 t1 0.3 t2 0.6",
    "B 0 2 0 S body 2 t1 1 t8 0.5 T body t2",
    "B 1 1 0 T body t0 S body 2 t3 1 t11 0.5",
    // CombinedFieldQuery
    "CF t1 2 body 1 title 1",
    "CF t2 2 body 1 title 3",
    "CF t5 2 body 2.5 title 1",
    "CF t0 1 title 1",
    "CF zz 2 body 1 title 1",
    "B 0 2 0 CF t3 2 body 1 title 2 T body t4",
    // Phrases with explicit positions
    "PP body 0 2 t0 0 t1 2",
    "PP body 0 3 t0 0 t1 1 t2 3",
    "PP body 1 2 t0 1 t2 3",
    "PP body 2 3 t1 0 t0 2 t1 3",
    "PP body 0 2 t0 0 t0 2",
    "B 0 2 0 PP body 0 2 t0 0 t2 2 T title t1",
    // NGramPhraseQuery
    "NG 2 gram 0 4 ab bc ca ab",
    "NG 2 gram 0 3 ab ba ab",
    "NG 2 gram 0 5 ab bc cd da ab",
    "NG 2 gram 1 3 ab bc cd",
    "NG 3 gram 0 2 ab bc",
    // MultiPhraseQuery (explicit positions, sloppy, in a boolean)
    "MP body 0 2 2 t0 t1 0 1 t2 1",
    "MP body 0 2 1 t0 0 2 t1 t3 2",
    "MP body 1 3 1 t1 0 2 t0 t2 1 1 t3 2",
    "B 0 2 0 MP body 0 2 2 t3 t4 0 1 t0 1 T body t5",
    // FuzzyQuery under a similarity
    "F body t1 1 0",
    "F title t10 1 1",
    // The rewrite methods
    "PF body t1 sb",
    "PF body t1 csbool",
    "PF body t1 tts:3",
    "PF body t1 ttb:3",
    "PF body t1 ttbf:3",
    "PF body t1 cs",
    "PF body t1 csb",
    "W body t?3 sb",
    "RE body t[0-2]. tts:2",
    "R body t10 t20 1 0 sb",
    "R body t2 t35 0 1 tts:4",
    "R title - t3 1 1 csb",
    "R body t30 - 0 1 cs",
    "A body t3.* sb",
    "A body (t1|t2)5? csb",
    "A title t1[01]? ttbf:5",
    // BlendedTermQuery
    "BT bool 3 body t1 1 body t2 1 title t1 1",
    "BT dismax:0.1 3 body t3 1 body t30 2 title t3 1",
    "BT dismax:0.01 2 body t0 1 title t0 0.5",
    // Indri
    "IA 2 T body t1 T body t2",
    "IA 1 T body t3",
    "IA 2 IA 2 T body t0 T body t4 T title t2",
    // Log-odds fusion
    "LO 0.5 - - 2 T body t1 T title t1",
    "LO 0 - - 3 T body t2 T body t3 T title t0",
    "LO 1 0.7,0.3 - 2 T body t4 T title t2",
    "LO 0.5 - -2,-1:3,4 2 T body t0 T body t5",
    "LO 0.5 - - 2 T body t1 T body zz",
    // Bayesian calibration
    "BY 1.5 1 0 T body t2",
    "BY 0.5 2 0.1 B 0 2 0 T body t1 T title t1",
    "BY 2 0.5 0.3 PP body 0 2 t0 0 t1 1",
  };

  /** Constant-scored or similarity-independent queries: run under BM25 only. */
  static final String[] UNSCORED = {
    // DocValuesRewriteMethod over SORTED_SET doc values
    "PF tag k1 dv",
    "RE tag k[0-9]5 dv",
    "R tag k03 k11 1 1 dv",
    "W tag k?? dv",
    // Numeric doc-values range, in and out of a boolean
    "NR num -20 40",
    "NR num 50 50",
    "NR num 1000 2000",
    "B 1 0 0 T body t1 F2 NR num 0 100",
    // The index-sort range
    "ISR snum 10 60 -",
    "ISR snum -5 5 -",
    "ISR snum 100 1000 -",
    "ISR num 0 50 NR num 0 50",
    "B 1 0 0 T body t0 F2 ISR snum 20 70 -",
    // Points: 2-D int, 1-D int, 16-byte
    "P2R p2 -10 -10 10 10",
    "P2R p2 0 -50 50 0",
    "I1R i1 -5 5",
    "I1S i1 3 -3 7 11 400",
    "LS num 5 7 -3 99 1000",
    "IPR ip 10.0.0.0 10.0.0.60",
    "IPS ip 10.0.0.3 10.0.0.9 10.0.0.77 10.0.1.1",
    "B 1 0 0 T body t2 F2 P2R p2 -30 -30 30 30",
    // Vector similarity thresholds
    "VSF vec 0.05",
    "VSF vec 0.2",
    "VSFF vec 0.05 T body t1",
    "VSB bvec 0.001",
    // KNN as a clause
    "KF vec 5",
    "KF vec 12",
    "KFF vec 8 T body t0",
    "KB bvec 6",
    "B 0 2 0 KF vec 5 T body t1",
    "B 1 1 0 T body t2 KF vec 30",
    "PKF vec 10",
    // KnnSearchStrategy.Hnsw(threshold): FilteredHnswGraphSearcher under a filter
    "KFS vec 3 60 T body t2",
    "KFS vec 2 90 T body t5",
    "KFS vec 4 100 T title t1",
    "KFS vec 3 100 B 0 2 0 T body t9 T body t12",
    "SKF vec 7 T title t1",
  };

  static float[] QVEC = {0.5f, -0.25f, 0.75f, 0.1f};
  static final byte[] QBVEC = {12, -7, 30, 1};

  static int pos;

  static float f(String[] tok) {
    return Float.parseFloat(tok[pos++]);
  }

  static int i(String[] tok) {
    return Integer.parseInt(tok[pos++]);
  }

  static MultiTermQuery.RewriteMethod method(String m) {
    if (m.equals("sb")) return MultiTermQuery.SCORING_BOOLEAN_REWRITE;
    if (m.equals("csbool")) return MultiTermQuery.CONSTANT_SCORE_BOOLEAN_REWRITE;
    if (m.equals("cs")) return MultiTermQuery.CONSTANT_SCORE_REWRITE;
    if (m.equals("csb")) return MultiTermQuery.CONSTANT_SCORE_BLENDED_REWRITE;
    if (m.equals("dv")) return MultiTermQuery.DOC_VALUES_REWRITE;
    String[] p = m.split(":");
    int n = Integer.parseInt(p[1]);
    switch (p[0]) {
      case "tts": return new MultiTermQuery.TopTermsScoringBooleanQueryRewrite(n);
      case "ttb": return new MultiTermQuery.TopTermsBoostOnlyBooleanQueryRewrite(n);
      case "ttbf": return new MultiTermQuery.TopTermsBlendedFreqScoringRewrite(n);
      default: throw new IllegalArgumentException(m);
    }
  }

  static float[] floats(String s) {
    if (s.equals("-")) return null;
    String[] p = s.split(",");
    float[] out = new float[p.length];
    for (int k = 0; k < p.length; k++) out[k] = Float.parseFloat(p[k]);
    return out;
  }

  static List<Query> clauses(String[] tok, int n) {
    List<Query> qs = new ArrayList<>();
    for (int k = 0; k < n; k++) qs.add(parse(tok));
    return qs;
  }

  /**
   * The prefix syntax: every token list starts with an op. {@code T field term}; {@code B nMust
   * nShould nMustNot [F2 nFilter...] clause...} (the filter clauses follow the must-not ones when
   * the {@code F2} marker is present: {@code B 1 0 0 T x y F2 <filter>}); {@code X boost clause};
   * {@code S field n (term boost)...}; {@code CF term n (field weight)...}; {@code PP field slop n
   * (term pos)...}; {@code NG n field slop k term...}; {@code MP field slop n (k term... pos)...};
   * {@code F field term maxEdits prefix}; {@code PF field prefix method}; {@code W field pattern
   * method}; {@code RE field regexp method}; {@code R field lower upper incL incU method} ({@code
   * -} for an open end); {@code A field regexp method} (an AutomatonQuery over the regexp's
   * automaton); {@code BT method n (field term boost)...}; {@code IA n clause...}; {@code LO alpha
   * weights min:max n clause...}; {@code BY alpha beta baseRate clause}; the doc-values, points
   * and vector ops are listed in {@link #UNSCORED}.
   */
  static Query parse(String[] tok) {
    String op = tok[pos++];
    switch (op) {
      case "T":
        return new TermQuery(new Term(tok[pos++], tok[pos++]));
      case "B": {
        int must = i(tok), should = i(tok), mustNot = i(tok);
        BooleanQuery.Builder b = new BooleanQuery.Builder();
        for (int k = 0; k < must; k++) b.add(parse(tok), BooleanClause.Occur.MUST);
        for (int k = 0; k < should; k++) b.add(parse(tok), BooleanClause.Occur.SHOULD);
        for (int k = 0; k < mustNot; k++) b.add(parse(tok), BooleanClause.Occur.MUST_NOT);
        if (pos < tok.length && tok[pos].equals("F2")) {
          pos++;
          b.add(parse(tok), BooleanClause.Occur.FILTER);
        }
        return b.build();
      }
      case "X": {
        float boost = f(tok);
        return new BoostQuery(parse(tok), boost);
      }
      case "S": {
        String field = tok[pos++];
        int n = i(tok);
        SynonymQuery.Builder b = new SynonymQuery.Builder(field);
        for (int k = 0; k < n; k++) {
          String t = tok[pos++];
          b.addTerm(new Term(field, t), f(tok));
        }
        return b.build();
      }
      case "CF": {
        CombinedFieldQuery.Builder b = new CombinedFieldQuery.Builder(tok[pos++]);
        int n = i(tok);
        for (int k = 0; k < n; k++) {
          String field = tok[pos++];
          b.addField(field, f(tok));
        }
        return b.build();
      }
      case "PP": {
        String field = tok[pos++];
        int slop = i(tok), n = i(tok);
        PhraseQuery.Builder b = new PhraseQuery.Builder();
        b.setSlop(slop);
        for (int k = 0; k < n; k++) {
          String t = tok[pos++];
          b.add(new Term(field, t), i(tok));
        }
        return b.build();
      }
      case "NG": {
        int n = i(tok);
        String field = tok[pos++];
        int slop = i(tok), k = i(tok);
        String[] terms = new String[k];
        for (int j = 0; j < k; j++) terms[j] = tok[pos++];
        return new NGramPhraseQuery(n, new PhraseQuery(slop, field, terms));
      }
      case "MP": {
        String field = tok[pos++];
        int slop = i(tok), n = i(tok);
        MultiPhraseQuery.Builder b = new MultiPhraseQuery.Builder();
        b.setSlop(slop);
        for (int k = 0; k < n; k++) {
          int m = i(tok);
          Term[] terms = new Term[m];
          for (int j = 0; j < m; j++) terms[j] = new Term(field, tok[pos++]);
          b.add(terms, i(tok));
        }
        return b.build();
      }
      case "F": {
        String field = tok[pos++];
        String t = tok[pos++];
        int edits = i(tok), prefix = i(tok);
        return new FuzzyQuery(new Term(field, t), edits, prefix);
      }
      case "PF": {
        String field = tok[pos++];
        String p = tok[pos++];
        return new PrefixQuery(new Term(field, p), method(tok[pos++]));
      }
      case "W": {
        String field = tok[pos++];
        String p = tok[pos++];
        return new WildcardQuery(
            new Term(field, p), Operations.DEFAULT_DETERMINIZE_WORK_LIMIT, method(tok[pos++]));
      }
      case "RE": {
        String field = tok[pos++];
        String p = tok[pos++];
        return new RegexpQuery(
            new Term(field, p),
            RegExp.ALL,
            0,
            RegexpQuery.DEFAULT_PROVIDER,
            Operations.DEFAULT_DETERMINIZE_WORK_LIMIT,
            method(tok[pos++]),
            true);
      }
      case "R": {
        String field = tok[pos++];
        String lo = tok[pos++], hi = tok[pos++];
        boolean incl = i(tok) == 1, incu = i(tok) == 1;
        return new TermRangeQuery(
            field,
            lo.equals("-") ? null : new BytesRef(lo),
            hi.equals("-") ? null : new BytesRef(hi),
            incl,
            incu,
            method(tok[pos++]));
      }
      case "A": {
        String field = tok[pos++];
        String re = tok[pos++];
        return new AutomatonQuery(
            new Term(field, re),
            Operations.determinize(
                new RegExp(re).toAutomaton(), Operations.DEFAULT_DETERMINIZE_WORK_LIMIT),
            false,
            method(tok[pos++]));
      }
      case "BT": {
        String m = tok[pos++];
        int n = i(tok);
        BlendedTermQuery.Builder b = new BlendedTermQuery.Builder();
        if (m.equals("bool")) {
          b.setRewriteMethod(BlendedTermQuery.BOOLEAN_REWRITE);
        } else {
          b.setRewriteMethod(
              new BlendedTermQuery.DisjunctionMaxRewrite(Float.parseFloat(m.split(":")[1])));
        }
        for (int k = 0; k < n; k++) {
          String field = tok[pos++];
          String t = tok[pos++];
          b.add(new Term(field, t), f(tok));
        }
        return b.build();
      }
      case "IA": {
        int n = i(tok);
        List<BooleanClause> cs = new ArrayList<>();
        for (int k = 0; k < n; k++) cs.add(new BooleanClause(parse(tok), BooleanClause.Occur.SHOULD));
        return new IndriAndQuery(cs);
      }
      case "LO": {
        float alpha = f(tok);
        float[] weights = floats(tok[pos++]);
        String mm = tok[pos++];
        float[] min = null, max = null;
        if (!mm.equals("-")) {
          String[] p = mm.split(":");
          min = floats(p[0]);
          max = floats(p[1]);
        }
        int n = i(tok);
        return new LogOddsFusionQuery(clauses(tok, n), alpha, weights, min, max);
      }
      case "BY": {
        float alpha = f(tok), beta = f(tok), base = f(tok);
        return new BayesianScoreQuery(parse(tok), alpha, beta, base);
      }
      case "NR": {
        String field = tok[pos++];
        long lo = Long.parseLong(tok[pos++]), hi = Long.parseLong(tok[pos++]);
        return NumericDocValuesField.newSlowRangeQuery(field, lo, hi);
      }
      case "ISR": {
        String field = tok[pos++];
        long lo = Long.parseLong(tok[pos++]), hi = Long.parseLong(tok[pos++]);
        Query fallback;
        if (tok[pos].equals("-")) {
          pos++;
          fallback = SortedNumericDocValuesField.newSlowRangeQuery(field, lo, hi);
        } else {
          fallback = parse(tok);
        }
        return new IndexSortSortedNumericDocValuesRangeQuery(field, lo, hi, fallback);
      }
      case "P2R": {
        String field = tok[pos++];
        int x0 = i(tok), y0 = i(tok), x1 = i(tok), y1 = i(tok);
        return IntPoint.newRangeQuery(field, new int[] {x0, y0}, new int[] {x1, y1});
      }
      case "I1R": {
        String field = tok[pos++];
        return IntPoint.newRangeQuery(field, i(tok), i(tok));
      }
      case "I1S": {
        String field = tok[pos++];
        List<Integer> vs = new ArrayList<>();
        while (pos < tok.length && !tok[pos].equals("F2")) vs.add(i(tok));
        return IntPoint.newSetQuery(field, vs.stream().mapToInt(Integer::intValue).toArray());
      }
      case "LS": {
        String field = tok[pos++];
        List<Long> vs = new ArrayList<>();
        while (pos < tok.length && !tok[pos].equals("F2")) vs.add(Long.parseLong(tok[pos++]));
        return LongPoint.newSetQuery(field, vs);
      }
      case "IPR": {
        String field = tok[pos++];
        return InetAddressPoint.newRangeQuery(field, addr(tok[pos++]), addr(tok[pos++]));
      }
      case "IPS": {
        String field = tok[pos++];
        List<InetAddress> vs = new ArrayList<>();
        while (pos < tok.length && !tok[pos].equals("F2")) vs.add(addr(tok[pos++]));
        return InetAddressPoint.newSetQuery(field, vs.toArray(InetAddress[]::new));
      }
      case "VSF": {
        String field = tok[pos++];
        return new FloatVectorSimilarityQuery(field, QVEC, f(tok));
      }
      case "VSFF": {
        String field = tok[pos++];
        float sim = f(tok);
        return new FloatVectorSimilarityQuery(field, QVEC, sim, parse(tok));
      }
      case "VSB": {
        String field = tok[pos++];
        return new ByteVectorSimilarityQuery(field, QBVEC, f(tok));
      }
      case "KF": {
        String field = tok[pos++];
        return new KnnFloatVectorQuery(field, QVEC, i(tok));
      }
      case "KFF": {
        String field = tok[pos++];
        int k = i(tok);
        return new KnnFloatVectorQuery(field, QVEC, k, parse(tok));
      }
      case "KFS": {
        String field = tok[pos++];
        int k = i(tok), threshold = i(tok);
        return new KnnFloatVectorQuery(
            field, QVEC, k, parse(tok), new KnnSearchStrategy.Hnsw(threshold));
      }
      case "KB": {
        String field = tok[pos++];
        return new KnnByteVectorQuery(field, QBVEC, i(tok));
      }
      case "PKF": {
        String field = tok[pos++];
        return PatienceKnnVectorQuery.fromFloatQuery(new KnnFloatVectorQuery(field, QVEC, i(tok)));
      }
      case "SKF": {
        String field = tok[pos++];
        int k = i(tok);
        return SeededKnnVectorQuery.fromFloatQuery(new KnnFloatVectorQuery(field, QVEC, k), parse(tok));
      }
      default:
        throw new IllegalArgumentException(op);
    }
  }

  static InetAddress addr(String s) {
    try {
      return InetAddress.getByName(s);
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  static String words(Random r, int n, int vocab) {
    StringBuilder sb = new StringBuilder();
    for (int k = 0; k < n; k++) {
      double x = r.nextDouble();
      int w = (int) (vocab * x * x * x);
      if (k > 0) sb.append(' ');
      sb.append('t').append(w);
    }
    return sb.toString();
  }

  /** A random word over {@code abcd} as its letter bigrams, one token each. */
  static String bigrams(Random r) {
    int len = 3 + r.nextInt(10);
    char[] w = new char[len];
    for (int k = 0; k < len; k++) w[k] = (char) ('a' + r.nextInt(r.nextInt(3) == 0 ? 4 : 3));
    StringBuilder sb = new StringBuilder();
    for (int k = 0; k + 1 < len; k++) {
      if (k > 0) sb.append(' ');
      sb.append(w[k]).append(w[k + 1]);
    }
    return sb.toString();
  }

  static void deleteRecursive(Path p) throws IOException {
    if (!Files.exists(p)) return;
    try (Stream<Path> s = Files.walk(p)) {
      s.sorted(Comparator.reverseOrder()).forEach(q -> q.toFile().delete());
    }
  }

  static String hits(TopDocs td) {
    StringBuilder sb = new StringBuilder();
    for (int k = 0; k < td.scoreDocs.length; k++) {
      ScoreDoc sd = td.scoreDocs[k];
      if (k > 0) sb.append(',');
      sb.append(sd.doc).append(':').append(Integer.toHexString(Float.floatToRawIntBits(sd.score)));
    }
    return sb.toString();
  }

  public static void main(String[] args) throws IOException {
    Path root = Path.of(args[0]);
    Path out = root.resolve("m7_queries_index");
    deleteRecursive(out);
    Files.createDirectories(out);
    Random r = new Random(20261001L);

    try (Directory dir = FSDirectory.open(out)) {
      IndexWriterConfig cfg = new IndexWriterConfig(new StandardAnalyzer());
      cfg.setUseCompoundFile(false);
      cfg.setMergePolicy(NoMergePolicy.INSTANCE);
      cfg.setIndexSort(new Sort(new SortedNumericSortField("snum", SortField.Type.LONG)));
      int id = 0;
      try (IndexWriter w = new IndexWriter(dir, cfg)) {
        for (int size : new int[] {90, 60, 30}) {
          for (int k = 0; k < size; k++) {
            Document doc = new Document();
            doc.add(new StringField("id", Integer.toString(id), Field.Store.NO));
            doc.add(new TextField("body", words(r, 1 + r.nextInt(r.nextInt(4) == 0 ? 50 : 12), 40), Field.Store.NO));
            if (r.nextInt(5) != 0) {
              doc.add(new TextField("title", words(r, 1 + r.nextInt(5), 12), Field.Store.NO));
            }
            doc.add(new TextField("gram", bigrams(r), Field.Store.NO));
            int tags = r.nextInt(4);
            for (int t = 0; t < tags; t++) {
              String tag = String.format("k%02d", r.nextInt(20));
              doc.add(new StringField("tag", tag, Field.Store.NO));
              doc.add(new SortedSetDocValuesField("tag", new BytesRef(tag)));
            }
            if (r.nextInt(6) != 0) {
              long v = r.nextInt(120) - 20;
              doc.add(new LongPoint("num", v));
              doc.add(new NumericDocValuesField("num", v));
            }
            doc.add(new SortedNumericDocValuesField("snum", r.nextInt(100)));
            if (r.nextInt(4) != 0) {
              doc.add(new IntPoint("p2", r.nextInt(101) - 50, r.nextInt(101) - 50));
            }
            if (r.nextInt(3) != 0) {
              doc.add(new IntPoint("i1", r.nextInt(31) - 10));
            }
            if (r.nextInt(3) != 0) {
              doc.add(new InetAddressPoint("ip", addr("10.0." + r.nextInt(2) + "." + r.nextInt(100))));
            }
            if (r.nextInt(7) != 0) {
              float[] v = new float[4];
              for (int j = 0; j < 4; j++) v[j] = r.nextFloat() * 2 - 1;
              doc.add(new KnnFloatVectorField("vec", v, VectorSimilarityFunction.EUCLIDEAN));
            }
            if (r.nextInt(5) != 0) {
              byte[] v = new byte[4];
              for (int j = 0; j < 4; j++) v[j] = (byte) (r.nextInt(81) - 40);
              doc.add(new KnnByteVectorField("bvec", v, VectorSimilarityFunction.EUCLIDEAN));
            }
            w.addDocument(doc);
            id++;
          }
          w.commit();
        }
        for (String del : new String[] {"3", "17", "40", "95", "101", "130"}) {
          w.deleteDocuments(new Term("id", del));
        }
        w.commit();
      }

      StringBuilder sb = new StringBuilder();
      StringBuilder est = new StringBuilder();
      try (DirectoryReader reader = DirectoryReader.open(dir)) {
        if (reader.leaves().size() != 3) {
          throw new AssertionError("expected three segments, got " + reader.leaves().size());
        }
        for (Map.Entry<String, Similarity> e : sims().entrySet()) {
          IndexSearcher searcher = new IndexSearcher(reader);
          searcher.setQueryCache(null);
          searcher.setSimilarity(e.getValue());
          List<String> qs = new ArrayList<>(List.of(SCORED));
          if (e.getKey().equals("bm25")) qs.addAll(List.of(UNSCORED));
          for (String q : qs) {
            pos = 0;
            String[] tok = q.split(" ");
            Query query = parse(tok);
            if (pos != tok.length) throw new AssertionError("trailing tokens in " + q);
            TopDocs td = searcher.search(query, TOP);
            sb.append(e.getKey()).append('\t').append(q).append('\t').append(hits(td)).append('\n');
          }
        }
        IndexSearcher searcher = new IndexSearcher(reader);
        searcher.setQueryCache(null);
        int[][] settings = {{50, 5, 42}, {10, 3, 7}, {4, 2, 1}};
        for (int[] s : settings) {
          BayesianScoreEstimator.Parameters p =
              BayesianScoreEstimator.estimate(searcher, "body", s[0], s[1], s[2]);
          est.append(s[0]).append('\t').append(s[1]).append('\t').append(s[2]).append('\t')
              .append(Integer.toHexString(Float.floatToRawIntBits(p.alpha()))).append('\t')
              .append(Integer.toHexString(Float.floatToRawIntBits(p.beta()))).append('\t')
              .append(Integer.toHexString(Float.floatToRawIntBits(p.baseRate()))).append('\n');
        }
      }
      Files.writeString(out.resolve("searches.tsv"), sb);
      Files.writeString(out.resolve("estimator.tsv"), est);
    }
    writeKnnIndex(root);
  }

  /** Vector queries over graphs large enough that filtered, patient and seeded walks happen. */
  static final String[] KNN = {
    "KF vec 10",
    "KF vec 50",
    "KFF vec 10 T color c1",
    "KFF vec 10 T color c0",
    "KFS vec 10 60 T color c1",
    "KFS vec 10 60 T color c3",
    "KFS vec 5 90 T color c0",
    "KFS vec 20 100 B 0 2 0 T color c2 T color c4",
    "PKF vec 10",
    "PKF vec 40",
    "SKF vec 10 T color c1",
    "SKF vec 25 T color c5",
    "VSF vec 0.35",
    "VSF vec 0.5",
    "VSFF vec 0.3 T color c2",
    "B 1 1 0 T color c0 KF vec 20",
  };

  /**
   * {@code m7_knn_index}: two segments of 8-d EUCLIDEAN vectors (1500 and 1000 documents, a few
   * deleted) with a skewed keyword {@code color}, searched with {@link #KNN} (the same prefix
   * syntax) into {@code searches.tsv}. The target is {@link #QVEC8}.
   */
  static void writeKnnIndex(Path root) throws IOException {
    Path out = root.resolve("m7_knn_index");
    deleteRecursive(out);
    Files.createDirectories(out);
    Random r = new Random(20261002L);
    try (Directory dir = FSDirectory.open(out)) {
      IndexWriterConfig cfg = new IndexWriterConfig(new StandardAnalyzer());
      cfg.setUseCompoundFile(false);
      cfg.setMergePolicy(NoMergePolicy.INSTANCE);
      cfg.setRAMBufferSizeMB(64);
      int id = 0;
      try (IndexWriter w = new IndexWriter(dir, cfg)) {
        for (int size : new int[] {1500, 1000}) {
          for (int k = 0; k < size; k++) {
            Document doc = new Document();
            doc.add(new StringField("id", Integer.toString(id++), Field.Store.NO));
            double x = r.nextDouble();
            doc.add(new StringField("color", "c" + (int) (10 * x * x), Field.Store.NO));
            if (r.nextInt(20) != 0) {
              float[] v = new float[8];
              for (int j = 0; j < 8; j++) v[j] = r.nextFloat() * 2 - 1;
              doc.add(new KnnFloatVectorField("vec", v, VectorSimilarityFunction.EUCLIDEAN));
            }
            w.addDocument(doc);
          }
          w.commit();
        }
        for (int d = 7; d < 2500; d += 97) {
          w.deleteDocuments(new Term("id", Integer.toString(d)));
        }
        w.commit();
      }
      StringBuilder sb = new StringBuilder();
      try (DirectoryReader reader = DirectoryReader.open(dir)) {
        if (reader.leaves().size() != 2) {
          throw new AssertionError("expected two segments, got " + reader.leaves().size());
        }
        IndexSearcher searcher = new IndexSearcher(reader);
        searcher.setQueryCache(null);
        float[] saved = QVEC;
        QVEC = QVEC8;
        for (String q : KNN) {
          pos = 0;
          String[] tok = q.split(" ");
          Query query = parse(tok);
          if (pos != tok.length) throw new AssertionError("trailing tokens in " + q);
          TopDocs td = searcher.search(query, 50);
          sb.append("bm25").append('\t').append(q).append('\t').append(hits(td)).append('\n');
        }
        QVEC = saved;
      }
      Files.writeString(out.resolve("searches.tsv"), sb);
    }
  }

  static final float[] QVEC8 = {0.1f, -0.3f, 0.25f, 0.6f, -0.05f, 0.4f, -0.2f, 0.15f};
}
