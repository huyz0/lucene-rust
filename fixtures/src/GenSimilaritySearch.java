import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.FieldType;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FieldInvertState;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.NumericDocValues;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.BoostQuery;
import org.apache.lucene.search.CollectionStatistics;
import org.apache.lucene.search.ConstantScoreQuery;
import org.apache.lucene.search.DisjunctionMaxQuery;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.PhraseQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.search.TermStatistics;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.similarities.AfterEffectB;
import org.apache.lucene.search.similarities.AxiomaticF2EXP;
import org.apache.lucene.search.similarities.BM25Similarity;
import org.apache.lucene.search.similarities.BasicModelIn;
import org.apache.lucene.search.similarities.BooleanSimilarity;
import org.apache.lucene.search.similarities.ClassicSimilarity;
import org.apache.lucene.search.similarities.DFISimilarity;
import org.apache.lucene.search.similarities.DFRSimilarity;
import org.apache.lucene.search.similarities.DistributionLL;
import org.apache.lucene.search.similarities.IBSimilarity;
import org.apache.lucene.search.similarities.IndependenceSaturated;
import org.apache.lucene.search.similarities.LMDirichletSimilarity;
import org.apache.lucene.search.similarities.LMJelinekMercerSimilarity;
import org.apache.lucene.search.similarities.LambdaDF;
import org.apache.lucene.search.similarities.MultiSimilarity;
import org.apache.lucene.search.similarities.NormalizationH2;
import org.apache.lucene.search.similarities.PerFieldSimilarityWrapper;
import org.apache.lucene.search.similarities.Similarity;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;

import java.io.IOException;
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
 * Searches under similarities other than the default, for
 * crates/lucene-search/tests/similarity_search_fixtures.rs: {@code IndexSearcher.setSimilarity}
 * end to end, and {@code IndexWriterConfig.setSimilarity}'s norms.
 *
 * <p>{@code similarity_search_index}: three segments (NoMergePolicy, a commit after each batch),
 * two of them with deleted documents, fields {@code body} (always) and {@code title} (on most
 * documents, so its norms are sparse), words drawn skewed from {@code t0..t39} so some terms are
 * everywhere and some are pulsed singletons in a segment. {@code searches.tsv}: {@code sim  query
 * hits}, the query in the prefix syntax {@link #parse} reads (and the Rust test reads too), the
 * top 20 as {@code doc:scoreBitsHex} with global doc ids.
 *
 * <p>{@code norms_docs.tsv} and {@code norms.tsv}: documents written with a per-field similarity
 * ({@code body}: {@code ClassicSimilarity(false)}; {@code title}: a similarity whose norm is
 * {@code length + 100 * uniqueTermCount}, wider than a byte; {@code tags}, a {@code DOCS} field
 * with repeated tokens: BM25), and the norm Lucene stored for each document and field
 * ({@code -} for none). The index itself is written to a temporary directory: only the norms are
 * evidence.
 */
public class GenSimilaritySearch {
  static final int TOP = 20;

  /** Norm {@code length + 100 * uniqueTermCount}: exercises a norm wider than a byte. */
  static final class WideNormSimilarity extends Similarity {
    @Override
    public long computeNorm(FieldInvertState state) {
      return state.getLength() + 100L * state.getUniqueTermCount();
    }

    @Override
    public SimScorer scorer(float boost, CollectionStatistics c, TermStatistics... t) {
      return new SimScorer() {
        @Override
        public float score(float freq, long norm) {
          return boost;
        }
      };
    }
  }

  static Map<String, Similarity> sims() {
    Map<String, Similarity> m = new LinkedHashMap<>();
    m.put("bm25", new BM25Similarity());
    m.put("bm25_k2_b03", new BM25Similarity(2.0f, 0.3f));
    m.put("classic", new ClassicSimilarity());
    m.put("boolean", new BooleanSimilarity());
    m.put("dfr_In_B_H2", new DFRSimilarity(new BasicModelIn(), new AfterEffectB(), new NormalizationH2()));
    m.put("ib_LL_DF_H2", new IBSimilarity(new DistributionLL(), new LambdaDF(), new NormalizationH2()));
    m.put("dfi_saturated", new DFISimilarity(new IndependenceSaturated()));
    m.put("lmdirichlet", new LMDirichletSimilarity());
    m.put("lmjm_0.7", new LMJelinekMercerSimilarity(0.7f));
    // Not IndriDirichletSimilarity: it scores below zero, which breaks the
    // non-negative-score contract every Scorer and collector relies on
    // (Lucene's own IndexSearcher returns no hits for a lone term under it).
    m.put("ax_f2exp", new AxiomaticF2EXP(0.5f, 0.2f));
    m.put("multi", new MultiSimilarity(new Similarity[] {new BM25Similarity(), new ClassicSimilarity()}));
    Similarity classic = new ClassicSimilarity();
    Similarity lmjm = new LMJelinekMercerSimilarity(0.7f);
    m.put("perfield", new PerFieldSimilarityWrapper() {
      @Override
      public Similarity get(String name) {
        return name.equals("title") ? classic : lmjm;
      }
    });
    return m;
  }

  /**
   * The queries, in a prefix syntax both sides parse: {@code T field term}, {@code P field slop n
   * term...}, {@code B nMust nShould nMustNot clause...}, {@code X boost clause}, {@code C score
   * clause}, {@code D tie n clause...}.
   */
  static final String[] QUERIES = {
    "T body t0",
    "T body t3",
    "T body t17",
    "T title t1",
    "B 0 3 0 T body t1 T body t5 T title t2",
    "B 1 1 0 T body t0 T body t3",
    "B 2 0 0 T body t1 T body t2",
    "B 1 1 1 T body t0 T title t1 T body t4",
    "B 0 2 0 X 2.5 T body t3 T body t6",
    "B 0 3 0 T body t8 T body t9 T body t10",
    "P body 0 2 t0 t1",
    "P body 0 3 t0 t1 t2",
    "P body 2 2 t0 t2",
    "P body 3 3 t3 t1 t0",
    "P body 0 2 t0 t21",
    "P body 1 2 t30 t0",
    "B 0 2 0 P body 0 2 t0 t1 T body t2",
    "B 1 1 0 T body t1 P body 2 2 t2 t4",
    "B 0 2 0 C 1.5 T body t2 T body t7",
    "B 0 1 0 D 0.3 2 T body t4 T title t4",
    "B 0 2 0 X 0.5 P body 1 2 t5 t6 T title t3",
  };

  static int pos;

  static Query parse(String[] tok) {
    String op = tok[pos++];
    switch (op) {
      case "T":
        return new TermQuery(new Term(tok[pos++], tok[pos++]));
      case "P": {
        String field = tok[pos++];
        int slop = Integer.parseInt(tok[pos++]);
        int n = Integer.parseInt(tok[pos++]);
        String[] terms = new String[n];
        for (int i = 0; i < n; i++) terms[i] = tok[pos++];
        return new PhraseQuery(slop, field, terms);
      }
      case "B": {
        int must = Integer.parseInt(tok[pos++]);
        int should = Integer.parseInt(tok[pos++]);
        int mustNot = Integer.parseInt(tok[pos++]);
        BooleanQuery.Builder b = new BooleanQuery.Builder();
        for (int i = 0; i < must; i++) b.add(parse(tok), BooleanClause.Occur.MUST);
        for (int i = 0; i < should; i++) b.add(parse(tok), BooleanClause.Occur.SHOULD);
        for (int i = 0; i < mustNot; i++) b.add(parse(tok), BooleanClause.Occur.MUST_NOT);
        return b.build();
      }
      case "X": {
        float boost = Float.parseFloat(tok[pos++]);
        return new BoostQuery(parse(tok), boost);
      }
      case "C": {
        float score = Float.parseFloat(tok[pos++]);
        return new BoostQuery(new ConstantScoreQuery(parse(tok)), score);
      }
      case "D": {
        float tie = Float.parseFloat(tok[pos++]);
        int n = Integer.parseInt(tok[pos++]);
        List<Query> qs = new ArrayList<>();
        for (int i = 0; i < n; i++) qs.add(parse(tok));
        return new DisjunctionMaxQuery(qs, tie);
      }
      default:
        throw new IllegalArgumentException(op);
    }
  }

  static String words(Random r, int n, int vocab) {
    StringBuilder sb = new StringBuilder();
    for (int i = 0; i < n; i++) {
      double x = r.nextDouble();
      int w = (int) (vocab * x * x * x);
      if (i > 0) sb.append(' ');
      sb.append('t').append(w);
    }
    return sb.toString();
  }

  static void deleteRecursive(Path p) throws IOException {
    if (!Files.exists(p)) return;
    try (Stream<Path> s = Files.walk(p)) {
      s.sorted(Comparator.reverseOrder()).forEach(q -> q.toFile().delete());
    }
  }

  public static void main(String[] args) throws IOException {
    Path root = Path.of(args[0]);
    Path out = root.resolve("similarity_search_index");
    deleteRecursive(out);
    Files.createDirectories(out);
    Random r = new Random(20260930L);

    try (Directory dir = FSDirectory.open(out)) {
      IndexWriterConfig cfg = new IndexWriterConfig(new StandardAnalyzer());
      cfg.setUseCompoundFile(false);
      cfg.setMergePolicy(NoMergePolicy.INSTANCE);
      int id = 0;
      try (IndexWriter w = new IndexWriter(dir, cfg)) {
        for (int size : new int[] {80, 50, 25}) {
          for (int i = 0; i < size; i++) {
            Document doc = new Document();
            doc.add(new StringField("id", Integer.toString(id++), Field.Store.NO));
            doc.add(new TextField("body", words(r, 1 + r.nextInt(r.nextInt(4) == 0 ? 60 : 12), 40), Field.Store.NO));
            if (r.nextInt(5) != 0) {
              doc.add(new TextField("title", words(r, 1 + r.nextInt(5), 12), Field.Store.NO));
            }
            w.addDocument(doc);
          }
          w.commit();
        }
        for (String del : new String[] {"3", "17", "40", "95", "101"}) {
          w.deleteDocuments(new Term("id", del));
        }
        w.commit();
      }

      StringBuilder sb = new StringBuilder();
      try (DirectoryReader reader = DirectoryReader.open(dir)) {
        if (reader.leaves().size() != 3) {
          throw new AssertionError("expected three segments, got " + reader.leaves().size());
        }
        for (Map.Entry<String, Similarity> e : sims().entrySet()) {
          IndexSearcher searcher = new IndexSearcher(reader);
          searcher.setQueryCache(null);
          searcher.setSimilarity(e.getValue());
          for (String q : QUERIES) {
            pos = 0;
            String[] tok = q.split(" ");
            Query query = parse(tok);
            if (pos != tok.length) throw new AssertionError("trailing tokens in " + q);
            TopDocs td = searcher.search(query, TOP);
            sb.append(e.getKey()).append('\t').append(q).append('\t');
            for (int i = 0; i < td.scoreDocs.length; i++) {
              ScoreDoc sd = td.scoreDocs[i];
              if (i > 0) sb.append(',');
              sb.append(sd.doc).append(':').append(Integer.toHexString(Float.floatToRawIntBits(sd.score)));
            }
            sb.append('\n');
          }
        }
      }
      Files.writeString(out.resolve("searches.tsv"), sb);
    }

    writeNorms(root, r);
  }

  /** Norms written under {@code IndexWriterConfig.setSimilarity}, per document and field. */
  static void writeNorms(Path root, Random r) throws IOException {
    Path out = root.resolve("similarity_norms");
    deleteRecursive(out);
    Files.createDirectories(out);
    Path tmp = Files.createTempDirectory("similarity-norms");
    FieldType tagsType = new FieldType();
    tagsType.setIndexOptions(IndexOptions.DOCS);
    tagsType.setTokenized(true);
    tagsType.freeze();
    Similarity wide = new WideNormSimilarity();
    Similarity classic = new ClassicSimilarity(false);
    Similarity bm25 = new BM25Similarity();
    Similarity perField = new PerFieldSimilarityWrapper() {
      @Override
      public Similarity get(String name) {
        switch (name) {
          case "body": return classic;
          case "title": return wide;
          default: return bm25;
        }
      }
    };
    StringBuilder docs = new StringBuilder();
    StringBuilder norms = new StringBuilder();
    int n = 60;
    try (Directory dir = FSDirectory.open(tmp)) {
      IndexWriterConfig cfg = new IndexWriterConfig(new StandardAnalyzer());
      cfg.setUseCompoundFile(false);
      cfg.setSimilarity(perField);
      try (IndexWriter w = new IndexWriter(dir, cfg)) {
        for (int i = 0; i < n; i++) {
          String body = words(r, 1 + r.nextInt(r.nextInt(4) == 0 ? 300 : 20), 30);
          // Absent on some documents (sparse norms), present without tokens on one.
          String title = i == 7 ? "..." : (r.nextInt(4) == 0 ? null : words(r, 1 + r.nextInt(r.nextInt(3) == 0 ? 40 : 6), 8));
          String tags = words(r, 1 + r.nextInt(12), 4);
          Document doc = new Document();
          doc.add(new TextField("body", body, Field.Store.NO));
          if (title != null) doc.add(new TextField("title", title, Field.Store.NO));
          doc.add(new Field("tags", tags, tagsType));
          w.addDocument(doc);
          docs.append(body).append('\t').append(title == null ? "-" : title).append('\t').append(tags).append('\n');
        }
      }
      try (DirectoryReader reader = DirectoryReader.open(dir)) {
        if (reader.leaves().size() != 1) throw new AssertionError("one segment expected");
        LeafReader leaf = reader.leaves().get(0).reader();
        for (String field : new String[] {"body", "title", "tags"}) {
          NumericDocValues nv = leaf.getNormValues(field);
          for (int d = 0; d < n; d++) {
            norms.append(field).append('\t').append(d).append('\t');
            if (nv != null && nv.advanceExact(d)) norms.append(nv.longValue());
            else norms.append('-');
            norms.append('\n');
          }
        }
      }
    }
    deleteRecursive(tmp);
    Files.writeString(out.resolve("norms_docs.tsv"), docs);
    Files.writeString(out.resolve("norms.tsv"), norms);
  }
}
