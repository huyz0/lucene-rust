import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import org.apache.lucene.index.FieldInvertState;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.search.CollectionStatistics;
import org.apache.lucene.search.TermStatistics;
import org.apache.lucene.search.similarities.*;
import org.apache.lucene.util.BytesRef;

/**
 * Every similarity in Lucene 10.5.0's {@code search/similarities}, scored on random statistics
 * within {@code CollectionStatistics}' and {@code TermStatistics}' own constraints, for
 * crates/lucene-search/tests/similarities_fixtures.rs.
 *
 * <p>{@code scores.tsv}: {@code sim  maxDoc,docCount,sumTTF,sumDF  df:ttf[;df:ttf...]  boost  freq
 * norm  scoreBits}, the score as {@code Float.floatToRawIntBits} in hex. {@code norms.tsv}:
 * {@code sim  docsOnly  length  numOverlap  uniqueTermCount  norm}.
 */
public class GenSimilarities {
  static Map<String, Similarity> sims() {
    Map<String, Similarity> m = new LinkedHashMap<>();
    m.put("bm25", new BM25Similarity());
    m.put("bm25_k2_b03", new BM25Similarity(2.0f, 0.3f));
    m.put("bm25_k0_b1", new BM25Similarity(0f, 1f));
    m.put("bm25_nodiscount", new BM25Similarity(false));
    m.put("classic", new ClassicSimilarity());
    m.put("classic_nodiscount", new ClassicSimilarity(false));
    m.put("boolean", new BooleanSimilarity());
    m.put("rawtf", new RawTFSimilarity());
    BasicModel[] bms = {new BasicModelG(), new BasicModelIF(), new BasicModelIn(), new BasicModelIne()};
    String[] bmNames = {"G", "IF", "In", "Ine"};
    AfterEffect[] aes = {new AfterEffectB(), new AfterEffectL()};
    String[] aeNames = {"B", "L"};
    Normalization[] ns = {
      new Normalization.NoNormalization(), new NormalizationH1(), new NormalizationH1(2.5f),
      new NormalizationH2(), new NormalizationH2(0.3f), new NormalizationH3(), new NormalizationH3(100f),
      new NormalizationZ(), new NormalizationZ(0.1f)
    };
    String[] nNames = {"none", "H1", "H1_2.5", "H2", "H2_0.3", "H3", "H3_100", "Z", "Z_0.1"};
    for (int b = 0; b < bms.length; b++)
      for (int a = 0; a < aes.length; a++)
        for (int n = 0; n < ns.length; n++)
          m.put("dfr_" + bmNames[b] + "_" + aeNames[a] + "_" + nNames[n], new DFRSimilarity(bms[b], aes[a], ns[n]));
    Distribution[] ds = {new DistributionLL(), new DistributionSPL()};
    String[] dNames = {"LL", "SPL"};
    Lambda[] ls = {new LambdaDF(), new LambdaTTF()};
    String[] lNames = {"DF", "TTF"};
    for (int d = 0; d < ds.length; d++)
      for (int l = 0; l < ls.length; l++)
        for (int n = 0; n < ns.length; n++)
          m.put("ib_" + dNames[d] + "_" + lNames[l] + "_" + nNames[n], new IBSimilarity(ds[d], ls[l], ns[n]));
    m.put("dfi_standardized", new DFISimilarity(new IndependenceStandardized()));
    m.put("dfi_saturated", new DFISimilarity(new IndependenceSaturated()));
    m.put("dfi_chisquared", new DFISimilarity(new IndependenceChiSquared()));
    m.put("lmdirichlet", new LMDirichletSimilarity());
    m.put("lmdirichlet_100", new LMDirichletSimilarity(100f));
    m.put("lmdirichlet_indri_500",
        new LMDirichletSimilarity(new IndriDirichletSimilarity.IndriCollectionModel(), 500f));
    m.put("lmjm_0.1", new LMJelinekMercerSimilarity(0.1f));
    m.put("lmjm_0.7", new LMJelinekMercerSimilarity(0.7f));
    m.put("lmjm_1", new LMJelinekMercerSimilarity(1f));
    m.put("indri", new IndriDirichletSimilarity());
    m.put("indri_100", new IndriDirichletSimilarity(100f));
    m.put("ax_f1exp", new AxiomaticF1EXP());
    m.put("ax_f1exp_0.5_0.2", new AxiomaticF1EXP(0.5f, 0.2f));
    m.put("ax_f1log", new AxiomaticF1LOG(0.1f));
    m.put("ax_f2exp", new AxiomaticF2EXP(0.5f, 0.2f));
    m.put("ax_f2log", new AxiomaticF2LOG());
    m.put("ax_f3exp", new AxiomaticF3EXP(0.25f, 3, 0.35f));
    m.put("ax_f3log", new AxiomaticF3LOG(0.4f, 2));
    m.put("multi", new MultiSimilarity(new Similarity[] {
      new BM25Similarity(), new ClassicSimilarity(),
      new DFRSimilarity(new BasicModelG(), new AfterEffectB(), new NormalizationH2())}));
    return m;
  }

  static long between(Random r, long lo, long hi) {
    if (hi <= lo) return lo;
    return lo + (long) (r.nextDouble() * (hi - lo + 1));
  }

  public static void main(String[] args) throws IOException {
    Path out = Path.of(args[0]).resolve("similarities");
    Files.createDirectories(out);
    Map<String, Similarity> sims = sims();
    Random r = new Random(20260930L);
    float[] boosts = {1f, 1f, 0.5f, 2f, 3.7f};
    StringBuilder scores = new StringBuilder();
    for (Map.Entry<String, Similarity> e : sims.entrySet()) {
      for (int c = 0; c < 80; c++) {
        long maxDoc = r.nextInt(4) == 0 ? between(r, 1, 20) : between(r, 1, 10_000_000);
        long docCount = between(r, 1, maxDoc);
        long sumDf = between(r, docCount, docCount * 30);
        long sumTtf = between(r, sumDf, sumDf * 4);
        CollectionStatistics cs =
            new CollectionStatistics("f", maxDoc, docCount, sumTtf, sumDf);
        int nTerms = r.nextInt(5) == 0 ? 2 + r.nextInt(2) : 1;
        TermStatistics[] ts = new TermStatistics[nTerms];
        StringBuilder tsText = new StringBuilder();
        for (int t = 0; t < nTerms; t++) {
          long df = between(r, 1, docCount);
          long ttf = between(r, df, Math.min(sumTtf, df * 20));
          ts[t] = new TermStatistics(new BytesRef("t" + t), df, ttf);
          if (t > 0) tsText.append(';');
          tsText.append(df).append(':').append(ttf);
        }
        float boost = boosts[r.nextInt(boosts.length)];
        Similarity.SimScorer scorer = e.getValue().scorer(boost, cs, ts);
        for (int k = 0; k < 3; k++) {
          float freq = r.nextInt(6) == 0 ? (1 + r.nextInt(40)) / 7f : 1 + r.nextInt(r.nextBoolean() ? 5 : 300);
          long norm = r.nextInt(12) == 0 ? 1L : (long) (byte) r.nextInt(256);
          float score = scorer.score(freq, norm);
          scores.append(e.getKey()).append('\t')
              .append(maxDoc).append(',').append(docCount).append(',').append(sumTtf).append(',').append(sumDf).append('\t')
              .append(tsText).append('\t')
              .append(Integer.toHexString(Float.floatToRawIntBits(boost))).append('\t')
              .append(Integer.toHexString(Float.floatToRawIntBits(freq))).append('\t')
              .append(norm).append('\t')
              .append(Integer.toHexString(Float.floatToRawIntBits(score))).append('\n');
        }
      }
    }
    Files.writeString(out.resolve("scores.tsv"), scores);

    StringBuilder norms = new StringBuilder();
    List<String> normSims = new ArrayList<>(List.of("bm25", "bm25_nodiscount", "classic", "classic_nodiscount", "multi"));
    for (String name : normSims) {
      Similarity s = sims.get(name);
      for (int c = 0; c < 300; c++) {
        boolean docsOnly = r.nextInt(4) == 0;
        int length = r.nextInt(3) == 0 ? r.nextInt(50) : r.nextInt(Integer.MAX_VALUE / 2);
        int overlap = length == 0 ? 0 : r.nextInt(length + 1);
        int unique = length == 0 ? 0 : 1 + r.nextInt(length);
        FieldInvertState st = new FieldInvertState(
            10, "f", docsOnly ? IndexOptions.DOCS : IndexOptions.DOCS_AND_FREQS_AND_POSITIONS,
            length, length, overlap, 0, 1, unique);
        norms.append(name).append('\t').append(docsOnly).append('\t').append(length).append('\t')
            .append(overlap).append('\t').append(unique).append('\t').append(s.computeNorm(st)).append('\n');
      }
    }
    Files.writeString(out.resolve("norms.tsv"), norms);
  }
}
