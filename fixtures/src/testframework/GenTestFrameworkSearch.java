import com.carrotsearch.randomizedtesting.RandomizedRunner;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.function.Function;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.SortedDocValuesField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.BooleanClause.Occur;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.BoostQuery;
import org.apache.lucene.search.CollectionStatistics;
import org.apache.lucene.search.ConstantScoreQuery;
import org.apache.lucene.search.DisjunctionMaxQuery;
import org.apache.lucene.search.Explanation;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.MatchAllDocsQuery;
import org.apache.lucene.search.MultiPhraseQuery;
import org.apache.lucene.search.PhraseQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.SynonymQuery;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.search.TermStatistics;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.similarities.AfterEffect;
import org.apache.lucene.search.similarities.AfterEffectB;
import org.apache.lucene.search.similarities.AfterEffectL;
import org.apache.lucene.search.similarities.AxiomaticF1EXP;
import org.apache.lucene.search.similarities.AxiomaticF1LOG;
import org.apache.lucene.search.similarities.AxiomaticF2EXP;
import org.apache.lucene.search.similarities.AxiomaticF2LOG;
import org.apache.lucene.search.similarities.AxiomaticF3EXP;
import org.apache.lucene.search.similarities.AxiomaticF3LOG;
import org.apache.lucene.search.similarities.BM25Similarity;
import org.apache.lucene.search.similarities.BasicModel;
import org.apache.lucene.search.similarities.BasicModelG;
import org.apache.lucene.search.similarities.BasicModelIF;
import org.apache.lucene.search.similarities.BasicModelIn;
import org.apache.lucene.search.similarities.BasicModelIne;
import org.apache.lucene.search.similarities.BooleanSimilarity;
import org.apache.lucene.search.similarities.ClassicSimilarity;
import org.apache.lucene.search.similarities.DFISimilarity;
import org.apache.lucene.search.similarities.DFRSimilarity;
import org.apache.lucene.search.similarities.Distribution;
import org.apache.lucene.search.similarities.DistributionLL;
import org.apache.lucene.search.similarities.DistributionSPL;
import org.apache.lucene.search.similarities.IBSimilarity;
import org.apache.lucene.search.similarities.IndependenceChiSquared;
import org.apache.lucene.search.similarities.IndependenceSaturated;
import org.apache.lucene.search.similarities.IndependenceStandardized;
import org.apache.lucene.search.similarities.IndriDirichletSimilarity;
import org.apache.lucene.search.similarities.LMDirichletSimilarity;
import org.apache.lucene.search.similarities.LMJelinekMercerSimilarity;
import org.apache.lucene.search.similarities.Lambda;
import org.apache.lucene.search.similarities.LambdaDF;
import org.apache.lucene.search.similarities.LambdaTTF;
import org.apache.lucene.search.similarities.Normalization;
import org.apache.lucene.search.similarities.NormalizationH1;
import org.apache.lucene.search.similarities.NormalizationH2;
import org.apache.lucene.search.similarities.NormalizationH3;
import org.apache.lucene.search.similarities.NormalizationZ;
import org.apache.lucene.search.similarities.RawTFSimilarity;
import org.apache.lucene.search.similarities.Similarity;
import org.apache.lucene.search.similarities.Similarity.BulkSimScorer;
import org.apache.lucene.search.similarities.Similarity.SimScorer;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.tests.analysis.MockAnalyzer;
import org.apache.lucene.tests.analysis.MockTokenizer;
import org.apache.lucene.tests.search.CheckHits;
import org.apache.lucene.tests.search.similarities.RandomSimilarity;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.tests.util.TestUtil;
import org.apache.lucene.util.BytesRef;
import org.apache.lucene.util.SmallFloat;
import org.apache.lucene.util.automaton.Automata;
import org.apache.lucene.util.automaton.CharacterRunAutomaton;

import org.junit.Test;
import org.junit.runner.JUnitCore;
import org.junit.runner.Result;
import org.junit.runner.RunWith;
import org.junit.runner.notification.Failure;
/**
 * Differential fixtures for lucene-test-framework's search bases (M14.h h7): what Lucene 10.5.0
 * scores, explains and draws from a seeded {@code java.util.Random}.
 *
 * <p>Writes {@code search.txt}, five sections:
 *
 * <ul>
 *   <li>{@code sim}: each {@code BaseSimilarityTestCase} subclass's {@code testRandomScoring} and
 *       {@code testBulkScore}, replayed over {@code new Random(1000 * i + 2 * rep)} (scoring) and
 *       {@code + 1} (bulk), {@code random()} and the {@code random} argument being one stream as
 *       they are inside a test. Each run is one line: the case count and an FNV-1a hash of every
 *       statistic, frequency, boost and score bit pattern it compared; the first cases also
 *       verbatim, so a divergence can be located.
 *   <li>{@code shuffle}: {@code Collections.shuffle} of 0..9 under {@code new Random(42)}.
 *   <li>{@code rsim}: the {@code allSims} index {@code RandomSimilarity.get} picks per field.
 *   <li>{@code exp}: {@code BaseExplanationTestCase}'s four documents searched under BM25 and
 *       {@code ClassicSimilarity}: top docs, score bits and each hit's explanation.
 *   <li>{@code eq}: {@code SearchEquivalenceTestBase}'s corpus draws (in the order the Rust base
 *       draws them: no {@code RandomIndexWriter}) and BM25 top docs over it.
 * </ul>
 *
 * <p>Run with no {@code tests.*} system properties: {@code atLeast}/{@code rarely} see multiplier
 * 1, not nightly. Usage: java GenTestFrameworkSearch <outdir> [ignored]
 */
public class GenTestFrameworkSearch {
  static final StringBuilder OUT = new StringBuilder();
  static final int REPS = 2;
  static final int VERBATIM = 5;

  static void line(Object... parts) {
    for (int i = 0; i < parts.length; i++) {
      if (i > 0) OUT.append(' ');
      OUT.append(parts[i]);
    }
    OUT.append('\n');
  }

  static String bits(float f) {
    return String.format("%08x", Float.floatToRawIntBits(f));
  }

  // ---------------------------------------------------------------- hashing

  static long hash;

  static void feed(long v) {
    hash ^= v;
    hash *= 0x100000001b3L;
  }

  static void feed(float f) {
    feed((long) Float.floatToRawIntBits(f));
  }

  // ------------------------------------------------------- similarity tests

  /** One subclass's {@code getSimilarity(Random)}. */
  record SimTest(String name, Function<Random, Similarity> get) {}

  static Normalization randomNormalization(Random random) {
    final float c;
    switch (random.nextInt(4)) {
      case 0 -> c = 0;
      case 1 -> c = Float.MIN_VALUE;
      case 2 -> c = Integer.MAX_VALUE;
      default -> c = Integer.MAX_VALUE * random.nextFloat();
    }
    final float z;
    switch (random.nextInt(3)) {
      case 0 -> z = Float.MIN_VALUE;
      case 1 -> z = Math.nextDown(0.5f);
      default -> {
        float zcand = random.nextFloat() / 2;
        z = zcand == 0f ? Math.nextUp(zcand) : zcand;
      }
    }
    final float mu;
    switch (random.nextInt(4)) {
      case 0 -> mu = 0;
      case 1 -> mu = Float.MIN_VALUE;
      case 2 -> mu = Integer.MAX_VALUE;
      default -> mu = Integer.MAX_VALUE * random.nextFloat();
    }
    return switch (random.nextInt(5)) {
      case 0 -> new Normalization.NoNormalization();
      case 1 -> new NormalizationH1(c);
      case 2 -> new NormalizationH2(c);
      case 3 -> new NormalizationH3(mu);
      default -> new NormalizationZ(z);
    };
  }

  static Similarity basicModel(Random random, BasicModel model) {
    AfterEffect afterEffect = random.nextInt(2) == 0 ? new AfterEffectL() : new AfterEffectB();
    return new DFRSimilarity(model, afterEffect, randomNormalization(random));
  }

  static Similarity distribution(Random random, Distribution distribution) {
    Lambda lambda = random.nextBoolean() ? new LambdaDF() : new LambdaTTF();
    return new IBSimilarity(distribution, lambda, randomNormalization(random));
  }

  interface Axiomatic {
    Similarity make(float s, int queryLen, float k);
  }

  static Similarity axiomatic(Random random, Axiomatic model) {
    final float s;
    switch (random.nextInt(4)) {
      case 0 -> s = 0;
      case 1 -> s = Float.MIN_VALUE;
      case 2 -> s = 1;
      default -> s = random.nextFloat();
    }
    final int queryLen;
    switch (random.nextInt(4)) {
      case 0 -> queryLen = 0;
      case 1 -> queryLen = 1;
      case 2 -> queryLen = Integer.MAX_VALUE;
      default -> queryLen = random.nextInt(Integer.MAX_VALUE);
    }
    final float k;
    switch (random.nextInt(4)) {
      case 0 -> k = 0;
      case 1 -> k = Float.MIN_VALUE;
      case 2 -> k = 1;
      default -> k = random.nextFloat();
    }
    return model.make(s, queryLen, k);
  }

  static float unbounded(Random random) {
    return switch (random.nextInt(4)) {
      case 0 -> 0;
      case 1 -> Float.MIN_VALUE;
      case 2 -> Integer.MAX_VALUE;
      default -> Integer.MAX_VALUE * random.nextFloat();
    };
  }

  static final List<SimTest> SIM_TESTS =
      List.of(
          new SimTest(
              "BM25",
              r -> {
                float k1 = unbounded(r);
                final float b;
                switch (r.nextInt(4)) {
                  case 0 -> b = 0;
                  case 1 -> b = Float.MIN_VALUE;
                  case 2 -> b = 1;
                  default -> b = r.nextFloat();
                }
                return new BM25Similarity(k1, b);
              }),
          new SimTest("Boolean", r -> new BooleanSimilarity()),
          new SimTest("Classic", r -> new ClassicSimilarity()),
          new SimTest("IndependenceChiSquared", r -> new DFISimilarity(new IndependenceChiSquared())),
          new SimTest("IndependenceSaturated", r -> new DFISimilarity(new IndependenceSaturated())),
          new SimTest(
              "IndependenceStandardized", r -> new DFISimilarity(new IndependenceStandardized())),
          new SimTest("IndriDirichlet", r -> new IndriDirichletSimilarity(unbounded(r))),
          new SimTest("LMDirichlet", r -> new LMDirichletSimilarity(unbounded(r))),
          new SimTest(
              "LMJelinekMercer",
              r -> {
                final float lambda;
                switch (r.nextInt(3)) {
                  case 0 -> lambda = Float.MIN_VALUE;
                  case 1 -> lambda = 1;
                  default -> lambda = r.nextFloat();
                }
                return new LMJelinekMercerSimilarity(lambda);
              }),
          new SimTest("RawTF", r -> new RawTFSimilarity()),
          new SimTest("AxiomaticF1EXP", r -> axiomatic(r, (s, q, k) -> new AxiomaticF1EXP(s, k))),
          new SimTest("AxiomaticF1LOG", r -> axiomatic(r, (s, q, k) -> new AxiomaticF1LOG(s))),
          new SimTest("AxiomaticF2EXP", r -> axiomatic(r, (s, q, k) -> new AxiomaticF2EXP(s, k))),
          new SimTest("AxiomaticF2LOG", r -> axiomatic(r, (s, q, k) -> new AxiomaticF2LOG(s))),
          new SimTest(
              "AxiomaticF3EXP", r -> axiomatic(r, (s, q, k) -> new AxiomaticF3EXP(s, q))),
          new SimTest("AxiomaticF3LOG", r -> axiomatic(r, (s, q, k) -> new AxiomaticF3LOG(s, q))),
          new SimTest("BasicModelG", r -> basicModel(r, new BasicModelG())),
          new SimTest("BasicModelIF", r -> basicModel(r, new BasicModelIF())),
          new SimTest("BasicModelIn", r -> basicModel(r, new BasicModelIn())),
          new SimTest("BasicModelIne", r -> basicModel(r, new BasicModelIne())),
          new SimTest("DistributionLL", r -> distribution(r, new DistributionLL())),
          new SimTest("DistributionSPL", r -> distribution(r, new DistributionSPL())));

  static final long MAXDOC_FORTESTING = 1L << 48;
  static final long MAXTOKENS_FORTESTING = 1L << 49;

  /** {@code BaseSimilarityTestCase.newCorpus}. */
  static CollectionStatistics newCorpus(Random random, int norm) {
    final int lowerBound = norm == 0 ? 1 : SmallFloat.byte4ToInt((byte) norm);
    final long maxDoc =
        switch (random.nextInt(6)) {
          case 0 -> 1;
          case 1 -> 2;
          case 2 -> TestUtil.nextLong(random, 3, 16);
          case 3 -> TestUtil.nextLong(random, 16, 100000);
          case 4 -> TestUtil.nextLong(random, 100000, MAXDOC_FORTESTING);
          default -> MAXDOC_FORTESTING;
        };
    final long docCount =
        switch (random.nextInt(3)) {
          case 0 -> 1;
          case 1 -> TestUtil.nextLong(random, 1, maxDoc);
          default -> maxDoc;
        };
    long upperBound;
    try {
      upperBound = Math.min(MAXTOKENS_FORTESTING, Math.multiplyExact(docCount, Integer.MAX_VALUE));
    } catch (ArithmeticException overflow) {
      upperBound = MAXTOKENS_FORTESTING;
    }
    final long sumDocFreq =
        switch (random.nextInt(3)) {
          case 0 -> docCount;
          case 1 -> upperBound + 1 - lowerBound;
          default -> TestUtil.nextLong(random, docCount, upperBound + 1 - lowerBound);
        };
    final long sumTotalTermFreq =
        switch (random.nextInt(4)) {
          case 0 -> sumDocFreq;
          case 1 -> sumDocFreq - 1 + lowerBound;
          case 2 -> upperBound;
          default -> TestUtil.nextLong(random, sumDocFreq - 1 + lowerBound, upperBound);
        };
    return new CollectionStatistics("field", maxDoc, docCount, sumTotalTermFreq, sumDocFreq);
  }

  static final BytesRef TERM = new BytesRef("term");

  /** {@code BaseSimilarityTestCase.newTerm}. */
  static TermStatistics newTerm(Random random, CollectionStatistics corpus) {
    final long docFreq =
        switch (random.nextInt(3)) {
          case 0 -> 1;
          case 1 -> corpus.docCount();
          default -> TestUtil.nextLong(random, 1, corpus.docCount());
        };
    long upperBound;
    try {
      upperBound =
          Math.min(corpus.sumTotalTermFreq(), Math.multiplyExact(docFreq, Integer.MAX_VALUE));
    } catch (ArithmeticException overflow) {
      upperBound = corpus.sumTotalTermFreq();
    }
    final long totalTermFreq;
    if (corpus.sumTotalTermFreq() == corpus.sumDocFreq()) {
      totalTermFreq = docFreq;
    } else {
      totalTermFreq =
          switch (random.nextInt(3)) {
            case 0 -> docFreq;
            case 1 -> upperBound;
            default -> TestUtil.nextLong(random, docFreq, upperBound);
          };
    }
    return new TermStatistics(TERM, docFreq, totalTermFreq);
  }

  static int cases;

  /** {@code testRandomScoring}. */
  static void randomScoring(String name, int seed, Function<Random, Similarity> get)
      throws Exception {
    Random random = new Random(seed);
    hash = 0xcbf29ce484222325L;
    cases = 0;
    final int iterations = LuceneTestCase.atLeast(random, 1);
    for (int i = 0; i < iterations; i++) {
      Similarity similarity = get.apply(random);
      for (int j = 0; j < 3; j++) {
        for (int k = 1; k < 256; k++) {
          CollectionStatistics corpus = newCorpus(random, k);
          for (int l = 0; l < 10; l++) {
            TermStatistics term = newTerm(random, corpus);
            final float freq;
            if (term.totalTermFreq() == term.docFreq()) {
              freq = 1;
            } else if (term.docFreq() == 1) {
              freq = Math.toIntExact(term.totalTermFreq());
            } else {
              int upperBound =
                  Math.toIntExact(
                      Math.min(term.totalTermFreq() - term.docFreq() + 1, Integer.MAX_VALUE));
              if (random.nextBoolean()) {
                freq =
                    switch (random.nextInt(3)) {
                      case 0 -> 1;
                      case 1 -> upperBound;
                      default -> TestUtil.nextInt(random, 1, upperBound);
                    };
              } else {
                float freqCandidate =
                    switch (random.nextInt(2)) {
                      case 0 -> Float.MIN_VALUE;
                      default -> upperBound * random.nextFloat();
                    };
                if (freqCandidate <= Float.MIN_VALUE) {
                  freqCandidate = Math.nextUp(Float.MIN_VALUE);
                }
                freq = freqCandidate;
              }
            }
            final float boost =
                switch (random.nextInt(5)) {
                  case 0 -> 0F;
                  case 1 -> Float.MIN_VALUE;
                  case 2 -> 1F;
                  case 3 -> Integer.MAX_VALUE;
                  default -> random.nextFloat() * Integer.MAX_VALUE;
                };
            doTestScoring(random, name, seed, similarity, corpus, term, boost, freq, k);
          }
        }
      }
    }
    line("sim", name, "scoring", seed, cases, String.format("%016x", hash));
  }

  static void checkExplanation(Random random, SimScorer scorer, float freq, long norm, float score) {
    Explanation explanation =
        scorer.explain(Explanation.match(freq, "freq, occurrences of term within document"), norm);
    if (score != explanation.getValue().doubleValue()) {
      throw new AssertionError("expected: " + score + ", got: " + explanation);
    }
    if (LuceneTestCase.rarely(random)) {
      CheckHits.verifyExplanation("<test query>", 0, score, true, explanation);
    }
  }

  /** {@code doTestScoring}, recording what it compared. */
  static void doTestScoring(
      Random random,
      String name,
      int seed,
      Similarity similarity,
      CollectionStatistics corpus,
      TermStatistics term,
      float boost,
      float freq,
      int norm)
      throws Exception {
    boolean nonNegative = !(similarity instanceof IndriDirichletSimilarity);
    SimScorer scorer = similarity.scorer(boost, corpus, term);
    float maxScore = scorer.score(Float.MAX_VALUE, 1);
    if (Float.isNaN(maxScore)) throw new AssertionError("maxScore is NaN");
    float score = scorer.score(freq, norm);
    if (!Float.isFinite(score) || (nonNegative && score < 0) || score > maxScore) {
      throw new AssertionError(name + " score " + score);
    }
    checkExplanation(random, scorer, freq, norm, score);

    final float prevFreq;
    if (random.nextBoolean() && freq == (int) freq && freq > 1 && term.docFreq() > 1) {
      prevFreq = freq - 1;
    } else {
      prevFreq = Math.nextDown(freq);
    }
    float prevScore = scorer.score(prevFreq, norm);
    if (!Float.isFinite(prevScore) || (nonNegative && prevScore < 0)) {
      throw new AssertionError(name + " prevScore " + prevScore);
    }
    checkExplanation(random, scorer, prevFreq, norm, prevScore);
    if (prevScore > score) throw new AssertionError(name + " prevScore > score");

    Float prevNormScore = null;
    if (norm > 1) {
      prevNormScore = scorer.score(freq, norm - 1);
      if (!Float.isFinite(prevNormScore) || (nonNegative && prevNormScore < 0)) {
        throw new AssertionError(name + " prevNormScore " + prevNormScore);
      }
      checkExplanation(random, scorer, freq, norm - 1, prevNormScore);
      if (prevNormScore < score) throw new AssertionError(name + " prevNormScore < score");
    }

    Float prevTermScore = null;
    if (term.docFreq() > 1 && freq < term.totalTermFreq()) {
      // as Java's test does: the rarer term's statistics are built, `term` is scored again
      SimScorer prevTermScorer = similarity.scorer(boost, corpus, term);
      prevTermScore = prevTermScorer.score(freq, norm);
      if (!Float.isFinite(prevTermScore) || (nonNegative && prevTermScore < 0)) {
        throw new AssertionError(name + " prevTermScore " + prevTermScore);
      }
      checkExplanation(random, prevTermScorer, freq, norm, prevTermScore);
      if (prevTermScore < score) throw new AssertionError(name + " prevTermScore < score");
    }

    feed(corpus.maxDoc());
    feed(corpus.docCount());
    feed(corpus.sumTotalTermFreq());
    feed(corpus.sumDocFreq());
    feed(term.docFreq());
    feed(term.totalTermFreq());
    feed(boost);
    feed(freq);
    feed(norm);
    feed(maxScore);
    feed(score);
    feed(prevFreq);
    feed(prevScore);
    if (prevNormScore == null) {
      feed(0);
    } else {
      feed(1);
      feed(prevNormScore);
    }
    if (prevTermScore == null) {
      feed(0);
    } else {
      feed(1);
      feed(prevTermScore);
    }
    if (cases < VERBATIM) {
      line(
          "case", name, seed, cases, corpus.maxDoc(), corpus.docCount(),
          corpus.sumTotalTermFreq(), corpus.sumDocFreq(), term.docFreq(), term.totalTermFreq(),
          bits(boost), bits(freq), norm, bits(maxScore), bits(score), bits(prevFreq),
          bits(prevScore), prevNormScore == null ? "-" : bits(prevNormScore),
          prevTermScore == null ? "-" : bits(prevTermScore));
    }
    cases++;
  }

  /** {@code testBulkScore}. */
  static void bulkScore(String name, int seed, Function<Random, Similarity> get) {
    Random random = new Random(seed);
    hash = 0xcbf29ce484222325L;
    Similarity similarity = get.apply(random);
    CollectionStatistics corpus = newCorpus(random, 1);
    TermStatistics term = newTerm(random, corpus);
    SimScorer scorer = similarity.scorer(random.nextFloat(5f), corpus, term);
    BulkSimScorer bulkScorer = scorer.asBulkSimScorer();
    int freqUpperBound =
        Math.toIntExact(Math.min(term.totalTermFreq() - term.docFreq() + 1, Integer.MAX_VALUE));
    int iters = LuceneTestCase.atLeast(random, 3);
    for (int iter = 0; iter < iters; ++iter) {
      int size = TestUtil.nextInt(random, 0, 200);
      float[] freqs = new float[size];
      long[] norms = new long[size];
      float[] scores = new float[size];
      for (int i = 0; i < size; ++i) {
        freqs[i] = TestUtil.nextInt(random, 1, freqUpperBound);
        norms[i] = TestUtil.nextLong(random, 1, 255);
      }
      bulkScorer.score(size, freqs, norms, scores);
      feed(size);
      for (int i = 0; i < size; ++i) {
        if (scores[i] != scorer.score(freqs[i], norms[i])) throw new AssertionError(name);
        feed(scores[i]);
      }
    }
    line("sim", name, "bulk", seed, iters, String.format("%016x", hash));
  }

  // ------------------------------------------------------- RandomSimilarity

  static final String[] RSIM_FIELDS = {
    "field", "body", "title", "id", "KEY", "alt", "f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7"
  };

  /** `toString()`, but `BooleanSimilarity` has none of its own (identity hash). */
  static String simName(Similarity s) {
    return s instanceof BooleanSimilarity ? "BooleanSimilarity" : s.toString();
  }

  static void randomSimilarity() throws Exception {
    // `allSims`, rebuilt in RandomSimilarity's order, to name what `get` picks
    List<String> all = new ArrayList<>();
    all.add(new ClassicSimilarity().toString());
    all.add(new BM25Similarity().toString());
    all.add(new AxiomaticF1EXP().toString());
    all.add(new AxiomaticF1LOG().toString());
    all.add(new AxiomaticF2EXP().toString());
    all.add(new AxiomaticF2LOG().toString());
    all.add(simName(new BooleanSimilarity()));
    Normalization[] norms = {
      new NormalizationH1(), new NormalizationH2(), new NormalizationH3(), new NormalizationZ()
    };
    for (BasicModel m :
        new BasicModel[] {new BasicModelG(), new BasicModelIF(), new BasicModelIn(), new BasicModelIne()}) {
      for (AfterEffect a : new AfterEffect[] {new AfterEffectB(), new AfterEffectL()}) {
        for (Normalization n : norms) all.add(new DFRSimilarity(m, a, n).toString());
      }
    }
    for (Distribution d : new Distribution[] {new DistributionLL(), new DistributionSPL()}) {
      for (Lambda l : new Lambda[] {new LambdaDF(), new LambdaTTF()}) {
        for (Normalization n : norms) all.add(new IBSimilarity(d, l, n).toString());
      }
    }
    all.add(new LMDirichletSimilarity().toString());
    all.add(new LMJelinekMercerSimilarity(0.1f).toString());
    all.add(new LMJelinekMercerSimilarity(0.7f).toString());
    all.add(new DFISimilarity(new IndependenceStandardized()).toString());
    all.add(new DFISimilarity(new IndependenceSaturated()).toString());
    all.add(new DFISimilarity(new IndependenceChiSquared()).toString());
    if (all.stream().distinct().count() != all.size()) throw new AssertionError("ambiguous names");
    for (int seed = 0; seed < 20; seed++) {
      RandomSimilarity sim = new RandomSimilarity(new Random(seed));
      StringBuilder picks = new StringBuilder();
      for (String f : RSIM_FIELDS) {
        int at = all.indexOf(simName(sim.get(f)));
        if (at < 0) throw new AssertionError(simName(sim.get(f)));
        if (picks.length() > 0) picks.append(',');
        picks.append(at);
      }
      line("rsim", seed, picks);
    }
  }

  // ----------------------------------------------------------- explanations

  static final String KEY = "KEY";
  static final String FIELD = "field";
  static final String ALTFIELD = "alt";
  static final String[] DOC_FIELDS = {
    "w1 w2 w3 w4 w5", "w1 w3 w2 w3 zz", "w1 xx w2 yy w3", "w1 w3 xx w2 yy w3 zz"
  };

  static Query t(String s) {
    return new TermQuery(new Term(FIELD, s));
  }

  static Query a(String s) {
    return new TermQuery(new Term(ALTFIELD, s));
  }

  static Query bq(int msm, Object... occurQuery) {
    BooleanQuery.Builder b = new BooleanQuery.Builder();
    for (int i = 0; i < occurQuery.length; i += 2) {
      b.add((Query) occurQuery[i + 1], (Occur) occurQuery[i]);
    }
    b.setMinimumNumberShouldMatch(msm);
    return b.build();
  }

  /** The queries, by the names the Rust test builds them under. */
  static List<Object[]> explanationQueries() {
    List<Object[]> qs = new ArrayList<>();
    qs.add(new Object[] {"t_w1", t("w1")});
    qs.add(new Object[] {"t_xx", t("xx")});
    qs.add(new Object[] {"bq_must", bq(0, Occur.MUST, t("w1"), Occur.MUST, t("w2"))});
    qs.add(
        new Object[] {
          "bq_should", bq(0, Occur.SHOULD, t("w1"), Occur.SHOULD, t("w5"), Occur.SHOULD, t("xx"))
        });
    qs.add(
        new Object[] {
          "bq_mixed",
          bq(
              0, Occur.MUST, t("w1"), Occur.MUST_NOT, t("xx"), Occur.SHOULD, t("w2"), Occur.SHOULD,
              a("w3"))
        });
    qs.add(
        new Object[] {
          "bq_msm",
          bq(
              2, Occur.SHOULD, t("w1"), Occur.SHOULD, t("w2"), Occur.SHOULD, t("w5"), Occur.SHOULD,
              t("zz"))
        });
    qs.add(
        new Object[] {
          // in occur order: the Rust model keeps clauses per occur
          "bq_filter", bq(0, Occur.FILTER, t("xx"), Occur.SHOULD, t("w3"))
        });
    qs.add(new Object[] {"boost", new BoostQuery(t("w2"), 1000)});
    qs.add(new Object[] {"phrase", new PhraseQuery(0, FIELD, "w1", "w2")});
    qs.add(new Object[] {"phrase_sloppy", new PhraseQuery(2, FIELD, "w1", "w3")});
    qs.add(
        new Object[] {
          "dmq", new DisjunctionMaxQuery(List.of(t("w1"), a("w5"), t("xx")), 0.5f)
        });
    qs.add(new Object[] {"csq", new ConstantScoreQuery(t("w1"))});
    qs.add(new Object[] {"match_all", MatchAllDocsQuery.INSTANCE});
    MultiPhraseQuery.Builder mpq = new MultiPhraseQuery.Builder();
    mpq.add(new Term(FIELD, "w1"));
    mpq.add(new Term[] {new Term(FIELD, "w2"), new Term(FIELD, "w3")});
    qs.add(new Object[] {"mpq", mpq.build()});
    qs.add(
        new Object[] {
          "synonym", new SynonymQuery.Builder(FIELD).addTerm(new Term(FIELD, "w2")).addTerm(new Term(FIELD, "w3")).build()
        });
    return qs;
  }

  static void explanations() throws Exception {
    ByteBuffersDirectory dir = new ByteBuffersDirectory();
    Analyzer analyzer = new MockAnalyzer(new Random(0));
    try (IndexWriter w = new IndexWriter(dir, new IndexWriterConfig(analyzer))) {
      for (int i = 0; i < DOC_FIELDS.length; i++) {
        Document doc = new Document();
        doc.add(new StringField(KEY, "" + i, Field.Store.NO));
        doc.add(new SortedDocValuesField(KEY, new BytesRef("" + i)));
        doc.add(new TextField(FIELD, DOC_FIELDS[i], Field.Store.NO));
        doc.add(new TextField(ALTFIELD, DOC_FIELDS[i], Field.Store.NO));
        w.addDocument(doc);
      }
    }
    try (DirectoryReader reader = DirectoryReader.open(dir)) {
      for (Similarity sim : new Similarity[] {new BM25Similarity(), new ClassicSimilarity()}) {
        IndexSearcher searcher = new IndexSearcher(reader);
        // Rust's explain has no query cache: a cached FILTER clause would
        // explain as CachingWrapperWeight's constant score, not as its weight
        searcher.setQueryCache(null);
        searcher.setSimilarity(sim);
        for (Object[] nq : explanationQueries()) {
          Query q = (Query) nq[1];
          TopDocs td = searcher.search(q, 10);
          line("exp", sim.getClass().getSimpleName(), nq[0], td.totalHits.value());
          for (ScoreDoc sd : td.scoreDocs) {
            line("hit", sd.doc, bits(sd.score));
            String[] lines = searcher.explain(q, sd.doc).toString().split("\n");
            line("expl", lines.length);
            for (String l : lines) line("|", l);
          }
        }
      }
    }
  }

  // ----------------------------------------------------- search equivalence

  static char randomChar(Random random) {
    char c = (char) TestUtil.nextInt(random, 'a', 'z');
    if (random.nextBoolean()) {
      c = (char) TestUtil.nextInt(random, 'a', c);
    }
    return c;
  }

  static String randomFieldContents(Random random) {
    StringBuilder sb = new StringBuilder();
    int numTerms = random.nextInt(15);
    for (int i = 0; i < numTerms; i++) {
      if (sb.length() > 0) sb.append(' ');
      sb.append(randomChar(random));
    }
    return sb.toString();
  }

  static void equivalence() throws Exception {
    for (int seed = 0; seed < 4; seed++) {
      Random random = new Random(seed);
      String stopword = "" + randomChar(random);
      CharacterRunAutomaton stopset = new CharacterRunAutomaton(Automata.makeString(stopword));
      Analyzer analyzer = new MockAnalyzer(random, MockTokenizer.WHITESPACE, false, stopset);
      int numDocs = LuceneTestCase.atLeast(random, 100);
      List<String> contents = new ArrayList<>();
      for (int i = 0; i < numDocs; i++) contents.add(randomFieldContents(random));
      List<Integer> deletes = new ArrayList<>();
      for (int i = 0; i < numDocs / 20; i++) {
        deletes.add(random.nextInt(numDocs));
        random.nextBoolean();
      }
      line("eq", seed, stopword, numDocs);
      for (int i = 0; i < numDocs; i++) line("doc", i, "[" + contents.get(i) + "]");
      line("del", deletes.toString().replace(" ", ""));
      ByteBuffersDirectory dir = new ByteBuffersDirectory();
      try (IndexWriter w = new IndexWriter(dir, new IndexWriterConfig(analyzer))) {
        for (int i = 0; i < numDocs; i++) {
          Document doc = new Document();
          doc.add(new StringField("id", "" + i, Field.Store.NO));
          doc.add(new TextField("field", contents.get(i), Field.Store.NO));
          w.addDocument(doc);
        }
        w.commit();
        for (int id : deletes) w.deleteDocuments(new Term("id", "" + id));
      }
      try (DirectoryReader reader = DirectoryReader.open(dir)) {
        IndexSearcher searcher = new IndexSearcher(reader);
        for (String term : new String[] {"a", "b", "c", "e", stopword}) {
          TopDocs td = searcher.search(new TermQuery(new Term("field", term)), 5);
          StringBuilder hits = new StringBuilder();
          for (ScoreDoc sd : td.scoreDocs) hits.append(' ').append(sd.doc).append(':').append(bits(sd.score));
          line("eqhits", term, td.totalHits.value() + hits.toString());
        }
      }
    }
  }

  static void dumpAll() throws Exception {
    for (int i = 0; i < SIM_TESTS.size(); i++) {
      SimTest t = SIM_TESTS.get(i);
      for (int rep = 0; rep < REPS; rep++) {
        randomScoring(t.name(), 1000 * i + 2 * rep, t.get());
        bulkScore(t.name(), 1000 * i + 2 * rep + 1, t.get());
      }
    }
    List<Integer> shuffled = new ArrayList<>();
    for (int i = 0; i < 10; i++) shuffled.add(i);
    java.util.Collections.shuffle(shuffled, new Random(42));
    line("shuffle", shuffled.toString().replace(" ", ""));
    randomSimilarity();
    explanations();
    equivalence();
  }

  /** {@link MockTokenizer} reads the randomized-testing context, so the dump runs as a test. */
  @RunWith(RandomizedRunner.class)
  public static class Dump {
    @Test
    public void dump() throws Exception {
      dumpAll();
    }
  }

  public static void main(String[] args) throws Exception {
    Result result = JUnitCore.runClasses(Dump.class);
    for (Failure f : result.getFailures()) {
      System.err.println(f.getTrace());
    }
    if (!result.wasSuccessful() || result.getRunCount() != 1) {
      throw new AssertionError("GenTestFrameworkSearch: the dump did not run cleanly");
    }
    Path out = Path.of(args[0]);
    Files.createDirectories(out);
    Files.writeString(out.resolve("search.txt"), OUT.toString());
  }
}
