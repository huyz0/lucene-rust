import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.*;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.tests.search.BlockScoreQueryWrapper;

/** Actual BlockScoreQueryWrapper scorers over a two-segment Lucene index. */
public class GenBlockScoreQueryWrapper {
  static final List<String> OUT = new ArrayList<>();
  static final int[] DOCS = {0, 2, 3, 6, 8, 9, 11, 14};
  static final float[] SCORES = {.25f, 2f, 1f, 5f, .5f, 3f, 7f, 4f};
  static final int[] LENGTHS = {1, 2, 3, 4, 7, 32, Integer.MAX_VALUE};
  static final int[] TARGETS = {-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 15, Integer.MAX_VALUE};
  static void line(Object... values) {
    StringBuilder s = new StringBuilder();
    for (Object value : values) { if (!s.isEmpty()) s.append('\t'); s.append(value); }
    OUT.add(s.toString());
  }
  static String bits(float value) { return String.format("%08x", Float.floatToRawIntBits(value)); }
  static Query query(boolean empty) {
    if (empty) return new TermQuery(new Term("id", "missing"));
    BooleanQuery.Builder b = new BooleanQuery.Builder();
    for (int i = 0; i < DOCS.length; i++) {
      Query term = new ConstantScoreQuery(new TermQuery(new Term("id", Integer.toString(DOCS[i]))));
      b.add(new BoostQuery(term, SCORES[i]), BooleanClause.Occur.SHOULD);
    }
    return b.build();
  }
  public static void main(String[] args) throws Exception {
    try (var dir = new ByteBuffersDirectory()) {
      var config = new IndexWriterConfig(new StandardAnalyzer());
      config.setMergePolicy(NoMergePolicy.INSTANCE);
      try (var writer = new IndexWriter(dir, config)) {
        for (int size : new int[] {16, 6}) {
          for (int i = 0; i < size; i++) {
            var doc = new Document();
            doc.add(new StringField("id", Integer.toString(i), Field.Store.NO));
            writer.addDocument(doc);
          }
          writer.commit();
        }
      }
      try (var reader = DirectoryReader.open(dir)) {
        var searcher = new IndexSearcher(reader);
        searcher.setQueryCache(null);
        for (boolean empty : new boolean[] {false, true}) {
          Query inner = query(empty);
          for (int length : LENGTHS) {
            Query wrapped = new BlockScoreQueryWrapper(inner, length);
            Weight weight = searcher.createWeight(searcher.rewrite(wrapped), ScoreMode.COMPLETE, 1);
            line("meta", empty, length, wrapped.toString().equals(inner.toString()),
                wrapped.equals(new BlockScoreQueryWrapper(inner, length)),
                wrapped.equals(new BlockScoreQueryWrapper(inner, length == 1 ? 2 : 1)));
            var terms = new java.util.HashSet<Term>();
            wrapped.visit(QueryVisitor.termCollector(terms));
            line("visit", empty, length, terms.size());
            for (var leaf : reader.leaves()) {
              Scorer scorer = weight.scorer(leaf);
              line("leaf", empty, length, leaf.ord, scorer == null, weight.isCacheable(leaf));
              if (scorer == null) continue;
              line("initial", empty, length, leaf.ord, scorer.docID(), bits(scorer.score()), scorer.iterator().cost(), scorer.twoPhaseIterator() == null);
              var it = scorer.iterator();
              while (true) {
                int doc = it.nextDoc();
                line("next", empty, length, leaf.ord, doc, bits(scorer.score()));
                if (doc == DocIdSetIterator.NO_MORE_DOCS) break;
              }
              for (int target : TARGETS) {
                scorer = weight.scorer(leaf);
                int doc = scorer.iterator().advance(target);
                line("advance", empty, length, leaf.ord, target, doc, bits(scorer.score()));
                scorer = weight.scorer(leaf);
                int end = scorer.advanceShallow(target);
                for (int upTo : new int[] {target, end, Integer.MAX_VALUE}) {
                  line("shallow", empty, length, leaf.ord, target, end, upTo, bits(scorer.getMaxScore(upTo)), scorer.docID());
                }
                scorer = weight.scorer(leaf);
                int first = scorer.iterator().nextDoc();
                end = scorer.advanceShallow(Math.max(first, target));
                line("mixed", empty, length, leaf.ord, target, end, bits(scorer.getMaxScore(Integer.MAX_VALUE)), scorer.docID());
              }
            }
          }
        }
        for (int length : new int[] {0, -1, Integer.MIN_VALUE}) {
          Query wrapped = new BlockScoreQueryWrapper(query(false), length);
          Weight weight = searcher.createWeight(searcher.rewrite(wrapped), ScoreMode.COMPLETE, 1);
          try { weight.scorer(reader.leaves().get(0)).advanceShallow(0); line("invalid", length, "none"); }
          catch (RuntimeException ex) { line("invalid", length, ex.getClass().getSimpleName()); }
          for (ScoreMode mode : new ScoreMode[] {ScoreMode.COMPLETE_NO_SCORES, ScoreMode.TOP_DOCS}) {
            weight = searcher.createWeight(searcher.rewrite(wrapped), mode, 1);
            Scorer scorer = weight.scorer(reader.leaves().get(0));
            int count = 0;
            while (scorer.iterator().nextDoc() != DocIdSetIterator.NO_MORE_DOCS) count++;
            line("noscores", length, mode, count);
          }
        }
        Query rewriteInner = new BoostQuery(query(false), 1);
        Query rewritten = searcher.rewrite(new BlockScoreQueryWrapper(rewriteInner, 3));
        line("rewrite", rewritten.equals(new BlockScoreQueryWrapper(searcher.rewrite(rewriteInner), 3)));
      }
    }
    Path out = Path.of(args[0]); Files.createDirectories(out);
    Files.write(out.resolve("block_score_query_wrapper.tsv"), OUT);
  }
}
