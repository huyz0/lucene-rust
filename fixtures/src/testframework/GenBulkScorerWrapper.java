import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.Tokenizer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.ConstantScoreQuery;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.MatchAllDocsQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreMode;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.search.BulkScorer;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.LeafCollector;
import org.apache.lucene.search.Scorable;
import org.apache.lucene.tests.search.BulkScorerWrapperScorer;
import org.apache.lucene.util.Bits;

/** Calls the real wrapper over a deterministic, range-respecting BulkScorer. */
public class GenBulkScorerWrapper {
  static final int END = DocIdSetIterator.NO_MORE_DOCS;
  static final StringBuilder OUT = new StringBuilder();
  static void row(Object... values) {
    for (int i = 0; i < values.length; ++i) {
      if (i != 0) OUT.append('\t');
      OUT.append(values[i]);
    }
    OUT.append('\n');
  }
  static final class Value extends Scorable {
    float value;
    public float score() { return value; }
  }
  static final class Source extends BulkScorer {
    final int[] docs, bits;
    final int failAt;
    final long cost;
    final IOException failure = new IOException("injected bulk failure");
    final StringBuilder trace = new StringBuilder();
    int index, calls;
    int failAfter;
    Source(int[] docs, int[] bits, int failAt, long cost) {
      this.docs = docs; this.bits = bits; this.failAt = failAt; this.cost = cost;
    }
    public long cost() { return cost; }
    public int score(LeafCollector collector, Bits accept, int min, int max) throws IOException {
      if (accept != null) throw new AssertionError("wrapper passed acceptDocs");
      if (trace.length() != 0) trace.append(',');
      trace.append(min).append(':').append(max);
      if (++calls == failAt && failAfter == 0) throw failure;
      Value value = new Value();
      collector.setScorer(value);
      while (index < docs.length && docs[index] < min) ++index;
      while (index < docs.length && docs[index] < max) {
        value.value = Float.intBitsToFloat(bits[index]);
        collector.collect(docs[index++]);
        if (calls == failAt && index == failAfter) throw failure;
      }
      // Deliberately return a legal lower bound, exposing empty refill windows.
      return index == docs.length ? END : max;
    }
    String takeTrace() { String text = trace.toString(); trace.setLength(0); return text; }
  }
  static void capture(String label, int buffer, int[] docs, int[] bits, int[] targets, int failAt)
      throws Exception {
    Source source = new Source(docs, bits, failAt, 0x123456789L);
    if (label.equals("partialFailure")) source.failAfter = 2;
    BulkScorerWrapperScorer wrapper = new BulkScorerWrapperScorer(source, buffer);
    DocIdSetIterator iterator = wrapper.iterator();
    row("initial", label, buffer, wrapper.docID(), iterator.docID(), iterator.cost(),
        Float.floatToRawIntBits(wrapper.getMaxScore(-1)));
    for (int target : targets) {
      try {
        int doc = target < 0 ? iterator.nextDoc() : iterator.advance(target);
        row("step", label, target, doc, wrapper.docID(), iterator.docID(),
            doc == END ? "-" : Integer.toUnsignedString(Float.floatToRawIntBits(wrapper.score())),
            source.takeTrace());
      } catch (IOException failure) {
        row("failure", label, target, failure == source.failure, wrapper.docID(),
            source.takeTrace());
      }
    }
  }
  static void indexed() throws Exception {
    try (var dir = new ByteBuffersDirectory(); var analyzer = new Analyzer() {
      protected TokenStreamComponents createComponents(String field) {
        return new TokenStreamComponents(new Tokenizer() {
          public boolean incrementToken() { return false; }
        });
      }
    }) {
      try (var writer = new IndexWriter(dir, new IndexWriterConfig(analyzer))) {
        for (int id = 0; id < 97; ++id) {
          var doc = new Document();
          doc.add(new StringField("tag", id % 11 == 0 ? "hit" : "miss", Field.Store.NO));
          writer.addDocument(doc);
        }
      }
      try (var reader = DirectoryReader.open(dir)) {
        var searcher = new IndexSearcher(reader);
        Query[] queries = {new MatchAllDocsQuery(),
            new ConstantScoreQuery(new TermQuery(new Term("tag", "hit")))};
        for (int q = 0; q < queries.length; ++q) {
          for (int buffer : new int[] {1, 4, 17, 128}) {
            for (boolean advance : new boolean[] {false, true}) {
              var weight = searcher.createWeight(searcher.rewrite(queries[q]), ScoreMode.COMPLETE, 1);
              var bulk = weight.bulkScorer(reader.leaves().get(0));
              var wrapper = new BulkScorerWrapperScorer(bulk, buffer);
              var iterator = wrapper.iterator();
              String label = q == 0 ? "all" : "term";
              row("indexed", label, buffer, advance, iterator.cost());
              for (int target = 0;; target += 7) {
                int doc = advance ? iterator.advance(target) : iterator.nextDoc();
                row("indexedStep", doc, doc == END ? "-" :
                    Integer.toUnsignedString(Float.floatToRawIntBits(wrapper.score())));
                if (doc == END) break;
                if (advance) target = Math.max(target, doc);
              }
            }
          }
        }
      }
    }
  }
  public static void main(String[] args) throws Exception {
    int[] docs = {0, 1, 3, 4, 9, 21, 22, 70, 71, 95};
    int[] bits = {0, 0x80000000, 1, 0x3f800000, 0xbf800000, 0x7f800000,
        0xff800000, 0x7fc01234, 0x3eaaaaab, 0x7f7fffff};
    for (int buffer : new int[] {1, 2, 4, 7, 128}) {
      capture("next" + buffer, buffer, docs, bits,
          new int[] {-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1}, -1);
      capture("advance" + buffer, buffer, docs, bits,
          new int[] {1,2,8,20,23,71,72,96}, -1);
      capture("mixed" + buffer, buffer, docs, bits,
          new int[] {-1,3,-1,21,-1,69,-1,90,-1}, -1);
    }
    capture("empty", 3, new int[0], new int[0], new int[] {-1}, -1);
    capture("failure", 4, docs, bits, new int[] {-1,-1,9,9,-1,70}, 2);
    capture("partialFailure", 4, docs, bits, new int[] {-1,-1,-1,9,-1}, 1);
    // A second iterator shares the outer cursor and score buffers.
    Source source = new Source(docs, bits, -1, 10);
    BulkScorerWrapperScorer wrapper = new BulkScorerWrapperScorer(source, 7);
    DocIdSetIterator a = wrapper.iterator(), b = wrapper.iterator();
    row("shared", a.nextDoc(), b.docID(), b.nextDoc(), a.docID(), wrapper.docID(),
        Integer.toUnsignedString(Float.floatToRawIntBits(wrapper.score())));
    indexed();
    Path out = Path.of(args[0]);
    Files.createDirectories(out);
    Files.writeString(out.resolve("bulk_scorer_wrapper.tsv"), OUT);
  }
}
