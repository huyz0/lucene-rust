import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Supplier;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.AnalyzerWrapper;
import org.apache.lucene.analysis.Tokenizer;
import org.apache.lucene.analysis.core.WhitespaceAnalyzer;
import org.apache.lucene.analysis.core.WhitespaceTokenizer;
import org.apache.lucene.analysis.payloads.DelimitedPayloadTokenFilter;
import org.apache.lucene.analysis.payloads.IdentityEncoder;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.BinaryDocValuesField;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.LongPoint;
import org.apache.lucene.document.NumericDocValuesField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.memory.MemoryIndex;
import org.apache.lucene.queryparser.classic.QueryParser;
import org.apache.lucene.search.FieldExistsQuery;
import org.apache.lucene.search.MatchAllDocsQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.similarities.BM25Similarity;
import org.apache.lucene.search.similarities.BooleanSimilarity;
import org.apache.lucene.search.similarities.ClassicSimilarity;
import org.apache.lucene.search.similarities.Similarity;
import org.apache.lucene.util.BytesRef;

/**
 * M13 T13.5: {@code MemoryIndex}. Every document of {@code fixtures/corpus/memory-index-docs.txt}
 * (a line: {@code field=value} pairs separated by tabs, a field repeated for several values) is
 * indexed under each configuration ({@link #CONFIGS}: analyzer, similarity, offsets, payloads), and
 * every query of {@code memory-index-queries.txt} (classic syntax, default field {@code body}, parsed
 * with the configuration's analyzer) searched. {@code memory_index/<config>.tsv} rows are {@code
 * D}, the document number and the escaped {@code toStringDebug()}; then {@code S}, the document
 * number, the escaped query and {@code Float.floatToIntBits(search(q))} in hex, or {@code X} and
 * the exception. {@code scenarios.tsv}: the named cases of {@link #scenarios}, each {@code name},
 * then the score bits or the exception. {@code
 * crates/lucene-search/tests/memory_index_fixtures.rs} compares.
 */
public class GenMemoryIndex {

  /** {@code StandardAnalyzer} with a position-increment gap of 100 and an offset gap of 10. */
  static Analyzer gapped() {
    return new AnalyzerWrapper(Analyzer.PER_FIELD_REUSE_STRATEGY) {
      final Analyzer inner = new StandardAnalyzer();

      @Override
      protected Analyzer getWrappedAnalyzer(String fieldName) {
        return inner;
      }

      @Override
      public int getPositionIncrementGap(String fieldName) {
        return 100;
      }

      @Override
      public int getOffsetGap(String fieldName) {
        return 10;
      }
    };
  }

  /** Whitespace, then {@code DelimitedPayloadTokenFilter('|', IdentityEncoder)}. */
  static Analyzer payloads() {
    return new Analyzer() {
      @Override
      protected TokenStreamComponents createComponents(String field) {
        Tokenizer t = new WhitespaceTokenizer();
        return new TokenStreamComponents(
            t, new DelimitedPayloadTokenFilter(t, '|', new IdentityEncoder()));
      }
    };
  }

  record Config(String name, Supplier<Analyzer> analyzer, Supplier<Similarity> similarity,
      boolean offsets, boolean payloads) {}

  static final List<Config> CONFIGS = List.of(
      new Config("std", StandardAnalyzer::new, BM25Similarity::new, false, false),
      new Config("std_offsets", StandardAnalyzer::new, BM25Similarity::new, true, false),
      new Config("tuned", StandardAnalyzer::new, () -> new BM25Similarity(1.5f, 0.5f), false, false),
      new Config("classic", StandardAnalyzer::new, ClassicSimilarity::new, true, false),
      new Config("boolean", StandardAnalyzer::new, BooleanSimilarity::new, false, false),
      new Config("ws", WhitespaceAnalyzer::new, BM25Similarity::new, true, true),
      new Config("gaps", GenMemoryIndex::gapped, BM25Similarity::new, true, false),
      new Config("payloads", GenMemoryIndex::payloads, BM25Similarity::new, true, true));

  /** One line of the documents corpus: its {@code field=value} pairs, in order. */
  static List<String[]> fields(String line) {
    List<String[]> out = new ArrayList<>();
    for (String pair : line.split("\t")) {
      int eq = pair.indexOf('=');
      out.add(new String[] {pair.substring(0, eq), pair.substring(eq + 1)});
    }
    return out;
  }

  static String bits(float f) {
    return Integer.toHexString(Float.floatToIntBits(f));
  }

  static String error(Throwable e) {
    return "X\t" + AnalysisRows.esc(e.getClass().getName() + ": " + e.getMessage());
  }

  interface Scenario {
    float run() throws Exception;
  }

  /** The named cases: the API's refusals and the non-text fields. */
  static Map<String, Scenario> scenarios() {
    Map<String, Scenario> s = new LinkedHashMap<>();
    s.put("empty_match_all", () -> new MemoryIndex().search(new MatchAllDocsQuery()));
    s.put("frozen_add", () -> {
      MemoryIndex mi = new MemoryIndex();
      mi.addField("body", "a b", new StandardAnalyzer());
      mi.freeze();
      mi.addField("body", "c", new StandardAnalyzer());
      return 0;
    });
    s.put("frozen_similarity", () -> {
      MemoryIndex mi = new MemoryIndex();
      mi.freeze();
      mi.setSimilarity(new ClassicSimilarity());
      return 0;
    });
    s.put("numeric_twice", () -> {
      MemoryIndex mi = new MemoryIndex();
      mi.addField(new NumericDocValuesField("n", 1), null);
      mi.addField(new NumericDocValuesField("n", 2), null);
      return 0;
    });
    s.put("binary_twice", () -> {
      MemoryIndex mi = new MemoryIndex();
      mi.addField(new BinaryDocValuesField("b", new BytesRef("x")), null);
      mi.addField(new BinaryDocValuesField("b", new BytesRef("y")), null);
      return 0;
    });
    s.put("doc_values_type_change", () -> {
      MemoryIndex mi = new MemoryIndex();
      mi.addField(new NumericDocValuesField("d", 1), null);
      mi.addField(new BinaryDocValuesField("d", new BytesRef("y")), null);
      return 0;
    });
    s.put("point_in_range", () -> {
      MemoryIndex mi = new MemoryIndex();
      mi.addField(new LongPoint("p", 5), null);
      mi.addField(new LongPoint("p", 50), null);
      return mi.search(LongPoint.newRangeQuery("p", 1, 10));
    });
    s.put("point_out_of_range", () -> {
      MemoryIndex mi = new MemoryIndex();
      mi.addField(new LongPoint("p", 5), null);
      return mi.search(LongPoint.newRangeQuery("p", 6, 10));
    });
    s.put("string_field", () -> {
      MemoryIndex mi = new MemoryIndex();
      mi.addField(new StringField("id", "Ab-1", Field.Store.NO), null);
      mi.addField("body", "some text", new StandardAnalyzer());
      return mi.search(new QueryParser("id", new WhitespaceAnalyzer()).parse("Ab-1 OR body:text"));
    });
    s.put("same_field_twice", () -> {
      MemoryIndex mi = new MemoryIndex();
      Analyzer a = new StandardAnalyzer();
      mi.addField("body", "the quick", a);
      mi.addField("body", "brown fox", a);
      return mi.search(new QueryParser("body", a).parse("\"quick brown\""));
    });
    s.put("keyword_stream", () -> {
      MemoryIndex mi = new MemoryIndex(true);
      mi.addField("tags", mi.keywordTokenStream(List.of("Red Fox", "blue", "Red Fox")));
      return mi.search(new QueryParser("tags", new WhitespaceAnalyzer()).parse("\"Red Fox\" OR blue"));
    });
    s.put("exists_text", () -> {
      MemoryIndex mi = new MemoryIndex();
      mi.addField("body", "a b", new StandardAnalyzer());
      return mi.search(new FieldExistsQuery("body"));
    });
    s.put("exists_missing", () -> {
      MemoryIndex mi = new MemoryIndex();
      mi.addField("body", "a b", new StandardAnalyzer());
      return mi.search(new FieldExistsQuery("title"));
    });
    s.put("exists_doc_values", () -> {
      MemoryIndex mi = new MemoryIndex();
      mi.addField(new NumericDocValuesField("n", 7), null);
      return mi.search(new FieldExistsQuery("n"));
    });
    s.put("text_beside_doc_values", () -> {
      MemoryIndex mi = new MemoryIndex(true);
      mi.addField(new NumericDocValuesField("n", 7), null);
      mi.addField("body", "the quick brown fox", new StandardAnalyzer());
      return mi.search(new QueryParser("body", new StandardAnalyzer()).parse("\"quick brown\" fox"));
    });
    s.put("payloads_beside_doc_values", () -> {
      MemoryIndex mi = new MemoryIndex(true, true);
      mi.addField(new NumericDocValuesField("n", 7), null);
      mi.addField("body", "quick|5 brown|x fox", payloads());
      return mi.search(new QueryParser("body", new WhitespaceAnalyzer()).parse("\"quick brown\" fox"));
    });
    s.put("similarity_after_reset", () -> {
      MemoryIndex mi = new MemoryIndex();
      mi.setSimilarity(new ClassicSimilarity());
      mi.addField("body", "a a b", new StandardAnalyzer());
      mi.reset();
      mi.addField("body", "a a b", new StandardAnalyzer());
      return mi.search(new QueryParser("body", new StandardAnalyzer()).parse("a"));
    });
    return s;
  }

  public static void main(String[] args) throws Exception {
    Path out = Path.of(args[0]).resolve("memory_index");
    Files.createDirectories(out);
    List<String> docs = AnalysisRows.corpus("memory-index-docs.txt");
    List<String> queries = AnalysisRows.corpus("memory-index-queries.txt");
    for (Config c : CONFIGS) {
      Analyzer analyzer = c.analyzer().get();
      QueryParser parser = new QueryParser("body", analyzer);
      List<Query> parsed = new ArrayList<>();
      for (String q : queries) {
        parsed.add(parser.parse(q));
      }
      StringBuilder b = new StringBuilder();
      for (int d = 0; d < docs.size(); d++) {
        MemoryIndex mi = new MemoryIndex(c.offsets(), c.payloads());
        mi.setSimilarity(c.similarity().get());
        for (String[] f : fields(docs.get(d))) {
          mi.addField(f[0], f[1], analyzer);
        }
        b.append("D\t").append(d).append('\t').append(AnalysisRows.esc(mi.toStringDebug())).append('\n');
        mi.freeze();
        for (int i = 0; i < queries.size(); i++) {
          b.append("S\t").append(d).append('\t').append(AnalysisRows.esc(queries.get(i))).append('\t');
          try {
            b.append(bits(mi.search(parsed.get(i))));
          } catch (RuntimeException e) {
            b.append(error(e));
          }
          b.append('\n');
        }
      }
      Files.writeString(out.resolve(c.name() + ".tsv"), b.toString(), StandardCharsets.UTF_8);
    }
    StringBuilder b = new StringBuilder();
    for (Map.Entry<String, Scenario> e : scenarios().entrySet()) {
      b.append(e.getKey()).append('\t');
      try {
        b.append(bits(e.getValue().run()));
      } catch (Exception ex) {
        b.append(error(ex));
      }
      b.append('\n');
    }
    Files.writeString(out.resolve("scenarios.tsv"), b.toString(), StandardCharsets.UTF_8);
  }
}
