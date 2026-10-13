import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.function.Supplier;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.index.Term;
import org.apache.lucene.queryparser.classic.ParseException;
import org.apache.lucene.queryparser.classic.QueryParser;
import org.apache.lucene.queryparser.ext.ExtendableQueryParser;
import org.apache.lucene.queryparser.ext.ExtensionQuery;
import org.apache.lucene.queryparser.ext.Extensions;
import org.apache.lucene.queryparser.ext.ParserExtension;
import org.apache.lucene.search.BoostQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.TermQuery;

/**
 * M13 T13.1: {@code ExtendableQueryParser} over {@code fixtures/corpus/queryparser-ext.txt} and
 * the classic corpus, with five extensions (upper-cased term, nested parse, {@code null}, a
 * {@code ParseException}, a boosted term) under the default {@code :} delimiter and {@code #}
 * ({@code <config>.tsv}; rows as {@link GenQueryParserClassic}'s). {@code
 * crates/lucene-search/tests/queryparser_ext_fixtures.rs} compares.
 */
public class GenQueryParserExt {

  static Extensions extensions(char delimiter) {
    Extensions e = new Extensions(delimiter);
    e.add("t", new ParserExtension() {
      @Override
      public Query parse(ExtensionQuery q) {
        return new TermQuery(new Term(q.field(), q.rawQueryString().toUpperCase(Locale.ROOT)));
      }
    });
    e.add("p", new ParserExtension() {
      @Override
      public Query parse(ExtensionQuery q) throws ParseException {
        return q.topLevelParser().parse(q.rawQueryString());
      }
    });
    e.add("n", new ParserExtension() {
      @Override
      public Query parse(ExtensionQuery q) {
        return null;
      }
    });
    e.add("e", new ParserExtension() {
      @Override
      public Query parse(ExtensionQuery q) throws ParseException {
        throw new ParseException("extension e failed: " + q.rawQueryString());
      }
    });
    e.add("b", new ParserExtension() {
      @Override
      public Query parse(ExtensionQuery q) {
        return new BoostQuery(new TermQuery(new Term(q.field(), q.rawQueryString())), 2.5f);
      }
    });
    return e;
  }

  public static void main(String[] args) throws Exception {
    Path out = Path.of(args[0]).resolve("queryparser_ext");
    Files.createDirectories(out);
    Map<String, Supplier<QueryParser>> c = new LinkedHashMap<>();
    c.put("colon", () -> new ExtendableQueryParser("f", new StandardAnalyzer(), extensions(':')));
    c.put("hash", () -> new ExtendableQueryParser("f", new StandardAnalyzer(), extensions('#')));
    c.put("none", () -> new ExtendableQueryParser("f", new StandardAnalyzer()));
    List<String> corpus = new java.util.ArrayList<>(AnalysisRows.corpus("queryparser-ext.txt"));
    corpus.addAll(AnalysisRows.corpus("queryparser-classic.txt"));
    GenQueryParserClassic.write(out, "", c, corpus);
    // buildExtensionField / escapeExtensionField / splitExtensionField
    Extensions e = extensions(':');
    StringBuilder b = new StringBuilder();
    for (String s : new String[] {"a", "a:b", "+x", ""}) {
      b.append(AnalysisRows.esc(s)).append('\t')
          .append(AnalysisRows.esc(e.buildExtensionField(s))).append('\t')
          .append(AnalysisRows.esc(e.buildExtensionField(s, "fld"))).append('\t')
          .append(AnalysisRows.esc(e.escapeExtensionField(s))).append('\n');
    }
    for (String s : new String[] {"title", "title:t", ":t", "a:b:c", "title:", ":"}) {
      Extensions.Pair<String, String> p = e.splitExtensionField("dflt", s);
      b.append("split\t").append(AnalysisRows.esc(s)).append('\t').append(p.cur()).append('\t')
          .append(p.cud() == null ? "<null>" : p.cud()).append('\n');
    }
    Files.writeString(out.resolve("helpers.tsv"), b.toString(), StandardCharsets.UTF_8);
  }
}
