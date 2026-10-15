import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.queryparser.classic.MultiFieldQueryParser;
import org.apache.lucene.queryparser.classic.ParseException;
import org.apache.lucene.queryparser.classic.QueryParser;
import org.apache.lucene.queryparser.complexPhrase.ComplexPhraseQueryParser;
import org.apache.lucene.queryparser.flexible.standard.StandardQueryParser;
import org.apache.lucene.queryparser.simple.SimpleQueryParser;
import org.apache.lucene.queryparser.xml.CoreParser;
import org.apache.lucene.queryparser.surround.query.SrndQuery;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import java.io.ByteArrayInputStream;
import org.apache.lucene.store.FSDirectory;

/**
 * M13 T13.1's pairs (the Rust twin is {@code benchmarks/rust-runner/src/micro_queryparser.rs}):
 * the classic {@code QueryParser} over {@code fixtures/corpus/queryparser-classic.txt} and over
 * {@code fixtures/data/queryparser_classic/fragments.txt} (most of which are syntax errors), and
 * {@code MultiFieldQueryParser} over the corpus; {@code ComplexPhraseQueryParser} over its corpus,
 * each query parsed and rewritten against {@code fixtures/data/queryparser_complexphrase/index}. Units are queries; output is {@code
 * name\tns_per_query\tqueries}.
 */
public class QueryParserMicro {
  /** {@code AnalysisRows.esc} undone. */
  static String unesc(String s) {
    StringBuilder b = new StringBuilder();
    for (int i = 0; i < s.length(); i++) {
      char c = s.charAt(i);
      if (c != '\\' || i + 1 == s.length()) {
        b.append(c);
        continue;
      }
      char n = s.charAt(++i);
      switch (n) {
        case 't' -> b.append('\t');
        case 'n' -> b.append('\n');
        case 'r' -> b.append('\r');
        case 'u' -> {
          b.append((char) Integer.parseInt(s.substring(i + 1, i + 5), 16));
          i += 4;
        }
        default -> b.append(n);
      }
    }
    return b.toString();
  }

  static void run(String name, QueryParser p, List<String> queries) throws Exception {
    SweepMicro.measure(
        name,
        () -> {
          for (String q : queries) {
            try {
              Query parsed = p.parse(q);
              SweepMicro.sink += parsed.hashCode();
            } catch (ParseException | RuntimeException e) {
              SweepMicro.sink += e.getMessage().length();
            }
          }
          return queries.size();
        });
  }

  static void runSimple(String name, SimpleQueryParser p, List<String> queries) throws Exception {
    SweepMicro.measure(
        name,
        () -> {
          for (String q : queries) {
            try {
              SweepMicro.sink += p.parse(q).hashCode();
            } catch (RuntimeException e) {
              SweepMicro.sink += e.getMessage().length();
            }
          }
          return queries.size();
        });
  }

  static void runSurround(String name, List<String> queries) throws Exception {
    SweepMicro.measure(
        name,
        () -> {
          for (String q : queries) {
            try {
              SrndQuery parsed = org.apache.lucene.queryparser.surround.parser.QueryParser.parse(q);
              SweepMicro.sink += parsed.toString().length();
            } catch (org.apache.lucene.queryparser.surround.parser.ParseException | RuntimeException e) {
              SweepMicro.sink += e.getMessage().length();
            }
          }
          return queries.size();
        });
  }

  public static void main(String[] args) throws Exception {
    List<String> corpus = new ArrayList<>();
    for (String line : Files.readAllLines(Path.of("fixtures/corpus/queryparser-classic.txt"), StandardCharsets.UTF_8)) {
      corpus.add(line);
    }
    List<String> fragments = new ArrayList<>();
    for (String line : Files.readAllLines(Path.of("fixtures/data/queryparser_classic/fragments.txt"), StandardCharsets.UTF_8)) {
      fragments.add(unesc(line));
    }
    run("qp_classic_corpus", new QueryParser("f", new StandardAnalyzer()), corpus);
    run("qp_classic_fragments", new QueryParser("f", new StandardAnalyzer()), fragments);
    List<String> simple = new ArrayList<>(corpus);
    simple.addAll(Files.readAllLines(Path.of("fixtures/corpus/queryparser-simple.txt"), StandardCharsets.UTF_8));
    runSimple("qp_simple_corpus", new SimpleQueryParser(new StandardAnalyzer(), "f"), simple);
    runSurround("qp_surround_corpus",
        Files.readAllLines(Path.of("fixtures/corpus/queryparser-surround.txt"), StandardCharsets.UTF_8));
    run("qp_multi_field_corpus", new MultiFieldQueryParser(new String[] {"title", "body"}, new StandardAnalyzer()), corpus);
    List<String> flexible = new ArrayList<>(corpus);
    flexible.addAll(Files.readAllLines(Path.of("fixtures/corpus/queryparser-flexible.txt"), StandardCharsets.UTF_8));
    StandardQueryParser flexibleParser = new StandardQueryParser(new StandardAnalyzer());
    SweepMicro.measure(
        "qp_flexible_corpus",
        () -> {
          for (String q : flexible) {
            try {
              SweepMicro.sink += flexibleParser.parse(q, "f").hashCode();
            } catch (org.apache.lucene.queryparser.flexible.core.QueryNodeException | RuntimeException e) {
              SweepMicro.sink += String.valueOf(e.getMessage()).length();
            }
          }
          return flexible.size();
        });
    List<String> xml = Files.readAllLines(Path.of("fixtures/corpus/queryparser-xml.txt"), StandardCharsets.UTF_8);
    CoreParser xmlParser = new CoreParser("body", new StandardAnalyzer());
    SweepMicro.measure(
        "qp_xml_corpus",
        () -> {
          for (String q : xml) {
            try {
              SweepMicro.sink += xmlParser.parse(new ByteArrayInputStream(q.getBytes(StandardCharsets.UTF_8))).hashCode();
            } catch (org.apache.lucene.queryparser.xml.ParserException | RuntimeException e) {
              SweepMicro.sink += String.valueOf(e.getMessage()).length();
            }
          }
          return xml.size();
        });
    List<String> complex =
        Files.readAllLines(Path.of("fixtures/corpus/queryparser-complexphrase.txt"), StandardCharsets.UTF_8);
    try (FSDirectory dir = FSDirectory.open(Path.of("fixtures/data/queryparser_complexphrase/index"));
        DirectoryReader reader = DirectoryReader.open(dir)) {
      IndexSearcher searcher = new IndexSearcher(reader);
      ComplexPhraseQueryParser p = new ComplexPhraseQueryParser("body", new StandardAnalyzer());
      SweepMicro.measure(
          "qp_complexphrase_corpus",
          () -> {
            for (String q : complex) {
              try {
                SweepMicro.sink += searcher.rewrite(p.parse(q)).hashCode();
              } catch (ParseException | RuntimeException e) {
                SweepMicro.sink += String.valueOf(e.getMessage()).length();
              }
            }
            return complex.size();
          });
    }
  }
}
