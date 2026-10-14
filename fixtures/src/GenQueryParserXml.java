import java.io.ByteArrayInputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.IntPoint;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.FieldType;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.queryparser.xml.CoreParser;
import org.apache.lucene.queryparser.xml.ParserException;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.store.FSDirectory;

/**
 * M13 T13.1: {@code CoreParser} over {@code fixtures/corpus/queryparser-xml.txt}, one XML document a
 * line, default field {@code body}, against ten documents ({@code index/}) with an {@code IntPoint} field {@code n} (1 to 10); no
 * document has {@code x}, so a range on it finds nothing. A row of {@code xml.tsv}
 * is the escaped document, then {@code Q}, {@code toString("body")}, the class tree, and the matching
 * documents; or {@code X} and the exception's class and message. A syntax error is recorded by its
 * class alone: Xerces' wording is not reproduced. {@code
 * crates/lucene-search/tests/queryparser_xml_fixtures.rs} compares.
 */
public class GenQueryParserXml {

  public static void main(String[] args) throws Exception {
    Path out = Path.of(args[0]).resolve("queryparser_xml");
    Path indexDir = out.resolve("index");
    Files.createDirectories(indexDir);
    FieldType type = new FieldType();
    type.setIndexOptions(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS);
    type.setTokenized(true);
    type.freeze();
    try (FSDirectory dir = FSDirectory.open(indexDir);
        IndexWriter w = new IndexWriter(dir, new IndexWriterConfig(new StandardAnalyzer()).setOpenMode(IndexWriterConfig.OpenMode.CREATE).setMergePolicy(NoMergePolicy.INSTANCE))) {
      for (int i = 0; i < GenQueryParserSurround.BODIES.length; i++) {
        Document d = new Document();
        d.add(new Field("body", GenQueryParserSurround.BODIES[i], type));
        d.add(new Field("title", GenQueryParserSurround.TITLES[i], type));
        d.add(new IntPoint("n", i + 1));
        w.addDocument(d);
      }
      w.commit();
    }
    List<String> corpus = AnalysisRows.corpus("queryparser-xml.txt");
    CoreParser parser = new CoreParser("body", new StandardAnalyzer());
    StringBuilder b = new StringBuilder();
    try (FSDirectory dir = FSDirectory.open(indexDir);
        DirectoryReader reader = DirectoryReader.open(dir)) {
      IndexSearcher searcher = new IndexSearcher(reader);
      for (String xml : corpus) {
        b.append(AnalysisRows.esc(xml)).append('\t');
        try {
          Query q = parser.parse(new ByteArrayInputStream(xml.getBytes(StandardCharsets.UTF_8)));
          b.append("Q\t").append(AnalysisRows.esc(q.toString("body"))).append('\t')
              .append(AnalysisRows.esc(GenQueryParserClassic.describe(q))).append('\t');
          try {
            b.append(GenQueryParserSurround.hits(searcher, searcher.rewrite(q)));
          } catch (RuntimeException e) {
            b.append("X\t").append(AnalysisRows.esc(exception(e)));
          }
        } catch (ParserException e) {
          b.append("X\t").append(AnalysisRows.esc(parserException(e)));
        } catch (RuntimeException e) {
          b.append("X\t").append(AnalysisRows.esc(exception(e)));
        }
        b.append('\n');
      }
    }
    Files.writeString(out.resolve("xml.tsv"), b.toString(), StandardCharsets.UTF_8);
  }

  /** The class and message of a runtime exception: {@code class: message}. */
  static String exception(RuntimeException e) {
    return e.getClass().getName() + ": " + e.getMessage();
  }

  /** {@code ParserException}: a syntax error by its class alone. */
  static String parserException(ParserException e) {
    String m = e.getMessage();
    if (m != null && m.startsWith("Error parsing XML stream")) {
      return ParserException.class.getName() + ": (syntax)";
    }
    return ParserException.class.getName() + ": " + m;
  }
}
