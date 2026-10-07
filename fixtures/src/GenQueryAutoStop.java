import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Random;
import java.util.stream.Stream;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.core.WhitespaceAnalyzer;
import org.apache.lucene.analysis.query.QueryAutoStopWordAnalyzer;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.analysis.tokenattributes.PositionIncrementAttribute;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.StoredField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.Term;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;

/**
 * M11 T11.6: {@code QueryAutoStopWordAnalyzer} over a Java-written index, recorded for
 * crates/lucene-search/tests/query_auto_stop_fixtures.rs.
 *
 * <p>{@code query_auto_stop/index}: three segments (no merging) of documents whose {@code body} and
 * {@code title} draw from a skewed vocabulary (some terms in nearly every document, some in few),
 * plus a {@code StringField} {@code id}, a stored-only {@code note} and a few deleted documents
 * (they count in {@code docFreq}, not in {@code numDocs}). {@code cases.txt}: for each
 * constructor in {@link #main}, a {@code #case} row, then {@code stop<TAB>field<TAB>words} (sorted)
 * for every field asked about, {@code all<TAB>field:word ...} ({@code getStopWords()}, sorted), and
 * {@code tokens<TAB>field<TAB>line<TAB>term:posInc ...} for {@link #LINES} through the analyzer.
 */
public class GenQueryAutoStop {
  static final String[] COMMON = {"the", "and", "of", "a"};
  static final String[] RARE = {
    "lucene", "rust", "index", "search", "query", "token", "stop", "word", "analyzer", "field",
    "segment", "merge", "écrit", "日本"
  };

  static final String[] LINES = {
    "the lucene index and the rust search",
    "a query of stop words",
    "日本 écrit",
    ""
  };

  static final String[] FIELDS_ASKED = {"body", "title", "id", "note", "missing"};

  static String words(Random r, int n) {
    StringBuilder b = new StringBuilder();
    for (int i = 0; i < n; i++) {
      if (i > 0) b.append(' ');
      // A quarter of the words from the four common ones.
      b.append(r.nextInt(4) == 0 ? COMMON[r.nextInt(COMMON.length)] : RARE[r.nextInt(RARE.length)]);
    }
    return b.toString();
  }

  static void record(StringBuilder o, String name, QueryAutoStopWordAnalyzer a) throws IOException {
    o.append("#case\t").append(name).append('\n');
    for (String f : FIELDS_ASKED) {
      String[] w = a.getStopWords(f);
      Arrays.sort(w);
      o.append("stop\t").append(f).append('\t').append(String.join(" ", w)).append('\n');
    }
    List<String> all = new ArrayList<>();
    for (Term t : a.getStopWords()) all.add(t.field() + ":" + t.text());
    all.sort(null);
    o.append("all\t").append(String.join(" ", all)).append('\n');
    for (String f : new String[] {"body", "title", "missing"}) {
      for (int ln = 0; ln < LINES.length; ln++) {
        o.append("tokens\t").append(f).append('\t').append(ln).append('\t');
        try (TokenStream ts = a.tokenStream(f, LINES[ln])) {
          CharTermAttribute term = ts.addAttribute(CharTermAttribute.class);
          PositionIncrementAttribute inc = ts.addAttribute(PositionIncrementAttribute.class);
          ts.reset();
          List<String> toks = new ArrayList<>();
          while (ts.incrementToken()) toks.add(term + ":" + inc.getPositionIncrement());
          ts.end();
          o.append(String.join(" ", toks));
        }
        o.append('\n');
      }
    }
  }

  public static void main(String[] args) throws IOException {
    Path out = Path.of(args[0]).resolve("query_auto_stop");
    if (Files.exists(out)) {
      try (Stream<Path> s = Files.walk(out)) {
        s.sorted(Comparator.reverseOrder()).forEach(p -> p.toFile().delete());
      }
    }
    Path index = out.resolve("index");
    Files.createDirectories(index);
    Random r = new Random(20261007L);
    StringBuilder o = new StringBuilder();
    try (Directory dir = FSDirectory.open(index)) {
      IndexWriterConfig cfg = new IndexWriterConfig(new WhitespaceAnalyzer());
      cfg.setUseCompoundFile(false);
      cfg.setMergePolicy(NoMergePolicy.INSTANCE);
      try (IndexWriter w = new IndexWriter(dir, cfg)) {
        int id = 0;
        for (int seg = 0; seg < 3; seg++) {
          for (int i = 0; i < 40; i++, id++) {
            Document doc = new Document();
            doc.add(new StringField("id", Integer.toString(id % 7), Field.Store.NO));
            doc.add(new TextField("body", words(r, 3 + r.nextInt(10)), Field.Store.NO));
            if (seg != 1) doc.add(new TextField("title", words(r, 1 + r.nextInt(3)), Field.Store.NO));
            doc.add(new StoredField("note", "the stored only"));
            w.addDocument(doc);
          }
          w.commit();
        }
        w.deleteDocuments(new Term("id", "3"));
        w.commit();
      }
      try (DirectoryReader reader = DirectoryReader.open(dir)) {
        Analyzer ws = new WhitespaceAnalyzer();
        record(o, "default", new QueryAutoStopWordAnalyzer(ws, reader));
        record(o, "maxDocFreq=10", new QueryAutoStopWordAnalyzer(ws, reader, 10));
        record(o, "percent=0.1", new QueryAutoStopWordAnalyzer(ws, reader, 0.1f));
        record(o, "fields=body,missing percent=0.25",
            new QueryAutoStopWordAnalyzer(ws, reader, List.of("body", "missing"), 0.25f));
        record(o, "fields=title,id maxDocFreq=3",
            new QueryAutoStopWordAnalyzer(ws, reader, List.of("title", "id"), 3));
        record(o, "maxDocFreq=1000", new QueryAutoStopWordAnalyzer(ws, reader, 1000));
      }
    }
    Files.writeString(out.resolve("cases.txt"), o.toString(), StandardCharsets.UTF_8);
  }
}
