import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.AnalyzerWrapper;
import org.apache.lucene.analysis.CharArraySet;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.FilterCodec;
import org.apache.lucene.codecs.PostingsFormat;
import org.apache.lucene.codecs.perfield.PerFieldPostingsFormat;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.suggest.document.CompletionAnalyzer;
import org.apache.lucene.search.suggest.document.ContextSuggestField;
import org.apache.lucene.search.suggest.document.SuggestField;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.Version;

/**
 * Writes {@code GenCompletionQuery}'s corpus (the same 90 documents, three segments, the same
 * deletions) with whichever Lucene release is on the classpath, its completion fields under that
 * release's newest {@code CompletionXXPostingsFormat} ({@code Completion90} on 9.0, {@code
 * Completion99} on 9.11, {@code Completion912} on 9.12, {@code Completion101} on 10.2 and 10.3,
 * over {@code Lucene101} and {@code Lucene103}). The codec
 * is the release's default under its own name with only the postings format overridden, so Lucene
 * 10.5.0 opens the index through backward-codecs as it would a real one; {@code
 * BwcCompletionDump} then records 10.5.0's answers.
 */
public class BwcCompletion {
  static final String[] FORMATS = {"Completion101", "Completion912", "Completion99", "Completion90"};

  public static void main(String[] args) throws Exception {
    Path out = Path.of(args[0]);
    Path index = out.resolve("index");
    Files.createDirectories(index);
    PostingsFormat completion = null;
    for (String name : FORMATS) {
      try {
        completion = PostingsFormat.forName(name);
        break;
      } catch (IllegalArgumentException e) {
        // not in this release
      }
    }
    if (completion == null) throw new IllegalStateException("no completion postings format");
    Codec base = Codec.getDefault();
    PerFieldPostingsFormat defaults = (PerFieldPostingsFormat) base.postingsFormat();
    PostingsFormat c = completion;
    PostingsFormat perField =
        new PerFieldPostingsFormat() {
          @Override
          public PostingsFormat getPostingsFormatForField(String field) {
            return field.equals("id") ? defaults.getPostingsFormatForField(field) : c;
          }
        };
    CharArraySet stops = new CharArraySet(List.of("the", "of"), false);
    Analyzer std = new StandardAnalyzer(stops);
    Analyzer nosep = new CompletionAnalyzer(new StandardAnalyzer(stops), false, true);
    Analyzer analyzer =
        new AnalyzerWrapper(Analyzer.PER_FIELD_REUSE_STRATEGY) {
          @Override
          protected Analyzer getWrappedAnalyzer(String field) {
            return field.equals("nosep") ? nosep : std;
          }
        };
    IndexWriterConfig cfg = new IndexWriterConfig(analyzer);
    cfg.setUseCompoundFile(false);
    cfg.setMergePolicy(NoMergePolicy.INSTANCE);
    cfg.setCodec(
        new FilterCodec(base.getName(), base) {
          @Override
          public PostingsFormat postingsFormat() {
            return perField;
          }
        });
    // GenCompletionQuery's corpus, draw for draw.
    Random r = new Random(13);
    String[] values = {
      "the beatles", "beach boys", "bee gees", "beyonce", "new york", "new jersey", "newark",
      "nirvana", "queen", "queen of the stone age", "cafe del mar", "café", "日本", "night",
      "nightwish", "music of the night", "the doors", "doors"
    };
    String[] contexts = {"music", "movies", "books", "m", "mus"};
    List<String> rows = new ArrayList<>();
    try (Directory dir = FSDirectory.open(index);
        IndexWriter w = new IndexWriter(dir, cfg)) {
      for (int i = 0; i < 90; i++) {
        String value = values[r.nextInt(values.length)];
        int weight = r.nextInt(4) == 0 ? 0 : r.nextInt(50);
        List<String> ctx = new ArrayList<>();
        for (String cx : contexts) if (r.nextInt(3) == 0) ctx.add(cx);
        Document d = new Document();
        d.add(new StringField("id", Integer.toString(i), Field.Store.NO));
        d.add(new SuggestField("suggest", value, weight));
        d.add(new SuggestField("nosep", value, weight));
        d.add(new ContextSuggestField("ctx", value, weight, ctx.toArray(new CharSequence[0])));
        w.addDocument(d);
        rows.add(i + "\t" + value + "\t" + weight + "\t" + String.join("\u001f", ctx));
        if (i == 39 || i == 69) w.commit();
      }
      for (int i = 41; i < 70; i += 3) w.deleteDocuments(new Term("id", Integer.toString(i)));
      w.commit();
    }
    Files.write(out.resolve("docs.tsv"), rows, StandardCharsets.UTF_8);
    Files.writeString(
        out.resolve("written_by.txt"),
        Version.LATEST + "\t" + completion.getName() + "\n",
        StandardCharsets.UTF_8);
  }
}
