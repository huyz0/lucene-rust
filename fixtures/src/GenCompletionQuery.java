import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Random;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.CharArraySet;
import org.apache.lucene.analysis.miscellaneous.PerFieldAnalyzerWrapper;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.codecs.PostingsFormat;
import org.apache.lucene.codecs.lucene104.Lucene104Codec;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.suggest.BitsProducer;
import org.apache.lucene.search.suggest.document.Completion104PostingsFormat;
import org.apache.lucene.search.suggest.document.CompletionAnalyzer;
import org.apache.lucene.search.suggest.document.CompletionQuery;
import org.apache.lucene.search.suggest.document.ContextQuery;
import org.apache.lucene.search.suggest.document.ContextSuggestField;
import org.apache.lucene.search.suggest.document.FuzzyCompletionQuery;
import org.apache.lucene.search.suggest.document.PrefixCompletionQuery;
import org.apache.lucene.search.suggest.document.RegexCompletionQuery;
import org.apache.lucene.search.suggest.document.SuggestField;
import org.apache.lucene.search.suggest.document.SuggestIndexSearcher;
import org.apache.lucene.search.suggest.document.TopSuggestDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.automaton.Operations;

/**
 * M13 T13.3d: the query side of {@code suggest.document}. Three segments (one with deletions) of
 * documents holding a {@link SuggestField} {@code suggest}, the same values under a {@link
 * CompletionAnalyzer} that drops separators ({@code nosep}) and a {@link ContextSuggestField}
 * {@code ctx}, all analyzed by a StandardAnalyzer with "the" and "of" as stop words (so values
 * have holes). {@code index/} is the index, {@code docs.tsv} its documents ({@code id TAB value
 * TAB weight TAB contexts}), and {@code queries.tsv} every query with Lucene's answer: {@code
 * kind TAB field TAB text TAB options TAB n TAB dedup TAB filter TAB hits}, hits {@code
 * \u001f}-separated, each {@code key \u001e doc \u001e score \u001e context}.
 */
public class GenCompletionQuery {
  static final CharArraySet STOPS = new CharArraySet(List.of("the", "of"), false);

  /** The even segment-local docs: a {@code BitsProducer} both engines can compute. */
  static final BitsProducer EVEN =
      new BitsProducer() {
        @Override
        public Bits getBits(LeafReaderContext context) {
          int max = context.reader().maxDoc();
          return new Bits() {
            @Override
            public boolean get(int index) {
              return index % 2 == 0;
            }

            @Override
            public int length() {
              return max;
            }
          };
        }
      };

  /** A filter matching nothing: {@code Bits.MatchNoBits} skips the segment. */
  static final BitsProducer NONE =
      new BitsProducer() {
        @Override
        public Bits getBits(LeafReaderContext context) {
          return new Bits.MatchNoBits(context.reader().maxDoc());
        }
      };

  public static void main(String[] args) throws Exception {
    Path out = Path.of(args[0]).resolve("completion_query");
    Path index = out.resolve("index");
    Files.createDirectories(index);
    try (var old = Files.list(index)) {
      for (Path p : old.toList()) Files.delete(p);
    }
    Random r = new Random(13);
    String[] values = {
      "the beatles", "beach boys", "bee gees", "beyonce", "new york", "new jersey", "newark",
      "nirvana", "queen", "queen of the stone age", "cafe del mar", "café", "日本", "night",
      "nightwish", "music of the night", "the doors", "doors"
    };
    String[] contexts = {"music", "movies", "books", "m", "mus"};
    List<String> rows = new ArrayList<>();
    Analyzer std = new StandardAnalyzer(STOPS);
    Analyzer analyzer =
        new PerFieldAnalyzerWrapper(
            std, Map.of("nosep", new CompletionAnalyzer(new StandardAnalyzer(STOPS), false, true)));
    IndexWriterConfig cfg = new IndexWriterConfig(analyzer);
    cfg.setUseCompoundFile(false);
    cfg.setMergePolicy(NoMergePolicy.INSTANCE);
    // One instance for every completion field, as OpenSearch's codec has: the
    // fields share one Completion104 suffix (a new instance per field would
    // give each its own).
    PostingsFormat completion = new Completion104PostingsFormat();
    cfg.setCodec(
        new Lucene104Codec() {
          @Override
          public PostingsFormat getPostingsFormatForField(String field) {
            return field.equals("id")
                ? super.getPostingsFormatForField(field)
                : completion;
          }
        });
    try (Directory dir = FSDirectory.open(index);
        IndexWriter w = new IndexWriter(dir, cfg)) {
      for (int i = 0; i < 90; i++) {
        String value = values[r.nextInt(values.length)];
        int weight = r.nextInt(4) == 0 ? 0 : r.nextInt(50);
        List<String> ctx = new ArrayList<>();
        for (String c : contexts) if (r.nextInt(3) == 0) ctx.add(c);
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

    StringBuilder q = new StringBuilder();
    try (Directory dir = FSDirectory.open(index);
        DirectoryReader reader = DirectoryReader.open(dir)) {
      if (reader.leaves().size() != 3) throw new IllegalStateException("segments");
      SuggestIndexSearcher s = new SuggestIndexSearcher(reader);
      CompletionAnalyzer sep = new CompletionAnalyzer(new StandardAnalyzer(STOPS));
      CompletionAnalyzer nosep = new CompletionAnalyzer(new StandardAnalyzer(STOPS), false, true);
      String[] prefixes = {
        "b", "be", "the", "the b", "beatles", "new", "new y", "newy", "n", "queen of", "queen o",
        "music of", "caf", "café", "日", "x", "the "
      };
      for (String field : new String[] {"suggest", "nosep", "ctx"}) {
        CompletionAnalyzer a = field.equals("nosep") ? nosep : sep;
        for (String p : prefixes) {
          Term t = new Term(field, p);
          run(q, s, "prefix", field, p, "", new PrefixCompletionQuery(a, t));
          run(q, s, "prefix", field, p, "even", new PrefixCompletionQuery(a, t, EVEN));
          run(q, s, "fuzzy", field, p, "", new FuzzyCompletionQuery(a, t));
          run(
              q, s, "fuzzy", field, p, "2,false,0,1,true",
              new FuzzyCompletionQuery(
                  a, t, null, 2, false, 0, 1, true, Operations.DEFAULT_DETERMINIZE_WORK_LIMIT));
        }
        for (String p : new String[] {"nwe", "beatels", "bech", "quen", "cafe", "日夲", "nigth"}) {
          Term t = new Term(field, p);
          run(q, s, "fuzzy", field, p, "", new FuzzyCompletionQuery(a, t));
          run(
              q, s, "fuzzy", field, p, "2,true,1,3,true|even",
              new FuzzyCompletionQuery(
                  a, t, EVEN, 2, true, 1, 3, true, Operations.DEFAULT_DETERMINIZE_WORK_LIMIT));
        }
        for (String re : new String[] {"b[a-z]*", "ne.", "(the|be)", "n.*t", "", "q.*e", "[^n]"}) {
          Term t = new Term(field, re);
          run(q, s, "regex", field, re, "", new RegexCompletionQuery(t));
          run(q, s, "regex", field, re, "even", new RegexCompletionQuery(t, EVEN));
        }
      }
      run(q, s, "prefix", "suggest", "b", "none", new PrefixCompletionQuery(sep, new Term("suggest", "b"), NONE));
      // contexts
      String[][] contextSets = {
        {}, {"music"}, {"music:2", "movies:3"}, {"m*:1", "music:4"}, {"mus*:2"}, {"books:0"},
        {"*all"}, {"music:2", "*all"}, {"absent"}
      };
      for (String[] cs : contextSets) {
        String opts = String.join(",", cs);
        for (String p : new String[] {"b", "the", "new y", "n", "x"}) {
          ContextQuery cq = new ContextQuery(new PrefixCompletionQuery(sep, new Term("ctx", p)));
          addContexts(cq, cs);
          run(q, s, "context-prefix", "ctx", p, opts, cq);
        }
        for (String p : new String[] {"nwe", "beatels"}) {
          ContextQuery cq = new ContextQuery(new FuzzyCompletionQuery(sep, new Term("ctx", p)));
          addContexts(cq, cs);
          run(q, s, "context-fuzzy", "ctx", p, opts, cq);
        }
        for (String re : new String[] {"b[a-z]*", "n.*"}) {
          ContextQuery cq = new ContextQuery(new RegexCompletionQuery(new Term("ctx", re), EVEN));
          addContexts(cq, cs);
          run(q, s, "context-regex", "ctx", re, opts + "|even", cq);
        }
      }
    }
    Files.writeString(out.resolve("queries.tsv"), q.toString(), StandardCharsets.UTF_8);
    System.out.println("GenCompletionQuery: ok");
  }

  /** {@code name}, {@code name:boost}, {@code prefix*:boost} or {@code *all}. */
  static void addContexts(ContextQuery cq, String[] specs) {
    for (String spec : specs) {
      if (spec.equals("*all")) {
        cq.addAllContexts();
        continue;
      }
      String[] nb = spec.split(":");
      boolean exact = !nb[0].endsWith("*");
      String name = exact ? nb[0] : nb[0].substring(0, nb[0].length() - 1);
      if (nb.length == 1) cq.addContext(name);
      else cq.addContext(name, Float.parseFloat(nb[1]), exact);
    }
  }

  static void run(
      StringBuilder out,
      SuggestIndexSearcher s,
      String kind,
      String field,
      String text,
      String options,
      CompletionQuery query)
      throws IOException {
    for (int n : new int[] {1, 3, 10, 100}) {
      for (boolean dedup : new boolean[] {false, true}) {
        TopSuggestDocs top = s.suggest(query, n, dedup);
        out.append(kind).append('\t').append(field).append('\t').append(text).append('\t');
        out.append(options).append('\t').append(n).append('\t').append(dedup ? 1 : 0).append('\t');
        TopSuggestDocs.SuggestScoreDoc[] hits = top.scoreLookupDocs();
        for (int i = 0; i < hits.length; i++) {
          if (i > 0) out.append('\u001f');
          var h = hits[i];
          out.append(h.key).append('\u001e').append(h.doc).append('\u001e');
          out.append(Float.floatToIntBits(h.score)).append('\u001e');
          out.append(h.context == null ? "" : h.context);
        }
        out.append('\n');
      }
    }
  }
}
