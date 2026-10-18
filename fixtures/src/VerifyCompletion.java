import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.codecs.PostingsFormat;
import org.apache.lucene.codecs.lucene104.Lucene104Codec;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.CheckIndex;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.SegmentInfos;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.search.suggest.document.Completion104PostingsFormat;
import org.apache.lucene.search.suggest.document.CompletionQuery;
import org.apache.lucene.search.suggest.document.PrefixCompletionQuery;
import org.apache.lucene.search.suggest.document.RegexCompletionQuery;
import org.apache.lucene.search.suggest.document.SuggestField;
import org.apache.lucene.search.suggest.document.SuggestIndexSearcher;
import org.apache.lucene.search.suggest.document.TopSuggestDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;

/**
 * Verifies {@code crates/lucene-search/examples/write_completion_fixture.rs}: in {@code
 * <dir>/flushed} (one segment) and {@code <dir>/merged} (three commits force-merged), {@code
 * suggest} is recorded as {@code Completion104} and {@code body} as {@code Lucene104}, CheckIndex
 * is clean, {@code body}'s terms are found, and every prefix and regex suggestion -- key, doc and
 * score, with and without duplicates skipped -- is the one Lucene gives over the index it builds
 * itself from {@code <dir>/docs.tsv} with the same commits.
 */
public class VerifyCompletion {
  record Doc(String surface, int weight) {}

  public static void main(String[] args) throws Exception {
    Path out = Path.of(args[0]);
    List<Doc> docs = new ArrayList<>();
    for (String line : Files.readAllLines(out.resolve("docs.tsv"), StandardCharsets.UTF_8)) {
      String[] c = line.split("\t");
      docs.add(new Doc(c[0], Integer.parseInt(c[1])));
    }
    verify(out.resolve("flushed"), docs, new int[0]);
    verify(out.resolve("merged"), docs, new int[] {70, 100});
    System.out.println("VerifyCompletion: ok");
  }

  static Path lucenesIndex(List<Doc> docs, int[] commits) throws Exception {
    Path idx = Files.createTempDirectory("verifycompletion");
    IndexWriterConfig cfg = new IndexWriterConfig(new StandardAnalyzer());
    cfg.setUseCompoundFile(false);
    cfg.setCodec(
        new Lucene104Codec() {
          @Override
          public PostingsFormat getPostingsFormatForField(String field) {
            return field.equals("suggest")
                ? new Completion104PostingsFormat()
                : super.getPostingsFormatForField(field);
          }
        });
    try (Directory dir = FSDirectory.open(idx);
        IndexWriter w = new IndexWriter(dir, cfg)) {
      for (int i = 0; i < docs.size(); i++) {
        Document d = new Document();
        d.add(new SuggestField("suggest", docs.get(i).surface(), docs.get(i).weight()));
        d.add(new TextField("body", docs.get(i).surface(), Field.Store.NO));
        w.addDocument(d);
        for (int c : commits) if (c == i + 1) w.commit();
      }
      w.commit();
      if (commits.length > 0) {
        w.forceMerge(1);
        w.commit();
      }
    }
    return idx;
  }

  static String answer(SuggestIndexSearcher s, CompletionQuery q, int n, boolean dedup)
      throws Exception {
    TopSuggestDocs top = s.suggest(q, n, dedup);
    StringBuilder b = new StringBuilder();
    for (TopSuggestDocs.SuggestScoreDoc h : top.scoreLookupDocs()) {
      b.append(h.key).append('/').append(h.doc).append('/').append(h.score).append(' ');
    }
    return b.toString();
  }

  static void verify(Path path, List<Doc> docs, int[] commits) throws Exception {
    try (Directory dir = FSDirectory.open(path)) {
      if (SegmentInfos.readLatestCommit(dir).size() != 1) {
        throw new AssertionError(path + ": expected one segment");
      }
      try (CheckIndex check = new CheckIndex(dir)) {
        if (!check.checkIndex().clean) throw new AssertionError(path + ": CheckIndex not clean");
      }
    }
    Path lucene = lucenesIndex(docs, commits);
    Analyzer analyzer = new StandardAnalyzer();
    try (Directory ours = FSDirectory.open(path);
        Directory theirs = FSDirectory.open(lucene);
        DirectoryReader a = DirectoryReader.open(ours);
        DirectoryReader b = DirectoryReader.open(theirs)) {
      for (LeafReaderContext leaf : a.leaves()) {
        FieldInfo s = leaf.reader().getFieldInfos().fieldInfo("suggest");
        FieldInfo body = leaf.reader().getFieldInfos().fieldInfo("body");
        if (!"Completion104".equals(s.getAttribute("PerFieldPostingsFormat.format"))
            || !"Lucene104".equals(body.getAttribute("PerFieldPostingsFormat.format"))) {
          throw new AssertionError(s.attributes() + " " + body.attributes());
        }
      }
      SuggestIndexSearcher sa = new SuggestIndexSearcher(a);
      SuggestIndexSearcher sb = new SuggestIndexSearcher(b);
      if (sa.count(new TermQuery(new Term("body", "york")))
          != sb.count(new TermQuery(new Term("body", "york")))) {
        throw new AssertionError("body:york count");
      }
      Set<String> prefixes = new LinkedHashSet<>();
      for (Doc d : docs) {
        for (int i = 1; i <= d.surface().length(); i++) prefixes.add(d.surface().substring(0, i));
      }
      prefixes.add("x");
      int checked = 0;
      for (String p : prefixes) {
        for (int n : new int[] {1, 3, 10, 200}) {
          for (boolean dedup : new boolean[] {false, true}) {
            Term t = new Term("suggest", p);
            String want = answer(sb, new PrefixCompletionQuery(analyzer, t), n, dedup);
            String got = answer(sa, new PrefixCompletionQuery(analyzer, t), n, dedup);
            if (!want.equals(got)) {
              throw new AssertionError(
                  path + " prefix " + p + " n=" + n + " dedup=" + dedup
                      + "\n want " + want + "\n got  " + got);
            }
            checked++;
          }
        }
      }
      for (String re : new String[] {"c[a-z]*", "[cd][a-z]", "new.*", "t.*|d.*", "[a-z]"}) {
        for (int n : new int[] {1, 2, 5, 200}) {
          for (boolean dedup : new boolean[] {false, true}) {
            String want = answer(sb, new RegexCompletionQuery(new Term("suggest", re)), n, dedup);
            String got = answer(sa, new RegexCompletionQuery(new Term("suggest", re)), n, dedup);
            if (!want.equals(got)) {
              throw new AssertionError(
                  path + " regex " + re + " n=" + n + " dedup=" + dedup
                      + "\n want " + want + "\n got  " + got);
            }
            checked++;
          }
        }
      }
      if (checked < 200) throw new AssertionError("checked " + checked);
    }
  }
}
