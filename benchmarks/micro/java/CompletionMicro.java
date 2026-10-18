import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import org.apache.lucene.analysis.CharArraySet;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.suggest.document.CompletionAnalyzer;
import org.apache.lucene.search.suggest.document.CompletionQuery;
import org.apache.lucene.search.suggest.document.ContextQuery;
import org.apache.lucene.search.suggest.document.FuzzyCompletionQuery;
import org.apache.lucene.search.suggest.document.PrefixCompletionQuery;
import org.apache.lucene.search.suggest.document.RegexCompletionQuery;
import org.apache.lucene.search.suggest.document.SuggestIndexSearcher;
import org.apache.lucene.store.FSDirectory;

/**
 * M13 T13.3d's pairs (the Rust twin is {@code benchmarks/rust-runner/src/micro_completion.rs}):
 * {@code SuggestIndexSearcher.suggest} over {@code GenCompletionQuery}'s index (three segments,
 * one with deletions), top 10 with duplicates skipped. {@code completion_prefix}: 12 prefixes on
 * {@code suggest}; {@code completion_fuzzy}: 6 misspellings, default fuzzy options; {@code
 * completion_regex}: 6 regular expressions; {@code completion_context}: 6 prefixes on {@code ctx}
 * boosted by two contexts. Units: queries. Output is {@code name\tns_per_unit\tunits}.
 */
public class CompletionMicro {
  static final String[] PREFIXES = {
    "b", "be", "the b", "beatles", "new", "new y", "n", "queen o", "music of", "caf", "日", "x"
  };
  static final String[] MISSPELT = {"nwe", "beatels", "bech", "quen", "cafe", "nigth"};
  static final String[] REGEXES = {"b[a-z]*", "ne.", "(the|be)", "n.*t", "q.*e", "[^n]"};
  static final String[] CONTEXT_PREFIXES = {"b", "the", "new y", "n", "q", "x"};

  static void run(String name, SuggestIndexSearcher s, List<CompletionQuery> queries)
      throws Exception {
    SweepMicro.measure(name, () -> {
      for (CompletionQuery q : queries) {
        try {
          SweepMicro.sink += s.suggest(q, 10, true).scoreLookupDocs().length;
        } catch (Exception e) {
          throw new RuntimeException(e);
        }
      }
      return queries.size();
    });
  }

  public static void main(String[] args) throws Exception {
    CharArraySet stops = new CharArraySet(List.of("the", "of"), false);
    CompletionAnalyzer a = new CompletionAnalyzer(new StandardAnalyzer(stops));
    try (FSDirectory fs = FSDirectory.open(Path.of("fixtures/data/completion_query/index"));
        DirectoryReader reader = DirectoryReader.open(fs)) {
      SuggestIndexSearcher s = new SuggestIndexSearcher(reader);
      List<CompletionQuery> prefix = new ArrayList<>();
      for (String p : PREFIXES) prefix.add(new PrefixCompletionQuery(a, new Term("suggest", p)));
      List<CompletionQuery> fuzzy = new ArrayList<>();
      for (String p : MISSPELT) fuzzy.add(new FuzzyCompletionQuery(a, new Term("suggest", p)));
      List<CompletionQuery> regex = new ArrayList<>();
      for (String r : REGEXES) regex.add(new RegexCompletionQuery(new Term("suggest", r)));
      List<CompletionQuery> context = new ArrayList<>();
      for (String p : CONTEXT_PREFIXES) {
        ContextQuery q = new ContextQuery(new PrefixCompletionQuery(a, new Term("ctx", p)));
        q.addContext("music", 2);
        q.addContext("mov", 3, false);
        context.add(q);
      }
      run("completion_prefix", s, prefix);
      run("completion_fuzzy", s, fuzzy);
      run("completion_regex", s, regex);
      run("completion_context", s, context);
    }
  }
}
