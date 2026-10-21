import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.queryparser.classic.QueryParser;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.uhighlight.UnifiedHighlighter;
import org.apache.lucene.search.vectorhighlight.FastVectorHighlighter;
import org.apache.lucene.search.vectorhighlight.FieldQuery;
import org.apache.lucene.store.FSDirectory;

/**
 * M13 T13.2's pairs (the Rust twin is {@code benchmarks/rust-runner/src/micro_highlight.rs}): the
 * unified highlighter and the fast vector highlighter over the fixture indexes, every templated
 * query of {@code uhighlight-queries.txt} / {@code vhighlight-queries.txt} against every document,
 * default configuration (but {@code uh_phrases}). {@code uh_postings}: field {@code offs}; {@code uh_analysis}: field {@code
 * docs}; {@code uh_tv}: field {@code tv}; {@code uh_phrases}: field {@code offs} without weight
 * matches (phrases through {@code PhraseHelper}); {@code fvh}: the fast vector highlighter on field {@code
 * tv} (30 characters, three fragments). Units: query x document. Output is {@code
 * name\tns_per_unit\tunits}.
 */
public class HighlightMicro {
  static List<Query> templated(String file, String field, Analyzer a) throws Exception {
    QueryParser parser = new QueryParser("offs", a);
    List<Query> out = new ArrayList<>();
    for (String q : Files.readAllLines(Path.of(file), StandardCharsets.UTF_8)) {
      if (q.contains("F:")) out.add(parser.parse(q.replace("F:", field + ":")));
    }
    return out;
  }

  public static void main(String[] args) throws Exception {
    Analyzer a = new StandardAnalyzer();
    try (FSDirectory fs = FSDirectory.open(Path.of("fixtures/data/uhighlight/index"));
        DirectoryReader reader = DirectoryReader.open(fs)) {
      IndexSearcher searcher = new IndexSearcher(reader);
      int[] docs = new int[reader.maxDoc()];
      for (int i = 0; i < docs.length; i++) docs[i] = i;
      UnifiedHighlighter defaults = UnifiedHighlighter.builder(searcher, a).build();
      UnifiedHighlighter classic =
          UnifiedHighlighter.builder(searcher, a).withWeightMatches(false).build();
      String[][] cases = {
        {"uh_postings", "offs"}, {"uh_analysis", "docs"}, {"uh_tv", "tv"}, {"uh_phrases", "offs"}
      };
      for (String[] c : cases) {
        UnifiedHighlighter uh = c[0].equals("uh_phrases") ? classic : defaults;
        List<Query> queries = templated("fixtures/corpus/uhighlight-queries.txt", c[1], a);
        SweepMicro.measure(c[0], () -> {
          for (Query q : queries) {
            try {
              SweepMicro.sink +=
                  uh.highlightFields(new String[] {c[1]}, q, docs, new int[] {2}).size();
            } catch (Exception e) {
              throw new RuntimeException(e);
            }
          }
          return queries.size() * docs.length;
        });
      }
    }
    try (FSDirectory fs = FSDirectory.open(Path.of("fixtures/data/vhighlight/index"));
        DirectoryReader reader = DirectoryReader.open(fs)) {
      List<Query> queries = templated("fixtures/corpus/vhighlight-queries.txt", "tv", a);
      FastVectorHighlighter fvh = new FastVectorHighlighter();
      SweepMicro.measure("fvh", () -> {
        for (Query q : queries) {
          try {
            FieldQuery fq = fvh.getFieldQuery(q, reader);
            for (int doc = 0; doc < reader.maxDoc(); doc++) {
              String[] f = fvh.getBestFragments(fq, reader, doc, "tv", 30, 3);
              SweepMicro.sink += f == null ? 0 : f.length;
            }
          } catch (Exception e) {
            throw new RuntimeException(e);
          }
        }
        return queries.size() * reader.maxDoc();
      });
    }
  }
}
