import java.io.IOException;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Comparator;
import java.util.Random;
import org.apache.lucene.analysis.core.WhitespaceAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.StoredField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.misc.HighFreqTerms;
import org.apache.lucene.misc.TermStats;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;

/**
 * M14 T14.3 `misc`: the index `HighFreqTerms` reads, and what
 * {@link HighFreqTerms#getHighFreqTerms} returns for it under each
 * configuration. Writes {@code misc/high_freq} (an index over three commits,
 * so three segments) and {@code misc/high_freq.tsv}:
 *
 * <pre>config  field  numTerms  term  totalTF  docFreq</pre>
 *
 * Read by {@code crates/lucene-search/tests/misc_fixtures.rs}.
 */
public class GenMisc {

  private static final String[] VOCAB = {
    "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa",
    "lambda", "mu", "nu", "xi", "omicron", "pi", "rho", "sigma", "tau", "upsilon", "phi", "chi",
    "psi", "omega", "index", "search", "term", "field", "doc", "score", "query", "merge", "segment",
    "commit", "lucene", "rust", "port", "vector", "bytes", "codec", "store", "reader", "writer",
    "analyzer", "token", "filter", "stored", "value", "length", "norm", "freq", "posting", "skip",
    "block", "tree", "fst", "prefix", "suffix", "range", "point"
  };

  public static void main(String[] args) throws Exception {
    Path out = Path.of(args[0]).resolve("misc");
    Files.createDirectories(out);
    Path index = out.resolve("high_freq");
    Random rnd = new Random(1464L);
    try (Directory dir = FSDirectory.open(index);
        IndexWriter w =
            new IndexWriter(dir, new IndexWriterConfig(new WhitespaceAnalyzer()))) {
      // Three commits: three segments, so the terms are merged across leaves.
      for (int commit = 0; commit < 3; commit++) {
        for (int d = 0; d < 40; d++) {
          Document doc = new Document();
          doc.add(new StoredField("id", commit * 100 + d));
          doc.add(new TextField("body", text(rnd, 6 + rnd.nextInt(20)), Field.Store.NO));
          if (d % 3 != 0) {
            doc.add(new TextField("title", text(rnd, 2 + rnd.nextInt(4)), Field.Store.NO));
          }
          w.addDocument(doc);
        }
        w.commit();
      }
    }

    try (Directory dir = FSDirectory.open(index);
        PrintWriter tsv =
            new PrintWriter(
                Files.newBufferedWriter(out.resolve("high_freq.tsv"), StandardCharsets.UTF_8))) {
      IndexReader reader = DirectoryReader.open(dir);
      // config, field (or "-"), numTerms, comparator
      Object[][] cases = {
        {"doc-freq", null, 100, new HighFreqTerms.DocFreqComparator()},
        {"total-tf", null, 100, new HighFreqTerms.TotalTermFreqComparator()},
        {"doc-freq", "body", 5, new HighFreqTerms.DocFreqComparator()},
        {"total-tf", "title", 3, new HighFreqTerms.TotalTermFreqComparator()},
        {"doc-freq", null, 1000, new HighFreqTerms.DocFreqComparator()},
        {"total-tf", "body", 1, new HighFreqTerms.TotalTermFreqComparator()},
      };
      for (Object[] c : cases) {
        String field = (String) c[1];
        TermStats[] terms =
            HighFreqTerms.getHighFreqTerms(
                reader, (Integer) c[2], field, (Comparator<TermStats>) c[3]);
        for (TermStats t : terms) {
          tsv.printf(
              "%s\t%s\t%d\t%s\t%d\t%d%n",
              c[0],
              field == null ? "-" : field,
              (Integer) c[2],
              t.termtext.utf8ToString(),
              t.totalTermFreq,
              t.docFreq);
        }
      }
      reader.close();
    }
  }

  /** Zipf-like: the earlier words of the vocabulary come up more often. */
  private static String text(Random rnd, int words) {
    StringBuilder sb = new StringBuilder();
    for (int i = 0; i < words; i++) {
      int a = rnd.nextInt(VOCAB.length);
      int b = rnd.nextInt(VOCAB.length);
      if (i > 0) {
        sb.append(' ');
      }
      sb.append(VOCAB[Math.min(a, b)]);
    }
    return sb.toString();
  }
}
