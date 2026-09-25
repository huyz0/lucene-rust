import java.nio.file.*;
import java.util.*;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.*;
import org.apache.lucene.index.*;
import org.apache.lucene.store.*;

/** The e2e verify corpus's shape (verify_opensearch.py load()), in process. */
public class GenRestCorpus {
  static final String[] W = ("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho "
      + "sigma tau upsilon phi chi psi omega").split(" ");
  static String word(Random r) { return W[Math.min(W.length - 1, (int) (Math.pow(r.nextDouble(), 2.2) * W.length))]; }
  static String words(Random r, int lo, int hi) {
    int n = lo + r.nextInt(hi - lo + 1); StringBuilder b = new StringBuilder();
    for (int i = 0; i < n; i++) { if (i > 0) b.append(' '); b.append(word(r)); }
    return b.toString();
  }
  public static void main(String[] a) throws Exception {
    int docs = Integer.parseInt(a[1]); boolean merge = a[2].equals("merged");
    Random r = new Random(42);
    IndexWriterConfig c = new IndexWriterConfig(new StandardAnalyzer());
    c.setUseCompoundFile(false); c.setRAMBufferSizeMB(64);
    try (Directory d = FSDirectory.open(Paths.get(a[0])); IndexWriter w = new IndexWriter(d, c)) {
      for (int i = 0; i < docs; i++) {
        Document doc = new Document();
        doc.add(new TextField("body", words(r, 1, 40), Field.Store.NO));
        doc.add(new TextField("title", words(r, 1, 5), Field.Store.NO));
        doc.add(new StringField("tag", word(r), Field.Store.NO));
        w.addDocument(doc);
        if (!merge && i % (docs / 8) == docs / 8 - 1) w.commit();
      }
      if (merge) w.forceMerge(1);
      w.commit();
    }
  }
}
