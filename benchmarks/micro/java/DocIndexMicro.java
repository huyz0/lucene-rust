import java.util.ArrayList;
import java.util.List;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.LongPoint;
import org.apache.lucene.document.NumericDocValuesField;
import org.apache.lucene.document.SortedDocValuesField;
import org.apache.lucene.document.SortedSetDocValuesField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.util.BytesRef;

/**
 * The document API's indexing cost (the Rust twin is {@code
 * benchmarks/rust-runner/src/micro_doc_index.rs}): each case builds a fresh in-memory index of
 * 5,000 documents through {@code IndexWriter.addDocument}, commits and closes it. Units: documents.
 *
 * <ul>
 *   <li>{@code doc_strings}: {@code SpellChecker}'s document shape -- the word, stored, and its
 *       start/inner/end 3- and 4-grams, each an unstored {@code StringField};
 *   <li>{@code doc_text}: a stored {@code StringField} id and a 30-word {@code TextField} through
 *       {@code StandardAnalyzer};
 *   <li>{@code doc_values}: a {@code NumericDocValuesField}, a {@code SortedDocValuesField}, three
 *       {@code SortedSetDocValuesField} values and a {@code LongPoint}.
 * </ul>
 *
 * Output is {@code name	ns_per_unit	units}.
 */
public class DocIndexMicro {
  static final int DOCS = 5000;
  static long state = 7;

  /** {@code SpellMicro.next}'s LCG, on its own state. */
  static long next(long bound) {
    state = state * 6364136223846793005L + 1442695040888963407L;
    return Long.remainderUnsigned(state >>> 33, bound);
  }

  static List<Document> stringDocs(List<String> words) {
    List<Document> out = new ArrayList<>();
    for (int i = 0; i < DOCS; i++) {
      String w = words.get(i);
      Document d = new Document();
      d.add(new StringField("word", w, Field.Store.YES));
      for (int n = 3; n <= 4; n++) {
        if (w.length() < n) continue;
        d.add(new StringField("start" + n, w.substring(0, n), Field.Store.NO));
        for (int k = 0; k + n <= w.length(); k++) {
          d.add(new StringField("gram" + n, w.substring(k, k + n), Field.Store.NO));
        }
        d.add(new StringField("end" + n, w.substring(w.length() - n), Field.Store.NO));
      }
      out.add(d);
    }
    return out;
  }

  static List<Document> textDocs(List<String> words) {
    List<Document> out = new ArrayList<>();
    for (int i = 0; i < DOCS; i++) {
      StringBuilder body = new StringBuilder();
      for (int k = 0; k < 30; k++) {
        if (k > 0) body.append(' ');
        body.append(words.get((int) next(2000)));
      }
      Document d = new Document();
      d.add(new StringField("id", Integer.toString(i), Field.Store.YES));
      d.add(new TextField("body", body.toString(), Field.Store.NO));
      out.add(d);
    }
    return out;
  }

  static List<Document> valueDocs(List<String> words) {
    List<Document> out = new ArrayList<>();
    for (int i = 0; i < DOCS; i++) {
      Document d = new Document();
      d.add(new NumericDocValuesField("n", next(1_000_000)));
      d.add(new SortedDocValuesField("s", new BytesRef(words.get((int) next(500)))));
      for (int k = 0; k < 3; k++) {
        d.add(new SortedSetDocValuesField("tags", new BytesRef(words.get((int) next(100)))));
      }
      d.add(new LongPoint("p", next(1_000_000_000)));
      out.add(d);
    }
    return out;
  }

  static long index(List<Document> docs) throws java.io.IOException {
    ByteBuffersDirectory dir = new ByteBuffersDirectory();
    try (IndexWriter w = new IndexWriter(dir, new IndexWriterConfig(new StandardAnalyzer()))) {
      for (Document d : docs) w.addDocument(d);
      w.commit();
    }
    SweepMicro.sink += dir.listAll().length;
    return docs.size();
  }

  public static void main(String[] args) throws Exception {
    List<String> words = SpellMicro.words();
    List<Document> strings = stringDocs(words);
    List<Document> text = textDocs(words);
    List<Document> values = valueDocs(words);
    SweepMicro.measure("doc_strings", () -> index(strings));
    SweepMicro.measure("doc_text", () -> index(text));
    SweepMicro.measure("doc_values", () -> index(values));
  }
}
