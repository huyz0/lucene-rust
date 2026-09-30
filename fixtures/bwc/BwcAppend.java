import java.nio.file.Path;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.IntPoint;
import org.apache.lucene.document.NumericDocValuesField;
import org.apache.lucene.document.StoredField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;

/**
 * M8 T8.4: turns a {@code fixtures/data/bwc/<version>} copy into a mixed-version index, the shape
 * an upgraded OpenSearch shard has -- the old release's segments plus {@code segments} new ones
 * Lucene 10.5.0 flushes on top (codec {@code Lucene104}), each of three documents, one of them
 * deleting an old document -- so an ordinary (policy-driven) merge in this port has old and new
 * sources at once.
 *
 * <pre>java BwcAppend &lt;index-dir&gt; &lt;segments&gt;</pre>
 */
public class BwcAppend {
  public static void main(String[] args) throws Exception {
    int segments = Integer.parseInt(args[1]);
    IndexWriterConfig cfg = new IndexWriterConfig();
    cfg.setUseCompoundFile(false);
    cfg.setMergePolicy(NoMergePolicy.INSTANCE);
    try (Directory dir = FSDirectory.open(Path.of(args[0]));
        IndexWriter w = new IndexWriter(dir, cfg)) {
      for (int s = 0; s < segments; s++) {
        for (int i = 0; i < 3; i++) {
          Document d = new Document();
          d.add(new StringField("id", "new-" + s + "-" + i, Field.Store.YES));
          d.add(new Field("body", "alpha beta new" + s + " gamma", TextField.TYPE_STORED));
          d.add(new NumericDocValuesField("num", 1000L * s + i));
          d.add(new IntPoint("ipt", s * 10 + i));
          d.add(new StoredField("ipt", s * 10 + i));
          w.addDocument(d);
        }
        w.deleteDocuments(new org.apache.lucene.index.Term("id", Integer.toString(1 + s * 5)));
        w.commit();
      }
    }
  }
}
