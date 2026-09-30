import java.nio.file.Files;
import java.nio.file.Path;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.FieldType;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.Version;

/**
 * M8's skip-data fixture: one segment of 20,000 documents whose terms are long enough to need
 * every level of a retired postings format's skip data -- the trailing multi-level skip list of
 * {@code Lucene90}/{@code Lucene99} (levels at 128, 1,024 and 8,192 documents) and the inline
 * level-1 entries of {@code Lucene912}/{@code Lucene101} (every 4,096). {@code BwcWrite}'s
 * 3,000-document segments reach neither.
 *
 * <p>Compiled and run against one release's own {@code lucene-core} jar by {@code
 * scripts/gen-bwc-fixtures.sh --big <version>}, like {@code BwcWrite}. Deterministic content:
 *
 * <ul>
 *   <li>{@code f} (positions): {@code all} in every document, {@code all} repeated {@code 1 + i %
 *       7} times so frequencies (and impacts) vary; {@code half} in even documents; {@code
 *       third} in every third; {@code rare} in every 97th; {@code run} in documents 5,000-12,999
 *       only (a dense run in the middle); {@code peak} in every document, 30 times in one of
 *       every 1,500 and once elsewhere (impacts that differ between levels); and a filler word
 *       per document for varied lengths.
 *   <li>{@code d} (docs only): the same {@code all}/{@code half}/{@code run} terms.
 * </ul>
 */
public class BwcBig {
  public static void main(String[] args) throws Exception {
    Path out = Path.of(args[0]);
    Files.createDirectories(out);
    FieldType docsOnly = new FieldType(TextField.TYPE_NOT_STORED);
    docsOnly.setIndexOptions(IndexOptions.DOCS);
    docsOnly.setOmitNorms(true);
    docsOnly.freeze();
    try (Directory dir = FSDirectory.open(out)) {
      IndexWriterConfig cfg = new IndexWriterConfig(new StandardAnalyzer());
      cfg.setUseCompoundFile(false);
      cfg.setMergePolicy(NoMergePolicy.INSTANCE);
      cfg.setMaxBufferedDocs(IndexWriterConfig.DISABLE_AUTO_FLUSH);
      cfg.setRAMBufferSizeMB(256);
      try (IndexWriter w = new IndexWriter(dir, cfg)) {
        for (int i = 0; i < 20000; i++) {
          StringBuilder f = new StringBuilder();
          StringBuilder d = new StringBuilder("all");
          for (int k = 0; k <= i % 7; k++) f.append("all ");
          // One document in 1,500 with a far higher frequency: most blocks'
          // level-0 impacts stay at 1 while the level-1 span holding the
          // peak must report 30, so a level-1 bound taken from the wrong
          // level cannot pass for a sound one.
          int peak = i % 1500 == 777 ? 30 : 1;
          for (int k = 0; k < peak; k++) f.append("peak ");
          if (i % 2 == 0) {
            f.append("half ");
            d.append(" half");
          }
          if (i % 3 == 0) f.append("third ");
          if (i % 97 == 0) f.append("rare ");
          if (i >= 5000 && i < 13000) {
            f.append("run ");
            d.append(" run");
          }
          for (int k = 0; k < i % 11; k++) f.append("w").append(k).append(' ');
          Document doc = new Document();
          doc.add(new Field("f", f.toString(), TextField.TYPE_NOT_STORED));
          doc.add(new Field("d", d.toString(), docsOnly));
          w.addDocument(doc);
        }
        w.commit();
      }
    }
    Files.writeString(out.resolve("written_by.txt"), Version.LATEST.toString() + "\n");
  }
}
