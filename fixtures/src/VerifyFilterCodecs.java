import org.apache.lucene.codecs.Codec;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.NumericDocValuesField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.CheckIndex;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.NumericDocValues;
import org.apache.lucene.index.SegmentCommitInfo;
import org.apache.lucene.index.SegmentInfos;
import org.apache.lucene.index.StoredFields;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.Bits;

import java.nio.file.Path;
import java.util.function.IntPredicate;

/**
 * Verifies {@code crates/lucene-test-framework/examples/write_filter_codecs_fixture.rs}:
 * for each of the test framework's {@code CheapBastard}, {@code ConfigurableMCodec} and
 * {@code Asserting},
 * {@code <dir>/<codec>/flushed} (three segments, {@code d5} and {@code d150} deleted) and
 * {@code <dir>/<codec>/merged} (the same force-merged into one) were written by this port's
 * IndexWriter with the codec installed. Lucene must find the codec's name on every segment,
 * read it through that codec ({@code Codec.forName}, from the test framework's SPI), find
 * the direct format's files without a codec suffix and its fields without per-field
 * attributes ({@code Asserting}: every field's postings, doc values and vectors in one
 * {@code Asserting_0} instance of each), see every document, term, doc value and vector, and
 * pass CheckIndex.
 *
 * <p>Then Lucene appends documents 300..399 to each {@code merged} index with its own
 * IndexWriter under the same codec (a compound segment, no merge) and checks again, leaving
 * the index for {@code filter_codecs_write_path::java_written_segments} to read and merge.
 */
public class VerifyFilterCodecs {
  static final String[] CODECS = {"CheapBastard", "ConfigurableMCodec", "Asserting"};
  static final String[] PER_FIELD_KEYS = {
    "PerFieldPostingsFormat.format", "PerFieldDocValuesFormat.format", "PerFieldKnnVectorsFormat.format"
  };
  static final int PER_SEGMENT = 100;

  public static void main(String[] args) throws Exception {
    Path root = Path.of(args[0]);
    for (String codec : CODECS) {
      verify(root.resolve(codec).resolve("flushed"), codec, 3, 300);
      Path merged = root.resolve(codec).resolve("merged");
      verify(merged, codec, 1, 300);
      append(merged, codec);
      verify(merged, codec, 2, 400);
    }
    System.out.println("VerifyFilterCodecs: ok");
  }

  /** {@code write_filter_codecs_fixture::vector}. */
  static float[] vector(int i) {
    return new float[] {i % 10, i % 3, 1f, (i % 7) / 7f};
  }

  /** {@code write_filter_codecs_fixture::doc}. */
  static Document doc(int i) {
    Document d = new Document();
    d.add(new StringField("id", "d" + i, Field.Store.YES));
    d.add(new TextField("body", "w" + (i % 5) + " x" + (i % 7) + " common", Field.Store.NO));
    d.add(new KnnFloatVectorField("v", vector(i)));
    d.add(new NumericDocValuesField("n", i));
    return d;
  }

  static boolean live(int i) {
    return i != 5 && i != 150;
  }

  static int count(int total, IntPredicate p) {
    int n = 0;
    for (int i = 0; i < total; i++) {
      if (live(i) && p.test(i)) n++;
    }
    return n;
  }

  static void append(Path path, String codec) throws Exception {
    try (Directory dir = FSDirectory.open(path)) {
      IndexWriterConfig config = new IndexWriterConfig();
      config.setCodec(Codec.forName(codec));
      config.setMergePolicy(NoMergePolicy.INSTANCE);
      try (IndexWriter w = new IndexWriter(dir, config)) {
        for (int i = 3 * PER_SEGMENT; i < 4 * PER_SEGMENT; i++) {
          w.addDocument(doc(i));
        }
        w.commit();
      }
    }
  }

  static void verify(Path path, String codec, int segments, int total) throws Exception {
    try (Directory dir = FSDirectory.open(path)) {
      SegmentInfos sis = SegmentInfos.readLatestCommit(dir);
      if (sis.size() != segments) {
        throw new AssertionError(path + ": expected " + segments + " segments, got " + sis.size());
      }
      for (SegmentCommitInfo sci : sis) {
        if (!codec.equals(sci.info.getCodec().getName())) {
          throw new AssertionError(path + ": segment " + sci.info.name + " has codec "
              + sci.info.getCodec().getName());
        }
        String direct = switch (codec) {
          case "CheapBastard" -> ".tim";
          case "ConfigurableMCodec" -> ".vemf";
          default -> "_Asserting_0.tim";
        };
        String name = sci.info.name + direct;
        if (!sci.files().contains(name) && !sci.info.getUseCompoundFile()) {
          throw new AssertionError(path + ": no " + name + " in " + sci.files());
        }
      }
      try (DirectoryReader reader = DirectoryReader.open(dir)) {
        if (reader.numDocs() != count(total, i -> true)) {
          throw new AssertionError(path + ": numDocs " + reader.numDocs());
        }
        IndexSearcher searcher = new IndexSearcher(reader);
        for (int k = 0; k < 5; k++) {
          final int kk = k;
          int got = searcher.count(new TermQuery(new Term("body", "w" + k)));
          if (got != count(total, i -> i % 5 == kk)) {
            throw new AssertionError(path + ": body:w" + k + " count " + got);
          }
        }
        for (int k = 0; k < 7; k++) {
          final int kk = k;
          int got = searcher.count(new TermQuery(new Term("body", "x" + k)));
          if (got != count(total, i -> i % 7 == kk)) {
            throw new AssertionError(path + ": body:x" + k + " count " + got);
          }
        }
        if (searcher.search(new KnnFloatVectorQuery("v", vector(7), 10), 10).scoreDocs.length
            != 10) {
          throw new AssertionError(path + ": knn hits");
        }
        String directKey = codec.equals("CheapBastard")
            ? "PerFieldPostingsFormat.format"
            : "PerFieldKnnVectorsFormat.format";
        for (LeafReaderContext ctx : reader.leaves()) {
          LeafReader leaf = ctx.reader();
          for (FieldInfo fi : leaf.getFieldInfos()) {
            if (codec.equals("Asserting")) {
              for (String key : PER_FIELD_KEYS) {
                String format = fi.getAttribute(key);
                if (format != null && !format.equals("Asserting")) {
                  throw new AssertionError(path + ": " + fi.name + " has " + fi.attributes());
                }
              }
            } else if (fi.getAttribute(directKey) != null) {
              throw new AssertionError(path + ": " + fi.name + " has " + fi.attributes());
            }
          }
          if (codec.equals("Asserting")) {
            FieldInfo body = leaf.getFieldInfos().fieldInfo("body");
            FieldInfo n = leaf.getFieldInfos().fieldInfo("n");
            FieldInfo v = leaf.getFieldInfos().fieldInfo("v");
            if (!"Asserting".equals(body.getAttribute(PER_FIELD_KEYS[0]))
                || !"Asserting".equals(n.getAttribute(PER_FIELD_KEYS[1]))
                || !"Asserting".equals(v.getAttribute(PER_FIELD_KEYS[2]))) {
              throw new AssertionError(path + ": per-field formats are not Asserting");
            }
          }
          if (leaf.getFloatVectorValues("v").size() != leaf.maxDoc()) {
            throw new AssertionError(path + ": vector count");
          }
          StoredFields stored = leaf.storedFields();
          NumericDocValues n = leaf.getNumericDocValues("n");
          Bits liveDocs = leaf.getLiveDocs();
          for (int d = 0; d < leaf.maxDoc(); d++) {
            int i = Integer.parseInt(stored.document(d).get("id").substring(1));
            if (!n.advanceExact(d) || n.longValue() != i) {
              throw new AssertionError(path + ": d" + i + " doc value");
            }
            if ((liveDocs == null || liveDocs.get(d)) != live(i)) {
              throw new AssertionError(path + ": d" + i + " liveness");
            }
          }
        }
      }
      try (CheckIndex checker = new CheckIndex(dir)) {
        CheckIndex.Status status = checker.checkIndex();
        if (!status.clean) {
          throw new AssertionError(path + ": CheckIndex is not clean");
        }
      }
    }
  }
}
