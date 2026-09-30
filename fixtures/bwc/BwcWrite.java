import java.io.IOException;
import java.lang.reflect.Constructor;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Random;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.TokenFilter;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.Tokenizer;
import org.apache.lucene.analysis.standard.StandardTokenizer;
import org.apache.lucene.analysis.tokenattributes.PayloadAttribute;
import org.apache.lucene.document.BinaryDocValuesField;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.FieldType;
import org.apache.lucene.document.IntPoint;
import org.apache.lucene.document.LongPoint;
import org.apache.lucene.document.NumericDocValuesField;
import org.apache.lucene.document.SortedDocValuesField;
import org.apache.lucene.document.SortedNumericDocValuesField;
import org.apache.lucene.document.SortedSetDocValuesField;
import org.apache.lucene.document.StoredField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.IndexableField;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.BytesRef;
import org.apache.lucene.util.Version;

/**
 * Writes one index with whichever Lucene version is on the classpath (9.0.0 through 10.4.0), for
 * the backward-codecs differential tests (M8). Only APIs every one of those versions has are used
 * directly; vector fields, whose classes were renamed in 9.5, are built by reflection.
 *
 * <p>Two segments (no merging), a few deletions, and every field kind: text with positions,
 * offsets and payloads, term vectors, all five doc-values types, norms, 1-D and 2-D points, float
 * vectors, and byte vectors where the version has them. {@code BwcDump} (on 10.5.0 with
 * backward-codecs) records what the index contains.
 */
public class BwcWrite {
  static final String[] WORDS = {
    "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa",
    "lambda", "mu", "nu", "xi", "omicron", "pi", "rho", "sigma", "tau", "upsilon"
  };

  /** Adds a payload (the token's position modulo 7, as one byte) to every other token. */
  static final class PayloadFilter extends TokenFilter {
    final PayloadAttribute payload = addAttribute(PayloadAttribute.class);
    int n;

    PayloadFilter(TokenStream in) {
      super(in);
    }

    @Override
    public boolean incrementToken() throws IOException {
      if (!input.incrementToken()) return false;
      payload.setPayload(n % 2 == 0 ? new BytesRef(new byte[] {(byte) (n % 7)}) : null);
      n++;
      return true;
    }

    @Override
    public void reset() throws IOException {
      super.reset();
      n = 0;
    }
  }

  static Analyzer analyzer() {
    return new Analyzer(Analyzer.PER_FIELD_REUSE_STRATEGY) {
      @Override
      protected TokenStreamComponents createComponents(String field) {
        Tokenizer t = new StandardTokenizer();
        return new TokenStreamComponents(t, field.equals("pay") ? new PayloadFilter(t) : t);
      }
    };
  }

  static String text(Random r, int n) {
    StringBuilder sb = new StringBuilder();
    for (int i = 0; i < n; i++) {
      if (i > 0) sb.append(' ');
      // Skewed: low words are frequent, so some terms fill whole postings blocks.
      sb.append(WORDS[(int) Math.min(WORDS.length - 1, Math.abs(r.nextGaussian()) * 5)]);
    }
    return sb.toString();
  }

  static IndexableField vectorField(String name, Object value, VectorSimilarityFunction sim)
      throws Exception {
    boolean isFloat = value instanceof float[];
    for (String cls :
        isFloat
            ? new String[] {"org.apache.lucene.document.KnnFloatVectorField", "org.apache.lucene.document.KnnVectorField"}
            : new String[] {"org.apache.lucene.document.KnnByteVectorField"}) {
      try {
        Class<?> c = Class.forName(cls);
        Constructor<?> k =
            c.getConstructor(String.class, isFloat ? float[].class : byte[].class, VectorSimilarityFunction.class);
        return (IndexableField) k.newInstance(name, value, sim);
      } catch (ClassNotFoundException | NoSuchMethodException e) {
        // try the next name
      }
    }
    return null;
  }

  public static void main(String[] args) throws Exception {
    Path out = Path.of(args[0]);
    Files.createDirectories(out);
    Random r = new Random(20260930L);
    FieldType withVectors = new FieldType(TextField.TYPE_STORED);
    withVectors.setStoreTermVectors(true);
    withVectors.setStoreTermVectorPositions(true);
    withVectors.setStoreTermVectorOffsets(true);
    withVectors.freeze();
    FieldType offsets = new FieldType(TextField.TYPE_NOT_STORED);
    offsets.setIndexOptions(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS_AND_OFFSETS);
    offsets.freeze();
    FieldType docsOnly = new FieldType(TextField.TYPE_NOT_STORED);
    docsOnly.setIndexOptions(IndexOptions.DOCS);
    docsOnly.setOmitNorms(true);
    docsOnly.freeze();
    FieldType freqsOnly = new FieldType(TextField.TYPE_NOT_STORED);
    freqsOnly.setIndexOptions(IndexOptions.DOCS_AND_FREQS);
    freqsOnly.freeze();

    try (Directory dir = FSDirectory.open(out)) {
      IndexWriterConfig cfg = new IndexWriterConfig(analyzer());
      cfg.setUseCompoundFile(false);
      cfg.setMergePolicy(NoMergePolicy.INSTANCE);
      cfg.setMaxBufferedDocs(IndexWriterConfig.DISABLE_AUTO_FLUSH);
      cfg.setRAMBufferSizeMB(256);
      try (IndexWriter w = new IndexWriter(dir, cfg)) {
        int docs = 0;
        for (int seg = 0; seg < 2; seg++) {
          // The first segment is large enough for full postings blocks
          // (128 docs in 9.x/10.x's formats) and multi-level skip data.
          int n = seg == 0 ? 3000 : 400;
          for (int i = 0; i < n; i++, docs++) {
            Document d = new Document();
            d.add(new StringField("id", Integer.toString(docs), Field.Store.YES));
            d.add(new Field("body", text(r, 3 + r.nextInt(40)), TextField.TYPE_STORED));
            if (i % 3 == 0) d.add(new Field("title", text(r, 1 + r.nextInt(6)), withVectors));
            d.add(new Field("off", text(r, 2 + r.nextInt(10)), offsets));
            d.add(new Field("pay", text(r, 2 + r.nextInt(10)), TextField.TYPE_NOT_STORED));
            d.add(new Field("docs", text(r, 1 + r.nextInt(5)), docsOnly));
            d.add(new Field("freqs", text(r, 1 + r.nextInt(8)), freqsOnly));
            d.add(new NumericDocValuesField("num", r.nextInt(5) == 0 ? Long.MIN_VALUE + i : r.nextInt(1_000_000) - 500_000L));
            d.add(new SortedDocValuesField("sorted", new BytesRef(WORDS[r.nextInt(WORDS.length)])));
            for (int k = 0, m = r.nextInt(4); k < m; k++) {
              d.add(new SortedSetDocValuesField("sset", new BytesRef(WORDS[r.nextInt(WORDS.length)])));
              d.add(new SortedNumericDocValuesField("snum", r.nextInt(100) - 50));
            }
            if (i % 5 != 0) d.add(new BinaryDocValuesField("bin", new BytesRef(text(r, 1 + r.nextInt(3)))));
            int iv = r.nextInt(20000) - 10000;
            d.add(new IntPoint("ipt", iv));
            d.add(new StoredField("ipt", iv));
            d.add(new LongPoint("lpt", r.nextLong() >> 8));
            d.add(new IntPoint("pt2", r.nextInt(1000), r.nextInt(1000)));
            if (i % 2 == 0) {
              float[] v = new float[8];
              for (int k = 0; k < v.length; k++) v[k] = r.nextFloat() * 2 - 1;
              IndexableField f = vectorField("fvec", v, VectorSimilarityFunction.EUCLIDEAN);
              if (f != null) d.add(f);
            }
            if (i % 2 == 1) {
              byte[] v = new byte[8];
              for (int k = 0; k < v.length; k++) v[k] = (byte) (r.nextInt(256) - 128);
              IndexableField f = vectorField("bvec", v, VectorSimilarityFunction.DOT_PRODUCT);
              if (f != null) d.add(f);
            }
            w.addDocument(d);
          }
          w.flush();
        }
        for (int id = 0; id < docs; id += 17) {
          w.deleteDocuments(new Term("id", Integer.toString(id)));
        }
        w.commit();
      }
    }
    Files.writeString(out.resolve("written_by.txt"), Version.LATEST.toString() + "\n");
  }
}
