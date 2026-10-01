import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.CheckIndex;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.PostingsEnum;
import org.apache.lucene.index.SegmentInfos;
import org.apache.lucene.index.Terms;
import org.apache.lucene.index.TermsEnum;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnByteVectorQuery;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.BytesRef;

import java.nio.file.Path;
import java.util.Arrays;
import java.util.Map;

/**
 * Verifies {@code crates/lucene-index/examples/write_per_field_knn_vectors_fixture.rs} (the
 * documents and routing of {@code GenPerFieldKnnVectors}): in {@code <dir>/flushed} (two segments)
 * and {@code <dir>/merged} (one, {@code d5} and {@code d302} deleted) every vector field records
 * the {@code PerFieldKnnVectorsFormat} format it was routed to -- {@code v_small} and {@code
 * v_bytes} on one {@code Lucene99HnswVectorsFormat} instance, {@code v_default} on another -- and
 * Lucene's {@code FieldsReader} reads every document's vector of every field back from that
 * instance's files, a KNN query over each field returns ten hits, and {@code CheckIndex} is clean.
 */
public class VerifyPerFieldKnnVectors {
  static final int PER_SEGMENT = 300;
  static final int DIM = 16;

  static long value(int i, int k) {
    long x = (i + 1) * 2654435761L + k * 40503L;
    return (x ^ (x >>> 13)) % 100000;
  }

  static float[] floats(int i, int salt) {
    float[] v = new float[DIM];
    for (int j = 0; j < DIM; j++) {
      v[j] = (value(i, salt * 100 + j) % 2001 - 1000) / 100f;
    }
    return v;
  }

  static byte[] bytes(int i) {
    byte[] v = new byte[DIM];
    for (int j = 0; j < DIM; j++) {
      v[j] = (byte) (value(i, 500 + j) % 255 - 127);
    }
    return v;
  }

  static float[] expected(String field, int i) {
    switch (field) {
      case "v_sq":
        return i % 7 != 0 ? floats(i, 1) : null;
      case "v_default":
        return i != 0 ? floats(i, 2) : null;
      case "v_small":
        return i != PER_SEGMENT ? floats(i, 3) : null;
      default:
        return floats(i, 4);
    }
  }

  static final Map<String, String> FORMATS =
      Map.of(
          "v_sq", "Lucene104HnswBinaryQuantizedVectorsFormat",
          "v_default", "Lucene99HnswVectorsFormat",
          "v_small", "Lucene99HnswVectorsFormat",
          "v_bytes", "Lucene99HnswVectorsFormat",
          "v_flat", "Lucene104ScalarQuantizedVectorsFormat");

  static void check(boolean ok, String what) {
    if (!ok) throw new AssertionError(what);
  }

  public static void main(String[] args) throws Exception {
    verify(Path.of(args[0]).resolve("flushed"), 2, 600);
    verify(Path.of(args[0]).resolve("merged"), 1, 598);
    System.out.println("VerifyPerFieldKnnVectors: ok");
  }

  static void verify(Path path, int segments, int docs) throws Exception {
    try (Directory dir = FSDirectory.open(path)) {
      check(SegmentInfos.readLatestCommit(dir).size() == segments, path + ": segment count");
      try (DirectoryReader reader = DirectoryReader.open(dir)) {
        check(reader.numDocs() == docs, path + ": documents");
        for (LeafReaderContext ctx : reader.leaves()) {
          LeafReader leaf = ctx.reader();
          for (String field : FORMATS.keySet()) {
            FieldInfo fi = leaf.getFieldInfos().fieldInfo(field);
            check(
                FORMATS.get(field).equals(fi.getAttribute("PerFieldKnnVectorsFormat.format")),
                field + " attributes " + fi.attributes());
          }
          String smallSuffix =
              leaf.getFieldInfos().fieldInfo("v_small").getAttribute("PerFieldKnnVectorsFormat.suffix");
          check(
              smallSuffix.equals(
                  leaf.getFieldInfos()
                      .fieldInfo("v_bytes")
                      .getAttribute("PerFieldKnnVectorsFormat.suffix")),
              "v_small and v_bytes share an instance");
          check(
              !smallSuffix.equals(
                  leaf.getFieldInfos()
                      .fieldInfo("v_default")
                      .getAttribute("PerFieldKnnVectorsFormat.suffix")),
              "v_default has an instance of its own");
          // Which document is which: the id terms.
          int[] original = new int[leaf.maxDoc()];
          Arrays.fill(original, -1);
          Terms terms = leaf.terms("id");
          TermsEnum te = terms.iterator();
          for (BytesRef t = te.next(); t != null; t = te.next()) {
            PostingsEnum pe = te.postings(null, PostingsEnum.NONE);
            for (int d = pe.nextDoc(); d != DocIdSetIterator.NO_MORE_DOCS; d = pe.nextDoc()) {
              original[d] = Integer.parseInt(t.utf8ToString().substring(1));
            }
          }
          for (String field : new String[] {"v_sq", "v_default", "v_small", "v_flat"}) {
            FloatVectorValues values = leaf.getFloatVectorValues(field);
            KnnVectorValues.DocIndexIterator it = values.iterator();
            int seen = 0;
            for (int d = it.nextDoc(); d != DocIdSetIterator.NO_MORE_DOCS; d = it.nextDoc()) {
              if (leaf.getLiveDocs() != null && !leaf.getLiveDocs().get(d)) {
                continue;
              }
              float[] want = expected(field, original[d]);
              check(want != null, field + ": doc " + original[d] + " has no vector");
              check(
                  Arrays.equals(want, values.vectorValue(it.index())),
                  field + ": doc " + original[d]);
              seen++;
            }
            int indexed = 0;
            for (int d = 0; d < leaf.maxDoc(); d++) {
              if ((leaf.getLiveDocs() == null || leaf.getLiveDocs().get(d))
                  && expected(field, original[d]) != null) {
                indexed++;
              }
            }
            check(seen == indexed, field + ": " + seen + " vectors, " + indexed + " indexed");
          }
          ByteVectorValues bytesValues = leaf.getByteVectorValues("v_bytes");
          KnnVectorValues.DocIndexIterator it = bytesValues.iterator();
          for (int d = it.nextDoc(); d != DocIdSetIterator.NO_MORE_DOCS; d = it.nextDoc()) {
            check(
                Arrays.equals(bytes(original[d]), bytesValues.vectorValue(it.index())),
                "v_bytes: doc " + original[d]);
          }
        }
        IndexSearcher searcher = new IndexSearcher(reader);
        for (String field : FORMATS.keySet()) {
          Query query =
              field.equals("v_bytes")
                  ? new KnnByteVectorQuery(field, bytes(1000), 10)
                  : new KnnFloatVectorQuery(field, floats(1000, 7), 10);
          check(searcher.search(query, 10).scoreDocs.length == 10, field + ": ten hits");
        }
      }
      try (CheckIndex checker = new CheckIndex(dir)) {
        check(checker.checkIndex().clean, path + ": CheckIndex");
      }
    }
  }
}
