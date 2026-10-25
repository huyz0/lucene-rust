import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashSet;
import org.apache.lucene.index.*;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.AcceptDocs;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.document.*;

/** Leaf searches on the existing Java-produced vectors_index, with finite visit budgets. */
public class GenTestFrameworkKnnVisits {
  public static void main(String[] args) throws Exception {
    StringBuilder out = new StringBuilder();
    try (var dir = FSDirectory.open(Path.of("fixtures/data/vectors_index"));
        var reader = DirectoryReader.open(dir)) {
      if (reader.leaves().size() != 1) throw new AssertionError("one fixture leaf");
      var leaf = reader.leaves().get(0).reader();
      for (FieldInfo field : leaf.getFieldInfos()) {
        if (field.getVectorDimension() == 0) continue;
        boolean floats = field.getVectorEncoding() == VectorEncoding.FLOAT32;
        var fv = floats ? leaf.getFloatVectorValues(field.name) : null;
        var bv = floats ? null : leaf.getByteVectorValues(field.name);
        int size = floats ? fv.size() : bv.size();
        for (int queryOrd : new int[] {0, size / 2}) {
          var ks = new LinkedHashSet<Integer>();
          ks.add(0);
          ks.add(1);
          ks.add(Math.min(10, size));
          ks.add(size);
          ks.add(size + 20);
          for (int k : ks) {
            var limits = new LinkedHashSet<Integer>();
            for (int limit : new int[] {0, 1, k, k + 3, size + 30, Integer.MAX_VALUE})
              limits.add(limit);
            for (int limit : limits) {
              var accept = AcceptDocs.fromLiveDocs(leaf.getLiveDocs(), leaf.maxDoc());
              TopDocs hits = floats
                  ? leaf.searchNearestVectors(field.name, fv.vectorValue(queryOrd), k, accept, limit)
                  : leaf.searchNearestVectors(field.name, bv.vectorValue(queryOrd), k, accept, limit);
              out.append(field.name).append('\t').append(queryOrd).append('\t').append(k)
                  .append('\t').append(limit).append('\t').append(hits.totalHits.value())
                  .append('\t').append(hits.totalHits.relation());
              for (var hit : hits.scoreDocs)
                out.append('\t').append(hit.doc).append(':').append(Float.floatToIntBits(hit.score));
              out.append('\n');
            }
          }
        }
      }
    }
    Path root = Path.of(args[0]);
    Files.createDirectories(root);
    Files.writeString(root.resolve("knn-visits.tsv"), out);
    StringBuilder empty = new StringBuilder();
    try (var dir = new ByteBuffersDirectory()) {
      try (var writer = new IndexWriter(dir, new IndexWriterConfig())) {
        var vector = new Document();
        vector.add(new StringField("id", "0", Field.Store.NO));
        vector.add(new KnnFloatVectorField("float", new float[] {1, 0}));
        vector.add(new KnnByteVectorField("byte", new byte[] {1, 0}));
        writer.addDocument(vector);
        var survivor = new Document();
        survivor.add(new StringField("id", "1", Field.Store.NO));
        writer.addDocument(survivor);
        writer.commit();
        writer.deleteDocuments(new Term("id", "0"));
        writer.forceMerge(1);
      }
      try (var reader = DirectoryReader.open(dir)) {
        var leaf = reader.leaves().get(0).reader();
        if (leaf.getFloatVectorValues("float").size() != 0
            || leaf.getByteVectorValues("byte").size() != 0)
          throw new AssertionError("empty vector fields");
        for (String encoding : new String[] {"float", "byte"}) {
          for (String field : new String[] {encoding, "id", "missing"}) {
            for (int k : new int[] {0, 1, 20}) {
              for (int limit : new int[] {0, Integer.MAX_VALUE}) {
                var accept = AcceptDocs.fromLiveDocs(leaf.getLiveDocs(), leaf.maxDoc());
                TopDocs hits = encoding.equals("float")
                    ? leaf.searchNearestVectors(field, new float[] {1, 0}, k, accept, limit)
                    : leaf.searchNearestVectors(field, new byte[] {1, 0}, k, accept, limit);
                if (hits.scoreDocs.length != 0) throw new AssertionError("empty result");
                empty.append(encoding).append('\t').append(field).append('\t').append(k)
                    .append('\t').append(limit).append('\t').append(hits.totalHits.value())
                    .append('\t').append(hits.totalHits.relation()).append('\n');
              }
            }
          }
        }
      }
    }
    Files.writeString(root.resolve("knn-empty.tsv"), empty);
  }
}
