import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.FieldType;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.LongPoint;
import org.apache.lucene.document.NumericDocValuesField;
import org.apache.lucene.document.SortedDocValuesField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.SegmentCommitInfo;
import org.apache.lucene.index.SegmentInfos;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.BytesRef;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * Generates a real `.fnm` (Lucene94FieldInfosFormat) fixture: a single-segment
 * index with a field of each notable shape (plain indexed, term-vectors, doc
 * values of a couple types, points, a KNN vector field, a soft-deletes field),
 * so the fixture exercises real field-number assignment and bit-packing rather
 * than hand-built bytes.
 */
public class GenFieldInfos {
  static final String SOFT_DELETES_FIELD = "__soft_deletes";

  public static void main(String[] args) throws IOException {
    Path out = Path.of(args[0]).resolve("field_infos_index");
    if (Files.exists(out)) {
      deleteRecursive(out);
    }
    Files.createDirectories(out);

    try (Directory dir = FSDirectory.open(out)) {
      IndexWriterConfig cfg = new IndexWriterConfig();
      cfg.setUseCompoundFile(false);
      cfg.setMergePolicy(NoMergePolicy.INSTANCE);
      cfg.setSoftDeletesField(SOFT_DELETES_FIELD);

      try (IndexWriter w = new IndexWriter(dir, cfg)) {
        // Two docs so soft-deleting one later still leaves a live segment
        // (a fully-soft-deleted segment is dropped on flush, which would
        // defeat the point of exercising the soft-deletes field bit).
        for (String id : new String[] {"1", "2"}) {
          Document doc = new Document();
          doc.add(new StringField("id", id, Field.Store.YES));
          doc.add(new TextField("body", "hello world from lucene rust", Field.Store.NO));

          FieldType withTermVectors = new FieldType(TextField.TYPE_NOT_STORED);
          withTermVectors.setStoreTermVectors(true);
          withTermVectors.setIndexOptions(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS);
          doc.add(new Field("with_tv", "term vector field contents", withTermVectors));

          doc.add(new NumericDocValuesField("num_dv", 42L));
          doc.add(new SortedDocValuesField("sorted_dv", new BytesRef("sorted-value")));
          doc.add(new LongPoint("point_field", 12345L));
          doc.add(new KnnFloatVectorField("vector_field", new float[] {0.1f, 0.2f, 0.3f},
              VectorSimilarityFunction.COSINE));
          w.addDocument(doc);
        }
        w.commit();

        // Soft-delete doc "1" via a doc-values update on the configured
        // soft-deletes field — this is the real mechanism (see
        // IndexWriter's isFullyDeleted handling); setting the field at
        // addDocument time instead marks the doc dead immediately.
        w.updateDocValues(new Term("id", "1"), new NumericDocValuesField(SOFT_DELETES_FIELD, 1L));
        w.commit();
      }

      SegmentInfos sis = SegmentInfos.readLatestCommit(dir);
      if (sis.size() != 1) {
        throw new AssertionError("expected exactly one segment, got " + sis.size());
      }
      SegmentCommitInfo sci = sis.info(0);

      // The soft-deletes update introduced a NEW field ("__soft_deletes") not
      // present in the original flush's FieldInfos, so it lives in a
      // generation-suffixed `.fnm` file, not the segment's original one (see
      // SegmentReader.getFieldInfos: suffix = Long.toString(fieldInfosGen, 36)
      // when fieldInfosGen != -1). Read via that same generation-aware path
      // rather than guessing a filename.
      long fieldInfosGen = sci.getFieldInfosGen();
      String segmentSuffix =
          fieldInfosGen != -1 ? Long.toString(fieldInfosGen, Character.MAX_RADIX) : "";
      String fnmFileName =
          org.apache.lucene.index.IndexFileNames.segmentFileName(
              sci.info.name, segmentSuffix, "fnm");

      try (IndexInput in = dir.openInput(fnmFileName, IOContext.READONCE)) {
        byte[] bytes = new byte[(int) in.length()];
        in.readBytes(bytes, 0, bytes.length);
        Files.write(out.resolve(fnmFileName + ".raw"), bytes);
      }

      List<String> fieldOrder = new ArrayList<>();
      // Re-read via FieldInfos to dump the exact assigned field numbers/types,
      // since IndexWriter assigns field numbers itself (not caller-controlled).
      org.apache.lucene.index.FieldInfos fis =
          sci.info.getCodec().fieldInfosFormat().read(dir, sci.info, segmentSuffix, IOContext.READONCE);

      StringBuilder m = new StringBuilder();
      m.append("fnm_file_name=").append(fnmFileName).append('\n');
      m.append("segment_name=").append(sci.info.name).append('\n');
      m.append("segment_suffix=").append(segmentSuffix).append('\n');
      m.append("id_hex=").append(hex(sci.info.getId())).append('\n');
      m.append("field_count=").append(fis.size()).append('\n');
      for (org.apache.lucene.index.FieldInfo fi : fis) {
        fieldOrder.add(fi.name);
        String prefix = "field." + fi.name + ".";
        m.append(prefix).append("number=").append(fi.number).append('\n');
        m.append(prefix).append("index_options=").append(fi.getIndexOptions()).append('\n');
        m.append(prefix).append("doc_values_type=").append(fi.getDocValuesType()).append('\n');
        m.append(prefix).append("has_term_vectors=").append(fi.hasTermVectors()).append('\n');
        m.append(prefix).append("is_soft_deletes=").append(fi.isSoftDeletesField()).append('\n');
        m.append(prefix).append("point_dimension_count=").append(fi.getPointDimensionCount()).append('\n');
        m.append(prefix).append("point_num_bytes=").append(fi.getPointNumBytes()).append('\n');
        m.append(prefix).append("vector_dimension=").append(fi.getVectorDimension()).append('\n');
        m.append(prefix).append("vector_similarity=").append(fi.getVectorSimilarityFunction()).append('\n');
      }
      m.append("field_order=").append(String.join(",", fieldOrder)).append('\n');
      Files.writeString(out.resolve("manifest.properties"), m.toString());
    }

    System.out.println("wrote field_infos_index/ fixture directory");
  }

  static void deleteRecursive(Path p) throws IOException {
    if (Files.isDirectory(p)) {
      try (var entries = Files.list(p)) {
        for (Path child : (Iterable<Path>) entries::iterator) {
          deleteRecursive(child);
        }
      }
    }
    Files.deleteIfExists(p);
  }

  static String hex(byte[] b) {
    StringBuilder sb = new StringBuilder();
    for (byte x : b) sb.append(String.format("%02x", x));
    return sb.toString();
  }
}
