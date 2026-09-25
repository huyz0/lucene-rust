import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Comparator;
import java.util.stream.Stream;

import org.apache.lucene.document.Document;
import org.apache.lucene.document.NumericDocValuesField;
import org.apache.lucene.document.SortedDocValuesField;
import org.apache.lucene.document.SortedNumericDocValuesField;
import org.apache.lucene.document.SortedSetDocValuesField;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.SegmentCommitInfo;
import org.apache.lucene.index.SegmentInfos;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.BytesRef;

/**
 * Doc-values skip indexes in the shapes GenDocValuesSkipIndex does not cover, in one segment, so
 * the Rust writer's .dvs can be compared with Lucene's byte for byte:
 *
 * <ul>
 *   <li>{@code run}: NUMERIC, a constant run longer than one interval (isDone's dense-run
 *       extension), then varying values;
 *   <li>{@code sorted}: SORTED, sparse -- the skip index covers ordinals;
 *   <li>{@code set_multi}: SORTED_SET, sparse and multi-valued;
 *   <li>{@code set_single}: SORTED_SET with one value per document (written in the SORTED shape,
 *       the skip summary between the type byte and the multiValued byte);
 *   <li>{@code sn}: SORTED_NUMERIC, one or two values per document.
 * </ul>
 *
 * Every value is a function of the document number, so the Rust test rebuilds the same columns
 * without reading them back. Keep the formulas in step with
 * crates/lucene-codecs/tests/doc_values_skip_index_fixtures.rs.
 */
public class GenDocValuesSkipIndexShapes {
  static final int NUM_DOCS = 9000;

  public static void main(String[] args) throws IOException {
    Path out = Path.of(args[0]).resolve("doc_values_skip_index_shapes");
    if (Files.exists(out)) {
      try (Stream<Path> s = Files.walk(out)) {
        s.sorted(Comparator.reverseOrder()).forEach(p -> p.toFile().delete());
      }
    }
    Files.createDirectories(out);

    try (Directory dir = FSDirectory.open(out)) {
      IndexWriterConfig cfg = new IndexWriterConfig();
      cfg.setUseCompoundFile(false);
      cfg.setMergePolicy(NoMergePolicy.INSTANCE);
      cfg.setRAMBufferSizeMB(256);
      cfg.setMaxBufferedDocs(IndexWriterConfig.DISABLE_AUTO_FLUSH);
      try (IndexWriter w = new IndexWriter(dir, cfg)) {
        for (int i = 0; i < NUM_DOCS; i++) {
          Document doc = new Document();
          doc.add(NumericDocValuesField.indexedField("run", i < 5000 ? 42 : (long) i * 3 - 7));
          if (i % 3 != 0) {
            doc.add(SortedDocValuesField.indexedField("sorted", new BytesRef("s" + (i * 7 % 50))));
          }
          if (i % 4 != 0) {
            for (int k = 0; k < 1 + i % 3; k++) {
              doc.add(SortedSetDocValuesField.indexedField("set_multi", new BytesRef("m" + ((i + k * 11) % 40))));
            }
          }
          if (i % 5 != 0) {
            doc.add(SortedSetDocValuesField.indexedField("set_single", new BytesRef("o" + (i % 17))));
          }
          doc.add(SortedNumericDocValuesField.indexedField("sn", i * 2L));
          if (i % 2 == 0) {
            doc.add(SortedNumericDocValuesField.indexedField("sn", -i));
          }
          w.addDocument(doc);
        }
        w.commit();
      }

      SegmentInfos sis = SegmentInfos.readLatestCommit(dir);
      if (sis.size() != 1) {
        throw new AssertionError("expected exactly one segment, got " + sis.size());
      }
      SegmentCommitInfo sci = sis.info(0);
      org.apache.lucene.index.FieldInfos fis =
          sci.info.getCodec().fieldInfosFormat().read(dir, sci.info, "", IOContext.READONCE);
      StringBuilder m = new StringBuilder();
      for (String f : sci.files()) {
        String ext = f.substring(f.lastIndexOf('.') + 1);
        if (ext.equals("dvm") || ext.equals("dvd") || ext.equals("dvs") || ext.equals("fnm")) {
          try (IndexInput in = dir.openInput(f, IOContext.READONCE)) {
            byte[] b = new byte[(int) in.length()];
            in.readBytes(b, 0, b.length);
            Files.write(out.resolve(f + ".raw"), b);
          }
          m.append(ext).append("_file_name=").append(f).append('\n');
        }
      }
      m.append("segment_name=").append(sci.info.name).append('\n');
      m.append("id_hex=").append(hex(sci.info.getId())).append('\n');
      m.append("max_doc=").append(sci.info.maxDoc()).append('\n');
      for (org.apache.lucene.index.FieldInfo fi : fis) {
        m.append("field.").append(fi.name).append('=').append(fi.number).append('\n');
      }
      Files.writeString(out.resolve("manifest.properties"), m.toString());
    }
    System.out.println("wrote doc_values_skip_index_shapes/ fixture directory");
  }

  static String hex(byte[] b) {
    StringBuilder s = new StringBuilder();
    for (byte x : b) {
      s.append(String.format("%02x", x));
    }
    return s.toString();
  }
}
