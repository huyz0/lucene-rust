import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.SegmentCommitInfo;
import org.apache.lucene.index.SegmentInfos;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Generates real `.nvm`/`.nvd` (Lucene90NormsFormat) fixtures: a single
 * dense-norms field ("body", a TextField indexed with norms on, varying doc
 * lengths so bytesPerNorm ends up > 0) across several documents, all in one
 * unmerged segment.
 */
public class GenNorms {
  public static void main(String[] args) throws IOException {
    Path out = Path.of(args[0]).resolve("norms_index");
    if (Files.exists(out)) {
      deleteRecursive(out);
    }
    Files.createDirectories(out);

    try (Directory dir = FSDirectory.open(out)) {
      IndexWriterConfig cfg = new IndexWriterConfig();
      cfg.setUseCompoundFile(false);
      cfg.setMergePolicy(NoMergePolicy.INSTANCE);

      // Deliberately different lengths so per-doc norms aren't all identical
      // (a single-length corpus can collapse bytesPerNorm to 0, the constant
      // case, which we want to exercise via a different, more controlled
      // fixture if ever needed -- this one targets bytesPerNorm > 0).
      String[] bodies = {
        "a",
        "a b c d e f g h",
        "a b",
        "a b c d e f g h i j k l m n o p q r s t",
        "a b c"
      };

      try (IndexWriter w = new IndexWriter(dir, cfg)) {
        for (String body : bodies) {
          Document doc = new Document();
          doc.add(new TextField("body", body, Field.Store.NO));
          w.addDocument(doc);
        }
        w.commit();
      }

      SegmentInfos sis = SegmentInfos.readLatestCommit(dir);
      if (sis.size() != 1) {
        throw new AssertionError("expected exactly one segment, got " + sis.size());
      }
      SegmentCommitInfo sci = sis.info(0);

      String nvmFileName = null;
      String nvdFileName = null;
      for (String f : sci.files()) {
        if (f.endsWith(".nvm")) nvmFileName = f;
        if (f.endsWith(".nvd")) nvdFileName = f;
      }
      if (nvmFileName == null || nvdFileName == null) {
        throw new AssertionError("expected .nvm/.nvd files, files=" + sci.files());
      }

      dump(dir, nvmFileName, out);
      dump(dir, nvdFileName, out);

      org.apache.lucene.index.FieldInfos fis =
          sci.info.getCodec().fieldInfosFormat().read(dir, sci.info, "", IOContext.READONCE);
      org.apache.lucene.index.FieldInfo bodyField = fis.fieldInfo("body");

      // Read norms directly through the codec's NormsProducer, so the
      // manifest's expected values come from Lucene itself rather than our
      // own arithmetic on token counts.
      org.apache.lucene.codecs.NormsProducer normsProducer =
          sci.info
              .getCodec()
              .normsFormat()
              .normsProducer(
                  new org.apache.lucene.index.SegmentReadState(
                      dir, sci.info, fis, IOContext.READONCE));
      org.apache.lucene.index.NumericDocValues norms = normsProducer.getNorms(bodyField);

      StringBuilder m = new StringBuilder();
      m.append("nvm_file_name=").append(nvmFileName).append('\n');
      m.append("nvd_file_name=").append(nvdFileName).append('\n');
      m.append("segment_name=").append(sci.info.name).append('\n');
      m.append("id_hex=").append(hex(sci.info.getId())).append('\n');
      m.append("field_number=").append(bodyField.number).append('\n');
      m.append("max_doc=").append(sci.info.maxDoc()).append('\n');

      StringBuilder normValues = new StringBuilder();
      for (int doc = 0; doc < sci.info.maxDoc(); doc++) {
        if (doc > 0) normValues.append(',');
        if (norms.advanceExact(doc)) {
          normValues.append(norms.longValue());
        } else {
          normValues.append("NONE");
        }
      }
      m.append("norm_values=").append(normValues).append('\n');
      Files.writeString(out.resolve("manifest.properties"), m.toString());

      normsProducer.close();
    }

    System.out.println("wrote norms_index/ fixture directory");
  }

  static void dump(Directory dir, String fileName, Path out) throws IOException {
    try (IndexInput in = dir.openInput(fileName, IOContext.READONCE)) {
      byte[] bytes = new byte[(int) in.length()];
      in.readBytes(bytes, 0, bytes.length);
      Files.write(out.resolve(fileName + ".raw"), bytes);
    }
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
