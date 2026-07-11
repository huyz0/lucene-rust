import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.StoredField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.SegmentCommitInfo;
import org.apache.lucene.index.SegmentInfos;
import org.apache.lucene.index.StoredFieldVisitor;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.BytesRef;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HexFormat;

/**
 * Generates real `.fdt`/`.fdx`/`.fdm` (Lucene90StoredFieldsFormat,
 * Mode.BEST_SPEED) fixtures: several documents each with one field of every
 * supported type (string, binary, int, long, float, double), enough docs
 * that the chunk uses the bulk (non-single-doc) framing, plus one larger
 * document to exercise a non-trivial per-doc byte length.
 */
public class GenStoredFields {
  public static void main(String[] args) throws IOException {
    Path out = Path.of(args[0]).resolve("stored_fields_index");
    if (Files.exists(out)) {
      deleteRecursive(out);
    }
    Files.createDirectories(out);

    try (Directory dir = FSDirectory.open(out)) {
      IndexWriterConfig cfg = new IndexWriterConfig();
      cfg.setUseCompoundFile(false);
      cfg.setMergePolicy(NoMergePolicy.INSTANCE);

      int numDocs = 6;
      try (IndexWriter w = new IndexWriter(dir, cfg)) {
        for (int i = 0; i < numDocs; i++) {
          Document doc = new Document();
          doc.add(new StringField("id", Integer.toString(i), Field.Store.NO));
          doc.add(new StoredField("str", "hello-" + i + "-" + "x".repeat(i * 3)));
          doc.add(new StoredField("bin", new BytesRef(("bytes" + i).getBytes(StandardCharsets.UTF_8))));
          doc.add(new StoredField("int", -1000 + i));
          doc.add(new StoredField("long", 1_000_000_000_000L + i));
          doc.add(new StoredField("float", 1.5f + i));
          doc.add(new StoredField("double", 2.25d + i));
          w.addDocument(doc);
        }
        w.commit();
      }

      SegmentInfos sis = SegmentInfos.readLatestCommit(dir);
      if (sis.size() != 1) {
        throw new AssertionError("expected exactly one segment, got " + sis.size());
      }
      SegmentCommitInfo sci = sis.info(0);

      String fdtFileName = null;
      String fdxFileName = null;
      String fdmFileName = null;
      for (String f : sci.info.files()) {
        if (f.endsWith(".fdt")) fdtFileName = f;
        if (f.endsWith(".fdx")) fdxFileName = f;
        if (f.endsWith(".fdm")) fdmFileName = f;
      }
      if (fdtFileName == null || fdxFileName == null || fdmFileName == null) {
        throw new AssertionError("expected .fdt/.fdx/.fdm files, files=" + sci.info.files());
      }

      dump(dir, fdtFileName, out);
      dump(dir, fdxFileName, out);
      dump(dir, fdmFileName, out);

      StringBuilder m = new StringBuilder();
      m.append("fdt_file_name=").append(fdtFileName).append('\n');
      m.append("fdx_file_name=").append(fdxFileName).append('\n');
      m.append("fdm_file_name=").append(fdmFileName).append('\n');
      m.append("segment_name=").append(sci.info.name).append('\n');
      m.append("id_hex=").append(hex(sci.info.getId())).append('\n');
      m.append("max_doc=").append(sci.info.maxDoc()).append('\n');

      org.apache.lucene.codecs.StoredFieldsReader fieldsReader =
          sci.info
              .getCodec()
              .storedFieldsFormat()
              .fieldsReader(
                  dir,
                  sci.info,
                  sci.info.getCodec().fieldInfosFormat().read(dir, sci.info, "", IOContext.READONCE),
                  IOContext.READONCE);

      for (int doc = 0; doc < sci.info.maxDoc(); doc++) {
        DumpVisitor visitor = new DumpVisitor();
        fieldsReader.document(doc, visitor);
        m.append("doc.").append(doc).append('.').append("fields=").append(visitor.render()).append('\n');
      }
      fieldsReader.close();

      Files.writeString(out.resolve("manifest.properties"), m.toString());
    }

    System.out.println("wrote stored_fields_index/ fixture directory");
  }

  /** Renders each visited field as `name:type:hexOrValue`, joined by ';'. */
  static class DumpVisitor extends StoredFieldVisitor {
    private final StringBuilder sb = new StringBuilder();

    private void sep() {
      if (sb.length() > 0) sb.append(';');
    }

    String render() {
      return sb.toString();
    }

    @Override
    public Status needsField(FieldInfo fieldInfo) {
      return "id".equals(fieldInfo.name) ? Status.NO : Status.YES;
    }

    @Override
    public void stringField(FieldInfo fieldInfo, String value) {
      sep();
      sb.append(fieldInfo.name).append(":string:").append(value);
    }

    @Override
    public void binaryField(FieldInfo fieldInfo, byte[] value) {
      sep();
      sb.append(fieldInfo.name).append(":binary:").append(HexFormat.of().formatHex(value));
    }

    @Override
    public void intField(FieldInfo fieldInfo, int value) {
      sep();
      sb.append(fieldInfo.name).append(":int:").append(value);
    }

    @Override
    public void longField(FieldInfo fieldInfo, long value) {
      sep();
      sb.append(fieldInfo.name).append(":long:").append(value);
    }

    @Override
    public void floatField(FieldInfo fieldInfo, float value) {
      sep();
      sb.append(fieldInfo.name).append(":float:").append(value);
    }

    @Override
    public void doubleField(FieldInfo fieldInfo, double value) {
      sep();
      sb.append(fieldInfo.name).append(":double:").append(value);
    }
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
