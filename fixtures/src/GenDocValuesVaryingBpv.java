import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.NumericDocValuesField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.NumericDocValues;
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
 * Generates a real `.dvm`/`.dvd` (Lucene90DocValuesFormat) fixture for a
 * NUMERIC field whose values force the writer's `doBlocks` varying-bits-
 * per-value split (`Lucene90DocValuesConsumer.writeValues`): the field is
 * split into `NUMERIC_BLOCK_SIZE` (16384)-value blocks, each independently
 * bit-packed, whenever doing so would save at least 10% versus one
 * whole-field width (`blockMinMax.spaceInBits / minMax.spaceInBits <= 0.9`).
 *
 * <p>Two full blocks (32768 docs): the first block's values fit in a small
 * range (needs few bits per value), the second block's values span a huge
 * range (needs many more bits per value) -- so the per-block average width
 * is much smaller than the single width the whole field would need,
 * comfortably tripping the 10%-savings heuristic and forcing `doBlocks`.
 */
public class GenDocValuesVaryingBpv {
  static final int BLOCK_SIZE = 1 << 14; // Lucene90DocValuesFormat.NUMERIC_BLOCK_SIZE
  static final int NUM_DOCS = 2 * BLOCK_SIZE + 100; // 2 full blocks + a partial third

  public static void main(String[] args) throws IOException {
    Path out = Path.of(args[0]).resolve("doc_values_varying_bpv");
    if (Files.exists(out)) {
      deleteRecursive(out);
    }
    Files.createDirectories(out);

    try (Directory dir = FSDirectory.open(out)) {
      IndexWriterConfig cfg = new IndexWriterConfig();
      cfg.setUseCompoundFile(false);
      cfg.setMergePolicy(NoMergePolicy.INSTANCE);

      try (IndexWriter w = new IndexWriter(dir, cfg)) {
        for (int i = 0; i < NUM_DOCS; i++) {
          Document doc = new Document();
          doc.add(new StringField("id", Integer.toString(i), Field.Store.NO));
          doc.add(new NumericDocValuesField("varying_bpv", valueFor(i)));
          w.addDocument(doc);
        }
        w.commit();
        w.forceMerge(1);
      }

      SegmentInfos sis = SegmentInfos.readLatestCommit(dir);
      if (sis.size() != 1) {
        throw new AssertionError("expected exactly one segment, got " + sis.size());
      }
      SegmentCommitInfo sci = sis.info(0);

      String dvmFileName = null;
      String dvdFileName = null;
      String fnmFileName = null;
      for (String f : sci.files()) {
        if (f.endsWith(".dvm")) dvmFileName = f;
        if (f.endsWith(".dvd")) dvdFileName = f;
        if (f.endsWith(".fnm")) fnmFileName = f;
      }
      if (dvmFileName == null || dvdFileName == null || fnmFileName == null) {
        throw new AssertionError("expected .dvm/.dvd/.fnm files, files=" + sci.files());
      }

      dump(dir, dvmFileName, out);
      dump(dir, dvdFileName, out);
      dump(dir, fnmFileName, out);

      org.apache.lucene.index.FieldInfos fis =
          sci.info.getCodec().fieldInfosFormat().read(dir, sci.info, "", IOContext.READONCE);
      org.apache.lucene.index.FieldInfo field = fis.fieldInfo("varying_bpv");

      StringBuilder m = new StringBuilder();
      m.append("dvm_file_name=").append(dvmFileName).append('\n');
      m.append("dvd_file_name=").append(dvdFileName).append('\n');
      m.append("fnm_file_name=").append(fnmFileName).append('\n');
      m.append("segment_name=").append(sci.info.name).append('\n');
      m.append("id_hex=").append(hex(sci.info.getId())).append('\n');
      m.append("max_doc=").append(sci.info.maxDoc()).append('\n');
      m.append("field_numbers=varying_bpv:").append(field.number).append('\n');

      org.apache.lucene.codecs.DocValuesProducer dvProducer =
          sci.info
              .getCodec()
              .docValuesFormat()
              .fieldsProducer(
                  new org.apache.lucene.index.SegmentReadState(
                      dir, sci.info, fis, IOContext.READONCE));

      NumericDocValues values = dvProducer.getNumeric(field);
      StringBuilder vals = new StringBuilder();
      for (int doc = 0; doc < sci.info.maxDoc(); doc++) {
        if (doc > 0) vals.append(',');
        if (values.advanceExact(doc)) {
          vals.append(values.longValue());
        } else {
          vals.append("NONE");
        }
      }
      m.append("field.varying_bpv.values=").append(vals).append('\n');
      dvProducer.close();

      Files.writeString(out.resolve("manifest.properties"), m.toString());
    }

    System.out.println("wrote doc_values_varying_bpv/ fixture directory");
  }

  static long valueFor(int doc) {
    if (doc < BLOCK_SIZE) {
      // First block: tiny range, few bits per value.
      return doc % 50;
    } else if (doc < 2 * BLOCK_SIZE) {
      // Second block: huge range, many bits per value.
      return ((long) (doc - BLOCK_SIZE)) * 1_000_000_000L;
    } else {
      // Trailing partial block: back to a tiny range.
      return (doc - 2 * BLOCK_SIZE) % 7;
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
