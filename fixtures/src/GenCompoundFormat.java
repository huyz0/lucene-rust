import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.NumericDocValuesField;
import org.apache.lucene.document.StringField;
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
 * Generates a real `.cfs`/`.cfe` (Lucene90CompoundFormat) fixture: a single
 * segment with `useCompoundFile=true` forced via `NoMergePolicy.NO_COMPOUND_FILES`
 * being avoided -- we explicitly ask for compound files on the writer config
 * so the segment's sub-files (`.si`, `.fnm`, `.dvd`, `.dvm`, ...) get packed
 * into one `.cfs`/`.cfe` pair instead of being written loose.
 */
public class GenCompoundFormat {
  public static void main(String[] args) throws IOException {
    Path out = Path.of(args[0]).resolve("compound_index");
    if (Files.exists(out)) {
      deleteRecursive(out);
    }
    Files.createDirectories(out);

    try (Directory dir = FSDirectory.open(out)) {
      IndexWriterConfig cfg = new IndexWriterConfig();
      cfg.setUseCompoundFile(true);
      cfg.setMergePolicy(NoMergePolicy.INSTANCE);

      try (IndexWriter w = new IndexWriter(dir, cfg)) {
        for (int i = 0; i < 5; i++) {
          org.apache.lucene.document.Document doc = new Document();
          doc.add(new StringField("id", Integer.toString(i), Field.Store.NO));
          doc.add(new NumericDocValuesField("num", i * 10L));
          w.addDocument(doc);
        }
        w.commit();
      }

      SegmentInfos sis = SegmentInfos.readLatestCommit(dir);
      if (sis.size() != 1) {
        throw new AssertionError("expected exactly one segment, got " + sis.size());
      }
      SegmentCommitInfo sci = sis.info(0);
      if (!sci.info.getUseCompoundFile()) {
        throw new AssertionError("expected segment to use a compound file");
      }

      String cfsFileName = null;
      String cfeFileName = null;
      for (String f : sci.info.files()) {
        if (f.endsWith(".cfs")) cfsFileName = f;
        if (f.endsWith(".cfe")) cfeFileName = f;
      }
      if (cfsFileName == null || cfeFileName == null) {
        throw new AssertionError("expected .cfs/.cfe files, files=" + sci.info.files());
      }

      dump(dir, cfsFileName, out);
      dump(dir, cfeFileName, out);

      // Also read the .cfs/.cfe pair back through Lucene's own compound
      // reader, to get the real sub-file names/lengths for the manifest --
      // this is what actually gets packed, e.g. "_0.si", "_0.fnm", etc.
      org.apache.lucene.codecs.CompoundDirectory cfsDir =
          sci.info.getCodec().compoundFormat().getCompoundReader(dir, sci.info);

      StringBuilder m = new StringBuilder();
      m.append("cfs_file_name=").append(cfsFileName).append('\n');
      m.append("cfe_file_name=").append(cfeFileName).append('\n');
      m.append("segment_name=").append(sci.info.name).append('\n');
      m.append("id_hex=").append(hex(sci.info.getId())).append('\n');

      StringBuilder subFiles = new StringBuilder();
      for (String name : cfsDir.listAll()) {
        if (subFiles.length() > 0) subFiles.append(',');
        // Strip the segment-name prefix to get the entries-table id, and
        // record the length read back through the compound reader itself
        // (not re-derived) so the manifest is an independent cross-check.
        String id = org.apache.lucene.index.IndexFileNames.stripSegmentName(name);
        subFiles.append(id).append(':').append(cfsDir.fileLength(name));
      }
      m.append("sub_files=").append(subFiles).append('\n');
      cfsDir.close();

      Files.writeString(out.resolve("manifest.properties"), m.toString());
    }

    System.out.println("wrote compound_index/ fixture directory");
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
