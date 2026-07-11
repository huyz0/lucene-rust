import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.SegmentCommitInfo;
import org.apache.lucene.index.SegmentInfos;
import org.apache.lucene.index.Term;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Generates a real `.liv` (Lucene90LiveDocsFormat) fixture: a single-segment index
 * with 5 docs, two of which are deleted by term after the initial commit (forcing a
 * live-docs generation on the next commit), so the fixture is real deletion bytes
 * rather than hand-built ones.
 */
public class GenLiveDocs {
  public static void main(String[] args) throws IOException {
    Path out = Path.of(args[0]).resolve("live_docs_index");
    if (Files.exists(out)) {
      deleteRecursive(out);
    }
    Files.createDirectories(out);

    try (Directory dir = FSDirectory.open(out)) {
      IndexWriterConfig cfg = new IndexWriterConfig();
      cfg.setUseCompoundFile(false);
      cfg.setMergePolicy(NoMergePolicy.INSTANCE);
      try (IndexWriter w = new IndexWriter(dir, cfg)) {
        for (int i = 0; i < 5; i++) {
          Document doc = new Document();
          doc.add(new StringField("id", Integer.toString(i), Field.Store.YES));
          w.addDocument(doc);
        }
        w.commit(); // single segment, no deletions yet

        w.deleteDocuments(new Term("id", "1"), new Term("id", "3"));
        w.commit(); // same segment now has a .liv file
      }

      SegmentInfos sis = SegmentInfos.readLatestCommit(dir);
      if (sis.size() != 1) {
        throw new AssertionError("expected exactly one segment, got " + sis.size());
      }
      SegmentCommitInfo sci = sis.info(0);

      String livFileName = null;
      for (String f : sci.files()) {
        if (f.endsWith(".liv")) {
          livFileName = f;
        }
      }
      if (livFileName == null) {
        throw new AssertionError("expected a .liv file, files=" + sci.files());
      }

      try (IndexInput in = dir.openInput(livFileName, IOContext.READONCE)) {
        byte[] bytes = new byte[(int) in.length()];
        in.readBytes(bytes, 0, bytes.length);
        Files.write(out.resolve(livFileName + ".raw"), bytes);
      }

      StringBuilder m = new StringBuilder();
      m.append("liv_file_name=").append(livFileName).append('\n');
      m.append("segment_name=").append(sci.info.name).append('\n');
      m.append("id_hex=").append(hex(sci.info.getId())).append('\n');
      m.append("del_gen=").append(sci.getDelGen()).append('\n');
      m.append("max_doc=").append(sci.info.maxDoc()).append('\n');
      m.append("del_count=").append(sci.getDelCount()).append('\n');
      // Doc ids 1 and 3 were deleted by term; with a single unmerged segment
      // and no updates, internal doc ids equal add order (0..4).
      m.append("deleted_doc_ids=1,3\n");
      m.append("live_doc_ids=0,2,4\n");
      Files.writeString(out.resolve("manifest.properties"), m.toString());
    }

    System.out.println("wrote live_docs_index/ fixture directory");
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
