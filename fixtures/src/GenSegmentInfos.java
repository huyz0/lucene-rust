import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.SegmentCommitInfo;
import org.apache.lucene.index.SegmentInfos;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IOContext;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;

/**
 * Generates a real `segments_N` commit file (and its sibling `.si` files) using an
 * actual IndexWriter over multiple commits, so the fixture exercises real segment
 * merging/generation counters rather than a hand-built one. Copies the whole index
 * directory plus a manifest describing what Rust should find when it parses
 * `segments_N`.
 */
public class GenSegmentInfos {
  public static void main(String[] args) throws IOException {
    Path out = Path.of(args[0]).resolve("segments_index");
    if (Files.exists(out)) {
      deleteRecursive(out);
    }
    Files.createDirectories(out);

    try (Directory dir = FSDirectory.open(out)) {
      IndexWriterConfig cfg = new IndexWriterConfig();
      cfg.setUseCompoundFile(false);
      try (IndexWriter w = new IndexWriter(dir, cfg)) {
        addDoc(w, "1", "the quick brown fox");
        addDoc(w, "2", "jumps over the lazy dog");
        w.commit(); // flush -> first segment, first commit generation

        addDoc(w, "3", "pack my box with five dozen liquor jugs");
        Map<String, String> userData = new HashMap<>();
        userData.put("lucene-rust-test", "true");
        w.setLiveCommitData(userData.entrySet());
        w.commit(); // second segment, second commit generation
      }

      SegmentInfos sis = SegmentInfos.readLatestCommit(dir);
      String segmentsFileName = sis.getSegmentsFileName();

      StringBuilder m = new StringBuilder();
      m.append("segments_file_name=").append(segmentsFileName).append('\n');
      m.append("generation=").append(sis.getGeneration()).append('\n');
      m.append("counter=").append(sis.counter).append('\n');
      m.append("num_segments=").append(sis.size()).append('\n');
      m.append("user_data=").append(joinMap(sis.getUserData())).append('\n');

      StringBuilder segNames = new StringBuilder();
      StringBuilder segDocCounts = new StringBuilder();
      StringBuilder segDelCounts = new StringBuilder();
      for (int i = 0; i < sis.size(); i++) {
        SegmentCommitInfo sci = sis.info(i);
        if (i > 0) {
          segNames.append(',');
          segDocCounts.append(',');
          segDelCounts.append(',');
        }
        segNames.append(sci.info.name);
        segDocCounts.append(sci.info.maxDoc());
        segDelCounts.append(sci.getDelCount());
      }
      m.append("segment_names=").append(segNames).append('\n');
      m.append("segment_doc_counts=").append(segDocCounts).append('\n');
      m.append("segment_del_counts=").append(segDelCounts).append('\n');

      Files.writeString(out.resolve("manifest.properties"), m.toString());

      // also dump raw bytes of the segments_N file itself (already on disk in `out`,
      // but copy explicitly so the fixture's shape is obvious / stable under FSDirectory
      // internals changing).
      try (IndexInput in = dir.openInput(segmentsFileName, IOContext.READONCE)) {
        byte[] bytes = new byte[(int) in.length()];
        in.readBytes(bytes, 0, bytes.length);
        Files.write(out.resolve(segmentsFileName + ".raw"), bytes);
      }
    }

    System.out.println("wrote segments_index/ fixture directory");
  }

  static void addDoc(IndexWriter w, String id, String body) throws IOException {
    Document doc = new Document();
    doc.add(new StringField("id", id, Field.Store.YES));
    doc.add(new TextField("body", body, Field.Store.NO));
    w.addDocument(doc);
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

  static String joinMap(Map<String, String> m) {
    StringBuilder sb = new StringBuilder();
    boolean first = true;
    for (Map.Entry<String, String> e : m.entrySet()) {
      if (!first) sb.append(';');
      first = false;
      sb.append(e.getKey()).append('=').append(e.getValue());
    }
    return sb.toString();
  }
}
