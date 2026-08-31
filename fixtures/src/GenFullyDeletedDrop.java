import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.StoredField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.SegmentInfos;
import org.apache.lucene.index.StoredFields;
import org.apache.lucene.index.Term;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * Cross-engine ground truth for {@code IndexWriter}'s <b>100%-deleted segment
 * drop</b> (ledger item 11b), recorded as an outcome rather than as bytes.
 *
 * <p>{@code IndexWriter.finishApply} removes from the in-memory
 * {@code SegmentInfos} every segment the just-applied deletes left fully
 * deleted:
 *
 * <pre>{@code
 * // closeSegmentStates
 * if (segState.rld.isFullyDeleted()
 *     && getConfig().getMergePolicy().keepFullyDeletedSegment(() -> segState.reader) == false) {
 *   allDeleted.add(segState.reader.getOriginalSegmentInfo());
 * }
 * // finishApply
 * for (SegmentCommitInfo info : result.allDeleted()) { dropDeletedSegment(info); }
 * }</pre>
 *
 * <p>{@code PendingDeletes.isFullyDeleted} is {@code getDelCount() ==
 * info.info.maxDoc()} -- <b>hard</b> deletes only -- and
 * {@code MergePolicy.keepFullyDeletedSegment} returns {@code false} in the base
 * class, which {@code TieredMergePolicy} (the {@code IndexWriterConfig}
 * default) does not override.
 *
 * <p>This port kept such a segment in the commit forever: a segment nothing can
 * ever match, carried by every later open, merge and {@code CheckIndex}.
 *
 * <p>No index is committed to {@code fixtures/data/}: what is checked is a
 * writer's <i>behaviour</i>, and the bytes of an index with a dropped segment
 * are indistinguishable from those of an index that never had it. So each
 * scenario runs in a {@link ByteBuffersDirectory} and only its outcome --
 * committed segment count, per-segment {@code maxDoc}/{@code delCount}, and the
 * visible ids -- is recorded. The Rust test replays the same script through
 * this port's own {@code IndexWriter} and compares.
 *
 * <p>Scenarios, and why each one is here:
 *
 * <ul>
 *   <li>{@code drop} -- the headline: two segments, the older one's every
 *       document deleted. Lucene commits <b>one</b> segment.
 *   <li>{@code partial} -- the control: the same shape with one of the older
 *       segment's two documents deleted. Lucene commits two, so the drop is
 *       conditional and not "any segment with deletes".
 *   <li>{@code all} -- every document of every segment deleted, i.e. the
 *       commit is empty. This is the case where a per-segment fix that only
 *       looked at the *first* segment would still pass {@code drop}.
 *   <li>{@code block} -- {@code updateDocuments} replacing a whole block, the
 *       shape this port's own `update_documents` test drives.
 * </ul>
 */
public class GenFullyDeletedDrop {

  public static void main(String[] args) throws IOException {
    Path out = Path.of(args[0]).resolve("fully_deleted_drop");
    if (Files.exists(out)) {
      deleteRecursive(out);
    }
    Files.createDirectories(out);

    StringBuilder m = new StringBuilder();
    m.append("scenarios=drop,partial,all,block\n");
    drop(m);
    partial(m);
    all(m);
    block(m);
    Files.writeString(out.resolve("manifest.properties"), m.toString());
    System.out.println("wrote fully_deleted_drop/ fixture directory");
  }

  /** Two segments; every document of the older one is deleted. */
  private static void drop(StringBuilder m) throws IOException {
    try (Directory dir = new ByteBuffersDirectory()) {
      try (IndexWriter w = writer(dir)) {
        w.addDocument(doc("a", "shared"));
        w.addDocument(doc("b", "shared"));
        w.flush();
        w.deleteDocuments(new Term("body", "shared"));
        w.addDocument(doc("c", "other"));
        w.addDocument(doc("d", "other"));
        w.commit();
      }
      record(m, "drop", dir);
    }
  }

  /** The control: only one of the older segment's two documents is deleted. */
  private static void partial(StringBuilder m) throws IOException {
    try (Directory dir = new ByteBuffersDirectory()) {
      try (IndexWriter w = writer(dir)) {
        w.addDocument(doc("a", "shared"));
        w.addDocument(doc("b", "kept"));
        w.flush();
        w.deleteDocuments(new Term("body", "shared"));
        w.addDocument(doc("c", "other"));
        w.addDocument(doc("d", "other"));
        w.commit();
      }
      record(m, "partial", dir);
    }
  }

  /** Every document of every segment deleted: the commit is empty. */
  private static void all(StringBuilder m) throws IOException {
    try (Directory dir = new ByteBuffersDirectory()) {
      try (IndexWriter w = writer(dir)) {
        w.addDocument(doc("a", "shared"));
        w.addDocument(doc("b", "shared"));
        w.flush();
        w.addDocument(doc("c", "shared"));
        w.addDocument(doc("d", "shared"));
        w.flush();
        w.deleteDocuments(new Term("body", "shared"));
        w.commit();
      }
      record(m, "all", dir);
    }
  }

  /** `updateDocuments` replacing a whole block, emptying the block's segment. */
  private static void block(StringBuilder m) throws IOException {
    try (Directory dir = new ByteBuffersDirectory()) {
      try (IndexWriter w = writer(dir)) {
        w.addDocuments(List.of(doc("p1", "key"), doc("c1", "key")));
        w.commit();
        w.updateDocuments(new Term("body", "key"), List.of(doc("p2", "key"), doc("c2", "key")));
        w.commit();
      }
      record(m, "block", dir);
    }
  }

  private static IndexWriter writer(Directory dir) throws IOException {
    IndexWriterConfig cfg = new IndexWriterConfig();
    cfg.setUseCompoundFile(false);
    cfg.setMergePolicy(NoMergePolicy.INSTANCE);
    return new IndexWriter(dir, cfg);
  }

  private static Document doc(String id, String body) {
    Document doc = new Document();
    doc.add(new StringField("id", id, Field.Store.NO));
    doc.add(new StoredField("id", id));
    doc.add(new StringField("body", body, Field.Store.NO));
    return doc;
  }

  private static void record(StringBuilder m, String name, Directory dir) throws IOException {
    SegmentInfos sis = SegmentInfos.readLatestCommit(dir);
    String prefix = name + ".";
    m.append(prefix).append("segment_count=").append(sis.size()).append('\n');
    StringBuilder shape = new StringBuilder();
    for (int i = 0; i < sis.size(); i++) {
      if (shape.length() > 0) shape.append(',');
      shape.append(sis.info(i).info.maxDoc()).append(':').append(sis.info(i).getDelCount());
    }
    m.append(prefix).append("segment_shape=").append(shape).append('\n');

    List<String> ids = new ArrayList<>();
    if (sis.size() > 0) {
      try (DirectoryReader reader = DirectoryReader.open(dir)) {
        StoredFields stored = reader.storedFields();
        var live = org.apache.lucene.index.MultiBits.getLiveDocs(reader);
        for (int doc = 0; doc < reader.maxDoc(); doc++) {
          if (live == null || live.get(doc)) {
            ids.add(stored.document(doc).get("id"));
          }
        }
      }
    }
    ids.sort(String::compareTo);
    m.append(prefix).append("visible_ids=").append(String.join(",", ids)).append('\n');
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
}
