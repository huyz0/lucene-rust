import org.apache.lucene.document.Document;
import org.apache.lucene.document.StoredField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FilterMergePolicy;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.MergePolicy;
import org.apache.lucene.index.MergeTrigger;
import org.apache.lucene.index.SegmentCommitInfo;
import org.apache.lucene.index.SegmentInfos;
import org.apache.lucene.index.SegmentReader;
import org.apache.lucene.index.SerialMergeScheduler;
import org.apache.lucene.index.TieredMergePolicy;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * Merge-on-commit and merge-on-refresh ({@code IndexWriterConfig.setMaxFullFlushMergeWaitMillis},
 * {@code MergePolicy.findFullFlushMerges}): the segments each commit and each near-real-time reader
 * holds, for a writer that waits for full-flush merges (the default 500 ms) and for one that does
 * not (0).
 *
 * <p>The policy is {@link TieredMergePolicy} ({@code segmentsPerTier} 2, {@code maxMergeAtOnce} 2,
 * everything else default, so every segment is below the 16 MB floor and eligible) wrapped so that
 * it proposes merges <i>only</i> for the {@code COMMIT} and {@code GET_READER} triggers. The
 * natural merges Lucene runs after a flush would otherwise land between commits, where this port
 * writes them as commits of their own; with them off, every merge here is a point-in-time merge
 * and the commit that contains it is exactly comparable. {@link SerialMergeScheduler} runs those on
 * the committing thread, so the wait always sees them finish.
 *
 * <p>Every document is one stored field of the same length, one document per flush, so the
 * segments tie on size in both engines and the policy's choice does not depend on how many bytes
 * either codec writes.
 *
 * <p>Output {@code merge_on_commit/{wait,nowait}/manifest.txt}: one line per step, {@code
 * commit <gen> <name>:<maxDoc> ...} for a commit (read back with {@code SegmentInfos.readCommit})
 * and {@code reader <name>:<maxDoc> ...} for a {@code DirectoryReader.open(writer)}, then {@code
 * docs <id> ...}, the stored ids of the final commit in doc order.
 */
public class GenMergeOnCommit {
  /** The steps both writers run: {@code a<n>} adds and commits n documents one flush each, {@code r} opens an NRT reader. */
  static final String[] STEPS = {"c1", "c1", "c1", "c1", "r1", "c1", "c1", "r1", "r1", "c2", "c1"};

  public static void main(String[] args) throws IOException {
    Path root = Path.of(args[0]).resolve("merge_on_commit");
    run(root.resolve("wait"), 500);
    run(root.resolve("nowait"), 0);
  }

  static final class FullFlushOnly extends FilterMergePolicy {
    FullFlushOnly(MergePolicy in) {
      super(in);
    }

    @Override
    public MergeSpecification findMerges(
        MergeTrigger trigger, SegmentInfos infos, MergeContext ctx) throws IOException {
      if (trigger != MergeTrigger.COMMIT && trigger != MergeTrigger.GET_READER) {
        return null;
      }
      return in.findMerges(trigger, infos, ctx);
    }

    @Override
    public MergeSpecification findFullFlushMerges(
        MergeTrigger trigger, SegmentInfos infos, MergeContext ctx) throws IOException {
      // MergePolicy's default, over this policy's own findMerges (FilterMergePolicy would
      // delegate to the wrapped policy's, which ignores the trigger).
      MergeSpecification spec = findMerges(trigger, infos, ctx);
      if (spec == null) {
        return null;
      }
      MergeSpecification out = null;
      for (OneMerge merge : spec.merges) {
        boolean below = true;
        for (SegmentCommitInfo sci : merge.segments) {
          if (size(sci, ctx) >= maxFullFlushMergeSize()) {
            below = false;
            break;
          }
        }
        if (below) {
          if (out == null) {
            out = new MergeSpecification();
          }
          out.add(merge);
        }
      }
      return out;
    }

    @Override
    protected long maxFullFlushMergeSize() {
      return (long) (((TieredMergePolicy) in).getFloorSegmentMB() * 1024 * 1024);
    }
  }

  static void run(Path out, long waitMillis) throws IOException {
    if (Files.exists(out)) {
      try (var walk = Files.walk(out)) {
        walk.sorted((a, b) -> b.compareTo(a)).forEach(p -> p.toFile().delete());
      }
    }
    Files.createDirectories(out);
    List<String> lines = new ArrayList<>();
    try (Directory dir = FSDirectory.open(out)) {
      TieredMergePolicy tmp = new TieredMergePolicy();
      tmp.setSegmentsPerTier(2);
      tmp.setMaxMergeAtOnce(2);
      tmp.setNoCFSRatio(0.0);
      IndexWriterConfig cfg = new IndexWriterConfig();
      cfg.setUseCompoundFile(false);
      cfg.setMergePolicy(new FullFlushOnly(tmp));
      cfg.setMergeScheduler(new SerialMergeScheduler());
      cfg.setMaxFullFlushMergeWaitMillis(waitMillis);
      int next = 0;
      try (IndexWriter w = new IndexWriter(dir, cfg)) {
        for (String step : STEPS) {
          int n = Integer.parseInt(step.substring(1));
          for (int i = 0; i < n; i++) {
            Document doc = new Document();
            doc.add(new StoredField("id", String.format("d%03d", next++)));
            w.addDocument(doc);
            w.flush();
          }
          if (step.charAt(0) == 'c') {
            w.commit();
            SegmentInfos sis = SegmentInfos.readLatestCommit(dir);
            StringBuilder sb = new StringBuilder("commit ").append(sis.getGeneration());
            for (SegmentCommitInfo sci : sis) {
              sb.append(' ').append(sci.info.name).append(':').append(sci.info.maxDoc());
            }
            lines.add(sb.toString());
          } else {
            try (DirectoryReader r = DirectoryReader.open(w)) {
              StringBuilder sb = new StringBuilder("reader");
              for (LeafReaderContext leaf : r.leaves()) {
                SegmentCommitInfo sci = ((SegmentReader) leaf.reader()).getSegmentInfo();
                sb.append(' ').append(sci.info.name).append(':').append(sci.info.maxDoc());
              }
              lines.add(sb.toString());
            }
          }
        }
      }
      try (DirectoryReader r = DirectoryReader.open(dir)) {
        StringBuilder sb = new StringBuilder("docs");
        for (LeafReaderContext leaf : r.leaves()) {
          var stored = leaf.reader().storedFields();
          for (int d = 0; d < leaf.reader().maxDoc(); d++) {
            sb.append(' ').append(stored.document(d).get("id"));
          }
        }
        lines.add(sb.toString());
      }
    }
    Files.write(out.resolve("manifest.txt"), lines, StandardCharsets.UTF_8);
  }
}
