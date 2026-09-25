import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.PostingsEnum;
import org.apache.lucene.index.TermsEnum;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.BytesRef;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Cross-engine ground truth for {@code PostingsEnum.docIDRunEnd()} -- what
 * {@code ReqExclBulkScorer} skips a whole run of excluded documents with --
 * appended to {@code postings_skip_index}'s manifest without regenerating it (see
 * {@link AppendScoringManifest} for why).
 *
 * <p>{@code skipterm} is in all 8,500 documents: its full level-0 blocks are 256
 * consecutive documents (Lucene104's unary bit-set encoding, all ones) and its first
 * level-1 span is dense too, so a run ends at the span's end, 8192. Its tail block is
 * not a bit set and ends every run at {@code doc + 1}. {@code gapterm} has no dense
 * block at all.
 *
 * <p>Each key is {@code runend.TERM.TARGET=LANDED:RUN_END} for a fresh enum's
 * {@code advance(TARGET)}, with and without frequencies.
 */
public class AppendRunEndManifest {
  static final int[] TARGETS = {0, 1, 255, 256, 300, 4096, 8191, 8192, 8200, 8499};

  public static void main(String[] args) throws IOException {
    Path indexDir = Path.of(args[0]).resolve("postings_skip_index");
    Path manifestPath = indexDir.resolve("manifest.properties");
    StringBuilder out = new StringBuilder();
    try (Directory dir = FSDirectory.open(indexDir);
        DirectoryReader reader = DirectoryReader.open(dir)) {
      LeafReader leaf = reader.leaves().get(0).reader();
      for (String term : new String[] {"skipterm", "gapterm"}) {
        for (int flags : new int[] {PostingsEnum.NONE, PostingsEnum.FREQS}) {
          String name = flags == PostingsEnum.NONE ? "docs" : "freqs";
          for (int target : TARGETS) {
            TermsEnum te = leaf.terms("pskip").iterator();
            if (!te.seekExact(new BytesRef(term))) {
              throw new AssertionError("missing " + term);
            }
            PostingsEnum pe = te.postings(null, flags);
            int landed = pe.advance(target);
            int runEnd = landed == DocIdSetIterator.NO_MORE_DOCS ? -1 : pe.docIDRunEnd();
            out.append("runend.")
                .append(term)
                .append('.')
                .append(name)
                .append('.')
                .append(target)
                .append('=')
                .append(landed)
                .append(':')
                .append(runEnd)
                .append('\n');
          }
        }
      }
    }

    String existing = Files.readString(manifestPath);
    StringBuilder kept = new StringBuilder();
    for (String line : existing.split("\n", -1)) {
      if (line.startsWith("runend.")) {
        continue;
      }
      kept.append(line).append('\n');
    }
    String base = kept.toString();
    while (base.endsWith("\n\n")) {
      base = base.substring(0, base.length() - 1);
    }
    Files.writeString(manifestPath, base + out);
    System.out.println("appended runend.* ground truth to " + manifestPath);
  }
}
