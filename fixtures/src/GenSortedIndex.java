import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.NumericDocValuesField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.NumericDocValues;
import org.apache.lucene.index.SegmentCommitInfo;
import org.apache.lucene.index.SegmentInfos;
import org.apache.lucene.search.Sort;
import org.apache.lucene.search.SortField;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * Generates a real <b>index-sorted</b> index: an {@code IndexWriter} configured
 * with {@code IndexWriterConfig.setIndexSort} over a two-field {@code Sort},
 * whose first tier is <b>reversed</b> and whose second tier breaks its ties.
 *
 * <p>This is the read-direction fixture for two things this port has to get
 * exactly right and that nothing else pins together:
 *
 * <ol>
 *   <li>the {@code .si}'s {@code numSortFields} block for a real
 *       multi-field sort ({@code SortFieldProvider} bytes -- {@code b11} proved
 *       the encoding, this proves it against a sort a real {@code IndexWriter}
 *       chose rather than one a test handed to {@code SegmentInfo} directly);
 *   <li><b>what the sort actually means.</b> The missing value is an ordinary
 *       sentinel inside Lucene's comparator, so a reversed sort moves the
 *       missing documents to the <i>other</i> end. The manifest records the
 *       physical order Lucene put the documents in, so the Rust comparator is
 *       checked against Lucene's behaviour rather than against a reading of
 *       its source.
 * </ol>
 *
 * <p>Two segments, deliberately: one flushed and then a second flushed and
 * force-merged with it, so the fixture carries both a <i>sort-on-flush</i>
 * segment and a <i>sort-preserving merge</i> segment.
 */
public class GenSortedIndex {
  /** {@code rank} descending, missing last (== {@code Long.MAX_VALUE}). */
  private static SortField rankSort() {
    SortField sf = new SortField("rank", SortField.Type.LONG, true);
    sf.setMissingValue(Long.MAX_VALUE);
    return sf;
  }

  /** {@code tie} ascending, missing first (== {@code Long.MIN_VALUE}). */
  private static SortField tieSort() {
    SortField sf = new SortField("tie", SortField.Type.LONG, false);
    sf.setMissingValue(Long.MIN_VALUE);
    return sf;
  }

  public static void main(String[] args) throws IOException {
    Path out = Path.of(args[0]).resolve("sorted_index");
    if (Files.exists(out)) {
      deleteRecursive(out);
    }
    Files.createDirectories(out);

    // id, rank, tie. A negative rank, a rank of 0, duplicate ranks that the
    // second tier has to break, three rows (so six documents, across the two
    // batches below) with *no* rank at all -- which a reversed missing-last
    // sort must place first, not last -- and two rows with no `tie`, which
    // the ascending missing-first second tier must place first within their
    // rank group.
    long[][] rows = {
      // {rank, tie}; a rank/tie of Long.MIN_VALUE means "field absent".
      {5, 1}, {5, 0}, {-3, 7}, {0, 2}, {Long.MIN_VALUE, 4},
      {12, 9}, {5, 2}, {Long.MIN_VALUE, 1}, {-3, 3}, {7, 0},
      {5, Long.MIN_VALUE}, {Long.MIN_VALUE, Long.MIN_VALUE},
    };

    try (Directory dir = FSDirectory.open(out)) {
      IndexWriterConfig cfg = new IndexWriterConfig();
      cfg.setUseCompoundFile(false);
      cfg.setIndexSort(new Sort(rankSort(), tieSort()));
      try (IndexWriter w = new IndexWriter(dir, cfg)) {
        for (int i = 0; i < rows.length; i++) {
          w.addDocument(doc("a" + i, rows[i]));
        }
        w.commit();
        for (int i = 0; i < rows.length; i++) {
          // A second batch whose keys interleave with the first, so the
          // force-merge below is a genuine k-way merge and not a
          // concatenation that happens to be ordered.
          long rank = rows[i][0] == Long.MIN_VALUE ? Long.MIN_VALUE : rows[i][0] + 1;
          w.addDocument(doc("b" + i, new long[] {rank, rows[i][1]}));
        }
        w.commit();
        w.forceMerge(1);
        w.commit();
      }

      StringBuilder m = new StringBuilder();
      SegmentInfos sis = SegmentInfos.readLatestCommit(dir);
      m.append("segments_file_name=").append(sis.getSegmentsFileName()).append('\n');
      m.append("num_segments=").append(sis.size()).append('\n');
      List<String> names = new ArrayList<>();
      for (SegmentCommitInfo sci : sis) {
        names.add(sci.info.name);
        m.append("segment.").append(sci.info.name).append(".id_hex=")
            .append(hex(sci.info.getId())).append('\n');
        m.append("segment.").append(sci.info.name).append(".max_doc=")
            .append(sci.info.maxDoc()).append('\n');
        m.append("segment.").append(sci.info.name).append(".sort=")
            .append(sci.info.getIndexSort()).append('\n');
      }
      m.append("segment_names=").append(String.join(",", names)).append('\n');

      try (DirectoryReader reader = DirectoryReader.open(dir)) {
        if (reader.leaves().size() != 1) {
          throw new IllegalStateException("expected one leaf after forceMerge(1)");
        }
        LeafReaderContext ctx = reader.leaves().get(0);
        m.append("leaf_sort=").append(ctx.reader().getMetaData().sort()).append('\n');
        m.append("max_doc=").append(ctx.reader().maxDoc()).append('\n');

        // The physical order Lucene put the documents in, and the two sort
        // columns as a reader sees them. The Rust side asserts its own
        // comparator reproduces this exact order from these exact columns.
        StringBuilder ids = new StringBuilder();
        for (int d = 0; d < ctx.reader().maxDoc(); d++) {
          if (d > 0) {
            ids.append(',');
          }
          ids.append(ctx.reader().storedFields().document(d).get("id"));
        }
        m.append("docs_in_order=").append(ids).append('\n');
        m.append("rank_column=").append(numericColumn(ctx, "rank")).append('\n');
        m.append("tie_column=").append(numericColumn(ctx, "tie")).append('\n');
      }

      Files.writeString(out.resolve("manifest.properties"), m.toString());
    }

    System.out.println("wrote sorted_index/ fixture directory");
  }

  private static Document doc(String id, long[] row) {
    Document d = new Document();
    d.add(new StringField("id", id, Field.Store.YES));
    if (row[0] != Long.MIN_VALUE) {
      d.add(new NumericDocValuesField("rank", row[0]));
    }
    if (row[1] != Long.MIN_VALUE) {
      d.add(new NumericDocValuesField("tie", row[1]));
    }
    return d;
  }

  /** Every doc's value, comma separated; an absent value is the empty string. */
  private static String numericColumn(LeafReaderContext ctx, String field) throws IOException {
    String[] cells = new String[ctx.reader().maxDoc()];
    java.util.Arrays.fill(cells, "");
    NumericDocValues values = ctx.reader().getNumericDocValues(field);
    if (values != null) {
      for (int doc = values.nextDoc();
          doc != NumericDocValues.NO_MORE_DOCS;
          doc = values.nextDoc()) {
        cells[doc] = Long.toString(values.longValue());
      }
    }
    return String.join(",", cells);
  }

  private static String hex(byte[] b) {
    StringBuilder sb = new StringBuilder(b.length * 2);
    for (byte x : b) {
      sb.append(String.format("%02x", x));
    }
    return sb.toString();
  }

  private static void deleteRecursive(Path p) throws IOException {
    try (var walk = Files.walk(p)) {
      walk.sorted(java.util.Comparator.reverseOrder()).forEach(f -> {
        try {
          Files.delete(f);
        } catch (IOException e) {
          throw new RuntimeException(e);
        }
      });
    }
  }
}
