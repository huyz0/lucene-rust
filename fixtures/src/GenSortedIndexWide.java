import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.NumericDocValuesField;
import org.apache.lucene.document.SortedDocValuesField;
import org.apache.lucene.document.SortedNumericDocValuesField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.NumericDocValues;
import org.apache.lucene.index.SegmentCommitInfo;
import org.apache.lucene.index.SegmentInfos;
import org.apache.lucene.index.SortedDocValues;
import org.apache.lucene.index.SortedNumericDocValues;
import org.apache.lucene.search.Sort;
import org.apache.lucene.search.SortField;
import org.apache.lucene.search.SortedNumericSelector;
import org.apache.lucene.search.SortedNumericSortField;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.BytesRef;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * Generates a real index-sorted index whose {@code Sort} is deliberately
 * <b>outside</b> what {@code segment_info::IndexSortField} could represent
 * before batch {@code c35}, so that opening it at all is the thing under test.
 *
 * <p>{@code GenSortedIndex} covers the shape this port itself writes: two
 * {@code SortField(field, LONG, reverse)} tiers whose missing values are
 * exactly {@code Long.MIN_VALUE}/{@code Long.MAX_VALUE}. That was the whole of
 * the old model, and {@code parse} rejected everything else with
 * {@code Error::UnsupportedSortField} -- honestly, but the consequence was
 * that an index a real {@code IndexWriter} wrote with an ordinary sort could
 * not be opened by this port at all.
 *
 * <p>Each of the three tiers here was individually unrepresentable:
 *
 * <ol>
 *   <li>{@code rank}: {@code SortField(LONG, reverse)} with an
 *       <b>arbitrary</b> missing value ({@code 42}), which is neither the
 *       sort-first nor the sort-last sentinel. It is also a sentinel that sits
 *       in the middle of the data, so a document with no {@code rank} lands
 *       between documents that have one -- something a "missing first or last"
 *       model cannot express even approximately.
 *   <li>{@code multi}: a {@code SortedNumericSortField} with the
 *       <b>{@code MAX} selector</b> and <b>no missing value at all</b> (Java
 *       then sorts such a document as if it held {@code 0}). The old model had
 *       no selector, so it refused any selector but {@code MIN}, and no way to
 *       say "no missing value".
 *   <li>{@code name}: a {@code SortField(STRING, reverse)} over
 *       {@code SortedDocValues}, compared by <b>term ordinal</b>. The old
 *       model could lower its missing value onto first/last but nothing
 *       downstream could read an ordinal column as a sort key.
 * </ol>
 *
 * <p>Two commits and a {@code forceMerge(1)}, as in {@code GenSortedIndex}, so
 * the fixture is a genuine sort-preserving merge rather than a single flush.
 *
 * <p>The manifest records the physical document order Lucene chose and all
 * three sort columns as a reader sees them, so the Rust side checks its
 * comparator against Lucene's behaviour rather than against a reading of
 * Lucene's source.
 */
public class GenSortedIndexWide {
  /** {@code rank} descending, missing value 42 -- an ordinary long, not a sentinel. */
  private static SortField rankSort() {
    SortField sf = new SortField("rank", SortField.Type.LONG, true);
    sf.setMissingValue(42L);
    return sf;
  }

  /** {@code multi} ascending, MAX selector, no missing value (Java uses 0). */
  private static SortField multiSort() {
    return new SortedNumericSortField(
        "multi", SortField.Type.INT, false, SortedNumericSelector.Type.MAX);
  }

  /** {@code name} descending by term ordinal, missing first. */
  private static SortField nameSort() {
    SortField sf = new SortField("name", SortField.Type.STRING, true);
    sf.setMissingValue(SortField.STRING_FIRST);
    return sf;
  }

  /**
   * {@code {rank, multiA, multiB, nameIndex}}. {@code Long.MIN_VALUE} means
   * "field absent"; {@code nameIndex} indexes into {@link #NAMES}.
   */
  private static final long[][] ROWS = {
    {5, 1, 3, 0},
    {5, 4, 4, 1},
    {-3, 7, Long.MIN_VALUE, 2},
    {0, 2, 9, Long.MIN_VALUE},
    {Long.MIN_VALUE, 4, 1, 0},
    {12, 9, 2, 3},
    {5, 2, 2, 1},
    {Long.MIN_VALUE, 1, 8, Long.MIN_VALUE},
    {-3, 3, 3, 2},
    {42, 0, 0, 3},
    {5, Long.MIN_VALUE, Long.MIN_VALUE, 0},
    {Long.MIN_VALUE, Long.MIN_VALUE, Long.MIN_VALUE, Long.MIN_VALUE},
  };

  private static final String[] NAMES = {"alpha", "bravo", "charlie", "delta"};

  public static void main(String[] args) throws IOException {
    Path out = Path.of(args[0]).resolve("sorted_index_wide");
    if (Files.exists(out)) {
      deleteRecursive(out);
    }
    Files.createDirectories(out);

    try (Directory dir = FSDirectory.open(out)) {
      IndexWriterConfig cfg = new IndexWriterConfig();
      cfg.setUseCompoundFile(false);
      cfg.setIndexSort(new Sort(rankSort(), multiSort(), nameSort()));
      try (IndexWriter w = new IndexWriter(dir, cfg)) {
        for (int i = 0; i < ROWS.length; i++) {
          w.addDocument(doc("a" + i, ROWS[i]));
        }
        w.commit();
        for (int i = 0; i < ROWS.length; i++) {
          // Keys that interleave with the first batch, so the force-merge is a
          // real k-way merge and not a concatenation that happens to be
          // ordered.
          long[] row = ROWS[i].clone();
          if (row[0] != Long.MIN_VALUE) {
            row[0] = row[0] + 1;
          }
          w.addDocument(doc("b" + i, row));
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

        StringBuilder ids = new StringBuilder();
        for (int d = 0; d < ctx.reader().maxDoc(); d++) {
          if (d > 0) {
            ids.append(',');
          }
          ids.append(ctx.reader().storedFields().document(d).get("id"));
        }
        m.append("docs_in_order=").append(ids).append('\n');
        m.append("rank_column=").append(numericColumn(ctx, "rank")).append('\n');
        m.append("multi_column=").append(sortedNumericColumn(ctx, "multi")).append('\n');
        m.append("name_ord_column=").append(sortedOrdColumn(ctx, "name")).append('\n');
        m.append("name_dictionary=").append(sortedDictionary(ctx, "name")).append('\n');
      }

      Files.writeString(out.resolve("manifest.properties"), m.toString());
    }

    System.out.println("wrote sorted_index_wide/ fixture directory");
  }

  private static Document doc(String id, long[] row) {
    Document d = new Document();
    d.add(new StringField("id", id, Field.Store.YES));
    if (row[0] != Long.MIN_VALUE) {
      d.add(new NumericDocValuesField("rank", row[0]));
    }
    if (row[1] != Long.MIN_VALUE) {
      d.add(new SortedNumericDocValuesField("multi", row[1]));
    }
    if (row[2] != Long.MIN_VALUE) {
      d.add(new SortedNumericDocValuesField("multi", row[2]));
    }
    if (row[3] != Long.MIN_VALUE) {
      d.add(new SortedDocValuesField("name", new BytesRef(NAMES[(int) row[3]])));
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

  /** Every doc's values, space separated within a doc and comma separated across docs. */
  private static String sortedNumericColumn(LeafReaderContext ctx, String field)
      throws IOException {
    String[] cells = new String[ctx.reader().maxDoc()];
    java.util.Arrays.fill(cells, "");
    SortedNumericDocValues values = ctx.reader().getSortedNumericDocValues(field);
    if (values != null) {
      for (int doc = values.nextDoc();
          doc != SortedNumericDocValues.NO_MORE_DOCS;
          doc = values.nextDoc()) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < values.docValueCount(); i++) {
          if (i > 0) {
            sb.append(' ');
          }
          sb.append(values.nextValue());
        }
        cells[doc] = sb.toString();
      }
    }
    return String.join(",", cells);
  }

  /** Every doc's term ordinal; an absent value is the empty string. */
  private static String sortedOrdColumn(LeafReaderContext ctx, String field) throws IOException {
    String[] cells = new String[ctx.reader().maxDoc()];
    java.util.Arrays.fill(cells, "");
    SortedDocValues values = ctx.reader().getSortedDocValues(field);
    if (values != null) {
      for (int doc = values.nextDoc();
          doc != SortedDocValues.NO_MORE_DOCS;
          doc = values.nextDoc()) {
        cells[doc] = Integer.toString(values.ordValue());
      }
    }
    return String.join(",", cells);
  }

  /** The SORTED field's dictionary, ordinal order, comma separated. */
  private static String sortedDictionary(LeafReaderContext ctx, String field) throws IOException {
    SortedDocValues values = ctx.reader().getSortedDocValues(field);
    if (values == null) {
      return "";
    }
    List<String> terms = new ArrayList<>();
    for (int ord = 0; ord < values.getValueCount(); ord++) {
      BytesRef term = values.lookupOrd(ord);
      terms.add(new String(term.bytes, term.offset, term.length, StandardCharsets.UTF_8));
    }
    return String.join(",", terms);
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
