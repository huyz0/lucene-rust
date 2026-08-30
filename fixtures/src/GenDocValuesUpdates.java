import org.apache.lucene.document.BinaryDocValuesField;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.NumericDocValuesField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.BinaryDocValues;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.NumericDocValues;
import org.apache.lucene.index.SegmentCommitInfo;
import org.apache.lucene.index.SegmentInfos;
import org.apache.lucene.index.Term;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.BytesRef;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;

/**
 * Generates a real index whose doc-values have been <b>updated in place</b>
 * through {@code IndexWriter.updateNumericDocValue} /
 * {@code updateBinaryDocValue}, across four update rounds.
 *
 * <p>This is the read-direction fixture for the generation-suffixed doc-values
 * files Lucene writes for a field update: {@code _0_<base36 gen>_Lucene90_0
 * .dvm/.dvd/.dvs} holding the field's whole rewritten column, plus a
 * {@code FieldInfos} generation {@code _0_<base36 gen>.fnm} carrying the
 * field's new {@code FieldInfo.docValuesGen}, plus the {@code docValuesGen} and
 * per-field {@code dvUpdatesFiles} entries in {@code segments_N}. Three fields
 * so the fixture covers all three states at once: {@code val} (NUMERIC,
 * updated twice and then partly <b>reset</b>, so its newest generation is a
 * genuinely sparse {@code IndexedDISI}-backed column), {@code tag} (BINARY,
 * updated once, at a <i>different</i> generation number than {@code val}) and
 * {@code keep} (NUMERIC, never updated, so it stays on the base column at
 * generation -1 -- the case a reader gets wrong by resolving *every* field to
 * the newest generation).
 *
 * <p>The manifest records what Rust must read back for every document, so the
 * differential test asserts against Lucene's own answers rather than against a
 * re-derivation of the format.
 */
public class GenDocValuesUpdates {
  private static final int NUM_DOCS = 100;

  /** Documents {@code 0..NUM_RESET} have their {@code val} removed in the last round. */
  private static final int NUM_RESET = 30;

  public static void main(String[] args) throws IOException {
    Path out = Path.of(args[0]).resolve("doc_values_updates_index");
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
          doc.add(new StringField("id", "doc" + i, Field.Store.YES));
          doc.add(new StringField("body", i % 2 == 0 ? "even" : "odd", Field.Store.NO));
          doc.add(new NumericDocValuesField("val", i));
          doc.add(new BinaryDocValuesField("tag", new BytesRef("base-" + i)));
          doc.add(new NumericDocValuesField("keep", 1000 + i));
          w.addDocument(doc);
        }
        w.commit();

        w.updateNumericDocValue(new Term("body", "even"), "val", 1_000L);
        w.commit();

        w.updateBinaryDocValue(new Term("body", "odd"), "tag", new BytesRef("updated"));
        w.commit();

        w.updateNumericDocValue(new Term("body", "even"), "val", 7_000L);
        w.commit();

        // Fourth round: *remove* the first NUM_RESET documents' `val`.
        // `updateDocValues` with a null-valued field is
        // `DocValuesFieldUpdates.reset(doc)`, and it is the only thing that
        // makes a rewritten column genuinely sparse -- so this is what puts an
        // `IndexedDISI` structure (rather than the dense marker) into the
        // generation the Rust side has to decode.
        for (int i = 0; i < NUM_RESET; i++) {
          w.updateDocValues(new Term("id", "doc" + i), new NumericDocValuesField("val", null));
        }
        w.commit();
      }

      StringBuilder m = new StringBuilder();
      SegmentInfos sis = SegmentInfos.readLatestCommit(dir);
      if (sis.size() != 1) {
        throw new IllegalStateException("expected exactly one segment, got " + sis.size());
      }
      SegmentCommitInfo sci = sis.info(0);
      m.append("segments_file_name=").append(sis.getSegmentsFileName()).append('\n');
      m.append("segment_name=").append(sci.info.name).append('\n');
      m.append("id_hex=").append(hex(sci.info.getId())).append('\n');
      m.append("max_doc=").append(sci.info.maxDoc()).append('\n');
      m.append("doc_values_gen=").append(sci.getDocValuesGen()).append('\n');
      m.append("field_infos_gen=").append(sci.getFieldInfosGen()).append('\n');
      m.append("field_infos_files=").append(String.join(",", sorted(sci.getFieldInfosFiles())))
          .append('\n');

      Map<Integer, Set<String>> dvFiles = new TreeMap<>(sci.getDocValuesUpdatesFiles());
      List<String> keys = new ArrayList<>();
      for (Map.Entry<Integer, Set<String>> e : dvFiles.entrySet()) {
        keys.add(String.valueOf(e.getKey()));
        m.append("dv_update_files.")
            .append(e.getKey())
            .append('=')
            .append(String.join(",", sorted(e.getValue())))
            .append('\n');
      }
      m.append("dv_update_fields=").append(String.join(",", keys)).append('\n');

      // Field numbers and per-field generations, read from the newest
      // FieldInfos -- the one a reader must actually consult.
      try (DirectoryReader reader = DirectoryReader.open(dir)) {
        LeafReaderContext ctx = reader.leaves().get(0);
        for (FieldInfo fi : ctx.reader().getFieldInfos()) {
          m.append("field_number.").append(fi.name).append('=').append(fi.number).append('\n');
          m.append("field_dv_gen.")
              .append(fi.name)
              .append('=')
              .append(fi.getDocValuesGen())
              .append('\n');
        }

        m.append("expected_val=").append(numericColumn(ctx, "val")).append('\n');
        m.append("expected_keep=").append(numericColumn(ctx, "keep")).append('\n');
        m.append("expected_tag=").append(binaryColumn(ctx, "tag")).append('\n');
      }

      Files.writeString(out.resolve("manifest.properties"), m.toString());
    }

    System.out.println("wrote doc_values_updates_index/ fixture directory");
  }

  /** Every doc's value, comma separated; an absent value is the empty string. */
  private static String numericColumn(LeafReaderContext ctx, String field) throws IOException {
    String[] cells = new String[ctx.reader().maxDoc()];
    java.util.Arrays.fill(cells, "");
    NumericDocValues values = ctx.reader().getNumericDocValues(field);
    if (values != null) {
      for (int doc = values.nextDoc();
          doc != org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;
          doc = values.nextDoc()) {
        cells[doc] = Long.toString(values.longValue());
      }
    }
    return String.join(",", cells);
  }

  private static String binaryColumn(LeafReaderContext ctx, String field) throws IOException {
    String[] cells = new String[ctx.reader().maxDoc()];
    java.util.Arrays.fill(cells, "");
    BinaryDocValues values = ctx.reader().getBinaryDocValues(field);
    if (values != null) {
      for (int doc = values.nextDoc();
          doc != org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;
          doc = values.nextDoc()) {
        BytesRef v = values.binaryValue();
        cells[doc] = new String(v.bytes, v.offset, v.length, StandardCharsets.UTF_8);
      }
    }
    return String.join(",", cells);
  }

  private static List<String> sorted(Set<String> in) {
    List<String> out = new ArrayList<>(in);
    java.util.Collections.sort(out);
    return out;
  }

  private static String hex(byte[] bytes) {
    StringBuilder sb = new StringBuilder(bytes.length * 2);
    for (byte b : bytes) {
      sb.append(String.format("%02x", b));
    }
    return sb.toString();
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
