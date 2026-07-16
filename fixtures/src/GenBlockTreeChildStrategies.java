import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.FieldType;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.SegmentCommitInfo;
import org.apache.lucene.index.SegmentInfos;
import org.apache.lucene.index.Terms;
import org.apache.lucene.index.TermsEnum;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.BytesRef;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

/**
 * Generates real `.tim`/`.tip`/`.tmd` fixtures whose `.tip` trie **root node**
 * is deliberately shaped (via `TrieBuilder.ChildSaveStrategy.choose`'s own
 * cost formulas, `needBytes(minLabel, maxLabel, labelCnt)` for each of
 * `BITS`/`ARRAY`/`REVERSE_ARRAY`) to force real Lucene's writer to pick
 * `ChildSaveStrategy.ARRAY` for one field and `ChildSaveStrategy.BITS` for
 * another -- the two strategies `GenBlockTreeMultilevel`'s "many" field does
 * *not* exercise (that fixture's root node happens to land on
 * `REVERSE_ARRAY`, verified separately -- see
 * `crates/lucene-codecs/src/blocktree.rs`'s `multilevel_fixture_reaches_a_genuine_non_leaf_block`
 * test, which now also asserts the exact strategy code).
 *
 * <p>Each field's terms all share a distinct **leading byte** (one of a
 * hand-picked small set of ASCII bytes), with enough terms per leading byte
 * (30, comfortably above the default `minItemsInBlock=25`) that real
 * Lucene's writer gives each leading-byte group its own `.tim` block and
 * therefore an actual `.tip` trie child at the root for that label -- the
 * same mechanism `GenBlockTreeMultilevel`'s many distinct leading letters
 * already rely on to produce a multi-children root, just with a
 * *deliberately chosen* label set instead of a random one so the resulting
 * `(minLabel, maxLabel, labelCnt)` triple is arithmetically known in advance
 * to make one specific strategy win `ChildSaveStrategy.choose`'s
 * lowest-cost comparison:
 *
 * <ul>
 *   <li>"arraystrat": 5 labels spread across the full printable-ASCII range
 *       (0x21..0x7e, distance 94) -- {@code needBytes}: BITS=ceil(94/8)=12,
 *       ARRAY=5-1=4, REVERSE_ARRAY=94-5+1=90. ARRAY (4) wins outright.
 *   <li>"bitsstrat": 9 labels spaced 5 apart starting at 0x21 (distance 41)
 *       -- {@code needBytes}: BITS=ceil(41/8)=6, ARRAY=9-1=8,
 *       REVERSE_ARRAY=41-9+1=33. BITS (6) wins outright.
 * </ul>
 *
 * Both computations use exactly `TrieBuilder.ChildSaveStrategy`'s own
 * formulas (see that class's `needBytes` overrides), so this is a
 * self-checked construction, not a guess -- and the Rust differential test
 * for this fixture additionally asserts the decoded `child_save_strategy`
 * code equals the expected one, so any future Lucene cost-heuristic change
 * would fail loudly here rather than silently start testing the wrong
 * strategy.
 */
public class GenBlockTreeChildStrategies {

  static final class CannedTokenStream extends TokenStream {
    private final List<String> tokens;
    private int index = 0;
    private final CharTermAttribute termAtt = addAttribute(CharTermAttribute.class);

    CannedTokenStream(List<String> tokens) {
      this.tokens = tokens;
    }

    @Override
    public boolean incrementToken() {
      if (index >= tokens.size()) {
        return false;
      }
      clearAttributes();
      termAtt.append(tokens.get(index++));
      return true;
    }

    @Override
    public void reset() throws IOException {
      super.reset();
      index = 0;
    }
  }

  public static void main(String[] args) throws IOException {
    Path out = Path.of(args[0]).resolve("blocktree_child_strategies_index");
    if (Files.exists(out)) {
      deleteRecursive(out);
    }
    Files.createDirectories(out);

    FieldType idType = new FieldType();
    idType.setIndexOptions(IndexOptions.DOCS);
    idType.setTokenized(true);
    idType.freeze();

    final int perLabel = 30; // > default minItemsInBlock=25

    // ARRAY: 5 labels, distance 94 (full printable-ASCII span).
    int[] arrayLabels = {0x21, 0x35, 0x49, 0x61, 0x7e};
    // BITS: 9 labels, distance 41 (spaced 5 apart from 0x21).
    int[] bitsLabels = {0x21, 0x26, 0x2b, 0x30, 0x35, 0x3a, 0x3f, 0x44, 0x49};

    try (Directory dir = FSDirectory.open(out)) {
      IndexWriterConfig cfg = new IndexWriterConfig();
      cfg.setUseCompoundFile(false);
      cfg.setMergePolicy(NoMergePolicy.INSTANCE);

      try (IndexWriter w = new IndexWriter(dir, cfg)) {
        addLabelGroup(w, "arraystrat", arrayLabels, perLabel, idType);
        addLabelGroup(w, "bitsstrat", bitsLabels, perLabel, idType);
        w.commit();
      }

      SegmentInfos sis = SegmentInfos.readLatestCommit(dir);
      if (sis.size() != 1) {
        throw new AssertionError("expected exactly one segment, got " + sis.size());
      }
      SegmentCommitInfo sci = sis.info(0);

      String timFileName = null, tipFileName = null, tmdFileName = null;
      String fnmFileName = null, siFileName = null;
      for (String f : sci.info.files()) {
        if (f.endsWith(".tim")) timFileName = f;
        if (f.endsWith(".tip")) tipFileName = f;
        if (f.endsWith(".tmd")) tmdFileName = f;
        if (f.endsWith(".fnm")) fnmFileName = f;
        if (f.endsWith(".si")) siFileName = f;
      }
      if (timFileName == null || tipFileName == null || tmdFileName == null) {
        throw new AssertionError("expected .tim/.tip/.tmd files, files=" + sci.info.files());
      }
      if (fnmFileName == null || siFileName == null) {
        throw new AssertionError("expected .fnm/.si files, files=" + sci.info.files());
      }

      dump(dir, timFileName, out);
      dump(dir, tipFileName, out);
      dump(dir, tmdFileName, out);
      dump(dir, fnmFileName, out);
      dump(dir, siFileName, out);

      StringBuilder m = new StringBuilder();
      m.append("tim_file_name=").append(timFileName).append('\n');
      m.append("tip_file_name=").append(tipFileName).append('\n');
      m.append("tmd_file_name=").append(tmdFileName).append('\n');
      m.append("fnm_file_name=").append(fnmFileName).append('\n');
      m.append("si_file_name=").append(siFileName).append('\n');
      String prefix = sci.info.name + "_";
      String segmentSuffix = timFileName.substring(prefix.length(), timFileName.length() - 4);
      m.append("segment_name=").append(sci.info.name).append('\n');
      m.append("segment_suffix=").append(segmentSuffix).append('\n');
      m.append("id_hex=").append(hex(sci.info.getId())).append('\n');
      m.append("max_doc=").append(sci.info.maxDoc()).append('\n');

      try (DirectoryReader reader = DirectoryReader.open(dir)) {
        LeafReader leaf = reader.leaves().get(0).reader();
        for (String field : new String[] {"arraystrat", "bitsstrat"}) {
          Terms leafTerms = leaf.terms(field);
          if (leafTerms == null) {
            throw new AssertionError("expected terms for field " + field);
          }
          m.append("field.").append(field).append(".numTerms=").append(leafTerms.size()).append('\n');
          StringBuilder allTerms = new StringBuilder();
          TermsEnum te = leafTerms.iterator();
          BytesRef term;
          int count = 0;
          while ((term = te.next()) != null) {
            allTerms.append(term.utf8ToString()).append('\n');
            count++;
          }
          Files.writeString(out.resolve(field + ".terms.tsv"), allTerms.toString());
          m.append("field.").append(field).append(".termsFile=").append(field).append(".terms.tsv\n");
          m.append("field.").append(field).append(".count=").append(count).append('\n');
        }
      }

      Files.writeString(out.resolve("manifest.properties"), m.toString());
    }

    System.out.println("wrote blocktree_child_strategies_index/ fixture directory");
  }

  static void addLabelGroup(
      IndexWriter w, String field, int[] labels, int perLabel, FieldType idType)
      throws IOException {
    for (int label : labels) {
      for (int i = 0; i < perLabel; i++) {
        String term = "" + (char) label + String.format("%04d", i);
        Document doc = new Document();
        doc.add(new Field(field, new CannedTokenStream(List.of(term)), idType));
        w.addDocument(doc);
      }
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
    for (byte value : b) sb.append(String.format("%02x", value));
    return sb.toString();
  }
}
