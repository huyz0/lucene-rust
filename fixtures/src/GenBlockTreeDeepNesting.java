import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.PostingsFormat;
import org.apache.lucene.codecs.lucene104.Lucene104Codec;
import org.apache.lucene.codecs.lucene104.Lucene104PostingsFormat;
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
import java.util.Random;
import java.util.TreeSet;

/**
 * Generates a real `.tim`/`.tip`/`.tmd` fixture engineered to force real Lucene
 * to write a `.tim` block tree that is **4+ levels deep** -- two or more
 * layers of non-leaf ("internal") blocks stacked between the root block and
 * the leaf blocks, not just the single layer of non-leaf nesting that
 * `GenBlockTreeMultilevel.java`'s 8000-random-term/default-block-size fixture
 * produces (empirically, root -> one internal layer -> leaves only, see that
 * class's own module doc and
 * `crates/lucene-codecs/tests/blocktree_multilevel_fixture.rs`).
 *
 * <p>Two levers control how deep {@code Lucene103BlockTreeTermsWriter}'s
 * recursive block-splitting goes for a shared prefix: (1) {@code
 * minItemsInBlock}/{@code maxItemsInBlock} (writer-configurable via {@link
 * Lucene104PostingsFormat#Lucene104PostingsFormat(int, int)} -- default 25/48,
 * see {@code Lucene103BlockTreeTermsWriter.DEFAULT_MIN_BLOCK_SIZE}/{@code
 * DEFAULT_MAX_BLOCK_SIZE}) -- smaller thresholds mean any given prefix group
 * splits into sub-blocks sooner; and (2) the term alphabet's branching factor
 * at each byte position -- a wide alphabet (e.g. all 26 lowercase letters, as
 * `GenBlockTreeMultilevel` uses) fans a large group out into many small
 * children after just one extra prefix byte, capping nesting depth at ~3
 * levels no matter how many terms are added (empirically verified: 2000,
 * 3000, 5000, and 8000 random 26-letter terms at {@code minItemsInBlock=2},
 * {@code maxItemsInBlock=4} all plateaued at depth 3). A **narrow** alphabet
 * (here, just {@code 'a'}/{@code 'b'}) makes each extra prefix byte only
 * halve a group's size instead of dividing it by 26, so with the same small
 * block-size thresholds a group needs many more extra prefix bytes -- and
 * therefore many more recursion levels -- before every child shrinks below
 * {@code maxItemsInBlock}. Empirically, 2000 distinct 16-byte strings over
 * {@code {a,b}} with {@code minItemsInBlock=2}/{@code maxItemsInBlock=4}
 * reaches a genuine depth of 6 non-leaf-chained physical `.tim` blocks
 * (verified with a throwaway depth-probing walk against this exact
 * generator's output before this fixture was made permanent) -- comfortably
 * past the 4-level bar this fixture exists to clear, with margin against
 * incidental fixture-regen variance.
 *
 * <p>See `crates/lucene-codecs/src/blocktree.rs`'s
 * `deep_nesting_fixture_reaches_at_least_four_levels` test for this
 * fixture's structural ground truth (independently re-deriving the actual
 * chained-sub-block nesting depth, not just "some non-leaf block exists"),
 * and `crates/lucene-codecs/tests/blocktree_deep_nesting_fixture.rs` for the
 * full public-API differential (every term still findable, matching this
 * fixture's own manifest).
 */
public class GenBlockTreeDeepNesting {

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
    Path out = Path.of(args[0]).resolve("blocktree_deep_nesting_index");
    if (Files.exists(out)) {
      deleteRecursive(out);
    }
    Files.createDirectories(out);

    FieldType idType = new FieldType();
    idType.setIndexOptions(IndexOptions.DOCS);
    idType.setTokenized(true);
    idType.freeze();

    // Fixed seed: this fixture must be byte-for-byte reproducible on every
    // regen, same as every other generator in this directory.
    final int numTerms = 2000;
    final int termLen = 16;
    final String alphabet = "ab";
    final int minItemsInBlock = 2;
    final int maxItemsInBlock = 4;
    final long seed = 12345L;
    Random rnd = new Random(seed);
    TreeSet<String> distinct = new TreeSet<>();
    while (distinct.size() < numTerms) {
      StringBuilder sb = new StringBuilder(termLen);
      for (int i = 0; i < termLen; i++) {
        sb.append(alphabet.charAt(rnd.nextInt(alphabet.length())));
      }
      distinct.add(sb.toString());
    }
    String[] terms = distinct.toArray(new String[0]);

    // Custom min/max block size, wired through a Codec subclass rather than
    // relying on Lucene104PostingsFormat's SPI-registered default-settings
    // constructor -- see `Lucene104PostingsFormat`'s two-arg constructor doc.
    Codec codec =
        new Lucene104Codec() {
          @Override
          public PostingsFormat getPostingsFormatForField(String field) {
            return new Lucene104PostingsFormat(minItemsInBlock, maxItemsInBlock);
          }
        };

    try (Directory dir = FSDirectory.open(out)) {
      IndexWriterConfig cfg = new IndexWriterConfig();
      cfg.setUseCompoundFile(false);
      cfg.setMergePolicy(NoMergePolicy.INSTANCE);
      cfg.setCodec(codec);

      try (IndexWriter w = new IndexWriter(dir, cfg)) {
        for (String term : terms) {
          Document doc = new Document();
          doc.add(new Field("many", new CannedTokenStream(List.of(term)), idType));
          w.addDocument(doc);
        }
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
        Terms leafTerms = leaf.terms("many");
        if (leafTerms == null) {
          throw new AssertionError("expected terms for field many");
        }
        m.append("field.many.numTerms=").append(leafTerms.size()).append('\n');

        // Ground truth for every single one of the numTerms terms
        // (docFreq/totalTermFreq are always 1/1 for this field -- one
        // distinct token per doc -- so the manifest only needs to record
        // the sorted term list itself for the Rust side to check against).
        StringBuilder allTerms = new StringBuilder();
        TermsEnum te = leafTerms.iterator();
        BytesRef term;
        int count = 0;
        while ((term = te.next()) != null) {
          allTerms.append(term.utf8ToString()).append('\n');
          count++;
        }
        if (count != numTerms) {
          throw new AssertionError("expected " + numTerms + " terms, got " + count);
        }
        Files.writeString(out.resolve("many.terms.tsv"), allTerms.toString());
        m.append("field.many.termsFile=many.terms.tsv\n");
      }

      Files.writeString(out.resolve("manifest.properties"), m.toString());
    }

    System.out.println(
        "wrote blocktree_deep_nesting_index/ fixture directory (" + terms.length + " terms)");
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
