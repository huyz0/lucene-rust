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
import java.util.Random;
import java.util.TreeSet;

/**
 * Generates a real `.tim`/`.tip`/`.tmd` (Lucene103BlockTreeTermsWriter, via
 * Lucene104PostingsFormat) fixture large/varied enough to force a genuine
 * **multi-level blocktree**: a `.tim` block that is itself non-leaf (some of
 * its entries are pointers to further-nested sub-blocks, `isLeafBlock ==
 * false`), not just a deeper `.tip` trie or more floor sub-blocks under one
 * node -- see `crates/lucene-codecs/src/blocktree.rs`'s module doc for the
 * three distinct "more than one block" mechanisms this format has, and why
 * this one is the one that was still unimplemented before this fixture
 * existed.
 *
 * <p>{@code GenBlockTree}'s own "many" field (400 terms, `"term0000".."term0399"`,
 * a sequential zero-padded numeric suffix) already forces multiple `.tim`
 * blocks and a multi-child `.tip` trie, but never a non-leaf block: every
 * over-large group under that field's writer resolved via *floor* sub-blocks
 * (same trie node, several physical blocks) rather than a deeper, separately
 * unindexed sub-block. Empirically (see this class's git history / the task
 * that added it), forcing an actual non-leaf `.tim` block needs a term set
 * with genuinely irregular, deep branching -- a purely sequential numeric
 * suffix doesn't produce one no matter how large. 8000 pseudo-random
 * lowercase strings (4-12 bytes, drawn from a fixed-seed `java.util.Random`
 * so this fixture is exactly reproducible without any external word list)
 * does: real dictionary words were tried first and also produced a non-leaf
 * block, but a `Random`-seeded synthetic set keeps this generator
 * self-contained (no dependency on `/usr/share/dict/words` or similar being
 * present/identical across machines).
 *
 * <p>See `crates/lucene-codecs/tests/blocktree_multilevel_fixture.rs` for
 * this fixture's Rust-side ground truth and the structural (not just
 * "lookups still work") proof that a real non-leaf block was reached.
 */
public class GenBlockTreeMultilevel {

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
    Path out = Path.of(args[0]).resolve("blocktree_multilevel_index");
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
    final int numTerms = 8000;
    final long seed = 12345L;
    Random rnd = new Random(seed);
    String alphabet = "abcdefghijklmnopqrstuvwxyz";
    TreeSet<String> distinct = new TreeSet<>();
    while (distinct.size() < numTerms) {
      int len = 4 + rnd.nextInt(9); // 4..12 bytes
      StringBuilder sb = new StringBuilder(len);
      for (int i = 0; i < len; i++) {
        sb.append(alphabet.charAt(rnd.nextInt(alphabet.length())));
      }
      distinct.add(sb.toString());
    }
    String[] terms = distinct.toArray(new String[0]);

    try (Directory dir = FSDirectory.open(out)) {
      IndexWriterConfig cfg = new IndexWriterConfig();
      cfg.setUseCompoundFile(false);
      cfg.setMergePolicy(NoMergePolicy.INSTANCE);

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

    System.out.println("wrote blocktree_multilevel_index/ fixture directory (" + terms.length + " terms)");
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
