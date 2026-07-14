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
 * Generates a real `.tim`/`.tip`/`.tmd` (Lucene103BlockTreeTermsWriter, via
 * Lucene104PostingsFormat) fixture whose term dictionary blocks actually get
 * LZ4-compressed by the real writer, plus the `.fnm`/`.si` this port's own
 * readers need to open it.
 *
 * <p>One field, "lz4field" (`IndexOptions.DOCS`), 200 distinct terms sharing a
 * long, highly-repetitive suffix -- `"commonprefixforcompression" +
 * "abcdabcd".repeat(6) + "%03d"` -- one document per term (docFreq=1,
 * totalTermFreq=1 for every term). Deliberately past the default
 * `minItemsInBlock=25`/`maxItemsInBlock=48` thresholds so the field's term
 * dictionary splits across multiple `.tim` blocks, and the shared
 * `"abcdabcd"` repetition is long/regular enough that
 * `Lucene103BlockTreeTermsWriter`'s LZ4 attempt (only tried when a block's
 * average suffix length exceeds 6 bytes and the shared prefix length exceeds
 * 2, see that class's `writeBlock`) actually saves more than 25%, so real
 * Lucene picks `CompressionAlgorithm.LZ4` for the compressible blocks rather
 * than falling back to `NO_COMPRESSION`.
 *
 * <p>See `crates/lucene-codecs/tests/blocktree_compressed_fixture.rs` for
 * this fixture's Rust-side ground truth (verbatim in expected_terms()) and
 * `crates/lucene-codecs/src/blocktree.rs`'s module doc for the compressed
 * suffix decode this exercises. `LOWERCASE_ASCII` could not be forced out of
 * a real `IndexWriter` in reasonable effort here either (same finding
 * recorded in that Rust test's module doc) -- this generator stays LZ4-only.
 */
public class GenBlockTreeCompressed {

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
    Path out = Path.of(args[0]).resolve("blocktree_compressed_index");
    if (Files.exists(out)) {
      deleteRecursive(out);
    }
    Files.createDirectories(out);

    FieldType idType = new FieldType();
    idType.setIndexOptions(IndexOptions.DOCS);
    idType.setTokenized(true);
    idType.freeze();

    // Digits FIRST, long repetitive filler AFTER: this matters because the
    // block-tree trie shares whatever *leading* bytes a group of sibling
    // terms have in common, and only the bytes *after* that shared prefix
    // are written as a block's compressible "suffix" bytes. A shared prefix
    // followed by varying digits (as tried first, and rejected -- see this
    // class's git history / commit message) leaves almost nothing in the
    // suffix once the digits are stripped out as the differentiator, so
    // every block falls under the writer's `suffixLength > 2 * numEntries`
    // "not worth compressing" gate. Putting the 3 varying digits right after
    // a short literal prefix ("id"), with the long repetitive filler
    // afterwards, means the *filler* itself ends up inside each block's
    // suffix bytes (since only ~2 sibling digits, not the filler, are ever
    // shared across a leaf block's terms) -- long, repetitive suffix bytes
    // are exactly what trips both the `suffixLength > 6 * numEntries`
    // average-length gate and LZ4's own opportunistic back-referencing.
    final String filler = "abcdabcd".repeat(8);
    final int numTerms = 200;
    String[] terms = new String[numTerms];
    for (int i = 0; i < numTerms; i++) {
      terms[i] = String.format("id%03d%s", i, filler);
    }

    try (Directory dir = FSDirectory.open(out)) {
      IndexWriterConfig cfg = new IndexWriterConfig();
      cfg.setUseCompoundFile(false);
      cfg.setMergePolicy(NoMergePolicy.INSTANCE);

      try (IndexWriter w = new IndexWriter(dir, cfg)) {
        for (String term : terms) {
          Document doc = new Document();
          doc.add(new Field("lz4field", new CannedTokenStream(List.of(term)), idType));
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
        org.apache.lucene.index.Terms leafTerms = leaf.terms("lz4field");
        if (leafTerms == null) {
          throw new AssertionError("expected terms for field lz4field");
        }
        m.append("field.lz4field.numTerms=").append(leafTerms.size()).append('\n');
      }

      Files.writeString(out.resolve("manifest.properties"), m.toString());
    }

    if (!terms[0].startsWith("id000")) {
      throw new AssertionError("sanity check failed");
    }

    System.out.println("wrote blocktree_compressed_index/ fixture directory");
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
