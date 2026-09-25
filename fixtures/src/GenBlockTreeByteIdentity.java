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
 * Generates a term dictionary for a <b>byte-identity</b> differential test of
 * this port's block-tree writer ({@code crates/lucene-codecs/src/
 * blocktree_writer.rs}): the Rust test hands the same terms, postings, segment
 * id and suffix to {@code postings_writer::write_fields} and requires the
 * {@code .tim}, {@code .tip}, {@code .tmd} and {@code .doc} it produces to be
 * these files, byte for byte.
 *
 * <p>The port deviates from Java in exactly two places, and this term set is
 * built so Java takes neither path:
 *
 * <ul>
 *   <li><b>Suffix compression.</b> Java only tries LZ4/LOWERCASE_ASCII when a
 *       block's {@code prefixLength > 2}. Every term here is at most three
 *       bytes, so no block's prefix is longer than two.
 *   <li><b>The zigzag singleton-delta branch of {@code encodeTerm}</b>, taken
 *       only between consecutive terms with {@code docFreq == 1}. Every term
 *       here is in both documents.
 * </ul>
 *
 * <p>Within those limits the shape is made irregular on purpose: two-byte
 * prefixes carry between 0 and 64 three-byte children drawn from a
 * fixed-seed {@link Random}, so some prefixes stay inline as terms in their
 * parent (mixed blocks), some become their own leaf blocks, and dense ones
 * floor-split; third bytes range over 64 labels with random gaps, so trie
 * nodes pick all three child-label strategies.
 */
public class GenBlockTreeByteIdentity {

  static final String LABELS =
      "-0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_abcdefghijklmnopqrstuvwxyz";

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
    Path out = Path.of(args[0]).resolve("blocktree_byte_identity_index");
    if (Files.exists(out)) {
      deleteRecursive(out);
    }
    Files.createDirectories(out);

    Random rnd = new Random(20260925L);
    TreeSet<String> distinct = new TreeSet<>();
    String firsts = "abcdefghijklmnopqrstuvwxyz";
    for (char a : firsts.toCharArray()) {
      if (rnd.nextInt(3) == 0) {
        distinct.add("" + a);
      }
      for (int j = 0; j < LABELS.length(); j++) {
        char b = LABELS.charAt(j);
        // Most two-byte prefixes are empty or sparse; a few are dense.
        int roll = rnd.nextInt(10);
        int children = roll < 5 ? 0 : roll < 8 ? rnd.nextInt(12) : 20 + rnd.nextInt(45);
        if (children > 0 && rnd.nextBoolean()) {
          distinct.add("" + a + b);
        }
        for (int k = 0; k < children; k++) {
          distinct.add("" + a + b + LABELS.charAt(rnd.nextInt(LABELS.length())));
        }
      }
    }
    List<String> terms = List.copyOf(distinct);

    FieldType type = new FieldType();
    type.setIndexOptions(IndexOptions.DOCS);
    type.setOmitNorms(true);
    type.setTokenized(true);
    type.freeze();

    try (Directory dir = FSDirectory.open(out)) {
      IndexWriterConfig cfg = new IndexWriterConfig();
      cfg.setUseCompoundFile(false);
      cfg.setMergePolicy(NoMergePolicy.INSTANCE);
      cfg.setRAMBufferSizeMB(256);
      try (IndexWriter w = new IndexWriter(dir, cfg)) {
        for (int d = 0; d < 2; d++) {
          Document doc = new Document();
          doc.add(new Field("t", new CannedTokenStream(terms), type));
          w.addDocument(doc);
        }
        w.commit();
      }

      SegmentInfos sis = SegmentInfos.readLatestCommit(dir);
      if (sis.size() != 1) {
        throw new AssertionError("expected exactly one segment, got " + sis.size());
      }
      SegmentCommitInfo sci = sis.info(0);
      StringBuilder m = new StringBuilder();
      for (String f : sci.info.files()) {
        for (String ext : new String[] {".tim", ".tip", ".tmd", ".doc", ".psm", ".fnm"}) {
          if (f.endsWith(ext)) {
            dump(dir, f, out);
            m.append(ext.substring(1)).append("_file_name=").append(f).append('\n');
          }
        }
      }
      String tim = sci.info.files().stream().filter(f -> f.endsWith(".tim")).findFirst().get();
      String prefix = sci.info.name + "_";
      m.append("segment_suffix=")
          .append(tim.substring(prefix.length(), tim.length() - 4))
          .append('\n');
      m.append("id_hex=").append(hex(sci.info.getId())).append('\n');
      m.append("max_doc=").append(sci.info.maxDoc()).append('\n');
      m.append("num_terms=").append(terms.size()).append('\n');

      try (DirectoryReader reader = DirectoryReader.open(dir)) {
        LeafReader leaf = reader.leaves().get(0).reader();
        TermsEnum te = leaf.terms("t").iterator();
        int count = 0;
        for (BytesRef t = te.next(); t != null; t = te.next()) {
          if (te.docFreq() != 2) {
            throw new AssertionError("every term must be in both documents: " + t.utf8ToString());
          }
          count++;
        }
        if (count != terms.size()) {
          throw new AssertionError("expected " + terms.size() + " terms, got " + count);
        }
      }
      Files.writeString(out.resolve("terms.txt"), String.join("\n", terms) + "\n");
      Files.writeString(out.resolve("manifest.properties"), m.toString());
    }
    System.out.println(
        "wrote blocktree_byte_identity_index/ fixture directory (" + terms.size() + " terms)");
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
