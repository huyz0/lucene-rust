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
import org.apache.lucene.index.PostingsEnum;
import org.apache.lucene.index.SegmentCommitInfo;
import org.apache.lucene.index.SegmentInfos;
import org.apache.lucene.index.TermsEnum;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.BytesRef;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.TreeMap;
import java.util.TreeSet;

/**
 * Generates a term dictionary for the <b>byte-identity</b> test of the two
 * choices {@code GenBlockTreeByteIdentity} deliberately avoids: {@code .tim}
 * suffix compression ({@code LZ4} and {@code LOWERCASE_ASCII}) and the zigzag
 * singleton-doc-delta branch of {@code Lucene104PostingsWriter.encodeTerm}.
 *
 * <p>The Rust test ({@code crates/lucene-codecs/tests/
 * blocktree_byte_identity_fixture.rs}) hands the same terms, doc lists,
 * segment id and suffix to {@code postings_writer::write_fields} and requires
 * {@code .tim}, {@code .tip}, {@code .tmd}, {@code .doc} and {@code .psm} to be
 * these files byte for byte.
 *
 * <p>Four term families, each under prefixes longer than two bytes:
 *
 * <ul>
 *   <li>URL-like terms whose suffixes repeat long fragments: LZ4 saves more
 *       than 25%.
 *   <li>random lowercase words of 5..12 letters: LZ4 finds little,
 *       LOWERCASE_ASCII packs them.
 *   <li>random upper-case and punctuation suffixes: neither pays, the block
 *       stays uncompressed.
 *   <li>auto-increment style IDs, each in exactly one document in ascending,
 *       sometimes repeating and sometimes descending order: runs of singleton
 *       terms that take {@code encodeTerm}'s zigzag branch, interleaved with
 *       multi-document terms that break the run.
 * </ul>
 */
public class GenBlockTreeSuffixCompression {

  static final int NUM_DOCS = 64;

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

  static String word(Random rnd, String alphabet, int min, int max) {
    int n = min + rnd.nextInt(max - min + 1);
    StringBuilder sb = new StringBuilder();
    for (int i = 0; i < n; i++) {
      sb.append(alphabet.charAt(rnd.nextInt(alphabet.length())));
    }
    return sb.toString();
  }

  static TreeSet<Integer> randomDocs(Random rnd) {
    TreeSet<Integer> docs = new TreeSet<>();
    int n = 2 + rnd.nextInt(5);
    while (docs.size() < n) {
      docs.add(rnd.nextInt(NUM_DOCS));
    }
    return docs;
  }

  public static void main(String[] args) throws IOException {
    Path out = Path.of(args[0]).resolve("blocktree_suffix_compression_index");
    if (Files.exists(out)) {
      deleteRecursive(out);
    }
    Files.createDirectories(out);

    Random rnd = new Random(20260930L);
    TreeMap<String, TreeSet<Integer>> postings = new TreeMap<>();
    String lower = "abcdefghijklmnopqrstuvwxyz";
    String upper = "ABCDEFGHIJKLMNOPQRSTUVWXYZ!#$%&*+<>?@[]^{|}~";
    String[] fragments = {
      "/index.html", "/search?q=", "/products/", "/category/", "&page=", "/static/"
    };
    for (int host = 0; host < 12; host++) {
      String prefix = "http://www." + word(rnd, lower, 4, 8) + ".com";
      int n = 20 + rnd.nextInt(60);
      for (int i = 0; i < n; i++) {
        StringBuilder sb = new StringBuilder(prefix);
        int parts = 2 + rnd.nextInt(3);
        for (int p = 0; p < parts; p++) {
          sb.append(fragments[rnd.nextInt(fragments.length)]);
        }
        sb.append(rnd.nextInt(1000));
        postings.put(sb.toString(), randomDocs(rnd));
      }
    }
    for (int w = 0; w < 30; w++) {
      String prefix = "lc" + word(rnd, lower, 2, 3);
      int n = 10 + rnd.nextInt(70);
      for (int i = 0; i < n; i++) {
        postings.put(prefix + word(rnd, lower, 3, 10), randomDocs(rnd));
      }
    }
    for (int w = 0; w < 10; w++) {
      String prefix = "UC" + word(rnd, upper, 2, 3);
      int n = 10 + rnd.nextInt(70);
      for (int i = 0; i < n; i++) {
        postings.put(prefix + word(rnd, upper, 3, 10), randomDocs(rnd));
      }
    }
    int doc = 0;
    for (int id = 0; id < 3000; id++) {
      String term = String.format("id%07d", id * 7 + rnd.nextInt(7));
      if (rnd.nextInt(40) == 0) {
        postings.put(term, randomDocs(rnd));
        continue;
      }
      int roll = rnd.nextInt(10);
      if (roll < 6) {
        doc = (doc + 1) % NUM_DOCS;
      } else if (roll < 8) {
        doc = rnd.nextInt(NUM_DOCS);
      }
      TreeSet<Integer> one = new TreeSet<>();
      one.add(doc);
      postings.put(term, one);
    }

    List<List<String>> docTokens = new ArrayList<>();
    for (int d = 0; d < NUM_DOCS; d++) {
      docTokens.add(new ArrayList<>());
    }
    for (Map.Entry<String, TreeSet<Integer>> e : postings.entrySet()) {
      for (int d : e.getValue()) {
        docTokens.get(d).add(e.getKey());
      }
    }

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
        for (int d = 0; d < NUM_DOCS; d++) {
          Document document = new Document();
          document.add(new Field("t", new CannedTokenStream(docTokens.get(d)), type));
          w.addDocument(document);
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
      m.append("num_terms=").append(postings.size()).append('\n');

      // Round-trip: Lucene reads back exactly the postings the manifest lists.
      try (DirectoryReader reader = DirectoryReader.open(dir)) {
        LeafReader leaf = reader.leaves().get(0).reader();
        TermsEnum te = leaf.terms("t").iterator();
        int count = 0;
        for (BytesRef t = te.next(); t != null; t = te.next()) {
          TreeSet<Integer> want = postings.get(t.utf8ToString());
          PostingsEnum pe = te.postings(null, PostingsEnum.NONE);
          TreeSet<Integer> got = new TreeSet<>();
          for (int d = pe.nextDoc(); d != DocIdSetIterator.NO_MORE_DOCS; d = pe.nextDoc()) {
            got.add(d);
          }
          if (!got.equals(want)) {
            throw new AssertionError("postings differ for " + t.utf8ToString());
          }
          count++;
        }
        if (count != postings.size()) {
          throw new AssertionError("expected " + postings.size() + " terms, got " + count);
        }
      }
      StringBuilder terms = new StringBuilder();
      for (Map.Entry<String, TreeSet<Integer>> e : postings.entrySet()) {
        terms.append(e.getKey()).append('\t');
        StringBuilder docs = new StringBuilder();
        for (int d : e.getValue()) {
          if (docs.length() > 0) {
            docs.append(',');
          }
          docs.append(d);
        }
        terms.append(docs).append('\n');
      }
      Files.writeString(out.resolve("terms.txt"), terms.toString());
      Files.writeString(out.resolve("manifest.properties"), m.toString());
    }
    System.out.println(
        "wrote blocktree_suffix_compression_index/ fixture directory ("
            + postings.size()
            + " terms)");
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
