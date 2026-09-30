import org.apache.lucene.codecs.lucene104.Lucene104Codec;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.StoredField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.SegmentCommitInfo;
import org.apache.lucene.index.SegmentInfos;
import org.apache.lucene.index.StoredFields;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.BytesRef;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;

/**
 * Generates a {@code Mode.BEST_COMPRESSION} stored-fields segment for the
 * <b>byte-identity</b> test of this port's DEFLATE path ({@code
 * crates/lucene-codecs/src/deflater.rs}, a port of the zlib {@code deflate}
 * behind {@code java.util.zip.Deflater}): {@code
 * crates/lucene-codecs/tests/stored_fields_best_compression_fixtures.rs}
 * rebuilds the same documents from the same LCG and requires {@code .fdt},
 * {@code .fdx} and {@code .fdm} to be these files byte for byte.
 *
 * <p>The documents are chosen to reach every branch of zlib's deflate:
 * word-salad text (dynamic Huffman blocks, lazy matches, the preset
 * dictionary), random bytes (stored blocks), long runs (maximum-length
 * matches), tiny documents (static trees), and a few documents large enough
 * to slide the 32 KB window within one sub-block and to fill a block's
 * 16 383-symbol budget, and a last chunk small enough for static trees --
 * spread over several chunks, so the {@code Deflater}
 * carries its state across many {@code reset()}s.
 */
public class GenStoredFieldsDeflate {

  static final String[] WORDS = {
    "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog", "lorem", "ipsum", "dolor",
    "sit", "amet", "consectetur", "adipiscing", "elit", "sed", "do", "eiusmod", "tempor",
    "incididunt", "labore", "magna", "aliqua", "lucene", "rust", "segment", "merge"
  };

  static long seed = 20260930L;

  static int next(int bound) {
    seed = seed * 6364136223846793005L + 1442695040888963407L;
    return (int) Long.remainderUnsigned(seed >>> 33, bound);
  }

  /** Document {@code i}'s fields; the Rust test mirrors this exactly. */
  static Document doc(int i) {
    Document d = new Document();
    int kind = i >= 497 ? 0 : next(10);
    StringBuilder sb = new StringBuilder();
    int words;
    if (kind == 0) {
      words = 1 + next(3);
    } else if (i % 83 == 0) {
      words = 5000 + next(2000);
    } else {
      words = 20 + next(400);
    }
    for (int w = 0; w < words; w++) {
      sb.append(WORDS[next(WORDS.length)]).append(next(5) == 0 ? ". " : " ");
    }
    d.add(new StoredField("text", sb.toString()));
    byte[] blob = new byte[i % 101 == 50 ? 20000 + next(5000) : next(kind == 1 ? 1200 : 40)];
    for (int b = 0; b < blob.length; b++) {
      blob[b] = (byte) next(256);
    }
    d.add(new StoredField("blob", new BytesRef(blob)));
    if (kind == 2) {
      byte[] run = new byte[next(2000)];
      Arrays.fill(run, (byte) ('a' + next(3)));
      d.add(new StoredField("run", new BytesRef(run)));
    }
    d.add(new StoredField("num", next(1_000_000) - 500_000));
    if (i == 496) {
      // One document past the chunk size on its own: its sub-blocks are
      // longer than the window, so zlib slides it; the three tiny documents
      // after it make a last chunk small enough for static trees.
      String[] sentences = new String[40];
      for (int k = 0; k < sentences.length; k++) {
        StringBuilder s = new StringBuilder();
        for (int w = 0; w < 12; w++) {
          s.append(WORDS[next(WORDS.length)]).append(' ');
        }
        sentences[k] = s.append(". ").toString();
      }
      StringBuilder big = new StringBuilder();
      for (int k = 0; k < 9000; k++) {
        big.append(sentences[next(sentences.length)]);
      }
      d.add(new StoredField("big", big.toString()));
    }
    return d;
  }

  public static void main(String[] args) throws IOException {
    Path out = Path.of(args[0]).resolve("stored_fields_deflate_index");
    if (Files.exists(out)) {
      try (var walk = Files.walk(out)) {
        for (Path p : (Iterable<Path>) walk.sorted(java.util.Comparator.reverseOrder())::iterator) {
          Files.delete(p);
        }
      }
    }
    Files.createDirectories(out);
    int numDocs = 500;
    try (Directory dir = FSDirectory.open(out)) {
      IndexWriterConfig cfg = new IndexWriterConfig();
      cfg.setCodec(new Lucene104Codec(Lucene104Codec.Mode.BEST_COMPRESSION));
      cfg.setUseCompoundFile(false);
      cfg.setMergePolicy(NoMergePolicy.INSTANCE);
      cfg.setRAMBufferSizeMB(256);
      try (IndexWriter w = new IndexWriter(dir, cfg)) {
        for (int i = 0; i < numDocs; i++) {
          w.addDocument(doc(i));
        }
        w.commit();
      }
      SegmentInfos sis = SegmentInfos.readLatestCommit(dir);
      if (sis.size() != 1) {
        throw new AssertionError("expected one segment, got " + sis.size());
      }
      SegmentCommitInfo sci = sis.info(0);
      StringBuilder m = new StringBuilder();
      for (String f : sci.info.files()) {
        for (String ext : new String[] {".fdt", ".fdx", ".fdm"}) {
          if (f.endsWith(ext)) {
            try (IndexInput in = dir.openInput(f, IOContext.READONCE)) {
              byte[] bytes = new byte[(int) in.length()];
              in.readBytes(bytes, 0, bytes.length);
              Files.write(out.resolve(f + ".raw"), bytes);
            }
            m.append(ext.substring(1)).append("_file_name=").append(f).append('\n');
          }
        }
      }
      m.append("id_hex=").append(hex(sci.info.getId())).append('\n');
      m.append("num_docs=").append(numDocs).append('\n');
      // Round trip: Lucene reads back what was indexed.
      seed = 20260930L;
      try (DirectoryReader r = DirectoryReader.open(dir)) {
        StoredFields sf = r.leaves().get(0).reader().storedFields();
        for (org.apache.lucene.index.FieldInfo fi : r.leaves().get(0).reader().getFieldInfos()) {
          m.append("field.").append(fi.name).append('=').append(fi.number).append('\n');
        }
        for (int i = 0; i < numDocs; i++) {
          Document want = doc(i);
          Document got = sf.document(i);
          if (!want.get("text").equals(got.get("text"))
              || !want.getBinaryValue("blob").equals(got.getBinaryValue("blob"))) {
            throw new AssertionError("document " + i + " differs");
          }
        }
      }
      Files.writeString(out.resolve("manifest.properties"), m.toString());
    }
    System.out.println("wrote stored_fields_deflate_index/");
  }

  static String hex(byte[] b) {
    StringBuilder sb = new StringBuilder();
    for (byte v : b) sb.append(String.format("%02x", v));
    return sb.toString();
  }
}
