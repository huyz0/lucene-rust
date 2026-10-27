import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Random;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FilterLeafReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.PostingsEnum;
import org.apache.lucene.index.TermsEnum;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.tests.util.TestUtil;
import org.apache.lucene.util.BytesRef;

/**
 * Differential fixture for {@code TestUtil.docs}, {@code bytesToCharSequence} and {@code
 * bytesRefToString} (M14.h): the postings flags Lucene 10.5.0's {@code TestUtil.docs} asks a terms
 * enum for, given the caller's flags and a seeded {@code java.util.Random}.
 *
 * <p>Writes {@code test_util_docs.txt}: for each of 200 seeds and each caller flag ({@code NONE},
 * {@code FREQS}, {@code ALL}), {@code docs <seed> <in> <asked> <next>}; then {@code chars <seed>
 * <next>} for {@code stringToCharSequence}; then {@code bytes <hex> <string>} lines from {@code
 * bytesRefToString}.
 *
 * <p>Usage: java GenTestFrameworkTestUtilDocs <outdir>
 */
public class GenTestFrameworkTestUtilDocs {
  static String name(int flags) {
    switch (flags) {
      case PostingsEnum.NONE:
        return "None";
      case PostingsEnum.FREQS:
        return "Freqs";
      case PostingsEnum.POSITIONS:
        return "Positions";
      case PostingsEnum.OFFSETS:
        return "Offsets";
      case PostingsEnum.PAYLOADS:
        return "Payloads";
      case PostingsEnum.ALL:
        return "All";
      default:
        return "?" + flags;
    }
  }

  static final class Recording extends FilterLeafReader.FilterTermsEnum {
    int asked = -1;

    Recording(TermsEnum in) {
      super(in);
    }

    @Override
    public PostingsEnum postings(PostingsEnum reuse, int flags) throws IOException {
      asked = flags;
      return super.postings(reuse, flags);
    }
  }

  public static void main(String[] args) throws Exception {
    ByteBuffersDirectory dir = new ByteBuffersDirectory();
    try (IndexWriter w = new IndexWriter(dir, new IndexWriterConfig())) {
      Document doc = new Document();
      doc.add(new TextField("f", "a b", Field.Store.NO));
      w.addDocument(doc);
    }
    StringBuilder out = new StringBuilder();
    try (DirectoryReader reader = DirectoryReader.open(dir)) {
      TermsEnum te = reader.leaves().get(0).reader().terms("f").iterator();
      te.next();
      Recording rec = new Recording(te);
      int[] ins = {PostingsEnum.NONE, PostingsEnum.FREQS, PostingsEnum.ALL};
      for (int seed = 0; seed < 200; seed++) {
        for (int in : ins) {
          Random r = new Random(seed);
          TestUtil.docs(r, rec, null, in);
          out.append("docs ").append(seed).append(' ').append(name(in)).append(' ');
          out.append(name(rec.asked)).append(' ').append(r.nextInt()).append('\n');
        }
      }
    }
    for (int seed = 0; seed < 20; seed++) {
      Random r = new Random(seed);
      CharSequence cs = TestUtil.stringToCharSequence("héllo", r);
      if (!cs.toString().equals("héllo")) throw new AssertionError(cs);
      out.append("chars ").append(seed).append(' ').append(r.nextInt()).append('\n');
    }
    byte[][] samples = {{}, {'a', 'b'}, {(byte) 0xc3, (byte) 0xa9}};
    for (byte[] b : samples) {
      StringBuilder hex = new StringBuilder("-");
      for (byte x : b) hex.append(String.format("%02x", x & 0xff));
      out.append("bytes ").append(hex).append(' ');
      out.append(TestUtil.bytesRefToString(new BytesRef(b))).append('\n');
    }
    Path outDir = Path.of(args[0]);
    Files.createDirectories(outDir);
    Files.writeString(outDir.resolve("test_util_docs.txt"), out.toString());
  }
}
