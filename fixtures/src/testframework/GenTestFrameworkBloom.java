import java.nio.file.Path;
import org.apache.lucene.codecs.PostingsFormat;
import org.apache.lucene.tests.codecs.bloom.TestBloomFilteredLucenePostings;

/** The actual low-memory test format, including a deliberately saturated filter. */
public class GenTestFrameworkBloom {
  public static void main(String[] args) throws Exception {
    Path root = Path.of(args[0]).resolve("bloom");
    GenTestFrameworkBlockTerms.generate(root,
        new PostingsFormat[] {new TestBloomFilteredLucenePostings()},
        new String[] {"sparse"}, 500);
    GenTestFrameworkBlockTerms.generate(root,
        new PostingsFormat[] {new TestBloomFilteredLucenePostings()},
        new String[] {"saturated"}, 30000);
  }
}
