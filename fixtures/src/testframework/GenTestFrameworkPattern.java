import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Random;
import java.util.regex.Pattern;
import org.apache.lucene.tests.util.TestUtil;

/**
 * Differential fixture for {@code TestUtil.randomPattern} (M14.h): which regexpish string Lucene
 * 10.5.0's test framework settles on for a seeded {@code java.util.Random}, after retrying those
 * that do not compile or that split {@code "AB𠀀C"}'s surrogate pair.
 *
 * <p>Writes {@code pattern.txt}: for each of 500 seeds, {@code <seed> <pattern> <next>}, the
 * pattern's UTF-16 units in hex ({@code -} when empty) and the random's next {@code nextInt()}, which
 * shows the generator drew exactly as many values (retries included) as Java did.
 *
 * <p>Usage: java GenTestFrameworkPattern <outdir>
 */
public class GenTestFrameworkPattern {
  static final int SEEDS = 500;

  static String units(String s) {
    if (s.isEmpty()) return "-";
    StringBuilder sb = new StringBuilder();
    for (int i = 0; i < s.length(); i++) sb.append(String.format("%04x", (int) s.charAt(i)));
    return sb.toString();
  }

  public static void main(String[] args) throws Exception {
    StringBuilder out = new StringBuilder();
    for (int seed = 0; seed < SEEDS; seed++) {
      Random r = new Random(seed);
      Pattern p = TestUtil.randomPattern(r);
      out.append(seed).append(' ').append(units(p.pattern())).append(' ').append(r.nextInt());
      out.append('\n');
    }
    Path dir = Path.of(args[0]);
    Files.createDirectories(dir);
    Files.writeString(dir.resolve("pattern.txt"), out.toString());
  }
}
