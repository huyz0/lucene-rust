import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import org.apache.lucene.tests.util.StringMockResourceLoader;
import org.apache.lucene.util.RamUsageEstimator;

/** Open and consume 256 identical UTF-8 resources, matching the test-framework example. */
public class StringMockResourceLoaderMicro {
  static volatile long sink;

  static long batch(StringMockResourceLoader loader) throws IOException {
    long sum = 0;
    for (int i = 0; i < 256; i++) {
      try (InputStream stream = loader.openResource("ignored")) {
        int b;
        while ((b = stream.read()) != -1) sum += b;
      }
    }
    return sum;
  }

  static long[] run(StringMockResourceLoader loader, long ms) throws IOException {
    long start = System.nanoTime();
    long batches = 0;
    do {
      sink = batch(loader);
      batches++;
    } while (System.nanoTime() - start < ms * 1_000_000L);
    return new long[] {batches, System.nanoTime() - start};
  }

  public static void main(String[] args) throws Exception {
    for (int repeat : new int[] {8, 256}) {
      String text = "caf\u00e9 \ud83d\ude00\u0000\n".repeat(repeat);
      StringMockResourceLoader loader = new StringMockResourceLoader(text);
      String name = "string_resource_loader_" + repeat;
      // Every text contains supplementary characters: its String uses UTF-16,
      // which is exactly the storage model of RamUsageEstimator.sizeOf(String).
      long retained = RamUsageEstimator.shallowSizeOf(loader) + RamUsageEstimator.sizeOf(text);
      long openBytes;
      try (InputStream stream = loader.openResource("ignored")) {
        openBytes = RamUsageEstimator.shallowSizeOf(stream)
            + RamUsageEstimator.sizeOf(text.getBytes(StandardCharsets.UTF_8));
      }
      System.out.printf("%s_retained_bytes\t%d\t1%n", name, retained);
      System.out.printf("%s_open_bytes\t%d\t1%n", name, openBytes);
      System.out.printf("#check\t%s\t%d\t256%n", name, batch(loader));
      run(loader, 1000);
      long[] result = run(loader, 1500);
      long opens = result[0] * 256;
      System.out.printf(java.util.Locale.ROOT, "%s\t%.3f\t%d%n", name, (double) result[1] / opens, opens);
    }
  }
}
