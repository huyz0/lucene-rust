import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.store.ChecksumIndexInput;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.util.OfflineSorter;

/**
 * Differential fixtures for {@link OfflineSorter}.
 *
 * <p>The inputs are generated from a formula both sides share (so no multi-megabyte input is
 * checked in): variable-length item {@code i} is {@code Long.toHexString((i * 2654435761) %
 * 1000003)} followed by {@code i % 37} letters {@code k}; fixed-length item {@code i} is the
 * big-endian int {@code (i * 7919) % 100003}. Each case sorts in a fresh directory and records
 * the result file's name, {@code SortInfo}'s temp-file/merge/line counts, and the result's length
 * and footer checksum. Writes `offline_sorter/cases.txt`.
 *
 * <p>Usage: java GenOfflineSorter <outdir>
 */
public class GenOfflineSorter {
  static byte[] variable(long i) {
    return (Long.toHexString((i * 2654435761L) % 1000003) + "k".repeat((int) (i % 37)))
        .getBytes(StandardCharsets.US_ASCII);
  }

  static byte[] fixed(long i) {
    int v = (int) ((i * 7919) % 100003);
    return new byte[] {(byte) (v >>> 24), (byte) (v >>> 16), (byte) (v >>> 8), (byte) v};
  }

  public static void main(String[] args) throws Exception {
    Path out = Path.of(args[0], "offline_sorter");
    Files.createDirectories(out);
    StringBuilder sb = new StringBuilder();
    Object[][] cases = {
      // name, items, fixed?, buffer MB (x 1/2), maxTempFiles
      {"var_min_2", 120_000, false, 1, 2},
      {"var_min_3", 120_000, false, 1, 3},
      {"var_min_10", 120_000, false, 1, 10},
      {"var_1mb_2", 120_000, false, 2, 2},
      {"var_small", 5_000, false, 1, 10},
      {"var_empty", 0, false, 1, 10},
      {"fixed_min_2", 400_000, true, 1, 2},
      {"fixed_min_10", 400_000, true, 1, 10},
    };
    for (Object[] c : cases) {
      String name = (String) c[0];
      int n = (Integer) c[1];
      boolean isFixed = (Boolean) c[2];
      long buffer = (Integer) c[3] * OfflineSorter.ABSOLUTE_MIN_SORT_BUFFER_SIZE;
      int maxTemp = (Integer) c[4];
      Path tmp = Files.createTempDirectory("offline_sorter");
      try (Directory dir = FSDirectory.open(tmp)) {
        try (IndexOutput o = dir.createOutput("in", IOContext.DEFAULT)) {
          OfflineSorter.ByteSequencesWriter w = new OfflineSorter.ByteSequencesWriter(o);
          for (long i = 0; i < n; i++) w.write(isFixed ? fixed(i) : variable(i));
          CodecUtil.writeFooter(o);
        }
        java.lang.reflect.Method bufMethod =
            OfflineSorter.BufferSize.class.getDeclaredMethod("megabytes", long.class);
        java.lang.reflect.Constructor<OfflineSorter.BufferSize> ctor =
            OfflineSorter.BufferSize.class.getDeclaredConstructor(long.class);
        ctor.setAccessible(true);
        OfflineSorter.BufferSize bs = ctor.newInstance(buffer);
        // `getWriter` sees every temp file with its item count: the exact
        // partition sizes the RAM accounting produced.
        StringBuilder writes = new StringBuilder();
        OfflineSorter sorter =
            new OfflineSorter(
                dir,
                "t",
                OfflineSorter.DEFAULT_COMPARATOR,
                bs,
                maxTemp,
                isFixed ? 4 : -1,
                null,
                0) {
              @Override
              protected ByteSequencesWriter getWriter(IndexOutput out, long itemCount)
                  throws java.io.IOException {
                writes.append(writes.length() == 0 ? "" : ",").append(out.getName()).append(':').append(itemCount);
                return super.getWriter(out, itemCount);
              }
            };
        String result = sorter.sort("in");
        java.lang.reflect.Field f = OfflineSorter.class.getDeclaredField("sortInfo");
        f.setAccessible(true);
        OfflineSorter.SortInfo info = (OfflineSorter.SortInfo) f.get(sorter);
        long len;
        long checksum;
        try (ChecksumIndexInput in = dir.openChecksumInput(result)) {
          len = in.length();
          checksum = CodecUtil.retrieveChecksum(in);
        }
        sb.append(name)
            .append(' ')
            .append(n)
            .append(' ')
            .append(isFixed)
            .append(' ')
            .append(buffer)
            .append(' ')
            .append(maxTemp)
            .append(' ')
            .append(result)
            .append(' ')
            .append(info.tempMergeFiles)
            .append(' ')
            .append(info.mergeRounds)
            .append(' ')
            .append(info.lineCount)
            .append(' ')
            .append(len)
            .append(' ')
            .append(checksum)
            .append(' ')
            .append(writes.length() == 0 ? "-" : writes)
            .append('\n');
      } finally {
        try (var s = Files.walk(tmp)) {
          s.sorted(java.util.Comparator.reverseOrder()).forEach(p -> p.toFile().delete());
        }
      }
    }
    Files.writeString(out.resolve("cases.txt"), sb.toString());
  }
}
