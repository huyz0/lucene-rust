import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HexFormat;
import java.util.Map;
import java.util.Random;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.FilterCodec;
import org.apache.lucene.codecs.LiveDocsFormat;
import org.apache.lucene.index.SegmentCommitInfo;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.tests.codecs.cranky.CrankyCodec;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.FixedBitSet;
import org.apache.lucene.util.Version;

/** Calls the actual package-private CrankyLiveDocsFormat via CrankyCodec. */
public class GenCrankyLiveDocs {
  static final byte[] ID = "0123456789abcdef".getBytes(StandardCharsets.US_ASCII);
  static SegmentCommitInfo info(Directory dir, int maxDoc, long delGen, int delCount) {
    SegmentInfo si = new SegmentInfo(dir, Version.LATEST, Version.LATEST, "_0", maxDoc,
        false, false, Codec.getDefault(), Map.of(), ID, Map.of(), null);
    return new SegmentCommitInfo(si, delCount, 0, delGen, -1, -1, null);
  }
  static LiveDocsFormat cranky(LiveDocsFormat liveDocsDelegate, Random random) {
    Codec codec = new FilterCodec("LiveDocsProbe", Codec.getDefault()) {
      @Override public LiveDocsFormat liveDocsFormat() { return liveDocsDelegate; }
    };
    return new CrankyCodec(codec, random).liveDocsFormat();
  }
  static class Probe extends LiveDocsFormat {
    int reads, writes, listings;
    final FixedBitSet bits = new FixedBitSet(1);
    @Override public Bits readLiveDocs(Directory dir, SegmentCommitInfo info, IOContext context) {
      reads++;
      return bits;
    }
    @Override public void writeLiveDocs(Bits bits, Directory dir, SegmentCommitInfo info,
        int newDelCount, IOContext context) throws IOException {
      if (++writes % 7 == 0) throw new IOException("delegate write");
    }
    @Override public void files(SegmentCommitInfo info, Collection<String> files) {
      listings++;
      files.add("delegate-file");
    }
  }
  static String sequence(long seed) throws Exception {
    Random random = new Random(seed);
    Probe probe = new Probe();
    LiveDocsFormat[] formats = {cranky(probe, random), cranky(probe, random)};
    try (Directory dir = new ByteBuffersDirectory()) {
      SegmentCommitInfo info = info(dir, 1, -1, 0);
      StringBuilder outcomes = new StringBuilder();
      for (int i = 0; i < 1024; i++) {
        if (i % 13 == 0) random.nextBoolean();
        LiveDocsFormat format = formats[i & 1];
        if (format.readLiveDocs(dir, info, IOContext.DEFAULT) != probe.bits) throw new AssertionError();
        ArrayList<String> files = new ArrayList<>();
        files.add("existing");
        format.files(info, files);
        if (!files.equals(java.util.List.of("existing", "delegate-file"))) throw new AssertionError();
        try {
          format.writeLiveDocs(probe.bits, dir, info, 0, IOContext.DEFAULT);
          outcomes.append('O');
        } catch (IOException error) {
          if (error.getMessage().equals("Fake IOException from LiveDocsFormat.writeLiveDocs()")) outcomes.append('F');
          else if (error.getMessage().equals("delegate write")) outcomes.append('D');
          else throw error;
        }
      }
      return "sequence\t" + seed + "\t" + outcomes + "\t" + random.nextLong()
          + "\t" + probe.reads + "\t" + probe.writes + "\t" + probe.listings + "\n";
    }
  }
  static String bytes(int maxDoc, long generation, boolean mismatch) throws Exception {
    try (Directory dir = new ByteBuffersDirectory()) {
      FixedBitSet bits = new FixedBitSet(maxDoc);
      bits.set(0, maxDoc);
      bits.clear(maxDoc - 1);
      Random random = new Random(42); // First nextInt(100) is nonzero.
      LiveDocsFormat format = cranky(Codec.getDefault().liveDocsFormat(), random);
      SegmentCommitInfo before = info(dir, maxDoc, generation == 1 ? -1 : generation - 1, 0);
      boolean failed = false;
      try {
        format.writeLiveDocs(bits, dir, before, mismatch ? 0 : 1, IOContext.DEFAULT);
      } catch (IOException error) {
        if (!mismatch || !error.getMessage().startsWith("bits.deleted=1 info.delcount=0 newdelcount=0")) throw error;
        failed = true;
      }
      if (failed != mismatch) throw new AssertionError();
      SegmentCommitInfo after = info(dir, maxDoc, generation, 1);
      ArrayList<String> files = new ArrayList<>();
      format.files(after, files);
      if (files.size() != 1) throw new AssertionError();
      byte[] bytes;
      try (var input = dir.openInput(files.getFirst(), IOContext.DEFAULT)) {
        bytes = new byte[Math.toIntExact(input.length())];
        input.readBytes(bytes, 0, bytes.length);
      }
      Bits reread = format.readLiveDocs(dir, after, IOContext.DEFAULT);
      for (int i = 0; i < maxDoc; i++) if (reread.get(i) != bits.get(i)) throw new AssertionError();
      // A damaged footer must propagate without drawing from Random.
      byte[] damaged = bytes.clone();
      damaged[damaged.length - 1] ^= 1;
      dir.deleteFile(files.getFirst());
      try (var output = dir.createOutput(files.getFirst(), IOContext.DEFAULT)) {
        output.writeBytes(damaged, damaged.length);
      }
      try {
        format.readLiveDocs(dir, after, IOContext.DEFAULT);
        throw new AssertionError("corruption accepted");
      } catch (IOException expected) { }
      return "bytes\t" + maxDoc + "\t" + generation + "\t" + mismatch + "\t"
          + files.getFirst() + "\t" + HexFormat.of().formatHex(bytes) + "\t" + random.nextLong() + "\n";
    }
  }
  public static void main(String[] args) throws Exception {
    StringBuilder result = new StringBuilder();
    for (long seed : new long[] {0, 1, 42, -1, 123456789, Long.MIN_VALUE, Long.MAX_VALUE}) result.append(sequence(seed));
    int[] lengths = {1, 63, 64, 65, 130, 4096};
    long[] generations = {1, 2, 35, 36, 1295, 1296};
    for (int i = 0; i < lengths.length; i++) {
      result.append(bytes(lengths[i], generations[i], false));
      result.append(bytes(lengths[i], generations[i], true));
    }
    Path out = Path.of(args[0]);
    Files.createDirectories(out);
    Files.writeString(out.resolve("cranky_live_docs.tsv"), result, StandardCharsets.UTF_8);
  }
}

