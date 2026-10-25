import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import java.util.Random;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.FieldInfosFormat;
import org.apache.lucene.codecs.lucene94.Lucene94FieldInfosFormat;
import org.apache.lucene.index.*;
import org.apache.lucene.store.*;
import org.apache.lucene.util.Version;

/** The actual package-private CrankyFieldInfosFormat, reached only by constructor reflection. */
public class GenCrankyFieldInfos {
  static FieldInfosFormat cranky(FieldInfosFormat delegate, Random random) throws Exception {
    var constructor = Class.forName("org.apache.lucene.tests.codecs.cranky.CrankyFieldInfosFormat")
        .getDeclaredConstructor(FieldInfosFormat.class, Random.class);
    constructor.setAccessible(true);
    return (FieldInfosFormat) constructor.newInstance(delegate, random);
  }

  static final FieldInfos EMPTY = new FieldInfos(new FieldInfo[0]);
  static class Recording extends FieldInfosFormat {
    int reads, writes;
    boolean fail;
    @Override public FieldInfos read(Directory d, SegmentInfo s, String suffix, IOContext c)
        throws IOException {
      reads++;
      if (fail) throw new IOException("delegate read");
      return EMPTY;
    }
    @Override public void write(Directory d, SegmentInfo s, String suffix, FieldInfos f, IOContext c)
        throws IOException {
      writes++;
      if (fail) throw new IOException("delegate write");
    }
  }

  public static void main(String[] args) throws Exception {
    Path out = Path.of(args[0]);
    Files.createDirectories(out);
    StringBuilder result = new StringBuilder();
    long firstFailure = 0;
    while (new Random(firstFailure).nextInt(100) != 0) firstFailure++;
    for (long seed : new long[] {0, 1, 42, -1, Long.MIN_VALUE, Long.MAX_VALUE, firstFailure}) {
      Random random = new Random(seed);
      Recording delegate = new Recording();
      FieldInfosFormat format = cranky(delegate, random);
      for (int step = 0; step < 256; step++) {
        boolean read = step % 3 == 1;
        delegate.fail = step % 17 == 0;
        String status = "ok";
        try {
          if (read) {
            if (format.read(null, null, "suffix", IOContext.DEFAULT) != EMPTY)
              throw new AssertionError("read did not return delegate value");
          } else format.write(null, null, "suffix", EMPTY, IOContext.DEFAULT);
        } catch (IOException e) {
          status = e.getMessage();
        }
        result.append(seed).append('\t').append(step).append('\t')
            .append(read ? "read" : "write").append('\t').append(delegate.fail)
            .append('\t').append(status).append('\t').append(delegate.reads)
            .append('\t').append(delegate.writes).append('\n');
      }
      result.append(seed).append("\ttail\t").append(random.nextInt()).append('\n');
    }
    Files.writeString(out.resolve("cranky_field_infos.tsv"), result);

    // Exercise the real production delegate and export its deterministic bytes.
    try (Directory directory = new ByteBuffersDirectory()) {
      byte[] id = new byte[16];
      for (int i = 0; i < id.length; i++) id[i] = (byte) i;
      SegmentInfo segment = new SegmentInfo(directory, Version.LATEST, null, "_cfi", 0,
          false, false, Codec.getDefault(), Map.of(), id, Map.of(), null);
      FieldInfos infos = new FieldInfos(new FieldInfo[] {new FieldInfo("field", 0, false,
          false, false, IndexOptions.DOCS, DocValuesType.NONE, DocValuesSkipIndexType.NONE,
          -1, Map.of(), 0, 0, 0, 0, VectorEncoding.FLOAT32,
          VectorSimilarityFunction.EUCLIDEAN, false, false)});
      FieldInfosFormat format = cranky(new Lucene94FieldInfosFormat(), new Random(0));
      format.write(directory, segment, "s", infos, IOContext.DEFAULT);
      FieldInfos read = format.read(directory, segment, "s", IOContext.DEFAULT);
      if (read.size() != 1 || read.fieldInfo("field").getIndexOptions() != IndexOptions.DOCS)
        throw new AssertionError("real delegate round trip");
      try (IndexInput input = directory.openInput("_cfi_s.fnm", IOContext.DEFAULT)) {
        byte[] bytes = new byte[(int) input.length()];
        input.readBytes(bytes, 0, bytes.length);
        Files.write(out.resolve("cranky_field_infos.fnm"), bytes);
      }
    }
  }
}
