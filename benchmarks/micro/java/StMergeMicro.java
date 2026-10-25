import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Executor;
import org.apache.lucene.codecs.FieldsConsumer;
import org.apache.lucene.codecs.FieldsProducer;
import org.apache.lucene.codecs.uniformsplit.sharedterms.STUniformSplitPostingsFormat;
import org.apache.lucene.index.CodecReader;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.MergePolicy;
import org.apache.lucene.index.MergeState;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.index.SegmentInfos;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.InfoStream;

/** Specialized shared-terms merge, over the same deleted three-segment fixture as Rust.
 * Reader opening and MergeState construction are outside the measured operation.
 * Only the shared-terms/postings consumer is timed, including output allocation and close.
 */
public final class StMergeMicro {
  static final String SUFFIX = "SharedTermsUniformSplit_0";
  static final String[] EXTENSIONS = {"stustb", "stustd", "doc", "pos", "pay", "psm"};
  static long sink;

  interface Op { long run() throws IOException; }

  static long loop(Op op, long millis) throws IOException {
    long start = System.nanoTime(), units = 0;
    do { units += op.run(); } while (System.nanoTime() - start < millis * 1_000_000L);
    return units;
  }

  static String suffix(FieldInfos fields) {
    var field = fields.fieldInfo("docs");
    return field.getAttribute("PerFieldPostingsFormat.format") + "_"
        + field.getAttribute("PerFieldPostingsFormat.suffix");
  }

  static long hash(Directory out, String segment) throws IOException {
    long h = 0xcbf29ce484222325L;
    for (String ext : EXTENSIONS) {
      try (IndexInput in = out.openInput(segment + "_" + SUFFIX + "." + ext, IOContext.DEFAULT)) {
        h = (h ^ in.length()) * 0x100000001b3L;
        while (in.getFilePointer() < in.length()) h = (h ^ (in.readByte() & 255)) * 0x100000001b3L;
      }
    }
    return h;
  }

  public static void main(String[] args) throws Exception {
    Path root = Path.of(args[0]);
    List<Directory> directories = new ArrayList<>();
    List<DirectoryReader> readers = new ArrayList<>();
    List<FieldsProducer> producers = new ArrayList<>();
    var format = new STUniformSplitPostingsFormat(8, 2, null, null, true);
    try {
      List<CodecReader> leaves = new ArrayList<>();
      for (int i = 0; i < 3; i++) {
        Directory dir = FSDirectory.open(root.resolve("source" + i));
        directories.add(dir);
        DirectoryReader reader = DirectoryReader.open(dir);
        readers.add(reader);
        leaves.add((CodecReader) reader.leaves().getFirst().reader());
        SegmentInfo info = SegmentInfos.readLatestCommit(dir).info(0).info;
        FieldInfos fields = leaves.getLast().getFieldInfos();
        producers.add(format.fieldsProducer(
            new SegmentReadState(dir, info, fields, IOContext.DEFAULT, suffix(fields))));
      }
      Directory targetDir = FSDirectory.open(root.resolve("merged"));
      directories.add(targetDir);
      DirectoryReader targetReader = DirectoryReader.open(targetDir);
      readers.add(targetReader);
      CodecReader target = (CodecReader) targetReader.leaves().getFirst().reader();
      SegmentInfo original = SegmentInfos.readLatestCommit(targetDir).info(0).info;
      // MergeState assigns maxDoc once. Preserve the fixture's output id and
      // segment name so the complete byte checksum agrees across languages.
      SegmentInfo info = new SegmentInfo(targetDir, original.getVersion(), original.getMinVersion(),
          original.name, -1, false, original.getHasBlocks(), original.getCodec(),
          original.getDiagnostics(), original.getId(), original.getAttributes(), null);
      var constructor = MergeState.class.getDeclaredConstructor(List.class, SegmentInfo.class,
          InfoStream.class, Executor.class, MergePolicy.OneMerge.class);
      constructor.setAccessible(true);
      MergeState merge = constructor.newInstance(leaves, info, InfoStream.NO_OUTPUT,
          (Executor) Runnable::run, null);
      merge.mergeFieldInfos = target.getFieldInfos();
      // PerFieldFieldsProducer would trigger the generic fallback. Supply
      // each direct ST producer, exactly as per-field merge dispatch does.
      for (int i = 0; i < 3; i++) merge.fieldsProducers[i] = producers.get(i);
      if (merge.needsIndexSort) throw new AssertionError("specialized fixture must be unsorted");
      Op operation = () -> {
        try (Directory out = new ByteBuffersDirectory()) {
          var state = new SegmentWriteState(InfoStream.NO_OUTPUT, out, info,
              merge.mergeFieldInfos, null, IOContext.DEFAULT, SUFFIX);
          try (FieldsConsumer consumer = format.fieldsConsumer(state)) {
            consumer.merge(merge, target.getNormsReader());
          }
          long size = 0;
          for (String ext : EXTENSIONS) size += out.fileLength(info.name + "_" + SUFFIX + "." + ext);
          sink = size;
          return 1;
        }
      };
      try (Directory out = new ByteBuffersDirectory()) {
        var state = new SegmentWriteState(InfoStream.NO_OUTPUT, out, info,
            merge.mergeFieldInfos, null, IOContext.DEFAULT, SUFFIX);
        try (FieldsConsumer consumer = format.fieldsConsumer(state)) {
          consumer.merge(merge, target.getNormsReader());
        }
        long actual = hash(out, info.name);
        if (actual != hash(targetDir, info.name)) throw new AssertionError("Java fixture mismatch");
        System.out.printf("#check\tcx_stsplit_merge\t%016x\t1%n", actual);
      }
      loop(operation, Long.getLong("warmupMs", 1500));
      long start = System.nanoTime();
      long units = loop(operation, Long.getLong("measureMs", 2000));
      System.out.printf("cx_stsplit_merge\t%.3f\t%d%n", (double) (System.nanoTime() - start) / units, units);
      if (sink == 0xdeadbeefL) System.err.print("");
    } finally {
      for (FieldsProducer producer : producers) producer.close();
      for (DirectoryReader reader : readers) reader.close();
      for (Directory dir : directories) dir.close();
    }
  }
}
