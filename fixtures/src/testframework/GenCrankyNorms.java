import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import java.util.Random;
import org.apache.lucene.codecs.*;
import org.apache.lucene.codecs.lucene90.Lucene90NormsFormat;
import org.apache.lucene.index.*;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.store.*;
import org.apache.lucene.util.InfoStream;
import org.apache.lucene.util.Version;

/** Actual package-private cranky classes, reflected only to construct them. */
public class GenCrankyNorms {
  static NormsFormat cranky(NormsFormat delegate, Random random) throws Exception {
    var c = Class.forName("org.apache.lucene.tests.codecs.cranky.CrankyNormsFormat")
        .getDeclaredConstructor(NormsFormat.class, Random.class);
    c.setAccessible(true);
    return (NormsFormat) c.newInstance(delegate, random);
  }
  static NormsConsumer crankyConsumer(NormsConsumer delegate, Random random) throws Exception {
    var c = Class.forName("org.apache.lucene.tests.codecs.cranky.CrankyNormsFormat$CrankyNormsConsumer")
        .getDeclaredConstructor(NormsConsumer.class, Random.class);
    c.setAccessible(true);
    return (NormsConsumer) c.newInstance(delegate, random);
  }
  static class Values extends NormsProducer {
    @Override public NumericDocValues getNorms(FieldInfo field) {
      int[] docs = field.number == 0 ? new int[] {0,1,2,3,4} : new int[] {0,2,4};
      long[] values = field.number == 0 ? new long[] {-129,0,127,128,32768} : new long[] {7,8,9};
      return new NumericDocValues() {
        int position = -1, doc = -1;
        @Override public int docID() { return doc; }
        @Override public int nextDoc() {
          return doc = ++position < docs.length ? docs[position] : DocIdSetIterator.NO_MORE_DOCS;
        }
        @Override public int advance(int target) {
          while (nextDoc() < target) {}
          return doc;
        }
        @Override public boolean advanceExact(int target) {
          while (doc < target) nextDoc();
          return doc == target;
        }
        @Override public long cost() { return docs.length; }
        @Override public long longValue() { return values[position]; }
      };
    }
    @Override public void checkIntegrity() {}
    @Override public void close() {}
  }
  static class RecordingConsumer extends NormsConsumer {
    int adds, closes;
    boolean fail;
    @Override public void addNormsField(FieldInfo f, NormsProducer p) throws IOException {
      adds++;
      if (fail) throw new IOException("delegate add");
    }
    @Override public void close() throws IOException {
      closes++;
      if (fail) throw new IOException("delegate close");
    }
  }
  static class Recording extends NormsFormat {
    final RecordingConsumer consumer = new RecordingConsumer();
    final Values producer = new Values();
    int opens, reads;
    boolean fail;
    @Override public NormsConsumer normsConsumer(SegmentWriteState state) throws IOException {
      opens++;
      if (fail) throw new IOException("delegate open");
      return consumer;
    }
    @Override public NormsProducer normsProducer(SegmentReadState state) throws IOException {
      reads++;
      if (fail) throw new IOException("delegate read");
      return producer;
    }
  }
  static FieldInfo field(String name, int number) {
    return new FieldInfo(name, number, false, false, false, IndexOptions.DOCS,
        DocValuesType.NONE, DocValuesSkipIndexType.NONE, -1, Map.of(), 0,0,0,0,
        VectorEncoding.FLOAT32, VectorSimilarityFunction.EUCLIDEAN, false, false);
  }
  public static void main(String[] args) throws Exception {
    Path out = Path.of(args[0]);
    Files.createDirectories(out);
    StringBuilder trace = new StringBuilder();
    long firstFailure = 0;
    while (new Random(firstFailure).nextInt(100) != 0) firstFailure++;
    for (long seed : new long[] {0,1,42,-1,Long.MIN_VALUE,Long.MAX_VALUE,firstFailure}) {
      Random random = new Random(seed);
      Recording delegate = new Recording();
      NormsFormat format = cranky(delegate, random);
      NormsConsumer consumer = crankyConsumer(delegate.consumer, random);
      for (int step = 0; step < 512; step++) {
        delegate.fail = delegate.consumer.fail = step % 17 == 0;
        String operation = new String[] {"open","read","add","close"}[step % 4];
        String status = "ok";
        try {
          switch (operation) {
            case "open" -> format.normsConsumer(null);
            case "read" -> {
              if (format.normsProducer(null) != delegate.producer)
                throw new AssertionError("producer identity changed");
            }
            case "add" -> consumer.addNormsField(null, null);
            case "close" -> consumer.close();
          }
        } catch (IOException e) { status = e.getMessage(); }
        trace.append(seed).append('\t').append(step).append('\t').append(operation)
            .append('\t').append(delegate.fail).append('\t').append(status)
            .append('\t').append(delegate.opens).append('\t').append(delegate.reads)
            .append('\t').append(delegate.consumer.adds).append('\t')
            .append(delegate.consumer.closes).append('\n');
      }
      trace.append(seed).append("\ttail\t").append(random.nextInt()).append('\n');
    }
    Files.writeString(out.resolve("cranky_norms.tsv"), trace);
    try (Directory directory = new ByteBuffersDirectory()) {
      byte[] id = new byte[16];
      for (int i = 0; i < id.length; i++) id[i] = (byte) i;
      SegmentInfo segment = new SegmentInfo(directory, Version.LATEST, null, "_cn", 5,
          false, false, Codec.getDefault(), Map.of(), id, Map.of(), null);
      FieldInfos fields = new FieldInfos(new FieldInfo[] {field("dense",0),field("sparse",1)});
      NormsFormat format = cranky(new Lucene90NormsFormat(), new Random(0));
      try (NormsConsumer consumer = format.normsConsumer(new SegmentWriteState(
          InfoStream.NO_OUTPUT, directory, segment, fields, null, IOContext.DEFAULT, "s"))) {
        for (FieldInfo f : fields) consumer.addNormsField(f, new Values());
      }
      try (NormsProducer producer = format.normsProducer(new SegmentReadState(
          directory, segment, fields, IOContext.DEFAULT, "s"))) {
        producer.checkIntegrity();
        for (FieldInfo f : fields) {
          NumericDocValues actual = producer.getNorms(f), expected = new Values().getNorms(f);
          for (int doc = 0; doc < 5; doc++) {
            boolean present = expected.advanceExact(doc);
            if (actual.advanceExact(doc) != present || present && actual.longValue() != expected.longValue())
              throw new AssertionError("real delegate values");
          }
        }
      }
      for (String extension : new String[] {"nvm","nvd"}) {
        try (IndexInput input = directory.openInput("_cn_s." + extension, IOContext.DEFAULT)) {
          byte[] bytes = new byte[(int) input.length()];
          input.readBytes(bytes,0,bytes.length);
          Files.write(out.resolve("cranky_norms." + extension), bytes);
        }
      }
    }
    // A successful delegate close is idempotent, but the cranky guard still draws.
    try (Directory directory = new ByteBuffersDirectory()) {
      Random random = new Random(0);
      byte[] id = new byte[16];
      for (int i = 0; i < id.length; i++) id[i] = (byte) i;
      SegmentInfo segment = new SegmentInfo(directory, Version.LATEST, null, "_empty", 5,
          false, false, Codec.getDefault(), Map.of(), id, Map.of(), null);
      FieldInfos fields = new FieldInfos(new FieldInfo[0]);
      NormsFormat format = cranky(new Lucene90NormsFormat(), random);
      NormsConsumer consumer = format.normsConsumer(new SegmentWriteState(InfoStream.NO_OUTPUT,
          directory, segment, fields, null, IOContext.DEFAULT, "s"));
      consumer.close();
      consumer.close();
      String afterClose = "ok";
      try { consumer.addNormsField(field("dense",0),new Values()); }
      catch (NullPointerException e) { afterClose = "closed consumer"; }
      Files.writeString(out.resolve("cranky_norms_lifecycle.tsv"),
          "close\tok\nclose\tok\nadd\t" + afterClose + "\ntail\t" + random.nextInt() + "\n");
      try (NormsProducer producer = format.normsProducer(new SegmentReadState(directory,
          segment, fields, IOContext.DEFAULT, "s"))) { producer.checkIntegrity(); }
      for (String extension : new String[] {"nvm","nvd"}) {
        try (IndexInput input = directory.openInput("_empty_s." + extension, IOContext.DEFAULT)) {
          byte[] bytes = new byte[(int) input.length()];
          input.readBytes(bytes,0,bytes.length);
          Files.write(out.resolve("cranky_norms_empty." + extension),bytes);
        }
      }
    }
    StringBuilder opening = new StringBuilder();
    for (String failExtension : new String[] {"nvd","nvm"}) {
      StringBuilder calls = new StringBuilder();
      Random random = new Random(0);
      try (Directory directory = new FilterDirectory(new ByteBuffersDirectory()) {
        @Override public IndexOutput createOutput(String name, IOContext context) throws IOException {
          if (!calls.isEmpty()) calls.append(',');
          calls.append(name);
          if (name.endsWith("." + failExtension)) throw new IOException("create " + failExtension);
          return in.createOutput(name, context);
        }
      }) {
        SegmentInfo segment = new SegmentInfo(directory, Version.LATEST, null, "_fail", 5,
            false, false, Codec.getDefault(), Map.of(), new byte[16], Map.of(), null);
        String status = "ok";
        try {
          cranky(new Lucene90NormsFormat(),random).normsConsumer(new SegmentWriteState(
              InfoStream.NO_OUTPUT,directory,segment,new FieldInfos(new FieldInfo[0]),null,IOContext.DEFAULT));
        } catch (IOException e) { status = e.getMessage(); }
        opening.append(failExtension).append('\t').append(calls).append('\t').append(status)
            .append('\t').append(random.nextInt()).append('\n');
        if (failExtension.equals("nvm")) {
          try (IndexInput input = directory.openInput("_fail.nvd",IOContext.DEFAULT)) {
            byte[] bytes = new byte[(int) input.length()];
            input.readBytes(bytes,0,bytes.length);
            Files.write(out.resolve("cranky_norms_open_failure.nvd"),bytes);
          }
        }
      }
    }
    Files.writeString(out.resolve("cranky_norms_open_failures.tsv"),opening);
  }
}
