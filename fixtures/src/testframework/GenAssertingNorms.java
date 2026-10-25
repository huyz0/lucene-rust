import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicReference;
import org.apache.lucene.codecs.*;
import org.apache.lucene.codecs.lucene90.Lucene90NormsFormat;
import org.apache.lucene.index.*;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.store.*;
import org.apache.lucene.tests.codecs.asserting.AssertingNormsFormat;
import org.apache.lucene.util.InfoStream;
import org.apache.lucene.util.Version;

/** Actual asserting wrappers with -ea, recording callbacks and production norms bytes. */
public class GenAssertingNorms {
  static FieldInfo field(String name, int number, boolean norms) {
    return new FieldInfo(name,number,false,!norms,false,IndexOptions.DOCS,DocValuesType.NONE,
        DocValuesSkipIndexType.NONE,-1,Map.of(),0,0,0,0,VectorEncoding.FLOAT32,
        VectorSimilarityFunction.EUCLIDEAN,false,false);
  }
  static final FieldInfo FIELD = field("norm",0,true), NO_NORMS = field("none",1,false);
  static final FieldInfos FIELDS = new FieldInfos(new FieldInfo[] {FIELD});
  static NormsConsumer consumer(NormsConsumer delegate) throws Exception {
    var c = Class.forName("org.apache.lucene.tests.codecs.asserting.AssertingNormsFormat$AssertingNormsConsumer")
        .getDeclaredConstructor(NormsConsumer.class,int.class);
    c.setAccessible(true);
    return (NormsConsumer) c.newInstance(delegate,5);
  }
  static NormsProducer producer(NormsProducer delegate) throws Exception {
    var c = Class.forName("org.apache.lucene.tests.codecs.asserting.AssertingNormsFormat$AssertingNormsProducer")
        .getDeclaredConstructor(NormsProducer.class,int.class,boolean.class);
    c.setAccessible(true);
    return (NormsProducer) c.newInstance(delegate,5,false);
  }
  static AssertingNormsFormat format(NormsFormat delegate) throws Exception {
    var result = new AssertingNormsFormat();
    var f = AssertingNormsFormat.class.getDeclaredField("in");
    f.setAccessible(true);
    f.set(result,delegate);
    return result;
  }
  static class Values extends NormsProducer {
    @Override public NumericDocValues getNorms(FieldInfo field) {
      int[] docs = field.number == 0 ? new int[] {0,1,2,3,4}
          : field.number == 1 ? new int[] {0,2,4} : new int[0];
      long[] values = field.number == 0 ? new long[] {-129,0,127,128,32768}
          : field.number == 1 ? new long[] {7,8,9} : new long[0];
      return numeric(docs,values,new ArrayList<>(),false,-1);
    }
    @Override public void checkIntegrity() {}
    @Override public void close() {}
  }
  static NumericDocValues numeric(int[] docs,long[] values,List<String> events,
      boolean failNext,int initialDoc) {
    return new NumericDocValues() {
      int position = -1, doc = initialDoc;
      @Override public int docID() { return doc; }
      @Override public int nextDoc() throws IOException {
        if (failNext) { events.add("next-error"); throw new IOException("next failure"); }
        doc = ++position < docs.length ? docs[position] : DocIdSetIterator.NO_MORE_DOCS;
        events.add("next:" + doc);
        return doc;
      }
      @Override public int advance(int target) throws IOException { while (nextDoc() < target) {} return doc; }
      @Override public boolean advanceExact(int target) {
        while (position + 1 < docs.length && docs[position + 1] < target) position++;
        boolean exists = position >= 0 && docs[position] == target;
        if (!exists && position + 1 < docs.length && docs[position + 1] == target) { position++; exists = true; }
        doc = target;
        events.add("exact:" + target + ":" + exists);
        return exists;
      }
      @Override public long longValue() { events.add("value:" + values[position]); return values[position]; }
      @Override public long cost() { return docs.length; }
    };
  }
  static class RecordingProducer extends NormsProducer {
    final List<String> events;
    int[] docs = {0,2,4};
    long[] values = {10,20,30};
    boolean failGet, failNext, failIntegrity;
    int initialDoc = -1, closes, failClose;
    RecordingProducer(List<String> events) { this.events = events; }
    @Override public NumericDocValues getNorms(FieldInfo field) throws IOException {
      events.add("get");
      if (failGet) throw new IOException("get failure");
      return numeric(docs,values,events,failNext,initialDoc);
    }
    @Override public void checkIntegrity() throws IOException {
      events.add("integrity");
      if (failIntegrity) throw new IOException("integrity failure");
    }
    @Override public void close() throws IOException {
      events.add("pclose:" + ++closes);
      if (closes == failClose) throw new IOException("producer close failure");
    }
    @Override public NormsProducer getMergeInstance() { events.add("merge"); return this; }
  }
  static class RecordingConsumer extends NormsConsumer {
    final List<String> events;
    final NormsProducer expected;
    int closes, failClose;
    boolean failAdd;
    RecordingConsumer(List<String> events,NormsProducer expected) { this.events = events; this.expected = expected; }
    @Override public void addNormsField(FieldInfo field,NormsProducer producer) throws IOException {
      if (producer != expected) throw new AssertionError("producer identity changed");
      events.add("delegate-add");
      if (failAdd) throw new IOException("add failure");
    }
    @Override public void close() throws IOException {
      events.add("cclose:" + ++closes);
      if (closes == failClose) throw new IOException("consumer close failure");
    }
  }
  static class RecordingFormat extends NormsFormat {
    final List<String> events;
    final NormsProducer producer;
    final NormsConsumer consumer;
    boolean fail;
    RecordingFormat(List<String> events,NormsProducer producer,NormsConsumer consumer) {
      this.events = events; this.producer = producer; this.consumer = consumer;
    }
    @Override public NormsConsumer normsConsumer(SegmentWriteState state) throws IOException {
      events.add("open-consumer");
      if (fail) throw new IOException("open failure");
      return consumer;
    }
    @Override public NormsProducer normsProducer(SegmentReadState state) throws IOException {
      events.add("open-producer");
      if (fail) throw new IOException("open failure");
      return producer;
    }
  }
  static SegmentInfo segment(Directory directory) {
    byte[] id = new byte[16];
    for (int i = 0; i < id.length; i++) id[i] = (byte)i;
    return new SegmentInfo(directory,Version.LATEST,null,"_an",5,false,false,
        Codec.getDefault(),Map.of(),id,Map.of(),null);
  }
  static String run(String kind) throws Exception {
    List<String> events = new ArrayList<>();
    RecordingProducer delegate = new RecordingProducer(events);
    RecordingConsumer target = new RecordingConsumer(events,delegate);
    String status = "ok", message = "-";
    try (Directory directory = new ByteBuffersDirectory()) {
      try {
        if (kind.startsWith("consumer_")) {
          switch (kind) {
            case "consumer_duplicate" -> delegate.docs = new int[] {0,0,4};
            case "consumer_backwards" -> delegate.docs = new int[] {2,1,4};
            case "consumer_negative" -> delegate.docs = new int[] {-1,2,4};
            case "consumer_past_end" -> delegate.docs = new int[] {0,2,5};
            case "consumer_get_error" -> delegate.failGet = true;
            case "consumer_next_error" -> delegate.failNext = true;
            case "consumer_delegate_error" -> target.failAdd = true;
            case "consumer_close_first" -> target.failClose = 1;
            case "consumer_close_second" -> target.failClose = 2;
          }
          NormsConsumer wrapped = consumer(target);
          if (kind.contains("close")) wrapped.close();
          else wrapped.addNormsField(FIELD,delegate);
        } else if (kind.startsWith("format_")) {
          RecordingFormat f = new RecordingFormat(events,delegate,target);
          f.fail = kind.endsWith("error");
          AssertingNormsFormat wrapped = format(f);
          if (kind.contains("consumer")) wrapped.normsConsumer(new SegmentWriteState(
              InfoStream.NO_OUTPUT,directory,segment(directory),FIELDS,null,IOContext.DEFAULT));
          else wrapped.normsProducer(new SegmentReadState(directory,segment(directory),
              kind.equals("format_no_norms") ? new FieldInfos(new FieldInfo[0]) : FIELDS,IOContext.DEFAULT));
        } else {
          delegate.failGet = kind.equals("producer_get_error");
          delegate.failIntegrity = kind.equals("producer_integrity_error");
          if (kind.equals("producer_positioned")) delegate.initialDoc = 0;
          if (kind.equals("producer_close_first")) delegate.failClose = 1;
          if (kind.equals("producer_close_second")) delegate.failClose = 2;
          NormsProducer wrapped = producer(delegate);
          if (kind.contains("close")) wrapped.close();
          else if (kind.contains("integrity")) wrapped.checkIntegrity();
          else if (kind.equals("producer_merge")) wrapped.getMergeInstance().getNorms(FIELD).nextDoc();
          else if (kind.equals("producer_merge_thread")) {
            NormsProducer merging = wrapped.getMergeInstance();
            AtomicReference<String> outcome = new AtomicReference<>();
            Thread thread = new Thread(() -> {
              try { merging.getNorms(FIELD); outcome.set("ok"); }
              catch (AssertionError e) { outcome.set("assert"); }
              catch (IOException e) { outcome.set("io"); }
            });
            thread.start(); thread.join();
            if (!"assert".equals(outcome.get())) throw new AssertionError("merge thread check missing");
            status = "assert";
          } else {
            NumericDocValues values = wrapped.getNorms(kind.equals("producer_no_norms") ? NO_NORMS : FIELD);
            switch (kind) {
              case "producer_before_value" -> values.longValue();
              case "producer_exact_bounds" -> values.advanceExact(5);
              case "producer_backwards" -> { values.nextDoc(); values.advance(0); }
              case "producer_missing_value" -> { values.advanceExact(1); values.longValue(); }
              default -> { values.nextDoc(); values.longValue(); values.advance(2); values.longValue();
                values.advanceExact(4); values.longValue(); values.nextDoc(); }
            }
          }
        }
      } catch (AssertionError e) { status = "assert"; }
      catch (IOException e) { status = "io"; message = e.getMessage(); }
    }
    return "case\t" + kind + "\t" + status + "\t" + message + "\t" + String.join(",",events) + "\n";
  }
  public static void main(String[] args) throws Exception {
    if (!GenAssertingNorms.class.desiredAssertionStatus()) throw new AssertionError("run with -ea");
    Path out = Path.of(args[0]);
    Files.createDirectories(out);
    StringBuilder trace = new StringBuilder();
    for (String kind : new String[] {"consumer_valid","consumer_duplicate","consumer_backwards",
        "consumer_negative","consumer_past_end","consumer_get_error","consumer_next_error",
        "consumer_delegate_error","consumer_close_valid","consumer_close_first","consumer_close_second",
        "format_consumer","format_consumer_error","format_producer","format_producer_error","format_no_norms",
        "producer_valid","producer_no_norms","producer_positioned","producer_get_error",
        "producer_before_value","producer_exact_bounds","producer_backwards","producer_missing_value",
        "producer_integrity","producer_integrity_error","producer_close_valid","producer_close_first",
        "producer_close_second","producer_merge","producer_merge_thread"}) trace.append(run(kind));
    Files.writeString(out.resolve("asserting_norms.tsv"),trace);
    for (boolean empty : new boolean[] {false,true}) {
      try (Directory directory = new ByteBuffersDirectory()) {
        FieldInfos fields = empty ? new FieldInfos(new FieldInfo[0]) : new FieldInfos(new FieldInfo[] {
            field("dense",0,true),field("sparse",1,true),field("empty",2,true)});
        NormsFormat actual = format(new Lucene90NormsFormat());
        try (NormsConsumer consumer = actual.normsConsumer(new SegmentWriteState(InfoStream.NO_OUTPUT,
            directory,segment(directory),fields,null,IOContext.DEFAULT,"s"))) {
          for (FieldInfo field : fields) consumer.addNormsField(field,new Values());
        }
        // Empty field metadata intentionally rejects the asserting producer.
        NormsFormat reader = empty ? new Lucene90NormsFormat() : actual;
        try (NormsProducer producer = reader.normsProducer(new SegmentReadState(directory,
            segment(directory),fields,IOContext.DEFAULT,"s"))) {
          producer.checkIntegrity();
          for (FieldInfo field : fields) {
            NumericDocValues expected = new Values().getNorms(field), values = producer.getNorms(field);
            for (int doc = 0; doc < 5; doc++) {
              boolean present = expected.advanceExact(doc);
              if (values.advanceExact(doc) != present || present && values.longValue() != expected.longValue())
                throw new AssertionError("real norm values");
            }
          }
        }
        if (!empty) {
          StringBuilder lifecycle = new StringBuilder();
          NormsProducer original = actual.normsProducer(new SegmentReadState(directory,
              segment(directory),fields,IOContext.DEFAULT,"s"));
          NormsProducer merging = original.getMergeInstance();
          original.close();
          NumericDocValues values = merging.getNorms(fields.fieldInfo(1));
          if (values.nextDoc() != 0 || values.longValue() != 7) throw new AssertionError("clone after original close");
          merging.checkIntegrity();
          lifecycle.append("origin-close\t").append(values.longValue()).append('\n');
          merging.close();
          original = actual.normsProducer(new SegmentReadState(directory,
              segment(directory),fields,IOContext.DEFAULT,"s"));
          merging = original.getMergeInstance();
          merging.close();
          values = original.getNorms(fields.fieldInfo(0));
          if (values.nextDoc() != 0 || values.longValue() != -129) throw new AssertionError("original after clone close");
          original.checkIntegrity();
          lifecycle.append("merge-close\t").append(values.longValue()).append('\n');
          original.close();
          Files.writeString(out.resolve("asserting_norms_merge_close.tsv"),lifecycle);
        }
        for (String ext : new String[] {"nvm","nvd"}) {
          try (IndexInput input = directory.openInput("_an_s." + ext,IOContext.DEFAULT)) {
            byte[] bytes = new byte[(int)input.length()]; input.readBytes(bytes,0,bytes.length);
            Files.write(out.resolve("asserting_norms" + (empty ? "_empty" : "") + "." + ext),bytes);
          }
        }
      }
    }
  }
}
