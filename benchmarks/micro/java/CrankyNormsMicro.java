package org.apache.lucene.tests.codecs.cranky;

import java.io.IOException;
import java.util.Locale;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.atomic.AtomicLong;
import org.apache.lucene.codecs.*;
import org.apache.lucene.codecs.lucene90.Lucene90NormsFormat;
import org.apache.lucene.index.*;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.store.*;
import org.apache.lucene.util.InfoStream;
import org.apache.lucene.util.RamUsageEstimator;
import org.apache.lucene.util.Version;

/** Actual cranky wrapper and production Lucene90 norms, including real directory I/O. */
public class CrankyNormsMicro {
  static volatile long sink;
  static final byte[] ID = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
  static final FieldInfos FIELDS = new FieldInfos(new FieldInfo[] {field("dense",0),field("sparse",1)});
  static final NormsProducer VALUES = new Values();
  static FieldInfo field(String name, int number) {
    return new FieldInfo(name,number,false,false,false,IndexOptions.DOCS,DocValuesType.NONE,
        DocValuesSkipIndexType.NONE,-1,Map.of(),0,0,0,0,VectorEncoding.FLOAT32,
        VectorSimilarityFunction.EUCLIDEAN,false,false);
  }
  static class Values extends NormsProducer {
    @Override public NumericDocValues getNorms(FieldInfo field) {
      int[] docs = field.number == 0 ? new int[] {0,1,2,3,4} : new int[] {0,2,4};
      long[] values = field.number == 0 ? new long[] {-129,0,127,128,32768} : new long[] {7,8,9};
      return new NumericDocValues() {
        int position = -1, doc = -1;
        @Override public int docID() { return doc; }
        @Override public int nextDoc() { return doc = ++position < docs.length ? docs[position] : DocIdSetIterator.NO_MORE_DOCS; }
        @Override public int advance(int target) { while (nextDoc() < target) {} return doc; }
        @Override public boolean advanceExact(int target) { while (doc < target) nextDoc(); return doc == target; }
        @Override public long longValue() { return values[position]; }
        @Override public long cost() { return docs.length; }
      };
    }
    @Override public void close() {}
    @Override public void checkIntegrity() {}
  }
  static SegmentInfo segment(Directory directory) {
    return new SegmentInfo(directory,Version.LATEST,null,"_cn",5,false,false,
        Codec.getDefault(),Map.of(),ID,Map.of(),null);
  }
  static SegmentWriteState writeState(Directory directory) {
    return new SegmentWriteState(InfoStream.NO_OUTPUT,directory,segment(directory),FIELDS,
        null,IOContext.DEFAULT,"s");
  }
  static void expected(IOException error, String operation) {
    String message = "Fake IOException from " + operation + "()";
    if (!message.equals(error.getMessage())) throw new AssertionError(error);
  }
  static long batch(Directory saved, boolean write) throws Exception {
    Random random = new Random(0);
    NormsFormat format = new CrankyNormsFormat(new Lucene90NormsFormat(),random);
    long digest = 0;
    for (int i = 0; i < 256; i++) {
      if (write) {
        try (Directory directory = new ByteBuffersDirectory()) {
          NormsConsumer consumer;
          try { consumer = format.normsConsumer(writeState(directory)); }
          catch (IOException e) {
            expected(e,"NormsFormat.normsConsumer");
            digest = digest * 31 + 101;
            if (directory.listAll().length != 0) throw new AssertionError("opening guard created files");
            continue;
          }
          for (FieldInfo field : FIELDS) {
            try { consumer.addNormsField(field,VALUES); }
            catch (IOException e) { expected(e,"NormsConsumer.addNormsField"); digest = digest * 31 + 103; }
          }
          try { consumer.close(); }
          catch (IOException e) { expected(e,"NormsConsumer.close"); digest = digest * 31 + 107; }
          for (String name : new String[] {"_cn_s.nvm","_cn_s.nvd"}) {
            try (IndexInput input = directory.openInput(name,IOContext.DEFAULT)) {
              long length = input.length();
              for (long p = 0; p < length; p++) digest = digest * 31 + Byte.toUnsignedInt(input.readByte());
            }
          }
        }
      } else {
        try (NormsProducer producer = format.normsProducer(new SegmentReadState(saved,
            segment(saved),FIELDS,IOContext.DEFAULT,"s"))) {
          for (FieldInfo field : FIELDS) {
            NumericDocValues values = producer.getNorms(field);
            for (int doc = 0; doc < 5; doc++) digest = digest * 31 + (values.advanceExact(doc) ? values.longValue() : -777);
          }
        }
      }
    }
    return digest + Integer.toUnsignedLong(random.nextInt());
  }
  static long[] run(Directory directory, boolean write, long millis) throws Exception {
    long start = System.nanoTime(), count = 0;
    do { sink = batch(directory,write); count += 256; }
    while (System.nanoTime() - start < millis * 1_000_000L);
    return new long[] {System.nanoTime() - start,count};
  }
  public static void main(String[] args) throws Exception {
    long warmup = Long.parseLong(args[0]), measure = Long.parseLong(args[1]);
    Random random = new Random(0);
    NormsFormat delegate = new Lucene90NormsFormat();
    NormsFormat format = new CrankyNormsFormat(delegate,random);
    long owned = RamUsageEstimator.shallowSizeOf(format) + RamUsageEstimator.shallowSizeOf(delegate)
        + RamUsageEstimator.shallowSizeOf(random) + RamUsageEstimator.shallowSizeOfInstance(AtomicLong.class);
    System.out.printf("cranky_norms_format_owned_bytes\t%d%n",owned);
    try (Directory directory = new ByteBuffersDirectory()) {
      try (NormsConsumer consumer = delegate.normsConsumer(writeState(directory))) {
        for (FieldInfo field : FIELDS) consumer.addNormsField(field,VALUES);
      }
      for (boolean write : new boolean[] {false,true}) {
        String name = write ? "cranky_norms_lifecycle" : "cranky_norms_read";
        System.out.printf("#check\t%s\t%s\t256%n",name,Long.toUnsignedString(batch(directory,write)));
        run(directory,write,warmup);
        long[] result = run(directory,write,measure);
        System.out.printf(Locale.ROOT,"%s\t%.3f\t%d%n",name,(double)result[0]/result[1],result[1]);
      }
    }
  }
}
