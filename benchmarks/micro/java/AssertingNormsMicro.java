import java.io.IOException;
import java.util.Locale;
import java.util.Map;
import org.apache.lucene.codecs.*;
import org.apache.lucene.codecs.lucene90.Lucene90NormsFormat;
import org.apache.lucene.index.*;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.store.*;
import org.apache.lucene.tests.codecs.asserting.AssertingNormsFormat;
import org.apache.lucene.util.InfoStream;
import org.apache.lucene.util.RamUsageEstimator;
import org.apache.lucene.util.Version;

/** Actual asserting wrappers: counted validation and production norms read/write. */
public class AssertingNormsMicro {
  static volatile long sink;
  static final int BATCH = 256;
  static final byte[] ID = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
  static final Codec CODEC = Codec.getDefault();
  static final FieldInfos FIELDS = new FieldInfos(new FieldInfo[] {field("dense",0),field("sparse",1)});
  static final FieldInfo LARGE_FIELD = field("large",0);
  static FieldInfo field(String name,int number) {
    return new FieldInfo(name,number,false,false,false,IndexOptions.DOCS,DocValuesType.NONE,
        DocValuesSkipIndexType.NONE,-1,Map.of(),0,0,0,0,VectorEncoding.FLOAT32,
        VectorSimilarityFunction.EUCLIDEAN,false,false);
  }
  static SegmentInfo segment(Directory directory) {
    return new SegmentInfo(directory,Version.LATEST,null,"_an",65,false,false,
        CODEC,Map.of(),ID,Map.of(),null);
  }
  static SegmentWriteState writeState(Directory directory) {
    return new SegmentWriteState(InfoStream.NO_OUTPUT,directory,segment(directory),FIELDS,null,IOContext.DEFAULT,"s");
  }
  static class Source extends NormsProducer {
    final int maxDoc;
    final boolean count;
    long gets,nexts,values;
    Source(int maxDoc,boolean count) { this.maxDoc=maxDoc; this.count=count; }
    void reset() { gets=nexts=values=0; }
    @Override public NumericDocValues getNorms(FieldInfo field) {
      if (count) gets++;
      boolean sparse = field.number == 1;
      return new NumericDocValues() {
        int doc=-1;
        @Override public int docID() { return doc; }
        @Override public int nextDoc() {
          if (count) nexts++;
          int stride=sparse?3:1;
          int next=doc<0?0:(doc/stride+1)*stride;
          return doc=next>=maxDoc?DocIdSetIterator.NO_MORE_DOCS:next;
        }
        @Override public int advance(int target) { while(nextDoc()<target) {} return doc; }
        @Override public boolean advanceExact(int target) { doc=target; return !sparse || target%3==0; }
        @Override public long longValue() {
          if (count) values++;
          return sparse?(long)doc*17-33:(long)doc*37-500;
        }
        @Override public long cost() { return sparse?(maxDoc+2)/3:maxDoc; }
      };
    }
    @Override public void close() {}
    @Override public void checkIntegrity() {}
  }
  static class Target extends NormsConsumer {
    final NormsProducer expected;
    long digest,calls;
    Target(NormsProducer expected) { this.expected=expected; }
    @Override public void addNormsField(FieldInfo field,NormsProducer producer) throws IOException {
      if (producer!=expected) throw new AssertionError("producer identity changed");
      calls++;
      NumericDocValues values=producer.getNorms(field);
      while(values.nextDoc()!=DocIdSetIterator.NO_MORE_DOCS) digest=digest*31+values.longValue();
    }
    @Override public void close() {}
  }
  static class Harness implements AutoCloseable {
    final Source source=new Source(65,false),large=new Source(4096,true);
    final Target target=new Target(large);
    final NormsConsumer validator;
    final NormsFormat delegate=new Lucene90NormsFormat();
    final AssertingNormsFormat format=new AssertingNormsFormat();
    final Directory saved=new ByteBuffersDirectory();
    final SegmentReadState savedRead;
    Harness() throws Exception {
      // Both reflection and default-codec construction happen once, before timing.
      var constructor=Class.forName("org.apache.lucene.tests.codecs.asserting.AssertingNormsFormat$AssertingNormsConsumer")
          .getDeclaredConstructor(NormsConsumer.class,int.class);
      constructor.setAccessible(true);
      validator=(NormsConsumer)constructor.newInstance(target,4096);
      var in=AssertingNormsFormat.class.getDeclaredField("in"); in.setAccessible(true); in.set(format,delegate);
      try(NormsConsumer consumer=format.normsConsumer(writeState(saved))) {
        for(FieldInfo field:FIELDS) consumer.addNormsField(field,source);
      }
      savedRead=new SegmentReadState(saved,segment(saved),FIELDS,IOContext.DEFAULT,"s");
    }
    long batch(int kind) throws Exception {
      long digest=0;
      if(kind==0) { large.reset(); target.digest=target.calls=0; }
      for(int i=0;i<BATCH;i++) {
        if(kind==0) validator.addNormsField(LARGE_FIELD,large);
        else if(kind==1) {
          try(NormsProducer producer=format.normsProducer(savedRead)) {
            for(FieldInfo field:FIELDS) {
              NumericDocValues values=producer.getNorms(field);
              digest=digest*31+values.cost();
              while(values.nextDoc()!=DocIdSetIterator.NO_MORE_DOCS) {
                digest=digest*31+values.docID(); digest=digest*31+values.longValue();
              }
            }
          }
        } else {
          try(Directory directory=new ByteBuffersDirectory()) {
            try(NormsConsumer writer=format.normsConsumer(writeState(directory))) {
              for(FieldInfo field:FIELDS) writer.addNormsField(field,source);
            }
            for(String name:new String[] {"_an_s.nvm","_an_s.nvd"}) {
              try(IndexInput input=directory.openInput(name,IOContext.DEFAULT)) {
                long length=input.length();
                for(long p=0;p<length;p++) digest=digest*31+Byte.toUnsignedInt(input.readByte());
              }
            }
          }
        }
      }
      if(kind==0) {
        if(large.gets!=BATCH*2L || large.nexts!=BATCH*2L*4097 || large.values!=BATCH*2L*4096 || target.calls!=BATCH)
          throw new AssertionError("validation/delegate work changed");
        digest=target.digest;
        for(long value:new long[] {large.gets,large.nexts,large.values,target.calls}) digest=digest*31+value;
      }
      return digest;
    }
    long[] run(int kind,long millis) throws Exception {
      long expected=batch(kind);
      long start=System.nanoTime(),count=0;
      do {
        long actual=batch(kind);
        if(actual!=expected)throw new AssertionError("timed batch digest changed");
        sink=actual; count+=BATCH;
      } while(System.nanoTime()-start<millis*1_000_000L);
      return new long[] {System.nanoTime()-start,count};
    }
    @Override public void close() throws Exception { saved.close(); }
  }
  public static void main(String[] args) throws Exception {
    if(!AssertingNormsMicro.class.desiredAssertionStatus()) throw new AssertionError("run with -ea");
    long warmup=Long.parseLong(args[0]),measure=Long.parseLong(args[1]);
    try(Harness harness=new Harness()) {
      System.out.printf("asserting_norms_format_owned_bytes\t%d%n",
          RamUsageEstimator.shallowSizeOf(harness.format)+RamUsageEstimator.shallowSizeOf(harness.delegate));
      String[] names={"asserting_norms_validation","asserting_norms_read","asserting_norms_lifecycle"};
      for(int kind=0;kind<names.length;kind++) {
        System.out.printf("#check\t%s\t%s\t%d%n",names[kind],Long.toUnsignedString(harness.batch(kind)),BATCH);
        harness.run(kind,warmup);
        long[] result=harness.run(kind,measure);
        System.out.printf(Locale.ROOT,"%s\t%.3f\t%d%n",names[kind],(double)result[0]/result[1],result[1]);
      }
    }
  }
}
