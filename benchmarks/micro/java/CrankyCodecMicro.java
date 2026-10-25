import java.io.IOException;
import java.util.Collection;
import java.util.Random;
import org.apache.lucene.codecs.*;
import org.apache.lucene.index.*;
import org.apache.lucene.store.*;
import org.apache.lucene.tests.codecs.cranky.CrankyCodec;
import org.apache.lucene.util.*;

/** Actual CrankyCodec over inert probes: composition costs, not native codec I/O. */
public class CrankyCodecMicro {
  static volatile Object objectSink;
  static volatile long sink;
  static long hash(long digest,long value) { return (digest^value)*0x100000001b3L; }
  static final class Probe extends Codec {
    long factories,operations,displays,trace;
    boolean alternate;
    Probe() { super("CodecProbe"); }
    void call() { operations++; }
    <T> T component(int kind,T value) { factories++;trace=hash(trace,kind);return value; }
    final DocValuesFormat dv=new DocValuesFormat("ProbeDV") {
      public DocValuesConsumer fieldsConsumer(SegmentWriteState s) { throw new AssertionError("unused"); }
      public DocValuesProducer fieldsProducer(SegmentReadState s) { throw new AssertionError("unused"); }
    };
    final FieldInfosFormat fi=new FieldInfosFormat() {
      public FieldInfos read(Directory d,SegmentInfo s,String suffix,IOContext c) {call();return FieldInfos.EMPTY;}
      public void write(Directory d,SegmentInfo s,String suffix,FieldInfos f,IOContext c) {call();}
    };
    final LiveDocsFormat live=new LiveDocsFormat() {
      public Bits readLiveDocs(Directory d,SegmentCommitInfo s,IOContext c) {call();return new FixedBitSet(1);}
      public void writeLiveDocs(Bits b,Directory d,SegmentCommitInfo s,int n,IOContext c) {call();}
      public void files(SegmentCommitInfo s,Collection<String> files) {call();files.add("delegate-file");}
    };
    final NormsFormat norms=new NormsFormat() {
      public NormsConsumer normsConsumer(SegmentWriteState s) {throw new AssertionError("unused");}
      public NormsProducer normsProducer(SegmentReadState s) {throw new AssertionError("unused");}
    };
    final PostingsFormat postings=new PostingsFormat("ProbePostings") {
      public FieldsConsumer fieldsConsumer(SegmentWriteState s) {throw new AssertionError("unused");}
      public FieldsProducer fieldsProducer(SegmentReadState s) {throw new AssertionError("unused");}
    };
    final SegmentInfoFormat segment=new SegmentInfoFormat() {
      public SegmentInfo read(Directory d,String name,byte[] id,IOContext c) {throw new AssertionError("unused");}
      public void write(Directory d,SegmentInfo s,IOContext c) {call();}
    };
    final StoredFieldsFormat stored=new StoredFieldsFormat() {
      public StoredFieldsReader fieldsReader(Directory d,SegmentInfo s,FieldInfos f,IOContext c) {throw new AssertionError("unused");}
      public StoredFieldsWriter fieldsWriter(Directory d,SegmentInfo s,IOContext c) {throw new AssertionError("unused");}
    };
    final TermVectorsFormat vectors=new TermVectorsFormat() {
      public TermVectorsReader vectorsReader(Directory d,SegmentInfo s,FieldInfos f,IOContext c) {throw new AssertionError("unused");}
      public TermVectorsWriter vectorsWriter(Directory d,SegmentInfo s,IOContext c) {throw new AssertionError("unused");}
    };
    final CompoundFormat compound=new CompoundFormat() {
      public CompoundDirectory getCompoundReader(Directory d,SegmentInfo s) {throw new AssertionError("unused");}
      public void write(Directory d,SegmentInfo s,IOContext c) {call();}
    };
    final PointsFormat points=new PointsFormat() {
      public PointsReader fieldsReader(SegmentReadState s) {throw new AssertionError("unused");}
      public PointsWriter fieldsWriter(SegmentWriteState s) {throw new AssertionError("unused");}
    };
    final KnnVectorsFormat knn=new KnnVectorsFormat("ProbeKnn") {
      public KnnVectorsReader fieldsReader(SegmentReadState s) {throw new AssertionError("unused");}
      public KnnVectorsWriter fieldsWriter(SegmentWriteState s) {throw new AssertionError("unused");}
      public int getMaxDimensions(String field) {throw new AssertionError("unused");}
    };
    public DocValuesFormat docValuesFormat() {return component(0,dv);}
    public FieldInfosFormat fieldInfosFormat() {return component(1,fi);}
    public LiveDocsFormat liveDocsFormat() {return component(2,live);}
    public NormsFormat normsFormat() {return component(3,norms);}
    public PostingsFormat postingsFormat() {return component(4,postings);}
    public SegmentInfoFormat segmentInfoFormat() {return component(5,segment);}
    public StoredFieldsFormat storedFieldsFormat() {return component(6,stored);}
    public TermVectorsFormat termVectorsFormat() {return component(7,vectors);}
    public CompoundFormat compoundFormat() {return component(8,compound);}
    public PointsFormat pointsFormat() {return component(9,points);}
    public KnnVectorsFormat knnVectorsFormat() {return component(10,knn);}
    public String toString() {displays++;return alternate?"DelegateNext":"DelegateFirst";}
  }
  static long pair(Object first,Object second,long digest) {
    objectSink=first;objectSink=second;
    if(first==second)throw new AssertionError("wrapper reused");
    return hash(digest,1);
  }
  static long batch(int kind,CrankyCodec codec,Probe delegate,Random random) throws Exception {
    random.setSeed(42);delegate.factories=0;delegate.trace=0xcbf29ce484222325L;delegate.operations=0;delegate.displays=0;
    long digest=0xcbf29ce484222325L;
    switch(kind) {
      case 0->{
        for(int step=0;step<256;step++) {
          digest=pair(codec.docValuesFormat(),codec.docValuesFormat(),digest);
          digest=pair(codec.fieldInfosFormat(),codec.fieldInfosFormat(),digest);
          digest=pair(codec.liveDocsFormat(),codec.liveDocsFormat(),digest);
          digest=pair(codec.normsFormat(),codec.normsFormat(),digest);
          digest=pair(codec.postingsFormat(),codec.postingsFormat(),digest);
          digest=pair(codec.segmentInfoFormat(),codec.segmentInfoFormat(),digest);
          digest=pair(codec.storedFieldsFormat(),codec.storedFieldsFormat(),digest);
          digest=pair(codec.termVectorsFormat(),codec.termVectorsFormat(),digest);
          digest=pair(codec.compoundFormat(),codec.compoundFormat(),digest);
          digest=pair(codec.pointsFormat(),codec.pointsFormat(),digest);
        }
        if(delegate.factories!=5120||delegate.operations!=0) throw new AssertionError("factory count");
      }
      case 1->{
        for(int step=0;step<256;step++) {
          var first=codec.knnVectorsFormat();objectSink=first;
          var second=codec.knnVectorsFormat();objectSink=second;
          if(first!=delegate.knn||first!=second) throw new AssertionError("knn identity");
          digest=hash(digest,1);
        }
        if(delegate.factories!=512) throw new AssertionError("knn factory count");
      }
      case 2->{
        FieldInfosFormat[] held={codec.fieldInfosFormat(),codec.fieldInfosFormat()};
        var live=codec.liveDocsFormat();var segment=codec.segmentInfoFormat();var compound=codec.compoundFormat();
        objectSink=held;objectSink=live;objectSink=segment;objectSink=compound;
        if(held[0]==held[1]) throw new AssertionError("held wrapper reused");
        for(int step=0;step<256;step++) {
          String error=null;
          try {
            switch(step%5) {
              case 0,1->held[step%2].write(null,null,"",FieldInfos.EMPTY,IOContext.DEFAULT);
              case 2->live.writeLiveDocs(null,null,null,0,IOContext.DEFAULT);
              case 3->segment.write(null,null,IOContext.DEFAULT);
              case 4->compound.write(null,null,IOContext.DEFAULT);
            }
          } catch(IOException e) {error=e.getMessage();}
          if(error==null) digest=hash(digest,1);
          else {digest=hash(digest,0);for(int i=0;i<error.length();i++)digest=hash(digest,error.charAt(i));}
        }
        if(delegate.factories!=5)throw new AssertionError("held factory count");
      }
      case 3->{
        for(int step=0;step<256;step++) {
          delegate.alternate=step%2==1;
          String value=codec.toString();objectSink=value;
          for(int i=0;i<value.length();i++)digest=hash(digest,value.charAt(i));
        }
        if(delegate.factories!=0||delegate.displays!=256)throw new AssertionError("display count");
      }
      default->throw new AssertionError(kind);
    }
    digest=hash(digest,delegate.factories);digest=hash(digest,delegate.operations);digest=hash(digest,delegate.displays);
    digest=hash(digest,delegate.trace);digest=hash(digest,random.nextInt());sink=digest;return digest;
  }
  public static void main(String[] args) throws Exception {
    if(!CrankyCodecMicro.class.desiredAssertionStatus())throw new AssertionError("run with -ea");
    long warmup=Long.parseLong(args[0]),measure=Long.parseLong(args[1]);
    var delegate=new Probe();var random=new Random(42);var codec=new CrankyCodec(delegate,random);
    if(codec.getName()!=delegate.getName())throw new AssertionError("codec name not shared");
    System.out.printf("cranky_codec_shallow_adapter_bytes\t%d%n",RamUsageEstimator.shallowSizeOf(codec));
    System.out.println("cranky_codec_name_owned_bytes\t0");
    System.out.printf("cranky_codec_shared_rng_bytes\t%d%n",RamUsageEstimator.shallowSizeOf(random)+RamUsageEstimator.shallowSizeOf(new java.util.concurrent.atomic.AtomicLong()));
    String[] names={"cranky_codec_components","cranky_codec_knn","cranky_codec_shared_guards","cranky_codec_display"};
    for(int kind=0;kind<names.length;kind++) {
      long expected=batch(kind,codec,delegate,random);
      for(int phase=0;phase<2;phase++) {
        long budget=phase==0?warmup:measure,start=System.nanoTime(),operations=0;
        do {if(batch(kind,codec,delegate,random)!=expected)throw new AssertionError("unstable digest");operations+=256;}
        while(System.nanoTime()-start<budget*1000000);
        if(phase==1)System.out.printf(java.util.Locale.ROOT,"%s\t%.3f\t%d%n",names[kind],(double)(System.nanoTime()-start)/operations,operations);
      }
      System.out.printf("#check\t%s\t%s\t256%n",names[kind],Long.toUnsignedString(expected));
    }
  }
}
