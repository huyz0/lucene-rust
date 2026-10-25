import java.lang.reflect.Field;
import java.util.ArrayList;
import org.apache.lucene.codecs.*;
import org.apache.lucene.codecs.compressing.CompressionMode;
import org.apache.lucene.codecs.lucene104.Lucene104Codec;
import org.apache.lucene.index.*;
import org.apache.lucene.store.*;
import org.apache.lucene.tests.codecs.compressing.*;
import org.apache.lucene.tests.codecs.compressing.dummy.DummyCompressingCodec;
import org.apache.lucene.util.*;

/** Actual Java composition/accessors/display; constructors and I/O are not timed. */
public class CompressingCodecMicro {
  static volatile Object objectSink;
  static volatile long sink;
  static final long BASIS=0xcbf29ce484222325L;
  static final String[] LABELS={"fast","fast_decompression","high_compression","dummy","deflate_preset","lz4_preset"};
  static final String[] COMPONENTS={"docValuesFormat","fieldInfosFormat","liveDocsFormat","normsFormat","postingsFormat","segmentInfoFormat","compoundFormat","pointsFormat","knnVectorsFormat"};
  static long hash(long digest,long value) { return (digest^value)*0x100000001b3L; }
  static long text(long digest,String value) {
    digest=hash(digest,value.length());
    for(int i=0;i<value.length();i++) {
      digest=hash(digest,value.charAt(i));
    }
    return digest;
  }
  static Object member(Object value,String name) throws Exception {
    for(Class<?> type=value.getClass();type!=null;type=type.getSuperclass()) {
      try { Field f=type.getDeclaredField(name);f.setAccessible(true);return f.get(value); }
      catch(NoSuchFieldException ignored) {}
    }
    throw new NoSuchFieldException(name);
  }
  static CompressingCodec defaults(int kind) {
    return switch(kind) {
      case 0->new FastCompressingCodec();case 1->new FastDecompressionCompressingCodec();
      case 2->new HighCompressionCompressingCodec();case 3->new DummyCompressingCodec();
      case 4->new DeflateWithPresetCompressingCodec();case 5->new LZ4WithPresetCompressingCodec();
      default->throw new AssertionError(kind);
    };
  }
  static CompressingCodec configured(int kind,boolean suffix,int shift) {
    return switch(kind) {
      case 0->new FastCompressingCodec(37,3,suffix,shift);
      case 1->new FastDecompressionCompressingCodec(37,3,suffix,shift);
      case 2->new HighCompressionCompressingCodec(37,3,suffix,shift);
      case 3->new DummyCompressingCodec(37,3,suffix,shift);
      case 4->new DeflateWithPresetCompressingCodec(37,3,suffix,shift);
      case 5->new LZ4WithPresetCompressingCodec(37,3,suffix,shift);
      default->throw new AssertionError(kind);
    };
  }
  static final class BaseCodec extends CompressingCodec {
    BaseCodec(String name,String suffix,CompressionMode mode,int chunk,int docs,int shift) {
      super(name,suffix,mode,chunk,docs,shift);
    }
  }
  static CompressingCodec[] configurations() {
    var codecs=new ArrayList<CompressingCodec>();
    for(int kind=0;kind<6;kind++) {
      codecs.add(defaults(kind));codecs.add(configured(kind,false,2));codecs.add(configured(kind,true,22));
    }
    codecs.add(new BaseCodec("CustomCompressingData","",CompressionMode.FAST,11,7,2));
    codecs.add(new BaseCodec("CustomCompressingData","privateS",CompressionMode.FAST,11,7,22));
    codecs.add(new BaseCodec("CustomCompressingData","privateS",CompressionMode.FAST,Integer.MAX_VALUE,Integer.MAX_VALUE,22));
    codecs.add(new BaseCodec("","",CompressionMode.FAST,1,1,2));
    codecs.add(new BaseCodec("CustomCompressingData","",null,1,1,2));
    return codecs.toArray(CompressingCodec[]::new);
  }
  static long configurationCheck(CompressingCodec[] codecs) throws Exception {
    long digest=BASIS;
    for(var codec:codecs) {
      digest=text(digest,codec.getClass().getSimpleName());digest=text(digest,codec.getName());digest=text(digest,codec.toString());
      var sf=codec.storedFieldsFormat();var tv=codec.termVectorsFormat();
      if(sf!=codec.storedFieldsFormat()||tv!=codec.termVectorsFormat())throw new AssertionError("unstable selected formats");
      if(member(sf,"compressionMode")!=member(tv,"compressionMode"))throw new AssertionError("mode identity");
      digest=hash(digest,1);
      for(Object settings:new Object[]{sf,tv}) {
        digest=text(digest,(String)member(settings,"formatName"));
        digest=text(digest,(String)member(settings,"segmentSuffix"));
        digest=text(digest,String.valueOf(member(settings,"compressionMode")));
        digest=hash(digest,(int)member(settings,"chunkSize"));
        digest=hash(digest,(int)member(settings,"maxDocsPerChunk"));
        digest=hash(digest,(int)member(settings,settings==sf?"blockShift":"blockSize"));
      }
    }
    for(int kind=0;kind<6;kind++) {
      var first=defaults(kind);var second=defaults(kind);
      boolean shared=member(first.storedFieldsFormat(),"compressionMode")==member(second.storedFieldsFormat(),"compressionMode");
      if(shared!=(kind<4))throw new AssertionError("cross-codec mode allocation");
      digest=hash(digest,shared?1:0);
    }
    sink=digest;return digest;
  }
  static final class State { long calls,trace;int fail=-1; }
  static final class Probe extends FilterCodec {
    final State state;
    final IllegalStateException[] errors=new IllegalStateException[9];
    Object last;
    Probe(State state) {
      super("CompressingProbe",new Lucene104Codec());this.state=state;
      for(int i=0;i<9;i++)errors[i]=new IllegalStateException("delegate failure:"+COMPONENTS[i]);
    }
    <T> T component(int kind,T value) {
      state.calls++;state.trace=hash(state.trace,kind);
      if(state.fail==kind)throw errors[kind];
      last=value;return value;
    }
    final DocValuesFormat dv=new DocValuesFormat("ProbeDV") {
      public DocValuesConsumer fieldsConsumer(SegmentWriteState s){throw new AssertionError("unused");}
      public DocValuesProducer fieldsProducer(SegmentReadState s){throw new AssertionError("unused");}
    };
    final FieldInfosFormat fi=new FieldInfosFormat() {
      public FieldInfos read(Directory d,SegmentInfo s,String suffix,IOContext c){throw new AssertionError("unused");}
      public void write(Directory d,SegmentInfo s,String suffix,FieldInfos f,IOContext c){throw new AssertionError("unused");}
    };
    final LiveDocsFormat live=new LiveDocsFormat() {
      public Bits readLiveDocs(Directory d,SegmentCommitInfo s,IOContext c){throw new AssertionError("unused");}
      public void writeLiveDocs(Bits b,Directory d,SegmentCommitInfo s,int n,IOContext c){throw new AssertionError("unused");}
      public void files(SegmentCommitInfo s,java.util.Collection<String> files){throw new AssertionError("unused");}
    };
    final NormsFormat norms=new NormsFormat() {
      public NormsConsumer normsConsumer(SegmentWriteState s){throw new AssertionError("unused");}
      public NormsProducer normsProducer(SegmentReadState s){throw new AssertionError("unused");}
    };
    final PostingsFormat postings=new PostingsFormat("ProbePostings") {
      public FieldsConsumer fieldsConsumer(SegmentWriteState s){throw new AssertionError("unused");}
      public FieldsProducer fieldsProducer(SegmentReadState s){throw new AssertionError("unused");}
    };
    final SegmentInfoFormat segment=new SegmentInfoFormat() {
      public SegmentInfo read(Directory d,String name,byte[] id,IOContext c){throw new AssertionError("unused");}
      public void write(Directory d,SegmentInfo s,IOContext c){throw new AssertionError("unused");}
    };
    final CompoundFormat compound=new CompoundFormat() {
      public CompoundDirectory getCompoundReader(Directory d,SegmentInfo s){throw new AssertionError("unused");}
      public void write(Directory d,SegmentInfo s,IOContext c){throw new AssertionError("unused");}
    };
    static PointsFormat freshPoints() {
      return new PointsFormat() {
        public PointsReader fieldsReader(SegmentReadState s){throw new AssertionError("unused");}
        public PointsWriter fieldsWriter(SegmentWriteState s){throw new AssertionError("unused");}
      };
    }
    final KnnVectorsFormat knn=new KnnVectorsFormat("ProbeKnn") {
      public KnnVectorsReader fieldsReader(SegmentReadState s){throw new AssertionError("unused");}
      public KnnVectorsWriter fieldsWriter(SegmentWriteState s){throw new AssertionError("unused");}
      public int getMaxDimensions(String field){throw new AssertionError("unused");}
    };
    public DocValuesFormat docValuesFormat(){return component(0,dv);}
    public FieldInfosFormat fieldInfosFormat(){return component(1,fi);}
    public LiveDocsFormat liveDocsFormat(){return component(2,live);}
    public NormsFormat normsFormat(){return component(3,norms);}
    public PostingsFormat postingsFormat(){return component(4,postings);}
    public SegmentInfoFormat segmentInfoFormat(){return component(5,segment);}
    public CompoundFormat compoundFormat(){return component(6,compound);}
    public PointsFormat pointsFormat(){
      state.calls++;state.trace=hash(state.trace,7);
      if(state.fail==7)throw errors[7];
      var value=freshPoints();last=value;return value;
    }
    public KnnVectorsFormat knnVectorsFormat(){return component(8,knn);}
  }
  static Object component(CompressingCodec codec,int kind) {
    return switch(kind) {
      case 0->codec.docValuesFormat();case 1->codec.fieldInfosFormat();case 2->codec.liveDocsFormat();
      case 3->codec.normsFormat();case 4->codec.postingsFormat();case 5->codec.segmentInfoFormat();
      case 6->codec.compoundFormat();case 7->codec.pointsFormat();case 8->codec.knnVectorsFormat();
      default->throw new AssertionError(kind);
    };
  }
  static long batch(int kind,CompressingCodec[] codecs,CompressingCodec[] delegates,Probe[] probes,State state) {
    state.calls=0;state.trace=BASIS;state.fail=-1;long digest=BASIS;
    for(int step=0;step<256;step++) {
      var codec=codecs[step%codecs.length];
      switch(kind) {
        case 0->{
          var sf=codec.storedFieldsFormat();objectSink=sf;var tv=codec.termVectorsFormat();objectSink=tv;
          var sf2=codec.storedFieldsFormat();objectSink=sf2;var tv2=codec.termVectorsFormat();objectSink=tv2;
          if(sf!=sf2||tv!=tv2)throw new AssertionError("cached identity");
          digest=hash(digest,1);String name=codec.getName();objectSink=name;digest=text(digest,name);
        }
        case 1->{String value=codec.storedFieldsFormat().toString();objectSink=value;digest=text(digest,value);}
        case 2->{String value=codec.termVectorsFormat().toString();objectSink=value;digest=text(digest,value);}
        case 3->{String value=codec.toString();objectSink=value;digest=text(digest,value);}
        case 4->{
          int index=step%delegates.length;
          for(int selected=0;selected<9;selected++) {
            Object first=component(delegates[index],selected);objectSink=first;
            if(first!=probes[index].last)throw new AssertionError("first exact handle");
            Object second=component(delegates[index],selected);objectSink=second;
            if(second!=probes[index].last)throw new AssertionError("second exact handle");
            boolean same=first==second;if(same!=(selected!=7))throw new AssertionError("cached/fresh probe");
            digest=hash(digest,same?1:0);
          }
        }
        case 5->{
          int index=step%delegates.length,selected=step%9;state.fail=selected;
          try {component(delegates[index],selected);throw new AssertionError("missing error");}
          catch(IllegalStateException error) {
            objectSink=error;if(error!=probes[index].errors[selected])throw new AssertionError("exact error");
            digest=hash(digest,1);digest=text(digest,error.getMessage());
          }
        }
        default->throw new AssertionError(kind);
      }
    }
    if(state.calls!=(kind==4?256*18:kind==5?256:0))throw new AssertionError("callback count");
    digest=hash(digest,state.calls);digest=hash(digest,state.trace);sink=digest;return digest;
  }
  static void memoryRows() throws Exception {
    for(int kind=0;kind<6;kind++) {
      var codec=defaults(kind);var sf=codec.storedFieldsFormat();var tv=codec.termVectorsFormat();
      long shallow=RamUsageEstimator.shallowSizeOf(codec)+RamUsageEstimator.shallowSizeOf(sf)+RamUsageEstimator.shallowSizeOf(tv);
      long mode=kind<4?0:RamUsageEstimator.shallowSizeOf(member(sf,"compressionMode"));
      System.out.printf("compressing_%s_selected_shallow_bytes\t%d%n",LABELS[kind],shallow);
      System.out.printf("compressing_%s_owned_string_bytes\t0%n",LABELS[kind]);
      System.out.printf("compressing_%s_owned_mode_bytes\t%d%n",LABELS[kind],mode);
    }
  }
  public static void main(String[] args) throws Exception {
    if(!CompressingCodecMicro.class.desiredAssertionStatus())throw new AssertionError("run with -ea");
    long warmup=Long.parseLong(args[0]),measure=Long.parseLong(args[1]);
    var codecs=configurations();long config=configurationCheck(codecs);var state=new State();
    var delegates=new CompressingCodec[6];var probes=new Probe[6];
    Field delegateField=FilterCodec.class.getDeclaredField("delegate");delegateField.setAccessible(true);
    for(int kind=0;kind<6;kind++) {
      delegates[kind]=defaults(kind);probes[kind]=new Probe(state);delegateField.set(delegates[kind],probes[kind]);
    }
    memoryRows();System.out.printf("#setup\tcompressing_configuration\t%s\t23%n",Long.toUnsignedString(config));
    String[] names={"compressing_cached","compressing_stored_display","compressing_vectors_display","compressing_codec_display","compressing_delegation","compressing_delegate_errors"};
    for(int kind=0;kind<names.length;kind++) {
      long expected=batch(kind,codecs,delegates,probes,state);
      for(int phase=0;phase<2;phase++) {
        long budget=phase==0?warmup:measure,start=System.nanoTime(),operations=0;
        do {if(batch(kind,codecs,delegates,probes,state)!=expected)throw new AssertionError("unstable digest");operations+=256;}
        while(System.nanoTime()-start<budget*1000000);
        if(phase==1)System.out.printf(java.util.Locale.ROOT,"%s\t%.3f\t%d%n",names[kind],(double)(System.nanoTime()-start)/operations,operations);
      }
      System.out.printf("#check\t%s\t%s\t256\t%s%n",names[kind],Long.toUnsignedString(expected),Long.toUnsignedString(config));
    }
  }
}
