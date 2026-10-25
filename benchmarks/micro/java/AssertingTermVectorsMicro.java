import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Executor;
import org.apache.lucene.codecs.*;
import org.apache.lucene.codecs.simpletext.SimpleTextTermVectorsFormat;
import org.apache.lucene.index.*;
import org.apache.lucene.store.*;
import org.apache.lucene.tests.codecs.asserting.AssertingTermVectorsFormat;
import org.apache.lucene.util.*;

/** Assertions and full callback/value/output consumption stay in timed batches. */
public class AssertingTermVectorsMicro {
  static final int N=256;
  static final byte[] POS={3,2,0,10,4,1,0}, REAL_POS={3,2,0,10,4,0}, OFF={0,2,1,3,0,0};
  static final FieldInfo FIELD=field(0);
  static volatile long sink;
  static FieldInfo field(int n){return field("f",n);}
  static FieldInfo field(String name,int n){return new FieldInfo(name,n,true,false,false,IndexOptions.DOCS_AND_FREQS_AND_POSITIONS_AND_OFFSETS,
      DocValuesType.NONE,DocValuesSkipIndexType.NONE,-1,Map.of(),0,0,0,0,VectorEncoding.FLOAT32,VectorSimilarityFunction.EUCLIDEAN,false,false);}
  static SegmentInfo info(Directory d,int docs){return new SegmentInfo(d,Version.LATEST,Version.LATEST,"_atv",docs,false,false,null,Map.of(),"0123456789abcdef".getBytes(StandardCharsets.US_ASCII),Map.of(),null);}
  static TermVectorsFormat wrap(TermVectorsFormat delegate)throws Exception {
    var format=new AssertingTermVectorsFormat();var f=AssertingTermVectorsFormat.class.getDeclaredField("in");f.setAccessible(true);f.set(format,delegate);return format;
  }
  static class Digest extends TermVectorsWriter {
    long value;
    void mix(long n){value=value*31+n;}
    void bytes(byte[] b,int offset,int length){mix(length);for(int i=offset;i<offset+length;i++)mix(b[i]&255);}
    void bytes(BytesRef b){bytes(b.bytes,b.offset,b.length);}
    @Override public void startDocument(int n){mix(1);mix(n);}
    @Override public void finishDocument(){mix(2);}
    @Override public void startField(FieldInfo f,int n,boolean p,boolean o,boolean y){mix(3);byte[] name=f.name.getBytes(StandardCharsets.UTF_8);bytes(name,0,name.length);mix(f.number);mix(n);mix(p?1:0);mix(o?1:0);mix(y?1:0);}
    @Override public void finishField(){mix(4);}
    @Override public void startTerm(BytesRef t,int n){mix(5);bytes(t);mix(n);}
    @Override public void finishTerm(){mix(6);}
    @Override public void addPosition(int p,int s,int e,BytesRef y){mix(7);mix(p);mix(s);mix(e);if(y==null)mix(-1);else bytes(y);}
    @Override public void finish(int n){mix(8);mix(n);}
    @Override public void close(){mix(9);}
    @Override public long ramBytesUsed(){return 37;}
  }
  static class ProbeFormat extends TermVectorsFormat {
    Digest last;
    @Override public TermVectorsWriter vectorsWriter(Directory d,SegmentInfo s,IOContext c){last=new Digest();return last;}
    @Override public TermVectorsReader vectorsReader(Directory d,SegmentInfo s,FieldInfos f,IOContext c){throw new AssertionError("unused");}
  }
  static void document(TermVectorsWriter w,byte[] positions)throws IOException {
    w.startDocument(1);w.startField(FIELD,1,true,true,true);w.startTerm(new BytesRef(new byte[]{0,10,92,(byte)255}),3);
    w.addProx(3,new ByteArrayDataInput(positions),new ByteArrayDataInput(OFF));w.finishTerm();w.finishField();w.finishDocument();
  }
  static void consume(TermVectorsReader r,int doc,Digest d)throws IOException {
    Fields fields=r.get(doc);d.startDocument(fields==null?0:fields.size());
    if(fields!=null)for(String name:fields){Terms terms=fields.terms(name);d.startField(field(name,7),(int)terms.size(),terms.hasPositions(),terms.hasOffsets(),terms.hasPayloads());TermsEnum te=terms.iterator();for(BytesRef term=te.next();term!=null;term=te.next()){int freq=(int)te.totalTermFreq();d.startTerm(term,freq);PostingsEnum p=te.postings(null,PostingsEnum.ALL);p.nextDoc();for(int i=0;i<freq;i++)d.addPosition(p.nextPosition(),p.startOffset(),p.endOffset(),p.getPayload());d.finishTerm();}d.finishField();}d.finishDocument();
  }
  static void assign(MergeState s,String n,Object v)throws Exception{var f=MergeState.class.getField(n);f.setAccessible(true);f.set(s,v);}
  static MergeState state(Directory dir,TermVectorsReader r)throws Exception {
    var c=MergeState.class.getDeclaredConstructor(List.class,SegmentInfo.class,InfoStream.class,Executor.class,MergePolicy.OneMerge.class);c.setAccessible(true);
    MergeState s=c.newInstance(List.of(),info(dir,-1),InfoStream.NO_OUTPUT,(Executor)Runnable::run,null);
    int[][] maps={{2,-1},{0},{1,3}};MergeState.DocMap[] dm=new MergeState.DocMap[3];for(int i=0;i<3;i++){final int[] m=maps[i];dm[i]=new MergeState.DocMap(){@Override public int get(int d){return m[d];}};}
    assign(s,"termVectorsReaders",new TermVectorsReader[]{r,null,r});assign(s,"maxDocs",new int[]{2,1,2});assign(s,"docMaps",dm);s.mergeFieldInfos=new FieldInfos(new FieldInfo[]{field(7)});s.needsIndexSort=true;return s;
  }
  static class Bench implements AutoCloseable {
    final Directory source=new ByteBuffersDirectory();
    final ProbeFormat probe=new ProbeFormat();
    final TermVectorsFormat callbacks=wrap(probe),simple=wrap(new SimpleTextTermVectorsFormat());
    final TermVectorsReader reader;
    final MergeState merge;
    final SegmentInfo output;
    Bench()throws Exception {
      byte[] bytes=Files.readAllBytes(Path.of("fixtures/testframework/data/asserting_term_vectors.tvc"));try(var out=source.createOutput("_atv.tvc",IOContext.DEFAULT)){out.writeBytes(bytes,0,bytes.length);}
      output=info(source,2);reader=simple.vectorsReader(source,output,new FieldInfos(new FieldInfo[]{FIELD}),IOContext.DEFAULT);merge=state(source,reader);
    }
    long batch(String name)throws Exception {
      Digest digest=new Digest();
      if(name.equals("validation")) {
        TermVectorsWriter w=callbacks.vectorsWriter(source,output,IOContext.DEFAULT);
        for(int i=0;i<N;i++)document(w,POS);
        w.finish(N);w.close();digest.mix(probe.last.value);
      } else for(int i=0;i<N;i++)switch(name) {
        case "merge"->{TermVectorsWriter w=callbacks.vectorsWriter(source,output,IOContext.DEFAULT);int count=w.merge(merge);w.close();digest.mix(count);digest.mix(probe.last.value);}
        case "write"->{try(Directory dir=new ByteBuffersDirectory()){TermVectorsWriter w=simple.vectorsWriter(dir,output,IOContext.DEFAULT);document(w,REAL_POS);w.startDocument(0);w.finishDocument();w.finish(2);w.close();try(var input=dir.openInput("_atv.tvc",IOContext.DEFAULT)){byte[] b=new byte[Math.toIntExact(input.length())];input.readBytes(b,0,b.length);digest.bytes(b,0,b.length);}}}
        case "read"->{TermVectorsReader r=reader.clone();r.checkIntegrity();consume(r,0,digest);consume(r,1,digest);r.close();}
        default->throw new AssertionError(name);
      }
      return digest.value;
    }
    long[] run(String name,int ms)throws Exception {long expected=batch(name),start=System.nanoTime(),batches=0;do{long actual=batch(name);if(actual!=expected)throw new AssertionError("timed digest changed");sink=actual;batches++;}while(System.nanoTime()-start<ms*1_000_000L);return new long[]{batches,System.nanoTime()-start};}
    @Override public void close()throws Exception {reader.close();source.close();}
  }
  public static void main(String[] args)throws Exception {
    if(!AssertingTermVectorsMicro.class.desiredAssertionStatus())throw new AssertionError("run with -ea");
    int warmup=args.length>0?Integer.parseInt(args[0]):1000,measure=args.length>1?Integer.parseInt(args[1]):1500;
    try(Bench b=new Bench()) {
      TermVectorsWriter sample=b.callbacks.vectorsWriter(b.source,b.output,IOContext.DEFAULT);
      System.out.printf("#memory\tasserting_term_vectors_writer_wrapper\t%d%n",RamUsageEstimator.shallowSizeOf(sample));
      for(String name:new String[]{"validation","merge","write","read"}){System.out.printf("#check\tasserting_term_vectors_%s\t%d\t%d%n",name,b.batch(name),N);b.run(name,warmup);long[] r=b.run(name,measure);long count=r[0]*N;System.out.printf(java.util.Locale.ROOT,"asserting_term_vectors_%s\t%.3f\t%d%n",name,(double)r[1]/count,count);}
    }
  }
}
