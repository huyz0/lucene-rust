import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.Executor;
import java.util.concurrent.atomic.AtomicLong;
import org.apache.lucene.codecs.*;
import org.apache.lucene.codecs.simpletext.SimpleTextTermVectorsFormat;
import org.apache.lucene.index.*;
import org.apache.lucene.store.*;
import org.apache.lucene.util.*;

/** Actual Cranky wrapper, superclass proximity/merge, and real SimpleText bytes. */
public class CrankyTermVectorsMicro {
  static volatile long sink;
  static final int N=256;
  static final byte[] POS={3,2,0,10,4,1,0}, OFF={0,2,1,3,0,0};
  static final byte[] ID="0123456789abcdef".getBytes(StandardCharsets.US_ASCII);
  static final FieldInfo FIELD=field(0);
  static FieldInfo field(int number){return field("f",number);}
  static FieldInfo field(String name,int number){return new FieldInfo(name,number,true,false,false,IndexOptions.DOCS_AND_FREQS_AND_POSITIONS_AND_OFFSETS,DocValuesType.NONE,DocValuesSkipIndexType.NONE,-1,Map.of(),0,0,0,0,VectorEncoding.FLOAT32,VectorSimilarityFunction.EUCLIDEAN,false,false);}
  static SegmentInfo info(Directory dir,int docs){return new SegmentInfo(dir,Version.LATEST,Version.LATEST,"_ctv",docs,false,false,null,Map.of(),ID,Map.of(),null);}
  static TermVectorsFormat format(TermVectorsFormat delegate,Random random)throws Exception{var c=Class.forName("org.apache.lucene.tests.codecs.cranky.CrankyTermVectorsFormat").getDeclaredConstructor(TermVectorsFormat.class,Random.class);c.setAccessible(true);return (TermVectorsFormat)c.newInstance(delegate,random);}
  static TermVectorsWriter wrap(TermVectorsWriter delegate,Random random)throws Exception{var c=Class.forName("org.apache.lucene.tests.codecs.cranky.CrankyTermVectorsFormat$CrankyTermVectorsWriter").getDeclaredConstructor(TermVectorsWriter.class,Random.class);c.setAccessible(true);return (TermVectorsWriter)c.newInstance(delegate,random);}
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
    @Override public long ramBytesUsed(){return 0;}
  }
  static void assign(MergeState s,String name,Object value)throws Exception{var f=MergeState.class.getField(name);f.setAccessible(true);f.set(s,value);}
  static MergeState state(Directory dir,TermVectorsReader reader)throws Exception{
    var c=MergeState.class.getDeclaredConstructor(List.class,SegmentInfo.class,InfoStream.class,Executor.class,MergePolicy.OneMerge.class);c.setAccessible(true);
    MergeState s=c.newInstance(List.of(),info(dir,-1),InfoStream.NO_OUTPUT,(Executor)Runnable::run,null);
    int[][] maps={{2,-1},{0},{1,3}};MergeState.DocMap[] docMaps=new MergeState.DocMap[3];
    for(int i=0;i<3;i++){final int[] map=maps[i];docMaps[i]=new MergeState.DocMap(){@Override public int get(int doc){return map[doc];}};}
    assign(s,"termVectorsReaders",new TermVectorsReader[]{reader,null,reader});assign(s,"maxDocs",new int[]{2,1,2});assign(s,"docMaps",docMaps);s.mergeFieldInfos=new FieldInfos(new FieldInfo[]{field(7)});s.needsIndexSort=true;return s;
  }
  static void consume(TermVectorsReader reader,int doc,Digest d)throws IOException{
    Fields fields=reader.get(doc);d.startDocument(fields==null?0:fields.size());
    if(fields!=null)for(String name:fields){Terms terms=fields.terms(name);d.startField(field(name,7),(int)terms.size(),terms.hasPositions(),terms.hasOffsets(),terms.hasPayloads());TermsEnum te=terms.iterator();for(BytesRef term=te.next();term!=null;term=te.next()){int freq=(int)te.totalTermFreq();d.startTerm(term,freq);if(terms.hasPositions()||terms.hasOffsets()){PostingsEnum p=te.postings(null,PostingsEnum.ALL);p.nextDoc();for(int i=0;i<freq;i++)d.addPosition(p.nextPosition(),p.startOffset(),p.endOffset(),p.getPayload());}d.finishTerm();}d.finishField();}d.finishDocument();
  }
  static class Bench implements AutoCloseable {
    final Random random=new Random(42);
    final Digest counter=new Digest();
    final TermVectorsWriter guard=wrap(counter,random);
    final TermVectorsFormat format=format(new SimpleTextTermVectorsFormat(),random);
    final Directory source=new ByteBuffersDirectory();
    final TermVectorsReader reader;
    final MergeState merge;
    final SegmentInfo writeInfo;
    Bench()throws Exception{
      byte[] bytes=Files.readAllBytes(Path.of("fixtures/testframework/data/cranky_term_vectors.tvc"));try(var output=source.createOutput("_ctv.tvc",IOContext.DEFAULT)){output.writeBytes(bytes,0,bytes.length);}
      reader=new SimpleTextTermVectorsFormat().vectorsReader(source,info(source,2),new FieldInfos(new FieldInfo[]{FIELD}),IOContext.DEFAULT);merge=state(source,reader);writeInfo=info(source,2);
    }
    long batch(String name)throws Exception{
      random.setSeed(42);counter.value=0;Digest digest=new Digest();
      for(int i=0;i<N;i++){
        switch(name){
          case "prox"->{try{guard.addProx(3,new ByteArrayDataInput(POS),new ByteArrayDataInput(OFF));}catch(IOException e){if(!e.getMessage().startsWith("Fake IOException"))throw e;counter.mix(99);}}
          case "merge"->{try{counter.mix(guard.merge(merge));}catch(IOException e){if(!e.getMessage().startsWith("Fake IOException"))throw e;counter.mix(99);}}
          case "write"->{random.setSeed(42);try(Directory dir=new ByteBuffersDirectory()){TermVectorsWriter w=format.vectorsWriter(dir,writeInfo,IOContext.DEFAULT);w.startDocument(1);w.startField(FIELD,1,true,true,true);w.startTerm(new BytesRef(new byte[]{0,10,92,(byte)255}),3);w.addProx(3,new ByteArrayDataInput(POS),new ByteArrayDataInput(OFF));w.finishTerm();w.finishField();w.finishDocument();w.startDocument(0);w.finishDocument();w.finish(2);w.close();try(var input=dir.openInput("_ctv.tvc",IOContext.DEFAULT)){int length=Math.toIntExact(input.length());byte[] bytes=new byte[length];input.readBytes(bytes,0,length);digest.bytes(bytes,0,length);}}}
          case "read"->{consume(reader,0,digest);consume(reader,1,digest);}
          default->throw new AssertionError(name);
        }
      }
      if(name.equals("prox")||name.equals("merge"))digest.value=counter.value;
      digest.mix(random.nextInt());digest.mix(random.nextLong());return digest.value;
    }
    long[] run(String name,int ms)throws Exception{long expected=batch(name),start=System.nanoTime(),batches=0;do{long actual=batch(name);if(actual!=expected)throw new AssertionError("unstable digest");sink=actual;batches++;}while(System.nanoTime()-start<ms*1_000_000L);return new long[]{batches,System.nanoTime()-start};}
    @Override public void close()throws Exception{reader.close();source.close();}
  }
  public static void main(String[] args)throws Exception{
    try(Bench b=new Bench()){
      // Format + writer wrappers, shared Random and its AtomicLong once;
      // delegates, decoded documents, inputs and I/O transients excluded.
      long bytes=RamUsageEstimator.shallowSizeOf(b.format)+RamUsageEstimator.shallowSizeOf(b.guard)+RamUsageEstimator.shallowSizeOf(b.random)+RamUsageEstimator.shallowSizeOfInstance(AtomicLong.class);
      System.out.printf("#memory\tcranky_term_vectors_wrappers_rng\t%d%n",bytes);
      for(String name:new String[]{"prox","merge","write","read"}){System.out.printf("#check\tcranky_term_vectors_%s\t%d\t%d%n",name,b.batch(name),N);b.run(name,1000);long[] result=b.run(name,1500);long operations=result[0]*N;System.out.printf(java.util.Locale.ROOT,"cranky_term_vectors_%s\t%.3f\t%d%n",name,(double)result[1]/operations,operations);}
    }
  }
}
