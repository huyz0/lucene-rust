package org.apache.lucene.tests.codecs.cranky;
import java.nio.file.*;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.concurrent.atomic.AtomicLong;
import org.apache.lucene.codecs.*;
import org.apache.lucene.codecs.lucene104.*;
import org.apache.lucene.index.*;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.store.*;
import org.apache.lucene.util.*;
/** Actual cranky wrapper and native Lucene104 codec; consumes full values and files. */
public class CrankyPostingsMicro {
 static volatile long sink;
 static final byte[] ID={0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
 static final String SUFFIX="s";
 static final String[] EXTENSIONS={"doc","psm","pos","pay","tim","tip","tmd"};
 static final PostingsFormat NATIVE=new Lucene104PostingsFormat();
  static SegmentInfo segment(Directory directory,int maxDoc) {
    return new SegmentInfo(directory,Version.fromBits(10,5,0),null,"_cp",maxDoc,false,false,
        new Lucene104Codec(),Map.of(),ID.clone(),Map.of(),null);
  }
  static SegmentWriteState writeState(Directory directory,FieldInfos fields,int maxDoc) {
    return new SegmentWriteState(InfoStream.NO_OUTPUT,directory,segment(directory,maxDoc),fields,null,IOContext.DEFAULT,SUFFIX);
  }
  static SegmentReadState readState(Directory directory,FieldInfos fields,int maxDoc) {
    return new SegmentReadState(directory,segment(directory,maxDoc),fields,IOContext.DEFAULT,SUFFIX);
  }
  static MergeState mergeState(Directory directory,FieldInfos fields,FieldsProducer[] readers,
      int[] maxDocs,int[][] maps,boolean sorted) throws Exception {
    var constructor=MergeState.class.getDeclaredConstructor(List.class,SegmentInfo.class,
        InfoStream.class,java.util.concurrent.Executor.class,MergePolicy.OneMerge.class);
    constructor.setAccessible(true);
    MergeState state=constructor.newInstance(List.of(),segment(directory,-1),InfoStream.NO_OUTPUT,null,null);
    MergeState.DocMap[] docMaps=new MergeState.DocMap[maps.length];
    for(int i=0;i<maps.length;i++){final int[] map=maps[i];docMaps[i]=new MergeState.DocMap(){public int get(int doc){return map[doc];}};}
    Map<String,Object> members=new HashMap<>();members.put("fieldsProducers",readers);
    members.put("maxDocs",maxDocs);members.put("docMaps",docMaps);
    FieldInfos[] sourceFields=new FieldInfos[readers.length];Arrays.fill(sourceFields,fields);
    members.put("fieldInfos",sourceFields);members.put("needsIndexSort",sorted);
    for(var member:members.entrySet()){var field=MergeState.class.getField(member.getKey());field.setAccessible(true);field.set(state,member.getValue());}
    state.mergeFieldInfos=fields;return state;
  }
  static NormsProducer norms(Map<String,long[]> columns) {
    return new NormsProducer(){
      @Override public NumericDocValues getNorms(FieldInfo field){long[] values=columns.get(field.name);return new NumericDocValues(){int doc=-1;
        @Override public int docID(){return doc;}
        @Override public long cost(){return values.length;}
        @Override public boolean advanceExact(int target){doc=target;return values[target]!=-1;}
        @Override public long longValue(){return values[doc];}
        @Override public int nextDoc(){return advance(doc+1);}
        @Override public int advance(int target){for(doc=target;doc<values.length;doc++)if(values[doc]!=-1)return doc;return doc=DocIdSetIterator.NO_MORE_DOCS;}
      };}
      @Override public void checkIntegrity(){}
      @Override public void close(){}
    };
  }

 static long bytes(long digest,byte[] value,int offset,int length){digest=digest*31+length;for(int i=0;i<length;i++)digest=digest*31+Byte.toUnsignedInt(value[offset+i]);return digest;}
 static long read(long digest,FieldsProducer producer,FieldInfos infos) throws Exception {
  producer.checkIntegrity();
  for(String name:producer){byte[] nameBytes=name.getBytes(StandardCharsets.UTF_8);digest=bytes(digest,nameBytes,0,nameBytes.length);FieldInfo info=infos.fieldInfo(name);
   TermsEnum terms=producer.terms(name).iterator();for(BytesRef term=terms.next();term!=null;term=terms.next()){
    digest=bytes(digest,term.bytes,term.offset,term.length);PostingsEnum postings=terms.postings(null,PostingsEnum.ALL);
    for(int doc=postings.nextDoc();doc!=DocIdSetIterator.NO_MORE_DOCS;doc=postings.nextDoc()){
     digest=digest*31+doc;int freq=postings.freq();digest=digest*31+freq;
     if(info.getIndexOptions().compareTo(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS)>=0)for(int i=0;i<freq;i++){
      digest=digest*31+postings.nextPosition();digest=digest*31+postings.startOffset();digest=digest*31+postings.endOffset();BytesRef payload=postings.getPayload();digest=payload==null?digest*31:bytes(digest,payload.bytes,payload.offset,payload.length);
     }
    }
   }
  }return digest;
 }
 static long batch(Directory saved,FieldInfos infos,FieldsProducer source,NormsProducer[] norms,MergeState[] merges,int which) throws Exception {
  long digest=0;
  for(int i=0;i<16;i++){
   Random random=new Random(42);PostingsFormat format=new CrankyPostingsFormat(NATIVE,random);
   if(which==0){try(FieldsProducer reader=format.fieldsProducer(readState(saved,infos,9))){digest=read(digest,reader,infos);}}
   else try(Directory directory=new ByteBuffersDirectory()){
    try(FieldsConsumer consumer=format.fieldsConsumer(writeState(directory,infos,which==1?9:16))){
     if(which==1)consumer.write(source,norms[0]);else{
      consumer.merge(merges[which-2],norms[which-1]);
     }
    }
    for(String extension:EXTENSIONS)try(IndexInput input=directory.openInput("_cp_s."+extension,IOContext.DEFAULT)){long length=input.length();digest=digest*31+length;for(long p=0;p<length;p++)digest=digest*31+Byte.toUnsignedInt(input.readByte());}
   }
   digest=digest*31+Integer.toUnsignedLong(random.nextInt());
  }return digest;
 }
 static long[] run(Directory saved,FieldInfos infos,FieldsProducer source,NormsProducer[] norms,MergeState[] merges,int which,long millis,long expected) throws Exception {
  long start=System.nanoTime(),count=0;do{long actual=batch(saved,infos,source,norms,merges,which);if(actual!=expected)throw new AssertionError("unstable digest");sink=actual;count+=16;}while(System.nanoTime()-start<millis*1_000_000L);return new long[]{System.nanoTime()-start,count};
 }
 public static void main(String[] args) throws Exception {
  long warmup=Long.parseLong(args[0]),measure=Long.parseLong(args[1]);
  Random random=new Random(42);PostingsFormat format=new CrankyPostingsFormat(NATIVE,random);
  long owned=RamUsageEstimator.shallowSizeOf(format)+RamUsageEstimator.shallowSizeOf(NATIVE)+RamUsageEstimator.shallowSizeOf(random)+RamUsageEstimator.shallowSizeOfInstance(AtomicLong.class);
  System.out.printf("cranky_postings_format_owned_bytes\t%d%n",owned);
  try(Directory saved=new ByteBuffersDirectory()){
   for(String extension:EXTENSIONS)try(IndexOutput out=saved.createOutput("_cp_s."+extension,IOContext.DEFAULT)){byte[] data=Files.readAllBytes(Path.of("fixtures/testframework/data/cranky_postings."+extension));out.writeBytes(data,0,data.length);}
   try(IndexOutput out=saved.createOutput("_cp.fnm",IOContext.DEFAULT)){byte[] data=Files.readAllBytes(Path.of("fixtures/testframework/data/cranky_postings.fnm"));out.writeBytes(data,0,data.length);}
   FieldInfos infos=new Lucene104Codec().fieldInfosFormat().read(saved,segment(saved,9),"",IOContext.DEFAULT);
   NormsProducer[] norms=new NormsProducer[3];for(int c=0;c<3;c++){long[] values=new long[c==0?9:16];Arrays.fill(values,-1);for(int side=0;side<(c==0?1:2);side++)for(int doc=0;doc<9;doc++){if(c>0&&doc==1)continue;int at=doc==0?0:doc-1;int dest=c==0?doc:c==1?8*side+at:2*at+side;values[dest]=doc==1?4:3;}norms[c]=norms(Map.of("rich",values));}
   try(FieldsProducer source=NATIVE.fieldsProducer(readState(saved,infos,9))){
    MergeState[] merges=new MergeState[2];for(int kind=0;kind<2;kind++){int[][] maps=new int[2][9];for(int side=0;side<2;side++)for(int doc=0;doc<9;doc++){int at=doc==0?0:doc-1;maps[side][doc]=doc==1?-1:kind==1?2*at+side:8*side+at;}merges[kind]=mergeState(saved,infos,new FieldsProducer[]{source,source},new int[]{9,9},maps,kind==1);}
    String[] cases={"cranky_postings_read","cranky_postings_write","cranky_postings_merge_sequential","cranky_postings_merge_sorted"};
    for(int which=0;which<cases.length;which++){long expected=batch(saved,infos,source,norms,merges,which);System.out.printf("#check\t%s\t%s\t16%n",cases[which],Long.toUnsignedString(expected));run(saved,infos,source,norms,merges,which,warmup,expected);long[] result=run(saved,infos,source,norms,merges,which,measure,expected);System.out.printf(Locale.ROOT,"%s\t%.3f\t%d%n",cases[which],(double)result[0]/result[1],result[1]);}
   }
  }
 }
}
