package org.apache.lucene.tests.codecs.cranky;
import java.nio.file.*;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.concurrent.atomic.AtomicLong;
import org.apache.lucene.codecs.*;
import org.apache.lucene.codecs.lucene104.Lucene104Codec;
import org.apache.lucene.index.*;
import org.apache.lucene.store.*;
import org.apache.lucene.util.*;
/** Reuses fixture state construction; measures the actual Java cranky wrapper. */
public class CrankyDocValuesMicro extends GenCrankyDocValues {
 static volatile long sink;
 static long bytes(long digest,byte[] value){digest=digest*31+value.length;for(byte item:value)digest=digest*31+Byte.toUnsignedInt(item);return digest;}
 static long bytes(long digest,BytesRef value){digest=digest*31+value.length;for(int i=0;i<value.length;i++)digest=digest*31+Byte.toUnsignedInt(value.bytes[value.offset+i]);return digest;}
 static long consume(long digest,DocValuesProducer producer,FieldInfos infos)throws Exception{
  for(FieldInfo field:infos){
   digest=bytes(digest,field.name.getBytes(StandardCharsets.UTF_8));
   switch(field.getDocValuesType()){
    case NONE->{continue;}
    case NUMERIC->{var values=producer.getNumeric(field);for(int doc=values.nextDoc();doc!=org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;doc=values.nextDoc()){digest=digest*31+doc;digest=digest*31+values.longValue();}}
    case BINARY->{var values=producer.getBinary(field);for(int doc=values.nextDoc();doc!=org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;doc=values.nextDoc()){digest=digest*31+doc;BytesRef value=values.binaryValue();digest=bytes(digest,value);}}
    case SORTED->{var values=producer.getSorted(field);for(int ord=0;ord<values.getValueCount();ord++){BytesRef value=values.lookupOrd(ord);digest=bytes(digest,value);}for(int doc=values.nextDoc();doc!=org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;doc=values.nextDoc()){digest=digest*31+doc;digest=digest*31+values.ordValue();}}
    case SORTED_NUMERIC->{var values=producer.getSortedNumeric(field);for(int doc=values.nextDoc();doc!=org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;doc=values.nextDoc()){digest=digest*31+doc;int count=values.docValueCount();digest=digest*31+count;for(int i=0;i<count;i++)digest=digest*31+values.nextValue();}}
    case SORTED_SET->{var values=producer.getSortedSet(field);for(long ord=0;ord<values.getValueCount();ord++){BytesRef value=values.lookupOrd(ord);digest=bytes(digest,value);}for(int doc=values.nextDoc();doc!=org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;doc=values.nextDoc()){digest=digest*31+doc;int count=values.docValueCount();digest=digest*31+count;for(int i=0;i<count;i++)digest=digest*31+values.nextOrd();}}
   }
   if(field.docValuesSkipIndexType()!=DocValuesSkipIndexType.NONE){DocValuesSkipper skip=producer.getSkipper(field);digest=digest*31+skip.minValue();digest=digest*31+skip.maxValue();digest=digest*31+skip.docCount();digest=digest*31+skip.maxValueCount();for(int target=0;target<12;){skip.advance(target);if(skip.minDocID(0)==org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS)break;for(int level=0;level<skip.numLevels();level++){digest=digest*31+skip.minDocID(level);digest=digest*31+skip.maxDocID(level);digest=digest*31+skip.minValue(level);digest=digest*31+skip.maxValue(level);digest=digest*31+skip.docCount(level);}target=skip.maxDocID(0)+1;}}
  }return digest;
 }
 static long batch(int which,Directory input,FieldInfos infos,FieldInfos empty,DocValuesProducer source,MergeState[] merges)throws Exception{
  long digest=0;
  for(int iteration=0;iteration<16;iteration++){
   Random random=new Random(42);DocValuesFormat format=new CrankyDocValuesFormat(NATIVE,random);
   if(which==0){try(DocValuesProducer producer=format.fieldsProducer(read(input,infos,12))){producer.checkIntegrity();digest=consume(digest,producer,infos);}}
   else try(Directory output=new ByteBuffersDirectory()){
    try(DocValuesConsumer consumer=format.fieldsConsumer(write(output,which==4?empty:infos,which==1?12:which==4?0:20))){
     if(which==1){for(FieldInfo field:infos)add(consumer,field,source);}else if(which!=4)consumer.merge(merges[which-2]);
    }
    for(String extension:EXTENSIONS)try(IndexInput file=output.openInput("_cdv_s."+extension,IOContext.DEFAULT)){byte[] value=new byte[Math.toIntExact(file.length())];file.readBytes(value,0,value.length);digest=bytes(digest,value);}
   }
   digest=digest*31+Integer.toUnsignedLong(random.nextInt());
  }return digest;
 }
 static long[] run(int which,Directory input,FieldInfos infos,FieldInfos empty,DocValuesProducer source,MergeState[] merges,long millis,long expected)throws Exception{long start=System.nanoTime(),count=0;do{long actual=batch(which,input,infos,empty,source,merges);if(actual!=expected)throw new AssertionError("unstable full digest");sink=actual;count+=16;}while(System.nanoTime()-start<millis*1_000_000L);return new long[]{System.nanoTime()-start,count};}
 static void load(Directory directory,String prefix)throws Exception{for(String extension:EXTENSIONS)try(IndexOutput output=directory.createOutput("_cdv_s."+extension,IOContext.DEFAULT)){byte[] value=Files.readAllBytes(Path.of("fixtures/testframework/data/"+prefix+"."+extension));output.writeBytes(value,0,value.length);}}
 public static void main(String[] args)throws Exception{
  long warmup=Long.parseLong(args[0]),measure=Long.parseLong(args[1]);Random random=new Random(42);DocValuesFormat format=new CrankyDocValuesFormat(NATIVE,random);
  long owned=RamUsageEstimator.shallowSizeOf(format)+RamUsageEstimator.shallowSizeOf(NATIVE)+RamUsageEstimator.shallowSizeOf(random)+RamUsageEstimator.shallowSizeOfInstance(AtomicLong.class);System.out.printf("cranky_doc_values_format_owned_bytes\t%d%n",owned);
  try(Directory left=new ByteBuffersDirectory();Directory right=new ByteBuffersDirectory()){
   load(left,"cranky_doc_values_left");load(right,"cranky_doc_values_right");try(IndexOutput output=left.createOutput("_cdv.fnm",IOContext.DEFAULT)){byte[] value=Files.readAllBytes(Path.of("fixtures/testframework/data/cranky_doc_values.fnm"));output.writeBytes(value,0,value.length);}
   FieldInfos infos=new Lucene104Codec().fieldInfosFormat().read(left,segment(left,12),"",IOContext.DEFAULT),empty=new FieldInfos(new FieldInfo[0]);
   try(DocValuesProducer lp=NATIVE.fieldsProducer(read(left,infos,12));DocValuesProducer rp=NATIVE.fieldsProducer(read(right,infos,12))){
    MergeState[] merges=new MergeState[2];Bits live=new Bits(){public boolean get(int doc){return doc!=3&&doc!=10;}public int length(){return 12;}};
    for(int kind=0;kind<2;kind++){int[][] maps=new int[2][12];for(int side=0;side<2;side++)for(int doc=0;doc<12;doc++)maps[side][doc]=mapped(side,doc,kind==1);merges[kind]=merge(left,infos,new DocValuesProducer[]{lp,rp},new FieldInfos[]{infos,infos},new int[]{12,12},maps,new Bits[]{live,live},kind==1);}
    String[] cases={"cranky_doc_values_read","cranky_doc_values_write","cranky_doc_values_merge_sequential","cranky_doc_values_merge_sorted","cranky_doc_values_empty_close"};
    for(int which=0;which<cases.length;which++){long expected=batch(which,left,infos,empty,lp,merges);System.out.printf("#check\t%s\t%s\t16%n",cases[which],Long.toUnsignedString(expected));run(which,left,infos,empty,lp,merges,warmup,expected);long[] result=run(which,left,infos,empty,lp,merges,measure,expected);System.out.printf(Locale.ROOT,"%s\t%.3f\t%d%n",cases[which],(double)result[0]/result[1],result[1]);}
   }
  }
 }
}
