import java.io.IOException;
import java.lang.reflect.Constructor;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import org.apache.lucene.codecs.*;
import org.apache.lucene.codecs.lucene90.Lucene90StoredFieldsFormat;
import org.apache.lucene.index.*;
import org.apache.lucene.store.*;
import org.apache.lucene.tests.codecs.asserting.AssertingStoredFieldsFormat;
import org.apache.lucene.util.*;

/** Actual AssertingStoredFieldsFormat, complete digests and ordinary native codec work. */
public class AssertingStoredFieldsMicro {
  static final int BATCH=32;
  static final long OFFSET=0xcbf29ce484222325L;
  static final byte[] ID={0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
  static final FieldInfo FIELD=new FieldInfo("f",7,false,false,false,IndexOptions.NONE,DocValuesType.NONE,DocValuesSkipIndexType.NONE,-1,Map.of(),0,0,0,0,VectorEncoding.FLOAT32,VectorSimilarityFunction.EUCLIDEAN,false,false);
  static final FieldInfos FIELDS=new FieldInfos(new FieldInfo[]{FIELD});
  static final BytesRef BINARY=new BytesRef(new byte[]{0,1,(byte)255});
  static final byte[] INPUT={9,8};
  static volatile long sink;
  static long hash(long digest,long value){return (digest^value)*0x100000001b3L;}
  static long bytes(long digest,byte[] bytes){digest=hash(digest,bytes.length);for(byte value:bytes)digest=hash(digest,value&255);return digest;}
  static SegmentInfo info(Directory dir,int docs){return new SegmentInfo(dir,Version.LATEST,null,"_asf",docs,false,false,null,Map.of(),ID,Map.of(),null);}
  static AssertingStoredFieldsFormat format(StoredFieldsFormat delegate)throws Exception{var format=new AssertingStoredFieldsFormat();var f=AssertingStoredFieldsFormat.class.getDeclaredField("in");f.setAccessible(true);f.set(format,delegate);return format;}
  static class Visitor extends StoredFieldVisitor {
    long value=OFFSET;
    void mix(long v){value=hash(value,v);}
    void field(FieldInfo f,long kind){mix(f.number);mix(kind);}
    @Override public Status needsField(FieldInfo f){return Status.YES;}
    @Override public void intField(FieldInfo f,int v){field(f,1);mix(v);}
    @Override public void longField(FieldInfo f,long v){field(f,2);mix(v);}
    @Override public void floatField(FieldInfo f,float v){field(f,3);mix(Integer.toUnsignedLong(Float.floatToRawIntBits(v)));}
    @Override public void doubleField(FieldInfo f,double v){field(f,4);mix(Double.doubleToRawLongBits(v));}
    @Override public void binaryField(FieldInfo f,byte[] v){field(f,5);value=bytes(value,v);}
    @Override public void stringField(FieldInfo f,String v){field(f,6);value=bytes(value,v.getBytes(StandardCharsets.UTF_8));}
  }
  static class Counter extends StoredFieldsWriter {
    final Visitor visitor=new Visitor();
    @Override public void startDocument(){visitor.mix(10);}
    @Override public void finishDocument(){visitor.mix(11);}
    @Override public void writeField(FieldInfo f,int v){visitor.intField(f,v);}
    @Override public void writeField(FieldInfo f,long v){visitor.longField(f,v);}
    @Override public void writeField(FieldInfo f,float v){visitor.floatField(f,v);}
    @Override public void writeField(FieldInfo f,double v){visitor.doubleField(f,v);}
    @Override public void writeField(FieldInfo f,BytesRef v){visitor.field(f,5);visitor.mix(v.length);for(int i=0;i<v.length;i++)visitor.mix(v.bytes[v.offset+i]&255);}
    @Override public void writeField(FieldInfo f,String v){visitor.stringField(f,v);}
    @Override public void writeField(FieldInfo f,StoredFieldDataInput input)throws IOException{if(input.length()!=2)throw new AssertionError();byte[] b=new byte[2];input.getDataInput().readBytes(b,0,2);visitor.binaryField(f,b);}
    @Override public void finish(int docs){visitor.mix(12);visitor.mix(docs);}
    @Override public void close(){visitor.mix(13);}
    @Override public long ramBytesUsed(){return 0;}
  }
  static class CountingFormat extends StoredFieldsFormat {
    Counter last;
    @Override public StoredFieldsWriter fieldsWriter(Directory dir,SegmentInfo si,IOContext context){last=new Counter();return last;}
    @Override public StoredFieldsReader fieldsReader(Directory dir,SegmentInfo si,FieldInfos fields,IOContext context){throw new AssertionError("validation format reader");}
  }
  static void writeDocuments(StoredFieldsWriter writer)throws IOException{
    writer.startDocument();writer.writeField(FIELD,-17);writer.writeField(FIELD,Long.MIN_VALUE);writer.writeField(FIELD,-0.0f);writer.writeField(FIELD,Double.longBitsToDouble(0x7ff8000000000042L));writer.writeField(FIELD,BINARY);
    var input=new ByteArrayDataInput(INPUT);writer.writeField(FIELD,new StoredFieldDataInput(input,2));if(input.getPosition()!=2)throw new AssertionError("input consumption");writer.writeField(FIELD,"hello λ");writer.finishDocument();writer.startDocument();writer.finishDocument();writer.finish(2);writer.close();
  }
  static long consume(long digest,Directory directory)throws IOException{
    for(String ext:new String[]{"fdt","fdx","fdm"}){String name="_asf."+ext;try(var in=directory.openInput(name,IOContext.DEFAULT)){byte[] b=new byte[(int)in.length()];in.readBytes(b,0,b.length);digest=bytes(digest,b);}directory.deleteFile(name);}
    if(directory.listAll().length!=0)throw new AssertionError("leftover files");return digest;
  }
  @FunctionalInterface interface Batch{long run()throws Exception;}
  static void measure(String name,long warmup,long duration,Batch batch)throws Exception{
    long expected=batch.run();int phase=0;
    for(long millis:new long[]{warmup,duration}){long start=System.nanoTime(),operations=0;do{long result=batch.run();if(result!=expected)throw new AssertionError("timed batch digest changed");sink=result;operations+=BATCH;}while(System.nanoTime()-start<millis*1000000L);if(phase++==1)System.out.printf(Locale.ROOT,"%s\t%.3f\t%d%n",name,(double)(System.nanoTime()-start)/operations,operations);}
    System.out.printf("#check\t%s\t%s\t%d%n",name,Long.toUnsignedString(expected),BATCH);
  }
  static MergeState mergeState(Directory dir,StoredFieldsReader reader,boolean sorted)throws Exception{
    Constructor<MergeState> c=MergeState.class.getDeclaredConstructor(List.class,SegmentInfo.class,InfoStream.class,java.util.concurrent.Executor.class,MergePolicy.OneMerge.class);c.setAccessible(true);var state=c.newInstance(List.of(),info(dir,-1),InfoStream.NO_OUTPUT,null,null);
    int[][] maps=sorted?new int[][]{{1,-1},{0,2}}:new int[][]{{0,-1},{1,2}};MergeState.DocMap[] docMaps=new MergeState.DocMap[2];for(int i=0;i<2;i++){int[] map=maps[i];docMaps[i]=doc->map[doc];}
    for(var entry:Map.of("storedFieldsReaders",new StoredFieldsReader[]{reader,reader},"fieldInfos",new FieldInfos[]{FIELDS,FIELDS},"maxDocs",new int[]{2,2},"docMaps",docMaps).entrySet()){var f=MergeState.class.getField(entry.getKey());f.setAccessible(true);f.set(state,entry.getValue());}
    state.mergeFieldInfos=FIELDS;state.needsIndexSort=sorted;return state;
  }
  public static void main(String[] args)throws Exception{
    if(!AssertingStoredFieldsMicro.class.desiredAssertionStatus())throw new AssertionError("run with -ea");long warmup=Long.parseLong(args[0]),duration=Long.parseLong(args[1]);
    CountingFormat counter=new CountingFormat();var validation=format(counter);
    try(var validationDirectory=new ByteBuffersDirectory();var speed=new ByteBuffersDirectory();var compression=new ByteBuffersDirectory()){
      var validationInfo=info(validationDirectory,2);
      var sample=validation.fieldsWriter(validationDirectory,validationInfo,IOContext.DEFAULT);
      System.out.printf("asserting_stored_fields_writer_overhead_bytes\t%d%n",RamUsageEstimator.shallowSizeOf(sample));
      System.out.printf("asserting_stored_fields_format_owned_bytes\t%d%n",RamUsageEstimator.shallowSizeOf(validation)+RamUsageEstimator.shallowSizeOfInstance(Lucene90StoredFieldsFormat.class));
      measure("asserting_stored_fields_validation",warmup,duration,()->{long digest=OFFSET;for(int i=0;i<BATCH;i++){var writer=validation.fieldsWriter(validationDirectory,validationInfo,IOContext.DEFAULT);writeDocuments(writer);digest=hash(digest,counter.last.visitor.value);}return digest;});
      Directory[] directories={speed,compression};StoredFieldsFormat[] formats={format(new Lucene90StoredFieldsFormat(Lucene90StoredFieldsFormat.Mode.BEST_SPEED)),format(new Lucene90StoredFieldsFormat(Lucene90StoredFieldsFormat.Mode.BEST_COMPRESSION))};SegmentInfo[] infos={info(speed,2),info(compression,2)};
      for(int mode=0;mode<2;mode++){
        String label=mode==0?"speed":"compression";Directory saved=directories[mode];var format=formats[mode];var segment=infos[mode];segment.putAttribute(Lucene90StoredFieldsFormat.MODE_KEY,mode==0?"BEST_SPEED":"BEST_COMPRESSION");
        for(String ext:new String[]{"fdt","fdx","fdm"}){byte[] b=Files.readAllBytes(Path.of("fixtures/testframework/data/asserting_stored_fields_"+label+"."+ext));try(var out=saved.createOutput("_asf."+ext,IOContext.DEFAULT)){out.writeBytes(b,b.length);}}
        measure("asserting_stored_fields_read_"+label,warmup,duration,()->{var visitor=new Visitor();for(int i=0;i<BATCH;i++){try(var reader=format.fieldsReader(saved,segment,FIELDS,IOContext.DEFAULT)){reader.checkIntegrity();visitor.mix(2);for(int doc=0;doc<2;doc++){visitor.mix(doc);reader.document(doc,visitor);}}}return visitor.value;});
        measure("asserting_stored_fields_write_"+label,warmup,duration,()->{long digest=OFFSET;for(int i=0;i<BATCH;i++){try(var directory=new ByteBuffersDirectory()){var writer=format.fieldsWriter(directory,segment,IOContext.DEFAULT);writeDocuments(writer);digest=consume(digest,directory);}}return digest;});
      }
      try(var reader=formats[0].fieldsReader(speed,infos[0],FIELDS,IOContext.DEFAULT)){
        for(boolean sorted:new boolean[]{false,true}){var merge=mergeState(speed,reader,sorted);
          measure(sorted?"asserting_stored_fields_merge_sorted":"asserting_stored_fields_merge_sequential",warmup,duration,()->{long digest=OFFSET;for(int i=0;i<BATCH;i++){try(var directory=new ByteBuffersDirectory()){var writer=formats[0].fieldsWriter(directory,infos[0],IOContext.DEFAULT);digest=hash(digest,writer.merge(merge));writer.close();digest=consume(digest,directory);}}return digest;});
        }
      }
    }
  }
}
