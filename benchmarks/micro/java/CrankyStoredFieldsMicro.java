import java.io.IOException;
import java.lang.reflect.Constructor;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.Executor;
import java.util.concurrent.atomic.AtomicLong;
import org.apache.lucene.codecs.*;
import org.apache.lucene.codecs.lucene90.Lucene90StoredFieldsFormat;
import org.apache.lucene.index.*;
import org.apache.lucene.store.*;
import org.apache.lucene.util.*;

/** Matched full stored-fields reads, output bytes, base visitor merges and faults. */
public class CrankyStoredFieldsMicro {
  static final byte[] ID={0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
  static final byte[] BINARY={0,1,(byte)255}, INPUT={9,8};
  static final long BATCH=32, OFFSET=0xcbf29ce484222325L;
  static final String[] NAMES={"int","long","float","double","binary","input","string"};
  static final FieldInfo[] FIELDS=new FieldInfo[7];
  static final FieldInfos FIELD_INFOS;
  static final String[] STRINGS={"hello λ 0","hello λ 1","hello λ 2"};
  static final Constructor<?> FORMAT, WRITER, MERGE;
  static volatile long sink;
  static {
    try {
      for(int i=0;i<7;i++)FIELDS[i]=field(NAMES[i],i);
      FIELD_INFOS=new FieldInfos(FIELDS);
      FORMAT=Class.forName("org.apache.lucene.tests.codecs.cranky.CrankyStoredFieldsFormat").getDeclaredConstructor(StoredFieldsFormat.class,Random.class);
      WRITER=Class.forName("org.apache.lucene.tests.codecs.cranky.CrankyStoredFieldsFormat$CrankyStoredFieldsWriter").getDeclaredConstructor(StoredFieldsWriter.class,Random.class);
      MERGE=MergeState.class.getDeclaredConstructor(List.class,SegmentInfo.class,InfoStream.class,Executor.class,MergePolicy.OneMerge.class);
      FORMAT.setAccessible(true);WRITER.setAccessible(true);MERGE.setAccessible(true);
    } catch(Exception e) {throw new ExceptionInInitializerError(e);}
  }
  static FieldInfo field(String name,int number) {
    return new FieldInfo(name,number,false,false,false,IndexOptions.NONE,DocValuesType.NONE,DocValuesSkipIndexType.NONE,-1,Map.of(),0,0,0,0,VectorEncoding.FLOAT32,VectorSimilarityFunction.EUCLIDEAN,false,false);
  }
  static SegmentInfo info(Directory directory,int docs) {
    return new SegmentInfo(directory,Version.fromBits(10,5,0),null,"_real",docs,false,false,null,Map.of(),ID.clone(),Map.of(),null);
  }
  static long hash(long digest,long value) {return (digest^value)*0x100000001b3L;}
  static long bytes(long digest,byte[] value) {digest=hash(digest,value.length);for(byte b:value)digest=hash(digest,b&255);return digest;}
  static class Visitor extends StoredFieldVisitor {
    long digest;
    Visitor(long digest) {this.digest=digest;}
    void field(FieldInfo info,long kind) {digest=hash(hash(digest,info.number),kind);}
    @Override public Status needsField(FieldInfo f) {return Status.YES;}
    @Override public void intField(FieldInfo f,int v) {field(f,1);digest=hash(digest,v);}
    @Override public void longField(FieldInfo f,long v) {field(f,2);digest=hash(digest,v);}
    @Override public void floatField(FieldInfo f,float v) {field(f,3);digest=hash(digest,Integer.toUnsignedLong(Float.floatToRawIntBits(v)));}
    @Override public void doubleField(FieldInfo f,double v) {field(f,4);digest=hash(digest,Double.doubleToRawLongBits(v));}
    @Override public void binaryField(FieldInfo f,byte[] v) {field(f,5);digest=bytes(digest,v);}
    @Override public void stringField(FieldInfo f,String v) {field(f,6);digest=bytes(digest,v.getBytes(StandardCharsets.UTF_8));}
  }
  static void put(Directory directory,String name,byte[] bytes) throws IOException {
    try(var output=directory.createOutput(name,IOContext.DEFAULT)) {output.writeBytes(bytes,bytes.length);}
  }
  static long consume(long digest,Directory directory) throws IOException {
    for(String name:new String[]{"_real.fdt","_real.fdx","_real.fdm"}) {
      try(var input=directory.openInput(name,IOContext.DEFAULT)) {
        digest=hash(digest,input.length());
        while(input.getFilePointer()<input.length())digest=hash(digest,input.readByte()&255);
      }
      directory.deleteFile(name);
    }
    if(directory.listAll().length!=0)throw new AssertionError("leftover outputs");
    return digest;
  }
  interface Batch {long run() throws Exception;}
  static void measure(String name,long warmup,long duration,Batch batch) throws Exception {
    long expected=batch.run();
    for(int phase=0;phase<2;phase++) {
      long budget=phase==0?warmup:duration,start=System.nanoTime(),operations=0;
      do {long actual=batch.run();if(actual!=expected)throw new AssertionError("unstable digest");sink=actual;operations+=BATCH;}
      while(System.nanoTime()-start<budget*1000000);
      if(phase==1)System.out.printf(Locale.ROOT,"%s\t%.3f\t%d%n",name,(double)(System.nanoTime()-start)/operations,operations);
    }
    System.out.printf("#check\t%s\t%s\t%d%n",name,Long.toUnsignedString(expected),BATCH);
  }
  static MergeState mergeState(Directory directory,StoredFieldsReader reader,boolean sorted) throws Exception {
    MergeState state=(MergeState)MERGE.newInstance(List.of(),info(directory,-1),InfoStream.NO_OUTPUT,null,null);
    StoredFieldsReader[] readers={reader,reader};
    FieldInfos[] fields={FIELD_INFOS,FIELD_INFOS};
    int[][] values=sorted?new int[][]{{0,-1,2},{1,3,-1}}:new int[][]{{0,-1,1},{2,3,-1}};
    MergeState.DocMap[] maps={doc->values[0][doc],doc->values[1][doc]};
    for(var entry:Map.of("storedFieldsReaders",readers,"fieldInfos",fields,"maxDocs",new int[]{3,3},"docMaps",maps).entrySet()) {
      var f=MergeState.class.getDeclaredField(entry.getKey());f.setAccessible(true);f.set(state,entry.getValue());
    }
    FieldInfo[] target=new FieldInfo[7];for(int i=0;i<7;i++)target[i]=field(NAMES[i],6-i);
    state.mergeFieldInfos=new FieldInfos(target);state.needsIndexSort=sorted;return state;
  }
  static class CountWriter extends StoredFieldsWriter {
    long closed;
    @Override public void startDocument() {throw new AssertionError("start guard bypassed");}
    @Override public void finishDocument() {throw new AssertionError("finish-document guard bypassed");}
    @Override public void finish(int docs) {throw new AssertionError("finish guard bypassed");}
    @Override public void close() {closed++;}
    @Override public void writeField(FieldInfo f,int v) {throw new AssertionError("field guard bypassed");}
    @Override public void writeField(FieldInfo f,long v) {throw new AssertionError("field guard bypassed");}
    @Override public void writeField(FieldInfo f,float v) {throw new AssertionError("field guard bypassed");}
    @Override public void writeField(FieldInfo f,double v) {throw new AssertionError("field guard bypassed");}
    @Override public void writeField(FieldInfo f,BytesRef v) {throw new AssertionError("field guard bypassed");}
    @Override public void writeField(FieldInfo f,StoredFieldDataInput v) {throw new AssertionError("input guard bypassed");}
    @Override public void writeField(FieldInfo f,String v) {throw new AssertionError("field guard bypassed");}
    @Override public int merge(MergeState state) {throw new AssertionError("delegate merge called");}
    @Override public long ramBytesUsed() {return 0;}
  }
  static long first(int bound) {long seed=0;while(new Random(seed).nextInt(bound)!=0)seed++;return seed;}
  public static void main(String[] args) throws Exception {
    long warmup=Long.parseLong(args[0]),duration=Long.parseLong(args[1]);
    Path fixtures=Path.of("fixtures/testframework/data");
    Random random=new Random(0);
    var delegate=new Lucene90StoredFieldsFormat();
    var primary=(StoredFieldsFormat)FORMAT.newInstance(delegate,random);
    long owned=RamUsageEstimator.shallowSizeOf(primary)+RamUsageEstimator.shallowSizeOf(delegate)+RamUsageEstimator.shallowSizeOf(random)+RamUsageEstimator.shallowSizeOf(new AtomicLong());
    System.out.printf("cranky_stored_fields_owned_bytes\t%d%n",owned);
    try(var directory=new ByteBuffersDirectory();var speedBase=new ByteBuffersDirectory();var compressionBase=new ByteBuffersDirectory()) {
      var output=info(directory,3);
      StoredFieldsFormat[] formats={primary,(StoredFieldsFormat)FORMAT.newInstance(new Lucene90StoredFieldsFormat(Lucene90StoredFieldsFormat.Mode.BEST_COMPRESSION),random)};
      Directory[] bases={speedBase,compressionBase};
      SegmentInfo[] baseInfos={info(speedBase,3),info(compressionBase,3)};
      for(int mode=0;mode<2;mode++) {
        String label=mode==0?"speed":"compression";
        for(String extension:new String[]{"fdt","fdx","fdm"})put(bases[mode],"_real."+extension,Files.readAllBytes(fixtures.resolve("cranky_stored_fields_"+label+"."+extension)));
        baseInfos[mode].putAttribute(Lucene90StoredFieldsFormat.MODE_KEY,mode==0?"BEST_SPEED":"BEST_COMPRESSION");
      }
      ByteArrayDataInput input=new ByteArrayDataInput(INPUT);
      for(int mode=0;mode<2;mode++) {
        String label=mode==0?"speed":"compression";
        StoredFieldsFormat format=formats[mode];Directory base=bases[mode];SegmentInfo baseInfo=baseInfos[mode];
        measure("cranky_stored_fields_read_"+label,warmup,duration,()-> {
          random.setSeed(0);Visitor visitor=new Visitor(OFFSET);
          for(int op=0;op<BATCH;op++) {
            try(var reader=format.fieldsReader(base,baseInfo,FIELD_INFOS,IOContext.DEFAULT)) {
              reader.checkIntegrity();visitor.digest=hash(visitor.digest,3);
              for(int doc=0;doc<3;doc++)reader.document(doc,visitor);
            }
          }
          return hash(visitor.digest,random.nextInt());
        });
        // SegmentInfo's mode attribute is written by fieldsWriter. A fresh
        // metadata object per mode avoids a conflicting mode on the same info.
        SegmentInfo outputInfo=info(directory,3);
        measure("cranky_stored_fields_write_"+label,warmup,duration,()-> {
          long digest=OFFSET;
          for(int op=0;op<BATCH;op++) {
            random.setSeed(0);
            try(var writer=format.fieldsWriter(directory,outputInfo,IOContext.DEFAULT)) {
              for(int doc=0;doc<3;doc++) {
                writer.startDocument();
                writer.writeField(FIELDS[0],-3+doc);writer.writeField(FIELDS[1],Long.MIN_VALUE+doc);
                writer.writeField(FIELDS[2],-0.0f);writer.writeField(FIELDS[3],Double.longBitsToDouble(0x7ff8000000000042L));
                writer.writeField(FIELDS[4],new BytesRef(BINARY));input.reset(INPUT);writer.writeField(FIELDS[5],new StoredFieldDataInput(input,2));
                writer.writeField(FIELDS[6],STRINGS[doc]);writer.finishDocument();
              }
              writer.finish(3);
            }
            digest=consume(digest,directory);digest=hash(digest,random.nextInt());
          }
          return digest;
        });
      }
      try(var reader=primary.fieldsReader(speedBase,baseInfos[0],FIELD_INFOS,IOContext.DEFAULT)) {
        for(boolean sorted:new boolean[]{false,true}) {
          MergeState merge=mergeState(directory,reader,sorted);
          measure(sorted?"cranky_stored_fields_merge_sorted":"cranky_stored_fields_merge_sequential",warmup,duration,()-> {
            long digest=OFFSET;
            for(int op=0;op<BATCH;op++) {
              random.setSeed(0);
              try(var writer=primary.fieldsWriter(directory,output,IOContext.DEFAULT)) {digest=hash(digest,writer.merge(merge));}
              digest=consume(digest,directory);digest=hash(digest,random.nextInt());
            }
            return digest;
          });
        }
      }
      long[] seeds={first(100),first(1000),first(10000)};
      CountWriter count=new CountWriter();StoredFieldsWriter writer=(StoredFieldsWriter)WRITER.newInstance(count,random);
      String[] expected={"Fake IOException from StoredFieldsFormat.fieldsWriter()","Fake IOException from StoredFieldsWriter.close()","Fake IOException from StoredFieldsWriter.startDocument()","Fake IOException from StoredFieldsWriter.writeField()","Fake IOException from StoredFieldsWriter.finish()","Fake IOException from StoredFieldsWriter.merge()"};
      measure("cranky_stored_fields_faults",warmup,duration,()-> {
        count.closed=0;long digest=OFFSET;
        for(int iteration=0;iteration<BATCH;iteration++) {
          for(int op=0;op<6;op++) {
            random.setSeed(seeds[op==1?1:(op==2||op==3?2:0)]);input.reset(INPUT);
            String message;
            try {
              switch(op) {
                case 0->primary.fieldsWriter(directory,output,IOContext.DEFAULT);
                case 1->writer.close();case 2->writer.startDocument();
                case 3->writer.writeField(FIELDS[0],new StoredFieldDataInput(input,2));
                case 4->writer.finish(0);case 5->writer.merge(null);
              }
              throw new AssertionError("missing injection");
            } catch(IOException e) {message=e.getMessage();}
            if(!message.equals(expected[op])||input.getPosition()!=0)throw new AssertionError("wrong fault or consumed input");
            digest=bytes(digest,message.getBytes(StandardCharsets.UTF_8));digest=hash(digest,random.nextInt());
          }
        }
        if(count.closed!=BATCH||directory.listAll().length!=0)throw new AssertionError("wrong close count or output creation");
        return hash(digest,count.closed);
      });
    }
  }
}