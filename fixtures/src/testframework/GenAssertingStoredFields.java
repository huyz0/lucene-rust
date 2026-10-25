import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.atomic.AtomicReference;
import org.apache.lucene.codecs.*;
import org.apache.lucene.codecs.lucene90.Lucene90StoredFieldsFormat;
import org.apache.lucene.index.*;
import org.apache.lucene.store.*;
import org.apache.lucene.tests.codecs.asserting.AssertingStoredFieldsFormat;
import org.apache.lucene.util.*;

/** Actual Lucene asserting stored-fields wrappers, run with -ea. */
public class GenAssertingStoredFields {
  static final byte[] ID={0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
  static FieldInfo field(String name,int number) {
    return new FieldInfo(name,number,false,false,false,IndexOptions.NONE,DocValuesType.NONE,
        DocValuesSkipIndexType.NONE,-1,Map.of(),0,0,0,0,VectorEncoding.FLOAT32,
        VectorSimilarityFunction.EUCLIDEAN,false,false);
  }
  static final FieldInfo FIELD=field("f",7);
  static final FieldInfos FIELDS=new FieldInfos(new FieldInfo[]{FIELD});
  static StoredFieldsWriter writer(StoredFieldsWriter delegate)throws Exception {
    var c=Class.forName("org.apache.lucene.tests.codecs.asserting.AssertingStoredFieldsFormat$AssertingStoredFieldsWriter").getDeclaredConstructor(StoredFieldsWriter.class);
    c.setAccessible(true);return (StoredFieldsWriter)c.newInstance(delegate);
  }
  static StoredFieldsReader reader(StoredFieldsReader delegate,boolean merging)throws Exception {
    var c=Class.forName("org.apache.lucene.tests.codecs.asserting.AssertingStoredFieldsFormat$AssertingStoredFieldsReader").getDeclaredConstructor(StoredFieldsReader.class,int.class,boolean.class);
    c.setAccessible(true);return (StoredFieldsReader)c.newInstance(delegate,2,merging);
  }
  static AssertingStoredFieldsFormat format(StoredFieldsFormat delegate)throws Exception {
    var result=new AssertingStoredFieldsFormat();var f=AssertingStoredFieldsFormat.class.getDeclaredField("in");f.setAccessible(true);f.set(result,delegate);return result;
  }
  @FunctionalInterface interface Action {void run()throws Exception;}
  static String outcome(Action action)throws Exception {
    try {action.run();return "ok";}catch(AssertionError e){return "assert";}catch(IOException e){return "io:"+e.getMessage();}
  }
  static class ProbeWriter extends StoredFieldsWriter {
    final List<String> events;boolean fail;int closes,failClose;
    final Collection<Accountable> children=List.of(Accountable.NULL_ACCOUNTABLE);
    ProbeWriter(List<String> events){this.events=events;}
    void call(String op)throws IOException {events.add(op);if(fail)throw new IOException("delegate failure");}
    @Override public void startDocument()throws IOException {call("start");}
    @Override public void finishDocument()throws IOException {call("end");}
    @Override public void finish(int docs)throws IOException {call("finish:"+docs);}
    @Override public void close()throws IOException {events.add("close:"+ ++closes);if(closes==failClose)throw new IOException("close failure");}
    @Override public void writeField(FieldInfo f,int v)throws IOException {if(f!=FIELD||v!=-17)throw new AssertionError();call("int");}
    @Override public void writeField(FieldInfo f,long v)throws IOException {if(f!=FIELD||v!=Long.MIN_VALUE)throw new AssertionError();call("long");}
    @Override public void writeField(FieldInfo f,float v)throws IOException {if(f!=FIELD||Float.floatToRawIntBits(v)!=0x80000000)throw new AssertionError();call("float");}
    @Override public void writeField(FieldInfo f,double v)throws IOException {if(f!=FIELD||Double.doubleToRawLongBits(v)!=0x7ff8000000000042L)throw new AssertionError();call("double");}
    @Override public void writeField(FieldInfo f,BytesRef v)throws IOException {if(f!=FIELD||!v.equals(new BytesRef(new byte[]{0,1,(byte)255})))throw new AssertionError();call("binary");}
    @Override public void writeField(FieldInfo f,String v)throws IOException {if(f!=FIELD||!v.equals("hello λ"))throw new AssertionError();call("string");}
    @Override public void writeField(FieldInfo f,StoredFieldDataInput v)throws IOException {call("input");byte[] bytes=new byte[v.length()];v.getDataInput().readBytes(bytes,0,bytes.length);if(f!=FIELD||!Arrays.equals(bytes,new byte[]{9,8}))throw new AssertionError();}
    @Override public int merge(MergeState s){throw new AssertionError("delegate merge called");}
    @Override public long ramBytesUsed(){events.add("ram");return 73;}
    @Override public Collection<Accountable> getChildResources(){events.add("children");return children;}
  }
  static void writeTrace(String sequence,StringBuilder out)throws Exception {
    var events=new ArrayList<String>();var probe=new ProbeWriter(events);var wrapped=writer(probe);
    var statuses=new ArrayList<String>();int consumed=0;
    for(String token:sequence.split(",")) {
      boolean fail=token.endsWith("!");String op=fail?token.substring(0,token.length()-1):token;
      probe.fail=fail;if(op.startsWith("close")) {probe.fail=false;probe.failClose=fail?probe.closes+(op.equals("close2")?2:1):0;}
      var input=new ByteArrayDataInput(new byte[]{9,8});
      statuses.add(outcome(()->{
        switch(op) {
          case "start"->wrapped.startDocument();case "end"->wrapped.finishDocument();
          case "int"->wrapped.writeField(FIELD,-17);case "long"->wrapped.writeField(FIELD,Long.MIN_VALUE);
          case "float"->wrapped.writeField(FIELD,-0.0f);case "double"->wrapped.writeField(FIELD,Double.longBitsToDouble(0x7ff8000000000042L));
          case "binary"->wrapped.writeField(FIELD,new BytesRef(new byte[]{0,1,(byte)255}));case "string"->wrapped.writeField(FIELD,"hello λ");
          case "input"->wrapped.writeField(FIELD,new StoredFieldDataInput(input,2));
          case "close","close2"->wrapped.close();
          case "resources"->{if(wrapped.ramBytesUsed()!=73||wrapped.getChildResources()!=probe.children)throw new AssertionError("resource identity");}
          default->{if(!op.startsWith("finish"))throw new AssertionError(op);wrapped.finish(Integer.parseInt(op.substring(6)));}
        }
      }));consumed+=input.getPosition();
    }
    out.append("writer\t").append(sequence).append('\t').append(String.join(";",statuses)).append('\t').append(String.join(",",events)).append('\t').append(consumed).append('\n');
  }
  static class ProbeReader extends StoredFieldsReader {
    final List<String> events;final String name;String fail;int closes,failClose;
    ProbeReader(List<String> events,String name){this.events=events;this.name=name;}
    void call(String op)throws IOException {events.add(name+":"+op);if(op.equals(fail))throw new IOException("reader failure");}
    @Override public void document(int doc,StoredFieldVisitor v)throws IOException {call("doc:"+doc);if(v.needsField(FIELD)==StoredFieldVisitor.Status.YES)v.intField(FIELD,17);}
    @Override public void checkIntegrity()throws IOException {call("integrity");}
    @Override public void close()throws IOException {events.add(name+":close:"+ ++closes);if(closes==failClose)throw new IOException("close failure");}
    @Override public StoredFieldsReader clone(){events.add(name+":clone");return new ProbeReader(events,name+"c");}
    @Override public StoredFieldsReader getMergeInstance(){events.add(name+":merge");return new ProbeReader(events,name+"m");}
    @Override public String toString(){return "ProbeReader";}
  }
  static final StoredFieldVisitor VISITOR=new StoredFieldVisitor(){
    @Override public Status needsField(FieldInfo f){return Status.YES;}
    @Override public void intField(FieldInfo f,int v){if(f!=FIELD||v!=17)throw new AssertionError();}
  };
  static void readTrace(String name,StringBuilder out)throws Exception {
    var events=new ArrayList<String>();var probe=new ProbeReader(events,"r");var wrapped=reader(probe,name.equals("merge-clone"));
    String result=outcome(()->{
      switch(name){
        case "valid"->{wrapped.document(0,VISITOR);wrapped.document(1,VISITOR);wrapped.checkIntegrity();wrapped.close();}
        case "negative"->wrapped.document(-1,VISITOR);case "past-end"->wrapped.document(2,VISITOR);
        case "document-error"->{probe.fail="doc:0";wrapped.document(0,VISITOR);}
        case "visitor-error"->wrapped.document(0,new StoredFieldVisitor(){@Override public Status needsField(FieldInfo f)throws IOException{throw new IOException("visitor failure");}});
        case "integrity-error"->{probe.fail="integrity";wrapped.checkIntegrity();}
        case "close-first","close-second"->{probe.failClose=name.equals("close-first")?1:2;wrapped.close();}
        case "clone"->{var copy=wrapped.clone();wrapped.close();copy.document(1,VISITOR);copy.close();}
        case "merge"->{var copy=wrapped.getMergeInstance();wrapped.close();copy.document(1,VISITOR);copy.close();}
        case "merge-close"->{var copy=wrapped.getMergeInstance();copy.close();wrapped.document(0,VISITOR);wrapped.close();}
        case "merge-clone"->wrapped.clone();case "merge-merge"->wrapped.getMergeInstance().getMergeInstance().document(0,VISITOR);
        case "display"->{if(!wrapped.toString().equals("AssertingStoredFieldsReader(ProbeReader)"))throw new AssertionError();}
        case "thread","merge-thread","thread-clone"->{
          StoredFieldsReader target=name.equals("merge-thread")?wrapped.getMergeInstance():wrapped;
          var resultOnThread=new AtomicReference<String>();Thread thread=new Thread(()->{
            try{resultOnThread.set(outcome(()->{if(name.equals("thread-clone"))target.clone().document(0,VISITOR);else target.document(0,VISITOR);}));}
            catch(Exception e){throw new RuntimeException(e);}
          });thread.start();thread.join();if(resultOnThread.get().equals("assert"))throw new AssertionError();if(!resultOnThread.get().equals("ok"))throw new IOException(resultOnThread.get());
        }
        default->throw new AssertionError(name);
      }
    });
    out.append("reader\t").append(name).append('\t').append(result).append('\t').append(String.join(",",events)).append('\n');
  }
  static class ProbeFormat extends StoredFieldsFormat {
    final List<String> events;final boolean fail;
    ProbeFormat(List<String> events,boolean fail){this.events=events;this.fail=fail;}
    @Override public StoredFieldsWriter fieldsWriter(Directory d,SegmentInfo si,IOContext ctx)throws IOException{events.add("open-writer");if(fail)throw new IOException("open failure");return new ProbeWriter(events);}
    @Override public StoredFieldsReader fieldsReader(Directory d,SegmentInfo si,FieldInfos f,IOContext ctx)throws IOException{events.add("open-reader");if(fail)throw new IOException("open failure");return new ProbeReader(events,"r");}
  }
  static void formatTrace(String name,StringBuilder out)throws Exception{
    var events=new ArrayList<String>();var wrapped=format(new ProbeFormat(events,name.endsWith("error")));
    try(var directory=new ByteBuffersDirectory()){
      String status=outcome(()->{if(name.startsWith("writer"))wrapped.fieldsWriter(directory,info(directory,2),IOContext.DEFAULT);else wrapped.fieldsReader(directory,info(directory,2),FIELDS,IOContext.DEFAULT);});
      out.append("format\t").append(name).append('\t').append(status).append('\t').append(String.join(",",events)).append('\n');
    }
  }
  static void mergeTrace(boolean sorted,StringBuilder out)throws Exception{
    var events=new ArrayList<String>();var wrapped=writer(new ProbeWriter(events));
    try(var directory=new ByteBuffersDirectory()){
      var constructor=MergeState.class.getDeclaredConstructor(List.class,SegmentInfo.class,InfoStream.class,java.util.concurrent.Executor.class,MergePolicy.OneMerge.class);
      constructor.setAccessible(true);var state=(MergeState)constructor.newInstance(List.of(),info(directory,-1),InfoStream.NO_OUTPUT,null,null);
      StoredFieldsReader[] readers=new StoredFieldsReader[2];
      for(int i=0;i<2;i++){readers[i]=new ProbeReader(events,i==0?"a":"b"){
        @Override public void document(int doc,StoredFieldVisitor visitor)throws IOException{call("doc:"+doc);if(visitor.needsField(FIELD)==StoredFieldVisitor.Status.YES)visitor.intField(FIELD,-17);}
      };}
      int[][] maps=sorted?new int[][]{{1,-1},{0,2}}:new int[][]{{0,-1},{1,2}};
      MergeState.DocMap[] docMaps=new MergeState.DocMap[2];for(int i=0;i<2;i++){int[] map=maps[i];docMaps[i]=doc->map[doc];}
      for(var entry:Map.of("storedFieldsReaders",readers,"fieldInfos",new FieldInfos[]{FIELDS,FIELDS},"maxDocs",new int[]{2,2},"docMaps",docMaps).entrySet()){
        var f=MergeState.class.getField(entry.getKey());f.setAccessible(true);f.set(state,entry.getValue());
      }
      state.mergeFieldInfos=FIELDS;state.needsIndexSort=sorted;
      String result=outcome(()->{if(wrapped.merge(state)!=3)throw new AssertionError("merge count");});
      out.append("merge\t").append(sorted?"sorted":"sequential").append('\t').append(result).append('\t').append(String.join(",",events)).append('\n');
    }
  }
  static SegmentInfo info(Directory dir,int docs){return new SegmentInfo(dir,Version.LATEST,null,"_asf",docs,false,false,null,Map.of(),ID,Map.of(),null);}
  static void real(Path out,Lucene90StoredFieldsFormat.Mode mode,boolean empty)throws Exception {
    String stem=empty?"empty":mode==Lucene90StoredFieldsFormat.Mode.BEST_SPEED?"speed":"compression";
    try(var dir=new ByteBuffersDirectory()) {
      var format=format(new Lucene90StoredFieldsFormat(mode));var si=info(dir,empty?0:2);
      var w=format.fieldsWriter(dir,si,IOContext.DEFAULT);
      if(!empty){w.startDocument();w.writeField(FIELD,-17);w.writeField(FIELD,Long.MIN_VALUE);w.writeField(FIELD,-0.0f);w.writeField(FIELD,Double.longBitsToDouble(0x7ff8000000000042L));w.writeField(FIELD,new BytesRef(new byte[]{0,1,(byte)255}));w.writeField(FIELD,new StoredFieldDataInput(new ByteArrayDataInput(new byte[]{9,8}),2));w.writeField(FIELD,"hello λ");w.finishDocument();w.startDocument();w.finishDocument();}
      w.finish(empty?0:2);w.close();
      try(var r=format.fieldsReader(dir,si,FIELDS,IOContext.DEFAULT)) {
        r.checkIntegrity();
        if(!empty){int[] seen={0};r.document(0,new StoredFieldVisitor(){
          @Override public Status needsField(FieldInfo f){return Status.YES;}
          @Override public void intField(FieldInfo f,int v){if(v!=-17)throw new AssertionError();seen[0]++;}
          @Override public void longField(FieldInfo f,long v){if(v!=Long.MIN_VALUE)throw new AssertionError();seen[0]++;}
          @Override public void floatField(FieldInfo f,float v){if(Float.floatToRawIntBits(v)!=0x80000000)throw new AssertionError();seen[0]++;}
          @Override public void doubleField(FieldInfo f,double v){if(Double.doubleToRawLongBits(v)!=0x7ff8000000000000L)throw new AssertionError();seen[0]++;}
          @Override public void binaryField(FieldInfo f,byte[] v){if(!Arrays.equals(v,seen[0]==4?new byte[]{0,1,(byte)255}:new byte[]{9,8}))throw new AssertionError();seen[0]++;}
          @Override public void stringField(FieldInfo f,String v){if(!v.equals("hello λ"))throw new AssertionError();seen[0]++;}
        });if(seen[0]!=7)throw new AssertionError("fields");r.document(1,VISITOR);}
      }
      for(String extension:new String[]{"fdt","fdx","fdm"})try(var in=dir.openInput("_asf."+extension,IOContext.DEFAULT)){byte[] bytes=new byte[(int)in.length()];in.readBytes(bytes,0,bytes.length);Files.write(out.resolve("asserting_stored_fields_"+stem+"."+extension),bytes);}
    }
  }
  public static void main(String[] args)throws Exception {
    if(!GenAssertingStoredFields.class.desiredAssertionStatus())throw new AssertionError("run with -ea");
    Path out=Path.of(args[0]);Files.createDirectories(out);StringBuilder trace=new StringBuilder();
    for(String sequence:new String[]{"finish0,close,close","finish-1","start,int,long,float,double,binary,input,string,end,finish1,resources,close",
      "end","start,start,end,end","start,finish1,end,finish1","start,end,finish0","start,end,finish2","start,end,finish2!","start!,start,end,finish1","start,end!,end,finish1","start,int!,int,end,finish1","start,input!,input,end,finish1","start,end,finish1,finish1","close!","close2!","close,close","start,end,start,end,finish2"})writeTrace(sequence,trace);
    for(String kind:new String[]{"int","long","float","double","binary","input","string"}){writeTrace(kind,trace);writeTrace("start,end,"+kind,trace);}
    for(String name:new String[]{"valid","negative","past-end","document-error","visitor-error","integrity-error","close-first","close-second","clone","merge","merge-close","merge-clone","merge-merge","display","thread","merge-thread","thread-clone"})readTrace(name,trace);
    for(String name:new String[]{"writer","writer-error","reader","reader-error"})formatTrace(name,trace);
    mergeTrace(false,trace);mergeTrace(true,trace);
    Files.writeString(out.resolve("asserting_stored_fields.tsv"),trace);
    real(out,Lucene90StoredFieldsFormat.Mode.BEST_SPEED,false);real(out,Lucene90StoredFieldsFormat.Mode.BEST_COMPRESSION,false);real(out,Lucene90StoredFieldsFormat.Mode.BEST_SPEED,true);
  }
}
