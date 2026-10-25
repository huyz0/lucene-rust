import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import org.apache.lucene.codecs.*;
import org.apache.lucene.codecs.lucene90.Lucene90StoredFieldsFormat;
import org.apache.lucene.index.*;
import org.apache.lucene.store.*;
import org.apache.lucene.util.*;

/** Actual cranky stored-fields format, callback/RNG traces and real codec bytes. */
public class GenCrankyStoredFields {
  static final byte[] ID = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
  static final FieldInfo FIELD = field("shared",7);
  static StoredFieldsFormat wrap(StoredFieldsFormat delegate, Random random) throws Exception {
    var c = Class.forName("org.apache.lucene.tests.codecs.cranky.CrankyStoredFieldsFormat")
        .getDeclaredConstructor(StoredFieldsFormat.class,Random.class);
    c.setAccessible(true);
    return (StoredFieldsFormat)c.newInstance(delegate,random);
  }
  static StoredFieldsWriter wrapWriter(StoredFieldsWriter delegate, Random random) throws Exception {
    var c = Class.forName("org.apache.lucene.tests.codecs.cranky.CrankyStoredFieldsFormat$CrankyStoredFieldsWriter")
        .getDeclaredConstructor(StoredFieldsWriter.class,Random.class);
    c.setAccessible(true);
    return (StoredFieldsWriter)c.newInstance(delegate,random);
  }
  static FieldInfo field(String name,int number) {
    return new FieldInfo(name,number,false,false,false,IndexOptions.NONE,DocValuesType.NONE,
        DocValuesSkipIndexType.NONE,-1,Map.of(),0,0,0,0,VectorEncoding.FLOAT32,
        VectorSimilarityFunction.EUCLIDEAN,false,false);
  }
  static SegmentInfo info(Directory dir,String name,int docs) {
    return new SegmentInfo(dir,Version.LATEST,null,name,docs,false,false,null,Map.of(),ID.clone(),Map.of(),null);
  }
  static class State { boolean fail; final List<String> events = new ArrayList<>(); }
  static class RecordingWriter extends StoredFieldsWriter {
    final State state;
    final Collection<Accountable> children = List.of(Accountable.NULL_ACCOUNTABLE);
    RecordingWriter(State state) { this.state = state; }
    void event(String name) throws IOException { state.events.add(name); if(state.fail)throw new IOException("delegate failure"); }
    @Override public void startDocument() throws IOException { event("start"); }
    @Override public void finishDocument() throws IOException { event("finishDoc"); }
    @Override public void finish(int docs) throws IOException { event("finish:"+docs); }
    @Override public void close() throws IOException { event("close"); }
    @Override public int merge(MergeState state) { throw new AssertionError("delegate merge called"); }
    @Override public void writeField(FieldInfo f,int v) throws IOException { event("int:"+f.number+":"+v); }
    @Override public void writeField(FieldInfo f,long v) throws IOException { event("long:"+f.number+":"+v); }
    @Override public void writeField(FieldInfo f,float v) throws IOException { event("float:"+f.number+":"+Integer.toUnsignedString(Float.floatToRawIntBits(v))); }
    @Override public void writeField(FieldInfo f,double v) throws IOException { event("double:"+f.number+":"+Long.toUnsignedString(Double.doubleToRawLongBits(v))); }
    @Override public void writeField(FieldInfo f,BytesRef v) throws IOException { event("binary:"+f.number+":"+HexFormat.of().formatHex(v.bytes,v.offset,v.offset+v.length)); }
    @Override public void writeField(FieldInfo f,StoredFieldDataInput v) throws IOException {
      event("input:"+f.number+":"+v.length());
      byte[] bytes = new byte[v.length()];v.getDataInput().readBytes(bytes,0,bytes.length);
      if(!Arrays.equals(bytes,new byte[]{9,8}))throw new AssertionError("input value");
    }
    @Override public void writeField(FieldInfo f,String v) throws IOException { event("string:"+f.number+":"+v); }
    @Override public long ramBytesUsed() { state.events.add("ram");return 73; }
    @Override public Collection<Accountable> getChildResources() { state.events.add("children");return children; }
  }
  static class RecordingFormat extends StoredFieldsFormat {
    final State state; final RecordingWriter writer;
    final StoredFieldsReader reader = new Source(new State(),0,false,false);
    RecordingFormat(State state) { this.state=state;writer=new RecordingWriter(state); }
    @Override public StoredFieldsWriter fieldsWriter(Directory d,SegmentInfo si,IOContext ctx) throws IOException {
      writer.event("open");return writer;
    }
    @Override public StoredFieldsReader fieldsReader(Directory d,SegmentInfo si,FieldInfos f,IOContext ctx) throws IOException {
      writer.event("read");return reader;
    }
  }
  static MergeState mergeState(StoredFieldsReader[] readers,boolean sorted) throws Exception {
    try(var dir=new ByteBuffersDirectory()) {
      var c=MergeState.class.getDeclaredConstructor(List.class,SegmentInfo.class,InfoStream.class,java.util.concurrent.Executor.class,MergePolicy.OneMerge.class);
      c.setAccessible(true);
      MergeState state=(MergeState)c.newInstance(List.of(),info(dir,"_merge",-1),InfoStream.NO_OUTPUT,null,null);
      FieldInfos[] fields=new FieldInfos[readers.length];
      int[] maxDocs=new int[readers.length];
      MergeState.DocMap[] maps=new MergeState.DocMap[readers.length];
      for(int i=0;i<readers.length;i++) {
        int base=10+i*10;
        fields[i]=new FieldInfos(new FieldInfo[]{field("shared",base),field("binary",base+1)});
        maxDocs[i]=3;
        int[] values=i==0 ? (sorted?new int[]{0,-1,2}:new int[]{0,-1,1}) : (sorted?new int[]{1,3,-1}:new int[]{2,3,-1});
        maps[i]=doc->values[doc];
      }
      for(var entry:Map.of("storedFieldsReaders",readers,"fieldInfos",fields,"maxDocs",maxDocs,"docMaps",maps).entrySet()) {
        var f=MergeState.class.getDeclaredField(entry.getKey());f.setAccessible(true);f.set(state,entry.getValue());
      }
      state.mergeFieldInfos=new FieldInfos(new FieldInfo[]{field("shared",0),field("binary",1)});
      state.needsIndexSort=sorted;
      return state;
    }
  }
  static class Source extends StoredFieldsReader {
    final State state;final int source;final boolean failIntegrity,failVisit;
    Source(State state,int source,boolean failIntegrity,boolean failVisit) {this.state=state;this.source=source;this.failIntegrity=failIntegrity;this.failVisit=failVisit;}
    @Override public void checkIntegrity() throws IOException {
      state.events.add("integrity:"+source);if(failIntegrity)throw new IOException("source integrity");
    }
    @Override public void document(int doc,StoredFieldVisitor visitor) throws IOException {
      state.events.add("visit:"+source+":"+doc);
      int number=10+source*10;
      FieldInfo f=field("shared",number);
      if(visitor.needsField(f)==StoredFieldVisitor.Status.YES)visitor.intField(f,source*100+doc);
      if(failVisit)throw new IOException("source visit");
      f=field("binary",number+1);
      if(visitor.needsField(f)==StoredFieldVisitor.Status.YES)visitor.binaryField(f,new byte[]{(byte)source,(byte)doc});
    }
    @Override public StoredFieldsReader clone() {return this;}
    @Override public void close() {}
  }
  static long first(int bound) {long seed=0;while(new Random(seed).nextInt(bound)!=0)seed++;return seed;}
  static String sequence(long seed) throws Exception {
    State state=new State();Random random=new Random(seed);
    RecordingFormat delegate=new RecordingFormat(state);
    StoredFieldsFormat format=wrap(delegate,random);
    StoredFieldsWriter writer=wrapWriter(delegate.writer,random);
    // A real MergeState contains at least one reader: DocIDMerger rejects
    // zero subs internally. Use one zero-document source for an empty merge.
    MergeState empty=mergeState(new StoredFieldsReader[]{new Source(state,0,false,false)},false);
    var maxDocsField=MergeState.class.getDeclaredField("maxDocs");
    maxDocsField.setAccessible(true);maxDocsField.set(empty,new int[]{0});
    StringBuilder out=new StringBuilder();
    String[] operations={"open","read","start","finishDoc","int","long","float","double","binary","input","string","finish","close","ram","children","merge"};
    for(int step=0;step<2048;step++) {
      state.fail=step%17==0;state.events.clear();
      String op=operations[step%operations.length],status="ok";
      ByteArrayDataInput input=new ByteArrayDataInput(new byte[]{9,8});
      try {
        switch(op) {
          case "open"->format.fieldsWriter(null,null,null);
          case "read"->{if(format.fieldsReader(null,null,null,null)!=delegate.reader)throw new AssertionError("reader identity");}
          case "start"->writer.startDocument();
          case "finishDoc"->writer.finishDocument();
          case "int"->writer.writeField(FIELD,-3);
          case "long"->writer.writeField(FIELD,Long.MIN_VALUE);
          case "float"->writer.writeField(FIELD,-0.0f);
          case "double"->writer.writeField(FIELD,Double.longBitsToDouble(0x7ff8000000000042L));
          case "binary"->writer.writeField(FIELD,new BytesRef(new byte[]{0,1,(byte)255}));
          case "input"->writer.writeField(FIELD,new StoredFieldDataInput(input,2));
          case "string"->writer.writeField(FIELD,"hello λ");
          case "finish"->writer.finish(3);
          case "close"->writer.close();
          case "ram"->{if(writer.ramBytesUsed()!=73)throw new AssertionError("ram forwarding");}
          case "children"->{if(writer.getChildResources()!=delegate.writer.children)throw new AssertionError("children identity");}
          case "merge"->{if(writer.merge(empty)!=0)throw new AssertionError("empty count");}
        }
      }catch(IOException e){status=e.getMessage();}
      out.append("op\t").append(seed).append('\t').append(step).append('\t').append(op).append('\t').append(state.fail)
          .append('\t').append(status).append('\t').append(String.join(",",state.events)).append('\t').append(input.getPosition()).append('\n');
    }
    out.append("rng\t").append(seed).append('\t').append(random.nextInt()).append('\n');
    return out.toString();
  }
  static String mergeCase(boolean sorted,boolean integrity,boolean visit) throws Exception {
    State state=new State();Random random=new Random(42);
    StoredFieldsWriter writer=wrapWriter(new RecordingWriter(state),random);
    StoredFieldsReader[] readers={new Source(state,0,false,visit),new Source(state,1,integrity,false)};
    String status="ok";int count=-1;
    try {count=writer.merge(mergeState(readers,sorted));}catch(IOException e){status=e.getMessage();}
    return "merge\t"+sorted+"\t"+integrity+"\t"+visit+"\t"+status+"\t"+count+"\t"+String.join(",",state.events)+"\t"+random.nextInt()+"\n";
  }
  static void writeReal(Path out,Lucene90StoredFieldsFormat.Mode mode,boolean finished) throws Exception {
    try(var dir=new ByteBuffersDirectory()) {
      String stem=mode==Lucene90StoredFieldsFormat.Mode.BEST_SPEED?"speed":"compression";
      if(!finished)stem="unfinished";
      SegmentInfo segment=info(dir,"_real",3);
      Random random=new Random(0);
      StoredFieldsFormat format=wrap(new Lucene90StoredFieldsFormat(mode),random);
      StoredFieldsWriter writer=format.fieldsWriter(dir,segment,IOContext.DEFAULT);
      if(finished) {
        for(int doc=0;doc<3;doc++) {
          writer.startDocument();
          writer.writeField(field("int",0),-3+doc);
          writer.writeField(field("long",1),Long.MIN_VALUE+doc);
          writer.writeField(field("float",2),-0.0f);
          writer.writeField(field("double",3),Double.longBitsToDouble(0x7ff8000000000042L));
          writer.writeField(field("binary",4),new BytesRef(new byte[]{0,1,(byte)255}));
          writer.writeField(field("input",5),new StoredFieldDataInput(new ByteArrayDataInput(new byte[]{9,8}),2));
          writer.writeField(field("string",6),"hello λ "+doc);
          writer.finishDocument();
        }
        writer.finish(3);
      }
      writer.close();
      if(finished) {
        var fields=new FieldInfos(new FieldInfo[]{field("int",0),field("long",1),field("float",2),field("double",3),field("binary",4),field("input",5),field("string",6)});
        try(var reader=format.fieldsReader(dir,segment,fields,IOContext.DEFAULT)) {
          reader.checkIntegrity();
          for(int doc=0;doc<3;doc++) {
            final int expected=doc;
            int[] count={0};
            reader.document(doc,new StoredFieldVisitor(){
              @Override public Status needsField(FieldInfo f){return Status.YES;}
              @Override public void intField(FieldInfo f,int v){if(v!=-3+expected)throw new AssertionError();count[0]++;}
              @Override public void longField(FieldInfo f,long v){if(v!=Long.MIN_VALUE+expected)throw new AssertionError();count[0]++;}
              @Override public void floatField(FieldInfo f,float v){if(Float.floatToRawIntBits(v)!=0x80000000)throw new AssertionError();count[0]++;}
              @Override public void doubleField(FieldInfo f,double v){if(!Double.isNaN(v))throw new AssertionError();count[0]++;}
              @Override public void binaryField(FieldInfo f,byte[] v){if(!Arrays.equals(v,f.number==4?new byte[]{0,1,(byte)255}:new byte[]{9,8}))throw new AssertionError();count[0]++;}
              @Override public void stringField(FieldInfo f,String v){if(!v.equals("hello λ "+expected))throw new AssertionError();count[0]++;}
            });
            if(count[0]!=7)throw new AssertionError("missing fields");
          }
        }
      }
      for(String extension:finished?new String[]{"fdt","fdx","fdm"}:new String[]{"fdt","fdm"}) {
        try(var input=dir.openInput("_real."+extension,IOContext.DEFAULT)) {
          byte[] bytes=new byte[(int)input.length()];input.readBytes(bytes,0,bytes.length);
          Files.write(out.resolve("cranky_stored_fields_"+stem+"."+extension),bytes);
        }
      }
      if(!finished&&Arrays.asList(dir.listAll()).contains("_real.fdx"))throw new AssertionError("unfinished index");
      Files.writeString(out.resolve("cranky_stored_fields_"+stem+".rng"),Integer.toString(random.nextInt())+"\n");
    }
  }
  static class FailureDirectory extends FilterDirectory {
    final String failure;
    FailureDirectory(Directory delegate,String failure) {super(delegate);this.failure=failure;}
    @Override public IndexOutput createOutput(String name,IOContext context) throws IOException {
      if(name.endsWith(".fdx")&&failure.equals("create"))throw new IOException("fdx create");
      IndexOutput output=super.createOutput(name,context);
      if(name.endsWith(".fdx")&&failure.equals("close")
          ||name.endsWith(".fdt")&&(failure.equals("data_close")||failure.equals("both_close"))
          ||name.endsWith(".fdm")&&failure.equals("both_close"))return new FilterIndexOutput("fault",name,output) {
        @Override public void close() throws IOException {super.close();throw new IOException(name.substring(name.lastIndexOf('.')+1)+" close");}
      };
      return output;
    }
  }
  static void finishFailure(Path out,String failure) throws Exception {
    try(var dir=new FailureDirectory(new ByteBuffersDirectory(),failure)) {
      var random=new Random(42);
      var format=wrap(new Lucene90StoredFieldsFormat(),random);
      var writer=format.fieldsWriter(dir,info(dir,"_real",1),IOContext.DEFAULT);
      writer.startDocument();
      if(failure.equals("auto"))writer.writeField(field("int",0),"x".repeat(180000));
      else writer.writeField(field("int",0),17);
      writer.finishDocument();
      String status;boolean closed=false;
      if(failure.equals("auto"))status="unfinished automatic chunk";
      else if(failure.equals("data_close")||failure.equals("both_close")) {
        writer.finish(1);
        try {writer.close();throw new AssertionError("missing close failure");}
        catch(IOException e) {status=e.getMessage();closed=true;}
      } else {
        try {writer.finish(failure.equals("count")||failure.equals("retry")?2:1);throw new AssertionError("missing finish failure");}
        catch(IOException|RuntimeException e) {status=e.getMessage();}
        if(failure.equals("retry"))writer.finish(1);
      }
      if(!closed)writer.close();
      Files.writeString(out.resolve("cranky_stored_fields_failure_"+failure+".status"),status+"\n"+random.nextInt()+"\n");
      for(String extension:new String[]{"fdt","fdx","fdm"}) {
        String name="_real."+extension;
        if(Arrays.asList(dir.listAll()).contains(name))try(var input=dir.openInput(name,IOContext.DEFAULT)) {
          byte[] data=new byte[Math.toIntExact(input.length())];input.readBytes(data,0,data.length);
          Files.write(out.resolve("cranky_stored_fields_failure_"+failure+"."+extension),data);
        }
      }
    }
  }
  public static void main(String[] args) throws Exception {
    Path out=Path.of(args[0]);Files.createDirectories(out);
    StringBuilder trace=new StringBuilder();
    for(long seed:new long[]{0,1,42,-1,Long.MIN_VALUE,Long.MAX_VALUE,first(100),first(1000),first(10000)})trace.append(sequence(seed));
    for(boolean sorted:new boolean[]{false,true}) {
      trace.append(mergeCase(sorted,false,false));trace.append(mergeCase(sorted,true,false));trace.append(mergeCase(sorted,false,true));
    }
    Files.writeString(out.resolve("cranky_stored_fields.tsv"),trace);
    writeReal(out,Lucene90StoredFieldsFormat.Mode.BEST_SPEED,true);
    writeReal(out,Lucene90StoredFieldsFormat.Mode.BEST_COMPRESSION,true);
    writeReal(out,Lucene90StoredFieldsFormat.Mode.BEST_SPEED,false);
    for(String failure:new String[]{"count","create","close","auto","retry","data_close","both_close"})finishFailure(out,failure);
  }
}
