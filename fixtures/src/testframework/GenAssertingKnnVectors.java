import java.io.IOException;
import java.lang.reflect.Constructor;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.atomic.AtomicReference;
import org.apache.lucene.codecs.*;
import org.apache.lucene.codecs.hnsw.HnswGraphProvider;
import org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsFormat;
import org.apache.lucene.index.*;
import org.apache.lucene.search.*;
import org.apache.lucene.store.*;
import org.apache.lucene.tests.codecs.asserting.AssertingKnnVectorsFormat;
import org.apache.lucene.util.*;
import org.apache.lucene.util.hnsw.*;

/** Actual asserting-vector wrappers, callback identities and real unquantized graph files. */
public class GenAssertingKnnVectors {
  static final byte[] ID = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
  static FieldInfo field(String name,int number,int dimension,VectorEncoding encoding) {
    return new FieldInfo(name,number,false,false,false,IndexOptions.NONE,DocValuesType.NONE,
        DocValuesSkipIndexType.NONE,-1,Map.of(),0,0,0,dimension,encoding,
        VectorSimilarityFunction.EUCLIDEAN,false,false);
  }
  static final FieldInfo FLOAT = field("float",0,2,VectorEncoding.FLOAT32);
  static final FieldInfo BYTE = field("byte",1,2,VectorEncoding.BYTE);
  static final FieldInfos FIELDS = new FieldInfos(new FieldInfo[]{FLOAT,BYTE,field("plain",2,0,VectorEncoding.FLOAT32)});
  @FunctionalInterface interface Action {void run()throws Exception;}
  static String outcome(Action action)throws Exception {
    try {action.run();return "ok";}
    catch(AssertionError e){return "assert";}
    catch(IOException e){return "io:"+e.getMessage();}
  }
  static KnnVectorsReader wrap(KnnVectorsReader reader,FieldInfos fields)throws Exception {
    var c=Class.forName("org.apache.lucene.tests.codecs.asserting.AssertingKnnVectorsFormat$AssertingKnnVectorsReader")
        .getDeclaredConstructor(KnnVectorsReader.class,FieldInfos.class);
    c.setAccessible(true);return (KnnVectorsReader)c.newInstance(reader,fields);
  }
  static KnnVectorsWriter wrap(KnnVectorsWriter writer)throws Exception {
    var c=Class.forName("org.apache.lucene.tests.codecs.asserting.AssertingKnnVectorsFormat$AssertingKnnVectorsWriter")
        .getDeclaredConstructor(KnnVectorsWriter.class);
    c.setAccessible(true);return (KnnVectorsWriter)c.newInstance(writer);
  }
  static final class State {
    final List<String> events=new ArrayList<>();String fail="";int size=3,dimension=2,initial=-1,closes;boolean missing;
    void call(String event)throws IOException {events.add(event);if(event.equals("close"))closes++;if(event.equals(fail)||(fail.equals("close2")&&closes==2&&event.equals("close")))throw new IOException("delegate failure");}
  }
  static KnnVectorValues.DocIndexIterator iterator(State s) {
    s.events.add("iterator");
    return new KnnVectorValues.DocIndexIterator(){int doc=s.initial;
      public int docID(){s.events.add("docID");return doc;}
      public int index(){return doc;}
      public int nextDoc(){return doc=doc+1<s.size?doc+1:NO_MORE_DOCS;}
      public int advance(int target){return doc=target<s.size?target:NO_MORE_DOCS;}
      public long cost(){return s.size;}
    };
  }
  static FloatVectorValues floats(State s) {
    return new FloatVectorValues(){
      public int size(){s.events.add("size");return s.size;}
      public int dimension(){s.events.add("dimension");return s.dimension;}
      public float[] vectorValue(int ord){return new float[]{ord,ord+1};}
      public FloatVectorValues copy(){return this;}
      public DocIndexIterator iterator(){return GenAssertingKnnVectors.iterator(s);}
    };
  }
  static ByteVectorValues bytes(State s) {
    return new ByteVectorValues(){
      public int size(){s.events.add("size");return s.size;}
      public int dimension(){s.events.add("dimension");return s.dimension;}
      public byte[] vectorValue(int ord){return new byte[]{(byte)ord,(byte)(ord+1)};}
      public ByteVectorValues copy(){return this;}
      public DocIndexIterator iterator(){return GenAssertingKnnVectors.iterator(s);}
    };
  }
  static final class ProbeReader extends KnnVectorsReader implements HnswGraphProvider {
    final State s;final FloatVectorValues f;final ByteVectorValues b;
    ProbeReader(State s){this.s=s;f=floats(s);b=bytes(s);}
    public FloatVectorValues getFloatVectorValues(String name)throws IOException{s.call("float:"+name);return s.missing?null:f;}
    public ByteVectorValues getByteVectorValues(String name)throws IOException{s.call("byte:"+name);return s.missing?null:b;}
    public void checkIntegrity()throws IOException{s.call("integrity");}
    public void close()throws IOException{s.call("close");}
    public KnnVectorsReader getMergeInstance()throws IOException{s.call("merge");return s.missing?null:new ProbeReader(s);}
    public void finishMerge()throws IOException{s.call("finishMerge");}
    public Map<String,Long> getOffHeapByteSize(FieldInfo field){s.events.add("memory:"+field.name);return Map.of("vec",17L);}
    public HnswGraph getGraph(String field)throws IOException{s.call("graph:"+field);return HnswGraph.EMPTY;}
    void search(String field,AcceptDocs accept)throws IOException {
      s.call("search:"+field);s.events.add("acceptWrapped:"+accept.getClass().getSimpleName());
      accept.bits();accept.cost();accept.iterator();
    }
    public void search(String field,float[] target,KnnCollector c,AcceptDocs a)throws IOException{search(field,a);}
    public void search(String field,byte[] target,KnnCollector c,AcceptDocs a)throws IOException{search(field,a);}
  }
  static AcceptDocs accept(State s) {
    return new AcceptDocs(){
      public Bits bits()throws IOException{s.call("bits");return null;}
      public int cost()throws IOException{s.call("cost");return 3;}
      public DocIdSetIterator iterator()throws IOException{s.call("acceptIterator");var it=DocIdSetIterator.all(3);if(s.initial!=-1)it.nextDoc();return it;}
    };
  }
  static void readerTrace(String program,StringBuilder out)throws Exception {
    var s=new State();var raw=new ProbeReader(s);var base=wrap(raw,FIELDS);KnnVectorsReader[] merged={null,null};
    var results=new ArrayList<String>();var acceptance=accept(s);
    for(String token:program.split(","))results.add(outcome(()->{
      String[] p=token.split(":");KnnVectorsReader r=p[0].startsWith("m")?merged[0]:base;
      switch(p[0]) {
        case "f" -> {if(base.getFloatVectorValues(p.length>1?p[1]:"float")!=raw.f)throw new AssertionError("identity");}
        case "b" -> {if(base.getByteVectorValues(p.length>1?p[1]:"byte")!=raw.b)throw new AssertionError("identity");}
        case "sf" -> base.search(p.length>1?p[1]:"float",new float[]{0,1},new TopKnnCollector(1,100),acceptance);
        case "sb" -> base.search(p.length>1?p[1]:"byte",new byte[]{0,1},new TopKnnCollector(1,100),acceptance);
        case "merge" -> merged[0]=base.getMergeInstance();
        case "merge2" -> merged[1]=base.getMergeInstance();
        case "mf" -> r.getFloatVectorValues("float");
        case "mb" -> r.getByteVectorValues("byte");
        case "msf" -> r.search("float",new float[]{0,1},new TopKnnCollector(1,100),acceptance);
        case "msb" -> r.search("byte",new byte[]{0,1},new TopKnnCollector(1,100),acceptance);
        case "mm" -> r.getMergeInstance();
        case "mc" -> r.close();
        case "finish" -> base.finishMerge();
        case "mfinish" -> r.finishMerge();
        case "finish2" -> merged[1].finishMerge();
        case "close" -> base.close();
        case "integrity" -> base.checkIntegrity();
        case "memory" -> {if(!base.getOffHeapByteSize(FLOAT).equals(Map.of("vec",17L)))throw new AssertionError();}
        case "graph" -> {if(((HnswGraphProvider)base).getGraph("float")!=HnswGraph.EMPTY)throw new AssertionError();}
        case "fail" -> s.fail=token.substring(5);
        case "ok" -> s.fail="";
        case "size" -> s.size=Integer.parseInt(p[1]);
        case "dim" -> s.dimension=Integer.parseInt(p[1]);
        case "initial" -> s.initial=Integer.parseInt(p[1]);
        case "missing" -> s.missing=true;
        default -> throw new AssertionError(token);
      }
    }));
    out.append("reader\t").append(program).append('\t').append(String.join(";",results)).append('\t').append(String.join(",",s.events)).append('\n');
  }
  static void acceptTrace(String name,StringBuilder out)throws Exception {
    var s=new State();var raw=accept(s);var wrapped=org.apache.lucene.tests.search.AssertingAcceptDocs.wrap(raw);
    var results=new ArrayList<String>();
    switch(name) {
      case "all" -> results.add(outcome(()->{if(org.apache.lucene.tests.search.AssertingAcceptDocs.wrap(wrapped)!=wrapped)throw new AssertionError("identity");wrapped.bits();wrapped.cost();wrapped.iterator();}));
      case "positioned" -> {s.initial=0;results.add(outcome(()->wrapped.iterator()));}
      case "thread" -> {for(String op:new String[]{"bits","cost","iterator"}){var result=new AtomicReference<String>();var thread=new Thread(()->{try{result.set(outcome(()->{switch(op){case "bits"->wrapped.bits();case "cost"->wrapped.cost();case "iterator"->wrapped.iterator();}}));}catch(Exception e){throw new RuntimeException(e);}});thread.start();thread.join();results.add(result.get());}}
      default -> {s.fail=name;results.add(outcome(()->{switch(name){case "bits"->wrapped.bits();case "cost"->wrapped.cost();case "acceptIterator"->wrapped.iterator();default->throw new AssertionError(name);}}));}
    }
    out.append("accept\t").append(name).append('\t').append(String.join(";",results)).append('\t').append(String.join(",",s.events)).append('\n');
  }
  static final class ProbeWriter extends KnnVectorsWriter {
    final State s;final IORunnable task;final KnnFieldVectorsWriter<float[]> field;
    ProbeWriter(State s){this.s=s;task=()->s.call("deferred");field=new KnnFieldVectorsWriter<>(){
      public void addValue(int doc,float[] value)throws IOException{s.call("addValue:"+doc);}
      public float[] copyValue(float[] value){return value.clone();}
      public long ramBytesUsed(){return 5;}
    };}
    public KnnFieldVectorsWriter<?> addField(FieldInfo f)throws IOException{s.call("add:"+f.name);return field;}
    public void flush(int max,org.apache.lucene.index.Sorter.DocMap map)throws IOException{s.call("flush:"+max+":"+(map==null));}
    public IORunnable mergeOneField(FieldInfo f,MergeState m)throws IOException{s.call("mergeOne:"+f.name);return s.missing?null:task;}
    public void finish()throws IOException{s.call("finish");}
    public void close()throws IOException{s.call("close");}
    public long ramBytesUsed(){s.events.add("ram");return 17;}
  }
  static SegmentInfo segment(Directory dir,int maxDoc){return new SegmentInfo(dir,Version.fromBits(10,5,0),null,"_ak",maxDoc,false,false,null,Map.of(),ID,Map.of(),null);}
  static MergeState emptyMerge(Directory dir)throws Exception {
    var c=MergeState.class.getDeclaredConstructor(List.class,SegmentInfo.class,InfoStream.class,java.util.concurrent.Executor.class,MergePolicy.OneMerge.class);
    // MergeState assigns the target document count, even for an empty input.
    c.setAccessible(true);var m=c.newInstance(List.of(),segment(dir,-1),InfoStream.NO_OUTPUT,(java.util.concurrent.Executor)Runnable::run,null);m.mergeFieldInfos=FIELDS;return m;
  }
  static void writerTrace(String name,StringBuilder out)throws Exception {
    if(name.startsWith("merge-reversed")){reversedMergeTrace(name,out);return;}
    var s=new State();var raw=new ProbeWriter(s);var w=wrap(raw);var results=new ArrayList<String>();
    try(var dir=new ByteBuffersDirectory()) {var merge=emptyMerge(dir);
      switch(name){
        case "all" -> {results.add(outcome(()->{if(w.addField(FLOAT)!=raw.field)throw new AssertionError("field identity");w.flush(7,null);if(w.mergeOneField(FLOAT,merge)!=raw.task)throw new AssertionError("task identity");raw.task.run();w.finish();if(w.ramBytesUsed()!=17)throw new AssertionError();w.close();}));}
        case "null-field" -> results.add(outcome(()->w.mergeOneField(null,merge)));
        case "null-state" -> results.add(outcome(()->w.mergeOneField(FLOAT,null)));
        case "no-task" -> {s.missing=true;results.add(outcome(()->{if(w.mergeOneField(FLOAT,merge)!=null)throw new AssertionError();}));}
        case "deferred-error" -> {s.fail="deferred";results.add(outcome(()->w.mergeOneField(FLOAT,merge).run()));}
        case "merge-order" -> results.add(outcome(()->w.merge(merge)));
        default -> {s.fail=name;results.add(outcome(()->{switch(name){case "add:float"->w.addField(FLOAT);case "flush:7:true"->w.flush(7,null);case "mergeOne:float"->w.mergeOneField(FLOAT,merge);case "finish"->w.finish();case "close"->w.close();default->throw new AssertionError(name);}}));}
      }
    }
    out.append("writer\t").append(name).append('\t').append(String.join(";",results)).append('\t').append(String.join(",",s.events)).append('\n');
  }
  static void reversedMergeTrace(String name,StringBuilder out)throws Exception {
    var s=new State();s.fail=name.equals("merge-reversed-field-error")?"mergeOne:byte":name.equals("merge-reversed-deferred-error")?"deferred:float":"";
    KnnVectorsWriter delegate=new KnnVectorsWriter(){
      public KnnFieldVectorsWriter<?> addField(FieldInfo field){throw new AssertionError();}
      public void flush(int maxDoc,org.apache.lucene.index.Sorter.DocMap map){throw new AssertionError();}
      public IORunnable mergeOneField(FieldInfo field,MergeState state)throws IOException{s.call("mergeOne:"+field.name);return ()->s.call("deferred:"+field.name);}
      public void finish()throws IOException{s.call("finish");}
      public void close(){throw new AssertionError();}
      public long ramBytesUsed(){throw new AssertionError();}
    };
    try(var dir=new ByteBuffersDirectory()) {var merge=emptyMerge(dir);merge.mergeFieldInfos=new FieldInfos(new FieldInfo[]{FIELDS.fieldInfo("plain"),BYTE,FLOAT});
      String status=outcome(()->wrap(delegate).merge(merge));out.append("writer\t").append(name).append('\t').append(status).append('\t').append(String.join(",",s.events)).append('\n');}
  }
  static void real(Path out)throws Exception {
    var format=new Lucene99HnswVectorsFormat(4,10,0);HnswGraphBuilder.randSeed=42;
    try(var dir=new ByteBuffersDirectory()) {
      var si=segment(dir,16);var fis=new FieldInfos(new FieldInfo[]{FLOAT,BYTE});
      try(var w=wrap(format.fieldsWriter(new SegmentWriteState(InfoStream.NO_OUTPUT,dir,si,fis,null,IOContext.DEFAULT,"s")))) {
        @SuppressWarnings("unchecked") var f=(KnnFieldVectorsWriter<float[]>)w.addField(FLOAT);
        @SuppressWarnings("unchecked") var b=(KnnFieldVectorsWriter<byte[]>)w.addField(BYTE);
        for(int doc=0;doc<16;doc++){if(doc%3!=1)f.addValue(doc,new float[]{doc*0.25f,16-doc});if(doc%4!=2)b.addValue(doc,new byte[]{(byte)(doc-8),(byte)(doc*2)});}
        w.flush(16,null);w.finish();
      }
      var manifest=new StringBuilder();var order=new StringBuilder();var metadata=new StringBuilder();
      for(var field:new FieldInfo[]{FLOAT,BYTE}) {
        FieldInfo changed=new FieldInfo(field.name,field.number,false,false,false,IndexOptions.NONE,DocValuesType.NONE,DocValuesSkipIndexType.NONE,-1,Map.of(),0,0,0,2,field.getVectorEncoding(),VectorSimilarityFunction.DOT_PRODUCT,false,false);
        FieldInfos wrong=new FieldInfos(field==FLOAT?new FieldInfo[]{changed,BYTE}:new FieldInfo[]{FLOAT,changed});String result;
        try(var ignored=format.fieldsReader(new SegmentReadState(dir,si,wrong,IOContext.DEFAULT,"s"))){result="ok";}
        catch(IllegalStateException error){result="metadata-error";}
        metadata.append(field.name).append("\tsimilarity\t").append(result).append('\n');
      }
      try(var r=wrap(format.fieldsReader(new SegmentReadState(dir,si,fis,IOContext.DEFAULT,"s")),fis)) {
        r.checkIntegrity();
        for(var fi:fis)for(String probe:new String[]{"normal","query-error","zero-k","cost-error"}) {
          var calls=new ArrayList<String>();var accept=new AcceptDocs(){
            public int cost()throws IOException{calls.add("cost");if(probe.equals("cost-error"))throw new IOException("cost failure");return 16;}
            public Bits bits(){calls.add("bits");return null;}
            public DocIdSetIterator iterator(){throw new AssertionError("native search must use bits");}
          };
          // TopKnnCollector's heap rejects capacity 0; a real collector decorator
          // supplies the supported reader-level k=0 boundary without that heap.
          KnnCollector collector=probe.equals("zero-k")?new KnnCollector.Decorator(new TopKnnCollector(1,100)){public int k(){return 0;}}:new TopKnnCollector(4,100);boolean shortTarget=probe.equals("query-error")||probe.equals("zero-k");String result;
          try {if(fi==FLOAT)r.search(fi.name,shortTarget?new float[]{0}:new float[]{0,1},collector,accept);else r.search(fi.name,shortTarget?new byte[]{0}:new byte[]{0,1},collector,accept);result="ok";}
          catch(IllegalArgumentException error){result="query-error";}
          catch(IOException error){result="io:"+error.getMessage();}
          order.append(fi.name).append('\t').append(probe).append('\t').append(result).append('\t').append(String.join(",",calls)).append('\n');
        }
        for(var fi:fis){var values=fi==FLOAT?r.getFloatVectorValues(fi.name):r.getByteVectorValues(fi.name);var it=values.iterator();
          for(int doc=it.nextDoc();doc!=DocIdSetIterator.NO_MORE_DOCS;doc=it.nextDoc()) {
            manifest.append("value\t").append(fi.name).append('\t').append(doc).append('\t').append(it.index());
            if(fi==FLOAT)for(float value:((FloatVectorValues)values).vectorValue(it.index()))manifest.append('\t').append(Float.floatToRawIntBits(value));
            else for(byte value:((ByteVectorValues)values).vectorValue(it.index()))manifest.append('\t').append(value);
            manifest.append('\n');
          }
          var graph=((HnswGraphProvider)r).getGraph(fi.name);manifest.append("graph\t").append(fi.name).append('\t').append(graph.size()).append('\t').append(graph.numLevels()).append('\t').append(graph.entryNode()).append('\n');
          for(int level=0;level<graph.numLevels();level++){var nodes=graph.getNodesOnLevel(level);while(nodes.hasNext()){int node=nodes.nextInt();manifest.append("neighbors\t").append(fi.name).append('\t').append(level).append('\t').append(node);graph.seek(level,node);for(int neighbor=graph.nextNeighbor();neighbor!=DocIdSetIterator.NO_MORE_DOCS;neighbor=graph.nextNeighbor())manifest.append('\t').append(neighbor);manifest.append('\n');}}
          for(int q:new int[]{0,5,15})for(boolean filtered:new boolean[]{false,true}) {
            Bits mask=filtered?new Bits(){public boolean get(int doc){return doc%3!=0;}public int length(){return 16;}}:null;
            var collector=new TopKnnCollector(4,100);
            if(fi==FLOAT)r.search(fi.name,new float[]{q*0.25f,16-q},collector,AcceptDocs.fromLiveDocs(mask,16));
            else r.search(fi.name,new byte[]{(byte)(q-8),(byte)(q*2)},collector,AcceptDocs.fromLiveDocs(mask,16));
            var hits=collector.topDocs();manifest.append("search\t").append(fi.name).append('\t').append(q).append('\t').append(filtered).append('\t').append(hits.totalHits.value()).append('\t').append(hits.totalHits.relation());
            for(var hit:hits.scoreDocs)manifest.append('\t').append(hit.doc).append(':').append(Float.floatToRawIntBits(hit.score));manifest.append('\n');
          }
        }
        var merged=r.getMergeInstance();merged.getFloatVectorValues("float");merged.getByteVectorValues("byte");merged.finishMerge();
      }
      for(String file:dir.listAll())try(var in=dir.openInput(file,IOContext.DEFAULT)){byte[] data=new byte[(int)in.length()];in.readBytes(data,0,data.length);Files.write(out.resolve("asserting_knn_vectors."+file.substring(file.lastIndexOf('.')+1)),data);}
      Files.writeString(out.resolve("asserting_knn_vectors_values.tsv"),manifest);
      Files.writeString(out.resolve("asserting_knn_vectors_native_order.tsv"),order);
      Files.writeString(out.resolve("asserting_knn_vectors_metadata.tsv"),metadata);
    }
  }
  public static void main(String[] args)throws Exception {
    if(!GenAssertingKnnVectors.class.desiredAssertionStatus())throw new AssertionError("run with -ea");
    Path out=Path.of(args[0]);Files.createDirectories(out);var text=new StringBuilder();
    for(String program:new String[]{
      "f,b,sf,sb,integrity,memory,graph,close,close", "f:byte,b:float,f:plain,f:missing,sf:byte,sb:float",
      "size:-1,f", "dim:0,b", "initial:0,f", "missing,f,b,merge", "fail:float:float,f", "fail:integrity,integrity",
      "merge,mf,mb,msf,msb,mm,mc,finish,mfinish,mfinish,close", "merge,merge2,mfinish,close,finish2,close",
      "merge,close", "merge,fail:finishMerge,mfinish,ok,mfinish,close", "merge,merge2,mfinish,fail:finishMerge,finish2,ok,close",
      "fail:merge,merge,close", "fail:close,close", "fail:close2,close", "initial:0,sf", "fail:bits,sf", "fail:cost,sb", "fail:acceptIterator,sf", "size:0,f,b", "dim:0,f", "initial:0,b"
    })readerTrace(program,text);
    for(String name:new String[]{"all","null-field","null-state","no-task","deferred-error","merge-order","merge-reversed","merge-reversed-field-error","merge-reversed-deferred-error","add:float","flush:7:true","mergeOne:float","finish","close"})writerTrace(name,text);
    for(String name:new String[]{"all","positioned","thread","bits","cost","acceptIterator"})acceptTrace(name,text);
    Files.writeString(out.resolve("asserting_knn_vectors.tsv"),text);real(out);
  }
}
