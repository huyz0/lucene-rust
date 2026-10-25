import java.io.IOException;
import java.lang.reflect.Constructor;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.atomic.AtomicReference;
import org.apache.lucene.codecs.*;
import org.apache.lucene.codecs.lucene90.*;
import org.apache.lucene.index.*;
import org.apache.lucene.store.*;
import org.apache.lucene.tests.codecs.asserting.AssertingPointsFormat;
import org.apache.lucene.util.*;

/** Actual Lucene assertions, including the nested point visitor, with -ea. */
public class GenAssertingPoints {
  static final byte[] ID=GenCrankyPoints.ID;
  static final FieldInfo FIELD=GenCrankyPoints.field("f",0,1,1);
  static final FieldInfos FIELDS=new FieldInfos(new FieldInfo[]{FIELD});
  @FunctionalInterface interface Action {void run()throws Exception;}
  static String outcome(Action a)throws Exception {try{a.run();return "ok";}catch(AssertionError e){return "assert";}catch(IOException e){return "io:"+e.getMessage();}catch(IllegalArgumentException e){return "illegal:"+e.getMessage();}}
  static PointValues.IntersectVisitor assertingVisitor(int dims,int indexDims,int width,PointValues.IntersectVisitor delegate)throws Exception {
    var c=Class.forName("org.apache.lucene.tests.index.AssertingLeafReader$AssertingIntersectVisitor").getDeclaredConstructor(int.class,int.class,int.class,PointValues.IntersectVisitor.class);
    c.setAccessible(true);return (PointValues.IntersectVisitor)c.newInstance(dims,indexDims,width,delegate);
  }
  static class Visitor implements PointValues.IntersectVisitor {
    final List<String> events;PointValues.Relation relation=PointValues.Relation.CELL_CROSSES_QUERY;boolean fail;
    Visitor(List<String> events){this.events=events;}
    @Override public void grow(int n){events.add("grow:"+n);}
    @Override public void visit(int doc)throws IOException {events.add("doc:"+doc);if(fail)throw new IOException("visitor failure");}
    @Override public void visit(int doc,byte[] value)throws IOException {events.add("value:"+doc+":"+HexFormat.of().formatHex(value));if(fail)throw new IOException("visitor failure");}
    @Override public PointValues.Relation compare(byte[] min,byte[] max){events.add("compare:"+HexFormat.of().formatHex(min)+":"+HexFormat.of().formatHex(max));return relation;}
  }
  static void visitorTrace(String program,StringBuilder out)throws Exception {
    var events=new ArrayList<String>();var delegate=new Visitor(events);var visitor=assertingVisitor(1,1,1,delegate);var results=new ArrayList<String>();
    for(String token:program.split(","))results.add(outcome(()->{String[] args=token.split(":");switch(args[0]){
      case "g"->visitor.grow(Integer.parseInt(args[1]));case "d"->visitor.visit(Integer.parseInt(args[1]));
      case "v"->visitor.visit(Integer.parseInt(args[1]),HexFormat.of().parseHex(args[2]));
      case "c"->visitor.compare(HexFormat.of().parseHex(args[1]),HexFormat.of().parseHex(args[2]));
      case "in"->delegate.relation=PointValues.Relation.CELL_INSIDE_QUERY;case "out"->delegate.relation=PointValues.Relation.CELL_OUTSIDE_QUERY;
      case "cross"->delegate.relation=PointValues.Relation.CELL_CROSSES_QUERY;case "fail"->delegate.fail=true;case "ok"->delegate.fail=false;
      default->throw new AssertionError(token);
    }}));
    out.append("visitor\t").append(program).append('\t').append(String.join(";",results)).append('\t').append(String.join(",",events)).append('\n');
  }
  static class State {final List<String> events=new ArrayList<>();String fail="";long size=3,treeSize=3;int docs=2;boolean missing;
    void call(String event)throws IOException{events.add(event);if(event.equals(fail))throw new IOException("delegate failure");}}
  static class Values extends PointValues {
    final State s;Values(State s){this.s=s;}
    @Override public long size(){s.events.add("size");return s.size;}
    @Override public int getDocCount(){s.events.add("docs");return s.docs;}
    @Override public int getNumDimensions()throws IOException{s.call("dims");return 1;}
    @Override public int getNumIndexDimensions()throws IOException{s.call("index");return 1;}
    @Override public int getBytesPerDimension()throws IOException{s.call("bytes");return 1;}
    @Override public byte[] getMinPackedValue()throws IOException{s.call("min");return new byte[]{0};}
    @Override public byte[] getMaxPackedValue()throws IOException{s.call("max");return new byte[]{(byte)255};}
    @Override public PointTree getPointTree()throws IOException{s.call("tree");return new Tree(s);}
  }
  static class Tree implements PointValues.PointTree {
    final State s;Tree(State s){this.s=s;}
    @Override public PointValues.PointTree clone(){s.events.add("clone");return new Tree(s);}
    @Override public boolean moveToChild()throws IOException{s.call("child");return false;}
    @Override public boolean moveToSibling()throws IOException{s.call("sibling");return false;}
    @Override public boolean moveToParent()throws IOException{s.call("parent");return false;}
    @Override public byte[] getMinPackedValue(){s.events.add("treeMin");return new byte[]{0};}
    @Override public byte[] getMaxPackedValue(){s.events.add("treeMax");return new byte[]{(byte)255};}
    @Override public long size(){s.events.add("treeSize");return s.treeSize;}
    @Override public void visitDocIDs(PointValues.IntersectVisitor v)throws IOException{s.call("ids");v.grow(1);v.visit(1);}
    @Override public void visitDocValues(PointValues.IntersectVisitor v)throws IOException{s.call("values");v.grow(2);v.visit(0,new byte[]{0});v.visit(1,new byte[]{(byte)255});}
  }
  static class Reader extends PointsReader {
    final State s;Reader(State s){this.s=s;}
    @Override public PointValues getValues(String f)throws IOException{s.call("get:"+f);return s.missing?null:new Values(s);}
    @Override public void checkIntegrity()throws IOException{s.call("integrity");}
    @Override public void close()throws IOException{s.call("close");}
    @Override public PointsReader getMergeInstance(){s.events.add("merge");return new Reader(s);}
    @Override public String toString(){return "ProbeReader";}
  }
  static PointsReader reader(State s,boolean merging)throws Exception {
    var c=Class.forName("org.apache.lucene.tests.codecs.asserting.AssertingPointsFormat$AssertingPointsReader").getDeclaredConstructor(int.class,PointsReader.class,FieldInfos.class,boolean.class);
    c.setAccessible(true);return (PointsReader)c.newInstance(2,new Reader(s),FIELDS,merging);
  }
  static void readerTrace(String name,StringBuilder out)throws Exception {
    State s=new State();PointsReader r=reader(s,false);String status=outcome(()->{switch(name){
      case "missing-field"->r.getValues("missing");case "null"->{s.missing=true;if(r.getValues("f")!=null)throw new AssertionError();}
      case "size0"->{s.size=0;r.getValues("f");}case "docs0"->{s.docs=0;r.getValues("f");}case "docs-size"->{s.docs=4;r.getValues("f");}case "docs-max"->{s.docs=3;r.getValues("f");}
      case "all"->{var v=r.getValues("f");v.getMinPackedValue();v.getMaxPackedValue();v.getNumDimensions();v.getNumIndexDimensions();v.getBytesPerDimension();v.size();v.getDocCount();var t=v.getPointTree();t.moveToChild();t.moveToSibling();t.moveToParent();t.getMinPackedValue();t.getMaxPackedValue();t.size();var visitor=new Visitor(s.events);t.visitDocIDs(visitor);t.visitDocValues(visitor);t.clone().visitDocValues(visitor);r.checkIntegrity();r.close();}
      case "tree0"->{s.treeSize=0;r.getValues("f").getPointTree().size();}
      case "tree-error"->{s.fail="tree";r.getValues("f").getPointTree();}case "dims-error"->{s.fail="dims";r.getValues("f").getPointTree().visitDocIDs(new Visitor(s.events));}
      case "close-error"->{s.fail="close";r.close();}case "integrity-error"->{s.fail="integrity";r.checkIntegrity();}
      case "merge"->{r.getMergeInstance().getValues("f");r.close();}
      case "thread","merge-thread","values-thread"->{PointValues values=name.equals("values-thread")?r.getValues("f"):null;PointsReader target=name.equals("merge-thread")?r.getMergeInstance():r;var result=new AtomicReference<String>();var t=new Thread(()->{try{result.set(outcome(()->{if(values!=null)values.size();else target.getValues("f");}));}catch(Exception e){throw new RuntimeException(e);}});t.start();t.join();if(result.get().equals("assert"))throw new AssertionError();if(!result.get().equals("ok"))throw new IOException(result.get());}
      case "display"->{if(!r.toString().equals("AssertingPointsReader(ProbeReader)"))throw new AssertionError();}
      default->throw new AssertionError(name);
    }});out.append("reader\t").append(name).append('\t').append(status).append('\t').append(String.join(",",s.events)).append('\n');
  }
  static void writerTrace(String name,StringBuilder out)throws Exception {
    State s=new State();var delegate=new PointsWriter(){
      @Override public void writeField(FieldInfo f,PointsReader r)throws IOException{s.call("write");}
      @Override public void finish()throws IOException{s.call("finish");}
      @Override public void merge(MergeState state)throws IOException{s.call("merge");}
      @Override public void close()throws IOException{s.call("close");}};
    var c=Class.forName("org.apache.lucene.tests.codecs.asserting.AssertingPointsFormat$AssertingPointsWriter").getDeclaredConstructor(SegmentWriteState.class,PointsWriter.class);c.setAccessible(true);var w=(PointsWriter)c.newInstance(null,delegate);
    String status=outcome(()->{switch(name){case "all"->{w.writeField(FIELD,new Reader(s));w.merge(null);w.finish();w.close();}case "zero"->w.writeField(new FieldInfo("f",0,false,false,false,IndexOptions.NONE,DocValuesType.NONE,DocValuesSkipIndexType.NONE,-1,Map.of(),0,0,0,0,VectorEncoding.FLOAT32,VectorSimilarityFunction.EUCLIDEAN,false,false),new Reader(s));case "write-error"->{s.fail="write";w.writeField(FIELD,new Reader(s));}case "close-error"->{s.fail="close";w.close();}default->throw new AssertionError(name);}});
    out.append("writer\t").append(name).append('\t').append(status).append('\t').append(String.join(",",s.events)).append('\n');
  }
  static void real(Path out)throws Exception {
    var fis=new FieldInfos(new FieldInfo[]{GenCrankyPoints.field("one",0,1,1),GenCrankyPoints.field("two",1,2,2)});
    PointsFormat nativeFormat=new PointsFormat(){@Override public PointsWriter fieldsWriter(SegmentWriteState s)throws IOException{return new Lucene90PointsWriter(s,4,16.0);}@Override public PointsReader fieldsReader(SegmentReadState s)throws IOException{return new Lucene90PointsReader(s);}};
    var format=new AssertingPointsFormat(nativeFormat);
    try(var dir=new ByteBuffersDirectory()) {
      var si=GenCrankyPoints.segment(dir);try(var w=format.fieldsWriter(new SegmentWriteState(InfoStream.NO_OUTPUT,dir,si,fis,null,IOContext.DEFAULT,"s"))){var source=new PointsReader(){@Override public PointValues getValues(String f){return new GenCrankyPoints.InputValues(f.equals("one")?1:2);}@Override public void checkIntegrity(){}@Override public void close(){}};for(var f:fis)w.writeField(f,source);w.finish();}
      StringBuilder manifest=new StringBuilder();try(var r=format.fieldsReader(new SegmentReadState(dir,si,fis,IOContext.DEFAULT,"s"))){r.checkIntegrity();for(var f:fis){var v=r.getValues(f.name);var points=new ArrayList<String>();v.getPointTree().visitDocValues(new PointValues.IntersectVisitor(){@Override public void visit(int doc){throw new AssertionError();}@Override public void visit(int doc,byte[] value){points.add(doc+":"+HexFormat.of().formatHex(value));}@Override public PointValues.Relation compare(byte[] min,byte[] max){return PointValues.Relation.CELL_CROSSES_QUERY;}});Collections.sort(points);var expected=new ArrayList<String>();for(int p=0;p<13;p++)expected.add((p%10)+":"+HexFormat.of().formatHex(GenCrankyPoints.packed(p,f.getPointDimensionCount())));Collections.sort(expected);if(!points.equals(expected))throw new AssertionError("real points");for(String p:points)manifest.append(f.number).append('\t').append(p).append('\n');}
        try(var mergedDir=new ByteBuffersDirectory()){var mi=new SegmentInfo(mergedDir,Version.fromBits(10,5,0),null,"_cp",18,false,false,null,Map.of(),ID,Map.of(),null);try(var w=format.fieldsWriter(new SegmentWriteState(InfoStream.NO_OUTPUT,mergedDir,mi,fis,null,IOContext.DEFAULT,"s"))){w.merge(GenCrankyPoints.mergeState(mergedDir,fis,r));}save(out,mergedDir,"asserting_points_merge");try(var mr=format.fieldsReader(new SegmentReadState(mergedDir,mi,fis,IOContext.DEFAULT,"s"))){mr.checkIntegrity();for(var f:fis)if(mr.getValues(f.name).size()!=22)throw new AssertionError();}}
      }save(out,dir,"asserting_points");Files.writeString(out.resolve("asserting_points_real.tsv"),manifest);
    }
  }
  static void nativeLifecycle(Path out)throws Exception {
    StringBuilder trace=new StringBuilder();
    var fis=new FieldInfos(new FieldInfo[]{GenCrankyPoints.field("one",0,1,1),GenCrankyPoints.field("two",1,2,2)});
    PointsFormat nativeFormat=new PointsFormat(){@Override public PointsWriter fieldsWriter(SegmentWriteState s)throws IOException{return new Lucene90PointsWriter(s,4,16.0);}@Override public PointsReader fieldsReader(SegmentReadState s)throws IOException{return new Lucene90PointsReader(s);}};
    try(var directory=new ByteBuffersDirectory()) {
      for(String ext:new String[]{"kdm","kdi","kdd"}){byte[] b=Files.readAllBytes(out.resolve("asserting_points."+ext));try(var output=directory.createOutput("_cp_s."+ext,IOContext.DEFAULT)){output.writeBytes(b,b.length);}}
      for(boolean closeMerge:new boolean[]{false,true}) {
        var state=new SegmentReadState(directory,GenCrankyPoints.segment(directory),fis,IOContext.DEFAULT,"s");
        var raw=nativeFormat.fieldsReader(state);if(raw!=raw.getMergeInstance())throw new AssertionError("native merge identity");raw.close();
        var r=new AssertingPointsFormat(nativeFormat).fieldsReader(state);var m=r.getMergeInstance();var values=r.getValues("one");var tree=values.getPointTree();
        if(closeMerge)m.close();else r.close();
        if(r.getValues("one")!=null||m.getValues("one")!=null)throw new AssertionError("closed aliases return values");
        if(values.size()!=13)throw new AssertionError("acquired metadata lost");
        int[] count={0};tree.visitDocValues(new PointValues.IntersectVisitor(){@Override public void visit(int doc){throw new AssertionError();}@Override public void visit(int doc,byte[] value){count[0]++;}@Override public PointValues.Relation compare(byte[] min,byte[] max){return PointValues.Relation.CELL_CROSSES_QUERY;}});
        if(count[0]!=13)throw new AssertionError("acquired tree lost");
        String newTree="unexpected",integrity="unexpected";try{values.getPointTree();}catch(AlreadyClosedException expected){newTree="closed";}try{r.checkIntegrity();}catch(AlreadyClosedException expected){integrity="closed";}
        if(!newTree.equals("closed")||!integrity.equals("closed")||r.getMergeInstance().getValues("one")!=null)throw new AssertionError("closed acquisition");
        trace.append(closeMerge?"merge":"original").append("\tnull\t13\t13\t").append(newTree).append('\t').append(integrity).append("\tnull\n");r.close();m.close();
      }
    }Files.writeString(out.resolve("asserting_points_native_lifecycle.tsv"),trace);
  }  static void save(Path out,Directory dir,String stem)throws IOException {for(String ext:new String[]{"kdm","kdi","kdd"})try(var in=dir.openInput("_cp_s."+ext,IOContext.DEFAULT)){byte[] b=new byte[Math.toIntExact(in.length())];in.readBytes(b,0,b.length);Files.write(out.resolve(stem+"."+ext),b);}}
  public static void main(String[] args)throws Exception {
    if(!GenAssertingPoints.class.desiredAssertionStatus())throw new AssertionError("run with -ea");Path out=Path.of(args[0]);Files.createDirectories(out);StringBuilder trace=new StringBuilder();
    for(String p:new String[]{"g:2,v:0:00,v:1:ff","v:0:00","g:1,d:0,d:1","g:1,c:00:ff,v:0:00","g:1,in,c:00:ff,d:0","g:1,in,c:00:ff,v:0:00","g:1,c:00:ff,d:0","g:1,out,c:00:ff,d:0","g:1,c:20:10,v:0:20","g:1,c:20:30,v:0:10","g:1,c:20:30,v:0:40","g:2,v:0:80,v:1:7f","g:2,v:1:80,v:0:80","g:2,v:0:80,v:1:80","g:1,v:0:0001","g:2,c:10:20,v:0:15,v:1:30","g:1,fail,v:0:10,ok,v:1:20","g:1,d:0,g:1,d:1"})visitorTrace(p,trace);
    for(String name:new String[]{"missing-field","null","size0","docs0","docs-size","docs-max","all","tree0","tree-error","dims-error","close-error","integrity-error","merge","thread","merge-thread","values-thread","display"})readerTrace(name,trace);
    for(String name:new String[]{"all","zero","write-error","close-error"})writerTrace(name,trace);
    Files.writeString(out.resolve("asserting_points.tsv"),trace);real(out);nativeLifecycle(out);
  }
}


