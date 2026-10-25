import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import org.apache.lucene.codecs.*;
import org.apache.lucene.codecs.lucene90.*;
import org.apache.lucene.index.*;
import org.apache.lucene.store.*;
import org.apache.lucene.util.*;

/** Actual package-private cranky wrappers; reflection constructs them only. */
public class GenCrankyPoints {
  static final byte[] ID={0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
  static FieldInfo field(String name,int number,int dims,int indexDims) {
    return new FieldInfo(name,number,false,false,false,IndexOptions.NONE,DocValuesType.NONE,
        DocValuesSkipIndexType.NONE,-1,Map.of(),dims,indexDims,2,0,VectorEncoding.FLOAT32,
        VectorSimilarityFunction.EUCLIDEAN,false,false);
  }
  static final FieldInfo FIELD=field("f",0,1,1);
  static PointsFormat wrap(PointsFormat delegate,Random random) throws Exception {
    var constructor=Class.forName("org.apache.lucene.tests.codecs.cranky.CrankyPointsFormat")
        .getDeclaredConstructor(PointsFormat.class,Random.class);
    constructor.setAccessible(true);return (PointsFormat)constructor.newInstance(delegate,random);
  }
  static class State {
    final List<String> events=new ArrayList<>();boolean fail,visitorFail;
    void event(String name) throws IOException {events.add(name);if(fail)throw new IOException("delegate failure");}
    void visit(String name) throws IOException {events.add(name);if(visitorFail)throw new IOException("visitor failure");}
  }
  static class RecordingTree implements PointValues.PointTree {
    final State state;int node,root;
    RecordingTree(State state) {this(state,1,1);}
    RecordingTree(State state,int node,int root) {this.state=state;this.node=node;this.root=root;}
    @Override public PointValues.PointTree clone() {state.events.add("clone");return new RecordingTree(state,node,node);}
    @Override public boolean moveToChild() throws IOException {state.event("child");if(node!=1)return false;node=2;return true;}
    @Override public boolean moveToSibling() throws IOException {state.event("sibling");if(node==root||node!=2)return false;node=3;return true;}
    @Override public boolean moveToParent() throws IOException {state.event("parent");if(node==root)return false;node=1;return true;}
    @Override public byte[] getMinPackedValue() {state.events.add("treeMin");return new byte[]{1,2};}
    @Override public byte[] getMaxPackedValue() {state.events.add("treeMax");return new byte[]{3,4};}
    @Override public long size() {state.events.add("treeSize");return node==1?3:1;}
    @Override public void visitDocIDs(PointValues.IntersectVisitor visitor) throws IOException {state.event("ids");visitor.grow(1);visitor.visit(node);}
    @Override public void visitDocValues(PointValues.IntersectVisitor visitor) throws IOException {state.event("values");visitor.grow(1);visitor.visit(node,new byte[]{1,2});}
  }
  static class RecordingValues extends PointValues {
    final State state;RecordingValues(State state) {this.state=state;}
    @Override public PointTree getPointTree() throws IOException {state.event("tree");return new RecordingTree(state);}
    @Override public byte[] getMinPackedValue() throws IOException {state.event("min");return new byte[]{1,2};}
    @Override public byte[] getMaxPackedValue() throws IOException {state.event("max");return new byte[]{3,4};}
    @Override public int getNumDimensions() throws IOException {state.event("dims");return 1;}
    @Override public int getNumIndexDimensions() throws IOException {state.event("indexDims");return 1;}
    @Override public int getBytesPerDimension() throws IOException {state.event("bytes");return 2;}
    @Override public long size() {state.events.add("size");return 3;}
    @Override public int getDocCount() {state.events.add("docCount");return 2;}
  }
  static class RecordingReader extends PointsReader {
    final State state;RecordingReader(State state) {this.state=state;}
    @Override public PointValues getValues(String field) throws IOException {state.event("get:"+field);return field.equals("missing")?null:new RecordingValues(state);}
    @Override public void checkIntegrity() throws IOException {state.event("integrity");}
    @Override public void close() throws IOException {state.event("readerClose");}
  }
  static class RecordingFormat extends PointsFormat {
    final State state;final RecordingReader reader;final PointsWriter writer;
    RecordingFormat(State state) {
      this.state=state;reader=new RecordingReader(state);
      writer=new PointsWriter() {
        @Override public void writeField(FieldInfo field,PointsReader source) throws IOException {state.event("write:"+(source==reader));}
        @Override public void finish() throws IOException {state.event("finish");}
        @Override public void merge(MergeState merge) throws IOException {state.event("merge");}
        @Override public void close() throws IOException {state.event("writerClose");}
      };
    }
    @Override public PointsWriter fieldsWriter(SegmentWriteState state) throws IOException {this.state.event("writer");return writer;}
    @Override public PointsReader fieldsReader(SegmentReadState state) throws IOException {this.state.event("reader");return reader;}
  }
  static String recording(long seed) throws Exception {
    State state=new State();Random random=new Random(seed);RecordingFormat delegate=new RecordingFormat(state);
    PointsFormat format=wrap(delegate,random);PointsWriter writer=format.fieldsWriter(null);PointsReader reader=format.fieldsReader(null);
    PointValues values=reader.getValues("f");PointValues.PointTree tree=values.getPointTree();
    PointValues.IntersectVisitor visitor=new PointValues.IntersectVisitor() {
      @Override public void visit(int doc) throws IOException {state.visit("doc:"+doc);}
      @Override public void visit(int doc,byte[] value) throws IOException {state.visit("value:"+doc+":"+HexFormat.of().formatHex(value));}
      @Override public void grow(int count) {state.events.add("grow:"+count);}
      @Override public PointValues.Relation compare(byte[] min,byte[] max) {return PointValues.Relation.CELL_CROSSES_QUERY;}
    };
    StringBuilder out=new StringBuilder();
    for(int step=0;step<2800;step++) {
      int op=step%28;state.events.clear();state.fail=step%17==0;state.visitorFail=step%19==0;String status="ok";
      try {
        switch(op) {
          case 0->format.fieldsWriter(null);case 1->format.fieldsReader(null);
          case 2->writer.writeField(FIELD,delegate.reader);case 3->writer.finish();case 4->writer.merge(null);case 5->writer.close();
          case 6->reader.checkIntegrity();case 7->reader.close();
          case 8->status=Boolean.toString(reader.getValues("missing")==null);case 9->status=Boolean.toString(reader.getValues("f")!=null);
          case 10->status=HexFormat.of().formatHex(values.getMinPackedValue());case 11->status=HexFormat.of().formatHex(values.getMaxPackedValue());
          case 12->status=Integer.toString(values.getNumDimensions());case 13->status=Integer.toString(values.getNumIndexDimensions());
          case 14->status=Integer.toString(values.getBytesPerDimension());case 15->status=Long.toString(values.size());case 16->status=Integer.toString(values.getDocCount());
          case 17->values.getPointTree();case 18->status=Boolean.toString(tree.moveToChild());case 19->status=Boolean.toString(tree.moveToSibling());case 20->status=Boolean.toString(tree.moveToParent());
          case 21->status=HexFormat.of().formatHex(tree.getMinPackedValue());case 22->status=HexFormat.of().formatHex(tree.getMaxPackedValue());case 23->status=Long.toString(tree.size());
          case 24->tree.visitDocIDs(visitor);case 25->tree.visitDocValues(visitor);case 26->tree.clone().visitDocIDs(visitor);case 27->tree.clone().visitDocValues(visitor);
        }
      } catch(IOException e) {status=e.getMessage();}
      out.append(seed).append('\t').append(step).append('\t').append(op).append('\t').append(status).append('\t')
          .append(String.join(",",state.events)).append('\t').append(random.nextInt()).append('\n');
    }
    return out.toString();
  }
  static byte[] packed(int point,int dims) {
    byte[] value=new byte[dims*2];for(int dim=0;dim<dims;dim++) {int number=(point*37+dim*101)&65535;value[dim*2]=(byte)(number>>>8);value[dim*2+1]=(byte)number;}return value;
  }
  static class InputValues extends PointValues {
    final int dims;InputValues(int dims) {this.dims=dims;}
    @Override public PointTree getPointTree() {
      return new PointTree() {
        @Override public PointTree clone() {return getPointTree();}
        @Override public boolean moveToChild() {return false;}
        @Override public boolean moveToSibling() {return false;}
        @Override public boolean moveToParent() {return false;}
        @Override public byte[] getMinPackedValue() {return packed(0,dims);}
        @Override public byte[] getMaxPackedValue() {return packed(12,dims);}
        @Override public long size() {return 13;}
        @Override public void visitDocIDs(IntersectVisitor visitor) throws IOException {for(int point=0;point<13;point++)visitor.visit(point%10);}
        @Override public void visitDocValues(IntersectVisitor visitor) throws IOException {for(int point=0;point<13;point++)visitor.visit(point%10,packed(point,dims));}
      };
    }
    @Override public byte[] getMinPackedValue() {return packed(0,dims);}
    @Override public byte[] getMaxPackedValue() {return packed(12,dims);}
    @Override public int getNumDimensions() {return dims;}
    @Override public int getNumIndexDimensions() {return dims;}
    @Override public int getBytesPerDimension() {return 2;}
    @Override public long size() {return 13;}
    @Override public int getDocCount() {return 10;}
  }
  static SegmentInfo segment(Directory directory) {
    return new SegmentInfo(directory,Version.fromBits(10,5,0),null,"_cp",10,false,false,null,Map.of(),ID.clone(),Map.of(),null);
  }
  static MergeState mergeState(Directory directory,FieldInfos fields,PointsReader reader) throws Exception {
    var constructor=MergeState.class.getDeclaredConstructor(List.class,SegmentInfo.class,InfoStream.class,
        java.util.concurrent.Executor.class,MergePolicy.OneMerge.class);
    constructor.setAccessible(true);
    SegmentInfo dummy=new SegmentInfo(directory,Version.fromBits(10,5,0),null,"_unused",-1,false,false,null,Map.of(),ID.clone(),Map.of(),null);
    MergeState state=constructor.newInstance(List.of(),dummy,InfoStream.NO_OUTPUT,null,null);
    int[][] maps={{0,-1,1,2,3,4,5,6,7,8},{9,-1,10,11,12,13,14,15,16,17}};
    Map<String,Object> members=new HashMap<>();
    members.put("pointsReaders",new PointsReader[]{reader,reader});
    members.put("fieldInfos",new FieldInfos[]{fields,fields});
    members.put("maxDocs",new int[]{10,10});
    members.put("docMaps",new MergeState.DocMap[]{new MergeState.DocMap(){public int get(int doc){return maps[0][doc];}},
        new MergeState.DocMap(){public int get(int doc){return maps[1][doc];}}});
    for(var member:members.entrySet()) {var field=MergeState.class.getField(member.getKey());field.setAccessible(true);field.set(state,member.getValue());}
    state.mergeFieldInfos=fields;return state;
  }
  static void realMerges(Path out,PointsFormat nativeFormat,Directory source,SegmentInfo sourceInfo,FieldInfos fields) throws Exception {
    for(boolean wrapped:new boolean[]{false,true})try(var directory=new ByteBuffersDirectory()) {
      PointsFormat sourceFormat=wrapped?wrap(nativeFormat,new Random(42)):nativeFormat;
      try(var reader=sourceFormat.fieldsReader(new SegmentReadState(source,sourceInfo,fields,IOContext.DEFAULT,"s"))) {
        SegmentInfo info=new SegmentInfo(directory,Version.fromBits(10,5,0),null,"_cp",18,false,false,null,Map.of(),ID.clone(),Map.of(),null);
        try(var writer=nativeFormat.fieldsWriter(new SegmentWriteState(InfoStream.NO_OUTPUT,directory,info,fields,null,IOContext.DEFAULT,"s"))) {
          writer.merge(mergeState(directory,fields,reader));
        }
        try(var merged=nativeFormat.fieldsReader(new SegmentReadState(directory,info,fields,IOContext.DEFAULT,"s"))) {
          merged.checkIntegrity();
          for(FieldInfo field:fields) {
            PointValues values=merged.getValues(field.name);if(values.size()!=22||values.getDocCount()!=18)throw new AssertionError("merged counts");
            List<String> actual=new ArrayList<>();values.getPointTree().visitDocValues(new PointValues.IntersectVisitor() {
              @Override public void visit(int doc) {throw new AssertionError("missing packed value");}
              @Override public void visit(int doc,byte[] value) {actual.add(doc+":"+HexFormat.of().formatHex(value));}
              @Override public PointValues.Relation compare(byte[] min,byte[] max) {return PointValues.Relation.CELL_CROSSES_QUERY;}
            });
            List<String> expected=new ArrayList<>();for(int offset:new int[]{0,9})for(int point=0;point<13;point++) {
              int old=point%10;if(old==1)continue;int doc=offset+(old==0?0:old-1);expected.add(doc+":"+HexFormat.of().formatHex(packed(point,field.getPointDimensionCount())));
            }
            Collections.sort(actual);Collections.sort(expected);if(!actual.equals(expected))throw new AssertionError("merged points");
          }
        }
        for(String extension:new String[]{"kdm","kdi","kdd"})try(var input=directory.openInput("_cp_s."+extension,IOContext.DEFAULT)) {
          byte[] bytes=new byte[Math.toIntExact(input.length())];input.readBytes(bytes,0,bytes.length);
          Files.write(out.resolve("cranky_points_merge_"+(wrapped?"wrapped":"native")+"."+extension),bytes);
        }
      }
    }
  }
  static void real(Path out) throws Exception {
    FieldInfos fields=new FieldInfos(new FieldInfo[]{field("one",0,1,1),field("two",1,2,2)});
    PointsFormat nativeFormat=new PointsFormat() {
      @Override public PointsWriter fieldsWriter(SegmentWriteState state) throws IOException {return new Lucene90PointsWriter(state,4,16.0);}
      @Override public PointsReader fieldsReader(SegmentReadState state) throws IOException {return new Lucene90PointsReader(state);}
    };
    try(var directory=new ByteBuffersDirectory()) {
      SegmentInfo info=segment(directory);Random random=new Random(42);PointsFormat format=wrap(nativeFormat,random);
      PointsWriter writer=format.fieldsWriter(new SegmentWriteState(InfoStream.NO_OUTPUT,directory,info,fields,null,IOContext.DEFAULT,"s"));
      PointsReader source=new PointsReader() {
        @Override public PointValues getValues(String field) {return new InputValues(field.equals("one")?1:2);}
        @Override public void checkIntegrity() {}
        @Override public void close() {}
      };
      for(FieldInfo field:fields)writer.writeField(field,source);
      writer.finish();writer.close();
      StringBuilder manifest=new StringBuilder();
      try(var reader=nativeFormat.fieldsReader(new SegmentReadState(directory,info,fields,IOContext.DEFAULT,"s"))) {
        reader.checkIntegrity();
        for(FieldInfo field:fields) {
          PointValues values=reader.getValues(field.name);
          manifest.append("field\t").append(field.number).append('\t').append(values.size()).append('\t').append(values.getDocCount()).append('\t')
              .append(HexFormat.of().formatHex(values.getMinPackedValue())).append('\t').append(HexFormat.of().formatHex(values.getMaxPackedValue())).append('\n');
          List<String> actual=new ArrayList<>();values.getPointTree().visitDocValues(new PointValues.IntersectVisitor() {
            @Override public void visit(int doc) {throw new AssertionError("missing packed value");}
            @Override public void visit(int doc,byte[] value) {actual.add(doc+":"+HexFormat.of().formatHex(value));}
            @Override public PointValues.Relation compare(byte[] min,byte[] max) {return PointValues.Relation.CELL_CROSSES_QUERY;}
          });
          List<String> expected=new ArrayList<>();for(int point=0;point<13;point++)expected.add((point%10)+":"+HexFormat.of().formatHex(packed(point,field.getPointDimensionCount())));
          Collections.sort(actual);Collections.sort(expected);if(!actual.equals(expected))throw new AssertionError("point roundtrip");
          for(String point:actual)manifest.append("point\t").append(field.number).append('\t').append(point).append('\n');
        }
      }
      for(String extension:new String[]{"kdm","kdi","kdd"})try(var input=directory.openInput("_cp_s."+extension,IOContext.DEFAULT)) {
        byte[] bytes=new byte[Math.toIntExact(input.length())];input.readBytes(bytes,0,bytes.length);Files.write(out.resolve("cranky_points."+extension),bytes);
      }
      Files.writeString(out.resolve("cranky_points_real.tsv"),manifest+"rng\t"+random.nextInt()+"\n");
      realMerges(out,nativeFormat,directory,info,fields);
    }
  }
  public static void main(String[] args) throws Exception {
    Path out=Path.of(args[0]);Files.createDirectories(out);StringBuilder trace=new StringBuilder();
    for(long seed:new long[]{0,1,42,-1,Long.MIN_VALUE,Long.MAX_VALUE})trace.append(recording(seed));
    Files.writeString(out.resolve("cranky_points.tsv"),trace);real(out);
  }
}
