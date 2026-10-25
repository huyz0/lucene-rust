import java.io.IOException;
import java.nio.file.*;
import java.util.*;
import org.apache.lucene.codecs.*;
import org.apache.lucene.codecs.lucene90.Lucene90DocValuesFormat;
import org.apache.lucene.codecs.lucene104.Lucene104Codec;
import org.apache.lucene.document.*;
import org.apache.lucene.index.*;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.store.*;
import org.apache.lucene.util.*;

/** Actual cranky wrapper and inherited five-type merge, never hand-written codec bytes. */
public class GenCrankyDocValues {
  static final byte[] ID={0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
  static final int DOCS=12;
  static final String[] EXTENSIONS={"dvm","dvd","dvs"};
  static final DocValuesFormat NATIVE=new Lucene90DocValuesFormat(4);
  static DocValuesFormat wrap(DocValuesFormat delegate,Random random) throws Exception {
    var constructor=Class.forName("org.apache.lucene.tests.codecs.cranky.CrankyDocValuesFormat").getDeclaredConstructor(DocValuesFormat.class,Random.class);
    constructor.setAccessible(true);return (DocValuesFormat)constructor.newInstance(delegate,random);
  }
  static SegmentInfo segment(Directory directory,int maxDoc){return new SegmentInfo(directory,Version.fromBits(10,5,0),null,"_cdv",maxDoc,false,false,new Lucene104Codec(),Map.of(),ID,Map.of(),null);}
  static SegmentWriteState write(Directory directory,FieldInfos fields,int maxDoc){return new SegmentWriteState(InfoStream.NO_OUTPUT,directory,segment(directory,maxDoc),fields,null,IOContext.DEFAULT,"s");}
  static SegmentReadState read(Directory directory,FieldInfos fields,int maxDoc){return new SegmentReadState(directory,segment(directory,maxDoc),fields,IOContext.DEFAULT,"s");}
  static FieldInfo field(String name,int number,DocValuesType type){return new FieldInfo(name,number,false,true,false,IndexOptions.NONE,type,DocValuesSkipIndexType.NONE,-1,Map.of(),0,0,0,0,VectorEncoding.FLOAT32,VectorSimilarityFunction.EUCLIDEAN,false,false);}
  static FieldInfos traceFields(){return new FieldInfos(new FieldInfo[]{field("n",0,DocValuesType.NUMERIC),field("b",1,DocValuesType.BINARY),field("s",2,DocValuesType.SORTED),field("sn",3,DocValuesType.SORTED_NUMERIC),field("ss",4,DocValuesType.SORTED_SET)});}
  static MergeState merge(Directory directory,FieldInfos fields,DocValuesProducer[] producers,FieldInfos[] sourceFields,int[] maxDocs,int[][] maps,Bits[] liveDocs,boolean sorted)throws Exception {
    var constructor=MergeState.class.getDeclaredConstructor(List.class,SegmentInfo.class,InfoStream.class,java.util.concurrent.Executor.class,MergePolicy.OneMerge.class);constructor.setAccessible(true);
    MergeState state=constructor.newInstance(List.of(),segment(directory,-1),InfoStream.NO_OUTPUT,null,null);
    MergeState.DocMap[] docMaps=new MergeState.DocMap[maps.length];for(int i=0;i<maps.length;i++){final int[] map=maps[i];docMaps[i]=new MergeState.DocMap(){@Override public int get(int doc){return map[doc];}};}
    Map<String,Object> members=new HashMap<>();members.put("docValuesProducers",producers);members.put("fieldInfos",sourceFields);members.put("maxDocs",maxDocs);members.put("docMaps",docMaps);members.put("liveDocs",liveDocs);members.put("needsIndexSort",sorted);
    for(var member:members.entrySet()){var f=MergeState.class.getField(member.getKey());f.setAccessible(true);f.set(state,member.getValue());}state.mergeFieldInfos=fields;return state;
  }
  static void add(DocValuesConsumer consumer,FieldInfo field,DocValuesProducer producer)throws IOException {
    switch(field.getDocValuesType()){
      case NUMERIC->consumer.addNumericField(field,producer);case BINARY->consumer.addBinaryField(field,producer);case SORTED->consumer.addSortedField(field,producer);
      case SORTED_NUMERIC->consumer.addSortedNumericField(field,producer);case SORTED_SET->consumer.addSortedSetField(field,producer);default->throw new AssertionError();
    }
  }
  static class State {final List<String> events=new ArrayList<>();boolean fail;void event(String value)throws IOException{events.add(value);if(fail)throw new IOException("delegate failure");}}
  static class RecordingProducer extends DocValuesProducer {
    final State state;RecordingProducer(State state){this.state=state;}
    @Override public NumericDocValues getNumeric(FieldInfo field)throws IOException{state.event("getNumeric:"+field.name);return DocValues.emptyNumeric();}
    @Override public BinaryDocValues getBinary(FieldInfo field)throws IOException{state.event("getBinary:"+field.name);return DocValues.emptyBinary();}
    @Override public SortedDocValues getSorted(FieldInfo field)throws IOException{state.event("getSorted:"+field.name);return DocValues.emptySorted();}
    @Override public SortedNumericDocValues getSortedNumeric(FieldInfo field)throws IOException{state.event("getSortedNumeric:"+field.name);return DocValues.emptySortedNumeric();}
    @Override public SortedSetDocValues getSortedSet(FieldInfo field)throws IOException{state.event("getSortedSet:"+field.name);return DocValues.emptySortedSet();}
    @Override public DocValuesSkipper getSkipper(FieldInfo field)throws IOException{state.event("getSkipper:"+field.name);return null;}
    @Override public void checkIntegrity()throws IOException{state.event("integrity");}
    @Override public void close()throws IOException{state.event("producerClose");}
  }
  static class RecordingFormat extends DocValuesFormat {
    final State state;final FieldInfos fields=traceFields();final RecordingProducer producer;
    RecordingFormat(State state){super("RecordingDocValues");this.state=state;producer=new RecordingProducer(state);}
    @Override public DocValuesProducer fieldsProducer(SegmentReadState ignored)throws IOException{state.event("producer");return producer;}
    @Override public DocValuesConsumer fieldsConsumer(SegmentWriteState ignored)throws IOException{
      state.event("consumer");return new DocValuesConsumer(){
        void record(String type,FieldInfo field,DocValuesProducer values)throws IOException{state.event(type+":"+(field==fields.fieldInfo(field.number))+":"+(values==producer));}
        @Override public void addNumericField(FieldInfo f,DocValuesProducer p)throws IOException{record("numeric",f,p);}
        @Override public void addBinaryField(FieldInfo f,DocValuesProducer p)throws IOException{record("binary",f,p);}
        @Override public void addSortedField(FieldInfo f,DocValuesProducer p)throws IOException{record("sorted",f,p);}
        @Override public void addSortedNumericField(FieldInfo f,DocValuesProducer p)throws IOException{record("sortedNumeric",f,p);}
        @Override public void addSortedSetField(FieldInfo f,DocValuesProducer p)throws IOException{record("sortedSet",f,p);}
        @Override public void close()throws IOException{state.event("consumerClose");}
        @Override public void merge(MergeState ignored){throw new AssertionError("wrapper must inherit superclass merge");}
      };
    }
  }
  static String trace()throws Exception{
    StringBuilder out=new StringBuilder();
    for(long seed:new long[]{0,1,42,-1,Long.MIN_VALUE,Long.MAX_VALUE})try(var directory=new ByteBuffersDirectory()){
      State state=new State();RecordingFormat delegate=new RecordingFormat(state);Random random=new Random(seed);DocValuesFormat format=wrap(delegate,random);DocValuesConsumer consumer=null;
      while(consumer==null)try{consumer=format.fieldsConsumer(null);}catch(IOException expected){}
      MergeState merge=merge(directory,delegate.fields,new DocValuesProducer[]{delegate.producer,null,delegate.producer},new FieldInfos[]{delegate.fields,new FieldInfos(new FieldInfo[0]),delegate.fields},new int[]{2,0,2},new int[][]{{0,1},{},{2,3}},new Bits[]{null,null,null},false);
      for(int step=0;step<2400;step++){
        state.events.clear();state.fail=step%11==0;int op=step%14;String status="ok";
        try{switch(op){
          case 0->status=format.getName();case 1->format.fieldsConsumer(null);
          case 2->consumer.addNumericField(delegate.fields.fieldInfo("n"),delegate.producer);
          case 3->consumer.addBinaryField(delegate.fields.fieldInfo("b"),delegate.producer);
          case 4->consumer.addSortedField(delegate.fields.fieldInfo("s"),delegate.producer);
          case 5->consumer.addSortedNumericField(delegate.fields.fieldInfo("sn"),delegate.producer);
          case 6->consumer.addSortedSetField(delegate.fields.fieldInfo("ss"),delegate.producer);
          case 7->consumer.close();case 8->status=Boolean.toString(format.fieldsProducer(null)==delegate.producer);
          case 9->delegate.producer.checkIntegrity();case 10->delegate.producer.close();case 11->consumer.merge(merge);
          case 12->consumer.mergeSortedField(delegate.fields.fieldInfo("s"),merge);case 13->consumer.mergeSortedSetField(delegate.fields.fieldInfo("ss"),merge);
        }}catch(IOException error){status=error.getMessage();}
        out.append(seed).append('\t').append(step).append('\t').append(op).append('\t').append(status).append('\t').append(String.join(",",state.events)).append('\t').append(random.nextInt()).append('\n');
      }
    }return out.toString();
  }
  static String hex(BytesRef value){return value.length==0?"-":HexFormat.of().formatHex(value.bytes,value.offset,value.offset+value.length);}
  static String snapshot(DocValuesProducer producer,FieldInfos fields,int maxDoc)throws Exception {
    StringBuilder out=new StringBuilder("maxDoc\t"+maxDoc+"\n");
    for(FieldInfo field:fields){
      DocValuesType type=field.getDocValuesType();if(type==DocValuesType.NONE)continue;
      out.append("field\t").append(field.number).append('\t').append(field.name).append('\t').append(type).append('\t').append(field.docValuesSkipIndexType()).append('\n');
      switch(type){
        case NUMERIC->{var values=producer.getNumeric(field);for(int doc=values.nextDoc();doc!=DocIdSetIterator.NO_MORE_DOCS;doc=values.nextDoc())out.append("numeric\t").append(field.number).append('\t').append(doc).append('\t').append(values.longValue()).append('\n');}
        case BINARY->{var values=producer.getBinary(field);for(int doc=values.nextDoc();doc!=DocIdSetIterator.NO_MORE_DOCS;doc=values.nextDoc())out.append("binary\t").append(field.number).append('\t').append(doc).append('\t').append(hex(values.binaryValue())).append('\n');}
        case SORTED->{var values=producer.getSorted(field);for(int ord=0;ord<values.getValueCount();ord++)out.append("term\t").append(field.number).append('\t').append(ord).append('\t').append(hex(values.lookupOrd(ord))).append('\n');for(int doc=values.nextDoc();doc!=DocIdSetIterator.NO_MORE_DOCS;doc=values.nextDoc())out.append("sorted\t").append(field.number).append('\t').append(doc).append('\t').append(values.ordValue()).append('\n');}
        case SORTED_NUMERIC->{var values=producer.getSortedNumeric(field);for(int doc=values.nextDoc();doc!=DocIdSetIterator.NO_MORE_DOCS;doc=values.nextDoc()){out.append("sortedNumeric\t").append(field.number).append('\t').append(doc);int count=values.docValueCount();for(int i=0;i<count;i++)out.append('\t').append(values.nextValue());out.append('\n');}}
        case SORTED_SET->{var values=producer.getSortedSet(field);for(long ord=0;ord<values.getValueCount();ord++)out.append("term\t").append(field.number).append('\t').append(ord).append('\t').append(hex(values.lookupOrd(ord))).append('\n');for(int doc=values.nextDoc();doc!=DocIdSetIterator.NO_MORE_DOCS;doc=values.nextDoc()){out.append("sortedSet\t").append(field.number).append('\t').append(doc);int count=values.docValueCount();for(int i=0;i<count;i++)out.append('\t').append(values.nextOrd());out.append('\n');}}
        default->throw new AssertionError();
      }
      if(field.docValuesSkipIndexType()!=DocValuesSkipIndexType.NONE){
        DocValuesSkipper skip=producer.getSkipper(field);out.append("skipGlobal\t").append(field.number).append('\t').append(skip.minValue()).append('\t').append(skip.maxValue()).append('\t').append(skip.docCount()).append('\t').append(skip.maxValueCount()).append('\n');
        for(int target=0;target<maxDoc;){skip.advance(target);if(skip.minDocID(0)==DocIdSetIterator.NO_MORE_DOCS)break;for(int level=0;level<skip.numLevels();level++)out.append("skip\t").append(field.number).append('\t').append(level).append('\t').append(skip.minDocID(level)).append('\t').append(skip.maxDocID(level)).append('\t').append(skip.minValue(level)).append('\t').append(skip.maxValue(level)).append('\t').append(skip.docCount(level)).append('\n');target=skip.maxDocID(0)+1;}
      }
    }return out.toString();
  }
  static BytesRef term(int side,int doc){return new BytesRef(doc==3?"gone-"+side:doc%6==0?"":doc%2==0?"aa":side==0?"left":"right");}
  static void populate(Directory directory,int side)throws Exception{
    try(IndexWriter writer=new IndexWriter(directory,new IndexWriterConfig().setUseCompoundFile(false).setCodec(new Lucene104Codec()))){
      for(int doc=0;doc<DOCS;doc++){
        Document document=new Document();
        if(doc%3!=2)document.add(NumericDocValuesField.indexedField("numeric",doc==0?Long.MIN_VALUE:doc==9?Long.MAX_VALUE:(side==0?-37L:71L)*doc));
        if(doc%4!=1)document.add(new BinaryDocValuesField("binary",new BytesRef(doc==0?new byte[0]:new byte[]{(byte)side,(byte)doc,(byte)255})));
        if(doc%3!=1)document.add(SortedDocValuesField.indexedField("sorted",term(side,doc)));
        if(doc%4!=2){document.add(SortedNumericDocValuesField.indexedField("sortedNumeric",doc));if(doc%2==1){document.add(SortedNumericDocValuesField.indexedField("sortedNumeric",-doc));document.add(SortedNumericDocValuesField.indexedField("sortedNumeric",doc));}}
        if(doc%4!=2){document.add(SortedSetDocValuesField.indexedField("sortedSet",term(side,doc)));if(doc!=3)document.add(SortedSetDocValuesField.indexedField("sortedSet",new BytesRef("cc")));}
        if(doc==3){document.add(new NumericDocValuesField("emptyNumeric",1));document.add(new BinaryDocValuesField("emptyBinary",new BytesRef("removed")));document.add(new SortedDocValuesField("emptySorted",new BytesRef("removed")));document.add(new SortedNumericDocValuesField("emptySortedNumeric",1));document.add(new SortedSetDocValuesField("emptySortedSet",new BytesRef("removed")));}
        writer.addDocument(document);
      }writer.forceMerge(1);
    }
  }
  static void save(Path out,String prefix,Directory directory)throws Exception{
    for(String extension:EXTENSIONS)try(IndexInput input=directory.openInput("_cdv_s."+extension,IOContext.DEFAULT)){byte[] bytes=new byte[Math.toIntExact(input.length())];input.readBytes(bytes,0,bytes.length);Files.write(out.resolve(prefix+"."+extension),bytes);}
  }
  static void saveInfos(Path out,FieldInfos fields,Directory directory)throws Exception{
    new Lucene104Codec().fieldInfosFormat().write(directory,segment(directory,DOCS),"",fields,IOContext.DEFAULT);
    try(IndexInput input=directory.openInput("_cdv.fnm",IOContext.DEFAULT)){byte[] bytes=new byte[Math.toIntExact(input.length())];input.readBytes(bytes,0,bytes.length);Files.write(out.resolve("cranky_doc_values.fnm"),bytes);}
  }
  static int mapped(int side,int doc,boolean sorted){if(doc==3||doc==10)return -1;int at=doc-(doc>3?1:0)-(doc>10?1:0);return sorted?2*at+side:10*side+at;}
  static void empty(Path out)throws Exception{
    try(Directory directory=new ByteBuffersDirectory()){
      FieldInfos fields=new FieldInfos(new FieldInfo[0]);Random random=new Random(42);DocValuesFormat cranky=wrap(NATIVE,random);
      try(DocValuesConsumer consumer=cranky.fieldsConsumer(write(directory,fields,0))){}
      save(out,"cranky_doc_values_empty",directory);
      try(DocValuesProducer producer=cranky.fieldsProducer(read(directory,fields,0))){producer.checkIntegrity();Files.writeString(out.resolve("cranky_doc_values_empty.tsv"),snapshot(producer,fields,0)+"rng\t"+random.nextInt()+"\n");}
    }
  }
  static void real(Path out)throws Exception{
    try(Directory left=new ByteBuffersDirectory();Directory right=new ByteBuffersDirectory();Directory savedLeft=new ByteBuffersDirectory();Directory savedRight=new ByteBuffersDirectory()){
      populate(left,0);populate(right,1);
      try(DirectoryReader l=DirectoryReader.open(left);DirectoryReader r=DirectoryReader.open(right)){
        SegmentReader[] leaves={(SegmentReader)l.leaves().get(0).reader(),(SegmentReader)r.leaves().get(0).reader()};FieldInfos fields=leaves[0].getFieldInfos();Directory[] saved={savedLeft,savedRight};
        for(int side=0;side<2;side++){
          Random random=new Random(42);DocValuesFormat cranky=wrap(NATIVE,random);
          try(DocValuesConsumer consumer=cranky.fieldsConsumer(write(saved[side],fields,DOCS))){for(FieldInfo field:fields)add(consumer,field,leaves[side].getDocValuesReader());}
          String prefix="cranky_doc_values_"+(side==0?"left":"right");save(out,prefix,saved[side]);
          try(DocValuesProducer producer=cranky.fieldsProducer(read(saved[side],fields,DOCS))){producer.checkIntegrity();String actual=snapshot(producer,fields,DOCS),expected=snapshot(leaves[side].getDocValuesReader(),fields,DOCS);if(!actual.lines().filter(line->!line.startsWith("skip\t")).toList().equals(expected.lines().filter(line->!line.startsWith("skip\t")).toList()))throw new AssertionError("native readback");Files.writeString(out.resolve(prefix+".tsv"),actual+"rng\t"+random.nextInt()+"\n");}
        }
        saveInfos(out,fields,savedLeft);
        try(DocValuesProducer lp=NATIVE.fieldsProducer(read(savedLeft,fields,DOCS));DocValuesProducer rp=NATIVE.fieldsProducer(read(savedRight,fields,DOCS))){
          for(boolean sorted:new boolean[]{false,true})try(Directory target=new ByteBuffersDirectory()){
            int[][] maps=new int[2][DOCS];for(int side=0;side<2;side++)for(int doc=0;doc<DOCS;doc++)maps[side][doc]=mapped(side,doc,sorted);
            Bits live=new Bits(){@Override public boolean get(int doc){return doc!=3&&doc!=10;}@Override public int length(){return DOCS;}};
            MergeState state=merge(target,fields,new DocValuesProducer[]{lp,rp},new FieldInfos[]{fields,fields},new int[]{DOCS,DOCS},maps,new Bits[]{live,live},sorted);
            Random random=new Random(42);DocValuesFormat cranky=wrap(NATIVE,random);
            try(DocValuesConsumer consumer=cranky.fieldsConsumer(write(target,fields,20))){consumer.merge(state);}
            String prefix="cranky_doc_values_merge_"+(sorted?"sorted":"sequential");save(out,prefix,target);
            try(DocValuesProducer producer=NATIVE.fieldsProducer(read(target,fields,20))){producer.checkIntegrity();String actual=snapshot(producer,fields,20);if(actual.contains("676f6e652d")||actual.contains("72656d6f766564"))throw new AssertionError("deleted-only dictionary term survived");Files.writeString(out.resolve(prefix+".tsv"),actual+"rng\t"+random.nextInt()+"\n");}
          }
        }
      }
    }
  }
  public static void main(String[] args)throws Exception{
    if(!GenCrankyDocValues.class.desiredAssertionStatus())throw new AssertionError("run with -ea");Path out=Path.of(args[0]);Files.createDirectories(out);Files.writeString(out.resolve("cranky_doc_values.tsv"),trace());empty(out);real(out);
  }
}

