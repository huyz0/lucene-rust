import java.io.IOException;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.atomic.AtomicReference;
import org.apache.lucene.codecs.*;
import org.apache.lucene.codecs.lucene90.Lucene90DocValuesFormat;
import org.apache.lucene.codecs.lucene104.Lucene104Codec;
import org.apache.lucene.index.*;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.store.*;
import org.apache.lucene.util.*;

/** Actual asserting classes under -ea: callback/error traces and native all-five files. */
public class GenAssertingDocValues {
  static final String[] KINDS={"numeric","binary","sorted","sortedNumeric","sortedSet"};
  static final DocValuesType[] TYPES={DocValuesType.NUMERIC,DocValuesType.BINARY,DocValuesType.SORTED,DocValuesType.SORTED_NUMERIC,DocValuesType.SORTED_SET};
  static FieldInfo field(String name,int number,DocValuesType type,boolean skip){return new FieldInfo(name,number,false,true,false,IndexOptions.NONE,type,skip?DocValuesSkipIndexType.RANGE:DocValuesSkipIndexType.NONE,-1,Map.of(),0,0,0,0,VectorEncoding.FLOAT32,VectorSimilarityFunction.EUCLIDEAN,false,false);}
  static final FieldInfos FIELDS=new FieldInfos(new FieldInfo[]{field(KINDS[0],0,TYPES[0],true),field(KINDS[1],1,TYPES[1],false),field(KINDS[2],2,TYPES[2],false),field(KINDS[3],3,TYPES[3],false),field(KINDS[4],4,TYPES[4],false)});
  static DocValuesConsumer wrap(DocValuesConsumer delegate,int maxDoc)throws Exception {var c=Class.forName("org.apache.lucene.tests.codecs.asserting.AssertingDocValuesFormat$AssertingDocValuesConsumer").getDeclaredConstructor(DocValuesConsumer.class,int.class);c.setAccessible(true);return (DocValuesConsumer)c.newInstance(delegate,maxDoc);}
  static DocValuesProducer wrap(DocValuesProducer delegate,FieldInfos fields,int maxDoc)throws Exception {var c=Class.forName("org.apache.lucene.tests.codecs.asserting.AssertingDocValuesFormat$AssertingDocValuesProducer").getDeclaredConstructor(DocValuesProducer.class,FieldInfos.class,int.class,boolean.class);c.setAccessible(true);return (DocValuesProducer)c.newInstance(delegate,fields,maxDoc,false);}
  static DocValuesFormat format()throws Exception {var format=new org.apache.lucene.tests.codecs.asserting.AssertingDocValuesFormat();var field=format.getClass().getDeclaredField("in");field.setAccessible(true);field.set(format,new Lucene90DocValuesFormat());return format;}
  @FunctionalInterface interface Action {void run()throws Exception;}
  static String outcome(Action action)throws Exception {try{action.run();return "ok";}catch(AssertionError e){return "assert";}catch(IndexOutOfBoundsException e){return "assert";}catch(NullPointerException e){return "null-field";}catch(AlreadyClosedException e){return "closed";}catch(IOException e){return "io:"+e.getMessage();}}
  static final class State {
    final List<String> calls=new ArrayList<>();final String scenario;int closes;
    State(String scenario){this.scenario=scenario;}
    void note(String call){calls.add(call);}
    void call(String call)throws IOException {note(call);if(scenario.equals("fail-"+call)||scenario.equals("fail-close2")&&call.equals("close")&&++closes==2)throw new IOException(call+" failure");}
    int[] docs(){return switch(scenario){case "empty"->new int[0];case "bounds"->new int[]{0,7};case "reverse"->new int[]{2,0};case "duplicate"->new int[]{0,0};case "zero"->new int[]{0};default->new int[]{0,2};};}
    int dictSize(){return switch(scenario){case "empty","zero"->0;case "orphan"->3;default->2;};}
    long number(int row,int index){if(scenario.equals("descending")&&row==0)return index==0?3:-1;return row==0?index==0?-1:3:4;}
    int count(int row){return scenario.equals("zero")?0:2;}
    long ord(int row,int index,boolean set){if(scenario.equals("badord")&&row==1)return 5;if(scenario.equals("negative")&&row==1)return -1;if(scenario.equals("descending")&&set&&row==0)return index==0?1:0;return set?index:row;}
  }
  static final class Cursor {
    final State s;int row=-1,upto,doc;
    Cursor(State s){this.s=s;doc=s.scenario.equals("positioned")?1:-1;}
    int next()throws IOException{s.call("next");upto=0;return doc=++row<s.docs().length?s.docs()[row]:DocIdSetIterator.NO_MORE_DOCS;}
    int id(){s.note("docID");return doc;}
    long value(){s.note("long");return row+1;}
    BytesRef binary(){s.note("binary");return new BytesRef(new byte[]{(byte)row});}
    int count(){s.note("count");return s.count(row);}
    long nextValue()throws IOException{s.call("value");return s.number(row,upto++);}
    long nextOrd()throws IOException{s.call("ord");return s.ord(row,upto++,true);}
    int ord(){s.note("ordValue");return (int)s.ord(row,0,false);}
    int valueCount(){s.note("valueCount");return s.dictSize();}
    BytesRef lookup(long ord)throws IOException{s.call("lookup:"+ord);return new BytesRef(new byte[]{(byte)(s.scenario.equals("dictionary")?'z'-ord:'a'+ord)});}
  }
  static final class Producer extends DocValuesProducer {
    final State s;Producer(State s){this.s=s;}
    public NumericDocValues getNumeric(FieldInfo f)throws IOException{s.call("get:numeric");var c=new Cursor(s);return new NumericDocValues(){public int docID(){return c.id();}public int nextDoc()throws IOException{return c.next();}public int advance(int target)throws IOException{while(nextDoc()<target){}return c.doc;}public boolean advanceExact(int target){throw new UnsupportedOperationException();}public long longValue(){return c.value();}public long cost(){return 2;}};}
    public BinaryDocValues getBinary(FieldInfo f)throws IOException{s.call("get:binary");var c=new Cursor(s);return new BinaryDocValues(){public int docID(){return c.id();}public int nextDoc()throws IOException{return c.next();}public int advance(int target)throws IOException{while(nextDoc()<target){}return c.doc;}public boolean advanceExact(int target){throw new UnsupportedOperationException();}public BytesRef binaryValue(){return c.binary();}public long cost(){return 2;}};}
    public SortedDocValues getSorted(FieldInfo f)throws IOException{s.call("get:sorted");var c=new Cursor(s);return new SortedDocValues(){public int docID(){return c.id();}public int nextDoc()throws IOException{return c.next();}public int advance(int target)throws IOException{while(nextDoc()<target){}return c.doc;}public boolean advanceExact(int target){throw new UnsupportedOperationException();}public int ordValue(){return c.ord();}public BytesRef lookupOrd(int ord)throws IOException{return c.lookup(ord);}public int getValueCount(){return c.valueCount();}public long cost(){return 2;}};}
    public SortedNumericDocValues getSortedNumeric(FieldInfo f)throws IOException{s.call("get:sortedNumeric");var c=new Cursor(s);return new SortedNumericDocValues(){public int docID(){return c.id();}public int nextDoc()throws IOException{return c.next();}public int advance(int target)throws IOException{while(nextDoc()<target){}return c.doc;}public boolean advanceExact(int target){throw new UnsupportedOperationException();}public int docValueCount(){return c.count();}public long nextValue()throws IOException{return c.nextValue();}public long cost(){return 2;}};}
    public SortedSetDocValues getSortedSet(FieldInfo f)throws IOException{s.call("get:sortedSet");var c=new Cursor(s);return new SortedSetDocValues(){public int docID(){return c.id();}public int nextDoc()throws IOException{return c.next();}public int advance(int target)throws IOException{while(nextDoc()<target){}return c.doc;}public boolean advanceExact(int target){throw new UnsupportedOperationException();}public int docValueCount(){return c.count();}public long nextOrd()throws IOException{return c.nextOrd();}public BytesRef lookupOrd(long ord)throws IOException{return c.lookup(ord);}public long getValueCount(){return c.valueCount();}public long cost(){return 2;}};}
    public DocValuesSkipper getSkipper(FieldInfo f)throws IOException{s.call("get:skipper");return s.scenario.equals("null-skipper")?null:new Skipper(s);}
    public void close()throws IOException{s.call("close");}
    public void checkIntegrity()throws IOException{s.call("integrity");}
    public DocValuesProducer getMergeInstance(){s.note("merge");return new Producer(s);}
    public String toString(){s.note("toString");return "ProbeProducer";}
  }
  static final class Consumer extends DocValuesConsumer {
    final State s;final Producer producer;Consumer(State s,Producer producer){this.s=s;this.producer=producer;}
    void add(FieldInfo f,DocValuesProducer p)throws IOException {if(p!=producer)throw new AssertionError("producer identity");if(f!=FIELDS.fieldInfo(f.name))throw new AssertionError("field identity");s.call("delegate");}
    public void addNumericField(FieldInfo f,DocValuesProducer p)throws IOException{add(f,p);}public void addBinaryField(FieldInfo f,DocValuesProducer p)throws IOException{add(f,p);}public void addSortedField(FieldInfo f,DocValuesProducer p)throws IOException{add(f,p);}public void addSortedNumericField(FieldInfo f,DocValuesProducer p)throws IOException{add(f,p);}public void addSortedSetField(FieldInfo f,DocValuesProducer p)throws IOException{add(f,p);}public void close()throws IOException{s.call("close");}
  }
  static final class Skipper extends DocValuesSkipper {
    final State s;int doc=-1;Skipper(State s){this.s=s;if(s.scenario.equals("positioned"))doc=0;}
    public void advance(int target)throws IOException{s.call("skipAdvance");doc=target<5?target:DocIdSetIterator.NO_MORE_DOCS;}
    public int numLevels(){return 2;}public int minDocID(int level){return doc;}public int maxDocID(int level){return doc== -1||doc==DocIdSetIterator.NO_MORE_DOCS?doc:level==0?doc+1:doc+2;}
    public long minValue(int level){return -2;}public long maxValue(int level){return 9;}public int docCount(int level){return 2;}public long minValue(){return -2;}public long maxValue(){return 9;}public int docCount(){return 4;}public int maxValueCount(){return 2;}
  }
  static void getter(DocValuesProducer p,int kind,FieldInfo field)throws IOException{switch(kind){case 0->p.getNumeric(field);case 1->p.getBinary(field);case 2->p.getSorted(field);case 3->p.getSortedNumeric(field);case 4->p.getSortedSet(field);default->throw new AssertionError();}}
  static void row(StringBuilder out,String group,String kind,String scenario,String status,State s){out.append(group).append('\t').append(kind).append('\t').append(scenario).append('\t').append(status).append('\t').append(String.join(",",s.calls)).append('\n');}
  static void consumerTraces(StringBuilder out)throws Exception {
    for(int kind=0;kind<5;kind++)for(String scenario:new String[]{"good","empty","bounds","reverse","duplicate","dictionary","orphan","badord","negative","descending","zero","fail-get:"+KINDS[kind],"fail-next","fail-lookup:0","fail-value","fail-ord","fail-delegate"}){
      var s=new State(scenario);var producer=new Producer(s);var consumer=wrap(new Consumer(s,producer),5);FieldInfo field=FIELDS.fieldInfo(kind);String status=outcome(()->GenCrankyDocValues.add(consumer,field,producer));row(out,"consumer",KINDS[kind],scenario,status,s);
    }
    for(String scenario:new String[]{"good","fail-close","fail-close2"}){var s=new State(scenario);var producer=new Producer(s);var consumer=wrap(new Consumer(s,producer),5);row(out,"consumer","close",scenario,outcome(consumer::close),s);}
  }
  static void producerTraces(StringBuilder out)throws Exception {
    for(int kind=0;kind<5;kind++)for(String scenario:new String[]{"good","positioned","wrong-number","wrong-type","missing","fail-get:"+KINDS[kind],"merge"}){
      var s=new State(scenario);var p=wrap(new Producer(s),FIELDS,5);if(scenario.equals("merge"))p=p.getMergeInstance();final DocValuesProducer producer=p;final int k=kind;
      FieldInfo field=scenario.equals("wrong-number")?field(KINDS[kind],17,TYPES[kind],false):scenario.equals("wrong-type")?field(KINDS[kind],kind,TYPES[(kind+1)%5],false):scenario.equals("missing")?field("absent",kind,TYPES[kind],false):FIELDS.fieldInfo(kind);
      row(out,"producer",KINDS[kind],scenario,outcome(()->getter(producer,k,field)),s);
    }
    for(String scenario:new String[]{"good","fail-close","fail-close2","integrity","fail-integrity"}){var s=new State(scenario);var p=wrap(new Producer(s),FIELDS,5);String status=outcome(()->{if(scenario.contains("integrity"))p.checkIntegrity();else p.close();});row(out,"producer","lifecycle",scenario,status,s);}
    for(String scenario:new String[]{"good","positioned","unpositioned","level","advance","fail-skipAdvance","none","missing","wrong-number","null-skipper","fail-get:skipper"}){var s=new State(scenario);var p=wrap(new Producer(s),FIELDS,5);String status=outcome(()->{FieldInfo field=scenario.equals("none")?field("numeric",0,TYPES[0],false):scenario.equals("missing")?field("absent",0,TYPES[0],true):scenario.equals("wrong-number")?field("numeric",17,TYPES[0],true):FIELDS.fieldInfo(0);var skipper=p.getSkipper(field);switch(scenario){case "unpositioned"->skipper.minValue(0);case "level"->skipper.minDocID(2);case "advance"->{skipper.advance(0);skipper.advance(1);}default->{skipper.advance(0);if(skipper.numLevels()!=2||skipper.minDocID(1)!=0||skipper.maxDocID(1)!=2||skipper.minValue(0)!=-2||skipper.maxValue(0)!=9||skipper.docCount(0)!=2||skipper.minValue()!=-2||skipper.maxValue()!=9||skipper.docCount()!=4||skipper.maxValueCount()!=2)throw new AssertionError();}}});row(out,"skipper","numeric",scenario,status,s);}
    {var s=new State("thread");var p=wrap(new Producer(s),FIELDS,5);var skipper=p.getSkipper(FIELDS.fieldInfo(0));var statuses=new ArrayList<String>();for(int op=0;op<11;op++){final int operation=op;var status=new AtomicReference<String>();var thread=new Thread(()->{try{status.set(outcome(()->{switch(operation){case 0->skipper.advance(0);case 1->skipper.numLevels();case 2->skipper.minDocID(0);case 3->skipper.maxDocID(0);case 4->skipper.minValue(0);case 5->skipper.maxValue(0);case 6->skipper.docCount(0);case 7->skipper.minValue();case 8->skipper.maxValue();case 9->skipper.docCount();case 10->skipper.maxValueCount();}}));}catch(Exception error){throw new RuntimeException(error);}});thread.start();thread.join();statuses.add(status.get());}row(out,"skipper","numeric","thread",String.join(";",statuses),s);}
    for(int kind=0;kind<5;kind++){final int k=kind;var s=new State("thread");var p=wrap(new Producer(s),FIELDS,5).getMergeInstance();var status=new AtomicReference<String>();var thread=new Thread(()->{try{status.set(outcome(()->getter(p,k,FIELDS.fieldInfo(k))));}catch(Exception error){throw new RuntimeException(error);}});thread.start();thread.join();row(out,"producer",KINDS[kind],"thread",status.get(),s);}
  }
  static void real(Path out)throws Exception {
    var format=format();
    try(var source=new ByteBuffersDirectory();var target=new ByteBuffersDirectory()) {
      GenCrankyDocValues.populate(source,0);
      try(var reader=DirectoryReader.open(source)) {
        var leaf=(SegmentReader)reader.leaves().get(0).reader();var fields=leaf.getFieldInfos();
        try(var consumer=format.fieldsConsumer(GenCrankyDocValues.write(target,fields,12))){for(var field:fields)GenCrankyDocValues.add(consumer,field,leaf.getDocValuesReader());}
        for(String ext:GenCrankyDocValues.EXTENSIONS)try(var input=target.openInput("_cdv_s."+ext,IOContext.DEFAULT)){byte[] bytes=new byte[Math.toIntExact(input.length())];input.readBytes(bytes,0,bytes.length);Files.write(out.resolve("asserting_doc_values."+ext),bytes);}
        new Lucene104Codec().fieldInfosFormat().write(target,GenCrankyDocValues.segment(target,12),"",fields,IOContext.DEFAULT);
        try(var input=target.openInput("_cdv.fnm",IOContext.DEFAULT)){byte[] bytes=new byte[Math.toIntExact(input.length())];input.readBytes(bytes,0,bytes.length);Files.write(out.resolve("asserting_doc_values.fnm"),bytes);}
        try(var producer=format.fieldsProducer(GenCrankyDocValues.read(target,fields,12))){producer.checkIntegrity();String actual=GenCrankyDocValues.snapshot(producer,fields,12),expected=GenCrankyDocValues.snapshot(leaf.getDocValuesReader(),fields,12);if(!actual.lines().filter(line->!line.startsWith("skip\t")).toList().equals(expected.lines().filter(line->!line.startsWith("skip\t")).toList()))throw new AssertionError("native readback");Files.writeString(out.resolve("asserting_doc_values_values.tsv"),actual);var merge=producer.getMergeInstance();GenCrankyDocValues.snapshot(merge,fields,12);merge.close();}
        var lifecycle=new StringBuilder();
        for(String probe:new String[]{"original-close-merged-read","merged-close-original-read","closed-original-merge"}){
          var original=format.fieldsProducer(GenCrankyDocValues.read(target,fields,12));var merged=original.getMergeInstance();String expected=GenCrankyDocValues.snapshot(original,fields,12);String status;
          switch(probe){
            case "original-close-merged-read"->{original.close();status=outcome(()->{if(!GenCrankyDocValues.snapshot(merged,fields,12).equals(expected))throw new AssertionError("merge read changed");});merged.close();}
            case "merged-close-original-read"->{merged.close();status=outcome(()->{if(!GenCrankyDocValues.snapshot(original,fields,12).equals(expected))throw new AssertionError("original read changed");});original.close();}
            case "closed-original-merge"->{original.close();status=outcome(()->original.getMergeInstance());merged.close();}
            default->throw new AssertionError();
          }
          lifecycle.append(probe).append('\t').append(status).append('\n');
        }
        Files.writeString(out.resolve("asserting_doc_values_lifecycle.tsv"),lifecycle);
      }
    }
  }
  public static void main(String[] args)throws Exception {if(!GenAssertingDocValues.class.desiredAssertionStatus())throw new AssertionError("run with -ea");Path out=Path.of(args[0]);Files.createDirectories(out);var traces=new StringBuilder();consumerTraces(traces);producerTraces(traces);Files.writeString(out.resolve("asserting_doc_values.tsv"),traces);real(out);}
}