import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import org.apache.lucene.index.*;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.tests.index.AssertingLeafReader.*;
import org.apache.lucene.util.BytesRef;

/** Actual asserting leaf iterators, scripted adversarial delegates and complete callback traces. */
public class GenAssertingDocValuesIterators {
  static final int MAX_DOC = 8;
  static final StringBuilder OUT = new StringBuilder();
  static class Probe {
    int doc=-1, next=0, advanced=2, exactDoc=2, report=Integer.MIN_VALUE;
    boolean exists=true;
    int count=2, count2=2, countCalls=0, dictionary=4, ord=1, value=7, lookup=1;
    long cost=3;
    String fail="none";
    final List<String> events=new ArrayList<>();
    void event(String name) throws IOException {
      events.add(name);
      String base=name.split(":")[0];
      if(fail.equals(base+":runtime") || (fail.equals(base+"-second:runtime") && events.stream().filter(e->e.split(":")[0].equals(base)).count()==2))
        throw new IllegalStateException("delegate-runtime");
      if(fail.equals(base+":io")) throw new IOException("delegate-io");
      if(base.equals("count") && fail.equals("count-second:runtime") && countCalls==1)
        throw new IllegalStateException("delegate-runtime");
    }
    void scalar(String name) {
      try {event(name);} catch(IOException e) {throw new IllegalStateException("delegate-runtime",e);}
    }
    int docID(){scalar("docID");return report==Integer.MIN_VALUE?doc:report;}
    int nextDoc()throws IOException{event("next");doc=next;return next;}
    int advance(int target)throws IOException{event("advance:"+target);doc=advanced;return advanced;}
    boolean exact(int target)throws IOException{event("exact:"+target);doc=exactDoc;return exists;}
    long cost(){scalar("cost");return cost;}
    int count(){scalar("count");return countCalls++==0?count:count2;}
    int dictionary(){scalar("dictionary");return dictionary;}
    long value()throws IOException{event("value");return value;}
    int ord()throws IOException{event("ord");return ord;}
    BytesRef binary()throws IOException{event("binary");return new BytesRef("v");}
    BytesRef lookupOrd(long ord)throws IOException{event("lookupOrd:"+ord);return new BytesRef("t"+ord);}
    int lookupTerm(BytesRef key)throws IOException{event("lookupTerm:"+key.utf8ToString());return lookup;}
    void set(String command){
      String[] pair=command.split("=",2);String key=pair[0],v=pair[1];
      switch(key){
        case "doc"->doc=Integer.parseInt(v);case "next"->next=Integer.parseInt(v);
        case "advanced"->advanced=Integer.parseInt(v);case "exactDoc"->exactDoc=Integer.parseInt(v);
        case "report"->report=v.equals("none")?Integer.MIN_VALUE:Integer.parseInt(v);
        case "exists"->exists=Boolean.parseBoolean(v);
        case "count"->{count=Integer.parseInt(v);count2=count;countCalls=0;}
        case "count2"->count2=Integer.parseInt(v);
        case "dictionary"->dictionary=Integer.parseInt(v);case "ord"->ord=Integer.parseInt(v);
        case "value"->value=Integer.parseInt(v);case "lookup"->lookup=Integer.parseInt(v);
        case "cost"->cost=Long.parseLong(v);case "fail"->fail=v;
        default->throw new AssertionError(key);
      }
    }
  }
  static class N extends NumericDocValues {
    final Probe p;N(Probe p){this.p=p;}
    public int docID(){return p.docID();}public int nextDoc()throws IOException{return p.nextDoc();}
    public int advance(int t)throws IOException{return p.advance(t);}public boolean advanceExact(int t)throws IOException{return p.exact(t);}
    public long cost(){return p.cost();}public long longValue()throws IOException{return p.value();}
  }
  static class B extends BinaryDocValues {
    final Probe p;B(Probe p){this.p=p;}
    public int docID(){return p.docID();}public int nextDoc()throws IOException{return p.nextDoc();}
    public int advance(int t)throws IOException{return p.advance(t);}public boolean advanceExact(int t)throws IOException{return p.exact(t);}
    public long cost(){return p.cost();}public BytesRef binaryValue()throws IOException{return p.binary();}
  }
  static class S extends SortedDocValues {
    final Probe p;S(Probe p){this.p=p;}
    public int docID(){return p.docID();}public int nextDoc()throws IOException{return p.nextDoc();}
    public int advance(int t)throws IOException{return p.advance(t);}public boolean advanceExact(int t)throws IOException{return p.exact(t);}
    public long cost(){return p.cost();}public int ordValue()throws IOException{return p.ord();}
    public BytesRef lookupOrd(int o)throws IOException{return p.lookupOrd(o);}public int getValueCount(){return p.dictionary();}
    public int lookupTerm(BytesRef key)throws IOException{return p.lookupTerm(key);}
  }
  static class SN extends SortedNumericDocValues {
    final Probe p;SN(Probe p){this.p=p;}
    public int docID(){return p.docID();}public int nextDoc()throws IOException{return p.nextDoc();}
    public int advance(int t)throws IOException{return p.advance(t);}public boolean advanceExact(int t)throws IOException{return p.exact(t);}
    public long cost(){return p.cost();}public long nextValue()throws IOException{return p.value();}public int docValueCount(){return p.count();}
  }
  static class SS extends SortedSetDocValues {
    final Probe p;SS(Probe p){this.p=p;}
    public int docID(){return p.docID();}public int nextDoc()throws IOException{return p.nextDoc();}
    public int advance(int t)throws IOException{return p.advance(t);}public boolean advanceExact(int t)throws IOException{return p.exact(t);}
    public long cost(){return p.cost();}public long nextOrd()throws IOException{return p.ord();}public int docValueCount(){return p.count();}
    public BytesRef lookupOrd(long o)throws IOException{return p.lookupOrd(o);}public long getValueCount(){return p.dictionary();}
    public long lookupTerm(BytesRef key)throws IOException{return p.lookupTerm(key);}
  }
  static DocIdSetIterator wrap(String kind,Probe p){
    return switch(kind){
      case "numeric"->new AssertingNumericDocValues(new N(p),MAX_DOC);
      case "binary"->new AssertingBinaryDocValues(new B(p),MAX_DOC);
      case "sorted"->new AssertingSortedDocValues(new S(p),MAX_DOC);
      case "sortedNumeric"->AssertingSortedNumericDocValues.create(new SN(p),MAX_DOC);
      case "sortedSet"->AssertingSortedSetDocValues.create(new SS(p),MAX_DOC);
      default->throw new AssertionError(kind);
    };
  }
  static String call(DocIdSetIterator dv,String op)throws IOException{
    String[] parts=op.split(":",2);int arg=parts.length==2&&!parts[0].equals("term")?Integer.parseInt(parts[1]):0;
    return switch(parts[0]){
      case "docID"->""+dv.docID();case "next"->""+dv.nextDoc();case "advance"->""+dv.advance(arg);
      case "exact"->""+(dv instanceof NumericDocValues n?n.advanceExact(arg):dv instanceof BinaryDocValues b?b.advanceExact(arg):dv instanceof SortedDocValues s?s.advanceExact(arg):dv instanceof SortedNumericDocValues sn?sn.advanceExact(arg):((SortedSetDocValues)dv).advanceExact(arg));
      case "cost"->""+dv.cost();
      case "value"->dv instanceof NumericDocValues n?""+n.longValue():dv instanceof BinaryDocValues b?b.binaryValue().utf8ToString():dv instanceof SortedDocValues s?""+s.ordValue():dv instanceof SortedNumericDocValues sn?""+sn.nextValue():""+((SortedSetDocValues)dv).nextOrd();
      case "count"->""+(dv instanceof SortedNumericDocValues sn?sn.docValueCount():((SortedSetDocValues)dv).docValueCount());
      case "dictionary"->""+(dv instanceof SortedDocValues s?s.getValueCount():((SortedSetDocValues)dv).getValueCount());
      case "lookup"->(dv instanceof SortedDocValues s?s.lookupOrd(arg):((SortedSetDocValues)dv).lookupOrd(arg)).utf8ToString();
      case "term"->""+(dv instanceof SortedDocValues s?s.lookupTerm(new BytesRef(parts[1])):((SortedSetDocValues)dv).lookupTerm(new BytesRef(parts[1])));
      default->throw new AssertionError(op);
    };
  }
  static void one(String kind,String name,String script){
    Probe p=new Probe();DocIdSetIterator dv=null;List<String> results=new ArrayList<>();
    for(String op:("new;"+script).split(";")){
      if(op.contains("=")){p.set(op);continue;}
      p.events.clear();String status;
      try{if(op.equals("new")){dv=wrap(kind,p);status="ok";}else status="ok:"+call(dv,op);}
      catch(AssertionError e){status="assert";}catch(IOException e){status="io";}catch(RuntimeException e){status="runtime";}
      results.add(op+"~"+status+"~"+String.join(",",p.events));
    }
    String trace=String.join("|",results);
    if(name.equals("advance-no-postdoc") && !trace.contains(",advance:4")) throw new AssertionError("advance probe not reached");
    if((name.equals("advance-outside-order")||name.equals("advance-below-order")) && !trace.contains(",advance:1")) throw new AssertionError("advance probe not reached");
    if((name.equals("exact-true-postassert")||name.equals("exact-false-postassert")||name.equals("counter-reset-after-failed-exact"))
        && !trace.contains("exact:2~assert~docID,exact:2,docID")) throw new AssertionError("exact postcondition probe not reached: "+trace);
    OUT.append(kind).append('\t').append(name).append('\t').append(script).append('\t').append(String.join("|",results)).append('\n');
  }
  public static void main(String[] args)throws Exception{
    boolean assertions=false;assert assertions=true;if(!assertions)throw new AssertionError("run with -ea");
    for(String kind:new String[]{"numeric","binary","sorted","sortedNumeric","sortedSet"}){
      one(kind,"negative-advance","fail=docID:runtime;advance:-1");
      one(kind,"negative-exact","fail=docID:runtime;exact:-1");
      one(kind,"next-backward","next=-1;fail=docID:runtime;next");
      one(kind,"next-outside","next=8;fail=docID:runtime;next");
      one(kind,"next-mismatch","next=2;report=3;next;report=none;value");
      one(kind,"next-doc-runtime","fail=docID:runtime;next;fail=none;value;next=1;next;value");
      one(kind,"advance-no-postdoc","advanced=4;report=3;advance:4;report=none;value");
      one(kind,"advance-outside-order","advanced=8;report=0;advance:1");
      one(kind,"advance-below-order","advanced=0;report=-1;advance:1");
      one(kind,"advance-below-target","advanced=0;advance:1");
      one(kind,"advance-doc-runtime","fail=docID:runtime;advance:1");
      one(kind,"advance-postdoc-runtime","advance:0;doc=-1;fail=docID-second:runtime;advance:1");
      one(kind,"exact-true-postassert","report=1;exact:2;report=none;value");
      one(kind,"exact-false-postassert","next;exists=false;report=1;exact:2;report=none;value");
      one(kind,"exact-upper-order","report=0;exact:8");
      one(kind,"exact-backward","doc=3;exact:2");
      one(kind,"iterator-normal","docID;cost;next;value;advanced=3;advance:2;value;exactDoc=3;exact:3;value;exists=false;exactDoc=4;exact:4;value;next=2147483647;next;value");
      one(kind,"next-io","fail=next:io;next;fail=none;value;next;value");
      one(kind,"next-runtime","fail=next:runtime;next;fail=none;value;next;value");
      one(kind,"advance-io","fail=advance:io;advance:1;fail=none;value");
      one(kind,"advance-runtime","fail=advance:runtime;advance:1;fail=none;value");
      one(kind,"exact-io","next;fail=exact:io;exact:2;fail=none;value");
      one(kind,"exact-runtime","next;fail=exact:runtime;exact:2;fail=none;value");
      one(kind,"value-runtime","next;fail="+(kind.equals("binary")?"binary":kind.equals("sorted")||kind.equals("sortedSet")?"ord":"value")+":runtime;value;fail=none;value");
      one(kind,"negative-cost","cost=-1;cost");
      if(kind.equals("sortedNumeric")||kind.equals("sortedSet")){
        one(kind,"count-unpositioned","count");
        one(kind,"count-zero-unpositioned","count=0;count");
        one(kind,"count-zero-positioned","next;count=0;count");
        one(kind,"count-double-getter","next;count=1;count2=3;count");
        one(kind,"count-second-runtime","next;fail=count-second:runtime;count");
        one(kind,"value-count-failed-message","next;count=0;value");
        one(kind,"value-count-second-runtime","next;count=0;fail=count-second:runtime;value");
        one(kind,"value-failure-increments","next;count=1;fail="+(kind.equals("sortedNumeric")?"value":"ord")+":io;value;fail=none;value;count");
        one(kind,"counter-reset-after-failed-exact","next;count=1;value;report=1;exact:2;report=none;value");
        one(kind,"counter-reset-after-success","next;count=1;value;next=1;next;value;exactDoc=2;exact:2;value");
      }
      if(kind.equals("sorted")||kind.equals("sortedSet")){
        one(kind,"dictionary-normal","dictionary;lookup:0;term:t1");
        one(kind,"dictionary-changed","dictionary=3;dictionary");
        one(kind,"lookup-negative","lookup:-1");
        one(kind,"lookup-outside","lookup:4");
        one(kind,"lookup-io","fail=lookupOrd:io;lookup:1");
        one(kind,"term-outside","lookup=4;term:t1");
        one(kind,"term-negative","lookup=-5;term:z");
        one(kind,"ord-outside","next;ord=4;value");
        one(kind,"ord-negative","next;ord=-2;value");
      }
    }
    Path out=Path.of(args[0]);Files.createDirectories(out);Files.writeString(out.resolve("asserting_doc_values_iterators.tsv"),OUT);
  }
}

