import java.io.EOFException;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HexFormat;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Executor;
import org.apache.lucene.codecs.*;
import org.apache.lucene.codecs.simpletext.SimpleTextTermVectorsFormat;
import org.apache.lucene.index.*;
import org.apache.lucene.store.*;
import org.apache.lucene.tests.codecs.asserting.AssertingTermVectorsFormat;
import org.apache.lucene.util.*;

/** Actual asserting wrappers, with assertions enabled, mutable failure probes and real bytes. */
public class GenAssertingTermVectors {
  static final byte[] POS={3,2,0,10,4,1,0}, OFF={0,2,1,3,0,0};
  // SimpleText asserts that a nonnull payload is nonempty. The last code is
  // payload-absent here; the probe-only POS retains explicit empty coverage.
  static final byte[] REAL_POS={3,2,0,10,4,0};
  static final FieldInfo FIELD=field(7);
  static final StringBuilder OUT=new StringBuilder();
  static FieldInfo field(int number) {
    return new FieldInfo("f",number,true,false,false,IndexOptions.DOCS_AND_FREQS_AND_POSITIONS_AND_OFFSETS,
        DocValuesType.NONE,DocValuesSkipIndexType.NONE,-1,Map.of(),0,0,0,0,
        VectorEncoding.FLOAT32,VectorSimilarityFunction.EUCLIDEAN,false,false);
  }
  static SegmentInfo info(Directory dir,int docs) {
    return new SegmentInfo(dir,Version.LATEST,Version.LATEST,"_atv",docs,false,false,null,
        Map.of(),"0123456789abcdef".getBytes(StandardCharsets.US_ASCII),Map.of(),null);
  }
  static TermVectorsWriter writer(TermVectorsWriter delegate)throws Exception {
    var c=Class.forName("org.apache.lucene.tests.codecs.asserting.AssertingTermVectorsFormat$AssertingTermVectorsWriter")
        .getDeclaredConstructor(TermVectorsWriter.class);c.setAccessible(true);
    return (TermVectorsWriter)c.newInstance(delegate);
  }
  static TermVectorsReader reader(TermVectorsReader delegate)throws Exception {
    var c=Class.forName("org.apache.lucene.tests.codecs.asserting.AssertingTermVectorsFormat$AssertingTermVectorsReader")
        .getDeclaredConstructor(TermVectorsReader.class);c.setAccessible(true);
    return (TermVectorsReader)c.newInstance(delegate);
  }
  static TermVectorsFormat format(TermVectorsFormat delegate)throws Exception {
    var f=new AssertingTermVectorsFormat();var field=AssertingTermVectorsFormat.class.getDeclaredField("in");
    field.setAccessible(true);field.set(f,delegate);return f;
  }
  static String hex(BytesRef value) {
    if(value==null)return "-";if(value.length==0)return "e";
    return HexFormat.of().formatHex(value.bytes,value.offset,value.offset+value.length);
  }
  static class Input extends DataInput {
    final byte[] bytes;int pos;
    Input(byte[] bytes){this.bytes=bytes;}
    @Override public byte readByte()throws IOException {if(pos==bytes.length)throw new EOFException("input");return bytes[pos++];}
    @Override public void readBytes(byte[] b,int o,int n)throws IOException {
      if(n>bytes.length-pos)throw new EOFException("input");System.arraycopy(bytes,pos,b,o,n);pos+=n;
    }
    @Override public void skipBytes(long n)throws IOException {
      if(n<0||n>bytes.length-pos)throw new EOFException("input");pos+=Math.toIntExact(n);
    }
  }
  static class Probe extends TermVectorsWriter {
    final List<String> log;String fail="-";int closes;
    Probe(List<String> log){this.log=log;}
    void call(String name,String detail)throws IOException {
      log.add(name+detail);
      if(fail.equals(name)||fail.equals(name+"#"+closes)){fail="-";throw new IOException("delegate "+name);}
    }
    @Override public void startDocument(int n)throws IOException {call("doc",":"+n);}
    @Override public void finishDocument()throws IOException {call("enddoc","");}
    @Override public void startField(FieldInfo f,int n,boolean p,boolean o,boolean y)throws IOException {
      call("field",":"+f.number+":"+n+":"+p+":"+o+":"+y);
    }
    @Override public void finishField()throws IOException {call("endfield","");}
    @Override public void startTerm(BytesRef t,int n)throws IOException {call("term",":"+hex(t)+":"+n);}
    @Override public void finishTerm()throws IOException {call("endterm","");}
    @Override public void addPosition(int p,int s,int e,BytesRef y)throws IOException {call("position",":"+p+":"+s+":"+e+":"+hex(y));}
    @Override public void finish(int n)throws IOException {call("finish",":"+n);}
    @Override public void close()throws IOException {closes++;call("close",":"+closes);}
    @Override public long ramBytesUsed(){log.add("ram");return 37;}
    @Override public Collection<Accountable> getChildResources(){throw new AssertionError("delegate child resources called");}
    @Override public void addProx(int n,DataInput p,DataInput o){throw new AssertionError("delegate prox called");}
    @Override public int merge(MergeState state){throw new AssertionError("delegate merge called");}
  }
  interface Operation {void run()throws Exception;}
  static String outcome(Operation op)throws Exception {
    try {op.run();return "ok";}catch(AssertionError e){return "assert";}
    catch(IOException e){return "io:"+e.getMessage();}
  }
  static void emit(String kind,String name,String program,List<String> statuses,List<String> log) {
    OUT.append(kind).append('\t').append(name).append('\t').append(program).append('\t')
        .append(String.join(";",statuses)).append('\t').append(log.isEmpty()?"-":String.join(";",log)).append('\n');
  }
  static void command(TermVectorsWriter w,Probe p,String token)throws Exception {
    String[] a=token.split(":");
    switch(a[0]) {
      case "fail"->p.fail=a[1];
      case "doc"->w.startDocument(Integer.parseInt(a[1]));
      case "enddoc"->w.finishDocument();
      case "field"->w.startField(FIELD,Integer.parseInt(a[1]),a[2].contains("p"),a[2].contains("o"),a[2].contains("y"));
      case "endfield"->w.finishField();
      case "term"->w.startTerm(new BytesRef(new byte[]{0,10,92,(byte)255}),Integer.parseInt(a[1]));
      case "endterm"->w.finishTerm();
      case "position"->w.addPosition(7,10,13,new BytesRef(new byte[0]));
      case "finish"->w.finish(Integer.parseInt(a[1]));
      case "close"->w.close();
      case "resources"->{if(w.ramBytesUsed()!=37||w.getChildResources().size()!=1||w.getChildResources().iterator().next()!=p)throw new AssertionError("identity");}
      case "prox","shortpos","shortoff"->{
        Input pi=new Input(a[0].equals("shortpos")?new byte[]{3}:POS);
        Input oi=new Input(a[0].equals("shortoff")?new byte[]{0}:OFF);
        String result=outcome(()->w.addProx(3,pi,oi));p.log.add("inputs:"+pi.pos+":"+oi.pos);
        if(!result.equals("ok"))throw new IOException(result);
      }
      default->throw new IllegalArgumentException(token);
    }
  }
  static void program(String name,String program)throws Exception {
    List<String> log=new ArrayList<>(),statuses=new ArrayList<>();Probe p=new Probe(log);TermVectorsWriter w=writer(p);
    for(String token:program.split(";"))statuses.add(outcome(()->command(w,p,token)));
    emit("writer",name,program,statuses,log);
  }
  static void writerCases()throws Exception {
    program("empty","finish:0;close;resources");
    program("zero-fields","doc:0;enddoc;finish:1;finish:1;close");
    for(String flags:new String[]{"n","p","o","y","poy"})
      program("flags-"+flags,"doc:1;field:1:"+flags+";term:1;"+(flags.equals("n")?"":"position;")+"endterm;endfield;enddoc;finish:1");
    program("multiple","doc:2;field:2:n;term:7;endterm;term:0;endterm;endfield;field:0:n;endfield;enddoc;doc:0;enddoc;finish:2");
    for(String token:new String[]{"enddoc","field:0:n","endfield","term:1","endterm","position","finish:1"})program("initial-"+token,token);
    program("doc-reentry","doc:0;doc:0;enddoc;doc:0;enddoc;finish:2");
    program("field-reentry","doc:1;field:0:n;field:0:n;endfield;enddoc");
    program("term-reentry","doc:1;field:1:n;term:0;term:0;endterm;endfield;enddoc");
    program("missing-field","doc:1;enddoc;field:0:n;endfield;enddoc;finish:1");
    program("extra-field","doc:0;field:0:n;endfield;enddoc;doc:0");
    program("missing-term","doc:1;field:1:n;endfield;term:0;endterm;endfield;enddoc");
    program("extra-term","doc:1;field:0:n;term:0;endterm;endfield");
    program("missing-position","doc:1;field:1:p;term:1;endterm;position;endterm;endfield;enddoc");
    program("extra-position","doc:1;field:1:p;term:0;position;endterm");
    program("negative-counts","doc:-1;field:-1:p;term:-1;endterm;endfield;enddoc;finish:1");
    program("finish-open-doc","doc:0;finish:1;enddoc;finish:0;finish:1");
    program("finish-open-field","doc:1;field:0:n;enddoc;finish:1");
    program("finish-open-term","doc:1;field:1:n;term:0;finish:1");
    for(String operation:new String[]{"doc","field","term","position","endterm","endfield","enddoc","finish"}) {
      String prefix=switch(operation) {
        case "doc"->"";case "field"->"doc:1;";case "term"->"doc:1;field:1:p;";
        case "position"->"doc:1;field:1:p;term:1;";
        case "endterm"->"doc:1;field:1:p;term:1;position;";
        case "endfield"->"doc:1;field:1:n;term:0;endterm;";
        case "enddoc"->"doc:0;";default->"doc:0;enddoc;";
      };
      String token=switch(operation){case "doc"->"doc:0";case "field"->"field:1:p";case "term"->"term:1";case "finish"->"finish:1";default->operation;};
      String suffix=switch(operation) {
        case "doc"->";enddoc;finish:1";
        case "field"->";term:1;position;endterm;endfield;enddoc;finish:1";
        case "term"->";position;endterm;endfield;enddoc;finish:1";
        case "position"->";endterm;endfield;enddoc;finish:1";
        case "endterm"->";endfield;enddoc;finish:1";
        case "endfield"->";enddoc;finish:1";
        case "enddoc"->";finish:1";default->"";
      };
      program("retry-"+operation,prefix+"fail:"+operation+";"+token+";"+token+suffix);
    }
    for(String failure:new String[]{"close#1","close#2"})program(failure,"fail:"+failure+";close;close");
    for(String op:new String[]{"prox","shortpos","shortoff"})
      program(op,"doc:1;field:1:poy;term:3;"+op+";endterm;endfield;enddoc;finish:1");
    program("prox-position-error","doc:1;field:1:poy;term:3;fail:position;prox;endterm");
  }
  static class Reader extends TermVectorsReader {
    final List<String> log;String fail="-";int closes;
    Reader(List<String> log){this.log=log;}
    void call(String name)throws IOException {log.add(name);if(fail.equals(name)){fail="-";throw new IOException("delegate "+name);}}
    @Override public Fields get(int doc)throws IOException {call("get:"+doc);return null;}
    @Override public void checkIntegrity()throws IOException {call("integrity");}
    @Override public TermVectorsReader clone(){log.add("clone");return new Reader(log);}
    @Override public TermVectorsReader getMergeInstance(){log.add("merge");return this;}
    @Override public void close()throws IOException {call("close:"+(++closes));}
    @Override public String toString(){return "ProbeReader";}
  }
  static void readerCase(String kind)throws Exception {
    List<String> log=new ArrayList<>(),statuses=new ArrayList<>();Reader p=new Reader(log);TermVectorsReader w=reader(p);
    if(kind.startsWith("fail-"))p.fail=kind.substring(5);
    statuses.add(outcome(()->{
      switch(kind) {
        case "clone"->readerClone(w).close();
        case "merge"->w.getMergeInstance().close();
        case "close","fail-close:1","fail-close:2"->w.close();
        case "integrity","fail-integrity"->w.checkIntegrity();
        case "string"->{if(!w.toString().equals("AssertingTermVectorsReader(ProbeReader)"))throw new AssertionError("string");}
        default->{if(w.get(0)!=null)throw new AssertionError("null get");}
      }
    }));emit("reader",kind,"-",statuses,log);
  }
  static TermVectorsReader readerClone(TermVectorsReader r){return r.clone();}
  static void document(TermVectorsWriter w)throws IOException {
    w.startDocument(1);w.startField(field(0),1,true,true,true);w.startTerm(new BytesRef(new byte[]{0,10,92,(byte)255}),3);
    w.addProx(3,new Input(REAL_POS),new Input(OFF));w.finishTerm();w.finishField();w.finishDocument();
    w.startDocument(0);w.finishDocument();w.finish(2);
  }
  static void assign(MergeState s,String name,Object value)throws Exception {var f=MergeState.class.getField(name);f.setAccessible(true);f.set(s,value);}
  static MergeState state(Directory dir,TermVectorsReader r,boolean sorted)throws Exception {
    var c=MergeState.class.getDeclaredConstructor(List.class,SegmentInfo.class,InfoStream.class,Executor.class,MergePolicy.OneMerge.class);c.setAccessible(true);
    MergeState s=c.newInstance(List.of(),info(dir,-1),InfoStream.NO_OUTPUT,(Executor)Runnable::run,null);
    int[][] maps=sorted?new int[][]{{2,-1},{0},{1,3}}:new int[][]{{0,-1},{1},{2,3}};
    MergeState.DocMap[] dm=new MergeState.DocMap[3];for(int i=0;i<3;i++){final int[] map=maps[i];dm[i]=new MergeState.DocMap(){@Override public int get(int d){return map[d];}};}
    assign(s,"termVectorsReaders",new TermVectorsReader[]{r,null,r});assign(s,"maxDocs",new int[]{2,1,2});assign(s,"docMaps",dm);
    s.mergeFieldInfos=new FieldInfos(new FieldInfo[]{FIELD});s.needsIndexSort=sorted;return s;
  }
  public static void main(String[] args)throws Exception {
    if(!GenAssertingTermVectors.class.desiredAssertionStatus())throw new AssertionError("run with -ea");
    writerCases();for(String k:new String[]{"get","fail-get:0","integrity","fail-integrity","clone","merge","string","close","fail-close:1","fail-close:2"})readerCase(k);
    Path out=Path.of(args[0]);Files.createDirectories(out);
    try(Directory dir=new ByteBuffersDirectory()) {
      TermVectorsFormat f=format(new SimpleTextTermVectorsFormat());TermVectorsWriter w=f.vectorsWriter(dir,info(dir,2),IOContext.DEFAULT);document(w);w.close();
      try(TermVectorsReader r=f.vectorsReader(dir,info(dir,2),new FieldInfos(new FieldInfo[]{field(0)}),IOContext.DEFAULT)) {
        r.checkIntegrity();if(r.get(1)!=null||r.get(0).terms("f").size()!=1)throw new AssertionError("readback");
        TermsEnum te=r.get(0).terms("f").iterator();te.next();PostingsEnum p=te.postings(null,PostingsEnum.ALL);p.nextDoc();
        int[] ps={1,3,3},ss={0,3,6},es={2,6,6};for(int i=0;i<3;i++){if(p.nextPosition()!=ps[i]||p.startOffset()!=ss[i]||p.endOffset()!=es[i])throw new AssertionError("occurrence");if(i==0?!new BytesRef(new byte[]{0,10}).equals(p.getPayload()):p.getPayload()!=null)throw new AssertionError("payload");}
        for(boolean sorted:new boolean[]{false,true}) {
          List<String> log=new ArrayList<>(),statuses=new ArrayList<>();Probe probe=new Probe(log);TermVectorsWriter target=writer(probe);
          statuses.add(outcome(()->{int n=target.merge(state(dir,r,sorted));log.add("merged:"+n);}));emit("merge",sorted?"sorted":"sequential","-",statuses,log);
        }
        TermVectorsReader clone=r.clone();clone.close();if(r.get(0)==null)throw new AssertionError("clone close invalidates original");
      }
      try(var input=dir.openInput("_atv.tvc",IOContext.DEFAULT)){byte[] bytes=new byte[Math.toIntExact(input.length())];input.readBytes(bytes,0,bytes.length);Files.write(out.resolve("asserting_term_vectors.tvc"),bytes);}
    }
    Files.writeString(out.resolve("asserting_term_vectors.tsv"),OUT,StandardCharsets.UTF_8);
  }
}
