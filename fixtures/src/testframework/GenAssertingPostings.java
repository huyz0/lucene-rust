import java.io.IOException;
import java.nio.file.*;
import java.util.*;
import org.apache.lucene.codecs.*;
import org.apache.lucene.codecs.lucene104.*;
import org.apache.lucene.document.*;
import org.apache.lucene.index.*;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.store.*;
import org.apache.lucene.tests.codecs.asserting.AssertingPostingsFormat;
import org.apache.lucene.util.*;

/** Actual asserting postings wrappers; run with -ea. No copied assertion logic. */
public class GenAssertingPostings {
  static FieldsConsumer consumer(SegmentWriteState state,FieldsConsumer delegate) throws Exception {
    var c=Class.forName("org.apache.lucene.tests.codecs.asserting.AssertingPostingsFormat$AssertingFieldsConsumer")
        .getDeclaredConstructor(SegmentWriteState.class,FieldsConsumer.class);
    c.setAccessible(true);return (FieldsConsumer)c.newInstance(state,delegate);
  }
  static FieldsProducer producer(FieldsProducer delegate) throws Exception {
    var c=Class.forName("org.apache.lucene.tests.codecs.asserting.AssertingPostingsFormat$AssertingFieldsProducer")
        .getDeclaredConstructor(FieldsProducer.class);
    c.setAccessible(true);return (FieldsProducer)c.newInstance(delegate);
  }
  static AssertingPostingsFormat format(PostingsFormat delegate) throws Exception {
    var f=new AssertingPostingsFormat();var member=AssertingPostingsFormat.class.getDeclaredField("in");
    member.setAccessible(true);member.set(f,delegate);return f;
  }
  interface Action {void run() throws Exception;}
  static String outcome(Action action) throws Exception {
    try {action.run();return "ok";}catch(AssertionError e){return "assert";}catch(IOException e){return "io:"+e.getMessage();}
  }
  static FieldInfo field(String name,int number,IndexOptions options) {
    return new FieldInfo(name,number,false,true,false,options,DocValuesType.NONE,DocValuesSkipIndexType.NONE,
        -1,Map.of(),0,0,0,0,VectorEncoding.FLOAT32,VectorSimilarityFunction.EUCLIDEAN,false,false);
  }
  static class Case {
    String name;IndexOptions options=IndexOptions.DOCS_AND_FREQS_AND_POSITIONS_AND_OFFSETS;
    String[] fields={"f"};byte[][] terms={{97},{(byte)128}};int[] docs={0,2},positions={0,2},starts={0,4},ends={2,6};
    int freq=2;boolean termFreqs=true,termPositions=true,termOffsets=true,payloads=true;
    boolean nullTerms,delegateFail,delegateScan,iteratorFail,nextFail,postingsFail,docFail;
    Case(String name){this.name=name;}
  }
  static class ProbeFields extends Fields {
    final Case c;final List<String> events;
    ProbeFields(Case c,List<String> events){this.c=c;this.events=events;}
    @Override public Iterator<String> iterator(){events.add("fields");return Arrays.asList(c.fields).iterator();}
    @Override public int size(){events.add("size");return c.fields.length;}
    @Override public Terms terms(String field){events.add("terms:"+field);return c.nullTerms?null:new ProbeTerms(c,events);}
  }
  static class ProbeTerms extends Terms {
    final Case c;final List<String> events;
    ProbeTerms(Case c,List<String> events){this.c=c;this.events=events;}
    @Override public TermsEnum iterator() throws IOException {
      events.add("iterator");if(c.iteratorFail)throw new IOException("iterator failure");
      return new BaseTermsEnum(){int at=-1;PostingsEnum previous;
        @Override public BytesRef next() throws IOException {events.add("term-next");if(c.nextFail)throw new IOException("next failure");return ++at<c.terms.length?new BytesRef(c.terms[at]):null;}
        @Override public BytesRef term(){return at>=0&&at<c.terms.length?new BytesRef(c.terms[at]):null;}
        @Override public int docFreq(){events.add("doc-freq");return c.docs.length;}
        @Override public long totalTermFreq(){events.add("total-freq");return (long)c.docs.length*c.freq;}
        @Override public SeekStatus seekCeil(BytesRef text){throw new UnsupportedOperationException();}
        @Override public void seekExact(long ord){throw new UnsupportedOperationException();}
        @Override public long ord(){return at;}
        @Override public ImpactsEnum impacts(int flags){throw new UnsupportedOperationException();}
        @Override public PostingsEnum postings(PostingsEnum reuse,int flags) throws IOException {
          events.add("postings:"+flags+":"+(reuse!=null)+":"+(reuse==previous));
          if(c.postingsFail)throw new IOException("postings failure");
          previous=new PostingsEnum(){int doc=-1,di=-1,pi;
            @Override public int docID(){return doc;}
            @Override public int nextDoc() throws IOException {events.add("doc-next");if(c.docFail)throw new IOException("doc failure");pi=0;return doc=++di<c.docs.length?c.docs[di]:NO_MORE_DOCS;}
            @Override public int advance(int target) throws IOException {while(nextDoc()<target){}return doc;}
            @Override public long cost(){return c.docs.length;}
            @Override public int freq(){events.add("freq");return c.freq;}
            @Override public int nextPosition(){events.add("position");return c.positions[pi++];}
            @Override public int startOffset(){events.add("start");return c.starts[pi-1];}
            @Override public int endOffset(){events.add("end");return c.ends[pi-1];}
            @Override public BytesRef getPayload(){events.add("payload");return null;}
          };return previous;
        }
      };
    }
    @Override public long size(){events.add("term-size");return c.terms.length;}
    @Override public long getSumTotalTermFreq(){events.add("sum-total");return (long)c.terms.length*c.docs.length*c.freq;}
    @Override public long getSumDocFreq(){events.add("sum-doc");return (long)c.terms.length*c.docs.length;}
    @Override public int getDocCount(){events.add("doc-count");return c.docs.length;}
    @Override public boolean hasFreqs(){events.add("has-freqs");return c.termFreqs;}
    @Override public boolean hasPositions(){events.add("has-positions");return c.termPositions;}
    @Override public boolean hasOffsets(){events.add("has-offsets");return c.termOffsets;}
    @Override public boolean hasPayloads(){events.add("has-payloads");return c.payloads;}
  }
  static List<Case> cases(){
    List<Case> all=new ArrayList<>();Case c;
    all.add(new Case("full"));
    c=new Case("docs");c.options=IndexOptions.DOCS;c.termPositions=false;c.termOffsets=false;c.payloads=false;all.add(c);
    c=new Case("freqs");c.options=IndexOptions.DOCS_AND_FREQS;c.termPositions=false;c.termOffsets=false;c.payloads=false;all.add(c);
    c=new Case("positions");c.options=IndexOptions.DOCS_AND_FREQS_AND_POSITIONS;c.termOffsets=false;c.payloads=false;all.add(c);
    c=new Case("positions-payloads");c.options=IndexOptions.DOCS_AND_FREQS_AND_POSITIONS;c.termOffsets=false;all.add(c);
    c=new Case("offsets-no-payloads");c.payloads=false;all.add(c);
    c=new Case("empty-fields");c.fields=new String[0];all.add(c);
    c=new Case("null-terms");c.nullTerms=true;all.add(c);
    c=new Case("empty-terms");c.terms=new byte[0][];all.add(c);
    c=new Case("empty-docs");c.docs=new int[0];all.add(c);
    c=new Case("unknown-field");c.fields=new String[]{"missing"};all.add(c);
    c=new Case("duplicate-fields");c.fields=new String[]{"f","f"};all.add(c);
    c=new Case("reversed-fields");c.fields=new String[]{"g","f"};all.add(c);
    c=new Case("unicode-fields");c.fields=new String[]{"\uD800\uDC00","\uE000"};all.add(c);
    c=new Case("unicode-fields-reversed");c.fields=new String[]{"\uE000","\uD800\uDC00"};all.add(c);
    c=new Case("duplicate-terms");c.terms=new byte[][]{{97},{97}};all.add(c);
    c=new Case("reversed-terms");c.terms=new byte[][]{{(byte)128},{97}};all.add(c);
    c=new Case("empty-term-first");c.terms=new byte[][]{{},{97}};all.add(c);
    c=new Case("positions-mismatch");c.termPositions=false;all.add(c);
    c=new Case("offsets-mismatch");c.termOffsets=false;all.add(c);
    c=new Case("freqs-metadata-unchecked");c.termFreqs=false;all.add(c);
    c=new Case("doc-negative");c.docs=new int[]{-1};all.add(c);
    c=new Case("doc-duplicate");c.docs=new int[]{0,0};all.add(c);
    c=new Case("doc-descending");c.docs=new int[]{2,0};all.add(c);
    c=new Case("doc-maxdoc-unchecked");c.docs=new int[]{100};all.add(c);
    c=new Case("freq-zero");c.freq=0;all.add(c);
    c=new Case("position-descending");c.positions=new int[]{2,1};all.add(c);
    c=new Case("position-negative-one-allowed");c.positions=new int[]{-1,0};all.add(c);
    c=new Case("position-too-negative");c.positions=new int[]{-2,0};all.add(c);
    c=new Case("position-max");c.positions=new int[]{0,IndexWriter.MAX_POSITION};all.add(c);
    c=new Case("position-over-max");c.positions=new int[]{0,IndexWriter.MAX_POSITION+1};all.add(c);
    c=new Case("offset-end-before-start");c.ends=new int[]{-1,6};all.add(c);
    c=new Case("offset-descending");c.starts=new int[]{2,1};all.add(c);
    c=new Case("offset-negative-one-allowed");c.starts=new int[]{-1,0};c.ends=new int[]{-1,2};all.add(c);
    c=new Case("delegate-failure-before-invalid-fields");c.delegateFail=true;c.fields=new String[]{"missing"};all.add(c);
    c=new Case("delegate-enumerates-before-validation");c.delegateScan=true;all.add(c);
    c=new Case("iterator-error");c.iteratorFail=true;all.add(c);
    c=new Case("next-error");c.nextFail=true;all.add(c);
    c=new Case("postings-error");c.postingsFail=true;all.add(c);
    c=new Case("doc-error");c.docFail=true;all.add(c);
    return all;
  }
  static void consumerTrace(Path out) throws Exception {
    StringBuilder text=new StringBuilder();
    try(Directory directory=new ByteBuffersDirectory()){
      for(Case c:cases()){
        List<String> events=new ArrayList<>();ProbeFields fields=new ProbeFields(c,events);
        FieldInfos infos=new FieldInfos(new FieldInfo[]{field("f",0,c.options),field("g",1,c.options),field("\uD800\uDC00",2,c.options),field("\uE000",3,c.options)});
        FieldsConsumer delegate=new FieldsConsumer(){
          @Override public void write(Fields incoming,NormsProducer norms) throws IOException {
            events.add("delegate:"+(incoming==fields)+":"+(norms==null));
            if(c.delegateFail)throw new IOException("delegate failure");
            if(c.delegateScan)for(String name:incoming){Terms terms=incoming.terms(name);if(terms!=null){TermsEnum e=terms.iterator();while(e.next()!=null){}}}
          }
          @Override public void close(){events.add("close");}
        };
        FieldsConsumer wrapper=consumer(GenCrankyPostings.writeState(directory,infos,5),delegate);
        String status=outcome(()->wrapper.write(fields,null));
        text.append(c.name).append('\t').append(status).append('\t').append(String.join(",",events)).append('\n');
      }
      for(int failAt:new int[]{0,1,2}){
        List<String> events=new ArrayList<>();int[] calls={0};
        FieldsConsumer delegate=new FieldsConsumer(){@Override public void write(Fields f,NormsProducer n){}@Override public void close() throws IOException {events.add("close:"+ ++calls[0]);if(calls[0]==failAt)throw new IOException("close failure");}};
        FieldsConsumer wrapper=consumer(null,delegate);
        text.append("close-").append(failAt).append('\t').append(outcome(wrapper::close)).append('\t').append(String.join(",",events)).append('\n');
      }
    }
    Files.writeString(out.resolve("asserting_postings_consumer.tsv"),text);
  }
  static class ProbeProducer extends FieldsProducer {
    final List<String> events;final Case c=new Case("producer");int closes,failClose;boolean failTerms,failIntegrity;ProbeProducer merge;
    ProbeProducer(List<String> events){this.events=events;}
    @Override public Iterator<String> iterator(){events.add("producer-fields");return List.of("f").iterator();}
    @Override public int size(){events.add("producer-size");return 1;}
    @Override public Terms terms(String name) throws IOException {events.add("producer-terms:"+name);if(failTerms)throw new IOException("terms failure");return name.equals("f")?new ProbeTerms(c,events):null;}
    @Override public void checkIntegrity() throws IOException {events.add("integrity");if(failIntegrity)throw new IOException("integrity failure");}
    @Override public void close() throws IOException {events.add("pclose:"+ ++closes);if(closes==failClose)throw new IOException("producer close failure");}
    @Override public FieldsProducer getMergeInstance(){events.add("merge-instance");return merge==null?this:merge;}
  }
  static void producerTrace(Path out) throws Exception {
    StringBuilder text=new StringBuilder();
    for(String op:List.of("fields","size","absent","terms","iterate","twice","stats","integrity","integrity-error","terms-error","next-error","merge","merge-distinct","merge-close","close","close-first-error","close-second-error")){
      List<String> events=new ArrayList<>();ProbeProducer delegate=new ProbeProducer(events);FieldsProducer wrapper=producer(delegate);
      delegate.failTerms=op.equals("terms-error");delegate.failIntegrity=op.equals("integrity-error");delegate.c.nextFail=op.equals("next-error");
      delegate.failClose=op.equals("close-first-error")?1:op.equals("close-second-error")?2:0;
      if(op.equals("merge-distinct"))delegate.merge=new ProbeProducer(events);
      String status=outcome(()->{
        switch(op){
          case "fields" -> {Iterator<String> values=wrapper.iterator();events.add("same-fields:"+values.next());}
          case "size" -> events.add("value:"+wrapper.size());
          case "absent" -> events.add("absent:"+(wrapper.terms("missing")==null));
          case "terms","terms-error" -> events.add("wrapped:"+(wrapper.terms("f").getClass().getSimpleName()));
          case "iterate","twice","next-error" -> {for(int i=0;i<(op.equals("twice")?2:1);i++){TermsEnum e=wrapper.terms("f").iterator();while(e.next()!=null){}}}
          case "stats" -> {Terms t=wrapper.terms("f");events.add("count:"+t.getDocCount());events.add("sum:"+t.getSumTotalTermFreq());}
          case "integrity","integrity-error" -> wrapper.checkIntegrity();
          case "merge","merge-distinct","merge-close" -> {FieldsProducer m=wrapper.getMergeInstance();events.add("wrapped-merge:"+(m!=delegate)+(":"+m.getClass().getSimpleName()));if(op.equals("merge-close"))m.close();else m.terms("f").iterator().next();}
          default -> wrapper.close();
        }
      });
      text.append(op).append('\t').append(status).append('\t').append(String.join(",",events)).append('\n');
    }
    Files.writeString(out.resolve("asserting_postings_producer.tsv"),text);
  }
  static void real(Path out) throws Exception {
    try(Directory source=new ByteBuffersDirectory()){
      try(IndexWriter writer=new IndexWriter(source,new IndexWriterConfig().setUseCompoundFile(false).setCodec(new Lucene104Codec()))){
        String[] names={"docs","freqs","positions","rich","empty","\uD800\uDC00","\uE000"};
        IndexOptions[] options={IndexOptions.DOCS,IndexOptions.DOCS_AND_FREQS,IndexOptions.DOCS_AND_FREQS_AND_POSITIONS,IndexOptions.DOCS_AND_FREQS_AND_POSITIONS_AND_OFFSETS,IndexOptions.DOCS,IndexOptions.DOCS_AND_FREQS_AND_POSITIONS_AND_OFFSETS,IndexOptions.DOCS_AND_FREQS_AND_POSITIONS_AND_OFFSETS};
        for(int doc=0;doc<9;doc++){Document document=new Document();for(int i=0;i<names.length;i++){if(doc%3==2&&(i==1||i==2))continue;document.add(new Field(names[i],GenCrankyPostings.tokens(doc,i==4,i>=3),GenCrankyPostings.type(options[i],i==3)));}writer.addDocument(document);}writer.forceMerge(1);
      }
      try(DirectoryReader reader=DirectoryReader.open(source)){
        SegmentReader leaf=(SegmentReader)reader.leaves().get(0).reader();FieldInfos fields=leaf.getFieldInfos();
        PostingsFormat nativeFormat=new Lucene104PostingsFormat();Map<String,long[]> columns=GenCrankyPostings.norms(leaf.getNormsReader(),fields,9);
        for(int kind=0;kind<3;kind++)try(Directory target=new ByteBuffersDirectory()){
          String prefix="asserting_postings"+(kind==0?"":kind==1?"_merge_sequential":"_merge_sorted");
          int maxDoc=kind==0?9:16;PostingsFormat asserting=format(nativeFormat);Map<String,long[]> outputColumns=columns;
          try(FieldsConsumer writer=asserting.fieldsConsumer(GenCrankyPostings.writeState(target,fields,maxDoc))){
            if(kind==0)writer.write(leaf.getPostingsReader(),GenCrankyPostings.norms(columns));
            else{
              int[][] maps=new int[2][9];for(int side=0;side<2;side++)for(int doc=0;doc<9;doc++)maps[side][doc]=doc==1?-1:kind==2?2*(doc==0?0:doc-1)+side:8*side+(doc==0?0:doc-1);
              outputColumns=new TreeMap<>();for(var entry:columns.entrySet()){long[] values=new long[16];Arrays.fill(values,-1);for(int side=0;side<2;side++)for(int doc=0;doc<9;doc++)if(maps[side][doc]>=0)values[maps[side][doc]]=entry.getValue()[doc];outputColumns.put(entry.getKey(),values);}
              writer.merge(GenCrankyPostings.mergeState(target,fields,new FieldsProducer[]{leaf.getPostingsReader(),leaf.getPostingsReader()},new int[]{9,9},maps,kind==2),GenCrankyPostings.norms(outputColumns));
            }
          }
          GenCrankyPostings.save(out,prefix,target);
          try(FieldsProducer output=asserting.fieldsProducer(GenCrankyPostings.readState(target,fields,maxDoc))){output.checkIntegrity();Files.writeString(out.resolve(prefix+".tsv"),GenCrankyPostings.snapshot(output,fields,maxDoc,GenCrankyPostings.norms(outputColumns)));}
          if(kind==0){new Lucene104Codec().fieldInfosFormat().write(target,GenCrankyPostings.segment(target,9),"",fields,IOContext.DEFAULT);try(IndexInput input=target.openInput("_cp.fnm",IOContext.DEFAULT)){byte[] data=new byte[Math.toIntExact(input.length())];input.readBytes(data,0,data.length);Files.write(out.resolve(prefix+".fnm"),data);}}
        }
      }
    }
  }
  public static void main(String[] args) throws Exception {Path out=Path.of(args[0]);Files.createDirectories(out);consumerTrace(out);producerTrace(out);real(out);}
}
