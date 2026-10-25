import java.io.IOException;
import java.nio.file.*;
import java.util.*;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.tokenattributes.*;
import org.apache.lucene.codecs.*;
import org.apache.lucene.codecs.lucene104.*;
import org.apache.lucene.document.*;
import org.apache.lucene.index.*;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.store.*;
import org.apache.lucene.util.*;

/** Actual package-private CrankyPostingsFormat, fixed-id Lucene104 byte fixtures. */
public class GenCrankyPostings {
  static final byte[] ID={0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
  static final int DOCS=9;
  static final String SUFFIX="s";
  static PostingsFormat wrap(PostingsFormat delegate,Random random) throws Exception {
    var constructor=Class.forName("org.apache.lucene.tests.codecs.cranky.CrankyPostingsFormat")
        .getDeclaredConstructor(PostingsFormat.class,Random.class);
    constructor.setAccessible(true);return (PostingsFormat)constructor.newInstance(delegate,random);
  }
  static SegmentInfo segment(Directory directory,int maxDoc) {
    return new SegmentInfo(directory,Version.fromBits(10,5,0),null,"_cp",maxDoc,false,false,
        new Lucene104Codec(),Map.of(),ID.clone(),Map.of(),null);
  }
  static SegmentWriteState writeState(Directory directory,FieldInfos fields,int maxDoc) {
    return new SegmentWriteState(InfoStream.NO_OUTPUT,directory,segment(directory,maxDoc),fields,null,IOContext.DEFAULT,SUFFIX);
  }
  static SegmentReadState readState(Directory directory,FieldInfos fields,int maxDoc) {
    return new SegmentReadState(directory,segment(directory,maxDoc),fields,IOContext.DEFAULT,SUFFIX);
  }
  static MergeState mergeState(Directory directory,FieldInfos fields,FieldsProducer[] readers,
      int[] maxDocs,int[][] maps,boolean sorted) throws Exception {
    var constructor=MergeState.class.getDeclaredConstructor(List.class,SegmentInfo.class,
        InfoStream.class,java.util.concurrent.Executor.class,MergePolicy.OneMerge.class);
    constructor.setAccessible(true);
    MergeState state=constructor.newInstance(List.of(),segment(directory,-1),InfoStream.NO_OUTPUT,null,null);
    MergeState.DocMap[] docMaps=new MergeState.DocMap[maps.length];
    for(int i=0;i<maps.length;i++){final int[] map=maps[i];docMaps[i]=new MergeState.DocMap(){public int get(int doc){return map[doc];}};}
    Map<String,Object> members=new HashMap<>();members.put("fieldsProducers",readers);
    members.put("maxDocs",maxDocs);members.put("docMaps",docMaps);
    FieldInfos[] sourceFields=new FieldInfos[readers.length];Arrays.fill(sourceFields,fields);
    members.put("fieldInfos",sourceFields);members.put("needsIndexSort",sorted);
    for(var member:members.entrySet()){var field=MergeState.class.getField(member.getKey());field.setAccessible(true);field.set(state,member.getValue());}
    state.mergeFieldInfos=fields;return state;
  }
  static class State {
    final List<String> events=new ArrayList<>();boolean fail;
    void event(String event) throws IOException {events.add(event);if(fail)throw new IOException("delegate failure");}
  }
  static final class RecordingProducer extends FieldsProducer {
    final State state;RecordingProducer(State state){this.state=state;}
    @Override public Iterator<String> iterator(){return List.of("f").iterator();}
    @Override public Terms terms(String field){return null;}
    @Override public int size(){return 1;}
    @Override public void checkIntegrity() throws IOException {state.event("integrity");}
    @Override public void close() throws IOException {state.event("readerClose");}
  }
  static final class RecordingNorms extends NormsProducer {
    @Override public NumericDocValues getNorms(FieldInfo field){return null;}
    @Override public void checkIntegrity(){}
    @Override public void close(){}
  }
  static final class RecordingFormat extends PostingsFormat {
    final State state;final RecordingProducer producer;final RecordingNorms norms=new RecordingNorms();
    RecordingFormat(State state){super("RecordingPostings");this.state=state;producer=new RecordingProducer(state);}
    @Override public FieldsConsumer fieldsConsumer(SegmentWriteState ignored) throws IOException {
      state.event("consumer");return new FieldsConsumer(){
        @Override public void write(Fields fields,NormsProducer n) throws IOException {state.event("write:"+(fields==producer)+":"+(n==norms));}
        @Override public void close() throws IOException {state.event("writerClose");}
      };
    }
    @Override public FieldsProducer fieldsProducer(SegmentReadState ignored) throws IOException {state.event("producer");return producer;}
  }
  static String trace() throws Exception {
    StringBuilder out=new StringBuilder();
    for(long seed:new long[]{0,1,42,-1,Long.MIN_VALUE,Long.MAX_VALUE})try(var directory=new ByteBuffersDirectory()) {
      State state=new State();RecordingFormat delegate=new RecordingFormat(state);Random random=new Random(seed);
      PostingsFormat format=wrap(delegate,random);
      // Acquire the consumer before fault scheduling and keep its exact RNG draw in the stream.
      FieldsConsumer consumer=null;
      while(consumer==null)try{consumer=format.fieldsConsumer(null);}catch(IOException expected){}
      FieldInfos fields=new FieldInfos(new FieldInfo[0]);
      MergeState merge=mergeState(directory,fields,new FieldsProducer[]{delegate.producer,null,delegate.producer},
          new int[]{2,0,2},new int[][]{{0,1},{},{2,3}},false);
      for(int step=0;step<2400;step++) {
        int op=step%8;state.events.clear();state.fail=step%11==0;String status="ok";
        try {
          switch(op){
            case 0->status=format.getName();
            case 1->format.fieldsConsumer(null);
            case 2->consumer.write(delegate.producer,delegate.norms);
            case 3->consumer.close();
            case 4->status=Boolean.toString(format.fieldsProducer(null)==delegate.producer);
            case 5->delegate.producer.checkIntegrity();
            case 6->delegate.producer.close();
            case 7->consumer.merge(merge,delegate.norms);
          }
        }catch(IOException error){status=error.getMessage();}
        out.append(seed).append('\t').append(step).append('\t').append(op).append('\t').append(status)
            .append('\t').append(String.join(",",state.events)).append('\t').append(random.nextInt()).append('\n');
      }
    }
    return out.toString();
  }
  static TokenStream tokens(int doc,boolean empty,boolean payloads) {
    return new TokenStream(){
      final CharTermAttribute term=addAttribute(CharTermAttribute.class);
      final PositionIncrementAttribute increment=addAttribute(PositionIncrementAttribute.class);
      final OffsetAttribute offset=addAttribute(OffsetAttribute.class);
      final PayloadAttribute payload=addAttribute(PayloadAttribute.class);int at;
      @Override public void reset() throws IOException {super.reset();at=0;}
      @Override public boolean incrementToken(){
        int count=empty?0:(doc==1?4:3);if(at==count)return false;clearAttributes();
        term.append(at==3?"gone":at==1?"bb":"aa");increment.setPositionIncrement(at==1?2:1);
        offset.setOffset(at*4,at*4+2);
        payload.setPayload(payloads&&at%2==0?new BytesRef(new byte[]{(byte)doc,(byte)(at+1),(byte)255}):null);
        at++;return true;
      }
      @Override public void end() throws IOException {super.end();offset.setOffset(at*4,at*4);}
    };
  }
  static FieldType type(IndexOptions options,boolean norms) {
    FieldType type=new FieldType();type.setTokenized(true);type.setIndexOptions(options);type.setOmitNorms(!norms);type.freeze();return type;
  }
  static String hex(BytesRef value) {return value==null||value.length==0?"-":HexFormat.of().formatHex(value.bytes,value.offset,value.offset+value.length);}
  static String snapshot(FieldsProducer producer,FieldInfos infos,int maxDoc,NormsProducer norms) throws Exception {
    StringBuilder out=new StringBuilder("maxDoc\t"+maxDoc+"\n");
    List<FieldInfo> sorted=new ArrayList<>();for(FieldInfo field:infos)if(field.getIndexOptions()!=IndexOptions.NONE)sorted.add(field);
    sorted.sort(Comparator.comparing(field->field.name));
    for(FieldInfo field:sorted) {
      out.append("field\t").append(field.name).append('\t').append(field.number).append('\t')
          .append(field.getIndexOptions().ordinal()).append('\t').append(field.hasPayloads()).append('\t').append(field.omitsNorms()).append('\n');
      if(!field.omitsNorms()&&norms!=null){NumericDocValues values=norms.getNorms(field);for(int doc=values.nextDoc();doc!=DocIdSetIterator.NO_MORE_DOCS;doc=values.nextDoc())out.append("norm\t").append(field.name).append('\t').append(doc).append('\t').append(values.longValue()).append('\n');}
      Terms terms=producer.terms(field.name);if(terms==null)continue;TermsEnum walk=terms.iterator();
      while(walk.next()!=null){String term=hex(walk.term());PostingsEnum postings=walk.postings(null,PostingsEnum.ALL);
        for(int doc=postings.nextDoc();doc!=DocIdSetIterator.NO_MORE_DOCS;doc=postings.nextDoc()){
          StringJoiner positions=new StringJoiner(","),offsets=new StringJoiner(","),payloads=new StringJoiner(",");
          int freq=postings.freq();if(field.getIndexOptions().compareTo(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS)>=0)
            for(int i=0;i<freq;i++){positions.add(Integer.toString(postings.nextPosition()));offsets.add(postings.startOffset()+":"+postings.endOffset());payloads.add(hex(postings.getPayload()));}
          out.append("posting\t").append(field.name).append('\t').append(term).append('\t').append(doc).append('\t').append(freq).append('\t')
              .append(positions.length()==0?"-":positions).append('\t').append(offsets.length()==0?"-":offsets).append('\t').append(payloads.length()==0?"-":payloads).append('\n');
        }
      }
    }
    return out.toString();
  }
  static Map<String,long[]> norms(NormsProducer producer,FieldInfos fields,int maxDoc) throws Exception {
    Map<String,long[]> result=new TreeMap<>();
    for(FieldInfo field:fields)if(field.hasNorms()){
      long[] values=new long[maxDoc];Arrays.fill(values,-1);NumericDocValues walk=producer.getNorms(field);
      for(int doc=walk.nextDoc();doc!=DocIdSetIterator.NO_MORE_DOCS;doc=walk.nextDoc())values[doc]=walk.longValue();result.put(field.name,values);
    }
    return result;
  }
  static NormsProducer norms(Map<String,long[]> columns) {
    return new NormsProducer(){
      @Override public NumericDocValues getNorms(FieldInfo field){long[] values=columns.get(field.name);return new NumericDocValues(){int doc=-1;
        @Override public int docID(){return doc;}
        @Override public long cost(){return values.length;}
        @Override public boolean advanceExact(int target){doc=target;return values[target]!=-1;}
        @Override public long longValue(){return values[doc];}
        @Override public int nextDoc(){return advance(doc+1);}
        @Override public int advance(int target){for(doc=target;doc<values.length;doc++)if(values[doc]!=-1)return doc;return doc=DocIdSetIterator.NO_MORE_DOCS;}
      };}
      @Override public void checkIntegrity(){}
      @Override public void close(){}
    };
  }
  static void save(Path out,String prefix,Directory directory) throws Exception {
    for(String extension:new String[]{"doc","psm","pos","pay","tim","tip","tmd"})try(IndexInput input=directory.openInput("_cp_s."+extension,IOContext.DEFAULT)){
      byte[] bytes=new byte[Math.toIntExact(input.length())];input.readBytes(bytes,0,bytes.length);Files.write(out.resolve(prefix+"."+extension),bytes);
    }
  }
  static void real(Path out) throws Exception {
    try(Directory source=new ByteBuffersDirectory()) {
      try(IndexWriter writer=new IndexWriter(source,new IndexWriterConfig().setUseCompoundFile(false).setCodec(new Lucene104Codec()))) {
        String[] names={"docs","freqs","positions","rich","empty"};
        IndexOptions[] options={IndexOptions.DOCS,IndexOptions.DOCS_AND_FREQS,IndexOptions.DOCS_AND_FREQS_AND_POSITIONS,
            IndexOptions.DOCS_AND_FREQS_AND_POSITIONS_AND_OFFSETS,IndexOptions.DOCS};
        for(int doc=0;doc<DOCS;doc++){Document document=new Document();for(int i=0;i<names.length;i++){
          if(doc%3==2&&(i==1||i==2))continue;
          document.add(new Field(names[i],tokens(doc,i==4,i==3),type(options[i],i==3)));
        }writer.addDocument(document);}writer.forceMerge(1);
      }
      try(DirectoryReader reader=DirectoryReader.open(source)) {
        SegmentReader leaf=(SegmentReader)reader.leaves().get(0).reader();FieldInfos fields=leaf.getFieldInfos();
        PostingsFormat nativeFormat=new Lucene104PostingsFormat();Map<String,long[]> columns=norms(leaf.getNormsReader(),fields,DOCS);
        String expected=snapshot(leaf.getPostingsReader(),fields,DOCS,norms(columns));
        try(Directory target=new ByteBuffersDirectory()){
          Random random=new Random(42);PostingsFormat cranky=wrap(nativeFormat,random);
          try(FieldsConsumer consumer=cranky.fieldsConsumer(writeState(target,fields,DOCS))){consumer.write(leaf.getPostingsReader(),norms(columns));}
          try(FieldsProducer producer=cranky.fieldsProducer(readState(target,fields,DOCS))){producer.checkIntegrity();
            if(!snapshot(producer,fields,DOCS,norms(columns)).equals(expected))throw new AssertionError("readback");}
          save(out,"cranky_postings",target);
          new Lucene104Codec().fieldInfosFormat().write(target,segment(target,DOCS),"",fields,IOContext.DEFAULT);
          try(IndexInput input=target.openInput("_cp.fnm",IOContext.DEFAULT)){byte[] bytes=new byte[Math.toIntExact(input.length())];input.readBytes(bytes,0,bytes.length);Files.write(out.resolve("cranky_postings.fnm"),bytes);}
          Files.writeString(out.resolve("cranky_postings_real.tsv"),expected+"rng\t"+random.nextInt()+"\n");
        }
        for(boolean sorted:new boolean[]{false,true})try(Directory target=new ByteBuffersDirectory()){
          int[][] maps=new int[2][DOCS];for(int side=0;side<2;side++)for(int doc=0;doc<DOCS;doc++)maps[side][doc]=doc==1?-1:(sorted?2*(doc==0?0:doc-1)+side:(DOCS-1)*side+(doc==0?0:doc-1));
          Map<String,long[]> mergedColumns=new TreeMap<>();for(var entry:columns.entrySet()){
            long[] values=new long[2*(DOCS-1)];Arrays.fill(values,-1);for(int side=0;side<2;side++)for(int doc=0;doc<DOCS;doc++)if(maps[side][doc]>=0)values[maps[side][doc]]=entry.getValue()[doc];mergedColumns.put(entry.getKey(),values);
          }
          Random random=new Random(42);PostingsFormat cranky=wrap(nativeFormat,random);
          MergeState state=mergeState(target,fields,new FieldsProducer[]{leaf.getPostingsReader(),leaf.getPostingsReader()},new int[]{DOCS,DOCS},maps,sorted);
          try(FieldsConsumer consumer=cranky.fieldsConsumer(writeState(target,fields,2*(DOCS-1)))){consumer.merge(state,norms(mergedColumns));}
          String prefix="cranky_postings_merge_"+(sorted?"sorted":"sequential");
          try(FieldsProducer producer=nativeFormat.fieldsProducer(readState(target,fields,2*(DOCS-1)))){
            producer.checkIntegrity();String snapshot=snapshot(producer,fields,2*(DOCS-1),norms(mergedColumns));
            if(snapshot.contains("676f6e65"))throw new AssertionError("deleted-only term survived");
            Files.writeString(out.resolve(prefix+".tsv"),snapshot+"rng\t"+random.nextInt()+"\n");
          }
          save(out,prefix,target);
        }
      }
    }
  }
  /** MultiFields orders names with String.compareTo (UTF-16), not UTF-8 bytes. */
  static void unicodeFields(Path out) throws Exception {
    String[] names={"\uD800\uDC00","\uE000"};
    try(Directory source=new ByteBuffersDirectory()) {
      try(IndexWriter writer=new IndexWriter(source,new IndexWriterConfig().setUseCompoundFile(false).setCodec(new Lucene104Codec()))) {
        for(int doc=0;doc<DOCS;doc++) {
          Document document=new Document();
          for(String name:names)document.add(new Field(name,tokens(doc,false,true),type(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS_AND_OFFSETS,false)));
          writer.addDocument(document);
        }
        writer.forceMerge(1);
      }
      try(DirectoryReader reader=DirectoryReader.open(source)) {
        SegmentReader leaf=(SegmentReader)reader.leaves().get(0).reader();FieldInfos fields=leaf.getFieldInfos();
        PostingsFormat nativeFormat=new Lucene104PostingsFormat();
        try(Directory target=new ByteBuffersDirectory()) {
          Random random=new Random(42);PostingsFormat cranky=wrap(nativeFormat,random);
          try(FieldsConsumer consumer=cranky.fieldsConsumer(writeState(target,fields,DOCS))){consumer.write(leaf.getPostingsReader(),null);}
          save(out,"cranky_postings_unicode",target);
          new Lucene104Codec().fieldInfosFormat().write(target,segment(target,DOCS),"",fields,IOContext.DEFAULT);
          try(IndexInput input=target.openInput("_cp.fnm",IOContext.DEFAULT)){byte[] bytes=new byte[Math.toIntExact(input.length())];input.readBytes(bytes,0,bytes.length);Files.write(out.resolve("cranky_postings_unicode.fnm"),bytes);}
          try(FieldsProducer producer=nativeFormat.fieldsProducer(readState(target,fields,DOCS))){
            producer.checkIntegrity();if(!snapshot(producer,fields,DOCS,null).equals(snapshot(leaf.getPostingsReader(),fields,DOCS,null)))throw new AssertionError("Unicode source readback");
            List<String> observed=new ArrayList<>();for(String name:producer)observed.add(name);
            if(!observed.equals(List.of(names)))throw new AssertionError("UTF-16 native field order: "+observed);
            StringBuilder order=new StringBuilder();for(String name:observed)order.append("field\t").append(name).append('\n');
            Files.writeString(out.resolve("cranky_postings_unicode_order.tsv"),order+"rng\t"+random.nextInt()+"\n");
          }
        }
        for(boolean sorted:new boolean[]{false,true})try(Directory target=new ByteBuffersDirectory()) {
          String prefix="cranky_postings_unicode_merge_"+(sorted?"sorted":"sequential");
          List<String> observed=new ArrayList<>();
          PostingsFormat recording=new PostingsFormat("UnicodeOrder") {
            @Override public FieldsProducer fieldsProducer(SegmentReadState state) throws IOException{return nativeFormat.fieldsProducer(state);}
            @Override public FieldsConsumer fieldsConsumer(SegmentWriteState state) throws IOException {
              FieldsConsumer delegate=nativeFormat.fieldsConsumer(state);
              return new FieldsConsumer(){
                @Override public void write(Fields input,NormsProducer norms) throws IOException {for(String field:input)observed.add(field);delegate.write(input,norms);}
                @Override public void close() throws IOException {delegate.close();}
              };
            }
          };
          int[][] maps=new int[2][DOCS];for(int side=0;side<2;side++)for(int doc=0;doc<DOCS;doc++)maps[side][doc]=doc==1?-1:(sorted?2*(doc==0?0:doc-1)+side:(DOCS-1)*side+(doc==0?0:doc-1));
          Random random=new Random(42);PostingsFormat cranky=wrap(recording,random);
          MergeState state=mergeState(target,fields,new FieldsProducer[]{leaf.getPostingsReader(),leaf.getPostingsReader()},new int[]{DOCS,DOCS},maps,sorted);
          try(FieldsConsumer consumer=cranky.fieldsConsumer(writeState(target,fields,2*(DOCS-1)))){consumer.merge(state,null);}
          if(!observed.equals(List.of(names)))throw new AssertionError("UTF-16 mapped field order: "+observed);
          StringBuilder order=new StringBuilder();for(String name:observed)order.append("field\t").append(name).append('\n');
          Files.writeString(out.resolve(prefix+"_order.tsv"),order+"rng\t"+random.nextInt()+"\n");
          save(out,prefix,target);
          try(FieldsProducer producer=nativeFormat.fieldsProducer(readState(target,fields,2*(DOCS-1)))){producer.checkIntegrity();if(snapshot(producer,fields,2*(DOCS-1),null).contains("676f6e65"))throw new AssertionError("Unicode deleted term survived");}
        }
      }
    }
  }
  public static void main(String[] args) throws Exception {
    Path out=Path.of(args[0]);Files.createDirectories(out);Files.writeString(out.resolve("cranky_postings.tsv"),trace());real(out);unicodeFields(out);
  }
}