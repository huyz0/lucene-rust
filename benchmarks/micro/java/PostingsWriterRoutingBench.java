import java.io.IOException;
import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.nio.file.Path;
import java.util.*;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.analysis.tokenattributes.*;
import org.apache.lucene.codecs.*;
import org.apache.lucene.codecs.lucene104.*;
import org.apache.lucene.codecs.perfield.PerFieldPostingsFormat;
import org.apache.lucene.document.*;
import org.apache.lucene.index.*;
import org.apache.lucene.search.Sort;
import org.apache.lucene.search.SortField;
import org.apache.lucene.store.*;
import org.apache.lucene.util.BytesRef;
import org.apache.lucene.util.SmallFloat;

/** Installed writer benchmark. Actual nested AssertingFieldsConsumer; native readers. */
public final class PostingsWriterRoutingBench {
  private static final int DOCS=8192, SOURCES=4;
  private static final String[] FIELDS={"docs","freqs","positions","offsets","sparse","empty"};
  private static volatile long sink;
  private static final class WriterOnlyAssertingCodec extends FilterCodec {
    private final PostingsFormat format;
    WriterOnlyAssertingCodec() {
      super("Lucene104",new Lucene104Codec());
      var nativeFormat=new Lucene104PostingsFormat();
      final Constructor<?> constructor;
      try {
        constructor=Class.forName("org.apache.lucene.tests.codecs.asserting.AssertingPostingsFormat$AssertingFieldsConsumer").getDeclaredConstructor(SegmentWriteState.class,FieldsConsumer.class);
        constructor.setAccessible(true);
      } catch (ReflectiveOperationException e) {throw new IllegalStateException(e);}
      var asserting=new PostingsFormat(nativeFormat.getName()) {
        @Override public FieldsConsumer fieldsConsumer(SegmentWriteState state) throws IOException {
          FieldsConsumer consumer=nativeFormat.fieldsConsumer(state);
          try {return (FieldsConsumer)constructor.newInstance(state,consumer);}
          catch (InvocationTargetException e) {
            try {consumer.close();}catch(Throwable closing){e.getCause().addSuppressed(closing);}
            Throwable cause=e.getCause();if(cause instanceof IOException io)throw io;if(cause instanceof RuntimeException runtime)throw runtime;if(cause instanceof Error error)throw error;throw new IOException(cause);
          } catch(ReflectiveOperationException e){try{consumer.close();}catch(Throwable closing){e.addSuppressed(closing);}throw new IOException(e);}
        }
        @Override public FieldsProducer fieldsProducer(SegmentReadState state) throws IOException {return nativeFormat.fieldsProducer(state);}
      };
      format=new PerFieldPostingsFormat(){@Override public PostingsFormat getPostingsFormatForField(String field){return asserting;}};
    }
    @Override public PostingsFormat postingsFormat(){return format;}
  }
  private static IndexOptions options(int field){return switch(field){case 0->IndexOptions.DOCS;case 1->IndexOptions.DOCS_AND_FREQS;case 3->IndexOptions.DOCS_AND_FREQS_AND_POSITIONS_AND_OFFSETS;default->IndexOptions.DOCS_AND_FREQS_AND_POSITIONS;};}
  private static long rank(int doc){return (long)(doc%(DOCS/SOURCES))*SOURCES+doc/(DOCS/SOURCES);}
  private record Token(String term,int increment,int start,int end,byte[] payload){}
  private static List<Token> tokens(int doc,int field){
    if(field==5||(field==4&&doc%3==0))return List.of();
    var terms=new ArrayList<Token>();
    terms.add(new Token("common",1,0,6,null));terms.add(new Token("common",0,10,16,null));
    terms.add(new Token(String.format(Locale.ROOT,"t%03d",doc%257),2,20,24,null));
    terms.add(new Token(String.format(Locale.ROOT,"u%05d",doc),1,30,36,null));
    terms.add(new Token(String.format(Locale.ROOT,"bucket%02d",doc%11),1,40,48,null));
    for(int i=0;i<doc%5;i++)terms.add(new Token("common",1,50+10*i,56+10*i,null));
    if(field==3)for(int i=0;i<terms.size();i++){var t=terms.get(i);terms.set(i,new Token(t.term,t.increment,t.start,t.end,(doc+i)%3==0?null:new byte[]{(byte)doc,(byte)i,(byte)255}));}
    return terms;
  }
  private static final class CorpusTokens extends TokenStream {
    final List<Token> tokens;int cursor;
    final CharTermAttribute term=addAttribute(CharTermAttribute.class);
    final PositionIncrementAttribute increment=addAttribute(PositionIncrementAttribute.class);
    final OffsetAttribute offset=addAttribute(OffsetAttribute.class);
    final PayloadAttribute payload=addAttribute(PayloadAttribute.class);
    CorpusTokens(List<Token> tokens){this.tokens=tokens;}
    @Override public void reset() throws IOException{super.reset();cursor=0;}
    @Override public boolean incrementToken(){if(cursor==tokens.size())return false;clearAttributes();var t=tokens.get(cursor++);term.append(t.term);increment.setPositionIncrement(t.increment);offset.setOffset(t.start,t.end);payload.setPayload(t.payload==null?null:new BytesRef(t.payload));return true;}
    @Override public void end()throws IOException{super.end();increment.setPositionIncrement(0);int end=tokens.isEmpty()?0:tokens.get(tokens.size()-1).end;offset.setOffset(end,end);}
  }
  private static Document document(int doc){
    var result=new Document();result.add(new StringField("id",Integer.toString(doc),Field.Store.YES));result.add(new NumericDocValuesField("rank",rank(doc)));
    for(int field=0;field<FIELDS.length;field++){if(field==4&&doc%3==0)continue;var type=new FieldType();type.setTokenized(true);type.setIndexOptions(options(field));type.setOmitNorms(field==0);type.freeze();result.add(new Field(FIELDS[field],new CorpusTokens(tokens(doc,field)),type));}return result;
  }
  private static IndexWriter prepare(Directory directory,boolean asserting,int mode)throws IOException{
    var config=new IndexWriterConfig(new StandardAnalyzer());config.setCodec(asserting?new WriterOnlyAssertingCodec():new Lucene104Codec());config.setUseCompoundFile(false);config.setMaxBufferedDocs(DOCS*2);config.setRAMBufferSizeMB(IndexWriterConfig.DISABLE_AUTO_FLUSH);config.setMaxFullFlushMergeWaitMillis(0);var policy=new LogDocMergePolicy();policy.setMergeFactor(1000);config.setMergePolicy(policy);config.setMergeScheduler(new SerialMergeScheduler());if(mode==2)config.setIndexSort(new Sort(new SortField("rank",SortField.Type.LONG)));
    var writer=new IndexWriter(directory,config);
    for(int doc=0;doc<DOCS;doc++){writer.addDocument(document(doc));if(mode!=0&&(doc+1)%(DOCS/SOURCES)==0)writer.commit();}
    if(mode!=0){var deletes=new ArrayList<Term>();for(int doc=0;doc<DOCS;doc++)if(doc%7==0)deletes.add(new Term("id",Integer.toString(doc)));writer.deleteDocuments(deletes.toArray(Term[]::new));writer.commit();require(SegmentInfos.readLatestCommit(directory).size()==SOURCES,"source count");}return writer;
  }
  private static void require(boolean value,String message){if(!value)throw new AssertionError(message);}
  private static long mix(long digest,long value){return(digest^value)*0x100000001b3L;}
  private static long bytes(long digest,byte[] value){digest=mix(digest,value.length);for(byte b:value)digest=mix(digest,Byte.toUnsignedInt(b));return digest;}
  private record Occurrence(int position,Token token){}
  private record Check(long digest,int docs,int terms,int postings,int occurrences,int payloadBytes,long encodedBytes){}
  private static Check verify(Directory directory,int mode)throws IOException{
    try(var reader=DirectoryReader.open(directory)){
      var ids=new ArrayList<Integer>();for(int doc=0;doc<DOCS;doc++)if(mode==0||doc%7!=0)ids.add(doc);if(mode==2)ids.sort(Comparator.comparingLong(PostingsWriterRoutingBench::rank));
      require(reader.maxDoc()==ids.size()&&reader.numDocs()==ids.size()&&reader.leaves().size()==1,"document counts");var segment=reader.leaves().get(0).reader();long digest=0xcbf29ce484222325L;var ranks=segment.getNumericDocValues("rank");var stored=segment.storedFields();
      for(int physical=0;physical<ids.size();physical++){int id=ids.get(physical);require(Integer.parseInt(stored.document(physical).get("id"))==id,"stored id");digest=mix(digest,id);require(ranks.advanceExact(physical)&&ranks.longValue()==rank(id),"rank");digest=mix(digest,rank(id));}
      int termCount=0,postingCount=0,occurrences=0,payloadBytes=0;
      for(int field=0;field<FIELDS.length;field++){
        String name=FIELDS[field];var info=segment.getFieldInfos().fieldInfo(name);
        require(info.getIndexOptions()==options(field),"options for "+name);
        require(info.omitsNorms()==(field==0),"omit norms for "+name);
        require(info.hasPayloads()==(field==3),"payload flag for "+name);
        require(Objects.equals(info.getAttribute(PerFieldPostingsFormat.PER_FIELD_FORMAT_KEY),field==5?null:"Lucene104"),"format for "+name);
        require(Objects.equals(info.getAttribute(PerFieldPostingsFormat.PER_FIELD_SUFFIX_KEY),field==5?null:"0"),"suffix for "+name);
        var norms=segment.getNormValues(name);require((norms!=null)==(field!=0),"norms presence");var expected=new TreeMap<String,TreeMap<Integer,List<Occurrence>>>();int docCount=0;
        for(int physical=0;physical<ids.size();physical++){int id=ids.get(physical);var values=tokens(id,field);if(!values.isEmpty())docCount++;if(norms!=null){boolean present=field!=4||id%3!=0;require(norms.advanceExact(physical)==present,"norm presence");if(present){int norm=Byte.toUnsignedInt(SmallFloat.intToByte4(values.size()-(values.isEmpty()?0:1)));require(norms.longValue()==norm,"norm");digest=mix(digest,norm);}}int position=-1;for(var token:values){position+=token.increment;expected.computeIfAbsent(token.term,k->new TreeMap<>()).computeIfAbsent(physical,k->new ArrayList<>()).add(new Occurrence(position,token));}}
        digest=mix(digest,field);digest=mix(digest,expected.size());var actual=segment.terms(name);if(expected.isEmpty()){require(actual==null,"empty terms");continue;}
        require(actual.size()==expected.size()&&actual.getDocCount()==docCount,"term counts");require(actual.hasFreqs()==(field!=0)&&actual.hasPositions()==(field>=2)&&actual.hasOffsets()==(field==3)&&actual.hasPayloads()==(field==3),"term flags");long sumDocs=0,sumTotal=0;for(var docs:expected.values()){sumDocs+=docs.size();for(var values:docs.values())sumTotal+=field==0?1:values.size();}require(actual.getSumDocFreq()==sumDocs&&actual.getSumTotalTermFreq()==sumTotal,"sum statistics");digest=mix(digest,sumDocs);digest=mix(digest,sumTotal);digest=mix(digest,docCount);
        var iterator=actual.iterator();for(var entry:expected.entrySet()){var term=iterator.next();require(term!=null&&term.utf8ToString().equals(entry.getKey()),"term order");digest=bytes(digest,entry.getKey().getBytes(java.nio.charset.StandardCharsets.UTF_8));termCount++;var docs=entry.getValue();long total=0;for(var values:docs.values())total+=field==0?1:values.size();require(iterator.docFreq()==docs.size()&&iterator.totalTermFreq()==total,"term statistics");var postings=iterator.postings(null,PostingsEnum.ALL);
          for(var doc:docs.entrySet()){require(postings.nextDoc()==doc.getKey(),"posting document");int freq=field==0?1:doc.getValue().size();require(postings.freq()==freq,"frequency");digest=mix(digest,doc.getKey());digest=mix(digest,freq);postingCount++;occurrences+=freq;if(field>=2)for(var occurrence:doc.getValue()){var token=occurrence.token;require(postings.nextPosition()==occurrence.position,"position");digest=mix(digest,occurrence.position);int start=field==3?token.start:-1,end=field==3?token.end:-1;require(postings.startOffset()==start&&postings.endOffset()==end,"offsets");digest=mix(digest,start);digest=mix(digest,end);BytesRef payload=postings.getPayload();byte[] data=payload==null?new byte[0]:Arrays.copyOfRange(payload.bytes,payload.offset,payload.offset+payload.length);require(Arrays.equals(data,token.payload==null?new byte[0]:token.payload),"payload");digest=bytes(digest,data);payloadBytes+=data.length;}}
          require(postings.nextDoc()==PostingsEnum.NO_MORE_DOCS,"posting end");}require(iterator.next()==null,"term end");
      }
      segment.checkIntegrity();long encoded=0;for(String file:directory.listAll())for(String extension:new String[]{"doc","psm","pos","pay","tim","tip","tmd"})if(file.endsWith("."+extension))encoded+=directory.fileLength(file);sink=digest;return new Check(digest,ids.size(),termCount,postingCount,occurrences,payloadBytes,encoded);
    }
  }
  private static String name(boolean asserting,int mode){return "postings_routing_"+(asserting?"asserting":"default")+"_"+new String[]{"flush","merge","sorted_merge"}[mode]+"_commit";}
  private static void output(String name,double ns,long iterations,Check c,long accounted){System.out.printf(Locale.ROOT,"#check\t%s\t%s\t%d\t%d\t%d\t%d\t%d\t%d%n",name,Long.toUnsignedString(c.digest),c.docs,FIELDS.length,c.terms,c.postings,c.occurrences,c.payloadBytes);System.out.printf(Locale.ROOT,"%s\t%.3f\t%d%n",name,ns,iterations);System.out.printf("%s_encoded_bytes\t%d%n%s_writer_accounted_bytes\t%d%n%s_logical_payload_bytes\t%d%n",name,c.encodedBytes,name,accounted,name,c.payloadBytes);}
  private static void run(boolean asserting,int mode,long millis,long fixed,boolean print)throws IOException{
    long nanos=0,iterations=0;for(;;){try(var directory=new ByteBuffersDirectory();var writer=prepare(directory,asserting,mode)){long accounted=writer.ramBytesUsed();long start=System.nanoTime();if(mode!=0)writer.forceMerge(1);writer.commit();nanos+=System.nanoTime()-start;iterations++;var check=verify(directory,mode);if((fixed>0&&iterations==fixed)||(fixed==0&&nanos>=millis*1000000L)){if(print)output(name(asserting,mode),(double)nanos/iterations,iterations,check,accounted);return;}}}
  }
  public static void main(String[] args)throws Exception{
    if(args[0].equals("--verify-dir")){try(var directory=FSDirectory.open(Path.of(args[1]))){int mode=Integer.parseInt(args[2]);output(name(true,mode),1,1,verify(directory,mode),0);}return;}
    if(args[0].equals("--memory-case")){long fixed=Long.parseLong(args[2]);require(fixed>0,"iterations");for(int mode=0;mode<3;mode++)for(boolean asserting:new boolean[]{false,true})if(name(asserting,mode).equals(args[1])){run(asserting,mode,0,fixed,true);return;}throw new IllegalArgumentException("unknown case");}
    long warmup=Long.parseLong(args[0]),measure=Long.parseLong(args[1]);require(measure>0,"measure duration");for(int mode=0;mode<3;mode++)for(boolean asserting:new boolean[]{false,true}){run(asserting,mode,warmup,0,false);run(asserting,mode,measure,0,true);}
  }
}
