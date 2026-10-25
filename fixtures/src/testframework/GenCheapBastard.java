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
import org.apache.lucene.tests.codecs.cheapbastard.CheapBastardCodec;
import org.apache.lucene.util.*;

/** Actual CheapBastardCodec components; no renamed native Codec installation. */
public class GenCheapBastard {
  static final int DOCS=9,TERMS=1536;
  static CheapBastardCodec codec(){return new CheapBastardCodec();}
  static String configuration()throws Exception {
    var c=codec();var member=FilterCodec.class.getDeclaredField("delegate");member.setAccessible(true);var d=(Codec)member.get(c);
    var p=c.postingsFormat();var min=p.getClass().getDeclaredField("minTermBlockSize");var max=p.getClass().getDeclaredField("maxTermBlockSize");min.setAccessible(true);max.setAccessible(true);
    var ordinary=new Lucene104PostingsFormat();StringBuilder out=new StringBuilder();
    out.append("name\t").append(c.getName()).append("\ndelegate\t").append(d.getName()).append("\npostings\t").append(p.getName()).append('\n');
    out.append("blocks\t").append(min.getInt(p)).append('\t').append(max.getInt(p)).append("\ndefault-blocks\t").append(min.getInt(ordinary)).append('\t').append(max.getInt(ordinary)).append('\n');
    out.append("same-postings\t").append(p==c.postingsFormat()).append('\n');
    for(String method:new String[]{"docValuesFormat","fieldInfosFormat","liveDocsFormat","normsFormat","segmentInfoFormat","storedFieldsFormat","termVectorsFormat","compoundFormat","pointsFormat","knnVectorsFormat"}){
      var accessor=Codec.class.getMethod(method);out.append("delegate-identity\t").append(method).append('\t').append(accessor.invoke(c)==accessor.invoke(d)).append('\n');
    }
    return out.toString();
  }
  static TokenStream tokens(int doc,boolean empty,boolean rich) {
    return new TokenStream(){
      final CharTermAttribute term=addAttribute(CharTermAttribute.class);final PositionIncrementAttribute inc=addAttribute(PositionIncrementAttribute.class);
      final OffsetAttribute offset=addAttribute(OffsetAttribute.class);final PayloadAttribute payload=addAttribute(PayloadAttribute.class);int at,position;
      @Override public void reset()throws IOException{super.reset();at=0;position=0;}
      @Override public boolean incrementToken(){
        if(empty)return false;
        while(at<TERMS && (at+doc)%3==0)at++;
        if(at>=TERMS)return false;clearAttributes();
        int n=at++;term.append(String.format(Locale.ROOT,"p%02x/t%04x",n/128,n));inc.setPositionIncrement(position%7==0?2:1);
        offset.setOffset(position*13,position*13+11);payload.setPayload(rich&&position%5==0?new BytesRef(new byte[]{(byte)doc,(byte)n,(byte)255}):null);position++;return true;
      }
      @Override public void end()throws IOException{super.end();offset.setOffset(position*13,position*13);}
    };
  }
  static int[][] maps(boolean sorted){int[][] maps=new int[2][DOCS];for(int side=0;side<2;side++)for(int doc=0;doc<DOCS;doc++)maps[side][doc]=doc==1?-1:sorted?2*(doc==0?0:doc-1)+side:8*side+(doc==0?0:doc-1);return maps;}
  static Map<String,long[]> mappedNorms(Map<String,long[]> columns,int[][] maps){Map<String,long[]> result=new TreeMap<>();for(var entry:columns.entrySet()){long[] v=new long[16];Arrays.fill(v,-1);for(int side=0;side<2;side++)for(int doc=0;doc<DOCS;doc++)if(maps[side][doc]>=0)v[maps[side][doc]]=entry.getValue()[doc];result.put(entry.getKey(),v);}return result;}
  static String seeks(FieldsProducer reader)throws Exception {
    StringBuilder text=new StringBuilder();
    for(String field:reader)for(String target:new String[]{"","p00/t0000","p00/t0001","p00/t007f","p01/t0080","p05/t02ff","p0b/t05ff","p0b/t05ffx","z"}){
      var e=reader.terms(field).iterator();var status=e.seekCeil(new BytesRef(target));text.append(field).append('\t').append(target).append('\t').append(status).append('\t');
      if(status!=TermsEnum.SeekStatus.END)text.append(e.term().utf8ToString()).append('\t').append(e.docFreq()).append('\t').append(e.totalTermFreq());text.append('\n');
    }return text.toString();
  }
  static void real(Path out)throws Exception {
    try(var source=new ByteBuffersDirectory()) {
      try(var writer=new IndexWriter(source,new IndexWriterConfig().setUseCompoundFile(false).setCodec(new Lucene104Codec()))){
        for(int doc=0;doc<DOCS;doc++){var d=new Document();d.add(new Field("docs",tokens(doc,false,false),GenCrankyPostings.type(IndexOptions.DOCS,false)));d.add(new Field("empty",tokens(doc,true,false),GenCrankyPostings.type(IndexOptions.DOCS,false)));d.add(new Field("rich",tokens(doc,false,true),GenCrankyPostings.type(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS_AND_OFFSETS,true)));writer.addDocument(d);}writer.forceMerge(1);
      }
      try(var reader=DirectoryReader.open(source)){
        var leaf=(SegmentReader)reader.leaves().get(0).reader();var infos=leaf.getFieldInfos();var columns=GenCrankyPostings.norms(leaf.getNormsReader(),infos,DOCS);var format=codec().postingsFormat();
        for(int kind=0;kind<4;kind++)try(var target=new ByteBuffersDirectory()){
          String prefix=kind==0?"cheap_bastard":kind==1?"cheap_bastard_merge_sequential":kind==2?"cheap_bastard_merge_sorted":"cheap_bastard_default";
          int maxDoc=kind==1||kind==2?16:DOCS;var norms=columns;PostingsFormat selected=kind==3?new Lucene104PostingsFormat():format;
          try(var writer=selected.fieldsConsumer(GenCrankyPostings.writeState(target,infos,maxDoc))){
            if(kind==0||kind==3)writer.write(leaf.getPostingsReader(),GenCrankyPostings.norms(norms));
            else{int[][] maps=maps(kind==2);norms=mappedNorms(columns,maps);writer.merge(GenCrankyPostings.mergeState(target,infos,new FieldsProducer[]{leaf.getPostingsReader(),leaf.getPostingsReader()},new int[]{DOCS,DOCS},maps,kind==2),GenCrankyPostings.norms(norms));}
          }
          GenCrankyPostings.save(out,prefix,target);
          try(var produced=selected.fieldsProducer(GenCrankyPostings.readState(target,infos,maxDoc))){produced.checkIntegrity();String snapshot=GenCrankyPostings.snapshot(produced,infos,maxDoc,GenCrankyPostings.norms(norms));if(kind==3){if(!snapshot.equals(Files.readString(out.resolve("cheap_bastard.tsv"))))throw new AssertionError("default block settings change values");}else Files.writeString(out.resolve(prefix+".tsv"),snapshot);Files.writeString(out.resolve(prefix+"_seeks.tsv"),seeks(produced));if(produced.terms("rich").size()!=TERMS)throw new AssertionError("term count");}
          if(kind==0){new Lucene104Codec().fieldInfosFormat().write(target,GenCrankyPostings.segment(target,DOCS),"",infos,IOContext.DEFAULT);try(var in=target.openInput("_cp.fnm",IOContext.DEFAULT)){byte[] bytes=new byte[Math.toIntExact(in.length())];in.readBytes(bytes,0,bytes.length);Files.write(out.resolve("cheap_bastard.fnm"),bytes);}}
        }
      }
    }
    if(Arrays.equals(Files.readAllBytes(out.resolve("cheap_bastard.tim")),Files.readAllBytes(out.resolve("cheap_bastard_default.tim"))))throw new AssertionError("block sizes must affect actual dictionary");
  }
  public static void main(String[] args)throws Exception {Path out=Path.of(args[0]);Files.createDirectories(out);Files.writeString(out.resolve("cheap_bastard_config.tsv"),configuration());real(out);}
}
