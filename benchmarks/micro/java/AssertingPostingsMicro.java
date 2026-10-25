import java.nio.file.*;
import java.nio.charset.StandardCharsets;
import java.util.*;
import org.apache.lucene.codecs.*;
import org.apache.lucene.codecs.lucene104.*;
import org.apache.lucene.index.*;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.store.*;
import org.apache.lucene.util.*;
/** Full actual asserting-wrapper work; helper classes are generators, never run here. */
public class AssertingPostingsMicro {
  static volatile long sink;
  static final String[] EXTENSIONS={"doc","psm","pos","pay","tim","tip","tmd"};
  static class Sink extends FieldsConsumer {
    @Override public void write(Fields fields,NormsProducer norms){}
    @Override public void close(){}
  }
  static long bytes(long d,byte[] value,int offset,int length){d=d*31+length;for(int i=0;i<length;i++)d=d*31+Byte.toUnsignedInt(value[offset+i]);return d;}
  static long read(long d,FieldsProducer reader) throws Exception {
    reader.checkIntegrity();d=d*31+reader.size();
    for(String name:reader){byte[] nameBytes=name.getBytes(StandardCharsets.UTF_8);d=bytes(d,nameBytes,0,nameBytes.length);Terms terms=reader.terms(name);
      long[] metadata={terms.size(),terms.getSumDocFreq(),terms.getSumTotalTermFreq(),terms.getDocCount(),terms.hasFreqs()?1:0,terms.hasPositions()?1:0,terms.hasOffsets()?1:0,terms.hasPayloads()?1:0};for(long value:metadata)d=d*31+value;
      TermsEnum e=terms.iterator();for(BytesRef term=e.next();term!=null;term=e.next()){
        d=bytes(d,term.bytes,term.offset,term.length);PostingsEnum p=e.postings(null,PostingsEnum.ALL);
        for(int doc=p.nextDoc();doc!=DocIdSetIterator.NO_MORE_DOCS;doc=p.nextDoc()){
          d=d*31+doc;int freq=p.freq();d=d*31+freq;if(terms.hasPositions())for(int i=0;i<freq;i++){
            d=d*31+p.nextPosition();d=d*31+p.startOffset();d=d*31+p.endOffset();BytesRef payload=p.getPayload();d=payload==null?d*31:bytes(d,payload.bytes,payload.offset,payload.length);
          }
        }
      }
    }return d;
  }
  static long batch(Directory saved,FieldInfos infos,FieldsProducer source,NormsProducer[] norms,MergeState[] merges,PostingsFormat format,FieldsConsumer validation,int which) throws Exception {
    long d=0;
    for(int i=0;i<16;i++){
      if(which==0){try(FieldsProducer reader=format.fieldsProducer(GenCrankyPostings.readState(saved,infos,9))){d=read(d,reader);}}
      else if(which==4){validation.write(source,norms[0]);validation.close();d=d*31+1;}
      else try(Directory target=new ByteBuffersDirectory()){
        try(FieldsConsumer writer=format.fieldsConsumer(GenCrankyPostings.writeState(target,infos,which==1?9:16))){if(which==1)writer.write(source,norms[0]);else writer.merge(merges[which-2],norms[which-1]);}
        for(String extension:EXTENSIONS)try(IndexInput input=target.openInput("_cp_s."+extension,IOContext.DEFAULT)){long length=input.length();d=d*31+length;for(long p=0;p<length;p++)d=d*31+Byte.toUnsignedInt(input.readByte());}
      }
    }return d;
  }
  static long[] run(Directory saved,FieldInfos infos,FieldsProducer source,NormsProducer[] norms,MergeState[] merges,PostingsFormat format,FieldsConsumer validation,int which,long ms,long expected) throws Exception {
    long start=System.nanoTime(),count=0;do{long actual=batch(saved,infos,source,norms,merges,format,validation,which);if(actual!=expected)throw new AssertionError("unstable digest");sink=actual;count+=16;}while(System.nanoTime()-start<ms*1_000_000);return new long[]{System.nanoTime()-start,count};
  }
  public static void main(String[] args) throws Exception {
    long warmup=Long.parseLong(args[0]),measure=Long.parseLong(args[1]);
    try(Directory saved=new ByteBuffersDirectory()){
      for(String ext:EXTENSIONS)try(IndexOutput out=saved.createOutput("_cp_s."+ext,IOContext.DEFAULT)){byte[] data=Files.readAllBytes(Path.of("fixtures/testframework/data/asserting_postings."+ext));out.writeBytes(data,0,data.length);}
      try(IndexOutput out=saved.createOutput("_cp.fnm",IOContext.DEFAULT)){byte[] data=Files.readAllBytes(Path.of("fixtures/testframework/data/asserting_postings.fnm"));out.writeBytes(data,0,data.length);}
      FieldInfos infos=new Lucene104Codec().fieldInfosFormat().read(saved,GenCrankyPostings.segment(saved,9),"",IOContext.DEFAULT);
      Sink sink=new Sink();FieldsConsumer wrapper=GenAssertingPostings.consumer(GenCrankyPostings.writeState(saved,infos,9),sink);
      System.out.printf("asserting_postings_validation_shallow_bytes\t%d%n",RamUsageEstimator.shallowSizeOf(wrapper)+RamUsageEstimator.shallowSizeOf(sink));
      PostingsFormat format=GenAssertingPostings.format(new Lucene104PostingsFormat());
      NormsProducer[] norms=new NormsProducer[3];
      for(int kind=0;kind<3;kind++){
        int maxDoc=kind==0?9:16;String prefix="asserting_postings"+(kind==0?"":kind==1?"_merge_sequential":"_merge_sorted");Map<String,long[]> columns=new TreeMap<>();
        for(String line:Files.readAllLines(Path.of("fixtures/testframework/data/"+prefix+".tsv"))){String[] row=line.split("\t");if(row[0].equals("norm")){long[] values=columns.computeIfAbsent(row[1],n->{long[] v=new long[maxDoc];Arrays.fill(v,-1);return v;});values[Integer.parseInt(row[2])]=Long.parseLong(row[3]);}}
        norms[kind]=GenCrankyPostings.norms(columns);
      }
      try(FieldsProducer source=new Lucene104PostingsFormat().fieldsProducer(GenCrankyPostings.readState(saved,infos,9))){
        MergeState[] merges=new MergeState[2];for(int kind=0;kind<2;kind++){int[][] maps=new int[2][9];for(int side=0;side<2;side++)for(int doc=0;doc<9;doc++)maps[side][doc]=doc==1?-1:kind==1?2*(doc==0?0:doc-1)+side:8*side+(doc==0?0:doc-1);merges[kind]=GenCrankyPostings.mergeState(saved,infos,new FieldsProducer[]{source,source},new int[]{9,9},maps,kind==1);}
        String[] names={"read","write","merge_sequential","merge_sorted","validation"};
        for(int which=0;which<names.length;which++){long expected=batch(saved,infos,source,norms,merges,format,wrapper,which);System.out.printf("#check\tasserting_postings_%s\t%s\t16%n",names[which],Long.toUnsignedString(expected));run(saved,infos,source,norms,merges,format,wrapper,which,warmup,expected);long[] result=run(saved,infos,source,norms,merges,format,wrapper,which,measure,expected);System.out.printf(Locale.ROOT,"asserting_postings_%s\t%.3f\t%d%n",names[which],(double)result[0]/result[1],result[1]);}
      }
    }
  }
}
