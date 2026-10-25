import java.nio.file.*;
import java.util.*;
import org.apache.lucene.codecs.*;
import org.apache.lucene.codecs.hnsw.HnswGraphProvider;
import org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsFormat;
import org.apache.lucene.index.*;
import org.apache.lucene.search.*;
import org.apache.lucene.store.*;
import org.apache.lucene.util.*;

/** Actual Java assertion wrappers over the same unquantized sparse float/byte files. */
public class AssertingKnnVectorsMicro {
  static volatile long sink;
  static final FieldInfos FIELDS=new FieldInfos(new FieldInfo[]{GenAssertingKnnVectors.FLOAT,GenAssertingKnnVectors.BYTE});
  static final KnnVectorsFormat FORMAT=new Lucene99HnswVectorsFormat(4,10,0);
  static final KnnVectorsFormat ASSERTING=assertingFormat();
  static KnnVectorsFormat assertingFormat() {
    try {
      var format=new org.apache.lucene.tests.codecs.asserting.AssertingKnnVectorsFormat();
      var delegate=format.getClass().getDeclaredField("delegate");delegate.setAccessible(true);delegate.set(format,FORMAT);
      return format;
    } catch(ReflectiveOperationException error) {throw new ExceptionInInitializerError(error);}
  }
  static final class Digest {long value;void mix(long n){value=value*31+n;}}
  static KnnVectorsReader open(Directory directory)throws Exception {
    return ASSERTING.fieldsReader(new SegmentReadState(directory,GenAssertingKnnVectors.segment(directory,16),FIELDS,IOContext.DEFAULT,"s"));
  }
  static void readDigest(KnnVectorsReader reader,Digest d)throws Exception {
    reader.checkIntegrity();
    for(String field:new String[]{"float","byte"}) {
      if(field.equals("float")){var v=reader.getFloatVectorValues(field);d.mix(v.size());d.mix(v.dimension());var it=v.iterator();for(int doc=it.nextDoc();doc!=DocIdSetIterator.NO_MORE_DOCS;doc=it.nextDoc()){d.mix(doc);for(float n:v.vectorValue(it.index()))d.mix(Float.floatToRawIntBits(n));}}
      else{var v=reader.getByteVectorValues(field);d.mix(v.size());d.mix(v.dimension());var it=v.iterator();for(int doc=it.nextDoc();doc!=DocIdSetIterator.NO_MORE_DOCS;doc=it.nextDoc()){d.mix(doc);for(byte n:v.vectorValue(it.index()))d.mix(n);}}
      var graph=((HnswGraphProvider)reader).getGraph(field);d.mix(graph.size());d.mix(graph.numLevels());d.mix(graph.entryNode());
      for(int level=0;level<graph.numLevels();level++){var nodes=graph.getNodesOnLevel(level);while(nodes.hasNext()){int node=nodes.nextInt();d.mix(node);graph.seek(level,node);var neighbors=new ArrayList<Integer>();for(int n=graph.nextNeighbor();n!=DocIdSetIterator.NO_MORE_DOCS;n=graph.nextNeighbor())neighbors.add(n);d.mix(neighbors.size());for(int n:neighbors)d.mix(n);}}
    }
  }
  static long batch(String name,Directory directory,KnnVectorsReader prepared)throws Exception {
    Digest d=new Digest();for(int i=0;i<16;i++)switch(name) {
      case "values"->{var f=prepared.getFloatVectorValues("float");var b=prepared.getByteVectorValues("byte");d.mix(f.size());d.mix(f.dimension());d.mix(b.size());d.mix(b.dimension());}
      case "read"->{try(var reader=open(directory)){readDigest(reader,d);}}
      case "search"->{for(String field:new String[]{"float","byte"})for(int q:new int[]{0,5,15})for(boolean filtered:new boolean[]{false,true}){
        // Both twins allocate this small mask inside the timed operation.
        boolean[] bits=new boolean[16];for(int doc=0;doc<16;doc++)bits[doc]=doc%3!=0;
        Bits mask=filtered?new Bits(){public boolean get(int doc){return bits[doc];}public int length(){return 16;}}:null;
        var accept=AcceptDocs.fromLiveDocs(mask,16);var collector=new TopKnnCollector(4,100);
        if(field.equals("float"))prepared.search(field,new float[]{q*0.25f,16-q},collector,accept);
        else prepared.search(field,new byte[]{(byte)(q-8),(byte)(q*2)},collector,accept);
        var hits=collector.topDocs();d.mix(hits.totalHits.value());d.mix(hits.totalHits.relation()==TotalHits.Relation.GREATER_THAN_OR_EQUAL_TO?1:0);for(var hit:hits.scoreDocs){d.mix(hit.doc);d.mix(Float.floatToRawIntBits(hit.score));}
      }}
      case "merge_lifecycle"->{var merge=prepared.getMergeInstance();d.mix(merge.getFloatVectorValues("float").size());d.mix(merge.getByteVectorValues("byte").size());merge.finishMerge();}
      default->throw new AssertionError(name);
    }return d.value;
  }
  static long[] run(String name,Directory dir,KnnVectorsReader reader,int ms)throws Exception {
    long expected=batch(name,dir,reader),start=System.nanoTime(),count=0;
    do{long got=batch(name,dir,reader);if(got!=expected)throw new AssertionError("timed digest changed");sink=got;count+=16;}while(System.nanoTime()-start<ms*1_000_000L);
    return new long[]{count,System.nanoTime()-start};
  }
  public static void main(String[] args)throws Exception {
    if(!AssertingKnnVectorsMicro.class.desiredAssertionStatus())throw new AssertionError("run with -ea");int warmup=args.length>0?Integer.parseInt(args[0]):1000,measure=args.length>1?Integer.parseInt(args[1]):1500;
    try(var directory=new ByteBuffersDirectory()) {
      for(String ext:new String[]{"vec","vemf","vex","vem"}){byte[] data=Files.readAllBytes(Path.of("fixtures/testframework/data/asserting_knn_vectors."+ext));try(var out=directory.createOutput("_ak_s."+ext,IOContext.DEFAULT)){out.writeBytes(data,data.length);}}
      try(var reader=open(directory)) {
        long wrapper=RamUsageEstimator.shallowSizeOf(reader);for(String field:new String[]{"mergeInstanceCount","finishMergeCount"}){var reflected=reader.getClass().getDeclaredField(field);reflected.setAccessible(true);wrapper+=RamUsageEstimator.shallowSizeOf(reflected.get(reader));}
        var accept=org.apache.lucene.tests.search.AssertingAcceptDocs.wrap(AcceptDocs.fromLiveDocs(null,16));
        System.out.printf("#memory\tasserting_knn_vectors_reader_wrapper\t%d%n",wrapper);System.out.printf("#memory\tasserting_knn_vectors_accept_wrapper\t%d%n",RamUsageEstimator.shallowSizeOf(accept));
        for(String name:new String[]{"values","read","search","merge_lifecycle"}){System.out.printf("#check\tasserting_knn_vectors_%s\t%d\t16%n",name,batch(name,directory,reader));run(name,directory,reader,warmup);long[] result=run(name,directory,reader,measure);System.out.printf(Locale.ROOT,"asserting_knn_vectors_%s\t%.3f\t%d%n",name,(double)result[1]/result[0],result[0]);}
      }
    }
  }
}
