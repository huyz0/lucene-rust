import org.apache.lucene.index.*; import org.apache.lucene.document.*; import org.apache.lucene.store.*;
import org.apache.lucene.codecs.lucene104.Lucene104Codec; import org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsFormat; import org.apache.lucene.codecs.KnnVectorsFormat;
import java.nio.*; import java.nio.file.*;
public class KnnBuild { public static void main(String[] a) throws Exception {
  ByteBuffer b=ByteBuffer.wrap(Files.readAllBytes(Path.of(a[1]+".base.fbin"))).order(ByteOrder.LITTLE_ENDIAN); int n=b.getInt(), d=b.getInt();
  IndexWriterConfig c=new IndexWriterConfig(); c.setRAMBufferSizeMB(4096); c.setMergePolicy(NoMergePolicy.INSTANCE); c.setUseCompoundFile(false);
  c.setCodec(new Lucene104Codec(){ @Override public KnnVectorsFormat getKnnVectorsFormatForField(String f){ return new Lucene99HnswVectorsFormat(16,100);} });
  long t0=System.nanoTime();
  try (var dir=FSDirectory.open(Path.of(a[0])); var w=new IndexWriter(dir,c)) {
    for (int i=0;i<n;i++){ float[] v=new float[d]; for(int j=0;j<d;j++)v[j]=b.getFloat(); Document doc=new Document(); doc.add(new StoredField("id",i)); doc.add(new KnnFloatVectorField("v",v,VectorSimilarityFunction.COSINE)); w.addDocument(doc);}
    w.commit(); }
  System.out.printf("build_s %.1f%n",(System.nanoTime()-t0)/1e9);
}}
