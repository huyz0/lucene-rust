import org.apache.lucene.index.*;
import org.apache.lucene.search.*;
import org.apache.lucene.store.MMapDirectory;
import java.nio.*; import java.nio.file.*;
public class KnnBench {
  static float[][] rdf(String p) throws Exception { ByteBuffer b=ByteBuffer.wrap(Files.readAllBytes(Path.of(p))).order(ByteOrder.LITTLE_ENDIAN); int n=b.getInt(), d=b.getInt(); float[][] o=new float[n][d]; for(int i=0;i<n;i++)for(int j=0;j<d;j++)o[i][j]=b.getFloat(); return o; }
  static int[][] rdi(String p) throws Exception { ByteBuffer b=ByteBuffer.wrap(Files.readAllBytes(Path.of(p))).order(ByteOrder.LITTLE_ENDIAN); int n=b.getInt(), d=b.getInt(); int[][] o=new int[n][d]; for(int i=0;i<n;i++)for(int j=0;j<d;j++)o[i][j]=b.getInt(); return o; }
  public static void main(String[] a) throws Exception {
    String idx=a[0], pre=a[1]; String[] efs=a[2].split(","); int nq=Integer.parseInt(a[3]);
    float[][] q=rdf(pre+".query.fbin"); int[][] gt=rdi(pre+".gt.ibin");
    try (var dir=new MMapDirectory(Path.of(idx)); var r=DirectoryReader.open(dir)) {
      IndexSearcher s=new IndexSearcher(r); s.setQueryCache(null);
      System.out.println("segments "+r.leaves().size());
      System.out.println("ef,recall@10,mean_us,p50_us");
      for (String e: efs) { int ef=Integer.parseInt(e);
        for (int rep=0; rep<3; rep++) for (int i=nq;i<nq+2000;i++) s.search(new KnnFloatVectorQuery("v",q[i],ef),10);
        long[] t=new long[nq]; long found=0;
        for (int i=0;i<nq;i++){ long t0=System.nanoTime(); TopDocs td=s.search(new KnnFloatVectorQuery("v",q[i],ef),10); t[i]=System.nanoTime()-t0;
          for (ScoreDoc sd: td.scoreDocs) for (int k=0;k<10;k++) if (gt[i][k]==sd.doc){found++;break;} }
        java.util.Arrays.sort(t); double mean=0; for(long x:t)mean+=x; mean/=nq;
        System.out.printf("%d,%.4f,%.0f,%.0f%n",ef,found/(nq*10.0),mean/1e3,t[nq/2]/1e3);
      }
    }
  }
}
