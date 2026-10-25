// Same inputs and actual wrapper implementations as GenScorerIndexSearcher.
import java.util.Arrays;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.*;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.tests.search.DisablingBulkScorerQuery;
import org.apache.lucene.tests.search.ScorerIndexSearcher;
import org.apache.lucene.util.RamUsageEstimator;

public class ScorerIndexSearcherMicro {
  static final long BASIS=0xcbf29ce484222325L;
  static volatile long sink;
  static long hash(long digest,long value){return(digest^value)*0x100000001b3L;}
  static final class Digest implements Collector {
    final ScoreMode mode;
    long digest=BASIS,count;
    Digest(ScoreMode mode){this.mode=mode;}
    public ScoreMode scoreMode(){return mode;}
    public LeafCollector getLeafCollector(LeafReaderContext ctx){return new LeafCollector(){
      Scorable scorer;
      public void setScorer(Scorable scorer){this.scorer=scorer;}
      public void collect(int doc)throws java.io.IOException{
        count++;digest=hash(digest,ctx.docBase+doc);
        digest=hash(digest,mode.needsScores()?Integer.toUnsignedLong(Float.floatToRawIntBits(scorer.score())):0);
      }
    };}
  }
  static long batch(IndexSearcher searcher,Query[] queries,boolean noScores)throws Exception{
    long digest=BASIS;
    for(int q=0;q<queries.length;q++){
      Digest c=new Digest(noScores?ScoreMode.COMPLETE_NO_SCORES:ScoreMode.COMPLETE);
      searcher.search(queries[q],c);
      digest=hash(digest,q);digest=hash(digest,c.count);digest=hash(digest,c.digest);
    }
    return digest;
  }
  public static void main(String[] args)throws Exception{
    long warmup=Long.parseLong(args.length>0?args[0]:"1000"),measure=Long.parseLong(args.length>1?args[1]:"1500");
    try(ByteBuffersDirectory directory=new ByteBuffersDirectory()){
      GenScorerIndexSearcher.write(directory,64);
      try(DirectoryReader reader=DirectoryReader.open(directory)){
        IndexSearcher forced=new ScorerIndexSearcher(reader),plain=new IndexSearcher(reader);
        forced.setQueryCache(null);plain.setQueryCache(null);
        Query[] raw=Arrays.copyOf(GenScorerIndexSearcher.queries(),9);
        Query[] wrapped=Arrays.stream(raw).map(DisablingBulkScorerQuery::new).toArray(Query[]::new);
        System.out.println("scorer_searcher_wrapper_extra_bytes\t"+(RamUsageEstimator.shallowSizeOf(forced)-RamUsageEstimator.shallowSizeOf(plain)));
        for(int c=0;c<4;c++){
          String name=(c<2?"scorer_searcher":"disabling_query")+(c%2==0?"_complete":"_no_scores");
          IndexSearcher searcher=c<2?forced:plain;Query[] queries=c<2?raw:wrapped;boolean noScores=c%2!=0;
          long expected=batch(searcher,queries,noScores);
          for(int phase=0;phase<2;phase++){long budget=phase==0?warmup:measure;
            long start=System.nanoTime(),operations=0;
            do{long actual=batch(searcher,queries,noScores);if(actual!=expected)throw new AssertionError("digest");sink=actual;operations++;}while(System.nanoTime()-start<budget*1_000_000);
            if(phase==1)System.out.printf(java.util.Locale.ROOT,"%s\t%.3f\t%d%n",name,(double)(System.nanoTime()-start)/operations,operations);
          }
          System.out.println("#check\t"+name+"\t"+Long.toUnsignedString(expected)+"\t9");
        }
      }
    }
  }
}
