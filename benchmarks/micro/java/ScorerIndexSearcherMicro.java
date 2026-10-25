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
    final boolean competitive;
    long digest=BASIS,count;
    Digest(ScoreMode mode,boolean competitive){this.mode=mode;this.competitive=competitive;}
    public ScoreMode scoreMode(){return mode;}
    public LeafCollector getLeafCollector(LeafReaderContext ctx){return new LeafCollector(){
      Scorable scorer;
      public void setScorer(Scorable scorer)throws java.io.IOException{
        this.scorer=scorer;
        if(competitive && count>0)scorer.setMinCompetitiveScore(Float.POSITIVE_INFINITY);
      }
      public void collect(int doc)throws java.io.IOException{
        count++;digest=hash(digest,ctx.docBase+doc);
        digest=hash(digest,mode.needsScores()?Integer.toUnsignedLong(Float.floatToRawIntBits(scorer.score())):0);
        if(competitive)scorer.setMinCompetitiveScore(Float.POSITIVE_INFINITY);
      }
    };}
  }
  static long batch(IndexSearcher searcher,Query[] queries,ScoreMode mode,boolean competitive)throws Exception{
    long digest=BASIS;
    for(int q=0;q<queries.length;q++){
      Digest c=new Digest(mode,competitive);
      searcher.search(queries[q],c);
      if(competitive && c.count!=1)throw new AssertionError("constant term must stop after the first live hit");
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
        Query constant=new ConstantScoreQuery(GenScorerIndexSearcher.term("a"));
        Query[] constantRaw={constant},constantWrapped={new DisablingBulkScorerQuery(constant)};
        for(int c=0;c<6;c++){
          boolean competitive=c>=4,scorerSearcher=c<2||c==4;
          String name=(scorerSearcher?"scorer_searcher":"disabling_query")+
              (competitive?"_constant_term_top_scores":c%2==0?"_complete":"_no_scores");
          IndexSearcher searcher=scorerSearcher?forced:plain;
          Query[] queries=competitive?(scorerSearcher?constantRaw:constantWrapped):(scorerSearcher?raw:wrapped);
          ScoreMode mode=competitive?ScoreMode.TOP_SCORES:c%2==0?ScoreMode.COMPLETE:ScoreMode.COMPLETE_NO_SCORES;
          long expected=batch(searcher,queries,mode,competitive);
          for(int phase=0;phase<2;phase++){long budget=phase==0?warmup:measure;
            long start=System.nanoTime(),operations=0;
            do{long actual=batch(searcher,queries,mode,competitive);if(actual!=expected)throw new AssertionError("digest");sink=actual;operations++;}while(System.nanoTime()-start<budget*1_000_000);
            if(phase==1)System.out.printf(java.util.Locale.ROOT,"%s\t%.3f\t%d%n",name,(double)(System.nanoTime()-start)/operations,operations);
          }
          System.out.println("#check\t"+name+"\t"+Long.toUnsignedString(expected)+"\t"+queries.length);
        }
      }
    }
  }
}
