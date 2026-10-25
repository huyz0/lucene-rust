import com.carrotsearch.randomizedtesting.RandomizedRunner;
import java.lang.reflect.Field;
import java.util.*;
import org.apache.lucene.codecs.*;
import org.apache.lucene.codecs.lucene104.Lucene104Codec;
import org.apache.lucene.codecs.perfield.*;
import org.apache.lucene.tests.codecs.asserting.*;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.util.RamUsageEstimator;
import org.junit.Test;
import org.junit.runner.JUnitCore;
import org.junit.runner.RunWith;

/** Actual AssertingCodec; all native graphs and stable delegate formats are prepared outside timing. */
public class AssertingCodecMicro {
  static final long BASIS=0xcbf29ce484222325L;
  static final int[] MASKS={0,21,63}, BITS={-1,-1,3,2,-1,-1,0,1,-1,4,5};
  static final int[] COMPONENT_KINDS={4,0,10}, CASE_MASKS={0,2,1};
  static final long[] FACTORY_TOTALS={1536,4608,3072}, ERROR_TOTALS={768,1536,2304};
  static final String[] FIELDS={"","field","alternate",null};
  static volatile Object objectSink;
  static volatile long sink;
  static long hash(long d,long v){return(d^v)*0x100000001b3L;}
  static void check(boolean b,String label){if(!b)throw new AssertionError(label);}
  static int token(String f){if(f==null)return 3;return switch(f){case ""->0;case "field"->1;case "alternate"->2;default->throw new AssertionError(f);};}
  static Object member(Object object,String name)throws Exception{for(Class<?> c=object.getClass();c!=null;c=c.getSuperclass()){try{Field f=c.getDeclaredField(name);f.setAccessible(true);return f.get(object);}catch(NoSuchFieldException missing){}}throw new NoSuchFieldException(name);}
  static Object component(Codec c,int k){return switch(k){case 0->c.docValuesFormat();case 1->c.fieldInfosFormat();case 2->c.liveDocsFormat();case 3->c.normsFormat();case 4->c.postingsFormat();case 5->c.segmentInfoFormat();case 6->c.storedFieldsFormat();case 7->c.termVectorsFormat();case 8->c.compoundFormat();case 9->c.pointsFormat();case 10->c.knnVectorsFormat();default->throw new AssertionError(k);};}
  static boolean routed(int mask,int k){return k==1||k==5||k==8||(BITS[k]>=0&&(mask&(1<<BITS[k]))!=0);}
  static final class Probe extends FilterCodec {
    final long[] counts=new long[11];long trace=BASIS,displays,nativeSelects,nativeTrace=BASIS;boolean fail,next;
    final IllegalStateException failure=new IllegalStateException("prebuilt probe error");
    final Object[] formats=new Object[11];
    final KnnVectorsFormat selected;
    Probe(){super("Probe",new Lucene104Codec());for(int k=0;k<11;k++)formats[k]=component(delegate,k);
      selected=((PerFieldKnnVectorsFormat)formats[10]).getKnnVectorsFormatForField("field");
      formats[10]=new PerFieldKnnVectorsFormat(){public KnnVectorsFormat getKnnVectorsFormatForField(String f){nativeSelects++;nativeTrace=hash(nativeTrace,token(f));return selected;}};
    }
    void reset(){Arrays.fill(counts,0);trace=BASIS;displays=0;nativeSelects=0;nativeTrace=BASIS;fail=false;}
    Object get(int k){counts[k]++;trace=hash(trace,k);if(fail)throw failure;return formats[k];}
    public DocValuesFormat docValuesFormat(){return(DocValuesFormat)get(0);}
    public FieldInfosFormat fieldInfosFormat(){return(FieldInfosFormat)get(1);}
    public LiveDocsFormat liveDocsFormat(){return(LiveDocsFormat)get(2);}
    public NormsFormat normsFormat(){return(NormsFormat)get(3);}
    public PostingsFormat postingsFormat(){return(PostingsFormat)get(4);}
    public SegmentInfoFormat segmentInfoFormat(){return(SegmentInfoFormat)get(5);}
    public StoredFieldsFormat storedFieldsFormat(){return(StoredFieldsFormat)get(6);}
    public TermVectorsFormat termVectorsFormat(){return(TermVectorsFormat)get(7);}
    public CompoundFormat compoundFormat(){return(CompoundFormat)get(8);}
    public PointsFormat pointsFormat(){return(PointsFormat)get(9);}
    public KnnVectorsFormat knnVectorsFormat(){return(KnnVectorsFormat)get(10);}
    public String toString(){displays++;return next?"ProbeNext":"ProbeFirst";}
  }
  static final class HookCodec extends AssertingCodec {
    final long[] counts=new long[3];long trace=BASIS;
    final Object[] alternatives={new AssertingPostingsFormat(),new AssertingDocValuesFormat(),new AssertingKnnVectorsFormat()};
    void reset(){Arrays.fill(counts,0);trace=BASIS;}
    Object hook(int k,String f,Object normal){counts[k]++;trace=hash(trace,k*4+token(f));return"alternate".equals(f)?alternatives[k]:normal;}
    public PostingsFormat getPostingsFormatForField(String f){return(PostingsFormat)hook(0,f,super.getPostingsFormatForField(f));}
    public DocValuesFormat getDocValuesFormatForField(String f){return(DocValuesFormat)hook(1,f,super.getDocValuesFormatForField(f));}
    public KnnVectorsFormat getKnnVectorsFormatForField(String f){return(KnnVectorsFormat)hook(2,f,super.getKnnVectorsFormatForField(f));}
  }
  static Object direct(AssertingCodec c,int k,String f){return switch(k){case 0->c.getPostingsFormatForField(f);case 1->c.getDocValuesFormatForField(f);case 2->c.getKnnVectorsFormatForField(f);default->throw new AssertionError(k);};}
  static Object route(Object c,int k,String f){return switch(k){case 0->((PerFieldPostingsFormat)c).getPostingsFormatForField(f);case 1->((PerFieldDocValuesFormat)c).getDocValuesFormatForField(f);case 2->((PerFieldKnnVectorsFormat)c).getKnnVectorsFormatForField(f);default->throw new AssertionError(k);};}
  static final class Bundle {
    final HookCodec codec=new HookCodec();final Probe probe=new Probe();final Object[] expected=new Object[11],defaults=new Object[3],routers=new Object[3];final String[] display=new String[2];
    Bundle()throws Exception{Field f=FilterCodec.class.getDeclaredField("delegate");f.setAccessible(true);f.set(codec,probe);for(int k=0;k<11;k++)expected[k]=component(codec,k);
      String[] cache={"postings","docValues","knnVectorsFormat"};
      for(int k=0;k<3;k++){defaults[k]=direct(codec,k,"field");routers[k]=member(codec,cache[k]);}
      for(int i=0;i<2;i++){probe.next=i!=0;display[i]=codec.toString();}
    }
    Object select(int k,int path,String f){if(path==0)return direct(codec,k,f);return route(path==1?routers[k]:component(codec,COMPONENT_KINDS[k]),k,f);}
  }
  static final Bundle[] BUNDLES=new Bundle[3];static AssertingCodec memoryCodec;
  @RunWith(RandomizedRunner.class)
  public static class ContextBase {
    @Test public void prepare()throws Exception{Class<?> c=getClass();int index=c==Mask0.class?0:c==Mask21.class?1:2;BUNDLES[index]=new Bundle();if(index==0)memoryCodec=new AssertingCodec();}
  }
  @LuceneTestCase.SuppressAssertingFormats({}) public static class Mask0 extends ContextBase {}
  @LuceneTestCase.SuppressAssertingFormats({AssertingCodec.Format.STORED_FIELDS,AssertingCodec.Format.NORMS,AssertingCodec.Format.POINTS}) public static class Mask21 extends ContextBase {}
  @LuceneTestCase.SuppressAssertingFormats({AssertingCodec.Format.STORED_FIELDS,AssertingCodec.Format.TERM_VECTORS,AssertingCodec.Format.NORMS,AssertingCodec.Format.LIVE_DOCS,AssertingCodec.Format.POINTS,AssertingCodec.Format.KNN_VECTORS}) public static class Mask63 extends ContextBase {}
  static long observe(Object actual,Object expected,long d,long event){objectSink=actual;check(actual==expected,"identity");return hash(d,event);}
  static long total(long[] values){long sum=0;for(long v:values)sum+=v;return sum;}
  static long batch(int which){for(Bundle b:BUNDLES){b.probe.reset();b.codec.reset();b.probe.fail=which==6;}long d=BASIS;
    switch(which){
      case 0,1,2->{int m=CASE_MASKS[which];Bundle b=BUNDLES[m];for(int step=0;step<256;step++)for(int k=0;k<11;k++)for(int pair=0;pair<2;pair++)d=observe(component(b.codec,k),b.expected[k],d,k*3+(routed(MASKS[m],k)?2:0));check(total(b.probe.counts)==FACTORY_TOTALS[which],"factories");}
      case 3,4->{int m=which==3?0:2;Bundle b=BUNDLES[m];for(int step=0;step<256;step++)for(int k=0;k<3;k++)for(int f=0;f<4;f++)for(int path=0;path<3;path++){boolean bypass=which==4&&k==2&&path==2;int category=bypass?2:f==2?1:0;Object expected=bypass?b.probe.selected:f==2?b.codec.alternatives[k]:b.defaults[k];d=observe(b.select(k,path,FIELDS[f]),expected,d,((k*4+f)*3+path)*3+category);}check(total(b.codec.counts)==(which==3?9216:8192),"hooks");check(b.probe.counts[10]==(which==3?0:1024),"KNN factories");check(b.probe.nativeSelects==(which==3?0:1024),"native selections");}
      case 5->{for(int step=0;step<256;step++)for(Bundle b:BUNDLES){b.probe.next=step%2!=0;String value=b.codec.toString();objectSink=value;check(value.equals(b.display[step%2]),"display");check(b.codec.getName().equals("Asserting"),"name");for(int i=0;i<value.length();i++)d=hash(d,value.charAt(i));}for(Bundle b:BUNDLES)check(b.probe.displays==256,"display count");}
      case 6->{for(int step=0;step<256;step++)for(int m=0;m<3;m++){Bundle b=BUNDLES[m];for(int k=0;k<11;k++){boolean forwarded=routed(MASKS[m],k);try{Object value=component(b.codec,k);check(!forwarded,"missing failure");d=observe(value,b.expected[k],d,(m*11+k)*2);}catch(IllegalStateException error){check(forwarded&&error==b.probe.failure,"error identity");objectSink=error;d=hash(d,(m*11+k)*2+1);}}}for(int m=0;m<3;m++)check(total(BUNDLES[m].probe.counts)==ERROR_TOTALS[m],"failed factories");}
      default->throw new AssertionError(which);
    }
    for(Bundle b:BUNDLES){Probe p=b.probe;d=hash(d,total(p.counts));for(long c:p.counts)d=hash(d,c);d=hash(d,p.displays);d=hash(d,p.nativeSelects);d=hash(d,p.nativeTrace);d=hash(d,p.trace);for(long c:b.codec.counts)d=hash(d,c);d=hash(d,b.codec.trace);}sink=d;return d;
  }
  static void memory()throws Exception{
    long shallow=RamUsageEstimator.shallowSizeOf(memoryCodec),wrappers=0,routers=0;
    for(String f:new String[]{"vectors","storedFields","norms","liveDocs","defaultFormat","defaultDVFormat","pointsFormat","defaultKnnVectorsFormat"})wrappers+=RamUsageEstimator.shallowSizeOf(member(memoryCodec,f));
    for(String f:new String[]{"postings","docValues","knnVectorsFormat"})routers+=RamUsageEstimator.shallowSizeOf(member(memoryCodec,f));
    long suppression=RamUsageEstimator.shallowSizeOf(member(memoryCodec,"suppressedFormats"));
    String[] names={"shallow_codec","cached_wrappers","cached_routers","shared_selector","suppression","selected_sum"};long[] values={shallow,wrappers,routers,0,suppression,shallow+wrappers+routers+suppression};
    for(int i=0;i<names.length;i++)System.out.println("asserting_codec_"+names[i]+"_bytes\t"+values[i]);
  }
  public static void main(String[] args)throws Exception{
    for(Class<?> c:new Class<?>[]{Mask0.class,Mask21.class,Mask63.class}){var r=JUnitCore.runClasses(c);if(!r.wasSuccessful()||r.getRunCount()!=1){for(var f:r.getFailures())System.err.println(f.getTrace());throw new AssertionError("context failed: "+c);}}
    long warmup=args.length>0?Long.parseLong(args[0]):1000,measure=args.length>1?Long.parseLong(args[1]):1500;memory();
    String[] names={"asserting_codec_cached","asserting_codec_suppressed","asserting_codec_mixed","asserting_codec_selectors","asserting_codec_knn_bypass","asserting_codec_display","asserting_codec_errors"};
    for(int which=0;which<names.length;which++){long expected=batch(which);for(int phase=0;phase<2;phase++){long start=System.nanoTime(),operations=0,budget=(phase==0?warmup:measure)*1000000L;do{check(batch(which)==expected,"digest changed");operations+=256;}while(System.nanoTime()-start<budget);if(phase==1)System.out.printf(Locale.ROOT,"%s\t%.3f\t%d%n",names[which],(double)(System.nanoTime()-start)/operations,operations);}System.out.println("#check\t"+names[which]+"\t"+Long.toUnsignedString(expected)+"\t256");}
  }
}
