import com.carrotsearch.randomizedtesting.*;
import java.util.*;
import java.util.concurrent.atomic.AtomicLong;
import org.apache.lucene.geo.*;
import org.apache.lucene.tests.geo.ShapeTestUtil;
import org.apache.lucene.util.RamUsageEstimator;
import org.junit.Test;
import org.junit.runner.*;
import org.junit.runner.notification.Failure;

/** Actual Java utility, context setup excluded; context lookups remain timed. */
public class ShapeTestUtilMicro {
 static volatile long sink;static long warmup,measure;
 static long values(long d,float[] values){d=d*31+values.length;for(float v:values)d=d*31+Integer.toUnsignedLong(Float.floatToRawIntBits(v));return d;}
 static long polygon(long d,XYPolygon p){return values(values(d,p.getPolyX()),p.getPolyY());}
 static long batch(int which,Random random,XYPolygon oracle){
  random.setSeed(42);long d=0;
  switch(which){
   case 0->{for(int i=0;i<512;i++)d=d*31+Integer.toUnsignedLong(Float.floatToRawIntBits(ShapeTestUtil.nextFloat(random)));}
   case 1->{for(int i=0;i<32;i++){var b=ShapeTestUtil.nextBox(random);d=values(d,new float[]{b.minX,b.maxX,b.minY,b.maxY});var p=ShapeTestUtil.nextXYPoint();d=values(d,new float[]{p.getX(),p.getY()});var c=ShapeTestUtil.nextCircle();d=values(d,new float[]{c.getX(),c.getY(),c.getRadius()});}}
   case 2->{for(int i=0;i<8;i++){d=polygon(d,ShapeTestUtil.nextPolygon());var l=ShapeTestUtil.nextLine();d=values(values(d,l.getX()),l.getY());}}
   case 3->{for(int i=0;i<16;i++)d=polygon(d,ShapeTestUtil.createRegularPolygon(1,7,5,64));}
   case 4->{for(int i=0;i<1024;i++)d=d*31+(ShapeTestUtil.containsSlowly(oracle,(i%17)/4.0-1,(i%19)/4.0-1)?1:0);}
   default->throw new AssertionError();
  }
  return d*31+Integer.toUnsignedLong(random.nextInt());
 }
 static long[] run(int which,Random random,XYPolygon oracle,long millis,long expected){long start=System.nanoTime(),count=0;do{long actual=batch(which,random,oracle);if(actual!=expected)throw new AssertionError("full digest changed");sink=actual;count++;}while(System.nanoTime()-start<millis*1_000_000);return new long[]{System.nanoTime()-start,count};}
 @RunWith(RandomizedRunner.class) public static class Work {
  @Test public void benchmark()throws Exception{
   Random random=new Random(42);
   XYPolygon oracle=new XYPolygon(new float[]{0,3,3,1,0},new float[]{0,0,2,4,0});
   System.out.printf("shape_test_util_rng_owned_bytes\t%d%n",RamUsageEstimator.shallowSizeOf(random)+RamUsageEstimator.shallowSizeOfInstance(AtomicLong.class));
   System.out.printf("shape_test_util_polygon_owned_bytes\t%d%n",RamUsageEstimator.shallowSizeOf(oracle)+RamUsageEstimator.sizeOf(oracle.getPolyX())+RamUsageEstimator.sizeOf(oracle.getPolyY())+RamUsageEstimator.shallowSizeOf(oracle.getHoles()));
   RandomizedContext.current().runWithPrivateRandomness(new Randomness(42,seed->random),()->{
    String[] names={"shape_biased_floats","shape_primitives","shape_polygon_line","shape_regular_polygon","shape_contains_slowly"};
    for(int which=0;which<names.length;which++){long expected=batch(which,random,oracle);System.out.printf("#check\t%s\t%s\t1%n",names[which],Long.toUnsignedString(expected));run(which,random,oracle,warmup,expected);long[] result=run(which,random,oracle,measure,expected);System.out.printf(Locale.ROOT,"%s\t%.3f\t%d%n",names[which],(double)result[0]/result[1],result[1]);}
    return null;
   });
  }
 }
 public static void main(String[] args){warmup=Long.parseLong(args[0]);measure=Long.parseLong(args[1]);Result result=JUnitCore.runClasses(Work.class);for(Failure f:result.getFailures())System.err.println(f.getTrace());if(!result.wasSuccessful()||result.getRunCount()!=1)throw new AssertionError("benchmark context failed");}
}
