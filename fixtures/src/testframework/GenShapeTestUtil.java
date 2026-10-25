import com.carrotsearch.randomizedtesting.RandomizedContext;
import com.carrotsearch.randomizedtesting.RandomizedRunner;
import com.carrotsearch.randomizedtesting.Randomness;
import java.nio.file.*;
import java.util.*;
import org.apache.lucene.geo.*;
import org.apache.lucene.tests.geo.ShapeTestUtil;
import org.junit.Test;
import org.junit.runner.*;
import org.junit.runner.notification.Failure;

/** Actual ShapeTestUtil, with java.util.Random explicitly installed in context. */
public class GenShapeTestUtil {
  static final StringBuilder OUT=new StringBuilder();
  static String bits(float... values){StringJoiner out=new StringJoiner(",");for(float value:values)out.add(Integer.toUnsignedString(Float.floatToRawIntBits(value)));return out.toString();}
  static String polygon(XYPolygon value){return bits(value.getPolyX())+";"+bits(value.getPolyY());}
  static String result(int op,Random random){
    return switch(op){
      case 0->bits(ShapeTestUtil.nextFloat(random));
      case 1->{var b=ShapeTestUtil.nextBox(random);yield bits(b.minX,b.maxX,b.minY,b.maxY);}
      case 2->{var p=ShapeTestUtil.nextXYPoint();yield bits(p.getX(),p.getY());}
      case 3->{var p=ShapeTestUtil.nextCircle();yield bits(p.getX(),p.getY(),p.getRadius());}
      case 4->polygon(ShapeTestUtil.nextPolygon());
      case 5->{var p=ShapeTestUtil.nextLine();yield bits(p.getX())+";"+bits(p.getY());}
      default->throw new AssertionError();
    };
  }
  @RunWith(RandomizedRunner.class)
  public static class Dump {
    @Test public void dump()throws Exception{
      for(long seed=0;seed<128;seed++)for(int op=0;op<6;op++){
        final long s=seed;final int operation=op;
        RandomizedContext.current().runWithPrivateRandomness(new Randomness(seed,Random::new),()->{
          Random r=RandomizedContext.current().getRandom();
          for(int at=0;at<(operation==0?32:operation<4?8:4);at++)
            OUT.append(s).append('\t').append(operation).append('\t').append(at).append('\t').append(result(operation,r)).append('\t').append(r.nextInt()).append('\n');
          return null;
        });
      }
    }
  }
  static class PickRandom extends Random {
    final int pick;boolean first=true;
    PickRandom(long seed,int pick){super(seed);this.pick=pick;}
    @Override public int nextInt(int bound){if(first){if(bound!=47)throw new AssertionError();first=false;return pick;}return super.nextInt(bound);}
  }
  static void scripted(Path out)throws Exception{
    StringBuilder text=new StringBuilder();
    for(int pick=0;pick<47;pick++)for(long seed=0;seed<8;seed++){
      Random r=new PickRandom(seed,pick);
      text.append(pick).append('\t').append(seed).append('\t').append(bits(ShapeTestUtil.nextFloat(r))).append('\t').append(r.nextInt()).append('\n');
    }
    Files.writeString(out.resolve("shape_test_util_scripted.tsv"),text);
    text.setLength(0);
    double[][] centers={{1,7,5},{0,0,0},{Float.MAX_VALUE,-Float.MAX_VALUE,Double.MAX_VALUE},{1e10,-1e10,1e8},{-0.0,0.0,0.25},
      {0.0,0.0,-0.0},{-0.0,-0.0,-0.0},{1,7,-5},
      {Double.NaN,0,1},{0,Double.NaN,1},{0,0,Double.NaN},
      {Double.POSITIVE_INFINITY,0,1},{Double.NEGATIVE_INFINITY,0,1},
      {0,Double.POSITIVE_INFINITY,1},{0,Double.NEGATIVE_INFINITY,1},
      {1,7,Double.POSITIVE_INFINITY},{1,7,Double.NEGATIVE_INFINITY},
      {Float.MAX_VALUE,-Float.MAX_VALUE,-0.0}};
    for(int c=0;c<centers.length;c++)for(int n:new int[]{1,2,3,4,7,64,500}){
      var v=centers[c];String value;
      try{value=polygon(ShapeTestUtil.createRegularPolygon(v[0],v[1],v[2],n));}catch(IllegalArgumentException e){value="error";}
      text.append(c).append('\t').append(n).append('\t').append(value).append('\n');
    }
    Files.writeString(out.resolve("shape_test_util_regular.tsv"),text);
    text.setLength(0);
    XYPolygon outer=new XYPolygon(new float[]{0,4,4,0,0},new float[]{0,0,4,4,0});
    XYPolygon triangle=new XYPolygon(new float[]{-2,3,0,-2},new float[]{0,0,5,0});
    XYPolygon hole=new XYPolygon(new float[]{1,2,2,1,1},new float[]{1,1,2,2,1});
    XYPolygon holes=new XYPolygon(outer.getPolyX(),outer.getPolyY(),hole);
    for(int poly=0;poly<3;poly++)for(double x:new double[]{-3,-2,-0.0,0,1,2,3,4,5,Double.NaN})for(double y:new double[]{-1,0,1,2,4,5,Double.NaN}){
      String value;try{value=""+ShapeTestUtil.containsSlowly(new XYPolygon[]{outer,triangle,holes}[poly],x,y);}catch(UnsupportedOperationException e){value=e.getMessage();}
      text.append(poly).append('\t').append(Long.toUnsignedString(Double.doubleToRawLongBits(x))).append('\t').append(Long.toUnsignedString(Double.doubleToRawLongBits(y))).append('\t').append(value).append('\n');
    }
    Files.writeString(out.resolve("shape_test_util_contains.tsv"),text);
  }
  public static void main(String[] args)throws Exception{
    Path out=Path.of(args[0]);Files.createDirectories(out);
    if(args.length>=3&&args[1].equals("--worker")){
      Result result=JUnitCore.runClasses(Dump.class);for(Failure f:result.getFailures())System.err.println(f.getTrace());if(!result.wasSuccessful()||result.getRunCount()!=1)throw new AssertionError("context fixture failed");
      Files.writeString(out.resolve("shape_test_util_"+args[2]+".tsv"),OUT);return;
    }
    scripted(out);
    for(boolean nightly:new boolean[]{false,true}){
      Process process=new ProcessBuilder(Path.of(System.getProperty("java.home"),"bin","java").toString(),"-ea","-Dtests.nightly="+nightly,"-cp",System.getProperty("java.class.path"),"GenShapeTestUtil",out.toString(),"--worker",""+nightly).inheritIO().start();
      if(process.waitFor()!=0)throw new AssertionError("ShapeTestUtil worker failed");
    }
  }
}
