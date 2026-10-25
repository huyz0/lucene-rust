package org.apache.lucene.tests.index;

import java.nio.file.*;
import java.util.*;
import org.apache.lucene.tests.codecs.asserting.AssertingPointsFormat;
import org.apache.lucene.codecs.*;
import org.apache.lucene.codecs.lucene90.*;
import org.apache.lucene.index.*;
import org.apache.lucene.store.*;
import org.apache.lucene.util.*;

/** Actual asserting points and visitor; native Lucene90 bytes and both merge paths. */
public class AssertingPointsMicro {
  static volatile long sink;
  static final byte[] ID={0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
  static final int[][] MAPS={{0,-1,1,2,3,4,5,6,7,8},{9,-1,10,11,12,13,14,15,16,17}};
  static final FieldInfos FIELDS=new FieldInfos(new FieldInfo[]{field("one",0,1),field("two",1,2)});
  static final PointsFormat NATIVE=new PointsFormat() {
    @Override public PointsWriter fieldsWriter(SegmentWriteState state) throws java.io.IOException {return new Lucene90PointsWriter(state,4,16.0);}
    @Override public PointsReader fieldsReader(SegmentReadState state) throws java.io.IOException {return new Lucene90PointsReader(state);}
  };
  static FieldInfo field(String name,int number,int dims) {
    return new FieldInfo(name,number,false,false,false,IndexOptions.NONE,DocValuesType.NONE,
        DocValuesSkipIndexType.NONE,-1,Map.of(),dims,dims,2,0,VectorEncoding.FLOAT32,VectorSimilarityFunction.EUCLIDEAN,false,false);
  }
  static SegmentInfo segment(Directory directory,int maxDoc) {
    return new SegmentInfo(directory,Version.fromBits(10,5,0),null,"_cp",maxDoc,false,false,null,Map.of(),ID,Map.of(),null);
  }
  static SegmentReadState readState(Directory directory) {return new SegmentReadState(directory,segment(directory,10),FIELDS,IOContext.DEFAULT,"s");}
  static SegmentWriteState writeState(Directory directory,int maxDoc) {return new SegmentWriteState(InfoStream.NO_OUTPUT,directory,segment(directory,maxDoc),FIELDS,null,IOContext.DEFAULT,"s");}
  static MergeState mergeState(Directory directory,PointsReader reader) throws Exception {
    var constructor=MergeState.class.getDeclaredConstructor(List.class,SegmentInfo.class,InfoStream.class,java.util.concurrent.Executor.class,MergePolicy.OneMerge.class);
    constructor.setAccessible(true);MergeState state=constructor.newInstance(List.of(),segment(directory,-1),InfoStream.NO_OUTPUT,null,null);
    Map<String,Object> members=new HashMap<>();members.put("pointsReaders",new PointsReader[]{reader,reader});
    members.put("fieldInfos",new FieldInfos[]{FIELDS,FIELDS});members.put("maxDocs",new int[]{10,10});
    members.put("docMaps",new MergeState.DocMap[]{new MergeState.DocMap(){public int get(int doc){return MAPS[0][doc];}},new MergeState.DocMap(){public int get(int doc){return MAPS[1][doc];}}});
    for(var member:members.entrySet()) {var field=MergeState.class.getField(member.getKey());field.setAccessible(true);field.set(state,member.getValue());}
    state.mergeFieldInfos=FIELDS;return state;
  }
  static class Visitor implements PointValues.IntersectVisitor {
    long digest;Visitor(long digest){this.digest=digest;}
    @Override public void visit(int doc){digest=digest*31+doc;}
    @Override public void visit(int doc,byte[] value){visit(doc);for(byte b:value)digest=digest*31+Byte.toUnsignedInt(b);}
    @Override public PointValues.Relation compare(byte[] min,byte[] max){return PointValues.Relation.CELL_CROSSES_QUERY;}
  }
  static long read(PointsReader reader,long digest) throws Exception {
    reader.checkIntegrity();Visitor visitor=new Visitor(digest);
    for(String name:new String[]{"one","two"}) {
      PointValues values=reader.getValues(name);
      for(byte[] bytes:new byte[][]{values.getMinPackedValue(),values.getMaxPackedValue()})for(byte b:bytes)visitor.digest=visitor.digest*31+Byte.toUnsignedInt(b);
      for(long value:new long[]{values.getNumDimensions(),values.getNumIndexDimensions(),values.getBytesPerDimension(),values.size(),values.getDocCount()})visitor.digest=visitor.digest*31+value;
      PointValues.PointTree tree=values.getPointTree();
      for(byte[] bytes:new byte[][]{tree.getMinPackedValue(),tree.getMaxPackedValue()})for(byte b:bytes)visitor.digest=visitor.digest*31+Byte.toUnsignedInt(b);
      visitor.digest=visitor.digest*31+tree.size();tree.visitDocIDs(visitor);tree.visitDocValues(visitor);tree.clone().visitDocValues(visitor);
    }
    return visitor.digest;
  }
  static long batch(Directory saved,PointsReader source,MergeState nativeMerge,MergeState wrappedMerge,int which) throws Exception {
    long digest=0;
    for(int i=0;i<16;i++) {

      PointsFormat format=new AssertingPointsFormat(NATIVE);
      if(which==4){Visitor consumer=new Visitor(digest);var visitor=new AssertingLeafReader.AssertingIntersectVisitor(1,1,2,consumer);visitor.grow(13);visitor.compare(new byte[]{0,0},new byte[]{1,(byte)188});for(int point=0;point<13;point++){int value=point*37;visitor.visit(point%10,new byte[]{(byte)(value>>>8),(byte)value});}digest=consumer.digest;}
      else if(which==0)try(var reader=format.fieldsReader(readState(saved))) {digest=read(reader,digest);}
      else try(var directory=new ByteBuffersDirectory()) {
        try(var writer=(which==3?NATIVE:format).fieldsWriter(writeState(directory,which==1?10:18))) {
          if(which==1){for(FieldInfo field:FIELDS)writer.writeField(field,source);writer.finish();}
          else writer.merge(which==2?nativeMerge:wrappedMerge);
        }
        for(String extension:new String[]{"kdm","kdi","kdd"})try(var input=directory.openInput("_cp_s."+extension,IOContext.DEFAULT)) {
          long length=input.length();for(long p=0;p<length;p++)digest=digest*31+Byte.toUnsignedInt(input.readByte());
        }
      }

    }
    return digest;
  }
  static long[] run(Directory saved,PointsReader source,MergeState nativeMerge,MergeState wrappedMerge,int which,long millis,long expected) throws Exception {
    long start=System.nanoTime(),count=0;
    do {long actual=batch(saved,source,nativeMerge,wrappedMerge,which);if(actual!=expected)throw new AssertionError("unstable digest");sink=actual;count+=16;}
    while(System.nanoTime()-start<millis*1_000_000L);return new long[]{System.nanoTime()-start,count};
  }
  public static void main(String[] args) throws Exception {
    if(!AssertingPointsMicro.class.desiredAssertionStatus())throw new AssertionError("run with -ea");
    long warmup=Long.parseLong(args[0]),measure=Long.parseLong(args[1]);
    Visitor counter=new Visitor(0);var sample=new AssertingLeafReader.AssertingIntersectVisitor(1,1,2,counter);
    long owned=RamUsageEstimator.shallowSizeOf(sample)+RamUsageEstimator.sizeOf(sample.lastDocValue)+RamUsageEstimator.sizeOf(sample.lastMinPackedValue)+RamUsageEstimator.sizeOf(sample.lastMaxPackedValue);
    System.out.printf("asserting_points_visitor_owned_bytes\t%d%n",owned);
    try(var saved=new ByteBuffersDirectory()) {
      for(String extension:new String[]{"kdm","kdi","kdd"})try(var output=saved.createOutput("_cp_s."+extension,IOContext.DEFAULT)) {byte[] bytes=Files.readAllBytes(Path.of("fixtures/testframework/data/asserting_points."+extension));output.writeBytes(bytes,0,bytes.length);}

      try(var source=NATIVE.fieldsReader(readState(saved));var wrapped=new AssertingPointsFormat(NATIVE).fieldsReader(readState(saved))) {
        MergeState nativeMerge=mergeState(saved,source),wrappedMerge=mergeState(saved,wrapped);
        String[] cases={"asserting_points_read","asserting_points_write","asserting_points_merge_native","asserting_points_merge_wrapped","asserting_points_validation"};
        for(int which=0;which<cases.length;which++) {
          long expected=batch(saved,source,nativeMerge,wrappedMerge,which);
          System.out.printf("#check\t%s\t%s\t16%n",cases[which],Long.toUnsignedString(expected));
          run(saved,source,nativeMerge,wrappedMerge,which,warmup,expected);
          long[] result=run(saved,source,nativeMerge,wrappedMerge,which,measure,expected);
          System.out.printf(Locale.ROOT,"%s\t%.3f\t%d%n",cases[which],(double)result[0]/result[1],result[1]);
        }


      }
    }
  }
}
