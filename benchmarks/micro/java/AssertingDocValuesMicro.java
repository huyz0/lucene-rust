import java.nio.file.*;
import java.util.*;
import org.apache.lucene.codecs.*;
import org.apache.lucene.codecs.lucene90.Lucene90DocValuesFormat;
import org.apache.lucene.codecs.lucene104.Lucene104Codec;
import org.apache.lucene.index.*;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.store.*;
import org.apache.lucene.util.*;

/** Paired actual asserting wrappers: same native bytes, complete data/output digests. */
public class AssertingDocValuesMicro {
  static volatile long sink;
  static final DocValuesFormat FORMAT=createFormat();
  static DocValuesFormat createFormat(){try{return GenAssertingDocValues.format();}catch(Exception error){throw new ExceptionInInitializerError(error);}}
  static final class Digest {long value;void mix(long n){value=value*31+n;}void bytes(BytesRef bytes){mix(bytes.length);for(int i=0;i<bytes.length;i++)mix(Byte.toUnsignedInt(bytes.bytes[bytes.offset+i]));}}
  static void digestValues(DocValuesProducer producer,FieldInfos fields,Digest digest)throws Exception {
    for(FieldInfo field:fields){digest.mix(field.number);switch(field.getDocValuesType()){
      case NUMERIC->{var v=producer.getNumeric(field);for(int doc=v.nextDoc();doc!=DocIdSetIterator.NO_MORE_DOCS;doc=v.nextDoc()){digest.mix(doc);digest.mix(v.longValue());}}
      case BINARY->{var v=producer.getBinary(field);for(int doc=v.nextDoc();doc!=DocIdSetIterator.NO_MORE_DOCS;doc=v.nextDoc()){digest.mix(doc);digest.bytes(v.binaryValue());}}
      case SORTED->{var v=producer.getSorted(field);digest.mix(v.getValueCount());for(int ord=0;ord<v.getValueCount();ord++)digest.bytes(v.lookupOrd(ord));for(int doc=v.nextDoc();doc!=DocIdSetIterator.NO_MORE_DOCS;doc=v.nextDoc()){digest.mix(doc);digest.mix(v.ordValue());}}
      case SORTED_NUMERIC->{var v=producer.getSortedNumeric(field);for(int doc=v.nextDoc();doc!=DocIdSetIterator.NO_MORE_DOCS;doc=v.nextDoc()){digest.mix(doc);int count=v.docValueCount();digest.mix(count);for(int i=0;i<count;i++)digest.mix(v.nextValue());}}
      case SORTED_SET->{var v=producer.getSortedSet(field);digest.mix(v.getValueCount());for(long ord=0;ord<v.getValueCount();ord++)digest.bytes(v.lookupOrd(ord));for(int doc=v.nextDoc();doc!=DocIdSetIterator.NO_MORE_DOCS;doc=v.nextDoc()){digest.mix(doc);int count=v.docValueCount();digest.mix(count);for(int i=0;i<count;i++)digest.mix(v.nextOrd());}}
      default->{}
    }
    if(field.docValuesSkipIndexType()!=DocValuesSkipIndexType.NONE){var s=producer.getSkipper(field);digest.mix(s.minValue());digest.mix(s.maxValue());digest.mix(s.docCount());digest.mix(s.maxValueCount());for(int target=0;target<12;){s.advance(target);if(s.minDocID(0)==DocIdSetIterator.NO_MORE_DOCS)break;for(int level=0;level<s.numLevels();level++){digest.mix(s.minDocID(level));digest.mix(s.maxDocID(level));digest.mix(s.minValue(level));digest.mix(s.maxValue(level));digest.mix(s.docCount(level));}target=s.maxDocID(0)+1;}}
    }
  }
  static final class NoopConsumer extends DocValuesConsumer {
    long calls;
    void add(FieldInfo field){calls=calls*31+field.number;}
    public void addNumericField(FieldInfo f,DocValuesProducer p){add(f);}public void addBinaryField(FieldInfo f,DocValuesProducer p){add(f);}public void addSortedField(FieldInfo f,DocValuesProducer p){add(f);}public void addSortedNumericField(FieldInfo f,DocValuesProducer p){add(f);}public void addSortedSetField(FieldInfo f,DocValuesProducer p){add(f);}public void close(){}
  }
  static long batch(String name,Directory directory,FieldInfos fields,DocValuesProducer prepared,DocValuesProducer source,NoopConsumer noop,DocValuesConsumer validator)throws Exception {
    var digest=new Digest();for(int i=0;i<16;i++)switch(name){
      case "validate"->{noop.calls=0;for(FieldInfo field:fields)GenCrankyDocValues.add(validator,field,source);digest.mix(noop.calls);}
      case "read"->{try(var producer=FORMAT.fieldsProducer(GenCrankyDocValues.read(directory,fields,12))){producer.checkIntegrity();digestValues(producer,fields,digest);}}
      case "write"->{try(var target=new ByteBuffersDirectory()){try(var consumer=FORMAT.fieldsConsumer(GenCrankyDocValues.write(target,fields,12))){for(FieldInfo field:fields)GenCrankyDocValues.add(consumer,field,source);}for(String ext:GenCrankyDocValues.EXTENSIONS)try(var input=target.openInput("_cdv_s."+ext,IOContext.DEFAULT)){byte[] bytes=new byte[Math.toIntExact(input.length())];input.readBytes(bytes,0,bytes.length);digest.bytes(new BytesRef(bytes));}}}
      case "merge_lifecycle"->{try(var producer=prepared.getMergeInstance()){digestValues(producer,fields,digest);}}
      default->throw new AssertionError(name);
    }return digest.value;
  }
  static long[] run(String name,Directory directory,FieldInfos fields,DocValuesProducer prepared,DocValuesProducer source,NoopConsumer noop,DocValuesConsumer validator,int ms)throws Exception {long expected=batch(name,directory,fields,prepared,source,noop,validator),start=System.nanoTime(),count=0;do{long got=batch(name,directory,fields,prepared,source,noop,validator);if(got!=expected)throw new AssertionError("timed digest changed");sink=got;count+=16;}while(System.nanoTime()-start<ms*1_000_000L);return new long[]{count,System.nanoTime()-start};}
  public static void main(String[] args)throws Exception {
    if(!AssertingDocValuesMicro.class.desiredAssertionStatus())throw new AssertionError("run with -ea");int warmup=args.length>0?Integer.parseInt(args[0]):1000,measure=args.length>1?Integer.parseInt(args[1]):1500;
    try(var directory=new ByteBuffersDirectory()){
      for(String ext:new String[]{"dvm","dvd","dvs","fnm"}){byte[] bytes=Files.readAllBytes(Path.of("fixtures/testframework/data/asserting_doc_values."+ext));try(var output=directory.createOutput(ext.equals("fnm")?"_cdv.fnm":"_cdv_s."+ext,IOContext.DEFAULT)){output.writeBytes(bytes,bytes.length);}}
      var fields=new Lucene104Codec().fieldInfosFormat().read(directory,GenCrankyDocValues.segment(directory,12),"",IOContext.DEFAULT);
      var noop=new NoopConsumer();var validator=GenAssertingDocValues.wrap(noop,12);
      try(var source=new Lucene90DocValuesFormat().fieldsProducer(GenCrankyDocValues.read(directory,fields,12));var prepared=FORMAT.fieldsProducer(GenCrankyDocValues.read(directory,fields,12))){
        System.out.printf("#memory\tasserting_doc_values_producer_wrapper\t%d%n",RamUsageEstimator.shallowSizeOf(prepared));System.out.printf("#memory\tasserting_doc_values_consumer_wrapper\t%d%n",RamUsageEstimator.shallowSizeOf(validator));
        var skipper=prepared.getSkipper(fields.fieldInfo("numeric"));System.out.printf("#memory\tasserting_doc_values_skipper_wrapper\t%d%n",RamUsageEstimator.shallowSizeOf(skipper));
        for(String name:new String[]{"validate","read","write","merge_lifecycle"}){System.out.printf("#check\tasserting_doc_values_%s\t%d\t16%n",name,batch(name,directory,fields,prepared,source,noop,validator));run(name,directory,fields,prepared,source,noop,validator,warmup);long[] result=run(name,directory,fields,prepared,source,noop,validator,measure);System.out.printf(Locale.ROOT,"asserting_doc_values_%s\t%.3f\t%d%n",name,(double)result[1]/result[0],result[0]);}
      }
    }
  }
}