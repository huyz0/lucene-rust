import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.HexFormat;
import java.util.List;
import java.util.Set;
import org.apache.lucene.index.FreqAndNormBuffer;
import org.apache.lucene.index.Impacts;
import org.apache.lucene.index.ImpactsEnum;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.util.BytesRef;

/** Actual AssertingLeafReader.AssertingImpactsEnum and CheckIndex checks, with -ea.
 * The borrowed AssertingImpacts stale-view guard is outside the Rust owned-Vec API.
 * Java CheckIndexException is recorded separately from AssertionError; Rust's
 * infallible impacts() channel can only report both by panic.
 */
public class GenAssertingImpacts {
  static final int END = DocIdSetIterator.NO_MORE_DOCS;
  static final StringBuilder OUT = new StringBuilder();
  static String hex(String value) { return HexFormat.of().formatHex(value.getBytes(StandardCharsets.UTF_8)); }
  static String ints(int[] values) { return Arrays.stream(values).mapToObj(Integer::toString).reduce((a,b)->a+","+b).orElse(""); }
  static void row(Object... fields) {
    for (int i=0;i<fields.length;i++) { if(i!=0) OUT.append('\t'); OUT.append(fields[i]); }
    OUT.append('\n');
  }
  record Level(int upto, int[] freqs, long[] norms) {}
  static Level level(int upto, long... pairs) {
    int[] freqs = new int[pairs.length/2]; long[] norms = new long[freqs.length];
    for(int i=0;i<freqs.length;i++) { freqs[i]=Math.toIntExact(pairs[2*i]); norms[i]=pairs[2*i+1]; }
    return new Level(upto,freqs,norms);
  }
  static String pairs(Level level) {
    List<String> pairs=new ArrayList<>();
    for(int i=0;i<level.freqs.length;i++) pairs.add(level.freqs[i]+"/"+Long.toUnsignedString(level.norms[i]));
    return String.join(",",pairs);
  }
  static class Case {
    final String name, actions;
    int initial=-1, freq=2;
    int[] docs={0,2,7}, positions={0,2}, starts={0,4}, ends={2,6};
    byte[] payload={1,(byte)255};
    String faults="";
    List<Level> levels=new ArrayList<>(List.of(level(END,1,1,3,7)));
    Case(String name,String actions) { this.name=name; this.actions=actions; }
    Case levels(Level... values) { levels=new ArrayList<>(List.of(values)); return this; }
    Case fault(String name) { faults=name; return this; }
  }
  static final class Probe extends ImpactsEnum {
    final Case c; final List<String> events=new ArrayList<>(); final Set<String> faults;
    int doc, index=-1, position;
    Probe(Case c) { this.c=c; doc=c.initial; faults=new HashSet<>(List.of(c.faults.split(","))); }
    void operation(String name,String event) throws IOException {
      events.add(event);
      if(faults.remove(name)) throw new IOException("fixture "+name+" failure");
    }
    @Override public int docID() { return doc; }
    @Override public int nextDoc() throws IOException {
      operation("next","next"); position=0; index++;
      return doc=index<c.docs.length?c.docs[index]:END;
    }
    @Override public int advance(int target) throws IOException {
      operation("advance","advance:"+target); position=0;
      do { index++; } while(index<c.docs.length && c.docs[index]<target);
      return doc=index<c.docs.length?c.docs[index]:END;
    }
    @Override public long cost() { events.add("cost"); return c.docs.length; }
    @Override public int freq() { events.add("freq"); return c.freq; }
    @Override public int nextPosition() throws IOException {
      operation("position","position"); int at=position++;
      return at<c.positions.length?c.positions[at]:-1;
    }
    @Override public int startOffset() { events.add("start"); int at=position-1; return at<c.starts.length?c.starts[at]:-1; }
    @Override public int endOffset() { events.add("end"); int at=position-1; return at<c.ends.length?c.ends[at]:-1; }
    @Override public BytesRef getPayload() { events.add("payload"); return c.payload==null?null:new BytesRef(c.payload); }
    @Override public void advanceShallow(int target) throws IOException { operation("shallow","shallow:"+target); }
    @Override public Impacts getImpacts() {
      events.add("impacts");
      return new Impacts() {
        @Override public int numLevels() { return c.levels.size(); }
        @Override public int getDocIdUpTo(int i) { return c.levels.get(i).upto; }
        @Override public FreqAndNormBuffer getImpacts(int i) {
          Level level=c.levels.get(i); FreqAndNormBuffer buffer=new FreqAndNormBuffer();
          buffer.freqs=level.freqs.clone(); buffer.norms=level.norms.clone(); buffer.size=level.freqs.length;
          return buffer;
        }
      };
    }
    @Override public String toString() { return "ProbeImpacts"; }
  }
  static ImpactsEnum wrap(ImpactsEnum delegate) throws Exception {
    var type=Class.forName("org.apache.lucene.tests.index.AssertingLeafReader$AssertingImpactsEnum");
    var ctor=type.getDeclaredConstructor(ImpactsEnum.class); ctor.setAccessible(true);
    return (ImpactsEnum)ctor.newInstance(delegate);
  }
  static String snapshot(Impacts impacts) {
    List<String> levels=new ArrayList<>();
    for(int i=0;i<impacts.numLevels();i++) {
      FreqAndNormBuffer buffer=impacts.getImpacts(i); List<String> pairs=new ArrayList<>();
      for(int j=0;j<buffer.size;j++) pairs.add(buffer.freqs[j]+"/"+Long.toUnsignedString(buffer.norms[j]));
      levels.add(impacts.getDocIdUpTo(i)+":"+String.join(",",pairs));
    }
    return String.join(";",levels);
  }
  static String action(ImpactsEnum wrapper,Probe probe,String op,int arg) throws Exception {
    return switch(op) {
      case "D" -> Integer.toString(wrapper.docID());
      case "C" -> Long.toString(wrapper.cost());
      case "N" -> Integer.toString(wrapper.nextDoc());
      case "A" -> Integer.toString(wrapper.advance(arg));
      case "S" -> { wrapper.advanceShallow(arg); yield ""; }
      case "I" -> snapshot(wrapper.getImpacts());
      case "F" -> Integer.toString(wrapper.freq());
      case "P" -> Integer.toString(wrapper.nextPosition());
      case "O" -> Integer.toString(wrapper.startOffset());
      case "E" -> Integer.toString(wrapper.endOffset());
      case "Y" -> { BytesRef payload=wrapper.getPayload(); yield payload==null?"null":HexFormat.of().formatHex(Arrays.copyOfRange(payload.bytes,payload.offset,payload.offset+payload.length)); }
      // Test setup: arm a delegate fault after the wrapper has consumed a position.
      case "X" -> { probe.faults.add(switch(arg) { case 1 -> "next"; case 2 -> "advance"; default -> throw new IllegalArgumentException("fault "+arg); }); yield ""; }
      default -> throw new IllegalArgumentException(op);
    };
  }
  static void run(Case c) throws Exception {
    row("case",c.name,c.initial,c.freq,ints(c.docs),ints(c.positions),ints(c.starts),ints(c.ends),
        c.payload==null?"null":HexFormat.of().formatHex(c.payload),c.faults);
    for(Level level:c.levels) row("level",c.name,level.upto,pairs(level));
    Probe probe=new Probe(c); ImpactsEnum wrapper=wrap(probe);
    for(String command:c.actions.split(" ")) {
      String[] parts=command.split(":"); String op=parts[0]; int arg=parts.length==1?0:Integer.parseInt(parts[1]);
      probe.events.clear(); String status="ok",type="",value="",message="";
      try { value=action(wrapper,probe,op,arg); }
      catch(AssertionError error) { status="assert"; type=error.getClass().getSimpleName(); message=String.valueOf(error.getMessage()); }
      catch(IOException error) { status="io"; type=error.getClass().getSimpleName(); message=String.valueOf(error.getMessage()); }
      catch(RuntimeException error) {
        if(!error.getClass().getSimpleName().equals("CheckIndexException")) throw error;
        status="check"; type=error.getClass().getSimpleName(); message=String.valueOf(error.getMessage());
      }
      row("step",c.name,op,arg,status,type,hex(value),hex(message),String.join(",",probe.events),probe.doc,probe.position);
    }
  }
  static List<Case> cases() {
    List<Case> all=new ArrayList<>(); Case c;
    all.add(new Case("valid_walk","D C S:0 I N F P O E Y P Y N F A:7 P N I"));
    all.add(new Case("unpositioned_impacts","I S:-1 I S:0 I"));
    all.add(new Case("repeated_shallow","S:2 S:2 I S:1 I A:2 D"));
    all.add(new Case("shallow_before_doc","A:7 S:6 S:7 I"));
    all.add(new Case("shallow_blocks_next","S:2 N A:1 A:2 N"));
    all.add(new Case("shallow_error_retains_target","S:7 S:6 N A:7 I").fault("shallow"));
    all.add(new Case("next_error_retains_state","N F N F P").fault("next"));
    all.add(new Case("advance_error_retains_state","A:2 F N A:2 I").fault("advance"));
    all.add(new Case("next_error_retains_consumed_position","N P X:1 N O P P"));
    all.add(new Case("advance_error_retains_consumed_position","N P X:2 A:2 O P P"));
    all.add(new Case("position_error_retains_budget","N P P P P").fault("position"));
    all.add(new Case("postings_unpositioned","F P O E Y C"));
    all.add(new Case("postings_before_position","N O E Y P O E Y"));
    all.add(new Case("position_budget_and_reset","N P P P N O P"));
    all.add(new Case("advance_equal_doc","N A:0 D"));
    all.add(new Case("finished_guards","A:2147483647 F P O E Y N A:2147483647 I"));
    all.add(new Case("finished_shallow_overflow","A:2147483647 S:2147483647 N"));
    c=new Case("initially_positioned_impacts","D I F"); c.initial=2; all.add(c);
    c=new Case("duplicate_doc","N N D"); c.docs=new int[]{0,0}; all.add(c);
    c=new Case("descending_doc","N N D"); c.docs=new int[]{2,0}; all.add(c);
    c=new Case("negative_doc","N D"); c.docs=new int[]{-1}; all.add(c);
    c=new Case("zero_freq","N F P"); c.freq=0; all.add(c);
    c=new Case("negative_position","N P P P"); c.positions=new int[]{-2,0}; all.add(c);
    c=new Case("minus_one_position","N P O E Y"); c.positions=new int[]{-1,0}; all.add(c);
    c=new Case("descending_positions_allowed","N P P"); c.positions=new int[]{9,1}; all.add(c);
    c=new Case("max_position_allowed","N P P"); c.positions=new int[]{Integer.MAX_VALUE,1}; all.add(c);
    c=new Case("empty_payload","N P Y"); c.payload=new byte[0]; all.add(c);
    c=new Case("null_payload","N P Y"); c.payload=null; all.add(c);
    c=new Case("offset_values_unchecked","N P O E"); c.starts=new int[]{-9}; c.ends=new int[]{-20}; all.add(c);
    all.add(new Case("no_levels","S:0 I").levels());
    all.add(new Case("level_below_target","S:5 I").levels(level(4,1,1)));
    all.add(new Case("level_below_doc","A:7 I").levels(level(6,1,1)));
    all.add(new Case("decreasing_level_bounds","S:0 I").levels(level(9,1,1),level(8,1,1)));
    all.add(new Case("equal_level_bounds_allowed","S:0 I").levels(level(9,1,1),level(9,1,1)));
    all.add(new Case("empty_impacts","S:0 I").levels(level(9)));
    all.add(new Case("empty_upper_impacts","S:0 I").levels(level(9,1,1),level(20)));
    all.add(new Case("first_freq_zero","S:0 I").levels(level(9,0,1)));
    all.add(new Case("first_freq_negative","S:0 I").levels(level(9,-1,1)));
    all.add(new Case("first_norm_zero","S:0 I").levels(level(9,1,0)));
    all.add(new Case("duplicate_freq","S:0 I").levels(level(9,1,1,1,2)));
    all.add(new Case("duplicate_norm","S:0 I").levels(level(9,1,1,2,1)));
    all.add(new Case("decreasing_freq","S:0 I").levels(level(9,3,1,2,2)));
    all.add(new Case("decreasing_unsigned_norm","S:0 I").levels(level(9,1,Long.MIN_VALUE,2,Long.MAX_VALUE)));
    all.add(new Case("unsigned_norm_extremes","S:0 I").levels(level(9,1,1,2,Long.MAX_VALUE,3,Long.MIN_VALUE,4,-1)));
    all.add(new Case("later_pairs_compare_only_first","S:0 I").levels(level(9,1,1,5,5,3,3)));
    all.add(new Case("later_pair_equal_first_rejected","S:0 I").levels(level(9,1,1,5,5,1,3)));
    all.add(new Case("dominance_same_pair","S:0 I").levels(level(9,2,7),level(20,2,7)));
    all.add(new Case("dominance_better_upper","S:0 I").levels(level(9,1,3,3,9),level(20,5,1)));
    all.add(new Case("dominance_no_upper_frequency","S:0 I").levels(level(9,3,1),level(20,2,1)));
    all.add(new Case("dominance_no_upper_norm","S:0 I").levels(level(9,1,1),level(20,2,2)));
    all.add(new Case("dominance_cursor_last_unchecked","S:0 I").levels(level(9,3,1),level(20,1,1,2,2)));
    all.add(new Case("dominance_cursor_exhausted","S:0 I").levels(level(9,3,1,4,2),level(20,1,1,2,2)));
    all.add(new Case("dominance_cursor_does_not_retry","S:0 I").levels(level(9,3,1,4,2),level(20,1,1,2,2,5,3)));
    all.add(new Case("dominance_unsigned_valid","S:0 I").levels(level(9,1,-1),level(20,1,Long.MIN_VALUE)));
    all.add(new Case("dominance_unsigned_invalid","S:0 I").levels(level(9,1,Long.MAX_VALUE),level(20,1,Long.MIN_VALUE)));
    all.add(new Case("three_levels","S:0 I").levels(level(9,1,3),level(20,2,2),level(END,3,1)));
    return all;
  }
  public static void main(String[] args) throws Exception {
    if(!GenAssertingImpacts.class.desiredAssertionStatus()) throw new AssertionError("generator requires -ea");
    for(Case c:cases()) run(c);
    Path out=Path.of(args[0]); Files.createDirectories(out);
    Files.writeString(out.resolve("asserting_impacts.tsv"),OUT.toString(),StandardCharsets.UTF_8);
  }
}
