package org.apache.lucene.tests.index;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import org.apache.lucene.index.FreqAndNormBuffer;
import org.apache.lucene.index.Impacts;
import org.apache.lucene.index.ImpactsEnum;
import org.apache.lucene.util.BytesRef;
import org.apache.lucene.util.RamUsageEstimator;

/** Actual asserting impacts wrapper over shared synthetic inputs and traces. */
public final class AssertingImpactsMicro {
  static volatile long sink;
  record Op(int kind, int arg) {}
  static final class Input {
    int[] docs, freqs, bounds;
    FreqAndNormBuffer[] levels;
    Op[] trace;
    long expected;
    Input(Path path) throws Exception {
      List<Integer> docsList = new ArrayList<>(), freqsList = new ArrayList<>();
      List<Integer> boundsList = new ArrayList<>();
      List<FreqAndNormBuffer> buffers = new ArrayList<>();
      List<Op> operations = new ArrayList<>();
      for (String line : Files.readAllLines(path)) {
        String[] row = line.split("\t");
        switch (row[0]) {
          case "D" -> { docsList.add(Integer.parseInt(row[1])); freqsList.add(Integer.parseInt(row[2])); }
          case "L" -> {
            int level = Integer.parseInt(row[1]);
            if (buffers.size() == level) { buffers.add(new FreqAndNormBuffer()); boundsList.add(Integer.parseInt(row[2])); }
            buffers.get(level).add(Integer.parseInt(row[3]), Long.parseUnsignedLong(row[4]));
          }
          case "T" -> operations.add(new Op(Integer.parseInt(row[1]), Integer.parseInt(row[2])));
          case "Z" -> expected = Long.parseUnsignedLong(row[1]);
          default -> throw new AssertionError("unknown input row");
        }
      }
      docs = docsList.stream().mapToInt(Integer::intValue).toArray();
      freqs = freqsList.stream().mapToInt(Integer::intValue).toArray();
      bounds = boundsList.stream().mapToInt(Integer::intValue).toArray();
      levels = buffers.toArray(FreqAndNormBuffer[]::new);
      trace = operations.toArray(Op[]::new);
    }
  }
  static final class Delegate extends ImpactsEnum {
    final Input input;
    int index = -1, position;
    final byte[] payload = new byte[2];
    final BytesRef payloadRef = new BytesRef(payload);
    final Impacts impacts;
    Delegate(Input input) {
      this.input = input;
      impacts = new Impacts() {
        @Override public int numLevels() { return input.levels.length; }
        @Override public int getDocIdUpTo(int level) { return input.bounds[level]; }
        @Override public FreqAndNormBuffer getImpacts(int level) { return input.levels[level]; }
      };
    }
    @Override public int docID() { return index < 0 ? -1 : index == input.docs.length ? NO_MORE_DOCS : input.docs[index]; }
    @Override public int nextDoc() { index++; position = 0; return docID(); }
    @Override public int advance(int target) { do { index++; } while (index < input.docs.length && input.docs[index] < target); position = 0; return docID(); }
    @Override public long cost() { return input.docs.length; }
    @Override public int freq() { return input.freqs[index]; }
    @Override public int nextPosition() {
      int value = 1 + 3 * position++;
      payload[0] = (byte) docID(); payload[1] = (byte) position;
      return value;
    }
    @Override public int startOffset() { return (1 + 3 * (position - 1)) * 5; }
    @Override public int endOffset() { return startOffset() + 2; }
    @Override public BytesRef getPayload() {
      if ((position & 1) == 0) return null;
      return payloadRef;
    }
    @Override public void advanceShallow(int target) {}
    @Override public Impacts getImpacts() { return impacts; }
  }
  static long fold(long hash, long value) { return hash * 31 + value; }
  static long traverse(ImpactsEnum iterator, Input input) throws Exception {
    long hash = 0;
    for (Op op : input.trace) {
      hash = fold(hash, op.kind);
      switch (op.kind) {
        case 0 -> iterator.advanceShallow(op.arg);
        case 1 -> hash = fold(hash, iterator.nextDoc());
        case 2 -> hash = fold(hash, iterator.advance(op.arg));
        case 3 -> {
          Impacts impacts = iterator.getImpacts();
          int levels = impacts.numLevels(); hash = fold(hash, levels);
          for (int level = 0; level < levels; level++) {
            hash = fold(hash, impacts.getDocIdUpTo(level));
            FreqAndNormBuffer pairs = impacts.getImpacts(level); hash = fold(hash, pairs.size);
            for (int i = 0; i < pairs.size; i++) { hash = fold(hash, pairs.freqs[i]); hash = fold(hash, pairs.norms[i]); }
          }
        }
        case 4 -> hash = fold(hash, iterator.freq());
        case 5 -> hash = fold(hash, iterator.nextPosition());
        case 6 -> { hash = fold(hash, iterator.startOffset()); hash = fold(hash, iterator.endOffset()); }
        case 7 -> {
          BytesRef payload = iterator.getPayload(); hash = fold(hash, payload == null ? -1 : payload.length);
          if (payload != null) for (int i = 0; i < payload.length; i++) hash = fold(hash, Byte.toUnsignedInt(payload.bytes[payload.offset + i]));
        }
        case 8 -> hash = fold(hash, iterator.docID());
        case 9 -> hash = fold(hash, iterator.cost());
        default -> throw new AssertionError("unknown operation");
      }
    }
    return hash;
  }
  static long[] run(Input input, long millis) throws Exception {
    long deadline = System.nanoTime() + millis * 1_000_000L, nanos = 0, count = 0;
    do {
      ImpactsEnum iterator = new AssertingLeafReader.AssertingImpactsEnum(new Delegate(input));
      long start = System.nanoTime();
      long actual = traverse(iterator, input);
      nanos += System.nanoTime() - start;
      if (actual != input.expected) throw new AssertionError("trace digest differs: " + Long.toUnsignedString(actual));
      sink = actual; count++;
    } while (System.nanoTime() < deadline);
    return new long[] {nanos, count};
  }
  // Use in a separate JVM so the raw delegate cannot pollute the normal
  // wrapper's type profile at traverse's virtual call sites.
  static long[] runRaw(Input input, long millis) throws Exception {
    long deadline = System.nanoTime() + millis * 1_000_000L, nanos = 0, count = 0;
    do {
      ImpactsEnum iterator = new Delegate(input);
      long start = System.nanoTime();
      long actual = traverse(iterator, input);
      nanos += System.nanoTime() - start;
      if (actual != input.expected) throw new AssertionError("trace digest differs: " + Long.toUnsignedString(actual));
      sink = actual; count++;
    } while (System.nanoTime() < deadline);
    return new long[] {nanos, count};
  }
  public static void main(String[] args) throws Exception {
    if (!AssertingImpactsMicro.class.desiredAssertionStatus()) throw new AssertionError("requires -ea");
    long warmup = Long.parseLong(args[1]), measure = Long.parseLong(args[2]);
    if (warmup < 0 || measure <= 0) throw new IllegalArgumentException("invalid durations");
    if (args.length > 3 && args[3].equals("raw")) {
      Input input = new Input(Path.of(args[0], "single.tsv"));
      runRaw(input, warmup); long[] result = runRaw(input, measure);
      System.out.printf(Locale.ROOT, "asserting_impacts_single_raw\t%.3f\t%d%n", (double) result[0] / result[1], result[1]);
      System.out.printf("#check\tasserting_impacts_single_raw\t%s\t%d\t%d%n", Long.toUnsignedString(input.expected), input.trace.length, input.docs.length);
      return;
    }
    boolean memoryReported = false;
    for (String name : new String[] {"single", "multilevel", "positions"}) {
      Input input = new Input(Path.of(args[0], name + ".tsv"));
      if (!memoryReported) {
        var wrapper = new AssertingLeafReader.AssertingImpactsEnum(new Delegate(input));
        var field = AssertingLeafReader.AssertingImpactsEnum.class.getDeclaredField("assertingPostings");
        field.setAccessible(true);
        System.out.println("asserting_impacts_wrapper_shallow_bytes\t"
            + (RamUsageEstimator.shallowSizeOf(wrapper) + RamUsageEstimator.shallowSizeOf(field.get(wrapper))));
        memoryReported = true;
      }
      run(input, warmup); long[] result = run(input, measure);
      System.out.printf(Locale.ROOT, "asserting_impacts_%s\t%.3f\t%d%n", name, (double) result[0] / result[1], result[1]);
      System.out.printf("#check\tasserting_impacts_%s\t%s\t%d\t%d%n", name, Long.toUnsignedString(input.expected), input.trace.length, input.docs.length);
    }
  }
}
