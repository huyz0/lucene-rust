import com.carrotsearch.randomizedtesting.RandomizedContext;
import com.carrotsearch.randomizedtesting.RandomizedRunner;
import com.carrotsearch.randomizedtesting.Randomness;
import java.lang.reflect.InvocationTargetException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.atomic.AtomicReference;
import org.apache.lucene.codecs.*;
import org.apache.lucene.codecs.lucene104.Lucene104Codec;
import org.apache.lucene.codecs.perfield.*;
import org.apache.lucene.tests.codecs.asserting.*;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.junit.Test;
import org.junit.runner.JUnitCore;
import org.junit.runner.RunWith;

/** Actual AssertingCodec composition and selectors; no component I/O or SPI claims. */
public class GenAssertingCodec {
  static final StringBuilder OUT = new StringBuilder();
  static final String[] KINDS = {"docValues", "fieldInfos", "liveDocs", "norms", "postings",
      "segmentInfo", "storedFields", "termVectors", "compound", "points", "knnVectors"};
  static final String[] CACHE = {"docValues", null, "liveDocs", "norms", "postings", null,
      "storedFields", "vectors", null, "pointsFormat", "knnVectorsFormat"};
  static final String[] EAGER = {"postings", "docValues", "knnVectorsFormat", "vectors",
      "storedFields", "norms", "liveDocs", "defaultFormat", "defaultDVFormat",
      "pointsFormat", "defaultKnnVectorsFormat"};
  static final int[] SUPPRESSION = {-1, -1, 3, 2, -1, -1, 0, 1, -1, 4, 5};
  static void check(boolean value, String label) { if (!value) throw new AssertionError(label); }
  static void row(Object... fields) {
    for (int i = 0; i < fields.length; i++) { if (i != 0) OUT.append('\t'); OUT.append(fields[i]); }
    OUT.append('\n');
  }
  static Object member(Object object, String name) throws Exception {
    for (Class<?> type = object.getClass(); type != null; type = type.getSuperclass()) {
      try { var field = type.getDeclaredField(name); field.setAccessible(true); return field.get(object); }
      catch (NoSuchFieldException missing) { /* Search inherited state. */ }
    }
    throw new NoSuchFieldException(name);
  }
  static Object component(Codec codec, int kind) {
    return switch (kind) {
      case 0 -> codec.docValuesFormat(); case 1 -> codec.fieldInfosFormat();
      case 2 -> codec.liveDocsFormat(); case 3 -> codec.normsFormat();
      case 4 -> codec.postingsFormat(); case 5 -> codec.segmentInfoFormat();
      case 6 -> codec.storedFieldsFormat(); case 7 -> codec.termVectorsFormat();
      case 8 -> codec.compoundFormat(); case 9 -> codec.pointsFormat();
      case 10 -> codec.knnVectorsFormat(); default -> throw new AssertionError(kind);
    };
  }
  static String type(Object value) {
    Class<?> type = value.getClass();
    while (type.isAnonymousClass()) type = type.getSuperclass();
    return type.getSimpleName();
  }
  static String name(Object value) {
    if (value instanceof PostingsFormat format) return format.getName();
    if (value instanceof DocValuesFormat format) return format.getName();
    if (value instanceof KnnVectorsFormat format) return format.getName();
    return "-";
  }
  static boolean routed(int mask, int kind) {
    return CACHE[kind] == null || (SUPPRESSION[kind] >= 0 && (mask & (1 << SUPPRESSION[kind])) != 0);
  }
  static String display(String delegate, int mask) {
    var flags = new StringJoiner(", ");
    for (var format : AssertingCodec.Format.values())
      if ((mask & (1 << format.ordinal())) != 0) flags.add(format.toString());
    return "Asserting(" + delegate + (mask == 0 ? "" : ", suppressed: " + flags) + ")";
  }
  static void replaceDelegate(AssertingCodec codec, Codec delegate) throws Exception {
    var field = FilterCodec.class.getDeclaredField("delegate"); field.setAccessible(true); field.set(codec, delegate);
  }
  static final class Probe extends FilterCodec {
    final List<String> events = new ArrayList<>();
    final Object[] last = new Object[KINDS.length];
    final IllegalStateException failure = new IllegalStateException("probe");
    int failKind = -1;
    boolean failDisplay;
    String display = "ProbeFirst";
    Probe() { super("Probe", new Lucene104Codec()); }
    void before(int kind) { events.add("factory:" + KINDS[kind]); if (kind == failKind) throw failure; }
    <T> T result(int kind, T result) { last[kind] = result; return result; }
    @Override public DocValuesFormat docValuesFormat() { before(0); return result(0, delegate.docValuesFormat()); }
    @Override public FieldInfosFormat fieldInfosFormat() { before(1); return result(1, delegate.fieldInfosFormat()); }
    @Override public LiveDocsFormat liveDocsFormat() { before(2); return result(2, delegate.liveDocsFormat()); }
    @Override public NormsFormat normsFormat() { before(3); return result(3, delegate.normsFormat()); }
    @Override public PostingsFormat postingsFormat() { before(4); return result(4, delegate.postingsFormat()); }
    @Override public SegmentInfoFormat segmentInfoFormat() { before(5); return result(5, delegate.segmentInfoFormat()); }
    @Override public StoredFieldsFormat storedFieldsFormat() { before(6); return result(6, delegate.storedFieldsFormat()); }
    @Override public TermVectorsFormat termVectorsFormat() { before(7); return result(7, delegate.termVectorsFormat()); }
    @Override public CompoundFormat compoundFormat() { before(8); return result(8, delegate.compoundFormat()); }
    @Override public PointsFormat pointsFormat() { before(9); return result(9, delegate.pointsFormat()); }
    @Override public KnnVectorsFormat knnVectorsFormat() { before(10); return result(10, delegate.knnVectorsFormat()); }
    @Override public String toString() { events.add("display"); if (failDisplay) throw failure; return display; }
  }
  static Object select(AssertingCodec codec, int kind, String field) {
    return switch (kind) {
      case 0 -> codec.getPostingsFormatForField(field);
      case 1 -> codec.getDocValuesFormatForField(field);
      case 2 -> codec.getKnnVectorsFormatForField(field);
      default -> throw new AssertionError(kind);
    };
  }
  static Object route(Object router, int kind, String field) {
    return switch (kind) {
      case 0 -> ((PerFieldPostingsFormat) router).getPostingsFormatForField(field);
      case 1 -> ((PerFieldDocValuesFormat) router).getDocValuesFormatForField(field);
      case 2 -> ((PerFieldKnnVectorsFormat) router).getKnnVectorsFormatForField(field);
      default -> throw new AssertionError(kind);
    };
  }
  static class HookCodec extends AssertingCodec {
    final List<String> events = new ArrayList<>();
    final Object[] alternatives = {new AssertingPostingsFormat(), new AssertingDocValuesFormat(), new AssertingKnnVectorsFormat()};
    final IllegalStateException failure = new IllegalStateException("selector");
    int failKind = -1;
    Object hook(int kind, String field, Object normal) {
      check(events != null, "constructor selected a field");
      events.add(kind + ":" + String.valueOf(field));
      if (failKind == kind) throw failure;
      return "alternate".equals(field) ? alternatives[kind] : normal;
    }
    @Override public PostingsFormat getPostingsFormatForField(String field) {
      return (PostingsFormat) hook(0, field, super.getPostingsFormatForField(field));
    }
    @Override public DocValuesFormat getDocValuesFormatForField(String field) {
      return (DocValuesFormat) hook(1, field, super.getDocValuesFormatForField(field));
    }
    @Override public KnnVectorsFormat getKnnVectorsFormatForField(String field) {
      return (KnnVectorsFormat) hook(2, field, super.getKnnVectorsFormatForField(field));
    }
  }
  static void hooks(String label, int mask) throws Exception {
    var codec = new HookCodec();
    check(codec.events.isEmpty(), "eager selector callback");
    String[] cacheNames = {"postings", "docValues", "knnVectorsFormat"};
    int[] componentKinds = {4, 0, 10};
    for (int kind = 0; kind < 3; kind++) {
      Object internal = member(codec, cacheNames[kind]);
      for (String field : new String[] {"", "field", "alternate", null}) {
        codec.events.clear(); Object selected = select(codec, kind, field);
        check(codec.events.equals(List.of(kind + ":" + String.valueOf(field))), "direct selector trace");
        codec.events.clear(); Object actual = route(internal, kind, field);
        check(actual == selected && codec.events.equals(List.of(kind + ":" + String.valueOf(field))), "internal selector trace");
        codec.events.clear(); Object publicResult = route(component(codec, componentKinds[kind]), kind, field);
        boolean bypass = kind == 2 && (mask & 32) != 0;
        check(bypass ? codec.events.isEmpty() : publicResult == selected && codec.events.equals(List.of(kind + ":" + String.valueOf(field))), "public selector trace");
        row("hook", label, kind, String.valueOf(field), type(selected), name(selected), "identity", bypass ? "delegate" : "hook", type(publicResult), name(publicResult));
      }
      codec.failKind = kind;
      for (int path = 0; path < 3; path++) {
        codec.events.clear(); boolean failed = false;
        try {
          if (path == 0) select(codec, kind, "failure");
          else route(path == 1 ? internal : component(codec, componentKinds[kind]), kind, "failure");
        } catch (IllegalStateException error) { check(error == codec.failure, "selector error identity"); failed = true; }
        boolean bypass = path == 2 && kind == 2 && (mask & 32) != 0;
        check(failed != bypass && codec.events.equals(bypass ? List.of() : List.of(kind + ":failure")), "selector failure trace");
        row("hook-failure", label, kind, path, failed ? "runtime:selector" : "ok", String.join(",", codec.events));
      }
      codec.failKind = -1;
    }
  }
  static void snapshot(String label, int mask, Random random, Random control) throws Exception {
    var codec = new AssertingCodec(); var second = new AssertingCodec();
    Codec nativeDelegate = (Codec) member(codec, "delegate");
    check(codec.getName().equals("Asserting") && codec.toString().equals(display(nativeDelegate.toString(), mask)), "constructor metadata");
    check(member(codec, "suppressedFormats").toString().equals(maskSet(mask).toString()), "suppression set");
    row("config", label, mask, codec.getName(), type(nativeDelegate), nativeDelegate.getName(), codec.toString());
    // Reflection proves even suppressed objects exist. Lexical constructor order is source-only evidence.
    for (String field : EAGER) {
      Object value = member(codec, field);
      check(value != null && value != member(second, field), "independent eager cache");
      if (value instanceof PerFieldPostingsFormat || value instanceof PerFieldDocValuesFormat || value instanceof PerFieldKnnVectorsFormat)
        row("cache", label, field, type(value), name(value), "independent");
      else {
        Object inner = member(value, field.equals("defaultKnnVectorsFormat") ? "delegate" : "in");
        row("cache", label, field, type(value), name(value), "independent", type(inner), name(inner));
      }
    }
    Object assertingKnn = member(codec, "defaultKnnVectorsFormat");
    Object defaultKnn = member(assertingKnn, "delegate");
    Object delegateKnn = ((PerFieldKnnVectorsFormat) nativeDelegate.knnVectorsFormat()).getKnnVectorsFormatForField("field");
    row("knn-defaults", label, member(defaultKnn, "maxConn"), member(defaultKnn, "beamWidth"), member(defaultKnn, "tinySegmentsThreshold"), member(defaultKnn, "numMergeWorkers"), member(delegateKnn, "tinySegmentsThreshold"), ((KnnVectorsFormat) assertingKnn).getMaxDimensions("field"));
    for (int kind = 0; kind < 3; kind++) {
      Object expected = member(codec, new String[] {"defaultFormat", "defaultDVFormat", "defaultKnnVectorsFormat"}[kind]);
      for (String field : new String[] {"", "field", null}) {
        Object actual = select(codec, kind, field); check(actual == expected, "default selector identity");
        row("default-selector", label, kind, String.valueOf(field), type(actual), name(actual), "cached");
      }
    }
    for (int kind = 0; kind < KINDS.length; kind++) {
      Object first = component(codec, kind), again = component(codec, kind);
      check((first == again) == !(kind == 9 && routed(mask, kind)), "native accessor identity");
      if (!routed(mask, kind)) check(first == member(codec, CACHE[kind]), "cached accessor identity");
      row("accessor", label, KINDS[kind], type(first), name(first), first == again, routed(mask, kind) ? "delegate" : "cached");
    }
    var probe = new Probe(); replaceDelegate(codec, probe);
    for (int kind = 0; kind < KINDS.length; kind++) {
      probe.events.clear(); Object first = component(codec, kind), firstDelegate = probe.last[kind], again = component(codec, kind);
      List<String> expected = routed(mask, kind) ? List.of("factory:" + KINDS[kind], "factory:" + KINDS[kind]) : List.of();
      check(probe.events.equals(expected), "factory trace");
      check(routed(mask, kind) ? first == firstDelegate && again == probe.last[kind] : first == member(codec, CACHE[kind]) && first == again, "factory identity");
      row("probe", label, KINDS[kind], first == again, String.join(",", probe.events));
      probe.events.clear(); probe.failKind = kind; boolean failed = false;
      try { component(codec, kind); } catch (IllegalStateException error) { check(error == probe.failure, "factory error identity"); failed = true; }
      check(failed == routed(mask, kind) && probe.events.equals(failed ? List.of("factory:" + KINDS[kind]) : List.of()), "factory failure route");
      row("probe-failure", label, KINDS[kind], failed ? "runtime:probe" : "ok", String.join(",", probe.events));
      probe.failKind = -1;
    }
    for (String value : new String[] {"ProbeFirst", "ProbeNext"}) {
      probe.display = value; probe.events.clear(); String actual = codec.toString();
      check(actual.equals(display(value, mask)) && probe.events.equals(List.of("display")), "live display");
      row("display", label, actual, String.join(",", probe.events), codec.getName());
    }
    probe.failDisplay = true; probe.events.clear(); boolean failed = false;
    try { codec.toString(); } catch (IllegalStateException error) { check(error == probe.failure, "display error identity"); failed = true; }
    check(failed && probe.events.equals(List.of("display")), "display error trace");
    row("display-failure", label, "runtime:probe", String.join(",", probe.events));
    hooks(label, mask);
    int tail = random.nextInt(); check(tail == control.nextInt(), "unexpected random draw");
    row("random-tail", label, tail);
  }
  static EnumSet<AssertingCodec.Format> maskSet(int mask) {
    var set = EnumSet.noneOf(AssertingCodec.Format.class);
    for (var value : AssertingCodec.Format.values()) if ((mask & (1 << value.ordinal())) != 0) set.add(value);
    return set;
  }
  @RunWith(RandomizedRunner.class)
  public static class ContextBase {
    @Test public void dump() throws Exception {
      Class<?> target = RandomizedContext.current().getTargetClass();
      var annotation = target.getAnnotation(LuceneTestCase.SuppressAssertingFormats.class);
      int mask = 0;
      if (annotation != null) for (var format : annotation.value()) mask |= 1 << format.ordinal();
      int selected = mask; var random = new Random(42); var control = new Random(42);
      RandomizedContext.current().runWithPrivateRandomness(new Randomness(42, seed -> random), () -> {
        snapshot(target.getSimpleName(), selected, random, control); return null;
      });
    }
  }
  public static class Unannotated extends ContextBase {}
  @LuceneTestCase.SuppressAssertingFormats({AssertingCodec.Format.KNN_VECTORS, AssertingCodec.Format.POINTS, AssertingCodec.Format.STORED_FIELDS, AssertingCodec.Format.POINTS})
  public static class DuplicatedReversed extends ContextBase {}
  @LuceneTestCase.SuppressAssertingFormats({AssertingCodec.Format.NORMS, AssertingCodec.Format.TERM_VECTORS})
  public static class AnnotatedParent extends ContextBase {}
  public static class Inherited extends AnnotatedParent {}
  @LuceneTestCase.SuppressAssertingFormats({})
  public static class Mask00 extends ContextBase {}
  @LuceneTestCase.SuppressAssertingFormats({AssertingCodec.Format.STORED_FIELDS})
  public static class Mask01 extends ContextBase {}
  @LuceneTestCase.SuppressAssertingFormats({AssertingCodec.Format.TERM_VECTORS})
  public static class Mask02 extends ContextBase {}
  @LuceneTestCase.SuppressAssertingFormats({AssertingCodec.Format.STORED_FIELDS, AssertingCodec.Format.TERM_VECTORS})
  public static class Mask03 extends ContextBase {}
  @LuceneTestCase.SuppressAssertingFormats({AssertingCodec.Format.NORMS})
  public static class Mask04 extends ContextBase {}
  @LuceneTestCase.SuppressAssertingFormats({AssertingCodec.Format.STORED_FIELDS, AssertingCodec.Format.NORMS})
  public static class Mask05 extends ContextBase {}
  @LuceneTestCase.SuppressAssertingFormats({AssertingCodec.Format.TERM_VECTORS, AssertingCodec.Format.NORMS})
  public static class Mask06 extends ContextBase {}
  @LuceneTestCase.SuppressAssertingFormats({AssertingCodec.Format.STORED_FIELDS, AssertingCodec.Format.TERM_VECTORS, AssertingCodec.Format.NORMS})
  public static class Mask07 extends ContextBase {}
  @LuceneTestCase.SuppressAssertingFormats({AssertingCodec.Format.LIVE_DOCS})
  public static class Mask08 extends ContextBase {}
  @LuceneTestCase.SuppressAssertingFormats({AssertingCodec.Format.STORED_FIELDS, AssertingCodec.Format.LIVE_DOCS})
  public static class Mask09 extends ContextBase {}
  @LuceneTestCase.SuppressAssertingFormats({AssertingCodec.Format.TERM_VECTORS, AssertingCodec.Format.LIVE_DOCS})
  public static class Mask10 extends ContextBase {}
  @LuceneTestCase.SuppressAssertingFormats({AssertingCodec.Format.STORED_FIELDS, AssertingCodec.Format.TERM_VECTORS, AssertingCodec.Format.LIVE_DOCS})
  public static class Mask11 extends ContextBase {}
  @LuceneTestCase.SuppressAssertingFormats({AssertingCodec.Format.NORMS, AssertingCodec.Format.LIVE_DOCS})
  public static class Mask12 extends ContextBase {}
  @LuceneTestCase.SuppressAssertingFormats({AssertingCodec.Format.STORED_FIELDS, AssertingCodec.Format.NORMS, AssertingCodec.Format.LIVE_DOCS})
  public static class Mask13 extends ContextBase {}
  @LuceneTestCase.SuppressAssertingFormats({AssertingCodec.Format.TERM_VECTORS, AssertingCodec.Format.NORMS, AssertingCodec.Format.LIVE_DOCS})
  public static class Mask14 extends ContextBase {}
  @LuceneTestCase.SuppressAssertingFormats({AssertingCodec.Format.STORED_FIELDS, AssertingCodec.Format.TERM_VECTORS, AssertingCodec.Format.NORMS, AssertingCodec.Format.LIVE_DOCS})
  public static class Mask15 extends ContextBase {}
  @LuceneTestCase.SuppressAssertingFormats({AssertingCodec.Format.POINTS})
  public static class Mask16 extends ContextBase {}
  @LuceneTestCase.SuppressAssertingFormats({AssertingCodec.Format.STORED_FIELDS, AssertingCodec.Format.POINTS})
  public static class Mask17 extends ContextBase {}
  @LuceneTestCase.SuppressAssertingFormats({AssertingCodec.Format.TERM_VECTORS, AssertingCodec.Format.POINTS})
  public static class Mask18 extends ContextBase {}
  @LuceneTestCase.SuppressAssertingFormats({AssertingCodec.Format.STORED_FIELDS, AssertingCodec.Format.TERM_VECTORS, AssertingCodec.Format.POINTS})
  public static class Mask19 extends ContextBase {}
  @LuceneTestCase.SuppressAssertingFormats({AssertingCodec.Format.NORMS, AssertingCodec.Format.POINTS})
  public static class Mask20 extends ContextBase {}
  @LuceneTestCase.SuppressAssertingFormats({AssertingCodec.Format.STORED_FIELDS, AssertingCodec.Format.NORMS, AssertingCodec.Format.POINTS})
  public static class Mask21 extends ContextBase {}
  @LuceneTestCase.SuppressAssertingFormats({AssertingCodec.Format.TERM_VECTORS, AssertingCodec.Format.NORMS, AssertingCodec.Format.POINTS})
  public static class Mask22 extends ContextBase {}
  @LuceneTestCase.SuppressAssertingFormats({AssertingCodec.Format.STORED_FIELDS, AssertingCodec.Format.TERM_VECTORS, AssertingCodec.Format.NORMS, AssertingCodec.Format.POINTS})
  public static class Mask23 extends ContextBase {}
  @LuceneTestCase.SuppressAssertingFormats({AssertingCodec.Format.LIVE_DOCS, AssertingCodec.Format.POINTS})
  public static class Mask24 extends ContextBase {}
  @LuceneTestCase.SuppressAssertingFormats({AssertingCodec.Format.STORED_FIELDS, AssertingCodec.Format.LIVE_DOCS, AssertingCodec.Format.POINTS})
  public static class Mask25 extends ContextBase {}
  @LuceneTestCase.SuppressAssertingFormats({AssertingCodec.Format.TERM_VECTORS, AssertingCodec.Format.LIVE_DOCS, AssertingCodec.Format.POINTS})
  public static class Mask26 extends ContextBase {}
  @LuceneTestCase.SuppressAssertingFormats({AssertingCodec.Format.STORED_FIELDS, AssertingCodec.Format.TERM_VECTORS, AssertingCodec.Format.LIVE_DOCS, AssertingCodec.Format.POINTS})
  public static class Mask27 extends ContextBase {}
  @LuceneTestCase.SuppressAssertingFormats({AssertingCodec.Format.NORMS, AssertingCodec.Format.LIVE_DOCS, AssertingCodec.Format.POINTS})
  public static class Mask28 extends ContextBase {}
  @LuceneTestCase.SuppressAssertingFormats({AssertingCodec.Format.STORED_FIELDS, AssertingCodec.Format.NORMS, AssertingCodec.Format.LIVE_DOCS, AssertingCodec.Format.POINTS})
  public static class Mask29 extends ContextBase {}
  @LuceneTestCase.SuppressAssertingFormats({AssertingCodec.Format.TERM_VECTORS, AssertingCodec.Format.NORMS, AssertingCodec.Format.LIVE_DOCS, AssertingCodec.Format.POINTS})
  public static class Mask30 extends ContextBase {}
  @LuceneTestCase.SuppressAssertingFormats({AssertingCodec.Format.STORED_FIELDS, AssertingCodec.Format.TERM_VECTORS, AssertingCodec.Format.NORMS, AssertingCodec.Format.LIVE_DOCS, AssertingCodec.Format.POINTS})
  public static class Mask31 extends ContextBase {}
  @LuceneTestCase.SuppressAssertingFormats({AssertingCodec.Format.KNN_VECTORS})
  public static class Mask32 extends ContextBase {}
  @LuceneTestCase.SuppressAssertingFormats({AssertingCodec.Format.STORED_FIELDS, AssertingCodec.Format.KNN_VECTORS})
  public static class Mask33 extends ContextBase {}
  @LuceneTestCase.SuppressAssertingFormats({AssertingCodec.Format.TERM_VECTORS, AssertingCodec.Format.KNN_VECTORS})
  public static class Mask34 extends ContextBase {}
  @LuceneTestCase.SuppressAssertingFormats({AssertingCodec.Format.STORED_FIELDS, AssertingCodec.Format.TERM_VECTORS, AssertingCodec.Format.KNN_VECTORS})
  public static class Mask35 extends ContextBase {}
  @LuceneTestCase.SuppressAssertingFormats({AssertingCodec.Format.NORMS, AssertingCodec.Format.KNN_VECTORS})
  public static class Mask36 extends ContextBase {}
  @LuceneTestCase.SuppressAssertingFormats({AssertingCodec.Format.STORED_FIELDS, AssertingCodec.Format.NORMS, AssertingCodec.Format.KNN_VECTORS})
  public static class Mask37 extends ContextBase {}
  @LuceneTestCase.SuppressAssertingFormats({AssertingCodec.Format.TERM_VECTORS, AssertingCodec.Format.NORMS, AssertingCodec.Format.KNN_VECTORS})
  public static class Mask38 extends ContextBase {}
  @LuceneTestCase.SuppressAssertingFormats({AssertingCodec.Format.STORED_FIELDS, AssertingCodec.Format.TERM_VECTORS, AssertingCodec.Format.NORMS, AssertingCodec.Format.KNN_VECTORS})
  public static class Mask39 extends ContextBase {}
  @LuceneTestCase.SuppressAssertingFormats({AssertingCodec.Format.LIVE_DOCS, AssertingCodec.Format.KNN_VECTORS})
  public static class Mask40 extends ContextBase {}
  @LuceneTestCase.SuppressAssertingFormats({AssertingCodec.Format.STORED_FIELDS, AssertingCodec.Format.LIVE_DOCS, AssertingCodec.Format.KNN_VECTORS})
  public static class Mask41 extends ContextBase {}
  @LuceneTestCase.SuppressAssertingFormats({AssertingCodec.Format.TERM_VECTORS, AssertingCodec.Format.LIVE_DOCS, AssertingCodec.Format.KNN_VECTORS})
  public static class Mask42 extends ContextBase {}
  @LuceneTestCase.SuppressAssertingFormats({AssertingCodec.Format.STORED_FIELDS, AssertingCodec.Format.TERM_VECTORS, AssertingCodec.Format.LIVE_DOCS, AssertingCodec.Format.KNN_VECTORS})
  public static class Mask43 extends ContextBase {}
  @LuceneTestCase.SuppressAssertingFormats({AssertingCodec.Format.NORMS, AssertingCodec.Format.LIVE_DOCS, AssertingCodec.Format.KNN_VECTORS})
  public static class Mask44 extends ContextBase {}
  @LuceneTestCase.SuppressAssertingFormats({AssertingCodec.Format.STORED_FIELDS, AssertingCodec.Format.NORMS, AssertingCodec.Format.LIVE_DOCS, AssertingCodec.Format.KNN_VECTORS})
  public static class Mask45 extends ContextBase {}
  @LuceneTestCase.SuppressAssertingFormats({AssertingCodec.Format.TERM_VECTORS, AssertingCodec.Format.NORMS, AssertingCodec.Format.LIVE_DOCS, AssertingCodec.Format.KNN_VECTORS})
  public static class Mask46 extends ContextBase {}
  @LuceneTestCase.SuppressAssertingFormats({AssertingCodec.Format.STORED_FIELDS, AssertingCodec.Format.TERM_VECTORS, AssertingCodec.Format.NORMS, AssertingCodec.Format.LIVE_DOCS, AssertingCodec.Format.KNN_VECTORS})
  public static class Mask47 extends ContextBase {}
  @LuceneTestCase.SuppressAssertingFormats({AssertingCodec.Format.POINTS, AssertingCodec.Format.KNN_VECTORS})
  public static class Mask48 extends ContextBase {}
  @LuceneTestCase.SuppressAssertingFormats({AssertingCodec.Format.STORED_FIELDS, AssertingCodec.Format.POINTS, AssertingCodec.Format.KNN_VECTORS})
  public static class Mask49 extends ContextBase {}
  @LuceneTestCase.SuppressAssertingFormats({AssertingCodec.Format.TERM_VECTORS, AssertingCodec.Format.POINTS, AssertingCodec.Format.KNN_VECTORS})
  public static class Mask50 extends ContextBase {}
  @LuceneTestCase.SuppressAssertingFormats({AssertingCodec.Format.STORED_FIELDS, AssertingCodec.Format.TERM_VECTORS, AssertingCodec.Format.POINTS, AssertingCodec.Format.KNN_VECTORS})
  public static class Mask51 extends ContextBase {}
  @LuceneTestCase.SuppressAssertingFormats({AssertingCodec.Format.NORMS, AssertingCodec.Format.POINTS, AssertingCodec.Format.KNN_VECTORS})
  public static class Mask52 extends ContextBase {}
  @LuceneTestCase.SuppressAssertingFormats({AssertingCodec.Format.STORED_FIELDS, AssertingCodec.Format.NORMS, AssertingCodec.Format.POINTS, AssertingCodec.Format.KNN_VECTORS})
  public static class Mask53 extends ContextBase {}
  @LuceneTestCase.SuppressAssertingFormats({AssertingCodec.Format.TERM_VECTORS, AssertingCodec.Format.NORMS, AssertingCodec.Format.POINTS, AssertingCodec.Format.KNN_VECTORS})
  public static class Mask54 extends ContextBase {}
  @LuceneTestCase.SuppressAssertingFormats({AssertingCodec.Format.STORED_FIELDS, AssertingCodec.Format.TERM_VECTORS, AssertingCodec.Format.NORMS, AssertingCodec.Format.POINTS, AssertingCodec.Format.KNN_VECTORS})
  public static class Mask55 extends ContextBase {}
  @LuceneTestCase.SuppressAssertingFormats({AssertingCodec.Format.LIVE_DOCS, AssertingCodec.Format.POINTS, AssertingCodec.Format.KNN_VECTORS})
  public static class Mask56 extends ContextBase {}
  @LuceneTestCase.SuppressAssertingFormats({AssertingCodec.Format.STORED_FIELDS, AssertingCodec.Format.LIVE_DOCS, AssertingCodec.Format.POINTS, AssertingCodec.Format.KNN_VECTORS})
  public static class Mask57 extends ContextBase {}
  @LuceneTestCase.SuppressAssertingFormats({AssertingCodec.Format.TERM_VECTORS, AssertingCodec.Format.LIVE_DOCS, AssertingCodec.Format.POINTS, AssertingCodec.Format.KNN_VECTORS})
  public static class Mask58 extends ContextBase {}
  @LuceneTestCase.SuppressAssertingFormats({AssertingCodec.Format.STORED_FIELDS, AssertingCodec.Format.TERM_VECTORS, AssertingCodec.Format.LIVE_DOCS, AssertingCodec.Format.POINTS, AssertingCodec.Format.KNN_VECTORS})
  public static class Mask59 extends ContextBase {}
  @LuceneTestCase.SuppressAssertingFormats({AssertingCodec.Format.NORMS, AssertingCodec.Format.LIVE_DOCS, AssertingCodec.Format.POINTS, AssertingCodec.Format.KNN_VECTORS})
  public static class Mask60 extends ContextBase {}
  @LuceneTestCase.SuppressAssertingFormats({AssertingCodec.Format.STORED_FIELDS, AssertingCodec.Format.NORMS, AssertingCodec.Format.LIVE_DOCS, AssertingCodec.Format.POINTS, AssertingCodec.Format.KNN_VECTORS})
  public static class Mask61 extends ContextBase {}
  @LuceneTestCase.SuppressAssertingFormats({AssertingCodec.Format.TERM_VECTORS, AssertingCodec.Format.NORMS, AssertingCodec.Format.LIVE_DOCS, AssertingCodec.Format.POINTS, AssertingCodec.Format.KNN_VECTORS})
  public static class Mask62 extends ContextBase {}
  @LuceneTestCase.SuppressAssertingFormats({AssertingCodec.Format.STORED_FIELDS, AssertingCodec.Format.TERM_VECTORS, AssertingCodec.Format.NORMS, AssertingCodec.Format.LIVE_DOCS, AssertingCodec.Format.POINTS, AssertingCodec.Format.KNN_VECTORS})
  public static class Mask63 extends ContextBase {}
  static void threadChecks() throws Exception {
    var method = AssertingCodec.class.getDeclaredMethod("assertThread", String.class, Thread.class);
    method.setAccessible(true); Thread owner = Thread.currentThread();
    method.invoke(null, "values", owner); row("thread", "same", "ok");
    for (String object : new String[] {"values", "", null}) {
      AtomicReference<Throwable> error = new AtomicReference<>(); AtomicReference<String> expectedMessage = new AtomicReference<>();
      Thread worker = new Thread(() -> {
        expectedMessage.set(object + " are only supposed to be consumed in the thread in which they have been acquired. But was acquired in " + owner + " and consumed in " + Thread.currentThread() + ".");
        try { method.invoke(null, object, owner); }
        catch (InvocationTargetException failure) { error.set(failure.getCause()); }
        catch (Exception failure) { error.set(failure); }
      }, "asserting-codec-probe");
      worker.start(); worker.join(); Throwable actual = error.get();
      String expected = expectedMessage.get();
      check(actual instanceof AssertionError && actual.getMessage().equals(expected), "thread failure message");
      row("thread", "different", String.valueOf(object), "assertion", "message-exact");
    }
    boolean failed = false;
    try { method.invoke(null, "values", null); }
    catch (InvocationTargetException error) { check(error.getCause() instanceof AssertionError, "null owner error"); failed = true; }
    check(failed, "null owner accepted"); row("thread", "null-owner", "assertion");
  }
  public static void main(String[] args) throws Exception {
    check(GenAssertingCodec.class.desiredAssertionStatus(), "run fixture with -ea");
    boolean absent = false;
    try { RandomizedContext.current(); } catch (IllegalStateException expected) { absent = true; }
    check(absent, "no-context case unexpectedly has context");
    snapshot("NoContext", 0, new Random(42), new Random(42));
    var contexts = new ArrayList<Class<?>>();
    for (int mask = 0; mask < 64; mask++) contexts.add(Class.forName("GenAssertingCodec$Mask" + String.format(Locale.ROOT, "%02d", mask)));
    contexts.add(Unannotated.class); contexts.add(DuplicatedReversed.class); contexts.add(Inherited.class);
    for (Class<?> context : contexts) {
      var result = JUnitCore.runClasses(context);
      if (!result.wasSuccessful() || result.getRunCount() != 1) {
        for (var failure : result.getFailures()) System.err.println(failure.getTrace());
        throw new AssertionError("context failed: " + context.getSimpleName());
      }
    }
    threadChecks();
    Files.createDirectories(Path.of(args[0]));
    Files.writeString(Path.of(args[0], "asserting_codec.tsv"), OUT.toString());
  }
}
