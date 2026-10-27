import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Comparator;
import java.util.HashSet;
import java.util.HexFormat;
import java.util.List;
import java.util.TreeMap;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.tokenattributes.BytesTermAttribute;
import org.apache.lucene.analysis.tokenattributes.OffsetAttribute;
import org.apache.lucene.analysis.tokenattributes.PayloadAttribute;
import org.apache.lucene.analysis.tokenattributes.PositionIncrementAttribute;
import org.apache.lucene.codecs.FilterCodec;
import org.apache.lucene.codecs.TermVectorsFormat;
import org.apache.lucene.codecs.TermVectorsReader;
import org.apache.lucene.codecs.TermVectorsWriter;
import org.apache.lucene.codecs.lucene104.Lucene104Codec;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.FieldType;
import org.apache.lucene.document.NumericDocValuesField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LogDocMergePolicy;
import org.apache.lucene.index.MergeState;
import org.apache.lucene.index.PostingsEnum;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.index.SegmentReader;
import org.apache.lucene.index.SerialMergeScheduler;
import org.apache.lucene.search.Sort;
import org.apache.lucene.search.SortField;
import org.apache.lucene.store.DataInput;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.tests.codecs.asserting.AssertingTermVectorsFormat;
import org.apache.lucene.util.Accountable;
import org.apache.lucene.util.BytesRef;

/** Shared exact corpus and independent native-reader oracle for routing and timings. */
public final class TermVectorsRoutingSupport {
  private TermVectorsRoutingSupport() {}

  public record Definition(String name, boolean positions, boolean offsets, boolean payloads) {}
  // Deliberately not field-name order. Java orders vector fields by UTF-16 names,
  // so the supplementary name sorts before U+E000, opposite their UTF-8 ordering.
  public static final List<Definition> DEFINITIONS = List.of(
      new Definition("z_full", true, true, true),
      new Definition("\uE000", false, false, false),
      new Definition("\uD800\uDC00", true, false, false),
      new Definition("a_offsets", false, true, false),
      new Definition("b_payloads", true, false, true),
      new Definition("m_sparse", true, true, false),
      new Definition("x_empty", true, true, true),
      new Definition("q_plain", false, false, false));
  /** Axes representable by the configured Rust facade, all with positions. */
  public static Definition definition(int field, Profile profile) {
    var definition = DEFINITIONS.get(field);
    return profile.kind.startsWith("configured")
        ? new Definition(definition.name, true, definition.offsets, definition.payloads)
        : definition;
  }
  public record Input(byte[] term, int increment, int start, int end, byte[] payload) {}
  public record Profile(String kind, int termBytes) {
    public static Profile rich() { return new Profile("rich", 0); }
    public static Profile tiny() { return new Profile("tiny", 0); }
    public static Profile bytes(int length) { return new Profile("bytes", length); }
  }
  @FunctionalInterface public interface Rows { void row(Object... cells); }
  @FunctionalInterface public interface Events {
    void event(String layer, String operation, String details) throws IOException;
  }

  public static void require(boolean condition, String message) {
    if (!condition) throw new AssertionError(message);
  }
  public static String hex(byte[] bytes) {
    return bytes.length == 0 ? "-" : HexFormat.of().formatHex(bytes);
  }
  public static String hex(BytesRef bytes) {
    return bytes == null || bytes.length == 0 ? "-"
        : HexFormat.of().formatHex(bytes.bytes, bytes.offset, bytes.offset + bytes.length);
  }
  public static String hex(String text) { return hex(text.getBytes(StandardCharsets.UTF_8)); }
  public static long rank(int doc, int docs, int sources) {
    int perSource = docs / sources;
    require(perSource > 0 && docs % sources == 0, "rank source geometry");
    return (long) (doc % perSource) * sources + doc / perSource;
  }
  public static String inputKind(int doc, Profile profile) {
    if (profile.kind.startsWith("configured")) {
      if (profile.kind.equals("configured-missing") || doc % 7 == 0) return "missing";
      if (profile.kind.equals("configured-empty") || doc % 7 == 1) return "empty";
      return "configured";
    }
    if (!profile.kind.equals("rich")) return profile.kind;
    return doc % 17 == 0 ? "missing" : doc % 17 == 1 ? "empty" : "rich";
  }
  public static List<Input> inputs(int doc, int field, Profile profile) {
    if (profile.kind.startsWith("configured")) {
      String kind = inputKind(doc, profile);
      if (kind.equals("missing")) return null;
      if (kind.equals("empty") || field == 6) return field == 6 ? List.of() : null;
      if (field == 5 && doc % 3 == 0) return null;
      var values = new ArrayList<Input>();
      for (int ordinal = 0; ordinal < 3; ordinal++) {
        String term = ordinal == 1 ? "\u03b2" : "a";
        byte[] payload = !definition(field, profile).payloads || doc % 3 == 0 ? null
            : doc % 3 == 1 ? new byte[0]
            : ordinal == 0 ? null : ordinal == 1 ? new byte[0] : new byte[] {0, (byte) doc, (byte) 255};
        values.add(new Input(term.getBytes(StandardCharsets.UTF_8), 1,
            ordinal * 2, ordinal * 2 + 1, payload));
      }
      return values;
    }
    if (profile.kind.equals("missing")) return null;
    if (profile.kind.equals("empty")) return field == 6 ? List.of() : null;
    if (profile.kind.equals("tiny")) {
      return field == 7 ? List.of(new Input(new byte[] {120}, 1, 0, 1, null)) : null;
    }
    if (profile.kind.equals("bytes")) {
      byte[] term = new byte[profile.termBytes];
      Arrays.fill(term, (byte) 97);
      return field == 7 ? List.of(new Input(term, 1, 0, 1, null)) : null;
    }
    if (doc % 17 == 0) return null;
    if (doc % 17 == 1) return field == 6 ? List.of() : null;
    if (field == 6) return List.of();
    if (field == 5 && doc % 3 == 0) return null;
    // Raw 0x00/0x80/0xff terms, shared prefixes, a zero position increment,
    // repeated terms, empty payloads and variable-length nonempty payloads.
    var values = new ArrayList<Input>();
    values.add(new Input(new byte[] {0}, 1, 0, 2, null));
    values.add(new Input(new byte[] {(byte) 255, 0, (byte) (doc % 13)}, 2, 4, 7, null));
    values.add(new Input(new byte[] {0}, 0, 4, 7, null));
    values.add(new Input(new byte[] {97, 0, (byte) 128}, 1, 10, 14, null));
    values.add(new Input(new byte[] {97, 0, (byte) 255}, 1, 16, 21, null));
    for (int i = 0; i < doc % 3; i++)
      values.add(new Input(new byte[] {0}, 1, 24 + i * 4, 27 + i * 4, null));
    if (DEFINITIONS.get(field).payloads) {
      for (int i = 0; i < values.size(); i++) {
        Input token = values.get(i);
        byte[] payload = (doc + i) % 3 == 0 ? null
            : (doc + i) % 3 == 1 ? new byte[0]
            : new byte[] {(byte) doc, (byte) i, 0, (byte) 255};
        values.set(i, new Input(token.term, token.increment, token.start, token.end, payload));
      }
    }
    return values;
  }

  private static final class Tokens extends TokenStream {
    final List<Input> inputs;
    int cursor;
    final BytesTermAttribute term = addAttribute(BytesTermAttribute.class);
    final PositionIncrementAttribute increment = addAttribute(PositionIncrementAttribute.class);
    final OffsetAttribute offset = addAttribute(OffsetAttribute.class);
    final PayloadAttribute payload = addAttribute(PayloadAttribute.class);
    Tokens(List<Input> inputs) { this.inputs = inputs; }
    @Override public void reset() throws IOException { super.reset(); cursor = 0; }
    @Override public boolean incrementToken() {
      if (cursor == inputs.size()) return false;
      clearAttributes();
      Input input = inputs.get(cursor++);
      term.setBytesRef(new BytesRef(input.term));
      increment.setPositionIncrement(input.increment);
      offset.setOffset(input.start, input.end);
      payload.setPayload(input.payload == null ? null : new BytesRef(input.payload));
      return true;
    }
    @Override public void end() throws IOException {
      super.end();
      increment.setPositionIncrement(0);
      int end = inputs.isEmpty() ? 0 : inputs.get(inputs.size() - 1).end;
      offset.setOffset(end, end);
    }
  }

  public static Document document(int doc, int docs, int sources, int fieldOrder, Profile profile) {
    var result = new Document();
    result.add(new StringField("id", Integer.toString(doc), Field.Store.YES));
    result.add(new NumericDocValuesField("rank", rank(doc, docs, sources)));
    for (int i = 0; i < DEFINITIONS.size(); i++) {
      int field = switch (fieldOrder % 3) {
        case 1 -> DEFINITIONS.size() - 1 - i;
        case 2 -> (i + 3) % DEFINITIONS.size();
        default -> i;
      };
      var definition = definition(field, profile);
      var tokens = inputs(doc, field, profile);
      if (tokens == null) continue;
      var type = new FieldType();
      type.setTokenized(true);
      type.setIndexOptions(profile.kind.startsWith("configured") && !definition.offsets
          ? IndexOptions.DOCS_AND_FREQS_AND_POSITIONS
          : IndexOptions.DOCS_AND_FREQS_AND_POSITIONS_AND_OFFSETS);
      type.setOmitNorms(true);
      type.setStoreTermVectors(true);
      type.setStoreTermVectorPositions(definition.positions);
      type.setStoreTermVectorOffsets(definition.offsets);
      type.setStoreTermVectorPayloads(definition.payloads);
      type.freeze();
      result.add(new Field(definition.name, new Tokens(tokens), type));
    }
    return result;
  }

  /** Same codec identity and native readers; only the lower writer is wrapped. */
  public static final class RoutingCodec extends FilterCodec {
    final TermVectorsFormat format;
    public RoutingCodec(boolean asserting, Events events) {
      super("Lucene104", new Lucene104Codec());
      final TermVectorsFormat nativeFormat = delegate.termVectorsFormat();
      final TermVectorsFormat traced = new TermVectorsFormat() {
        @Override public TermVectorsReader vectorsReader(
            Directory directory, SegmentInfo info, FieldInfos fields, IOContext context)
            throws IOException {
          return nativeFormat.vectorsReader(directory, info, fields, context);
        }
        @Override public TermVectorsWriter vectorsWriter(
            Directory directory, SegmentInfo info, IOContext context) throws IOException {
          if (events != null) events.event("format", "acquire", info.name);
          TermVectorsWriter writer = nativeFormat.vectorsWriter(directory, info, context);
          if (events != null) writer = new TraceWriter(writer, events, "native");
          return writer;
        }
      };
      final TermVectorsFormat selected;
      if (asserting) {
        var checked = new AssertingTermVectorsFormat();
        try {
          var member = AssertingTermVectorsFormat.class.getDeclaredField("in");
          member.setAccessible(true);
          member.set(checked, traced);
        } catch (ReflectiveOperationException error) {
          throw new IllegalStateException("cannot select native asserting delegate", error);
        }
        selected = checked;
      } else selected = traced;
      format = new TermVectorsFormat() {
        @Override public TermVectorsReader vectorsReader(
            Directory directory, SegmentInfo info, FieldInfos fields, IOContext context)
            throws IOException {
          return nativeFormat.vectorsReader(directory, info, fields, context);
        }
        @Override public TermVectorsWriter vectorsWriter(
            Directory directory, SegmentInfo info, IOContext context) throws IOException {
          TermVectorsWriter writer = selected.vectorsWriter(directory, info, context);
          return events == null ? writer : new TraceWriter(writer, events, "writer");
        }
      };
    }
    @Override public TermVectorsFormat termVectorsFormat() { return format; }
  }

  /** Trace both the selected writer API and the actual native delegate API. */
  private static final class TraceWriter extends TermVectorsWriter {
    final TermVectorsWriter delegate;
    final Events events;
    final String layer;
    TraceWriter(TermVectorsWriter delegate, Events events, String layer) {
      this.delegate = delegate; this.events = events; this.layer = layer;
    }
    void event(String op, String details) throws IOException { events.event(layer, op, details); }
    @Override public void startDocument(int fields) throws IOException {
      event("start-document", Integer.toString(fields)); delegate.startDocument(fields);
    }
    @Override public void finishDocument() throws IOException {
      event("finish-document", "-"); delegate.finishDocument();
    }
    @Override public void startField(FieldInfo field, int terms, boolean positions,
        boolean offsets, boolean payloads) throws IOException {
      event("start-field", hex(field.name) + ":" + field.number + ":" + terms
          + ":" + positions + ":" + offsets + ":" + payloads);
      delegate.startField(field, terms, positions, offsets, payloads);
    }
    @Override public void finishField() throws IOException {
      event("finish-field", "-"); delegate.finishField();
    }
    @Override public void startTerm(BytesRef term, int freq) throws IOException {
      event("start-term", hex(term) + ":" + freq); delegate.startTerm(term, freq);
    }
    @Override public void finishTerm() throws IOException { event("finish-term", "-"); delegate.finishTerm(); }
    @Override public void addPosition(int pos, int start, int end, BytesRef payload) throws IOException {
      event("position", pos + ":" + start + ":" + end + ":" + hex(payload));
      delegate.addPosition(pos, start, end, payload);
    }
    @Override public void addProx(int count, DataInput positions, DataInput offsets) throws IOException {
      event("prox", count + ":" + (positions != null) + ":" + (offsets != null));
      delegate.addProx(count, positions, offsets);
    }
    @Override public int merge(MergeState state) throws IOException {
      event("merge", state.termVectorsReaders.length + ":" + state.needsIndexSort);
      return delegate.merge(state);
    }
    @Override public void finish(int docs) throws IOException {
      event("finish", Integer.toString(docs)); delegate.finish(docs);
    }
    @Override public void close() throws IOException { event("close", "-"); delegate.close(); }
    @Override public long ramBytesUsed() { return delegate.ramBytesUsed(); }
    @Override public Collection<Accountable> getChildResources() { return delegate.getChildResources(); }
  }

  public static IndexWriterConfig config(boolean asserting, boolean sorted, int docs, Events events) {
    var config = new IndexWriterConfig();
    config.setCodec(new RoutingCodec(asserting, events));
    config.setUseCompoundFile(false);
    config.setMaxBufferedDocs(Math.max(2, docs * 2));
    config.setRAMBufferSizeMB(IndexWriterConfig.DISABLE_AUTO_FLUSH);
    config.setMaxFullFlushMergeWaitMillis(0);
    var policy = new LogDocMergePolicy();
    policy.setMergeFactor(1000);
    config.setMergePolicy(policy);
    config.setMergeScheduler(new SerialMergeScheduler());
    if (sorted) config.setIndexSort(new Sort(new SortField("rank", SortField.Type.LONG)));
    return config;
  }

  public static List<Integer> ids(int docs, int sources, boolean deleted, boolean sorted) {
    var ids = new ArrayList<Integer>();
    for (int doc = 0; doc < docs; doc++) if (!deleted || doc % 7 != 0) ids.add(doc);
    if (sorted) ids.sort(Comparator.comparingLong(doc -> rank(doc, docs, sources)));
    return ids;
  }
  public static long mix(long digest, long value) { return (digest ^ value) * 0x100000001b3L; }
  public static long bytes(long digest, byte[] value) {
    digest = mix(digest, value.length);
    for (byte b : value) digest = mix(digest, Byte.toUnsignedInt(b));
    return digest;
  }
  private record Occurrence(int position, Input input) {}
  public record Check(long digest, int docs, int fields, int terms, int occurrences,
                      int payloadBytes, long encodedBytes) {}

  /**
   * Verify every live document through a native Lucene reader against the input
   * corpus, not against another serializer. Optional rows form the fixture TSV.
   * Vectorless missing and explicit-empty inputs both read as null; their INPUT
   * rows retain that distinct cause. Numeric source field IDs are not digest input.
   */
  public static Check verify(Directory directory, List<Integer> expectedIds, int docs,
      int sources, Profile profile, boolean singleLeaf, Rows rows) throws IOException {
    try (var reader = DirectoryReader.open(directory)) {
      require(reader.numDocs() == expectedIds.size(), "live document count");
      if (singleLeaf) require(reader.leaves().size() == 1 && reader.maxDoc() == expectedIds.size(),
                              "merged or flushed document geometry");
      var expectedSet = new HashSet<>(expectedIds);
      var seen = new HashSet<Integer>();
      long digest = 0xcbf29ce484222325L;
      int fieldCount = 0, termCount = 0, occurrences = 0, payloadBytes = 0, ordinal = 0;
      for (var leaf : reader.leaves()) {
        var segment = leaf.reader();
        var stored = segment.storedFields();
        var vectors = segment.termVectors();
        var live = segment.getLiveDocs();
        var norms = segment.getNumericDocValues("rank");
        if (rows != null) for (var field : segment.getFieldInfos())
          rows.row("FIELD", leaf.ord, hex(field.name), field.number, field.hasTermVectors());
        for (int physical = 0; physical < segment.maxDoc(); physical++) {
          if (live != null && !live.get(physical)) continue;
          int id = Integer.parseInt(stored.document(physical).get("id"));
          require(expectedSet.contains(id) && seen.add(id), "unexpected/duplicate stored id");
          if (singleLeaf) require(id == expectedIds.get(ordinal), "document ordering");
          require(norms.advanceExact(physical) && norms.longValue() == rank(id, docs, sources), "rank");
          digest = mix(digest, id);
          digest = mix(digest, rank(id, docs, sources));
          var expectedFields = new TreeMap<String, Integer>();
          for (int field = 0; field < DEFINITIONS.size(); field++) {
            var tokens = inputs(id, field, profile);
            if (tokens != null && !tokens.isEmpty()) expectedFields.put(DEFINITIONS.get(field).name, field);
          }
          var actualFields = vectors.get(physical);
          if (rows != null) rows.row("DOC", ordinal, leaf.docBase + physical, id,
                                    inputKind(id, profile), expectedFields.size(),
                                    actualFields == null ? "null" : "fields");
          digest = mix(digest, expectedFields.size());
          if (expectedFields.isEmpty()) {
            require(actualFields == null, "missing/empty vector document must read null");
            ordinal++;
            continue;
          }
          require(actualFields != null && actualFields.size() == expectedFields.size(), "vector field count");
          var names = new ArrayList<String>();
          actualFields.forEach(names::add);
          require(names.equals(new ArrayList<>(expectedFields.keySet())), "UTF-16 field order");
          for (var fieldEntry : expectedFields.entrySet()) {
            String name = fieldEntry.getKey();
            int field = fieldEntry.getValue();
            var definition = definition(field, profile);
            var terms = actualFields.terms(name);
            var expectedTerms = new TreeMap<BytesRef, List<Occurrence>>();
            int position = -1;
            boolean hasPayloads = false;
            for (var token : inputs(id, field, profile)) {
              position += token.increment;
              expectedTerms.computeIfAbsent(new BytesRef(token.term), key -> new ArrayList<>())
                  .add(new Occurrence(position, token));
              hasPayloads |= token.payload != null && token.payload.length != 0;
            }
            hasPayloads &= definition.payloads;
            require(terms.hasFreqs() && terms.hasPositions() == definition.positions
                && terms.hasOffsets() == definition.offsets && terms.hasPayloads() == hasPayloads,
                "vector flags for " + name);
            require(terms.size() == expectedTerms.size(), "vector term count");
            if (rows != null) rows.row("VECTOR", ordinal, hex(name), terms.size(),
                definition.positions, definition.offsets, hasPayloads);
            digest = bytes(digest, name.getBytes(StandardCharsets.UTF_8));
            digest = mix(digest, definition.positions ? 1 : 0);
            digest = mix(digest, definition.offsets ? 1 : 0);
            digest = mix(digest, hasPayloads ? 1 : 0);
            digest = mix(digest, expectedTerms.size());
            fieldCount++;
            var iterator = terms.iterator();
            for (var entry : expectedTerms.entrySet()) {
              var term = iterator.next();
              require(term != null && term.equals(entry.getKey()), "unsigned binary term order");
              int freq = entry.getValue().size();
              require(iterator.docFreq() == 1 && iterator.totalTermFreq() == freq, "vector term stats");
              var posting = iterator.postings(null, PostingsEnum.ALL);
              require(posting.nextDoc() == 0 && posting.freq() == freq, "vector posting");
              digest = bytes(digest, Arrays.copyOfRange(term.bytes, term.offset, term.offset + term.length));
              digest = mix(digest, freq);
              termCount++;
              occurrences += freq;
              if (rows != null) rows.row("TERM", ordinal, hex(name), hex(term), freq);
              if (definition.positions || definition.offsets) for (var occurrence : entry.getValue()) {
                var token = occurrence.input;
                int p = definition.positions ? occurrence.position : -1;
                int start = definition.offsets ? token.start : -1;
                int end = definition.offsets ? token.end : -1;
                require(posting.nextPosition() == p && posting.startOffset() == start
                    && posting.endOffset() == end, "vector occurrence");
                byte[] expectedPayload = definition.payloads && token.payload != null
                    ? token.payload : new byte[0];
                var payload = posting.getPayload();
                byte[] actualPayload = payload == null ? new byte[0]
                    : Arrays.copyOfRange(payload.bytes, payload.offset, payload.offset + payload.length);
                require(Arrays.equals(actualPayload, expectedPayload), "vector payload");
                digest = mix(digest, p); digest = mix(digest, start); digest = mix(digest, end);
                digest = bytes(digest, actualPayload);
                payloadBytes += actualPayload.length;
                if (rows != null) rows.row("POSITION", ordinal, hex(name), hex(term),
                                          p, start, end, hex(actualPayload));
              }
              require(posting.nextDoc() == PostingsEnum.NO_MORE_DOCS, "vector posting end");
            }
            require(iterator.next() == null, "vector term end");
          }
          ordinal++;
        }
        segment.checkIntegrity();
      }
      require(seen.equals(expectedSet), "live id set");
      long encoded = 0;
      for (String file : directory.listAll())
        if (file.endsWith(".tvd") || file.endsWith(".tvx") || file.endsWith(".tvm"))
          encoded += directory.fileLength(file);
      return new Check(digest, ordinal, fieldCount, termCount, occurrences, payloadBytes, encoded);
    }
  }

  /** Native metadata counters prove which 128-doc / 4096-byte boundary fired. */
  public static void chunks(Directory directory, Rows rows) throws IOException {
    try (var reader = DirectoryReader.open(directory)) {
      for (var leaf : reader.leaves()) {
        var vectors = ((SegmentReader) leaf.reader()).getTermVectorsReader();
        if (vectors == null) continue;
        try {
          var type = vectors.getClass();
          long[] counters = new long[3];
          String[] methods = {"getNumChunks", "getNumDirtyChunks", "getNumDirtyDocs"};
          for (int i = 0; i < methods.length; i++) {
            var method = type.getDeclaredMethod(methods[i]);
            method.setAccessible(true);
            counters[i] = ((Number) method.invoke(vectors)).longValue();
          }
          rows.row("CHUNKS", leaf.ord, counters[0], counters[1], counters[2]);
        } catch (ReflectiveOperationException error) {
          throw new IOException("native chunk metadata access failed", error);
        }
      }
    }
  }
}
