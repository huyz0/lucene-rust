import java.io.IOException;
import java.net.URLClassLoader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.HexFormat;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.TreeSet;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.FieldsConsumer;
import org.apache.lucene.codecs.FieldsProducer;
import org.apache.lucene.codecs.FilterCodec;
import org.apache.lucene.codecs.NormsProducer;
import org.apache.lucene.codecs.PostingsFormat;
import org.apache.lucene.codecs.lucene104.Lucene104Codec;
import org.apache.lucene.codecs.lucene104.Lucene104PostingsFormat;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.FieldType;
import org.apache.lucene.document.NumericDocValuesField;
import org.apache.lucene.document.StoredField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.Fields;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LogDocMergePolicy;
import org.apache.lucene.index.MergeState;
import org.apache.lucene.index.PostingsEnum;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentReader;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.SerialMergeScheduler;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.Sort;
import org.apache.lucene.search.SortField;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.analysis.CannedTokenStream;
import org.apache.lucene.tests.analysis.Token;
import org.apache.lucene.util.BytesRef;

/** Real IndexWriter sessions through Java's actual asserting postings consumer. */
public class GenPostingsWriterRouting {
  private static final StringBuilder OUT = new StringBuilder();
  private static final List<Event> EVENTS = new ArrayList<>();
  private static final Map<Integer, String> CONSUMER_SEGMENTS = new HashMap<>();
  private static final Map<Integer, String> CONSUMER_SUFFIXES = new HashMap<>();
  private static final Map<String, Integer> SEGMENT_ORDER = new LinkedHashMap<>();
  private static final HashSet<String> CLOSED_SEGMENTS = new HashSet<>();
  private static final int[] RANKS = {50, 10, 40, 20, 80, 60, 0, 30, 70, 90, 100};
  private static int nextConsumer;
  private static String phase;

  private record Definition(String name, IndexOptions options, boolean omitNorms,
                            String group, int minBlock, int maxBlock) {}
  private static final List<Definition> DEFINITIONS = List.of(
      definition("id", IndexOptions.DOCS, true),
      definition("docs", IndexOptions.DOCS, true),
      definition("freq", IndexOptions.DOCS_AND_FREQS, true),
      definition("pos", IndexOptions.DOCS_AND_FREQS_AND_POSITIONS, false),
      definition("offset", IndexOptions.DOCS_AND_FREQS_AND_POSITIONS_AND_OFFSETS, false),
      definition("empty", IndexOptions.DOCS_AND_FREQS_AND_POSITIONS, false),
      definition("late", IndexOptions.DOCS_AND_FREQS_AND_POSITIONS_AND_OFFSETS, true),
      definition("\uD800\uDC00", IndexOptions.DOCS, true),
      new Definition("\uE000", IndexOptions.DOCS, true, "small", 2, 3),
      new Definition("vanish", IndexOptions.DOCS, true, "tombstone", 4, 7),
      definition("boundary127", IndexOptions.DOCS_AND_FREQS, true),
      definition("boundary128", IndexOptions.DOCS_AND_FREQS, true),
      definition("boundary129", IndexOptions.DOCS_AND_FREQS, true));
  private record Input(String term, int increment, int start, int end, byte[] payload) {}
  private record Occurrence(int position, int start, int end, String payload) {}
  private record Event(String phase, int consumer, String suffix, String operation, String details) {}
  private record Scenario(String name, boolean sorted, boolean empty, int richDocs) {
    int rank(int ordinal) { return richDocs == 9 ? RANKS[ordinal] : ordinal * 71 % 137; }
    String id(int ordinal) { return "d" + ordinal; }
  }

  private static Definition definition(String name, IndexOptions options, boolean omitNorms) {
    return new Definition(name, options, omitNorms, "default", 25, 48);
  }
  private static void row(Object... cells) {
    for (int cell = 0; cell < cells.length; cell++) {
      if (cell != 0) OUT.append('\t');
      OUT.append(cells[cell]);
    }
    OUT.append('\n');
  }
  private static String hex(byte[] bytes) { return bytes == null || bytes.length == 0 ? "-" : HexFormat.of().formatHex(bytes); }
  private static String termHex(String text) { return HexFormat.of().formatHex(text.getBytes(StandardCharsets.UTF_8)); }
  private static String bytesHex(BytesRef bytes) {
    return bytes == null || bytes.length == 0 ? "-"
        : HexFormat.of().formatHex(bytes.bytes, bytes.offset, bytes.offset + bytes.length);
  }
  private static String names(Iterable<String> fields) {
    var result = new ArrayList<String>(); fields.forEach(result::add); return String.join(",", result);
  }
  private static void event(int id, String suffix, String operation, String details) {
    EVENTS.add(new Event(phase, id, suffix, operation, details));
  }
  private static FieldsConsumer asserting(SegmentWriteState state, FieldsConsumer delegate) {
    try {
      // Use the pinned Java wrapper itself, as GenAssertingPostings does.
      var constructor = Class.forName("org.apache.lucene.tests.codecs.asserting.AssertingPostingsFormat$AssertingFieldsConsumer")
          .getDeclaredConstructor(SegmentWriteState.class, FieldsConsumer.class);
      constructor.setAccessible(true);
      return (FieldsConsumer) constructor.newInstance(state, delegate);
    } catch (ReflectiveOperationException error) { throw new IllegalStateException(error); }
  }

  /** Native per-field identity and reader; only its writer enters assertions. */
  private static final class RoutingFormat extends PostingsFormat {
    final PostingsFormat delegate;
    RoutingFormat(int minBlock, int maxBlock) {
      super("Lucene104"); delegate = new Lucene104PostingsFormat(minBlock, maxBlock);
    }
    @Override public FieldsProducer fieldsProducer(SegmentReadState state) throws IOException {
      return delegate.fieldsProducer(state);
    }
    @Override public FieldsConsumer fieldsConsumer(SegmentWriteState state) throws IOException {
      String operationSegment = phase + ":" + state.segmentInfo.name;
      if (CLOSED_SEGMENTS.contains(operationSegment)) throw new AssertionError("group opened after group close");
      int id = nextConsumer++; String suffix = state.segmentSuffix;
      CONSUMER_SEGMENTS.put(id, operationSegment);
      CONSUMER_SUFFIXES.put(id, suffix);
      SEGMENT_ORDER.computeIfAbsent(operationSegment, key -> SEGMENT_ORDER.size());
      var nativeConsumer = delegate.fieldsConsumer(state);
      event(id, suffix, "open", "maxDoc=" + state.segmentInfo.maxDoc());
      var trace = new FieldsConsumer() {
        @Override public void write(Fields fields, NormsProducer norms) throws IOException {
          if (CLOSED_SEGMENTS.contains(operationSegment)) throw new AssertionError("group write after group close");
          event(id, suffix, "native-write", names(fields));
          nativeConsumer.write(fields, norms);
          event(id, suffix, "native-write-done", "-");
        }
        @Override public void close() throws IOException {
          event(id, suffix, "native-close", "-"); nativeConsumer.close();
        }
      };
      var checked = asserting(state, trace);
      return new FieldsConsumer() {
        @Override public void write(Fields fields, NormsProducer norms) throws IOException {
          event(id, suffix, "write", names(fields)); checked.write(fields, norms);
          event(id, suffix, "write-done", "-");
        }
        @Override public void merge(MergeState state, NormsProducer norms) throws IOException {
          event(id, suffix, "merge", "sources=" + state.fieldsProducers.length + ",sorted=" + state.needsIndexSort);
          for (int source = 0; source < state.fieldsProducers.length; source++) {
            int liveCount = 0;
            for (int doc = 0; doc < state.maxDocs[source]; doc++) {
              if (state.liveDocs[source] == null || state.liveDocs[source].get(doc)) liveCount++;
            }
            event(id, suffix, "source", source + ":" + state.maxDocs[source] + ":" + liveCount + ":"
                + (state.fieldsProducers[source] == null ? "null" : names(state.fieldsProducers[source])));
          }
          // Preserve PerFieldMergeState's restricted source objects and maps.
          checked.merge(state, norms); event(id, suffix, "merge-done", "-");
        }
        @Override public void close() throws IOException {
          CLOSED_SEGMENTS.add(CONSUMER_SEGMENTS.get(id));
          event(id, suffix, "close", "-"); checked.close();
        }
      };
    }
  }
  public static final class RoutingCodec extends FilterCodec {
    // Cache identity: equal descriptors share an actual PostingsFormat instance.
    final PostingsFormat defaults = new RoutingFormat(25, 48);
    final PostingsFormat small = new RoutingFormat(2, 3);
    final PostingsFormat tombstone = new RoutingFormat(4, 7);
    final PostingsFormat perField = new org.apache.lucene.codecs.perfield.PerFieldPostingsFormat() {
      @Override public PostingsFormat getPostingsFormatForField(String field) {
        return field.equals("\uE000") ? small : field.equals("vanish") ? tombstone : defaults;
      }
    };
    public RoutingCodec() { super("PostingsRoutingOracle", new Lucene104Codec()); }
    @Override public PostingsFormat postingsFormat() { return perField; }
  }

  private static List<Input> inputs(Scenario scenario, int ordinal, String field) {
    if (scenario.empty) return field.equals("empty") ? List.of() : null;
    if (ordinal >= scenario.richDocs) return null;
    if (field.equals("id")) return List.of(new Input(scenario.id(ordinal), 1, 0, 1, null));
    if (field.equals("empty")) return ordinal % 2 == 0 ? List.of() : null;
    if (field.equals("vanish")) return ordinal == 1 || ordinal == 4 || ordinal == 6
        ? List.of(new Input("gone", 1, 0, 4, null)) : null;
    if (field.equals("late")) return ordinal >= 6
        ? List.of(new Input("later", 1, 0, 5, null)) : null;
    if (field.startsWith("boundary")) {
      if (scenario.richDocs != 132 || ordinal >= Integer.parseInt(field.substring(8)) + 3) return null;
      return List.of(new Input("boundary", 1, 0, 8, null));
    }
    if (field.equals("\uE000") || field.equals("\uD800\uDC00")) {
      return List.of(new Input("\u00E9", 1, 0, 1, null), new Input("z", 1, 2, 3, null));
    }
    if (field.equals("docs")) return List.of(new Input("a", 1, 0, 1, null),
        new Input("a", 1, 2, 3, null), new Input("z", 1, 4, 5, null));
    if (field.equals("freq")) {
      var result = new ArrayList<Input>();
      for (int i = 0; i <= ordinal % 3; i++) result.add(new Input("same", 1, i * 2, i * 2 + 1, null));
      result.add(new Input("u" + ordinal % 5, 1, 8, 10, null)); return result;
    }
    byte[] payload = field.equals("offset") ? new byte[] {(byte) ordinal, (byte) 255} : null;
    return List.of(new Input("same", 1, 0, 2, payload),
        new Input("same", 0, 0, 2, field.equals("offset") ? new byte[0] : null),
        new Input("other", ordinal % 3 + 1, 5, 9, null));
  }
  private static Document document(Scenario scenario, int ordinal) {
    var document = new Document();
    document.add(new StoredField("stored_id", scenario.id(ordinal)));
    document.add(new NumericDocValuesField("rank", scenario.rank(ordinal)));
    for (var definition : DEFINITIONS) {
      var inputs = inputs(scenario, ordinal, definition.name); if (inputs == null) continue;
      var tokens = new ArrayList<Token>(); int finalOffset = 0;
      for (var input : inputs) {
        var token = new Token(input.term, input.increment, input.start, input.end);
        token.setPayload(input.payload == null ? null : new BytesRef(input.payload));
        tokens.add(token); finalOffset = Math.max(finalOffset, input.end);
      }
      var type = new FieldType(); type.setTokenized(true); type.setIndexOptions(definition.options);
      type.setOmitNorms(definition.omitNorms); type.freeze();
      document.add(new Field(definition.name, new CannedTokenStream(0, finalOffset, tokens.toArray(Token[]::new)), type));
    }
    return document;
  }
  private static IndexWriterConfig config(boolean sorted) {
    var config = new IndexWriterConfig(); config.setCodec(new RoutingCodec()); config.setUseCompoundFile(false);
    config.setMaxFullFlushMergeWaitMillis(0);
    var policy = new LogDocMergePolicy(); policy.setMergeFactor(1000);
    config.setMergePolicy(policy); config.setMergeScheduler(new SerialMergeScheduler());
    if (sorted) config.setIndexSort(new Sort(new SortField("rank", SortField.Type.LONG)));
    return config;
  }

  private static TreeMap<String, TreeMap<Integer, List<Occurrence>>> expected(
      Scenario scenario, List<Integer> ordinals, Definition definition) {
    var result = new TreeMap<String, TreeMap<Integer, List<Occurrence>>>();
    for (int doc = 0; doc < ordinals.size(); doc++) {
      var tokens = inputs(scenario, ordinals.get(doc), definition.name); if (tokens == null) continue;
      int position = -1;
      for (var token : tokens) {
        position += token.increment;
        result.computeIfAbsent(termHex(token.term), key -> new TreeMap<>())
            .computeIfAbsent(doc, key -> new ArrayList<>())
            .add(new Occurrence(position, token.start, token.end, hex(token.payload)));
      }
    }
    return result;
  }
  private static void snapshot(Scenario scenario, Directory directory) throws IOException {
    try (var reader = DirectoryReader.open(directory)) {
      row("CASE", scenario.name, phase, reader.maxDoc(), reader.numDocs(), reader.leaves().size());
      for (var leaf : reader.leaves()) {
        var segment = (SegmentReader) leaf.reader(); segment.checkIntegrity();
        var ordinals = new ArrayList<Integer>(); var live = segment.getLiveDocs();
        for (int doc = 0; doc < segment.maxDoc(); doc++) {
          String id = segment.storedFields().document(doc).get("stored_id");
          int ordinal = Integer.parseInt(id.substring(1)); ordinals.add(ordinal);
          row("DOC", scenario.name, phase, leaf.ord, doc, live == null || live.get(doc) ? 1 : 0, id, scenario.rank(ordinal));
        }
        for (var definition : DEFINITIONS) {
          var info = segment.getFieldInfos().fieldInfo(definition.name); var terms = segment.terms(definition.name);
          var expected = expected(scenario, ordinals, definition);
          if ((terms == null) != expected.isEmpty()) throw new AssertionError("field term presence " + definition.name);
          row("FIELD", scenario.name, phase, leaf.ord, definition.name, info == null ? -1 : info.number,
              info == null ? "NONE" : info.getIndexOptions(), info == null || info.omitsNorms() ? 1 : 0,
              info != null && info.hasPayloads() ? 1 : 0,
              info == null || info.getAttribute("PerFieldPostingsFormat.format") == null ? "-" : info.getAttribute("PerFieldPostingsFormat.format"),
              info == null || info.getAttribute("PerFieldPostingsFormat.suffix") == null ? "-" : info.getAttribute("PerFieldPostingsFormat.suffix"),
              terms == null ? 0 : 1, terms == null ? 0 : terms.size(), terms == null ? 0 : terms.getSumDocFreq(),
              terms == null ? 0 : terms.getSumTotalTermFreq(), terms == null ? 0 : terms.getDocCount());
          var norms = segment.getNormValues(definition.name);
          for (int doc = 0; doc < segment.maxDoc(); doc++) {
            row("NORM", scenario.name, phase, leaf.ord, definition.name, doc,
                norms != null && norms.advanceExact(doc) ? Long.toString(norms.longValue()) : "-");
          }
          if (terms == null) continue;
          if (terms.size() != expected.size()) throw new AssertionError("term count");
          long sumDocFreq = 0, sumTotal = 0; var docsSeen = new TreeSet<Integer>();
          var iterator = terms.iterator(); int termCount = 0;
          for (BytesRef term; (term = iterator.next()) != null;) {
            String termHex = bytesHex(term); var postingsExpected = expected.get(termHex);
            if (postingsExpected == null) throw new AssertionError("unexpected term"); termCount++;
            boolean freqs = definition.options != IndexOptions.DOCS;
            long total = freqs ? postingsExpected.values().stream().mapToLong(List::size).sum() : postingsExpected.size();
            if (iterator.docFreq() != postingsExpected.size() || iterator.totalTermFreq() != total) throw new AssertionError("term statistics");
            sumDocFreq += postingsExpected.size(); sumTotal += total;
            row("TERM", scenario.name, phase, leaf.ord, definition.name, termHex, iterator.docFreq(), iterator.totalTermFreq());
            var postings = iterator.postings(null, PostingsEnum.ALL); int count = 0;
            for (int doc; (doc = postings.nextDoc()) != DocIdSetIterator.NO_MORE_DOCS;) {
              var occurrences = postingsExpected.get(doc); if (occurrences == null) throw new AssertionError("posting document");
              count++; docsSeen.add(doc); int frequency = freqs ? occurrences.size() : 1;
              if (postings.freq() != frequency) throw new AssertionError("posting frequency");
              row("POSTING", scenario.name, phase, leaf.ord, definition.name, termHex, doc, frequency);
              if (definition.options.compareTo(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS) < 0) continue;
              for (int occurrence = 0; occurrence < frequency; occurrence++) {
                var wanted = occurrences.get(occurrence); int position = postings.nextPosition();
                int start = postings.startOffset(), end = postings.endOffset(); String payload = bytesHex(postings.getPayload());
                boolean offsets = definition.options == IndexOptions.DOCS_AND_FREQS_AND_POSITIONS_AND_OFFSETS;
                if (position != wanted.position || start != (offsets ? wanted.start : -1)
                    || end != (offsets ? wanted.end : -1) || !payload.equals(wanted.payload)) throw new AssertionError("posting occurrence");
                row("POSITION", scenario.name, phase, leaf.ord, definition.name, termHex, doc, occurrence, position, start, end, payload);
              }
            }
            if (count != postingsExpected.size()) throw new AssertionError("posting count");
          }
          if (termCount != expected.size() || terms.getSumDocFreq() != sumDocFreq
              || terms.getSumTotalTermFreq() != sumTotal || terms.getDocCount() != docsSeen.size()) throw new AssertionError("field statistics");
        }
        for (String file : new TreeSet<>(segment.getSegmentInfo().info.files())) {
          String prefix = segment.getSegmentName() + "_";
          if (!file.startsWith(prefix)) continue; String rest = file.substring(prefix.length());
          int dot = rest.lastIndexOf('.'); if (dot < 0) continue;
          String suffix = rest.substring(0, dot), extension = rest.substring(dot + 1);
          if (suffix.startsWith("Lucene104_") && List.of("psm", "doc", "pos", "pay", "tim", "tip", "tmd").contains(extension)) {
            row("GROUP_FILE", scenario.name, phase, leaf.ord, suffix, extension, directory.fileLength(file));
          }
        }
      }
    }
    row("CHECKPOINT", scenario.name, phase, nextConsumer, EVENTS.size());
  }
  private static void flush(Scenario scenario, IndexWriter writer, int ordinal) throws IOException {
    writer.commit(); row("FLUSH", scenario.name, phase, ordinal);
  }
  private static void scenario(Scenario scenario) throws Exception {
    EVENTS.clear(); CONSUMER_SEGMENTS.clear(); CONSUMER_SUFFIXES.clear(); SEGMENT_ORDER.clear();
    CLOSED_SEGMENTS.clear(); nextConsumer = 0; phase = "initial";
    for (var definition : DEFINITIONS) row("FIELD_DEF", scenario.name, definition.name, definition.options,
        definition.omitNorms ? 1 : 0, definition.group, definition.minBlock, definition.maxBlock);
    for (int ordinal = 0; ordinal < scenario.richDocs + 2; ordinal++) {
      row("INPUT_DOC", scenario.name, ordinal, scenario.id(ordinal), scenario.rank(ordinal));
      for (var definition : DEFINITIONS) {
        var inputs = inputs(scenario, ordinal, definition.name); if (inputs == null) continue;
        if (inputs.isEmpty()) row("INPUT_EMPTY", scenario.name, ordinal, definition.name);
        for (var input : inputs) row("INPUT_TOKEN", scenario.name, ordinal, definition.name, termHex(input.term),
            input.increment, input.start, input.end, hex(input.payload));
      }
    }
    int firstEnd = scenario.richDocs == 9 ? 5 : 65;
    try (var directory = new ByteBuffersDirectory()) {
      try (var writer = new IndexWriter(directory, config(scenario.sorted))) {
        for (int ordinal = 0; ordinal <= firstEnd; ordinal++) {
          writer.addDocument(document(scenario, ordinal));
          if (scenario.richDocs == 9 && ordinal == 2 || ordinal == firstEnd) flush(scenario, writer, ordinal);
        }
        snapshot(scenario, directory);
      }
      try (var writer = new IndexWriter(directory, config(scenario.sorted))) {
        phase = "append";
        for (int ordinal = firstEnd + 1; ordinal < scenario.richDocs; ordinal++) writer.addDocument(document(scenario, ordinal));
        flush(scenario, writer, scenario.richDocs - 1);
        writer.addDocument(document(scenario, scenario.richDocs)); flush(scenario, writer, scenario.richDocs);
        snapshot(scenario, directory);
        if (!scenario.empty) {
          phase = "deleted";
          for (int ordinal : new int[] {1, 4, 6}) {
            writer.deleteDocuments(new Term("id", scenario.id(ordinal))); row("DELETE", scenario.name, phase, scenario.id(ordinal));
          }
          writer.commit(); snapshot(scenario, directory);
        }
        phase = "merged"; writer.forceMerge(1); writer.commit(); snapshot(scenario, directory);
        phase = "reappend"; writer.addDocument(document(scenario, scenario.richDocs + 1));
        flush(scenario, writer, scenario.richDocs + 1); snapshot(scenario, directory);
        phase = "remerged"; writer.forceMerge(1); writer.commit(); snapshot(scenario, directory);
      }
    }
    if (scenario.empty && !EVENTS.isEmpty()) throw new AssertionError("no per-group consumers without terms");
    // HashMap iteration chooses group execution order. Canonicalize whole
    // consumer traces, preserving each consumer's observed partial order.
    var consumers = new ArrayList<>(CONSUMER_SEGMENTS.keySet());
    consumers.sort((left, right) -> {
      int batch = Integer.compare(SEGMENT_ORDER.get(CONSUMER_SEGMENTS.get(left)), SEGMENT_ORDER.get(CONSUMER_SEGMENTS.get(right)));
      return batch != 0 ? batch : CONSUMER_SUFFIXES.get(left).compareTo(CONSUMER_SUFFIXES.get(right));
    });
    int sequence = 0;
    for (int consumer = 0; consumer < consumers.size(); consumer++) {
      int original = consumers.get(consumer);
      for (var event : EVENTS) if (event.consumer == original) {
        row("EVENT", scenario.name, event.phase, sequence++, consumer, event.suffix, event.operation, event.details);
      }
    }
  }
  public static void main(String[] args) throws Exception {
    if (!GenPostingsWriterRouting.class.desiredAssertionStatus()) throw new AssertionError("run with -ea");
    Path output = Path.of(args[0]); Files.createDirectories(output);
    Path providers = Files.createTempDirectory("postings-routing-provider");
    Path service = providers.resolve("META-INF/services/org.apache.lucene.codecs.Codec");
    Files.createDirectories(service.getParent()); Files.writeString(service, RoutingCodec.class.getName() + "\n");
    try (var loader = new URLClassLoader(new java.net.URL[] {providers.toUri().toURL()}, GenPostingsWriterRouting.class.getClassLoader())) {
      Codec.reloadCodecs(loader);
      row("# Lucene 10.5.0 actual reflected AssertingFieldsConsumer; native Lucene104 group identity/readers; no copied assertions");
      row("# FIELD_DEF case field indexOptions omitNorms groupLabel minBlock maxBlock; cached descriptors, UTF-16 field order");
      row("# INPUT_DOC case ordinal storedId rank; INPUT_TOKEN case ordinal field termHex positionIncrement startOffset endOffset payloadHex(-=none/empty); INPUT_EMPTY case ordinal field");
      row("# FLUSH case phase afterOrdinal; DELETE case phase storedId; CASE case phase maxDoc numDocs leaves; DOC case phase leaf localDoc live storedId rank");
      row("# FIELD case phase leaf field number indexOptions omitNorms payloads format suffix termsPresent termCount sumDocFreq sumTotalTermFreq docCount");
      row("# NORM case phase leaf field localDoc value(-=absent); TERM case phase leaf field termHex docFreq totalTermFreq");
      row("# POSTING case phase leaf field termHex localDoc freq; POSITION case phase leaf field termHex localDoc occurrence position startOffset endOffset payloadHex");
      row("# GROUP_FILE case phase leaf codecSuffix extension encodedLength(provenance); CHECKPOINT case phase consumersOpened eventCount");
      row("# EVENT case phase sequence consumerId codecSuffix operation details; canonicalized whole consumer traces preserve partial order; generator requires all group writes before outer closes");
      for (boolean empty : new boolean[] {false, true}) for (boolean sorted : new boolean[] {false, true}) {
        scenario(new Scenario((empty ? "empty" : "rich") + (sorted ? "_sorted" : "_unsorted"), sorted, empty, 9));
      }
      scenario(new Scenario("block_boundary", false, false, 132));
      Files.writeString(output.resolve("postings_writer_routing.tsv"), OUT);
    } finally {
      Files.deleteIfExists(service); Files.deleteIfExists(service.getParent());
      Files.deleteIfExists(service.getParent().getParent()); Files.deleteIfExists(providers);
    }
  }
}
