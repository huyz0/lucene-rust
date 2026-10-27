import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.Term;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;

/** Real default-binary IndexWriter sessions with optional Java asserting writers.
 * No SimpleText format, synthetic codec bytes or custom reader wrapper is used.
 */
public final class GenTermVectorsWriterRouting {
  private static final StringBuilder OUT = new StringBuilder();
  private static final List<String> EVENTS = new ArrayList<>();
  private static String phase;

  private static void row(Object... cells) {
    for (int i = 0; i < cells.length; i++) {
      if (i != 0) OUT.append('\t');
      OUT.append(cells[i]);
    }
    OUT.append('\n');
  }
  private static TermVectorsRoutingSupport.Events events() {
    return (layer, operation, details) -> EVENTS.add(phase + "\t" + layer + "\t" + operation + "\t" + details);
  }
  private static TermVectorsRoutingSupport.Rows prefixed(String name, String snapshot) {
    return cells -> {
      Object[] record = new Object[cells.length + 2];
      record[0] = cells[0]; record[1] = name; record[2] = snapshot;
      System.arraycopy(cells, 1, record, 3, cells.length - 1);
      row(record);
    };
  }
  private static void snapshot(Directory directory, String name, String snapshot,
      int docs, int sources, TermVectorsRoutingSupport.Profile profile,
      boolean deleted, boolean sorted, boolean singleLeaf) throws IOException {
    var rows = prefixed(name, snapshot);
    var check = TermVectorsRoutingSupport.verify(directory,
        TermVectorsRoutingSupport.ids(docs, sources, deleted, sorted && singleLeaf),
        docs, sources, profile, singleLeaf, rows);
    row("CHECK", name, snapshot, Long.toUnsignedString(check.digest()), check.docs(),
        check.fields(), check.terms(), check.occurrences(), check.payloadBytes(), check.encodedBytes());
    TermVectorsRoutingSupport.chunks(directory, rows);
  }
  private static void inputs(String name, int docs, int sources,
      TermVectorsRoutingSupport.Profile profile) {
    for (int doc = 0; doc < docs; doc++) {
      row("INPUT", name, doc, TermVectorsRoutingSupport.rank(doc, docs, sources),
          TermVectorsRoutingSupport.inputKind(doc, profile));
      for (int field = 0; field < TermVectorsRoutingSupport.DEFINITIONS.size(); field++) {
        var definition = TermVectorsRoutingSupport.DEFINITIONS.get(field);
        var tokens = TermVectorsRoutingSupport.inputs(doc, field, profile);
        if (tokens == null) continue;
        row("INPUT_FIELD", name, doc, TermVectorsRoutingSupport.hex(definition.name()),
            tokens.isEmpty() ? "empty" : "tokens", tokens.size());
        for (int ordinal = 0; ordinal < tokens.size(); ordinal++) {
          var token = tokens.get(ordinal);
          row("INPUT_TOKEN", name, doc, TermVectorsRoutingSupport.hex(definition.name()), ordinal,
              TermVectorsRoutingSupport.hex(token.term()), token.increment(),
              token.start(), token.end(),
              token.payload() == null ? "null" : TermVectorsRoutingSupport.hex(token.payload()));
        }
      }
    }
  }
  private static void traces(String name) {
    for (int event = 0; event < EVENTS.size(); event++) row("TRACE", name, event, EVENTS.get(event));
  }

  private static void mergeScenario(boolean asserting, boolean imported, boolean sorted)
      throws Exception {
    String name = (asserting ? "asserting" : "native") + "_"
        + (imported ? "imported" : "direct") + (sorted ? "_sorted" : "_unsorted");
    int docs = 12, sources = 3;
    var profile = TermVectorsRoutingSupport.Profile.rich();
    inputs(name, docs, sources, profile);
    EVENTS.clear();
    try (var directory = new ByteBuffersDirectory();
         var writer = new IndexWriter(directory,
             TermVectorsRoutingSupport.config(asserting, sorted, docs, events()))) {
      phase = "initial";
      if (imported) {
        for (int source = 0; source < sources; source++) {
          try (var input = new ByteBuffersDirectory()) {
            try (var sourceWriter = new IndexWriter(input,
                TermVectorsRoutingSupport.config(asserting, sorted, docs, events()))) {
              for (int doc = source * 4; doc < (source + 1) * 4; doc++)
                sourceWriter.addDocument(TermVectorsRoutingSupport.document(doc, docs, sources, source, profile));
              sourceWriter.commit();
            }
            writer.addIndexes(input);
          }
        }
      } else {
        for (int doc = 0; doc < docs; doc++) {
          writer.addDocument(TermVectorsRoutingSupport.document(doc, docs, sources, 0, profile));
          if ((doc + 1) % 4 == 0) writer.flush();
        }
      }
      writer.commit();
      snapshot(directory, name, "initial", docs, sources, profile, false, sorted, false);
      phase = "delete";
      for (int doc = 0; doc < docs; doc++) if (doc % 7 == 0)
        writer.deleteDocuments(new Term("id", Integer.toString(doc)));
      writer.commit();
      snapshot(directory, name, "before", docs, sources, profile, true, sorted, false);
      phase = "merge";
      writer.forceMerge(1);
      writer.commit();
      snapshot(directory, name, "after", docs, sources, profile, true, sorted, true);
    }
    traces(name);
  }

  private static void flushScenario(boolean asserting, boolean explicit, String label,
      int docs, TermVectorsRoutingSupport.Profile profile) throws Exception {
    String name = (asserting ? "asserting_" : "native_")
        + (explicit ? "explicit_" : "buffered_") + label;
    inputs(name, docs, 1, profile);
    EVENTS.clear(); phase = "flush";
    try (var directory = new ByteBuffersDirectory();
         var writer = new IndexWriter(directory,
             TermVectorsRoutingSupport.config(asserting, false, docs, events()))) {
      for (int doc = 0; doc < docs; doc++)
        writer.addDocument(TermVectorsRoutingSupport.document(doc, docs, 1, 0, profile));
      if (explicit) writer.flush();
      writer.commit();
      snapshot(directory, name, "after", docs, 1, profile, false, false, true);
      long vectorFiles = java.util.Arrays.stream(directory.listAll())
          .filter(file -> file.endsWith(".tvd") || file.endsWith(".tvx") || file.endsWith(".tvm")).count();
      // An explicitly empty vector field makes Lucene initialize its writer,
      // unlike a whole session with no declared vector fields.
      TermVectorsRoutingSupport.require(vectorFiles == (label.endsWith("missing") ? 0 : 3),
                                        "missing/empty writer acquisition");
      row("FILES", name, vectorFiles);
      if (label.startsWith("docs") || label.startsWith("bytes")) {
        long expectedChunks = label.equals("docs129") ? 2 : 1;
        long expectedDirty = label.equals("docs128") || label.equals("bytes4096")
            || label.equals("bytes4097") ? 0 : 1;
        long expectedDirtyDocs = expectedDirty == 0 ? 0 : label.equals("docs127") ? 127 : 1;
        int[] leaves = {0};
        TermVectorsRoutingSupport.chunks(directory, cells -> {
          leaves[0]++;
          TermVectorsRoutingSupport.require(((Number) cells[2]).longValue() == expectedChunks
              && ((Number) cells[3]).longValue() == expectedDirty
              && ((Number) cells[4]).longValue() == expectedDirtyDocs,
              "native chunk boundary " + label);
        });
        TermVectorsRoutingSupport.require(leaves[0] == 1, "boundary source count");
      }
    }
    traces(name);
  }

  private static void failureScenario(String operation) throws Exception {
    String name = "asserting_failure_" + operation;
    EVENTS.clear(); phase = "failure";
    int[] fires = {0};
    var profile = TermVectorsRoutingSupport.Profile.rich();
    var injected = new IOException("injected-" + operation);
    TermVectorsRoutingSupport.Events events = (layer, op, details) -> {
      EVENTS.add(phase + "\t" + layer + "\t" + op + "\t" + details);
      if (fires[0] == 0 && (operation.equals("acquire") ? layer.equals("format")
          : layer.equals("native")) && op.equals(operation)) {
        fires[0]++;
        throw injected;
      }
    };
    try (var directory = new ByteBuffersDirectory()) {
      IndexWriter writer = null;
      Throwable primary = null;
      try {
        writer = new IndexWriter(directory, TermVectorsRoutingSupport.config(true, false, 12, events));
        for (int doc = 0; doc < 12; doc++)
          writer.addDocument(TermVectorsRoutingSupport.document(doc, 12, 3, 0, profile));
        writer.commit();
      } catch (Throwable error) {
        primary = error;
      } finally {
        if (writer != null) try { writer.close(); }
        catch (Throwable closing) {
          if (primary == null) primary = closing;
          else primary.addSuppressed(closing);
        }
      }
      TermVectorsRoutingSupport.require(fires[0] == 1 && primary != null, "injected failure fired");
      row("FAILURE", name, primary.getClass().getSimpleName(), primary.getMessage(),
          primary.getSuppressed().length, fires[0]);
    }
    traces(name);
  }

  public static void main(String[] args) throws Exception {
    if (!GenTermVectorsWriterRouting.class.desiredAssertionStatus())
      throw new AssertionError("run native term-vector routing oracle with -ea");
    if (args.length == 3 && args[0].equals("--verify-dir")) {
      verifyExport(Path.of(args[1]), args[2]);
      return;
    }
    Path output = Path.of(args[0]);
    Files.createDirectories(output);
    row("# Scope: Lucene104Codec native binary term-vector writer routing; native readers; optional actual Java asserting writer.");
    row("# Names and raw binary terms are hex. '-' is empty bytes; null payload is distinct in INPUT_TOKEN.");
    row("# Missing and explicit-empty input documents both decode as null vectors; INPUT/FILES/TRACE retain their distinct causes.");
    row("# TRACE records phase, layer, operation and details. Native bulk merge may bypass document callbacks; asserting inherited merge does not.");
    for (var definition : TermVectorsRoutingSupport.DEFINITIONS)
      row("DEFINITION", TermVectorsRoutingSupport.hex(definition.name()), definition.positions(),
          definition.offsets(), definition.payloads());
    for (int field = 0; field < TermVectorsRoutingSupport.DEFINITIONS.size(); field++) {
      var definition = TermVectorsRoutingSupport.definition(field,
          new TermVectorsRoutingSupport.Profile("configured-rich", 0));
      row("DEFINITION_CONFIG", TermVectorsRoutingSupport.hex(definition.name()), definition.positions(),
          definition.offsets(), definition.payloads());
    }
    for (boolean asserting : new boolean[] {false, true}) {
      for (boolean imported : new boolean[] {false, true})
        for (boolean sorted : new boolean[] {false, true}) mergeScenario(asserting, imported, sorted);
      for (boolean explicit : new boolean[] {false, true}) {
        flushScenario(asserting, explicit, "rich", 12, TermVectorsRoutingSupport.Profile.rich());
        flushScenario(asserting, explicit, "empty", 3, new TermVectorsRoutingSupport.Profile("empty", 0));
        flushScenario(asserting, explicit, "missing", 3, new TermVectorsRoutingSupport.Profile("missing", 0));
        flushScenario(asserting, explicit, "configured_rich", 12,
            new TermVectorsRoutingSupport.Profile("configured-rich", 0));
        flushScenario(asserting, explicit, "configured_empty", 3,
            new TermVectorsRoutingSupport.Profile("configured-empty", 0));
        flushScenario(asserting, explicit, "configured_missing", 3,
            new TermVectorsRoutingSupport.Profile("configured-missing", 0));
      }
      for (int docs : new int[] {127, 128, 129})
        flushScenario(asserting, true, "docs" + docs, docs, TermVectorsRoutingSupport.Profile.tiny());
      for (int bytes : new int[] {4095, 4096, 4097})
        flushScenario(asserting, true, "bytes" + bytes, 1, TermVectorsRoutingSupport.Profile.bytes(bytes));
    }
    for (String operation : new String[] {"acquire", "start-document", "finish", "close"})
      failureScenario(operation);
    Files.writeString(output.resolve("term_vectors_writer_routing.tsv"), OUT);
  }

  /** Full native-reader reverse proof of each final Rust fixture replay. */
  private static void verifyExport(Path path, String name) throws Exception {
    boolean merge = name.contains("_direct_") || name.contains("_imported_");
    boolean sorted = name.endsWith("_sorted") && !name.endsWith("_unsorted");
    int docs = 12, sources = merge ? 3 : 1;
    var profile = TermVectorsRoutingSupport.Profile.rich();
    String label = name.substring(name.lastIndexOf('_') + 1);
    if (name.contains("_configured_")) {
      profile = new TermVectorsRoutingSupport.Profile("configured-" + label, 0);
      if (!label.equals("rich")) docs = 3;
    } else if (label.equals("empty") || label.equals("missing")) {
      docs = 3;
      profile = new TermVectorsRoutingSupport.Profile(label, 0);
    } else if (label.startsWith("docs")) {
      docs = Integer.parseInt(label.substring(4));
      profile = TermVectorsRoutingSupport.Profile.tiny();
    } else if (label.startsWith("bytes")) {
      docs = 1;
      profile = TermVectorsRoutingSupport.Profile.bytes(Integer.parseInt(label.substring(5)));
    }
    try (var directory = FSDirectory.open(path)) {
      var check = TermVectorsRoutingSupport.verify(directory,
          TermVectorsRoutingSupport.ids(docs, sources, merge, merge && sorted),
          docs, sources, profile, true, null);
      System.out.println("READBACK\t" + name + "\t" + Long.toUnsignedString(check.digest())
          + "\t" + check.docs() + "\t" + check.fields() + "\t" + check.terms()
          + "\t" + check.occurrences() + "\t" + check.payloadBytes());
    }
  }
}
