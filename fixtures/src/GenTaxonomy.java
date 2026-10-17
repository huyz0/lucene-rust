import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.function.Supplier;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.SortedNumericDocValuesField;
import org.apache.lucene.document.SortedSetDocValuesField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.facet.FacetField;
import org.apache.lucene.facet.FacetsConfig;
import org.apache.lucene.facet.sortedset.SortedSetDocValuesFacetField;
import org.apache.lucene.facet.taxonomy.FacetLabel;
import org.apache.lucene.facet.taxonomy.ParallelTaxonomyArrays;
import org.apache.lucene.facet.taxonomy.TaxonomyReader;
import org.apache.lucene.facet.taxonomy.directory.DirectoryTaxonomyReader;
import org.apache.lucene.facet.taxonomy.directory.DirectoryTaxonomyWriter;
import org.apache.lucene.facet.taxonomy.writercache.LruTaxonomyWriterCache;
import org.apache.lucene.facet.taxonomy.writercache.TaxonomyWriterCache;
import org.apache.lucene.index.IndexWriterConfig.OpenMode;
import org.apache.lucene.index.IndexableField;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.IOUtils;

/**
 * M13 T13.4a: the taxonomy index. Writes {@code taxonomy/}:
 *
 * <ul>
 *   <li>one directory per writer-cache configuration ({@code default}, {@code lru_hashed_3},
 *       {@code lru_string_3}, {@code lru_unbounded}), each holding {@code stage1/}..{@code
 *       stage3/}: the taxonomy after stage 1 (a new index), stage 2 (reopened
 *       CREATE_OR_APPEND, new categories mixed with existing ones) and stage 3 (reopened with
 *       CREATE: a new epoch). {@code ops.tsv} has every {@code addCategory} with the ordinal it
 *       returned, every {@code getParent} and the size after each stage; {@code stageN.tsv} the
 *       reader's dump of {@code stageN/} (see {@link #dump}).
 *   <li>{@code add_taxonomy/}: {@code dest/} after {@code addTaxonomy(default/stage2,
 *       MemoryOrdinalMap)}, its dump and the ordinal map.
 *   <li>{@code replace/}: {@code dest/} after {@code replaceTaxonomy(lru_string_3/stage2)} and
 *       three more categories, its dump.
 *   <li>{@code build/}: {@code FacetsConfig.build} over {@link #docs}: {@code build.tsv} (doc,
 *       field name, kind, value), {@code taxo/} the taxonomy the builds wrote, {@code
 *       taxo.tsv} its dump.
 *   <li>{@code labels.tsv}: {@code FacetLabel} hashCode, longHashCode, toString, subpath and
 *       compareTo over {@link #LABELS}; {@code errors.tsv}: the refusals and their messages.
 * </ul>
 *
 * Fields are tab separated; a path is its components, one per column. No label used here
 * contains a tab or a newline.
 */
public class GenTaxonomy {
  static final String LONG_EXACT = "x".repeat(FacetLabel.MAX_CATEGORY_PATH_LENGTH - 5);

  /** The categories of each stage, in the order they are added. */
  static List<String[]> stage(int stage) {
    List<String[]> out = new ArrayList<>();
    switch (stage) {
      case 1 -> {
        for (int i = 0; i < 12; i++) out.add(new String[] {"Author", "author" + i});
        for (int y = 2000; y < 2003; y++) {
          for (int m = 1; m <= 3; m++) {
            out.add(new String[] {"Pub Date", "" + y, "" + m, "1"});
            out.add(new String[] {"Pub Date", "" + y, "" + m, "15"});
          }
        }
        out.add(new String[] {"Ünïcødé", "日本", "語"});
        out.add(new String[] {"Ünïcødé", "中文"});
        out.add(new String[] {"Ünïcødé", "￿"});
        out.add(new String[] {"Ünïcødé", "😀z"});
        out.add(new String[] {"sep\u001Fdim", "a\u001Fb", "c\u001E"});
        out.add(new String[] {"esc\u001Edim", "\u001E\u001E", "\u001F"});
        out.add(new String[] {"long", LONG_EXACT});
        out.add(new String[] {"long", "y".repeat(1000), "z".repeat(1000)});
        out.add(new String[] {"Author", "author3"}); // already there
      }
      case 2 -> {
        // Existing categories first (cache misses, cache fill), then new ones.
        for (int i = 11; i >= 0; i -= 2) out.add(new String[] {"Author", "author" + i});
        out.add(new String[] {"Pub Date", "2001", "2", "15"});
        out.add(new String[] {"Pub Date", "2001", "2", "16"});
        out.add(new String[] {"Pub Date", "2004"});
        out.add(new String[] {"Pub Date", "2004", "1", "1"});
        for (int i = 0; i < 15; i++) out.add(new String[] {"Tag", "tag" + (i * 7 % 15)});
        for (int i = 0; i < 12; i++) out.add(new String[] {"Author", "author" + i});
        List<String> deep = new ArrayList<>();
        deep.add("deep");
        for (int i = 0; i < 30; i++) deep.add("d" + i);
        out.add(deep.toArray(new String[0]));
        out.add(new String[] {"Ünïcødé", "日本"});
        out.add(new String[] {"Ünïcødé", "한국어"});
        out.add(new String[] {"long", LONG_EXACT});
      }
      case 3 -> {
        out.add(new String[] {"Tag", "tag1"});
        out.add(new String[] {"Author", "author5"});
        out.add(new String[] {"Pub Date", "1999", "12", "31"});
        out.add(new String[] {"Tag", "tag2"});
        out.add(new String[] {"Author", "author5"});
      }
      default -> throw new AssertionError();
    }
    return out;
  }

  /** Paths every dump looks up with {@code getOrdinal}, present or not. */
  static final String[][] PROBES = {
    {},
    {"Author"},
    {"Author", "author3"},
    {"Author", "author99"},
    {"Pub Date", "2001", "2"},
    {"Pub Date", "2004", "1", "1"},
    {"Tag", "tag1"},
    {"Ünïcødé", "😀z"},
    {"sep\u001Fdim", "a\u001Fb", "c\u001E"},
    {"sep\u001Fdim", "a", "b"},
    {"esc\u001Edim", "\u001E\u001E", "\u001F"},
    {"long", LONG_EXACT},
    {"deep", "d0", "d1", "d2"},
    {"nope"},
  };

  record Scenario(String name, Supplier<TaxonomyWriterCache> cache) {}

  static final Scenario[] SCENARIOS = {
    new Scenario("default", () -> null),
    new Scenario("lru_hashed_3",
        () -> new LruTaxonomyWriterCache(3, LruTaxonomyWriterCache.LRUType.LRU_HASHED)),
    new Scenario("lru_string_3",
        () -> new LruTaxonomyWriterCache(3, LruTaxonomyWriterCache.LRUType.LRU_STRING)),
    new Scenario("lru_unbounded", () -> new LruTaxonomyWriterCache(Integer.MAX_VALUE)),
  };

  public static void main(String[] args) throws Exception {
    Path out = Path.of(args[0]).resolve("taxonomy");
    if (Files.exists(out)) IOUtils.rm(out);
    Files.createDirectories(out);

    for (Scenario s : SCENARIOS) {
      Path base = out.resolve(s.name);
      Files.createDirectories(base);
      StringBuilder ops = new StringBuilder();
      for (int st = 1; st <= 3; st++) {
        Path dir = base.resolve("stage" + st);
        if (st > 1) copy(base.resolve("stage" + (st - 1)), dir);
        OpenMode mode = st == 1 ? OpenMode.CREATE : st == 2 ? OpenMode.CREATE_OR_APPEND : OpenMode.CREATE;
        try (Directory d = FSDirectory.open(dir)) {
          TaxonomyWriterCache cache = s.cache.get();
          DirectoryTaxonomyWriter w =
              cache == null ? new DirectoryTaxonomyWriter(d, mode) : new DirectoryTaxonomyWriter(d, mode, cache);
          for (String[] path : stage(st)) {
            int ord = w.addCategory(new FacetLabel(path));
            ops.append(st).append("\tadd\t").append(ord);
            for (String c : path) ops.append('\t').append(c);
            ops.append('\n');
          }
          for (int o = 0; o < w.getSize(); o++) {
            ops.append(st).append("\tparent\t").append(o).append('\t').append(w.getParent(o)).append('\n');
          }
          ops.append(st).append("\tsize\t").append(w.getSize()).append('\n');
          if (st == 1) w.setLiveCommitData(Map.of("user", "stage1").entrySet());
          w.close();
        }
        try (Directory d = FSDirectory.open(dir);
            DirectoryTaxonomyReader r = new DirectoryTaxonomyReader(d)) {
          write(base.resolve("stage" + st + ".tsv"), dump(r));
          Files.writeString(base.resolve("stage" + st + ".tostring.txt"), r.toString(1000),
              StandardCharsets.UTF_8);
        }
      }
      write(base.resolve("ops.tsv"), ops.toString());
    }

    // addTaxonomy
    {
      Path base = out.resolve("add_taxonomy");
      Path dest = base.resolve("dest");
      Files.createDirectories(dest);
      StringBuilder sb = new StringBuilder();
      try (Directory d = FSDirectory.open(dest);
          Directory src = FSDirectory.open(out.resolve("default").resolve("stage2"))) {
        DirectoryTaxonomyWriter w = new DirectoryTaxonomyWriter(d);
        w.addCategory(new FacetLabel("Tag", "tag3"));
        w.addCategory(new FacetLabel("Other", "x", "y"));
        DirectoryTaxonomyWriter.MemoryOrdinalMap map = new DirectoryTaxonomyWriter.MemoryOrdinalMap();
        w.addTaxonomy(src, map);
        int[] m = map.getMap();
        for (int i = 0; i < m.length; i++) sb.append("map\t").append(i).append('\t').append(m[i]).append('\n');
        sb.append("size\t").append(w.getSize()).append('\n');
        w.close();
      }
      write(base.resolve("map.tsv"), sb.toString());
      try (Directory d = FSDirectory.open(dest); DirectoryTaxonomyReader r = new DirectoryTaxonomyReader(d)) {
        write(base.resolve("dest.tsv"), dump(r));
      }
    }

    // replaceTaxonomy
    {
      Path base = out.resolve("replace");
      Path dest = base.resolve("dest");
      copy(out.resolve("default").resolve("stage1"), dest);
      StringBuilder sb = new StringBuilder();
      try (Directory d = FSDirectory.open(dest);
          Directory src = FSDirectory.open(out.resolve("lru_string_3").resolve("stage2"))) {
        DirectoryTaxonomyWriter w = new DirectoryTaxonomyWriter(d);
        w.replaceTaxonomy(src);
        sb.append("epoch\t").append(Long.toString(w.getTaxonomyEpoch(), 16)).append('\n');
        for (String[] p : new String[][] {{"Author", "author1"}, {"New", "one"}, {"Tag", "tag9"}}) {
          sb.append("add\t").append(w.addCategory(new FacetLabel(p)));
          for (String c : p) sb.append('\t').append(c);
          sb.append('\n');
        }
        sb.append("size\t").append(w.getSize()).append('\n');
        w.close();
      }
      write(base.resolve("ops.tsv"), sb.toString());
      try (Directory d = FSDirectory.open(dest); DirectoryTaxonomyReader r = new DirectoryTaxonomyReader(d)) {
        write(base.resolve("dest.tsv"), dump(r));
      }
    }

    build(out.resolve("build"));
    labels(out);
    errors(out);
  }

  static FacetsConfig config() {
    FacetsConfig c = new FacetsConfig();
    c.setHierarchical("Publish Date", true);
    c.setMultiValued("Tags", true);
    c.setHierarchical("Cat", true);
    c.setMultiValued("Cat", true);
    c.setMultiValued("Year", true);
    c.setRequireDimCount("Year", true);
    c.setHierarchical("Loc", true);
    c.setMultiValued("Loc", true);
    c.setRequireDimCount("Loc", true);
    c.setIndexFieldName("Loc", "$loc");
    c.setIndexFieldName("Price", "$loc");
    c.setIndexFieldName("Color", "$color");
    c.setDrillDownTermsIndexing("Color", FacetsConfig.DrillDownTermsIndexing.NONE);
    c.setHierarchical("Size", true);
    c.setDrillDownTermsIndexing("Size", FacetsConfig.DrillDownTermsIndexing.DIMENSION_AND_FULL_PATH);
    c.setHierarchical("Shape", true);
    c.setDrillDownTermsIndexing("Shape", FacetsConfig.DrillDownTermsIndexing.ALL_PATHS_NO_DIM);
    c.setDrillDownTermsIndexing("Flat", FacetsConfig.DrillDownTermsIndexing.FULL_PATH_ONLY);
    c.setHierarchical("Flat", true);
    c.setIndexFieldName("A1", "$a");
    c.setIndexFieldName("B1", "$b");
    c.setIndexFieldName("Z1", "$z9");
    c.setIndexFieldName("E1", "$€");
    c.setHierarchical("E1", true);
    c.setMultiValued("E1", true);
    c.setIndexFieldName("Brand", "$ssdv");
    c.setIndexFieldName("Region", "$ssdv");
    c.setHierarchical("Region", true);
    c.setMultiValued("Kind", true);
    c.setRequireDimCount("Kind", true);
    return c;
  }

  /**
   * The documents: {@code F} entries are {@code FacetField}s, {@code S} entries {@code
   * SortedSetDocValuesFacetField}s, {@code I} the non-facet {@code id} field.
   */
  static List<List<String[]>> docs() {
    List<List<String[]>> docs = new ArrayList<>();
    docs.add(List.<String[]>of(
        new String[] {"I", "d0"},
        new String[] {"F", "Author", "Bob"},
        new String[] {"F", "Publish Date", "2010", "10", "15"},
        new String[] {"F", "Tags", "x"},
        new String[] {"F", "Tags", "y"}));
    docs.add(List.<String[]>of(
        new String[] {"F", "Cat", "a", "b", "c"},
        new String[] {"I", "d1"},
        new String[] {"F", "Cat", "a", "d"},
        new String[] {"F", "Year", "2010"},
        new String[] {"F", "Year", "2012"},
        new String[] {"F", "Loc", "eu", "fr", "paris"},
        new String[] {"F", "Price", "cheap"}));
    docs.add(List.<String[]>of(
        new String[] {"F", "Color", "red"},
        new String[] {"F", "Size", "s", "m", "l"},
        new String[] {"F", "Shape", "round", "small", "tiny"},
        new String[] {"F", "Flat", "a", "b"},
        new String[] {"I", "d2"}));
    docs.add(List.<String[]>of(
        new String[] {"F", "A1", "v"},
        new String[] {"F", "B1", "v"},
        new String[] {"F", "Z1", "v"},
        new String[] {"F", "E1", "€", "¢"},
        new String[] {"F", "E1", "€", "£"},
        new String[] {"F", "Author", "Lisa"},
        new String[] {"S", "Brand", "acme"},
        new String[] {"S", "Region", "eu", "de"},
        new String[] {"S", "Kind", "k1"},
        new String[] {"S", "Kind", "k2"},
        new String[] {"I", "d3"}));
    docs.add(List.<String[]>of(
        new String[] {"F", "Author", "Bob"},
        new String[] {"F", "Cat", "a", "b", "c"},
        new String[] {"F", "Cat", "a", "b"},
        new String[] {"F", "Loc", "eu", "fr"},
        new String[] {"F", "sep\u001Fdim", "a\u001Fb"},
        new String[] {"F", "Ünïcødé", "😀"}));
    docs.add(List.<String[]>of(new String[] {"I", "only"}));
    return docs;
  }

  static Document toDoc(List<String[]> spec) {
    Document doc = new Document();
    for (String[] f : spec) {
      String[] path = Arrays.copyOfRange(f, 2, f.length);
      switch (f[0]) {
        case "F" -> doc.add(new FacetField(f[1], path));
        case "S" -> doc.add(new SortedSetDocValuesFacetField(f[1], path));
        case "I" -> doc.add(new StringField("id", f[1], Field.Store.YES));
        default -> throw new AssertionError();
      }
    }
    return doc;
  }

  static void build(Path base) throws Exception {
    Path taxo = base.resolve("taxo");
    Files.createDirectories(taxo);
    FacetsConfig config = config();
    StringBuilder sb = new StringBuilder();
    try (Directory d = FSDirectory.open(taxo)) {
      DirectoryTaxonomyWriter w = new DirectoryTaxonomyWriter(d);
      List<List<String[]>> docs = docs();
      for (int i = 0; i < docs.size(); i++) {
        Document built = config.build(w, toDoc(docs.get(i)));
        for (IndexableField f : built.getFields()) {
          sb.append(i).append('\t').append(f.name()).append('\t');
          if (f instanceof SortedNumericDocValuesField) {
            sb.append("sndv\t").append(f.numericValue());
          } else if (f instanceof SortedSetDocValuesField) {
            sb.append("ssdv\t").append(f.binaryValue().utf8ToString());
          } else if (f instanceof StringField) {
            sb.append(f.fieldType().stored() ? "string_stored\t" : "string\t").append(f.stringValue());
          } else {
            throw new AssertionError(f.getClass().toString());
          }
          sb.append('\n');
        }
      }
      w.close();
    }
    write(base.resolve("build.tsv"), sb.toString());
    try (Directory d = FSDirectory.open(taxo); DirectoryTaxonomyReader r = new DirectoryTaxonomyReader(d)) {
      write(base.resolve("taxo.tsv"), dump(r));
    }
  }

  static final String[][] LABELS = {
    {},
    {"a"},
    {"a", "b"},
    {"a", "b", "c"},
    {"b"},
    {"ab"},
    {"a", "bc"},
    {"Ünïcødé", "日本"},
    {"￿"},
    {"😀"},
    {"😀z"},
    {"sep\u001Fdim", "a\u001Fb"},
    {"Author", "author3"},
    {"Pub Date", "2001", "2", "15"},
    {"long", LONG_EXACT},
  };

  static void labels(Path out) throws Exception {
    StringBuilder sb = new StringBuilder();
    for (String[] p : LABELS) {
      FacetLabel l = new FacetLabel(p);
      sb.append("label\t").append(l.hashCode()).append('\t').append(l.longHashCode()).append('\t')
          .append(l.toString().length() > 200 ? "LONG" : l.toString());
      for (String c : p) sb.append('\t').append(c.length() > 200 ? "LONG" : c);
      sb.append('\n');
      for (int n = 0; n <= p.length; n++) {
        FacetLabel sub = l.subpath(n);
        sb.append("subpath\t").append(n).append('\t').append(sub.length).append('\t')
            .append(sub.hashCode()).append('\t').append(sub.longHashCode()).append('\t')
            .append(p.length == 0 ? "-" : sub.lastComponent().length() > 200 ? "LONG" : sub.lastComponent())
            .append('\n');
      }
    }
    for (int i = 0; i < LABELS.length; i++) {
      for (int j = 0; j < LABELS.length; j++) {
        FacetLabel a = new FacetLabel(LABELS[i]);
        FacetLabel b = new FacetLabel(LABELS[j]);
        sb.append("compare\t").append(i).append('\t').append(j).append('\t')
            .append(Integer.signum(a.compareTo(b))).append('\t').append(a.equals(b)).append('\n');
      }
    }
    write(out.resolve("labels.tsv"), sb.toString());
  }

  interface Thrower {
    void run() throws Exception;
  }

  static void error(StringBuilder sb, String name, Thrower t) {
    try {
      t.run();
      sb.append(name).append("\tnone\t\n");
    } catch (Exception e) {
      sb.append(name).append('\t').append(e.getClass().getName()).append('\t')
          .append(e.getMessage() == null ? "" : e.getMessage()).append('\n');
    }
  }

  static void errors(Path out) throws Exception {
    StringBuilder sb = new StringBuilder();
    String tooLong = "x".repeat(FacetLabel.MAX_CATEGORY_PATH_LENGTH - 4);
    error(sb, "label_too_long", () -> new FacetLabel("long", tooLong));
    error(sb, "label_exact_max", () -> new FacetLabel("long", LONG_EXACT));
    error(sb, "label_too_long_one_component", () -> new FacetLabel("y".repeat(8192)));
    error(sb, "label_empty_component", () -> new FacetLabel("a", "", "b"));
    error(sb, "label_empty_dim", () -> new FacetLabel("", new String[] {"b"}));
    error(sb, "last_component_empty", () -> new FacetLabel().lastComponent());
    error(sb, "facet_field_empty_dim", () -> new FacetField("", "a"));
    error(sb, "facet_field_empty_label", () -> new FacetField("a", "b", ""));
    error(sb, "facet_field_no_path", () -> new FacetField("a"));
    FacetsConfig config = config();
    try (Directory d = new org.apache.lucene.store.ByteBuffersDirectory()) {
      DirectoryTaxonomyWriter w = new DirectoryTaxonomyWriter(d);
      error(sb, "build_not_hierarchical",
          () -> config.build(w, toDoc(List.<String[]>of(new String[] {"F", "Author", "a", "b"}))));
      error(sb, "build_not_multivalued",
          () -> config.build(w, toDoc(List.<String[]>of(
              new String[] {"F", "Author", "a"}, new String[] {"F", "Author", "b"}))));
      error(sb, "build_not_multivalued_mixed",
          () -> config.build(w, toDoc(List.<String[]>of(
              new String[] {"S", "Brand", "a"}, new String[] {"F", "Brand", "b"}))));
      error(sb, "build_ssdv_not_hierarchical",
          () -> config.build(w, toDoc(List.<String[]>of(new String[] {"S", "Brand", "a", "b"}))));
      error(sb, "build_no_taxo_writer",
          () -> config.build(toDoc(List.<String[]>of(new String[] {"F", "Author", "a"}))));
      error(sb, "build_ssdv_only_no_taxo_writer",
          () -> config.build(toDoc(List.<String[]>of(new String[] {"S", "Brand", "a"}))));
      error(sb, "writer_parent_out_of_range", () -> w.getParent(w.getSize()));
      error(sb, "writer_parent_negative", () -> w.getParent(-1));
      w.commit();
      try (DirectoryTaxonomyReader r = new DirectoryTaxonomyReader(d)) {
        error(sb, "reader_path_out_of_range", () -> r.getPath(r.getSize()));
        error(sb, "reader_path_negative", () -> r.getPath(-1));
        error(sb, "reader_bulk_path_out_of_range", () -> r.getBulkPath(0, 99));
      }
      w.close();
      error(sb, "writer_closed", () -> w.addCategory(new FacetLabel("a")));
      error(sb, "writer_closed_size", () -> w.getSize());
    }
    write(out.resolve("errors.tsv"), sb.toString());
  }

  /**
   * A reader's dump: {@code size}, {@code epoch} (or {@code -}), {@code user} (each commit user
   * data entry, by key), {@code ord} (ordinal, parent, youngest child, older sibling, the
   * path), {@code children} (the ordinal, then {@code getChildren}' ordinals), {@code lookup}
   * (each {@link #PROBES} path's {@code getOrdinal}), {@code bulk_ord} ({@code getBulkOrdinals}
   * over every probe), {@code bulk_path} ({@code getBulkPath} over every ordinal, last first).
   */
  static String dump(DirectoryTaxonomyReader r) throws Exception {
    StringBuilder sb = new StringBuilder();
    sb.append("size\t").append(r.getSize()).append('\n');
    Map<String, String> user = new TreeMap<>(r.getCommitUserData());
    for (Map.Entry<String, String> e : user.entrySet()) {
      sb.append("user\t").append(e.getKey()).append('\t').append(e.getValue()).append('\n');
    }
    ParallelTaxonomyArrays arrays = r.getParallelTaxonomyArrays();
    for (int o = 0; o < r.getSize(); o++) {
      sb.append("ord\t").append(o).append('\t').append(arrays.parents().get(o)).append('\t')
          .append(arrays.children().get(o)).append('\t').append(arrays.siblings().get(o));
      for (String c : r.getPath(o).components) sb.append('\t').append(c);
      sb.append('\n');
    }
    for (int o = 0; o < r.getSize(); o++) {
      sb.append("children\t").append(o);
      TaxonomyReader.ChildrenIterator it = r.getChildren(o);
      for (int c = it.next(); c != TaxonomyReader.INVALID_ORDINAL; c = it.next()) sb.append('\t').append(c);
      sb.append('\n');
    }
    FacetLabel[] probes = new FacetLabel[PROBES.length];
    for (int i = 0; i < PROBES.length; i++) {
      probes[i] = new FacetLabel(PROBES[i]);
      sb.append("lookup\t").append(i).append('\t').append(r.getOrdinal(probes[i])).append('\n');
    }
    sb.append("bulk_ord");
    for (int o : r.getBulkOrdinals(probes)) sb.append('\t').append(o);
    sb.append('\n');
    int[] ords = new int[r.getSize()];
    for (int i = 0; i < ords.length; i++) ords[i] = ords.length - 1 - i;
    sb.append("bulk_path");
    for (FacetLabel l : r.getBulkPath(ords)) sb.append('\t').append(l.length);
    sb.append('\n');
    return sb.toString();
  }

  static void copy(Path from, Path to) throws Exception {
    Files.createDirectories(to);
    try (var files = Files.list(from)) {
      for (Path f : (Iterable<Path>) files::iterator) {
        if (f.getFileName().toString().equals("write.lock")) continue;
        Files.copy(f, to.resolve(f.getFileName()));
      }
    }
  }

  static void write(Path p, String s) throws Exception {
    Files.writeString(p, s, StandardCharsets.UTF_8);
  }
}
