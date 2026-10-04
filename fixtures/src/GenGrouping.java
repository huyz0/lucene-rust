import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Comparator;
import java.util.List;
import java.util.Random;
import java.util.function.Supplier;
import java.util.stream.Stream;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.DoubleDocValuesField;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.NumericDocValuesField;
import org.apache.lucene.document.SortedDocValuesField;
import org.apache.lucene.document.SortedNumericDocValuesField;
import org.apache.lucene.document.SortedSetDocValuesField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.DoubleValuesSource;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.LongValuesSource;
import org.apache.lucene.search.MatchAllDocsQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.Sort;
import org.apache.lucene.search.SortField;
import org.apache.lucene.search.SortedNumericSelector;
import org.apache.lucene.search.SortedNumericSortField;
import org.apache.lucene.search.SortedSetSelector;
import org.apache.lucene.search.SortedSetSortField;
import org.apache.lucene.search.FieldDoc;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.search.grouping.AllGroupHeadsCollectorManager;
import org.apache.lucene.search.grouping.AllGroupsCollectorManager;
import org.apache.lucene.search.grouping.DistinctValuesCollector;
import org.apache.lucene.search.grouping.DistinctValuesCollectorManager;
import org.apache.lucene.search.grouping.DoubleRange;
import org.apache.lucene.search.grouping.DoubleRangeFactory;
import org.apache.lucene.search.grouping.DoubleRangeGroupSelector;
import org.apache.lucene.search.grouping.FirstPassGroupingCollectorManager;
import org.apache.lucene.search.grouping.GroupDocs;
import org.apache.lucene.search.grouping.GroupFacetCollector;
import org.apache.lucene.search.grouping.GroupSelector;
import org.apache.lucene.search.grouping.GroupingSearch;
import org.apache.lucene.search.grouping.LongRange;
import org.apache.lucene.search.grouping.LongRangeFactory;
import org.apache.lucene.search.grouping.LongRangeGroupSelector;
import org.apache.lucene.search.grouping.SearchGroup;
import org.apache.lucene.search.grouping.TermGroupFacetCollector;
import org.apache.lucene.search.grouping.TermGroupSelector;
import org.apache.lucene.search.grouping.TopGroups;
import org.apache.lucene.search.grouping.TopGroupsCollectorManager;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.BytesRef;
import org.apache.lucene.util.NumericUtils;

/**
 * M10 T10.4's grouping fixture: {@code grouping/index}, a four-segment index with deletions in two
 * segments, and {@code grouping/blocks}, a three-segment index of document blocks (each closed by
 * a document with {@code end:x}) with deletions, some of them of a block's closing document.
 *
 * <p>Fields of {@code index}: {@code body} (text, for scores), {@code g} (the {@code SORTED} group
 * field, sometimes missing), {@code g2} ({@code SORTED}, the distinct values), {@code s} ({@code
 * SORTED} sort key), {@code n} ({@code NUMERIC} long), {@code ni} ({@code NUMERIC} int), {@code d}
 * ({@code NUMERIC} double bits, for {@code DoubleValuesSource.fromDoubleField}), {@code ds}/{@code
 * fs} ({@code SORTED_NUMERIC} sortable double/float), {@code mv} ({@code SORTED_NUMERIC}, zero to
 * three longs), {@code fsv}/{@code fmv} (the facet fields, {@code SORTED} and {@code SORTED_SET}).
 * Values repeat, so sorts tie.
 *
 * <p>{@code searches.tsv}: one search per line, {@code kind \t spec \t answer}. Kinds: {@code gs}
 * ({@code GroupingSearch} by a selector: its {@code TopGroups}, matching groups and group heads),
 * {@code mgr} (the collector managers -- first pass, top groups, all groups, group heads, distinct
 * values -- over a one-slice and a two-slice searcher), {@code block} ({@code GroupingSearch} by
 * blocks) and {@code facet} ({@code TermGroupFacetCollector}), or the exception Lucene threw. The
 * spec grammar is the Rust test's ({@code grouping_fixtures.rs}).
 *
 * <p>Usage: {@code java GenGrouping <fixtures-data-dir>}.
 */
public class GenGrouping {
  static final String[] WORDS = {"red", "blue", "green", "fast", "slow", "big", "small", "old"};

  static void clean(Path root) throws IOException {
    if (Files.exists(root)) {
      try (Stream<Path> walk = Files.walk(root)) {
        for (Path p : walk.sorted(Comparator.reverseOrder()).toList()) Files.delete(p);
      }
    }
  }

  static String word(Random r) {
    int i = (int) (Math.pow(r.nextDouble(), 1.5) * WORDS.length);
    return WORDS[Math.min(i, WORDS.length - 1)];
  }

  static String body(Random r) {
    int n = 1 + r.nextInt(6);
    StringBuilder b = new StringBuilder();
    for (int i = 0; i < n; i++) {
      if (i > 0) b.append(' ');
      b.append(word(r));
    }
    return b.toString();
  }

  static Document doc(Random r, int id) {
    Document d = new Document();
    d.add(new StringField("id", "d" + id, Field.Store.NO));
    d.add(new TextField("body", body(r), Field.Store.NO));
    if (r.nextInt(8) != 0) {
      d.add(new SortedDocValuesField("g", new BytesRef("g" + r.nextInt(9))));
    }
    if (r.nextInt(5) != 0) {
      d.add(new SortedDocValuesField("g2", new BytesRef("v" + r.nextInt(5))));
    }
    if (r.nextInt(6) != 0) {
      d.add(new SortedDocValuesField("s", new BytesRef("s" + r.nextInt(12))));
    }
    if (r.nextInt(6) != 0) {
      d.add(new NumericDocValuesField("n", r.nextInt(30) - 6));
    }
    if (r.nextInt(6) != 0) {
      d.add(new NumericDocValuesField("ni", r.nextInt(7)));
    }
    if (r.nextInt(6) != 0) {
      double v = (r.nextInt(40) - 10) / 4.0;
      d.add(new DoubleDocValuesField("d", v));
      d.add(new SortedNumericDocValuesField("ds", NumericUtils.doubleToSortableLong(v)));
    }
    if (r.nextInt(6) != 0) {
      float v = (r.nextInt(20) - 5) / 2f;
      d.add(new SortedNumericDocValuesField("fs", NumericUtils.floatToSortableInt(v)));
    }
    int m = r.nextInt(4);
    for (int i = 0; i < m; i++) {
      d.add(new SortedNumericDocValuesField("mv", r.nextInt(20)));
    }
    if (r.nextInt(5) != 0) {
      d.add(new SortedDocValuesField("fsv", new BytesRef((r.nextBoolean() ? "a" : "b") + r.nextInt(4))));
    }
    int fm = r.nextInt(4);
    for (int i = 0; i < fm; i++) {
      d.add(new SortedSetDocValuesField("fmv", new BytesRef((r.nextBoolean() ? "a" : "b") + r.nextInt(4))));
    }
    return d;
  }

  static IndexWriterConfig config() {
    IndexWriterConfig cfg = new IndexWriterConfig(new StandardAnalyzer());
    cfg.setUseCompoundFile(false);
    cfg.setMergePolicy(NoMergePolicy.INSTANCE);
    cfg.setMaxBufferedDocs(IndexWriterConfig.DISABLE_AUTO_FLUSH);
    cfg.setRAMBufferSizeMB(256);
    return cfg;
  }

  public static void main(String[] args) throws IOException {
    Path root = Path.of(args[0]).resolve("grouping");
    clean(root);
    Path indexDir = root.resolve("index");
    Path blocksDir = root.resolve("blocks");
    Files.createDirectories(indexDir);
    Files.createDirectories(blocksDir);
    Random r = new Random(0x10_4_2026_1004L);
    try (Directory dir = FSDirectory.open(indexDir);
        IndexWriter w = new IndexWriter(dir, config())) {
      int id = 0;
      int[] sizes = {110, 80, 95, 30};
      for (int seg = 0; seg < sizes.length; seg++) {
        for (int i = 0; i < sizes[seg]; i++, id++) {
          w.addDocument(doc(r, id));
        }
        w.commit();
        if (seg == 1 || seg == 2) {
          for (int k = 0; k < 14; k++) {
            w.deleteDocuments(new Term("id", "d" + (id - 1 - r.nextInt(sizes[seg]))));
          }
          w.commit();
        }
      }
    }
    try (Directory dir = FSDirectory.open(blocksDir);
        IndexWriter w = new IndexWriter(dir, config())) {
      int id = 0;
      int[] blocks = {30, 22, 12};
      for (int seg = 0; seg < blocks.length; seg++) {
        for (int b = 0; b < blocks[seg]; b++) {
          List<Document> block = new ArrayList<>();
          int n = 1 + r.nextInt(6);
          for (int i = 0; i < n; i++, id++) {
            Document d = doc(r, id);
            if (i == n - 1) d.add(new StringField("end", "x", Field.Store.NO));
            block.add(d);
          }
          w.addDocuments(block);
        }
        w.commit();
        if (seg >= 1) {
          for (int k = 0; k < 10; k++) {
            w.deleteDocuments(new Term("id", "d" + (id - 1 - r.nextInt(40))));
          }
          w.commit();
        }
      }
    }

    StringBuilder out = new StringBuilder();
    try (Directory dir = FSDirectory.open(indexDir);
        DirectoryReader reader = DirectoryReader.open(dir);
        Directory bdir = FSDirectory.open(blocksDir);
        DirectoryReader blocks = DirectoryReader.open(bdir)) {
      if (reader.leaves().size() != 4) throw new AssertionError("segments");
      if (blocks.leaves().size() != 3) throw new AssertionError("block segments");
      new GenGrouping(r, reader, blocks, out).run();
    }
    Files.writeString(root.resolve("searches.tsv"), out.toString(), StandardCharsets.UTF_8);
  }

  final Random r;
  final DirectoryReader reader;
  final IndexSearcher s;
  final IndexSearcher sliced;
  final IndexSearcher bs;
  final IndexSearcher bsliced;
  final StringBuilder out;

  static IndexSearcher.LeafSlice slice(LeafReaderContext... leaves) {
    List<IndexSearcher.LeafReaderContextPartition> parts = new ArrayList<>();
    for (LeafReaderContext l : leaves) {
      parts.add(IndexSearcher.LeafReaderContextPartition.createForEntireSegment(l));
    }
    return new IndexSearcher.LeafSlice(parts);
  }

  /** A searcher whose slices are {@code [[0,2],[1,3]]} (or {@code [[0],[1,2]]} for three). */
  static IndexSearcher sliced(DirectoryReader reader) {
    IndexSearcher searcher =
        new IndexSearcher(reader, Runnable::run) {
          @Override
          protected LeafSlice[] slices(List<LeafReaderContext> leaves) {
            if (leaves.size() == 4) {
              return new LeafSlice[] {
                slice(leaves.get(0), leaves.get(2)),
                slice(leaves.get(1), leaves.get(3))
              };
            }
            return new LeafSlice[] {
              slice(leaves.get(0)),
              slice(leaves.get(1), leaves.get(2))
            };
          }
        };
    searcher.setQueryCache(null);
    return searcher;
  }

  GenGrouping(Random r, DirectoryReader reader, DirectoryReader blocks, StringBuilder out) {
    this.r = r;
    this.reader = reader;
    this.out = out;
    this.s = new IndexSearcher(reader);
    s.setQueryCache(null);
    this.sliced = sliced(reader);
    this.bs = new IndexSearcher(blocks);
    bs.setQueryCache(null);
    this.bsliced = sliced(blocks);
  }

  // ---------------------------------------------------------------------------------------------
  // Specs: a query, a selector, a sort -- each as text the Rust test parses, and the Java object.
  // ---------------------------------------------------------------------------------------------

  record Q(String spec, Query query) {}

  Q query() {
    return switch (r.nextInt(5)) {
      case 0 -> new Q("all", new MatchAllDocsQuery());
      case 1, 2 -> {
        String w = word(r);
        yield new Q("t:" + w, new TermQuery(new Term("body", w)));
      }
      case 3 -> {
        String a = word(r), b = word(r);
        BooleanQuery.Builder q = new BooleanQuery.Builder();
        q.add(new TermQuery(new Term("body", a)), BooleanClause.Occur.SHOULD);
        q.add(new TermQuery(new Term("body", b)), BooleanClause.Occur.SHOULD);
        yield new Q("or:" + a + "," + b, q.build());
      }
      default -> {
        String a = word(r), b = word(r);
        BooleanQuery.Builder q = new BooleanQuery.Builder();
        q.add(new TermQuery(new Term("body", a)), BooleanClause.Occur.MUST);
        q.add(new TermQuery(new Term("body", b)), BooleanClause.Occur.MUST_NOT);
        yield new Q("not:" + a + "," + b, q.build());
      }
    };
  }

  /** A sort key: {@code name[:r]}. */
  SortField key(String name, boolean reverse) {
    return switch (name) {
      case "score" -> new SortField(null, SortField.Type.SCORE, reverse);
      case "doc" -> new SortField(null, SortField.Type.DOC, reverse);
      case "s" -> new SortField("s", SortField.Type.STRING, reverse);
      case "slast" -> {
        SortField f = new SortField("s", SortField.Type.STRING, reverse);
        f.setMissingValue(SortField.STRING_LAST);
        yield f;
      }
      case "g" -> new SortField("g", SortField.Type.STRING, reverse);
      case "n" -> new SortField("n", SortField.Type.LONG, reverse);
      case "nm" -> {
        SortField f = new SortField("n", SortField.Type.LONG, reverse);
        f.setMissingValue(7L);
        yield f;
      }
      case "ni" -> new SortField("ni", SortField.Type.INT, reverse);
      case "ds" -> new SortedNumericSortField("ds", SortField.Type.DOUBLE, reverse);
      case "fs" -> new SortedNumericSortField("fs", SortField.Type.FLOAT, reverse);
      case "mvmax" ->
          new SortedNumericSortField("mv", SortField.Type.LONG, reverse, SortedNumericSelector.Type.MAX);
      case "fmv" -> new SortedSetSortField("fmv", reverse, SortedSetSelector.Type.MIN);
      default -> throw new IllegalArgumentException(name);
    };
  }

  static final String[] KEYS = {
    "score", "doc", "s", "slast", "g", "n", "nm", "ni", "ds", "fs", "mvmax", "fmv"
  };

  record S(String spec, Sort sort) {}

  S sort() {
    switch (r.nextInt(6)) {
      case 0:
        return new S("REL", Sort.RELEVANCE);
      case 1:
        return new S("IDX", Sort.INDEXORDER);
      default:
        int n = 1 + r.nextInt(3);
        List<SortField> fields = new ArrayList<>();
        StringBuilder spec = new StringBuilder();
        for (int i = 0; i < n; i++) {
          String k = KEYS[r.nextInt(KEYS.length)];
          boolean rev = r.nextInt(3) == 0;
          if (i > 0) spec.append(',');
          spec.append(k).append(rev ? ":r" : "");
          fields.add(key(k, rev));
        }
        return new S(spec.toString(), new Sort(fields.toArray(new SortField[0])));
    }
  }

  record Sel(String spec, Supplier<GroupSelector<?>> make) {}

  Sel selector(boolean scores) {
    int c = r.nextInt(scores ? 6 : 5);
    return switch (c) {
      case 0, 1, 2 -> new Sel("term:g", () -> new TermGroupSelector("g"));
      case 3 -> {
        long min = r.nextInt(5) - 2, width = 2 + r.nextInt(5), max = min + width * (2 + r.nextInt(4));
        yield new Sel(
            "long:n:" + min + ":" + width + ":" + max,
            () ->
                new LongRangeGroupSelector(
                    LongValuesSource.fromLongField("n"), new LongRangeFactory(min, width, max)));
      }
      case 4 -> {
        double min = (r.nextInt(8) - 4) / 2.0, width = (1 + r.nextInt(6)) / 2.0;
        double max = min + width * (2 + r.nextInt(4));
        yield new Sel(
            "double:d:" + min + ":" + width + ":" + max,
            () ->
                new DoubleRangeGroupSelector(
                    DoubleValuesSource.fromDoubleField("d"), new DoubleRangeFactory(min, width, max)));
      }
      default -> {
        double width = (1 + r.nextInt(4)) / 4.0;
        yield new Sel(
            "dscore:0.0:" + width + ":2.0",
            () ->
                new DoubleRangeGroupSelector(
                    DoubleValuesSource.SCORES, new DoubleRangeFactory(0.0, width, 2.0)));
      }
    };
  }

  // ---------------------------------------------------------------------------------------------
  // Formatting
  // ---------------------------------------------------------------------------------------------

  static String hex(float f) {
    return Integer.toHexString(Float.floatToIntBits(f));
  }

  static String hex(double d) {
    return Long.toHexString(Double.doubleToLongBits(d));
  }

  static String val(Object o) {
    if (o == null) return "null";
    if (o instanceof BytesRef b) return "b:" + b.utf8ToString();
    if (o instanceof Float f) return "f:" + hex(f);
    if (o instanceof Double d) return "d:" + hex(d);
    if (o instanceof Long l) return "l:" + l;
    if (o instanceof Integer i) return "i:" + i;
    if (o instanceof LongRange lr) return "L(" + lr.min + "," + lr.max + ")";
    if (o instanceof DoubleRange dr) return "D(" + hex(dr.min) + "," + hex(dr.max) + ")";
    throw new IllegalArgumentException(o.getClass().toString());
  }

  static String vals(Object[] values) {
    if (values == null) return "null";
    StringBuilder b = new StringBuilder();
    for (int i = 0; i < values.length; i++) {
      if (i > 0) b.append(',');
      b.append(val(values[i]));
    }
    return b.toString();
  }

  static String topGroups(TopGroups<?> tg) {
    if (tg == null) return "null";
    StringBuilder b = new StringBuilder();
    b.append("thc=").append(tg.totalHitCount);
    b.append(" tghc=").append(tg.totalGroupedHitCount);
    b.append(" tgc=").append(tg.totalGroupCount);
    b.append(" ms=").append(hex(tg.maxScore));
    b.append(" gsn=").append(tg.groupSort.length);
    b.append(" wsn=").append(tg.withinGroupSort.length);
    for (GroupDocs<?> g : tg.groups) {
      b.append(" |gv=").append(val(g.groupValue()));
      b.append(" sv=").append(vals(g.groupSortValues()));
      b.append(" s=").append(hex(g.score()));
      b.append(" ms=").append(hex(g.maxScore()));
      b.append(" th=").append(g.totalHits().value());
      b.append(" docs=");
      ScoreDoc[] docs = g.scoreDocs();
      for (int i = 0; i < docs.length; i++) {
        if (i > 0) b.append(';');
        b.append(docs[i].doc).append(':').append(hex(docs[i].score));
        if (docs[i] instanceof FieldDoc fd) b.append(':').append(vals(fd.fields));
      }
    }
    return b.toString();
  }

  static String sortedVals(Collection<?> values) {
    List<String> l = new ArrayList<>();
    for (Object o : values) l.add(val(o));
    l.sort(null);
    return String.join(",", l);
  }

  static String bits(Bits bits) {
    StringBuilder b = new StringBuilder();
    for (int i = 0; i < bits.length(); i++) {
      if (bits.get(i)) {
        if (b.length() > 0) b.append(',');
        b.append(i);
      }
    }
    return b.toString();
  }

  static String searchGroups(Collection<? extends SearchGroup<?>> groups) {
    if (groups == null) return "null";
    StringBuilder b = new StringBuilder();
    for (SearchGroup<?> g : groups) {
      if (b.length() > 0) b.append(' ');
      b.append(val(g.groupValue)).append('=').append(vals(g.sortValues));
    }
    return b.toString();
  }

  interface Answer {
    String get() throws Exception;
  }

  void emit(String kind, String spec, Answer answer) {
    String a;
    try {
      a = answer.get();
    } catch (Exception e) {
      a = "ERR " + e.getClass().getSimpleName();
    }
    out.append(kind).append('\t').append(spec).append('\t').append(a).append('\n');
  }

  int small(int bound) {
    return r.nextInt(bound);
  }

  // ---------------------------------------------------------------------------------------------
  // Searches
  // ---------------------------------------------------------------------------------------------

  void run() {
    for (int i = 0; i < 260; i++) groupingSearch();
    for (int i = 0; i < 70; i++) managers();
    for (int i = 0; i < 90; i++) block();
    for (int i = 0; i < 70; i++) facet();
    errors();
  }

  @SuppressWarnings({"unchecked", "rawtypes"})
  void groupingSearch() {
    Q q = query();
    S gs = sort(), ws = sort();
    boolean scoreSel = gs.spec.equals("REL");
    Sel sel = selector(scoreSel);
    int go = small(3), gl = 1 + small(6), gdo = small(3), gdl = 1 + small(4);
    boolean ms = sel.spec.startsWith("dscore") || r.nextInt(4) != 0;
    boolean ag = r.nextBoolean(), ah = r.nextBoolean(), ign = r.nextInt(4) == 0;
    String cache = "none";
    if (!sel.spec.startsWith("dscore")) {
      cache =
          switch (r.nextInt(5)) {
            // Without cached scores Lucene's replay hands the second pass no
            // scorer, and a range selector then reads the first pass's last
            // segment's values: only a term selector is cached that way.
            case 0 ->
                "docs:"
                    + (small(2) == 0 ? 5 : 100000)
                    + ":"
                    + (r.nextBoolean() || !sel.spec.startsWith("term") ? 1 : 0);
            case 1 -> "mb:4.0:1";
            default -> "none";
          };
    }
    String spec =
        "q="
            + q.spec
            + ";sel="
            + sel.spec
            + ";gs="
            + gs.spec
            + ";ws="
            + ws.spec
            + ";go="
            + go
            + ";gl="
            + gl
            + ";gdo="
            + gdo
            + ";gdl="
            + gdl
            + ";ms="
            + (ms ? 1 : 0)
            + ";ag="
            + (ag ? 1 : 0)
            + ";ah="
            + (ah ? 1 : 0)
            + ";ign="
            + (ign ? 1 : 0)
            + ";cache="
            + cache;
    String c = cache;
    emit(
        "gs",
        spec,
        () -> {
          GroupingSearch g = new GroupingSearch(sel.make.get());
          g.setGroupSort(gs.sort);
          g.setSortWithinGroup(ws.sort);
          g.setGroupDocsOffset(gdo);
          g.setGroupDocsLimit(gdl);
          g.setIncludeMaxScore(ms);
          g.setAllGroups(ag);
          g.setAllGroupHeads(ah);
          g.setIgnoreDocsWithoutGroupField(ign);
          String[] cp = c.split(":");
          if (cp[0].equals("docs")) g.setCaching(Integer.parseInt(cp[1]), cp[2].equals("1"));
          if (cp[0].equals("mb")) g.setCachingInMB(Double.parseDouble(cp[1]), cp[2].equals("1"));
          TopGroups<?> tg = g.search(s, q.query, go, gl);
          return topGroups(tg)
              + " #groups="
              + sortedVals((Collection) g.getAllMatchingGroups())
              + " #heads="
              + bits(g.getAllGroupHeads());
        });
  }

  @SuppressWarnings({"unchecked", "rawtypes"})
  void managers() {
    Q q = query();
    S gs = sort(), ws = sort();
    Sel sel = selector(false);
    int go = small(3), gl = 1 + small(6), gdo = small(3), gdl = 1 + small(4);
    boolean ms = r.nextBoolean(), ign = r.nextInt(4) == 0;
    TopGroups.ScoreMergeMode smm = TopGroups.ScoreMergeMode.values()[small(3)];
    for (int slicedIdx = 0; slicedIdx < 2; slicedIdx++) {
      IndexSearcher searcher = slicedIdx == 0 ? s : sliced;
      String spec =
          "q="
              + q.spec
              + ";sel="
              + sel.spec
              + ";gs="
              + gs.spec
              + ";ws="
              + ws.spec
              + ";go="
              + go
              + ";gl="
              + gl
              + ";gdo="
              + gdo
              + ";gdl="
              + gdl
              + ";ms="
              + (ms ? 1 : 0)
              + ";ign="
              + (ign ? 1 : 0)
              + ";smm="
              + smm
              + ";sliced="
              + slicedIdx;
      emit(
          "mgr",
          spec,
          () -> {
            StringBuilder b = new StringBuilder();
            Collection<SearchGroup> first =
                (Collection<SearchGroup>)
                    searcher.search(
                    q.query,
                    new FirstPassGroupingCollectorManager(
                        (Supplier) sel.make, gs.sort, go, gl, ign));
            b.append("first=").append(searchGroups((Collection) first));
            if (!first.isEmpty()) {
              TopGroups<?> tg =
                  (TopGroups<?>)
                      searcher.search(
                      q.query,
                      new TopGroupsCollectorManager(
                          (Supplier) sel.make, first, gs.sort, ws.sort, gdo, gdl, ms, smm));
              b.append(" #top=").append(topGroups(tg));
              List<DistinctValuesCollector.GroupCount> distinct =
                  (List)
                      searcher.search(
                          q.query,
                          new DistinctValuesCollectorManager(
                              (Supplier) sel.make,
                              first,
                              (Supplier<GroupSelector<BytesRef>>) () -> new TermGroupSelector("g2")));
              b.append(" #distinct=");
              for (DistinctValuesCollector.GroupCount gc : distinct) {
                b.append(val(gc.groupValue)).append('{').append(sortedVals(gc.uniqueValues)).append('}');
              }
            }
            // A range selector is never handed a scorer by these two
            // collectors, so it has no values: Lucene's NullPointerException.
            try {
              Collection<?> all =
                  (Collection<?>)
                      searcher.search(q.query, new AllGroupsCollectorManager((Supplier) sel.make));
              b.append(" #all=").append(sortedVals(all));
            } catch (NullPointerException e) {
              b.append(" #all=ERR");
            }
            try {
              AllGroupHeadsCollectorManager.GroupHeadsResult heads =
                  (AllGroupHeadsCollectorManager.GroupHeadsResult)
                      searcher.search(
                          q.query, new AllGroupHeadsCollectorManager((Supplier) sel.make, ws.sort));
              int[] h = heads.retrieveGroupHeads().clone();
              java.util.Arrays.sort(h);
              b.append(" #heads=").append(java.util.Arrays.toString(h).replace(" ", ""));
            } catch (NullPointerException e) {
              b.append(" #heads=ERR");
            }
            return b.toString();
          });
    }
  }

  void block() {
    Q q = query();
    S gs = sort(), ws = sort();
    int go = small(3), gl = 1 + small(6), gdo = small(3), gdl = 1 + small(4);
    for (int slicedIdx = 0; slicedIdx < 2; slicedIdx++) {
      IndexSearcher searcher = slicedIdx == 0 ? bs : bsliced;
      String spec =
          "q="
              + q.spec
              + ";gs="
              + gs.spec
              + ";ws="
              + ws.spec
              + ";go="
              + go
              + ";gl="
              + gl
              + ";gdo="
              + gdo
              + ";gdl="
              + gdl
              + ";sliced="
              + slicedIdx;
      emit(
          "block",
          spec,
          () -> {
            GroupingSearch g = new GroupingSearch(new TermQuery(new Term("end", "x")));
            g.setGroupSort(gs.sort);
            g.setSortWithinGroup(ws.sort);
            g.setGroupDocsOffset(gdo);
            g.setGroupDocsLimit(gdl);
            return topGroups(g.search(searcher, q.query, go, gl));
          });
    }
  }

  void facet() {
    Q q = query();
    boolean mv = r.nextBoolean();
    String field = mv ? "fmv" : "fsv";
    String prefix = switch (small(4)) {
      case 0 -> "a";
      case 1 -> "b1";
      case 2 -> "c";
      default -> "-";
    };
    String groupField = r.nextInt(4) == 0 ? "g2" : "g";
    int size = 1 + small(10), minCount = small(3), offset = small(3), limit = 1 + small(8);
    boolean byCount = r.nextBoolean();
    facet(q, groupField, field, mv, prefix, size, minCount, byCount, offset, limit);
  }

  void facet(
      Q q,
      String groupField,
      String field,
      boolean mv,
      String prefix,
      int size,
      int minCount,
      boolean byCount,
      int offset,
      int limit) {
    String spec =
        "q="
            + q.spec
            + ";group="
            + groupField
            + ";field="
            + field
            + ";mv="
            + (mv ? 1 : 0)
            + ";prefix="
            + prefix
            + ";size="
            + size
            + ";min="
            + minCount
            + ";bycount="
            + (byCount ? 1 : 0)
            + ";offset="
            + offset
            + ";limit="
            + limit;
    emit(
        "facet",
        spec,
        () -> {
          TermGroupFacetCollector c =
              TermGroupFacetCollector.createTermGroupFacetCollector(
                  groupField, field, mv, prefix.equals("-") ? null : new BytesRef(prefix), 128);
          s.search(q.query, c);
          GroupFacetCollector.GroupedFacetResult res = c.mergeSegmentResults(size, minCount, byCount);
          StringBuilder b = new StringBuilder();
          b.append("total=").append(res.getTotalCount());
          b.append(" missing=").append(res.getTotalMissingCount());
          for (GroupFacetCollector.FacetEntry e : res.getFacetEntries(offset, limit)) {
            b.append(' ').append(e.value().utf8ToString()).append(':').append(e.count());
          }
          return b.toString();
        });
  }

  void errors() {
    Q all = new Q("all", new MatchAllDocsQuery());
    // Facets over a field no document has, and prefixes past every term.
    for (boolean mv : new boolean[] {true, false}) {
      for (String prefix : new String[] {"-", "a", "zz"}) {
        facet(all, "g", "nofield", mv, prefix, 5, 0, true, 0, 5);
        facet(all, "g", mv ? "fmv" : "fsv", mv, prefix.equals("a") ? "b3" : prefix, 5, 0, false, 0, 9);
      }
    }
    // A group field with doc values of another type.
    emit(
        "gs",
        "q=all;sel=term:fmv;gs=REL;ws=REL;go=0;gl=3;gdo=0;gdl=1;ms=1;ag=0;ah=0;ign=0;cache=none",
        () -> topGroups(new GroupingSearch("fmv").search(s, all.query, 0, 3)));
    // No groups to return: fewer groups than the offset.
    emit(
        "gs",
        "q=all;sel=term:g;gs=REL;ws=REL;go=40;gl=3;gdo=0;gdl=1;ms=1;ag=1;ah=1;ign=0;cache=none",
        () -> {
          GroupingSearch g = new GroupingSearch("g");
          g.setAllGroups(true);
          g.setAllGroupHeads(true);
          TopGroups<?> tg = g.search(s, all.query, 40, 3);
          return topGroups(tg)
              + " #groups="
              + sortedVals(g.getAllMatchingGroups())
              + " #heads="
              + bits(g.getAllGroupHeads());
        });
    // A cache without scores replayed into a second pass that reads them.
    emit(
        "gs",
        "q=t:red;sel=term:g;gs=REL;ws=REL;go=0;gl=3;gdo=0;gdl=2;ms=1;ag=0;ah=0;ign=0;cache=docs:100000:0",
        () -> {
          GroupingSearch g = new GroupingSearch("g");
          g.setCaching(100000, false);
          g.setGroupDocsLimit(2);
          return topGroups(g.search(s, new TermQuery(new Term("body", "red")), 0, 3));
        });
    // A group limit of zero, and a within-group limit of zero.
    emit(
        "gs",
        "q=all;sel=term:g;gs=REL;ws=REL;go=0;gl=0;gdo=0;gdl=1;ms=1;ag=0;ah=0;ign=0;cache=none",
        () -> topGroups(new GroupingSearch("g").search(s, all.query, 0, 0)));
    emit(
        "gs",
        "q=all;sel=term:g;gs=REL;ws=IDX;go=0;gl=3;gdo=0;gdl=0;ms=1;ag=0;ah=0;ign=0;cache=none",
        () -> {
          GroupingSearch g = new GroupingSearch("g");
          g.setSortWithinGroup(Sort.INDEXORDER);
          g.setGroupDocsLimit(0);
          return topGroups(g.search(s, all.query, 0, 3));
        });
    // Blocks sorted by relevance within the group without scores.
    emit(
        "block",
        "q=all;gs=IDX;ws=score;go=0;gl=3;gdo=0;gdl=2;sliced=0",
        () -> {
          GroupingSearch g = new GroupingSearch(new TermQuery(new Term("end", "x")));
          g.setGroupSort(Sort.INDEXORDER);
          g.setSortWithinGroup(new Sort(new SortField(null, SortField.Type.SCORE)));
          g.setGroupDocsLimit(2);
          return topGroups(g.search(bs, all.query, 0, 3));
        });
  }
}
