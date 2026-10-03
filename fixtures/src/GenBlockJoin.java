import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;
import java.util.Random;
import java.util.function.Supplier;
import java.util.stream.Stream;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.KnnByteVectorField;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.NumericDocValuesField;
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
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.BoostQuery;
import org.apache.lucene.search.ConstantScoreQuery;
import org.apache.lucene.search.FieldDoc;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.MatchAllDocsQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.Sort;
import org.apache.lucene.search.SortField;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TopFieldDocs;
import org.apache.lucene.search.join.BitSetProducer;
import org.apache.lucene.search.join.CheckJoinIndex;
import org.apache.lucene.search.join.DiversifyingChildrenByteKnnVectorQuery;
import org.apache.lucene.search.join.DiversifyingChildrenFloatKnnVectorQuery;
import org.apache.lucene.search.join.ParentChildrenBlockJoinQuery;
import org.apache.lucene.search.join.ParentsChildrenBlockJoinQuery;
import org.apache.lucene.search.join.QueryBitSetProducer;
import org.apache.lucene.search.join.ScoreMode;
import org.apache.lucene.search.join.ToChildBlockJoinQuery;
import org.apache.lucene.search.join.ToParentBlockJoinQuery;
import org.apache.lucene.search.join.ToParentBlockJoinSortField;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.BytesRef;
import org.apache.lucene.util.NumericUtils;

/**
 * M10 T10.2's block-join fixture: {@code block_join/index}, a four-segment index of two-level
 * document blocks -- grandchildren ({@code type:grand}) closed by their child ({@code
 * type:child}), children closed by their parent ({@code type:parent}) -- with empty and
 * single-child blocks, written with the parent field {@code _parent}; then deletions: whole blocks
 * in two segments, and lone children, grandchildren and parents in a third (a block index
 * {@code CheckJoinIndex} rejects, which the queries must still answer as Lucene answers it).
 *
 * <p>{@code searches.tsv}: random block-join queries -- {@code ToParentBlockJoinQuery} in every
 * score mode, {@code ToChildBlockJoinQuery}, {@code ParentChildrenBlockJoinQuery}, {@code
 * ParentsChildrenBlockJoinQuery}, two levels deep, alone and inside booleans, boosted and
 * constant-scored -- each with every hit and its score bits ({@code all}, a search of more hits
 * than the index has), the top ten ({@code top}), or the exception Lucene threw; and searches
 * sorted by a {@code ToParentBlockJoinSortField} of every type with the sort values ({@code
 * sort}); and {@code DiversifyingChildrenFloatKnnVectorQuery} / {@code
 * DiversifyingChildrenByteKnnVectorQuery} searches over the children's vectors, with and without
 * a child filter, each with its child hits and score bits ({@code knn}). The query grammar is the
 * Rust test's ({@code block_join_fixtures.rs}).
 *
 * <p>Usage: {@code java GenBlockJoin <fixtures-data-dir>}.
 */
public class GenBlockJoin {
  static final String[] WORDS = {
    "red", "blue", "green", "fast", "slow", "big", "small", "shiny", "old", "new", "cheap", "rare"
  };
  static final int SEGMENTS = 4;
  static final int[] BLOCKS = {150, 150, 150, 30};

  static void clean(Path root) throws IOException {
    if (Files.exists(root)) {
      try (Stream<Path> walk = Files.walk(root)) {
        for (Path p : walk.sorted(Comparator.reverseOrder()).toList()) Files.delete(p);
      }
    }
  }

  /** A skewed word: the first words far more often, so frequencies vary. */
  static String word(Random r) {
    int i = (int) (Math.pow(r.nextDouble(), 1.7) * WORDS.length);
    return WORDS[Math.min(i, WORDS.length - 1)];
  }

  static String body(Random r) {
    int n = 1 + r.nextInt(7);
    StringBuilder b = new StringBuilder();
    for (int i = 0; i < n; i++) {
      if (i > 0) b.append(' ');
      b.append(word(r));
    }
    return b.toString();
  }

  /** The doc-values a child or grandchild sorts by: zero to two values of each. */
  static void values(Random r, Document d) {
    int n = r.nextInt(4) == 0 ? 0 : 1 + r.nextInt(2);
    for (int i = 0; i < n; i++) {
      long l = r.nextInt(2000) - 500;
      d.add(new SortedNumericDocValuesField("lp", l));
      d.add(new SortedNumericDocValuesField("ip", r.nextInt(200) - 100));
      d.add(
          new SortedNumericDocValuesField(
              "fp", NumericUtils.floatToSortableInt((r.nextInt(4000) - 2000) / 7f)));
      d.add(
          new SortedNumericDocValuesField(
              "dp", NumericUtils.doubleToSortableLong((r.nextInt(4000) - 2000) / 3.0)));
      d.add(new SortedSetDocValuesField("color", new BytesRef(word(r))));
    }
  }

  static Document doc(String id, int block, String type, Random r) {
    Document d = new Document();
    d.add(new StringField("id", id, Field.Store.YES));
    d.add(new StringField("bid", Integer.toString(block), Field.Store.NO));
    d.add(new StringField("type", type, Field.Store.NO));
    d.add(new TextField("body", body(r), Field.Store.NO));
    return d;
  }

  public static void main(String[] args) throws IOException {
    Path root = Path.of(args[0]).resolve("block_join");
    clean(root);
    Path indexDir = root.resolve("index");
    Files.createDirectories(indexDir);
    Random r = new Random(0xB10C_2026_1003L);
    try (Directory dir = FSDirectory.open(indexDir)) {
      IndexWriterConfig cfg = new IndexWriterConfig(new StandardAnalyzer());
      cfg.setUseCompoundFile(false);
      cfg.setMergePolicy(NoMergePolicy.INSTANCE);
      cfg.setMaxBufferedDocs(IndexWriterConfig.DISABLE_AUTO_FLUSH);
      cfg.setRAMBufferSizeMB(256);
      cfg.setParentField("_parent");
      try (IndexWriter w = new IndexWriter(dir, cfg)) {
        int block = 0;
        List<String> loneDeletes = new ArrayList<>();
        for (int seg = 0; seg < SEGMENTS; seg++) {
          List<Integer> segBlocks = new ArrayList<>();
          for (int i = 0; i < BLOCKS[seg]; i++, block++) {
            segBlocks.add(block);
            List<Document> docs = new ArrayList<>();
            int roll = r.nextInt(100);
            int children = roll < 15 ? 0 : roll < 40 ? 1 : 2 + r.nextInt(3);
            for (int c = 0; c < children; c++) {
              int grand = r.nextInt(2) == 0 ? 0 : 1 + r.nextInt(2);
              for (int g = 0; g < grand; g++) {
                Document gd = doc("g" + block + "." + c + "." + g, block, "grand", r);
                values(r, gd);
                docs.add(gd);
                if (seg == 2 && r.nextInt(12) == 0) loneDeletes.add("g" + block + "." + c + "." + g);
              }
              Document cd = doc("c" + block + "." + c, block, "child", r);
              values(r, cd);
              float[] v = new float[4];
              byte[] bv = new byte[4];
              for (int k = 0; k < 4; k++) {
                v[k] = (r.nextInt(2000) - 1000) / 250f;
                bv[k] = (byte) (r.nextInt(200) - 100);
              }
              cd.add(new KnnFloatVectorField("fvec", v, VectorSimilarityFunction.EUCLIDEAN));
              cd.add(new KnnByteVectorField("bvec", bv, VectorSimilarityFunction.EUCLIDEAN));
              docs.add(cd);
              if (seg == 2 && r.nextInt(10) == 0) loneDeletes.add("c" + block + "." + c);
            }
            Document p = doc("p" + block, block, "parent", r);
            p.add(new NumericDocValuesField("rank", r.nextInt(100)));
            docs.add(p);
            if (seg == 2 && r.nextInt(60) == 0) loneDeletes.add("p" + block);
            w.addDocuments(docs);
          }
          w.commit();
          // Whole blocks in segments 1 and 3, by the term all of a block has.
          if (seg == 1 || seg == 3) {
            for (int k = 0; k < (seg == 1 ? 10 : 1); k++) {
              int b = segBlocks.get(r.nextInt(segBlocks.size()));
              w.deleteDocuments(new Term("bid", Integer.toString(b)));
            }
          }
          if (seg == 2) {
            for (String id : loneDeletes) {
              w.deleteDocuments(new Term("id", id));
            }
          }
          w.commit();
        }
      }
    }

    StringBuilder out = new StringBuilder();
    try (Directory dir = FSDirectory.open(indexDir);
        DirectoryReader reader = DirectoryReader.open(dir)) {
      if (reader.leaves().size() != SEGMENTS) throw new AssertionError("segments");
      IndexSearcher s = new IndexSearcher(reader);
      s.setQueryCache(null);
      new GenBlockJoin(r, s, reader, out).run();
    }
    Files.writeString(root.resolve("searches.tsv"), out.toString(), StandardCharsets.UTF_8);
  }

  // ---------------------------------------------------------------------------------------------
  // Queries: each a Lucene query and the spec the Rust test parses into the same query.
  // ---------------------------------------------------------------------------------------------

  record Q(String spec, Query query) {}

  final Random r;
  final IndexSearcher s;
  final DirectoryReader reader;
  final StringBuilder out;
  /** {@code P0}: the parents; {@code P1}: the parents and the children. */
  final BitSetProducer p0 = new QueryBitSetProducer(new TermQuery(new Term("type", "parent")));

  final BitSetProducer p1 =
      new QueryBitSetProducer(
          new BooleanQuery.Builder()
              .add(new TermQuery(new Term("type", "parent")), BooleanClause.Occur.SHOULD)
              .add(new TermQuery(new Term("type", "child")), BooleanClause.Occur.SHOULD)
              .build());

  final BitSetProducer kids = new QueryBitSetProducer(new TermQuery(new Term("type", "child")));
  final BitSetProducer grands = new QueryBitSetProducer(new TermQuery(new Term("type", "grand")));

  GenBlockJoin(Random r, IndexSearcher s, DirectoryReader reader, StringBuilder out) {
    this.r = r;
    this.s = s;
    this.reader = reader;
    this.out = out;
  }

  BitSetProducer filter(String name) {
    return switch (name) {
      case "P0" -> p0;
      case "P1" -> p1;
      case "KIDS" -> kids;
      case "GRANDS" -> grands;
      default -> throw new IllegalArgumentException(name);
    };
  }

  static Q term(String field, String value) {
    return new Q("t(" + field + "," + value + ")", new TermQuery(new Term(field, value)));
  }

  static Q bool(List<String> occurs, List<Q> clauses) {
    BooleanQuery.Builder b = new BooleanQuery.Builder();
    StringBuilder spec = new StringBuilder("bool(");
    for (int i = 0; i < clauses.size(); i++) {
      if (i > 0) spec.append(',');
      String o = occurs.get(i);
      spec.append(o).append(':').append(clauses.get(i).spec());
      b.add(
          clauses.get(i).query(),
          switch (o) {
            case "must" -> BooleanClause.Occur.MUST;
            case "should" -> BooleanClause.Occur.SHOULD;
            case "filter" -> BooleanClause.Occur.FILTER;
            case "not" -> BooleanClause.Occur.MUST_NOT;
            default -> throw new IllegalArgumentException(o);
          });
    }
    return new Q(spec.append(')').toString(), b.build());
  }

  static Q boost(float f, Q q) {
    return new Q(
        "boost(" + Float.toString(f) + "," + q.spec() + ")", new BoostQuery(q.query(), f));
  }

  static Q constant(Q q) {
    return new Q("cs(" + q.spec() + ")", new ConstantScoreQuery(q.query()));
  }

  static final ScoreMode[] MODES = ScoreMode.values();

  Q toParent(ScoreMode mode, String filter, Q child) {
    return new Q(
        "tp(" + mode + "," + filter + "," + child.spec() + ")",
        new ToParentBlockJoinQuery(child.query(), filter(filter), mode));
  }

  Q toChild(String filter, Q parent) {
    return new Q(
        "tc(" + filter + "," + parent.spec() + ")",
        new ToChildBlockJoinQuery(parent.query(), filter(filter)));
  }

  /** A query over one level's documents only: some body words, filtered to the level's type. */
  Q level(String type) {
    List<String> occ = new ArrayList<>();
    List<Q> cl = new ArrayList<>();
    occ.add("filter");
    cl.add(term("type", type));
    int shape = r.nextInt(5);
    switch (shape) {
      case 0 -> {
        occ.add("must");
        cl.add(term("body", word(r)));
      }
      case 1 -> {
        occ.add("should");
        cl.add(term("body", word(r)));
        occ.add("should");
        cl.add(term("body", word(r)));
      }
      case 2 -> {
        occ.add("must");
        cl.add(term("body", word(r)));
        occ.add("not");
        cl.add(term("body", word(r)));
      }
      case 3 -> {
        occ.add("must");
        cl.add(term("body", word(r)));
        occ.add("should");
        cl.add(boost(2.5f, term("body", word(r))));
      }
      default -> {
        // no scoring clause: every document of the level, scoring 0
      }
    }
    return bool(occ, cl);
  }

  /** A child-level query: plain, or itself a join up from the grandchildren. */
  Q children() {
    if (r.nextInt(4) == 0) {
      Q up = toParent(MODES[r.nextInt(MODES.length)], "P1", level("grand"));
      // The grandchildren's parents are children only; restrict to keep that explicit.
      return bool(List.of("must", "filter"), List.of(up, term("type", "child")));
    }
    return level("child");
  }

  void emit(String kind, Q q, Supplier<String> answer) {
    String a;
    try {
      a = answer.get();
    } catch (RuntimeException e) {
      a = "ERR " + e.getClass().getSimpleName();
    }
    out.append(kind).append('\t').append(q.spec()).append('\t').append(a).append('\n');
  }

  String hits(Query q, int n) {
    try {
      TopDocs top = s.search(q, n);
      StringBuilder b = new StringBuilder();
      for (ScoreDoc sd : top.scoreDocs) {
        if (b.length() > 0) b.append(' ');
        b.append(sd.doc).append(':').append(Integer.toHexString(Float.floatToIntBits(sd.score)));
      }
      return b.length() == 0 ? "-" : b.toString();
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  void search(Q q) {
    emit("all", q, () -> hits(q.query(), 100_000));
    emit("top", q, () -> hits(q.query(), 10));
  }

  int randomParentDoc() {
    while (true) {
      LeafReaderContext leaf = reader.leaves().get(r.nextInt(reader.leaves().size()));
      int doc = r.nextInt(leaf.reader().maxDoc());
      try {
        String id = leaf.reader().storedFields().document(doc).get("id");
        if (id.startsWith("p")) return leaf.docBase + doc;
      } catch (IOException e) {
        throw new RuntimeException(e);
      }
    }
  }

  void run() throws IOException {
    for (LeafReaderContext leaf : reader.leaves()) {
      if (leaf.reader().getMetaData().hasBlocks() == false) throw new AssertionError("blocks");
    }
    // The three segments with whole-block deletes or none pass CheckJoinIndex; the one with lone
    // deletes does not, and is meant not to.
    try {
      CheckJoinIndex.check(reader, p0);
      throw new AssertionError("segment 2's lone deletes should fail CheckJoinIndex");
    } catch (IllegalStateException expected) {
      // as intended
    }

    // ToParentBlockJoinQuery, every score mode, top level.
    for (int i = 0; i < 40; i++) {
      for (ScoreMode mode : MODES) {
        search(toParent(mode, "P0", children()));
      }
    }
    // Two levels: grandchildren to children to parents.
    for (int i = 0; i < 25; i++) {
      ScoreMode m1 = MODES[r.nextInt(MODES.length)];
      ScoreMode m2 = MODES[r.nextInt(MODES.length)];
      search(toParent(m2, "P0", toParent(m1, "P1", level("grand"))));
    }
    // Inside booleans (the scorer path, not the bulk one), boosted and constant.
    for (int i = 0; i < 60; i++) {
      Q join = toParent(MODES[r.nextInt(MODES.length)], "P0", children());
      Q other = level("parent");
      Q q =
          switch (r.nextInt(6)) {
            case 0 -> bool(List.of("must", "should"), List.of(join, term("body", word(r))));
            case 1 -> bool(List.of("must", "must"), List.of(join, other));
            case 2 -> bool(List.of("should", "should"), List.of(join, other));
            case 3 -> bool(List.of("filter", "must"), List.of(join, other));
            case 4 -> boost(0.5f + r.nextInt(8) / 2f, join);
            default -> bool(List.of("must", "not"), List.of(constant(join), term("body", word(r))));
          };
      search(q);
    }
    // ToChildBlockJoinQuery, top level and inside booleans, and parents to grandchildren.
    for (int i = 0; i < 60; i++) {
      Q parent = level("parent");
      Q q =
          switch (r.nextInt(5)) {
            case 0, 1 -> toChild("P0", parent);
            case 2 -> bool(List.of("must", "should"), List.of(toChild("P0", parent), term("body", word(r))));
            case 3 -> bool(List.of("filter", "must"), List.of(toChild("P0", parent), level("child")));
            default -> toChild("P1", bool(List.of("must", "filter"), List.of(toChild("P0", parent), term("type", "child"))));
          };
      search(q);
    }
    // Round trips: children of parents of children.
    for (int i = 0; i < 15; i++) {
      search(toChild("P0", toParent(MODES[r.nextInt(MODES.length)], "P0", level("child"))));
    }
    // ParentChildrenBlockJoinQuery over random parents, deleted ones included.
    for (int i = 0; i < 40; i++) {
      int parent = randomParentDoc();
      Q child = level("child");
      Q q =
          new Q(
              "pc(P0," + parent + "," + child.spec() + ")",
              new ParentChildrenBlockJoinQuery(p0, child.query(), parent));
      search(q);
    }
    // ParentsChildrenBlockJoinQuery: matching children of matching parents, limited per parent.
    for (int i = 0; i < 30; i++) {
      Q parent = level("parent");
      Q child = level("child");
      int limit = r.nextInt(3) == 0 ? Integer.MAX_VALUE : 1 + r.nextInt(3);
      Q q =
          new Q(
              "pcs(P0," + limit + "," + parent.spec() + "," + child.spec() + ")",
              new ParentsChildrenBlockJoinQuery(p0, parent.query(), child.query(), limit));
      search(q);
    }
    // Misuse Lucene detects: a child query matching parents, a parent query matching children.
    for (int i = 0; i < 4; i++) {
      search(toParent(MODES[1 + r.nextInt(MODES.length - 1)], "P0", term("body", word(r))));
      search(toChild("P0", term("body", word(r))));
    }
    // Sorted by the children's values.
    String[][] sortFields = {
      {"lp", "LONG"}, {"ip", "INT"}, {"fp", "FLOAT"}, {"dp", "DOUBLE"}, {"color", "STRING"}
    };
    for (int i = 0; i < 60; i++) {
      String[] f = sortFields[i % sortFields.length];
      SortField.Type type = SortField.Type.valueOf(f[1]);
      boolean grand = r.nextInt(4) == 0;
      boolean reverseParents = r.nextBoolean();
      boolean reverseChildren = r.nextInt(3) == 0 ? !reverseParents : reverseParents;
      Object parentMissing = null;
      Object childMissing = null;
      if (r.nextInt(3) == 0) {
        parentMissing = missing(type);
      }
      if (r.nextInt(3) == 0) {
        childMissing = missing(type);
      }
      String parentFilter = grand ? "P1" : "P0";
      String childFilter = grand ? "GRANDS" : "KIDS";
      ToParentBlockJoinSortField sf =
          new ToParentBlockJoinSortField(
              f[0],
              type,
              reverseParents,
              reverseChildren,
              parentMissing,
              childMissing,
              filter(parentFilter),
              filter(childFilter));
      Q q = grand ? level("child") : (r.nextBoolean() ? level("parent") : toParent(ScoreMode.Max, "P0", level("child")));
      String spec =
          "sort("
              + f[0]
              + ","
              + f[1]
              + ","
              + reverseParents
              + ","
              + reverseChildren
              + ","
              + missingSpec(parentMissing)
              + ","
              + missingSpec(childMissing)
              + ","
              + parentFilter
              + ","
              + childFilter
              + ","
              + q.spec()
              + ")";
      emit("sort", new Q(spec, q.query()), () -> sorted(q.query(), new Sort(sf)));
    }
    // DiversifyingChildren{Float,Byte}KnnVectorQuery: the best child per parent, unfiltered (the
    // graph, optimistic per-leaf collectors and the re-entrant second pass) and filtered (small
    // filters take the parent-grouping exact search, larger ones the graph).
    int[] ks = {1, 3, 5, 10, 30};
    for (int i = 0; i < 120; i++) {
      boolean bytes = (i & 1) == 1;
      int k = ks[r.nextInt(ks.length)];
      StringBuilder vec = new StringBuilder();
      float[] fv = new float[4];
      byte[] bv = new byte[4];
      for (int j = 0; j < 4; j++) {
        fv[j] = (r.nextInt(2000) - 1000) / 250f;
        bv[j] = (byte) (r.nextInt(200) - 100);
        if (j > 0) vec.append(';');
        vec.append(bytes ? Integer.toString(bv[j]) : Integer.toHexString(Float.floatToIntBits(fv[j])));
      }
      Q filter =
          switch (r.nextInt(6)) {
            case 0, 1 -> null;
            case 2 -> level("child");
            case 3 -> term("body", word(r));
            default -> blocks(1 + r.nextInt(r.nextBoolean() ? 4 : 40));
          };
      Query filterQuery = filter == null ? null : filter.query();
      Query q =
          bytes
              ? new DiversifyingChildrenByteKnnVectorQuery("bvec", bv, filterQuery, k, p0)
              : new DiversifyingChildrenFloatKnnVectorQuery("fvec", fv, filterQuery, k, p0);
      String spec =
          (bytes ? "dknnb(bvec," : "dknnf(fvec,")
              + k
              + ","
              + vec
              + ",P0,"
              + (filter == null ? "-" : filter.spec())
              + ")";
      emit("knn", new Q(spec, q), () -> hits(q, 1000));
    }
  }

  /** Any of {@code n} random blocks: a filter small enough for the exact search. */
  Q blocks(int n) {
    List<String> occurs = new ArrayList<>();
    List<Q> clauses = new ArrayList<>();
    for (int i = 0; i < n; i++) {
      occurs.add("should");
      clauses.add(term("bid", Integer.toString(r.nextInt(480))));
    }
    return bool(occurs, clauses);
  }

  Object missing(SortField.Type type) {
    return switch (type) {
      case LONG -> (long) (r.nextInt(3000) - 1500);
      case INT -> r.nextInt(300) - 150;
      case FLOAT -> (r.nextInt(6000) - 3000) / 7f;
      case DOUBLE -> (r.nextInt(6000) - 3000) / 3.0;
      case STRING -> r.nextBoolean() ? SortField.STRING_FIRST : SortField.STRING_LAST;
      default -> throw new IllegalArgumentException();
    };
  }

  static String missingSpec(Object m) {
    if (m == null) return "null";
    if (m == SortField.STRING_FIRST) return "first";
    if (m == SortField.STRING_LAST) return "last";
    if (m instanceof Float f) return "f" + Integer.toHexString(Float.floatToIntBits(f));
    if (m instanceof Double d) return "d" + Long.toHexString(Double.doubleToLongBits(d));
    if (m instanceof Integer i) return "i" + i;
    return "l" + m;
  }

  String sorted(Query q, Sort sort) {
    try {
      TopFieldDocs top = s.search(q, 50, sort);
      StringBuilder b = new StringBuilder();
      for (ScoreDoc sd : top.scoreDocs) {
        FieldDoc fd = (FieldDoc) sd;
        if (b.length() > 0) b.append(' ');
        b.append(fd.doc).append(':').append(value(fd.fields[0]));
      }
      return b.length() == 0 ? "-" : b.toString();
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  static String value(Object v) {
    if (v == null) return "null";
    if (v instanceof BytesRef br) return "s" + br.utf8ToString();
    if (v instanceof Float f) return "f" + Integer.toHexString(Float.floatToIntBits(f));
    if (v instanceof Double d) return "d" + Long.toHexString(Double.doubleToLongBits(d));
    if (v instanceof Integer i) return "i" + i;
    if (v instanceof Long l) return "l" + l;
    return String.format(Locale.ROOT, "?%s", v);
  }
}
