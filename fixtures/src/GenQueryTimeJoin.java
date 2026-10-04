import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Random;
import java.util.function.Supplier;
import java.util.stream.Stream;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.DoublePoint;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.FloatPoint;
import org.apache.lucene.document.IntPoint;
import org.apache.lucene.document.LongPoint;
import org.apache.lucene.document.NumericDocValuesField;
import org.apache.lucene.document.SortedDocValuesField;
import org.apache.lucene.document.SortedNumericDocValuesField;
import org.apache.lucene.document.SortedSetDocValuesField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.DocValues;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.OrdinalMap;
import org.apache.lucene.index.SortedDocValues;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.BoostQuery;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.MatchAllDocsQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.join.JoinUtil;
import org.apache.lucene.search.join.ScoreMode;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.BytesRef;
import org.apache.lucene.util.packed.PackedInts;

/**
 * M10 T10.3's query-time join fixture: {@code query_time_join/index}, a four-segment index of
 * "from" documents ({@code type:from}) that reference "to" documents ({@code type:to}) by key,
 * with deletions in two segments.
 *
 * <p>Keys: a to document has the indexed key {@code pk} (one term), {@code pkm} (one to three
 * terms), the points {@code npl}/{@code npi}/{@code npf}/{@code npd} (long, int, float, double,
 * one or two values) and the {@code SORTED} join field {@code gj} (its own key). A from document
 * has {@code fk} ({@code SORTED}, sometimes missing), {@code fkm} ({@code SORTED_SET}, zero to
 * three values), {@code nfk}/{@code nfkm} ({@code NUMERIC}/{@code SORTED_NUMERIC} longs),
 * {@code nff}/{@code nfd} (a float's or double's raw bits) and {@code gj} (the key it
 * references). Keys repeat across documents, so a join value has several from documents and
 * several to documents.
 *
 * <p>{@code searches.tsv}: {@code JoinUtil.createJoinQuery} in every form -- terms (single- and
 * multi-valued, every score mode), numeric (every type, single- and multi-valued, every score
 * mode, including the long orders Lucene refuses as points), and global ordinals (every score
 * mode, with and without min/max) -- each searched alone ({@code all}: every hit and its score
 * bits) and inside a boolean with a filter and a boost ({@code bool}), or the exception Lucene
 * threw. The query grammar is the Rust test's ({@code query_time_join_fixtures.rs}).
 *
 * <p>Usage: {@code java GenQueryTimeJoin <fixtures-data-dir>}.
 */
public class GenQueryTimeJoin {
  static final String[] WORDS = {"red", "blue", "green", "fast", "slow", "big", "small", "old"};
  static final int SEGMENTS = 4;
  static final int KEYS = 40;

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

  static String key(int k) {
    return "k" + k;
  }

  /** A key's float: one negative, so long order and point order disagree for a few queries. */
  static float keyFloat(int k) {
    return (k == 37) ? -k / 4f : k / 4f;
  }

  static double keyDouble(int k) {
    return (k == 31) ? -k / 3.0 : k / 3.0;
  }

  static Document to(Random r, int id) {
    Document d = new Document();
    int k = r.nextInt(KEYS);
    d.add(new StringField("id", "t" + id, Field.Store.YES));
    d.add(new StringField("type", "to", Field.Store.NO));
    d.add(new TextField("body", body(r), Field.Store.NO));
    d.add(new StringField("pk", key(k), Field.Store.NO));
    int extra = r.nextInt(3);
    d.add(new StringField("pkm", key(k), Field.Store.NO));
    for (int i = 0; i < extra; i++) {
      d.add(new StringField("pkm", key(r.nextInt(KEYS)), Field.Store.NO));
    }
    d.add(new LongPoint("npl", k));
    d.add(new IntPoint("npi", k));
    d.add(new FloatPoint("npf", keyFloat(k)));
    d.add(new DoublePoint("npd", keyDouble(k)));
    if (r.nextInt(3) == 0) {
      int k2 = r.nextInt(KEYS);
      d.add(new LongPoint("npl", k2));
      d.add(new IntPoint("npi", k2));
      d.add(new FloatPoint("npf", keyFloat(k2)));
      d.add(new DoublePoint("npd", keyDouble(k2)));
    }
    if (r.nextInt(8) != 0) {
      d.add(new SortedDocValuesField("gj", new BytesRef(key(k))));
    }
    return d;
  }

  static Document from(Random r, int id) {
    Document d = new Document();
    int k = r.nextInt(KEYS);
    d.add(new StringField("id", "f" + id, Field.Store.YES));
    d.add(new StringField("type", "from", Field.Store.NO));
    d.add(new TextField("body", body(r), Field.Store.NO));
    if (r.nextInt(6) != 0) {
      d.add(new SortedDocValuesField("fk", new BytesRef(key(k))));
      d.add(new NumericDocValuesField("nfk", k));
      d.add(new NumericDocValuesField("nff", Float.floatToRawIntBits(keyFloat(k))));
      d.add(new NumericDocValuesField("nfd", Double.doubleToRawLongBits(keyDouble(k))));
      d.add(new SortedDocValuesField("gj", new BytesRef(key(k))));
    }
    int n = r.nextInt(4);
    for (int i = 0; i < n; i++) {
      int km = r.nextInt(KEYS);
      d.add(new SortedSetDocValuesField("fkm", new BytesRef(key(km))));
      d.add(new SortedNumericDocValuesField("nfkm", km));
    }
    return d;
  }

  public static void main(String[] args) throws IOException {
    Path root = Path.of(args[0]).resolve("query_time_join");
    clean(root);
    Path indexDir = root.resolve("index");
    Files.createDirectories(indexDir);
    Random r = new Random(0x10_1_2026_1004L);
    try (Directory dir = FSDirectory.open(indexDir)) {
      IndexWriterConfig cfg = new IndexWriterConfig(new StandardAnalyzer());
      cfg.setUseCompoundFile(false);
      cfg.setMergePolicy(NoMergePolicy.INSTANCE);
      cfg.setMaxBufferedDocs(IndexWriterConfig.DISABLE_AUTO_FLUSH);
      cfg.setRAMBufferSizeMB(256);
      try (IndexWriter w = new IndexWriter(dir, cfg)) {
        int id = 0;
        int[] sizes = {120, 90, 100, 25};
        for (int seg = 0; seg < SEGMENTS; seg++) {
          for (int i = 0; i < sizes[seg]; i++, id++) {
            w.addDocument(r.nextInt(5) < 2 ? to(r, id) : from(r, id));
          }
          w.commit();
          if (seg == 1 || seg == 2) {
            for (int k = 0; k < 12; k++) {
              w.deleteDocuments(new Term("id", (r.nextBoolean() ? "t" : "f") + (id - 1 - r.nextInt(sizes[seg]))));
            }
            w.commit();
          }
        }
      }
    }

    StringBuilder out = new StringBuilder();
    try (Directory dir = FSDirectory.open(indexDir);
        DirectoryReader reader = DirectoryReader.open(dir)) {
      if (reader.leaves().size() != SEGMENTS) throw new AssertionError("segments");
      IndexSearcher s = new IndexSearcher(reader);
      s.setQueryCache(null);
      new GenQueryTimeJoin(r, s, reader, out).run();
    }
    Files.writeString(root.resolve("searches.tsv"), out.toString(), StandardCharsets.UTF_8);
  }

  record Q(String spec, Query query) {}

  final Random r;
  final IndexSearcher s;
  final DirectoryReader reader;
  final StringBuilder out;
  final OrdinalMap gjMap;

  GenQueryTimeJoin(Random r, IndexSearcher s, DirectoryReader reader, StringBuilder out)
      throws IOException {
    this.r = r;
    this.s = s;
    this.reader = reader;
    this.out = out;
    SortedDocValues[] values = new SortedDocValues[reader.leaves().size()];
    for (LeafReaderContext leaf : reader.leaves()) {
      values[leaf.ord] = DocValues.getSorted(leaf.reader(), "gj");
    }
    gjMap = OrdinalMap.build(null, values, PackedInts.DEFAULT);
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

  /** A from-side query: some body words over the from documents. */
  Q fromQuery() {
    List<String> occ = new ArrayList<>();
    List<Q> cl = new ArrayList<>();
    occ.add("filter");
    cl.add(term("type", "from"));
    switch (r.nextInt(4)) {
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
      default -> {}
    }
    return bool(occ, cl);
  }

  Q toQuery() {
    return switch (r.nextInt(3)) {
      case 0 -> term("type", "to");
      case 1 -> bool(List.of("filter", "should"), List.of(term("type", "to"), term("body", word(r))));
      default -> new Q("all()", new MatchAllDocsQuery());
    };
  }

  void emit(String kind, String spec, Supplier<String> answer) {
    String a;
    try {
      a = answer.get();
    } catch (RuntimeException e) {
      Throwable c = e.getCause() != null && e instanceof Wrapped ? e.getCause() : e;
      a = "ERR " + c.getClass().getSimpleName();
    }
    out.append(kind).append('\t').append(spec).append('\t').append(a).append('\n');
  }

  static final class Wrapped extends RuntimeException {
    Wrapped(Exception e) {
      super(e);
    }
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
      throw new Wrapped(e);
    }
  }

  interface JoinBuilder {
    Query build() throws IOException;
  }

  /** The join alone, every hit; and boosted inside a boolean with a filter, the top ten. */
  void search(String spec, JoinBuilder join) {
    Query[] built = new Query[1];
    emit(
        "all",
        spec,
        () -> {
          try {
            built[0] = join.build();
          } catch (IOException e) {
            throw new Wrapped(e);
          }
          return hits(built[0], 100_000);
        });
    if (built[0] == null) return;
    String word = word(r);
    Query wrapped =
        new BooleanQuery.Builder()
            .add(new BoostQuery(built[0], 2f), BooleanClause.Occur.MUST)
            .add(new TermQuery(new Term("type", "to")), BooleanClause.Occur.FILTER)
            .add(new TermQuery(new Term("body", word)), BooleanClause.Occur.SHOULD)
            .build();
    emit("bool", spec + "|" + word, () -> hits(wrapped, 10));
  }

  void run() {
    ScoreMode[] modes = ScoreMode.values();
    // Terms joins: from a SORTED or SORTED_SET field to a single- or multi-term field.
    for (int i = 0; i < 12; i++) {
      for (ScoreMode mode : modes) {
        for (boolean mv : new boolean[] {false, true}) {
          String fromField = mv ? "fkm" : "fk";
          String toField = r.nextBoolean() ? "pk" : "pkm";
          Q fq = fromQuery();
          String spec = "terms(" + fromField + "," + mv + "," + toField + "," + mode + "," + fq.spec() + ")";
          search(spec, () -> JoinUtil.createJoinQuery(fromField, mv, toField, fq.query(), s, mode));
        }
      }
    }
    // Numeric joins: every type, single- and multi-valued.
    String[][] types = {
      {"Long", "npl", "nfk", "nfkm"},
      {"Integer", "npi", "nfk", "nfkm"},
      {"Float", "npf", "nff", "nff"},
      {"Double", "npd", "nfd", "nfd"}
    };
    for (int i = 0; i < 8; i++) {
      for (ScoreMode mode : modes) {
        for (String[] t : types) {
          boolean mv = r.nextBoolean();
          String fromField = mv ? t[3] : t[2];
          Class<? extends Number> type =
              switch (t[0]) {
                case "Long" -> Long.class;
                case "Integer" -> Integer.class;
                case "Float" -> Float.class;
                default -> Double.class;
              };
          Q fq = fromQuery();
          String spec =
              "num(" + fromField + "," + mv + "," + t[1] + "," + t[0] + "," + mode + "," + fq.spec() + ")";
          search(
              spec,
              () -> JoinUtil.createJoinQuery(fromField, mv, t[1], type, fq.query(), s, mode));
        }
      }
    }
    // Global ordinals: every mode, with and without bounds.
    int[][] bounds = {{0, Integer.MAX_VALUE}, {2, Integer.MAX_VALUE}, {0, 2}, {2, 3}, {1, 1}};
    for (int i = 0; i < 6; i++) {
      for (ScoreMode mode : modes) {
        for (int[] b : bounds) {
          Q fq = fromQuery();
          Q tq = toQuery();
          String spec =
              "gord(" + mode + "," + b[0] + "," + b[1] + "," + fq.spec() + "," + tq.spec() + ")";
          search(
              spec,
              () -> JoinUtil.createJoinQuery("gj", fq.query(), tq.query(), s, mode, gjMap, b[0], b[1]));
        }
      }
    }
    // Global ordinals over a single-segment searcher (the first segment): no map, the
    // segment's own ordinals.
    IndexSearcher one = new IndexSearcher(reader.leaves().get(0).reader());
    one.setQueryCache(null);
    IndexSearcher all = s;
    for (int i = 0; i < 5; i++) {
      for (ScoreMode mode : modes) {
        int[] b = bounds[r.nextInt(bounds.length)];
        Q fq = fromQuery();
        Q tq = toQuery();
        String spec =
            "gord1(" + mode + "," + b[0] + "," + b[1] + "," + fq.spec() + "," + tq.spec() + ")";
        String a;
        try {
          Query q = JoinUtil.createJoinQuery("gj", fq.query(), tq.query(), one, mode, null, b[0], b[1]);
          TopDocs top = one.search(q, 100_000);
          StringBuilder sb = new StringBuilder();
          for (ScoreDoc sd : top.scoreDocs) {
            if (sb.length() > 0) sb.append(' ');
            sb.append(sd.doc).append(':').append(Integer.toHexString(Float.floatToIntBits(sd.score)));
          }
          a = sb.length() == 0 ? "-" : sb.toString();
        } catch (IOException | RuntimeException e) {
          a = "ERR " + e.getClass().getSimpleName();
        }
        out.append("one\t").append(spec).append('\t').append(a).append('\n');
      }
    }
    // Errors: a global-ordinals join over several segments needs a map.
    search(
        "gord-nomap(None)",
        () ->
            JoinUtil.createJoinQuery(
                "gj", new TermQuery(new Term("type", "from")), new MatchAllDocsQuery(), s, ScoreMode.None, null));
    // Join values that encode out of order as points: a double's raw bits cast to `int`.
    for (ScoreMode mode : new ScoreMode[] {ScoreMode.None, ScoreMode.Total}) {
      search(
          "num(nfd,false,npi,Integer," + mode + ",t(type,from))",
          () ->
              JoinUtil.createJoinQuery(
                  "nfd", false, "npi", Integer.class, new TermQuery(new Term("type", "from")), s, mode));
    }
    // Every from document: the negative float and double keys too.
    for (ScoreMode mode : new ScoreMode[] {ScoreMode.None, ScoreMode.Max}) {
      search(
          "num(nff,false,npf,Float," + mode + ",t(type,from))",
          () ->
              JoinUtil.createJoinQuery(
                  "nff", false, "npf", Float.class, new TermQuery(new Term("type", "from")), s, mode));
      search(
          "num(nfd,true,npd,Double," + mode + ",t(type,from))",
          () ->
              JoinUtil.createJoinQuery(
                  "nfd", true, "npd", Double.class, new TermQuery(new Term("type", "from")), s, mode));
    }
    // A from field with doc values of another kind.
    search(
        "terms(nfk,false,pk,None,t(type,from))",
        () ->
            JoinUtil.createJoinQuery(
                "nfk", false, "pk", new TermQuery(new Term("type", "from")), s, ScoreMode.None));
  }
}
