import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.stream.Stream;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.CharArraySet;
import org.apache.lucene.analysis.LowerCaseFilter;
import org.apache.lucene.analysis.StopFilter;
import org.apache.lucene.analysis.TokenFilter;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.Tokenizer;
import org.apache.lucene.analysis.miscellaneous.PerFieldAnalyzerWrapper;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.analysis.standard.StandardTokenizer;
import org.apache.lucene.analysis.tokenattributes.PayloadAttribute;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.FieldType;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.ReaderUtil;
import org.apache.lucene.index.Term;
import org.apache.lucene.queries.payloads.AveragePayloadFunction;
import org.apache.lucene.queries.payloads.MaxPayloadFunction;
import org.apache.lucene.queries.payloads.MinPayloadFunction;
import org.apache.lucene.queries.payloads.PayloadDecoder;
import org.apache.lucene.queries.payloads.PayloadFunction;
import org.apache.lucene.queries.payloads.PayloadScoreQuery;
import org.apache.lucene.queries.payloads.SpanPayloadCheckQuery;
import org.apache.lucene.queries.payloads.SumPayloadFunction;
import org.apache.lucene.queries.spans.FieldMaskingSpanQuery;
import org.apache.lucene.queries.spans.SpanContainingQuery;
import org.apache.lucene.queries.spans.SpanFirstQuery;
import org.apache.lucene.queries.spans.SpanMultiTermQueryWrapper;
import org.apache.lucene.queries.spans.SpanNearQuery;
import org.apache.lucene.queries.spans.SpanNotQuery;
import org.apache.lucene.queries.spans.SpanOrQuery;
import org.apache.lucene.queries.spans.SpanPositionRangeQuery;
import org.apache.lucene.queries.spans.SpanQuery;
import org.apache.lucene.queries.spans.SpanTermQuery;
import org.apache.lucene.queries.spans.SpanWithinQuery;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.BoostQuery;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Matches;
import org.apache.lucene.search.MatchesIterator;
import org.apache.lucene.search.MultiTermQuery;
import org.apache.lucene.search.PrefixQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.RegexpQuery;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.ScoreMode;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.search.TermRangeQuery;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.Weight;
import org.apache.lucene.search.WildcardQuery;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.BytesRef;

/**
 * M10 T10.6's spans and payloads fixture: {@code spans/index}, a four-segment index with deletions
 * in two segments, and {@code spans/queries.tsv}, the span queries of {@code lucene-queries}' {@code
 * spans} package and the {@code payloads} package's queries over it.
 *
 * <p>Fields: {@code id} (StringField), {@code body} (positions and offsets, a stop word leaving
 * holes), {@code body2} (the same words reversed, for {@code FieldMaskingSpanQuery}), and four
 * payload fields with the same text: {@code pay} (a one-byte payload, none on every fifth
 * position), {@code ipay} (a big-endian int), {@code fpay} (a big-endian float) and {@code spay} (a
 * short UTF-8 string), the last three with none on every sixth. One word ({@code zeta}) is in a
 * single document (a pulsed singleton term); a few documents have no text at all.
 *
 * <p>For each query spec (the grammar is the Rust test's, {@code spans_fixtures.rs}): its {@code
 * toString}; then, for each variant (plain, boosted, the required clause, the filter or the prohibited clause of a boolean with an
 * optional term), every hit with its score bits and five documents' explanations; then the {@code
 * Matches} of every hit and of the explained documents -- each span's positions and offsets, and its
 * terms as sub-matches with their {@code TermQuery}. An exception is recorded as its class name.
 *
 * <p>Usage: {@code java GenSpans <fixtures-data-dir>}.
 */
public class GenSpans {
  static final String[] WORDS = {
    "apple", "apply", "ape", "bank", "band", "banana", "cat", "car", "cart", "dog", "door", "dot",
    "egg", "eel", "fig", "fish"
  };

  static final CharArraySet STOP = new CharArraySet(List.of("the"), false);

  static final String[] STRINGS = {"a", "b", "ab", "ba", "\u00e9"};

  static void clean(Path root) throws IOException {
    if (Files.exists(root)) {
      try (Stream<Path> walk = Files.walk(root)) {
        for (Path p : walk.sorted(Comparator.reverseOrder()).toList()) Files.delete(p);
      }
    }
  }

  /** The payload of the token at position {@code p} of a field of kind {@code kind}. */
  static BytesRef payload(String kind, int p) {
    switch (kind) {
      case "pay":
        if (p % 5 == 4) return null;
        if (p % 7 == 3) return new BytesRef(new byte[] {(byte) -2});
        return new BytesRef(new byte[] {(byte) (p % 4)});
      case "ipay":
        if (p % 6 == 5) return null;
        return new BytesRef(ByteBuffer.allocate(4).putInt((p * 37) % 11 - 3).array());
      case "fpay":
        if (p % 6 == 5) return null;
        return new BytesRef(ByteBuffer.allocate(4).putFloat((p % 5) * 0.75f - 1f).array());
      case "spay":
        if (p % 6 == 5) return null;
        return new BytesRef(STRINGS[p % 5].getBytes(StandardCharsets.UTF_8));
      default:
        throw new AssertionError(kind);
    }
  }

  /** {@code StandardTokenizer}, lower-cased, {@code the} removed, a payload per position. */
  static final class PayloadAnalyzer extends Analyzer {
    final String kind;

    PayloadAnalyzer(String kind) {
      this.kind = kind;
    }

    @Override
    protected TokenStreamComponents createComponents(String fieldName) {
      Tokenizer t = new StandardTokenizer();
      TokenStream ts = new StopFilter(new LowerCaseFilter(t), STOP);
      ts =
          new TokenFilter(ts) {
            final PayloadAttribute pay = addAttribute(PayloadAttribute.class);
            int pos;

            @Override
            public boolean incrementToken() throws IOException {
              if (!input.incrementToken()) return false;
              pay.setPayload(payload(kind, pos++));
              return true;
            }

            @Override
            public void reset() throws IOException {
              super.reset();
              pos = 0;
            }
          };
      return new TokenStreamComponents(t, ts);
    }
  }

  static final String[] PAYLOAD_FIELDS = {"pay", "ipay", "fpay", "spay"};

  static Analyzer analyzer() {
    Map<String, Analyzer> per = new HashMap<>();
    for (String f : PAYLOAD_FIELDS) per.put(f, new PayloadAnalyzer(f));
    return new PerFieldAnalyzerWrapper(new StandardAnalyzer(STOP), per);
  }

  static IndexWriterConfig config() {
    IndexWriterConfig cfg = new IndexWriterConfig(analyzer());
    cfg.setUseCompoundFile(false);
    cfg.setMergePolicy(NoMergePolicy.INSTANCE);
    cfg.setMaxBufferedDocs(IndexWriterConfig.DISABLE_AUTO_FLUSH);
    cfg.setRAMBufferSizeMB(256);
    return cfg;
  }

  static final FieldType OFFSETS = new FieldType(TextField.TYPE_NOT_STORED);

  static {
    OFFSETS.setIndexOptions(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS_AND_OFFSETS);
    OFFSETS.freeze();
  }

  static String word(Random r) {
    int i = (int) (Math.pow(r.nextDouble(), 1.4) * WORDS.length);
    return WORDS[Math.min(i, WORDS.length - 1)];
  }

  static Document doc(Random r, int id) {
    Document d = new Document();
    d.add(new StringField("id", "d" + id, Field.Store.NO));
    int len = id % 11 == 5 ? 0 : 1 + r.nextInt(22);
    List<String> words = new ArrayList<>();
    for (int i = 0; i < len; i++) {
      if (r.nextInt(9) == 0) words.add("the");
      words.add(word(r));
    }
    if (id == 17) words.add(r.nextInt(words.size() + 1), "zeta");
    String text = String.join(" ", words);
    if (len > 0) {
      d.add(new Field("body", text, OFFSETS));
      List<String> rev = new ArrayList<>(words);
      java.util.Collections.reverse(rev);
      d.add(new Field("body2", String.join(" ", rev), OFFSETS));
      for (String f : PAYLOAD_FIELDS) d.add(new Field(f, text, OFFSETS));
    }
    return d;
  }

  public static void main(String[] args) throws IOException {
    Path root = Path.of(args[0]).resolve("spans");
    clean(root);
    Path indexDir = root.resolve("index");
    Files.createDirectories(indexDir);
    Random r = new Random(0x10_5_2026_1007L);
    try (Directory dir = FSDirectory.open(indexDir);
        IndexWriter w = new IndexWriter(dir, config())) {
      int id = 0;
      int[] sizes = {30, 25, 28, 12};
      for (int seg = 0; seg < sizes.length; seg++) {
        for (int i = 0; i < sizes[seg]; i++, id++) {
          w.addDocument(doc(r, id));
        }
        w.commit();
        if (seg == 1 || seg == 2) {
          for (int k = 0; k < 5; k++) {
            w.deleteDocuments(new Term("id", "d" + (id - 1 - r.nextInt(sizes[seg]))));
          }
          w.commit();
        }
      }
    }
    StringBuilder out = new StringBuilder();
    try (Directory dir = FSDirectory.open(indexDir);
        DirectoryReader reader = DirectoryReader.open(dir)) {
      if (reader.leaves().size() != 4) throw new AssertionError("segments");
      IndexSearcher searcher = new IndexSearcher(reader);
      searcher.setQueryCache(null);
      for (String spec : SPECS) {
        run(out, searcher, spec);
      }
    }
    Files.writeString(root.resolve("queries.tsv"), out.toString(), StandardCharsets.UTF_8);
  }

  // ---------------------------------------------------------------------------------------------
  // The spec grammar: name(arg,arg,...), args split at top-level commas.
  // ---------------------------------------------------------------------------------------------

  static List<String> split(String s) {
    List<String> out = new ArrayList<>();
    int depth = 0, start = 0;
    for (int i = 0; i < s.length(); i++) {
      char c = s.charAt(i);
      if (c == '(') depth++;
      else if (c == ')') depth--;
      else if (c == ',' && depth == 0) {
        out.add(s.substring(start, i));
        start = i + 1;
      }
    }
    if (start < s.length() || !out.isEmpty()) out.add(s.substring(start));
    return out;
  }

  static String name(String spec) {
    int p = spec.indexOf('(');
    return p < 0 ? spec : spec.substring(0, p);
  }

  static List<String> args(String spec) {
    int p = spec.indexOf('(');
    return p < 0 ? List.of() : split(spec.substring(p + 1, spec.length() - 1));
  }

  static SpanQuery[] queries(List<String> a, int from) {
    SpanQuery[] out = new SpanQuery[a.size() - from];
    for (int i = from; i < a.size(); i++) out[i - from] = query(a.get(i));
    return out;
  }

  static PayloadFunction function(String name) {
    return switch (name) {
      case "min" -> new MinPayloadFunction();
      case "max" -> new MaxPayloadFunction();
      case "avg" -> new AveragePayloadFunction();
      case "sum" -> new SumPayloadFunction();
      default -> throw new AssertionError(name);
    };
  }

  /** A payload to match, as its type encodes it: {@code null} for none. */
  static BytesRef payloadArg(String type, String v) {
    if (v.equals("null")) return null;
    return switch (type) {
      case "INT" -> new BytesRef(ByteBuffer.allocate(4).putInt(Integer.parseInt(v)).array());
      case "FLOAT" -> new BytesRef(ByteBuffer.allocate(4).putFloat(Float.parseFloat(v)).array());
      case "BYTE" -> new BytesRef(new byte[] {Byte.parseByte(v)});
      default -> new BytesRef(v.replace("e'", "\u00e9"));
    };
  }

  static <Q extends MultiTermQuery> SpanQuery wrap(Q q, int topN) {
    SpanMultiTermQueryWrapper<Q> w = new SpanMultiTermQueryWrapper<>(q);
    if (topN > 0) w.setRewriteMethod(new SpanMultiTermQueryWrapper.TopTermsSpanBooleanQueryRewrite(topN));
    return w;
  }

  /** A bare word is a term of {@code body}; {@code t(field,word)} one of another field. */
  static SpanQuery query(String spec) {
    List<String> a = args(spec);
    return switch (name(spec)) {
      case "t" -> new SpanTermQuery(new Term(a.get(0), a.get(1)));
      case "near" ->
          new SpanNearQuery(
              queries(a, 2), Integer.parseInt(a.get(0)), Boolean.parseBoolean(a.get(1)));
      case "or" -> new SpanOrQuery(queries(a, 0));
      case "first" -> new SpanFirstQuery(query(a.get(1)), Integer.parseInt(a.get(0)));
      case "range" ->
          new SpanPositionRangeQuery(
              query(a.get(2)), Integer.parseInt(a.get(0)), Integer.parseInt(a.get(1)));
      case "not" ->
          new SpanNotQuery(
              query(a.get(0)),
              query(a.get(1)),
              Integer.parseInt(a.get(2)),
              Integer.parseInt(a.get(3)));
      case "containing" -> new SpanContainingQuery(query(a.get(0)), query(a.get(1)));
      case "within" -> new SpanWithinQuery(query(a.get(0)), query(a.get(1)));
      case "mask" -> new FieldMaskingSpanQuery(query(a.get(1)), a.get(0));
      case "prefix" -> wrap(new PrefixQuery(new Term(a.get(0), a.get(1))), 0);
      case "wildcard" -> wrap(new WildcardQuery(new Term(a.get(0), a.get(1))), 0);
      case "regexp" -> wrap(new RegexpQuery(new Term(a.get(0), a.get(1))), 0);
      case "trange" ->
          wrap(
              new TermRangeQuery(
                  a.get(0),
                  new BytesRef(a.get(1)),
                  new BytesRef(a.get(2)),
                  Boolean.parseBoolean(a.get(3)),
                  Boolean.parseBoolean(a.get(4))),
              0);
      case "topprefix" ->
          wrap(new PrefixQuery(new Term(a.get(1), a.get(2))), Integer.parseInt(a.get(0)));
      case "check" -> {
        // check(TYPE,OP,query,payload|payload|...)
        String type = a.get(0);
        List<BytesRef> pays = new ArrayList<>();
        for (String v : a.get(3).split("\\|")) pays.add(payloadArg(type, v));
        SpanPayloadCheckQuery.PayloadType pt =
            type.equals("BYTE")
                ? SpanPayloadCheckQuery.PayloadType.STRING
                : SpanPayloadCheckQuery.PayloadType.valueOf(type);
        yield new SpanPayloadCheckQuery(
            query(a.get(2)), pays, pt, SpanPayloadCheckQuery.MatchOperation.valueOf(a.get(1)));
      }
      case "pscore" ->
          new PayloadScoreQuery(
              query(a.get(2)),
              function(a.get(0)),
              PayloadDecoder.FLOAT_DECODER,
              Boolean.parseBoolean(a.get(1)));
      default -> {
        if (!a.isEmpty()) throw new AssertionError(spec);
        yield new SpanTermQuery(new Term("body", spec));
      }
    };
  }

  // ---------------------------------------------------------------------------------------------
  // queries.tsv
  // ---------------------------------------------------------------------------------------------

  static String hex(float f) {
    return Integer.toHexString(Float.floatToIntBits(f));
  }

  static String err(Throwable e) {
    return "!" + e.getClass().getSimpleName();
  }

  static String clean(String s) {
    return s == null ? "null" : s.replace("\\", "\\\\").replace("\t", "\\t").replace("\n", "\\n");
  }

  static final int[] EXPLAIN = {0, 7, 17, 33, 60};

  static final String[] VARIANTS = {"plain", "boost", "bool", "filter", "mustnot"};

  static Query variant(String v, SpanQuery s) {
    return switch (v) {
      case "plain" -> s;
      case "boost" -> new BoostQuery(s, 2.5f);
      case "bool" ->
          new BooleanQuery.Builder()
              .add(s, BooleanClause.Occur.MUST)
              .add(new TermQuery(new Term("body", "egg")), BooleanClause.Occur.SHOULD)
              .build();
      case "filter" ->
          new BooleanQuery.Builder()
              .add(s, BooleanClause.Occur.FILTER)
              .add(new TermQuery(new Term("body", "egg")), BooleanClause.Occur.SHOULD)
              .build();
      case "mustnot" ->
          new BooleanQuery.Builder()
              .add(new TermQuery(new Term("body", "egg")), BooleanClause.Occur.SHOULD)
              .add(s, BooleanClause.Occur.MUST_NOT)
              .build();
      default -> throw new AssertionError(v);
    };
  }

  static void run(StringBuilder out, IndexSearcher searcher, String spec) {
    String head = spec;
    SpanQuery s;
    try {
      s = query(spec);
    } catch (Exception e) {
      out.append(head).append("\tquery\t").append(err(e)).append('\n');
      return;
    }
    String str;
    try {
      str = clean(s.toString());
    } catch (Exception e) {
      // `Term.toString(null)`: a payload check's `null` payload.
      str = err(e);
    }
    out.append(head).append("\tquery\t").append(str).append('\n');
    for (String v : VARIANTS) {
      Query q = variant(v, s);
      try {
        TopDocs td = searcher.search(q, 1000);
        StringBuilder b = new StringBuilder();
        b.append(td.totalHits.value()).append(' ');
        for (ScoreDoc sd : td.scoreDocs) {
          b.append(sd.doc).append(':').append(hex(sd.score)).append(',');
        }
        out.append(head).append('\t').append(v).append("\thits\t").append(b).append('\n');
      } catch (Exception e) {
        out.append(head).append('\t').append(v).append("\thits\t").append(err(e)).append('\n');
        continue;
      }
      for (int doc : EXPLAIN) {
        String e;
        try {
          e = clean(searcher.explain(q, doc).toString());
        } catch (Exception ex) {
          e = err(ex);
        }
        out.append(head).append('\t').append(v).append("\texplain ").append(doc).append('\t');
        out.append(e).append('\n');
      }
    }
    // `Weight.matches` of the plain query's hits and the explained documents.
    java.util.TreeSet<Integer> docs = new java.util.TreeSet<>();
    for (int doc : EXPLAIN) docs.add(doc);
    try {
      for (ScoreDoc sd : searcher.search(s, 1000).scoreDocs) docs.add(sd.doc);
    } catch (Exception e) {
      // The plain variant's hits line above records it.
    }
    for (int doc : docs) {
      String m;
      try {
        Weight w = searcher.createWeight(searcher.rewrite(s), ScoreMode.COMPLETE_NO_SCORES, 1);
        List<LeafReaderContext> leaves = searcher.getIndexReader().leaves();
        LeafReaderContext ctx = leaves.get(ReaderUtil.subIndex(doc, leaves));
        m = render(w.matches(ctx, doc - ctx.docBase));
      } catch (Exception ex) {
        m = err(ex);
      }
      out.append(head).append("\tmatches ").append(doc).append('\t').append(m).append('\n');
    }
  }

  /** Each field's matches: {@code start:end:startOffset:endOffset[sub ...]}, a sub with its term. */
  static String render(Matches m) throws IOException {
    if (m == null) return "none";
    StringBuilder b = new StringBuilder();
    for (String f : m) {
      if (b.length() > 0) b.append(';');
      b.append(f).append('|');
      MatchesIterator it = m.getMatches(f);
      boolean first = true;
      while (it != null && it.next()) {
        if (!first) b.append(',');
        first = false;
        b.append(it.startPosition()).append(':').append(it.endPosition()).append(':')
            .append(it.startOffset()).append(':').append(it.endOffset());
        MatchesIterator sub = it.getSubMatches();
        if (sub != null) {
          b.append('[');
          boolean f2 = true;
          while (sub.next()) {
            if (!f2) b.append(' ');
            f2 = false;
            b.append(sub.startPosition()).append(':').append(sub.endPosition()).append(':')
                .append(sub.startOffset()).append(':').append(sub.endOffset()).append('=')
                .append(sub.getQuery());
          }
          b.append(']');
        }
      }
    }
    return b.toString();
  }

  static final String[] SPECS = {
    // terms
    "apple", "zeta", "nosuch", "t(nofield,apple)", "t(pay,cat)",
    // near and or (the faster path's queries, here through the general one when nested)
    "near(0,true,apple,bank)", "near(2,false,apple,bank,cat)", "or(apple,zeta,nosuch)",
    "near(1,true,or(apple,ape),bank)", "near(3,false,near(0,true,apple,bank),cat)",
    // first and position ranges
    "first(3,apple)", "first(0,apple)", "first(5,near(1,true,apple,bank))", "first(4,or(cat,dog))",
    "range(2,6,apple)", "range(0,1,or(apple,bank))", "range(3,10,near(2,false,cat,dog))",
    "near(2,true,first(4,apple),bank)",
    // not
    "not(apple,bank,0,0)", "not(apple,bank,1,1)", "not(apple,bank,2,0)", "not(apple,bank,-1,-1)",
    "not(near(1,true,apple,bank),cat,0,3)", "not(or(apple,cat),near(0,true,bank,dog),1,1)",
    "not(apple,nosuch,0,0)", "not(nosuch,apple,0,0)",
    "not(apple,near(2,false,bank,cat),0,0)",
    // containing / within
    "containing(near(3,true,apple,cat),bank)", "containing(near(5,false,apple,bank),or(cat,dog))",
    "within(near(3,true,apple,cat),bank)", "within(near(5,false,apple,bank),or(cat,dog))",
    "containing(or(near(4,true,apple,cat),near(4,false,dog,egg)),bank)",
    "within(first(5,near(4,false,apple,bank)),apple)", "containing(apple,nosuch)",
    "within(near(6,false,cat,dog),or(apple,bank,egg))",
    // field masking
    "near(2,false,apple,mask(body,t(body2,bank)))", "mask(body,t(body2,apple))",
    "near(1,true,mask(body,t(body2,cat)),dog)",
    // multi-term
    "prefix(body,ap)", "prefix(body,ba)", "prefix(body,zz)", "wildcard(body,ca?)",
    "regexp(body,d[a-z]+)", "trange(body,c,e,true,false)", "topprefix(2,body,ba)",
    "near(1,true,prefix(body,ap),or(cat,car))", "not(prefix(body,ba),apple,1,1)",
    // payload checks
    "check(BYTE,EQ,t(pay,cat),1)", "check(BYTE,EQ,t(pay,cat),null)",
    "check(BYTE,EQ,near(0,true,t(pay,apple),t(pay,bank)),2|3)",
    "check(BYTE,EQ,near(1,false,t(pay,apple),t(pay,bank)),0|1)",
    "check(INT,GT,t(ipay,apple),0)", "check(INT,LTE,t(ipay,cat),-1)", "check(INT,EQ,t(ipay,dog),5)",
    "check(INT,GTE,near(2,true,t(ipay,apple),t(ipay,bank)),0|null)",
    "check(FLOAT,LT,t(fpay,apple),0.5)", "check(FLOAT,GTE,t(fpay,bank),0.5)",
    "check(STRING,LT,t(spay,cat),b)", "check(STRING,GTE,t(spay,apple),ba)",
    "check(STRING,GT,t(spay,dog),b)", "check(STRING,EQ,t(spay,egg),e')",
    "check(STRING,EQ,t(spay,cat),a|b)", "first(6,check(BYTE,EQ,t(pay,apple),0))",
    // payload scores
    "pscore(min,true,t(pay,apple))", "pscore(max,true,t(pay,apple))",
    "pscore(avg,true,t(pay,apple))", "pscore(sum,true,t(pay,apple))",
    "pscore(sum,false,t(pay,cat))", "pscore(max,false,near(2,false,t(pay,apple),t(pay,bank)))",
    "pscore(avg,true,or(t(pay,cat),t(pay,dog)))", "pscore(min,false,first(3,t(pay,bank)))",
    "pscore(sum,true,t(pay,nosuch))", "near(2,true,pscore(sum,true,t(pay,apple)),t(pay,bank))",
    // nested: the filters, containments and payload checks inside other span queries
    "near(3,true,containing(near(3,true,apple,cat),bank),dog)",
    "first(8,within(near(5,false,apple,bank),or(cat,dog)))",
    "not(within(near(4,false,apple,cat),bank),dog,0,0)",
    "or(containing(near(4,true,apple,cat),bank),first(1,dog))",
    "check(BYTE,EQ,containing(near(3,true,t(pay,apple),t(pay,cat)),t(pay,bank)),0|1|2)",
    "or(check(BYTE,EQ,t(pay,cat),1),t(pay,dog))",
    "near(1,false,not(apple,bank,0,0),range(0,5,cat))",
    "mask(body,within(t(body2,bank),near(3,true,t(body2,apple),t(body2,cat))))",
    // multi-term wrappers inside every wrapper
    "first(3,prefix(body,ap))", "range(0,4,wildcard(body,ca?))",
    "containing(near(4,true,prefix(body,ap),cat),bank)", "within(near(5,false,apple,cat),prefix(body,ba))",
    "mask(body,prefix(body2,ap))", "check(BYTE,EQ,prefix(pay,ca),1)",
    "pscore(sum,true,prefix(pay,ba))", "near(2,false,trange(body,a,c,false,true),dog)",
    // the pulsed singleton's payloads; a missing clause; refusals
    "check(BYTE,EQ,t(pay,zeta),1)", "pscore(max,true,t(pay,zeta))", "near(1,true,apple,nosuch)",
    "near(1,true,t(nofield,a),t(nofield,b))", "near(0,true,apple)", "t(id,d1)",
  };
}
