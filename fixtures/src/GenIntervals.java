import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.function.Predicate;
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
import org.apache.lucene.queries.intervals.IntervalQuery;
import org.apache.lucene.queries.intervals.Intervals;
import org.apache.lucene.queries.intervals.IntervalsSource;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.BoostQuery;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Matches;
import org.apache.lucene.search.MatchesIterator;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.ScoreMode;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.Weight;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.BytesRef;

/**
 * M10 T10.6's intervals fixture: {@code intervals/index}, a four-segment index with deletions in
 * two segments, and {@code intervals/queries.tsv}, every {@code Intervals} factory's source over
 * it.
 *
 * <p>Fields: {@code id} (StringField), {@code body} (positions and offsets, a stop word leaving
 * holes), {@code body2} (the same words reversed, for {@code fixField}), {@code pay} (positions,
 * offsets and payloads: a one-byte payload per position, none on every fifth), {@code nopos}
 * (docs only). One word ({@code zeta}) is in a single document (a pulsed singleton term).
 *
 * <p>For each source spec (the grammar is the Rust test's, {@code intervals_fixtures.rs}): its
 * {@code toString} and {@code minExtent}; then, for each scoring variant ({@code IntervalQuery}
 * plain, with a pivot, with a pivot and exponent, boosted, as the required clause of a boolean),
 * every hit with its score bits and five documents' explanations; then the {@code Matches} of every
 * hit -- positions, offsets and sub-matches. An exception is recorded as its class name.
 *
 * <p>Usage: {@code java GenIntervals <fixtures-data-dir>}.
 */
public class GenIntervals {
  static final String[] WORDS = {
    "apple", "apply", "ape", "bank", "band", "banana", "cat", "car", "cart", "dog", "door", "dot",
    "egg", "eel", "fig", "fish"
  };

  static final CharArraySet STOP = new CharArraySet(List.of("the"), false);

  static void clean(Path root) throws IOException {
    if (Files.exists(root)) {
      try (Stream<Path> walk = Files.walk(root)) {
        for (Path p : walk.sorted(Comparator.reverseOrder()).toList()) Files.delete(p);
      }
    }
  }

  /** {@code StandardTokenizer}, lower-cased, {@code the} removed, a payload per position. */
  static final class PayloadAnalyzer extends Analyzer {
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
              int p = pos++;
              if (p % 5 == 4) {
                pay.setPayload(null);
              } else if (p % 7 == 3) {
                pay.setPayload(new BytesRef(new byte[] {(byte) -2}));
              } else {
                pay.setPayload(new BytesRef(new byte[] {(byte) (p % 4)}));
              }
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

  static Analyzer analyzer() {
    Map<String, Analyzer> per = new HashMap<>();
    per.put("pay", new PayloadAnalyzer());
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
      d.add(new Field("pay", text, OFFSETS));
    }
    d.add(new StringField("nopos", words.isEmpty() ? "none" : words.get(0), Field.Store.NO));
    return d;
  }

  public static void main(String[] args) throws IOException {
    Path root = Path.of(args[0]).resolve("intervals");
    clean(root);
    Path indexDir = root.resolve("index");
    Files.createDirectories(indexDir);
    Random r = new Random(0x10_5_2026_1006L);
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
      for (String[] spec : SPECS) {
        run(out, searcher, spec[0], spec[1]);
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

  static Predicate<BytesRef> filter(String name) {
    return switch (name) {
      case "even" -> b -> b != null && b.bytes[b.offset] % 2 == 0;
      case "null" -> b -> b == null;
      case "ge1" -> b -> b != null && b.bytes[b.offset] >= 1;
      case "neg" -> b -> b != null && b.bytes[b.offset] < 0;
      default -> throw new AssertionError(name);
    };
  }

  static IntervalsSource[] sources(List<String> a, int from) throws IOException {
    IntervalsSource[] out = new IntervalsSource[a.size() - from];
    for (int i = from; i < a.size(); i++) out[i - from] = source(a.get(i));
    return out;
  }

  /** A bare word is a term. */
  static IntervalsSource source(String spec) throws IOException {
    List<String> a = args(spec);
    return switch (name(spec)) {
      case "pt" -> Intervals.term(a.get(0), filter(a.get(1)));
      case "phrase" -> Intervals.phrase(sources(a, 0));
      case "phraset" -> Intervals.phrase(a.toArray(String[]::new));
      case "or" -> Intervals.or(sources(a, 0));
      case "ornr" -> Intervals.or(false, sources(a, 0));
      case "ordered" -> Intervals.ordered(sources(a, 0));
      case "unordered" -> Intervals.unordered(sources(a, 0));
      case "uno" -> Intervals.unorderedNoOverlaps(source(a.get(0)), source(a.get(1)));
      case "maxgaps" -> Intervals.maxgaps(Integer.parseInt(a.get(0)), source(a.get(1)));
      case "maxwidth" -> Intervals.maxwidth(Integer.parseInt(a.get(0)), source(a.get(1)));
      case "extend" ->
          Intervals.extend(
              source(a.get(0)), Integer.parseInt(a.get(1)), Integer.parseInt(a.get(2)));
      case "containing" -> Intervals.containing(source(a.get(0)), source(a.get(1)));
      case "notcontaining" -> Intervals.notContaining(source(a.get(0)), source(a.get(1)));
      case "containedby" -> Intervals.containedBy(source(a.get(0)), source(a.get(1)));
      case "notcontainedby" -> Intervals.notContainedBy(source(a.get(0)), source(a.get(1)));
      case "overlapping" -> Intervals.overlapping(source(a.get(0)), source(a.get(1)));
      case "nonoverlapping" -> Intervals.nonOverlapping(source(a.get(0)), source(a.get(1)));
      case "before" -> Intervals.before(source(a.get(0)), source(a.get(1)));
      case "after" -> Intervals.after(source(a.get(0)), source(a.get(1)));
      case "within" ->
          Intervals.within(source(a.get(0)), Integer.parseInt(a.get(1)), source(a.get(2)));
      case "notwithin" ->
          Intervals.notWithin(source(a.get(0)), Integer.parseInt(a.get(1)), source(a.get(2)));
      case "atleast" -> Intervals.atLeast(Integer.parseInt(a.get(0)), sources(a, 1));
      case "fix" -> Intervals.fixField(a.get(0), source(a.get(1)));
      case "prefix" ->
          a.size() == 1
              ? Intervals.prefix(new BytesRef(a.get(0)))
              : Intervals.prefix(new BytesRef(a.get(0)), Integer.parseInt(a.get(1)));
      case "wildcard" -> Intervals.wildcard(new BytesRef(a.get(0)));
      case "regexp" -> Intervals.regexp(new BytesRef(a.get(0)));
      case "range" ->
          Intervals.range(
              a.get(0).equals("*") ? null : new BytesRef(a.get(0)),
              a.get(1).equals("*") ? null : new BytesRef(a.get(1)),
              Boolean.parseBoolean(a.get(2)),
              Boolean.parseBoolean(a.get(3)));
      case "fuzzy" ->
          a.size() == 2
              ? Intervals.fuzzyTerm(a.get(0), Integer.parseInt(a.get(1)))
              : Intervals.fuzzyTerm(
                  a.get(0),
                  Integer.parseInt(a.get(1)),
                  Integer.parseInt(a.get(2)),
                  Boolean.parseBoolean(a.get(3)),
                  Integer.parseInt(a.get(4)));
      case "none" -> Intervals.noIntervals(a.get(0));
      case "analyzed" ->
          Intervals.analyzedText(
              a.get(0).replace('_', ' '),
              new StandardAnalyzer(STOP),
              "body",
              Integer.parseInt(a.get(1)),
              Boolean.parseBoolean(a.get(2)));
      default -> {
        if (!a.isEmpty()) throw new AssertionError(spec);
        yield Intervals.term(spec);
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

  static final String[] VARIANTS = {"plain", "pivot", "sigmoid", "boost", "bool"};

  static Query variant(String v, String field, IntervalsSource s) {
    return switch (v) {
      case "plain" -> new IntervalQuery(field, s);
      case "pivot" -> new IntervalQuery(field, s, 2.5f);
      case "sigmoid" -> new IntervalQuery(field, s, 1.5f, 2.0f);
      case "boost" -> new BoostQuery(new IntervalQuery(field, s), 3.0f);
      case "bool" ->
          new BooleanQuery.Builder()
              .add(new IntervalQuery(field, s), BooleanClause.Occur.MUST)
              .add(new TermQuery(new Term("body", "egg")), BooleanClause.Occur.SHOULD)
              .build();
      default -> throw new AssertionError(v);
    };
  }

  static void run(StringBuilder out, IndexSearcher searcher, String field, String spec) {
    String head = field + "\t" + spec;
    IntervalsSource s;
    try {
      s = source(spec);
    } catch (Exception e) {
      out.append(head).append("\tsource\t").append(err(e)).append('\n');
      return;
    }
    out.append(head)
        .append("\tsource\t")
        .append(clean(s.toString()))
        .append('\t')
        .append(s.minExtent())
        .append('\n');
    List<Integer> hits = new ArrayList<>();
    for (String v : VARIANTS) {
      Query q = variant(v, field, s);
      try {
        TopDocs td = searcher.search(q, 1000);
        StringBuilder b = new StringBuilder();
        b.append(td.totalHits.value()).append(' ');
        for (ScoreDoc sd : td.scoreDocs) {
          b.append(sd.doc).append(':').append(hex(sd.score)).append(',');
          if (v.equals("plain")) hits.add(sd.doc);
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
    // The hits, and the explained documents whether they match or not.
    java.util.TreeSet<Integer> docs = new java.util.TreeSet<>(hits);
    for (int doc : EXPLAIN) docs.add(doc);
    Query plain = new IntervalQuery(field, s);
    for (int doc : docs) {
      String m;
      try {
        Weight w = searcher.createWeight(searcher.rewrite(plain), ScoreMode.COMPLETE_NO_SCORES, 1);
        List<LeafReaderContext> leaves = searcher.getIndexReader().leaves();
        LeafReaderContext ctx = leaves.get(ReaderUtil.subIndex(doc, leaves));
        m = render(w.matches(ctx, doc - ctx.docBase));
      } catch (Exception ex) {
        m = err(ex);
      }
      out.append(head).append("\tmatches ").append(doc).append('\t').append(m).append('\n');
    }
  }

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
        b.append(it.startPosition())
            .append(':')
            .append(it.endPosition())
            .append(':')
            .append(it.startOffset())
            .append(':')
            .append(it.endOffset());
        MatchesIterator sub = it.getSubMatches();
        if (sub != null) {
          b.append('[');
          boolean f2 = true;
          while (sub.next()) {
            if (!f2) b.append(' ');
            f2 = false;
            b.append(sub.startPosition())
                .append(':')
                .append(sub.endPosition())
                .append(':')
                .append(sub.startOffset())
                .append(':')
                .append(sub.endOffset());
          }
          b.append(']');
        }
      }
    }
    return b.toString();
  }

  /** {field, source spec}. */
  static final String[][] SPECS = {
    // terms
    {"body", "apple"}, {"body", "zeta"}, {"body", "nosuch"}, {"pay", "cat"}, {"nofield", "apple"},
    {"nopos", "apple"},
    // phrases and blocks
    {"body", "phrase(apple,bank)"}, {"body", "phraset(cat,dog,egg)"}, {"body", "phraset(fig)"},
    {"body", "phrase(apple,apple)"}, {"body", "phrase(or(apple,phrase(ape,bank)),cat)"},
    {"body", "phrase(ornr(apple,phrase(ape,bank)),cat)"},
    {"body", "phrase(phrase(apple,bank),cat)"}, {"body", "phrase(apple,extend(bank,0,2),cat)"},
    // disjunctions
    {"body", "or(apple,bank)"}, {"body", "or(apple,phrase(bank,cat))"},
    {"body", "or(apple,apple,or(bank,apple))"}, {"body", "or(nosuch,zeta)"},
    {"body", "ornr(apple,phrase(bank,cat))"},
    // ordered / unordered
    {"body", "ordered(apple,bank)"}, {"body", "ordered(apple,bank,cat)"},
    {"body", "ordered(apple,apple)"}, {"body", "ordered(apple,apple,bank)"},
    {"body", "ordered(bank,apple,apple)"}, {"body", "ordered(apple,or(bank,cat),dog)"},
    {"body", "ordered(phrase(apple,bank),cat)"}, {"body", "ordered(apple,nosuch)"},
    {"body", "unordered(apple,bank)"}, {"body", "unordered(apple,apple)"},
    {"body", "unordered(apple,bank,apple)"}, {"body", "unordered(apple,bank,cat,dog)"},
    {"body", "unordered(apple,ordered(bank,cat))"}, {"body", "uno(apple,bank)"},
    {"body", "unordered(apple)"},
    // filters
    {"body", "maxgaps(1,ordered(apple,bank))"}, {"body", "maxgaps(0,unordered(apple,bank,cat))"},
    {"body", "maxgaps(2,ordered(or(apple,phrase(ape,bank)),cat))"},
    {"body", "maxgaps(3,unordered(apple,apple))"}, {"body", "maxwidth(3,unordered(apple,bank))"},
    {"body", "maxwidth(4,or(ordered(apple,bank),phrase(cat,dog,egg)))"},
    {"body", "ordered(maxgaps(1,ordered(apple,bank)),cat)"},
    // extend
    {"body", "extend(apple,1,2)"}, {"body", "ordered(extend(apple,2,0),bank)"},
    {"body", "extend(ordered(apple,bank),3,1)"}, {"body", "phrase(extend(or(apple,bank),1,0),cat)"},
    // containment and overlap
    {"body", "containing(ordered(apple,cat),bank)"},
    {"body", "containing(unordered(apple,bank),or(cat,dog))"},
    {"body", "containing(or(ordered(apple,cat),phrase(dog,egg,fig)),bank)"},
    {"body", "notcontaining(ordered(apple,cat),bank)"},
    {"body", "notcontaining(or(ordered(apple,cat),ordered(dog,fig)),bank)"},
    {"body", "containedby(bank,ordered(apple,cat))"},
    {"body", "containedby(bank,or(ordered(apple,cat),phrase(dog,egg)))"},
    {"body", "notcontainedby(bank,ordered(apple,cat))"},
    {"body", "notcontainedby(bank,or(ordered(apple,cat),ordered(dog,fig)))"},
    {"body", "overlapping(ordered(apple,bank),ordered(cat,dog))"},
    {"body", "overlapping(or(apple,phrase(bank,cat)),unordered(cat,dog))"},
    {"body", "nonoverlapping(ordered(apple,bank),ordered(bank,cat))"},
    {"body", "nonoverlapping(apple,nosuch)"}, {"body", "notcontaining(apple,nosuch)"},
    {"body", "before(apple,bank)"}, {"body", "after(apple,bank)"},
    {"body", "before(or(apple,cat),phrase(bank,dog))"}, {"body", "within(apple,2,bank)"},
    {"body", "notwithin(apple,1,bank)"}, {"body", "within(or(apple,dog),1,or(bank,cat))"},
    // at least
    {"body", "atleast(2,apple,bank,cat)"}, {"body", "atleast(2,apple,bank,cat,dog)"},
    {"body", "atleast(3,apple,bank,cat,dog)"}, {"body", "atleast(2,apple,phrase(bank,cat),dog)"},
    {"body", "atleast(4,apple,bank,cat)"}, {"body", "atleast(3,apple,bank,cat)"},
    {"body", "atleast(2,apple,apple,bank)"}, {"body", "atleast(1,apple,nosuch)"},
    {"body", "ordered(atleast(2,apple,bank,cat),dog)"},
    // fixField
    {"body", "ordered(apple,fix(body2,bank))"}, {"body", "fix(body2,phrase(bank,apple))"},
    {"body", "fix(body2,or(apple,phrase(bank,cat)))"},
    // multi-term
    {"body", "prefix(ap)"}, {"body", "prefix(ba)"}, {"body", "prefix(zz)"},
    {"body", "wildcard(ca?)"}, {"body", "wildcard(d*)"}, {"body", "regexp(b[a-z]+)"},
    {"body", "range(c,e,true,false)"}, {"body", "range(*,b,true,true)"},
    {"body", "range(fig,*,false,true)"}, {"body", "fuzzy(cat,1)"}, {"body", "fuzzy(dor,2)"},
    {"body", "fuzzy(bnak,1,0,true,128)"}, {"body", "fuzzy(bnak,1,0,false,128)"},
    {"body", "fuzzy(ap,1,1,true,128)"}, {"body", "prefix(a,2)"}, {"body", "prefix(ba,2000)"},
    {"body", "ordered(prefix(ap),cat)"}, {"body", "phrase(prefix(ba),or(cat,car))"},
    {"pay", "prefix(do)"},
    // payload filters
    {"pay", "pt(apple,even)"}, {"pay", "pt(apple,null)"}, {"pay", "pt(cat,ge1)"},
    {"pay", "pt(bank,neg)"}, {"pay", "ordered(pt(cat,ge1),dog)"},
    {"pay", "or(pt(apple,even),pt(apple,null))"}, {"body", "pt(apple,even)"},
    // no intervals
    {"body", "none(nothing_here)"}, {"body", "ordered(apple,none(x))"},
    {"body", "or(apple,none(x))"},
    // analyzed text
    {"body", "analyzed(apple_bank,0,true)"}, {"body", "analyzed(the_apple_the_bank,0,true)"},
    {"body", "analyzed(apple_the_cat,2,false)"}, {"body", "analyzed(apple_bank_cat,-1,true)"},
    {"body", "analyzed(apple,0,true)"}, {"body", "analyzed(the,0,true)"},
    {"body", "analyzed(cat_dog,1,true)"},
  };
}
