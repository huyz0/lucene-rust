import java.io.*; import java.nio.charset.StandardCharsets; import java.nio.file.*; import java.util.*; import java.util.function.Predicate;
import org.apache.lucene.analysis.*; import org.apache.lucene.analysis.standard.StandardAnalyzer; import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.document.*; import org.apache.lucene.index.*; import org.apache.lucene.search.*; import org.apache.lucene.store.*;
import org.apache.lucene.util.*; import org.apache.lucene.util.automaton.*;

/**
 * Stock baselines for the substring-field bench, on stock Lucene 10.5.1:
 *   build-text <corpus> <out>  msg as text (StandardAnalyzer, positions), line numeric doc values
 *   build-os   <corpus> <out>  OpenSearch's wildcard field layout: msg as anchored 3-grams (docs only) + msg sorted-set doc values
 *   query <text> <os> <queries.tsv> [secs]  text-field wildcard/regexp/term/and, and OpenSearch's WildcardMatchingQuery
 * The wildcard tokenizer and query mirror OpenSearch's WildcardFieldMapper (Apache 2.0).
 * Builds flush 256 MB segments without merging, then forceMerge(1); both timed.
 */
public class SubstringBench {
  static final FieldType DOCS = new FieldType();
  static { DOCS.setIndexOptions(IndexOptions.DOCS); DOCS.setTokenized(true); DOCS.setOmitNorms(true); DOCS.freeze(); }
  static final int NGRAM_SIZE = 3;

  /** OpenSearch WildcardFieldMapper.WildcardFieldTokenizer. */
  static final class WildcardFieldTokenizer extends Tokenizer {
    private final CharTermAttribute charTermAttribute = addAttribute(CharTermAttribute.class);
    private final char[] buffer = new char[NGRAM_SIZE];
    private int offset = NGRAM_SIZE - 1;
    @Override public void reset() throws IOException { super.reset(); for (int i = 0; i < NGRAM_SIZE - 1; i++) buffer[i] = 0; offset = NGRAM_SIZE - 1; }
    @Override public boolean incrementToken() throws IOException {
      charTermAttribute.setLength(NGRAM_SIZE);
      int c = input.read(); c = c == -1 ? 0 : c;
      buffer[offset++ % NGRAM_SIZE] = (char) c;
      boolean has_next = false;
      for (int i = 0; i < NGRAM_SIZE; i++) { char cur = buffer[(offset + i) % NGRAM_SIZE]; charTermAttribute.buffer()[i] = cur; has_next |= cur != 0; }
      return has_next;
    }
  }

  static IndexWriter writer(String out, Analyzer a) throws IOException {
    IndexWriterConfig c = new IndexWriterConfig(a);
    c.setRAMBufferSizeMB(256); c.setUseCompoundFile(false); c.setMergePolicy(NoMergePolicy.INSTANCE); c.setOpenMode(IndexWriterConfig.OpenMode.CREATE);
    return new IndexWriter(FSDirectory.open(Paths.get(out)), c);
  }

  static void build(String corpus, String out, boolean os) throws IOException {
    long t0 = System.nanoTime(); int n = 0;
    try (IndexWriter w = writer(out, new StandardAnalyzer(CharArraySet.EMPTY_SET)); BufferedReader r = Files.newBufferedReader(Paths.get(corpus), StandardCharsets.UTF_8)) {
      for (String line; (line = r.readLine()) != null; n++) {
        Document d = new Document();
        if (os) {
          Tokenizer t = new WildcardFieldTokenizer(); t.setReader(new StringReader(line));
          d.add(new Field("msg", t, DOCS));
          d.add(new SortedSetDocValuesField("msg", new BytesRef(line)));
        } else {
          d.add(new TextField("msg", line, Field.Store.NO));
        }
        d.add(new NumericDocValuesField("line", n));
        w.addDocument(d);
      }
      w.commit();
      long t1 = System.nanoTime();
      int segs; try (DirectoryReader rr = DirectoryReader.open(w)) { segs = rr.leaves().size(); }
      w.getConfig().setMergePolicy(new TieredMergePolicy()); w.forceMerge(1); w.commit();
      long t2 = System.nanoTime();
      System.out.printf("build-%s docs=%d index_s=%.1f docs_per_s=%.0f flushed_segments=%d merge_s=%.1f%n", os ? "os" : "text", n, (t1 - t0) / 1e9, n / ((t1 - t0) / 1e9), segs, (t2 - t1) / 1e9);
    }
    sizes(out);
  }

  static void sizes(String out) throws IOException {
    TreeMap<String, Long> m = new TreeMap<>(); long t = 0;
    for (Path p : Files.list(Paths.get(out)).toList()) { String f = p.getFileName().toString(); int d = f.lastIndexOf('.'); String ext = d < 0 ? f : f.substring(d + 1); long s = Files.size(p); m.merge(ext, s, Long::sum); t += s; }
    StringBuilder sb = new StringBuilder("sizes MiB:");
    for (var e : m.entrySet()) if (e.getValue() > 1 << 16) sb.append(String.format(" %s=%.1f", e.getKey(), e.getValue() / 1048576.0));
    System.out.println(sb.append(String.format(" total=%.1f", t / 1048576.0)));
  }

  // ------------------------------------------------------------ OpenSearch WildcardFieldType, ported verbatim where it matters
  static final Set<Character> WILDCARD_SPECIAL = Set.of('?', '*', '\\');
  static final Set<Character> REGEXP_SPECIAL = Set.of('.', '^', '$', '*', '+', '?', '(', ')', '[', ']', '{', '}', '|', '/', '\\');

  static Set<String> getRequiredNGrams(String value, boolean regexpMode) {
    Set<String> terms = new HashSet<>();
    if (value.isEmpty()) return terms;
    int pos = 0; String rawSequence; String currentSequence; char[] buffer = new char[NGRAM_SIZE];
    List<String> possiblePrefixTerms = new ArrayList<>();
    if (!value.startsWith("?") && !value.startsWith("*")) {
      rawSequence = getNonWildcardSequence(value, 0);
      currentSequence = performEscape(rawSequence, regexpMode);
      Arrays.fill(buffer, (char) 0);
      int startIdx = Math.max(NGRAM_SIZE - currentSequence.length(), 1);
      for (int j = 0; j < currentSequence.length() && j < NGRAM_SIZE - 1; j++) buffer[startIdx + j] = currentSequence.charAt(j);
      terms.add(new String(buffer));
    } else {
      pos = findNonWildcardSequence(value, pos);
      rawSequence = getNonWildcardSequence(value, pos);
    }
    while (pos < value.length()) {
      boolean isEndOfValue = pos + rawSequence.length() == value.length();
      currentSequence = performEscape(rawSequence, regexpMode);
      if (currentSequence.length() >= NGRAM_SIZE) {
        for (int i = 0; i < currentSequence.length() - NGRAM_SIZE + 1; i++) terms.add(currentSequence.substring(i, i + 3));
      } else if (isEndOfValue != true && currentSequence.isEmpty() == false) {
        possiblePrefixTerms.add(currentSequence);
      }
      if (isEndOfValue) {
        Arrays.fill(buffer, (char) 0);
        if (pos == 0 && currentSequence.length() == NGRAM_SIZE - 2) {
          for (int i = 0; i < currentSequence.length(); i++) buffer[i + 1] = currentSequence.charAt(i);
          terms.add(new String(buffer));
          Arrays.fill(buffer, (char) 0);
        }
        int rightStartIdx = NGRAM_SIZE - currentSequence.length() - 2;
        rightStartIdx = rightStartIdx < 0 ? NGRAM_SIZE - 2 : rightStartIdx;
        for (int j = 0; j < currentSequence.length() && j < NGRAM_SIZE - 1; j++) buffer[rightStartIdx - j] = currentSequence.charAt(currentSequence.length() - j - 1);
        terms.add(new String(buffer));
      }
      pos = findNonWildcardSequence(value, pos + rawSequence.length());
      rawSequence = getNonWildcardSequence(value, pos);
    }
    if (terms.isEmpty()) terms.addAll(possiblePrefixTerms);
    return terms;
  }
  static String getNonWildcardSequence(String value, int startFrom) {
    int bs = 0;
    for (int i = startFrom; i < value.length(); i++) { char c = value.charAt(i);
      if (c == '\\') bs++; else { if ((c == '?' || c == '*') && bs % 2 == 0) return value.substring(startFrom, i); bs = 0; } }
    return value.substring(startFrom);
  }
  static int findNonWildcardSequence(String value, int startFrom) {
    for (int i = startFrom; i < value.length(); i++) { char c = value.charAt(i); if (c != '?' && c != '*') return i; }
    return value.length();
  }
  static String performEscape(String str, boolean regexpMode) {
    StringBuilder sb = new StringBuilder(); Set<Character> target = regexpMode ? REGEXP_SPECIAL : WILDCARD_SPECIAL;
    for (int i = 0; i < str.length(); i++) { if (str.charAt(i) == '\\' && (i + 1) < str.length()) { char c = str.charAt(i + 1); if (target.contains(c)) i++; } sb.append(str.charAt(i)); }
    return sb.toString();
  }
  static String quoteWildcard(String str) { StringBuilder sb = new StringBuilder(); for (char c : str.toCharArray()) { if (WILDCARD_SPECIAL.contains(c)) sb.append('\\'); sb.append(c); } return sb.toString(); }
  /** AutomatonQueries.caseInsensitive{Term,Prefix}Query: ASCII here (the corpus is ASCII). */
  static Query ciTerm(String f, String term, boolean prefix) {
    StringBuilder re = new StringBuilder();
    for (char c : term.toCharArray()) { if (Character.isLetter(c)) re.append('[').append(Character.toLowerCase(c)).append(Character.toUpperCase(c)).append(']'); else if (c == 0) re.append("\\u0000"); else re.append('\\').append(c); }
    Automaton a = Operations.concatenate(List.of(caseAutomaton(term)));
    if (prefix) a = Operations.concatenate(List.of(a, Automata.makeAnyString()));
    return new AutomatonQuery(new Term(f, term), Operations.determinize(a, Operations.DEFAULT_DETERMINIZE_WORK_LIMIT));
  }
  static Automaton caseAutomaton(String s) {
    List<Automaton> l = new ArrayList<>();
    for (int i = 0; i < s.length(); ) { int cp = s.codePointAt(i); i += Character.charCount(cp);
      int lo = Character.toLowerCase(cp), up = Character.toUpperCase(cp);
      l.add(lo == up ? Automata.makeChar(cp) : Operations.union(List.of(Automata.makeChar(lo), Automata.makeChar(up)))); }
    return Operations.concatenate(l);
  }
  static BooleanQuery matchAllTermsQuery(String f, Set<String> terms, boolean ci) {
    BooleanQuery.Builder b = new BooleanQuery.Builder();
    for (String t : terms) {
      Query q = t.length() == NGRAM_SIZE ? (ci ? ciTerm(f, t, false) : new TermQuery(new Term(f, t))) : (ci ? ciTerm(f, t, true) : new PrefixQuery(new Term(f, t)));
      b.add(q, BooleanClause.Occur.FILTER);
    }
    return b.build();
  }
  static Query regexpToQuery(String f, RegExp regExp, boolean ci, List<String> possiblePrefixTerms) {
    BooleanQuery query;
    if (regExp.kind == RegExp.Kind.REGEXP_UNION) {
      List<Query> clauses = new ArrayList<>();
      while (regExp.exp1.kind == RegExp.Kind.REGEXP_UNION) { clauses.add(regexpToQuery(f, regExp.exp2, ci, null)); regExp = regExp.exp1; }
      clauses.add(regexpToQuery(f, regExp.exp2, ci, null)); clauses.add(regexpToQuery(f, regExp.exp1, ci, null));
      BooleanQuery.Builder b = new BooleanQuery.Builder();
      for (int i = clauses.size() - 1; i >= 0; i--) { Query c = clauses.get(i); if (c instanceof MatchAllDocsQuery) return c; b.add(c, BooleanClause.Occur.SHOULD); }
      query = b.build();
    } else if (regExp.kind == RegExp.Kind.REGEXP_STRING) {
      BooleanQuery.Builder b = new BooleanQuery.Builder();
      for (String s : getRequiredNGrams("*" + regExp.s + "*", true)) {
        if (s.length() == NGRAM_SIZE) b.add(ci ? ciTerm(f, s, false) : new TermQuery(new Term(f, s)), BooleanClause.Occur.FILTER);
        else if (possiblePrefixTerms != null) possiblePrefixTerms.add(s);
      }
      query = b.build();
    } else if (regExp.kind == RegExp.Kind.REGEXP_CONCATENATION) {
      List<Query> clauses = new ArrayList<>();
      while (regExp.exp1.kind == RegExp.Kind.REGEXP_CONCATENATION) { clauses.add(regexpToQuery(f, regExp.exp2, ci, possiblePrefixTerms)); regExp = regExp.exp1; }
      clauses.add(regexpToQuery(f, regExp.exp2, ci, possiblePrefixTerms)); clauses.add(regexpToQuery(f, regExp.exp1, ci, possiblePrefixTerms));
      BooleanQuery.Builder b = new BooleanQuery.Builder();
      for (int i = clauses.size() - 1; i >= 0; i--) { Query c = clauses.get(i); if (!(c instanceof MatchAllDocsQuery)) b.add(c, BooleanClause.Occur.FILTER); }
      query = b.build();
    } else if ((regExp.kind == RegExp.Kind.REGEXP_REPEAT_MIN || regExp.kind == RegExp.Kind.REGEXP_REPEAT_MINMAX) && regExp.min > 0) {
      return regexpToQuery(f, regExp.exp1, ci, possiblePrefixTerms);
    } else if (regExp.kind == RegExp.Kind.REGEXP_CHAR && possiblePrefixTerms != null) {
      possiblePrefixTerms.add(String.valueOf((char) regExp.c)); return MatchAllDocsQuery.INSTANCE;
    } else return MatchAllDocsQuery.INSTANCE;
    if (query.clauses().size() == 1) return query.iterator().next().query();
    else if (query.clauses().size() == 0) return MatchAllDocsQuery.INSTANCE;
    return query;
  }

  record Os(Query approximation, Predicate<String> predicate) {}

  static Os wildcard(String f, String value, boolean ci) {
    final String finalValue = ci ? value.toLowerCase(Locale.ROOT) : value;
    Automaton automaton = WildcardQuery.toAutomaton(new Term(f, finalValue), Operations.DEFAULT_DETERMINIZE_WORK_LIMIT);
    CompiledAutomaton ca = new CompiledAutomaton(automaton);
    Predicate<String> p;
    if (ca.type == CompiledAutomaton.AUTOMATON_TYPE.SINGLE) p = s -> (ci ? s.toLowerCase(Locale.ROOT) : s).equals(performEscape(finalValue, false));
    else if (ca.type == CompiledAutomaton.AUTOMATON_TYPE.ALL) p = s -> true;
    else p = s -> { if (ci) s = s.toLowerCase(Locale.ROOT); BytesRef b = new BytesRef(s); return ca.runAutomaton.run(b.bytes, b.offset, b.length); };
    Set<String> grams = getRequiredNGrams(finalValue, false);
    Query approx = grams.isEmpty() ? new FieldExistsQuery(f) : matchAllTermsQuery(f, grams, ci);
    return new Os(approx, p);
  }

  static Os regexp(String f, String value, boolean ci) {
    RegExp regExp = new RegExp(value, RegExp.ALL, ci ? RegExp.ASCII_CASE_INSENSITIVE : 0);
    Automaton automaton = Operations.determinize(regExp.toAutomaton(), Operations.DEFAULT_DETERMINIZE_WORK_LIMIT);
    CompiledAutomaton ca = new CompiledAutomaton(automaton);
    Predicate<String> p;
    if (ca.type == CompiledAutomaton.AUTOMATON_TYPE.SINGLE) p = s -> (ci ? s.toLowerCase(Locale.ROOT) : s).equals(performEscape(value, true));
    else p = s -> { BytesRef b = new BytesRef(s); return ca.runAutomaton.run(b.bytes, b.offset, b.length); };
    List<String> prefixes = new ArrayList<>();
    Query approx = regexpToQuery(f, regExp, ci, prefixes);
    if (approx instanceof MatchAllDocsQuery) {
      if (prefixes.isEmpty()) approx = new FieldExistsQuery(f);
      else { String pt = prefixes.get(0); for (String c : prefixes) if (c.length() > pt.length()) pt = c; approx = ci ? ciTerm(f, pt, true) : new PrefixQuery(new Term(f, pt)); }
    }
    return new Os(approx, p);
  }

  /** WildcardMatchingQuery's scorer: the approximation, then every doc-values value of the candidate through the predicate. */
  static long[] osCount(IndexSearcher s, Os os) throws IOException {
    Weight w = s.createWeight(s.rewrite(os.approximation), ScoreMode.COMPLETE_NO_SCORES, 1f);
    long n = 0, h = 0;
    for (LeafReaderContext ctx : s.getIndexReader().leaves()) {
      Scorer sc = w.scorer(ctx); if (sc == null) continue;
      SortedSetDocValues dv = DocValues.getSortedSet(ctx.reader(), "msg");
      NumericDocValues line = DocValues.getNumeric(ctx.reader(), "line");
      DocIdSetIterator it = sc.iterator();
      for (int d = it.nextDoc(); d != DocIdSetIterator.NO_MORE_DOCS; d = it.nextDoc()) {
        if (!dv.advanceExact(d)) continue;
        boolean match = false;
        for (int k = 0; k < dv.docValueCount() && !match; k++) match = os.predicate.test(dv.lookupOrd(dv.nextOrd()).utf8ToString());
        if (match) { n++; line.advanceExact(d); h += mix(line.longValue()); }
      }
    }
    return new long[] {n, h};
  }

  static long mix(long x) { x *= 0x9E3779B97F4A7C15L; return x ^ (x >>> 31); }

  static Query textQuery(String kind, String p) {
    switch (kind) {
      case "wildcard": return new WildcardQuery(new Term("msg", p));
      case "regexp": return new RegexpQuery(new Term("msg", p));
      case "term": return new TermQuery(new Term("msg", p));
      case "and": { BooleanQuery.Builder b = new BooleanQuery.Builder(); for (String t : p.split(" ")) b.add(new TermQuery(new Term("msg", t)), BooleanClause.Occur.FILTER); return b.build(); }
      default: return null;
    }
  }

  interface Op { long run() throws IOException; }
  static double secs = 3;
  static String time(Op op) throws IOException {
    long[] ts = new long[100000]; int reps = 0; long hits = op.run();
    long warm = System.nanoTime() + (long) (secs * 0.3e9); while (System.nanoTime() < warm) op.run();
    long end = System.nanoTime() + (long) (secs * 1e9);
    while ((System.nanoTime() < end || reps < 3) && reps < ts.length) { long s = System.nanoTime(); op.run(); ts[reps++] = System.nanoTime() - s; }
    Arrays.sort(ts, 0, reps);
    return String.format("%.3f\t%d", ts[reps / 2] / 1e6, hits);
  }

  public static void main(String[] a) throws Exception {
    switch (a[0]) {
      case "build-text": build(a[1], a[2], false); break;
      case "build-os": build(a[1], a[2], true); break;
      case "clean": { new IndexWriter(FSDirectory.open(Paths.get(a[1])), new IndexWriterConfig()).close(); sizes(a[1]); break; }
      case "query": {
        if (a.length > 4) secs = Double.parseDouble(a[4]);
        IndexSearcher ts = new IndexSearcher(DirectoryReader.open(FSDirectory.open(Paths.get(a[1]))));
        IndexSearcher os = new IndexSearcher(DirectoryReader.open(FSDirectory.open(Paths.get(a[2]))));
        ts.setQueryCache(null); os.setQueryCache(null);
        System.out.println("id\ttext_ms\ttext_hits\tos_ms\tos_hits\tos_hash");
        for (String l : Files.readAllLines(Paths.get(a[3]))) {
          if (l.startsWith("#") || l.isBlank()) continue;
          String[] c = l.split("\t");
          boolean ci = c[2].equals("1");
          Os q = switch (c[1]) {
            case "contains" -> wildcard("msg", "*" + quoteWildcard(c[3]) + "*", ci);
            case "wildcard" -> wildcard("msg", c[3], ci);
            case "exact" -> wildcard("msg", quoteWildcard(c[3]), ci);
            default -> regexp("msg", c[3], ci);
          };
          Query tq = textQuery(c[4], c[5]);
          String t = tq == null ? "-\t-" : time(() -> ts.count(tq));
          long[] res = osCount(os, q);
          String o = time(() -> osCount(os, q)[0]);
          System.out.println(c[0] + "\t" + t + "\t" + o + "\t" + Long.toHexString(res[1]));
        }
        break;
      }
    }
  }
}
