import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.BitSet;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.apache.lucene.util.BytesRef;
import org.apache.lucene.util.BytesRefBuilder;
import org.apache.lucene.util.IntsRef;
import org.apache.lucene.util.automaton.Automata;
import org.apache.lucene.util.automaton.Automaton;
import org.apache.lucene.util.automaton.ByteRunnable;
import org.apache.lucene.util.automaton.CharacterRunAutomaton;
import org.apache.lucene.util.automaton.CompiledAutomaton;
import org.apache.lucene.util.automaton.LevenshteinAutomata;
import org.apache.lucene.util.automaton.LimitedFiniteStringsIterator;
import org.apache.lucene.util.automaton.NFARunAutomaton;
import org.apache.lucene.util.automaton.Operations;
import org.apache.lucene.util.automaton.RegExp;
import org.apache.lucene.util.automaton.TooComplexToDeterminizeException;
import org.apache.lucene.util.automaton.Transition;

/**
 * Lucene 10.5.0's public automaton API over a corpus of regexps, Levenshtein words, operations and
 * factories, for crates/lucene-util/tests/automaton_fixtures.rs.
 *
 * <p>{@code automata.tsv}: a {@code PROBES} line of global probe strings, then one block per case:
 * a {@code CASE} line naming the construction and its inputs, a {@code probes} line of extra
 * case-specific probe strings, then result lines the Rust test recomputes and compares verbatim.
 * Strings are code points, printable ASCII as itself and everything else (plus backslash and tab)
 * as a backslash, {@code u}, then the hex code point in braces.
 *
 * <p>Per automaton ({@code block}): raw state/transition counts and determinism, the full raw
 * transition table when small, dead-state/empty/total/finite flags, {@code
 * determinize(DEFAULT_DETERMINIZE_WORK_LIMIT)}'s counts and table (or the
 * TooComplexToDeterminizeException message), AutomatonTestUtil.minimizeSimple's counts, probe
 * acceptance through Operations.run, CharacterRunAutomaton, NFARunAutomaton and the
 * CompiledAutomaton's ByteRunnable, common prefix, singleton, topological order and finite
 * strings, and the CompiledAutomaton's type, term, common suffix, sink state and floor() of every
 * probe.
 */
public class GenAutomata {
  static final StringBuilder OUT = new StringBuilder();
  static final List<String> PROBES =
      List.of(
          "", "a", "b", "c", "d", "e", "ab", "ac", "ad", "abc", "abd", "aa", "aaa", "aaaa", "aaaaa",
          "z", "q", "x", "xy", "xyz", "xz", "abcd", "abde", "acd", "cd", "ba", "bb", "aab", "abab",
          "abba", "bab", "foo", "bar", "baz", "fo", "foobar", "foofoobar", "ing", "singing", "az",
          "a z", "azq", "qaz", "1", "5", "9", "10", "010", "05", "0005", "17", "100", "101", "500",
          "501", "0500", "99", "07", "7", "1234", "01234", "quick", "QUICK", "QuIcK", "Quick", "K",
          "k", "K", "S", "s", "ſ", "Σ", "σ", "ς", "Ǆ", "ǅ",
          "ǆ", "straße", "STRASSE", "Straße", "é", "éé", "É",
          "日本語", "日本", "😀", "😀😀",
          "😁", "😂😃", "\t", " ", "_", "-", "\\", ".", "quo\\ted",
          "~a", "a&b", "#", "@", "<1-5>", "a~b", "Hello", "hello", "HELLO", "kS", "KS", "Ks",
          "abcing", "a.b", "a|", "*a", "?", "hello world", "cat", "cats", "dog", "dogs", "ca",
          "a\u0000", "a\u0000\u0000", "\u0000", "ae", "ace", "abee", "xyzzy", "ÿ");

  static String esc(int[] cps) {
    StringBuilder b = new StringBuilder();
    for (int c : cps) {
      if (c >= 0x20 && c < 0x7f && c != '\\') {
        b.append((char) c);
      } else {
        b.append("\\u{").append(Integer.toHexString(c)).append('}');
      }
    }
    return b.toString();
  }

  static String esc(String s) {
    return esc(s.codePoints().toArray());
  }

  static String hex(BytesRef b) {
    if (b == null) return "-";
    StringBuilder s = new StringBuilder("x");
    for (int i = 0; i < b.length; i++) {
      s.append(String.format("%02x", b.bytes[b.offset + i] & 0xff));
    }
    return s.toString();
  }

  static void line(Object... fields) {
    for (int i = 0; i < fields.length; i++) {
      if (i > 0) OUT.append('\t');
      OUT.append(fields[i]);
    }
    OUT.append('\n');
  }

  static int[] labels(String s, boolean binary) {
    if (binary) {
      byte[] b = s.getBytes(StandardCharsets.UTF_8);
      int[] r = new int[b.length];
      for (int i = 0; i < b.length; i++) r[i] = b[i] & 0xff;
      return r;
    }
    return s.codePoints().toArray();
  }

  static String dump(Automaton a) {
    StringBuilder b = new StringBuilder();
    Transition t = new Transition();
    for (int s = 0; s < a.getNumStates(); s++) {
      if (s > 0) b.append(';');
      b.append(a.isAccept(s) ? 'A' : 'N');
      int n = a.initTransition(s, t);
      for (int i = 0; i < n; i++) {
        a.getNextTransition(t);
        b.append(',').append(t.dest).append('/').append(t.min).append('/').append(t.max);
      }
    }
    return b.toString();
  }

  static String flag(boolean b) {
    return b ? "1" : "0";
  }

  // ---- AutomatonTestUtil (lucene-test-framework) copies ------------------------------------

  static Automaton reverseOriginal(Automaton a, Set<Integer> initialStates) {
    if (Operations.isEmpty(a)) {
      return new Automaton();
    }
    int numStates = a.getNumStates();
    Automaton.Builder builder = new Automaton.Builder();
    builder.createState();
    for (int s = 0; s < numStates; s++) {
      builder.createState();
    }
    builder.setAccept(1, true);
    Transition t = new Transition();
    for (int s = 0; s < numStates; s++) {
      int numTransitions = a.getNumTransitions(s);
      a.initTransition(s, t);
      for (int i = 0; i < numTransitions; i++) {
        a.getNextTransition(t);
        builder.addTransition(t.dest + 1, s + 1, t.min, t.max);
      }
    }
    Automaton result = builder.finish();
    int s = 0;
    BitSet acceptStates = a.getAcceptStates();
    while (s < numStates && (s = acceptStates.nextSetBit(s)) != -1) {
      result.addEpsilon(0, s + 1);
      if (initialStates != null) {
        initialStates.add(s + 1);
      }
      s++;
    }
    result.finishState();
    return result;
  }

  static Automaton minimizeSimple(Automaton a) {
    Set<Integer> initialSet = new HashSet<Integer>();
    a = determinizeSimple(reverseOriginal(a, initialSet), initialSet);
    initialSet.clear();
    a = determinizeSimple(reverseOriginal(a, initialSet), initialSet);
    return a;
  }

  static Automaton determinizeSimple(Automaton a, Set<Integer> initialset) {
    if (a.getNumStates() == 0) {
      return a;
    }
    int[] points = a.getStartPoints();
    Map<Set<Integer>, Set<Integer>> sets = new HashMap<>();
    LinkedList<Set<Integer>> worklist = new LinkedList<>();
    Map<Set<Integer>, Integer> newstate = new HashMap<>();
    sets.put(initialset, initialset);
    worklist.add(initialset);
    Automaton.Builder result = new Automaton.Builder();
    result.createState();
    newstate.put(initialset, 0);
    Transition t = new Transition();
    while (worklist.size() > 0) {
      Set<Integer> s = worklist.removeFirst();
      int r = newstate.get(s);
      for (int q : s) {
        if (a.isAccept(q)) {
          result.setAccept(r, true);
          break;
        }
      }
      for (int n = 0; n < points.length; n++) {
        Set<Integer> p = new HashSet<>();
        for (int q : s) {
          int count = a.initTransition(q, t);
          for (int i = 0; i < count; i++) {
            a.getNextTransition(t);
            if (t.min <= points[n] && points[n] <= t.max) {
              p.add(t.dest);
            }
          }
        }
        if (!sets.containsKey(p)) {
          sets.put(p, p);
          worklist.add(p);
          newstate.put(p, result.createState());
        }
        int q = newstate.get(p);
        int min = points[n];
        int max;
        if (n + 1 < points.length) {
          max = points[n + 1] - 1;
        } else {
          max = Character.MAX_CODE_POINT;
        }
        result.addTransition(r, q, min, max);
      }
    }
    return Operations.removeDeadStates(result.finish());
  }

  static boolean isFinite(Automaton a) {
    if (a.getNumStates() == 0) {
      return true;
    }
    return isFinite(
        new Transition(), a, 0, new BitSet(a.getNumStates()), new BitSet(a.getNumStates()), 0);
  }

  static boolean isFinite(
      Transition scratch, Automaton a, int state, BitSet path, BitSet visited, int level) {
    if (level > 1000) {
      throw new IllegalArgumentException("input automaton is too large: " + level);
    }
    path.set(state);
    int numTransitions = a.initTransition(state, scratch);
    for (int t = 0; t < numTransitions; t++) {
      a.getTransition(state, t, scratch);
      if (path.get(scratch.dest)
          || (!visited.get(scratch.dest)
              && !isFinite(scratch, a, scratch.dest, path, visited, level + 1))) {
        return false;
      }
    }
    path.clear(state);
    visited.set(state);
    return true;
  }

  // ---- the per-automaton block ---------------------------------------------------------------

  static String bits(List<String> probes, java.util.function.Predicate<String> accepts) {
    StringBuilder b = new StringBuilder();
    for (String p : probes) b.append(accepts.test(p) ? '1' : '0');
    return b.toString();
  }

  static boolean nfaRun(ByteRunnable r, int[] labels) {
    int p = 0;
    for (int c : labels) {
      p = r.step(p, c);
      if (p == -1) return false;
    }
    return r.isAccept(p);
  }

  static void compiled(String key, Automaton a, boolean binary, List<String> probes, boolean finite) {
    CompiledAutomaton c = new CompiledAutomaton(a, false, true, binary);
    String term = c.term == null ? "-" : hex(c.term);
    String runBits = "-";
    String floors = "-";
    if (c.type == CompiledAutomaton.AUTOMATON_TYPE.NORMAL) {
      ByteRunnable r = c.getByteRunnable();
      runBits =
          bits(
              probes,
              p -> {
                byte[] b = p.getBytes(StandardCharsets.UTF_8);
                return r.run(b, 0, b.length);
              });
      // floor() walks max-label transitions to a leaf, so only a finite language terminates.
      if (c.runAutomaton != null && finite) {
        StringBuilder f = new StringBuilder();
        for (String p : probes) {
          if (f.length() > 0) f.append(',');
          BytesRef fl = c.floor(new BytesRef(p), new BytesRefBuilder());
          f.append(fl == null ? "null" : hex(fl));
        }
        floors = f.toString();
      }
    }
    line(
        key,
        c.type,
        term,
        hex(c.commonSuffixRef),
        c.sinkState,
        flag(c.finite),
        flag(c.type == CompiledAutomaton.AUTOMATON_TYPE.NORMAL && c.runAutomaton == null),
        runBits);
    line(key + "_floor", floors);
  }

  static void block(Automaton a, boolean binary, List<String> probes) {
    line("raw", a.getNumStates(), a.getNumTransitions(), flag(a.isDeterministic()));
    if (a.getNumStates() <= 40) line("rawdump", dump(a));
    line(
        "flags",
        flag(Operations.hasDeadStates(a)),
        flag(Operations.isEmpty(a)),
        flag(binary ? Operations.isTotal(a, 0, 255) : Operations.isTotal(a)),
        flag(isFinite(a)));
    NFARunAutomaton nfa = binary ? new NFARunAutomaton(a, 256) : new NFARunAutomaton(a);
    line("nfarun", bits(probes, p -> nfaRun(nfa, labels(p, binary))));
    compiled("compiled", a, binary, probes, isFinite(a));
    Automaton det;
    try {
      det = Operations.determinize(a, Operations.DEFAULT_DETERMINIZE_WORK_LIMIT);
    } catch (TooComplexToDeterminizeException e) {
      line("det", "TOO_COMPLEX", e.getMessage());
      return;
    }
    line("det", det.getNumStates(), det.getNumTransitions(), flag(det.isDeterministic()));
    if (det.getNumStates() <= 40) line("detdump", dump(det));
    if (det.getNumStates() <= 2000) {
      Automaton min = minimizeSimple(det);
      line("min", min.getNumStates(), min.getNumTransitions());
    }
    line("run", bits(probes, p -> Operations.run(det, new IntsRef(labels(p, binary), 0, labels(p, binary).length))));
    if (!binary) {
      CharacterRunAutomaton cra = new CharacterRunAutomaton(det);
      line("charrun", bits(probes, cra::run));
    }
    try {
      line("prefix", esc(Operations.getCommonPrefix(det)));
    } catch (IllegalArgumentException e) {
      line("prefix", "ERR", e.getMessage());
    }
    IntsRef single = Operations.getSingleton(det);
    line("singleton", single == null ? "null" : esc(Arrays.copyOfRange(single.ints, single.offset, single.offset + single.length)));
    boolean finite = isFinite(det);
    if (finite) {
      if (det.getNumStates() <= 60) {
        StringBuilder b = new StringBuilder();
        for (int s : Operations.topoSortStates(det)) {
          if (b.length() > 0) b.append(',');
          b.append(s);
        }
        line("topo", b);
      }
      LimitedFiniteStringsIterator it = new LimitedFiniteStringsIterator(det, 50);
      List<String> strs = new ArrayList<>();
      for (IntsRef s = it.next(); s != null; s = it.next()) {
        strs.add(esc(Arrays.copyOfRange(s.ints, s.offset, s.offset + s.length)));
      }
      List<Object> fields = new ArrayList<>();
      fields.add("strings");
      fields.add(strs.size());
      fields.addAll(strs);
      line(fields.toArray());
    }
    compiled("compiled_det", det, binary, probes, finite);
  }

  static void startCase(List<String> extraProbes, Object... fields) {
    Object[] f = new Object[fields.length + 1];
    f[0] = "CASE";
    System.arraycopy(fields, 0, f, 1, fields.length);
    line(f);
    List<Object> p = new ArrayList<>();
    p.add("probes");
    p.add(extraProbes.size());
    for (String s : extraProbes) p.add(esc(s));
    line(p.toArray());
  }

  static List<String> allProbes(List<String> extra) {
    List<String> all = new ArrayList<>(PROBES);
    all.addAll(extra);
    return all;
  }

  // ---- case families -------------------------------------------------------------------------

  static void regexp(String pattern, int syntax, int match) {
    startCase(List.of(), "regexp", esc(pattern), syntax, match);
    RegExp re;
    try {
      re = new RegExp(pattern, syntax, match);
    } catch (IllegalArgumentException e) {
      line("parse", "ERR", esc(e.getMessage()));
      return;
    }
    line("parse", "OK", esc(re.toString()));
    line("tree", esc(re.toStringTree()));
    Automaton a;
    try {
      a = re.toAutomaton();
    } catch (TooComplexToDeterminizeException e) {
      line("auto", "TOO_COMPLEX", esc(e.getMessage()));
      return;
    } catch (IllegalArgumentException e) {
      line("auto", "ERR", esc(e.getMessage()));
      return;
    }
    block(a, false, PROBES);
  }

  static List<String> edits(String word) {
    int[] w = word.codePoints().toArray();
    LinkedHashSet<String> out = new LinkedHashSet<>();
    out.add(word);
    int[] subs = {'x', w.length > 0 ? w[0] : 'y'};
    for (int i = 0; i <= w.length; i++) {
      for (int c : subs) {
        int[] ins = new int[w.length + 1];
        System.arraycopy(w, 0, ins, 0, i);
        ins[i] = c;
        System.arraycopy(w, i, ins, i + 1, w.length - i);
        out.add(new String(ins, 0, ins.length));
      }
      if (i < w.length) {
        int[] del = new int[w.length - 1];
        System.arraycopy(w, 0, del, 0, i);
        System.arraycopy(w, i + 1, del, i, w.length - i - 1);
        out.add(new String(del, 0, del.length));
        int[] sub = w.clone();
        sub[i] = 'x';
        out.add(new String(sub, 0, sub.length));
      }
      if (i + 1 < w.length) {
        int[] tr = w.clone();
        tr[i] = w[i + 1];
        tr[i + 1] = w[i];
        out.add(new String(tr, 0, tr.length));
      }
    }
    // Two and three edits: drop the first two / three, and swap-then-insert.
    if (w.length >= 2) out.add(new String(w, 2, w.length - 2));
    if (w.length >= 3) out.add(new String(w, 3, w.length - 3));
    out.add("xx" + word);
    out.add("xxx" + word);
    out.add(word + "yy");
    if (w.length >= 2) {
      int[] tr = w.clone();
      tr[0] = w[1];
      tr[1] = w[0];
      out.add(new String(tr, 0, tr.length) + "x");
    }
    return new ArrayList<>(out);
  }

  static void lev(String word, int n, boolean transpositions, String prefix) {
    List<String> extra = new ArrayList<>();
    for (String e : edits(word)) {
      extra.add(e);
      if (!prefix.isEmpty()) extra.add(prefix + e);
    }
    startCase(extra, "lev", esc(word), n, flag(transpositions), esc(prefix));
    Automaton a = new LevenshteinAutomata(word, transpositions).toAutomaton(n, prefix);
    if (a == null) {
      line("lev", "null");
      return;
    }
    block(a, false, allProbes(extra));
  }

  static Automaton re(String s) {
    return new RegExp(s).toAutomaton();
  }

  static void op(String name, String... args) throws Exception {
    startCase(List.of(), "op", name, String.join("\t", Arrays.stream(args).map(GenAutomata::esc).toArray(String[]::new)));
    Automaton a;
    int limit = Operations.DEFAULT_DETERMINIZE_WORK_LIMIT;
    try {
      switch (name) {
        case "concat" -> {
          List<Automaton> l = new ArrayList<>();
          for (String s : args) l.add(re(s));
          a = Operations.concatenate(l);
        }
        case "union" -> {
          List<Automaton> l = new ArrayList<>();
          for (String s : args) l.add(re(s));
          a = Operations.union(l);
        }
        case "intersection" -> a = Operations.intersection(re(args[0]), re(args[1]));
        case "minus" -> a = Operations.minus(re(args[0]), re(args[1]), limit);
        case "complement" -> a = Operations.complement(re(args[0]), limit);
        case "optional" -> a = Operations.optional(re(args[0]));
        case "repeat" -> a = Operations.repeat(re(args[0]));
        case "repeatmin" -> a = Operations.repeat(re(args[0]), Integer.parseInt(args[1]));
        case "repeatrange" ->
            a = Operations.repeat(re(args[0]), Integer.parseInt(args[1]), Integer.parseInt(args[2]));
        case "reverse" -> a = Operations.reverse(re(args[0]));
        default -> throw new IllegalStateException(name);
      }
    } catch (TooComplexToDeterminizeException e) {
      line("op", "TOO_COMPLEX", esc(e.getMessage()));
      return;
    }
    block(a, false, PROBES);
  }

  static BytesRef bytes(String hexOrNull) {
    if (hexOrNull.equals("null")) return null;
    byte[] b = new byte[hexOrNull.length() / 2];
    for (int i = 0; i < b.length; i++) {
      b[i] = (byte) Integer.parseInt(hexOrNull.substring(2 * i, 2 * i + 2), 16);
    }
    return new BytesRef(b);
  }

  static void automata(String name, String... args) {
    startCase(List.of(), "automata", name, String.join("\t", Arrays.stream(args).map(GenAutomata::esc).toArray(String[]::new)));
    Automaton a;
    boolean binary = false;
    try {
      switch (name) {
        case "string" -> a = Automata.makeString(args[0]);
        case "ci_string" -> a = Automata.makeCaseInsensitiveString(args[0]);
        case "char_range" -> a = Automata.makeCharRange(Integer.parseInt(args[0]), Integer.parseInt(args[1]));
        case "decimal" ->
            a = Automata.makeDecimalInterval(Integer.parseInt(args[0]), Integer.parseInt(args[1]), Integer.parseInt(args[2]));
        case "any_string" -> a = Automata.makeAnyString();
        case "any_char" -> a = Automata.makeAnyChar();
        case "empty" -> a = Automata.makeEmpty();
        case "empty_string" -> a = Automata.makeEmptyString();
        case "char_set" -> a = Automata.makeCharSet(Arrays.stream(args).mapToInt(Integer::parseInt).toArray());
        case "any_binary" -> {
          a = Automata.makeAnyBinary();
          binary = true;
        }
        case "non_empty_binary" -> {
          a = Automata.makeNonEmptyBinary();
          binary = true;
        }
        case "binary" -> {
          a = Automata.makeBinary(bytes(args[0]));
          binary = true;
        }
        case "binary_interval" -> {
          a = Automata.makeBinaryInterval(bytes(args[0]), args[1].equals("1"), bytes(args[2]), args[3].equals("1"));
          binary = true;
        }
        case "string_union", "binary_string_union" -> {
          List<BytesRef> l = new ArrayList<>();
          for (String s : args) l.add(new BytesRef(s));
          binary = name.startsWith("binary");
          a = binary ? Automata.makeBinaryStringUnion(l) : Automata.makeStringUnion(l);
        }
        default -> throw new IllegalStateException(name);
      }
    } catch (IllegalArgumentException e) {
      line("automata", "ERR", e.getMessage() == null ? "null" : esc(e.getMessage()));
      return;
    }
    block(a, binary, PROBES);
  }

  public static void main(String[] argv) throws Exception {
    Path out = Path.of(argv[0]).resolve("automata");
    Files.createDirectories(out);
    List<Object> p = new ArrayList<>();
    p.add("PROBES");
    p.add(PROBES.size());
    for (String s : PROBES) p.add(esc(s));
    line(p.toArray());

    int all = RegExp.ALL;
    String[] patterns = {
      "", "a", "abc", "a|b", "ab|ac", "a*", "a+", "a?", "(ab)*c", "a{2}", "a{2,}", "a{2,4}", "a{0,2}",
      "a{0}", "[a-z]", "[^a-z]", "[abc]", "[a-cx-z]", "[^abc]d", ".", ".*", "a.*", ".*ing", "a.*z",
      "@", "#", "a#", "a&b", "[a-z]+&.*q.*", "\"quo\\ted\"", "<1-10>", "<01-10>", "<5-500>",
      "<0-99>", "<0500-0501>", "<17-1234>", "<name>", "\\d+", "\\D", "\\s*", "\\S", "\\w+", "\\W",
      "[\\d]", "[\\w-]", "\\\\", "\\.", "é+", "日本.", "😀+",
      "[😀-😂]", "(a|b)*a(a|b){3}", "(a|b)*a(a|b){15}", "foo|bar|baz|fo",
      "(foo)+bar", "x(y|z)*", "[a-c]{3}", "a(b|c)d?e*", "((a))", "a|", "*a", "?", "[]", "a{",
      "a{2", "a{3,2}", "[a", "(a", "\"a", "<a", "<1-2-3>", "a)", "[z-a]", "\\p", "a\\",
      "a{99999999999}", "()", "a()b", "[\\\\]", "a|b|c|d|e|f", "(a|ab)(c|bcd)(d*)",
      "[^\\d]", "[a-z&&[b]]", "(cat|dog)s?", "hello world", ".{2,3}", "x{0,}", "a{1,1}",
      "\"\"", "[-a]", "[a-]", "\\S+\\s\\S+", "<-5>", "<5->", "<1-5>x", "~a", "a~b",
    };
    for (String s : patterns) regexp(s, all, 0);
    for (String s : new String[] {"a&b", "#", "@", "<1-5>", "a.*"}) regexp(s, RegExp.NONE, 0);
    for (String s : new String[] {"a&b", "#a", "<1-5>", "<name>"}) regexp(s, RegExp.INTERSECTION, 0);
    regexp("<name>", RegExp.INTERVAL, 0);
    regexp("<1-5>", RegExp.AUTOMATON, 0);
    int dc = all | RegExp.DEPRECATED_COMPLEMENT;
    for (String s : new String[] {"~a", "~(ab)", "a~b", "~(a|b)*", "~~a", "[a-z]+&~(.*q.*)", "~((a|b)*a(a|b){15})"}) {
      regexp(s, dc, 0);
    }
    for (String s : new String[] {"quick", "QuIcK", "[abc]", "[a-c]", "k", "s", "σ", "ǅ", "straße", "hello world", "k+s", "[^k]"}) {
      regexp(s, all, RegExp.CASE_INSENSITIVE);
    }
    for (String s : new String[] {"[a-c]", "[k-m]", "[a-cQ]"}) {
      regexp(s, all, RegExp.CASE_INSENSITIVE_RANGE);
      regexp(s, all, RegExp.CASE_INSENSITIVE | RegExp.CASE_INSENSITIVE_RANGE);
    }
    regexp("Hello", all, RegExp.ASCII_CASE_INSENSITIVE);
    regexp("a", 0x100, 0);
    regexp("a", all, 0x01);

    String[] words = {"", "a", "ab", "abc", "kitten", "lucene", "日本", "😀x", "aaaa", "abcdefghij"};
    for (String w : words) {
      for (int n = 0; n <= 3; n++) {
        lev(w, n, false, "");
        lev(w, n, true, "");
      }
    }
    lev("abc", 1, false, "pre");
    lev("abc", 2, true, "pre");
    lev("kitten", 1, true, "é");

    op("concat", "a*", "b");
    op("concat", "ab", "");
    op("concat", "a|b", "c|d", "e");
    op("concat", "a?", "b?", "c?");
    op("concat", "a", "#");
    op("union", "abc", "abd", "ab");
    op("union", "a*", "b*");
    op("union", "#", "a");
    op("union", "a", "b", "c", "");
    op("intersection", "a.*", ".*b");
    op("intersection", "[a-z]+", "x.*");
    op("intersection", "a", "b");
    op("minus", "[a-z]+", "abc");
    op("minus", ".*", "a.*");
    op("minus", "a", "#");
    op("minus", "#", "a");
    op("minus", "(a|b)*", "(a|b)*a(a|b){15}");
    op("complement", "a*");
    op("complement", "abc");
    op("complement", "(a|b)*a(a|b){15}");
    op("optional", "a");
    op("optional", "a*b");
    op("optional", "(ab)*");
    op("optional", "(ab)+");
    op("repeat", "ab");
    op("repeat", "a?b");
    op("repeat", "a*");
    op("repeat", "a|bc*");
    op("repeatmin", "ab", "2");
    op("repeatmin", "a|bc", "0");
    op("repeatrange", "a|bc", "1", "3");
    op("repeatrange", "ab", "0", "2");
    op("repeatrange", "a", "3", "2");
    op("repeatrange", "a*b", "2", "4");
    op("reverse", "abc|de");
    op("reverse", "a*b");
    op("reverse", "#");

    automata("string", "hello");
    automata("string", "");
    automata("string", "日😀");
    automata("ci_string", "Straße");
    automata("ci_string", "kS");
    automata("char_range", "97", "122");
    automata("char_range", "128512", "128591");
    automata("char_range", "5", "4");
    automata("decimal", "1", "100", "0");
    automata("decimal", "5", "500", "3");
    automata("decimal", "0", "0", "0");
    automata("decimal", "17", "1234", "0");
    automata("decimal", "123", "45678", "5");
    automata("decimal", "5", "1", "0");
    automata("decimal", "1", "500", "2");
    automata("any_string");
    automata("any_char");
    automata("empty");
    automata("empty_string");
    automata("char_set", "97", "99", "101");
    automata("any_binary");
    automata("non_empty_binary");
    automata("binary", "61ff00");
    automata("binary_interval", "null", "1", "null", "1");
    automata("binary_interval", "", "0", "null", "1");
    automata("binary_interval", "6162", "1", "6164", "0");
    automata("binary_interval", "61", "0", "610000", "1");
    automata("binary_interval", "61", "1", "610000", "0");
    automata("binary_interval", "616263", "1", "null", "1");
    automata("binary_interval", "616263", "0", "null", "1");
    automata("binary_interval", "61", "1", "62", "1");
    automata("binary_interval", "null", "1", "6d", "0");
    automata("binary_interval", "62", "1", "61", "1");
    automata("binary_interval", "61", "1", "61", "1");
    automata("binary_interval", "61", "1", "61", "0");
    automata("binary_interval", "6100", "1", "62ff", "1");
    automata("binary_interval", "7a", "0", "7a7a7a", "0");
    automata("binary_interval", "null", "0", "61", "1");
    automata("string_union", "cat", "cats", "dog", "dogs", "été");
    automata("string_union", "a", "ab", "abc", "b");
    automata("string_union");
    automata("string_union", "", "a");
    automata("string_union", "b", "a");
    automata("binary_string_union", "cat", "cats", "dog", "dogs", "été");
    automata("binary_string_union", "", "hello", "help", "hello");

    Files.writeString(out.resolve("automata.tsv"), OUT.toString());
  }
}
