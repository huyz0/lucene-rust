import java.io.ByteArrayInputStream;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Deque;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.function.Supplier;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.CharArraySet;
import org.apache.lucene.analysis.core.LowerCaseFilter;
import org.apache.lucene.analysis.core.WhitespaceTokenizer;
import org.apache.lucene.analysis.custom.CustomAnalyzer;
import org.apache.lucene.analysis.pl.PolishAnalyzer;
import org.apache.lucene.analysis.stempel.StempelFilter;
import org.apache.lucene.analysis.stempel.StempelStemmer;
import org.apache.lucene.internal.hppc.CharCursor;
import org.apache.lucene.internal.hppc.CharObjectHashMap;
import org.apache.lucene.util.BytesRef;
import org.egothor.stemmer.Trie;

/**
 * M12 T12.2: Lucene's analysis-stempel module -- the Egothor stemmer tables, {@code
 * StempelStemmer}, {@code StempelFilter}, {@code PolishAnalyzer} and its factory.
 *
 * <p>Writes {@code analysis_stempel/}:
 *
 * <ul>
 *   <li>{@code stems.tsv}: {@code word<TAB>stem|~} through Lucene's default Polish table ({@code
 *       stemmer_20000.tbl}) for edge cases, the training forms below and 30,000 words joined from
 *       Polish stems written here and word endings read off the table's own first trie (its paths
 *       from the root, reversed: the table is a suffix trie).
 *   <li>{@code tables/<method>.tbl}: tables Egothor's own {@code Compile} builds from {@code
 *       corpus/analysis-stempel-train.txt} for each method (forward and backward {@code Trie}s, a
 *       {@code MultiTrie2}, each optimiser), and {@code tables/<method>.tsv}: every training form
 *       and stems word through each (a {@code !Exception} where Java throws).
 *   <li>{@code <chain>.tsv}: {@code PolishAnalyzer} (default, exclusions, no stop words) and
 *       {@code StempelFilter} chains (minimum lengths 1 and 5) over {@code
 *       corpus/analysis-stempel.txt}, {@link AnalysisRows}' rows; {@code factory_*.tsv}: {@code
 *       stempelPolishStem} through {@code CustomAnalyzer}, as {@code GenAnalysisFactories} writes.
 * </ul>
 *
 * Runs with lucene-analysis-stempel on its own classpath ({@code generator_classpath} in {@code
 * scripts/gen-fixtures.sh}). Deterministic. Read by {@code
 * crates/lucene-analysis-stempel/tests/stempel_fixtures.rs}.
 */
public class GenAnalysisStempel {

  /** Polish stems (and a few whole words), written for this project. */
  static final String[] BASES = {
    "kot", "pies", "ps", "dom", "kobiet", "kobie", "miast", "mieś", "dzieck", "dzieci", "rę", "rąk",
    "noc", "mów", "czyt", "pis", "pisz", "był", "dobr", "now", "duż", "szkoł", "szkól", "człowiek",
    "ludz", "rok", "lat", "dzie", "dni", "książk", "książek", "las", "lis", "most", "brat", "siostr",
    "ojc", "matk", "syn", "córk", "drzew", "kwiat", "rzek", "gór", "morz", "słońc", "księżyc",
    "gwiazd", "chleb", "mlek", "wod", "ogień", "ognia", "wiatr", "deszcz", "śnieg", "lód", "lod",
    "zim", "lat", "wiosn", "jesień", "jesieni", "pracow", "rob", "chodz", "jeźdz", "jedz", "pij",
    "spa", "myśl", "wiedz", "widz", "słysz", "kocha", "lubi", "chci", "móg", "mog", "musi", "trzeb",
    "piękn", "brzydk", "mał", "wielk", "star", "młod", "szybk", "woln", "ciepł", "zimn", "jasn",
    "ciemn", "czarn", "biał", "czerwon", "zielon", "niebiesk", "żółt", "polsk", "angielsk",
    "niemieck", "francusk", "komputer", "program", "programist", "telefon", "samochod", "samochód",
    "pociąg", "tramwaj", "autobus", "ulic", "plac", "park", "sklep", "rynek", "rynk", "kości",
    "zamk", "zamek", "król", "królow", "książ", "rycerz", "smok", "wojn", "pokoj", "pokój", "ziem",
    "niebo", "nieb", "piekł", "anioł", "diabł", "święt", "kości", "Warszaw", "Krakow", "Kraków",
    "Gdańsk", "Wrocław", "Poznań", "Łódź", "Łodz", "Polsk", "Europ", "Ameryk", "Azj", "Afryk"
  };

  static final String[] EDGES = {
    "", "a", "ab", "abc", "ę", "ąę", "x", "XYZ", "KOTAMI", "Kotami", "123", "12ab", "a-b", "kot's",
    "😀kot", "kot😀", "żółć", "źdźbło", "chrząszcz", "przeciwdziałanie", "najprawdopodobniej",
    "konstantynopolitańczykowianeczka", "dziewięćdziesięciodziewięcioletni"
  };

  static String stem(StempelStemmer s, String w) {
    try {
      StringBuilder b = s.stem(w);
      return b == null ? "~" : "=" + AnalysisRows.esc(b.toString());
    } catch (Exception e) {
      return "!" + e.getClass().getSimpleName();
    }
  }

  /** Paths of the table's first trie up to {@code depth}, as word endings. */
  @SuppressWarnings("unchecked")
  static List<String> endings(Trie table, int depth) throws Exception {
    java.lang.reflect.Field triesF = Class.forName("org.egothor.stemmer.MultiTrie").getDeclaredField("tries");
    triesF.setAccessible(true);
    Trie first = ((List<Trie>) triesF.get(table)).get(0);
    java.lang.reflect.Field rowsF = Trie.class.getDeclaredField("rows");
    java.lang.reflect.Field rootF = Trie.class.getDeclaredField("root");
    rowsF.setAccessible(true);
    rootF.setAccessible(true);
    List<?> rows = (List<?>) rowsF.get(first);
    Class<?> rowC = Class.forName("org.egothor.stemmer.Row");
    java.lang.reflect.Method getRef = rowC.getDeclaredMethod("getRef", char.class);
    java.lang.reflect.Field cellsF = rowC.getDeclaredField("cells");
    cellsF.setAccessible(true);
    List<String> out = new ArrayList<>();
    Deque<Object[]> queue = new ArrayDeque<>();
    queue.add(new Object[] {rows.get((Integer) rootF.get(first)), ""});
    while (!queue.isEmpty()) {
      Object[] e = queue.poll();
      Object row = e[0];
      String path = (String) e[1];
      // CharObjectHashMap's keys, sorted for a stable order.
      CharObjectHashMap<?> cells = (CharObjectHashMap<?>) cellsF.get(row);
      char[] keys = new char[cells.size()];
      int k = 0;
      for (CharCursor c : cells.keys()) keys[k++] = c.value;
      Arrays.sort(keys);
      for (char c : keys) {
        String p = c + path; // backward trie: the walk reads the word from its end
        out.add(p);
        int ref = (Integer) getRef.invoke(row, c);
        if (ref >= 0 && p.length() < depth) queue.add(new Object[] {rows.get(ref), p});
      }
    }
    return out;
  }

  public static void main(String[] args) throws Exception {
    Path out = Path.of(args[0]).resolve("analysis_stempel");
    Files.createDirectories(out.resolve("tables"));
    String corpusDir = System.getenv().getOrDefault("FIXTURES_CORPUS", "fixtures/corpus");
    List<String> train = Files.readAllLines(Path.of(corpusDir, "analysis-stempel-train.txt"), StandardCharsets.UTF_8);
    List<String> forms = new ArrayList<>();
    for (String l : train) forms.addAll(Arrays.asList(l.trim().split("\\s+")));

    Trie table = PolishAnalyzer.getDefaultTable();
    StempelStemmer polish = new StempelStemmer(table);
    LinkedHashSet<String> words = new LinkedHashSet<>(Arrays.asList(EDGES));
    words.addAll(forms);
    List<String> ends = endings(table, 4);
    Random rnd = new Random(20261007L);
    for (int i = 0; i < 30000; i++) {
      String base = BASES[rnd.nextInt(BASES.length)];
      String end = ends.get(rnd.nextInt(ends.size()));
      words.add(base + end);
    }
    StringBuilder m = new StringBuilder();
    for (String w : words) m.append(AnalysisRows.esc(w)).append('\t').append(stem(polish, w)).append('\n');
    Files.writeString(out.resolve("stems.tsv"), m.toString(), StandardCharsets.UTF_8);

    // Tables built by Egothor's Compile from the training file, one per method.
    Path trainFile = Path.of(corpusDir, "analysis-stempel-train.txt");
    List<String> probe = new ArrayList<>(forms);
    probe.addAll(Arrays.asList(EDGES));
    probe.addAll(Arrays.asList("kotkami", "domkach", "piesek", "czytywali", "dobrzy", "szkółka"));
    for (String method : new String[] {"-0ME2", "-ME", "M1", "-0E", "0L", "-G", "-2", "E"}) {
      Path tmp = Files.createTempDirectory("stempel");
      Path in = tmp.resolve("train.txt");
      Files.copy(trainFile, in);
      java.io.PrintStream stdout = System.out;
      System.setOut(new java.io.PrintStream(java.io.OutputStream.nullOutputStream()));
      try {
        org.egothor.stemmer.Compile.main(new String[] {method, in.toString()});
      } finally {
        System.setOut(stdout);
      }
      byte[] bytes = Files.readAllBytes(tmp.resolve("train.txt.out"));
      String name = method.replace('-', 'b');
      Files.write(out.resolve("tables").resolve(name + ".tbl"), bytes);
      StempelStemmer s;
      try (InputStream is = new ByteArrayInputStream(bytes)) {
        s = new StempelStemmer(is);
      }
      StringBuilder t = new StringBuilder();
      for (String w : probe) t.append(AnalysisRows.esc(w)).append('\t').append(stem(s, w)).append('\n');
      Files.writeString(out.resolve("tables").resolve(name + ".tsv"), t.toString(), StandardCharsets.UTF_8);
      Files.delete(tmp.resolve("train.txt.out"));
      Files.delete(in);
      Files.delete(tmp);
    }

    List<String> lines = AnalysisRows.corpus("analysis-stempel.txt");
    Map<String, Supplier<Analyzer>> chains = new LinkedHashMap<>();
    chains.put("polish", PolishAnalyzer::new);
    chains.put("polish_exclusions", () -> new PolishAnalyzer(PolishAnalyzer.getDefaultStopSet(),
        new CharArraySet(List.of("kot", "domu", "dzieci", "Łodzi"), false)));
    chains.put("polish_nostop", () -> new PolishAnalyzer(CharArraySet.EMPTY_SET));
    chains.put("stempel_min1", () -> AnalysisRows.chain(WhitespaceTokenizer::new,
        t -> new StempelFilter(new LowerCaseFilter(t), new StempelStemmer(table), 1)));
    chains.put("stempel_min5", () -> AnalysisRows.chain(WhitespaceTokenizer::new,
        t -> new StempelFilter(t, new StempelStemmer(table), 5)));
    AnalysisRows.writeChains(out, chains, lines);

    String[][] factories = {
      {"factory_stempel", "tok:whitespace", "tf:lowercase", "tf:stempelPolishStem"},
      {"factory_stempel_keyword", "tok:whitespace", "tf:keywordMarker", "pattern=[A-Z].*", "tf:stempelPolishStem"},
      {"factory_stempel_param", "tok:whitespace", "tf:stempelPolishStem", "x=1"},
    };
    for (String[] fields : factories) {
      StringBuilder f = new StringBuilder();
      CustomAnalyzer a;
      try {
        a = GenAnalysisFactories.build(GenAnalysisFactories.parse(fields), Path.of(corpusDir));
      } catch (Exception | Error e) {
        f.append("B\t").append(e.getClass().getSimpleName()).append('\t')
            .append(AnalysisRows.esc(GenAnalysisFactories.stable(e))).append('\n');
        Files.writeString(out.resolve(fields[0] + ".tsv"), f.toString(), StandardCharsets.UTF_8);
        continue;
      }
      try (a) {
        f.append("S\t").append(a.toString().replaceAll("@[0-9a-f]+", "")).append('\n');
        f.append(AnalysisRows.rows(a, lines));
        for (int ln = 0; ln < lines.size(); ln++) {
          f.append("N\t").append(ln).append('\t');
          BytesRef n = a.normalize("f", lines.get(ln));
          f.append(AnalysisRows.hex(n)).append('\n');
        }
      }
      Files.writeString(out.resolve(fields[0] + ".tsv"), f.toString(), StandardCharsets.UTF_8);
    }
  }
}
