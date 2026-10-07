import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.TreeSet;
import org.apache.lucene.analysis.CharFilterFactory;
import org.apache.lucene.analysis.TokenFilterFactory;
import org.apache.lucene.analysis.TokenizerFactory;
import org.apache.lucene.analysis.custom.CustomAnalyzer;
import org.apache.lucene.util.BytesRef;
import org.apache.lucene.util.Version;

/**
 * M11 T11.7: the analysis factories and {@code CustomAnalyzer}, built from configuration text.
 *
 * <p>{@code corpus/analysis-factories.conf} holds one configuration per line: a name, then the
 * builder steps ({@code tok:}, {@code cf:}, {@code tf:}, {@code when:}, {@code whenTerm:N} -- a
 * term longer than N chars --, {@code endwhen}, {@code version:}, {@code posgap:}, {@code
 * offgap:}), each followed by its {@code key=value} arguments. Every configuration is built with
 * {@code CustomAnalyzer.builder(corpus/analysis-factories)} and written to {@code
 * analysis_factories/<name>.tsv}: either {@code B<TAB>ExceptionSimpleName<TAB>message} when the
 * build throws, or {@code S<TAB>toString} (identity hashes removed), {@link AnalysisRows}' rows over
 * {@code corpus/analysis-factories.txt}, and {@code N<TAB>line<TAB>hex|X:Exception} for {@code
 * normalize("f", line)}. Three messages are trimmed where Java's own text changes from run to run:
 * the SPI loader's list of names (a {@code Set.copyOf}), Hunspell's list of input streams, and a
 * SAX parse error's location (a system id resolved against the working directory, and the line
 * and column the reader's buffer had reached); a {@code StringIndexOutOfBoundsException}'s
 * message, the JDK's own wording, is left out.
 * {@code names.txt} lists every SPI name of the three kinds ({@code T}, {@code F}, {@code C}),
 * sorted. {@code crates/lucene-analysis/tests/analysis_factory_fixtures.rs} compares (the {@code
 * Word2VecSynonym} configurations in {@code lucene-search}, where that filter lives).
 */
public class GenAnalysisFactories {

  static String unesc(String s) {
    StringBuilder b = new StringBuilder();
    for (int i = 0; i < s.length(); i++) {
      char c = s.charAt(i);
      if (c != '\\' || i + 1 == s.length()) {
        b.append(c);
        continue;
      }
      char n = s.charAt(++i);
      switch (n) {
        case 't' -> b.append('\t');
        case 'n' -> b.append('\n');
        case 'r' -> b.append('\r');
        case 'u' -> {
          b.append((char) Integer.parseInt(s.substring(i + 1, i + 5), 16));
          i += 4;
        }
        default -> b.append(n);
      }
    }
    return b.toString();
  }

  /**
   * Java's messages that differ from run to run, cut where they start to; a {@code
   * StringIndexOutOfBoundsException}'s, whose wording is the JDK's, is dropped, and a SAX parse
   * error loses its location.
   */
  static String stable(Throwable e) {
    String message = e.getMessage();
    if (e instanceof StringIndexOutOfBoundsException) return "";
    if (message == null) return "null";
    int i = message.indexOf("The current classpath supports the following names: ");
    if (i >= 0) return message.substring(0, i);
    if (message.startsWith("Unable to load hunspell data!")) return "Unable to load hunspell data!";
    // A SAX parse error's location: the system id the JDK resolved against the working
    // directory, and the line and column its reader's buffer had reached.
    return message.replaceFirst(
        "^(org\\.xml\\.sax\\.SAXParseException; )systemId: [^;]*; lineNumber: -?\\d+; columnNumber:"
            + " -?\\d+; ",
        "$1");
  }

  static final class Step {
    final String kind;
    final String name;
    final List<String> params = new ArrayList<>();

    Step(String kind, String name) {
      this.kind = kind;
      this.name = name;
    }
  }

  static List<Step> parse(String[] fields) {
    List<Step> steps = new ArrayList<>();
    for (int i = 1; i < fields.length; i++) {
      String f = fields[i];
      int colon = f.indexOf(':');
      int eq = f.indexOf('=');
      if (f.equals("endwhen")) {
        steps.add(new Step("endwhen", null));
      } else if (colon > 0 && (eq < 0 || colon < eq)) {
        steps.add(new Step(f.substring(0, colon), unesc(f.substring(colon + 1))));
      } else {
        Step last = steps.get(steps.size() - 1);
        last.params.add(f.substring(0, eq));
        last.params.add(unesc(f.substring(eq + 1)));
      }
    }
    return steps;
  }

  static CustomAnalyzer build(List<Step> steps, Path dir) throws Exception {
    CustomAnalyzer.Builder b = CustomAnalyzer.builder(dir);
    CustomAnalyzer.ConditionBuilder cb = null;
    for (Step s : steps) {
      String[] p = s.params.toArray(new String[0]);
      switch (s.kind) {
        case "tok" -> b.withTokenizer(s.name, p);
        case "cf" -> b.addCharFilter(s.name, p);
        case "tf" -> {
          if (cb != null) cb.addTokenFilter(s.name, p);
          else b.addTokenFilter(s.name, p);
        }
        case "when" -> cb = b.when(s.name, p);
        case "whenTerm" -> {
          int len = Integer.parseInt(s.name);
          cb = b.whenTerm(t -> t.length() > len);
        }
        case "endwhen" -> {
          b = cb.endwhen();
          cb = null;
        }
        case "version" -> b.withDefaultMatchVersion(Version.parse(s.name));
        case "posgap" -> b.withPositionIncrementGap(Integer.parseInt(s.name));
        case "offgap" -> b.withOffsetGap(Integer.parseInt(s.name));
        default -> throw new AssertionError(s.kind);
      }
    }
    return b.build();
  }

  public static void main(String[] args) throws Exception {
    Path out = Path.of(args[0]).resolve("analysis_factories");
    Files.createDirectories(out);
    String corpusDir = System.getenv().getOrDefault("FIXTURES_CORPUS", "fixtures/corpus");
    Path resources = Path.of(corpusDir, "analysis-factories");
    List<String> lines = AnalysisRows.corpus("analysis-factories.txt");
    List<String> configs =
        Files.readAllLines(Path.of(corpusDir, "analysis-factories.conf"), StandardCharsets.UTF_8);
    for (String config : configs) {
      if (config.isEmpty() || config.startsWith("#")) continue;
      String[] fields = config.split("\t", -1);
      StringBuilder m = new StringBuilder();
      CustomAnalyzer a;
      try {
        a = build(parse(fields), resources);
      } catch (Exception | Error e) {
        m.append("B\t").append(e.getClass().getSimpleName()).append('\t')
            .append(AnalysisRows.esc(stable(e))).append('\n');
        Files.writeString(out.resolve(fields[0] + ".tsv"), m.toString(), StandardCharsets.UTF_8);
        continue;
      }
      try (a) {
        m.append("S\t").append(a.toString().replaceAll("@[0-9a-f]+", "")).append('\n');
        m.append(AnalysisRows.rows(a, lines));
        for (int ln = 0; ln < lines.size(); ln++) {
          m.append("N\t").append(ln).append('\t');
          try {
            BytesRef n = a.normalize("f", lines.get(ln));
            m.append(AnalysisRows.hex(n));
          } catch (Exception e) {
            m.append("X:").append(e.getClass().getSimpleName());
          }
          m.append('\n');
        }
      }
      Files.writeString(out.resolve(fields[0] + ".tsv"), m.toString(), StandardCharsets.UTF_8);
    }
    StringBuilder names = new StringBuilder();
    for (String n : new TreeSet<>(TokenizerFactory.availableTokenizers())) names.append("T\t").append(n).append('\n');
    for (String n : new TreeSet<>(TokenFilterFactory.availableTokenFilters())) names.append("F\t").append(n).append('\n');
    for (String n : new TreeSet<>(CharFilterFactory.availableCharFilters())) names.append("C\t").append(n).append('\n');
    Files.writeString(out.resolve("names.txt"), names.toString(), StandardCharsets.UTF_8);
    System.out.println(String.format(Locale.ROOT, "%d configurations", configs.size()));
  }
}
