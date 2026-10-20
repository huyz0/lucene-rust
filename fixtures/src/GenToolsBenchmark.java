import java.io.PrintWriter;
import java.lang.management.ManagementFactory;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.stream.Stream;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;

/**
 * M14.i: what the {@code benchmark} module's search-quality command lines, {@code QueryDriver}
 * and {@code QualityQueriesFinder}, print and write. Writes {@code tools/benchmark}:
 *
 * <ul>
 *   <li>{@code index/}: {@value #DOCS} documents of made-up words, Zipf-like so some are common
 *       and many rare, in two segments: {@code docname} ({@code doc<n>}, stored) and {@code
 *       body}, as the quality package expects;
 *   <li>{@code topics.txt}, TREC topics over those words (one id not a number, one title with
 *       query syntax to escape, one a single rare word); {@code topics-bad.txt}, a title the
 *       classic parser refuses ({@code AND} at the end, which {@code escape} keeps);
 *   <li>{@code qrels.txt}, judgements: each topic's relevant documents are those holding its
 *       first title word, with some judged not relevant, a comment, a query no topic has and a
 *       topic left unjudged; {@code qrels-none.txt} judges no topic at all, so the average is
 *       {@code NaN} and {@code QualityStats.log} throws;
 *   <li>{@code cases.tsv} and {@code out/<id>.out}, {@code .err}, {@code .code}, as {@code
 *       GenToolsDemo} writes them, and {@code out/<id>.sub}: the submission file a {@code
 *       QueryDriver} run wrote. The search and doc-name times are written {@code {s}}.
 * </ul>
 *
 * <p>Read by {@code crates/lucene-tools/tests/benchmark_fixtures.rs}.
 */
public class GenToolsBenchmark {

  private static final int DOCS = 120;
  private static final int VOCABULARY = 160;

  private static final Map<String, String> CLASSES =
      Map.of(
          "QueryDriver", "org.apache.lucene.benchmark.quality.trec.QueryDriver",
          "QualityQueriesFinder", "org.apache.lucene.benchmark.quality.utils.QualityQueriesFinder");

  private static final List<String[]> CASES = new ArrayList<>();

  static {
    c("qd-usage", "QueryDriver", "a", "b", "c");
    c("qd-usage-long", "QueryDriver", "a", "b", "c", "d", "e", "f");
    c("qd-title", "QueryDriver", "{fixture}/topics.txt", "{fixture}/qrels.txt",
        "{tmp}/title.sub", "{fixture}/index");
    c("qd-td", "QueryDriver", "{fixture}/topics.txt", "{fixture}/qrels.txt", "{tmp}/td.sub",
        "{fixture}/index", "TD");
    c("qd-tdn", "QueryDriver", "{fixture}/topics.txt", "{fixture}/qrels.txt", "{tmp}/tdn.sub",
        "{fixture}/index", "NDT");
    c("qd-narrative", "QueryDriver", "{fixture}/topics.txt", "{fixture}/qrels.txt",
        "{tmp}/n.sub", "{fixture}/index", "N");
    c("qd-no-part", "QueryDriver", "{fixture}/topics.txt", "{fixture}/qrels.txt",
        "{tmp}/x.sub", "{fixture}/index", "x");
    c("qd-unjudged", "QueryDriver", "{fixture}/topics.txt", "{fixture}/qrels-none.txt",
        "{tmp}/none.sub", "{fixture}/index");
    c("qd-parse-error", "QueryDriver", "{fixture}/topics-bad.txt", "{fixture}/qrels.txt",
        "{tmp}/bad.sub", "{fixture}/index");
    c("qqf-usage", "QualityQueriesFinder");
    c("qqf-index", "QualityQueriesFinder", "{fixture}/index");
  }

  private static void c(String... c) {
    CASES.add(c);
  }

  public static void main(String[] args) throws Exception {
    Path data = Path.of(args[0]).resolve("tools").resolve("benchmark");
    deleteTree(data);
    Files.createDirectories(data.resolve("out"));

    String[] words = vocabulary();
    Random rnd = new Random(20261021);
    List<List<String>> docs = new ArrayList<>();
    for (int d = 0; d < DOCS; d++) {
      int n = 6 + rnd.nextInt(10);
      List<String> body = new ArrayList<>();
      for (int i = 0; i < n; i++) {
        double r = rnd.nextDouble();
        body.add(words[(int) (VOCABULARY * r * r * r)]);
      }
      docs.add(body);
    }
    writeIndex(data, docs);

    // title, description, narrative; the first title word decides relevance
    String[][] topics = {
      // no part empty: the classic parser refuses an empty string
      {"12", words[0] + " " + words[5], "documents about " + words[5], words[9] + " " + words[2]},
      {"3", words[1] + " " + words[30], words[1], words[33]},
      {"7", words[2] + "-" + words[40] + " (" + words[3] + ")", words[3] + ": " + words[44], "n/a"},
      {"25", words[90], "a rare word", words[91] + " " + words[92]},
      {"b2", words[7] + " " + words[8] + " " + words[20], words[21], words[4]},
      {"8", words[11], "never judged", words[12]},
    };
    StringBuilder sb = new StringBuilder();
    for (String[] t : topics) {
      sb.append(topic(t[0], t[1], t[2], t[3]));
    }
    Files.writeString(data.resolve("topics.txt"), sb.toString(), StandardCharsets.UTF_8);
    Files.writeString(
        data.resolve("topics-bad.txt"),
        topic("1", words[0] + " AND", "", ""),
        StandardCharsets.UTF_8);

    StringBuilder qrels = new StringBuilder("# query 0 doc relevance\n\n");
    for (String[] t : topics) {
      if (t[0].equals("8")) {
        continue;
      }
      String first = t[1].split("[ \\-]")[0];
      for (int d = 0; d < DOCS; d++) {
        if (docs.get(d).contains(first)) {
          // every third holder judged not relevant
          qrels.append(t[0]).append(" 0 doc").append(d).append(d % 3 == 2 ? " 0\n" : " 1\n");
        }
      }
    }
    qrels.append("  99\t0 doc1 1  \n");
    Files.writeString(data.resolve("qrels.txt"), qrels.toString(), StandardCharsets.UTF_8);
    Files.writeString(
        data.resolve("qrels-none.txt"), "99 0 doc1 1\n98 0 doc2 1\n", StandardCharsets.UTF_8);

    Path tmp = Files.createTempDirectory("gen-tools-benchmark");
    Path logging = Files.createTempFile("gen-tools-benchmark", ".properties");
    Files.writeString(logging, ".level=OFF\n", StandardCharsets.UTF_8);
    try (PrintWriter tsv =
        new PrintWriter(
            Files.newBufferedWriter(data.resolve("cases.tsv"), StandardCharsets.UTF_8))) {
      for (String[] c : CASES) {
        tsv.print(String.join("\t", c) + "\n");
        List<String> command = new ArrayList<>();
        command.add(Path.of(System.getProperty("java.home"), "bin", "java").toString());
        command.addAll(ManagementFactory.getRuntimeMXBean().getInputArguments());
        command.add("--enable-native-access=ALL-UNNAMED");
        command.add("-Djava.util.logging.config.file=" + logging);
        command.add("-cp");
        command.add(System.getProperty("java.class.path"));
        command.add(CLASSES.get(c[1]));
        List<String> toolArgs = new ArrayList<>();
        for (int i = 2; i < c.length; i++) {
          toolArgs.add(c[i].replace("{fixture}", data.toString()).replace("{tmp}", tmp.toString()));
        }
        command.addAll(toolArgs);
        Path outFile = tmp.resolve(".out");
        Path errFile = tmp.resolve(".err");
        Process process =
            new ProcessBuilder(command)
                .redirectOutput(outFile.toFile())
                .redirectError(errFile.toFile())
                .start();
        process.getOutputStream().close();
        int exit = process.waitFor();
        String out = clean(Files.readString(outFile, StandardCharsets.UTF_8), data, tmp);
        String err = clean(Files.readString(errFile, StandardCharsets.UTF_8), data, tmp);
        Files.delete(outFile);
        Files.delete(errFile);
        String code = Integer.toString(exit);
        String prefix = "Exception in thread \"main\" ";
        int at = err.indexOf(prefix);
        if (at >= 0) {
          int trace = err.indexOf("\n\tat ", at);
          code = err.substring(at + prefix.length(), trace >= 0 ? trace : err.length() - 1);
          err = err.substring(0, at);
        }
        Path base = data.resolve("out");
        Files.writeString(base.resolve(c[0] + ".out"), out, StandardCharsets.UTF_8);
        Files.writeString(base.resolve(c[0] + ".err"), err, StandardCharsets.UTF_8);
        Files.writeString(base.resolve(c[0] + ".code"), code + "\n", StandardCharsets.UTF_8);
        if (c[1].equals("QueryDriver") && toolArgs.size() >= 4) {
          Path sub = Path.of(toolArgs.get(2));
          if (Files.exists(sub)) {
            Files.copy(sub, base.resolve(c[0] + ".sub"));
          }
        }
      }
    } finally {
      deleteTree(tmp);
      Files.deleteIfExists(logging);
    }
  }

  /** Two-syllable made-up words, distinct, in a fixed order. */
  private static String[] vocabulary() {
    String consonants = "bdfgklmnprstvz";
    String vowels = "aeiou";
    String[] words = new String[VOCABULARY];
    for (int i = 0; i < VOCABULARY; i++) {
      int a = i % 70;
      int b = i / 70;
      words[i] =
          ""
              + consonants.charAt(a % 14)
              + vowels.charAt(a / 14)
              + consonants.charAt((a * 3 + b) % 14)
              + vowels.charAt((a + b) % 5)
              + (b > 0 ? "n".repeat(b) : "");
    }
    return words;
  }

  private static String topic(String id, String title, String desc, String narr) {
    return "<top>\n<num> Number: "
        + id
        + "\n\n<title> "
        + title
        + "\n\n<desc> Description:\n"
        + desc
        + "\n\n<narr> Narrative:\n"
        + narr
        + "\n\n</top>\n\n";
  }

  /** The documents, a commit after the first half: two segments. */
  private static void writeIndex(Path data, List<List<String>> docs) throws Exception {
    try (Directory dir = FSDirectory.open(data.resolve("index"));
        IndexWriter w = new IndexWriter(dir, new IndexWriterConfig(new StandardAnalyzer()))) {
      for (int d = 0; d < docs.size(); d++) {
        Document doc = new Document();
        doc.add(new StringField("docname", "doc" + d, Field.Store.YES));
        doc.add(new TextField("body", String.join(" ", docs.get(d)), Field.Store.NO));
        w.addDocument(doc);
        if (d == docs.size() / 2) {
          w.commit();
        }
      }
    }
    Files.deleteIfExists(data.resolve("index").resolve("write.lock"));
  }

  /** What a tool printed, with the directories and the measured times written as names. */
  private static String clean(String text, Path data, Path tmp) {
    return text.replace(data.toString(), "{fixture}")
        .replace(tmp.toString(), "{tmp}")
        .replaceAll("(Search|DocName) Seconds: [^\n]*", "$1 Seconds: {s}");
  }

  private static void deleteTree(Path p) throws Exception {
    if (!Files.exists(p)) {
      return;
    }
    try (Stream<Path> walk = Files.walk(p)) {
      List<Path> all = new ArrayList<>(walk.sorted(Comparator.reverseOrder()).toList());
      for (Path f : all) {
        Files.delete(f);
      }
    }
  }
}
