import java.io.PrintWriter;
import java.lang.management.ManagementFactory;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;
import java.util.Random;
import java.util.stream.Stream;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.demo.knn.DemoEmbeddings;
import org.apache.lucene.demo.knn.KnnVectorDict;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.KeywordField;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.LongField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;

/**
 * M14.i: what the {@code demo} module's {@code IndexFiles} and {@code SearchFiles} print, and
 * the toy word-vector dictionary they share. Writes {@code tools/demo}:
 *
 * <ul>
 *   <li>{@code docs/}, the files the demo indexes (one in a subdirectory), {@code vectors.txt}, a
 *       GloVe-style dictionary source, and {@code queries.txt}, a {@code -queries} file;
 *   <li>{@code dict/knn-dict.fst} and {@code dict/knn-dict.bin}, {@code KnnVectorDict.build}'s
 *       output over {@code vectors.txt};
 *   <li>{@code embeddings.tsv}: {@code text, bits} -- {@code DemoEmbeddings.computeEmbedding} of
 *       the text, each float's bits in hex, comma-separated;
 *   <li>{@code index/}, a Java-written demo index with vectors: the documents of {@code docs/}
 *       as {@code IndexFiles.indexDoc} makes them, with the relative path {@code docs/<name>}
 *       and modification time 0, in name order, and the dictionary;
 *   <li>{@code cases.tsv}, one run per line: {@code id, stdin, tool, args...}. {@code stdin} is
 *       what {@code System.in} holds, newlines written {@code \n} ({@code -} for nothing). An
 *       argument may name {@code {fixture}} (this directory, read only) or {@code {tmp}} (one
 *       scratch directory the runs share, in order);
 *   <li>{@code out/<id>.out}, {@code .err} and {@code .code}, as {@code GenToolsMisc} writes
 *       them, with this directory written {@code {fixture}}, the scratch directory {@code {tmp}},
 *       and the times the tools measure {@code {ms}}.
 * </ul>
 *
 * Each case runs the tool's own {@code main} in a JVM of its own, so the exit codes and the
 * argument errors are Lucene's: an exception out of {@code main} is recorded as the code, {@code
 * <class>: <message>}, without its stack trace.
 *
 * <p>A {@code walk} case indexes a whole directory, in the order the file system lists it: its
 * lines are compared as a set.
 *
 * <p>Read by {@code crates/lucene-tools/tests/demo_fixtures.rs}.
 */
public class GenToolsDemo {

  private static final String[][] DOCS = {
    {"a.txt", "The quick brown fox jumps over the lazy dog."},
    {"b.txt", "A fast brown fox leaps over sleeping dogs in the park."},
    {"c.txt", "Lucene is a search engine library written in Java."},
    {"d.txt", "Rust is a systems programming language focused on safety and speed."},
    {"e.txt", "The search engine ranks documents by relevance using BM25 and the engine is fast."},
    {"f.txt", "Dogs and cats are common pets; the dog barks and the cat sleeps."},
    {"g.txt", "Vectors and embeddings let a search engine find similar documents."},
    {"h.txt", "The lazy cat sleeps all day in the warm sun, and the cat dreams."},
    {"sub/i.txt", "Foxes and dogs are both canines, but the fox is wild and the dog is not."},
    // no word of the dictionary: a zero vector
    {"sub/j.txt", "Zebras graze quietly."},
  };

  /** The dictionary's words after its first line, sorted. */
  private static final String[] WORDS = {
    "brown", "cat", "cats", "dog", "dogs", "engine", "fast", "fox", "java", "lazy", "lucene",
    "quick", "rust", "search", "sleeps", "sun", "vectors"
  };

  private static final String[] EMBED = {
    "fox", "lazy dog", "Search ENGINE vectors", "nothing known here", "the the the", "",
  };

  private static final List<String[]> CASES = new ArrayList<>();

  static {
    c("index-usage", "-", "IndexFiles", "-index", "{tmp}/x");
    c("index-unknown", "-", "IndexFiles", "-bogus");
    c("index-missing-docs", "-", "IndexFiles", "-index", "{tmp}/x", "-docs", "{tmp}/nowhere");
    c("index-a", "-", "IndexFiles", "-index", "{tmp}/idx", "-docs", "{fixture}/docs/a.txt");
    // Eight one-document segments at most: a ninth makes TieredMergePolicy merge them, and the
    // order the merged segment takes its documents in follows the segments' byte sizes, which
    // the Rust writer does not reproduce (its `.si` diagnostics are shorter).
    String[] rest = {"b", "c", "d", "e", "f", "g"};
    for (String r : rest) {
      c("index-" + r, "-", "IndexFiles", "-update", "-index", "{tmp}/idx", "-docs",
          "{fixture}/docs/" + r + ".txt");
    }
    c("index-i", "-", "IndexFiles", "-index", "{tmp}/idx", "-update", "-docs",
        "{fixture}/docs/sub//i.txt/");
    c("index-again-a", "-", "IndexFiles", "-update", "-index", "{tmp}/idx", "-docs",
        "{fixture}/docs/a.txt");
    c("search-fox", "-", "SearchFiles", "-index", "{tmp}/idx", "-query", "fox");
    c("search-raw", "-", "SearchFiles", "-index", "{tmp}/idx", "-query", "the", "-raw");
    c("search-bool", "-", "SearchFiles", "-index", "{tmp}/idx", "-query",
        "+search engine -java");
    c("search-phrase", "-", "SearchFiles", "-index", "{tmp}/idx", "-query", "\"lazy dog\"");
    c("search-wild", "-", "SearchFiles", "-index", "{tmp}/idx", "-query", "do*", "-paging",
        "3");
    c("search-none", "-", "SearchFiles", "-index", "{tmp}/idx", "-query", "zebra");
    c("search-path", "-", "SearchFiles", "-index", "{tmp}/idx", "-field", "path", "-query",
        "nothing");
    c("search-blank", "-", "SearchFiles", "-index", "{tmp}/idx", "-query", "  ");
    c("search-repeat", "-", "SearchFiles", "-index", "{tmp}/idx", "-query", "cat", "-repeat",
        "3");
    c("search-parse-error", "-", "SearchFiles", "-index", "{tmp}/idx", "-query", "fox AND");
    c("search-queries", "-", "SearchFiles", "-index", "{tmp}/idx", "-queries",
        "{fixture}/queries.txt", "-paging", "2");
    c("search-help", "-", "SearchFiles", "-h");
    c("search-unknown", "-", "SearchFiles", "-index", "{tmp}/idx", "-bogus");
    c("search-paging-zero", "-", "SearchFiles", "-paging", "0");
    c("search-interactive", "fox\\nq\\ndog\\n\\n", "SearchFiles", "-index", "{tmp}/idx");
    c("search-pages", "the\\nn\\nn\\np\\n2\\n9\\nn\\nq\\n", "SearchFiles", "-index",
        "{tmp}/idx", "-paging", "2");
    c("search-collect-more", "the is\\nn\\nn\\nn\\nn\\nn\\ny\\nn\\nn\\n\\n", "SearchFiles",
        "-index", "{tmp}/idx", "-paging", "1");
    c("search-collect-no", "the is\\nn\\nn\\nn\\nn\\nn\\nn\\n", "SearchFiles", "-index",
        "{tmp}/idx", "-paging", "1");
    c("search-bad-page", "the\\nx\\n", "SearchFiles", "-index", "{tmp}/idx", "-paging", "2");
    c("java-fox", "-", "SearchFiles", "-index", "{fixture}/index", "-query", "fox");
    c("java-knn", "-", "SearchFiles", "-index", "{fixture}/index", "-query", "lazy dog",
        "-knn_vector", "3");
    c("java-knn-raw", "-", "SearchFiles", "-index", "{fixture}/index", "-query", "+cat -sun",
        "-knn_vector", "4", "-raw");
    c("java-knn-no-terms", "-", "SearchFiles", "-index", "{fixture}/index", "-query",
        "path:docs -contents:fox", "-knn_vector", "2");
    c("java-knn-path", "-", "SearchFiles", "-index", "{fixture}/index", "-field", "path",
        "-query", "docs", "-knn_vector", "2");
    c("java-knn-unknown-words", "-", "SearchFiles", "-index", "{fixture}/index", "-query",
        "zebra", "-knn_vector", "2");
    c("walk-knn", "-", "IndexFiles", "-index", "{tmp}/knn", "-docs", "{fixture}/docs",
        "-knn_dict", "{fixture}/vectors.txt");
    c("walk-knn-search", "-", "SearchFiles", "-index", "{tmp}/knn", "-query", "brown fox",
        "-knn_vector", "3");
    c("walk-knn-search-engine", "-", "SearchFiles", "-index", "{tmp}/knn", "-query",
        "engine", "-knn_vector", "5");
    // the dictionary again: `createOutput` refuses the files the first run wrote
    c("walk-knn-again", "-", "IndexFiles", "-index", "{tmp}/knn", "-docs", "{fixture}/docs",
        "-knn_dict", "{fixture}/vectors.txt");
    c("walk-create-again", "-", "IndexFiles", "-index", "{tmp}/knn", "-create", "-docs",
        "{fixture}/docs/sub");
    c("walk-create-again-knn", "-", "SearchFiles", "-index", "{tmp}/knn", "-query", "dog",
        "-knn_vector", "2");
    c("walk-create-again-search", "-", "SearchFiles", "-index", "{tmp}/knn", "-query", "dog",
        "-raw");
  }

  private static void c(String... c) {
    CASES.add(c);
  }

  public static void main(String[] args) throws Exception {
    Path data = Path.of(args[0]).resolve("tools").resolve("demo");
    deleteTree(data);
    Files.createDirectories(data.resolve("out"));
    for (String[] d : DOCS) {
      Path f = data.resolve("docs").resolve(d[0]);
      Files.createDirectories(f.getParent());
      Files.writeString(f, d[1] + "\n", StandardCharsets.UTF_8);
    }
    Files.writeString(
        data.resolve("queries.txt"), "fox\n  quick brown\ncat\n\nnever read\n", StandardCharsets.UTF_8);
    writeVectors(data.resolve("vectors.txt"));

    try (Directory dict = FSDirectory.open(data.resolve("dict"))) {
      KnnVectorDict.build(data.resolve("vectors.txt"), dict, "knn-dict");
      try (KnnVectorDict vd = new KnnVectorDict(dict, "knn-dict");
          PrintWriter tsv =
              new PrintWriter(
                  Files.newBufferedWriter(data.resolve("embeddings.tsv"), StandardCharsets.UTF_8))) {
        DemoEmbeddings e = new DemoEmbeddings(vd);
        List<String> texts = new ArrayList<>(List.of(EMBED));
        for (String[] d : DOCS) {
          texts.add(d[1]);
        }
        for (String t : texts) {
          float[] v = e.computeEmbedding(t);
          StringBuilder sb = new StringBuilder();
          for (int i = 0; i < v.length; i++) {
            if (i > 0) {
              sb.append(',');
            }
            sb.append(Integer.toHexString(Float.floatToRawIntBits(v[i])));
          }
          tsv.print(t + "\t" + sb + "\n");
        }
      }
    }
    writeJavaIndex(data);

    Path tmp = Files.createTempDirectory("gen-tools-demo");
    // the forked JVMs log nothing (Lucene's vectorization notice would land on their stderr)
    Path logging = Files.createTempFile("gen-tools-demo", ".properties");
    Files.writeString(logging, ".level=OFF\n", StandardCharsets.UTF_8);
    try (PrintWriter tsv =
        new PrintWriter(
            Files.newBufferedWriter(data.resolve("cases.tsv"), StandardCharsets.UTF_8))) {
      for (String[] c : CASES) {
        tsv.print(String.join("\t", c) + "\n");
        List<String> command = new ArrayList<>();
        command.add(Path.of(System.getProperty("java.home"), "bin", "java").toString());
        // this JVM's own flags (the vector FMA pins among them), and none of the JDK's
        // native-access warnings on the forked JVM's stderr
        command.addAll(ManagementFactory.getRuntimeMXBean().getInputArguments());
        command.add("--enable-native-access=ALL-UNNAMED");
        command.add("-Djava.util.logging.config.file=" + logging);
        command.add("-cp");
        command.add(System.getProperty("java.class.path"));
        command.add("org.apache.lucene.demo." + c[2]);
        for (int i = 3; i < c.length; i++) {
          command.add(c[i].replace("{fixture}", data.toString()).replace("{tmp}", tmp.toString()));
        }
        String in = c[1].equals("-") ? "" : c[1].replace("\\n", "\n");
        Path outFile = tmp.resolve(".out");
        Path errFile = tmp.resolve(".err");
        Process process =
            new ProcessBuilder(command)
                .redirectOutput(outFile.toFile())
                .redirectError(errFile.toFile())
                .start();
        try (var stdin = process.getOutputStream()) {
          stdin.write(in.getBytes(StandardCharsets.UTF_8));
        }
        int exit = process.waitFor();
        String out = clean(Files.readString(outFile, StandardCharsets.UTF_8), data, tmp);
        String err = clean(Files.readString(errFile, StandardCharsets.UTF_8), data, tmp);
        Files.delete(outFile);
        Files.delete(errFile);
        // An exception out of `main`: its toString (without the JVM's prefix), every line of it,
        // is the code, and it and its stack trace are not part of stderr.
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
      }
    } finally {
      deleteTree(tmp);
      Files.deleteIfExists(logging);
    }
  }

  /** What a tool printed, with the directories and the measured times written as names. */
  private static String clean(String text, Path data, Path tmp) {
    return text
        .replace(data.toString(), "{fixture}")
        .replace(tmp.toString(), "{tmp}")
        .replaceAll("documents in [0-9]+ ms", "documents in {ms} ms")
        .replaceAll("Time: [0-9]+ms", "Time: {ms}ms");
  }

  /** A first line, then {@link #WORDS} with four components each, four decimals. */
  private static void writeVectors(Path path) throws Exception {
    Random rnd = new Random(1414);
    StringBuilder sb = new StringBuilder();
    List<String> words = new ArrayList<>();
    words.add("the");
    words.addAll(List.of(WORDS));
    for (String w : words) {
      sb.append(w);
      for (int i = 0; i < 4; i++) {
        sb.append(' ').append(String.format(Locale.ROOT, "%.4f", rnd.nextGaussian()));
      }
      sb.append('\n');
    }
    Files.writeString(path, sb.toString(), StandardCharsets.UTF_8);
  }

  /** {@code IndexFiles.indexDoc} for each document, with a relative path and time 0. */
  private static void writeJavaIndex(Path data) throws Exception {
    try (Directory dir = FSDirectory.open(data.resolve("index"))) {
      KnnVectorDict.build(data.resolve("vectors.txt"), dir, "knn-dict");
      try (KnnVectorDict vd = new KnnVectorDict(dir, "knn-dict");
          IndexWriter w = new IndexWriter(dir, new IndexWriterConfig(new StandardAnalyzer()))) {
        DemoEmbeddings e = new DemoEmbeddings(vd);
        for (String[] d : DOCS) {
          String text = d[1] + "\n";
          Document doc = new Document();
          doc.add(new KeywordField("path", "docs/" + d[0], Field.Store.YES));
          doc.add(new LongField("modified", 0L, Field.Store.NO));
          doc.add(new TextField("contents", new java.io.StringReader(text)));
          doc.add(
              new KnnFloatVectorField(
                  "contents-vector", e.computeEmbedding(text), VectorSimilarityFunction.DOT_PRODUCT));
          w.addDocument(doc);
        }
      }
    }
    Files.deleteIfExists(data.resolve("index").resolve("write.lock"));
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
