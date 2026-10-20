import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.io.PrintWriter;
import java.lang.reflect.Method;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Random;
import java.util.stream.Stream;
import org.apache.lucene.analysis.core.WhitespaceAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field.Store;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.Term;
import org.apache.lucene.misc.GetTermInfo;
import org.apache.lucene.misc.HighFreqTerms;
import org.apache.lucene.misc.IndexMergeTool;
import org.apache.lucene.misc.index.IndexSplitter;
import org.apache.lucene.misc.index.MultiPassIndexSplitter;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;

/**
 * M14.i: what the {@code misc} module's command-line tools print. Writes {@code tools/misc}:
 *
 * <ul>
 *   <li>{@code index} (three commits and a delete: three segments, one with deletions) and
 *       {@code index2} (one segment), the indexes the tools read;
 *   <li>{@code cases.tsv}, one tool run per line: {@code id, tool, args...}. An argument may
 *       name {@code {fixture}} (this directory, read only) or {@code {tmp}} (one scratch
 *       directory the runs share, in order). The tool {@code copy} copies directory
 *       {@code args[0]} to {@code args[1]} and prints nothing;
 *   <li>{@code out/<id>.out} and {@code out/<id>.err}, what the run wrote to {@code
 *       System.out} and {@code System.err}, and {@code out/<id>.code} its exit code -- or, for
 *       an exception thrown out of {@code main}, {@code class: message}. A run that would end
 *       in {@code System.exit} is stopped short of it, with what it printed and the code.
 * </ul>
 *
 * Read by {@code crates/lucene-tools/tests/misc_fixtures.rs}, which runs the same lines
 * through {@code lucene_tools::run}.
 */
public class GenToolsMisc {

  private static final String[] VOCAB = {
    "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa",
    "lambda", "mu", "nu", "xi", "omicron", "pi", "rho", "sigma", "tau", "upsilon"
  };

  private static final String[][] CASES = {
    {"term-common", "GetTermInfo", "{fixture}/index", "body", "common"},
    {"term-rare", "GetTermInfo", "{fixture}/index", "title", "upsilon"},
    {"term-absent", "GetTermInfo", "{fixture}/index", "body", "absent"},
    {"term-no-field", "GetTermInfo", "{fixture}/index", "nofield", "x"},
    {"term-usage", "GetTermInfo", "{fixture}/index", "body"},
    {"high-default", "HighFreqTerms", "{fixture}/index"},
    {"high-ttf-body", "HighFreqTerms", "{fixture}/index", "-t", "5", "body"},
    {"high-title-ttf", "HighFreqTerms", "{fixture}/index", "3", "title", "-t"},
    {"high-too-many", "HighFreqTerms", "{fixture}/index", "body", "4", "title", "2"},
    {"high-last-field-wins", "HighFreqTerms", "{fixture}/index", "body", "4", "title"},
    {"high-last-count-wins", "HighFreqTerms", "{fixture}/index", "7", "body", "3"},
    {"high-zero", "HighFreqTerms", "{fixture}/index", "0"},
    {"high-negative", "HighFreqTerms", "{fixture}/index", "-5", "body"},
    {"high-no-field", "HighFreqTerms", "{fixture}/index", "nofield"},
    {"high-usage", "HighFreqTerms"},
    {"split-list", "IndexSplitter", "{fixture}/index", "-l"},
    {"split-list2", "IndexSplitter", "{fixture}/index2", "-l"},
    {"split-usage", "IndexSplitter", "{fixture}/index"},
    {"split-copy", "IndexSplitter", "{fixture}/index", "{tmp}/split", "_0", "_2"},
    {"split-copy-list", "IndexSplitter", "{tmp}/split", "-l"},
    {"split-copy-terms", "HighFreqTerms", "{tmp}/split", "-t", "1000"},
    {"remove-copy", "copy", "{fixture}/index", "{tmp}/removed"},
    {"remove", "IndexSplitter", "{tmp}/removed", "-d", "_1", "_9"},
    {"remove-list", "IndexSplitter", "{tmp}/removed", "-l"},
    {"remove-terms", "HighFreqTerms", "{tmp}/removed", "1000"},
    {"merge", "IndexMergeTool", "{tmp}/merged", "{fixture}/index", "{fixture}/index2"},
    {"merge-list", "IndexSplitter", "{tmp}/merged", "-l"},
    {"merge-terms", "HighFreqTerms", "{tmp}/merged", "-t", "1000"},
    {
      "merge-again",
      "IndexMergeTool",
      "-max-segments",
      "1",
      "{tmp}/merged",
      "{fixture}/index2",
      "{fixture}/index"
    },
    {"merge-again-list", "IndexSplitter", "{tmp}/merged", "-l"},
    {"merge-again-terms", "HighFreqTerms", "{tmp}/merged", "1000", "-t"},
    {"merge-again-term", "GetTermInfo", "{tmp}/merged", "body", "common"},
    {
      "merge-logdoc",
      "IndexMergeTool",
      "-merge-policy",
      "org.apache.lucene.index.LogDocMergePolicy",
      "-max-segments",
      "2",
      "{tmp}/merged2",
      "{fixture}/index",
      "{fixture}/index2"
    },
    {"merge-logdoc-list", "IndexSplitter", "{tmp}/merged2", "-l"},
    {"merge-logdoc-terms", "HighFreqTerms", "{tmp}/merged2", "1000"},
    {"merge-unknown-option", "IndexMergeTool", "-x", "{tmp}/m", "a", "b"},
    {"merge-too-few", "IndexMergeTool", "-verbose", "{tmp}/m", "a"},
    {"merge-bad-number", "IndexMergeTool", "-max-segments", "two", "{tmp}/m", "a", "b"},
    {
      "multipass-seq",
      "MultiPassIndexSplitter",
      "-out",
      "{tmp}/seq",
      "-num",
      "3",
      "-seq",
      "{fixture}/index"
    },
    {"multipass-seq-0", "HighFreqTerms", "{tmp}/seq/part-0", "1000"},
    {"multipass-seq-1", "HighFreqTerms", "{tmp}/seq/part-1", "1000"},
    {"multipass-seq-2", "HighFreqTerms", "{tmp}/seq/part-2", "1000"},
    {
      "multipass-rr",
      "MultiPassIndexSplitter",
      "-num",
      "2",
      "{fixture}/index",
      "-out",
      "{tmp}/rr",
      "{fixture}/index2"
    },
    {"multipass-rr-0", "HighFreqTerms", "{tmp}/rr/part-0", "1000", "-t"},
    {"multipass-rr-1", "HighFreqTerms", "{tmp}/rr/part-1", "1000", "-t"},
    {"multipass-no-out", "MultiPassIndexSplitter", "-num", "2", "{fixture}/index", "{fixture}/index2", "-seq"},
    {"multipass-num-1", "MultiPassIndexSplitter", "-out", "{tmp}/mp1", "-num", "1", "{fixture}/index"},
    {"multipass-no-input", "MultiPassIndexSplitter", "-out", "{tmp}/mp2", "-num", "2", "no-such-input-dir"},
  };

  public static void main(String[] args) throws Exception {
    Path data = Path.of(args[0]).resolve("tools").resolve("misc");
    deleteTree(data);
    Files.createDirectories(data.resolve("out"));
    writeIndex(data.resolve("index"), 3, 120, 1414L, true);
    writeIndex(data.resolve("index2"), 1, 50, 2828L, false);

    Path tmp = Files.createTempDirectory("gen-tools-misc");
    PrintStream stdout = System.out;
    PrintStream stderr = System.err;
    try (PrintWriter tsv =
        new PrintWriter(
            Files.newBufferedWriter(data.resolve("cases.tsv"), StandardCharsets.UTF_8))) {
      for (String[] c : CASES) {
        tsv.print(String.join("\t", c) + "\n");
        String[] real = new String[c.length - 2];
        for (int i = 2; i < c.length; i++) {
          real[i - 2] =
              c[i].replace("{fixture}", data.toString()).replace("{tmp}", tmp.toString());
        }
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        ByteArrayOutputStream err = new ByteArrayOutputStream();
        System.setOut(new PrintStream(out, true, StandardCharsets.UTF_8));
        System.setErr(new PrintStream(err, true, StandardCharsets.UTF_8));
        String code;
        try {
          code = Integer.toString(run(c[1], real));
        } catch (Exception e) {
          code = e.getClass().getName() + ": " + e.getMessage();
        } finally {
          System.out.flush();
          System.err.flush();
          System.setOut(stdout);
          System.setErr(stderr);
        }
        Path base = data.resolve("out");
        Files.write(base.resolve(c[0] + ".out"), out.toByteArray());
        Files.write(base.resolve(c[0] + ".err"), err.toByteArray());
        Files.writeString(base.resolve(c[0] + ".code"), code + "\n", StandardCharsets.UTF_8);
      }
    } finally {
      deleteTree(tmp);
    }
  }

  /** {@code tool.usage()}, a private method. */
  private static void usage(Class<?> tool) throws Exception {
    Method usage = tool.getDeclaredMethod("usage");
    usage.setAccessible(true);
    usage.invoke(null);
  }

  /**
   * One tool's {@code main}, and the exit code it ends with. Where {@code main} would call
   * {@code System.exit}, this does what it does up to that call and returns the code.
   */
  private static int run(String tool, String[] args) throws Exception {
    switch (tool) {
      case "GetTermInfo":
        if (args.length != 3) {
          usage(GetTermInfo.class);
          return 1;
        }
        GetTermInfo.main(args);
        return 0;
      case "HighFreqTerms":
        if (args.length == 0 || args.length > 4) {
          usage(HighFreqTerms.class);
          return 1;
        }
        HighFreqTerms.main(args);
        return 0;
      case "IndexSplitter":
        IndexSplitter.main(args);
        return 0;
      case "IndexMergeTool":
        // main prints a parse error's message to System.err and exits 2: the same, without
        // the exit.
        try {
          Method parse =
              Class.forName("org.apache.lucene.misc.IndexMergeTool$Options")
                  .getDeclaredMethod("parse", String[].class);
          parse.setAccessible(true);
          parse.invoke(null, (Object) args);
        } catch (java.lang.reflect.InvocationTargetException e) {
          if (e.getCause() instanceof IllegalArgumentException) {
            System.err.println(e.getCause().getMessage());
            return 2;
          }
          throw e;
        }
        IndexMergeTool.main(args);
        return 0;
      case "MultiPassIndexSplitter":
        MultiPassIndexSplitter.main(args);
        return 0;
      case "copy":
        copyTree(Path.of(args[0]), Path.of(args[1]));
        return 0;
      default:
        throw new IllegalArgumentException(tool);
    }
  }

  private static void writeIndex(Path path, int commits, int docsPerCommit, long seed, boolean delete)
      throws IOException {
    Random rnd = new Random(seed);
    try (Directory dir = FSDirectory.open(path);
        IndexWriter w = new IndexWriter(dir, new IndexWriterConfig(new WhitespaceAnalyzer()))) {
      int id = 0;
      for (int commit = 0; commit < commits; commit++) {
        for (int d = 0; d < docsPerCommit; d++) {
          Document doc = new Document();
          doc.add(new StringField("id", Integer.toString(id++), Store.YES));
          // "common" in every document, nine times: totalTF passes 1,000 and is grouped.
          StringBuilder body = new StringBuilder("common common common common common");
          body.append(" common common common common ").append(text(rnd, 4 + rnd.nextInt(12)));
          doc.add(new TextField("body", body.toString(), Store.NO));
          if (d % 3 != 0) {
            doc.add(new TextField("title", text(rnd, 1 + rnd.nextInt(3)), Store.NO));
          }
          w.addDocument(doc);
        }
        w.commit();
      }
      if (delete) {
        // A deleted document still counts in docFreq and totalTermFreq.
        w.deleteDocuments(new Term("id", "5"), new Term("id", "130"));
        w.commit();
      }
    }
    Files.deleteIfExists(path.resolve("write.lock"));
  }

  /** Zipf-like: the earlier words of the vocabulary come up more often. */
  private static String text(Random rnd, int words) {
    StringBuilder sb = new StringBuilder();
    for (int i = 0; i < words; i++) {
      if (i > 0) {
        sb.append(' ');
      }
      sb.append(VOCAB[Math.min(rnd.nextInt(VOCAB.length), rnd.nextInt(VOCAB.length))]);
    }
    return sb.toString();
  }

  private static void copyTree(Path from, Path to) throws IOException {
    Files.createDirectories(to);
    try (Stream<Path> files = Files.list(from)) {
      for (Path f : (Iterable<Path>) files::iterator) {
        Files.copy(f, to.resolve(f.getFileName()));
      }
    }
  }

  private static void deleteTree(Path p) throws IOException {
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
