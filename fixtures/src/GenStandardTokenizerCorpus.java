import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.Reader;
import java.io.StringReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.zip.GZIPInputStream;
import java.util.zip.ZipFile;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.analysis.standard.StandardTokenizer;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.analysis.tokenattributes.OffsetAttribute;
import org.apache.lucene.analysis.tokenattributes.PositionIncrementAttribute;
import org.apache.lucene.analysis.tokenattributes.TypeAttribute;

/**
 * M7's corpus-scale differential fixture for {@code StandardTokenizer} ({@code
 * crates/lucene-analysis/tests/standard_tokenizer_corpus.rs}): Lucene 10.5.0's {@code
 * StandardTokenizer} and {@code StandardAnalyzer} over a few megabytes of real multilingual text,
 * recorded as digests rather than tokens.
 *
 * <p>The text is not committed. It is read from the Lucene 10.5.0 jars themselves, which every
 * machine that runs the fixtures already has: the first {@link #EUROPARL_LINES} lines of {@code
 * lucene-test-framework}'s {@code europarl.lines.txt.gz} (the {@code LineFileDocs} corpus:
 * European Parliament proceedings in twenty-one languages -- Latin, Greek and Cyrillic script,
 * diacritics, numbers, abbreviations, punctuation, and lines cut mid-character, which decode to
 * U+FFFD) and every stopword list {@code lucene-analysis-common} ships (Thai, Devanagari,
 * Bengali, Tamil, Telugu, Arabic, Persian, Sorani, Armenian, Greek, Cyrillic, Latin-script
 * languages). Each line is one document, decoded as UTF-8 with replacement, as {@code
 * LineFileDocs} reads it.
 *
 * <p>{@code standard_tokenizer_corpus/manifest.tsv}: {@code S} lines name each source (module,
 * jar entry, gzipped or not, lines taken); {@code C} lines cover {@link #CHUNK} lines of one
 * source under one configuration ({@code tok}: the bare tokenizer at the default {@code
 * maxTokenLength}; {@code std}: {@code StandardAnalyzer} without stopwords) with the chunk text's
 * FNV-1a digest, its token count, the digest of every token, and an 8-hex-digit digest per line,
 * so a mismatch is narrowed to the line. A token is digested as its term's UTF-8, then {@code
 * \0start,end,posInc,type\n}; each line ends with {@code end:finalOffset,posInc\n}.
 *
 * <p>{@code GenStandardTokenizerCorpus --dump <source> <line> <config>} prints one line's tokens,
 * one per line in that same form, for a diff against the Rust test's failure output.
 */
public class GenStandardTokenizerCorpus {

  static final int EUROPARL_LINES = 4000;
  static final int CHUNK = 100;

  /** {@code id, module, jar entry, gzipped, lines taken (-1: all)}. */
  static List<String[]> sources() {
    List<String[]> s = new ArrayList<>();
    s.add(
        new String[] {
          "europarl",
          "lucene-test-framework",
          "org/apache/lucene/tests/util/europarl.lines.txt.gz",
          "1",
          Integer.toString(EUROPARL_LINES)
        });
    String[] lists = {
      "ar/stopwords.txt", "bg/stopwords.txt", "bn/stopwords.txt", "br/stopwords.txt",
      "ca/stopwords.txt", "cjk/stopwords.txt", "ckb/stopwords.txt", "cz/stopwords.txt",
      "el/stopwords.txt", "et/stopwords.txt", "eu/stopwords.txt", "fa/stopwords.txt",
      "gl/stopwords.txt", "hi/stopwords.txt", "hy/stopwords.txt", "id/stopwords.txt",
      "lt/stopwords.txt", "lv/stopwords.txt", "ne/stopwords.txt", "ro/stopwords.txt",
      "snowball/danish_stop.txt", "snowball/dutch_stop.txt", "snowball/english_stop.txt",
      "snowball/finnish_stop.txt", "snowball/french_stop.txt", "snowball/german_stop.txt",
      "snowball/hungarian_stop.txt", "snowball/indonesian_stop.txt", "snowball/irish_stop.txt",
      "snowball/italian_stop.txt", "snowball/norwegian_stop.txt", "snowball/portuguese_stop.txt",
      "snowball/russian_stop.txt", "snowball/spanish_stop.txt", "snowball/swedish_stop.txt",
      "sr/stopwords.txt", "ta/stopwords.txt", "te/stopwords.txt", "th/stopwords.txt",
      "tr/stopwords.txt",
    };
    for (String l : lists) {
      s.add(
          new String[] {
            l.replace('/', '_').replace(".txt", ""),
            "lucene-analysis-common",
            "org/apache/lucene/analysis/" + l,
            "0",
            "-1"
          });
    }
    return s;
  }

  static List<String> lines(String[] src) throws IOException {
    InputStream in;
    if (src[1].equals("lucene-test-framework")) {
      // Read from the jar as data (scripts/gen-fixtures.sh exports its path):
      // on the classpath its SPI registrations would break other generators.
      String jar = System.getenv("LUCENE_TEST_FRAMEWORK_JAR");
      if (jar == null) {
        throw new IOException("LUCENE_TEST_FRAMEWORK_JAR is not set (run scripts/gen-fixtures.sh)");
      }
      ZipFile zip = new ZipFile(jar);
      in = zip.getInputStream(zip.getEntry(src[2]));
    } else {
      in = GenStandardTokenizerCorpus.class.getClassLoader().getResourceAsStream(src[2]);
    }
    if (in == null) {
      throw new IOException("not found: " + src[2]);
    }
    if (src[3].equals("1")) {
      in = new GZIPInputStream(in);
    }
    int limit = Integer.parseInt(src[4]);
    List<String> out = new ArrayList<>();
    // A plain InputStreamReader: malformed UTF-8 becomes U+FFFD.
    try (BufferedReader r = new BufferedReader(new InputStreamReader(in, StandardCharsets.UTF_8))) {
      String line;
      while ((line = r.readLine()) != null && (limit < 0 || out.size() < limit)) {
        out.add(line);
      }
    }
    return out;
  }

  static final long FNV_OFFSET = 0xcbf29ce484222325L;

  static long fnv(long h, byte[] bytes) {
    for (byte b : bytes) {
      h ^= b & 0xffL;
      h *= 0x100000001b3L;
    }
    return h;
  }

  static Analyzer analyzer(String config) {
    if (config.equals("std")) {
      return new StandardAnalyzer();
    }
    return new Analyzer() {
      @Override
      protected TokenStreamComponents createComponents(String fieldName) {
        return new TokenStreamComponents(new StandardTokenizer());
      }
    };
  }

  /** One line's tokens in digest form, and their count. */
  static String tokens(Analyzer a, String text, int[] count) throws IOException {
    StringBuilder sb = new StringBuilder();
    try (TokenStream ts = a.tokenStream("body", new StringReader(text))) {
      CharTermAttribute term = ts.addAttribute(CharTermAttribute.class);
      OffsetAttribute off = ts.addAttribute(OffsetAttribute.class);
      PositionIncrementAttribute inc = ts.addAttribute(PositionIncrementAttribute.class);
      TypeAttribute type = ts.addAttribute(TypeAttribute.class);
      ts.reset();
      while (ts.incrementToken()) {
        sb.append(term)
            .append('\0')
            .append(off.startOffset())
            .append(',')
            .append(off.endOffset())
            .append(',')
            .append(inc.getPositionIncrement())
            .append(',')
            .append(type.type())
            .append('\n');
        count[0]++;
      }
      ts.end();
      sb.append("end:")
          .append(off.endOffset())
          .append(',')
          .append(inc.getPositionIncrement())
          .append('\n');
    }
    return sb.toString();
  }

  public static void main(String[] args) throws Exception {
    if (args.length == 4 && args[0].equals("--dump")) {
      for (String[] src : sources()) {
        if (src[0].equals(args[1])) {
          String text = lines(src).get(Integer.parseInt(args[2]));
          System.out.print(tokens(analyzer(args[3]), text, new int[1]).replace('\0', '|'));
          return;
        }
      }
      throw new IllegalArgumentException("no source " + args[1]);
    }
    Path out = Path.of(args[0]).resolve("standard_tokenizer_corpus");
    Files.createDirectories(out);
    StringBuilder m = new StringBuilder();
    m.append("# GenStandardTokenizerCorpus: see its javadoc for the format.\n");
    long totalBytes = 0;
    long totalTokens = 0;
    List<String[]> sources = sources();
    List<List<String>> texts = new ArrayList<>();
    for (String[] src : sources) {
      List<String> ls = lines(src);
      texts.add(ls);
      m.append("S\t")
          .append(src[0])
          .append('\t')
          .append(src[1])
          .append('\t')
          .append(src[2])
          .append('\t')
          .append(src[3])
          .append('\t')
          .append(ls.size())
          .append('\n');
    }
    for (String config : new String[] {"tok", "std"}) {
      Analyzer a = analyzer(config);
      for (int s = 0; s < sources.size(); s++) {
        List<String> ls = texts.get(s);
        for (int first = 0; first < ls.size(); first += CHUNK) {
          int n = Math.min(CHUNK, ls.size() - first);
          long textDigest = FNV_OFFSET;
          long digest = FNV_OFFSET;
          int[] count = new int[1];
          StringBuilder perLine = new StringBuilder();
          for (int k = first; k < first + n; k++) {
            byte[] text = (ls.get(k) + "\n").getBytes(StandardCharsets.UTF_8);
            textDigest = fnv(textDigest, text);
            if (config.equals("tok")) {
              totalBytes += text.length;
            }
            byte[] toks = tokens(a, ls.get(k), count).getBytes(StandardCharsets.UTF_8);
            digest = fnv(digest, toks);
            if (k > first) {
              perLine.append(',');
            }
            perLine.append(String.format("%08x", (int) fnv(FNV_OFFSET, toks)));
          }
          totalTokens += count[0];
          m.append("C\t")
              .append(sources.get(s)[0])
              .append('\t')
              .append(first)
              .append('\t')
              .append(n)
              .append('\t')
              .append(String.format("%016x", textDigest))
              .append('\t')
              .append(config)
              .append('\t')
              .append(count[0])
              .append('\t')
              .append(String.format("%016x", digest))
              .append('\t')
              .append(perLine)
              .append('\n');
        }
      }
      a.close();
    }
    m.append("# ").append(totalBytes).append(" bytes of text, ").append(totalTokens).append(" tokens\n");
    Files.writeString(out.resolve("manifest.tsv"), m.toString());
  }
}
