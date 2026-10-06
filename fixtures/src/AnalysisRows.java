import java.io.Reader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.function.Supplier;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.CharArraySet;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.Tokenizer;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.analysis.tokenattributes.FlagsAttribute;
import org.apache.lucene.analysis.tokenattributes.KeywordAttribute;
import org.apache.lucene.analysis.tokenattributes.OffsetAttribute;
import org.apache.lucene.analysis.tokenattributes.PayloadAttribute;
import org.apache.lucene.analysis.tokenattributes.PositionIncrementAttribute;
import org.apache.lucene.analysis.tokenattributes.PositionLengthAttribute;
import org.apache.lucene.analysis.tokenattributes.TermFrequencyAttribute;
import org.apache.lucene.analysis.tokenattributes.TermToBytesRefAttribute;
import org.apache.lucene.analysis.tokenattributes.TypeAttribute;
import org.apache.lucene.util.BytesRef;

/**
 * The analysis harness's chain builders and row writer, shared by the analysis generators written
 * after {@code GenAnalysisCommon} ({@code GenAnalysisSynonym}, {@code GenAnalysisLanguages}, ...).
 * Rows are {@code GenAnalysisCommon}'s:
 *
 * <pre>
 *   T line term start end posInc posLen type flags payload(hex|-) keyword(0|1) termFreq
 *   E line finalStart finalEnd finalPosInc
 *   X line ExceptionSimpleName
 * </pre>
 *
 * {@code crates/lucene-analysis/tests/support/mod.rs} is the Rust twin.
 */
final class AnalysisRows {
  private AnalysisRows() {}

  static Analyzer chain(
      Function<Reader, Reader> charFilters,
      Supplier<Tokenizer> tokenizer,
      Function<TokenStream, TokenStream> filters) {
    return new Analyzer() {
      @Override
      protected TokenStreamComponents createComponents(String fieldName) {
        Tokenizer t = tokenizer.get();
        return new TokenStreamComponents(t, filters.apply(t));
      }

      @Override
      protected Reader initReader(String fieldName, Reader reader) {
        return charFilters.apply(reader);
      }
    };
  }

  static Analyzer chain(Supplier<Tokenizer> tokenizer, Function<TokenStream, TokenStream> filters) {
    return chain(r -> r, tokenizer, filters);
  }

  static Analyzer tok(Supplier<Tokenizer> tokenizer) {
    return chain(r -> r, tokenizer, t -> t);
  }

  static CharArraySet set(boolean ignoreCase, String... words) {
    return new CharArraySet(Arrays.asList(words), ignoreCase);
  }

  /** The corpus file's lines, split on '\n' only (a line may hold U+2028 or U+0085). */
  static List<String> corpus(String name) throws Exception {
    String dir = System.getenv().getOrDefault("FIXTURES_CORPUS", "fixtures/corpus");
    String text = Files.readString(Path.of(dir, name), StandardCharsets.UTF_8);
    List<String> lines = new ArrayList<>(Arrays.asList(text.split("\n", -1)));
    if (!lines.isEmpty() && lines.get(lines.size() - 1).isEmpty()) lines.remove(lines.size() - 1);
    return lines;
  }

  static String esc(String s) {
    StringBuilder b = new StringBuilder();
    for (int i = 0; i < s.length(); i++) {
      char c = s.charAt(i);
      switch (c) {
        case '\\' -> b.append("\\\\");
        case '\t' -> b.append("\\t");
        case '\n' -> b.append("\\n");
        case '\r' -> b.append("\\r");
        default -> {
          boolean lone =
              Character.isSurrogate(c)
                  && !(Character.isHighSurrogate(c) && i + 1 < s.length() && Character.isLowSurrogate(s.charAt(i + 1)))
                  && !(Character.isLowSurrogate(c) && i > 0 && Character.isHighSurrogate(s.charAt(i - 1)));
          if (c < 0x20 || lone) {
            b.append(String.format("\\u%04X", (int) c));
          } else {
            b.append(c);
          }
        }
      }
    }
    return b.toString();
  }

  static String hex(BytesRef b) {
    if (b == null) return "-";
    StringBuilder s = new StringBuilder();
    for (int i = 0; i < b.length; i++) s.append(String.format("%02x", b.bytes[b.offset + i] & 0xff));
    return s.toString();
  }

  /** Every line of {@code lines} through {@code a}, as rows. */
  static String rows(Analyzer a, List<String> lines) {
    StringBuilder m = new StringBuilder();
    for (int ln = 0; ln < lines.size(); ln++) {
      TokenStream ts = null;
      try {
        ts = a.tokenStream("f", lines.get(ln));
        CharTermAttribute term = ts.hasAttribute(CharTermAttribute.class) ? ts.getAttribute(CharTermAttribute.class) : null;
        TermToBytesRefAttribute bytes = ts.getAttribute(TermToBytesRefAttribute.class);
        OffsetAttribute off = ts.addAttribute(OffsetAttribute.class);
        PositionIncrementAttribute inc = ts.addAttribute(PositionIncrementAttribute.class);
        PositionLengthAttribute len = ts.addAttribute(PositionLengthAttribute.class);
        TypeAttribute type = ts.addAttribute(TypeAttribute.class);
        FlagsAttribute flags = ts.addAttribute(FlagsAttribute.class);
        PayloadAttribute payload = ts.addAttribute(PayloadAttribute.class);
        KeywordAttribute kw = ts.addAttribute(KeywordAttribute.class);
        TermFrequencyAttribute tf = ts.addAttribute(TermFrequencyAttribute.class);
        ts.reset();
        while (ts.incrementToken()) {
          String t = term != null ? esc(term.toString()) : "#" + hex(bytes.getBytesRef());
          m.append("T\t").append(ln).append('\t').append(t)
              .append('\t').append(off.startOffset()).append('\t').append(off.endOffset())
              .append('\t').append(inc.getPositionIncrement()).append('\t').append(len.getPositionLength())
              .append('\t').append(esc(type.type())).append('\t').append(flags.getFlags())
              .append('\t').append(hex(payload.getPayload())).append('\t').append(kw.isKeyword() ? 1 : 0)
              .append('\t').append(tf.getTermFrequency()).append('\n');
        }
        ts.end();
        m.append("E\t").append(ln).append('\t').append(off.startOffset()).append('\t').append(off.endOffset())
            .append('\t').append(inc.getPositionIncrement()).append('\n');
      } catch (Exception ex) {
        m.append("X\t").append(ln).append('\t').append(ex.getClass().getSimpleName()).append('\n');
      } finally {
        if (ts != null) {
          try {
            ts.close();
          } catch (Exception ignored) {
            // a stream that cannot close is dropped, as the Rust side drops it
          }
        }
      }
    }
    return m.toString();
  }

  /** Writes {@code <out>/<name>.tsv} for every chain. */
  static void writeChains(Path out, Map<String, Supplier<Analyzer>> chains, List<String> lines) throws Exception {
    Files.createDirectories(out);
    for (Map.Entry<String, Supplier<Analyzer>> e : chains.entrySet()) {
      try (Analyzer a = e.getValue().get()) {
        Files.writeString(out.resolve(e.getKey() + ".tsv"), rows(a, lines), StandardCharsets.UTF_8);
      }
    }
  }
}
