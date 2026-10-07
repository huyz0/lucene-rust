import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.text.BreakIterator;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Random;
import java.util.function.Supplier;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.analysis.tokenattributes.OffsetAttribute;
import org.apache.lucene.analysis.tokenattributes.PositionIncrementAttribute;
import org.apache.lucene.analysis.util.SegmentingTokenizerBase;

/**
 * M12 T12.6: {@code java.text.BreakIterator.getSentenceInstance(Locale.ROOT)} and {@code
 * SegmentingTokenizerBase} (crates/lucene-analysis/src/util/sentence_break.rs,
 * segmenting_tokenizer_base.rs; tests/analysis_segmenting_fixtures.rs compares).
 *
 * <ul>
 *   <li>{@code sentence_classes.txt}: {@code lo hi class} -- every code point's sentence class,
 *       told apart by {@link #PROBES} (each class answers them differently), leaving out the code
 *       points JDK 21 and 25 classify differently ({@link #jdkDependent}).
 *   <li>{@code sentences.txt}: {@code units<TAB>boundaries} (UTF-16 units in hex, so lone
 *       surrogates survive) -- {@link #TEXTS} and 20,000 seeded strings over {@link #POOL}.
 *   <li>{@code <chain>.tsv} over {@code texts.txt}: two {@code SegmentingTokenizerBase}
 *       subclasses written here over texts longer than its 1,024-unit buffer.
 * </ul>
 *
 * Byte-identical under JDK 21 and 25.
 */
public class GenAnalysisSegmenting {

  /** One representative per class, in sentence_break.rs's {@code Class} order. */
  static final String[] REPS = {
    "#", " ", "?", ")", "\"", "1", ".", "A", "a", "।", " ", "̀", "￿"
  };

  static final String[] ANCHORS = {"a", "A", ".", " ", "?", "̀", "1", "\"", "।", "#"};

  /** Contexts {@code (x, c at position pos, y)} that tell the thirteen classes apart. */
  static final List<String[]> PROBES = new ArrayList<>();

  static final BreakIterator BI = BreakIterator.getSentenceInstance(Locale.ROOT);

  static String bounds(String s) {
    BI.setText(s);
    StringBuilder r = new StringBuilder();
    int p = BI.first();
    while ((p = BI.next()) != BreakIterator.DONE) r.append(s.codePointCount(0, p)).append(' ');
    return r.toString();
  }

  static String probe(String c, String[] t) {
    return switch (t[2]) {
      case "0" -> bounds(c + t[0] + t[1]);
      case "1" -> bounds(t[0] + c + t[1]);
      default -> bounds(t[0] + t[1] + c);
    };
  }

  /** Picks probes greedily until every pair of representatives is told apart. */
  static void choose() {
    List<String[]> all = new ArrayList<>();
    for (String x : ANCHORS) for (String y : ANCHORS) for (String p : new String[] {"0", "1", "2"}) all.add(new String[] {x, y, p});
    List<int[]> pairs = new ArrayList<>();
    for (int i = 0; i < REPS.length; i++) for (int j = i + 1; j < REPS.length; j++) pairs.add(new int[] {i, j});
    while (!pairs.isEmpty()) {
      String[] best = null;
      int bestN = 0;
      for (String[] t : all) {
        int n = 0;
        for (int[] pr : pairs) if (!probe(REPS[pr[0]], t).equals(probe(REPS[pr[1]], t))) n++;
        if (n > bestN) {
          bestN = n;
          best = t;
        }
      }
      if (best == null) throw new IllegalStateException("classes not separable");
      PROBES.add(best);
      final String[] b = best;
      pairs.removeIf(pr -> !probe(REPS[pr[0]], b).equals(probe(REPS[pr[1]], b)));
    }
  }

  static String signature(String c) {
    StringBuilder b = new StringBuilder();
    for (String[] t : PROBES) b.append(probe(c, t)).append(',');
    return b.toString();
  }

  /**
   * GenAnalysisCommon's code points whose Character properties differ between JDK 21 and 25,
   * plus the code points the sentence tables class differently next to them: the mark before a
   * changed one, and the unassigned gaps beside Unicode 16's new blocks.
   */
  static boolean jdkDependent(int cp) {
    return GenAnalysisCommon.jdkDependent(cp)
        || GenAnalysisCommon.jdkDependent(cp + 1)
        || (cp >= 0x18CD6 && cp <= 0x18CFE)
        || (cp >= 0x2EBE1 && cp <= 0x2EBEF);
  }

  static String classes() {
    String[] reps = new String[REPS.length];
    for (int i = 0; i < REPS.length; i++) reps[i] = signature(REPS[i]);
    StringBuilder out = new StringBuilder();
    int start = -1, last = -1, cls = -1;
    for (int c = 0; c <= 0x110000; c++) {
      int k = -1;
      if (c <= 0x10FFFF && (c < 0xD800 || c > 0xDFFF)) {
        if (!jdkDependent(c)) {
          String sig = signature(new String(Character.toChars(c)));
          for (int i = 0; i < reps.length && k < 0; i++) if (reps[i].equals(sig)) k = i;
          if (k < 0) throw new IllegalStateException("U+" + Integer.toHexString(c));
        }
      }
      if (start >= 0 && (k != cls || c != last + 1)) {
        out.append(String.format("%X %X %d%n", start, last, cls));
        start = -1;
      }
      if (k >= 0) {
        if (start < 0) {
          start = c;
          cls = k;
        }
        last = c;
      }
    }
    return out.toString();
  }

  /** Code points of every class, each with the same class under JDK 21 and 25. */
  static final int[] POOL = {
    // Other: symbols, opening punctuation, Pi, Mc, controls, private use, unassigned, lone surrogates
    '#', '(', '-', '@', '*', 0xAB, 0x903, 0x85, 0x0B, 0xE000, 0x378, 0x1F600, 0xD800, 0xDC00,
    // Space
    ' ', ' ', ' ', ' ', '\t', '\n', '\r', '\f', 0xA0, 0x3000, 0x2028,
    // Term
    '!', '?', 0x3002, 0xFF01, 0xFF1F,
    // Close
    ')', ']', 0xBB, 0x300D,
    // Quote
    '"', '\'',
    // Digit (and the digit-classed exception U+1D169)
    '1', '7', ',', 0x661, 0x2160, 0xB2, 0x1D7CE, 0x1D169,
    // Period
    '.', '.', '.', 0xFF0E,
    // Upper (and the letter-classed gap U+2A6E0)
    'A', 'T', 'Z', 0xC9, 0x1C5, 0x2B0, 0x4E2D, 0x6587, 0xE01, 0x5D0, 0x10400, 0x20000, 0x2A6E0,
    // Lower
    'a', 'e', 'o', 'z', 0xE9, 0x3C3, 0x10428,
    // Danda, Para
    0x964, 0x965, 0x2029,
    // Ignore
    0x300, 0xAD, 0x200B, 0xE31, 0x1D167, 0xE0020,
    // Done
    0xFFFF
  };

  static final String[] TEXTS = {
    "Hello world. This is it. no more! Yes? ok.",
    "Mr. Smith went to Washington. He said \"Hi.\" Then (quietly) left.",
    "Version 3.5 is out. e.g. this one... And 1.2.3? Sure! (Really.) \"Yes.\" 'No.'",
    "这是一个句子。这是另一个！还有吗？没有。",
    "中文和English混合。Mixed text. 第二句",
    "यह एक वाक्य है। यह दूसरा है॥ तीसरा",
    "ภาษาไทยไม่มีจุด ประโยคที่สอง",
    "a.  b. c. D   e f. G",
    "",
  };

  static String hex(String s) {
    StringBuilder b = new StringBuilder();
    for (int i = 0; i < s.length(); i++) {
      if (i > 0) b.append(' ');
      b.append(String.format("%04X", (int) s.charAt(i)));
    }
    return b.toString();
  }

  static void sentence(StringBuilder out, String s) {
    out.append(hex(s)).append('\t');
    BI.setText(s);
    int p = BI.first();
    boolean first = true;
    while ((p = BI.next()) != BreakIterator.DONE) {
      if (!first) out.append(' ');
      out.append(p);
      first = false;
    }
    out.append('\n');
  }

  static String sentences() {
    StringBuilder out = new StringBuilder();
    for (String t : TEXTS) sentence(out, t);
    Random r = new Random(2026);
    for (int i = 0; i < 20000; i++) {
      int len = 1 + r.nextInt(i % 4 == 0 ? 40 : 10);
      StringBuilder b = new StringBuilder();
      for (int j = 0; j < len; j++) b.appendCodePoint(POOL[r.nextInt(POOL.length)]);
      sentence(out, b.toString());
    }
    return out.toString();
  }

  /** Each sentence one token. */
  static final class WholeSentenceTokenizer extends SegmentingTokenizerBase {
    final CharTermAttribute termAtt = addAttribute(CharTermAttribute.class);
    final OffsetAttribute offsetAtt = addAttribute(OffsetAttribute.class);
    int sentenceStart, sentenceEnd;
    boolean hasSentence;

    WholeSentenceTokenizer() {
      super(BreakIterator.getSentenceInstance(Locale.ROOT));
    }

    @Override
    protected void setNextSentence(int sentenceStart, int sentenceEnd) {
      this.sentenceStart = sentenceStart;
      this.sentenceEnd = sentenceEnd;
      hasSentence = true;
    }

    @Override
    protected boolean incrementWord() {
      if (!hasSentence) return false;
      hasSentence = false;
      clearAttributes();
      termAtt.copyBuffer(buffer, sentenceStart, sentenceEnd - sentenceStart);
      offsetAtt.setOffset(correctOffset(offset + sentenceStart), correctOffset(offset + sentenceEnd));
      return true;
    }
  }

  /** Letter-or-digit runs, the first of each sentence ten positions on. */
  static final class SentenceAndWordTokenizer extends SegmentingTokenizerBase {
    final CharTermAttribute termAtt = addAttribute(CharTermAttribute.class);
    final OffsetAttribute offsetAtt = addAttribute(OffsetAttribute.class);
    final PositionIncrementAttribute posIncAtt = addAttribute(PositionIncrementAttribute.class);
    int sentenceStart, sentenceEnd, wordStart, wordEnd, posBoost = -1;

    SentenceAndWordTokenizer() {
      super(BreakIterator.getSentenceInstance(Locale.ROOT));
    }

    @Override
    public void reset() throws java.io.IOException {
      super.reset();
      sentenceStart = sentenceEnd = wordStart = wordEnd = 0;
      posBoost = -1;
    }

    @Override
    protected void setNextSentence(int sentenceStart, int sentenceEnd) {
      this.wordStart = this.wordEnd = this.sentenceStart = sentenceStart;
      this.sentenceEnd = sentenceEnd;
      posBoost += 10;
    }

    @Override
    protected boolean incrementWord() {
      wordStart = wordEnd;
      while (wordStart < sentenceEnd && !Character.isLetterOrDigit(buffer[wordStart])) wordStart++;
      if (wordStart == sentenceEnd) return false;
      wordEnd = wordStart + 1;
      while (wordEnd < sentenceEnd && Character.isLetterOrDigit(buffer[wordEnd])) wordEnd++;
      clearAttributes();
      termAtt.copyBuffer(buffer, wordStart, wordEnd - wordStart);
      offsetAtt.setOffset(correctOffset(offset + wordStart), correctOffset(offset + wordEnd));
      posIncAtt.setPositionIncrement(posIncAtt.getPositionIncrement() + posBoost);
      posBoost = 0;
      return true;
    }
  }

  /** Texts around the 1,024-unit buffer: with and without safe ends, sentences across it. */
  static List<String> segmentingTexts() {
    List<String> texts = new ArrayList<>();
    for (String t : TEXTS) texts.add(t);
    Random r = new Random(7);
    String[] words = {"alpha", "Beta", "gamma", "中文", "😀x", "é", "3.14", "U.S.", "ok"};
    String[] seps = {" ", " ", " ", ". ", "! ", "? ", "\n", "\r\n", " ", "\u0085", " ", ", ", "。"};
    for (int i = 0; i < 24; i++) {
      StringBuilder b = new StringBuilder();
      int target = 200 + r.nextInt(i % 3 == 0 ? 400 : 5000);
      boolean noSafe = i % 6 == 5;
      while (b.length() < target) {
        b.append(words[r.nextInt(words.length)]);
        String sep = seps[r.nextInt(seps.length)];
        if (noSafe && (sep.contains("\n") || sep.contains("\r") || sep.contains(" ") || sep.contains("\u0085") || sep.contains(" "))) sep = " ";
        b.append(sep);
      }
      texts.add(b.toString());
    }
    return texts;
  }

  public static void main(String[] args) throws Exception {
    Path out = Path.of(args[0]).resolve("analysis_segmenting");
    Files.createDirectories(out);
    choose();
    Files.writeString(out.resolve("sentence_classes.txt"), classes(), StandardCharsets.UTF_8);
    Files.writeString(out.resolve("sentences.txt"), sentences(), StandardCharsets.UTF_8);
    List<String> texts = segmentingTexts();
    StringBuilder t = new StringBuilder();
    for (String s : texts) t.append(AnalysisRows.esc(s)).append('\n');
    Files.writeString(out.resolve("texts.txt"), t.toString(), StandardCharsets.UTF_8);
    Map<String, Supplier<Analyzer>> c = new LinkedHashMap<>();
    c.put("whole_sentence", () -> AnalysisRows.tok(WholeSentenceTokenizer::new));
    c.put("sentence_words", () -> AnalysisRows.tok(SentenceAndWordTokenizer::new));
    AnalysisRows.writeChains(out, c, texts);
  }
}
