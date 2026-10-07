/*
 * Prints the code points whose java.text.BreakIterator sentence class (JDK running this) is not the
 * one crates/lucene-analysis/src/util/sentence_break.rs derives from Character.getType, as the
 * Rust table `EXCEPTIONS` of that file. Black-box: each code point's class is the class
 * representative whose boundaries it reproduces in every two-anchor context below; no JDK code
 * or data is read. Run with JDK 25 (the JDK the port follows):
 *
 *   java crates/lucene-analysis/tools/GenSentenceBreakClasses.java
 */
import java.text.BreakIterator;
import java.util.Locale;

public class GenSentenceBreakClasses {
  // One representative per class, in the order of sentence_break.rs's `Class`.
  static final String[] REPS = {
    "#", " ", "?", ")", "\"", "1", ".", "A", "a", "।", " ", "̀", "￿"
  };
  static final String[] NAMES = {
    "Other", "Space", "Term", "Close", "Quote", "Digit", "Period", "Upper", "Lower", "Danda",
    "Para", "Ignore", "Done"
  };
  static final String[] ANCHORS = {"a", "A", ".", " ", "?", "̀", "1", "\"", "।", "#"};
  static final BreakIterator BI = BreakIterator.getSentenceInstance(Locale.ROOT);

  static void bounds(String s, StringBuilder out) {
    BI.setText(s);
    int p = BI.first();
    while ((p = BI.next()) != BreakIterator.DONE) out.append(s.codePointCount(0, p)).append(' ');
    out.append(',');
  }

  static String signature(String c) {
    StringBuilder b = new StringBuilder();
    for (String x : ANCHORS) {
      for (String y : ANCHORS) {
        bounds(c + x + y, b);
        bounds(x + c + y, b);
        bounds(x + y + c, b);
      }
    }
    return b.toString();
  }

  /** sentence_break.rs's `class_of` without its exceptions. */
  static int rule(int c) {
    int t = Character.getType(c);
    if (c == 0xFFFF) return 12;
    if (c == 0x2029) return 10;
    if (c == 0x964 || c == 0x965) return 9;
    if (c == '.' || c == 0xFF0E) return 6;
    if (c == '"' || c == '\'') return 4;
    if (c == '!' || c == '?' || c == 0x3002 || c == 0xFF01 || c == 0xFF1F) return 2;
    if (c == 9 || c == 10 || c == 12 || c == 13) return 1;
    switch (t) {
      case Character.SPACE_SEPARATOR, Character.LINE_SEPARATOR -> {
        return 1;
      }
      case Character.NON_SPACING_MARK, Character.ENCLOSING_MARK, Character.FORMAT -> {
        return 11;
      }
      case Character.END_PUNCTUATION, Character.FINAL_QUOTE_PUNCTUATION -> {
        return 3;
      }
      case Character.DECIMAL_DIGIT_NUMBER, Character.LETTER_NUMBER, Character.OTHER_NUMBER -> {
        return 5;
      }
      case Character.LOWERCASE_LETTER -> {
        return 8;
      }
      case Character.UPPERCASE_LETTER,
          Character.TITLECASE_LETTER,
          Character.MODIFIER_LETTER,
          Character.OTHER_LETTER -> {
        return 7;
      }
      default -> {
        return c == ',' ? 5 : 0;
      }
    }
  }

  public static void main(String[] args) {
    String[] reps = new String[REPS.length];
    for (int i = 0; i < REPS.length; i++) reps[i] = signature(REPS[i]);
    int start = -1, last = -1, cls = -1;
    StringBuilder out = new StringBuilder();
    for (int c = 0; c <= 0x110000; c++) {
      int k = -1;
      if (c <= 0x10FFFF && (c < 0xD800 || c > 0xDFFF)) {
        String sig = signature(new String(Character.toChars(c)));
        for (int i = 0; i < reps.length; i++) {
          if (reps[i].equals(sig)) k = i;
        }
        if (k < 0) throw new IllegalStateException("no class for U+" + Integer.toHexString(c));
        if (k == rule(c)) k = -1;
      }
      if (start >= 0 && (k != cls || c != last + 1)) {
        out.append(String.format("    (0x%X, 0x%X, Class::%s),%n", start, last, NAMES[cls]));
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
    System.out.print(out);
  }
}
