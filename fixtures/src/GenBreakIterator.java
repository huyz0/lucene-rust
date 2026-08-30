import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.text.BreakIterator;
import java.util.Locale;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.analysis.tokenattributes.OffsetAttribute;

/**
 * Generates `break_iterator/manifest.properties`: the sentence boundaries
 * `java.text.BreakIterator.getSentenceInstance(Locale.ROOT)` produces for a set
 * of texts.
 *
 * <p>Why this exists: `UnifiedHighlighter` cuts passages with
 * `BreakIterator.getSentenceInstance`, so the Rust port's
 * `highlighter::sentence_boundaries` has to agree with it, and until c12 it did
 * not (it carried a hand-rolled abbreviation list; the JDK suppresses nothing).
 * The replacement is UAX #29 via `unicode-segmentation`, which is the same
 * specification the JDK implements — but "the same specification" is an
 * argument, not evidence, and a JDK or CLDR data bump could reintroduce a
 * divergence silently. Writing the JDK's actual answers into a manifest puts
 * that claim under `scripts/gen-fixtures.sh --check` like every other
 * Java-derived expectation in this repo.
 *
 * <p><b>Both locales are emitted</b> because `UnifiedHighlighter`'s default is
 * `Locale.ROOT` while an application may pass `Locale.ENGLISH`; the Rust port
 * is untailored, so the fixture records whether the two agree rather than
 * assuming it.
 *
 * <p><b>Every text in {@link #TEXTS} is ASCII on purpose.</b> `BreakIterator`
 * reports UTF-16 offsets and the Rust port reports UTF-8 byte offsets; emitting
 * the sliced *substrings* rather than the offsets sidesteps the unit difference
 * entirely, so the Rust test compares text to text.
 *
 * <p><b>{@link #OFFSET_TEXTS} exists to make that unit difference itself
 * testable</b> (M2 sweep `c29-search-carryovers`, closing c23's F13). Lucene's
 * `OffsetAttribute` offsets are indices into the original `String`, i.e. **UTF-16
 * code units**, and every highlighter slices the stored text with them. A Rust
 * highlighter that reads them as UTF-8 byte offsets, or as Unicode scalar
 * (`char`) counts, produces snippets cut in the wrong place for non-ASCII text
 * -- and no fixture in this repo could catch that, because they are all ASCII,
 * where all three units coincide. These texts deliberately are not: they carry
 * 2-byte Latin-1, 3-byte CJK, combining marks, and supplementary-plane
 * characters (which are ONE Unicode scalar but TWO UTF-16 code units, the case
 * that separates "char count" from "UTF-16 code unit" as well). For each token
 * the manifest records `start,end,slice`, where `slice` is
 * `text.substring(start, end)` -- Java's own UTF-16-indexed slice. The Rust
 * test asserts its highlighter marks exactly that slice, so it compares text to
 * text and the offsets never have to be re-derived on the Rust side.
 */
public class GenBreakIterator {
  /** Manifest list separator; no text below contains it. */
  static final char SEP = '';

  static final String[] TEXTS = {
    "Mr. Smith went home. He slept well.",
    "He finished 21st. She started next.",
    "She said \"stop.\" Then she left.",
    "no terminator at all here",
    "Dr. Who visited St. Paul. Then left.",
    "One.\nTwo.\n\nThree.",
    "Ends without a terminator",
    "One.\n\n\n\n\n\n\n\nTwo.",
    "(See note.) Next sentence begins here.",
    "He finished. Then, after a pause, he began again. Finally he stopped.",
    "",
  };

  public static void main(String[] args) throws IOException {
    Path out = Path.of(args[0]).resolve("break_iterator");
    if (Files.exists(out)) {
      deleteRecursive(out);
    }
    Files.createDirectories(out);

    StringBuilder m = new StringBuilder();
    m.append("count=").append(TEXTS.length).append('\n');
    m.append("java_version=").append(System.getProperty("java.specification.version")).append('\n');
    for (int i = 0; i < TEXTS.length; i++) {
      m.append("text.").append(i).append('=').append(escape(TEXTS[i])).append('\n');
      m.append("root.").append(i).append('=').append(slices(TEXTS[i], Locale.ROOT)).append('\n');
      m.append("english.")
          .append(i)
          .append('=')
          .append(slices(TEXTS[i], Locale.ENGLISH))
          .append('\n');
    }
    m.append("offset_count=").append(OFFSET_TEXTS.length).append('\n');
    try (StandardAnalyzer analyzer = new StandardAnalyzer()) {
      for (int i = 0; i < OFFSET_TEXTS.length; i++) {
        String text = OFFSET_TEXTS[i];
        m.append("offset_text.").append(i).append('=').append(escape(text)).append('\n');
        m.append("offset_utf16_length.").append(i).append('=').append(text.length()).append('\n');
        m.append("offset_tokens.").append(i).append('=').append(tokens(analyzer, text)).append('\n');
      }
    }
    Files.writeString(out.resolve("manifest.properties"), m.toString());
    System.out.println("wrote break_iterator/ fixture directory");
  }

  /** The sentences `text` breaks into, `SEP`-joined and escaped. */
  static String slices(String text, Locale locale) {
    BreakIterator bi = BreakIterator.getSentenceInstance(locale);
    bi.setText(text);
    StringBuilder sb = new StringBuilder();
    int start = bi.first();
    for (int end = bi.next(); end != BreakIterator.DONE; start = end, end = bi.next()) {
      if (sb.length() > 0) {
        sb.append(SEP);
      }
      sb.append(escape(text.substring(start, end)));
    }
    return sb.toString();
  }

  /**
   * Texts whose UTF-8 byte length, Unicode scalar count and UTF-16 code-unit
   * length all differ, so a manifest built from them can only be reproduced by
   * a reader using the same unit Lucene does.
   */
  static final String[] OFFSET_TEXTS = {
    // 2-byte UTF-8, 1 UTF-16 unit each: byte offsets diverge, char offsets do not.
    "café naïve Zürich",
    // 3-byte UTF-8 CJK plus ASCII.
    "\u6771\u4EAC tokyo \u5927\u962A osaka",
    // Supplementary plane: U+1F600 and U+1D400 are one Unicode scalar each but
    // TWO UTF-16 code units, so a char-count reader diverges from Lucene here
    // even though a BMP-only text would not tell them apart.
    "alpha \uD83D\uDE00 beta \uD835\uDC00 gamma",
    // Combining mark: "cafe" + U+0301 renders as "café" but is 5 code units.
    "cafe\u0301 latte",
    // Mixed: everything above in one field value.
    "n\u00E4ive \u6771\u4EAC \uD83D\uDE00 plain words",
  };

  /**
   * `term:start,end:slice` for every token `analyzer` produces from `text`,
   * SEP-joined. `start`/`end` come straight off `OffsetAttribute` and `slice`
   * is `text.substring(start, end)` -- the same indices, used the way a
   * highlighter uses them.
   */
  static String tokens(StandardAnalyzer analyzer, String text) throws IOException {
    StringBuilder sb = new StringBuilder();
    try (TokenStream ts = analyzer.tokenStream("body", text)) {
      CharTermAttribute term = ts.addAttribute(CharTermAttribute.class);
      OffsetAttribute offset = ts.addAttribute(OffsetAttribute.class);
      ts.reset();
      while (ts.incrementToken()) {
        if (sb.length() > 0) {
          sb.append(SEP);
        }
        sb.append(escape(term.toString()))
            .append(':')
            .append(offset.startOffset())
            .append(',')
            .append(offset.endOffset())
            .append(':')
            .append(escape(text.substring(offset.startOffset(), offset.endOffset())));
      }
      ts.end();
    }
    return sb.toString();
  }

  static String escape(String s) {
    return s.replace("\\", "\\\\").replace("\n", "\\n").replace(String.valueOf(SEP), "\\u0001");
  }

  static void deleteRecursive(Path p) throws IOException {
    try (var s = Files.walk(p)) {
      s.sorted(java.util.Comparator.reverseOrder())
          .forEach(
              q -> {
                try {
                  Files.delete(q);
                } catch (IOException e) {
                  throw new RuntimeException(e);
                }
              });
    }
  }
}
