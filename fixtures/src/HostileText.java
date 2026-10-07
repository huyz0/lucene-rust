import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.function.Supplier;

/**
 * The hostile random-text sweep shared by {@code GenAnalysisKuromoji} and {@code GenAnalysisNori}:
 * a seeded corpus ({@link #lines}) drawn from pools no tokenizer is tuned for -- kana, half- and
 * full-width forms, kanji in and out of the BMP, emoji, combining marks, joiners, variation
 * selectors, byte order marks, Hangul syllables and jamo (including the extended ranges), Thai,
 * Greek, Cyrillic, Arabic digits, private use, C0 controls, radicals, compatibility ideographs,
 * specials and real words of both languages -- and its tokenizer rows summarised per line ({@link
 * #digests}), so every mode combination fits in one small file.
 *
 * <p>The corpus is identical under JDK 21 and JDK 25 ({@link java.util.Random} is specified), and
 * so are the digests: both generators' output was byte-identical under the two when it was
 * written (their Unicode versions, 15.0 and 16.0, differ in no character the tokenizers ask about
 * here).
 */
final class HostileText {
  private HostileText() {}

  static int[] codePoints(String s) {
    return s.codePoints().toArray();
  }

  /** {@code n} lines from {@code seed}; no line holds a line terminator. */
  static List<String> lines(long seed, int n) {
    Random r = new Random(seed);
    List<Supplier<String>> pools = new ArrayList<>();
    int[][] ranges = {
      {0x3041, 0x3096}, {0x30A1, 0x30FA}, {0xFF61, 0xFF9F}, {0xFF01, 0xFF5E}, {0x21, 0x7E},
      {0x4E00, 0x4FFF}, {0x20000, 0x2A6DF}, {0x1F300, 0x1FAFF}, {0xAC00, 0xD7A3}, {0x1100, 0x11FF},
      {0x3131, 0x318E}, {0xA960, 0xA97C}, {0xD7B0, 0xD7FB}, {0x0E01, 0x0E5B}, {0x0391, 0x03C9},
      {0x0410, 0x044F}, {0x0660, 0x0669}, {0xE000, 0xE0FF}, {0x01, 0x1F}, {0x2F00, 0x2FD5},
      {0x3400, 0x4DBF}, {0xF900, 0xFAFF}, {0x2E80, 0x2EF3}, {0x31F0, 0x31FF}, {0x1B000, 0x1B11E},
      {0xFFF0, 0xFFFD},
    };
    for (int[] range : ranges) {
      pools.add(() -> new String(Character.toChars(range[0] + r.nextInt(range[1] - range[0] + 1))));
    }
    String[] choices = {
      // iteration marks, the prolonged sound mark, the wave dash, the middle dot
      "々ゝゞヽヾ〃〻ー〜・",
      // (semi-)voiced sound marks, a combining acute, ZWJ, VS16, VS17, ZWSP, BOM
      "゙゚́‍️󠄀​﻿",
      // spaces: ASCII, ideographic, no-break, tab, thin, vertical tab
      " 　 \t \u000B",
      "、。！？「」（）,.!?;:-—…",
    };
    for (String choice : choices) {
      int[] cps = codePoints(choice);
      pools.add(() -> new String(Character.toChars(cps[r.nextInt(cps.length)])));
    }
    String[] words = {
      "関西国際空港", "東京都", "日本経済新聞", "형태소분석", "대한민국", "가곡역", "세종시", "삼성전자", "ｶﾞｷﾞ", "ﾊﾟﾋﾟ",
      "１２３４５", "一二三四五六", "三十五万", "오늘은", "했습니다", "ぁぃぅ", "ヴァ", "ヶ月", "〆切",
    };
    pools.add(() -> words[r.nextInt(words.length)]);
    int[] lengths = {1, 2, 5, 10, 30, 80, 200};
    List<String> out = new ArrayList<>();
    for (int i = 0; i < n; i++) {
      int len = lengths[r.nextInt(lengths.length)];
      StringBuilder b = new StringBuilder();
      for (int k = 0; k < len; k++) b.append(pools.get(r.nextInt(pools.size())).get());
      out.add(b.toString().replace("\n", "").replace("\r", ""));
    }
    return out;
  }

  /**
   * A row with {@link AnalysisRows#esc}'s lone-surrogate escapes read as U+FFFD, as the Rust
   * tests' {@code normalise_expected} reads them (a Rust string holds no lone surrogate).
   */
  static String normalise(String row) {
    StringBuilder out = new StringBuilder();
    int i = 0;
    for (int j; (j = row.indexOf("\\u", i)) >= 0; ) {
      String head = row.substring(i, j);
      out.append(head);
      int backslashes = 0;
      for (int k = head.length() - 1; k >= 0 && head.charAt(k) == '\\'; k--) backslashes++;
      int code = -1;
      if (j + 6 <= row.length()) {
        try {
          code = Integer.parseInt(row.substring(j + 2, j + 6), 16);
        } catch (NumberFormatException e) {
          code = -1;
        }
      }
      if (backslashes % 2 == 0 && code >= 0xD800 && code <= 0xDFFF) {
        out.append('�');
        i = j + 6;
      } else {
        out.append("\\u");
        i = j + 2;
      }
    }
    return out.append(row.substring(i)).toString();
  }

  /** FNV-1a, 64 bits, over UTF-8: the Rust test computes the same. */
  static long fnv1a(String s) {
    long h = 0xcbf29ce484222325L;
    for (byte x : s.getBytes(StandardCharsets.UTF_8)) {
      h ^= x & 0xff;
      h *= 0x100000001b3L;
    }
    return h;
  }

  /**
   * One configuration's rows ({@code T}/{@code E}/{@code X}, as the generator's {@code rows}
   * writes them over every line with one tokenizer, so reuse is exercised), as {@code config line
   * rows digest} per line: the number of rows the line produced and the FNV-1a of those rows,
   * {@link #normalise}d,
   * joined with {@code \n}. The {@code K} row of attribute keys is dropped.
   */
  static String digests(String config, String rows, int lines) {
    Map<Integer, StringBuilder> byLine = new LinkedHashMap<>();
    Map<Integer, Integer> counts = new LinkedHashMap<>();
    for (int i = 0; i < lines; i++) {
      byLine.put(i, new StringBuilder());
      counts.put(i, 0);
    }
    for (String row : rows.split("\n")) {
      if (row.isEmpty() || row.startsWith("K\t")) continue;
      int ln = Integer.parseInt(row.split("\t", 3)[1]);
      StringBuilder b = byLine.get(ln);
      if (b.length() > 0) b.append('\n');
      b.append(normalise(row));
      counts.merge(ln, 1, Integer::sum);
    }
    StringBuilder m = new StringBuilder();
    for (int i = 0; i < lines; i++) {
      m.append(config).append('\t').append(i).append('\t').append(counts.get(i)).append('\t')
          .append(String.format("%016x", fnv1a(byLine.get(i).toString()))).append('\n');
    }
    return m.toString();
  }
}
