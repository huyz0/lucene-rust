import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Generates {@code crates/lucene-analysis/src/java_unicode_script.rs}: the runs of {@code
 * Character.UnicodeScript.of(c)} over the BMP, which Nori's unknown-word grouping asks of each
 * UTF-16 unit. The script names are numbered in order of first appearance; the arrays are laid out
 * as rustfmt lays them out, so the output is the committed file byte for byte.
 *
 * <p>JDK 25 (Unicode 16.0) only: JDK 21 (Unicode 15.0) answers differently in a handful of ranges,
 * so the generator refuses any other feature release rather than emit a different table.
 *
 * <p>Usage: {@code java crates/lucene-analysis/tools/DumpUnicodeScripts.java >
 * crates/lucene-analysis/src/java_unicode_script.rs}
 */
public class DumpUnicodeScripts {
  /** rustfmt's `max_width`; a filled array line stays strictly below it. */
  static final int MAX_WIDTH = 100;

  static final String HEADER =
      """
      //! `java.lang.Character.UnicodeScript.of(int)` over the BMP, as JDK 25
      //! (Unicode 16.0) answers it -- what Nori's unknown-word grouping asks of
      //! each UTF-16 unit: the starts of the runs of one script, and the script
      //! of each run (an index into [`SCRIPT_NAMES`]). JDK 21 (Unicode 15.0)
      //! differs in a handful of rarely used ranges (a few Arabic, Balinese,
      //! Cyrillic Extended-C, CJK stroke and ideographic description code
      //! points); the fixtures avoid them.
      //!
      //! Generated -- do not edit: the output of
      //! `java crates/lucene-analysis/tools/DumpUnicodeScripts.java` on JDK 25. CI's `fixtures`
      //! job regenerates it and diffs.

      """;

  static final String FOOTER =
      """

      /// The script of a UTF-16 unit (`UnicodeScript.of(unit)`), as an index
      /// into [`SCRIPT_NAMES`]. Surrogates are `UNKNOWN`.
      pub fn unicode_script_of(unit: u16) -> u8 {
          let run = RUN_STARTS.partition_point(|&s| s <= unit).saturating_sub(1);
          RUN_SCRIPTS.get(run).copied().unwrap_or(0)
      }

      /// `UnicodeScript.name()` of [`unicode_script_of`]'s answer.
      pub fn unicode_script_name(script: u8) -> &'static str {
          SCRIPT_NAMES
              .get(usize::from(script))
              .copied()
              .unwrap_or("UNKNOWN")
      }

      #[cfg(test)]
      mod tests {
          use super::*;

          #[test]
          fn scripts_of_the_jdk() {
              let name = |u: u16| unicode_script_name(unicode_script_of(u));
              assert_eq!(name(0x41), "LATIN");
              assert_eq!(name(0x20), "COMMON");
              assert_eq!(name(0x0300), "INHERITED");
              assert_eq!(name(0xAC00), "HANGUL");
              assert_eq!(name(0x4E00), "HAN");
              assert_eq!(name(0x3042), "HIRAGANA");
              assert_eq!(name(0xD800), "UNKNOWN");
              assert_eq!(name(0x0), "COMMON");
              assert_eq!(name(0xFFFF), "UNKNOWN");
              assert_eq!(RUN_STARTS.len(), RUN_SCRIPTS.len());
              assert!(RUN_STARTS.windows(2).all(|w| w[0] < w[1]));
              assert_eq!(unicode_script_name(u8::MAX), "UNKNOWN");
          }
      }
      """;

  public static void main(String[] args) {
    if (Runtime.version().feature() != 25) {
      System.err.println(
          "DumpUnicodeScripts: needs JDK 25 (Unicode 16.0), running on "
              + Runtime.version()
              + "; the table would differ");
      System.exit(2);
    }
    Map<Character.UnicodeScript, Integer> index = new LinkedHashMap<>();
    List<String> starts = new ArrayList<>();
    List<String> scripts = new ArrayList<>();
    Character.UnicodeScript prev = null;
    for (int c = 0; c < 0x10000; c++) {
      Character.UnicodeScript s = Character.UnicodeScript.of(c);
      if (s != prev) {
        index.putIfAbsent(s, index.size());
        starts.add(String.format("0x%04X", c));
        scripts.add(Integer.toString(index.get(s)));
        prev = s;
      }
    }
    StringBuilder b = new StringBuilder(HEADER);
    b.append("/// `UnicodeScript.name()` of each script index.\n");
    b.append("pub static SCRIPT_NAMES: [&str; ").append(index.size()).append("] = [\n");
    for (Character.UnicodeScript s : index.keySet()) {
      b.append("    \"").append(s.name()).append("\",\n");
    }
    b.append("];\n\n/// The first unit of each run.\n");
    array(b, "RUN_STARTS", "u16", starts);
    b.append("\n/// The script index of each run.\n");
    array(b, "RUN_SCRIPTS", "u8", scripts);
    b.append(FOOTER);
    System.out.print(b);
  }

  /** A static array as rustfmt fills it: as many items per line as fit in 99 columns. */
  static void array(StringBuilder b, String name, String type, List<String> items) {
    b.append("static ").append(name).append(": [").append(type).append("; ");
    b.append(items.size()).append("] = [\n");
    StringBuilder line = new StringBuilder("   ");
    for (String item : items) {
      if (line.length() + 1 + item.length() + 1 >= MAX_WIDTH) {
        b.append(line).append('\n');
        line = new StringBuilder("   ");
      }
      line.append(' ').append(item).append(',');
    }
    b.append(line).append("\n];\n");
  }
}
