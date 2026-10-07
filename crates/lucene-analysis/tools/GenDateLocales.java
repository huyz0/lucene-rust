/*
 * Prints crates/lucene-analysis/src/miscellaneous/date_locales.rs: what
 * java.text.SimpleDateFormat's parse accepts in each available locale, as JDK 25 answers it --
 * for the text fields (G, M/L from three letters, E, a) the set of names a one-field pattern
 * parses whole (Pattern itself is the oracle: every display name the locale's Calendar and
 * DateFormatSymbols offer is tried), the default date pattern, and the number symbols a numeric
 * field reads (NaN, infinity, exponent separator, negative prefix). The data is CLDR's (Unicode
 * licence), read through the JDK's public API; no JDK code is copied. Run with JDK 25:
 *
 *   java crates/lucene-analysis/tools/GenDateLocales.java > crates/lucene-analysis/src/miscellaneous/date_locales.rs
 *
 * With --facts it prints each locale's record as text instead (compare JDK 21 and 25 with it).
 */
import java.text.DateFormat;
import java.text.DateFormatSymbols;
import java.text.DecimalFormat;
import java.text.ParsePosition;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Calendar;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.TreeMap;
import java.util.TreeSet;

public class GenDateLocales {
  static final int[] FIELDS = {Calendar.ERA, Calendar.MONTH, Calendar.MONTH, Calendar.MONTH, Calendar.DAY_OF_WEEK, Calendar.AM_PM};
  /**
   * One pattern per name set, and what follows the name in its text: {@code M} reads CLDR's
   * stand-alone names when it is the pattern's only field and the format names otherwise; {@code L}
   * reads both.
   */
  static final String[] PATTERNS = {"G", "MMM'\u0001'd", "MMM", "LLL", "E", "a"};
  static final String[] SUFFIXES = {"", "\u00011", "", "", "", ""};
  static final int[] STYLES = {
    Calendar.SHORT_FORMAT, Calendar.LONG_FORMAT, Calendar.NARROW_FORMAT,
    Calendar.SHORT_STANDALONE, Calendar.LONG_STANDALONE, Calendar.NARROW_STANDALONE
  };

  /** Where a parse of {@code s} with {@code pattern} ends, -1 where it fails. */
  static int end(String pattern, Locale l, String s) {
    ParsePosition p = new ParsePosition(0);
    new SimpleDateFormat(pattern, l).parse(s, p);
    return p.getErrorIndex() >= 0 ? -1 : p.getIndex();
  }

  /** The names {@code PATTERNS[k]} parses whole in {@code l}, sorted. */
  static List<String> accepted(Locale l, int k) {
    Calendar c = Calendar.getInstance(l);
    DateFormatSymbols d = DateFormatSymbols.getInstance(l);
    TreeSet<String> cand = new TreeSet<>();
    for (int st : STYLES) {
      Map<String, Integer> m = c.getDisplayNames(FIELDS[k], st, l);
      if (m != null) cand.addAll(m.keySet());
    }
    for (String[] x : new String[][] {d.getEras(), d.getMonths(), d.getShortMonths(), d.getWeekdays(), d.getShortWeekdays(), d.getAmPmStrings()}) {
      cand.addAll(Arrays.asList(x));
    }
    List<String> out = new ArrayList<>();
    for (String s : cand) if (!s.isEmpty() && end(PATTERNS[k], l, s + SUFFIXES[k]) == s.length() + SUFFIXES[k].length()) out.add(s);
    return out;
  }

  /** A locale's record: the six name sets, then the default pattern and the number symbols. */
  static List<Object> record(Locale l) {
    List<Object> r = new ArrayList<>();
    for (int k = 0; k < PATTERNS.length; k++) r.add(accepted(l, k));
    DecimalFormat df = (DecimalFormat) new SimpleDateFormat("y", l).getNumberFormat();
    if (!df.getPositivePrefix().isEmpty() || !df.getPositiveSuffix().isEmpty() || !df.getNegativeSuffix().isEmpty()) {
      throw new IllegalStateException(l + ": a number affix beyond the negative prefix");
    }
    r.add(((SimpleDateFormat) DateFormat.getDateInstance(DateFormat.DEFAULT, l)).toPattern());
    r.add(df.getDecimalFormatSymbols().getNaN());
    r.add(df.getDecimalFormatSymbols().getInfinity());
    r.add(df.getDecimalFormatSymbols().getExponentSeparator());
    r.add(df.getNegativePrefix());
    return r;
  }

  /** The locales the table holds: no variant (its tag round-trips) and a Gregorian or Buddhist calendar. */
  static Map<String, Locale> locales() {
    Map<String, Locale> m = new TreeMap<>();
    for (Locale l : Locale.getAvailableLocales()) {
      if (!l.getVariant().isEmpty()) continue;
      String cal = Calendar.getInstance(l).getCalendarType();
      if (!cal.equals("gregory") && !cal.equals("buddhist")) continue;
      m.put(l.toLanguageTag(), l);
    }
    return m;
  }

  static String lit(String s) {
    StringBuilder b = new StringBuilder("\"");
    s.codePoints().forEach(c -> {
      if (c == '"' || c == '\\') b.append('\\').appendCodePoint(c);
      else if (c < 0x20 || c == 0x7f || (c >= 0x80 && (Character.getType(c) == Character.FORMAT || Character.isSpaceChar(c)))) b.append(String.format("\\u{%x}", c));
      else b.appendCodePoint(c);
    });
    return b.append('"').toString();
  }

  public static void main(String[] args) {
    Map<String, Locale> locales = locales();
    if (args.length > 0 && args[0].equals("--facts")) {
      for (var e : locales.entrySet()) System.out.println(e.getKey() + "\t" + record(e.getValue()));
      return;
    }
    Map<String, Integer> strings = new LinkedHashMap<>();
    Map<List<Object>, Integer> records = new LinkedHashMap<>();
    Map<String, Integer> index = new TreeMap<>();
    for (var e : locales.entrySet()) {
      List<Object> r = record(e.getValue());
      Integer id = records.get(r);
      if (id == null) records.put(r, id = records.size());
      index.put(e.getKey(), id);
    }
    StringBuilder data = new StringBuilder();
    for (List<Object> r : records.keySet()) {
      data.append("    LocaleData {");
      String[] names = {"eras", "months", "months_alone", "standalone_months", "weekdays", "am_pm"};
      for (int k = 0; k < names.length; k++) {
        data.append(' ').append(names[k]).append(": &[");
        @SuppressWarnings("unchecked") List<String> set = (List<String>) r.get(k);
        for (int i = 0; i < set.size(); i++) data.append(i == 0 ? "" : ", ").append(strings.computeIfAbsent(set.get(i), x -> strings.size()));
        data.append("],");
      }
      String[] scalars = {"default_pattern", "nan", "infinity", "exponent", "minus"};
      for (int k = 0; k < 5; k++) data.append(' ').append(scalars[k]).append(": ").append(strings.computeIfAbsent((String) r.get(PATTERNS.length + k), x -> strings.size())).append(',');
      data.append(" },\n");
    }
    StringBuilder o = new StringBuilder();
    o.append("//! What `java.text.SimpleDateFormat`'s parse accepts in each available\n");
    o.append("//! locale, as JDK 25 answers it (CLDR data, Unicode licence). Generated by\n");
    o.append("//! `tools/GenDateLocales.java`; do not edit.\n\n");
    o.append("/// A locale's record: the names each text field accepts, the default date\n");
    o.append("/// pattern and a numeric field's symbols, as indices into [`STRINGS`].\n");
    o.append("pub(crate) struct LocaleData {\n");
    o.append("    /// `G`.\n    pub(crate) eras: &'static [u16],\n");
    o.append("    /// `M` from three letters, beside another field.\n    pub(crate) months: &'static [u16],\n");
    o.append("    /// `M` from three letters, the pattern's only field.\n    pub(crate) months_alone: &'static [u16],\n");
    o.append("    /// `L` from three letters.\n    pub(crate) standalone_months: &'static [u16],\n");
    o.append("    /// `E`.\n    pub(crate) weekdays: &'static [u16],\n");
    o.append("    /// `a`.\n    pub(crate) am_pm: &'static [u16],\n");
    o.append("    /// `DateFormat.getDateInstance(DEFAULT, locale)`'s pattern.\n    pub(crate) default_pattern: u16,\n");
    o.append("    /// `DecimalFormatSymbols.getNaN()`.\n    pub(crate) nan: u16,\n");
    o.append("    /// `getInfinity()`.\n    pub(crate) infinity: u16,\n");
    o.append("    /// `getExponentSeparator()`.\n    pub(crate) exponent: u16,\n");
    o.append("    /// The negative prefix (the exponent's sign too).\n    pub(crate) minus: u16,\n");
    o.append("}\n\n");
    o.append("/// The strings the records name.\n");
    o.append("pub(crate) static STRINGS: [&str; ").append(strings.size()).append("] = [\n");
    int col = 0;
    StringBuilder line = new StringBuilder("   ");
    for (String s : strings.keySet()) {
      String l = " " + lit(s) + ",";
      if (line.length() + l.length() > 100) { o.append(line).append('\n'); line = new StringBuilder("   "); }
      line.append(l);
    }
    o.append(line).append("\n];\n\n");
    o.append("/// The distinct records.\n");
    o.append("pub(crate) static DATA: [LocaleData; ").append(records.size()).append("] = [\n").append(data).append("];\n\n");
    o.append("/// Each available locale's `toLanguageTag()` (no variant; a Gregorian or\n");
    o.append("/// Buddhist calendar), sorted, and its record.\n");
    o.append("pub(crate) static LOCALES: [(&str, u16); ").append(index.size()).append("] = [\n");
    line = new StringBuilder("   ");
    for (var e : index.entrySet()) {
      String l = " (" + lit(e.getKey()) + ", " + e.getValue() + "),";
      if (line.length() + l.length() > 100) { o.append(line).append('\n'); line = new StringBuilder("   "); }
      line.append(l);
    }
    o.append(line).append("\n];\n");
    System.out.print(o);
  }
}
