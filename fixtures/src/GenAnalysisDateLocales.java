import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.text.DateFormat;
import java.text.DecimalFormat;
import java.text.DecimalFormatSymbols;
import java.text.ParsePosition;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Random;
import java.util.Set;
import java.util.TimeZone;

/**
 * M12 T12.7: {@code SimpleDateFormat}'s parse outside {@code Locale.ENGLISH}, for {@code
 * DateRecognizerFilterFactory}'s {@code locale} ({@code miscellaneous/date_locales.rs}, from
 * {@code tools/GenDateLocales.java}). The locales are {@code corpus/date-locales.txt}: those whose
 * record ({@code GenDateLocales --facts}) JDK 21 and JDK 25 agree on, so this file is the same under
 * both. Per locale, its default date instance over a formatted date ({@code
 * Locale.Builder().setLanguageTag(tag)}, as the factory builds it); per distinct record (its first
 * locale), every pattern of {@link #PATTERNS} over formatted dates and mutations of them, and numbers
 * in the locale's own symbols ({@code dates.txt}: {@code tag<TAB>pattern<TAB>text<TAB>end}, {@code
 * end} the parse position's index or -1). Texts whose space separators JDK 23+'s lenient matching
 * would read otherwise than JDK 21 are left out. Read by {@code
 * crates/lucene-analysis/tests/date_locale_fixtures.rs}.
 */
public class GenAnalysisDateLocales {
  static final String[] PATTERNS = {
    "G", "MMMM", "LLL", "EEEE", "a", "y", "d MMM y", "EEE, d MMMM yyyy G", "h:mm a", "yyyyMMdd",
    "dd.MM.yy", "MMMd", "EEEEd", "ya", "LLLLy"
  };

  static int end(DateFormat f, String text) {
    ParsePosition p = new ParsePosition(0);
    f.parse(text, p);
    return p.getErrorIndex() >= 0 ? -1 : p.getIndex();
  }

  static String spaces(String s) {
    StringBuilder b = new StringBuilder();
    s.chars().filter(c -> Character.getType(c) == Character.SPACE_SEPARATOR).distinct().forEach(c -> b.append((char) c));
    return b.toString();
  }

  /**
   * Kept: the pattern has no space separator, or one kind only and the text no other kind (JDK
   * 23+ matches any space separator for a pattern's, JDK 21 only the same one); no split pair.
   */
  static boolean kept(String pattern, String text) {
    for (int i = 0; i < text.length(); i++) {
      char c = text.charAt(i);
      if (Character.isHighSurrogate(c) && (i + 1 == text.length() || !Character.isLowSurrogate(text.charAt(i + 1)))) return false;
      if (Character.isLowSurrogate(c) && (i == 0 || !Character.isHighSurrogate(text.charAt(i - 1)))) return false;
    }
    String p = spaces(pattern.replaceAll("'[^']*'", "")), t = spaces(text);
    if (p.isEmpty()) return true;
    return p.length() == 1 && t.chars().allMatch(c -> c == p.charAt(0));
  }

  static String mutate(Random r, String text, List<String> bits) {
    int edits = r.nextInt(3);
    for (int e = 0; e < edits; e++) {
      int at = text.isEmpty() ? 0 : r.nextInt(text.length() + 1);
      if (at < text.length() && Character.isLowSurrogate(text.charAt(at))) at--;
      switch (r.nextInt(5)) {
        case 0 -> text = text.substring(0, at) + bits.get(r.nextInt(bits.size())) + text.substring(at);
        case 1 -> text = at < text.length() ? text.substring(0, at) + text.substring(at + Character.charCount(text.codePointAt(at))) : text;
        case 2 -> text = text.substring(0, at);
        case 3 -> text = text.toUpperCase(Locale.ROOT);
        default -> text = text.toLowerCase(Locale.ROOT);
      }
    }
    return text;
  }

  public static void main(String[] args) throws Exception {
    Path out = Path.of(args[0]).resolve("analysis_date_locales");
    Files.createDirectories(out);
    StringBuilder o = new StringBuilder();
    Set<String> records = new LinkedHashSet<>();
    Random r = new Random(0x5EED_DA7EL);
    for (String tag : AnalysisRows.corpus("date-locales.txt")) {
      Locale l = new Locale.Builder().setLanguageTag(tag).build();
      DateFormat def = DateFormat.getDateInstance(DateFormat.DEFAULT, l);
      def.setTimeZone(TimeZone.getTimeZone("UTC"));
      String text = def.format(new Date(r.nextLong() % 4_000_000_000_000L));
      if (kept(((SimpleDateFormat) def).toPattern(), text)) {
        o.append(tag).append("\tDEFAULT\t").append(AnalysisRows.esc(text)).append('\t').append(end(def, text)).append('\n');
      }
      DecimalFormat df = (DecimalFormat) new SimpleDateFormat("y", l).getNumberFormat();
      DecimalFormatSymbols sy = df.getDecimalFormatSymbols();
      // One battery per distinct set of what the parse reads: the symbols, the default pattern
      // and every display name of the four text fields.
      StringBuilder key = new StringBuilder(sy.getNaN() + "|" + sy.getInfinity() + "|" + sy.getExponentSeparator() + "|"
          + df.getNegativePrefix() + "|" + ((SimpleDateFormat) def).toPattern());
      java.util.Calendar cal = java.util.Calendar.getInstance(l);
      java.text.DateFormatSymbols dfs = java.text.DateFormatSymbols.getInstance(l);
      for (int field : new int[] {java.util.Calendar.ERA, java.util.Calendar.MONTH, java.util.Calendar.DAY_OF_WEEK, java.util.Calendar.AM_PM}) {
        key.append('|').append(new java.util.TreeMap<>(cal.getDisplayNames(field, java.util.Calendar.ALL_STYLES, l)));
      }
      for (String[] a : new String[][] {dfs.getEras(), dfs.getMonths(), dfs.getShortMonths(), dfs.getWeekdays(), dfs.getShortWeekdays(), dfs.getAmPmStrings()}) {
        key.append('|').append(String.join(",", a));
      }
      if (!records.add(key.toString())) continue;
      List<String> bits = new ArrayList<>(List.of("1", "12", "x", ".", "-", " ", ",", "0"));
      bits.add(sy.getNaN());
      bits.add(sy.getInfinity());
      bits.add(sy.getExponentSeparator());
      bits.add(df.getNegativePrefix());
      bits.add(String.valueOf((char) (sy.getZeroDigit() + 7)));
      for (String pattern : PATTERNS) {
        SimpleDateFormat f = new SimpleDateFormat(pattern, l);
        f.setTimeZone(TimeZone.getTimeZone("UTC"));
        Set<String> seen = new LinkedHashSet<>();
        for (int k = 0; k < 12 && seen.size() < 6; k++) {
          String t = f.format(new Date(r.nextLong() % 4_000_000_000_000L));
          if (k > 1) t = mutate(r, t, bits);
          if (kept(pattern, t) && seen.add(t)) {
            o.append(tag).append('\t').append(AnalysisRows.esc(pattern)).append('\t').append(AnalysisRows.esc(t)).append('\t').append(end(f, t)).append('\n');
          }
        }
      }
      SimpleDateFormat y = new SimpleDateFormat("y", l);
      String n = df.getNegativePrefix(), x = sy.getExponentSeparator();
      for (String t : new String[] {sy.getNaN(), n + "12", n + sy.getInfinity(), sy.getInfinity(), "1" + x + "2", "1" + x + n + "2", "1" + x, "1" + x + n, n, n + sy.getNaN(), "-3", "−3", "1E2", "1e2"}) {
        if (kept("y", t)) o.append(tag).append("\ty\t").append(AnalysisRows.esc(t)).append('\t').append(end(y, t)).append('\n');
      }
    }
    Files.writeString(out.resolve("dates.txt"), o.toString(), StandardCharsets.UTF_8);
  }
}
