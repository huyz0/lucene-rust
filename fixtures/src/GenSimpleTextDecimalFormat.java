import java.io.IOException;
import java.math.BigDecimal;
import java.math.BigInteger;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.text.DecimalFormat;
import java.text.DecimalFormatSymbols;
import java.text.ParseException;
import java.util.ArrayList;
import java.util.HexFormat;
import java.util.List;
import java.util.Locale;

/**
 * What {@code SimpleTextDocValuesReader}'s {@code DecimalFormat} gives for each probe string:
 * {@code codecs_simpletext/decimal_format.tsv}, one row per probe, for {@code
 * decimal_format.rs}'s tests.
 *
 * <p>Columns: the probe (UTF-8 hex), then with {@code setParseBigDecimal(true)} (the NUMERIC
 * reader) {@code BigInteger.valueOf(origin).add(bd.toBigIntegerExact()).longValue()} for origin 0
 * and for {@code Long.MIN_VALUE}, then without it (the ord readers) {@code intValue()} and {@code
 * longValue()}. A cell is {@code PARSE} for a {@code ParseException}, {@code ERR} for any other
 * exception (a {@code Double} where a {@code BigDecimal} was cast, a fraction).
 */
public class GenSimpleTextDecimalFormat {
  static final String[] PROBES = {
    "000123", "-0120", "0", "-0", "1,234", "1,234.5", ".5", "12 ", "12.50", "0.05", "12.5.6",
    "1.2,3", "1E3", "15E-1", "1E", "1E+3", "1e3", "1E99999999999999999999", "7 trailing", "NaN",
    "∞", "-∞", "", "-", "x1", " 1", "+1", ".", ",", "0005", "1.00", "100E-2", "-7",
    "18446744073709551615", "18446744073709551616", "1E64", "1E70", "1.5", "0042", "-3",
    "4294967297", "1.9", "-1.9", "1E30", "9223372036854775807", "-9223372036854775808",
    "9223372036854775808", "-9223372036854775809", "92233720368547758070", "0.5",
    "9223372036854775810", "-0.0", "00000000000000000008",
  };

  interface Cell {
    String get(DecimalFormat f, String probe) throws Exception;
  }

  static String cell(boolean bigDecimal, String probe, Cell c) {
    DecimalFormat f = new DecimalFormat("000", new DecimalFormatSymbols(Locale.ROOT));
    f.setParseBigDecimal(bigDecimal);
    try {
      return c.get(f, probe);
    } catch (ParseException e) {
      return "PARSE";
    } catch (Exception e) {
      return "ERR";
    }
  }

  static String exact(DecimalFormat f, String probe, long origin) throws Exception {
    BigDecimal bd = (BigDecimal) f.parse(probe);
    return Long.toString(BigInteger.valueOf(origin).add(bd.toBigIntegerExact()).longValue());
  }

  public static void main(String[] args) throws IOException {
    Path root = Path.of(args[0]).resolve("codecs_simpletext");
    Files.createDirectories(root);
    List<String> rows = new ArrayList<>();
    for (String p : PROBES) {
      rows.add(
          String.join(
              "\t",
              HexFormat.of().formatHex(p.getBytes(StandardCharsets.UTF_8)),
              cell(true, p, (f, s) -> exact(f, s, 0)),
              cell(true, p, (f, s) -> exact(f, s, Long.MIN_VALUE)),
              cell(false, p, (f, s) -> Integer.toString(f.parse(s).intValue())),
              cell(false, p, (f, s) -> Long.toString(f.parse(s).longValue()))));
    }
    Files.write(root.resolve("decimal_format.tsv"), rows);
  }
}
