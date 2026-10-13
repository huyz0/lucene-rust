/*
 * Writes crates/lucene-analysis-icu/src/resources/unames.bin.z: the Unicode character names ICU4J
 * 77.1 answers (UCD 16.0 as ICU implements it, Unicode-3.0 licensed), what a UnicodeSet's
 * \N{name} and [:name=...:] consult through UCharacter.getCharFromExtendedName. ICU4J is the
 * reader; nothing else is derived. Run:
 *
 *   java -cp icu4j-77.1.jar crates/lucene-analysis-icu/tools/GenIcuNames.java \
 *     crates/lucene-analysis-icu/src/resources/unames.bin.z
 *
 * Names ICU computes -- an ideograph's "PREFIX-HEX" (CJK, Tangut, Nushu, Khitan) and a Hangul
 * syllable's -- are not listed: the ideograph ranges are, and Hangul is the Unicode algorithm.
 *
 * Format (big-endian, then zlib): "LIN1"; u16 range count, each u8 prefix length + ASCII prefix
 * (ending in '-'), u32 first and last code point; u32 name count, then the names in ascending
 * order -- each name as u8 length of the prefix it shares with the previous name, u8 length of
 * the rest + ASCII, u32 code point (the primary names, then the name aliases ICU4J reports).
 */
import com.ibm.icu.lang.UCharacter;
import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.zip.Deflater;
import java.util.zip.DeflaterOutputStream;

public class GenIcuNames {
  record Range(String prefix, int first, int last) {}

  public static void main(String[] args) throws Exception {
    List<Range> ranges = new ArrayList<>();
    TreeMap<String, Integer> names = new TreeMap<>();
    for (int cp = 0; cp <= 0x10ffff; cp++) {
      String name = UCharacter.getName(cp);
      String alias = UCharacter.getNameAlias(cp);
      if (alias != null && !alias.isEmpty()) names.putIfAbsent(alias, cp);
      if (name == null || name.isEmpty()) continue;
      if (name.startsWith("HANGUL SYLLABLE ")) continue;
      int dash = name.lastIndexOf('-');
      if (dash > 0 && name.substring(dash + 1).equals(String.format("%04X", cp))) {
        String prefix = name.substring(0, dash + 1);
        Range last = ranges.isEmpty() ? null : ranges.get(ranges.size() - 1);
        if (last != null && last.prefix().equals(prefix) && last.last() == cp - 1) {
          ranges.set(ranges.size() - 1, new Range(prefix, last.first(), cp));
        } else {
          ranges.add(new Range(prefix, cp, cp));
        }
        continue;
      }
      names.put(name, cp);
    }
    // a primary name wins over an alias spelled the same
    for (int cp = 0; cp <= 0x10ffff; cp++) {
      String name = UCharacter.getName(cp);
      if (name != null && !name.isEmpty() && names.containsKey(name)) names.put(name, cp);
    }
    ByteArrayOutputStream raw = new ByteArrayOutputStream();
    DataOutputStream out = new DataOutputStream(raw);
    out.write("LIN1".getBytes(StandardCharsets.US_ASCII));
    out.writeShort(ranges.size());
    for (Range r : ranges) {
      byte[] p = r.prefix().getBytes(StandardCharsets.US_ASCII);
      out.writeByte(p.length);
      out.write(p);
      out.writeInt(r.first());
      out.writeInt(r.last());
    }
    out.writeInt(names.size());
    String prev = "";
    for (Map.Entry<String, Integer> e : names.entrySet()) {
      String n = e.getKey();
      int common = 0;
      while (common < prev.length() && common < n.length() && prev.charAt(common) == n.charAt(common)) common++;
      byte[] rest = n.substring(common).getBytes(StandardCharsets.US_ASCII);
      out.writeByte(common);
      out.writeByte(rest.length);
      out.write(rest);
      out.writeInt(e.getValue());
      prev = n;
    }
    out.flush();
    ByteArrayOutputStream z = new ByteArrayOutputStream();
    try (DeflaterOutputStream d = new DeflaterOutputStream(z, new Deflater(9))) {
      d.write(raw.toByteArray());
    }
    Files.write(Path.of(args[0]), z.toByteArray());
    System.out.println(ranges.size() + " ranges, " + names.size() + " names, " + z.size() + " bytes");
  }
}
