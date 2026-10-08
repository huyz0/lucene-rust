/*
 * Writes crates/lucene-analysis-icu/src/resources/coll.pack.z: ICU4J's collation and
 * transliteration data, as the ICU4J 77.1 jar carries it, for the collation port
 * (src/icu4j/coll/) and the transliterator port (src/icu4j/translit/).
 *
 * Format (big-endian, then zlib at level 9): "ICP1"; u32 entry count; per entry, sorted by name:
 * u16 name length, the name (ASCII), u32 data length, the data. The entries are every file under
 * com/ibm/icu/impl/data/icudata/coll/ of the jar (ucadata.icu, res_index.res and each locale's
 * .res), byte for byte, and translit/root.res (the transliterators' IDs and rules, entry
 * "translit/root.res"), plus one generated entry, "default_scripts.txt": the rows of
 * com.ibm.icu.impl.LocaleFallbackData.DEFAULT_SCRIPT_TABLE (what ICUResourceBundle's parent
 * locale fallback reads) whose language has a collation bundle -- "id=Script" per line, sorted.
 * Every other language falls back to the root collation whatever its script. And "lang3.txt" and
 * "region3.txt": com.ibm.icu.impl.LocaleIDs' three-to-two-letter language and region codes (what
 * new ULocale(id) canonicalizes "deu" and "DEU" to), "code=two" per changed code.
 *
 * Run:
 *
 *   java -cp fixtures/.jars/icu4j-77.1.jar crates/lucene-analysis-icu/tools/GenIcuCollPack.java \
 *     fixtures/.jars/icu4j-77.1.jar crates/lucene-analysis-icu/src/resources/coll.pack.z
 */

import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.io.InputStream;
import java.lang.reflect.Field;
import java.nio.charset.StandardCharsets;
import java.util.Map;
import java.util.TreeMap;
import java.util.TreeSet;
import java.util.jar.JarEntry;
import java.util.jar.JarFile;
import java.util.zip.Deflater;
import java.util.zip.DeflaterOutputStream;

public class GenIcuCollPack {
  private static final String DIR = "com/ibm/icu/impl/data/icudata/coll/";
  private static final String TRANSLIT_ROOT = "com/ibm/icu/impl/data/icudata/translit/root.res";

  public static void main(String[] args) throws Exception {
    TreeMap<String, byte[]> entries = new TreeMap<>();
    try (JarFile jar = new JarFile(args[0])) {
      for (JarEntry e : java.util.Collections.list(jar.entries())) {
        String name = e.getName();
        if (!name.startsWith(DIR) || e.isDirectory()) continue;
        String rest = name.substring(DIR.length());
        if (rest.contains("/")) continue;
        try (InputStream in = jar.getInputStream(e)) {
          entries.put(rest, in.readAllBytes());
        }
      }
      // The transliterators' rules and IDs (src/icu4j/translit/), read through the same reader.
      try (InputStream in = jar.getInputStream(jar.getJarEntry(TRANSLIT_ROOT))) {
        entries.put("translit/root.res", in.readAllBytes());
      }
    }
    TreeSet<String> languages = new TreeSet<>();
    for (String name : entries.keySet()) {
      if (!name.endsWith(".res")) continue;
      String id = name.substring(0, name.length() - 4);
      int us = id.indexOf('_');
      languages.add(us < 0 ? id : id.substring(0, us));
    }
    Class<?> data = Class.forName("com.ibm.icu.impl.LocaleFallbackData");
    Field f = data.getDeclaredField("DEFAULT_SCRIPT_TABLE");
    f.setAccessible(true);
    @SuppressWarnings("unchecked")
    Map<String, String> table = (Map<String, String>) f.get(null);
    StringBuilder sb = new StringBuilder();
    for (Map.Entry<String, String> e : new TreeMap<>(table).entrySet()) {
      String id = e.getKey();
      int us = id.indexOf('_');
      if (languages.contains(us < 0 ? id : id.substring(0, us))) {
        sb.append(id).append('=').append(e.getValue()).append('\n');
      }
    }
    entries.put("default_scripts.txt", sb.toString().getBytes(StandardCharsets.US_ASCII));
    // LocaleIDParser's three-to-two-letter language and region mappings (every three-letter code
    // that changes).
    StringBuilder lang3 = new StringBuilder();
    StringBuilder region3 = new StringBuilder();
    for (char a = 'a'; a <= 'z'; a++) {
      for (char b = 'a'; b <= 'z'; b++) {
        for (char c = 'a'; c <= 'z'; c++) {
          String code = "" + a + b + c;
          String two = com.ibm.icu.impl.LocaleIDs.threeToTwoLetterLanguage(code);
          if (two != null && !two.equals(code)) lang3.append(code).append('=').append(two).append('\n');
          String upper = code.toUpperCase(java.util.Locale.ROOT);
          two = com.ibm.icu.impl.LocaleIDs.threeToTwoLetterRegion(upper);
          if (two != null && !two.equals(upper)) region3.append(upper).append('=').append(two).append('\n');
        }
      }
    }
    entries.put("lang3.txt", lang3.toString().getBytes(StandardCharsets.US_ASCII));
    entries.put("region3.txt", region3.toString().getBytes(StandardCharsets.US_ASCII));

    ByteArrayOutputStream raw = new ByteArrayOutputStream();
    DataOutputStream out = new DataOutputStream(raw);
    out.writeBytes("ICP1");
    out.writeInt(entries.size());
    for (Map.Entry<String, byte[]> e : entries.entrySet()) {
      byte[] name = e.getKey().getBytes(StandardCharsets.US_ASCII);
      out.writeShort(name.length);
      out.write(name);
      out.writeInt(e.getValue().length);
      out.write(e.getValue());
    }
    out.flush();
    ByteArrayOutputStream z = new ByteArrayOutputStream();
    try (DeflaterOutputStream d = new DeflaterOutputStream(z, new Deflater(9))) {
      raw.writeTo(d);
    }
    java.nio.file.Files.write(java.nio.file.Path.of(args[1]), z.toByteArray());
    System.err.println(
        "coll pack: " + entries.size() + " entries, " + raw.size() + " bytes, " + z.size() + " compressed");
  }

  private GenIcuCollPack() {}

  static {
    // Fail early on a JDK without the jar on the class path.
    try {
      Class.forName("com.ibm.icu.util.ULocale");
    } catch (ClassNotFoundException e) {
      throw new IllegalStateException("run with icu4j-77.1.jar on the class path", e);
    }
  }
}
