/*
 * Writes crates/lucene-analysis-icu/src/resources/uprops.bin.z: the Unicode character properties
 * ICU4J 77.1 answers (UCD 16.0 as ICU implements it, Unicode-3.0) for every binary and enumerated
 * property, General_Category masks, Script_Extensions, and every property and value alias -- what
 * UnicodeSet patterns ([:Latin:], \p{gc=Lu}), UScript and UCharacter.getType consult. ICU4J is
 * the reader (its code and data are Unicode-licensed); nothing else is derived. Run:
 *
 *   java -cp icu4j-77.1.jar crates/lucene-analysis-icu/tools/GenIcuProperties.java \
 *     crates/lucene-analysis-icu/src/resources/uprops.bin.z
 *
 * Format (big-endian, then zlib): "LIP1"; u16 property count; per property: i32 UProperty id, u8
 * kind (0 binary, 1 enumerated, 2 General_Category_Mask, 3 Script_Extensions, 4 any other property:
 * names only, 5 Age: runs of the version packed major.minor.milli.micro into an int, 6 Numeric_Value: each
 * distinct double a value named by its bits in hex, runs of those values), names (u8 count,
 * each u8 length + ASCII), u16 value count with each value's i32 and names, u32 run count and the
 * runs (u32 first code point, i32 value; a run lasts until the next one starts), then for a binary
 * property the strings its set holds (u32 count, each u16 length + UTF-16 units: the emoji
 * properties of strings). Then u16 script list count, each u8 length + u16 scripts
 * (Script_Extensions' run values index these lists), and every script code's short and long name.
 */
import com.ibm.icu.lang.UCharacter;
import com.ibm.icu.lang.UProperty;
import com.ibm.icu.lang.UScript;
import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.zip.Deflater;
import java.util.zip.DeflaterOutputStream;

public class GenIcuProperties {
  static List<String> propNames(int p) {
    Set<String> names = new LinkedHashSet<>();
    for (int choice = 0; choice < 10; choice++) {
      try {
        String n = UCharacter.getPropertyName(p, choice);
        if (n != null) names.add(n);
      } catch (IllegalArgumentException e) {
        break;
      }
    }
    return new ArrayList<>(names);
  }

  static List<String> valueNames(int p, int v) {
    Set<String> names = new LinkedHashSet<>();
    for (int choice = 0; choice < 10; choice++) {
      try {
        String n = UCharacter.getPropertyValueName(p, v, choice);
        if (n != null) names.add(n);
      } catch (IllegalArgumentException e) {
        break;
      }
    }
    return new ArrayList<>(names);
  }

  static void names(DataOutputStream o, List<String> names) throws Exception {
    o.writeByte(names.size());
    for (String n : names) {
      byte[] b = n.getBytes("US-ASCII");
      o.writeByte(b.length);
      o.write(b);
    }
  }

  static void runs(DataOutputStream o, int[] values) throws Exception {
    List<int[]> runs = new ArrayList<>();
    for (int c = 0; c <= 0x10ffff; c++) {
      if (c == 0 || values[c] != values[c - 1]) runs.add(new int[] {c, values[c]});
    }
    o.writeInt(runs.size());
    for (int[] r : runs) {
      o.writeInt(r[0]);
      o.writeInt(r[1]);
    }
  }

  public static void main(String[] args) throws Exception {
    ByteArrayOutputStream bytes = new ByteArrayOutputStream();
    DataOutputStream o = new DataOutputStream(bytes);
    o.writeBytes("LIP1");
    List<Integer> props = new ArrayList<>();
    for (int p = UProperty.BINARY_START; p < UProperty.BINARY_LIMIT; p++) props.add(p);
    for (int p = UProperty.INT_START; p < UProperty.INT_LIMIT; p++) props.add(p);
    props.add(UProperty.GENERAL_CATEGORY_MASK);
    props.add(UProperty.SCRIPT_EXTENSIONS);
    // Names only: the double, string and other properties (UnicodeSet refuses most of them).
    List<Integer> others = new ArrayList<>();
    for (int p = UProperty.DOUBLE_START; p < UProperty.DOUBLE_LIMIT; p++) others.add(p);
    for (int p = UProperty.STRING_START; p < UProperty.STRING_LIMIT; p++) others.add(p);
    for (int p = UProperty.OTHER_PROPERTY_START; p < UProperty.OTHER_PROPERTY_LIMIT; p++) {
      if (p != UProperty.SCRIPT_EXTENSIONS) others.add(p);
    }
    props.addAll(others);
    o.writeShort(props.size());
    Map<String, Integer> lists = new LinkedHashMap<>();
    List<int[]> listValues = new ArrayList<>();
    for (int p : props) {
      o.writeInt(p);
      int kind =
          p < UProperty.BINARY_LIMIT ? 0
              : p < UProperty.INT_LIMIT ? 1
              : p == UProperty.GENERAL_CATEGORY_MASK ? 2
              : p == UProperty.SCRIPT_EXTENSIONS ? 3
              : p == UProperty.AGE ? 5
              : p == UProperty.NUMERIC_VALUE ? 6 : 4;
      o.writeByte(kind);
      names(o, propNames(p));
      if (kind == 0 || kind == 1) {
        int max = UCharacter.getIntPropertyMaxValue(p);
        int min = UCharacter.getIntPropertyMinValue(p);
        List<int[]> vals = new ArrayList<>();
        List<List<String>> vnames = new ArrayList<>();
        for (int v = min; v <= max; v++) {
          List<String> n = valueNames(p, v);
          if (!n.isEmpty()) {
            vals.add(new int[] {v});
            vnames.add(n);
          }
        }
        o.writeShort(vals.size());
        for (int i = 0; i < vals.size(); i++) {
          o.writeInt(vals.get(i)[0]);
          names(o, vnames.get(i));
        }
        int[] values = new int[0x110000];
        for (int c = 0; c <= 0x10ffff; c++) values[c] = UCharacter.getIntPropertyValue(c, p);
        runs(o, values);
        if (kind == 0) {
          List<String> strings = new ArrayList<>();
          for (String str : com.ibm.icu.lang.CharacterProperties.getBinaryPropertySet(p).strings()) {
            strings.add(str);
          }
          o.writeInt(strings.size());
          for (String str : strings) {
            o.writeShort(str.length());
            for (int i = 0; i < str.length(); i++) o.writeChar(str.charAt(i));
          }
        }
      } else if (kind == 2) {
        // Every single category and every group mask ICU names.
        Set<Integer> masks = new LinkedHashSet<>();
        for (int gc = 0; gc < 32; gc++) {
          if (!valueNames(UProperty.GENERAL_CATEGORY_MASK, 1 << gc).isEmpty()) masks.add(1 << gc);
        }
        for (String g : new String[] {"L", "LC", "M", "N", "P", "S", "Z", "C"}) {
          masks.add(UCharacter.getPropertyValueEnum(UProperty.GENERAL_CATEGORY_MASK, g));
        }
        o.writeShort(masks.size());
        for (int m : masks) {
          o.writeInt(m);
          names(o, valueNames(UProperty.GENERAL_CATEGORY_MASK, m));
        }
        o.writeInt(0);
      } else if (kind == 4) {
        o.writeShort(0);
        o.writeInt(0);
      } else if (kind == 6) {
        // Numeric_Value: each distinct double is a value named by its bits in hex.
        Map<Long, Integer> index = new LinkedHashMap<>();
        int[] values = new int[0x110000];
        for (int c = 0; c <= 0x10ffff; c++) {
          long bits = Double.doubleToLongBits(UCharacter.getUnicodeNumericValue(c));
          Integer i = index.get(bits);
          if (i == null) {
            i = index.size();
            index.put(bits, i);
          }
          values[c] = i;
        }
        o.writeShort(index.size());
        for (Map.Entry<Long, Integer> e : index.entrySet()) {
          o.writeInt(e.getValue());
          names(o, List.of(Long.toHexString(e.getKey())));
        }
        runs(o, values);
      } else if (kind == 5) {
        o.writeShort(0);
        int[] values = new int[0x110000];
        for (int c = 0; c <= 0x10ffff; c++) {
          com.ibm.icu.util.VersionInfo v = UCharacter.getAge(c);
          values[c] = (v.getMajor() << 24) | (v.getMinor() << 16) | (v.getMilli() << 8) | v.getMicro();
        }
        runs(o, values);
      } else {
        o.writeShort(0);
        int[] values = new int[0x110000];
        for (int c = 0; c <= 0x10ffff; c++) {
          java.util.BitSet set = new java.util.BitSet();
          UScript.getScriptExtensions(c, set);
          int[] l = set.stream().toArray();
          String key = Arrays.toString(l);
          Integer idx = lists.get(key);
          if (idx == null) {
            idx = listValues.size();
            lists.put(key, idx);
            listValues.add(l);
          }
          values[c] = idx;
        }
        runs(o, values);
      }
    }
    o.writeShort(listValues.size());
    for (int[] l : listValues) {
      o.writeByte(l.length);
      for (int s : l) o.writeShort(s);
    }
    // Script names for every code UScript knows (UScript.getName / getShortName), including the
    // ones no character has (Japanese, Chinese/Japanese for ICUTokenizer's combined runs).
    o.writeShort(UScript.CODE_LIMIT);
    for (int s = 0; s < UScript.CODE_LIMIT; s++) {
      names(o, List.of(UScript.getShortName(s), UScript.getName(s)));
    }
    o.flush();
    ByteArrayOutputStream z = new ByteArrayOutputStream();
    try (DeflaterOutputStream d = new DeflaterOutputStream(z, new Deflater(9))) {
      d.write(bytes.toByteArray());
    }
    Files.write(Path.of(args[0]), z.toByteArray());
    System.err.println("uprops: " + bytes.size() + " bytes, " + z.size() + " compressed");
  }
}
