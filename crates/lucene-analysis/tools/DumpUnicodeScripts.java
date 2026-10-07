/** Prints the runs of `Character.UnicodeScript.of(c)` over the BMP, `hex-start NAME` per line,
 * for `java_unicode_script.rs` (`java tools/DumpUnicodeScripts.java`, on JDK 25). */
public class DumpUnicodeScripts {
  public static void main(String[] a) {
    Character.UnicodeScript prev = null; int start = 0;
    StringBuilder b = new StringBuilder();
    for (int c = 0; c <= 0x10000; c++) {
      Character.UnicodeScript s = c < 0x10000 ? Character.UnicodeScript.of(c) : null;
      if (s != prev) {
        if (prev != null) b.append(Integer.toHexString(start)).append(' ').append(prev.name()).append('\n');
        prev = s; start = c;
      }
    }
    System.out.print(b);
  }
}
