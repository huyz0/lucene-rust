import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.nio.file.Path;
import org.apache.lucene.index.BinaryDocValues;
import org.apache.lucene.index.CheckIndex;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.NumericDocValues;
import org.apache.lucene.index.SortedDocValues;
import org.apache.lucene.index.SortedNumericDocValues;
import org.apache.lucene.index.StoredFields;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.Bits;

/**
 * Opens an index {@code write_index_format} wrote with whatever Lucene is on the class path, finds
 * it clean, and checks every live document's sparse doc values against the formulas the writer
 * used. Usage: {@code VerifyIndexFormat <index dir> <expected segments_N format: 10|11|any>}.
 */
public class VerifyIndexFormat {
  static void check(boolean ok, String msg) {
    if (!ok) throw new AssertionError(msg);
  }

  public static void main(String[] args) throws Exception {
    Path path = Path.of(args[0]);
    try (Directory dir = FSDirectory.open(path)) {
      ByteArrayOutputStream bytes = new ByteArrayOutputStream();
      try (CheckIndex ci = new CheckIndex(dir)) {
        ci.setInfoStream(new PrintStream(bytes, true, "UTF-8"));
        CheckIndex.Status status = ci.checkIndex();
        check(status.clean, "CheckIndex found problems:\n" + bytes);
      }
      long live = 0;
      int dvp = 0;
      for (String f : dir.listAll()) if (f.endsWith(".dvp")) dvp++;
      try (DirectoryReader reader = DirectoryReader.open(dir)) {
        for (LeafReaderContext ctx : reader.leaves()) {
          LeafReader leaf = ctx.reader();
          Bits liveDocs = leaf.getLiveDocs();
          StoredFields stored = leaf.storedFields();
          for (int d = 0; d < leaf.maxDoc(); d++) {
            if (liveDocs != null && !liveDocs.get(d)) continue;
            live++;
            String id = stored.document(d).get("id");
            long n = Long.parseLong(id.substring(3));
            NumericDocValues num = leaf.getNumericDocValues("num");
            boolean hasNum = num != null && num.advanceExact(d);
            check(hasNum == (n % 3 != 0), "num presence of doc " + n);
            if (hasNum) check(num.longValue() == n * 7 - 100, "num value of doc " + n);
            BinaryDocValues bin = leaf.getBinaryDocValues("bin");
            boolean hasBin = bin != null && bin.advanceExact(d);
            check(hasBin == (n % 5 != 0), "bin presence of doc " + n);
            if (hasBin) check(bin.binaryValue().utf8ToString().equals("b" + n), "bin value " + n);
            SortedNumericDocValues snum = leaf.getSortedNumericDocValues("snum");
            boolean hasSnum = snum != null && snum.advanceExact(d);
            check(hasSnum == (n % 4 != 1), "snum presence of doc " + n);
            if (hasSnum) {
              check(snum.docValueCount() == 2, "snum count of doc " + n);
              check(snum.nextValue() == n && snum.nextValue() == n + 1, "snum values of doc " + n);
            }
            SortedDocValues kw = leaf.getSortedDocValues("kw");
            boolean hasKw = kw != null && kw.advanceExact(d);
            check(hasKw == (n % 7 != 2), "kw presence of doc " + n);
            if (hasKw)
              check(kw.lookupOrd(kw.ordValue()).utf8ToString().equals("k" + (n % 50)), "kw " + n);
          }
        }
      }
      check(live > 0, "no live documents");
      System.out.println(
          "VerifyIndexFormat ok: " + live + " live documents, " + dvp + " .dvp files, Lucene "
              + org.apache.lucene.util.Version.LATEST);
      String want = args.length > 1 ? args[1] : "any";
      if (want.equals("with-dvp")) check(dvp > 0, "expected .dvp files");
      if (want.equals("no-dvp")) check(dvp == 0, "expected no .dvp files");
    }
  }
}
