import org.apache.lucene.index.BinaryDocValues;
import org.apache.lucene.index.CheckIndex;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.DocValuesSkipper;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.NumericDocValues;
import org.apache.lucene.index.SegmentInfos;
import org.apache.lucene.index.SortedDocValues;
import org.apache.lucene.index.SortedSetDocValues;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;

import java.nio.file.Path;
import java.util.TreeSet;

/**
 * Verifies {@code crates/lucene-index/examples/write_per_field_doc_values_fixture.rs}: in {@code
 * <dir>/flushed} (two segments) every {@code s_} field records {@code PerFieldDocValuesFormat}
 * suffix 0 and every {@code d_} field suffix 1 (the flush reached {@code s_key} first); in {@code
 * <dir>/merged} (one segment) the reverse (the merge reached {@code d_num} first). Lucene's {@code
 * PerFieldDocValuesFormat.FieldsReader} reads every document's value of every field back from its
 * own instance's files, the routed fields' skip indexes close their first interval after 16
 * documents and the default ones after the whole segment (under 4096), and {@code CheckIndex} is
 * clean.
 */
public class VerifyPerFieldDocValues {
  static final int PER_SEGMENT = 300;
  static final String[] WORDS = {
    "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliet"
  };

  static long value(int i, int k) {
    long x = (i + 1) * 2654435761L + k * 40503L;
    return (x ^ (x >>> 13)) % 100000;
  }

  static String word(int i, int k) {
    return WORDS[(int) (value(i, k) % WORDS.length)];
  }

  public static void main(String[] args) throws Exception {
    verify(Path.of(args[0]).resolve("flushed"), 2, "0", "1");
    verify(Path.of(args[0]).resolve("merged"), 1, "1", "0");
    System.out.println("VerifyPerFieldDocValues: ok");
  }

  static void check(boolean ok, String what) {
    if (!ok) throw new AssertionError(what);
  }

  static void verify(Path path, int segments, String routed, String standard) throws Exception {
    try (Directory dir = FSDirectory.open(path)) {
      check(SegmentInfos.readLatestCommit(dir).size() == segments, path + ": segment count");
      try (DirectoryReader reader = DirectoryReader.open(dir)) {
        // The documents in index order: the merge takes the larger segment first.
        int[] order = new int[2 * PER_SEGMENT];
        int at = 0;
        for (LeafReaderContext ctx : reader.leaves()) {
          LeafReader leaf = ctx.reader();
          for (FieldInfo fi : leaf.getFieldInfos()) {
            String want = fi.name.startsWith("s_") ? routed : standard;
            check(
                "Lucene90".equals(fi.getAttribute("PerFieldDocValuesFormat.format"))
                    && want.equals(fi.getAttribute("PerFieldDocValuesFormat.suffix")),
                path + ": " + fi.name + " attributes " + fi.attributes());
          }
          NumericDocValues dNum = leaf.getNumericDocValues("d_num");
          BinaryDocValues idBin = leaf.getBinaryDocValues("s_bin");
          // d_num and s_bin together identify the document.
          for (int doc = 0; doc < leaf.maxDoc(); doc++) {
            check(dNum.advanceExact(doc) && idBin.advanceExact(doc), "d_num/s_bin missing");
            String bin = idBin.binaryValue().utf8ToString();
            int found = -1;
            for (int i = 0; i < 2 * PER_SEGMENT; i++) {
              if (value(i, 0) == dNum.longValue() && bin.equals("b" + value(i, 5))) {
                found = i;
                break;
              }
            }
            check(found >= 0, "unknown d_num " + dNum.longValue());
            order[at++] = found;
          }
          int base = at - leaf.maxDoc();
          NumericDocValues sNum = leaf.getNumericDocValues("s_num");
          SortedDocValues sKey = leaf.getSortedDocValues("s_key");
          SortedSetDocValues dSet = leaf.getSortedSetDocValues("d_set");
          BinaryDocValues sBin = leaf.getBinaryDocValues("s_bin");
          for (int doc = 0; doc < leaf.maxDoc(); doc++) {
            int i = order[base + doc];
            check(sNum.advanceExact(doc) == (i % 5 != 0), "s_num presence " + i);
            if (i % 5 != 0) check(sNum.longValue() == value(i, 1) / 16, "s_num " + i);
            check(sKey.advanceExact(doc), "s_key missing");
            check(sKey.lookupOrd(sKey.ordValue()).utf8ToString().equals(word(i, 2)), "s_key " + i);
            check(dSet.advanceExact(doc), "d_set missing");
            TreeSet<String> want = new TreeSet<>();
            want.add(word(i, 3));
            want.add(word(i, 4));
            TreeSet<String> got = new TreeSet<>();
            for (int k = 0; k < dSet.docValueCount(); k++) {
              got.add(dSet.lookupOrd(dSet.nextOrd()).utf8ToString());
            }
            check(want.equals(got), "d_set " + i + " " + got);
            check(sBin.advanceExact(doc), "s_bin missing");
            check(sBin.binaryValue().utf8ToString().equals("b" + value(i, 5)), "s_bin " + i);
          }
          for (String field : new String[] {"s_key", "s_num", "d_num"}) {
            DocValuesSkipper skipper = leaf.getDocValuesSkipper(field);
            check(skipper != null, field + " has no skip index");
            skipper.advance(0);
            int interval = skipper.maxDocID(0) - skipper.minDocID(0) + 1;
            if (field.startsWith("s_")) {
              check(skipper.docCount(0) == 16, field + ": first interval holds " + skipper.docCount(0));
            } else {
              check(interval == leaf.maxDoc(), field + ": first interval spans " + interval);
            }
          }
        }
        check(at == 2 * PER_SEGMENT, "document count");
      }
      try (CheckIndex checker = new CheckIndex(dir)) {
        CheckIndex.Status status = checker.checkIndex();
        check(status.clean, path + ": CheckIndex");
      }
    }
  }
}
