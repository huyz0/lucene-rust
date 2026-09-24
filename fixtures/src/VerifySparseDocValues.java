import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.TreeSet;
import org.apache.lucene.index.BinaryDocValues;
import org.apache.lucene.index.CheckIndex;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.DocValues;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.NumericDocValues;
import org.apache.lucene.index.SortedDocValues;
import org.apache.lucene.index.SortedNumericDocValues;
import org.apache.lucene.index.SortedSetDocValues;
import org.apache.lucene.index.StoredFields;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.BytesRef;

/**
 * Reverse-direction verifier for doc values that are <b>sparse in every type</b>: reads the two
 * indexes {@code write_sparse_doc_values_fixture} writes -- three flushed segments, and the same
 * documents merged into one -- and checks, for every document and every one of the five doc-values
 * fields, that real Lucene sees a value exactly where the generator put one, and the right value.
 * Then {@link CheckIndex} on both.
 *
 * <p>Presence is the point: a sparse column whose {@code IndexedDISI} is off by one reads back a
 * plausible value on the wrong document, which a check of values alone would pass. The table of
 * which document has which value is the Rust example's module doc; it is repeated in {@link
 * #expected}.
 *
 * <p>Usage: {@code java VerifySparseDocValues <out-dir>}.
 */
public class VerifySparseDocValues {
  /** Must match {@code write_sparse_doc_values_fixture.rs}. */
  private static final int NUM_DOCS = 3_000;

  private static int failures = 0;

  private static void fail(String message) {
    if (++failures <= 30) {
      System.out.println("MISMATCH " + message);
    }
  }

  /** The value document {@code i} should have for {@code field}, or null for none. */
  static String expected(String field, int i) {
    switch (field) {
      case "num":
        return i % 3 != 0 ? Long.toString(7L * i - 1000) : null;
      case "bin":
        return i % 5 != 0 ? "b" + i : null;
      case "sorted":
        return i % 7 != 0 ? "s" + (i % 50) : null;
      case "snum":
        {
          if (i % 11 == 0) return null;
          long[] v = {i % 13, i, -i};
          Arrays.sort(v);
          return Arrays.toString(v);
        }
      case "sset":
        {
          if (i % 13 == 0) return null;
          return new TreeSet<>(List.of("t" + (i % 17), "t" + (i % 19))).toString();
        }
      default:
        throw new IllegalArgumentException(field);
    }
  }

  static String actual(LeafReader leaf, String field, int doc) throws IOException {
    switch (field) {
      case "num":
        {
          NumericDocValues v = DocValues.getNumeric(leaf, field);
          return v.advanceExact(doc) ? Long.toString(v.longValue()) : null;
        }
      case "bin":
        {
          BinaryDocValues v = DocValues.getBinary(leaf, field);
          return v.advanceExact(doc) ? v.binaryValue().utf8ToString() : null;
        }
      case "sorted":
        {
          SortedDocValues v = DocValues.getSorted(leaf, field);
          return v.advanceExact(doc) ? v.lookupOrd(v.ordValue()).utf8ToString() : null;
        }
      case "snum":
        {
          SortedNumericDocValues v = DocValues.getSortedNumeric(leaf, field);
          if (!v.advanceExact(doc)) return null;
          long[] out = new long[v.docValueCount()];
          for (int k = 0; k < out.length; k++) out[k] = v.nextValue();
          return Arrays.toString(out);
        }
      case "sset":
        {
          SortedSetDocValues v = DocValues.getSortedSet(leaf, field);
          if (!v.advanceExact(doc)) return null;
          List<String> out = new ArrayList<>();
          for (int k = 0; k < v.docValueCount(); k++) {
            BytesRef term = v.lookupOrd(v.nextOrd());
            out.add(term.utf8ToString());
          }
          return out.toString();
        }
      default:
        throw new IllegalArgumentException(field);
    }
  }

  static void verify(Path path, boolean merged) throws IOException {
    try (Directory dir = FSDirectory.open(path);
        DirectoryReader reader = DirectoryReader.open(dir)) {
      if (reader.maxDoc() != NUM_DOCS) {
        fail(path + ": maxDoc=" + reader.maxDoc());
      }
      if (merged ? reader.leaves().size() != 1 : reader.leaves().size() < 3) {
        fail(path + ": " + reader.leaves().size() + " segment(s)");
      }
      int seen = 0;
      for (LeafReaderContext ctx : reader.leaves()) {
        LeafReader leaf = ctx.reader();
        StoredFields stored = leaf.storedFields();
        for (int doc = 0; doc < leaf.maxDoc(); doc++) {
          int i = Integer.parseInt(stored.document(doc).get("id").substring(3));
          seen++;
          // A fresh iterator per lookup keeps every read an independent
          // advanceExact from the column's start: slower, but a positioning
          // bug cannot hide behind a previous document's state.
          for (String field : new String[] {"num", "bin", "sorted", "snum", "sset"}) {
            String want = expected(field, i);
            String got = actual(leaf, field, doc);
            if (want == null ? got != null : !want.equals(got)) {
              fail(path.getFileName() + " doc" + i + " " + field + ": got " + got + ", want " + want);
            }
          }
        }
      }
      if (seen != NUM_DOCS) {
        fail(path + ": walked " + seen + " documents");
      }
    }
    try (Directory dir = FSDirectory.open(path);
        CheckIndex checker = new CheckIndex(dir)) {
      ByteArrayOutputStream captured = new ByteArrayOutputStream();
      checker.setInfoStream(new PrintStream(captured, true, StandardCharsets.UTF_8));
      checker.setLevel(CheckIndex.Level.MIN_LEVEL_FOR_SLOW_CHECKS);
      if (!checker.checkIndex().clean) {
        fail(path + ": CheckIndex reported the index unclean");
        System.out.println(captured.toString(StandardCharsets.UTF_8));
      }
    }
  }

  public static void main(String[] args) throws IOException {
    Path out = Path.of(args[0]);
    verify(out.resolve("flushed"), false);
    verify(out.resolve("merged"), true);
    if (failures > 0) {
      System.out.println(failures + " check(s) failed");
      System.exit(1);
    }
    System.out.println("Sparse doc values in all five types verified against real Lucene. PASS");
  }
}
