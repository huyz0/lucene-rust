import org.apache.lucene.document.LongPoint;
import org.apache.lucene.index.BinaryDocValues;
import org.apache.lucene.index.CheckIndex;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FilterLeafReader;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.NumericDocValues;
import org.apache.lucene.index.SegmentReader;
import org.apache.lucene.index.SortedDocValues;
import org.apache.lucene.index.SortedSetDocValues;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.Terms;
import org.apache.lucene.index.TermsEnum;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.BytesRef;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.nio.file.Path;
import java.util.Set;
import java.util.TreeSet;

/**
 * Verifies the Rust-written compound segments of
 * {@code crates/lucene-index/examples/write_compound_segment_fixture.rs}: every
 * segment is compound and holds only its {@code .cfs}/{@code .cfe}/{@code .si};
 * every live document's stored fields, doc values of all five types, point,
 * vector, term vector and postings read back through Lucene's own
 * {@code Lucene90CompoundFormat} reader as the table in the example says; and
 * {@code CheckIndex} is clean.
 */
public class VerifyCompoundSegment {
  static final int NUM_DOCS = 1000;

  public static void main(String[] args) throws Exception {
    Path root = Path.of(args[0]);
    verify(root.resolve("flushed"), 3);
    verify(root.resolve("merged"), 1);
    System.out.println("ok");
  }

  static void check(boolean cond, String what) {
    if (!cond) {
      throw new AssertionError(what);
    }
  }

  static void verify(Path path, int segments) throws Exception {
    try (Directory dir = FSDirectory.open(path);
        DirectoryReader reader = DirectoryReader.open(dir)) {
      check(reader.leaves().size() == segments, path + ": " + reader.leaves().size() + " leaves");
      int live = 0;
      for (LeafReaderContext ctx : reader.leaves()) {
        SegmentReader sr = (SegmentReader) FilterLeafReader.unwrap(ctx.reader());
        String name = sr.getSegmentName();
        check(sr.getSegmentInfo().info.getUseCompoundFile(), name + " is not compound");
        Set<String> files = new TreeSet<>(sr.getSegmentInfo().info.files());
        check(
            files.equals(Set.of(name + ".cfs", name + ".cfe", name + ".si")),
            name + " lists " + files);
        live += verifyLeaf(ctx.reader());
      }
      check(live == NUM_DOCS - NUM_DOCS / 10, "live docs " + live);

      IndexSearcher searcher = new IndexSearcher(reader);
      for (int w = 0; w < 13; w++) {
        int expected = 0;
        for (int i = 0; i < NUM_DOCS; i++) {
          if (i % 10 != 0 && (i % 13 == w || i % 7 == w)) {
            expected++;
          }
        }
        int got = searcher.count(new TermQuery(new Term("body", "w" + w)));
        check(got == expected, "body:w" + w + " count " + got + " != " + expected);
      }
      int inRange = searcher.count(LongPoint.newRangeQuery("pt", 100, 499));
      check(inRange == 400 - 40, "pt range count " + inRange);

      try (CheckIndex checker = new CheckIndex(dir)) {
        checker.setLevel(CheckIndex.Level.MIN_LEVEL_FOR_SLOW_CHECKS);
        ByteArrayOutputStream log = new ByteArrayOutputStream();
        checker.setInfoStream(new PrintStream(log, true, "UTF-8"));
        CheckIndex.Status status = checker.checkIndex();
        if (!status.clean) {
          System.out.println(log.toString("UTF-8"));
          throw new AssertionError(path + ": CheckIndex is not clean");
        }
      }
    }
  }

  /** Checks every live document of the leaf against the table; returns how many. */
  static int verifyLeaf(LeafReader leaf) throws Exception {
    Bits liveDocs = leaf.getLiveDocs();
    NumericDocValues num = leaf.getNumericDocValues("num");
    SortedDocValues cat = leaf.getSortedDocValues("cat");
    SortedSetDocValues tags = leaf.getSortedSetDocValues("tags");
    BinaryDocValues blob = leaf.getBinaryDocValues("blob");
    FloatVectorValues vectors = leaf.getFloatVectorValues("v");
    KnnVectorValues.DocIndexIterator vit = vectors.iterator();
    int live = 0;
    for (int d = 0; d < leaf.maxDoc(); d++) {
      String id = leaf.storedFields().document(d).get("id");
      int i = Integer.parseInt(id.substring(3));
      boolean isLive = liveDocs == null || liveDocs.get(d);
      check(isLive == (i % 10 != 0), id + " liveness");
      if (isLive) {
        live++;
      }
      check(num.advanceExact(d) == (i % 4 != 0), id + " num presence");
      if (i % 4 != 0) {
        check(num.longValue() == 3L * i, id + " num=" + num.longValue());
      }
      check(cat.advanceExact(d), id + " cat");
      check(cat.lookupOrd(cat.ordValue()).utf8ToString().equals("c" + (i % 7)), id + " cat value");
      check(tags.advanceExact(d), id + " tags");
      Set<String> got = new TreeSet<>();
      for (int k = 0; k < tags.docValueCount(); k++) {
        got.add(tags.lookupOrd(tags.nextOrd()).utf8ToString());
      }
      check(got.equals(new TreeSet<>(java.util.List.of("t" + (i % 5), "t" + (i % 3)))), id + " tags " + got);
      check(blob.advanceExact(d), id + " blob");
      check(blob.binaryValue().equals(new BytesRef("b" + i)), id + " blob value");
      check(vit.advance(d) == d, id + " vector");
      float[] v = vectors.vectorValue(vit.index());
      check(v[0] == i && v[1] == i % 10 && v[2] == 1f && v[3] == 0f, id + " vector value");
      Terms tv = leaf.termVectors().get(d, "body");
      check(tv != null, id + " term vector");
      Set<String> terms = new TreeSet<>();
      TermsEnum te = tv.iterator();
      for (BytesRef t = te.next(); t != null; t = te.next()) {
        terms.add(t.utf8ToString());
      }
      Set<String> want = new TreeSet<>(java.util.List.of("w" + (i % 13), "w" + (i % 7), "common"));
      check(terms.equals(want), id + " term vector " + terms);
    }
    check(leaf.getNormValues("body") != null, "body norms");
    check(leaf.getPointValues("pt") != null, "pt points");
    return live;
  }
}
