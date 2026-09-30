import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.BinaryDocValuesField;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.NumericDocValuesField;
import org.apache.lucene.document.SortedSetDocValuesField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.BinarySortField;
import org.apache.lucene.search.FieldDoc;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.Sort;
import org.apache.lucene.search.SortField;
import org.apache.lucene.search.SortedSetSelector;
import org.apache.lucene.search.SortedSetSortField;
import org.apache.lucene.search.TopFieldCollectorManager;
import org.apache.lucene.search.TopFieldDocs;
import org.apache.lucene.search.TotalHits;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.BytesRef;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Comparator;
import java.util.Random;
import java.util.stream.Stream;

/**
 * Sorted searches by {@code SortField.Type.STRING_VAL} and {@code BinarySortField} (a {@code
 * BINARY} column through {@code TermValComparator}) and by {@code SortedSetSortField} with the
 * {@code MIDDLE_MIN}/{@code MIDDLE_MAX} selectors, recorded from Lucene for
 * crates/lucene-search/tests/sort_val_fixtures.rs.
 *
 * <p>Two segments (the first with deletions); per query, sort, {@code topN} and total-hits
 * threshold, the hits as {@code doc:v0:v1...} (bytes as {@code x}hex, {@code -} for missing, a
 * long as is, a score as its float bits), the total and its relation; then a {@code searchAfter}
 * page from the tenth hit.
 */
public class GenSortVal {
  static final int[] SEGMENTS = {400, 300};
  static final String[] QUERIES = {"(all)", "(t w0)", "(b 0 (? (t w1)) (? (t w2)))"};
  static final String[][] SORTS = {
    {"bv:val:false:first"},
    {"bv:val:true:last"},
    {"bv:binary:false:first"},
    {"bv:binary:true:last"},
    {"ss:middle_min:false:first"},
    {"ss:middle_max:true:first"},
    {"ss:middle_min:false:last", "n:long:false:first"},
    {"bv:val:false:last", "score:score:false:first"},
    {"ss:middle_max:false:first", "bv:val:true:first"},
    {"ss:min:false:first", "bv:binary:false:last"},
    {"n:custom_mod3:false:first"},
    {"n:custom_mod3:true:first", "bv:custom_len:false:first"},
    {"bv:custom_len:true:first", "score:score:false:first"},
  };
  static final String[] TERMS = {"a", "ab", "abc", "b", "ba", "été", "z", ""};

  public static void main(String[] args) throws IOException {
    Path out = Path.of(args[0]).resolve("sort_val_index");
    if (Files.exists(out)) {
      try (Stream<Path> s = Files.walk(out)) {
        s.sorted(Comparator.reverseOrder()).forEach(p -> p.toFile().delete());
      }
    }
    Files.createDirectories(out);
    StringBuilder m = new StringBuilder();
    Random r = new Random(20260930L);
    try (Directory dir = FSDirectory.open(out)) {
      IndexWriterConfig cfg = new IndexWriterConfig(new StandardAnalyzer());
      cfg.setUseCompoundFile(false);
      cfg.setMergePolicy(NoMergePolicy.INSTANCE);
      cfg.setMaxBufferedDocs(IndexWriterConfig.DISABLE_AUTO_FLUSH);
      cfg.setRAMBufferSizeMB(256);
      int id = 0;
      try (IndexWriter w = new IndexWriter(dir, cfg)) {
        for (int size : SEGMENTS) {
          for (int i = 0; i < size; i++, id++) {
            Document d = new Document();
            d.add(new StringField("id", Integer.toString(id), Field.Store.NO));
            d.add(new TextField("body", GenMixedBooleanScoring.body(r), Field.Store.NO));
            if (id % 6 != 5) {
              byte[] b = new byte[r.nextInt(4)];
              for (int k = 0; k < b.length; k++) b[k] = (byte) (r.nextInt(5) * 60 - 1);
              d.add(new BinaryDocValuesField("bv", new BytesRef(b)));
            }
            int n = r.nextInt(6);
            for (int k = 0; k < n; k++) {
              d.add(new SortedSetDocValuesField("ss", new BytesRef(TERMS[r.nextInt(TERMS.length)])));
            }
            d.add(new NumericDocValuesField("n", r.nextInt(7)));
            w.addDocument(d);
          }
          w.commit();
        }
        for (int d = 0; d < SEGMENTS[0]; d += 7) {
          w.deleteDocuments(new Term("id", Integer.toString(d)));
        }
        w.commit();
      }
      try (DirectoryReader reader = DirectoryReader.open(dir)) {
        IndexSearcher searcher = new IndexSearcher(reader);
        searcher.setQueryCache(null);
        int run = 0;
        for (String qs : QUERIES) {
          Query q = GenSortedSearch.parse(new GenMixedBooleanScoring.Tokens(qs));
          for (String[] spec : SORTS) {
            Sort sort = sort(spec);
            for (int topN : new int[] {10, 25}) {
              for (int threshold : new int[] {Integer.MAX_VALUE, 50}) {
                TopFieldDocs td =
                    searcher.search(q, new TopFieldCollectorManager(sort, topN, null, threshold));
                String k = "run." + run++;
                m.append(k).append(".query=").append(qs).append('\n');
                m.append(k).append(".sort=").append(String.join(",", spec)).append('\n');
                m.append(k).append(".top_n=").append(topN).append('\n');
                m.append(k).append(".threshold=").append(threshold == Integer.MAX_VALUE ? "max" : threshold).append('\n');
                m.append(k).append(".hits=").append(hits(td, spec)).append('\n');
                m.append(k).append(".total=").append(td.totalHits.value()).append('\n');
                m.append(k).append(".relation=").append(td.totalHits.relation() == TotalHits.Relation.EQUAL_TO ? "eq" : "gte").append('\n');
                if (threshold == Integer.MAX_VALUE && topN == 10 && td.scoreDocs.length == 10) {
                  FieldDoc after = (FieldDoc) td.scoreDocs[9];
                  TopFieldDocs page =
                      searcher.search(q, new TopFieldCollectorManager(sort, 10, after, Integer.MAX_VALUE));
                  m.append(k).append(".after=").append(hit(after, spec)).append('\n');
                  m.append(k).append(".page=").append(hits(page, spec)).append('\n');
                }
              }
            }
          }
        }
        m.append("run_count=").append(run).append('\n');
      }
    }
    Files.writeString(out.resolve("manifest.properties"), m.toString(), StandardCharsets.UTF_8);
  }

  static Sort sort(String[] spec) {
    SortField[] fields = new SortField[spec.length];
    for (int i = 0; i < spec.length; i++) {
      String[] p = spec[i].split(":");
      boolean reverse = Boolean.parseBoolean(p[2]);
      Object missing = p[3].equals("last") ? SortField.STRING_LAST : SortField.STRING_FIRST;
      fields[i] =
          switch (p[1]) {
            case "val" -> {
              SortField f = new SortField(p[0], SortField.Type.STRING_VAL, reverse);
              f.setMissingValue(missing);
              yield f;
            }
            case "binary" -> new BinarySortField(p[0], reverse, missing);
            case "min", "middle_min", "middle_max" -> {
              SortedSetSelector.Type t =
                  switch (p[1]) {
                    case "min" -> SortedSetSelector.Type.MIN;
                    case "middle_min" -> SortedSetSelector.Type.MIDDLE_MIN;
                    default -> SortedSetSelector.Type.MIDDLE_MAX;
                  };
              SortField f = new SortedSetSortField(p[0], reverse, t);
              f.setMissingValue(missing);
              yield f;
            }
            case "long" -> new SortField(p[0], SortField.Type.LONG, reverse);
            case "custom_mod3" -> new SortField(p[0], new Mod3Source(), reverse);
            case "custom_len" -> new SortField(p[0], new LengthSource(), reverse);
            case "score" -> new SortField(null, SortField.Type.SCORE, reverse);
            default -> throw new AssertionError(p[1]);
          };
    }
    return new Sort(fields);
  }

  /** A custom order over a NUMERIC column: by the value mod 3, then the value. */
  static final class Mod3Source extends org.apache.lucene.search.FieldComparatorSource {
    @Override
    public org.apache.lucene.search.FieldComparator<?> newComparator(
        String field, int numHits, org.apache.lucene.search.Pruning pruning, boolean reversed) {
      return new Mod3(field, numHits);
    }
  }

  static int mod3(long a, long b) {
    int c = Long.compare(Math.floorMod(a, 3), Math.floorMod(b, 3));
    return c != 0 ? c : Long.compare(a, b);
  }

  static final class Mod3 extends org.apache.lucene.search.FieldComparator<Long>
      implements org.apache.lucene.search.LeafFieldComparator {
    final String field;
    final long[] values;
    long bottom, top;
    org.apache.lucene.index.NumericDocValues dv;

    Mod3(String field, int numHits) {
      this.field = field;
      this.values = new long[numHits];
    }

    long get(int doc) throws IOException {
      return dv.advanceExact(doc) ? dv.longValue() : 0;
    }

    @Override public int compare(int a, int b) { return mod3(values[a], values[b]); }
    @Override public void setTopValue(Long v) { top = v; }
    @Override public Long value(int slot) { return values[slot]; }
    @Override public int compareValues(Long a, Long b) { return mod3(a, b); }
    @Override public org.apache.lucene.search.LeafFieldComparator getLeafComparator(
        org.apache.lucene.index.LeafReaderContext ctx) throws IOException {
      dv = org.apache.lucene.index.DocValues.getNumeric(ctx.reader(), field);
      return this;
    }
    @Override public void setBottom(int slot) { bottom = values[slot]; }
    @Override public int compareBottom(int doc) throws IOException { return mod3(bottom, get(doc)); }
    @Override public int compareTop(int doc) throws IOException { return mod3(top, get(doc)); }
    @Override public void copy(int slot, int doc) throws IOException { values[slot] = get(doc); }
    @Override public void setScorer(org.apache.lucene.search.Scorable scorer) {}
  }

  /** A custom order over a BINARY column: by length, then bytes; missing first. */
  static final class LengthSource extends org.apache.lucene.search.FieldComparatorSource {
    @Override
    public org.apache.lucene.search.FieldComparator<?> newComparator(
        String field, int numHits, org.apache.lucene.search.Pruning pruning, boolean reversed) {
      return new Length(field, numHits);
    }
  }

  static int byLength(BytesRef a, BytesRef b) {
    if (a == null || b == null) {
      return a == b ? 0 : (a == null ? -1 : 1);
    }
    int c = Integer.compare(a.length, b.length);
    return c != 0 ? c : a.compareTo(b);
  }

  static final class Length extends org.apache.lucene.search.FieldComparator<BytesRef>
      implements org.apache.lucene.search.LeafFieldComparator {
    final String field;
    final BytesRef[] values;
    BytesRef bottom, top;
    org.apache.lucene.index.BinaryDocValues dv;

    Length(String field, int numHits) {
      this.field = field;
      this.values = new BytesRef[numHits];
    }

    BytesRef get(int doc) throws IOException {
      return dv.advanceExact(doc) ? BytesRef.deepCopyOf(dv.binaryValue()) : null;
    }

    @Override public int compare(int a, int b) { return byLength(values[a], values[b]); }
    @Override public void setTopValue(BytesRef v) { top = v; }
    @Override public BytesRef value(int slot) { return values[slot]; }
    @Override public int compareValues(BytesRef a, BytesRef b) { return byLength(a, b); }
    @Override public org.apache.lucene.search.LeafFieldComparator getLeafComparator(
        org.apache.lucene.index.LeafReaderContext ctx) throws IOException {
      dv = org.apache.lucene.index.DocValues.getBinary(ctx.reader(), field);
      return this;
    }
    @Override public void setBottom(int slot) { bottom = values[slot]; }
    @Override public int compareBottom(int doc) throws IOException { return byLength(bottom, get(doc)); }
    @Override public int compareTop(int doc) throws IOException { return byLength(top, get(doc)); }
    @Override public void copy(int slot, int doc) throws IOException { values[slot] = get(doc); }
    @Override public void setScorer(org.apache.lucene.search.Scorable scorer) {}
  }

  static String hits(TopFieldDocs td, String[] spec) {
    StringBuilder b = new StringBuilder();
    for (ScoreDoc sd : td.scoreDocs) {
      if (b.length() > 0) b.append(',');
      b.append(hit((FieldDoc) sd, spec));
    }
    return b.toString();
  }

  static String hit(FieldDoc fd, String[] spec) {
    StringBuilder b = new StringBuilder();
    b.append(fd.doc);
    for (int k = 0; k < spec.length; k++) {
      Object v = fd.fields[k];
      b.append(':');
      if (v == null) {
        b.append('-');
      } else if (v instanceof BytesRef br) {
        b.append('x');
        for (int i = 0; i < br.length; i++) {
          b.append(String.format("%02x", br.bytes[br.offset + i] & 0xff));
        }
      } else if (v instanceof Float f) {
        b.append(Float.floatToRawIntBits(f));
      } else {
        b.append(v);
      }
    }
    return b.toString();
  }
}
