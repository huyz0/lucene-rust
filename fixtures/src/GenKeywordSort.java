import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.LongPoint;
import org.apache.lucene.document.SortedDocValuesField;
import org.apache.lucene.document.SortedNumericDocValuesField;
import org.apache.lucene.document.SortedSetDocValuesField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.DocValues;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.SegmentInfos;
import org.apache.lucene.index.SortedSetDocValues;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.FieldDoc;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.Sort;
import org.apache.lucene.search.SortField;
import org.apache.lucene.search.SortedNumericSortField;
import org.apache.lucene.search.SortedSetSelector;
import org.apache.lucene.search.SortedSetSortField;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TopFieldCollectorManager;
import org.apache.lucene.search.TotalHits;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.BytesRef;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HexFormat;
import java.util.List;
import java.util.Random;
import java.util.stream.Stream;

/**
 * Keyword sorts -- {@code SortedSetSortField} through {@code TermOrdValComparator} -- recorded from
 * Lucene for crates/lucene-search/tests/keyword_sort_fixtures.rs, plus each segment's {@code
 * lookupTerm}/{@code lookupOrd} answers for the doc-values terms dictionaries.
 *
 * <p>Three segments of 20,000 documents, the first and last with deletions:
 *
 * <pre>
 *   k   SORTED_SET, one value, ~8,000 distinct terms with shared prefixes, 15% missing; indexed
 *   km  SORTED_SET, 0-3 values per document (MIN and MAX selectors); indexed
 *   kd  SORTED, one value in every document; indexed
 *   kn  SORTED_SET, one value, doc values only (no postings)
 *   i   int, SortedNumeric (a numeric key beside a keyword one)
 * </pre>
 *
 * <p>Queries use GenSortedSearch's grammar. A sort key is written {@code
 * field:string:selector:reverse:missingLast} ({@code 1} for {@code STRING_LAST}) or as in
 * GenSortedSearch; a keyword value in a hit is {@code x} and its bytes in hex, or {@code n} for a
 * document without one.
 */
public class GenKeywordSort {
  static final int DOCS_PER_SEGMENT = 20_000;
  static final int SEGMENTS = 3;

  static final String[] QUERIES = {
    "(all)",
    "(t w0)",
    "(t w30)",
    "(b 0 (+ (t w1)) (- (t w2)))",
    "(b 0 (# (t w0)) (# (r 5000 45000)))",
    "(b 0 (+ (t w0)) (? (p w1 w2)))",
  };

  static final String[] SORTS = {
    "k:asc:first",
    "k:asc:last",
    "k:desc:first",
    "k:desc:last",
    "km:asc:last:min",
    "km:desc:last:max",
    "kd:asc:last",
    "kd:desc:first",
    "kn:desc:last",
    "k:asc:last,i",
    "i,k:desc:first",
    "score,k:asc:last",
    "k:asc:first,doc",
  };

  static String hex(BytesRef b) {
    return HexFormat.of().formatHex(b.bytes, b.offset, b.length);
  }

  static String term(Random r) {
    // Shared prefixes and varying lengths, so blocks and the reverse index both matter.
    String[] stems = {"alpha", "al", "beta", "b", "gamma-ray", "delta", "d", "zeta", "", "omega"};
    return stems[r.nextInt(stems.length)] + Integer.toString(r.nextInt(900), 36)
        + (r.nextInt(5) == 0 ? "-" + (char) ('a' + r.nextInt(26)) : "");
  }

  public static void main(String[] args) throws IOException {
    Path out = Path.of(args[0]).resolve("keyword_sort_index");
    if (Files.exists(out)) {
      try (Stream<Path> s = Files.walk(out)) {
        s.sorted(Comparator.reverseOrder()).forEach(p -> p.toFile().delete());
      }
    }
    Files.createDirectories(out);

    StringBuilder m = new StringBuilder();
    try (Directory dir = FSDirectory.open(out)) {
      IndexWriterConfig cfg = new IndexWriterConfig(new StandardAnalyzer());
      cfg.setUseCompoundFile(false);
      cfg.setMergePolicy(NoMergePolicy.INSTANCE);
      cfg.setRAMBufferSizeMB(256);
      cfg.setMaxBufferedDocs(IndexWriterConfig.DISABLE_AUTO_FLUSH);
      Random random = new Random(20260927L);
      try (IndexWriter w = new IndexWriter(dir, cfg)) {
        for (int seg = 0; seg < SEGMENTS; seg++) {
          for (int i = 0; i < DOCS_PER_SEGMENT; i++) {
            int id = seg * DOCS_PER_SEGMENT + i;
            Document doc = new Document();
            doc.add(new StringField("id", Integer.toString(id), Field.Store.NO));
            doc.add(new TextField("body", GenMixedBooleanScoring.body(random), Field.Store.NO));
            doc.add(new LongPoint("r", id));
            if (random.nextInt(100) >= 15) {
              String k = term(random);
              doc.add(new SortedSetDocValuesField("k", new BytesRef(k)));
              doc.add(new StringField("k", k, Field.Store.NO));
            }
            for (int v = 0, n = random.nextInt(4); v < n; v++) {
              String k = term(random);
              doc.add(new SortedSetDocValuesField("km", new BytesRef(k)));
              doc.add(new StringField("km", k, Field.Store.NO));
            }
            String kd = term(random);
            doc.add(new SortedDocValuesField("kd", new BytesRef(kd)));
            doc.add(new StringField("kd", kd, Field.Store.NO));
            if (random.nextInt(10) != 0) {
              doc.add(new SortedSetDocValuesField("kn", new BytesRef(term(random))));
            }
            doc.add(new SortedNumericDocValuesField("i", random.nextInt(50)));
            w.addDocument(doc);
          }
          w.commit();
        }
        for (int id = 0; id < DOCS_PER_SEGMENT; id += 31) {
          w.deleteDocuments(new Term("id", Integer.toString(id)));
          w.deleteDocuments(new Term("id", Integer.toString(2 * DOCS_PER_SEGMENT + id + 5)));
        }
        w.commit();
      }

      SegmentInfos sis = SegmentInfos.readLatestCommit(dir);
      if (sis.size() != SEGMENTS) {
        throw new AssertionError("expected " + SEGMENTS + " segments, got " + sis.size());
      }
      int run = 0;
      try (DirectoryReader reader = DirectoryReader.open(dir)) {
        // The terms dictionaries, probed: lookupTerm for present, absent, before-first and
        // after-last keys, and lookupOrd for a sample of ordinals.
        int probe = 0;
        Random pr = new Random(7);
        for (LeafReaderContext leaf : reader.leaves()) {
          for (String field : new String[] {"k", "kd", "km"}) {
            SortedSetDocValues dv = DocValues.getSortedSet(leaf.reader(), field);
            long count = dv.getValueCount();
            m.append("dict.").append(leaf.ord).append('.').append(field).append(".count=").append(count).append('\n');
            List<BytesRef> keys = new ArrayList<>();
            for (int j = 0; j < 150; j++) {
              long ord = (long) (pr.nextDouble() * count);
              BytesRef t = BytesRef.deepCopyOf(dv.lookupOrd(ord));
              m.append("ord.").append(leaf.ord).append('.').append(field).append('.').append(ord).append('=')
                  .append(hex(t)).append('\n');
              keys.add(t);
              BytesRef longer = new BytesRef(t.utf8ToString() + "\u0000");
              keys.add(longer);
              if (t.length > 1) {
                keys.add(new BytesRef(t.bytes, t.offset, t.length - 1));
              }
            }
            keys.add(new BytesRef(""));
            keys.add(new BytesRef("~~~~"));
            keys.add(new BytesRef("zzzzzzzz"));
            for (BytesRef key : keys) {
              m.append("probe.").append(probe++).append('=').append(leaf.ord).append(':').append(field)
                  .append(':').append(hex(key)).append(':').append(dv.lookupTerm(key)).append('\n');
            }
          }
        }
        m.append("probe_count=").append(probe).append('\n');

        IndexSearcher searcher = new IndexSearcher(reader);
        searcher.setQueryCache(null);
        for (String qs : QUERIES) {
          Query q = GenSortedSearch.parse(new GenMixedBooleanScoring.Tokens(qs));
          for (String ss : SORTS) {
            Sort sort = sort(ss);
            for (int topN : new int[] {10, 50}) {
              for (int threshold : new int[] {100, Integer.MAX_VALUE}) {
                TopDocs page1 =
                    searcher.search(q, new TopFieldCollectorManager(sort, topN, null, threshold));
                run = record(m, run, qs, sort, topN, threshold, null, page1);
                if (page1.scoreDocs.length == 0) {
                  continue;
                }
                FieldDoc last = (FieldDoc) page1.scoreDocs[page1.scoreDocs.length - 1];
                TopDocs page2 =
                    searcher.search(q, new TopFieldCollectorManager(sort, topN, last, threshold));
                run = record(m, run, qs, sort, topN, threshold, last, page2);
                FieldDoc byValues = new FieldDoc(Integer.MAX_VALUE, Float.NaN, last.fields);
                TopDocs page2v =
                    searcher.search(q, new TopFieldCollectorManager(sort, topN, byValues, threshold));
                run = record(m, run, qs, sort, topN, threshold, byValues, page2v);
              }
            }
          }
        }
      }
      m.insert(0, "run_count=" + run + "\n");
    }
    Files.writeString(out.resolve("manifest.properties"), m.toString());
    System.out.println("wrote " + out);
  }

  static int record(
      StringBuilder m, int run, String q, Sort sort, int topN, int threshold, FieldDoc after, TopDocs td) {
    String k = "run." + run;
    m.append(k).append(".query=").append(q).append('\n');
    m.append(k).append(".sort=").append(spec(sort)).append('\n');
    m.append(k).append(".top_n=").append(topN).append('\n');
    m.append(k).append(".threshold=").append(threshold == Integer.MAX_VALUE ? "max" : threshold).append('\n');
    if (after != null) {
      m.append(k).append(".after=").append(hit(sort, after)).append('\n');
    }
    StringBuilder hits = new StringBuilder();
    for (ScoreDoc sd : td.scoreDocs) {
      if (hits.length() > 0) {
        hits.append(',');
      }
      hits.append(hit(sort, (FieldDoc) sd));
    }
    m.append(k).append(".hits=").append(hits).append('\n');
    m.append(k).append(".total=").append(td.totalHits.value()).append('\n');
    m.append(k).append(".relation=")
        .append(td.totalHits.relation() == TotalHits.Relation.EQUAL_TO ? "eq" : "gte")
        .append('\n');
    return run + 1;
  }

  static String hit(Sort sort, FieldDoc fd) {
    StringBuilder b = new StringBuilder().append(fd.doc);
    for (int i = 0; i < fd.fields.length; i++) {
      SortField f = sort.getSort()[i];
      b.append(':');
      if (f instanceof SortedSetSortField) {
        b.append(fd.fields[i] == null ? "n" : "x" + hex((BytesRef) fd.fields[i]));
      } else {
        b.append(GenSortedSearch.comparable(f, fd.fields[i]));
      }
    }
    return b.toString();
  }

  static String spec(Sort sort) {
    StringBuilder b = new StringBuilder();
    for (SortField f : sort.getSort()) {
      if (b.length() > 0) {
        b.append(',');
      }
      if (f instanceof SortedSetSortField ss) {
        b.append(f.getField()).append(":string:")
            .append(ss.getSelector() == SortedSetSelector.Type.MAX ? "max" : "min")
            .append(':').append(f.getReverse())
            .append(':').append(f.getMissingValue() == SortField.STRING_LAST ? 1 : 0);
      } else {
        b.append(GenSortedSearch.spec(new Sort(f)));
      }
    }
    return b.toString();
  }

  static Sort sort(String s) {
    List<SortField> fields = new ArrayList<>();
    for (String k : s.split(",")) {
      switch (k) {
        case "score" -> fields.add(SortField.FIELD_SCORE);
        case "doc" -> fields.add(SortField.FIELD_DOC);
        case "i" -> fields.add(new SortedNumericSortField("i", SortField.Type.INT));
        default -> {
          String[] p = k.split(":");
          boolean reverse = p[1].equals("desc");
          SortedSetSelector.Type sel =
              p.length > 3 && p[3].equals("max") ? SortedSetSelector.Type.MAX : SortedSetSelector.Type.MIN;
          SortedSetSortField f = new SortedSetSortField(p[0], reverse, sel);
          f.setMissingValue(p[2].equals("last") ? SortField.STRING_LAST : SortField.STRING_FIRST);
          fields.add(f);
        }
      }
    }
    return new Sort(fields.toArray(new SortField[0]));
  }
}
