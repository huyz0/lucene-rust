import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.LongPoint;
import org.apache.lucene.document.SortedDocValuesField;
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
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreMode;
import org.apache.lucene.search.Scorer;
import org.apache.lucene.search.Weight;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.BytesRef;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.TreeMap;
import java.util.stream.Stream;

/**
 * OpenSearch's {@code terms} aggregation on keyword fields, as a shard computes it, recorded for
 * crates/lucene-search/tests/terms_aggs_fixtures.rs.
 *
 * <p>This generator runs on Lucene's jars alone, so the shard's steps are reproduced here: every
 * live match of the query counted once per distinct term it holds ({@code
 * GlobalOrdinalsStringTermsAggregator}), the top {@code shard_size} kept by count descending then
 * term ascending (unsigned bytes; the compound order {@code TermsAggregationBuilder} builds), the
 * rest summed into {@code otherDocCount}, the kept listed by term ascending ({@code reduceOrder}
 * {@code KEY_ASC}). Once over the whole shard, and once per slice of a concurrent search over the
 * slices {@code [[0,2],[1,3]]}, each counting from scratch.
 *
 * <p>Four segments of 10,000 documents, three with deletions:
 *
 * <pre>
 *   kw   keyword, every document, ~40 terms (SORTED_SET, single-valued)
 *   mkw  keyword, 0-4 values, ~300 terms (SORTED_SET, multi-valued)
 *   hk   keyword, half the documents, ~15,000 terms
 *   bk   keyword, raw bytes including 0x00 and 0xff (unsigned order)
 *   sk   a SORTED field (single-valued doc values of the other kind)
 *   pk   keyword as OpenSearch indexes one: postings and doc values, 0-2 values, some repeated
 * </pre>
 *
 * <p>Each run records {@code run.N.<field>.<shardSize>.<slice>=other|term:count,...} with the
 * terms in hex, {@code <slice>} being {@code all} or the slice's index.
 */
public class GenTermsAggs {
  static final int DOCS_PER_SEGMENT = 10_000;
  static final int SEGMENTS = 4;
  static final String[] FIELDS = {"kw", "mkw", "hk", "bk", "sk", "pk"};
  static final int[] SHARD_SIZES = {1, 3, 25, 1000};
  static final int[][] SLICES = {{0, 2}, {1, 3}};

  static final String[] QUERIES = {
    "(all)",
    "(t w0)",
    "(t w30)",
    "(b 0 (+ (t w1)) (- (t w2)))",
    "(b 0 (# (t w0)) (# (r 5000 25000)))",
    "(b 0 (? (t w3)) (? (t w4)))",
    "(t nosuchterm)",
  };

  public static void main(String[] args) throws IOException {
    Path out = Path.of(args[0]).resolve("terms_aggs_index");
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
      Random random = new Random(20260926L);
      try (IndexWriter w = new IndexWriter(dir, cfg)) {
        for (int seg = 0; seg < SEGMENTS; seg++) {
          for (int i = 0; i < DOCS_PER_SEGMENT; i++) {
            int id = seg * DOCS_PER_SEGMENT + i;
            Document doc = new Document();
            doc.add(new StringField("id", Integer.toString(id), Field.Store.NO));
            doc.add(new TextField("body", GenMixedBooleanScoring.body(random), Field.Store.NO));
            doc.add(new LongPoint("r", id));
            // Skewed, so counts tie and differ.
            int k = (int) Math.floor(Math.pow(random.nextDouble(), 2) * 40);
            doc.add(new SortedSetDocValuesField("kw", new BytesRef("k" + k)));
            for (int v = 0, n = random.nextInt(5); v < n; v++) {
              doc.add(new SortedSetDocValuesField("mkw", new BytesRef("m" + random.nextInt(300))));
            }
            if (random.nextBoolean()) {
              doc.add(new SortedSetDocValuesField("hk", new BytesRef("h" + random.nextInt(15_000))));
            }
            byte[] b = {(byte) (random.nextInt(4) * 85), (byte) random.nextInt(3)};
            doc.add(new SortedSetDocValuesField("bk", new BytesRef(b)));
            if (random.nextInt(3) != 0) {
              doc.add(new SortedDocValuesField("sk", new BytesRef("s" + random.nextInt(12))));
            }
            for (int v = 0, n = random.nextInt(3); v < n; v++) {
              String p = "p" + random.nextInt(v == 1 && random.nextBoolean() ? 3 : 60);
              doc.add(new StringField("pk", p, Field.Store.NO));
              doc.add(new SortedSetDocValuesField("pk", new BytesRef(p)));
            }
            w.addDocument(doc);
          }
          w.commit();
        }
        for (int id = 0; id < DOCS_PER_SEGMENT; id += 17) {
          w.deleteDocuments(new Term("id", Integer.toString(id)));
          w.deleteDocuments(new Term("id", Integer.toString(2 * DOCS_PER_SEGMENT + id + 3)));
          w.deleteDocuments(new Term("id", Integer.toString(3 * DOCS_PER_SEGMENT + id / 2)));
        }
        w.commit();
      }

      SegmentInfos sis = SegmentInfos.readLatestCommit(dir);
      if (sis.size() != SEGMENTS) {
        throw new AssertionError("expected " + SEGMENTS + " segments, got " + sis.size());
      }
      int run = 0;
      try (DirectoryReader reader = DirectoryReader.open(dir)) {
        IndexSearcher searcher = new IndexSearcher(reader);
        searcher.setQueryCache(null);
        int[] all = new int[reader.leaves().size()];
        for (int i = 0; i < all.length; i++) {
          all[i] = i;
        }
        for (String qs : QUERIES) {
          Query q = GenSortedSearch.parse(new GenMixedBooleanScoring.Tokens(qs));
          Weight weight = searcher.createWeight(searcher.rewrite(q), ScoreMode.COMPLETE_NO_SCORES, 1f);
          m.append("run.").append(run).append(".query=").append(qs).append('\n');
          for (String field : FIELDS) {
            TreeMap<BytesRef, Long> whole = counts(reader, weight, field, all);
            List<TreeMap<BytesRef, Long>> sliced = new ArrayList<>();
            for (int[] slice : SLICES) {
              sliced.add(counts(reader, weight, field, slice));
            }
            for (int shardSize : SHARD_SIZES) {
              String prefix = "run." + run + "." + field + "." + shardSize + ".";
              m.append(prefix).append("all=").append(select(whole, shardSize)).append('\n');
              for (int s = 0; s < SLICES.length; s++) {
                m.append(prefix).append(s).append('=').append(select(sliced.get(s), shardSize)).append('\n');
              }
            }
          }
          run++;
        }
      }
      m.insert(0, "run_count=" + run + "\n");
    }
    Files.writeString(out.resolve("manifest.properties"), m.toString());
    System.out.println("wrote " + out);
  }

  /** Every term's count over the live matches in the given leaves. */
  static TreeMap<BytesRef, Long> counts(DirectoryReader reader, Weight weight, String field, int[] leaves)
      throws IOException {
    TreeMap<BytesRef, Long> counts = new TreeMap<>();
    for (int ord : leaves) {
      LeafReaderContext leaf = reader.leaves().get(ord);
      SortedSetDocValues dv = DocValues.getSortedSet(leaf.reader(), field);
      Scorer scorer = weight.scorer(leaf);
      if (scorer == null) {
        continue;
      }
      Bits live = leaf.reader().getLiveDocs();
      DocIdSetIterator it = scorer.iterator();
      for (int doc = it.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS; doc = it.nextDoc()) {
        if (live != null && live.get(doc) == false) {
          continue;
        }
        if (dv.advanceExact(doc) == false) {
          continue;
        }
        for (int i = 0; i < dv.docValueCount(); i++) {
          BytesRef term = BytesRef.deepCopyOf(dv.lookupOrd(dv.nextOrd()));
          counts.merge(term, 1L, Long::sum);
        }
      }
    }
    return counts;
  }

  /** The top {@code shardSize} by count desc, term asc; the rest's sum; the kept by term. */
  static String select(TreeMap<BytesRef, Long> counts, int shardSize) {
    List<Map.Entry<BytesRef, Long>> ranked = new ArrayList<>(counts.entrySet());
    ranked.sort((a, b) -> {
      int c = Long.compare(b.getValue(), a.getValue());
      return c != 0 ? c : a.getKey().compareTo(b.getKey());
    });
    long other = 0;
    TreeMap<BytesRef, Long> kept = new TreeMap<>();
    for (int i = 0; i < ranked.size(); i++) {
      if (i < shardSize) {
        kept.put(ranked.get(i).getKey(), ranked.get(i).getValue());
      } else {
        other += ranked.get(i).getValue();
      }
    }
    StringBuilder b = new StringBuilder().append(other).append('|');
    boolean first = true;
    for (Map.Entry<BytesRef, Long> e : kept.entrySet()) {
      if (!first) {
        b.append(',');
      }
      first = false;
      b.append(hex(e.getKey())).append(':').append(e.getValue());
    }
    return b.toString();
  }

  static String hex(BytesRef b) {
    StringBuilder s = new StringBuilder();
    for (int i = 0; i < b.length; i++) {
      s.append(String.format("%02x", b.bytes[b.offset + i] & 0xff));
    }
    return s.toString();
  }
}
