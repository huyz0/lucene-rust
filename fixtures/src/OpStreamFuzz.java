import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.TreeSet;
import java.util.stream.Stream;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.LongPoint;
import org.apache.lucene.document.NumericDocValuesField;
import org.apache.lucene.document.SortedDocValuesField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.NumericDocValues;
import org.apache.lucene.index.PointValues;
import org.apache.lucene.index.SerialMergeScheduler;
import org.apache.lucene.index.SortedDocValues;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.PhraseQuery;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.BytesRef;

/**
 * The Java half of M4's differential operation-stream fuzzer (T4.5): the same seeded stream of
 * adds, updates, deletes (by id and by body word), doc-values updates, flushes and commits,
 * applied to real Lucene's {@code IndexWriter}, then dumped **semantically** -- live documents
 * with their doc values and points, the live documents matching every term and a set of
 * phrases -- so the dump is comparable with this port's regardless of how either engine laid out
 * its segments.
 *
 * <p>{@code crates/lucene-search/examples/op_stream_fuzz.rs} is the other half, with the same
 * generator; {@code scripts/op-stream-fuzz.sh} compares the two.
 *
 * <p>Usage: {@code OpStreamFuzz <out-dir> <first-seed> <end-seed> <ops>}: writes
 * {@code <out-dir>/<seed>.java.txt} for every seed in {@code [first, end)}.
 */
public class OpStreamFuzz {
  static final int WORDS = 10;
  static final int XS = 13;

  /** xorshift64 over a SplitMix64-spread seed; identical to the Rust side's {@code Rng}. */
  static final class Rng {
    long s;

    Rng(long seed) {
      long z = seed + 0x9e3779b97f4a7c15L;
      z = (z ^ (z >>> 30)) * 0xbf58476d1ce4e5b9L;
      z = (z ^ (z >>> 27)) * 0x94d049bb133111ebL;
      s = (z ^ (z >>> 31)) | 1;
    }

    long next() {
      s ^= s << 13;
      s ^= s >>> 7;
      s ^= s << 17;
      return s;
    }

    long below(long n) {
      return Long.remainderUnsigned(next(), n);
    }
  }

  static String body(long id, long v) {
    return "w" + ((id * 7 + v) % WORDS) + " w" + ((id + 3 * v) % WORDS) + " x" + (id % XS);
  }

  static String cat(long id, long v) {
    return (id + v) % 6 == 0 ? null : "c" + ((id + v) % 5);
  }

  static long pt(long id, long v) {
    return id * 3 - v;
  }

  static Document document(long id, long v) {
    Document d = new Document();
    d.add(new StringField("id", "i" + id, Field.Store.NO));
    d.add(new TextField("body", body(id, v), Field.Store.NO));
    d.add(new NumericDocValuesField("idn", id));
    d.add(new NumericDocValuesField("ver", v));
    d.add(new NumericDocValuesField("score", id * 10 + v));
    String c = cat(id, v);
    if (c != null) {
      d.add(new SortedDocValuesField("cat", new BytesRef(c)));
    }
    d.add(new LongPoint("pt", pt(id, v)));
    return d;
  }

  static void run(Path dirPath, long seed, int numOps) throws IOException {
    Rng rng = new Rng(seed);
    int maxBufferedDocs = 2 + (int) rng.below(20);
    // id -> version, for choosing an existing document exactly as the Rust side does.
    TreeMap<Long, Long> live = new TreeMap<>();
    long nextId = 0;
    IndexWriterConfig cfg = new IndexWriterConfig();
    cfg.setMaxBufferedDocs(maxBufferedDocs);
    cfg.setRAMBufferSizeMB(IndexWriterConfig.DISABLE_AUTO_FLUSH);
    cfg.setMergeScheduler(new SerialMergeScheduler());
    try (Directory dir = FSDirectory.open(dirPath);
        IndexWriter w = new IndexWriter(dir, cfg)) {
      for (int op = 0; op < numOps; op++) {
        long roll = rng.below(100);
        boolean needsPick = roll >= 48 && roll <= 77 && !(roll >= 67 && roll <= 70);
        Long pick = null;
        // One pick in five is any id ever issued -- possibly deleted; the rest
        // are a live id. Same draws, same order, as the Rust side.
        if (!needsPick) {
          pick = null;
        } else if (nextId > 0 && rng.below(5) == 0) {
          pick = rng.below(nextId);
        } else if (!live.isEmpty()) {
          long k = rng.below(live.size());
          pick = live.keySet().stream().skip(k).findFirst().get();
        }
        if (roll <= 47 || (roll <= 77 && !(roll >= 67 && roll <= 70) && pick == null)) {
          long id = nextId++;
          w.addDocument(document(id, 0));
          live.put(id, 0L);
        } else if (roll <= 59) {
          Long current = live.get(pick);
          long v = current == null ? 1 : current + 1;
          w.updateDocument(new Term("id", "i" + pick), document(pick, v));
          live.put(pick, v);
        } else if (roll <= 66) {
          w.deleteDocuments(new Term("id", "i" + pick));
          live.remove(pick);
        } else if (roll <= 70) {
          long word = rng.below(WORDS);
          w.deleteDocuments(new Term("body", "w" + word));
          live.entrySet()
              .removeIf(
                  e -> {
                    String[] ws = body(e.getKey(), e.getValue()).split(" ");
                    return ws[0].equals("w" + word) || ws[1].equals("w" + word);
                  });
        } else if (roll <= 77) {
          long value = rng.below(100_000);
          w.updateNumericDocValue(new Term("id", "i" + pick), "score", value);
        } else if (roll <= 83) {
          w.flush();
        } else {
          w.commit();
        }
      }
      w.commit();
    }
  }

  static void dump(Path dirPath, Path out) throws IOException {
    List<String> lines = new ArrayList<>();
    try (Directory dir = FSDirectory.open(dirPath);
        DirectoryReader reader = DirectoryReader.open(dir)) {
      // Global doc -> id, through the idn doc values.
      long[] idOf = new long[reader.maxDoc()];
      TreeMap<Long, String> docs = new TreeMap<>();
      for (LeafReaderContext ctx : reader.leaves()) {
        LeafReader leaf = ctx.reader();
        Bits liveDocs = leaf.getLiveDocs();
        NumericDocValues idn = leaf.getNumericDocValues("idn");
        NumericDocValues ver = leaf.getNumericDocValues("ver");
        NumericDocValues score = leaf.getNumericDocValues("score");
        SortedDocValues cat = leaf.getSortedDocValues("cat");
        Map<Integer, List<Long>> points = new TreeMap<>();
        PointValues pv = leaf.getPointValues("pt");
        if (pv != null) {
          pv.intersect(
              new PointValues.IntersectVisitor() {
                @Override
                public void visit(int docID) {
                  throw new IllegalStateException();
                }

                @Override
                public void visit(int docID, byte[] packed) {
                  points.computeIfAbsent(docID, d -> new ArrayList<>())
                      .add(LongPoint.decodeDimension(packed, 0));
                }

                @Override
                public PointValues.Relation compare(byte[] min, byte[] max) {
                  return PointValues.Relation.CELL_CROSSES_QUERY;
                }
              });
        }
        for (int doc = 0; doc < leaf.maxDoc(); doc++) {
          long id = idn.advanceExact(doc) ? idn.longValue() : -1;
          idOf[ctx.docBase + doc] = id;
          if (liveDocs != null && !liveDocs.get(doc)) {
            continue;
          }
          String v = ver.advanceExact(doc) ? Long.toString(ver.longValue()) : "-";
          String s = score.advanceExact(doc) ? Long.toString(score.longValue()) : "-";
          String c =
              cat != null && cat.advanceExact(doc)
                  ? cat.lookupOrd(cat.ordValue()).utf8ToString()
                  : "-";
          List<Long> p = points.getOrDefault(doc, List.of());
          if (docs.put(id, "doc " + id + " ver " + v + " score " + s + " cat " + c + " pt " + p)
              != null) {
            lines.add("DUPLICATE id " + id);
          }
        }
      }
      lines.addAll(docs.values());

      List<String> terms = new ArrayList<>();
      for (int i = 0; i < WORDS; i++) terms.add("w" + i);
      for (int i = 0; i < XS; i++) terms.add("x" + i);
      // Through a scored TermQuery, as the Rust side does.
      IndexSearcher searcher = new IndexSearcher(reader);
      searcher.setQueryCache(null);
      for (String t : terms) {
        lines.add("term " + t + " " + hits(searcher, new TermQuery(new Term("body", t)), idOf));
      }
      for (int a = 0; a < WORDS; a++) {
        for (int b = 0; b < WORDS; b += 3) {
          lines.add("phrase w" + a + " w" + b + " " + hits(searcher, new PhraseQuery("body", "w" + a, "w" + b), idOf));
        }
        lines.add("phrase w" + a + " x" + a + " " + hits(searcher, new PhraseQuery("body", "w" + a, "x" + a), idOf));
      }
      lines.add("range pt -20..150 " + hits(searcher, LongPoint.newRangeQuery("pt", -20, 150), idOf));
    }
    Files.writeString(out, String.join("\n", lines) + "\n", StandardCharsets.UTF_8);
    // Segment count and maxDoc: evidence that merges ran, beside the dump.
    try (Directory dir = FSDirectory.open(dirPath);
        DirectoryReader reader = DirectoryReader.open(dir)) {
      Path meta = out.resolveSibling(out.getFileName().toString().replace(".txt", ".meta"));
      Files.writeString(meta, reader.leaves().size() + " " + reader.maxDoc() + "\n");
    }
  }

  static TreeSet<Long> hits(IndexSearcher searcher, Query q, long[] idOf) throws IOException {
    TreeSet<Long> ids = new TreeSet<>();
    int n = Math.max(1, searcher.getIndexReader().maxDoc());
    for (ScoreDoc sd : searcher.search(q, n).scoreDocs) {
      ids.add(idOf[sd.doc]);
    }
    return ids;
  }

  static void delete(Path p) throws IOException {
    if (!Files.exists(p)) return;
    try (Stream<Path> walk = Files.walk(p)) {
      for (Path f : (Iterable<Path>) walk.sorted(Comparator.reverseOrder())::iterator) {
        Files.delete(f);
      }
    }
  }

  public static void main(String[] args) throws IOException {
    Path out = Path.of(args[0]);
    long first = Long.parseLong(args[1]);
    long end = Long.parseLong(args[2]);
    int numOps = Integer.parseInt(args[3]);
    Files.createDirectories(out);
    Path work = Files.createTempDirectory("op-stream-java");
    for (long seed = first; seed < end; seed++) {
      Path idx = work.resolve("s" + seed);
      run(idx, seed, numOps);
      dump(idx, out.resolve(seed + ".java.txt"));
      delete(idx);
    }
    delete(work);
    System.out.println("OpStreamFuzz: Lucene ran seeds " + first + ".." + end);
  }
}
