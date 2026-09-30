import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.IntPoint;
import org.apache.lucene.document.NumericDocValuesField;
import org.apache.lucene.document.SortedDocValuesField;
import org.apache.lucene.document.StoredField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.CheckIndex;
import org.apache.lucene.index.CodecReader;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FilterCodecReader;
import org.apache.lucene.index.FilterMergePolicy;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.MergePolicy;
import org.apache.lucene.index.MergeTrigger;
import org.apache.lucene.index.NumericDocValues;
import org.apache.lucene.index.SegmentCommitInfo;
import org.apache.lucene.index.SegmentInfos;
import org.apache.lucene.index.Sorter;
import org.apache.lucene.index.StoredFields;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.TieredMergePolicy;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.BytesRef;
import org.apache.lucene.util.FixedBitSet;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Executor;

/**
 * {@code OneMerge.wrapForMerge} and {@code OneMerge.reorder} through {@code IndexWriter}: a merge
 * policy ({@link Reordering}, a {@code FilterMergePolicy} over {@code TieredMergePolicy}) whose every
 * merge is a {@code OneMerge} subclass ({@link RankMerge}) that
 *
 * <ul>
 *   <li>wraps each source ({@code FilterCodecReader.wrapLiveDocs}) so a document whose {@code rank}
 *       is a multiple of 11 is not carried over, and
 *   <li>reorders the merged view by {@code rank} descending, ties by merged-view doc id -- the shape
 *       of {@code BPReorderingMergePolicy}, with a key a test can recompute.
 * </ul>
 *
 * <p>Three flushed segments of {@link #PER_SEGMENT} documents ({@code id} stored and indexed,
 * {@code body} text, {@code rank} NUMERIC doc values, {@code tag} SORTED doc values, {@code pt} an
 * {@code IntPoint}), every document whose number is a multiple of 13 deleted, then {@code
 * forceMerge(1)}. {@code merge_reorder/}: the merged index; {@code order.txt}: the merged segment's
 * ids in document order. The Rust test runs the same documents, deletes and hooks and compares every
 * merged file but the {@code .si} byte for byte (segment id normalised).
 */
public class GenMergeReorder {
  static final int PER_SEGMENT = 60;
  static final String[] WORDS = {
    "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliet"
  };

  static long value(int i, int k) {
    long x = (i + 1) * 2654435761L + k * 40503L;
    return (x ^ (x >>> 13)) % 100000;
  }

  static long rank(int i) {
    return value(i, 0) % 1000;
  }

  static Document doc(int i) {
    Document d = new Document();
    d.add(new StringField("id", "d" + i, Field.Store.YES));
    d.add(
        new TextField(
            "body",
            WORDS[(int) (value(i, 1) % WORDS.length)]
                + " "
                + WORDS[(int) (value(i, 2) % WORDS.length)]
                + " "
                + WORDS[(int) (value(i, 3) % WORDS.length)],
            Field.Store.NO));
    d.add(new NumericDocValuesField("rank", rank(i)));
    d.add(new SortedDocValuesField("tag", new BytesRef(WORDS[(int) (value(i, 4) % WORDS.length)])));
    d.add(new IntPoint("pt", (int) (value(i, 5) % 5000)));
    d.add(new StoredField("n", i));
    return d;
  }

  /** The {@code OneMerge} subclass: hides every rank multiple of 11, sorts by rank descending. */
  static final class RankMerge extends MergePolicy.OneMerge {
    RankMerge(List<SegmentCommitInfo> segments) {
      super(segments);
    }

    @Override
    public CodecReader wrapForMerge(CodecReader reader) throws IOException {
      Bits live = reader.getLiveDocs();
      NumericDocValues rank = reader.getNumericDocValues("rank");
      FixedBitSet keep = new FixedBitSet(reader.maxDoc());
      int numDocs = 0;
      for (int doc = rank.nextDoc(); doc != NumericDocValues.NO_MORE_DOCS; doc = rank.nextDoc()) {
        if ((live == null || live.get(doc)) && rank.longValue() % 11 != 0) {
          keep.set(doc);
          numDocs++;
        }
      }
      final int kept = numDocs;
      return new FilterCodecReader(reader) {
        @Override
        public Bits getLiveDocs() {
          return keep;
        }

        @Override
        public int numDocs() {
          return kept;
        }

        @Override
        public CacheHelper getCoreCacheHelper() {
          return null;
        }

        @Override
        public CacheHelper getReaderCacheHelper() {
          return null;
        }
      };
    }

    @Override
    public Sorter.DocMap reorder(CodecReader reader, Directory dir, Executor executor)
        throws IOException {
      int maxDoc = reader.maxDoc();
      long[] ranks = new long[maxDoc];
      NumericDocValues rank = reader.getNumericDocValues("rank");
      for (int doc = rank.nextDoc(); doc != NumericDocValues.NO_MORE_DOCS; doc = rank.nextDoc()) {
        ranks[doc] = rank.longValue();
      }
      Integer[] order = new Integer[maxDoc];
      for (int i = 0; i < maxDoc; i++) {
        order[i] = i;
      }
      Arrays.sort(order, (a, b) -> ranks[a] != ranks[b] ? Long.compare(ranks[b], ranks[a]) : Integer.compare(a, b));
      int[] newToOld = new int[maxDoc];
      int[] oldToNew = new int[maxDoc];
      for (int i = 0; i < maxDoc; i++) {
        newToOld[i] = order[i];
        oldToNew[order[i]] = i;
      }
      return new Sorter.DocMap() {
        @Override
        public int oldToNew(int docID) {
          return oldToNew[docID];
        }

        @Override
        public int newToOld(int docID) {
          return newToOld[docID];
        }

        @Override
        public int size() {
          return maxDoc;
        }
      };
    }
  }

  /** Every merge the delegate finds, as a {@link RankMerge}. */
  static final class Reordering extends FilterMergePolicy {
    Reordering(MergePolicy in) {
      super(in);
    }

    static MergeSpecification wrap(MergeSpecification spec) {
      if (spec == null) {
        return null;
      }
      MergeSpecification out = new MergeSpecification();
      for (OneMerge m : spec.merges) {
        out.add(new RankMerge(m.segments));
      }
      return out;
    }

    @Override
    public MergeSpecification findMerges(
        MergeTrigger trigger, SegmentInfos infos, MergeContext ctx) throws IOException {
      return wrap(in.findMerges(trigger, infos, ctx));
    }

    @Override
    public MergeSpecification findForcedMerges(
        SegmentInfos infos,
        int maxSegmentCount,
        Map<SegmentCommitInfo, Boolean> segmentsToMerge,
        MergeContext ctx)
        throws IOException {
      return wrap(in.findForcedMerges(infos, maxSegmentCount, segmentsToMerge, ctx));
    }
  }

  public static void main(String[] args) throws IOException {
    Path out = Path.of(args[0]).resolve("merge_reorder");
    if (Files.exists(out)) {
      try (var walk = Files.walk(out)) {
        walk.sorted(Comparator.reverseOrder()).forEach(p -> p.toFile().delete());
      }
    }
    Files.createDirectories(out);
    try (Directory dir = FSDirectory.open(out)) {
      IndexWriterConfig cfg = new IndexWriterConfig();
      cfg.setUseCompoundFile(false);
      cfg.setRAMBufferSizeMB(256);
      TieredMergePolicy tmp = new TieredMergePolicy();
      tmp.setNoCFSRatio(0.0);
      cfg.setMergePolicy(new Reordering(tmp));
      cfg.setMaxFullFlushMergeWaitMillis(0);
      try (IndexWriter w = new IndexWriter(dir, cfg)) {
        for (int seg = 0; seg < 3; seg++) {
          for (int i = seg * PER_SEGMENT; i < (seg + 1) * PER_SEGMENT; i++) {
            w.addDocument(doc(i));
          }
          w.commit();
        }
        for (int i = 0; i < 3 * PER_SEGMENT; i += 13) {
          w.deleteDocuments(new Term("id", "d" + i));
        }
        w.commit();
        w.forceMerge(1);
        w.commit();
      }
      List<String> ids = new ArrayList<>();
      try (DirectoryReader reader = DirectoryReader.open(dir)) {
        if (reader.leaves().size() != 1) {
          throw new AssertionError("one merged segment expected");
        }
        for (LeafReaderContext leaf : reader.leaves()) {
          StoredFields stored = leaf.reader().storedFields();
          for (int d = 0; d < leaf.reader().maxDoc(); d++) {
            ids.add(stored.document(d).get("id"));
          }
        }
      }
      Files.write(out.resolve("order.txt"), ids, StandardCharsets.UTF_8);
      try (CheckIndex check = new CheckIndex(dir)) {
        if (!check.checkIndex().clean) {
          throw new AssertionError("CheckIndex failed");
        }
      }
    }
    System.out.println("wrote merge_reorder/");
  }
}
