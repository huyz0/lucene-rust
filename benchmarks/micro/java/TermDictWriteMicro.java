import java.io.IOException;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.TreeSet;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.FieldsConsumer;
import org.apache.lucene.codecs.lucene104.Lucene104PostingsFormat;
import org.apache.lucene.index.BaseTermsEnum;
import org.apache.lucene.index.DocValuesSkipIndexType;
import org.apache.lucene.index.DocValuesType;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.Fields;
import org.apache.lucene.index.ImpactsEnum;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.PostingsEnum;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.Terms;
import org.apache.lucene.index.TermsEnum;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.util.BytesRef;
import org.apache.lucene.util.InfoStream;
import org.apache.lucene.util.StringHelper;
import org.apache.lucene.util.Version;

/**
 * Term-dictionary write throughput: {@code Lucene104PostingsFormat}'s {@code FieldsConsumer} --
 * {@code Lucene103BlockTreeTermsWriter} over {@code Lucene104PostingsWriter} -- writing one field
 * into an in-memory directory. The Rust side is {@code bench_term_dict_write} in {@code
 * benchmarks/rust-runner/src/micro.rs}, over the same generated terms.
 *
 * <p>Every term is a singleton ({@code docFreq == 1}, {@code IndexOptions.DOCS}), so the postings
 * writer emits no {@code .doc} bytes and what is measured is the term dictionary: block splitting,
 * suffix compression, the trie, and {@code encodeTerm}. For these inputs both engines write the
 * same bytes ({@code crates/lucene-codecs/tests/blocktree_writer_identity.rs}), so this is a
 * comparison of identical work.
 *
 * <p>Emits {@code case<TAB>ns_per_term<TAB>terms}.
 */
public final class TermDictWriteMicro {

  static long warmupMs = Long.getLong("warmupMs", 1500);
  static long measureMs = Long.getLong("measureMs", 2000);

  public static void main(String[] args) throws IOException {
    run("ids_1m", idTerms(1_000_000));
    run("words_200k", wordTerms(200_000));
  }

  /** {@code %08d} of 0..n, already in byte order. */
  static BytesRef[] idTerms(int n) {
    BytesRef[] out = new BytesRef[n];
    for (int i = 0; i < n; i++) {
      out[i] = new BytesRef(String.format("%08d", i));
    }
    return out;
  }

  /** Random lowercase words, 4-12 letters, xorshift64 identical to the Rust side; sorted. */
  static BytesRef[] wordTerms(int n) {
    long s = 42;
    TreeSet<String> set = new TreeSet<>();
    while (set.size() < n) {
      s ^= s << 13;
      s ^= s >>> 7;
      s ^= s << 17;
      int len = 4 + (int) Long.remainderUnsigned(s, 9);
      StringBuilder sb = new StringBuilder(len);
      for (int i = 0; i < len; i++) {
        s ^= s << 13;
        s ^= s >>> 7;
        s ^= s << 17;
        sb.append((char) ('a' + Long.remainderUnsigned(s, 26)));
      }
      set.add(sb.toString());
    }
    BytesRef[] out = new BytesRef[n];
    int i = 0;
    for (String t : set) {
      out[i++] = new BytesRef(t);
    }
    return out;
  }

  static void run(String name, BytesRef[] terms) throws IOException {
    long bytes = writeOnce(terms);
    System.err.printf("%s: %d terms, %d bytes written%n", name, terms.length, bytes);
    loop(terms, warmupMs);
    long start = System.nanoTime();
    long units = loop(terms, measureMs);
    long elapsed = System.nanoTime() - start;
    System.out.printf("%s\t%.3f\t%d%n", name, (double) elapsed / units, units);
    System.out.flush();
  }

  static long loop(BytesRef[] terms, long budgetMs) throws IOException {
    long budgetNs = budgetMs * 1_000_000L;
    long units = 0;
    long start = System.nanoTime();
    do {
      writeOnce(terms);
      units += terms.length;
    } while (System.nanoTime() - start < budgetNs);
    return units;
  }

  /** One flush of the whole field; returns the bytes written. */
  static long writeOnce(BytesRef[] terms) throws IOException {
    ByteBuffersDirectory dir = new ByteBuffersDirectory();
    FieldInfo fi =
        new FieldInfo(
            "f",
            0,
            false,
            true,
            false,
            IndexOptions.DOCS,
            DocValuesType.NONE,
            DocValuesSkipIndexType.NONE,
            -1,
            Collections.emptyMap(),
            0,
            0,
            0,
            0,
            VectorEncoding.FLOAT32,
            VectorSimilarityFunction.EUCLIDEAN,
            false,
            false);
    FieldInfos fis = new FieldInfos(new FieldInfo[] {fi});
    SegmentInfo si =
        new SegmentInfo(
            dir,
            Version.LATEST,
            Version.LATEST,
            "_0",
            terms.length,
            false,
            false,
            Codec.getDefault(),
            Collections.emptyMap(),
            StringHelper.randomId(),
            Collections.emptyMap(),
            null);
    SegmentWriteState state =
        new SegmentWriteState(InfoStream.NO_OUTPUT, dir, si, fis, null, IOContext.DEFAULT);
    try (FieldsConsumer consumer = new Lucene104PostingsFormat().fieldsConsumer(state)) {
      consumer.write(new ArrayFields(terms), null);
    }
    long total = 0;
    for (String f : dir.listAll()) {
      total += dir.fileLength(f);
    }
    return total;
  }

  /** One field, "f", whose i-th term occurs once, in document i. */
  static final class ArrayFields extends Fields {
    final BytesRef[] terms;

    ArrayFields(BytesRef[] terms) {
      this.terms = terms;
    }

    @Override
    public Iterator<String> iterator() {
      return List.of("f").iterator();
    }

    @Override
    public Terms terms(String field) {
      return new Terms() {
        @Override
        public TermsEnum iterator() {
          return new ArrayTermsEnum(terms);
        }

        @Override
        public long size() {
          return terms.length;
        }

        @Override
        public long getSumTotalTermFreq() {
          return terms.length;
        }

        @Override
        public long getSumDocFreq() {
          return terms.length;
        }

        @Override
        public int getDocCount() {
          return terms.length;
        }

        @Override
        public boolean hasFreqs() {
          return false;
        }

        @Override
        public boolean hasOffsets() {
          return false;
        }

        @Override
        public boolean hasPositions() {
          return false;
        }

        @Override
        public boolean hasPayloads() {
          return false;
        }
      };
    }

    @Override
    public int size() {
      return 1;
    }
  }

  static final class ArrayTermsEnum extends BaseTermsEnum {
    final BytesRef[] terms;
    int ord = -1;
    final SingletonPostings postings = new SingletonPostings();

    ArrayTermsEnum(BytesRef[] terms) {
      this.terms = terms;
    }

    @Override
    public BytesRef next() {
      return ++ord < terms.length ? terms[ord] : null;
    }

    @Override
    public BytesRef term() {
      return terms[ord];
    }

    @Override
    public int docFreq() {
      return 1;
    }

    @Override
    public long totalTermFreq() {
      return 1;
    }

    @Override
    public PostingsEnum postings(PostingsEnum reuse, int flags) {
      postings.reset(ord);
      return postings;
    }

    @Override
    public SeekStatus seekCeil(BytesRef text) {
      throw new UnsupportedOperationException();
    }

    @Override
    public void seekExact(long ord) {
      throw new UnsupportedOperationException();
    }

    @Override
    public long ord() {
      return ord;
    }

    @Override
    public ImpactsEnum impacts(int flags) {
      throw new UnsupportedOperationException();
    }
  }

  static final class SingletonPostings extends PostingsEnum {
    int target;
    int doc = -1;

    void reset(int target) {
      this.target = target;
      this.doc = -1;
    }

    @Override
    public int freq() {
      return 1;
    }

    @Override
    public int nextPosition() {
      return -1;
    }

    @Override
    public int startOffset() {
      return -1;
    }

    @Override
    public int endOffset() {
      return -1;
    }

    @Override
    public BytesRef getPayload() {
      return null;
    }

    @Override
    public int docID() {
      return doc;
    }

    @Override
    public int nextDoc() {
      return doc = (doc == -1 ? target : NO_MORE_DOCS);
    }

    @Override
    public int advance(int t) {
      throw new UnsupportedOperationException();
    }

    @Override
    public long cost() {
      return 1;
    }
  }
}
