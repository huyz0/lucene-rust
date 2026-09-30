import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import org.apache.lucene.index.BinaryDocValues;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.DocValuesType;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.Fields;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.NumericDocValues;
import org.apache.lucene.index.PointValues;
import org.apache.lucene.index.PostingsEnum;
import org.apache.lucene.index.SegmentCommitInfo;
import org.apache.lucene.index.SegmentInfos;
import org.apache.lucene.index.SegmentReader;
import org.apache.lucene.index.SortedDocValues;
import org.apache.lucene.index.SortedNumericDocValues;
import org.apache.lucene.index.SortedSetDocValues;
import org.apache.lucene.index.StoredFieldVisitor;
import org.apache.lucene.index.Terms;
import org.apache.lucene.index.TermsEnum;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.search.AcceptDocs;
import org.apache.lucene.search.ByteVectorSimilarityQuery;
import org.apache.lucene.search.ConstantScoreScorer;
import org.apache.lucene.search.ConstantScoreWeight;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.FloatVectorSimilarityQuery;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnByteVectorQuery;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.PatienceKnnVectorQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.QueryVisitor;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.ScoreMode;
import org.apache.lucene.search.ScorerSupplier;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.Weight;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.BytesRef;
import org.apache.lucene.util.BitSetIterator;
import org.apache.lucene.util.FixedBitSet;

/**
 * Records, with Lucene 10.5.0 and backward-codecs, what an index written by an older Lucene
 * ({@code BwcWrite}) contains, as {@code expected.txt} beside it.
 *
 * <p>Structure (segment and field infos) is written out in full. Content is written as FNV-1a
 * 64-bit digests over a canonical byte stream, with counts, so a Rust reader can prove it read the
 * same thing without the fixture holding every posting. The stream: {@code L(x)} is {@code x} as
 * 8 little-endian bytes; {@code B(b)} is {@code L(len)} then the bytes. Per kind:
 *
 * <ul>
 *   <li>live: {@code L(1|0)} per doc of {@code 0..maxDoc}.
 *   <li>postings: per term in order, {@code B(term) L(docFreq) L(totalTermFreq)}, then per
 *       posting (deleted documents included) {@code L(doc) L(freq)} and, when the field has
 *       positions, per position {@code L(pos)}, {@code L(start) L(end)} when it has offsets, and
 *       {@code L(-1)} or {@code B(payload)} when it has payloads.
 *   <li>norms: {@code L(doc) L(norm)} per document with a norm.
 *   <li>dv: NUMERIC {@code L(doc) L(v)}; BINARY {@code L(doc) B(v)}; SORTED {@code L(doc) L(ord)
 *       B(term)}; SORTED_SET {@code L(doc) L(count)} then {@code L(ord) B(term)} each;
 *       SORTED_NUMERIC {@code L(doc) L(count)} then {@code L(v)} each.
 *   <li>points: every {@code (doc, packedValue)} the tree visits with every cell crossing, sorted
 *       by doc then value, as {@code L(doc) B(value)}.
 *   <li>stored: per doc of {@code 0..maxDoc}, {@code L(doc)}, then per stored field {@code B(name)}
 *       and {@code L(0) B(utf8)} for a string, {@code L(1) B(bytes)} binary, {@code L(2) L(int)},
 *       {@code L(3) L(long)}, {@code L(4) L(floatBits)}, {@code L(5) L(doubleBits)}.
 *   <li>tv: per doc, {@code L(-1)} for none, else per field {@code B(name)} and its terms as for
 *       postings of the one document ({@code B(term) L(freq)} then positions/offsets/payloads).
 *   <li>vec: float {@code L(doc)} then {@code L(floatBits)} per dimension; byte {@code L(doc) B(v)}.
 * </ul>
 *
 * <p>{@code knn} lines give the ten nearest live documents to a fixed query vector, with score bits
 * ({@code LeafReader.searchNearestVectors}). The query-level lines run through an {@code
 * IndexSearcher} over the one segment, so they take {@code AbstractKnnVectorQuery}'s and {@code
 * AbstractVectorSimilarityQuery}'s own paths (filter cost heuristics, exact fallbacks):
 *
 * <ul>
 *   <li>{@code knnf}: the ten nearest among documents {@code doc % 3 == 0} ({@link ModQuery}).
 *   <li>{@code knne}: the twenty nearest among {@code doc % 89 == 0} -- few enough that the search
 *       is exact, or falls back to exact.
 *   <li>{@code patience}: {@code PatienceKnnVectorQuery} (saturation 0.5, patience 2) of the ten
 *       nearest.
 *   <li>{@code vsim}/{@code vsimf}: {@code *VectorSimilarityQuery} with the fifth {@code knn} hit's
 *       score as {@code resultSimilarity} ({@code thr=}, float bits), unfiltered and filtered by
 *       {@code doc % 3 == 0}, every hit sorted by doc.
 * </ul>
 */
public class BwcDump {
  /** Matches the documents {@code doc % mod == 0} of each leaf (deleted ones included). */
  static final class ModQuery extends Query {
    final int mod;

    ModQuery(int mod) {
      this.mod = mod;
    }

    @Override
    public Weight createWeight(IndexSearcher searcher, ScoreMode scoreMode, float boost) {
      return new ConstantScoreWeight(this, boost) {
        @Override
        public ScorerSupplier scorerSupplier(LeafReaderContext ctx) {
          int maxDoc = ctx.reader().maxDoc();
          FixedBitSet bits = new FixedBitSet(maxDoc);
          for (int d = 0; d < maxDoc; d += mod) bits.set(d);
          return new DefaultScorerSupplier(
              new ConstantScoreScorer(score(), scoreMode, new BitSetIterator(bits, bits.cardinality())));
        }

        @Override
        public boolean isCacheable(LeafReaderContext ctx) {
          return false;
        }
      };
    }

    @Override
    public void visit(QueryVisitor visitor) {
      visitor.visitLeaf(this);
    }

    @Override
    public String toString(String field) {
      return "mod(" + mod + ")";
    }

    @Override
    public boolean equals(Object o) {
      return o instanceof ModQuery m && m.mod == mod;
    }

    @Override
    public int hashCode() {
      return mod;
    }
  }

  static String byDoc(TopDocs td) {
    ScoreDoc[] sd = td.scoreDocs.clone();
    Arrays.sort(sd, (a, b) -> Integer.compare(a.doc, b.doc));
    return hits(new TopDocs(td.totalHits, sd));
  }

  static void queryLines(StringBuilder out, LeafReader r, String seg, String field, Object q, TopDocs knn)
      throws IOException {
    IndexSearcher s = new IndexSearcher(r);
    s.setQueryCache(null);
    boolean isFloat = q instanceof float[];
    Query knnf = isFloat
        ? new KnnFloatVectorQuery(field, (float[]) q, 10, new ModQuery(3))
        : new KnnByteVectorQuery(field, (byte[]) q, 10, new ModQuery(3));
    Query knne = isFloat
        ? new KnnFloatVectorQuery(field, (float[]) q, 20, new ModQuery(89))
        : new KnnByteVectorQuery(field, (byte[]) q, 20, new ModQuery(89));
    Query patience = isFloat
        ? PatienceKnnVectorQuery.fromFloatQuery(new KnnFloatVectorQuery(field, (float[]) q, 10), 0.5, 2)
        : PatienceKnnVectorQuery.fromByteQuery(new KnnByteVectorQuery(field, (byte[]) q, 10), 0.5, 2);
    out.append("knnf ").append(seg).append(' ').append(field).append(' ').append(hits(s.search(knnf, 10))).append('\n');
    out.append("knne ").append(seg).append(' ').append(field).append(' ').append(hits(s.search(knne, 20))).append('\n');
    out.append("patience ").append(seg).append(' ').append(field).append(' ').append(hits(s.search(patience, 10))).append('\n');
    ScoreDoc[] sd = knn.scoreDocs;
    float thr = sd.length == 0 ? 0f : sd[Math.min(4, sd.length - 1)].score;
    int n = Math.max(1, r.maxDoc());
    for (boolean filtered : new boolean[] {false, true}) {
      Query filter = filtered ? new ModQuery(3) : null;
      Query vsim = isFloat
          ? new FloatVectorSimilarityQuery(field, (float[]) q, thr, filter)
          : new ByteVectorSimilarityQuery(field, (byte[]) q, thr, filter);
      out.append(filtered ? "vsimf " : "vsim ").append(seg).append(' ').append(field)
          .append(" thr=").append(Integer.toHexString(Float.floatToRawIntBits(thr))).append(' ')
          .append(byDoc(s.search(vsim, n))).append('\n');
    }
  }

  static final class Fnv {
    long h = 0xcbf29ce484222325L;
    long n;

    void bytes(byte[] b, int off, int len) {
      for (int i = off; i < off + len; i++) {
        h ^= (b[i] & 0xff);
        h *= 0x100000001b3L;
      }
    }

    void l(long v) {
      for (int i = 0; i < 8; i++) {
        h ^= (v >>> (8 * i)) & 0xff;
        h *= 0x100000001b3L;
      }
      n++;
    }

    void b(BytesRef r) {
      l(r.length);
      bytes(r.bytes, r.offset, r.length);
    }

    void b(byte[] v) {
      l(v.length);
      bytes(v, 0, v.length);
    }

    String hex() {
      return String.format("%016x", h);
    }
  }

  static void postingsOf(TermsEnum te, IndexOptions io, boolean payloads, Fnv f, boolean withStats)
      throws IOException {
    boolean pos = io.compareTo(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS) >= 0;
    boolean off = io.compareTo(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS_AND_OFFSETS) >= 0;
    PostingsEnum pe = null;
    for (BytesRef t = te.next(); t != null; t = te.next()) {
      f.b(t);
      if (withStats) {
        f.l(te.docFreq());
        f.l(te.totalTermFreq());
      }
      pe = te.postings(pe, pos ? PostingsEnum.ALL : PostingsEnum.FREQS);
      for (int d = pe.nextDoc(); d != DocIdSetIterator.NO_MORE_DOCS; d = pe.nextDoc()) {
        if (withStats) f.l(d);
        int freq = pe.freq();
        f.l(freq);
        if (!pos) continue;
        for (int k = 0; k < freq; k++) {
          f.l(pe.nextPosition());
          if (off) {
            f.l(pe.startOffset());
            f.l(pe.endOffset());
          }
          if (payloads) {
            BytesRef p = pe.getPayload();
            if (p == null) f.l(-1);
            else f.b(p);
          }
        }
      }
    }
  }

  public static void main(String[] args) throws Exception {
    Path index = Path.of(args[0]);
    StringBuilder out = new StringBuilder();
    try (Directory dir = FSDirectory.open(index)) {
      SegmentInfos infos = SegmentInfos.readLatestCommit(dir);
      out.append("commit gen=").append(infos.getGeneration())
          .append(" version=").append(infos.getCommitLuceneVersion())
          .append(" created=").append(infos.getIndexCreatedVersionMajor())
          .append(" min=").append(infos.getMinSegmentLuceneVersion()).append('\n');
      for (SegmentCommitInfo sci : infos) {
        out.append("seg ").append(sci.info.name)
            .append(" codec=").append(sci.info.getCodec().getName())
            .append(" maxDoc=").append(sci.info.maxDoc())
            .append(" delCount=").append(sci.getDelCount())
            .append(" softDel=").append(sci.getSoftDelCount())
            .append(" version=").append(sci.info.getVersion())
            .append(" minVersion=").append(sci.info.getMinVersion())
            .append(" compound=").append(sci.info.getUseCompoundFile())
            .append(" files=").append(new java.util.TreeSet<>(sci.files()))
            .append('\n');
      }
      try (DirectoryReader reader = DirectoryReader.open(dir)) {
        for (LeafReaderContext ctx : reader.leaves()) {
          LeafReader r = ctx.reader();
          String seg = ((SegmentReader) r).getSegmentName();
          int maxDoc = r.maxDoc();
          for (FieldInfo fi : r.getFieldInfos()) {
            Map<String, String> attrs = new TreeMap<>(fi.attributes());
            out.append("field ").append(seg).append(' ').append(fi.name)
                .append(" num=").append(fi.number)
                .append(" index=").append(fi.getIndexOptions())
                .append(" tv=").append(fi.hasTermVectors())
                .append(" norms=").append(fi.hasNorms())
                .append(" payloads=").append(fi.hasPayloads())
                .append(" dv=").append(fi.getDocValuesType())
                .append(" points=").append(fi.getPointDimensionCount()).append(',')
                .append(fi.getPointIndexDimensionCount()).append(',').append(fi.getPointNumBytes())
                .append(" vec=").append(fi.getVectorDimension()).append(',')
                .append(fi.getVectorEncoding()).append(',').append(fi.getVectorSimilarityFunction())
                .append(" attrs=").append(attrs).append('\n');
          }
          Bits live = r.getLiveDocs();
          Fnv lf = new Fnv();
          int numLive = 0;
          for (int d = 0; d < maxDoc; d++) {
            boolean alive = live == null || live.get(d);
            lf.l(alive ? 1 : 0);
            if (alive) numLive++;
          }
          out.append("live ").append(seg).append(" n=").append(numLive).append(' ').append(lf.hex()).append('\n');

          for (FieldInfo fi : r.getFieldInfos()) {
            String field = fi.name;
            if (fi.getIndexOptions() != IndexOptions.NONE) {
              Terms terms = r.terms(field);
              Fnv f = new Fnv();
              postingsOf(terms.iterator(), fi.getIndexOptions(), fi.hasPayloads(), f, true);
              out.append("postings ").append(seg).append(' ').append(field)
                  .append(" terms=").append(terms.size())
                  .append(" sumDocFreq=").append(terms.getSumDocFreq())
                  .append(" sumTTF=").append(terms.getSumTotalTermFreq())
                  .append(" docCount=").append(terms.getDocCount())
                  .append(" min=").append(terms.getMin().utf8ToString())
                  .append(" max=").append(terms.getMax().utf8ToString())
                  .append(' ').append(f.hex()).append('\n');
            }
            if (fi.hasNorms()) {
              NumericDocValues norms = r.getNormValues(field);
              Fnv f = new Fnv();
              for (int d = norms.nextDoc(); d != DocIdSetIterator.NO_MORE_DOCS; d = norms.nextDoc()) {
                f.l(d);
                f.l(norms.longValue());
              }
              out.append("norms ").append(seg).append(' ').append(field).append(' ').append(f.hex()).append('\n');
            }
            if (fi.getDocValuesType() != DocValuesType.NONE) {
              Fnv f = new Fnv();
              switch (fi.getDocValuesType()) {
                case NUMERIC -> {
                  NumericDocValues v = r.getNumericDocValues(field);
                  for (int d = v.nextDoc(); d != DocIdSetIterator.NO_MORE_DOCS; d = v.nextDoc()) {
                    f.l(d);
                    f.l(v.longValue());
                  }
                }
                case BINARY -> {
                  BinaryDocValues v = r.getBinaryDocValues(field);
                  for (int d = v.nextDoc(); d != DocIdSetIterator.NO_MORE_DOCS; d = v.nextDoc()) {
                    f.l(d);
                    f.b(v.binaryValue());
                  }
                }
                case SORTED -> {
                  SortedDocValues v = r.getSortedDocValues(field);
                  f.l(v.getValueCount());
                  for (int d = v.nextDoc(); d != DocIdSetIterator.NO_MORE_DOCS; d = v.nextDoc()) {
                    f.l(d);
                    f.l(v.ordValue());
                    f.b(v.lookupOrd(v.ordValue()));
                  }
                }
                case SORTED_SET -> {
                  SortedSetDocValues v = r.getSortedSetDocValues(field);
                  f.l(v.getValueCount());
                  for (int d = v.nextDoc(); d != DocIdSetIterator.NO_MORE_DOCS; d = v.nextDoc()) {
                    f.l(d);
                    int c = v.docValueCount();
                    f.l(c);
                    for (int k = 0; k < c; k++) {
                      long ord = v.nextOrd();
                      f.l(ord);
                      f.b(v.lookupOrd(ord));
                    }
                  }
                }
                case SORTED_NUMERIC -> {
                  SortedNumericDocValues v = r.getSortedNumericDocValues(field);
                  for (int d = v.nextDoc(); d != DocIdSetIterator.NO_MORE_DOCS; d = v.nextDoc()) {
                    f.l(d);
                    int c = v.docValueCount();
                    f.l(c);
                    for (int k = 0; k < c; k++) f.l(v.nextValue());
                  }
                }
                default -> {}
              }
              out.append("dv ").append(seg).append(' ').append(field).append(' ')
                  .append(fi.getDocValuesType()).append(' ').append(f.hex()).append('\n');
            }
            if (fi.getPointDimensionCount() > 0) {
              PointValues pv = r.getPointValues(field);
              List<Object[]> seen = new ArrayList<>();
              pv.intersect(new PointValues.IntersectVisitor() {
                @Override public void visit(int docID) { throw new IllegalStateException(); }
                @Override public void visit(int docID, byte[] packed) { seen.add(new Object[] {docID, packed.clone()}); }
                @Override public PointValues.Relation compare(byte[] min, byte[] max) {
                  return PointValues.Relation.CELL_CROSSES_QUERY;
                }
              });
              seen.sort((a, b) -> {
                int c = Integer.compare((Integer) a[0], (Integer) b[0]);
                return c != 0 ? c : Arrays.compareUnsigned((byte[]) a[1], (byte[]) b[1]);
              });
              Fnv f = new Fnv();
              for (Object[] p : seen) {
                f.l((Integer) p[0]);
                f.b((byte[]) p[1]);
              }
              out.append("points ").append(seg).append(' ').append(field)
                  .append(" size=").append(pv.size()).append(" docCount=").append(pv.getDocCount())
                  .append(" min=").append(hex(pv.getMinPackedValue()))
                  .append(" max=").append(hex(pv.getMaxPackedValue()))
                  .append(' ').append(f.hex()).append('\n');
            }
            if (fi.getVectorDimension() > 0) {
              Fnv f = new Fnv();
              int count = 0;
              if (fi.getVectorEncoding() == VectorEncoding.FLOAT32) {
                FloatVectorValues v = r.getFloatVectorValues(field);
                KnnVectorValues.DocIndexIterator it = v.iterator();
                for (int d = it.nextDoc(); d != DocIdSetIterator.NO_MORE_DOCS; d = it.nextDoc()) {
                  f.l(d);
                  for (float x : v.vectorValue(it.index())) f.l(Float.floatToRawIntBits(x));
                  count++;
                }
                float[] q = new float[fi.getVectorDimension()];
                for (int k = 0; k < q.length; k++) q[k] = (float) Math.sin(k + 1);
                TopDocs td = r.searchNearestVectors(field, q, 10, AcceptDocs.fromLiveDocs(live, maxDoc), Integer.MAX_VALUE);
                out.append("knn ").append(seg).append(' ').append(field).append(' ').append(hits(td)).append('\n');
                queryLines(out, r, seg, field, q, td);
              } else {
                ByteVectorValues v = r.getByteVectorValues(field);
                KnnVectorValues.DocIndexIterator it = v.iterator();
                for (int d = it.nextDoc(); d != DocIdSetIterator.NO_MORE_DOCS; d = it.nextDoc()) {
                  f.l(d);
                  f.b(v.vectorValue(it.index()));
                  count++;
                }
                byte[] q = new byte[fi.getVectorDimension()];
                for (int k = 0; k < q.length; k++) q[k] = (byte) (k * 37 - 100);
                TopDocs td = r.searchNearestVectors(field, q, 10, AcceptDocs.fromLiveDocs(live, maxDoc), Integer.MAX_VALUE);
                out.append("knn ").append(seg).append(' ').append(field).append(' ').append(hits(td)).append('\n');
                queryLines(out, r, seg, field, q, td);
              }
              out.append("vec ").append(seg).append(' ').append(field).append(" n=").append(count)
                  .append(' ').append(f.hex()).append('\n');
            }
          }

          Fnv sf = new Fnv();
          var stored = r.storedFields();
          for (int d = 0; d < maxDoc; d++) {
            sf.l(d);
            stored.document(d, new StoredFieldVisitor() {
              @Override public Status needsField(FieldInfo fi) { return Status.YES; }
              @Override public void stringField(FieldInfo fi, String v) { sf.b(fi.name.getBytes(StandardCharsets.UTF_8)); sf.l(0); sf.b(v.getBytes(StandardCharsets.UTF_8)); }
              @Override public void binaryField(FieldInfo fi, byte[] v) { sf.b(fi.name.getBytes(StandardCharsets.UTF_8)); sf.l(1); sf.b(v); }
              @Override public void intField(FieldInfo fi, int v) { sf.b(fi.name.getBytes(StandardCharsets.UTF_8)); sf.l(2); sf.l(v); }
              @Override public void longField(FieldInfo fi, long v) { sf.b(fi.name.getBytes(StandardCharsets.UTF_8)); sf.l(3); sf.l(v); }
              @Override public void floatField(FieldInfo fi, float v) { sf.b(fi.name.getBytes(StandardCharsets.UTF_8)); sf.l(4); sf.l(Float.floatToRawIntBits(v)); }
              @Override public void doubleField(FieldInfo fi, double v) { sf.b(fi.name.getBytes(StandardCharsets.UTF_8)); sf.l(5); sf.l(Double.doubleToRawLongBits(v)); }
            });
          }
          out.append("stored ").append(seg).append(' ').append(sf.hex()).append('\n');

          Fnv tf = new Fnv();
          var tvs = r.termVectors();
          for (int d = 0; d < maxDoc; d++) {
            Fields fields = tvs.get(d);
            if (fields == null) {
              tf.l(-1);
              continue;
            }
            for (String name : fields) {
              tf.b(name.getBytes(StandardCharsets.UTF_8));
              Terms t = fields.terms(name);
              IndexOptions io = t.hasOffsets()
                  ? IndexOptions.DOCS_AND_FREQS_AND_POSITIONS_AND_OFFSETS
                  : t.hasPositions() ? IndexOptions.DOCS_AND_FREQS_AND_POSITIONS : IndexOptions.DOCS_AND_FREQS;
              postingsOf(t.iterator(), io, t.hasPayloads(), tf, false);
            }
          }
          out.append("tv ").append(seg).append(' ').append(tf.hex()).append('\n');
        }
      }
    }
    Files.writeString(index.resolve("expected.txt"), out);
  }

  static String hits(TopDocs td) {
    StringBuilder sb = new StringBuilder();
    for (ScoreDoc sd : td.scoreDocs) {
      if (sb.length() > 0) sb.append(',');
      sb.append(sd.doc).append(':').append(Integer.toHexString(Float.floatToRawIntBits(sd.score)));
    }
    return sb.toString();
  }

  static String hex(byte[] b) {
    StringBuilder sb = new StringBuilder();
    for (byte x : b) sb.append(String.format("%02x", x & 0xff));
    return sb.toString();
  }
}
