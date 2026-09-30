import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HexFormat;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.TreeSet;
import org.apache.lucene.document.Document;
import org.apache.lucene.index.BinaryDocValues;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.CheckIndex;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.Fields;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.IndexableField;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.NumericDocValues;
import org.apache.lucene.index.PointValues;
import org.apache.lucene.index.PostingsEnum;
import org.apache.lucene.index.SegmentCommitInfo;
import org.apache.lucene.index.SegmentInfos;
import org.apache.lucene.index.SortedDocValues;
import org.apache.lucene.index.SortedNumericDocValues;
import org.apache.lucene.index.SortedSetDocValues;
import org.apache.lucene.index.Terms;
import org.apache.lucene.index.TermsEnum;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.BytesRef;

/**
 * M8 T8.4's Java half: real Lucene 10.5.0 (with backward-codecs) checks an index this port merged
 * out of an older Lucene's segments.
 *
 * <pre>java BwcMergeCheck &lt;original-index&gt; &lt;merged-index&gt;</pre>
 *
 * <ol>
 *   <li>{@code CheckIndex} on the merged index must be clean.
 *   <li>Every merged segment must be written by the current codec: codec {@code Lucene104}, every
 *       postings field {@code Lucene104}, every vector field {@code Lucene99HnswVectorsFormat}.
 *   <li>The merged index must hold exactly the original's live documents, in the original's order,
 *       with the same content: for every field, the postings (docs, freqs, positions, offsets,
 *       payloads), norms, doc values of all five types, points, term vectors, stored fields and
 *       vectors, each compared as a SHA-256 digest over the documents renumbered by their rank
 *       among the live ones (a merge drops the deleted documents and packs the rest).
 * </ol>
 *
 * Prints one line per kind and exits non-zero on any difference.
 */
public class BwcMergeCheck {
  static final class Digest {
    final MessageDigest md;

    Digest() throws Exception {
      md = MessageDigest.getInstance("SHA-256");
    }

    void l(long v) {
      for (int i = 0; i < 8; i++) md.update((byte) (v >>> (8 * i)));
    }

    void b(BytesRef r) {
      if (r == null) {
        l(-1);
        return;
      }
      l(r.length);
      md.update(r.bytes, r.offset, r.length);
    }

    void s(String s) {
      b(new BytesRef(s.getBytes(StandardCharsets.UTF_8)));
    }

    String hex() {
      return HexFormat.of().formatHex(md.digest());
    }
  }

  /** One live document: its leaf and leaf-local id. */
  record Live(LeafReader leaf, int doc) {}

  static List<Live> live(DirectoryReader r) {
    List<Live> out = new ArrayList<>();
    for (LeafReaderContext ctx : r.leaves()) {
      Bits liveDocs = ctx.reader().getLiveDocs();
      for (int d = 0; d < ctx.reader().maxDoc(); d++) {
        if (liveDocs == null || liveDocs.get(d)) out.add(new Live(ctx.reader(), d));
      }
    }
    return out;
  }

  /** The rank of every live doc: `ranks[leafOrd][doc]`, -1 for a deleted one. */
  static int[][] ranks(DirectoryReader r) {
    int[][] out = new int[r.leaves().size()][];
    int next = 0;
    for (LeafReaderContext ctx : r.leaves()) {
      Bits liveDocs = ctx.reader().getLiveDocs();
      int[] m = new int[ctx.reader().maxDoc()];
      for (int d = 0; d < m.length; d++) m[d] = liveDocs == null || liveDocs.get(d) ? next++ : -1;
      out[ctx.ord] = m;
    }
    return out;
  }

  static TreeMap<String, FieldInfo> fields(DirectoryReader r) {
    TreeMap<String, FieldInfo> out = new TreeMap<>();
    for (LeafReaderContext ctx : r.leaves()) {
      for (FieldInfo fi : ctx.reader().getFieldInfos()) out.putIfAbsent(fi.name, fi);
    }
    return out;
  }

  /** A live document's stored fields, as one string: its identity across the two indices. */
  static String storedKey(LeafReader leaf, int doc) throws Exception {
    StringBuilder sb = new StringBuilder();
    for (IndexableField f : leaf.storedFields().document(doc).getFields()) {
      sb.append(f.name()).append('=');
      if (f.binaryValue() != null) sb.append(HexFormat.of().formatHex(BytesRef.deepCopyOf(f.binaryValue()).bytes));
      else if (f.numericValue() != null) sb.append(f.numericValue().getClass().getSimpleName()).append(f.numericValue());
      else sb.append(f.stringValue());
      sb.append('\u0001');
    }
    return sb.toString();
  }

  /**
   * Renumbers {@code orig}'s live documents by where the same document (same stored fields) sits in
   * {@code merged}: a merge may concatenate its sources in any order (this port's {@code
   * force_merge} takes the smallest first; {@code TieredMergePolicy} the largest), and what must
   * hold is that each document kept all of its content, whatever its new number.
   */
  static int[][] ranksByStoredFields(DirectoryReader orig, DirectoryReader merged) throws Exception {
    Map<String, Integer> where = new java.util.HashMap<>();
    List<Live> mergedLive = live(merged);
    for (int i = 0; i < mergedLive.size(); i++) {
      if (where.put(storedKey(mergedLive.get(i).leaf(), mergedLive.get(i).doc()), i) != null) {
        throw new IllegalStateException("two merged documents have the same stored fields");
      }
    }
    int[][] out = new int[orig.leaves().size()][];
    for (LeafReaderContext ctx : orig.leaves()) {
      Bits liveDocs = ctx.reader().getLiveDocs();
      int[] m = new int[ctx.reader().maxDoc()];
      for (int d = 0; d < m.length; d++) {
        if (liveDocs == null || liveDocs.get(d)) {
          Integer at = where.remove(storedKey(ctx.reader(), d));
          if (at == null) throw new IllegalStateException("original doc " + ctx.ord + "/" + d + " is missing from the merged index");
          m[d] = at;
        } else {
          m[d] = -1;
        }
      }
      out[ctx.ord] = m;
    }
    if (!where.isEmpty()) throw new IllegalStateException(where.size() + " merged documents are not in the original");
    return out;
  }

  static Map<String, String> describe(DirectoryReader r, int[][] ranks) throws Exception {
    TreeMap<String, String> out = new TreeMap<>();
    List<Live> live = new ArrayList<>();
    for (Live l : live(r)) live.add(null);
    for (LeafReaderContext ctx : r.leaves()) {
      for (int d = 0; d < ranks[ctx.ord].length; d++) {
        if (ranks[ctx.ord][d] >= 0) live.set(ranks[ctx.ord][d], new Live(ctx.reader(), d));
      }
    }
    out.put("docs", Integer.toString(live.size()));
    for (FieldInfo fi : fields(r).values()) {
      String f = fi.name;
      if (fi.getIndexOptions() != IndexOptions.NONE) {
        TreeSet<BytesRef> terms = new TreeSet<>();
        for (LeafReaderContext ctx : r.leaves()) {
          Terms t = ctx.reader().terms(f);
          if (t == null) continue;
          TermsEnum te = t.iterator();
          for (BytesRef term = te.next(); term != null; term = te.next()) terms.add(BytesRef.deepCopyOf(term));
        }
        Digest d = new Digest();
        long termCount = 0;
        for (BytesRef term : terms) {
          List<String> one = new ArrayList<>();
          for (LeafReaderContext ctx : r.leaves()) {
            Terms t = ctx.reader().terms(f);
            if (t == null) continue;
            TermsEnum te = t.iterator();
            if (!te.seekExact(term)) continue;
            PostingsEnum pe = te.postings(null, PostingsEnum.ALL);
            for (int doc = pe.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS; doc = pe.nextDoc()) {
              int rank = ranks[ctx.ord][doc];
              if (rank < 0) continue;
              int freq = pe.freq();
              StringBuilder p = new StringBuilder(String.format("%010d:%d", rank, freq));
              if (fi.getIndexOptions().compareTo(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS) >= 0) {
                for (int i = 0; i < freq; i++) {
                  p.append(',').append(pe.nextPosition());
                  p.append('/').append(pe.startOffset()).append('/').append(pe.endOffset());
                  BytesRef pay = pe.getPayload();
                  p.append('/').append(pay == null ? "-" : HexFormat.of().formatHex(BytesRef.deepCopyOf(pay).bytes));
                }
              }
              one.add(p.toString());
            }
          }
          // A term whose every posting was deleted does not survive a merge.
          if (!one.isEmpty()) {
            one.sort(null);
            termCount++;
            d.b(term);
            for (String p : one) d.s(p);
          }
        }
        out.put("postings " + f, termCount + " " + d.hex());
      }
      Digest norms = new Digest();
      Digest dv = new Digest();
      Digest pts = new Digest();
      Digest vec = new Digest();
      for (int i = 0; i < live.size(); i++) {
        LeafReader leaf = live.get(i).leaf();
        int doc = live.get(i).doc();
        FieldInfo lfi = leaf.getFieldInfos().fieldInfo(f);
        if (lfi == null) continue;
        if (lfi.hasNorms()) {
          NumericDocValues n = leaf.getNormValues(f);
          if (n != null && n.advanceExact(doc)) {
            norms.l(i);
            norms.l(n.longValue());
          }
        }
        switch (lfi.getDocValuesType()) {
          case NUMERIC -> {
            NumericDocValues v = leaf.getNumericDocValues(f);
            if (v.advanceExact(doc)) {
              dv.l(i);
              dv.l(v.longValue());
            }
          }
          case BINARY -> {
            BinaryDocValues v = leaf.getBinaryDocValues(f);
            if (v.advanceExact(doc)) {
              dv.l(i);
              dv.b(v.binaryValue());
            }
          }
          case SORTED -> {
            SortedDocValues v = leaf.getSortedDocValues(f);
            if (v.advanceExact(doc)) {
              dv.l(i);
              dv.b(v.lookupOrd(v.ordValue()));
            }
          }
          case SORTED_NUMERIC -> {
            SortedNumericDocValues v = leaf.getSortedNumericDocValues(f);
            if (v.advanceExact(doc)) {
              dv.l(i);
              int c = v.docValueCount();
              dv.l(c);
              for (int k = 0; k < c; k++) dv.l(v.nextValue());
            }
          }
          case SORTED_SET -> {
            SortedSetDocValues v = leaf.getSortedSetDocValues(f);
            if (v.advanceExact(doc)) {
              dv.l(i);
              int c = v.docValueCount();
              dv.l(c);
              for (int k = 0; k < c; k++) dv.b(v.lookupOrd(v.nextOrd()));
            }
          }
          default -> {}
        }
        if (lfi.getVectorDimension() > 0) {
          if (lfi.getVectorEncoding() == VectorEncoding.FLOAT32) {
            FloatVectorValues v = leaf.getFloatVectorValues(f);
            KnnVectorValues.DocIndexIterator it = v.iterator();
            if (it.advance(doc) == doc) {
              vec.l(i);
              for (float x : v.vectorValue(it.index())) vec.l(Float.floatToRawIntBits(x));
            }
          } else {
            ByteVectorValues v = leaf.getByteVectorValues(f);
            KnnVectorValues.DocIndexIterator it = v.iterator();
            if (it.advance(doc) == doc) {
              vec.l(i);
              vec.b(new BytesRef(v.vectorValue(it.index())));
            }
          }
        }
      }
      if (fi.hasNorms()) out.put("norms " + f, norms.hex());
      if (fi.getDocValuesType() != org.apache.lucene.index.DocValuesType.NONE) out.put("dv " + f, dv.hex());
      if (fi.getVectorDimension() > 0) out.put("vec " + f, vec.hex());
      if (fi.getPointDimensionCount() > 0) {
        List<String> seen = new ArrayList<>();
        for (LeafReaderContext ctx : r.leaves()) {
          PointValues pv = ctx.reader().getPointValues(f);
          if (pv == null) continue;
          int[] m = ranks[ctx.ord];
          pv.intersect(
              new PointValues.IntersectVisitor() {
                @Override
                public void visit(int docID) {
                  throw new UnsupportedOperationException();
                }

                @Override
                public void visit(int docID, byte[] packed) {
                  if (m[docID] >= 0) {
                    seen.add(String.format("%010d:%s", m[docID], HexFormat.of().formatHex(packed)));
                  }
                }

                @Override
                public PointValues.Relation compare(byte[] min, byte[] max) {
                  return PointValues.Relation.CELL_CROSSES_QUERY;
                }
              });
        }
        seen.sort(null);
        for (String s : seen) pts.s(s);
        out.put("points " + f, seen.size() + " " + pts.hex());
      }
    }
    Digest stored = new Digest();
    Digest tv = new Digest();
    for (int i = 0; i < live.size(); i++) {
      LeafReader leaf = live.get(i).leaf();
      int doc = live.get(i).doc();
      Document d = leaf.storedFields().document(doc);
      stored.l(i);
      for (IndexableField f : d.getFields()) {
        stored.s(f.name());
        if (f.binaryValue() != null) stored.b(f.binaryValue());
        else if (f.numericValue() != null) stored.s(f.numericValue().getClass().getSimpleName() + f.numericValue());
        else stored.s(f.stringValue());
      }
      Fields vectors = leaf.termVectors().get(doc);
      if (vectors == null) continue;
      tv.l(i);
      for (String f : vectors) {
        tv.s(f);
        TermsEnum te = vectors.terms(f).iterator();
        for (BytesRef term = te.next(); term != null; term = te.next()) {
          tv.b(term);
          PostingsEnum pe = te.postings(null, PostingsEnum.ALL);
          pe.nextDoc();
          int freq = pe.freq();
          tv.l(freq);
          for (int k = 0; k < freq; k++) {
            tv.l(pe.nextPosition());
            tv.l(pe.startOffset());
            tv.l(pe.endOffset());
            tv.b(pe.getPayload());
          }
        }
      }
    }
    out.put("stored", stored.hex());
    out.put("tv", tv.hex());
    return out;
  }

  public static void main(String[] args) throws Exception {
    int failures = 0;
    try (Directory orig = FSDirectory.open(Path.of(args[0]));
        Directory merged = FSDirectory.open(Path.of(args[1]))) {
      try (CheckIndex ci = new CheckIndex(merged)) {
        CheckIndex.Status st = ci.checkIndex();
        System.out.println("checkindex clean=" + st.clean + " segments=" + st.segmentInfos.size());
        if (!st.clean) failures++;
      }
      // A segment the merge did not touch (same name and id as in the original) keeps its old
      // codec; every segment the merge wrote must be the current one.
      java.util.Set<String> untouched = new java.util.HashSet<>();
      for (SegmentCommitInfo sci : SegmentInfos.readLatestCommit(orig)) {
        untouched.add(sci.info.name + "/" + HexFormat.of().formatHex(sci.info.getId()));
      }
      SegmentInfos infos = SegmentInfos.readLatestCommit(merged);
      java.util.Set<String> written = new java.util.HashSet<>();
      for (SegmentCommitInfo sci : infos) {
        String codec = sci.info.getCodec().getName();
        boolean kept = untouched.contains(sci.info.name + "/" + HexFormat.of().formatHex(sci.info.getId()));
        System.out.println("segment " + sci.info.name + " codec=" + codec + " maxDoc=" + sci.info.maxDoc() + " del=" + sci.getDelCount() + (kept ? " (not merged)" : " (merged)"));
        if (!kept) {
          written.add(sci.info.name);
          if (!codec.equals("Lucene104")) failures++;
        }
      }
      if (written.isEmpty()) {
        System.out.println("FAIL no segment was merged");
        failures++;
      }
      try (DirectoryReader mr = DirectoryReader.open(merged);
          DirectoryReader or = DirectoryReader.open(orig)) {
        for (LeafReaderContext ctx : mr.leaves()) {
          String name = org.apache.lucene.index.SegmentReader.class.cast(ctx.reader()).getSegmentName();
          if (!written.contains(name)) continue;
          for (FieldInfo fi : ctx.reader().getFieldInfos()) {
            String pf = fi.getAttribute("PerFieldPostingsFormat.format");
            String vf = fi.getAttribute("PerFieldKnnVectorsFormat.format");
            if (pf != null && !pf.equals("Lucene104")) {
              System.out.println("FAIL field " + fi.name + " postings format " + pf);
              failures++;
            }
            if (vf != null && !vf.equals("Lucene99HnswVectorsFormat")) {
              System.out.println("FAIL field " + fi.name + " vectors format " + vf);
              failures++;
            }
          }
        }
        int[][] mapped;
        try {
          mapped = ranksByStoredFields(or, mr);
        } catch (IllegalStateException e) {
          System.out.println("FAIL documents: " + e.getMessage());
          System.out.println("BwcMergeCheck: FAILURES");
          System.exit(1);
          return;
        }
        Map<String, String> want = describe(or, mapped);
        Map<String, String> got = describe(mr, ranks(mr));
        TreeSet<String> keys = new TreeSet<>(want.keySet());
        keys.addAll(got.keySet());
        for (String k : keys) {
          String w = want.get(k);
          String g = got.get(k);
          if (w != null && w.equals(g)) {
            System.out.println("ok   " + k + " " + w);
          } else {
            System.out.println("FAIL " + k + "\n  original " + w + "\n  merged   " + g);
            failures++;
          }
        }
      }
    }
    System.out.println(failures == 0 ? "BwcMergeCheck: PASS" : "BwcMergeCheck: " + failures + " FAILURES");
    System.exit(failures == 0 ? 0 : 1);
  }
}
