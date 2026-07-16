import org.apache.lucene.document.Document;
import org.apache.lucene.document.IntPoint;
import org.apache.lucene.document.LongPoint;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.PointValues;
import org.apache.lucene.index.SegmentCommitInfo;
import org.apache.lucene.index.SegmentInfos;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.NumericUtils;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HexFormat;
import java.util.List;

/**
 * Generates real `.kdm`/`.kdi`/`.kdd` (Lucene90PointsFormat / BKD tree)
 * fixtures: 2000 documents, most carrying a single-dimension `LongPoint`
 * ("val", a distinct value per doc) but every third doc skipping the field
 * entirely -- enough points to force several leaves (default
 * maxPointsInLeafNode=512) and non-continuous doc ids within each leaf, so
 * the doc-id encoding isn't trivially CONTINUOUS_IDS. Real Lucene picks
 * whichever `DocIdsWriter` encoding fits; the differential test just checks
 * our decode against Lucene's own `PointValues.intersect` output regardless
 * of which encoding was chosen for each leaf.
 */
public class GenPoints {
  public static void main(String[] args) throws Exception {
    Path out = Path.of(args[0]).resolve("points_index");
    if (Files.exists(out)) {
      deleteRecursive(out);
    }
    Files.createDirectories(out);

    int numDocs = 2000;
    try (Directory dir = FSDirectory.open(out)) {
      IndexWriterConfig cfg = new IndexWriterConfig();
      cfg.setUseCompoundFile(false);
      cfg.setMergePolicy(NoMergePolicy.INSTANCE);

      try (IndexWriter w = new IndexWriter(dir, cfg)) {
        for (int i = 0; i < numDocs; i++) {
          Document doc = new Document();
          doc.add(new StringField("id", Integer.toString(i), org.apache.lucene.document.Field.Store.NO));
          if (i % 3 != 0) {
            long value = (long) i * 7919L - 1_000_000L; // spread across pos/neg range
            doc.add(new LongPoint("val", value));
          }
          // Second, 2-dimension field, engineered so real Lucene's
          // BKDWriter picks dim1 (not dim0) as `sortedDim`/`compressedDim`
          // for at least one leaf -- see BKDWriter.build's per-leaf scan
          // (`usedBytes[dim].cardinality()`), which picks whichever
          // dimension has the fewest distinct byte values *at the leaf's
          // common-prefix offset* as `sortedDim`, ties going to the
          // lowest-numbered dimension. A naive "dim0 = i" (sequential, only
          // 0..2000) turned out to be the wrong shape for that: BKDWriter's
          // top-down build recursively splits leaves into ever-narrower
          // *contiguous* ranges of whichever dim is widest, and since dim0
          // was by far the widest, most splits happened on dim0 -- which
          // squeezes each leaf's dim0 range down to only 1-2 distinct bytes
          // at the differing position, at or below dim1's fixed cardinality
          // (i % 4, so <=4), so dim0 kept winning ties every time.
          //
          // dim0 here is instead `i` run through an odd (thus bijective mod
          // 2^32) multiplicative hash, spreading consecutive doc ids across
          // the *entire* 32-bit space essentially at random. BKDWriter still
          // narrows dim0's range by splitting on it, but halving a ~2^32-wide
          // range a few times barely dents the low-order byte's entropy, so
          // dim0's in-leaf cardinality at the differing byte stays close to
          // its true ceiling (min(leafSize, 256)) -- always well above
          // dim1's <=4, guaranteeing dim1 wins the `sortedDim` selection.
          // dim0 remains a bijection of `i`, so every (dim0, dim1) tuple
          // stays unique per doc (count == leafCardinality), which keeps the
          // high-cardinality encoding cheaper than the low-cardinality
          // (`-2`) one, so real Lucene writes `compressedDim == sortedDim`
          // (nonzero) rather than `-2` -- exercising the "real dimension
          // index > 0" branch of the leaf decoder (`compressed_byte_offset =
          // compressed_dim * bytes_per_dim + ...` with a nonzero
          // `compressed_dim`) that the single-dimension `val` field above
          // can never reach. `CompressedDimSpy` (see that file) mechanically
          // confirms this by reading the raw `compressedDim` byte back out
          // of the written leaves, independent of this repo's Rust decoder.
          int dim0 = (int) (i * 2654435761L); // Knuth multiplicative hash, odd multiplier
          doc.add(new IntPoint("multi", dim0, i % 4));
          w.addDocument(doc);
        }
        w.commit();
      }

      SegmentInfos sis = SegmentInfos.readLatestCommit(dir);
      if (sis.size() != 1) {
        throw new AssertionError("expected exactly one segment, got " + sis.size());
      }
      SegmentCommitInfo sci = sis.info(0);

      String kdmFileName = null;
      String kdiFileName = null;
      String kddFileName = null;
      for (String f : sci.info.files()) {
        if (f.endsWith(".kdm")) kdmFileName = f;
        if (f.endsWith(".kdi")) kdiFileName = f;
        if (f.endsWith(".kdd")) kddFileName = f;
      }
      if (kdmFileName == null || kdiFileName == null || kddFileName == null) {
        throw new AssertionError("expected .kdm/.kdi/.kdd files, files=" + sci.info.files());
      }

      dump(dir, kdmFileName, out);
      dump(dir, kdiFileName, out);
      dump(dir, kddFileName, out);

      StringBuilder m = new StringBuilder();
      m.append("kdm_file_name=").append(kdmFileName).append('\n');
      m.append("kdi_file_name=").append(kdiFileName).append('\n');
      m.append("kdd_file_name=").append(kddFileName).append('\n');
      m.append("segment_name=").append(sci.info.name).append('\n');
      m.append("id_hex=").append(hex(sci.info.getId())).append('\n');
      m.append("max_doc=").append(sci.info.maxDoc()).append('\n');

      FieldInfo fieldInfo =
          sci.info
              .getCodec()
              .fieldInfosFormat()
              .read(dir, sci.info, "", IOContext.READONCE)
              .fieldInfo("val");
      m.append("field_number=").append(fieldInfo.number).append('\n');

      org.apache.lucene.codecs.PointsReader pointsReader =
          sci.info.getCodec().pointsFormat().fieldsReader(
              new org.apache.lucene.index.SegmentReadState(
                  dir,
                  sci.info,
                  sci.info.getCodec().fieldInfosFormat().read(dir, sci.info, "", IOContext.READONCE),
                  IOContext.READONCE));

      PointValues values = pointsReader.getValues("val");
      m.append("num_dims=").append(values.getNumDimensions()).append('\n');
      m.append("num_index_dims=").append(values.getNumIndexDimensions()).append('\n');
      m.append("bytes_per_dim=").append(values.getBytesPerDimension()).append('\n');
      m.append("point_count=").append(values.size()).append('\n');
      m.append("doc_count=").append(values.getDocCount()).append('\n');

      List<long[]> collected = new ArrayList<>();
      values.intersect(
          new PointValues.IntersectVisitor() {
            @Override
            public void visit(int docID) {
              throw new AssertionError("should not be called: compare always returns CROSSES");
            }

            @Override
            public void visit(int docID, byte[] packedValue) {
              long decoded = NumericUtils.sortableBytesToLong(packedValue, 0);
              collected.add(new long[] {docID, decoded});
            }

            @Override
            public PointValues.Relation compare(byte[] minPackedValue, byte[] maxPackedValue) {
              return PointValues.Relation.CELL_CROSSES_QUERY;
            }
          });

      // Sort by docID for a stable, easy-to-check manifest ordering (our
      // Rust decode preserves leaf/in-leaf order, which is already
      // doc-id-ascending per leaf but not necessarily globally -- the test
      // itself will sort both sides the same way before comparing).
      collected.sort((a, b) -> Long.compare(a[0], b[0]));
      StringBuilder points = new StringBuilder();
      for (long[] entry : collected) {
        if (points.length() > 0) points.append(';');
        points.append(entry[0]).append(':').append(entry[1]);
      }
      m.append("points=").append(points).append('\n');

      FieldInfo multiFieldInfo =
          sci.info
              .getCodec()
              .fieldInfosFormat()
              .read(dir, sci.info, "", IOContext.READONCE)
              .fieldInfo("multi");
      m.append("multi_field_number=").append(multiFieldInfo.number).append('\n');

      PointValues multiValues = pointsReader.getValues("multi");
      m.append("multi_num_dims=").append(multiValues.getNumDimensions()).append('\n');
      m.append("multi_num_index_dims=").append(multiValues.getNumIndexDimensions()).append('\n');
      m.append("multi_bytes_per_dim=").append(multiValues.getBytesPerDimension()).append('\n');
      m.append("multi_point_count=").append(multiValues.size()).append('\n');
      m.append("multi_doc_count=").append(multiValues.getDocCount()).append('\n');

      List<int[]> multiCollected = new ArrayList<>();
      multiValues.intersect(
          new PointValues.IntersectVisitor() {
            @Override
            public void visit(int docID) {
              throw new AssertionError("should not be called: compare always returns CROSSES");
            }

            @Override
            public void visit(int docID, byte[] packedValue) {
              int dim0 = NumericUtils.sortableBytesToInt(packedValue, 0);
              int dim1 = NumericUtils.sortableBytesToInt(packedValue, Integer.BYTES);
              multiCollected.add(new int[] {docID, dim0, dim1});
            }

            @Override
            public PointValues.Relation compare(byte[] minPackedValue, byte[] maxPackedValue) {
              return PointValues.Relation.CELL_CROSSES_QUERY;
            }
          });
      multiCollected.sort((a, b) -> Integer.compare(a[0], b[0]));
      StringBuilder multiPoints = new StringBuilder();
      for (int[] entry : multiCollected) {
        if (multiPoints.length() > 0) multiPoints.append(';');
        multiPoints.append(entry[0]).append(':').append(entry[1]).append(':').append(entry[2]);
      }
      m.append("multi_points=").append(multiPoints).append('\n');

      // Mechanically verify (independent of this repo's Rust decoder) that
      // real Lucene actually wrote a nonzero `compressedDim` for at least
      // one leaf of the "multi" field -- see CompressedDimSpy's doc comment.
      if (!(multiValues instanceof org.apache.lucene.util.bkd.BKDReader multiBkdReader)) {
        throw new AssertionError(
            "expected \"multi\" field's PointValues to be a BKDReader, got "
                + multiValues.getClass());
      }
      int[] leafCompressedDims =
          org.apache.lucene.util.bkd.CompressedDimSpy.leafCompressedDims(multiBkdReader);
      boolean sawNonZeroCompressedDim = false;
      StringBuilder leafCompressedDimsStr = new StringBuilder();
      for (int cd : leafCompressedDims) {
        if (leafCompressedDimsStr.length() > 0) leafCompressedDimsStr.append(',');
        leafCompressedDimsStr.append(cd);
        if (cd >= 1) sawNonZeroCompressedDim = true;
      }
      if (!sawNonZeroCompressedDim) {
        throw new AssertionError(
            "expected at least one leaf of the \"multi\" field to have compressedDim >= 1, "
                + "but every leaf's compressedDim was in {-2,-1,0}: "
                + leafCompressedDimsStr);
      }
      m.append("multi_leaf_compressed_dims=").append(leafCompressedDimsStr).append('\n');

      pointsReader.close();

      Files.writeString(out.resolve("manifest.properties"), m.toString());
    }

    System.out.println("wrote points_index/ fixture directory");
  }

  static void dump(Directory dir, String fileName, Path out) throws IOException {
    try (IndexInput in = dir.openInput(fileName, IOContext.READONCE)) {
      byte[] bytes = new byte[(int) in.length()];
      in.readBytes(bytes, 0, bytes.length);
      Files.write(out.resolve(fileName + ".raw"), bytes);
    }
  }

  static void deleteRecursive(Path p) throws IOException {
    if (Files.isDirectory(p)) {
      try (var entries = Files.list(p)) {
        for (Path child : (Iterable<Path>) entries::iterator) {
          deleteRecursive(child);
        }
      }
    }
    Files.deleteIfExists(p);
  }

  static String hex(byte[] b) {
    return HexFormat.of().formatHex(b);
  }
}
