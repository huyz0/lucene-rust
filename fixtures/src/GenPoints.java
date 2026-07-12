import org.apache.lucene.document.Document;
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
  public static void main(String[] args) throws IOException {
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
