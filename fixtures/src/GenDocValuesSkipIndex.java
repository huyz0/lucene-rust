import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.NumericDocValuesField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.SegmentCommitInfo;
import org.apache.lucene.index.SegmentInfos;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Generates a real `.dvm`/`.dvd`/`.dvs` (Lucene90DocValuesFormat) fixture
 * for a NUMERIC field with a doc-values skip index
 * (`NumericDocValuesField.indexedField`, which sets
 * `DocValuesSkipIndexType.RANGE` -- see
 * `Lucene90DocValuesFormat.DEFAULT_SKIP_INDEX_INTERVAL_SIZE` / 4096 and
 * `SKIP_INDEX_LEVEL_SHIFT` / `SKIP_INDEX_MAX_LEVEL` in the real source).
 *
 * <p>Writes {@link #NUM_DOCS} docs (comfortably more than the 4096-doc
 * base-interval size, so the skip index gets a second base interval and at
 * least a 2-level structure) with strictly increasing values, forces a
 * single merged segment, then reads the field's `DocValuesSkipperEntry`
 * back out of `.dvm` and manually walks the raw `.dvs` bytes at
 * `[offset, offset + length)` -- the exact on-disk shape
 * `Lucene90DocValuesConsumer.writeLevels` produces -- dumping every
 * interval's every level to the manifest for the Rust differential test to
 * compare against.
 */
public class GenDocValuesSkipIndex {
  // >= 8 * 4096 base intervals so at least one level-1 group forms
  // (SKIP_INDEX_LEVEL_SHIFT == 3 -> groups of 8), giving the fixture a
  // genuinely multi-level structure instead of only ever level-1 rows.
  static final int NUM_DOCS = 36000;

  public static void main(String[] args) throws IOException {
    Path out = Path.of(args[0]).resolve("doc_values_skip_index");
    if (Files.exists(out)) {
      deleteRecursive(out);
    }
    Files.createDirectories(out);

    try (Directory dir = FSDirectory.open(out)) {
      IndexWriterConfig cfg = new IndexWriterConfig();
      cfg.setUseCompoundFile(false);
      cfg.setMergePolicy(NoMergePolicy.INSTANCE);

      try (IndexWriter w = new IndexWriter(dir, cfg)) {
        for (int i = 0; i < NUM_DOCS; i++) {
          Document doc = new Document();
          doc.add(new StringField("id", Integer.toString(i), Field.Store.NO));
          // Strictly increasing, occasionally-jumping values so min/max per
          // interval are non-trivial and never collapse to a single value.
          long value = (long) i * 7 - 3;
          doc.add(NumericDocValuesField.indexedField("skip_numeric", value));
          w.addDocument(doc);
        }
        w.commit();
        w.forceMerge(1);
      }

      SegmentInfos sis = SegmentInfos.readLatestCommit(dir);
      if (sis.size() != 1) {
        throw new AssertionError("expected exactly one segment, got " + sis.size());
      }
      SegmentCommitInfo sci = sis.info(0);

      String dvmFileName = null;
      String dvdFileName = null;
      String dvsFileName = null;
      String fnmFileName = null;
      for (String f : sci.files()) {
        if (f.endsWith(".dvm")) dvmFileName = f;
        if (f.endsWith(".dvd")) dvdFileName = f;
        if (f.endsWith(".dvs")) dvsFileName = f;
        if (f.endsWith(".fnm")) fnmFileName = f;
      }
      if (dvmFileName == null || dvdFileName == null || dvsFileName == null || fnmFileName == null) {
        throw new AssertionError("expected .dvm/.dvd/.dvs/.fnm files, files=" + sci.files());
      }

      dump(dir, dvmFileName, out);
      dump(dir, dvdFileName, out);
      dump(dir, dvsFileName, out);
      dump(dir, fnmFileName, out);

      org.apache.lucene.index.FieldInfos fis =
          sci.info.getCodec().fieldInfosFormat().read(dir, sci.info, "", IOContext.READONCE);
      org.apache.lucene.index.FieldInfo field = fis.fieldInfo("skip_numeric");

      StringBuilder m = new StringBuilder();
      m.append("dvm_file_name=").append(dvmFileName).append('\n');
      m.append("dvd_file_name=").append(dvdFileName).append('\n');
      m.append("dvs_file_name=").append(dvsFileName).append('\n');
      m.append("fnm_file_name=").append(fnmFileName).append('\n');
      m.append("segment_name=").append(sci.info.name).append('\n');
      m.append("id_hex=").append(hex(sci.info.getId())).append('\n');
      m.append("max_doc=").append(sci.info.maxDoc()).append('\n');
      m.append("field_number=").append(field.number).append('\n');

      // Read the DocValuesSkipperEntry straight out of the .dvm bytes we
      // already dumped, using the exact field order
      // Lucene90DocValuesProducer.readDocValueSkipperMeta reads: fieldNumber
      // (i32) + type byte, then offset/length/maxValue/minValue/docCount/
      // maxDocId/maxValueCount, immediately followed by the NUMERIC entry
      // body for this field (the only field in this segment, so there's no
      // ambiguity about which record it is).
      byte[] dvmBytes = Files.readAllBytes(out.resolve(dvmFileName + ".raw"));
      long[] skipper = readSkipperMetaFromDvm(dvmBytes, field.number);
      long skipOffset = skipper[0];
      long skipLength = skipper[1];
      long skipMaxValue = skipper[2];
      long skipMinValue = skipper[3];
      long skipDocCount = skipper[4];
      long skipMaxDocId = skipper[5];
      long skipMaxValueCount = skipper[6];

      m.append("skip.offset=").append(skipOffset).append('\n');
      m.append("skip.length=").append(skipLength).append('\n');
      m.append("skip.max_value=").append(skipMaxValue).append('\n');
      m.append("skip.min_value=").append(skipMinValue).append('\n');
      m.append("skip.doc_count=").append(skipDocCount).append('\n');
      m.append("skip.max_doc_id=").append(skipMaxDocId).append('\n');
      m.append("skip.max_value_count=").append(skipMaxValueCount).append('\n');

      // Now manually walk the raw .dvs bytes at [offset, offset+length),
      // one interval at a time, exactly mirroring
      // Lucene90DocValuesConsumer.writeLevels's on-disk shape: a level-count
      // byte, then per level (coarsest-first): maxDocID, minDocID, maxValue,
      // minValue, docCount.
      byte[] dvsBytes = Files.readAllBytes(out.resolve(dvsFileName + ".raw"));
      StringBuilder intervals = new StringBuilder();
      int pos = (int) skipOffset;
      int end = (int) (skipOffset + skipLength);
      int intervalCount = 0;
      while (pos < end) {
        int levels = dvsBytes[pos] & 0xFF;
        pos += 1;
        if (intervalCount > 0) intervals.append(';');
        intervals.append(levels);
        // Levels are written coarsest-first on disk; collect them, then
        // emit level0..levelN-1 (ascending) into the manifest to match this
        // port's `SkipIndexInterval.levels` ordering.
        long[][] rows = new long[levels][5];
        for (int level = levels - 1; level >= 0; level--) {
          long maxDocId = readIntBE_LE(dvsBytes, pos);
          pos += 4;
          long minDocId = readIntBE_LE(dvsBytes, pos);
          pos += 4;
          long maxValue = readLongLE(dvsBytes, pos);
          pos += 8;
          long minValue = readLongLE(dvsBytes, pos);
          pos += 8;
          long docCount = readIntBE_LE(dvsBytes, pos);
          pos += 4;
          rows[level] = new long[] {minDocId, maxDocId, minValue, maxValue, docCount};
        }
        for (int level = 0; level < levels; level++) {
          intervals
              .append(',')
              .append(rows[level][0])
              .append(':')
              .append(rows[level][1])
              .append(':')
              .append(rows[level][2])
              .append(':')
              .append(rows[level][3])
              .append(':')
              .append(rows[level][4]);
        }
        intervalCount++;
      }
      m.append("skip.interval_count=").append(intervalCount).append('\n');
      m.append("skip.intervals=").append(intervals).append('\n');

      Files.writeString(out.resolve("manifest.properties"), m.toString());
    }

    System.out.println("wrote doc_values_skip_index/ fixture directory");
  }

  /**
   * Reads exactly one field's record out of a `.dvm` byte array: skips the
   * codec header, then scans `fieldNumber:i32, type:u8, [skipper fields if
   * this field has a skip index]` entries until it finds `wantFieldNumber`.
   * Field-number/int/long fields are little-endian
   * ({@code DataOutput.writeInt/writeLong}); this only supports the single
   * NUMERIC-with-skip-index shape this generator itself writes.
   */
  static long[] readSkipperMetaFromDvm(byte[] b, int wantFieldNumber) {
    // Header: magic(4) + codec string (vint len + bytes) + version(4) +
    // id(16) + suffix (vint len + bytes, empty here -> 1 byte).
    int pos = 4;
    int codecLen = b[pos] & 0xFF;
    pos += 1 + codecLen;
    pos += 4; // version
    pos += 16; // id
    int suffixLen = b[pos] & 0xFF;
    pos += 1 + suffixLen;

    while (true) {
      int fieldNumber = (int) readIntBE_LE(b, pos);
      pos += 4;
      if (fieldNumber == -1) {
        throw new AssertionError("field " + wantFieldNumber + " not found in .dvm");
      }
      int type = b[pos] & 0xFF;
      pos += 1;
      long offset = readLongLE(b, pos);
      long length = readLongLE(b, pos + 8);
      long maxValue = readLongLE(b, pos + 16);
      long minValue = readLongLE(b, pos + 24);
      long docCount = readIntBE_LE(b, pos + 32);
      long maxDocId = readIntBE_LE(b, pos + 36);
      long maxValueCount = readIntBE_LE(b, pos + 40);
      int afterSkipper = pos + 44;
      if (fieldNumber == wantFieldNumber) {
        return new long[] {offset, length, maxValue, minValue, docCount, maxDocId, maxValueCount};
      }
      // This generator only ever writes one field, so we never need to
      // skip past a NUMERIC entry body to find the next field record.
      throw new AssertionError(
          "unexpected extra field " + fieldNumber + " (type " + type + ") before " + wantFieldNumber
              + "; afterSkipper=" + afterSkipper);
    }
  }

  static long readIntBE_LE(byte[] b, int pos) {
    // DataOutput.writeInt is little-endian in this codebase's port; real
    // Lucene's IndexOutput.writeInt is also little-endian internally
    // (Lucene stores ints LE apart from the BE codec header magics).
    return (b[pos] & 0xFFL)
        | ((b[pos + 1] & 0xFFL) << 8)
        | ((b[pos + 2] & 0xFFL) << 16)
        | ((long) b[pos + 3] << 24);
  }

  static long readLongLE(byte[] b, int pos) {
    long v = 0;
    for (int i = 0; i < 8; i++) {
      v |= (b[pos + i] & 0xFFL) << (8 * i);
    }
    return v;
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
    StringBuilder sb = new StringBuilder();
    for (byte x : b) sb.append(String.format("%02x", x));
    return sb.toString();
  }
}
