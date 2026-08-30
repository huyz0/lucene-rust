import org.apache.lucene.codecs.lucene99.Lucene99SegmentInfoFormat;
import org.apache.lucene.index.IndexFileNames;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.search.Sort;
import org.apache.lucene.search.SortField;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.StringHelper;
import org.apache.lucene.util.Version;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.TreeSet;

/**
 * Generates a `.si` (Lucene99SegmentInfoFormat) fixture: a real SegmentInfo written via
 * the actual codec, plus a manifest of every field so Rust can assert without parsing
 * Java. Two variants: with and without a minVersion, to exercise both branches.
 */
public class GenSegmentInfo {
  public static void main(String[] args) throws IOException {
    Path out = Path.of(args[0]);
    Files.createDirectories(out);

    gen(out, "_0", true);
    gen(out, "_1", false);
    genSorted(out, "_2");
  }

  /**
   * An index-sorted segment: the `.si` variant whose sort-field bytes are written by
   * `SortFieldProvider` (provider name + that provider's own bytestream), which is a
   * completely different layout from the rest of the file. Two sort fields, so the
   * fixture also pins the multi-field priority order, and a mix of
   * reverse/missing-first/missing-last so neither flag can be silently dropped.
   */
  static void genSorted(Path out, String segmentName) throws IOException {
    byte[] id = StringHelper.randomId();
    Version version = Version.LUCENE_10_0_0;

    SortField primary = new SortField("timestamp", SortField.Type.LONG, true);
    primary.setMissingValue(Long.MAX_VALUE); // missing sorts last
    SortField secondary = new SortField("price", SortField.Type.LONG, false);
    secondary.setMissingValue(Long.MIN_VALUE); // missing sorts first
    Sort indexSort = new Sort(primary, secondary);

    SegmentInfo si =
        new SegmentInfo(
            new ByteBuffersDirectory(),
            version,
            version,
            segmentName,
            100,
            false,
            false,
            null,
            new LinkedHashMap<>(),
            id,
            new LinkedHashMap<>(),
            indexSort);
    si.setFiles(new TreeSet<>());

    Directory dir = new ByteBuffersDirectory();
    Lucene99SegmentInfoFormat format = new Lucene99SegmentInfoFormat();
    format.write(dir, si, IOContext.DEFAULT);

    SegmentInfo readBack = format.read(dir, segmentName, id, IOContext.DEFAULT);
    if (!indexSort.equals(readBack.getIndexSort())) throw new AssertionError("sort round-trip mismatch");

    String fileName = IndexFileNames.segmentFileName(segmentName, "", "si");
    try (IndexInput in = dir.openInput(fileName, IOContext.DEFAULT)) {
      byte[] bytes = new byte[(int) in.length()];
      in.readBytes(bytes, 0, bytes.length);
      Files.write(out.resolve(segmentName + ".si"), bytes);
    }

    StringBuilder m = new StringBuilder();
    m.append("segment_name=").append(segmentName).append('\n');
    m.append("id_hex=").append(hex(id)).append('\n');
    m.append("version_major=").append(version.major).append('\n');
    m.append("version_minor=").append(version.minor).append('\n');
    m.append("version_bugfix=").append(version.bugfix).append('\n');
    m.append("has_min_version=1\n");
    m.append("min_version_major=").append(version.major).append('\n');
    m.append("min_version_minor=").append(version.minor).append('\n');
    m.append("min_version_bugfix=").append(version.bugfix).append('\n');
    m.append("doc_count=").append(si.maxDoc()).append('\n');
    m.append("is_compound_file=0\n");
    m.append("has_blocks=0\n");
    m.append("diagnostics=\n");
    m.append("attributes=\n");
    m.append("files=\n");
    // `field:reverse:missing` triples in priority order, the same rendering
    // `crates/lucene-index/examples/write_segment_info_fixture.rs` emits.
    m.append("index_sort=timestamp:1:last,price:0:first\n");
    Files.writeString(out.resolve(segmentName + ".manifest.properties"), m.toString());

    System.out.println("wrote " + segmentName + ".si (" + Files.size(out.resolve(segmentName + ".si")) + " bytes)");
  }

  static void gen(Path out, String segmentName, boolean withMinVersion) throws IOException {
    byte[] id = StringHelper.randomId();
    Version version = Version.LUCENE_10_0_0;
    Version minVersion = withMinVersion ? Version.LUCENE_9_12_0 : null;

    Map<String, String> diagnostics = new LinkedHashMap<>();
    diagnostics.put("source", "flush");
    diagnostics.put("lucene.version", version.toString());
    diagnostics.put("os", "Linux");

    Map<String, String> attributes = new LinkedHashMap<>();
    attributes.put("Lucene90StoredFieldsFormat.mode", "BEST_SPEED");

    // Files referred to by this segment must be prefixed with the segment name
    // (IndexFileNames.parseSegmentName is enforced by the writer).
    TreeSet<String> files = new TreeSet<>();
    files.add(segmentName + ".fdt");
    files.add(segmentName + ".fdx");
    files.add(segmentName + "_1.doc");

    SegmentInfo si =
        new SegmentInfo(
            new ByteBuffersDirectory(), // placeholder; not persisted, only used for asserts
            version,
            minVersion,
            segmentName,
            12345,
            true, // isCompoundFile
            false, // hasBlocks
            null,
            diagnostics,
            id,
            attributes,
            null); // no index sort
    si.setFiles(files);

    Directory dir = new ByteBuffersDirectory();
    Lucene99SegmentInfoFormat format = new Lucene99SegmentInfoFormat();
    format.write(dir, si, IOContext.DEFAULT);

    // sanity round-trip through Lucene itself before shipping the fixture
    SegmentInfo readBack = format.read(dir, segmentName, id, IOContext.DEFAULT);
    if (readBack.maxDoc() != si.maxDoc()) throw new AssertionError("round-trip mismatch");

    String fileName = IndexFileNames.segmentFileName(segmentName, "", "si");
    try (IndexInput in = dir.openInput(fileName, IOContext.DEFAULT)) {
      byte[] bytes = new byte[(int) in.length()];
      in.readBytes(bytes, 0, bytes.length);
      Files.write(out.resolve(segmentName + ".si"), bytes);
    }

    StringBuilder m = new StringBuilder();
    m.append("segment_name=").append(segmentName).append('\n');
    m.append("id_hex=").append(hex(id)).append('\n');
    m.append("version_major=").append(version.major).append('\n');
    m.append("version_minor=").append(version.minor).append('\n');
    m.append("version_bugfix=").append(version.bugfix).append('\n');
    m.append("has_min_version=").append(withMinVersion ? 1 : 0).append('\n');
    if (withMinVersion) {
      m.append("min_version_major=").append(minVersion.major).append('\n');
      m.append("min_version_minor=").append(minVersion.minor).append('\n');
      m.append("min_version_bugfix=").append(minVersion.bugfix).append('\n');
    }
    m.append("doc_count=").append(si.maxDoc()).append('\n');
    m.append("is_compound_file=1\n");
    m.append("has_blocks=0\n");
    m.append("diagnostics=").append(joinMap(diagnostics)).append('\n');
    m.append("attributes=").append(joinMap(attributes)).append('\n');
    m.append("files=").append(String.join(",", files)).append('\n');
    Files.writeString(out.resolve(segmentName + ".manifest.properties"), m.toString());

    System.out.println("wrote " + segmentName + ".si (" + Files.size(out.resolve(segmentName + ".si")) + " bytes)");
  }

  static String joinMap(Map<String, String> m) {
    StringBuilder sb = new StringBuilder();
    boolean first = true;
    for (Map.Entry<String, String> e : m.entrySet()) {
      if (!first) sb.append(';');
      first = false;
      sb.append(e.getKey()).append('=').append(e.getValue());
    }
    return sb.toString();
  }

  static String hex(byte[] b) {
    StringBuilder sb = new StringBuilder();
    for (byte x : b) sb.append(String.format("%02x", x));
    return sb.toString();
  }
}
