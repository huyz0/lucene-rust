import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HexFormat;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.apache.lucene.codecs.simpletext.SimpleTextSegmentInfoFormat;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.search.BinarySortField;
import org.apache.lucene.search.Sort;
import org.apache.lucene.search.SortField;
import org.apache.lucene.search.SortedNumericSelector;
import org.apache.lucene.search.SortedNumericSortField;
import org.apache.lucene.search.SortedSetSelector;
import org.apache.lucene.search.SortedSetSortField;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.util.Version;

/** Real SimpleText segment infos covering all four index-sort providers. */
public class GenSimpleTextIndexSort {
  static final HexFormat HEX = HexFormat.of();

  static Object[] missing(SortField.Type type) {
    return switch (type) {
      case INT -> new Object[] {null, Integer.MIN_VALUE, 17, Integer.MAX_VALUE};
      case LONG -> new Object[] {null, Long.MIN_VALUE, 42L, Long.MAX_VALUE};
      case FLOAT -> new Object[] {null, -0.0f, 0.0f, -Float.MAX_VALUE, 1.0e-20f,
          Float.NEGATIVE_INFINITY, Float.POSITIVE_INFINITY, Float.NaN};
      case DOUBLE -> new Object[] {null, -0.0d, 0.0d, -Double.MAX_VALUE, 1.0e-200d,
          Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY, Double.NaN};
      default -> throw new AssertionError(type);
    };
  }

  static List<Sort> sorts() {
    var sorts = new ArrayList<Sort>();
    sorts.add(null);
    for (boolean reverse : new boolean[] {false, true}) {
      for (var type : new SortField.Type[] {SortField.Type.INT, SortField.Type.LONG,
          SortField.Type.FLOAT, SortField.Type.DOUBLE}) {
        for (Object missing : missing(type)) {
          var plain = new SortField("plain\\\n\u00e9", type, reverse);
          if (missing != null) plain.setMissingValue(missing);
          sorts.add(new Sort(plain));
          for (var selector : SortedNumericSelector.Type.values()) {
            var numeric = new SortedNumericSortField("multi", type, reverse, selector);
            if (missing != null) numeric.setMissingValue(missing);
            sorts.add(new Sort(numeric));
          }
        }
      }
      for (Object missing : new Object[] {null, SortField.STRING_FIRST, SortField.STRING_LAST}) {
        var plain = new SortField("text", SortField.Type.STRING, reverse);
        if (missing != null) plain.setMissingValue(missing);
        sorts.add(new Sort(plain));
        var binary = new BinarySortField("binary", reverse, missing);
        sorts.add(new Sort(binary));
        for (var selector : SortedSetSelector.Type.values()) {
          var set = new SortedSetSortField("set", reverse, selector);
          if (missing != null) set.setMissingValue(missing);
          sorts.add(new Sort(set));
        }
      }
    }
    // A priority-ordered sort spanning all four provider implementations.
    sorts.add(new Sort(new SortField("first", SortField.Type.LONG, true),
        new SortedNumericSortField("second", SortField.Type.FLOAT, false,
            SortedNumericSelector.Type.MAX),
        new SortedSetSortField("third", true, SortedSetSelector.Type.MIDDLE_MAX),
        new BinarySortField("fourth", false)));
    // SimpleText retains Version's prerelease component; binary .si does not.
    sorts.add(null);
    sorts.add(null);
    return sorts;
  }

  public static void main(String[] args) throws Exception {
    Path out = Path.of(args[0]).resolve("simpletext_index_sort");
    Files.createDirectories(out);
    var manifest = new ArrayList<String>();
    var format = new SimpleTextSegmentInfoFormat();
    int ordinal = 0;
    var sorts = sorts();
    for (Sort sort : sorts) {
      String name = "_" + Integer.toString(ordinal, 36);
      byte[] id = new byte[16];
      id[0] = (byte) ordinal;
      id[1] = (byte) (ordinal >>> 8);
      try (var dir = new ByteBuffersDirectory()) {
        Version version = ordinal >= sorts.size() - 2
            ? Version.parse("10.0.0." + (ordinal - sorts.size() + 3)) : Version.LATEST;
        Version minVersion = ordinal % 2 != 0 ? null : ordinal >= sorts.size() - 2
            ? Version.parse("9.0.0.1") : Version.LUCENE_10_0_0;
        var info = new SegmentInfo(dir, version, minVersion,
            name, 7, ordinal % 2 == 0, ordinal % 3 == 0, null,
            Map.of("source", "fixture"), id, Map.of("note", "line\nbreak\\"), sort);
        info.setFiles(Set.of(name + ".other"));
        format.write(dir, info, IOContext.DEFAULT);
        var read = format.read(dir, name, id, IOContext.DEFAULT);
        if (!java.util.Objects.equals(sort, read.getIndexSort()))
          throw new AssertionError("sort round-trip: " + sort);
        String file = name + ".si";
        try (var input = dir.openInput(file, IOContext.DEFAULT)) {
          byte[] bytes = new byte[Math.toIntExact(input.length())];
          input.readBytes(bytes, 0, bytes.length);
          Files.write(out.resolve(file), bytes);
        }
        String description = sort == null ? "<none>" : read.getIndexSort().toString();
        manifest.add(name + "\t" + HEX.formatHex(id) + "\t"
            + HEX.formatHex(description.getBytes(java.nio.charset.StandardCharsets.UTF_8))
            + "\t" + read.getVersion() + "\t" + read.getMinVersion());
      }
      ordinal++;
    }
    Files.write(out.resolve("cases.tsv"), manifest);
    System.out.println("SimpleText index sort: " + ordinal + " cases");
  }
}
