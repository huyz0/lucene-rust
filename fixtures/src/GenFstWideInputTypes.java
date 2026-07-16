import org.apache.lucene.util.BytesRef;
import org.apache.lucene.util.IntsRef;
import org.apache.lucene.util.fst.ByteSequenceOutputs;
import org.apache.lucene.util.fst.FST;
import org.apache.lucene.util.fst.FSTCompiler;
import org.apache.lucene.util.fst.Util;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HexFormat;
import java.util.List;
import java.util.TreeMap;

/**
 * Generates two real on-heap FSTs whose {@code FST.INPUT_TYPE} is {@code
 * BYTE2} and {@code BYTE4} respectively -- unlike every other {@code GenFst*}
 * generator in this directory, which builds {@code BYTE1} (raw term bytes)
 * FSTs for the BlockTree term index. {@code FSTCompiler.Builder}'s public API
 * takes an {@code INPUT_TYPE} directly, and {@code FSTCompiler.add} takes an
 * {@link IntsRef} (not a {@link BytesRef}), so building a genuinely
 * wider-than-byte-alphabet FST needs no non-public API at all -- each key's
 * "characters" are just {@code int} label values devised directly rather
 * than encoded from a {@link BytesRef} via {@code Util.toIntsRef}, so the
 * label values can legitimately exceed 255 (up to 0xFFFF for {@code BYTE2},
 * matching a UTF-16 code unit's range; up to 0x10FFFF for {@code BYTE4},
 * matching a Unicode code point's range), which is the entire point of this
 * fixture: proving {@code lucene-codecs/src/fst.rs}'s {@code read_label}
 * decodes labels outside a raw byte's range.
 *
 * <p>{@code allowFixedLengthArcs(false)} is passed for both builds, matching
 * {@code GenFst.java}'s own choice, so only the (already-ported) list-encoded
 * node format is exercised here -- this fixture is about the arc *label*
 * width, not the node encoding, which is an orthogonal axis already covered
 * by {@code GenFstBinarySearch}/{@code GenFstDirectAddressing}/
 * {@code GenFstContinuous}.
 */
public class GenFstWideInputTypes {
  public static void main(String[] args) throws IOException {
    Path root = Path.of(args[0]);
    genByte2(root);
    genByte4(root);
  }

  private static void genByte2(Path root) throws IOException {
    Path out = root.resolve("fst_byte2");
    Files.createDirectories(out);

    // Keys are int-label sequences using UTF-16-code-unit-range values
    // (including some > 255, which a raw byte could never hold), sharing
    // prefixes/suffixes the same way GenFst.java's byte keys do.
    TreeMap<List<Integer>, String> entries = new TreeMap<>(GenFstWideInputTypes::compareLabels);
    entries.put(List.of(0x41, 0x300), "1"); // "A" + combining grave accent
    entries.put(List.of(0x41, 0x300, 0x42), "2");
    entries.put(List.of(0x41, 0x301), "3"); // "A" + combining acute accent
    entries.put(List.of(0x4e2d, 0x6587), "4"); // CJK: "中文"
    entries.put(List.of(0x4e2d, 0x6587, 0xff01), "5"); // + fullwidth "!"
    entries.put(List.of(0xffff), "6"); // max BYTE2 label value

    build(out, FST.INPUT_TYPE.BYTE2, entries);
  }

  private static void genByte4(Path root) throws IOException {
    Path out = root.resolve("fst_byte4");
    Files.createDirectories(out);

    // Keys are int-label sequences using full Unicode-code-point-range
    // values, including several past 0xFFFF (BYTE2's own ceiling) and up to
    // 0x10FFFF (the last valid code point) -- only representable at all
    // because BYTE4 labels are full ints, not bytes or even UTF-16 units.
    TreeMap<List<Integer>, String> entries = new TreeMap<>(GenFstWideInputTypes::compareLabels);
    entries.put(List.of(0x1f600), "1"); // U+1F600 GRINNING FACE
    entries.put(List.of(0x1f600, 0x1f601), "2");
    entries.put(List.of(0x1f602), "3");
    entries.put(List.of(0x10000), "4"); // first supplementary-plane code point
    entries.put(List.of(0x10000, 0x41), "5");
    entries.put(List.of(0x10ffff), "6"); // last valid Unicode code point

    build(out, FST.INPUT_TYPE.BYTE4, entries);
  }

  private static int compareLabels(List<Integer> a, List<Integer> b) {
    int n = Math.min(a.size(), b.size());
    for (int i = 0; i < n; i++) {
      int cmp = Integer.compare(a.get(i), b.get(i));
      if (cmp != 0) {
        return cmp;
      }
    }
    return Integer.compare(a.size(), b.size());
  }

  private static void build(Path out, FST.INPUT_TYPE inputType, TreeMap<List<Integer>, String> entries)
      throws IOException {
    ByteSequenceOutputs outputs = ByteSequenceOutputs.getSingleton();
    FSTCompiler<BytesRef> fstCompiler =
        new FSTCompiler.Builder<>(inputType, outputs).allowFixedLengthArcs(false).build();

    for (var e : entries.entrySet()) {
      IntsRef input = toIntsRef(e.getKey());
      BytesRef output = new BytesRef(e.getValue());
      fstCompiler.add(input, output);
    }
    FST.FSTMetadata<BytesRef> metadata = fstCompiler.compile();
    FST<BytesRef> fst = FST.fromFSTReader(metadata, fstCompiler.getFSTReader());

    Path fstFile = out.resolve("fst.bin");
    fst.save(fstFile);

    // Round-trip through Java Lucene itself before shipping the fixture.
    FST<BytesRef> reloaded = FST.read(fstFile, outputs);
    for (var e : entries.entrySet()) {
      BytesRef got = Util.get(reloaded, toIntsRef(e.getKey()));
      if (got == null || !got.utf8ToString().equals(e.getValue())) {
        throw new AssertionError(
            "self-check failed for key=" + e.getKey() + " got=" + got + " want=" + e.getValue());
      }
    }

    // A handful of absent label sequences that share no prefix with any
    // present key, mirroring GenFst.java's absent-key coverage.
    List<List<Integer>> absentKeys =
        List.of(List.of(0x20), List.of(0x39, 0x39, 0x39));
    for (List<Integer> k : absentKeys) {
      BytesRef got = Util.get(reloaded, toIntsRef(k));
      if (got != null) {
        throw new AssertionError("expected key=" + k + " to be absent, got=" + got);
      }
    }

    StringBuilder m = new StringBuilder();
    m.append("input_type=").append(inputType).append('\n');
    m.append("num_present=").append(entries.size()).append('\n');
    int i = 0;
    for (var e : entries.entrySet()) {
      m.append("present.").append(i).append(".key=").append(joinLabels(e.getKey())).append('\n');
      m.append("present.")
          .append(i)
          .append(".output_hex=")
          .append(hex(e.getValue()))
          .append('\n');
      i++;
    }
    m.append("num_absent=").append(absentKeys.size()).append('\n');
    for (int j = 0; j < absentKeys.size(); j++) {
      m.append("absent.").append(j).append(".key=").append(joinLabels(absentKeys.get(j))).append('\n');
    }
    Files.writeString(out.resolve("manifest.properties"), m.toString());

    System.out.println(inputType + " FST fixture written to " + out);
  }

  private static IntsRef toIntsRef(List<Integer> labels) {
    IntsRef ref = new IntsRef(labels.size());
    for (int label : labels) {
      ref.ints[ref.length++] = label;
    }
    return ref;
  }

  private static String joinLabels(List<Integer> labels) {
    StringBuilder sb = new StringBuilder();
    for (int i = 0; i < labels.size(); i++) {
      if (i > 0) {
        sb.append(',');
      }
      sb.append(labels.get(i));
    }
    return sb.toString();
  }

  private static String hex(String s) {
    return HexFormat.of().formatHex(s.getBytes(java.nio.charset.StandardCharsets.UTF_8));
  }
}
