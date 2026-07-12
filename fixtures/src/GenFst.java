import org.apache.lucene.util.BytesRef;
import org.apache.lucene.util.IntsRef;
import org.apache.lucene.util.IntsRefBuilder;
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
 * Generates a real on-heap {@code FST<BytesRef>} (the same output type real
 * Lucene's {@code Lucene90BlockTreeTermsReader} term index uses --
 * {@link ByteSequenceOutputs}), built via the actual {@link FSTCompiler}, and
 * dumps its raw bytes (metadata + body, exactly as {@link FST#save(Path)}
 * writes them) plus a manifest of every lookup this fixture exercises so the
 * Rust differential test can assert against real Lucene's own
 * {@link Util#get(FST, BytesRef)} results without parsing Java.
 *
 * <p>Keys deliberately share prefixes/suffixes ("app"/"apple"/"application",
 * "band"/"banana"/"bandana") so arc sharing (both prefix compression within
 * a single lookup path, and suffix sharing across different keys) is
 * actually exercised, not just a single trivial entry.
 *
 * <p>{@code allowFixedLengthArcs(false)} is passed so the compiler never
 * emits fixed-length-arc (binary search / direct addressing / continuous)
 * nodes -- this port's reader only supports variable-length ("list") arc
 * nodes in this slice (see {@code lucene-codecs/src/fst.rs}'s module docs),
 * and a handful of keys wouldn't trigger those encodings anyway, but
 * disabling them explicitly documents and guarantees the fixture stays in
 * scope for what's actually been ported.
 */
public class GenFst {
  public static void main(String[] args) throws IOException {
    Path out = Path.of(args[0]).resolve("fst");
    Files.createDirectories(out);

    // Sorted map: FSTCompiler requires keys added in sorted order.
    TreeMap<String, String> entries = new TreeMap<>();
    entries.put("app", "1");
    entries.put("apple", "2");
    entries.put("application", "3");
    entries.put("banana", "4");
    entries.put("band", "5");
    entries.put("bandana", "6");
    entries.put("z", "26");

    ByteSequenceOutputs outputs = ByteSequenceOutputs.getSingleton();
    FSTCompiler<BytesRef> fstCompiler =
        new FSTCompiler.Builder<>(FST.INPUT_TYPE.BYTE1, outputs)
            .allowFixedLengthArcs(false)
            .build();

    IntsRefBuilder scratch = new IntsRefBuilder();
    for (var e : entries.entrySet()) {
      BytesRef key = new BytesRef(e.getKey());
      BytesRef output = new BytesRef(e.getValue());
      IntsRef input = Util.toIntsRef(key, scratch);
      fstCompiler.add(input, output);
    }
    FST.FSTMetadata<BytesRef> metadata = fstCompiler.compile();
    FST<BytesRef> fst = FST.fromFSTReader(metadata, fstCompiler.getFSTReader());

    Path fstFile = out.resolve("fst.bin");
    fst.save(fstFile);

    // Round-trip through Java Lucene itself before shipping the fixture:
    // read it back from disk via FST.read and confirm every present key
    // resolves to the same output (self-check, not the Rust test).
    FST<BytesRef> reloaded = FST.read(fstFile, outputs);
    for (var e : entries.entrySet()) {
      BytesRef got = Util.get(reloaded, new BytesRef(e.getKey()));
      if (got == null || !got.utf8ToString().equals(e.getValue())) {
        throw new AssertionError(
            "self-check failed for key=" + e.getKey() + " got=" + got + " want=" + e.getValue());
      }
    }

    List<String> absentKeys =
        List.of(
            "", // empty string: this FST has no empty output
            "a", // proper prefix of "app", not itself accepted
            "appl", // proper prefix of "apple"/"application"
            "apples", // extends past an accepting node with no further arcs
            "ban", // proper prefix of "banana"/"band"/"bandana"
            "bandanas", // extends past "bandana"
            "cat", // shares no prefix with any key
            "zz" // extends past "z"
            );
    for (String k : absentKeys) {
      if (!k.isEmpty()) {
        BytesRef got = Util.get(reloaded, new BytesRef(k));
        if (got != null) {
          throw new AssertionError("expected key=" + k + " to be absent, got=" + got);
        }
      }
    }

    StringBuilder m = new StringBuilder();
    m.append("num_present=").append(entries.size()).append('\n');
    int i = 0;
    for (var e : entries.entrySet()) {
      m.append("present.").append(i).append(".key_hex=").append(hex(e.getKey())).append('\n');
      m.append("present.")
          .append(i)
          .append(".output_hex=")
          .append(hex(e.getValue()))
          .append('\n');
      i++;
    }
    m.append("num_absent=").append(absentKeys.size()).append('\n');
    for (int j = 0; j < absentKeys.size(); j++) {
      m.append("absent.").append(j).append(".key_hex=").append(hex(absentKeys.get(j))).append('\n');
    }
    Files.writeString(out.resolve("manifest.properties"), m.toString());

    System.out.println("FST fixture written to " + out);
  }

  private static String hex(String s) {
    return HexFormat.of().formatHex(s.getBytes(java.nio.charset.StandardCharsets.UTF_8));
  }
}
