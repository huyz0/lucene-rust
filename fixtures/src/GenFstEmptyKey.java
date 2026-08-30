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
 * Generates a real {@code FST<BytesRef>} that <em>accepts the empty string</em>
 * with a non-empty output -- the one shape no other {@code GenFst*} fixture
 * covers, and the shape that exercises {@code FST.FSTMetadata}'s
 * {@code emptyOutput} serialization.
 *
 * <p>That serialization is not simply "the output bytes, reversed":
 * {@code FSTMetadata.save} runs the value through
 * {@code outputs.writeFinalOutput} first (for {@link ByteSequenceOutputs}: a
 * {@code vint} length followed by the payload), reverses <em>that whole
 * buffer</em>, and writes {@code vint(len)} + the reversed buffer, so it
 * decodes with the same reverse {@code BytesReader} every arc output uses.
 * A reader that reverses the buffer and keeps it verbatim silently gains a
 * leading length byte -- which is exactly what this port did before this
 * fixture existed, and why hand-built bytes were not enough here.
 *
 * <p>{@code allowFixedLengthArcs(false)} keeps the body in the same
 * variable-length ("list") arc encoding {@code GenFst.java} produces, so this
 * fixture isolates the metadata concern rather than mixing in a node-encoding
 * change.
 */
public class GenFstEmptyKey {
  public static void main(String[] args) throws IOException {
    Path out = Path.of(args[0]).resolve("fst_empty_key");
    Files.createDirectories(out);

    TreeMap<String, String> entries = new TreeMap<>();
    entries.put("", "ROOT-OUTPUT"); // the empty string, with a real output
    entries.put("app", "1");
    entries.put("apple", "2");
    entries.put("banana", "4");
    entries.put("z", "26");

    ByteSequenceOutputs outputs = ByteSequenceOutputs.getSingleton();
    FSTCompiler<BytesRef> fstCompiler =
        new FSTCompiler.Builder<>(FST.INPUT_TYPE.BYTE1, outputs)
            .allowFixedLengthArcs(false)
            .build();

    IntsRefBuilder scratch = new IntsRefBuilder();
    for (var e : entries.entrySet()) {
      IntsRef input = Util.toIntsRef(new BytesRef(e.getKey()), scratch);
      fstCompiler.add(input, new BytesRef(e.getValue()));
    }
    FST.FSTMetadata<BytesRef> metadata = fstCompiler.compile();
    FST<BytesRef> fst = FST.fromFSTReader(metadata, fstCompiler.getFSTReader());

    Path fstFile = out.resolve("fst.bin");
    fst.save(fstFile);

    // Round-trip through Java Lucene itself before shipping the fixture.
    FST<BytesRef> reloaded = FST.read(fstFile, outputs);
    if (reloaded.getEmptyOutput() == null
        || !reloaded.getEmptyOutput().utf8ToString().equals(entries.get(""))) {
      throw new AssertionError("self-check failed: emptyOutput=" + reloaded.getEmptyOutput());
    }
    for (var e : entries.entrySet()) {
      BytesRef got = Util.get(reloaded, new BytesRef(e.getKey()));
      if (got == null || !got.utf8ToString().equals(e.getValue())) {
        throw new AssertionError(
            "self-check failed for key=" + e.getKey() + " got=" + got + " want=" + e.getValue());
      }
    }

    List<String> absentKeys = List.of("a", "appl", "apples", "ban", "cat", "zz");
    for (String k : absentKeys) {
      if (Util.get(reloaded, new BytesRef(k)) != null) {
        throw new AssertionError("expected key=" + k + " to be absent");
      }
    }

    StringBuilder m = new StringBuilder();
    m.append("empty_output_hex=").append(hex(entries.get(""))).append('\n');
    m.append("num_present=").append(entries.size()).append('\n');
    int i = 0;
    for (var e : entries.entrySet()) {
      m.append("present.").append(i).append(".key_hex=").append(hex(e.getKey())).append('\n');
      m.append("present.").append(i).append(".output_hex=").append(hex(e.getValue())).append('\n');
      i++;
    }
    m.append("num_absent=").append(absentKeys.size()).append('\n');
    for (int j = 0; j < absentKeys.size(); j++) {
      m.append("absent.").append(j).append(".key_hex=").append(hex(absentKeys.get(j))).append('\n');
    }
    Files.writeString(out.resolve("manifest.properties"), m.toString());

    System.out.println("FST empty-key fixture written to " + out);
  }

  private static String hex(String s) {
    return HexFormat.of().formatHex(s.getBytes(java.nio.charset.StandardCharsets.UTF_8));
  }
}
