import org.apache.lucene.util.BytesRef;
import org.apache.lucene.util.fst.ByteSequenceOutputs;
import org.apache.lucene.util.fst.FST;
import org.apache.lucene.util.fst.Util;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HexFormat;
import java.util.List;
import java.util.Properties;

/**
 * Verifies {@code fst::build_fst}/{@code fst::write_fst}
 * ({@code crates/lucene-codecs/src/fst.rs}) by opening the bytes written by
 * {@code crates/lucene-codecs/examples/write_fst_fixture.rs} through real
 * Lucene's {@link FST#read(Path, org.apache.lucene.util.fst.Outputs)} and
 * looking up every key via real {@link Util#get(FST, BytesRef)}.
 *
 * <p>This is the reverse direction of {@code GenFst.java}: instead of a real
 * {@code FSTCompiler}-built FST being read by this port's {@code Fst::read},
 * this port's own from-scratch, non-minimizing {@code build_fst} construction
 * writes the bytes and real Lucene reads them back. Passing here is the
 * concrete proof that the simplified construction (no suffix sharing, no
 * output pushing, no fixed-length-arc nodes -- see {@code fst.rs}'s module
 * doc) is still a wire-format-valid FST that unmodified Lucene code accepts.
 *
 * <p>Verifies two fixtures: the small 7-key set in {@code args[0]} (same
 * shape as {@code GenFst.java}'s read-side fixture), and a larger 200-key
 * set in {@code args[0]/large} that forces multi-byte {@code vlong}
 * node-address targets -- a shape only previously self-round-tripped
 * through this port's own reader, never checked against real Lucene.
 */
public class VerifyFst {
  public static void main(String[] args) throws IOException {
    verifyOne(Path.of(args[0]));
    verifyOne(Path.of(args[0]).resolve("large"));
  }

  private static void verifyOne(Path dir) throws IOException {
    Path fstFile = dir.resolve("fst.bin");

    Properties manifest = new Properties();
    try (var in = Files.newBufferedReader(dir.resolve("manifest.properties"))) {
      manifest.load(in);
    }

    ByteSequenceOutputs outputs = ByteSequenceOutputs.getSingleton();
    FST<BytesRef> fst = FST.read(fstFile, outputs);

    HexFormat hex = HexFormat.of();
    int numPresent = Integer.parseInt(manifest.getProperty("num_present"));
    if (numPresent <= 0) {
      throw new AssertionError("expected num_present > 0");
    }
    for (int i = 0; i < numPresent; i++) {
      byte[] key = hex.parseHex(manifest.getProperty("present." + i + ".key_hex"));
      byte[] wantOutput = hex.parseHex(manifest.getProperty("present." + i + ".output_hex"));
      BytesRef got = Util.get(fst, new BytesRef(key));
      if (got == null) {
        throw new AssertionError(
            "key " + new String(key, java.nio.charset.StandardCharsets.UTF_8) + " (index " + i
                + ") expected present but was not found");
      }
      byte[] gotBytes = java.util.Arrays.copyOfRange(got.bytes, got.offset, got.offset + got.length);
      if (!java.util.Arrays.equals(gotBytes, wantOutput)) {
        throw new AssertionError(
            "key "
                + new String(key, java.nio.charset.StandardCharsets.UTF_8)
                + " (index "
                + i
                + ") expected output "
                + hex.formatHex(wantOutput)
                + " but got "
                + hex.formatHex(gotBytes));
      }
    }

    int numAbsent = Integer.parseInt(manifest.getProperty("num_absent"));
    if (numAbsent <= 0) {
      throw new AssertionError("expected num_absent > 0");
    }
    List<String> failures = new java.util.ArrayList<>();
    for (int i = 0; i < numAbsent; i++) {
      byte[] key = hex.parseHex(manifest.getProperty("absent." + i + ".key_hex"));
      BytesRef got = Util.get(fst, new BytesRef(key));
      if (got != null) {
        failures.add(
            "key "
                + new String(key, java.nio.charset.StandardCharsets.UTF_8)
                + " (index "
                + i
                + ") expected absent but got "
                + got);
      }
    }
    if (!failures.isEmpty()) {
      throw new AssertionError(String.join("; ", failures));
    }

    System.out.println(
        "VerifyFst OK ("
            + dir.getFileName()
            + "): "
            + numPresent
            + " present keys resolved, "
            + numAbsent
            + " absent keys rejected");
  }
}
