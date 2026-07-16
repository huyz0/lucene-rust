import org.apache.lucene.util.BytesRef;
import org.apache.lucene.util.IntsRef;
import org.apache.lucene.util.IntsRefBuilder;
import org.apache.lucene.util.fst.BytesRefFSTEnum;
import org.apache.lucene.util.fst.ByteSequenceOutputs;
import org.apache.lucene.util.fst.FST;
import org.apache.lucene.util.fst.FSTCompiler;
import org.apache.lucene.util.fst.Util;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HexFormat;
import java.util.List;
import java.util.TreeMap;

/**
 * Generates a real on-heap {@code FST<BytesRef>} (`fst_deep_trie/` subdirectory)
 * whose keys are deliberately chosen so a single {@code seekCeil}/{@code
 * seekFloor}/{@code seekExact} call must descend (and, for several absent
 * targets, backtrack) across at least 3 distinct trie levels -- every prior
 * `GenFst*` fixture's interesting structure lives at or one level below the
 * root; none forces genuinely deep, multi-level branching the way a real
 * term-index FST over longer terms would.
 *
 * <p>Keys (9, sorted): {@code abcaa, abcab, abcz, abda, abdz, acaa, aczz,
 * baaa, bzzz}. With {@code allowFixedLengthArcs(false)} (list-encoded nodes
 * only, same scope as {@code GenFst.java}), the BYTE1 input type means every
 * byte of a shared prefix is its own trie node, so this key set produces a
 * real chain: root -'a'/'b'-> node1 -'b'/'c'-> node2(under "ab") -'c'/'d'->
 * node3(under "abc") -'a'/'z'-> node4(under "abca") -'a'/'b'-> leaves. That's
 * 5 levels along the "abcaa"/"abcab" path -- confirmed below by manually
 * walking arcs with {@code readFirstTargetArc}/{@code readNextArc} rather
 * than assumed.
 *
 * <p>Ground truth for every seek in the manifest comes from real Lucene's own
 * {@link BytesRefFSTEnum#seekCeil}/{@code seekFloor}/{@code seekExact} against
 * the reloaded FST -- not hand-derived -- so the Rust differential test is
 * checked against actual `FSTEnum` behavior for deep, multi-level
 * backtracking, the same standard every other fixture in this directory
 * holds to.
 */
public class GenFstDeepTrie {
  public static void main(String[] args) throws IOException {
    Path out = Path.of(args[0]).resolve("fst_deep_trie");
    Files.createDirectories(out);

    TreeMap<String, String> entries = new TreeMap<>();
    entries.put("abcaa", "1");
    entries.put("abcab", "2");
    entries.put("abcz", "3");
    entries.put("abda", "4");
    entries.put("abdz", "5");
    entries.put("acaa", "6");
    entries.put("aczz", "7");
    entries.put("baaa", "8");
    entries.put("bzzz", "9");

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

    FST<BytesRef> reloaded = FST.read(fstFile, outputs);

    // Self-check #1: every present key still resolves via real Util.get.
    for (var e : entries.entrySet()) {
      BytesRef got = Util.get(reloaded, new BytesRef(e.getKey()));
      if (got == null || !got.utf8ToString().equals(e.getValue())) {
        throw new AssertionError(
            "self-check failed for key=" + e.getKey() + " got=" + got + " want=" + e.getValue());
      }
    }

    // Self-check #2: the path to "abcaa" genuinely crosses >= 3 distinct
    // trie levels below the root (i.e. this is not secretly a shallow FST
    // that happens to have long keys) -- walk it by hand, byte by byte,
    // via readFirstTargetArc/readNextArc, and count how many *node
    // boundaries* (arcs consumed) it takes to reach the accepting arc.
    FST.BytesReader r = reloaded.getBytesReader();
    FST.Arc<BytesRef> arc = reloaded.getFirstArc(new FST.Arc<>());
    byte[] path = "abcaa".getBytes(StandardCharsets.UTF_8);
    int depth = 0;
    for (byte b : path) {
      int label = b & 0xff;
      FST.Arc<BytesRef> next = new FST.Arc<>();
      reloaded.readFirstTargetArc(arc, next, r);
      boolean found = false;
      while (true) {
        if (next.label() == label) {
          found = true;
          break;
        }
        if (next.isLast()) {
          break;
        }
        reloaded.readNextArc(next, r);
      }
      if (!found) {
        throw new AssertionError("manual walk lost the path to \"abcaa\" at byte " + (char) b);
      }
      arc = next;
      depth++;
    }
    if (depth < 3) {
      throw new AssertionError("expected >= 3 trie levels along \"abcaa\", got " + depth);
    }
    if (!arc.isFinal()) {
      throw new AssertionError("expected \"abcaa\" to land on a final arc");
    }

    // Seek targets: a mix of present keys and absent keys chosen to force
    // seekCeil/seekFloor to resolve or backtrack at every level of the trie
    // (root, node1 "a", node2 "ab", node3 "abc", node4 "abca").
    List<String> targets =
        List.of(
            "", // before everything
            "a", // ends exactly at node1 (root's 'a' arc), no bytes left
            "aaaa", // before node1's first arc ('b') -- root-level-adjacent backtrack
            "azzz", // past node1's last arc ('c')
            "ab", // ends exactly at node2 (under "a"'s 'b' arc)
            "abc", // ends exactly at node3 (under "ab"'s 'c' arc)
            "abcaa", // present, exact, deepest leaf
            "abcab", // present, exact, deepest leaf sibling
            "abcaz", // past node4's last arc ('b') -- must backtrack up to node3
            "abcb", // absent between node3's 'a' and 'z' arcs
            "abczz", // past "abcz" -- must cross into the "abd" subtree
            "abd", // ends exactly at node2's 'd' arc
            "abdzz", // past "abdz" -- must cross into the "ac" subtree
            "aca", // proper prefix of "acaa", before it
            "ad", // between "ac.." and "b.." subtrees -- root-level backtrack
            "b", // ends exactly at root's 'b' arc
            "c" // past everything
            );

    BytesRefFSTEnum<BytesRef> ceilEnum = new BytesRefFSTEnum<>(reloaded);
    BytesRefFSTEnum<BytesRef> floorEnum = new BytesRefFSTEnum<>(reloaded);
    BytesRefFSTEnum<BytesRef> exactEnum = new BytesRefFSTEnum<>(reloaded);

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

    m.append("num_targets=").append(targets.size()).append('\n');
    for (int j = 0; j < targets.size(); j++) {
      String t = targets.get(j);
      BytesRef bt = new BytesRef(t);
      m.append("target.").append(j).append(".key_hex=").append(hex(t)).append('\n');

      BytesRefFSTEnum.InputOutput<BytesRef> ceil = ceilEnum.seekCeil(bt);
      appendResult(m, "target." + j + ".ceil", ceil);

      BytesRefFSTEnum.InputOutput<BytesRef> floor = floorEnum.seekFloor(bt);
      appendResult(m, "target." + j + ".floor", floor);

      BytesRefFSTEnum.InputOutput<BytesRef> exact = exactEnum.seekExact(bt);
      appendResult(m, "target." + j + ".exact", exact);
    }
    Files.writeString(out.resolve("manifest.properties"), m.toString());

    System.out.println("FST deep-trie fixture written to " + out + " (manual-walk depth=" + depth + ")");
  }

  private static void appendResult(
      StringBuilder m, String prefix, BytesRefFSTEnum.InputOutput<BytesRef> io) {
    if (io == null) {
      m.append(prefix).append(".present=false").append('\n');
    } else {
      m.append(prefix).append(".present=true").append('\n');
      m.append(prefix).append(".key_hex=").append(hex(io.input)).append('\n');
      m.append(prefix).append(".output_hex=").append(hex(io.output)).append('\n');
    }
  }

  private static String hex(String s) {
    return HexFormat.of().formatHex(s.getBytes(StandardCharsets.UTF_8));
  }

  private static String hex(BytesRef b) {
    return HexFormat.of().formatHex(b.bytes, b.offset, b.offset + b.length);
  }
}
