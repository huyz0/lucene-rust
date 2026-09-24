import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.SegmentCommitInfo;
import org.apache.lucene.index.SegmentInfos;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.Terms;
import org.apache.lucene.index.TermsEnum;
import org.apache.lucene.search.RegexpQuery;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.BytesRef;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HexFormat;
import java.util.List;
import java.util.TreeSet;

/**
 * Ground truth for the lockstep regexp walk over a real, multi-level term
 * dictionary: which terms {@code RegexpQuery}'s own {@code CompiledAutomaton}
 * enumerates ({@code getTermsEnum(terms)}, i.e. {@code Terms.intersect} --
 * {@code IntersectTermsEnum}) for each pattern.
 *
 * <p>The dictionary is big and varied enough that Lucene's block-tree writer
 * lays it out with everything the walk has to navigate: floor blocks (60 000
 * base-36 ids under one leading {@code t}), nested sub-blocks, long shared
 * prefixes, multi-byte UTF-8 and ill-formed bytes (binary terms). This port's
 * own terms writer emits a single leaf block per field, so only a
 * Java-written index exercises any of that.
 *
 * <p>{@code cases.tsv}: {@code pattern TAB count TAB fnv TAB docFreqSum TAB
 * firstHex TAB lastHex}, where {@code fnv} is FNV-1a 64 over every matched
 * term in enumeration order as a 4-byte little-endian length followed by its
 * bytes. Deterministic: no randomness, one segment.
 */
public class GenRegexpIntersect {

  static final String[] PATTERNS = {
    "t1[0-9]",
    ".*z",
    "t[0-9a-f]+",
    "(t1|t2)[a-z]",
    "t.*9",
    "[a-z][0-9]{2}",
    ".*1.*2.*",
    "t",
    "",
    ".*",
    "t.",
    "tz.*",
    "[^t].*",
    "é[0-9]+€",
    "é.*",
    "a{16}[0-9]+",
    "a+1.*",
    ".x1[0-2]?",
    "t<100-2000>",
    "t[a-z]+&.*q.*",
    "#",
    "(t0|t1|tz|t10)",
    "t1a.",
    "t[0-9]{2}[a-z]",
    ".*[qz][0-9]",
    ".",
    ".{2,3}",
    "[0-9a-z]x1",
    "t9zz.*",
    "zzzz",
    "t.*(0|9)",
    "@",
    "t[^0-9].*",
    "t.{3}",
    ".*0",
    "(a|é|t).",
    "t[0-9]+[a-z]+[0-9]+",
    "<0-99>x.*",
  };

  static String base36(int n) {
    return Integer.toString(n, 36);
  }

  static List<byte[]> terms() {
    TreeSet<BytesRef> set = new TreeSet<>();
    for (int i = 0; i < 60_000; i++) {
      set.add(new BytesRef("t" + base36(i)));
    }
    for (int i = 0; i < 3_000; i++) {
      set.add(new BytesRef(base36(i % 97) + "x" + (i % 13)));
      set.add(new BytesRef("é" + (i % 500) + "€"));
      set.add(new BytesRef("aaaaaaaaaaaaaaaa" + i));
    }
    for (int i = 0; i < 256; i++) {
      set.add(new BytesRef(new byte[] {'t', (byte) 0xFF, (byte) i}));
      set.add(new BytesRef(new byte[] {(byte) i}));
    }
    List<byte[]> out = new ArrayList<>();
    for (BytesRef b : set) {
      out.add(BytesRef.deepCopyOf(b).bytes);
    }
    return out;
  }

  static long fnv(long h, byte[] bytes, int off, int len) {
    for (int i = 0; i < len; i++) {
      h ^= bytes[off + i] & 0xFF;
      h *= 0x100000001b3L;
    }
    return h;
  }

  public static void main(String[] args) throws IOException {
    Path out = Path.of(args[0]).resolve("regexp_intersect_index");
    if (Files.exists(out)) {
      try (var s = Files.list(out)) {
        for (Path p : s.toList()) {
          Files.delete(p);
        }
      }
    }
    Files.createDirectories(out);
    List<byte[]> terms = terms();
    int n = terms.size();

    try (Directory dir = FSDirectory.open(out)) {
      IndexWriterConfig cfg = new IndexWriterConfig();
      cfg.setUseCompoundFile(false);
      try (IndexWriter w = new IndexWriter(dir, cfg)) {
        // Document d holds term d and term (7d mod n): docFreq 1-3, so both
        // pulsed singletons and real `.doc` postings are in play.
        for (int d = 0; d < n; d++) {
          Document doc = new Document();
          doc.add(new StringField("body", new BytesRef(terms.get(d)), Field.Store.NO));
          int other = (int) ((7L * d) % n);
          if (other != d) {
            doc.add(new StringField("body", new BytesRef(terms.get(other)), Field.Store.NO));
          }
          w.addDocument(doc);
        }
        w.forceMerge(1);
        w.commit();
      }

      SegmentInfos sis = SegmentInfos.readLatestCommit(dir);
      SegmentCommitInfo sci = sis.info(0);
      StringBuilder m = new StringBuilder();
      for (String f : sci.info.files()) {
        String ext = f.substring(f.lastIndexOf('.') + 1);
        m.append(ext).append("_file_name=").append(f).append('\n');
      }
      m.append("id_hex=").append(HexFormat.of().formatHex(sci.info.getId())).append('\n');
      m.append("segment_name=").append(sci.info.name).append('\n');
      m.append("max_doc=").append(sci.info.maxDoc()).append('\n');
      m.append("num_terms=").append(n).append('\n');

      StringBuilder cases = new StringBuilder();
      try (DirectoryReader reader = DirectoryReader.open(dir)) {
        LeafReader leaf = reader.leaves().get(0).reader();
        Terms t = leaf.terms("body");
        for (String p : PATTERNS) {
          RegexpQuery q = new RegexpQuery(new Term("body", p));
          TermsEnum te = q.getCompiled().getTermsEnum(t);
          long h = 0xcbf29ce484222325L;
          long count = 0;
          long dfSum = 0;
          String first = "-";
          String last = "-";
          for (BytesRef b = te.next(); b != null; b = te.next()) {
            byte[] len = {
              (byte) b.length, (byte) (b.length >>> 8), (byte) (b.length >>> 16), (byte) (b.length >>> 24)
            };
            h = fnv(h, len, 0, 4);
            h = fnv(h, b.bytes, b.offset, b.length);
            String hex = HexFormat.of().formatHex(b.bytes, b.offset, b.offset + b.length);
            if (count == 0) {
              first = hex;
            }
            last = hex;
            count++;
            dfSum += te.docFreq();
          }
          cases
              .append(p)
              .append('\t')
              .append(count)
              .append('\t')
              .append(Long.toUnsignedString(h, 16))
              .append('\t')
              .append(dfSum)
              .append('\t')
              .append(first.isEmpty() ? "EMPTY" : first)
              .append('\t')
              .append(last.isEmpty() ? "EMPTY" : last)
              .append('\n');
        }
      }
      Files.writeString(out.resolve("manifest.properties"), m.toString(), StandardCharsets.UTF_8);
      Files.writeString(out.resolve("cases.tsv"), cases.toString(), StandardCharsets.UTF_8);
    }
  }
}
