import org.apache.lucene.codecs.FieldsProducer;
import org.apache.lucene.codecs.PostingsFormat;
import org.apache.lucene.codecs.uniformsplit.BlockDecoder;
import org.apache.lucene.codecs.uniformsplit.BlockEncoder;
import org.apache.lucene.codecs.uniformsplit.sharedterms.STUniformSplitPostingsFormat;
import org.apache.lucene.codecs.lucene104.Lucene104Codec;
import org.apache.lucene.index.DocValuesSkipIndexType;
import org.apache.lucene.index.DocValuesType;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.PostingsEnum;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.Terms;
import org.apache.lucene.index.TermsEnum;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.store.DataInput;
import org.apache.lucene.store.DataOutput;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.util.BytesRef;
import org.apache.lucene.util.Version;
import org.apache.lucene.util.automaton.CompiledAutomaton;
import org.apache.lucene.util.automaton.Operations;
import org.apache.lucene.util.automaton.RegExp;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HexFormat;
import java.util.List;
import java.util.Map;

/**
 * Reverse-direction check for lucene-codecs' {@code uniformsplit.sharedterms} package: {@code
 * write_sharedterms_fixture} (Rust) writes three {@code STUniformSplitPostingsFormat} groups
 * ({@code .stustb}/{@code .stustd} plus {@code Lucene104} postings, the three fields' terms in one
 * dictionary, {@code body} and {@code tag} sharing many; default 32/3 block lines, 8/2, and 16/3
 * through an XOR block encoder), and real Lucene opens each through its own {@code fieldsProducer}
 * with a hand-built {@link SegmentInfo}/{@link FieldInfos}, then:
 *
 * <ul>
 *   <li>runs {@code checkIntegrity} (every checksum);
 *   <li>walks every field's terms with {@code next()} (Java's {@code STBlockReader}, skipping the
 *       other fields' lines), requiring the exact term list, {@code docFreq}/{@code
 *       totalTermFreq}, and every document's freq, positions, offsets and payloads through {@code
 *       postings(ALL)} to equal what Rust wrote;
 *   <li>{@code seekExact}s every term, and {@code seekCeil}s just past it (the term followed by a
 *       zero byte), requiring {@code NOT_FOUND} on the field's next term, or {@code END} after its
 *       last;
 *   <li>runs {@code Terms.intersect} for every pattern and start term Rust ran, requiring the
 *       exact terms (with docFreq/totalTermFreq) Rust's own reader enumerated.
 * </ul>
 *
 * <p>The XOR codec is duplicated from {@code GenCodecsSharedTerms} (verify-write-path compiles
 * only the Verify classes).
 *
 * <p>What it cannot catch: anything that binds these files into a segment ({@code
 * PerFieldPostingsFormat} attributes, {@code SegmentInfo.files}), since this port's IndexWriter
 * does not route a field to this format; Java's merge path ({@code STMergingTermsEnum}), which
 * this port does not have; nor an encoder other than this one.
 */
public class VerifyCodecsSharedTerms {
  /** XORs every byte with 0x5a and appends the (byte) sum of the plain bytes. */
  static final class XorEncoder implements BlockEncoder {
    @Override
    public WritableBytes encode(DataInput in, long length) throws IOException {
      byte[] b = new byte[Math.toIntExact(length) + 1];
      byte sum = 0;
      for (int i = 0; i < length; i++) {
        byte x = in.readByte();
        sum += x;
        b[i] = (byte) (x ^ 0x5a);
      }
      b[b.length - 1] = sum;
      return new WritableBytes() {
        @Override
        public long size() {
          return b.length;
        }

        @Override
        public void writeTo(DataOutput out) throws IOException {
          out.writeBytes(b, 0, b.length);
        }
      };
    }
  }

  /** {@link XorEncoder}'s inverse; refuses a block whose trailing sum does not match. */
  static final class XorDecoder implements BlockDecoder {
    @Override
    public BytesRef decode(DataInput in, long length) throws IOException {
      if (length < 1) throw new IOException("empty xor block");
      byte[] b = new byte[Math.toIntExact(length - 1)];
      byte sum = 0;
      for (int i = 0; i < b.length; i++) {
        b[i] = (byte) (in.readByte() ^ 0x5a);
        sum += b[i];
      }
      if (in.readByte() != sum) throw new IOException("xor block sum mismatch");
      return new BytesRef(b);
    }
  }

  public static void main(String[] args) throws Exception {
    Path dir = Path.of(args[0]);
    Map<String, String> manifest = new HashMap<>();
    for (String line : Files.readAllLines(dir.resolve("manifest.properties"))) {
      int eq = line.indexOf('=');
      if (eq > 0) {
        manifest.put(line.substring(0, eq), line.substring(eq + 1));
      }
    }
    byte[] id = HexFormat.of().parseHex(manifest.get("id_hex"));
    int maxDoc = Integer.parseInt(manifest.get("max_doc"));

    String[] names = {"id", "tag", "body"};
    IndexOptions[] options = {
      IndexOptions.DOCS,
      IndexOptions.DOCS_AND_FREQS,
      IndexOptions.DOCS_AND_FREQS_AND_POSITIONS_AND_OFFSETS
    };
    FieldInfo[] infos = new FieldInfo[names.length];
    for (int i = 0; i < names.length; i++) {
      infos[i] =
          new FieldInfo(
              names[i],
              i,
              false,
              false,
              i == 2,
              options[i],
              DocValuesType.NONE,
              DocValuesSkipIndexType.NONE,
              -1,
              new HashMap<>(),
              0,
              0,
              0,
              0,
              VectorEncoding.FLOAT32,
              VectorSimilarityFunction.EUCLIDEAN,
              false,
              false);
    }
    FieldInfos fis = new FieldInfos(infos);

    int failures = 0;
    try (Directory directory = FSDirectory.open(dir)) {
      SegmentInfo si =
          new SegmentInfo(
              directory,
              Version.LATEST,
              Version.LATEST,
              "_0",
              maxDoc,
              false,
              false,
              new Lucene104Codec(),
              Collections.emptyMap(),
              id,
              new HashMap<>(),
              null);
      failures +=
          verify(
              directory,
              si,
              fis,
              new STUniformSplitPostingsFormat(),
              "SharedTermsUniformSplit_0",
              dir);
      failures +=
          verify(
              directory,
              si,
              fis,
              new STUniformSplitPostingsFormat(8, 2, null, null, true),
              "SharedTermsUniformSplit_1",
              dir);
      failures +=
          verify(
              directory,
              si,
              fis,
              new STUniformSplitPostingsFormat(16, 3, new XorEncoder(), new XorDecoder(), false),
              "SharedTermsUniformSplit_2",
              dir);
    }
    if (failures > 0) {
      System.out.println(failures + " failure(s)");
      System.exit(1);
    }
    System.out.println("SharedTermsUniformSplit groups verified against real Lucene. PASS");
  }

  static String hex(BytesRef b) {
    return HexFormat.of().formatHex(b.bytes, b.offset, b.offset + b.length);
  }

  /** {@code GenCodecsSharedTerms.dumpPostings}' line for the enum's current term. */
  static String termLine(String field, Terms terms, TermsEnum te) throws Exception {
    StringBuilder sb = new StringBuilder(field);
    sb.append('\t').append(hex(te.term())).append('\t').append(te.docFreq()).append('\t')
        .append(te.totalTermFreq());
    PostingsEnum pe = te.postings(null, PostingsEnum.ALL);
    for (int doc = pe.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS; doc = pe.nextDoc()) {
      int freq = pe.freq();
      sb.append('\t').append(doc).append(':').append(freq);
      if (terms.hasPositions()) {
        StringBuilder pos = new StringBuilder();
        StringBuilder offs = new StringBuilder();
        StringBuilder pays = new StringBuilder();
        for (int k = 0; k < freq; k++) {
          pos.append(k == 0 ? "" : ".").append(pe.nextPosition());
          offs.append(k == 0 ? "" : ".").append(pe.startOffset()).append('-')
              .append(pe.endOffset());
          BytesRef pl = pe.getPayload();
          pays.append(k == 0 ? "" : ".").append(pl == null || pl.length == 0 ? "-" : hex(pl));
        }
        sb.append(':').append(pos).append(':').append(offs).append(':').append(pays);
      }
    }
    return sb.toString();
  }

  /** Past the field's last term, {@code seekCeil} is {@code END}. */
  static int checkLastCeil(Terms terms, BytesRef past) throws Exception {
    TermsEnum.SeekStatus status = terms.iterator().seekCeil(past);
    if (status != TermsEnum.SeekStatus.END) {
      System.out.println("seekCeil past the last term gave " + status);
      return 1;
    }
    return 0;
  }

  static int verify(
      Directory directory,
      SegmentInfo si,
      FieldInfos fis,
      PostingsFormat format,
      String suffix,
      Path dir)
      throws Exception {
    int failures = 0;
    SegmentReadState state = new SegmentReadState(directory, si, fis, IOContext.DEFAULT, suffix);
    try (FieldsProducer fields = format.fieldsProducer(state)) {
      fields.checkIntegrity();
      String current = null;
      TermsEnum walk = null;
      Terms terms = null;
      long termCount = 0;
      BytesRef pendingCeil = null;
      for (String line : Files.readAllLines(dir.resolve("terms." + suffix + ".tsv"))) {
        String[] e = line.split("\t", 3);
        if (!e[0].equals(current)) {
          if (walk != null && walk.next() != null) {
            System.out.println(suffix + ": extra term in " + current);
            failures++;
          }
          if (pendingCeil != null) failures += checkLastCeil(terms, pendingCeil);
          pendingCeil = null;
          current = e[0];
          terms = fields.terms(current);
          walk = terms.iterator();
        }
        BytesRef want = new BytesRef(HexFormat.of().parseHex(e[1]));
        if (pendingCeil != null) {
          TermsEnum ceil = terms.iterator();
          TermsEnum.SeekStatus status = ceil.seekCeil(pendingCeil);
          if (status != TermsEnum.SeekStatus.NOT_FOUND || !ceil.term().equals(want)) {
            System.out.println(suffix + ": " + current + " seekCeil(" + hex(pendingCeil)
                + ") gave " + status + " expected NOT_FOUND on " + e[1]);
            failures++;
          }
        }
        BytesRef got = walk.next();
        if (got == null || !got.equals(want)) {
          System.out.println(suffix + ": " + current + " next() " + got + " expected " + want);
          return failures + 1;
        }
        String gotLine = termLine(current, terms, walk);
        if (!gotLine.equals(line)) {
          System.out.println(suffix + ": " + current + " " + e[1] + "\n  got  " + gotLine
              + "\n  want " + line);
          failures++;
        }
        if (!terms.iterator().seekExact(want)) {
          System.out.println(suffix + ": " + current + " seekExact missed " + e[1]);
          failures++;
        }
        pendingCeil = new BytesRef(Arrays.copyOf(want.bytes, want.length + 1));
        termCount++;
      }
      if (walk != null && walk.next() != null) {
        System.out.println(suffix + ": extra term in " + current);
        failures++;
      }
      if (pendingCeil != null) failures += checkLastCeil(terms, pendingCeil);

      long rows = 0;
      long matches = 0;
      for (String line : Files.readAllLines(dir.resolve("intersect." + suffix + ".tsv"))) {
        String[] e = line.split("\t");
        CompiledAutomaton compiled =
            new CompiledAutomaton(
                Operations.determinize(
                    new RegExp(e[1], RegExp.ALL, 0).toAutomaton(),
                    Operations.DEFAULT_DETERMINIZE_WORK_LIMIT),
                false,
                true,
                false);
        TermsEnum te =
            fields.terms(e[0])
                .intersect(compiled, e[2].equals("-") ? null : new BytesRef(e[2]));
        List<String> got = new ArrayList<>();
        for (BytesRef t = te.next(); t != null; t = te.next()) {
          got.add(hex(t) + ":" + te.docFreq() + ":" + te.totalTermFreq());
        }
        List<String> want = List.of(e).subList(3, e.length);
        if (!got.equals(want)) {
          System.out.println(suffix + ": intersect " + e[0] + " " + e[1] + " " + e[2]
              + "\n  got  " + got + "\n  want " + want);
          failures++;
        }
        rows++;
        matches += got.size();
      }
      System.out.println(
          suffix + ": " + termCount + " terms, " + rows + " intersections (" + matches
              + " terms), ok");
    }
    return failures;
  }
}
