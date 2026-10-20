import org.apache.lucene.codecs.FieldsProducer;
import org.apache.lucene.codecs.PostingsReaderBase;
import org.apache.lucene.codecs.blockterms.BlockTermsReader;
import org.apache.lucene.codecs.blockterms.FixedGapTermsIndexReader;
import org.apache.lucene.codecs.blockterms.TermsIndexReaderBase;
import org.apache.lucene.codecs.blockterms.VariableGapTermsIndexReader;
import org.apache.lucene.codecs.lucene104.Lucene104Codec;
import org.apache.lucene.codecs.lucene104.Lucene104PostingsReader;
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
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.util.BytesRef;
import org.apache.lucene.util.Version;

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
 * Reverse-direction check for lucene-codecs' {@code blockterms} package: {@code
 * write_blockterms_fixture} (Rust) writes four block terms groups ({@code .tib} plus {@code
 * Lucene104} postings, over a {@code .tii} fixed-gap index of interval 32 or 3, or a {@code
 * .tiv}/{@code .tmv} variable-gap index picking every 5th term or every 9th and those in 30+
 * documents), and real Lucene opens each through its own {@link BlockTermsReader} over its own
 * {@link FixedGapTermsIndexReader} or {@link VariableGapTermsIndexReader} and a {@link
 * Lucene104PostingsReader} (Lucene ships no postings format over this package) with a hand-built
 * {@link SegmentInfo}/{@link FieldInfos}, then:
 *
 * <ul>
 *   <li>runs {@code checkIntegrity} (the {@code .tib}'s and the postings' checksums; the index
 *       files' are checked as they open);
 *   <li>walks every term with {@code next()}, requiring the exact term list, {@code
 *       docFreq}/{@code totalTermFreq}, and every document's freq, positions, offsets and payloads
 *       through {@code postings(ALL)} to equal what Rust wrote -- and, over the fixed-gap index,
 *       {@code ord()} to be the term's line within its field;
 *   <li>{@code seekExact}s every term on a fresh enum, and {@code seekCeil}s just past it (the
 *       term followed by a zero byte) on one enum reused across the field, requiring {@code
 *       NOT_FOUND} on the next term, or {@code END} after the last;
 *   <li>over the fixed-gap index, {@code seekExact(ord)}s every 7th ord on a reused enum,
 *       requiring that ord's term and {@code docFreq}.
 * </ul>
 *
 * <p>What it cannot catch: anything that binds these files into a segment ({@code
 * PerFieldPostingsFormat} attributes, {@code SegmentInfo.files}), since this port's IndexWriter
 * does not route a field to this package; nor a selector other than these two.
 */
public class VerifyCodecsBlockTerms {
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
    List<String> lines = Files.readAllLines(dir.resolve("terms.tsv"));

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
      failures += verify(directory, si, fis, "BlockTerms_0", true, lines);
      failures += verify(directory, si, fis, "BlockTerms_1", true, lines);
      failures += verify(directory, si, fis, "BlockTerms_2", false, lines);
      failures += verify(directory, si, fis, "BlockTerms_3", false, lines);
    }
    if (failures > 0) {
      System.out.println(failures + " failure(s)");
      System.exit(1);
    }
    System.out.println("BlockTerms groups verified against real Lucene. PASS");
  }

  static String hex(BytesRef b) {
    return HexFormat.of().formatHex(b.bytes, b.offset, b.offset + b.length);
  }

  /** {@code write_blockterms_fixture}'s line for the enum's current term. */
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

  /** One field's lines: walk, seek, ceil and (with ords) seek by ord. */
  static int verifyField(
      String suffix, String field, Terms terms, boolean ords, List<String> lines)
      throws Exception {
    int failures = 0;
    TermsEnum walk = terms.iterator();
    TermsEnum ceil = terms.iterator();
    List<BytesRef> all = new ArrayList<>();
    List<Integer> docFreqs = new ArrayList<>();
    for (String line : lines) {
      String[] e = line.split("\t", 5);
      BytesRef want = new BytesRef(HexFormat.of().parseHex(e[1]));
      BytesRef got = walk.next();
      if (got == null || !got.equals(want)) {
        System.out.println(suffix + ": " + field + " next() " + got + " expected " + want);
        return failures + 1;
      }
      if (ords && walk.ord() != all.size()) {
        System.out.println(suffix + ": " + field + " " + e[1] + " ord " + walk.ord()
            + " expected " + all.size());
        failures++;
      }
      String gotLine = termLine(field, terms, walk);
      if (!gotLine.equals(line)) {
        System.out.println(suffix + ": " + field + " " + e[1] + "\n  got  " + gotLine
            + "\n  want " + line);
        failures++;
      }
      if (!terms.iterator().seekExact(want)) {
        System.out.println(suffix + ": " + field + " seekExact missed " + e[1]);
        failures++;
      }
      all.add(want);
      docFreqs.add(Integer.parseInt(e[2]));
    }
    if (walk.next() != null) {
      System.out.println(suffix + ": extra term in " + field);
      failures++;
    }
    for (int i = 0; i < all.size(); i++) {
      BytesRef past = new BytesRef(Arrays.copyOf(all.get(i).bytes, all.get(i).length + 1));
      TermsEnum.SeekStatus status = ceil.seekCeil(past);
      boolean last = i + 1 == all.size();
      if (last
          ? status != TermsEnum.SeekStatus.END
          : status != TermsEnum.SeekStatus.NOT_FOUND || !ceil.term().equals(all.get(i + 1))) {
        System.out.println(suffix + ": " + field + " seekCeil(" + hex(past) + ") gave " + status);
        failures++;
      }
    }
    if (ords) {
      TermsEnum byOrd = terms.iterator();
      for (int ord = all.size() - 1; ord >= 0; ord -= 7) {
        byOrd.seekExact(ord);
        if (!byOrd.term().equals(all.get(ord))
            || byOrd.ord() != ord
            || byOrd.docFreq() != docFreqs.get(ord)) {
          System.out.println(suffix + ": " + field + " seekExact(" + ord + ") gave "
              + hex(byOrd.term()) + " ord " + byOrd.ord());
          failures++;
        }
      }
    }
    return failures;
  }

  static int verify(
      Directory directory,
      SegmentInfo si,
      FieldInfos fis,
      String suffix,
      boolean fixedGap,
      List<String> lines)
      throws Exception {
    int failures = 0;
    SegmentReadState state = new SegmentReadState(directory, si, fis, IOContext.DEFAULT, suffix);
    PostingsReaderBase postings = new Lucene104PostingsReader(state);
    TermsIndexReaderBase index =
        fixedGap ? new FixedGapTermsIndexReader(state) : new VariableGapTermsIndexReader(state);
    try (FieldsProducer fields = new BlockTermsReader(index, postings, state)) {
      fields.checkIntegrity();
      Map<String, List<String>> byField = new HashMap<>();
      for (String line : lines) {
        byField.computeIfAbsent(line.substring(0, line.indexOf('\t')), k -> new ArrayList<>())
            .add(line);
      }
      long termCount = 0;
      for (String field : List.of("body", "id", "tag")) {
        List<String> fieldLines = byField.get(field);
        failures += verifyField(suffix, field, fields.terms(field), fixedGap, fieldLines);
        termCount += fieldLines.size();
      }
      System.out.println(suffix + ": " + termCount + " terms, ok");
    }
    return failures;
  }
}
