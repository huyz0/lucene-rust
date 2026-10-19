import org.apache.lucene.codecs.FieldsProducer;
import org.apache.lucene.codecs.PostingsFormat;
import org.apache.lucene.codecs.lucene104.Lucene104Codec;
import org.apache.lucene.codecs.memory.DirectPostingsFormat;
import org.apache.lucene.codecs.memory.FSTPostingsFormat;
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
import org.apache.lucene.util.automaton.CompiledAutomaton;
import org.apache.lucene.util.automaton.Operations;
import org.apache.lucene.util.automaton.RegExp;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HexFormat;
import java.util.List;
import java.util.Map;

/**
 * Reverse-direction check for lucene-codecs' {@code memory} package: {@code
 * write_memory_fixture} (Rust) writes an {@code FSTPostingsFormat} group ({@code .tfp} plus
 * {@code Lucene104} postings) and a {@code DirectPostingsFormat} group ({@code Lucene104}'s
 * files), and real Lucene opens each through its own {@code fieldsProducer} with a hand-built
 * {@link SegmentInfo}/{@link FieldInfos}, then:
 *
 * <ul>
 *   <li>runs {@code checkIntegrity} (every checksum);
 *   <li>walks every term with {@code next()}, requiring the exact term list, each term's {@code
 *       docFreq}/{@code totalTermFreq}, and every document's freq, positions, offsets and
 *       payloads through {@code postings(ALL)} to equal what Rust wrote;
 *   <li>{@code seekExact}s every term;
 *   <li>runs {@code Terms.intersect} for every pattern and start term Rust ran, requiring the
 *       exact terms (with docFreq/totalTermFreq) Rust's own reader enumerated.
 * </ul>
 *
 * <p>What it cannot catch: anything that binds these files into a segment ({@code
 * PerFieldPostingsFormat} attributes, {@code SegmentInfo.files}), since this port's IndexWriter
 * does not route a field to these formats.
 */
public class VerifyCodecsMemory {
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
      failures += verify(directory, si, fis, new FSTPostingsFormat(), "FST50_0", dir);
      failures += verify(directory, si, fis, new DirectPostingsFormat(), "Direct_0", dir);
    }
    if (failures > 0) {
      System.out.println(failures + " failure(s)");
      System.exit(1);
    }
    System.out.println("FST50 and Direct groups verified against real Lucene. PASS");
  }

  static String hex(BytesRef b) {
    return HexFormat.of().formatHex(b.bytes, b.offset, b.offset + b.length);
  }

  /** {@code GenCodecsMemory.dumpPostings}' line for the enum's current term. */
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
      for (String line : Files.readAllLines(dir.resolve("terms." + suffix + ".tsv"))) {
        String[] e = line.split("\t", 3);
        if (!e[0].equals(current)) {
          if (walk != null && walk.next() != null) {
            System.out.println(suffix + ": extra term in " + current);
            failures++;
          }
          current = e[0];
          terms = fields.terms(current);
          walk = terms.iterator();
        }
        BytesRef want = new BytesRef(HexFormat.of().parseHex(e[1]));
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
        termCount++;
      }
      if (walk != null && walk.next() != null) {
        System.out.println(suffix + ": extra term in " + current);
        failures++;
      }

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
