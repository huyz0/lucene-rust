import org.apache.lucene.codecs.FieldsConsumer;
import org.apache.lucene.codecs.FieldsProducer;
import org.apache.lucene.codecs.NormsProducer;
import org.apache.lucene.codecs.PostingsFormat;
import org.apache.lucene.codecs.lucene104.Lucene104Codec;
import org.apache.lucene.codecs.simpletext.SimpleTextCodec;
import org.apache.lucene.index.DocValuesSkipIndexType;
import org.apache.lucene.index.DocValuesType;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.NumericDocValues;
import org.apache.lucene.index.PostingsEnum;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.Terms;
import org.apache.lucene.index.TermsEnum;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.util.BytesRef;
import org.apache.lucene.util.InfoStream;
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
 * Reverse-direction check for lucene-codecs' {@code simpletext} postings: {@code
 * write_simpletext_fixture} (Rust) writes {@code _0_SimpleText_0.pst} (docs-only fields with the
 * empty term and with binary terms holding {@code \n}, {@code \} and {@code 0xff}; a freqs field
 * and a positions/offsets/payloads field, both with norms), and real Lucene opens it through its
 * own {@code SimpleTextCodec().postingsFormat()} with a hand-built {@link SegmentInfo}/{@link
 * FieldInfos}, then:
 *
 * <ul>
 *   <li>(at open) checks the {@code checksum} line, and runs {@code checkIntegrity};
 *   <li>walks every term with {@code next()}, requiring the exact term list, {@code
 *       docFreq}/{@code totalTermFreq}, and every document's freq, positions, offsets and payloads
 *       through {@code postings(ALL)} to equal what Rust wrote, and {@code seekExact}s each term
 *       on a fresh enum;
 *   <li>re-writes the file with its own {@code SimpleTextFieldsWriter} from its own reader and the
 *       norms Rust used, as {@code _0_SimpleText_1.pst}, and requires the two files to be equal
 *       byte for byte -- which pins every skip list, its levels and its competitive impacts.
 * </ul>
 *
 * <p>What it cannot catch: anything that binds the file into a segment ({@code
 * PerFieldPostingsFormat} attributes, {@code SegmentInfo.files}), since this port's IndexWriter
 * does not route a field to SimpleText.
 */
public class VerifyCodecsSimpleText {
  static final String[] NAMES = {"id", "bin", "tag", "body"};

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

    IndexOptions[] options = {
      IndexOptions.DOCS,
      IndexOptions.DOCS,
      IndexOptions.DOCS_AND_FREQS,
      IndexOptions.DOCS_AND_FREQS_AND_POSITIONS_AND_OFFSETS
    };
    FieldInfo[] infos = new FieldInfo[NAMES.length];
    for (int i = 0; i < NAMES.length; i++) {
      infos[i] =
          new FieldInfo(
              NAMES[i],
              i,
              false,
              i < 2,
              i == 3,
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
    Map<String, long[]> norms = new HashMap<>();
    for (String line : Files.readAllLines(dir.resolve("norms.tsv"))) {
      String[] e = line.split("\t");
      long[] values = new long[e.length - 1];
      for (int i = 1; i < e.length; i++) values[i - 1] = Long.parseLong(e[i]);
      norms.put(e[0], values);
    }
    Map<String, List<String>> byField = new HashMap<>();
    for (String line : Files.readAllLines(dir.resolve("terms.tsv"))) {
      byField.computeIfAbsent(line.substring(0, line.indexOf('\t')), k -> new ArrayList<>())
          .add(line);
    }

    int failures = 0;
    PostingsFormat format = new SimpleTextCodec().postingsFormat();
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
      SegmentReadState read =
          new SegmentReadState(directory, si, fis, IOContext.DEFAULT, "SimpleText_0");
      try (FieldsProducer fields = format.fieldsProducer(read)) {
        fields.checkIntegrity();
        long termCount = 0;
        for (String field : List.of("bin", "body", "id", "tag")) {
          List<String> lines = byField.get(field);
          failures += verifyField(field, fields.terms(field), lines);
          termCount += lines.size();
        }
        System.out.println("SimpleText_0: " + termCount + " terms, ok");

        SegmentWriteState write =
            new SegmentWriteState(
                InfoStream.NO_OUTPUT, directory, si, fis, null, IOContext.DEFAULT, "SimpleText_1");
        try (FieldsConsumer consumer = format.fieldsConsumer(write)) {
          consumer.write(fields, new ArrayNorms(norms));
        }
      }
    }
    byte[] rust = Files.readAllBytes(dir.resolve("_0_SimpleText_0.pst"));
    byte[] java = Files.readAllBytes(dir.resolve("_0_SimpleText_1.pst"));
    if (!Arrays.equals(rust, java)) {
      System.out.println("re-written .pst differs: Rust " + rust.length + " bytes, Java "
          + java.length + ", first difference at " + Arrays.mismatch(rust, java));
      failures++;
    }
    if (failures > 0) {
      System.out.println(failures + " failure(s)");
      System.exit(1);
    }
    System.out.println("SimpleText postings verified against real Lucene. PASS");
  }

  /** Norms out of {@code norms.tsv}, every document holding one. */
  static final class ArrayNorms extends NormsProducer {
    final Map<String, long[]> norms;

    ArrayNorms(Map<String, long[]> norms) {
      this.norms = norms;
    }

    @Override
    public NumericDocValues getNorms(FieldInfo field) {
      long[] values = norms.get(field.name);
      if (values == null) throw new IllegalStateException("no norms for " + field.name);
      return new NumericDocValues() {
        int doc = -1;

        @Override
        public long longValue() {
          return values[doc];
        }

        @Override
        public boolean advanceExact(int target) {
          doc = target;
          return target < values.length;
        }

        @Override
        public int docID() {
          return doc;
        }

        @Override
        public int nextDoc() {
          return advance(doc + 1);
        }

        @Override
        public int advance(int target) {
          doc = target < values.length ? target : DocIdSetIterator.NO_MORE_DOCS;
          return doc;
        }

        @Override
        public long cost() {
          return values.length;
        }
      };
    }

    @Override
    public void checkIntegrity() {}

    @Override
    public void close() {}
  }

  static String hex(BytesRef b) {
    return HexFormat.of().formatHex(b.bytes, b.offset, b.offset + b.length);
  }

  /** {@code write_simpletext_fixture}'s line for the enum's current term. */
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

  static int verifyField(String field, Terms terms, List<String> lines) throws Exception {
    int failures = 0;
    TermsEnum walk = terms.iterator();
    for (String line : lines) {
      String[] e = line.split("\t", 5);
      BytesRef want = new BytesRef(HexFormat.of().parseHex(e[1]));
      BytesRef got = walk.next();
      if (got == null || !got.equals(want)) {
        System.out.println(field + " next() " + got + " expected " + want);
        return failures + 1;
      }
      String gotLine = termLine(field, terms, walk);
      if (!gotLine.equals(line)) {
        System.out.println(field + " " + e[1] + "\n  got  " + gotLine + "\n  want " + line);
        failures++;
      }
      if (!terms.iterator().seekExact(want)) {
        System.out.println(field + " seekExact missed " + e[1]);
        failures++;
      }
    }
    if (walk.next() != null) {
      System.out.println("extra term in " + field);
      failures++;
    }
    return failures;
  }
}
