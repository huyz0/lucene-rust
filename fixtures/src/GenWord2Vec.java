import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Base64;
import java.util.List;
import java.util.Locale;
import java.util.Random;
import java.util.zip.CRC32;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.Tokenizer;
import org.apache.lucene.analysis.core.WhitespaceTokenizer;
import org.apache.lucene.analysis.synonym.word2vec.Dl4jModelReader;
import org.apache.lucene.analysis.synonym.word2vec.TermAndBoost;
import org.apache.lucene.analysis.synonym.word2vec.Word2VecModel;
import org.apache.lucene.analysis.synonym.word2vec.Word2VecSynonymFilter;
import org.apache.lucene.analysis.synonym.word2vec.Word2VecSynonymProvider;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.analysis.tokenattributes.OffsetAttribute;
import org.apache.lucene.analysis.tokenattributes.PositionIncrementAttribute;
import org.apache.lucene.analysis.tokenattributes.PositionLengthAttribute;
import org.apache.lucene.analysis.tokenattributes.TypeAttribute;
import org.apache.lucene.util.BytesRef;

/**
 * M11 T11.5: the {@code word2vec} synonym package, recorded for
 * crates/lucene-search/tests/word2vec_fixtures.rs.
 *
 * <p>Two Deeplearning4j-style model zips written here from seeded clustered vectors (80 terms, 12
 * dimensions): {@code model_b64.zip} (a {@code config.json} entry, then a deflated {@code syn0.txt}
 * with {@code B64:} terms and data descriptors) and {@code model_plain.zip} (one stored {@code
 * syn0.txt}, raw terms). {@code synonyms.txt}: {@code Word2VecSynonymProvider.getSynonyms} of every
 * term and a missing one, {@code term<TAB>max<TAB>min<TAB>synonym:boostBits ...}, for both models.
 * {@code filter.txt}: {@code Word2VecSynonymFilter} over {@link #LINES} after a {@code
 * WhitespaceTokenizer}, a row per token {@code line<TAB>term<TAB>start<TAB>end<TAB>posInc<TAB>
 * posLen<TAB>type}.
 */
public class GenWord2Vec {
  static final int TERMS = 80;
  static final int DIM = 12;
  /** Every entry's timestamp, fixed: the zip bytes are part of the fixture. */
  static final java.time.LocalDateTime TIME = java.time.LocalDateTime.of(2026, 1, 1, 0, 0);

  static String term(int i) {
    return switch (i) {
      case 7 -> "école";
      case 13 -> "日本";
      case 21 -> "a_b";
      default -> String.format(Locale.ROOT, "w%02d", i);
    };
  }

  static float[][] vectors() {
    Random r = new Random(7);
    float[][] centers = new float[8][DIM];
    for (float[] c : centers) for (int d = 0; d < DIM; d++) c[d] = (float) r.nextGaussian();
    float[][] v = new float[TERMS][DIM];
    for (int i = 0; i < TERMS; i++) {
      float[] c = centers[i % 8];
      for (int d = 0; d < DIM; d++) v[i][d] = c[d] + 0.35f * (float) r.nextGaussian();
    }
    return v;
  }

  static String syn0(boolean b64) {
    float[][] v = vectors();
    StringBuilder o = new StringBuilder();
    o.append(TERMS).append(' ').append(DIM).append('\n');
    for (int i = 0; i < TERMS; i++) {
      String t = term(i);
      o.append(b64 ? "B64:" + Base64.getEncoder().encodeToString(t.getBytes(StandardCharsets.UTF_8)) : t);
      for (float x : v[i]) o.append(' ').append(x);
      o.append(i % 3 == 0 ? "\r\n" : "\n");
    }
    return o.toString();
  }

  static byte[] zip(boolean b64) throws IOException {
    ByteArrayOutputStream bytes = new ByteArrayOutputStream();
    try (ZipOutputStream z = new ZipOutputStream(bytes)) {
      byte[] model = syn0(b64).getBytes(StandardCharsets.UTF_8);
      if (b64) {
        ZipEntry config = new ZipEntry("config.json");
        config.setTimeLocal(TIME);
        z.putNextEntry(config);
        z.write("{\"layer\":12}".getBytes(StandardCharsets.UTF_8));
        z.closeEntry();
        ZipEntry e = new ZipEntry("syn0.txt");
        e.setTimeLocal(TIME);
        z.putNextEntry(e);
      } else {
        ZipEntry e = new ZipEntry("syn0.txt");
        e.setTimeLocal(TIME);
        e.setMethod(ZipEntry.STORED);
        e.setSize(model.length);
        CRC32 crc = new CRC32();
        crc.update(model);
        e.setCrc(crc.getValue());
        z.putNextEntry(e);
      }
      z.write(model);
      z.closeEntry();
    }
    return bytes.toByteArray();
  }

  static final String[] LINES = {
    "w00 w01 w08 unknown w16",
    "école 日本 a_b",
    "w79",
    ""
  };

  public static void main(String[] args) throws IOException {
    Path out = Path.of(args[0]).resolve("word2vec");
    Files.createDirectories(out);
    StringBuilder syn = new StringBuilder();
    StringBuilder filter = new StringBuilder();
    for (boolean b64 : new boolean[] {true, false}) {
      byte[] z = zip(b64);
      String name = b64 ? "model_b64.zip" : "model_plain.zip";
      Files.write(out.resolve(name), z);
      Word2VecModel model;
      try (Dl4jModelReader reader = new Dl4jModelReader(new ByteArrayInputStream(z))) {
        model = reader.read();
      }
      Word2VecSynonymProvider provider = new Word2VecSynonymProvider(model);
      syn.append("#model\t").append(name).append('\n');
      List<String> terms = new ArrayList<>();
      for (int i = 0; i < TERMS; i++) terms.add(term(i));
      terms.add("missing");
      for (String t : terms) {
        for (Object[] p : new Object[][] {{3, 0.0f}, {10, 0.9f}, {1, 0.5f}}) {
          int max = (Integer) p[0];
          float min = (Float) p[1];
          syn.append(t).append('\t').append(max).append('\t').append(min).append('\t');
          List<String> found = new ArrayList<>();
          for (TermAndBoost tb : provider.getSynonyms(new BytesRef(t), max, min)) {
            found.add(tb.term().utf8ToString() + ":" + Integer.toHexString(Float.floatToIntBits(tb.boost())));
          }
          syn.append(String.join(" ", found)).append('\n');
        }
      }
      filter.append("#model\t").append(name).append('\n');
      for (int ln = 0; ln < LINES.length; ln++) {
        Tokenizer tok = new WhitespaceTokenizer();
        tok.setReader(new java.io.StringReader(LINES[ln]));
        try (TokenStream ts = new Word2VecSynonymFilter(tok, provider, 2, 0.8f)) {
          CharTermAttribute term = ts.addAttribute(CharTermAttribute.class);
          OffsetAttribute off = ts.addAttribute(OffsetAttribute.class);
          PositionIncrementAttribute inc = ts.addAttribute(PositionIncrementAttribute.class);
          PositionLengthAttribute len = ts.addAttribute(PositionLengthAttribute.class);
          TypeAttribute type = ts.addAttribute(TypeAttribute.class);
          ts.reset();
          while (ts.incrementToken()) {
            filter.append(ln).append('\t').append(term).append('\t').append(off.startOffset()).append('\t')
                .append(off.endOffset()).append('\t').append(inc.getPositionIncrement()).append('\t')
                .append(len.getPositionLength()).append('\t').append(type.type()).append('\n');
          }
          ts.end();
        }
      }
    }
    Files.writeString(out.resolve("synonyms.txt"), syn.toString(), StandardCharsets.UTF_8);
    Files.writeString(out.resolve("filter.txt"), filter.toString(), StandardCharsets.UTF_8);
  }
}
