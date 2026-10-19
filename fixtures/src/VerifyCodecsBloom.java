import org.apache.lucene.codecs.FieldsProducer;
import org.apache.lucene.codecs.bloom.BloomFilteringPostingsFormat;
import org.apache.lucene.codecs.bloom.FuzzySet;
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
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.util.BytesRef;
import org.apache.lucene.util.Version;

import java.lang.reflect.Field;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Collections;
import java.util.HashMap;
import java.util.HexFormat;
import java.util.Map;

/**
 * Reverse-direction check for lucene-codecs' {@code BloomFilteringPostingsFormat}:
 * {@code write_bloom_fixture} (Rust) writes two {@code .blm} groups with their
 * {@code Lucene104PostingsFormat} delegate files, and real Lucene opens each
 * through {@code BloomFilteringPostingsFormat.fieldsProducer} with a hand-built
 * {@link SegmentInfo}/{@link FieldInfos}, then:
 *
 * <ul>
 *   <li>opens the group (which verifies the {@code .blm}'s checksum) and runs
 *       {@code checkIntegrity} (the delegate's checksums);
 *   <li>reads the deserialized filters off the producer (reflection: the map is
 *       package-private) and requires exactly the expected fields, each one's
 *       {@code toString} (hash count, set bits, size) equal to this port's;
 *   <li>requires {@code contains} to answer {@code MAYBE} for every term written
 *       (a filter that lies is a lost document) and to give, for 3000 absent
 *       terms per field, exactly this port's {@code MAYBE}/{@code NO} -- which
 *       pins the bit positions, not only their count;
 *   <li>walks every term with {@code next()} and its documents with {@code
 *       postings}, and {@code seekExact}s every term (present) and every probe
 *       (absent) through the filtered enum.
 * </ul>
 *
 * <p>What it cannot catch: anything that binds these files into a segment
 * ({@code PerFieldPostingsFormat} attributes, {@code SegmentInfo.files}), since
 * this port's IndexWriter does not route a field to this format.
 */
public class VerifyCodecsBloom {
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
      IndexOptions.DOCS, IndexOptions.DOCS_AND_FREQS, IndexOptions.DOCS_AND_FREQS_AND_POSITIONS
    };
    FieldInfo[] infos = new FieldInfo[names.length];
    for (int i = 0; i < names.length; i++) {
      infos[i] =
          new FieldInfo(
              names[i],
              i,
              false,
              false,
              false,
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
      failures += verify(directory, si, fis, "BloomFilter_0", dir);
      failures += verify(directory, si, fis, "BloomFilter_1", dir);
    }
    if (failures > 0) {
      System.out.println(failures + " failure(s)");
      System.exit(1);
    }
    System.out.println("BloomFilter groups verified against real Lucene. PASS");
  }

  @SuppressWarnings("unchecked")
  static int verify(Directory directory, SegmentInfo si, FieldInfos fis, String suffix, Path dir)
      throws Exception {
    int failures = 0;
    SegmentReadState state = new SegmentReadState(directory, si, fis, IOContext.DEFAULT, suffix);
    try (FieldsProducer fields = new BloomFilteringPostingsFormat().fieldsProducer(state)) {
      fields.checkIntegrity();
      Field f = fields.getClass().getDeclaredField("bloomsByFieldName");
      f.setAccessible(true);
      Map<String, FuzzySet> blooms = (Map<String, FuzzySet>) f.get(fields);

      Map<String, String> expectedBlooms = new HashMap<>();
      for (String line : Files.readAllLines(dir.resolve("blooms." + suffix + ".tsv"))) {
        String[] e = line.split("\t");
        expectedBlooms.put(e[0], e[1]);
      }
      if (!blooms.keySet().equals(expectedBlooms.keySet())) {
        System.out.println(suffix + ": filters " + blooms.keySet() + " expected " + expectedBlooms.keySet());
        failures++;
      }
      for (Map.Entry<String, String> e : expectedBlooms.entrySet()) {
        FuzzySet set = blooms.get(e.getKey());
        if (set == null || !set.toString().equals(e.getValue())) {
          System.out.println(suffix + ": " + e.getKey() + " " + set + " expected " + e.getValue());
          failures++;
        }
      }

      // Every term, walked and seeked, with its documents.
      String current = null;
      TermsEnum walk = null;
      Terms terms = null;
      PostingsEnum pe = null;
      long termCount = 0;
      for (String line : Files.readAllLines(dir.resolve("terms." + suffix + ".tsv"))) {
        String[] e = line.split("\t");
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
        FuzzySet set = blooms.get(current);
        if (set != null && set.contains(want) != FuzzySet.ContainsResult.MAYBE) {
          System.out.println(suffix + ": " + current + " filter refuses present term " + e[1]);
          failures++;
        }
        String[] docs = e[2].split(",");
        pe = walk.postings(pe, PostingsEnum.NONE);
        for (String d : docs) {
          if (pe.nextDoc() != Integer.parseInt(d)) {
            System.out.println(suffix + ": " + current + " " + e[1] + " docs differ");
            failures++;
            break;
          }
        }
        if (pe.nextDoc() != DocIdSetIterator.NO_MORE_DOCS || walk.docFreq() != docs.length) {
          System.out.println(suffix + ": " + current + " " + e[1] + " docFreq " + walk.docFreq());
          failures++;
        }
        TermsEnum seek = terms.iterator();
        if (!seek.seekExact(want) || seek.docFreq() != docs.length) {
          System.out.println(suffix + ": " + current + " seekExact missed " + e[1]);
          failures++;
        }
        termCount++;
      }
      if (walk != null && walk.next() != null) {
        System.out.println(suffix + ": extra term in " + current);
        failures++;
      }

      // Absent terms: this port's filter answers, and seekExact misses.
      long probes = 0;
      for (String line : Files.readAllLines(dir.resolve("probes." + suffix + ".tsv"))) {
        String[] e = line.split("\t");
        BytesRef t = new BytesRef(HexFormat.of().parseHex(e[1]));
        FuzzySet set = blooms.get(e[0]);
        String answer = set == null ? "-" : set.contains(t).name();
        if (!answer.equals(e[2])) {
          System.out.println(suffix + ": " + e[0] + " contains(" + e[1] + ") " + answer + " expected " + e[2]);
          failures++;
        }
        if (fields.terms(e[0]).iterator().seekExact(t)) {
          System.out.println(suffix + ": " + e[0] + " absent term " + e[1] + " found");
          failures++;
        }
        probes++;
      }
      System.out.println(
          suffix + ": " + blooms.size() + " filters, " + termCount + " terms, " + probes + " probes, ok");
    }
    return failures;
  }
}
