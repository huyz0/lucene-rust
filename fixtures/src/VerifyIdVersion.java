import org.apache.lucene.codecs.FieldsProducer;
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
import org.apache.lucene.sandbox.codecs.idversion.IDVersionPostingsFormat;
import org.apache.lucene.sandbox.codecs.idversion.IDVersionSegmentTermsEnum;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.util.BytesRef;
import org.apache.lucene.util.Version;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HexFormat;
import java.util.List;
import java.util.Map;

/**
 * Reverse-direction check for lucene-sandbox's {@code IDVersionPostingsFormat}:
 * {@code write_idversion_fixture} (Rust) writes two {@code .tiv}/{@code .tipv}
 * groups, and real Lucene opens each through {@code
 * IDVersionPostingsFormat.fieldsProducer} with a hand-built {@link SegmentInfo}/
 * {@link FieldInfos}, then:
 *
 * <ul>
 *   <li>runs {@code checkIntegrity} (both files' checksums);
 *   <li>checks the field's size, docCount, sumDocFreq and sumTotalTermFreq;
 *   <li>walks every term with {@code next()}, requiring the expected term,
 *       {@code getVersion()}, document and payload -- the walk decodes every
 *       block's suffixes and the first-term {@code vLong}/{@code zLong}-delta
 *       versions;
 *   <li>on a fresh enum per sample, {@code seekExact(id, version)} must hit and
 *       {@code seekExact(id, version + 1)} must miss -- the miss is decided by
 *       the terms index's per-prefix {@code Long.MAX_VALUE - maxVersion}
 *       outputs as often as by the block itself, so an FST output written wrong
 *       fails here even when the blocks are right;
 *   <li>every deleted id must be absent.
 * </ul>
 *
 * <p>What it cannot catch: anything that binds these files into a segment
 * ({@code PerFieldPostingsFormat} attributes, {@code SegmentInfo.files}),
 * since this port's IndexWriter does not route a field to this format.
 */
public class VerifyIdVersion {
  public static void main(String[] args) throws IOException {
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

    FieldInfo[] infos = new FieldInfo[2];
    String[] names = {"id", "id2"};
    for (int i = 0; i < 2; i++) {
      infos[i] =
          new FieldInfo(
              names[i],
              i,
              false,
              true,
              true, // storePayloads
              IndexOptions.DOCS_AND_FREQS_AND_POSITIONS,
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
          verify(directory, si, fis, "IDVersion_0", "id", new IDVersionPostingsFormat(), dir);
      failures +=
          verify(
              directory, si, fis, "IDVersion_1", "id2", new IDVersionPostingsFormat(4, 9), dir);
    }
    if (failures > 0) {
      System.out.println(failures + " failure(s)");
      System.exit(1);
    }
    System.out.println("IDVersion groups verified against real Lucene. PASS");
  }

  static int verify(
      Directory directory,
      SegmentInfo si,
      FieldInfos fis,
      String suffix,
      String field,
      IDVersionPostingsFormat format,
      Path dir)
      throws IOException {
    List<String[]> expected = new ArrayList<>();
    for (String line : Files.readAllLines(dir.resolve("expected." + suffix + ".tsv"))) {
      expected.add(line.split("\t"));
    }
    List<String> deleted = Files.readAllLines(dir.resolve("deleted." + suffix + ".tsv"));

    int failures = 0;
    SegmentReadState state = new SegmentReadState(directory, si, fis, IOContext.DEFAULT, suffix);
    try (FieldsProducer fields = format.fieldsProducer(state)) {
      fields.checkIntegrity();
      if (fields.size() != 1) {
        System.out.println(suffix + ": size " + fields.size());
        failures++;
      }
      Terms terms = fields.terms(field);
      long n = expected.size();
      if (terms.size() != n
          || terms.getDocCount() != n
          || terms.getSumDocFreq() != n
          || terms.getSumTotalTermFreq() != n) {
        System.out.println(
            suffix
                + ": stats size="
                + terms.size()
                + " docCount="
                + terms.getDocCount()
                + " sumDocFreq="
                + terms.getSumDocFreq()
                + " sumTotalTermFreq="
                + terms.getSumTotalTermFreq()
                + " expected "
                + n);
        failures++;
      }

      IDVersionSegmentTermsEnum walk = (IDVersionSegmentTermsEnum) terms.iterator();
      PostingsEnum pe = null;
      for (String[] e : expected) {
        BytesRef want = new BytesRef(HexFormat.of().parseHex(e[0]));
        int doc = Integer.parseInt(e[1]);
        long version = Long.parseLong(e[2]);
        BytesRef got = walk.next();
        if (got == null || !got.equals(want)) {
          System.out.println(suffix + ": next() " + got + " expected " + want);
          return failures + 1;
        }
        pe = walk.postings(pe, PostingsEnum.PAYLOADS);
        int gotDoc = pe.nextDoc();
        pe.nextPosition();
        long payload = IDVersionPostingsFormat.bytesToLong(pe.getPayload());
        if (walk.getVersion() != version
            || walk.docFreq() != 1
            || gotDoc != doc
            || payload != version
            || pe.nextDoc() != DocIdSetIterator.NO_MORE_DOCS) {
          System.out.println(
              suffix
                  + ": "
                  + e[0]
                  + " doc="
                  + gotDoc
                  + " version="
                  + walk.getVersion()
                  + " payload="
                  + payload
                  + " expected doc="
                  + doc
                  + " version="
                  + version);
          failures++;
        }
      }
      if (walk.next() != null) {
        System.out.println(suffix + ": extra term after the last");
        failures++;
      }

      for (int i = 0; i < expected.size(); i += 7) {
        String[] e = expected.get(i);
        BytesRef term = new BytesRef(HexFormat.of().parseHex(e[0]));
        long version = Long.parseLong(e[2]);
        IDVersionSegmentTermsEnum te = (IDVersionSegmentTermsEnum) terms.iterator();
        if (!te.seekExact(term, version) || te.getVersion() != version) {
          System.out.println(suffix + ": seekExact(" + e[0] + ", " + version + ") missed");
          failures++;
        }
        te = (IDVersionSegmentTermsEnum) terms.iterator();
        if (version < IDVersionPostingsFormat.MAX_VERSION && te.seekExact(term, version + 1)) {
          System.out.println(suffix + ": seekExact(" + e[0] + ", " + (version + 1) + ") hit");
          failures++;
        }
      }
      TermsEnum te = terms.iterator();
      for (String d : deleted) {
        if (te.seekExact(new BytesRef(HexFormat.of().parseHex(d)))) {
          System.out.println(suffix + ": deleted id " + d + " present");
          failures++;
        }
      }
      System.out.println(
          suffix + ": " + n + " terms, " + deleted.size() + " deleted ids absent, ok");
    }
    return failures;
  }
}
