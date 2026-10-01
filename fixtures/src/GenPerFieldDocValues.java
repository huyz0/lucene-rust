import org.apache.lucene.codecs.DocValuesFormat;
import org.apache.lucene.codecs.lucene104.Lucene104Codec;
import org.apache.lucene.codecs.lucene90.Lucene90DocValuesFormat;
import org.apache.lucene.document.BinaryDocValuesField;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.NumericDocValuesField;
import org.apache.lucene.document.SortedDocValuesField;
import org.apache.lucene.document.SortedSetDocValuesField;
import org.apache.lucene.index.CheckIndex;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.MergePolicy;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.SegmentCommitInfo;
import org.apache.lucene.index.SegmentInfos;
import org.apache.lucene.index.SegmentReader;
import org.apache.lucene.index.TieredMergePolicy;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.BytesRef;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.TreeMap;

/**
 * {@code PerFieldDocValuesFormat} write routing across flush and merge: fields whose name starts
 * with {@code s_} go to {@code Lucene90DocValuesFormat(16)} (a skip-index interval of 16 documents
 * instead of 4096), every other field to the default {@code Lucene90DocValuesFormat()}.
 *
 * <p>Five doc-values fields, in this order in every document: {@code d_num} (NUMERIC, skip index,
 * default), {@code s_num} (NUMERIC, skip index, routed, missing from every fifth document), {@code
 * s_key} (SORTED, skip index, routed), {@code d_set} (SORTED_SET, default) and {@code s_bin}
 * (BINARY, routed). The instance a flush reaches first is {@code Lucene90_0}; the flush visits
 * fields in {@code IndexingChain}'s field-hash order and a merge in field-number order.
 *
 * <p>{@code per_field_doc_values/flushed}: two flushed segments of {@link #PER_SEGMENT} documents.
 * {@code per_field_doc_values/merged}: the same documents force-merged into one segment. {@code
 * manifest.txt} in each: per segment its doc-values files and each doc-values field's
 * {@code PerFieldDocValuesFormat} attributes. The Rust test writes the same documents with the same
 * routing and compares the files byte for byte (segment id normalised); the values themselves are
 * produced by {@link #value}.
 */
public class GenPerFieldDocValues {
  static final int PER_SEGMENT = 300;
  static final String[] WORDS = {
    "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliet"
  };

  /** Document {@code i}'s values; the Rust test computes the same. */
  static long value(int i, int k) {
    long x = (i + 1) * 2654435761L + k * 40503L;
    return (x ^ (x >>> 13)) % 100000;
  }

  public static void main(String[] args) throws IOException {
    Path root = Path.of(args[0]).resolve("per_field_doc_values");
    if (Files.exists(root)) {
      try (var walk = Files.walk(root)) {
        walk.sorted(Comparator.reverseOrder()).forEach(p -> p.toFile().delete());
      }
    }
    write(root.resolve("flushed"), false);
    write(root.resolve("merged"), true);
    System.out.println("wrote per_field_doc_values/");
  }

  static Document doc(int i) {
    Document d = new Document();
    d.add(NumericDocValuesField.indexedField("d_num", value(i, 0)));
    if (i % 5 != 0) {
      d.add(NumericDocValuesField.indexedField("s_num", value(i, 1) / 16));
    }
    d.add(SortedDocValuesField.indexedField("s_key", new BytesRef(WORDS[(int) (value(i, 2) % WORDS.length)])));
    d.add(new SortedSetDocValuesField("d_set", new BytesRef(WORDS[(int) (value(i, 3) % WORDS.length)])));
    d.add(new SortedSetDocValuesField("d_set", new BytesRef(WORDS[(int) (value(i, 4) % WORDS.length)])));
    d.add(new BinaryDocValuesField("s_bin", new BytesRef("b" + value(i, 5))));
    return d;
  }

  static void write(Path out, boolean merge) throws IOException {
    Files.createDirectories(out);
    DocValuesFormat small = new Lucene90DocValuesFormat(16);
    DocValuesFormat standard = new Lucene90DocValuesFormat();
    Lucene104Codec codec =
        new Lucene104Codec() {
          @Override
          public DocValuesFormat getDocValuesFormatForField(String field) {
            return field.startsWith("s_") ? small : standard;
          }
        };
    try (Directory dir = FSDirectory.open(out)) {
      IndexWriterConfig cfg = new IndexWriterConfig();
      cfg.setCodec(codec);
      cfg.setUseCompoundFile(false);
      cfg.setRAMBufferSizeMB(256);
      MergePolicy policy;
      if (merge) {
        TieredMergePolicy tmp = new TieredMergePolicy();
        tmp.setNoCFSRatio(0.0);
        policy = tmp;
      } else {
        policy = NoMergePolicy.INSTANCE;
      }
      cfg.setMergePolicy(policy);
      cfg.setMaxFullFlushMergeWaitMillis(0);
      try (IndexWriter w = new IndexWriter(dir, cfg)) {
        for (int seg = 0; seg < 2; seg++) {
          for (int i = seg * PER_SEGMENT; i < (seg + 1) * PER_SEGMENT; i++) {
            w.addDocument(doc(i));
          }
          w.commit();
        }
        if (merge) {
          w.forceMerge(1);
          w.commit();
        }
      }
      List<String> lines = new ArrayList<>();
      SegmentInfos sis = SegmentInfos.readLatestCommit(dir);
      try (DirectoryReader reader = DirectoryReader.open(dir)) {
        for (LeafReaderContext leaf : reader.leaves()) {
          SegmentCommitInfo sci = ((SegmentReader) leaf.reader()).getSegmentInfo();
          StringBuilder files = new StringBuilder("segment " + sci.info.name + " files");
          new java.util.TreeSet<>(sci.files()).stream()
              .filter(f -> f.contains("Lucene90"))
              .forEach(f -> files.append(' ').append(f));
          lines.add(files.toString());
          for (FieldInfo fi : leaf.reader().getFieldInfos()) {
            lines.add(
                "field "
                    + sci.info.name
                    + " "
                    + fi.name
                    + " "
                    + fi.number
                    + " "
                    + fi.getAttribute("PerFieldDocValuesFormat.format")
                    + " "
                    + fi.getAttribute("PerFieldDocValuesFormat.suffix"));
          }
        }
      }
      Files.write(out.resolve("manifest.txt"), lines, StandardCharsets.UTF_8);
      CheckIndex.Status status;
      try (CheckIndex check = new CheckIndex(dir)) {
        status = check.checkIndex();
      }
      if (!status.clean) {
        throw new AssertionError("CheckIndex failed on " + out);
      }
    }
  }
}
