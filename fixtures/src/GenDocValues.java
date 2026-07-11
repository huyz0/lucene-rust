import org.apache.lucene.document.BinaryDocValuesField;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.NumericDocValuesField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.BinaryDocValues;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.NumericDocValues;
import org.apache.lucene.index.SegmentCommitInfo;
import org.apache.lucene.index.SegmentInfos;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.BytesRef;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HexFormat;

/**
 * Generates real `.dvm`/`.dvd` (Lucene90DocValuesFormat) fixtures for
 * NUMERIC and BINARY fields in one unmerged segment:
 *
 * <ul>
 *   <li>"varying": dense numeric, arbitrary signed values (plain
 *       delta-compressed path with negative deltas).
 *   <li>"gcd": dense numeric, values sharing a large GCD (GCD-compressed
 *       path).
 *   <li>"sparse": numeric present on only some docs (IndexedDISI path, same
 *       as GenNorms.java).
 *   <li>"bin_fixed": dense binary, every value the same length (direct
 *       {@code doc * length} addressing, no monotonic address block).
 *   <li>"bin_var": dense binary, varying lengths (DirectMonotonicReader
 *       address block).
 *   <li>"bin_sparse": variable-length binary present on only some docs
 *       (IndexedDISI + DirectMonotonicReader together).
 * </ul>
 */
public class GenDocValues {
  public static void main(String[] args) throws IOException {
    Path out = Path.of(args[0]).resolve("doc_values_index");
    if (Files.exists(out)) {
      deleteRecursive(out);
    }
    Files.createDirectories(out);

    try (Directory dir = FSDirectory.open(out)) {
      IndexWriterConfig cfg = new IndexWriterConfig();
      cfg.setUseCompoundFile(false);
      cfg.setMergePolicy(NoMergePolicy.INSTANCE);

      long[] varying = {-100, 7, 42, 1000, -3};
      long[] gcdVals = {1000, 1025, 1075, 1200, 1050};
      boolean[] hasSparse = {true, false, true, false, true};
      long[] sparseVals = {5, 0, 15, 0, 25};

      String[] binFixed = {"aaaa", "bbbb", "cccc", "dddd", "eeee"};
      String[] binVar = {"a", "bb", "ccc", "dddd", "e"};
      boolean[] hasBinSparse = {true, false, true, false, true};
      String[] binSparse = {"x", "", "yyy", "", "z"};

      try (IndexWriter w = new IndexWriter(dir, cfg)) {
        for (int i = 0; i < varying.length; i++) {
          Document doc = new Document();
          doc.add(new StringField("id", Integer.toString(i), Field.Store.NO));
          doc.add(new NumericDocValuesField("varying", varying[i]));
          doc.add(new NumericDocValuesField("gcd", gcdVals[i]));
          if (hasSparse[i]) {
            doc.add(new NumericDocValuesField("sparse", sparseVals[i]));
          }
          doc.add(
              new BinaryDocValuesField(
                  "bin_fixed", new BytesRef(binFixed[i].getBytes(StandardCharsets.UTF_8))));
          doc.add(
              new BinaryDocValuesField(
                  "bin_var", new BytesRef(binVar[i].getBytes(StandardCharsets.UTF_8))));
          if (hasBinSparse[i]) {
            doc.add(
                new BinaryDocValuesField(
                    "bin_sparse", new BytesRef(binSparse[i].getBytes(StandardCharsets.UTF_8))));
          }
          w.addDocument(doc);
        }
        w.commit();
      }

      SegmentInfos sis = SegmentInfos.readLatestCommit(dir);
      if (sis.size() != 1) {
        throw new AssertionError("expected exactly one segment, got " + sis.size());
      }
      SegmentCommitInfo sci = sis.info(0);

      String dvmFileName = null;
      String dvdFileName = null;
      String fnmFileName = null;
      for (String f : sci.files()) {
        if (f.endsWith(".dvm")) dvmFileName = f;
        if (f.endsWith(".dvd")) dvdFileName = f;
        if (f.endsWith(".fnm")) fnmFileName = f;
      }
      if (dvmFileName == null || dvdFileName == null || fnmFileName == null) {
        throw new AssertionError("expected .dvm/.dvd/.fnm files, files=" + sci.files());
      }

      dump(dir, dvmFileName, out);
      dump(dir, dvdFileName, out);
      dump(dir, fnmFileName, out);

      org.apache.lucene.index.FieldInfos fis =
          sci.info.getCodec().fieldInfosFormat().read(dir, sci.info, "", IOContext.READONCE);

      StringBuilder m = new StringBuilder();
      m.append("dvm_file_name=").append(dvmFileName).append('\n');
      m.append("dvd_file_name=").append(dvdFileName).append('\n');
      m.append("fnm_file_name=").append(fnmFileName).append('\n');
      m.append("segment_name=").append(sci.info.name).append('\n');
      m.append("id_hex=").append(hex(sci.info.getId())).append('\n');
      m.append("max_doc=").append(sci.info.maxDoc()).append('\n');

      StringBuilder fieldNumbers = new StringBuilder();
      for (org.apache.lucene.index.FieldInfo fi : fis) {
        if (fieldNumbers.length() > 0) fieldNumbers.append(',');
        fieldNumbers.append(fi.name).append(':').append(fi.number);
      }
      m.append("field_numbers=").append(fieldNumbers).append('\n');

      org.apache.lucene.codecs.DocValuesProducer dvProducer =
          sci.info
              .getCodec()
              .docValuesFormat()
              .fieldsProducer(
                  new org.apache.lucene.index.SegmentReadState(
                      dir, sci.info, fis, IOContext.READONCE));

      for (String fieldName : new String[] {"varying", "gcd", "sparse"}) {
        org.apache.lucene.index.FieldInfo field = fis.fieldInfo(fieldName);
        NumericDocValues values = dvProducer.getNumeric(field);

        String prefix = "field." + fieldName + ".";
        StringBuilder vals = new StringBuilder();
        for (int doc = 0; doc < sci.info.maxDoc(); doc++) {
          if (doc > 0) vals.append(',');
          if (values.advanceExact(doc)) {
            vals.append(values.longValue());
          } else {
            vals.append("NONE");
          }
        }
        m.append(prefix).append("values=").append(vals).append('\n');
      }

      for (String fieldName : new String[] {"bin_fixed", "bin_var", "bin_sparse"}) {
        org.apache.lucene.index.FieldInfo field = fis.fieldInfo(fieldName);
        BinaryDocValues values = dvProducer.getBinary(field);

        String prefix = "field." + fieldName + ".";
        StringBuilder vals = new StringBuilder();
        for (int doc = 0; doc < sci.info.maxDoc(); doc++) {
          if (doc > 0) vals.append(',');
          if (values.advanceExact(doc)) {
            vals.append(HexFormat.of().formatHex(values.binaryValue().bytes,
                values.binaryValue().offset,
                values.binaryValue().offset + values.binaryValue().length));
          } else {
            vals.append("NONE");
          }
        }
        m.append(prefix).append("values_hex=").append(vals).append('\n');
      }
      dvProducer.close();

      Files.writeString(out.resolve("manifest.properties"), m.toString());
    }

    System.out.println("wrote doc_values_index/ fixture directory");
  }

  static void dump(Directory dir, String fileName, Path out) throws IOException {
    try (IndexInput in = dir.openInput(fileName, IOContext.READONCE)) {
      byte[] bytes = new byte[(int) in.length()];
      in.readBytes(bytes, 0, bytes.length);
      Files.write(out.resolve(fileName + ".raw"), bytes);
    }
  }

  static void deleteRecursive(Path p) throws IOException {
    if (Files.isDirectory(p)) {
      try (var entries = Files.list(p)) {
        for (Path child : (Iterable<Path>) entries::iterator) {
          deleteRecursive(child);
        }
      }
    }
    Files.deleteIfExists(p);
  }

  static String hex(byte[] b) {
    StringBuilder sb = new StringBuilder();
    for (byte x : b) sb.append(String.format("%02x", x));
    return sb.toString();
  }
}
