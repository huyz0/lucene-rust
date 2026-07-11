import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.SortedNumericDocValuesField;
import org.apache.lucene.document.SortedSetDocValuesField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.SegmentCommitInfo;
import org.apache.lucene.index.SegmentInfos;
import org.apache.lucene.index.SortedNumericDocValues;
import org.apache.lucene.index.SortedSetDocValues;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.BytesRef;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Generates real `.dvm`/`.dvd` (Lucene90DocValuesFormat) fixtures for
 * multi-valued doc values: a SORTED_NUMERIC field ("nums", 0-3 values per
 * doc) and a SORTED_SET field ("tags", 0-2 values per doc sharing a small
 * terms dictionary), across 5 docs, so some docs have zero values (the
 * IndexedDISI-sparse path, since not every doc has the field at all) and
 * others have more than one (the DirectMonotonicReader address-range path).
 */
public class GenMultiValuedDocValues {
  public static void main(String[] args) throws IOException {
    Path out = Path.of(args[0]).resolve("multi_valued_dv_index");
    if (Files.exists(out)) {
      deleteRecursive(out);
    }
    Files.createDirectories(out);

    try (Directory dir = FSDirectory.open(out)) {
      IndexWriterConfig cfg = new IndexWriterConfig();
      cfg.setUseCompoundFile(false);
      cfg.setMergePolicy(NoMergePolicy.INSTANCE);

      long[][] nums = {{5, 10}, {}, {7}, {1, 2, 3}, {}};
      String[][] tags = {
        {"red", "blue"}, {}, {"green"}, {"blue"}, {"red", "green"}
      };

      try (IndexWriter w = new IndexWriter(dir, cfg)) {
        for (int i = 0; i < nums.length; i++) {
          Document doc = new Document();
          doc.add(new StringField("id", Integer.toString(i), Field.Store.NO));
          for (long v : nums[i]) {
            doc.add(new SortedNumericDocValuesField("nums", v));
          }
          for (String t : tags[i]) {
            doc.add(
                new SortedSetDocValuesField(
                    "tags", new BytesRef(t.getBytes(StandardCharsets.UTF_8))));
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
      for (String f : sci.info.files()) {
        if (f.endsWith(".dvm")) dvmFileName = f;
        if (f.endsWith(".dvd")) dvdFileName = f;
        if (f.endsWith(".fnm")) fnmFileName = f;
      }
      if (dvmFileName == null || dvdFileName == null || fnmFileName == null) {
        throw new AssertionError("expected .dvm/.dvd/.fnm files, files=" + sci.info.files());
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

      org.apache.lucene.index.FieldInfo numsField = fis.fieldInfo("nums");
      SortedNumericDocValues sndv = dvProducer.getSortedNumeric(numsField);
      StringBuilder numsOut = new StringBuilder();
      for (int doc = 0; doc < sci.info.maxDoc(); doc++) {
        if (doc > 0) numsOut.append(';');
        if (sndv.advanceExact(doc)) {
          for (int i = 0; i < sndv.docValueCount(); i++) {
            if (i > 0) numsOut.append(',');
            numsOut.append(sndv.nextValue());
          }
        } else {
          numsOut.append("NONE");
        }
      }
      m.append("field.nums.values=").append(numsOut).append('\n');

      org.apache.lucene.index.FieldInfo tagsField = fis.fieldInfo("tags");
      SortedSetDocValues ssdv = dvProducer.getSortedSet(tagsField);
      StringBuilder tagsOut = new StringBuilder();
      for (int doc = 0; doc < sci.info.maxDoc(); doc++) {
        if (doc > 0) tagsOut.append(';');
        if (ssdv.advanceExact(doc)) {
          for (int i = 0; i < ssdv.docValueCount(); i++) {
            if (i > 0) tagsOut.append(',');
            tagsOut.append(ssdv.nextOrd());
          }
        } else {
          tagsOut.append("NONE");
        }
      }
      m.append("field.tags.ords=").append(tagsOut).append('\n');

      long tagsValueCount = ssdv.getValueCount();
      m.append("field.tags.value_count=").append(tagsValueCount).append('\n');
      StringBuilder tagsTerms = new StringBuilder();
      for (long ord = 0; ord < tagsValueCount; ord++) {
        if (ord > 0) tagsTerms.append(',');
        tagsTerms.append(ssdv.lookupOrd(ord).utf8ToString());
      }
      m.append("field.tags.terms=").append(tagsTerms).append('\n');

      dvProducer.close();

      Files.writeString(out.resolve("manifest.properties"), m.toString());
    }

    System.out.println("wrote multi_valued_dv_index/ fixture directory");
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
