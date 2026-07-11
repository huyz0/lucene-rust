import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.SortedDocValuesField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.SegmentCommitInfo;
import org.apache.lucene.index.SegmentInfos;
import org.apache.lucene.index.SortedDocValues;
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
 * Generates real `.dvm`/`.dvd` (Lucene90DocValuesFormat) fixtures for a
 * single-valued SORTED field: 5 docs with repeated values ("banana",
 * "apple", "cherry", "apple", "banana") so the terms dictionary has 3
 * unique, alphabetically-ordered terms (ords 0=apple, 1=banana, 2=cherry)
 * and the ordinal array has repeats -- exercising both the terms
 * dictionary decode and the ordinal (NUMERIC-shaped) decode together.
 */
public class GenSortedDocValues {
  public static void main(String[] args) throws IOException {
    Path out = Path.of(args[0]).resolve("sorted_dv_index");
    if (Files.exists(out)) {
      deleteRecursive(out);
    }
    Files.createDirectories(out);

    try (Directory dir = FSDirectory.open(out)) {
      IndexWriterConfig cfg = new IndexWriterConfig();
      cfg.setUseCompoundFile(false);
      cfg.setMergePolicy(NoMergePolicy.INSTANCE);

      String[] values = {"banana", "apple", "cherry", "apple", "banana"};

      try (IndexWriter w = new IndexWriter(dir, cfg)) {
        for (int i = 0; i < values.length; i++) {
          Document doc = new Document();
          doc.add(new StringField("id", Integer.toString(i), Field.Store.NO));
          doc.add(
              new SortedDocValuesField(
                  "sorted", new BytesRef(values[i].getBytes(StandardCharsets.UTF_8))));
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

      org.apache.lucene.index.FieldInfo field = fis.fieldInfo("sorted");
      SortedDocValues sdv = dvProducer.getSorted(field);

      StringBuilder ords = new StringBuilder();
      for (int doc = 0; doc < sci.info.maxDoc(); doc++) {
        if (doc > 0) ords.append(',');
        if (sdv.advanceExact(doc)) {
          ords.append(sdv.ordValue());
        } else {
          ords.append("NONE");
        }
      }
      m.append("field.sorted.ords=").append(ords).append('\n');

      int valueCount = sdv.getValueCount();
      m.append("field.sorted.value_count=").append(valueCount).append('\n');
      StringBuilder terms = new StringBuilder();
      for (int ord = 0; ord < valueCount; ord++) {
        if (ord > 0) terms.append(',');
        BytesRef term = sdv.lookupOrd(ord);
        terms.append(term.utf8ToString());
      }
      m.append("field.sorted.terms=").append(terms).append('\n');

      dvProducer.close();

      Files.writeString(out.resolve("manifest.properties"), m.toString());
    }

    System.out.println("wrote sorted_dv_index/ fixture directory");
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
