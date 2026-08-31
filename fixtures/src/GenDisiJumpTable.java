import org.apache.lucene.document.Document;
import org.apache.lucene.document.NumericDocValuesField;
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

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * A Java-written segment carrying a real {@code IndexedDISI} <b>block jump
 * table</b>, which nothing else in this tree does.
 *
 * <p>{@code c39-codecs-readpath} ported the jump table in both directions and
 * proved the <i>write</i> side against real Lucene
 * ({@code VerifySparseNumericDocValues}' block-jump pass, two independent
 * negative controls). The read side had never run over bytes Lucene wrote, for
 * a fixture reason rather than a design one: {@code IndexedDISI.writeBitSet}
 * emits {@code jumpTableEntryCount = 0} for anything under two logical blocks,
 * and the largest Java-written index in the tree was 36 000 documents against a
 * block size of 65 536. So there was no Java-written table to read.
 *
 * <p>That is the one direction of this format not covered by real bytes, and
 * this sweep has twice found the writer and the reader agreeing on a shared
 * mistake -- the FST framing bug and the invented {@code .si} sort encoding --
 * precisely where only one direction was checked.
 *
 * <p>{@link #NUM_DOCS} is 200 000, so the sparse field spans four logical
 * 65 536-document blocks and Lucene emits a four-entry table. Every third
 * document carries the field, which makes each block DENSE (21 846 present
 * documents per block, well over {@code MAX_ARRAY_LENGTH = 4095}) -- the shape
 * whose block header the jump table lets a seek skip. A second field,
 * {@code very_sparse}, is present on every 20 000th document only, so its
 * blocks are SPARSE and one logical block (the fourth) is empty, exercising
 * {@code flushBlockJumps}' empty-block fill.
 *
 * <p>The manifest records values only for a sample of documents: 200 000 lines
 * would dominate the fixture, and the property under test is that a *seek*
 * several blocks ahead lands on the right value, not that a full scan does
 * (which {@code doc_values_index} already covers over dense bytes).
 */
public class GenDisiJumpTable {

  private static final int NUM_DOCS = 200_000;
  /** Every third document carries `sparse` -- DENSE blocks. */
  private static final int SPARSE_STEP = 3;
  /** Every 20 000th document carries `very_sparse` -- SPARSE blocks, and the last block empty. */
  private static final int VERY_SPARSE_STEP = 20_000;

  public static void main(String[] args) throws IOException {
    Path out = Path.of(args[0]).resolve("disi_jump_table_index");
    if (Files.exists(out)) {
      deleteRecursive(out);
    }
    Files.createDirectories(out);

    try (Directory dir = FSDirectory.open(out)) {
      IndexWriterConfig cfg = new IndexWriterConfig();
      cfg.setUseCompoundFile(false);
      cfg.setMergePolicy(NoMergePolicy.INSTANCE);
      // One segment, or the jump table is split across several small ones.
      cfg.setMaxBufferedDocs(NUM_DOCS + 1);
      cfg.setRAMBufferSizeMB(IndexWriterConfig.DISABLE_AUTO_FLUSH);

      try (IndexWriter w = new IndexWriter(dir, cfg)) {
        for (int i = 0; i < NUM_DOCS; i++) {
          // No `id` field: an indexed term per document would add a 900 KB
          // term dictionary to a fixture whose whole subject is the `.dvd`.
          Document doc = new Document();
          if (i % SPARSE_STEP == 0) {
            doc.add(new NumericDocValuesField("sparse", sparseValue(i)));
          }
          if (i % VERY_SPARSE_STEP == 0) {
            doc.add(new NumericDocValuesField("very_sparse", verySparseValue(i)));
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
      m.append("sparse_step=").append(SPARSE_STEP).append('\n');
      m.append("very_sparse_step=").append(VERY_SPARSE_STEP).append('\n');

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

      // Sampled ground truth. The probes are chosen so that consecutive ones
      // are several 65 536-document blocks apart -- a cold seek to each is
      // exactly the access pattern `IndexedDISI.advanceBlock` consults the
      // jump table for -- and so that both present and absent documents, the
      // first and last present document, and every block boundary are covered.
      int[] probes = probes();
      StringBuilder probeList = new StringBuilder();
      for (int p : probes) {
        if (probeList.length() > 0) probeList.append(',');
        probeList.append(p);
      }
      m.append("probes=").append(probeList).append('\n');

      for (String fieldName : new String[] {"sparse", "very_sparse"}) {
        org.apache.lucene.index.FieldInfo field = fis.fieldInfo(fieldName);
        StringBuilder vals = new StringBuilder();
        long present = 0;
        // One producer per pass: `advanceExact` is forward-only, and the
        // probes are ascending, so a single instance is enough -- but a fresh
        // one per field keeps the two passes independent.
        NumericDocValues values = dvProducer.getNumeric(field);
        for (int p : probes) {
          if (vals.length() > 0) vals.append(',');
          if (values.advanceExact(p)) {
            vals.append(values.longValue());
            present++;
          } else {
            vals.append("NONE");
          }
        }
        m.append("field.").append(fieldName).append(".probe_values=").append(vals).append('\n');
        m.append("field.").append(fieldName).append(".probes_present=").append(present).append('\n');

        // A full scan's cardinality, which pins the column independently of
        // the sampled probes.
        NumericDocValues scan = dvProducer.getNumeric(field);
        long count = 0;
        long checksum = 0;
        for (int doc = 0; doc < sci.info.maxDoc(); doc++) {
          if (scan.advanceExact(doc)) {
            count++;
            checksum = checksum * 31 + scan.longValue();
          }
        }
        m.append("field.").append(fieldName).append(".count=").append(count).append('\n');
        m.append("field.").append(fieldName).append(".checksum=").append(checksum).append('\n');
      }
      dvProducer.close();

      Files.writeString(out.resolve("manifest.properties"), m.toString());
    }

    System.out.println("wrote disi_jump_table_index/ fixture directory");
  }

  /** Values a reader can re-derive, so a wrong seek is a wrong number rather than a plausible one. */
  private static long sparseValue(int doc) {
    return (long) doc * 7 + 11;
  }

  private static long verySparseValue(int doc) {
    return (long) doc * -13 - 5;
  }

  /**
   * Ascending probes several blocks apart, plus every 65 536-document block
   * boundary and its neighbours.
   */
  private static int[] probes() {
    java.util.TreeSet<Integer> set = new java.util.TreeSet<>();
    for (int block = 0; block * 65536 < NUM_DOCS; block++) {
      int base = block * 65536;
      set.add(base);
      set.add(base + 1);
      set.add(base + 2);
      if (base + 65535 < NUM_DOCS) {
        set.add(base + 65535);
      }
    }
    // Spread probes that skip two or three whole blocks at a time.
    set.add(0);
    set.add(131_073);
    set.add(196_608);
    set.add(199_998);
    set.add(199_999);
    for (int p = 5_000; p < NUM_DOCS; p += 37_777) {
      set.add(p);
    }
    // `very_sparse`'s own present documents, so that field's SPARSE blocks
    // are probed at a value and not only at absences.
    for (int p = 0; p < NUM_DOCS; p += VERY_SPARSE_STEP) {
      set.add(p);
      set.add(p + 1);
    }
    int[] out = new int[set.size()];
    int i = 0;
    for (int p : set) {
      out[i++] = p;
    }
    return out;
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
