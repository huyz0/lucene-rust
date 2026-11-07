/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.StoredFieldsReader;
import org.apache.lucene.codecs.lucene104.Lucene104Codec;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.StoredField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.SegmentReader;
import org.apache.lucene.index.StoredFieldVisitor;
import org.apache.lucene.index.StoredFields;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.BytesRef;
import org.opensearch.index.codec.customcodecs.Zstd104Codec;
import org.opensearch.index.codec.customcodecs.ZstdNoDict104Codec;

/**
 * Stock stored fields on a corpus of one JSON {@code _source} per line, in the columns
 * {@code crates/lucene-codecs/examples/stored_fields_compression.rs} prints: Lucene 10.5's LZ4
 * and DEFLATE modes and OpenSearch custom-codecs' zstd ones, with the plugin's classes and
 * zstd-jni on the classpath.
 *
 * <p>Each document holds {@code _id} and {@code _source} as binary stored fields, written by an
 * {@code IndexWriter} into one segment (the write column therefore includes the writer's own
 * overhead). The random column is one {@code StoredFields.document} at 100k seeded random ids;
 * the scan reads every document in order through the merge instance, as a merge does.
 *
 * <p>Usage: {@code StoredFieldsBench WORK_DIR FILE...}
 */
public class StoredFieldsBench {
  static final int RANDOM_READS = 100_000;

  record Mode(String name, Codec codec) {}

  static List<Mode> modes() {
    return List.of(
        new Mode("lz4 (default)", new Lucene104Codec(Lucene104Codec.Mode.BEST_SPEED)),
        new Mode("deflate (best_compression)", new Lucene104Codec(Lucene104Codec.Mode.BEST_COMPRESSION)),
        new Mode("zstd level 1", new Zstd104Codec(1)),
        new Mode("zstd level 3", new Zstd104Codec(3)),
        new Mode("zstd level 6", new Zstd104Codec(6)),
        new Mode("zstd_no_dict level 1", new ZstdNoDict104Codec(1)),
        new Mode("zstd_no_dict level 3", new ZstdNoDict104Codec(3)),
        new Mode("zstd_no_dict level 6", new ZstdNoDict104Codec(6)));
  }

  /** Counts what it is shown, so nothing is optimized away. */
  static final class Sink extends StoredFieldVisitor {
    long n;

    @Override
    public Status needsField(FieldInfo fieldInfo) {
      return Status.YES;
    }

    @Override
    public void binaryField(FieldInfo fieldInfo, byte[] value) {
      n += value.length;
    }
  }

  public static void main(String[] args) throws Exception {
    Path work = Path.of(args[0]);
    for (int f = 1; f < args.length; f++) {
      List<String> lines = Files.readAllLines(Path.of(args[f]), StandardCharsets.UTF_8);
      long raw = 0;
      byte[][] sources = new byte[lines.size()][];
      for (int i = 0; i < sources.length; i++) {
        sources[i] = lines.get(i).getBytes(StandardCharsets.UTF_8);
        raw += sources[i].length;
      }
      System.out.printf("%s: %d docs, %.1f MiB of _source%n", args[f], sources.length, raw / (double) (1 << 20));
      System.out.println("| mode | .fdt MiB | ratio | write MB/s | random doc us | scan MB/s |");
      System.out.println("|---|---|---|---|---|---|");
      long state = 0x9E3779B97F4A7C15L;
      int[] ids = new int[RANDOM_READS];
      for (int i = 0; i < ids.length; i++) {
        state ^= state << 13;
        state ^= state >>> 7;
        state ^= state << 17;
        ids[i] = (int) Long.remainderUnsigned(state, sources.length);
      }
      for (Mode mode : modes()) {
        Path dir = work.resolve(mode.name().replaceAll("[^a-z0-9]+", "_"));
        double writeS = Double.MAX_VALUE;
        for (int rep = 0; rep < 3; rep++) {
          deleteRecursive(dir);
          Files.createDirectories(dir);
          long t = System.nanoTime();
          try (FSDirectory d = FSDirectory.open(dir)) {
            IndexWriterConfig cfg = new IndexWriterConfig().setCodec(mode.codec())
                .setMergePolicy(NoMergePolicy.INSTANCE).setRAMBufferSizeMB(4096).setUseCompoundFile(false);
            try (IndexWriter w = new IndexWriter(d, cfg)) {
              for (int i = 0; i < sources.length; i++) {
                Document doc = new Document();
                doc.add(new StoredField("_id", new BytesRef(Integer.toString(i).getBytes(StandardCharsets.UTF_8))));
                doc.add(new StoredField("_source", new BytesRef(sources[i])));
                w.addDocument(doc);
              }
              w.commit();
            }
          }
          writeS = Math.min(writeS, (System.nanoTime() - t) / 1e9);
        }
        long fdt;
        try (var files = Files.list(dir)) {
          fdt = files.filter(p -> p.toString().endsWith(".fdt")).mapToLong(p -> p.toFile().length()).sum();
        }
        double randomS = Double.MAX_VALUE;
        double scanS = Double.MAX_VALUE;
        Sink sink = new Sink();
        try (DirectoryReader r = DirectoryReader.open(FSDirectory.open(dir))) {
          if (r.leaves().size() != 1) throw new AssertionError("one segment, got " + r.leaves().size());
          LeafReader leaf = r.leaves().get(0).reader();
          // Warm-up: one pass of each.
          for (int rep = 0; rep < 4; rep++) {
            StoredFields stored = leaf.storedFields();
            long t = System.nanoTime();
            for (int id : ids) stored.document(id, sink);
            if (rep > 0) randomS = Math.min(randomS, (System.nanoTime() - t) / 1e9);
            StoredFieldsReader merge = ((SegmentReader) leaf).getFieldsReader().getMergeInstance();
            t = System.nanoTime();
            for (int id = 0; id < leaf.maxDoc(); id++) merge.document(id, sink);
            if (rep > 0) scanS = Math.min(scanS, (System.nanoTime() - t) / 1e9);
          }
        }
        if (sink.n == 0) throw new AssertionError();
        double mb = raw / 1e6;
        System.out.printf("| %s | %.2f | %.2f | %.0f | %.2f | %.0f |%n", mode.name(), fdt / (double) (1 << 20),
            raw / (double) fdt, mb / writeS, randomS * 1e6 / RANDOM_READS, mb / scanS);
        deleteRecursive(dir);
      }
      System.out.println();
    }
  }

  static void deleteRecursive(Path p) throws Exception {
    if (Files.isDirectory(p)) {
      try (var entries = Files.list(p)) {
        for (Path child : (Iterable<Path>) entries::iterator) deleteRecursive(child);
      }
    }
    Files.deleteIfExists(p);
  }
}
