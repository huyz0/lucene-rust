/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.StoredField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.SegmentReader;
import org.apache.lucene.index.StoredFieldVisitor;
import org.apache.lucene.index.StoredFields;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.BytesRef;
import org.opensearch.index.codec.customcodecs.Zstd104Codec;
import org.opensearch.index.codec.customcodecs.ZstdNoDict104Codec;

/**
 * OpenSearch custom-codecs' zstd stored fields, with the plugin's own codec classes on the
 * classpath.
 *
 * <ul>
 *   <li>{@code write DIR zstd|zstd_no_dict LEVEL}: writes the documents {@link #docs} describes
 *       into two segments (the second compound) with {@code Zstd104Codec} or
 *       {@code ZstdNoDict104Codec}.
 *   <li>{@code dump DIR}: prints every segment's codec and stored-fields mode and every
 *       document's stored fields, in the format
 *       {@code crates/lucene-search/tests/opensearch_zstd_stored_fields.rs} prints.
 * </ul>
 */
public class ZstdStoredFields {
  public static void main(String[] args) throws Exception {
    Path dir = Path.of(args[1]);
    switch (args[0]) {
      case "write" -> write(dir, args[2], Integer.parseInt(args[3]));
      case "dump" -> dump(dir);
      default -> throw new IllegalArgumentException(args[0]);
    }
  }

  static final String[] WORDS = {
    "lucene", "zstd", "segment", "stored", "field", "chunk", "the", "of", "and", "index",
    "opensearch", "rust", "dictionary", "block", "frame", "日本語", "grüße",
    "{\"k\":", "1234", "-", "\n"
  };

  /** Knuth's MMIX LCG, the same one the Rust test uses. */
  static long state;

  static int next(int bound) {
    state = state * 6364136223846793005L + 1442695040888963407L;
    return (int) ((state >>> 33) % bound);
  }

  static String text(int len) {
    StringBuilder sb = new StringBuilder();
    while (sb.length() < len) {
      sb.append(WORDS[next(WORDS.length)]).append(' ');
    }
    return sb.toString();
  }

  static List<Document> docs(int from, int to) {
    List<Document> out = new ArrayList<>();
    for (int i = from; i < to; i++) {
      Document doc = new Document();
      doc.add(new StringField("id", Integer.toString(i), Field.Store.NO));
      if (i % 113 != 0) {
        doc.add(new StoredField("title", text(10 + next(150))));
        int bodyLen = i == 777 ? 1_300_000 : next(3000);
        doc.add(new StoredField("body", text(bodyLen)));
        byte[] bin = new byte[next(64)];
        for (int b = 0; b < bin.length; b++) bin[b] = (byte) next(256);
        doc.add(new StoredField("bin", new BytesRef(bin)));
        doc.add(new StoredField("n", next(1 << 30) - (1 << 29)));
        doc.add(new StoredField("l", (long) next(1 << 30) << 20));
        doc.add(new StoredField("f", next(100000) / 7.0f));
        doc.add(new StoredField("d", next(100000) / 13.0d));
        for (int t = 0; t < i % 4; t++) doc.add(new StoredField("tag", "t" + next(50)));
      }
      out.add(doc);
    }
    return out;
  }

  static void write(Path dir, String codec, int level) throws Exception {
    Files.createDirectories(dir);
    state = 42;
    Codec c = codec.equals("zstd") ? new Zstd104Codec(level) : new ZstdNoDict104Codec(level);
    try (FSDirectory d = FSDirectory.open(dir)) {
      IndexWriterConfig cfg = new IndexWriterConfig().setCodec(c).setMergePolicy(NoMergePolicy.INSTANCE);
      cfg.setUseCompoundFile(false);
      try (IndexWriter w = new IndexWriter(d, cfg)) {
        w.addDocuments(docs(0, 1200));
        w.commit();
        w.getConfig().setUseCompoundFile(true);
        w.addDocuments(docs(1200, 1700));
        w.commit();
      }
    }
  }

  static void dump(Path dir) throws Exception {
    try (DirectoryReader reader = DirectoryReader.open(FSDirectory.open(dir))) {
      for (LeafReaderContext leaf : reader.leaves()) {
        SegmentReader sr = (SegmentReader) leaf.reader();
        var info = sr.getSegmentInfo().info;
        String mode = "-";
        for (var e : info.getAttributes().entrySet()) {
          if (e.getKey().endsWith("StoredFieldsFormat.mode")) mode = e.getKey() + "=" + e.getValue();
        }
        System.out.println("segment\t" + info.name + "\t" + info.getCodec().getName() + "\t" + mode + "\t" + sr.maxDoc());
        StoredFields stored = sr.storedFields();
        for (int doc = 0; doc < sr.maxDoc(); doc++) {
          Render r = new Render();
          stored.document(doc, r);
          System.out.println(doc + "\t" + r.count + "\t" + r.sb.length() + "\t" + Long.toHexString(fnv(r.sb.toString())));
        }
      }
    }
  }

  static long fnv(String s) {
    long h = 0xcbf29ce484222325L;
    for (byte b : s.getBytes(StandardCharsets.UTF_8)) {
      h ^= b & 0xff;
      h *= 0x100000001b3L;
    }
    return h;
  }

  /** Each field as {@code name:type:value;}, strings and bytes as hex, floats as their bits. */
  static class Render extends StoredFieldVisitor {
    final StringBuilder sb = new StringBuilder();
    int count;

    void add(FieldInfo fi, String type, String value) {
      count++;
      sb.append(fi.name).append(':').append(type).append(':').append(value).append(';');
    }

    static String hex(byte[] b) {
      StringBuilder s = new StringBuilder();
      for (byte x : b) s.append(String.format("%02x", x));
      return s.toString();
    }

    @Override
    public Status needsField(FieldInfo fieldInfo) {
      return Status.YES;
    }

    @Override
    public void stringField(FieldInfo fi, String value) {
      add(fi, "s", hex(value.getBytes(StandardCharsets.UTF_8)));
    }

    @Override
    public void binaryField(FieldInfo fi, byte[] value) {
      add(fi, "b", hex(value));
    }

    @Override
    public void intField(FieldInfo fi, int value) {
      add(fi, "i", Integer.toString(value));
    }

    @Override
    public void longField(FieldInfo fi, long value) {
      add(fi, "l", Long.toString(value));
    }

    @Override
    public void floatField(FieldInfo fi, float value) {
      add(fi, "f", Integer.toHexString(Float.floatToRawIntBits(value)));
    }

    @Override
    public void doubleField(FieldInfo fi, double value) {
      add(fi, "d", Long.toHexString(Double.doubleToRawLongBits(value)));
    }
  }
}
