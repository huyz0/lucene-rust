import org.apache.lucene.document.Document;
import org.apache.lucene.document.SortedDocValuesField;
import org.apache.lucene.document.SortedSetDocValuesField;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.BytesRef;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Comparator;

/**
 * A SORTED and a SORTED_SET field whose terms dictionary blocks LZ4 can
 * compress against their first term: every term repeats long runs of the
 * block's first term, so `Lucene90DocValuesConsumer.addTermsDict`'s
 * `LZ4.compressWithDictionary` (first term as preset dictionary, one
 * `FastCompressionHashTable` per field) emits back-references into the
 * dictionary. The Rust writer must rebuild `.dvd`/`.dvm` byte for byte --
 * `crates/lucene-index/tests/sorted_doc_values_lz4_dict_fixtures.rs`.
 * 300 dense docs: five 64-term blocks, the last one partial.
 */
public class GenSortedDocValuesLz4Dict {
  public static void main(String[] args) throws IOException {
    Path out = Path.of(args[0]).resolve("sorted_dv_lz4_dict_index");
    if (Files.exists(out)) {
      try (var walk = Files.walk(out)) {
        walk.sorted(Comparator.reverseOrder()).forEach(p -> p.toFile().delete());
      }
    }
    Files.createDirectories(out);
    try (Directory dir = FSDirectory.open(out)) {
      IndexWriterConfig cfg = new IndexWriterConfig();
      cfg.setUseCompoundFile(false);
      cfg.setMergePolicy(NoMergePolicy.INSTANCE);
      try (IndexWriter w = new IndexWriter(dir, cfg)) {
        for (int i = 0; i < 300; i++) {
          Document d = new Document();
          d.add(new SortedDocValuesField("sorted", bytes(term(i))));
          d.add(new SortedSetDocValuesField("set", bytes(term(i))));
          d.add(new SortedSetDocValuesField("set", bytes(term(i + 1000))));
          w.addDocument(d);
        }
      }
    }
  }

  static String term(int i) {
    return String.format("%05d", i) + "-commonmiddle-" + (i % 7) + "-commonmiddle-tail" + (i % 3);
  }

  static BytesRef bytes(String s) {
    return new BytesRef(s.getBytes(StandardCharsets.UTF_8));
  }
}
