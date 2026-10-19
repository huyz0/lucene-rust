import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.Set;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.CodecReader;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.Sorter;
import org.apache.lucene.misc.index.BPIndexReorderer;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;

/**
 * M14 T14.3 `misc`: {@link BPIndexReorderer}. Writes {@code misc/bp_flat} (one segment of 700
 * documents drawn from overlapping topics), {@code misc/bp_blocks} (the same kind of documents in
 * parent blocks of 1 to 4 children) and {@code misc/bp.tsv}:
 *
 * <pre>
 * D  config  index  newToOld...   computeDocMap's new-to-old table, space separated
 * </pre>
 *
 * <p>Read by {@code crates/lucene-search/tests/misc_bp_fixtures.rs}.
 */
public class GenMiscBp {

  static final int DOCS = 700;
  static final int TOPICS = 12;
  static final int WORDS_PER_TOPIC = 9;

  static String body(Random r) {
    StringBuilder sb = new StringBuilder();
    int topic = r.nextInt(TOPICS);
    int terms = 4 + r.nextInt(5);
    for (int t = 0; t < terms; t++) {
      // mostly the topic's words, now and then a neighbouring topic's
      int tp = r.nextInt(5) == 0 ? (topic + 1) % TOPICS : topic;
      sb.append("w").append(tp).append('_').append(r.nextInt(WORDS_PER_TOPIC)).append(' ');
    }
    return sb.toString();
  }

  public static void main(String[] args) throws Exception {
    Path out = Path.of(args[0]).resolve("misc");
    Files.createDirectories(out);
    Random r = new Random(20260617);

    Path flat = out.resolve("bp_flat");
    try (Directory dir = FSDirectory.open(flat);
        IndexWriter w =
            new IndexWriter(dir, new IndexWriterConfig().setMergePolicy(NoMergePolicy.INSTANCE))) {
      for (int i = 0; i < DOCS; i++) {
        Document doc = new Document();
        doc.add(new StringField("id", "d" + i, Field.Store.NO));
        doc.add(new TextField("body", body(r), Field.Store.NO));
        w.addDocument(doc);
      }
      w.commit();
    }

    Path blocks = out.resolve("bp_blocks");
    try (Directory dir = FSDirectory.open(blocks);
        IndexWriter w =
            new IndexWriter(
                dir,
                new IndexWriterConfig()
                    .setMergePolicy(NoMergePolicy.INSTANCE)
                    .setParentField("_parent"))) {
      int made = 0;
      while (made < DOCS) {
        int children = r.nextInt(4);
        List<Document> block = new ArrayList<>();
        for (int c = 0; c <= children; c++) {
          Document doc = new Document();
          doc.add(new StringField("id", "d" + made, Field.Store.NO));
          doc.add(new TextField("body", body(r), Field.Store.NO));
          block.add(doc);
          made++;
        }
        w.addDocuments(block);
      }
      w.commit();
    }

    try (PrintWriter tsv =
        new PrintWriter(Files.newBufferedWriter(out.resolve("bp.tsv"), StandardCharsets.UTF_8))) {
      try (Directory dir = FSDirectory.open(flat);
          DirectoryReader reader = DirectoryReader.open(dir)) {
        CodecReader leaf = (CodecReader) reader.leaves().get(0).reader();
        dump(tsv, "default_small", "flat", leaf, 2, 1f, 4, 20, Set.of("body"));
        dump(tsv, "few_iters", "flat", leaf, 2, 1f, 16, 3, Set.of("body"));
        dump(tsv, "one_iter", "flat", leaf, 2, 1f, 2, 1, Set.of("body"));
        dump(tsv, "max_doc_freq", "flat", leaf, 2, 0.05f, 4, 20, Set.of("body"));
        dump(tsv, "min_doc_freq", "flat", leaf, 40, 1f, 4, 20, Set.of("body"));
        dump(tsv, "all_fields", "flat", leaf, 2, 1f, 8, 20, null);
        dump(tsv, "no_terms", "flat", leaf, 100000, 1f, 4, 20, Set.of("body"));
      }
      try (Directory dir = FSDirectory.open(blocks);
          DirectoryReader reader = DirectoryReader.open(dir)) {
        CodecReader leaf = (CodecReader) reader.leaves().get(0).reader();
        dump(tsv, "blocks", "blocks", leaf, 2, 1f, 4, 20, Set.of("body"));
        dump(tsv, "blocks_big_partition", "blocks", leaf, 2, 1f, 64, 20, Set.of("body"));
      }
    }
    System.exit(0);
  }

  static void dump(
      PrintWriter tsv,
      String config,
      String index,
      CodecReader leaf,
      int minDocFreq,
      float maxDocFreq,
      int minPartitionSize,
      int maxIters,
      Set<String> fields)
      throws Exception {
    BPIndexReorderer reorderer = new BPIndexReorderer();
    reorderer.setRAMBudgetMB(256);
    reorderer.setMinDocFreq(minDocFreq);
    reorderer.setMaxDocFreq(maxDocFreq);
    reorderer.setMinPartitionSize(minPartitionSize);
    reorderer.setMaxIters(maxIters);
    reorderer.setFields(fields);
    Sorter.DocMap map;
    try (Directory tmp = new ByteBuffersDirectory()) {
      map = reorderer.computeDocMap(leaf, tmp, null);
    }
    StringBuilder sb = new StringBuilder();
    for (int i = 0; i < map.size(); i++) {
      if (i > 0) sb.append(' ');
      sb.append(map.newToOld(i));
    }
    tsv.printf("D\t%s\t%s\t%d\t%d\t%s\t%d\t%d\t%s%n", config, index, minDocFreq, minPartitionSize,
        maxDocFreq, maxIters, map.size(), sb);
  }
}
