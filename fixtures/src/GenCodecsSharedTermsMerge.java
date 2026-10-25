import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import org.apache.lucene.codecs.PostingsFormat;
import org.apache.lucene.codecs.lucene104.Lucene104Codec;
import org.apache.lucene.codecs.uniformsplit.sharedterms.STUniformSplitPostingsFormat;
import org.apache.lucene.document.*;
import org.apache.lucene.index.*;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.Sort;
import org.apache.lucene.search.SortField;
import org.apache.lucene.store.FSDirectory;

/** Actual shared-terms merge inputs, deletion/sort maps, and the resulting Java segment. */
public class GenCodecsSharedTermsMerge {
  static final String[] FIELDS = {"id", "docs", "freq", "pay", "gone", "gonepos"};

  static IndexWriterConfig config(boolean sorted) {
    PostingsFormat format = new STUniformSplitPostingsFormat(8, 2, null, null, true);
    var cfg = new IndexWriterConfig();
    cfg.setOpenMode(IndexWriterConfig.OpenMode.CREATE);
    cfg.setUseCompoundFile(false);
    cfg.setMergePolicy(NoMergePolicy.INSTANCE);
    if (sorted) cfg.setIndexSort(new Sort(new SortField("sort", SortField.Type.INT)));
    cfg.setCodec(new Lucene104Codec() {
      @Override
      public PostingsFormat getPostingsFormatForField(String field) {
        return format;
      }
    });
    return cfg;
  }

  public static void main(String[] args) throws Exception {
    run(Path.of(args[0]), false);
  }

  static void run(Path output, boolean positionBlocks) throws Exception {
    Path root = output.resolve(positionBlocks
        ? "codecs_sharedterms_merge_blocks" : "codecs_sharedterms_merge");
    for (boolean sorted : new boolean[] {false, true}) {
      Path scenario = root.resolve(sorted ? "sorted" : "unsorted");
      Files.createDirectories(scenario);
      var dirs = new ArrayList<FSDirectory>();
      var readers = new ArrayList<DirectoryReader>();
      var leaves = new ArrayList<CodecReader>();
      try {
        for (int segment = 0; segment < 3; segment++) {
          Path source = scenario.resolve("source" + segment);
          Files.createDirectories(source);
          var dir = FSDirectory.open(source);
          dirs.add(dir);
          try (var writer = new IndexWriter(dir, config(sorted))) {
            var freqs = new FieldType(TextField.TYPE_NOT_STORED);
            freqs.setIndexOptions(IndexOptions.DOCS_AND_FREQS);
            freqs.freeze();
            var positions = new FieldType(TextField.TYPE_NOT_STORED);
            positions.setIndexOptions(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS_AND_OFFSETS);
            positions.freeze();
            String[] removedPositions = new String[32];
            java.util.Arrays.fill(removedPositions, "removed");
            for (int doc = 0; doc < 300; doc++) {
              String id = segment + "_" + doc;
              var fields = new ArrayList<IndexableField>();
              fields.add(new StringField("id", id, Field.Store.NO));
              fields.add(new StringField("docs", doc % 11 == 0 ? "" : "w" + doc % 19, Field.Store.NO));
              if (doc % 3 != 0)
                fields.add(new Field("freq", "w" + doc % 19 + " w" + doc % 19 + " common", freqs));
              if (doc % 5 != 0) {
                String[] words = positionBlocks
                    ? new String[] {"w" + doc % 19, "common", "common", "common", "s" + segment}
                    : new String[] {"w" + doc % 19, "common", "s" + segment};
                fields.add(new Field("pay", new GenCodecsSharedTerms.PayStream(
                    words), positions));
              }
              if (doc % 37 == 0) fields.add(new StringField("gone", "removed", Field.Store.NO));
              if (positionBlocks && doc % 37 == 0)
                fields.add(new Field("gonepos", new GenCodecsSharedTerms.PayStream(
                    removedPositions), positions));
              // Each independent index assigns the same names different numbers.
              java.util.Collections.rotate(fields, segment);
              var document = new Document();
              for (var field : fields) document.add(field);
              document.add(new StoredField("stored_id", id));
              document.add(new NumericDocValuesField("sort", 900 - (doc * 3 + segment)));
              writer.addDocument(document);
            }
            writer.commit();
            for (int doc = 0; doc < 300; doc++) {
              if (doc % 37 == 0 || doc % 13 == segment)
                writer.deleteDocuments(new Term("id", segment + "_" + doc));
            }
            writer.commit();
          }
          var reader = DirectoryReader.open(dir);
          readers.add(reader);
          if (reader.leaves().size() != 1) throw new AssertionError("one source leaf");
          leaves.add((CodecReader) reader.leaves().get(0).reader());
        }
        Path destination = scenario.resolve("merged");
        Files.createDirectories(destination);
        try (var dir = FSDirectory.open(destination)) {
          try (var writer = new IndexWriter(dir, config(sorted))) {
            writer.addIndexes(leaves.toArray(CodecReader[]::new));
            writer.commit();
          }
          try (var reader = DirectoryReader.open(dir)) {
            if (reader.leaves().size() != 1) throw new AssertionError("one merged leaf");
            var leaf = reader.leaves().get(0).reader();
            var outputDocs = new HashMap<String, Integer>();
            for (int doc = 0; doc < leaf.maxDoc(); doc++)
              outputDocs.put(leaf.storedFields().document(doc).get("stored_id"), doc);
            var maps = new ArrayList<String>();
            for (int segment = 0; segment < leaves.size(); segment++) {
              var source = leaves.get(segment);
              var line = new StringBuilder(Integer.toString(segment));
              for (int doc = 0; doc < source.maxDoc(); doc++) {
                String id = source.storedFields().document(doc).get("stored_id");
                int mapped = outputDocs.getOrDefault(id, -1);
                if ((source.getLiveDocs() == null || source.getLiveDocs().get(doc)) != (mapped >= 0))
                  throw new AssertionError("doc map disagrees with live docs");
                line.append('\t').append(mapped);
              }
              maps.add(line.toString());
            }
            Files.write(scenario.resolve("docmaps.tsv"), maps);
            Files.writeString(scenario.resolve("max-doc.tsv"), leaf.maxDoc() + "\n");
            var stats = new ArrayList<String>();
            var norms = new ArrayList<String>();
            for (String name : FIELDS) {
              Terms terms = leaf.terms(name);
              if (terms == null) continue;
              FieldInfo field = leaf.getFieldInfos().fieldInfo(name);
              stats.add(name + "\t" + terms.size() + "\t" + terms.getDocCount()
                  + "\t" + terms.getSumDocFreq() + "\t" + terms.getSumTotalTermFreq());
              Files.write(scenario.resolve("postings." + name + ".tsv"),
                  GenCodecsSharedTerms.dumpPostings(terms));
              var values = leaf.getNormValues(name);
              if (values != null) {
                long[] dense = new long[leaf.maxDoc()];
                for (int doc = values.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS; doc = values.nextDoc())
                  dense[doc] = values.longValue();
                var line = new StringBuilder(Integer.toString(field.number));
                for (long value : dense) line.append('\t').append(value);
                norms.add(line.toString());
              }
            }
            Files.write(scenario.resolve("stats.tsv"), stats);
            Files.write(scenario.resolve("norms.tsv"), norms);
          }
        }
      } finally {
        for (var reader : readers) reader.close();
        for (var dir : dirs) dir.close();
      }
    }
  }
}
