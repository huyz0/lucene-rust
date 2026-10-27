import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashSet;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.FieldType;
import org.apache.lucene.document.StoredField;
import org.apache.lucene.index.CheckIndex;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.PostingsEnum;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.BytesRef;
/** Checks native global schema validation and native append BEFORE Rust merge. Mutates exports. */
public class VerifyConfiguredTvGaps {
  static void require(boolean condition, String description) {
    if (!condition) throw new AssertionError(description);
  }
  static void verify(Path path, int expected) throws Exception {
    try (var dir = FSDirectory.open(path)) {
      try (var reader = DirectoryReader.open(dir)) {
        require(reader.numDocs() == expected, path + ": wrong original document count");
        require(reader.leaves().size() == expected, path + ": source segments unexpectedly merged");
        // DirectoryReader alone does not enforce cross-segment schema consistency.
        FieldInfos.getMergedFieldInfos(reader);
      }
      var type = new FieldType();
      type.setTokenized(true);
      type.setIndexOptions(IndexOptions.DOCS_AND_FREQS);
      type.setOmitNorms(true);
      type.setStoreTermVectors(true);
      type.setStoreTermVectorPositions(true);
      type.freeze();
      // Initialization validates every segment's FieldInfo via global FieldNumbers.
      try (var writer = new IndexWriter(dir, new IndexWriterConfig(new StandardAnalyzer()).setOpenMode(IndexWriterConfig.OpenMode.APPEND))) {
        var document = new Document();
        document.add(new StoredField("id", "c"));
        document.add(new Field("body", "x x", type));
        writer.addDocument(document);
        writer.commit();
        writer.forceMerge(1);
        writer.commit();
      }
      try (var reader = DirectoryReader.open(dir)) {
        require(reader.numDocs() == expected + 1, path + ": append lost documents");
        require(reader.leaves().size() == 1, path + ": Java forceMerge failed");
        require(FieldInfos.getMergedFieldInfos(reader).fieldInfo("body").hasTermVectors(), path + ": vector schema lost");
        var seen = new HashSet<String>();
        for (int doc = 0; doc < reader.maxDoc(); doc++) {
          String id = reader.storedFields().document(doc).get("id");
          require(seen.add(id), path + ": duplicate ID " + id);
          var fields = reader.termVectors().get(doc);
          if (id.equals("a")) { require(fields == null, path + ": empty source has terms"); continue; }
          require(fields != null, path + ": missing populated vector " + id);
          var terms = fields.terms("body").iterator();
          require(terms.seekExact(new BytesRef("x")), path + ": missing x for " + id);
          var postings = terms.postings(null, PostingsEnum.ALL);
          require(postings.nextDoc() == 0 && postings.freq() == 2, path + ": wrong vector frequency");
          require(postings.nextPosition() == 0 && postings.nextPosition() == 1, path + ": wrong vector positions");
        }
        require(seen.contains("a") && seen.contains("c") && (expected == 1 || seen.contains("b")), path + ": wrong IDs");
      }
      try (var check = new CheckIndex(dir)) { require(check.checkIndex().clean, path + ": CheckIndex failed"); }
    }
  }
  public static void main(String[] args) throws Exception {
    int verified = 0;
    try (var cases = Files.list(Path.of(args[0]))) {
      for (var path : cases.filter(Files::isDirectory).sorted().toList()) {
        verify(path.resolve("first"), 1);
        verify(path.resolve("two"), 2);
        verified += 2;
      }
    }
    require(verified == 24, "expected 24 first/two-segment exports, got " + verified);
    System.out.println("configured TV gaps: 24 native schema/append/merge/readback/CheckIndex checks pass");
  }
}
