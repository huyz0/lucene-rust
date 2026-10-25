import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HexFormat;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.NumericDocValues;
import org.apache.lucene.index.BinaryDocValues;
import org.apache.lucene.index.SortedDocValues;
import org.apache.lucene.index.SortedNumericDocValues;
import org.apache.lucene.index.SortedSetDocValues;
import org.apache.lucene.index.PostingsEnum;
import org.apache.lucene.index.SegmentReader;
import org.apache.lucene.index.TermsEnum;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.FieldExistsQuery;
import org.apache.lucene.search.MatchAllDocsQuery;
import org.apache.lucene.search.ScoreMode;
import org.apache.lucene.search.TermStatistics;
import org.apache.lucene.search.similarities.BM25Similarity;
import org.apache.lucene.search.similarities.ClassicSimilarity;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.BytesRef;

/** Reader-level expectations over GenCodecsSimpleTextCodec's real indexes. */
public final class GenSimpleTextDirectoryReader {
  static void existenceCases(Path root) throws Exception {
    // Mixed sources exercise priority; empty token streams distinguish norms from terms.
    for (boolean deleted : new boolean[] {false, true}) {
      Path out = root.resolve("simpletext_field_exists").resolve(deleted ? "deleted" : "clean");
      Files.createDirectories(out);
      Path index = out.resolve("index");
      // CREATE retains the old segment-name counter; reset this generator's
      // own index so repeated scoped runs have the same file tree.
      if (Files.exists(index)) {
        try (var files = Files.walk(index)) {
          for (var file : files.sorted(java.util.Comparator.reverseOrder()).toList()) {
            Files.delete(file);
          }
        }
      }
      try (var dir = FSDirectory.open(index)) {
        var config = new org.apache.lucene.index.IndexWriterConfig(
            new org.apache.lucene.analysis.core.WhitespaceAnalyzer())
            .setCodec(new org.apache.lucene.codecs.simpletext.SimpleTextCodec())
            .setOpenMode(org.apache.lucene.index.IndexWriterConfig.OpenMode.CREATE)
            .setMergePolicy(org.apache.lucene.index.NoMergePolicy.INSTANCE)
            .setUseCompoundFile(false);
        try (var writer = new org.apache.lucene.index.IndexWriter(dir, config)) {
          for (int docID = 0; docID < 8; docID++) {
            var doc = new org.apache.lucene.document.Document();
            doc.add(new org.apache.lucene.document.StringField("id", "" + docID,
                org.apache.lucene.document.Field.Store.NO));
            if (docID % 2 == 0) {
              doc.add(new org.apache.lucene.document.TextField("norms_dv", "term",
                  org.apache.lucene.document.Field.Store.NO));
              doc.add(new org.apache.lucene.document.KnnFloatVectorField("float_dv", new float[] {1, docID}));
              doc.add(new org.apache.lucene.document.KnnByteVectorField("byte_dv", new byte[] {1, (byte) docID}));
              doc.add(new org.apache.lucene.document.StringField("vector_terms", "term",
                  org.apache.lucene.document.Field.Store.NO));
              doc.add(new org.apache.lucene.document.KnnFloatVectorField("vector_terms", new float[] {1, docID}));
              for (var field : new String[] {"norms_dv", "float_dv", "byte_dv"}) {
                doc.add(new org.apache.lucene.document.NumericDocValuesField(field, docID));
              }
            }
            // A point or term count is a valid DV proxy when both share documents.
            if (docID % 3 != 0) {
              doc.add(new org.apache.lucene.document.IntPoint("points_dv", docID));
              doc.add(new org.apache.lucene.document.NumericDocValuesField("points_dv", docID));
              doc.add(new org.apache.lucene.document.StringField("terms_dv", "term",
                  org.apache.lucene.document.Field.Store.NO));
              doc.add(new org.apache.lucene.document.NumericDocValuesField("terms_dv", docID));
            }
            doc.add(new org.apache.lucene.document.TextField("dense_norms", "term",
                org.apache.lucene.document.Field.Store.NO));
            doc.add(new org.apache.lucene.document.TextField("empty_norms", "",
                org.apache.lucene.document.Field.Store.NO));
            doc.add(org.apache.lucene.document.NumericDocValuesField.indexedField("dense_skip", docID));
            writer.addDocument(doc);
          }
          writer.commit();
          if (deleted) {
            writer.deleteDocuments(new Term("id", "0"), new Term("id", "1"));
            writer.commit();
          }
        }
        try (var reader = DirectoryReader.open(dir)) {
          if (reader.leaves().size() != 1) throw new AssertionError("expected single leaf");
          var leaf = reader.leaves().get(0);
          Files.write(out.resolve("exists.tsv"), exists(reader, leaf));
          ((SegmentReader) leaf.reader()).checkIntegrity();
        }
        if (!deleted) vectorPostingsCorruption(out, dir);
      }
    }
  }

  static void vectorPostingsCorruption(Path out, FSDirectory dir) throws Exception {
    Path postings = out.resolve("index/_0.pst");
    byte[] original = Files.readAllBytes(postings);
    String contents = new String(original, java.nio.charset.StandardCharsets.UTF_8);
    int field = contents.indexOf("field vector_terms\n");
    int doc = contents.indexOf("    doc 0\n", field);
    if (field < 0 || doc < field) throw new AssertionError("missing mixed vector field");
    byte[] damaged = contents.substring(0, contents.lastIndexOf("checksum "))
        .getBytes(java.nio.charset.StandardCharsets.UTF_8);
    damaged[doc + "    doc ".length()] = 'x';
    var checksum = new java.util.zip.CRC32();
    checksum.update(damaged);
    byte[] footer = String.format(java.util.Locale.ROOT, "checksum %020d\n", checksum.getValue())
        .getBytes(java.nio.charset.StandardCharsets.UTF_8);
    var bytes = new java.io.ByteArrayOutputStream();
    bytes.write(damaged);
    bytes.write(footer);
    byte[] corrupted = bytes.toByteArray();
    // Recompute the real footer: the field's lazy document parser must fail,
    // while opening and vector existence operations must still succeed.
    Files.write(postings, corrupted);
    try (var reader = DirectoryReader.open(dir)) {
      var leaf = reader.leaves().get(0);
      var searcher = new IndexSearcher(reader);
      var query = new FieldExistsQuery("vector_terms");
      int count = query.createWeight(searcher, ScoreMode.COMPLETE_NO_SCORES, 1f).count(leaf);
      boolean rewrite = searcher.rewrite(query) instanceof MatchAllDocsQuery;
      try {
        leaf.reader().terms("vector_terms");
        throw new AssertionError("damaged document accepted");
      } catch (NumberFormatException expected) {
        Files.writeString(out.resolve("vector-corruption.tsv"),
            "count\t" + count + "\nrewrite\t" + rewrite + "\nterms\terror\n");
      }
      Files.write(out.resolve("vector-corruption.pst"), corrupted);
    } finally {
      Files.write(postings, original);
    }
  }

  static ArrayList<String> exists(DirectoryReader reader,
      org.apache.lucene.index.LeafReaderContext leaf) throws Exception {
    var searcher = new IndexSearcher(reader);
    var segment = leaf.reader();
    var names = new ArrayList<String>();
    for (var info : segment.getFieldInfos()) names.add(info.name);
    names.add("absent-probe");
    var lines = new ArrayList<String>();
    for (var name : names) {
      var query = new FieldExistsQuery(name);
      var weight = query.createWeight(searcher, ScoreMode.COMPLETE_NO_SCORES, 1f);
      var info = segment.getFieldInfos().fieldInfo(name);
      if (info != null && !info.hasNorms() && !info.hasVectorValues()
          && info.getDocValuesType() == org.apache.lucene.index.DocValuesType.NONE) {
        try {
          weight.scorerSupplier(leaf);
          throw new AssertionError("unsupported field accepted: " + name);
        } catch (IllegalStateException expected) {
          lines.add("unsupported\t" + hex(new BytesRef(name)));
        }
        continue;
      }
      int shortcut = weight.count(leaf);
      var raw = new ArrayList<String>();
      var live = new ArrayList<String>();
      var supplier = weight.scorerSupplier(leaf);
      if (supplier != null) {
        var cursor = supplier.get(Long.MAX_VALUE).iterator();
        for (int doc = cursor.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS;
            doc = cursor.nextDoc()) {
          raw.add(Integer.toString(doc));
          if (segment.getLiveDocs() == null || segment.getLiveDocs().get(doc)) {
            live.add(Integer.toString(doc));
          }
        }
      }
      if (shortcut >= 0 && shortcut != live.size()) throw new AssertionError(name);
      lines.add("field\t" + hex(new BytesRef(name)) + "\t" + shortcut);
      lines.add("raw\t" + String.join(",", raw));
      lines.add("live\t" + String.join(",", live));
      // Rewrite is reader-wide and is allowed to reject another leaf's unsupported field.
      boolean supportedEverywhere = true;
      for (var other : reader.leaves()) {
        var fi = other.reader().getFieldInfos().fieldInfo(name);
        if (fi != null && !fi.hasNorms() && !fi.hasVectorValues()
            && fi.getDocValuesType() == org.apache.lucene.index.DocValuesType.NONE) {
          supportedEverywhere = false;
        }
      }
      lines.add("rewrite\t" + (supportedEverywhere
          ? Boolean.toString(searcher.rewrite(query) instanceof MatchAllDocsQuery) : "unsupported"));
    }
    return lines;
  }

  static String hex(BytesRef bytes) {
    return HexFormat.of().formatHex(bytes.bytes, bytes.offset, bytes.offset + bytes.length);
  }

  static ArrayList<String> scores(DirectoryReader reader, SegmentReader segment) throws Exception {
    var searcher = new IndexSearcher(reader);
    var lines = new ArrayList<String>();
    for (var field : segment.getFieldInfos()) {
      var terms = segment.terms(field.name);
      if (terms == null || terms.getDocCount() == 0) continue;
      var collection = searcher.collectionStatistics(field.name);
      lines.add("field\t" + hex(new BytesRef(field.name)) + "\t" + collection.maxDoc()
          + "\t" + collection.docCount() + "\t" + collection.sumTotalTermFreq()
          + "\t" + collection.sumDocFreq());
      var cursor = terms.iterator();
      for (var bytes = cursor.next(); bytes != null; bytes = cursor.next()) {
        var term = new Term(field.name, BytesRef.deepCopyOf(bytes));
        var stats = new TermStatistics(term.bytes(), reader.docFreq(term), reader.totalTermFreq(term));
        var bm25 = new BM25Similarity().scorer(1f, collection, stats);
        var classic = new ClassicSimilarity().scorer(1f, collection, stats);
        lines.add("term\t" + hex(bytes) + "\t" + stats.docFreq() + "\t" + stats.totalTermFreq());
        var norms = segment.getNormValues(field.name);
        var postings = cursor.postings(null, PostingsEnum.FREQS);
        for (int doc = postings.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS;
            doc = postings.nextDoc()) {
          long norm = norms != null && norms.advanceExact(doc) ? norms.longValue() : 1L;
          int freq = postings.freq();
          lines.add("doc\t" + doc + "\t" + freq + "\t" + norm + "\t"
              + Integer.toUnsignedString(Float.floatToRawIntBits(bm25.score(freq, norm)), 16)
              + "\t" + Integer.toUnsignedString(Float.floatToRawIntBits(classic.score(freq, norm)), 16));
        }
        lines.add("endterm");
      }
      lines.add("endfield");
    }
    return lines;
  }

  static void values(ArrayList<String> lines, String name, String type, long gen,
      DocIdSetIterator values, org.apache.lucene.index.DocValuesSkipper skipper) throws Exception {
    lines.add("field\t" + hex(new BytesRef(name)) + "\t" + type + "\t" + gen
        + "\t" + values.cost());
    if (values instanceof SortedDocValues sorted) {
      for (int ord = 0; ord < sorted.getValueCount(); ord++) {
        lines.add("ord\t" + ord + "\t" + hex(sorted.lookupOrd(ord)));
      }
    } else if (values instanceof SortedSetDocValues sorted) {
      for (long ord = 0; ord < sorted.getValueCount(); ord++) {
        lines.add("ord\t" + ord + "\t" + hex(sorted.lookupOrd(ord)));
      }
    }
    if (skipper != null) {
      lines.add("skipall\t" + skipper.minValue() + "\t" + skipper.maxValue()
          + "\t" + skipper.docCount() + "\t" + skipper.maxValueCount());
      for (int target = 0;;) {
        skipper.advance(target);
        if (skipper.minDocID(0) == DocIdSetIterator.NO_MORE_DOCS) break;
        lines.add("skip\t" + skipper.minDocID(0) + "\t" + skipper.maxDocID(0)
            + "\t" + skipper.minValue(0) + "\t" + skipper.maxValue(0)
            + "\t" + skipper.docCount(0));
        target = skipper.maxDocID(0) + 1;
      }
    }
    for (int doc = values.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS;
        doc = values.nextDoc()) {
      String value;
      if (values instanceof NumericDocValues numeric) {
        value = Long.toString(numeric.longValue());
      } else if (values instanceof BinaryDocValues binary) {
        value = hex(binary.binaryValue());
      } else if (values instanceof SortedDocValues sorted) {
        value = Integer.toString(sorted.ordValue());
      } else if (values instanceof SortedNumericDocValues numeric) {
        var parts = new ArrayList<String>();
        for (int i = 0, count = numeric.docValueCount(); i < count; i++) {
          parts.add(Long.toString(numeric.nextValue()));
        }
        value = String.join(",", parts);
      } else if (values instanceof SortedSetDocValues sorted) {
        var parts = new ArrayList<String>();
        for (int i = 0, count = sorted.docValueCount(); i < count; i++) {
          parts.add(Long.toString(sorted.nextOrd()));
        }
        value = String.join(",", parts);
      } else {
        throw new AssertionError(values);
      }
      lines.add("d\t" + doc + "\t" + value);
    }
    lines.add("endfield");
  }

  public static void main(String[] args) throws Exception {
    Path root = Path.of(args[0]);
    existenceCases(root);
    for (String variant : new String[] {"flush", "merged"}) {
      Path out = root.resolve("simpletext_directory_reader").resolve(variant);
      Files.createDirectories(out);
      try (var dir = FSDirectory.open(root.resolve("codecs_simpletext/codec")
          .resolve(variant).resolve("index"));
          var reader = DirectoryReader.open(dir)) {
        var segments = new ArrayList<String>();
        for (var leaf : reader.leaves()) {
          var segment = (SegmentReader) leaf.reader();
          segments.add(segment.getSegmentName());
          var lines = new ArrayList<String>();
          for (var field : segment.getFieldInfos()) {
            var terms = segment.terms(field.name);
            if (terms == null) continue;
            lines.add("field\t" + field.number + "\t" + hex(new BytesRef(field.name))
                + "\t" + terms.size() + "\t" + terms.getSumTotalTermFreq()
                + "\t" + terms.getSumDocFreq() + "\t" + terms.getDocCount()
                + "\t" + terms.hasFreqs() + "\t" + terms.hasPositions()
                + "\t" + terms.hasOffsets() + "\t" + terms.hasPayloads());
            var cursor = terms.iterator();
            for (var term = cursor.next(); term != null; term = cursor.next()) {
              lines.add("term\t" + hex(term) + "\t" + cursor.docFreq()
                  + "\t" + cursor.totalTermFreq());
              var postings = cursor.postings(null, PostingsEnum.ALL);
              for (int doc = postings.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS;
                  doc = postings.nextDoc()) {
                int freq = postings.freq();
                lines.add("doc\t" + doc + "\t" + freq);
                if (terms.hasPositions()) {
                  for (int occurrence = 0; occurrence < freq; occurrence++) {
                    int position = postings.nextPosition();
                    var payload = postings.getPayload();
                    lines.add("pos\t" + position + "\t" + postings.startOffset()
                        + "\t" + postings.endOffset() + "\t"
                        + (payload == null ? "-" : hex(payload)));
                  }
                }
              }
              lines.add("endterm");
            }
            for (var target : new BytesRef[] {new BytesRef(), new BytesRef(new byte[] {0}),
                new BytesRef("absent-probe"), new BytesRef(new byte[] {(byte) 255})}) {
              var status = cursor.seekCeil(target);
              lines.add("ceil\t" + hex(target) + "\t" + status + "\t"
                  + (status == TermsEnum.SeekStatus.END ? "-" : hex(cursor.term())));
            }
            lines.add("endfield");
          }
          Files.write(out.resolve("terms." + segment.getSegmentName() + ".tsv"), lines);
          var values = new ArrayList<String>();
          for (var field : segment.getFieldInfos()) {
            DocIdSetIterator cursor = switch (field.getDocValuesType()) {
              case NONE -> null;
              case NUMERIC -> segment.getNumericDocValues(field.name);
              case BINARY -> segment.getBinaryDocValues(field.name);
              case SORTED -> segment.getSortedDocValues(field.name);
              case SORTED_NUMERIC -> segment.getSortedNumericDocValues(field.name);
              case SORTED_SET -> segment.getSortedSetDocValues(field.name);
            };
            if (cursor != null) {
              values(values, field.name, field.getDocValuesType().name(), field.getDocValuesGen(), cursor,
                  segment.getDocValuesSkipper(field.name));
            }
            var norms = segment.getNormValues(field.name);
            if (norms != null) values(values, field.name, "NORMS", -1, norms, null);
          }
          Files.write(out.resolve("values." + segment.getSegmentName() + ".tsv"), values);
          Files.write(out.resolve("scores." + segment.getSegmentName() + ".tsv"), scores(reader, segment));
          Files.write(out.resolve("exists." + segment.getSegmentName() + ".tsv"), exists(reader, leaf));
          segment.checkIntegrity();
        }
        Files.write(out.resolve("segments.tsv"), segments);
      }
    }
  }
}
