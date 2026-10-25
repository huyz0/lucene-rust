import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HexFormat;
import java.util.List;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.SegmentReader;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.PhraseQuery;
import org.apache.lucene.search.ScoreMode;
import org.apache.lucene.search.Scorer;
import org.apache.lucene.search.Weight;
import org.apache.lucene.search.similarities.BM25Similarity;
import org.apache.lucene.search.similarities.ClassicSimilarity;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.BytesRef;

/** Native phrase-scorer expectations over existing Java SimpleText indexes. */
public final class GenSimpleTextPhraseScorer {
  record Candidate(BytesRef term, int docFreq) {}
  record Spec(String field, List<BytesRef> terms, int[] positions, int slop) {
    PhraseQuery query() {
      var builder = new PhraseQuery.Builder().setSlop(slop);
      for (int i = 0; i < terms.size(); i++) {
        builder.add(new Term(field, terms.get(i)), positions[i]);
      }
      return builder.build();
    }
  }

  static String hex(BytesRef bytes) {
    return HexFormat.of().formatHex(bytes.bytes, bytes.offset, bytes.offset + bytes.length);
  }

  static String bits(float value) {
    return Integer.toUnsignedString(Float.floatToRawIntBits(value), 16);
  }

  static ArrayList<Spec> queries(SegmentReader leaf) throws Exception {
    var result = new ArrayList<Spec>();
    for (var info : leaf.getFieldInfos()) {
      var terms = leaf.terms(info.name);
      if (terms == null) continue;
      var candidates = new ArrayList<Candidate>();
      var cursor = terms.iterator();
      for (var term = cursor.next(); term != null; term = cursor.next()) {
        candidates.add(new Candidate(BytesRef.deepCopyOf(term), cursor.docFreq()));
      }
      if (candidates.isEmpty()) continue;
      candidates.sort(Comparator.comparingInt(Candidate::docFreq).reversed()
          .thenComparing(Candidate::term));
      var a = candidates.get(0).term();
      var b = candidates.get(Math.min(1, candidates.size() - 1)).term();
      // Unsupported fields must throw before looking up a missing term.
      if (!terms.hasPositions()) {
        result.add(new Spec(info.name, List.of(a, b), new int[] {0, 1}, 0));
        result.add(new Spec(info.name, List.of(a, new BytesRef("absent-probe")),
            new int[] {0, 1}, 0));
        continue;
      }
      result.add(new Spec(info.name, List.of(), new int[] {}, 0));
      result.add(new Spec(info.name, List.of(a), new int[] {3}, 0));
      result.add(new Spec(info.name, List.of(a, new BytesRef("absent-probe")),
          new int[] {0, 1}, 0));
      for (int slop : new int[] {0, 1, 2, 5}) {
        result.add(new Spec(info.name, List.of(a, b), new int[] {0, 1}, slop));
        result.add(new Spec(info.name, List.of(b, a), new int[] {0, 1}, slop));
        result.add(new Spec(info.name, List.of(a, a), new int[] {0, 1}, slop));
        result.add(new Spec(info.name, List.of(a, b, a), new int[] {0, 1, 2}, slop));
        result.add(new Spec(info.name, List.of(a, b), new int[] {3, 5}, slop));
        result.add(new Spec(info.name, List.of(a, b), new int[] {0, 0}, slop));
        result.add(new Spec(info.name, List.of(b, a), new int[] {0, 0}, slop));
        result.add(new Spec(info.name, List.of(a, a, a), new int[] {0, 0, 1}, slop));
      }
    }
    result.add(new Spec("absent-field", List.of(new BytesRef("a"), new BytesRef("b")),
        new int[] {0, 1}, 0));
    return result;
  }

  static Scorer scorer(Weight weight, LeafReaderContext leaf) throws Exception {
    var supplier = weight.scorerSupplier(leaf);
    if (supplier == null) return null;
    // IndexSearcher marks the top-level supplier before get/bulkScorer.
    // A one-term phrase rewrites to TermQuery, whose threshold is ignored
    // unless this flag enables its ImpactsDISI.
    supplier.setTopLevelScoringClause();
    return supplier.get(Long.MAX_VALUE);
  }

  static void deepIndex(Path index, boolean merged) throws Exception {
    if (Files.exists(index)) {
      try (var files = Files.walk(index)) {
        for (var file : files.sorted(Comparator.reverseOrder()).toList()) Files.delete(file);
      }
    }
    try (var directory = FSDirectory.open(index)) {
      var config = new org.apache.lucene.index.IndexWriterConfig(
          new org.apache.lucene.analysis.core.WhitespaceAnalyzer())
          .setCodec(new org.apache.lucene.codecs.simpletext.SimpleTextCodec())
          .setUseCompoundFile(merged);
      if (!merged) config.setMergePolicy(org.apache.lucene.index.NoMergePolicy.INSTANCE);
      try (var writer = new org.apache.lucene.index.IndexWriter(directory, config)) {
        for (int i = 0; i < 768; i++) {
          var doc = new org.apache.lucene.document.Document();
          doc.add(new org.apache.lucene.document.StringField("id", Integer.toString(i),
              org.apache.lucene.document.Field.Store.NO));
          String text = "alpha beta " + (i % 3 == 0 ? "alpha beta alpha " : "gap ")
              + "omega ".repeat(1 + i % 7) + (i % 11 == 0 ? "rare beta" : "");
          doc.add(new org.apache.lucene.document.TextField("body", text,
              org.apache.lucene.document.Field.Store.NO));
          // Equal doc frequencies with overlapping, shifted document ranges:
          // the two terms have different skip boundaries despite equal costs.
          String tied = (i < 384 ? "left " : "")
              + (i >= 128 && i < 512 ? "right" : "");
          doc.add(new org.apache.lucene.document.TextField("tied", tied,
              org.apache.lucene.document.Field.Store.NO));
          writer.addDocument(doc);
          if (i == 255 || i == 511) writer.flush();
        }
        if (merged) writer.forceMerge(1);
        writer.deleteDocuments(new Term("id", "0"), new Term("id", "333"),
            new Term("id", "767"));
        writer.commit();
      }
      try (var reader = DirectoryReader.open(directory)) {
        for (var leaf : reader.leaves()) {
          var terms = leaf.reader().terms("body").iterator();
          if (!terms.seekExact(new BytesRef("alpha"))) throw new AssertionError("missing alpha");
          var impacts = terms.impacts(org.apache.lucene.index.PostingsEnum.POSITIONS);
          // Target 0 retains SimpleTextSkipReader's initial dummy level.
          impacts.advanceShallow(1);
          if (impacts.getImpacts().numLevels() != 4) {
            throw new AssertionError("deep phrase index must exercise all four impacts levels");
          }
          ((SegmentReader) leaf.reader()).checkIntegrity();
          if (merged) {
            var tied = leaf.reader().terms("tied");
            var left = tied.iterator();
            var right = tied.iterator();
            if (!left.seekExact(new BytesRef("left"))
                || !right.seekExact(new BytesRef("right"))
                || left.docFreq() != right.docFreq() || left.docFreq() != 384) {
              throw new AssertionError("tied phrase costs must be equal and nonzero");
            }
            var leftImpacts = left.impacts(org.apache.lucene.index.PostingsEnum.POSITIONS);
            var rightImpacts = right.impacts(org.apache.lucene.index.PostingsEnum.POSITIONS);
            // The merge may place their shared documents first; target 154
            // reaches the shifted ranges and distinguishes the level-0 blocks.
            leftImpacts.advanceShallow(154);
            rightImpacts.advanceShallow(154);
            if (leftImpacts.getImpacts().getDocIdUpTo(0)
                == rightImpacts.getImpacts().getDocIdUpTo(0)) {
              throw new AssertionError("tied phrase terms must have different skip boundaries");
            }
          }
        }
      }
    }
  }

  static void write(Path out, DirectoryReader reader, LeafReaderContext leaf) throws Exception {
    var segment = (SegmentReader) leaf.reader();
    var lines = new ArrayList<String>();
    var searcher = new IndexSearcher(reader);
    for (var spec : queries(segment)) {
      lines.add("query\t" + hex(new BytesRef(spec.field())) + "\t" + spec.slop()
          + "\t" + spec.terms().size());
      var collection = searcher.collectionStatistics(spec.field());
      if (collection != null) {
        lines.add("collection\t" + collection.maxDoc() + "\t" + collection.docCount()
            + "\t" + collection.sumTotalTermFreq() + "\t" + collection.sumDocFreq());
      }
      for (int i = 0; i < spec.terms().size(); i++) {
        var term = new Term(spec.field(), spec.terms().get(i));
        lines.add("term\t" + hex(term.bytes()) + "\t" + spec.positions()[i]
            + "\t" + reader.docFreq(term) + "\t" + reader.totalTermFreq(term));
      }
      for (String similarity : new String[] {"bm25", "classic"}) {
        searcher.setSimilarity(similarity.equals("bm25")
            ? new BM25Similarity() : new ClassicSimilarity());
        for (float boost : new float[] {1f, 2.25f}) {
          for (var mode : new ScoreMode[] {
              ScoreMode.COMPLETE, ScoreMode.TOP_SCORES, ScoreMode.COMPLETE_NO_SCORES}) {
            lines.add("case\t" + similarity + "\t" + bits(boost) + "\t" + mode);
            Weight weight = searcher.createWeight(searcher.rewrite(spec.query()), mode, boost);
            Scorer scorer;
            try {
              scorer = scorer(weight, leaf);
            } catch (IllegalStateException expected) {
              if (segment.terms(spec.field()).hasPositions()) throw expected;
              lines.add("error\t" + hex(new BytesRef(expected.getMessage())));
              lines.add("endcase");
              continue;
            }
            if (scorer == null) {
              lines.add("absent");
              lines.add("endcase");
              continue;
            }
            var scores = new ArrayList<Float>();
            var iterator = scorer.iterator();
            for (int doc = iterator.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS;
                doc = iterator.nextDoc()) {
              float score = mode.needsScores() ? scorer.score() : 0f;
              scores.add(score);
              lines.add("doc\t" + doc + "\t" + bits(score));
            }
            if (mode == ScoreMode.TOP_SCORES) {
              // A fresh scorer keeps the shallow trace independent of document iteration.
              var bounds = scorer(weight, leaf);
              int[] targets = java.util.stream.IntStream.of(
                  0, 1, 7, 8, 41, 58, 154, 501, segment.maxDoc())
                  .filter(target -> target <= segment.maxDoc()).distinct().sorted().toArray();
              for (int target : targets) {
                int boundary = bounds.advanceShallow(target);
                if (bounds.docID() != -1) throw new AssertionError("shallow moved postings");
                for (int upTo : new int[] {boundary,
                    boundary == Integer.MAX_VALUE ? boundary : boundary + 1, Integer.MAX_VALUE}) {
                  lines.add("bound\t" + target + "\t" + boundary + "\t" + upTo
                      + "\t" + bits(bounds.getMaxScore(upTo)));
                }
              }
              if (!scores.isEmpty()) {
                scores.sort(Float::compare);
                float threshold = Math.nextUp(scores.get(scores.size() / 2));
                lines.add("threshold\t" + bits(threshold));
                var pruned = scorer(weight, leaf);
                pruned.setMinCompetitiveScore(threshold);
                var competitive = pruned.iterator();
                for (int doc = competitive.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS;
                    doc = competitive.nextDoc()) {
                  lines.add("competitive\t" + doc + "\t" + bits(pruned.score()));
                }
              }
            }
            lines.add("endcase");
          }
        }
      }
      lines.add("endquery");
    }
    Files.write(out.resolve("phrases." + segment.getSegmentName() + ".tsv"), lines);
  }

  public static void main(String[] args) throws Exception {
    // The postings-only fixture names the generator's public SimpleText SPI wrapper.
    GenCodecsSimpleText.registerFormats();
    Path root = Path.of(args[0]);
    for (String family : new String[] {"codec", "postings", "deep"}) {
      for (String variant : new String[] {"flush", "merged"}) {
        Path out = root.resolve("simpletext_phrase_scorer").resolve(family).resolve(variant);
        Files.createDirectories(out);
        Path index = family.equals("deep") ? out.resolve("index")
            : root.resolve("codecs_simpletext").resolve(family).resolve(variant).resolve("index");
        if (family.equals("deep")) deepIndex(index, variant.equals("merged"));
        try (var directory = FSDirectory.open(index);
            var reader = DirectoryReader.open(directory)) {
          var segments = new ArrayList<String>();
          for (var leaf : reader.leaves()) {
            segments.add(((SegmentReader) leaf.reader()).getSegmentName());
            write(out, reader, leaf);
          }
          Files.write(out.resolve("segments.tsv"), segments);
        }
      }
    }
  }
}
