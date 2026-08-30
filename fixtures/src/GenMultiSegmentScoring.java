import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.SegmentInfos;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.Terms;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Comparator;

/**
 * A genuine <b>two-segment</b> real-Lucene index whose two segments have
 * deliberately lopsided statistics, plus real {@link IndexSearcher} {@code TopDocs}
 * recorded bit-for-bit -- the only shape that can catch a per-leaf/reader-wide
 * statistics divergence.
 *
 * <p>Why it is needed. {@code IndexSearcher} computes {@code TermStats} and
 * {@code FieldStats} <i>once across the whole reader</i> and hands the same values to
 * every leaf: {@code docFreq}/{@code docCount} for the idf, and
 * {@code sumTotalTermFreq / docCount} for {@code BM25Similarity}'s {@code avgdl}.
 * A port that derives either from the leaf it happens to be scoring produces a
 * different score for the same document, and the top-k then fills from whichever
 * segment makes the term look rarest or its documents look shortest. Every other
 * scoring fixture in this tree is a single segment, where per-leaf and reader-wide
 * are the same number by construction, so none of them can see it.
 *
 * <p>The corpus is built so the two are as far apart as practical:
 *
 * <ul>
 *   <li><b>Segment 0</b> -- four very short documents (1-3 terms). Its own
 *       {@code avgdl} is around 2.
 *   <li><b>Segment 1</b> -- four long documents (40+ terms). Its own {@code avgdl}
 *       is around 40, i.e. ~20x segment 0's, and the reader-wide value sits
 *       between them, equal to neither.
 *   <li>{@code fox} appears in one of four documents in segment 0 and in three of
 *       four in segment 1, so the per-leaf and reader-wide idf differ as well.
 * </ul>
 *
 * <p>{@code NoMergePolicy} plus a {@code commit()} between the two batches is what
 * guarantees two segments survive to the committed index; without it Lucene is free
 * to merge them into one and the fixture silently stops testing anything.
 *
 * <p>Recorded doc ids are <b>global</b> (segment 1's are offset by segment 0's
 * {@code maxDoc}), which is exactly the space a multi-segment search returns.
 */
public class GenMultiSegmentScoring {

  private static final String[] SHORT_DOCS = {
    "fox",
    "cat dog",
    "bird cat dog",
    "dog",
  };

  public static void main(String[] args) throws IOException {
    Path out = Path.of(args[0]).resolve("multi_segment_scoring_index");
    if (Files.exists(out)) {
      deleteRecursive(out);
    }
    Files.createDirectories(out);

    StringBuilder m = new StringBuilder();

    try (Directory dir = FSDirectory.open(out)) {
      IndexWriterConfig cfg = new IndexWriterConfig(new StandardAnalyzer());
      cfg.setUseCompoundFile(false);
      cfg.setMergePolicy(NoMergePolicy.INSTANCE);
      try (IndexWriter w = new IndexWriter(dir, cfg)) {
        for (String body : SHORT_DOCS) {
          Document doc = new Document();
          doc.add(new TextField("body", body, Field.Store.NO));
          w.addDocument(doc);
        }
        w.commit(); // segment 0: short documents, `fox` in 1 of 4

        for (int i = 0; i < 4; i++) {
          Document doc = new Document();
          doc.add(new TextField("body", longBody(i), Field.Store.NO));
          w.addDocument(doc);
        }
        w.commit(); // segment 1: long documents, `fox` in 3 of 4
      }

      SegmentInfos sis = SegmentInfos.readLatestCommit(dir);
      if (sis.size() != 2) {
        throw new AssertionError("expected exactly two segments, got " + sis.size());
      }
      m.append("segment_count=").append(sis.size()).append('\n');
      int docBase = 0;
      for (int i = 0; i < sis.size(); i++) {
        m.append("segment.").append(i).append(".name=").append(sis.info(i).info.name).append('\n');
        m.append("segment.")
            .append(i)
            .append(".max_doc=")
            .append(sis.info(i).info.maxDoc())
            .append('\n');
        m.append("segment.").append(i).append(".doc_base=").append(docBase).append('\n');
        docBase += sis.info(i).info.maxDoc();
      }

      try (DirectoryReader reader = DirectoryReader.open(dir)) {
        if (reader.leaves().size() != 2) {
          throw new AssertionError("expected two leaves, got " + reader.leaves().size());
        }
        IndexSearcher searcher = new IndexSearcher(reader);

        // Per-leaf counters, and the reader-wide sums IndexSearcher.fieldStats
        // computes from them. A port that scores from the per-leaf numbers
        // produces `avgdl` = one of `segment.N.avgdl`; Lucene uses `avgdl`.
        long sumTotalTermFreq = 0;
        long docCount = 0;
        for (int i = 0; i < reader.leaves().size(); i++) {
          LeafReaderContext leaf = reader.leaves().get(i);
          Terms terms = leaf.reader().terms("body");
          sumTotalTermFreq += terms.getSumTotalTermFreq();
          docCount += terms.getDocCount();
          m.append("segment.")
              .append(i)
              .append(".sum_total_term_freq=")
              .append(terms.getSumTotalTermFreq())
              .append('\n');
          m.append("segment.")
              .append(i)
              .append(".doc_count=")
              .append(terms.getDocCount())
              .append('\n');
          m.append("segment.")
              .append(i)
              .append(".doc_freq.fox=")
              .append(leaf.reader().docFreq(new Term("body", "fox")))
              .append('\n');
          float leafAvgdl =
              (float) (terms.getSumTotalTermFreq() / (double) terms.getDocCount());
          m.append("segment.").append(i).append(".avgdl=").append(leafAvgdl).append('\n');
          m.append("segment.")
              .append(i)
              .append(".avgdl.bits=")
              .append(Float.floatToIntBits(leafAvgdl))
              .append('\n');
        }
        m.append("sum_total_term_freq=").append(sumTotalTermFreq).append('\n');
        m.append("doc_count=").append(docCount).append('\n');
        m.append("doc_freq.fox=")
            .append(reader.docFreq(new Term("body", "fox")))
            .append('\n');
        float avgdl = (float) (sumTotalTermFreq / (double) docCount);
        m.append("avgdl=").append(avgdl).append('\n');
        m.append("avgdl.bits=").append(Float.floatToIntBits(avgdl)).append('\n');

        record(m, "scoring.term.fox", searcher, new TermQuery(new Term("body", "fox")));
        record(m, "scoring.term.dog", searcher, new TermQuery(new Term("body", "dog")));
        record(
            m,
            "scoring.boolean.should.fox.dog",
            searcher,
            new BooleanQuery.Builder()
                .add(new TermQuery(new Term("body", "fox")), BooleanClause.Occur.SHOULD)
                .add(new TermQuery(new Term("body", "dog")), BooleanClause.Occur.SHOULD)
                .build());
      }
    }

    Files.writeString(out.resolve("manifest.properties"), m.toString());
    System.out.println("wrote " + out);
  }

  /**
   * A ~40-term document. Terms are drawn from a small vocabulary so `fox`'s
   * frequency, and therefore its score, differs per document; document 3 omits
   * `fox` entirely so `docFreq` is 3 of 4 here against 1 of 4 in segment 0.
   */
  private static String longBody(int i) {
    StringBuilder sb = new StringBuilder();
    for (int t = 0; t < 40; t++) {
      if (sb.length() > 0) {
        sb.append(' ');
      }
      if (i < 3 && t % (i + 2) == 0) {
        sb.append("fox");
      } else if (t % 3 == 0) {
        sb.append("dog");
      } else if (t % 3 == 1) {
        sb.append("cat");
      } else {
        sb.append("bird");
      }
    }
    return sb.toString();
  }

  /**
   * {@code <key>.docScores=doc:score,...} plus {@code <key>.bits=doc:<raw float
   * bits>,...}, in global doc-id space, exactly as {@link AppendScoringManifest}
   * records them for the single-segment fixture.
   */
  private static void record(StringBuilder out, String key, IndexSearcher searcher, Query query)
      throws IOException {
    TopDocs td = searcher.search(query, 20);
    StringBuilder scores = new StringBuilder();
    StringBuilder bits = new StringBuilder();
    for (var sd : td.scoreDocs) {
      if (scores.length() > 0) {
        scores.append(',');
        bits.append(',');
      }
      scores.append(sd.doc).append(':').append(sd.score);
      bits.append(sd.doc).append(':').append(Float.floatToIntBits(sd.score));
    }
    out.append(key).append(".query=").append(query).append('\n');
    out.append(key).append(".docScores=").append(scores).append('\n');
    out.append(key).append(".bits=").append(bits).append('\n');
  }

  private static void deleteRecursive(Path p) throws IOException {
    try (var walk = Files.walk(p)) {
      walk.sorted(Comparator.reverseOrder())
          .forEach(
              q -> {
                try {
                  Files.delete(q);
                } catch (IOException e) {
                  throw new RuntimeException(e);
                }
              });
    }
  }
}
