import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.analysis.tokenattributes.TermFrequencyAttribute;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.FieldType;
import org.apache.lucene.document.StoredField;
import org.apache.lucene.index.CheckIndex;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.NumericDocValues;
import org.apache.lucene.index.PostingsEnum;
import org.apache.lucene.index.StoredFields;
import org.apache.lucene.index.TermsEnum;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.BytesRef;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

/**
 * Verifies {@code crates/lucene-index/examples/write_custom_freq_norms_fixture.rs}: a {@code
 * DOCS_AND_CUSTOM_FREQS} field {@code score} that keeps its norms. For {@code <dir>/flushed} and
 * {@code <dir>/merged} (one segment each), Lucene indexes the same {@code (term, freq)} pairs
 * through its own {@code IndexWriter} -- a token stream setting {@code TermFrequencyAttribute} --
 * and every document's norm, and every posting's frequency, must equal what Lucene wrote; {@code
 * CheckIndex} must be clean. Documents are matched by their stored {@code id}.
 */
public class VerifyCustomFreqNorms {
  static final int DOCS = 120;

  /** Document {@code i}'s pairs, as {@code terms} in the Rust example. */
  static List<Map.Entry<String, Integer>> terms(int i) {
    List<Map.Entry<String, Integer>> out = new ArrayList<>();
    if (i % 7 == 0) {
      return out;
    }
    for (int k = 0; k < 1 + i % 5; k++) {
      out.add(Map.entry("t" + ((i + 3 * k) % 13), (i * 31 + k * 17) % 50 + 1));
    }
    return out;
  }

  /** One token per pair, its frequency in {@code TermFrequencyAttribute}. */
  static final class Pairs extends TokenStream {
    final CharTermAttribute term = addAttribute(CharTermAttribute.class);
    final TermFrequencyAttribute freq = addAttribute(TermFrequencyAttribute.class);
    final List<Map.Entry<String, Integer>> pairs;
    int next;

    Pairs(List<Map.Entry<String, Integer>> pairs) {
      this.pairs = pairs;
    }

    @Override
    public boolean incrementToken() {
      if (next == pairs.size()) {
        return false;
      }
      clearAttributes();
      Map.Entry<String, Integer> p = pairs.get(next++);
      term.setEmpty().append(p.getKey());
      freq.setTermFrequency(p.getValue());
      return true;
    }
  }

  public static void main(String[] args) throws Exception {
    FieldType type = new FieldType();
    type.setIndexOptions(IndexOptions.DOCS_AND_CUSTOM_FREQS);
    type.setTokenized(true);
    type.setOmitNorms(false);
    type.freeze();
    for (String sub : new String[] {"flushed", "merged"}) {
      int docs = sub.equals("flushed") ? DOCS : 3 * DOCS;
      try (Directory rust = FSDirectory.open(Path.of(args[0]).resolve(sub));
          Directory java = new ByteBuffersDirectory()) {
        try (IndexWriter w = new IndexWriter(java, new IndexWriterConfig())) {
          for (int i = 0; i < docs; i++) {
            Document d = new Document();
            d.add(new StoredField("id", Integer.toString(i)));
            List<Map.Entry<String, Integer>> pairs = terms(i);
            if (!pairs.isEmpty()) {
              d.add(new Field("score", new Pairs(pairs), type));
            }
            w.addDocument(d);
          }
          w.forceMerge(1);
        }
        Map<Integer, String> expected = dump(java);
        Map<Integer, String> actual = dump(rust);
        if (expected.size() != docs) {
          throw new AssertionError(sub + ": Lucene's own index has " + expected.size() + " docs");
        }
        for (int i = 0; i < docs; i++) {
          if (!expected.get(i).equals(actual.get(i))) {
            throw new AssertionError(
                sub + ": doc " + i + ": lucene " + expected.get(i) + ", rust " + actual.get(i));
          }
        }
        try (CheckIndex check = new CheckIndex(rust)) {
          if (!check.checkIndex().clean) {
            throw new AssertionError(sub + ": CheckIndex failed");
          }
        }
      }
    }
    System.out.println("VerifyCustomFreqNorms: ok");
  }

  /** Per stored id: the {@code score} norm (or "none") and every {@code term:freq}. */
  static Map<Integer, String> dump(Directory dir) throws Exception {
    Map<Integer, String> out = new TreeMap<>();
    try (DirectoryReader reader = DirectoryReader.open(dir)) {
      if (reader.leaves().size() != 1) {
        throw new AssertionError("one segment expected, got " + reader.leaves().size());
      }
      LeafReader leaf = reader.leaves().get(0).reader();
      StoredFields stored = leaf.storedFields();
      int[] ids = new int[leaf.maxDoc()];
      StringBuilder[] lines = new StringBuilder[leaf.maxDoc()];
      NumericDocValues norms = leaf.getNormValues("score");
      for (int doc = 0; doc < leaf.maxDoc(); doc++) {
        ids[doc] = Integer.parseInt(stored.document(doc).get("id"));
        String norm = "none";
        if (norms != null && norms.advanceExact(doc)) {
          norm = Long.toString(norms.longValue());
        }
        lines[doc] = new StringBuilder("norm=" + norm);
      }
      if (leaf.terms("score") != null) {
        TermsEnum te = leaf.terms("score").iterator();
        BytesRef term;
        while ((term = te.next()) != null) {
          PostingsEnum pe = te.postings(null, PostingsEnum.FREQS);
          while (pe.nextDoc() != DocIdSetIterator.NO_MORE_DOCS) {
            lines[pe.docID()].append(' ').append(term.utf8ToString()).append(':').append(pe.freq());
          }
        }
      }
      for (int doc = 0; doc < leaf.maxDoc(); doc++) {
        out.put(ids[doc], lines[doc].toString());
      }
    }
    return out;
  }
}
