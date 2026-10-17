import java.io.StringReader;
import java.util.ArrayList;
import java.util.List;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.spell.DirectSpellChecker;
import org.apache.lucene.search.spell.PlainTextDictionary;
import org.apache.lucene.search.spell.SpellChecker;
import org.apache.lucene.search.spell.WordBreakSpellChecker;
import org.apache.lucene.store.ByteBuffersDirectory;

/**
 * M13 T13.3e's spell pair (the Rust twin is {@code benchmarks/rust-runner/src/micro_spell.rs})
 * over 5,000 pseudo-words from a shared LCG, word {@code i} in {@code i % 5 + 1} documents of a
 * one-segment index. {@code direct_suggest}: {@code DirectSpellChecker} top 5 for 500
 * misspellings, units: lookups; {@code spell_suggest}: {@code SpellChecker} top 5 for the same,
 * units: lookups; {@code spell_index}: a spell index built from every word, units: words; {@code
 * word_break}: {@code WordBreakSpellChecker} breaks of 500 word pairs run together (max changes
 * 2), units: lookups. Output is {@code name	ns_per_unit	units}.
 */
public class SpellMicro {
  static long state = 42;

  static long next(long bound) {
    state = state * 6364136223846793005L + 1442695040888963407L;
    return Long.remainderUnsigned(state >>> 33, bound);
  }

  static List<String> words() {
    List<String> out = new ArrayList<>();
    for (int i = 0; i < 5000; i++) {
      long len = 4 + next(7);
      StringBuilder sb = new StringBuilder();
      for (long k = 0; k < len; k++) sb.append((char) ('a' + next(26)));
      out.add(sb.toString());
    }
    return out;
  }

  static List<String> misspellings(List<String> words) {
    List<String> out = new ArrayList<>();
    for (int i = 0; i * 10 < words.size(); i++) {
      StringBuilder b = new StringBuilder(words.get(i * 10));
      int at = i % (b.length() - 1);
      if (i % 2 == 0) {
        b.deleteCharAt(at);
      } else {
        char c = b.charAt(at);
        b.setCharAt(at, b.charAt(at + 1));
        b.setCharAt(at + 1, c);
      }
      out.add(b.toString());
    }
    return out;
  }

  static SpellChecker spellIndex(ByteBuffersDirectory dir, String text) throws java.io.IOException {
    SpellChecker s = new SpellChecker(dir);
    s.indexDictionary(new PlainTextDictionary(new StringReader(text)), new IndexWriterConfig(null), true);
    return s;
  }

  public static void main(String[] args) throws Exception {
    List<String> words = words();
    List<String> queries = misspellings(words);
    ByteBuffersDirectory user = new ByteBuffersDirectory();
    try (IndexWriter iw = new IndexWriter(user, new IndexWriterConfig(null))) {
      for (int i = 0; i < words.size(); i++) {
        for (int k = 0; k <= i % 5; k++) {
          Document doc = new Document();
          doc.add(new StringField("body", words.get(i), Field.Store.NO));
          iw.addDocument(doc);
        }
      }
      iw.forceMerge(1);
    }
    DirectoryReader reader = DirectoryReader.open(user);

    DirectSpellChecker direct = new DirectSpellChecker();
    SweepMicro.measure("direct_suggest", () -> {
      for (String q : queries) {
        SweepMicro.sink += direct.suggestSimilar(new Term("body", q), 5, reader).length;
      }
      return queries.size();
    });

    StringBuilder sb = new StringBuilder();
    for (String w : words) sb.append(w).append('\n');
    String text = sb.toString();
    SpellChecker spell = spellIndex(new ByteBuffersDirectory(), text);
    SweepMicro.measure("spell_suggest", () -> {
      for (String q : queries) SweepMicro.sink += spell.suggestSimilar(q, 5).length;
      return queries.size();
    });
    SweepMicro.measure("spell_index", () -> {
      try (SpellChecker s = spellIndex(new ByteBuffersDirectory(), text)) {
        SweepMicro.sink += s.exist(words.get(0)) ? 1 : 0;
      }
      return words.size();
    });

    List<Term> pairs = new ArrayList<>();
    for (int i = 0; i < 500; i++) {
      pairs.add(new Term("body", words.get(i * 7) + words.get(i * 7 + 3)));
    }
    WordBreakSpellChecker wb = new WordBreakSpellChecker();
    wb.setMaxChanges(2);
    SweepMicro.measure("word_break", () -> {
      for (Term t : pairs) {
        SweepMicro.sink += wb.suggestWordBreaks(t, 5, reader, null, null).length;
      }
      return pairs.size();
    });
  }
}
