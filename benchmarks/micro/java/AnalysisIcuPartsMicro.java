import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.core.KeywordTokenizer;
import org.apache.lucene.analysis.core.WhitespaceTokenizer;
import org.apache.lucene.analysis.icu.ICUFoldingFilter;
import org.apache.lucene.analysis.icu.segmentation.DefaultICUTokenizerConfig;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import com.ibm.icu.text.BreakIterator;
import com.ibm.icu.text.Collator;
import com.ibm.icu.text.Normalizer;
import com.ibm.icu.text.Normalizer2;
import com.ibm.icu.text.RawCollationKey;
import com.ibm.icu.text.ReplaceableString;
import com.ibm.icu.text.RuleBasedBreakIterator;
import com.ibm.icu.text.Transliterator;
import com.ibm.icu.util.ULocale;

/**
 * M12's ICU gaps taken apart (the Rust twin is {@code
 * benchmarks/rust-runner/src/micro_analysis_icu_parts.rs}): the tokenizers alone and each ICU
 * kernel alone over the inputs {@code AnalysisIcuMicro} analyzes.
 */
public class AnalysisIcuPartsMicro {
  static Analyzer only(java.util.function.Supplier<org.apache.lucene.analysis.Tokenizer> tok) {
    return new Analyzer() {
      @Override
      protected TokenStreamComponents createComponents(String fieldName) {
        return new TokenStreamComponents(tok.get());
      }
    };
  }

  static void chain(String name, Analyzer a, List<String> docs) throws IOException {
    SweepMicro.measure(
        name,
        () -> {
          long tokens = 0;
          for (String text : docs) tokens += SweepMicro.consume(a.tokenStream("body", text));
          return tokens;
        });
    a.close();
  }

  static List<String> terms(List<String> docs) throws IOException {
    List<String> out = new ArrayList<>();
    try (Analyzer a = only(WhitespaceTokenizer::new)) {
      for (String text : docs) {
        try (TokenStream ts = a.tokenStream("body", text)) {
          CharTermAttribute term = ts.addAttribute(CharTermAttribute.class);
          ts.reset();
          while (ts.incrementToken()) out.add(term.toString());
          ts.end();
        }
      }
    }
    return out;
  }

  public static void main(String[] args) throws Exception {
    List<String> docs = AnalysisIcuMicro.docs();
    List<String> terms = terms(docs);

    chain("whitespace_only", only(WhitespaceTokenizer::new), docs);
    chain("keyword_only", only(KeywordTokenizer::new), docs);

    Normalizer2 fold = ICUFoldingFilter.NORMALIZER;
    StringBuilder buf = new StringBuilder();
    SweepMicro.measure(
        "fold_kernel_utf16",
        () -> {
          for (String t : terms) {
            if (fold.quickCheck(t) != Normalizer.YES) {
              buf.setLength(0);
              fold.normalize(t, buf);
              SweepMicro.sink += buf.length();
            }
          }
          return terms.size();
        });
    SweepMicro.measure(
        "fold_kernel_from_utf8",
        () -> {
          for (String t : terms) {
            if (fold.quickCheck(t) != Normalizer.YES) {
              buf.setLength(0);
              fold.normalize(t, buf);
              SweepMicro.sink += buf.length();
            }
          }
          return terms.size();
        });

    Transliterator ts = Transliterator.getInstance("Traditional-Simplified");
    SweepMicro.measure(
        "trad_simp_kernel_utf16",
        () -> {
          for (String t : terms) {
            ReplaceableString r = new ReplaceableString(t);
            Transliterator.Position pos = new Transliterator.Position(0, t.length(), 0, t.length());
            ts.filteredTransliterate(r, pos, false);
            SweepMicro.sink += r.length();
          }
          return terms.size();
        });

    Collator root = Collator.getInstance(ULocale.ROOT);
    RawCollationKey key = new RawCollationKey();
    SweepMicro.measure(
        "coll_key_kernel_utf8",
        () -> {
          for (String d : docs) {
            root.getRawCollationKey(d, key);
            SweepMicro.sink += key.size;
          }
          return docs.size();
        });
    SweepMicro.measure(
        "coll_key_kernel_utf16",
        () -> {
          for (String d : docs) {
            root.getRawCollationKey(d, key);
            SweepMicro.sink += key.size;
          }
          return docs.size();
        });

    RuleBasedBreakIterator bi;
    try (InputStream in = DefaultICUTokenizerConfig.class.getResourceAsStream("Default.brk")) {
      bi = RuleBasedBreakIterator.getInstanceFromCompiledRules(in);
    }
    SweepMicro.measure(
        "default_brk_kernel",
        () -> {
          long n = 0;
          for (String d : docs) {
            bi.setText(d);
            for (int b = bi.first(); b != BreakIterator.DONE; b = bi.next()) n++;
          }
          return n;
        });
  }
}
