import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;
import java.util.function.Supplier;
import org.apache.commons.codec.language.Caverphone2;
import org.apache.commons.codec.language.ColognePhonetic;
import org.apache.commons.codec.language.DoubleMetaphone;
import org.apache.commons.codec.language.MatchRatingApproachEncoder;
import org.apache.commons.codec.language.Metaphone;
import org.apache.commons.codec.language.Nysiis;
import org.apache.commons.codec.language.RefinedSoundex;
import org.apache.commons.codec.language.Soundex;
import org.apache.commons.codec.language.bm.NameType;
import org.apache.commons.codec.language.bm.PhoneticEngine;
import org.apache.commons.codec.language.bm.RuleType;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.Tokenizer;
import org.apache.lucene.analysis.core.WhitespaceTokenizer;
import org.apache.lucene.analysis.phonetic.BeiderMorseFilter;
import org.apache.lucene.analysis.phonetic.DaitchMokotoffSoundexFilter;
import org.apache.lucene.analysis.phonetic.DoubleMetaphoneFilter;
import org.apache.lucene.analysis.phonetic.PhoneticFilter;
import org.apache.lucene.analysis.morfologik.MorfologikFilter;
import org.apache.lucene.analysis.pl.PolishAnalyzer;
import org.apache.lucene.analysis.stempel.StempelFilter;
import org.apache.lucene.analysis.stempel.StempelStemmer;
import org.apache.lucene.analysis.uk.UkrainianMorfologikAnalyzer;
import morfologik.stemming.polish.PolishStemmer;

/**
 * M12's pairs (the Rust twin is {@code benchmarks/rust-runner/src/micro_analysis_m12.rs}): the
 * language modules' filters over corpora both sides generate from {@link SweepMicro.Rng}. Units are
 * tokens; output is {@code name\tns_per_token\ttokens}.
 */
public class AnalysisM12Micro {
  /** Name fragments the generated names are joined from. */
  static final String[] SYLLABLES = {
    "an", "ber", "schm", "idt", "ko", "wal", "ski", "mc", "don", "ald", "ph", "ough", "tz", "sch",
    "ch", "cz", "rz", "ei", "ie", "ou", "th", "gh", "w", "y", "ss", "ll", "tt", "n", "m", "r", "l",
    "k", "s", "t", "d", "b", "g", "p", "f", "v", "z", "x", "q", "j", "h", "a", "e", "i", "o", "u"
  };

  static long rem(long x, long n) {
    return Long.remainderUnsigned(x, n);
  }

  /** {@code docs} documents of 50 names of 2 to 4 syllables, capitalised. */
  static List<String> names(long seed, int docs) {
    SweepMicro.Rng r = new SweepMicro.Rng(seed);
    List<String> out = new ArrayList<>();
    for (int d = 0; d < docs; d++) {
      StringBuilder s = new StringBuilder();
      for (int w = 0; w < 50; w++) {
        if (w > 0) s.append(' ');
        int n = 2 + (int) rem(r.next(), 3);
        StringBuilder name = new StringBuilder();
        for (int k = 0; k < n; k++) name.append(SYLLABLES[(int) rem(r.next(), SYLLABLES.length)]);
        name.setCharAt(0, Character.toUpperCase(name.charAt(0)));
        s.append(name);
      }
      out.add(s.toString());
    }
    return out;
  }

  /**
   * Documents of 50 words each, the first column of a fixture file (lines holding an escape
   * skipped), read from the repository root as the Rust side reads it.
   */
  static List<String> fixtureDocs(String file) throws IOException {
    List<String> words = new ArrayList<>();
    for (String line : Files.readAllLines(Path.of(file), StandardCharsets.UTF_8)) {
      String w = line.split("\t", -1)[0];
      if (!w.isEmpty() && w.indexOf('\\') < 0 && w.indexOf(' ') < 0) words.add(w);
    }
    List<String> docs = new ArrayList<>();
    for (int i = 0; i + 50 <= words.size(); i += 50) docs.add(String.join(" ", words.subList(i, i + 50)));
    return docs;
  }

  static Analyzer chain(Supplier<Tokenizer> tok, Function<TokenStream, TokenStream> filters) {
    return new Analyzer() {
      @Override
      protected TokenStreamComponents createComponents(String fieldName) {
        Tokenizer t = tok.get();
        return new TokenStreamComponents(t, filters.apply(t));
      }
    };
  }

  static void run(String name, Analyzer a, List<String> docs) throws IOException {
    SweepMicro.measure(
        name,
        () -> {
          long tokens = 0;
          for (String text : docs) tokens += SweepMicro.consume(a.tokenStream("body", text));
          return tokens;
        });
    a.close();
  }

  public static void main(String[] args) throws Exception {
    List<String> names = names(0x9E37_79B9_7F4A_7C15L, 100);
    List<String> few = names(0x2545_F491_4F6C_DD1DL, 10);
    run("ph_soundex", chain(WhitespaceTokenizer::new, t -> new PhoneticFilter(t, new Soundex(), true)), names);
    run("ph_refined_soundex", chain(WhitespaceTokenizer::new, t -> new PhoneticFilter(t, new RefinedSoundex(), false)), names);
    run("ph_metaphone", chain(WhitespaceTokenizer::new, t -> new PhoneticFilter(t, new Metaphone(), true)), names);
    run("ph_double_metaphone", chain(WhitespaceTokenizer::new, t -> new PhoneticFilter(t, new DoubleMetaphone(), true)), names);
    run("ph_caverphone2", chain(WhitespaceTokenizer::new, t -> new PhoneticFilter(t, new Caverphone2(), false)), names);
    run("ph_cologne", chain(WhitespaceTokenizer::new, t -> new PhoneticFilter(t, new ColognePhonetic(), true)), names);
    run("ph_nysiis", chain(WhitespaceTokenizer::new, t -> new PhoneticFilter(t, new Nysiis(), true)), names);
    run("ph_mra", chain(WhitespaceTokenizer::new, t -> new PhoneticFilter(t, new MatchRatingApproachEncoder(), true)), names);
    run("double_metaphone_filter", chain(WhitespaceTokenizer::new, t -> new DoubleMetaphoneFilter(t, 4, true)), names);
    run("daitch_mokotoff", chain(WhitespaceTokenizer::new, t -> new DaitchMokotoffSoundexFilter(t, true)), names);
    PhoneticEngine approx = new PhoneticEngine(NameType.GENERIC, RuleType.APPROX, true);
    PhoneticEngine exact = new PhoneticEngine(NameType.ASHKENAZI, RuleType.EXACT, true);
    run("beider_morse_gen_approx", chain(WhitespaceTokenizer::new, t -> new BeiderMorseFilter(t, approx)), few);
    run("beider_morse_ash_exact", chain(WhitespaceTokenizer::new, t -> new BeiderMorseFilter(t, exact)), few);
    List<String> polish = fixtureDocs("fixtures/data/analysis_stempel/stems.tsv");
    run("stempel_filter", chain(WhitespaceTokenizer::new,
        t -> new StempelFilter(t, new StempelStemmer(PolishAnalyzer.getDefaultTable()))), polish);
    run("polish_analyzer", new PolishAnalyzer(), polish);
    List<String> morf = fixtureDocs("fixtures/data/analysis_morfologik/lookups_polish.tsv");
    morfologik.stemming.Dictionary dict = new PolishStemmer().getDictionary();
    run("morfologik_filter", chain(WhitespaceTokenizer::new, t -> new MorfologikFilter(t, dict)), morf);
    run("ukrainian_analyzer", new UkrainianMorfologikAnalyzer(),
        fixtureDocs("fixtures/data/analysis_morfologik/lookups_ukrainian.tsv"));
  }
}
