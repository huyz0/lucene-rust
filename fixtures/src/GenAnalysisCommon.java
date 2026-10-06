import java.io.Reader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.Function;
import java.util.function.Supplier;
import java.util.regex.Pattern;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.CharArraySet;
import org.apache.lucene.analysis.LowerCaseFilter;
import org.apache.lucene.analysis.StopFilter;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.Tokenizer;
import org.apache.lucene.analysis.boost.DelimitedBoostTokenFilter;
import org.apache.lucene.analysis.charfilter.HTMLStripCharFilter;
import org.apache.lucene.analysis.charfilter.MappingCharFilter;
import org.apache.lucene.analysis.charfilter.NormalizeCharMap;
import org.apache.lucene.analysis.cjk.CJKAnalyzer;
import org.apache.lucene.analysis.cjk.CJKBigramFilter;
import org.apache.lucene.analysis.cjk.CJKWidthCharFilter;
import org.apache.lucene.analysis.cjk.CJKWidthFilter;
import org.apache.lucene.analysis.commongrams.CommonGramsFilter;
import org.apache.lucene.analysis.commongrams.CommonGramsQueryFilter;
import org.apache.lucene.analysis.core.DecimalDigitFilter;
import org.apache.lucene.analysis.core.FlattenGraphFilter;
import org.apache.lucene.analysis.core.KeywordAnalyzer;
import org.apache.lucene.analysis.core.KeywordTokenizer;
import org.apache.lucene.analysis.core.LetterTokenizer;
import org.apache.lucene.analysis.core.SimpleAnalyzer;
import org.apache.lucene.analysis.core.StopAnalyzer;
import org.apache.lucene.analysis.core.TypeTokenFilter;
import org.apache.lucene.analysis.core.UnicodeWhitespaceAnalyzer;
import org.apache.lucene.analysis.core.UnicodeWhitespaceTokenizer;
import org.apache.lucene.analysis.core.UpperCaseFilter;
import org.apache.lucene.analysis.core.WhitespaceAnalyzer;
import org.apache.lucene.analysis.core.WhitespaceTokenizer;
import org.apache.lucene.analysis.email.UAX29URLEmailAnalyzer;
import org.apache.lucene.analysis.email.UAX29URLEmailTokenizer;
import org.apache.lucene.analysis.en.EnglishAnalyzer;
import org.apache.lucene.analysis.en.EnglishMinimalStemFilter;
import org.apache.lucene.analysis.en.EnglishPossessiveFilter;
import org.apache.lucene.analysis.en.KStemFilter;
import org.apache.lucene.analysis.en.PorterStemFilter;
import org.apache.lucene.analysis.minhash.MinHashFilter;
import org.apache.lucene.analysis.miscellaneous.ASCIIFoldingFilter;
import org.apache.lucene.analysis.miscellaneous.CapitalizationFilter;
import org.apache.lucene.analysis.miscellaneous.CodepointCountFilter;
import org.apache.lucene.analysis.miscellaneous.ConcatenateGraphFilter;
import org.apache.lucene.analysis.miscellaneous.ConditionalTokenFilter;
import org.apache.lucene.analysis.miscellaneous.DelimitedTermFrequencyTokenFilter;
import org.apache.lucene.analysis.miscellaneous.DropIfFlaggedFilter;
import org.apache.lucene.analysis.miscellaneous.FingerprintFilter;
import org.apache.lucene.analysis.miscellaneous.FixBrokenOffsetsFilter;
import org.apache.lucene.analysis.miscellaneous.HyphenatedWordsFilter;
import org.apache.lucene.analysis.miscellaneous.KeepWordFilter;
import org.apache.lucene.analysis.miscellaneous.KeywordRepeatFilter;
import org.apache.lucene.analysis.miscellaneous.LengthFilter;
import org.apache.lucene.analysis.miscellaneous.LimitTokenCountFilter;
import org.apache.lucene.analysis.miscellaneous.LimitTokenOffsetFilter;
import org.apache.lucene.analysis.miscellaneous.LimitTokenPositionFilter;
import org.apache.lucene.analysis.miscellaneous.PatternKeywordMarkerFilter;
import org.apache.lucene.analysis.miscellaneous.ProtectedTermFilter;
import org.apache.lucene.analysis.miscellaneous.RemoveDuplicatesTokenFilter;
import org.apache.lucene.analysis.miscellaneous.ScandinavianFoldingFilter;
import org.apache.lucene.analysis.miscellaneous.ScandinavianNormalizationFilter;
import org.apache.lucene.analysis.miscellaneous.SetKeywordMarkerFilter;
import org.apache.lucene.analysis.miscellaneous.StemmerOverrideFilter;
import org.apache.lucene.analysis.miscellaneous.TrimFilter;
import org.apache.lucene.analysis.miscellaneous.TruncateTokenFilter;
import org.apache.lucene.analysis.miscellaneous.TypeAsSynonymFilter;
import org.apache.lucene.analysis.miscellaneous.WordDelimiterGraphFilter;
import org.apache.lucene.analysis.ngram.EdgeNGramTokenFilter;
import org.apache.lucene.analysis.ngram.EdgeNGramTokenizer;
import org.apache.lucene.analysis.ngram.NGramTokenFilter;
import org.apache.lucene.analysis.ngram.NGramTokenizer;
import org.apache.lucene.analysis.path.PathHierarchyTokenizer;
import org.apache.lucene.analysis.path.ReversePathHierarchyTokenizer;
import org.apache.lucene.analysis.pattern.PatternCaptureGroupTokenFilter;
import org.apache.lucene.analysis.pattern.PatternReplaceCharFilter;
import org.apache.lucene.analysis.pattern.PatternReplaceFilter;
import org.apache.lucene.analysis.pattern.PatternTokenizer;
import org.apache.lucene.analysis.pattern.SimplePatternSplitTokenizer;
import org.apache.lucene.analysis.pattern.SimplePatternTokenizer;
import org.apache.lucene.analysis.payloads.DelimitedPayloadTokenFilter;
import org.apache.lucene.analysis.payloads.FloatEncoder;
import org.apache.lucene.analysis.payloads.IdentityEncoder;
import org.apache.lucene.analysis.payloads.IntegerEncoder;
import org.apache.lucene.analysis.payloads.NumericPayloadTokenFilter;
import org.apache.lucene.analysis.payloads.TokenOffsetPayloadTokenFilter;
import org.apache.lucene.analysis.payloads.TypeAsPayloadTokenFilter;
import org.apache.lucene.analysis.shingle.FixedShingleFilter;
import org.apache.lucene.analysis.shingle.ShingleFilter;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.analysis.standard.StandardTokenizer;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.analysis.tokenattributes.FlagsAttribute;
import org.apache.lucene.analysis.tokenattributes.KeywordAttribute;
import org.apache.lucene.analysis.tokenattributes.OffsetAttribute;
import org.apache.lucene.analysis.tokenattributes.PayloadAttribute;
import org.apache.lucene.analysis.tokenattributes.PositionIncrementAttribute;
import org.apache.lucene.analysis.tokenattributes.PositionLengthAttribute;
import org.apache.lucene.analysis.tokenattributes.TermFrequencyAttribute;
import org.apache.lucene.analysis.tokenattributes.TermToBytesRefAttribute;
import org.apache.lucene.analysis.tokenattributes.TypeAttribute;
import org.apache.lucene.analysis.util.ElisionFilter;
import org.apache.lucene.util.BytesRef;

/**
 * M11 T11.1: the analysis-common differential harness. Runs each named analyzer chain over every
 * line of {@code fixtures/corpus/analysis-common.txt} (a multilingual corpus written for this
 * project) and writes {@code analysis_common/<chain>.tsv}: every token's term, offsets, position
 * increment and length, type, flags, payload, keyword flag and term frequency, then the state
 * {@code end()} leaves -- or the exception a line throws. {@code
 * crates/lucene-analysis/tests/analysis_common_fixtures.rs} rebuilds each chain in Rust and compares
 * token for token.
 *
 * <p>Rows ({@code \t}-separated):
 *
 * <pre>
 *   T line term start end posInc posLen type flags payload(hex|-) keyword(0|1) termFreq
 *   E line finalStart finalEnd finalPosInc
 *   X line ExceptionSimpleName
 * </pre>
 *
 * Terms and types escape {@code \\ \t \n \r} and every other char below U+0020 or a lone surrogate
 * as {@code \\uXXXX}. Each chain is one {@link Analyzer}, reused across the lines as Lucene reuses
 * it, so reset/reuse is exercised too.
 */
public class GenAnalysisCommon {

  /** A chain: optional char filters, a tokenizer, then filters. */
  static Analyzer chain(
      Function<Reader, Reader> charFilters,
      Supplier<Tokenizer> tokenizer,
      Function<TokenStream, TokenStream> filters) {
    return new Analyzer() {
      @Override
      protected TokenStreamComponents createComponents(String fieldName) {
        Tokenizer t = tokenizer.get();
        return new TokenStreamComponents(t, filters.apply(t));
      }

      @Override
      protected Reader initReader(String fieldName, Reader reader) {
        return charFilters.apply(reader);
      }
    };
  }

  static Analyzer chain(Supplier<Tokenizer> tokenizer, Function<TokenStream, TokenStream> filters) {
    return chain(r -> r, tokenizer, filters);
  }

  static Analyzer tok(Supplier<Tokenizer> tokenizer) {
    return chain(r -> r, tokenizer, t -> t);
  }

  static CharArraySet set(boolean ignoreCase, String... words) {
    return new CharArraySet(Arrays.asList(words), ignoreCase);
  }

  static final int WDGF_DEFAULT =
      WordDelimiterGraphFilter.GENERATE_WORD_PARTS
          | WordDelimiterGraphFilter.GENERATE_NUMBER_PARTS
          | WordDelimiterGraphFilter.SPLIT_ON_CASE_CHANGE
          | WordDelimiterGraphFilter.SPLIT_ON_NUMERICS
          | WordDelimiterGraphFilter.STEM_ENGLISH_POSSESSIVE;

  static Map<String, Supplier<Analyzer>> chains() throws Exception {
    Map<String, Supplier<Analyzer>> c = new LinkedHashMap<>();
    CharArraySet en = EnglishAnalyzer.ENGLISH_STOP_WORDS_SET;

    // ---- core
    c.put("standard_analyzer", StandardAnalyzer::new);
    c.put("keyword_analyzer", KeywordAnalyzer::new);
    c.put("whitespace_analyzer", WhitespaceAnalyzer::new);
    c.put("whitespace_max5", () -> new WhitespaceAnalyzer(5));
    c.put("unicode_whitespace_analyzer", UnicodeWhitespaceAnalyzer::new);
    c.put("letter_tokenizer", () -> tok(LetterTokenizer::new));
    c.put("letter_max3", () -> tok(() -> new LetterTokenizer(TokenStream.DEFAULT_TOKEN_ATTRIBUTE_FACTORY, 3)));
    c.put("simple_analyzer", SimpleAnalyzer::new);
    c.put("stop_analyzer", () -> new StopAnalyzer(en));
    c.put("ws_lowercase", () -> chain(WhitespaceTokenizer::new, LowerCaseFilter::new));
    c.put("ws_uppercase", () -> chain(WhitespaceTokenizer::new, UpperCaseFilter::new));
    c.put("ws_decimal_digit", () -> chain(WhitespaceTokenizer::new, DecimalDigitFilter::new));
    c.put("std_stop_lower", () -> chain(StandardTokenizer::new, t -> new StopFilter(new LowerCaseFilter(t), en)));
    c.put("std_type_drop_num", () -> chain(StandardTokenizer::new, t -> new TypeTokenFilter(t, Set.of("<NUM>"))));
    c.put("std_type_keep_alnum", () -> chain(StandardTokenizer::new, t -> new TypeTokenFilter(t, Set.of("<ALPHANUM>"), true)));
    c.put("ws_wdgf_flatten", () -> chain(WhitespaceTokenizer::new,
        t -> new FlattenGraphFilter(new WordDelimiterGraphFilter(t, WDGF_DEFAULT | WordDelimiterGraphFilter.CATENATE_ALL | WordDelimiterGraphFilter.PRESERVE_ORIGINAL, null))));
    c.put("keyword_tokenizer", () -> tok(KeywordTokenizer::new));

    // ---- miscellaneous
    c.put("std_ascii_folding", () -> chain(StandardTokenizer::new, ASCIIFoldingFilter::new));
    c.put("ws_ascii_folding_preserve", () -> chain(WhitespaceTokenizer::new, t -> new ASCIIFoldingFilter(t, true)));
    c.put("ws_wdgf_default", () -> chain(WhitespaceTokenizer::new, t -> new WordDelimiterGraphFilter(t, WDGF_DEFAULT, null)));
    c.put("ws_wdgf_catenate", () -> chain(WhitespaceTokenizer::new, t -> new WordDelimiterGraphFilter(t,
        WDGF_DEFAULT | WordDelimiterGraphFilter.CATENATE_WORDS | WordDelimiterGraphFilter.CATENATE_NUMBERS | WordDelimiterGraphFilter.CATENATE_ALL | WordDelimiterGraphFilter.PRESERVE_ORIGINAL,
        set(false, "AT&T", "j2se"))));
    c.put("ws_wdgf_offsets_off", () -> chain(WhitespaceTokenizer::new, t -> new WordDelimiterGraphFilter(t, false,
        org.apache.lucene.analysis.miscellaneous.WordDelimiterIterator.DEFAULT_WORD_DELIM_TABLE, WordDelimiterGraphFilter.GENERATE_WORD_PARTS | WordDelimiterGraphFilter.CATENATE_WORDS, null)));
    c.put("std_length_2_5", () -> chain(StandardTokenizer::new, t -> new LengthFilter(t, 2, 5)));
    c.put("std_codepoint_count_1_3", () -> chain(StandardTokenizer::new, t -> new CodepointCountFilter(t, 1, 3)));
    c.put("keyword_trim", () -> chain(KeywordTokenizer::new, TrimFilter::new));
    c.put("std_truncate_4", () -> chain(StandardTokenizer::new, t -> new TruncateTokenFilter(t, 4)));
    c.put("std_truncate_cp_2", () -> chain(StandardTokenizer::new, t -> TruncateTokenFilter.truncateAfterCodePoints(t, 2)));
    c.put("std_limit_count_3", () -> chain(StandardTokenizer::new, t -> new LimitTokenCountFilter(t, 3)));
    c.put("std_limit_count_3_all", () -> chain(StandardTokenizer::new, t -> new LimitTokenCountFilter(t, 3, true)));
    c.put("std_limit_offset_20", () -> chain(StandardTokenizer::new, t -> new LimitTokenOffsetFilter(t, 20)));
    c.put("std_limit_position_4", () -> chain(StandardTokenizer::new, t -> new LimitTokenPositionFilter(t, 4)));
    c.put("ws_keyword_marker_porter", () -> chain(WhitespaceTokenizer::new,
        t -> new PorterStemFilter(new SetKeywordMarkerFilter(new LowerCaseFilter(t), set(false, "running", "happiness")))));
    c.put("ws_pattern_keyword_porter", () -> chain(WhitespaceTokenizer::new,
        t -> new PorterStemFilter(new PatternKeywordMarkerFilter(new LowerCaseFilter(t), Pattern.compile("[a-z]+ing")))));
    c.put("ws_keyword_repeat_porter_dedup", () -> chain(WhitespaceTokenizer::new,
        t -> new RemoveDuplicatesTokenFilter(new PorterStemFilter(new KeywordRepeatFilter(new LowerCaseFilter(t))))));
    c.put("ws_stemmer_override_porter", () -> chain(WhitespaceTokenizer::new, t -> {
      try {
        StemmerOverrideFilter.Builder b = new StemmerOverrideFilter.Builder(true);
        b.add("running", "run!");
        b.add("Happiness", "joy");
        b.add("dogs", "dog");
        return new PorterStemFilter(new StemmerOverrideFilter(t, b.build()));
      } catch (Exception e) {
        throw new RuntimeException(e);
      }
    }));
    c.put("std_elision", () -> chain(StandardTokenizer::new,
        t -> new ElisionFilter(t, set(true, "l", "m", "t", "qu", "n", "s", "j", "d", "c", "jusqu", "quoiqu", "lorsqu", "puisqu"))));
    c.put("ws_capitalization", () -> chain(WhitespaceTokenizer::new, CapitalizationFilter::new));
    c.put("ws_capitalization_custom", () -> chain(WhitespaceTokenizer::new, t -> new CapitalizationFilter(t, false,
        set(true, "the", "and"), true, List.of("mc".toCharArray()), 2, 4, 6)));
    c.put("ws_remove_duplicates", () -> chain(WhitespaceTokenizer::new, t -> new RemoveDuplicatesTokenFilter(new LowerCaseFilter(t))));
    c.put("std_fingerprint", () -> chain(StandardTokenizer::new, t -> new FingerprintFilter(new LowerCaseFilter(t))));
    c.put("std_fingerprint_small", () -> chain(StandardTokenizer::new, t -> new FingerprintFilter(t, 20, '_')));
    c.put("std_concatenate_graph", () -> chain(StandardTokenizer::new, ConcatenateGraphFilter::new));
    c.put("ws_delimited_term_frequency", () -> chain(WhitespaceTokenizer::new, DelimitedTermFrequencyTokenFilter::new));
    c.put("ws_protected_term", () -> chain(WhitespaceTokenizer::new,
        t -> new ProtectedTermFilter(set(false, "BROWN", "The"), t, in -> new LowerCaseFilter(in))));
    c.put("ws_conditional_lower", () -> chain(WhitespaceTokenizer::new, t -> new ConditionalTokenFilter(t, LowerCaseFilter::new) {
      private final CharTermAttribute term = addAttribute(CharTermAttribute.class);

      @Override
      protected boolean shouldFilter() {
        return term.length() > 3;
      }
    }));
    c.put("ws_keep_word", () -> chain(WhitespaceTokenizer::new, t -> new KeepWordFilter(t, set(true, "the", "fox", "dog", "quick"))));
    c.put("ws_hyphenated_words", () -> chain(WhitespaceTokenizer::new, HyphenatedWordsFilter::new));
    c.put("std_type_as_synonym", () -> chain(StandardTokenizer::new, t -> new TypeAsSynonymFilter(t, "_type_")));
    c.put("ws_scandinavian_folding", () -> chain(WhitespaceTokenizer::new, ScandinavianFoldingFilter::new));
    c.put("ws_scandinavian_normalization", () -> chain(WhitespaceTokenizer::new, ScandinavianNormalizationFilter::new));
    c.put("std_fix_broken_offsets", () -> chain(StandardTokenizer::new, FixBrokenOffsetsFilter::new));
    c.put("std_drop_if_flagged", () -> chain(StandardTokenizer::new, t -> new DropIfFlaggedFilter(t, 1)));

    // ---- ngram
    c.put("ngram_tokenizer_1_2", () -> tok(() -> new NGramTokenizer(1, 2)));
    c.put("ngram_tokenizer_2_3", () -> tok(() -> new NGramTokenizer(2, 3)));
    c.put("edge_ngram_tokenizer_1_3", () -> tok(() -> new EdgeNGramTokenizer(1, 3)));
    c.put("std_ngram_filter_2_3", () -> chain(StandardTokenizer::new, t -> new NGramTokenFilter(t, 2, 3, false)));
    c.put("std_ngram_filter_2_3_preserve", () -> chain(StandardTokenizer::new, t -> new NGramTokenFilter(t, 2, 3, true)));
    c.put("std_edge_ngram_filter_1_4", () -> chain(StandardTokenizer::new, t -> new EdgeNGramTokenFilter(t, 1, 4, false)));
    c.put("std_edge_ngram_filter_2_3_preserve", () -> chain(StandardTokenizer::new, t -> new EdgeNGramTokenFilter(t, 2, 3, true)));

    // ---- shingle
    c.put("std_shingle_default", () -> chain(StandardTokenizer::new, ShingleFilter::new));
    c.put("std_shingle_2_3_no_unigrams", () -> chain(StandardTokenizer::new, t -> {
      ShingleFilter s = new ShingleFilter(new StopFilter(t, en), 2, 3);
      s.setOutputUnigrams(false);
      s.setTokenSeparator("+");
      s.setFillerToken("*");
      return s;
    }));
    c.put("std_shingle_unigrams_if_none", () -> chain(StandardTokenizer::new, t -> {
      ShingleFilter s = new ShingleFilter(t, 3, 3);
      s.setOutputUnigrams(false);
      s.setOutputUnigramsIfNoShingles(true);
      return s;
    }));
    c.put("std_fixed_shingle_3", () -> chain(StandardTokenizer::new, t -> new FixedShingleFilter(new StopFilter(t, en), 3)));

    // ---- pattern
    c.put("pattern_tokenizer_split", () -> tok(() -> new PatternTokenizer(Pattern.compile("[ ,;.]+"), -1)));
    c.put("pattern_tokenizer_group", () -> tok(() -> new PatternTokenizer(Pattern.compile("([a-z]+)([0-9]*)"), 1)));
    c.put("simple_pattern_tokenizer", () -> tok(() -> new SimplePatternTokenizer("[a-zA-Z]+[0-9]*")));
    c.put("simple_pattern_split_tokenizer", () -> tok(() -> new SimplePatternSplitTokenizer("[ \t,;.]+")));
    c.put("ws_pattern_replace_all", () -> chain(WhitespaceTokenizer::new, t -> new PatternReplaceFilter(t, Pattern.compile("[aeiou]"), "_", true)));
    c.put("ws_pattern_replace_first", () -> chain(WhitespaceTokenizer::new, t -> new PatternReplaceFilter(t, Pattern.compile("([a-z])([a-z]*)"), "$2$1", false)));
    c.put("ws_pattern_capture_group", () -> chain(WhitespaceTokenizer::new,
        t -> new PatternCaptureGroupTokenFilter(t, true, Pattern.compile("([A-Z][a-z]+)"), Pattern.compile("([0-9]+)"))));
    c.put("pattern_replace_char_filter", () -> chain(r -> new PatternReplaceCharFilter(Pattern.compile("([a-z]+)-([a-z]+)"), "$2_$1", r),
        WhitespaceTokenizer::new, t -> t));

    // ---- path
    c.put("path_hierarchy", () -> tok(PathHierarchyTokenizer::new));
    c.put("path_hierarchy_backslash_skip1", () -> tok(() -> new PathHierarchyTokenizer('\\', '/', 1)));
    c.put("reverse_path_hierarchy", () -> tok(ReversePathHierarchyTokenizer::new));
    c.put("reverse_path_hierarchy_dot_skip1", () -> tok(() -> new ReversePathHierarchyTokenizer('.', 1)));

    // ---- charfilter
    c.put("mapping_char_filter", () -> {
      NormalizeCharMap.Builder b = new NormalizeCharMap.Builder();
      b.add("ä", "ae");
      b.add("ß", "ss");
      b.add("fox", "wolf");
      b.add("qu", "kw");
      b.add("the", "");
      b.add("&", " and ");
      b.add("é", "e");
      NormalizeCharMap map = b.build();
      return chain(r -> new MappingCharFilter(map, r), WhitespaceTokenizer::new, t -> t);
    });
    c.put("html_strip_standard", () -> chain(HTMLStripCharFilter::new, StandardTokenizer::new, t -> t));
    c.put("html_strip_keyword", () -> chain(HTMLStripCharFilter::new, KeywordTokenizer::new, t -> t));
    c.put("html_strip_escaped_b", () -> chain(r -> new HTMLStripCharFilter(r, Set.of("b")), WhitespaceTokenizer::new, t -> t));

    // ---- commongrams
    c.put("std_common_grams", () -> chain(StandardTokenizer::new, t -> new CommonGramsFilter(new LowerCaseFilter(t), en)));
    c.put("std_common_grams_query", () -> chain(StandardTokenizer::new,
        t -> new CommonGramsQueryFilter(new CommonGramsFilter(new LowerCaseFilter(t), en))));

    // ---- cjk
    c.put("cjk_analyzer", CJKAnalyzer::new);
    c.put("std_cjk_bigram_unigrams", () -> chain(StandardTokenizer::new,
        t -> new CJKBigramFilter(t, CJKBigramFilter.HAN | CJKBigramFilter.HIRAGANA | CJKBigramFilter.KATAKANA | CJKBigramFilter.HANGUL, true)));
    c.put("std_cjk_bigram_han_only", () -> chain(StandardTokenizer::new, t -> new CJKBigramFilter(t, CJKBigramFilter.HAN)));
    c.put("ws_cjk_width", () -> chain(WhitespaceTokenizer::new, CJKWidthFilter::new));
    c.put("cjk_width_char_filter", () -> chain(CJKWidthCharFilter::new, WhitespaceTokenizer::new, t -> t));

    // ---- payloads / boost
    c.put("ws_delimited_payload_float", () -> chain(WhitespaceTokenizer::new, t -> new DelimitedPayloadTokenFilter(t, '|', new FloatEncoder())));
    c.put("ws_delimited_payload_int", () -> chain(WhitespaceTokenizer::new, t -> new DelimitedPayloadTokenFilter(t, '|', new IntegerEncoder())));
    c.put("ws_delimited_payload_identity", () -> chain(WhitespaceTokenizer::new, t -> new DelimitedPayloadTokenFilter(t, '|', new IdentityEncoder())));
    c.put("std_numeric_payload", () -> chain(StandardTokenizer::new, t -> new NumericPayloadTokenFilter(t, 3.5f, "<NUM>")));
    c.put("std_type_as_payload", () -> chain(StandardTokenizer::new, TypeAsPayloadTokenFilter::new));
    c.put("std_token_offset_payload", () -> chain(StandardTokenizer::new, TokenOffsetPayloadTokenFilter::new));
    c.put("ws_delimited_boost", () -> chain(WhitespaceTokenizer::new, t -> new DelimitedBoostTokenFilter(t, '|')));

    // ---- minhash
    c.put("ws_shingle_minhash", () -> chain(WhitespaceTokenizer::new, t -> {
      ShingleFilter s = new ShingleFilter(t, 2, 2);
      s.setOutputUnigrams(false);
      return new MinHashFilter(s, 4, 2, 1, true);
    }));
    c.put("ws_minhash_64_buckets", () -> chain(WhitespaceTokenizer::new, t -> new MinHashFilter(t, 1, 64, 1, true)));
    c.put("ws_minhash_no_rotation", () -> chain(WhitespaceTokenizer::new, t -> new MinHashFilter(t, 2, 8, 2, false)));

    // ---- email (UAX#29 + URLs + emails)
    c.put("uax29_url_email_analyzer", () -> new UAX29URLEmailAnalyzer(en));
    c.put("uax29_url_email_tokenizer", () -> tok(UAX29URLEmailTokenizer::new));

    // ---- en
    c.put("english_analyzer", EnglishAnalyzer::new);
    c.put("std_english_possessive", () -> chain(StandardTokenizer::new, EnglishPossessiveFilter::new));
    c.put("std_porter", () -> chain(StandardTokenizer::new, t -> new PorterStemFilter(new LowerCaseFilter(t))));
    c.put("std_kstem", () -> chain(StandardTokenizer::new, t -> new KStemFilter(new LowerCaseFilter(t))));
    c.put("std_english_minimal", () -> chain(StandardTokenizer::new, t -> new EnglishMinimalStemFilter(new LowerCaseFilter(t))));
    return c;
  }

  static String esc(String s) {
    StringBuilder b = new StringBuilder();
    for (int i = 0; i < s.length(); i++) {
      char ch = s.charAt(i);
      boolean lone =
          (Character.isHighSurrogate(ch) && (i + 1 >= s.length() || !Character.isLowSurrogate(s.charAt(i + 1))))
              || (Character.isLowSurrogate(ch) && (i == 0 || !Character.isHighSurrogate(s.charAt(i - 1))));
      if (ch == '\\') b.append("\\\\");
      else if (ch == '\t') b.append("\\t");
      else if (ch == '\n') b.append("\\n");
      else if (ch == '\r') b.append("\\r");
      else if (ch < 0x20 || lone) b.append(String.format("\\u%04X", (int) ch));
      else b.append(ch);
    }
    return b.toString();
  }

  static String hex(BytesRef b) {
    if (b == null) return "-";
    StringBuilder s = new StringBuilder();
    for (int i = 0; i < b.length; i++) s.append(String.format("%02x", b.bytes[b.offset + i] & 0xff));
    return s.toString();
  }

  public static void main(String[] args) throws Exception {
    Path out = Path.of(args[0]).resolve("analysis_common");
    Files.createDirectories(out);
    String corpusDir = System.getenv().getOrDefault("FIXTURES_CORPUS", "fixtures/corpus");
    String corpus = Files.readString(Path.of(corpusDir, "analysis-common.txt"), StandardCharsets.UTF_8);
    // Split on '\n' only: the corpus holds U+2028 and U+0085 inside lines.
    List<String> lines = new ArrayList<>(Arrays.asList(corpus.split("\n", -1)));
    if (!lines.isEmpty() && lines.get(lines.size() - 1).isEmpty()) lines.remove(lines.size() - 1);

    for (Map.Entry<String, Supplier<Analyzer>> e : chains().entrySet()) {
      StringBuilder m = new StringBuilder();
      try (Analyzer a = e.getValue().get()) {
        for (int ln = 0; ln < lines.size(); ln++) {
          TokenStream ts = a.tokenStream("f", lines.get(ln));
          try {
            CharTermAttribute term = ts.hasAttribute(CharTermAttribute.class) ? ts.getAttribute(CharTermAttribute.class) : null;
            TermToBytesRefAttribute bytes = ts.getAttribute(TermToBytesRefAttribute.class);
            OffsetAttribute off = ts.addAttribute(OffsetAttribute.class);
            PositionIncrementAttribute inc = ts.addAttribute(PositionIncrementAttribute.class);
            PositionLengthAttribute len = ts.addAttribute(PositionLengthAttribute.class);
            TypeAttribute type = ts.addAttribute(TypeAttribute.class);
            FlagsAttribute flags = ts.addAttribute(FlagsAttribute.class);
            PayloadAttribute payload = ts.addAttribute(PayloadAttribute.class);
            KeywordAttribute kw = ts.addAttribute(KeywordAttribute.class);
            TermFrequencyAttribute tf = ts.addAttribute(TermFrequencyAttribute.class);
            ts.reset();
            while (ts.incrementToken()) {
              String t = term != null ? esc(term.toString()) : "#" + hex(bytes.getBytesRef());
              m.append("T\t").append(ln).append('\t').append(t)
                  .append('\t').append(off.startOffset()).append('\t').append(off.endOffset())
                  .append('\t').append(inc.getPositionIncrement()).append('\t').append(len.getPositionLength())
                  .append('\t').append(esc(type.type())).append('\t').append(flags.getFlags())
                  .append('\t').append(hex(payload.getPayload())).append('\t').append(kw.isKeyword() ? 1 : 0)
                  .append('\t').append(tf.getTermFrequency()).append('\n');
            }
            ts.end();
            m.append("E\t").append(ln).append('\t').append(off.startOffset()).append('\t').append(off.endOffset())
                .append('\t').append(inc.getPositionIncrement()).append('\n');
          } catch (Exception ex) {
            m.append("X\t").append(ln).append('\t').append(ex.getClass().getSimpleName()).append('\n');
          } finally {
            ts.close();
          }
        }
      }
      Files.writeString(out.resolve(e.getKey() + ".tsv"), m.toString(), StandardCharsets.UTF_8);
    }
  }
}
