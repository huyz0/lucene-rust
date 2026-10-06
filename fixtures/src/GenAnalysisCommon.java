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
import java.util.regex.Matcher;
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
import org.apache.lucene.analysis.miscellaneous.LimitTokenCountAnalyzer;
import org.apache.lucene.analysis.miscellaneous.LimitTokenCountFilter;
import org.apache.lucene.analysis.miscellaneous.LimitTokenOffsetFilter;
import org.apache.lucene.analysis.miscellaneous.LimitTokenPositionFilter;
import org.apache.lucene.analysis.miscellaneous.PatternKeywordMarkerFilter;
import org.apache.lucene.analysis.miscellaneous.PerFieldAnalyzerWrapper;
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
import org.apache.lucene.analysis.pattern.PatternTypingFilter;
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
import org.apache.lucene.analysis.shingle.ShingleAnalyzerWrapper;
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
import org.apache.lucene.search.BoostAttribute;
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
 *   T line term start end posInc posLen type flags payload(hex|-) keyword(0|1) termFreq [boost]
 *   E line finalStart finalEnd finalPosInc
 *   X line ExceptionSimpleName
 * </pre>
 *
 * Terms and types escape {@code \\ \t \n \r} and every other char below U+0020 or a lone surrogate
 * as {@code \\uXXXX}. {@code boost} (the float's bits in hex) is there only for a chain with a
 * {@code BoostAttribute}. Each chain is one {@link Analyzer}, reused across the lines as Lucene reuses
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
    // A word-delimiter graph: up to DEFAULT_MAX_GRAPH_EXPANSIONS paths, produced one at a time.
    c.put("ws_wdgf_concatenate_graph", () -> chain(WhitespaceTokenizer::new,
        t -> new ConcatenateGraphFilter(new WordDelimiterGraphFilter(t,
            WordDelimiterGraphFilter.GENERATE_WORD_PARTS | WordDelimiterGraphFilter.CATENATE_ALL, null))));
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
    c.put("std_pattern_typing", () -> chain(StandardTokenizer::new, t -> new PatternTypingFilter(t,
        new PatternTypingFilter.PatternTypingRule(Pattern.compile("^(\\d+)\\.(\\d+)$"), 3, "decimal_$1"),
        new PatternTypingFilter.PatternTypingRule(Pattern.compile("^([A-Z])"), 4, "capital_$1"))));
    c.put("pattern_replace_char_filter", () -> chain(r -> new PatternReplaceCharFilter(Pattern.compile("([a-z]+)-([a-z]+)"), "$2_$1", r),
        WhitespaceTokenizer::new, t -> t));
    // java.util.regex semantics the port re-emits (see util/java_regex.rs): `.` and `$` with
    // Java's line terminators, Java's empty-match rule (inside surrogate pairs too), ASCII-only
    // and Unicode case folding, POSIX classes, \\h and \\v.
    c.put("keyword_pattern_replace_dot", () -> chain(KeywordTokenizer::new, t -> new PatternReplaceFilter(t, Pattern.compile("."), "_", true)));
    c.put("keyword_pattern_replace_dollar", () -> chain(KeywordTokenizer::new, t -> new PatternReplaceFilter(t, Pattern.compile("(\\S)\\s*$"), "[$1]", true)));
    c.put("keyword_pattern_replace_x_star", () -> chain(KeywordTokenizer::new, t -> new PatternReplaceFilter(t, Pattern.compile("x*"), "-", true)));
    c.put("pattern_replace_char_filter_x_star", () -> chain(r -> new PatternReplaceCharFilter(Pattern.compile("x*"), "-", r),
        KeywordTokenizer::new, t -> t));
    c.put("ws_pattern_replace_ascii_case", () -> chain(WhitespaceTokenizer::new, t -> new PatternReplaceFilter(t, Pattern.compile("(?i)[a-e\u00e9]|stra\u00dfe|k"), "#", true)));
    c.put("ws_pattern_replace_unicode_case", () -> chain(WhitespaceTokenizer::new, t -> new PatternReplaceFilter(t, Pattern.compile("(?iu)[a-e\u00e9]|stra\u00dfe|k|\u03c3"), "#", true)));
    c.put("ws_pattern_replace_posix", () -> chain(WhitespaceTokenizer::new, t -> new PatternReplaceFilter(t, Pattern.compile("\\p{Punct}|\\p{Upper}|[[:alpha:]]"), "_", true)));
    c.put("pattern_tokenizer_h_v", () -> tok(() -> new PatternTokenizer(Pattern.compile("[\\h\\v,]+"), -1)));
    c.put("pattern_tokenizer_categories", () -> tok(() -> new PatternTokenizer(Pattern.compile("(\\p{L}+)|(\\p{Nd}+)"), 0)));
    c.put("std_pattern_typing_classes", () -> chain(StandardTokenizer::new, t -> new PatternTypingFilter(t,
        new PatternTypingFilter.PatternTypingRule(Pattern.compile("^\\p{Lu}\\p{Ll}+$"), 1, "title"),
        new PatternTypingFilter.PatternTypingRule(Pattern.compile("(?iu)^\\w*(.)$"), 2, "end_$1"))));
    // Rejected by the port (Java's \\b counts a non-spacing mark after a letter as a word
    // character; MULTILINE ^): the Rust harness expects IllegalArgument for these.
    c.put("keyword_pattern_replace_word_boundary", () -> chain(KeywordTokenizer::new, t -> new PatternReplaceFilter(t, Pattern.compile("\\bthe\\b"), "THE", true)));
    c.put("keyword_pattern_replace_multiline", () -> chain(KeywordTokenizer::new, t -> new PatternReplaceFilter(t, Pattern.compile("(?m)^"), ">", true)));

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

    // ---- analyzer wrappers
    c.put("ws_limit_token_count_analyzer", () -> new LimitTokenCountAnalyzer(new WhitespaceAnalyzer(), 3));
    c.put("simple_limit_token_count_consume_all", () -> new LimitTokenCountAnalyzer(new SimpleAnalyzer(), 2, true));
    c.put("per_field_wrapper_field", () -> new PerFieldAnalyzerWrapper(new WhitespaceAnalyzer(), Map.of("f", new SimpleAnalyzer())));
    c.put("per_field_wrapper_default", () -> new PerFieldAnalyzerWrapper(new WhitespaceAnalyzer(), Map.of("g", new SimpleAnalyzer())));
    c.put("shingle_analyzer_wrapper", () -> new ShingleAnalyzerWrapper(new StandardAnalyzer(), 3));
    c.put("shingle_analyzer_wrapper_options", () -> new ShingleAnalyzerWrapper(new WhitespaceAnalyzer(), 2, 3, "_", false, true, "*"));
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

  /** The one token a keyword chain makes of {@code text}. */
  static String single(Analyzer a, String text) throws Exception {
    try (TokenStream ts = a.tokenStream("f", text)) {
      CharTermAttribute term = ts.addAttribute(CharTermAttribute.class);
      ts.reset();
      ts.incrementToken();
      String s = term.toString();
      ts.end();
      return s;
    }
  }

  static String hex(BytesRef b) {
    if (b == null) return "-";
    StringBuilder s = new StringBuilder();
    for (int i = 0; i < b.length; i++) s.append(String.format("%02x", b.bytes[b.offset + i] & 0xff));
    return s.toString();
  }

  /** Stems crossed with suffixes: every branch of KStem's (and Porter's) ending rules. */
  static final String[] STEM_BASES = {
    "abandon", "able", "accept", "act", "adapt", "admire", "agree", "allow", "analyze", "apply",
    "argue", "art", "attend", "automate", "bake", "beauty", "begin", "believe", "break", "build",
    "busy", "care", "carry", "cat", "celebrate", "certain", "chance", "child", "civil", "class",
    "clean", "commune", "compete", "complete", "compute", "condition", "connect", "consider",
    "control", "cook", "count", "create", "critic", "cry", "dance", "decide", "define", "deliver",
    "depend", "describe", "design", "develop", "die", "differ", "direct", "dog", "dream", "dry",
    "economy", "educate", "elect", "emerge", "employ", "energy", "enjoy", "equal", "excite",
    "exist", "expect", "fail", "family", "fly", "form", "frequent", "friend", "general", "give",
    "globe", "govern", "happy", "harm", "help", "hero", "history", "hope", "hop", "idea",
    "identify", "imagine", "improve", "industry", "inform", "invent", "ironic", "judge", "kind",
    "know", "lady", "lazy", "legal", "lie", "light", "logic", "love", "magic", "manage", "market",
    "mass", "material", "medic", "member", "mobile", "modern", "move", "nation", "nature", "need",
    "normal", "notice", "obey", "observe", "occur", "office", "operate", "organ", "organize",
    "paint", "person", "physic", "plan", "play", "please", "poet", "polite", "pony", "possible",
    "power", "practice", "prefer", "produce", "profess", "public", "quick", "radio", "rare",
    "rational", "real", "receive", "refer", "relate", "rely", "rest", "rot", "run", "sad", "safe",
    "sense", "sensible", "serve", "simple", "sing", "social", "special", "stop", "study",
    "succeed", "sun", "swim", "system", "teach", "tender", "terror", "tradition", "true", "try",
    "use", "value", "vary", "visit", "walk", "weak", "wonder", "write", "zeal", "access",
    "acquire", "admit", "bed", "bubble", "commit", "consist", "curricul", "dig", "drum", "fancy",
    "fit", "gamble", "infant", "medium", "museum", "music", "nuance", "permit", "red", "remedy",
    "rhythm", "satisfy", "speed", "stun", "tan", "terrible", "unfold", "unhook", "unload",
    "unseat", "usual", "vision"
  };

  static final String[] STEM_SUFFIXES = {
    "", "s", "es", "ies", "'s", "ed", "ied", "d", "ing", "ning", "ting", "ity", "ality", "ivity",
    "ility", "ness", "iness", "ion", "ation", "ication", "ization", "ition", "er", "ier", "or",
    "izer", "ly", "ily", "ally", "ably", "al", "ial", "ical", "ive", "ative", "ize", "ise", "ment",
    "able", "ible", "ism", "ic", "ancy", "ency", "ance", "ence", "ous", "ful", "less", "ned",
    "ted", "ble", "nce", "ncy", "ically", "fully", "edness", "um", "ium", "ize", "izing", "izer", "ped", "ging", "ming", "ning", "ping",
    "red", "ring", "bed", "ged", "med"
  };


  /**
   * Code points whose {@code java.lang.Character} properties (type, case mappings, digit value,
   * whitespace, ...) differ between JDK 21 (Unicode 15.0) and JDK 25 (Unicode 16.0): pairs of
   * inclusive bounds. The code-point fixtures leave them out, so they are byte-identical under
   * either JDK; the port follows JDK 25 for them (see {@code java_character.rs}).
   */
  static final int[] JDK21_JDK25_DIFFER = {
    0x19B, 0x19B, 0x264, 0x264, 0x363, 0x36F, 0x897, 0x897, 0x1B4E, 0x1B4F, 0x1B7F, 0x1B7F,
    0x1C89, 0x1C8A, 0x1DD3, 0x1DE6, 0x2427, 0x2429, 0x2FFC, 0x2FFF, 0x31E4, 0x31E5, 0x31EF, 0x31EF,
    0xA7CB, 0xA7CD, 0xA7DA, 0xA7DC, 0x105C0, 0x105F3, 0x10D40, 0x10D65, 0x10D69, 0x10D85,
    0x10D8E, 0x10D8F, 0x10EC2, 0x10EC4, 0x10EFC, 0x10EFC, 0x11380, 0x11389, 0x1138B, 0x1138B,
    0x1138E, 0x1138E, 0x11390, 0x113B5, 0x113B7, 0x113C0, 0x113C2, 0x113C2, 0x113C5, 0x113C5,
    0x113C7, 0x113CA, 0x113CC, 0x113D5, 0x113D7, 0x113D8, 0x113E1, 0x113E2, 0x116D0, 0x116E3,
    0x1171E, 0x1171E, 0x11BC0, 0x11BE1, 0x11BF0, 0x11BF9, 0x11F5A, 0x11F5A, 0x13460, 0x143FA,
    0x16100, 0x16139, 0x16D40, 0x16D79, 0x18CFF, 0x18CFF, 0x1CC00, 0x1CCF9, 0x1CD00, 0x1CEB3,
    0x1E5D0, 0x1E5FA, 0x1E5FF, 0x1E5FF, 0x1F8B2, 0x1F8BB, 0x1F8C0, 0x1F8C1, 0x1FA89, 0x1FA89,
    0x1FA8F, 0x1FA8F, 0x1FABE, 0x1FABE, 0x1FAC6, 0x1FAC6, 0x1FADC, 0x1FADC, 0x1FADF, 0x1FADF,
    0x1FAE9, 0x1FAE9, 0x1FBCB, 0x1FBEF, 0x2EBF0, 0x2EE5D
  };

  static boolean jdkDependent(int cp) {
    for (int i = 0; i < JDK21_JDK25_DIFFER.length; i += 2) {
      if (cp >= JDK21_JDK25_DIFFER[i] && cp <= JDK21_JDK25_DIFFER[i + 1]) return true;
    }
    return false;
  }

  /** java.util.regex constructs, each run over every input of {@link #REGEX_INPUTS}. */
  static final String[] REGEX_PATTERNS = {
    // classes
    "\\w+", "\\d+", "\\s+", "\\W", "\\D+", "\\S+", "[\\W]", "[^\\W]", "[\\d\\s]+", "[]a]", "[^]a]",
    "\\p{Alpha}+", "\\p{Upper}", "\\p{Lower}+", "\\p{Punct}", "\\p{Space}", "\\p{Digit}", "\\p{Alnum}+",
    "\\p{Graph}+", "\\p{Print}+", "\\p{Blank}", "\\p{Cntrl}", "\\p{XDigit}+", "\\p{ASCII}+",
    "\\p{L}+", "\\pL", "\\p{IsL}+", "\\p{Lu}", "\\p{IsLu}", "\\p{gc=Nd}", "\\p{general_category=Lu}",
    "\\p{LC}", "\\p{IsLC}", "\\p{LD}+", "\\p{L1}+", "\\p{all}", "\\P{L}+", "\\p{N}", "\\p{P}", "\\p{S}",
    "\\p{Z}", "\\p{M}", "\\p{C}", "\\p{Cs}", "\\p{Mn}", "\\p{So}",
    "[\\p{L}&&\\p{Lu}]", "[\\p{L}&&[^\\p{Lu}]]", "[a-z&&[^aeiou]]", "[^a-c&&b-d]", "[a[bc]d]", "[^a[b]]",
    "[[:alpha:]]", "[[:^alpha:]]", "[x[:digit:]]", "[a-c-e]", "[a-]", "[-a]",
    "\\h", "\\H+", "\\v", "\\V+", "[\\v]", "[^\\h]", "\\e", "\\x{1F600}", "\\uD83D\\uDE00", "\\u00e9", "\\x41",
    "\\Qa.b\\E", "[\\Q-]\\E]", "\\Q*", "\\<a\\>", "\\%\\@\\'\\\"\\#", "\\_",
    ".", "(?s).", ".+", "(?s).+", "[^a]*",
    // case
    "(?i)a", "(?i)abc", "(?i)é", "(?iu)é", "(?i)[a-c]+", "(?iu)[a-c]+", "(?i)[é]", "(?iu)[é]",
    "(?i)k", "(?iu)k", "(?iu)K", "(?i)[a-z]+", "(?iu)[a-z]+", "(?iu)[^k]", "(?i)[^a]", "(?iu)ß",
    "(?iu)ßx", "(?iu)ẞ", "(?iu)[ß]", "(?iu)i", "(?iu)[i]", "(?iu)İ", "(?iu)ǅ",
    "(?i)\\p{Lu}", "(?i)\\P{Lu}", "(?i)\\p{Upper}", "(?i)\\p{Lower}", "(?iu)\\w", "(?i:a)b", "a(?i)b|c",
    "(?i)(?-i:a)", "(?i)(?u)k", "(?-i)a",
    // anchors and $
    "^", "$", "^a", "a$", "^$", "\\A", "\\z", "\\n$", "a$\\n", "\\s+$", "\\s*$", "$\\s", "(\\w+)$",
    "[^a]$", "x*$", ".$", "(?s).$", "\\A|.", "\\A|(.)",
    // empty matches and find()
    "x*", "a*", "(a*)", "a*?", "(a)?", "(a|)?", "(?:a|)*", "((a)|b)*", "(a)|b", "(x)|(y)", "a{0}",
    "(?:)", "|", "a|", "|a", "\\b?",
    // repetition and groups
    "a{2}", "a{1,}", "a{1,2}", "a{0,1}?", "a+?", "(?<word>\\w+)", "(?<ab1>x)", "(a(b)?)+", "(?:ab)+",
    // rejected by the port (Java compiles them)
    "\\bfox", "\\b", "\\B", "(?m)^a", "(?m)a$", "(?x) a b", "(?U)\\w", "(a|)*", "(a*)+", "\\p{IsLatin}",
    "\\p{InGreek}", "\\p{IsAlphabetic}", "\\p{javaLowerCase}", "(a)\\1", "(?=a)", "(?<=a)b", "a++",
    "(?>a)", "\\Z", "\\G", "\\R", "\\X", "\\cA", "\\0101", "\\N{LATIN SMALL LETTER A}", "(?d).",
    "[a~~b]", "[&&a]", "[a&&]", "[a&&&b]", "\\p{gc=L}",
    // rejected by Java
    "(", "[a", "a{2,1}", "x{,3}", "\\y", "(?P<a>x)", "(?<a_b>x)", "\\u{e9}", "\\U000000e9", "[a--b]",
    "\\p{Latin}", "\\p{Uppercase_Letter}", "\\p{ Lu }", "a{ 2 }", "*",
  };

  static final String[] REGEX_INPUTS = {
    "", "a", "abxd", "aA", "Hello World", "foo-bar baz_qux", "x\n", "a\r\n", "a\r", "line1\nline2\n",
    "a\u0085", "b ", "a \n", "café naïve", "é", "😀", "baa😀",
    "a😀b😀\n", "ǅǄǆ ß ẞ İ ı K K k",
    "١٢ 123", "\t \u000B 　 ", "[:alpha:]-^", "<a>&b", "MS-DOS 3.14", "aaa",
    "xxxyyy", "a.b axb *", "\u001b%@'\"#_", "AbAB aBC",
  };

  /** One input's find() spans and groups, replaceAll("<$0>") and matches(); or the exception. */
  static String regexRun(String pattern, String input) {
    try {
      Pattern p = Pattern.compile(pattern);
      Matcher m = p.matcher(input);
      StringBuilder b = new StringBuilder();
      while (m.find()) {
        b.append('(').append(m.start()).append(',').append(m.end());
        for (int g = 1; g <= m.groupCount(); g++) b.append(' ').append(m.start(g)).append(':').append(m.end(g));
        b.append(')');
      }
      b.append(" rep=").append(esc(p.matcher(input).replaceAll("<$0>")));
      b.append(" m=").append(p.matcher(input).matches());
      return b.toString();
    } catch (Exception e) {
      return "EXC " + e.getClass().getSimpleName();
    }
  }

  static void writeRegexFixtures(Path out) throws Exception {
    StringBuilder r = new StringBuilder();
    for (String pattern : REGEX_PATTERNS) {
      for (String input : REGEX_INPUTS) {
        r.append(esc(pattern)).append('\t').append(esc(input)).append('\t').append(regexRun(pattern, input)).append('\n');
      }
    }
    Files.writeString(out.resolve("regex.words"), r.toString(), StandardCharsets.UTF_8);

    // Case-insensitivity: every code point with a simple case mapping (and a few without),
    // matched by each literal and class form of a set of probes. Line 1 is the input, line 2
    // the same input with U+E000 after each character (for the two-literal `Slice` form).
    StringBuilder in = new StringBuilder();
    StringBuilder in2 = new StringBuilder();
    List<Integer> probes = new ArrayList<>();
    for (int cp = 0; cp < 0x20000; cp++) {
      if (jdkDependent(cp) || Character.getType(cp) == Character.SURROGATE) continue;
      boolean cased = Character.toUpperCase(cp) != cp || Character.toLowerCase(cp) != cp;
      boolean extra = cp == 0xDF || cp == 0x138 || cp == 0x149 || cp == 0x390 || cp == 0x3B0
          || cp == 0x1FD3 || cp == 0x1FE3 || cp == 0x1F0 || cp == '1' || cp == '_';
      if (!cased && !extra) continue;
      in.appendCodePoint(cp);
      in2.appendCodePoint(cp).append('');
      if (cp < 0x250 || (cp >= 0x370 && cp < 0x530) || cp >= 0x1E00 && cp < 0x2200 || cp >= 0x10400 || extra) {
        probes.add(cp);
      }
    }
    StringBuilder ci = new StringBuilder();
    ci.append(esc(in.toString())).append('\n').append(esc(in2.toString())).append('\n');
    List<String[]> forms = new ArrayList<>();
    for (int cp : probes) {
      String x = "\\x{" + Integer.toHexString(cp) + "}";
      forms.add(new String[] {"(?i)" + x, "1"});
      forms.add(new String[] {"(?iu)" + x, "1"});
      forms.add(new String[] {"(?i)[" + x + "]", "1"});
      forms.add(new String[] {"(?iu)[" + x + "]", "1"});
      forms.add(new String[] {"(?iu)[" + x + "-" + x + "]", "1"});
      forms.add(new String[] {"(?iu)" + x + "\\x{e000}", "2"});
    }
    for (String p : new String[] {"(?i)[a-z]", "(?iu)[a-z]", "(?iu)[\\x{e0}-\\x{ff}]", "(?iu)[\\x{400}-\\x{42f}]",
        "(?i)[A-Z]", "(?iu)[^a-z]", "(?i)\\p{Lu}", "(?i)\\p{Ll}", "(?i)\\p{Lt}", "(?iu)\\p{Lu}", "(?i)\\p{Upper}",
        "(?i)\\p{Lower}", "(?iu)\\w", "\\p{Lu}", "\\p{Ll}", "\\p{Lt}", "(?iu)[\\x{1f00}-\\x{1fff}]"}) {
      forms.add(new String[] {p, "1"});
    }
    String input1 = in.toString(), input2 = in2.toString();
    for (String[] f : forms) {
      Matcher m = Pattern.compile(f[0]).matcher(f[1].equals("1") ? input1 : input2);
      StringBuilder pos = new StringBuilder();
      while (m.find()) pos.append(pos.length() == 0 ? "" : ",").append(m.start());
      ci.append(esc(f[0])).append('\t').append(f[1]).append('\t').append(pos).append('\n');
    }
    Files.writeString(out.resolve("regex_ci.words"), ci.toString(), StandardCharsets.UTF_8);
  }


  /**
   * {@code ConcatenateGraphFilter} with separators that are not ASCII (Java casts the separator to
   * a byte when escaping and when writing the label) over a few inputs: "sep\tinput\tbytes|...".
   * The bytes are {@code TermToBytesRefAttribute}'s, what an index receives.
   */
  static void writeConcatenateFixtures(Path out) throws Exception {
    StringBuilder b = new StringBuilder();
    Character[] seps = {null, '\u001f', 'x', '\u0080', '©', 'é', 'ÿ', 'Ā', 'Ł'};
    String[] inputs = {"a b", "aéb c", "xŁ y", "a-b c-d", "© x"};
    for (Character sep : seps) {
      for (String input : inputs) {
        WhitespaceTokenizer t = new WhitespaceTokenizer();
        t.setReader(new java.io.StringReader(input));
        TokenStream ts = new ConcatenateGraphFilter(t, sep, true, 10000);
        TermToBytesRefAttribute bytes = ts.getAttribute(TermToBytesRefAttribute.class);
        b.append(sep == null ? "-" : Integer.toHexString(sep)).append('\t').append(esc(input)).append('\t');
        ts.reset();
        while (ts.incrementToken()) b.append(hex(bytes.getBytesRef())).append('|');
        ts.end();
        ts.close();
        b.append('\n');
      }
    }
    Files.writeString(out.resolve("concatenate.words"), b.toString(), StandardCharsets.UTF_8);
  }


  /** The chains {@code codepoints.words} runs every code point through. */
  static Map<String, Analyzer> codePointChains() {
    Map<String, Analyzer> m = new LinkedHashMap<>();
    m.put("fold", chain(KeywordTokenizer::new, t -> new ASCIIFoldingFilter(t, true)));
    m.put("cjkw", chain(KeywordTokenizer::new, CJKWidthFilter::new));
    m.put("cjkwcf", chain(CJKWidthCharFilter::new, KeywordTokenizer::new, t -> t));
    m.put("lower", chain(KeywordTokenizer::new, LowerCaseFilter::new));
    m.put("upper", chain(KeywordTokenizer::new, UpperCaseFilter::new));
    m.put("digit", chain(KeywordTokenizer::new, DecimalDigitFilter::new));
    m.put("scf", chain(KeywordTokenizer::new, ScandinavianFoldingFilter::new));
    m.put("scn", chain(KeywordTokenizer::new, ScandinavianNormalizationFilter::new));
    m.put("letter", tok(LetterTokenizer::new));
    m.put("ws", tok(WhitespaceTokenizer::new));
    m.put("wdgf", chain(KeywordTokenizer::new, t -> new WordDelimiterGraphFilter(t,
        WordDelimiterGraphFilter.GENERATE_WORD_PARTS | WordDelimiterGraphFilter.GENERATE_NUMBER_PARTS
            | WordDelimiterGraphFilter.SPLIT_ON_CASE_CHANGE | WordDelimiterGraphFilter.SPLIT_ON_NUMERICS
            | WordDelimiterGraphFilter.STEM_ENGLISH_POSSESSIVE, null)));
    return m;
  }

  /**
   * One chain over {@code "a" + cp + "B"}: each token's UTF-16 units in hex (the code point's own
   * units written {@code @}), {@code :start-end/posInc}, then {@code |endOffset}.
   */
  static String codePointRun(Analyzer a, int cp) {
    String text = "a" + new String(Character.toChars(cp)) + "B";
    char[] own = Character.toChars(cp);
    StringBuilder sb = new StringBuilder();
    try (TokenStream ts = a.tokenStream("f", text)) {
      CharTermAttribute t = ts.addAttribute(CharTermAttribute.class);
      OffsetAttribute o = ts.addAttribute(OffsetAttribute.class);
      PositionIncrementAttribute p = ts.addAttribute(PositionIncrementAttribute.class);
      ts.reset();
      while (ts.incrementToken()) {
        for (int i = 0; i < t.length(); i++) {
          boolean mine = i + own.length <= t.length();
          for (int k = 0; mine && k < own.length; k++) mine = t.charAt(i + k) == own[k];
          if (mine) {
            sb.append("@.");
            i += own.length - 1;
          } else {
            sb.append(Integer.toHexString(t.charAt(i))).append('.');
          }
        }
        sb.append(':').append(o.startOffset()).append('-').append(o.endOffset()).append('/')
            .append(p.getPositionIncrement()).append(' ');
      }
      ts.end();
      sb.append('|').append(o.endOffset());
    } catch (Exception e) {
      sb.append('X').append(e.getClass().getSimpleName());
    }
    return sb.toString();
  }

  /**
   * {@code codepoints.words}: every code point (but surrogates and {@link #JDK21_JDK25_DIFFER})
   * through each of {@link #codePointChains}, as runs of consecutive code points with the same
   * result: "chain\tfirst\tlast\tresult" (hex bounds). The skipped ranges lead as "X\tfirst\tlast".
   */
  static void writeCodePointFixtures(Path out) throws Exception {
    StringBuilder b = new StringBuilder();
    for (int i = 0; i < JDK21_JDK25_DIFFER.length; i += 2) {
      b.append("X\t").append(Integer.toHexString(JDK21_JDK25_DIFFER[i])).append('\t')
          .append(Integer.toHexString(JDK21_JDK25_DIFFER[i + 1])).append('\n');
    }
    for (Map.Entry<String, Analyzer> e : codePointChains().entrySet()) {
      int first = -1, last = -1;
      String run = null;
      for (int cp = 0; cp <= Character.MAX_CODE_POINT + 1; cp++) {
        boolean skip = cp > Character.MAX_CODE_POINT || (cp >= 0xD800 && cp <= 0xDFFF) || jdkDependent(cp);
        String r = skip ? null : codePointRun(e.getValue(), cp);
        if (run != null && (r == null || !r.equals(run) || cp != last + 1)) {
          b.append(e.getKey()).append('\t').append(Integer.toHexString(first)).append('\t')
              .append(Integer.toHexString(last)).append('\t').append(run).append('\n');
          run = null;
        }
        if (r != null) {
          if (run == null) {
            run = r;
            first = cp;
          }
          last = cp;
        }
      }
      e.getValue().close();
    }
    Files.writeString(out.resolve("codepoints.words"), b.toString(), StandardCharsets.UTF_8);
  }

  public static void main(String[] args) throws Exception {
    Path out = Path.of(args[0]).resolve("analysis_common");
    Files.createDirectories(out);
    String corpusDir = System.getenv().getOrDefault("FIXTURES_CORPUS", "fixtures/corpus");
    String corpus = Files.readString(Path.of(corpusDir, "analysis-common.txt"), StandardCharsets.UTF_8);
    // Split on '\n' only: the corpus holds U+2028 and U+0085 inside lines.
    List<String> lines = new ArrayList<>(Arrays.asList(corpus.split("\n", -1)));
    if (!lines.isEmpty() && lines.get(lines.size() - 1).isEmpty()) lines.remove(lines.size() - 1);

    // KStem and Porter over the stem x suffix words: "word\tkstem\tporter".
    StringBuilder stems = new StringBuilder();
    try (Analyzer k = chain(KeywordTokenizer::new, t -> new KStemFilter(t));
        Analyzer p = chain(KeywordTokenizer::new, t -> new PorterStemFilter(t))) {
      // The listed bases, then every 80th KStem head word (read from the
      // package-private KStemData classes by reflection).
      List<String> bases = new ArrayList<>(Arrays.asList(STEM_BASES));
      int n = 0;
      for (int i = 1; i <= 8; i++) {
        java.lang.reflect.Field f =
            Class.forName("org.apache.lucene.analysis.en.KStemData" + i).getDeclaredField("data");
        f.setAccessible(true);
        for (String w : (String[]) f.get(null)) {
          if (n++ % 80 == 0) bases.add(w);
        }
      }
      for (String base : bases) {
        for (String suffix : STEM_SUFFIXES) {
          String w = base + suffix;
          stems.append(w).append('\t').append(single(k, w)).append('\t').append(single(p, w)).append('\n');
        }
      }
    }
    Files.writeString(out.resolve("stems.words"), stems.toString(), StandardCharsets.UTF_8);

    // UAX29URLEmailTokenizer over URL/email-shaped fragments joined at random
    // (fixed seed): "text\tterm type start end posInc|...|endOffset posInc".
    String[] frags = {
      "http://", "https://", "HTTP://", "www.", "x", "example", ".com", ".co.uk", ".", "/", ":", "@",
      "-", "_", "1", "42", "é", "😀", "中", "[", "]", "%20", "?q=1", "#f", "mailto:", "ftp://",
      "file:///", "localhost", ":8080", "a.b", "user", "+tag", "'", "\"", "<", ">", "(", ")", ",",
      " ", "  ", "xn--p1ai", "192.168.0.1", "::1", "&", "=", ";", "!", "~", "*", "$"
    };
    java.util.Random rnd = new java.util.Random(42);
    StringBuilder urls = new StringBuilder();
    org.apache.lucene.analysis.email.UAX29URLEmailTokenizer ut =
        new org.apache.lucene.analysis.email.UAX29URLEmailTokenizer();
    CharTermAttribute uTerm = ut.addAttribute(CharTermAttribute.class);
    OffsetAttribute uOff = ut.addAttribute(OffsetAttribute.class);
    TypeAttribute uType = ut.addAttribute(TypeAttribute.class);
    PositionIncrementAttribute uInc = ut.addAttribute(PositionIncrementAttribute.class);
    for (int i = 0; i < 3000; i++) {
      StringBuilder text = new StringBuilder();
      int n = 1 + rnd.nextInt(8);
      for (int j = 0; j < n; j++) text.append(frags[rnd.nextInt(frags.length)]);
      ut.setReader(new java.io.StringReader(text.toString()));
      ut.reset();
      urls.append(esc(text.toString())).append('\t');
      while (ut.incrementToken()) {
        urls.append(esc(uTerm.toString())).append(' ').append(uType.type()).append(' ')
            .append(uOff.startOffset()).append(' ').append(uOff.endOffset()).append(' ')
            .append(uInc.getPositionIncrement()).append('|');
      }
      ut.end();
      urls.append(uOff.endOffset()).append(' ').append(uInc.getPositionIncrement()).append('\n');
      ut.close();
    }
    Files.writeString(out.resolve("urls.words"), urls.toString(), StandardCharsets.UTF_8);

    writeRegexFixtures(out);
    writeConcatenateFixtures(out);
    writeCodePointFixtures(out);

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
            BoostAttribute boost = ts.hasAttribute(BoostAttribute.class) ? ts.getAttribute(BoostAttribute.class) : null;
            ts.reset();
            while (ts.incrementToken()) {
              String t = term != null ? esc(term.toString()) : "#" + hex(bytes.getBytesRef());
              m.append("T\t").append(ln).append('\t').append(t)
                  .append('\t').append(off.startOffset()).append('\t').append(off.endOffset())
                  .append('\t').append(inc.getPositionIncrement()).append('\t').append(len.getPositionLength())
                  .append('\t').append(esc(type.type())).append('\t').append(flags.getFlags())
                  .append('\t').append(hex(payload.getPayload())).append('\t').append(kw.isKeyword() ? 1 : 0)
                  .append('\t').append(tf.getTermFrequency());
              if (boost != null) {
                m.append('\t').append(Integer.toHexString(Float.floatToIntBits(boost.getBoost())));
              }
              m.append('\n');
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
