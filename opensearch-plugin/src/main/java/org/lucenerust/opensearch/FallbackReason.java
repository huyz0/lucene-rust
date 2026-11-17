/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch;

/**
 * Every reason the plugin hands a search (or a part of one) back to Lucene or Java. The
 * {@linkplain #key() key} is what {@code GET _plugins/accelerated/stats} reports in its
 * {@code fallbacks} map, so a constant is the one place a reason is named: {@code
 * scripts/check-fallbacks.py} checks the constants against {@code fallback-reasons.txt} and fails
 * on a reason that is not a constant, a constant nothing uses, or a new constant nobody approved.
 *
 * <p>A constant whose name ends in {@code ANY} is a prefix: its reasons are the key followed by a
 * detail ({@link #with}), for example {@code aggregation_} plus the aggregation's type.
 */
public enum FallbackReason {
    AGGREGATION_ANY("aggregation_"),
    AGGREGATION_SCRIPT("aggregation_script"),
    AGGREGATION_SCRIPT_OPTIONS("aggregation_script_options"),
    AGGREGATION_SCRIPT_SCORE("aggregation_script_score"),
    AGGREGATION_SHAPE("aggregation_shape"),
    AGGREGATIONS("aggregations"),
    ANALYSIS_ANY("analysis_"),
    ANALYSIS_HUNSPELL("analysis_hunspell"),
    ANALYSIS_JAPANESE_DICTIONARY("analysis_japanese_dictionary"),
    ANALYSIS_JAPANESE_USER_DICTIONARY("analysis_japanese_user_dictionary"),
    ANALYSIS_KOREAN_DICTIONARY("analysis_korean_dictionary"),
    ANALYSIS_KOREAN_USER_DICTIONARY("analysis_korean_user_dictionary"),
    ANALYSIS_PATTERN_FLAGS("analysis_pattern_flags"),
    ANALYSIS_REFLECT("analysis_reflect"),
    ANALYSIS_REVERSE_MARKER("analysis_reverse_marker"),
    ANALYSIS_SHINGLE_TYPE("analysis_shingle_type"),
    ANALYSIS_SNOWBALL("analysis_snowball"),
    ANALYSIS_STEMMER_OVERRIDE("analysis_stemmer_override"),
    ANALYSIS_SYNONYMS("analysis_synonyms"),
    ANALYSIS_WORD_DELIMITER_FLAGS("analysis_word_delimiter_flags"),
    ANALYSIS_WORD_DELIMITER_TYPES("analysis_word_delimiter_types"),
    ANALYSIS_WORDS("analysis_words"),
    APPROXIMATE("approximate"),
    BOOLEAN_MSM_NEGATIVE("boolean_msm_negative"),
    BOOST_INVALID("boost_invalid"),
    CANCELLED("cancelled"),
    CLAUSE_ANY("clause_"),
    COLLAPSE_FIELD("collapse_field"),
    COLLAPSE_REFUSED("collapse_refused"),
    COLLAPSE_SEARCH_AFTER("collapse_search_after"),
    COLLECTOR_SPEC("collector_spec"),
    COLLECTORS("collectors"),
    CONVERTER("converter"),
    COVERING_OPTIONS("covering_options"),
    COVERING_SCRIPT("covering_script"),
    DATA("data"),
    DFS("dfs"),
    DIRECTORY_MIXED("directory_mixed"),
    DIRECTORY_NOT_NATIVE("directory_not_native"),
    DISABLED("disabled"),
    ERROR("error"),
    EXTENDED_REFLECT("extended_reflect"),
    FETCH_DOC_VALUES("fetch_doc_values"),
    FIELD_SIMILARITY("field_similarity"),
    FUNCTION_SCORE_ANY("function_score_"),
    FUNCTION_SCORE_DECAY("function_score_decay"),
    FUNCTION_SCORE_FIELD_DATA("function_score_field_data"),
    FUNCTION_SCORE_SCRIPT("function_score_script"),
    FUZZY_AUTOMATON("fuzzy_automaton"),
    GEO_GEOMETRY("geo_geometry"),
    GEO_REFLECT("geo_reflect"),
    HIGHLIGHT_ANY("highlight_"),
    HIGHLIGHT_ANALYZER("highlight_analyzer"),
    HIGHLIGHT_BOUNDARY_SCANNER("highlight_boundary_scanner"),
    HIGHLIGHT_CLAUSE_ORDER("highlight_clause_order"),
    HIGHLIGHT_DERIVED_FIELD("highlight_derived_field"),
    HIGHLIGHT_FRAGMENT_OFFSET("highlight_fragment_offset"),
    HIGHLIGHT_FRAGMENT_SIZE("highlight_fragment_size"),
    HIGHLIGHT_FRAGMENTER("highlight_fragmenter"),
    HIGHLIGHT_FVH_HIT("highlight_fvh_hit"),
    HIGHLIGHT_FVH_REFLECTION("highlight_fvh_reflection"),
    HIGHLIGHT_FVH_TERM_VECTORS("highlight_fvh_term_vectors"),
    HIGHLIGHT_LIVE_VIEW("highlight_live_view"),
    HIGHLIGHT_LOCALE("highlight_locale"),
    HIGHLIGHT_LONE_SURROGATE("highlight_lone_surrogate"),
    HIGHLIGHT_MAX_ANALYZER_OFFSET("highlight_max_analyzer_offset"),
    HIGHLIGHT_NATIVE_ERROR("highlight_native_error"),
    HIGHLIGHT_NESTED_ANY("highlight_nested_"),
    HIGHLIGHT_PLAIN_HIT("highlight_plain_hit"),
    HIGHLIGHT_READER("highlight_reader"),
    INTERVAL_ANY("interval_"),
    INTERVAL_OPTIONS("interval_options"),
    INTERVAL_PAINLESS("interval_painless"),
    INTERVAL_REPEATING_COUNT("interval_repeating_count"),
    INTERVAL_SCORING("interval_scoring"),
    INTERVAL_SCRIPT("interval_script"),
    INTRA_SEGMENT("intra_segment"),
    JOIN_PARENTS("join_parents"),
    JOIN_UNREWRITTEN("join_unrewritten"),
    MIN_SCORE("min_score"),
    MODE_ANY("mode_"),
    NATIVE_ERROR("native_error"),
    NATIVE_OPEN_FAILED("native_open_failed"),
    NESTED("nested"),
    NESTED_PARENTS("nested_parents"),
    NESTED_QUERY("nested_query"),
    NOSKIP("noskip"),
    PATTERN_PHRASE_ANY("pattern_phrase_"),
    POINTS_WIDTH("points_width"),
    POSTINGS_FORMAT("postings_format"),
    QUERY_TOO_DEEP("query_too_deep"),
    QUERY_TOO_LARGE("query_too_large"),
    RANDOM_SCORE_ID("random_score_id"),
    READER_CLOSED("reader_closed"),
    READER_NOT_DIRECTORY("reader_not_directory"),
    READER_NOT_STANDARD("reader_not_standard"),
    READER_UNCACHEABLE("reader_uncacheable"),
    REALTIME_VIEW("realtime_view"),
    RESCORE("rescore"),
    RESCORE_ANY("rescore_"),
    RESCORE_APPROXIMATE("rescore_approximate"),
    RESCORE_QUERY("rescore_query"),
    SCALED_FLOAT("scaled_float"),
    SCRIPT_DATE_VALUES("script_date_values"),
    SCRIPT_EXPRESSION_PER_DOCUMENT("script_expression_per_document"),
    SCRIPT_QUERY_SCRIPT("script_query_script"),
    SCRIPT_SCORE_HAMMINGBIT("script_score_hammingbit"),
    SCRIPT_SCORE_SCRIPT("script_score_script"),
    SCRIPT_SORT_SCORE("script_sort_score"),
    SCROLL_AFTER("scroll_after"),
    SEARCH_AFTER("search_after"),
    SEARCH_AFTER_FIELDS("search_after_fields"),
    SEARCH_AFTER_NULL("search_after_null"),
    SEARCH_AFTER_TYPE("search_after_type"),
    SEGMENT_INFOS_WRITE("segment_infos_write"),
    SLOWER_SHAPE("slower_shape"),
    SORT_ANY("sort_"),
    SORT_GEO_FIELD_DATA("sort_geo_field_data"),
    SORT_GEO_NESTED("sort_geo_nested"),
    SORT_GEO_ORIGINS("sort_geo_origins"),
    SORT_GEO_REFLECT("sort_geo_reflect"),
    SORT_KEYS("sort_keys"),
    SORT_SCRIPT("sort_script"),
    SORT_SCRIPT_NESTED("sort_script_nested"),
    SORT_SCRIPT_OPTIONS("sort_script_options"),
    SORT_UNOPTIMIZED("sort_unoptimized"),
    SPAN_ANY("span_"),
    SUBSTRING_REGEXP_FLAGS("substring_regexp_flags"),
    TERM_STATES("term_states"),
    TERMINATE_AFTER("terminate_after"),
    TIME_SERIES_ORDER("time_series_order"),
    TIMEOUT("timeout"),
    TYPE_ANY("type_");

    private final String key;

    FallbackReason(String key) {
        this.key = key;
    }

    /** The reason as the stats report it. */
    public String key() {
        return key;
    }

    /** Whether this constant is a prefix that a detail is appended to. */
    public boolean isPrefix() {
        return key.endsWith("_");
    }

    /** The reason for {@code detail} under this prefix. */
    public String with(String detail) {
        if (isPrefix() == false) {
            throw new IllegalStateException(key + " is not a prefix");
        }
        return key + detail;
    }
}
