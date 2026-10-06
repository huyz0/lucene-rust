//! `org.apache.lucene.analysis.miscellaneous`.
//!
//! `ASCIIFoldingFilter`'s streaming form is [`AsciiFoldingTokenFilter`] (the
//! crate-root [`crate::AsciiFoldingFilter`] is the older `Vec<Token>` API
//! over the same table).

mod concatenate_graph;
mod conditional;
mod filtering;
mod keyword;
mod limit;
mod stateful;
mod term_filters;
mod word_delimiter;

pub use concatenate_graph::{ConcatenateGraphFilter, DEFAULT_MAX_GRAPH_EXPANSIONS, SEP_LABEL};
pub use conditional::{
    protected_term_filter, ConditionalRoot, ConditionalTokenFilter, NotProtected, OneTimeWrapper,
    ProtectedTermFilter, ShouldFilter,
};
pub use filtering::{CodepointCountFilter, DropIfFlaggedFilter, KeepWordFilter, LengthFilter};
pub use keyword::{
    KeywordMarkerFilter, KeywordRepeatFilter, KeywordTest, PatternKeywordMarkerFilter,
    RemoveDuplicatesTokenFilter, SetKeywordMarkerFilter, StemmerOverrideBuilder,
    StemmerOverrideFilter, StemmerOverrideMap,
};
pub use limit::{LimitTokenCountFilter, LimitTokenOffsetFilter, LimitTokenPositionFilter};
pub use stateful::{
    AsciiFoldingTokenFilter, FingerprintFilter, FixBrokenOffsetsFilter, HyphenatedWordsFilter,
    TypeAsSynonymFilter, FINGERPRINT_DEFAULT_MAX_OUTPUT_TOKEN_SIZE, FINGERPRINT_DEFAULT_SEPARATOR,
};
pub(crate) use term_filters::parse_int;
pub use term_filters::{
    CapitalizationFilter, DelimitedTermFrequencyTokenFilter, Folding, ScandinavianFoldingFilter,
    ScandinavianNormalizationFilter, ScandinavianNormalizer, TrimFilter, TruncateTokenFilter,
    CAPITALIZATION_DEFAULT_MAX, DEFAULT_TERM_FREQUENCY_DELIMITER,
};
pub use word_delimiter::{
    get_type as word_delimiter_type, WordDelimiterGraphFilter, ALPHA, ALPHANUM, CATENATE_ALL,
    CATENATE_NUMBERS, CATENATE_WORDS, DEFAULT_WORD_DELIM_TABLE, GENERATE_NUMBER_PARTS,
    GENERATE_WORD_PARTS, IGNORE_KEYWORDS, PRESERVE_ORIGINAL, SPLIT_ON_CASE_CHANGE,
    SPLIT_ON_NUMERICS, STEM_ENGLISH_POSSESSIVE,
};
