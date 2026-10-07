//! The token filter factories with arguments but no resources: boost,
//! CJK bigrams, MinHash, the miscellaneous filters, n-grams, patterns,
//! numeric payloads, shingles.

use std::collections::{BTreeSet, HashSet};
use std::sync::Arc;

use lucene_util::version::Version;

use super::args::{self, JavaArgs};
use super::{
    analysis_factory, factory_struct, FactoryBase, FactoryClass, FactoryError, JavaException,
    TokenFilterFactory,
};
use crate::miscellaneous as misc;
use crate::token_stream::TokenStream;
use crate::util::JavaPattern;
use crate::{AnalysisError, CharArraySet};

factory_struct! {
    /// `org.apache.lucene.analysis.boost.DelimitedBoostTokenFilterFactory` (`delimitedBoost`).
    DelimitedBoostTokenFilterFactory { delimiter: char }
}
analysis_factory!(DelimitedBoostTokenFilterFactory);

impl FactoryClass for DelimitedBoostTokenFilterFactory {
    const NAME: &'static str = "delimitedBoost";
    const CLASS_NAME: &'static str =
        "org.apache.lucene.analysis.boost.DelimitedBoostTokenFilterFactory";
    fn from_args(args: &mut JavaArgs) -> Result<Self, FactoryError> {
        let base = FactoryBase::new(Self::CLASS_NAME, args)?;
        let delimiter = args::get_char(args, "delimiter", u16::from(b'|'))?;
        args::reject_unknown(args)?;
        Ok(DelimitedBoostTokenFilterFactory {
            base,
            delimiter: args::unit_char(delimiter),
        })
    }
}

impl TokenFilterFactory for DelimitedBoostTokenFilterFactory {
    fn create(&self, input: Box<dyn TokenStream>) -> Result<Box<dyn TokenStream>, AnalysisError> {
        Ok(Box::new(crate::boost::DelimitedBoostTokenFilter::new(
            input,
            self.delimiter,
        )))
    }
}

factory_struct! {
    /// `org.apache.lucene.analysis.cjk.CJKBigramFilterFactory` (`cjkBigram`).
    CJKBigramFilterFactory { flags: i32, output_unigrams: bool }
}
analysis_factory!(CJKBigramFilterFactory);

impl FactoryClass for CJKBigramFilterFactory {
    const NAME: &'static str = "cjkBigram";
    const CLASS_NAME: &'static str = "org.apache.lucene.analysis.cjk.CJKBigramFilterFactory";
    fn from_args(args: &mut JavaArgs) -> Result<Self, FactoryError> {
        use crate::cjk::{HAN, HANGUL_FLAG, HIRAGANA_FLAG, KATAKANA_FLAG};
        let base = FactoryBase::new(Self::CLASS_NAME, args)?;
        let mut flags = 0;
        for (name, flag) in [
            ("han", HAN),
            ("hiragana", HIRAGANA_FLAG),
            ("katakana", KATAKANA_FLAG),
            ("hangul", HANGUL_FLAG),
        ] {
            if args::get_boolean(args, name, true) {
                flags |= flag;
            }
        }
        let output_unigrams = args::get_boolean(args, "outputUnigrams", false);
        args::reject_unknown(args)?;
        Ok(CJKBigramFilterFactory {
            base,
            flags,
            output_unigrams,
        })
    }
}

impl TokenFilterFactory for CJKBigramFilterFactory {
    fn create(&self, input: Box<dyn TokenStream>) -> Result<Box<dyn TokenStream>, AnalysisError> {
        Ok(Box::new(crate::cjk::CJKBigramFilter::new(
            input,
            self.flags,
            self.output_unigrams,
        )))
    }
}

factory_struct! {
    /// `org.apache.lucene.analysis.minhash.MinHashFilterFactory` (`minHash`;
    /// Java does not check for unknown arguments).
    MinHashFilterFactory { hash_count: i32, bucket_count: i32, hash_set_size: i32, with_rotation: bool }
}
analysis_factory!(MinHashFilterFactory);

impl FactoryClass for MinHashFilterFactory {
    const NAME: &'static str = "minHash";
    const CLASS_NAME: &'static str = "org.apache.lucene.analysis.minhash.MinHashFilterFactory";
    fn from_args(args: &mut JavaArgs) -> Result<Self, FactoryError> {
        use crate::minhash::{DEFAULT_BUCKET_COUNT, DEFAULT_HASH_COUNT, DEFAULT_HASH_SET_SIZE};
        let base = FactoryBase::new(Self::CLASS_NAME, args)?;
        let hash_count = args::get_int(args, "hashCount", DEFAULT_HASH_COUNT)?;
        let bucket_count = args::get_int(args, "bucketCount", DEFAULT_BUCKET_COUNT)?;
        let hash_set_size = args::get_int(args, "hashSetSize", DEFAULT_HASH_SET_SIZE)?;
        let with_rotation = args::get_boolean(args, "withRotation", bucket_count > 1);
        Ok(MinHashFilterFactory {
            base,
            hash_count,
            bucket_count,
            hash_set_size,
            with_rotation,
        })
    }
}

impl TokenFilterFactory for MinHashFilterFactory {
    fn create(&self, input: Box<dyn TokenStream>) -> Result<Box<dyn TokenStream>, AnalysisError> {
        Ok(Box::new(crate::minhash::MinHashFilter::new(
            input,
            self.hash_count,
            self.bucket_count,
            self.hash_set_size,
            self.with_rotation,
        )?))
    }
}

factory_struct! {
    /// `org.apache.lucene.analysis.miscellaneous.ASCIIFoldingFilterFactory`
    /// (`asciiFolding`); `normalize` never preserves the original.
    ASCIIFoldingFilterFactory { preserve_original: bool }
}
analysis_factory!(ASCIIFoldingFilterFactory);

impl FactoryClass for ASCIIFoldingFilterFactory {
    const NAME: &'static str = "asciiFolding";
    const CLASS_NAME: &'static str =
        "org.apache.lucene.analysis.miscellaneous.ASCIIFoldingFilterFactory";
    fn from_args(args: &mut JavaArgs) -> Result<Self, FactoryError> {
        let base = FactoryBase::new(Self::CLASS_NAME, args)?;
        let preserve_original = args::get_boolean(args, "preserveOriginal", false);
        args::reject_unknown(args)?;
        Ok(ASCIIFoldingFilterFactory {
            base,
            preserve_original,
        })
    }
}

impl TokenFilterFactory for ASCIIFoldingFilterFactory {
    fn create(&self, input: Box<dyn TokenStream>) -> Result<Box<dyn TokenStream>, AnalysisError> {
        Ok(Box::new(misc::AsciiFoldingTokenFilter::new(
            input,
            self.preserve_original,
        )))
    }

    fn normalize(&self, input: Box<dyn TokenStream>) -> Box<dyn TokenStream> {
        Box::new(misc::AsciiFoldingTokenFilter::new(input, false))
    }
}

factory_struct! {
    /// `org.apache.lucene.analysis.miscellaneous.CapitalizationFilterFactory` (`capitalization`).
    CapitalizationFilterFactory {
        keep: Option<Arc<CharArraySet>>,
        ok_prefix: Option<Vec<String>>,
        min_word_length: i32,
        max_word_count: i32,
        max_token_length: i32,
        only_first_word: bool,
        force_first_letter: bool,
    }
}
analysis_factory!(CapitalizationFilterFactory);

impl FactoryClass for CapitalizationFilterFactory {
    const NAME: &'static str = "capitalization";
    const CLASS_NAME: &'static str =
        "org.apache.lucene.analysis.miscellaneous.CapitalizationFilterFactory";
    fn from_args(args: &mut JavaArgs) -> Result<Self, FactoryError> {
        let base = FactoryBase::new(Self::CLASS_NAME, args)?;
        let ignore_case = args::get_boolean(args, "keepIgnoreCase", false);
        let keep =
            args::get_set(args, "keep").map(|k| Arc::new(CharArraySet::from_words(k, ignore_case)));
        let ok_prefix = args::get_set(args, "okPrefix").map(|k| k.into_iter().collect());
        let min_word_length = args::get_int(args, "minWordLength", 0)?;
        let max_word_count = args::get_int(args, "maxWordCount", misc::CAPITALIZATION_DEFAULT_MAX)?;
        let max_token_length =
            args::get_int(args, "maxTokenLength", misc::CAPITALIZATION_DEFAULT_MAX)?;
        let only_first_word = args::get_boolean(args, "onlyFirstWord", true);
        let force_first_letter = args::get_boolean(args, "forceFirstLetter", true);
        args::reject_unknown(args)?;
        Ok(CapitalizationFilterFactory {
            base,
            keep,
            ok_prefix,
            min_word_length,
            max_word_count,
            max_token_length,
            only_first_word,
            force_first_letter,
        })
    }
}

impl TokenFilterFactory for CapitalizationFilterFactory {
    fn create(&self, input: Box<dyn TokenStream>) -> Result<Box<dyn TokenStream>, AnalysisError> {
        Ok(Box::new(misc::CapitalizationFilter::with_options(
            input,
            self.only_first_word,
            self.keep.clone(),
            self.force_first_letter,
            self.ok_prefix.clone(),
            self.min_word_length,
            self.max_word_count,
            self.max_token_length,
        )?))
    }
}

factory_struct! {
    /// `org.apache.lucene.analysis.miscellaneous.CodepointCountFilterFactory` (`codepointCount`).
    CodepointCountFilterFactory { min: i32, max: i32 }
}
analysis_factory!(CodepointCountFilterFactory);

impl FactoryClass for CodepointCountFilterFactory {
    const NAME: &'static str = "codepointCount";
    const CLASS_NAME: &'static str =
        "org.apache.lucene.analysis.miscellaneous.CodepointCountFilterFactory";
    fn from_args(args: &mut JavaArgs) -> Result<Self, FactoryError> {
        let base = FactoryBase::new(Self::CLASS_NAME, args)?;
        let min = args::require_int(args, "min")?;
        let max = args::require_int(args, "max")?;
        args::reject_unknown(args)?;
        Ok(CodepointCountFilterFactory { base, min, max })
    }
}

impl TokenFilterFactory for CodepointCountFilterFactory {
    fn create(&self, input: Box<dyn TokenStream>) -> Result<Box<dyn TokenStream>, AnalysisError> {
        Ok(Box::new(misc::CodepointCountFilter::new(
            input, self.min, self.max,
        )?))
    }
}

factory_struct! {
    /// `org.apache.lucene.analysis.miscellaneous.LengthFilterFactory` (`length`).
    LengthFilterFactory { min: i32, max: i32 }
}
analysis_factory!(LengthFilterFactory);

impl FactoryClass for LengthFilterFactory {
    const NAME: &'static str = "length";
    const CLASS_NAME: &'static str = "org.apache.lucene.analysis.miscellaneous.LengthFilterFactory";
    fn from_args(args: &mut JavaArgs) -> Result<Self, FactoryError> {
        let base = FactoryBase::new(Self::CLASS_NAME, args)?;
        let min = args::require_int(args, "min")?;
        let max = args::require_int(args, "max")?;
        args::reject_unknown(args)?;
        Ok(LengthFilterFactory { base, min, max })
    }
}

impl TokenFilterFactory for LengthFilterFactory {
    fn create(&self, input: Box<dyn TokenStream>) -> Result<Box<dyn TokenStream>, AnalysisError> {
        Ok(Box::new(misc::LengthFilter::new(
            input, self.min, self.max,
        )?))
    }
}

factory_struct! {
    /// `org.apache.lucene.analysis.miscellaneous.ConcatenateGraphFilterFactory` (`concatenateGraph`).
    ConcatenateGraphFilterFactory {
        token_separator: Option<char>,
        preserve_position_increments: bool,
        max_graph_expansions: i32,
    }
}
analysis_factory!(ConcatenateGraphFilterFactory);

impl FactoryClass for ConcatenateGraphFilterFactory {
    const NAME: &'static str = "concatenateGraph";
    const CLASS_NAME: &'static str =
        "org.apache.lucene.analysis.miscellaneous.ConcatenateGraphFilterFactory";
    fn from_args(args: &mut JavaArgs) -> Result<Self, FactoryError> {
        let base = FactoryBase::new(Self::CLASS_NAME, args)?;
        // getCharacter: an empty value is no separator.
        let token_separator = match args.remove("tokenSeparator") {
            None => Some(misc::SEP_LABEL),
            Some(s) if s.is_empty() => None,
            Some(s) => {
                let mut units = s.encode_utf16();
                match (units.next(), units.next()) {
                    (Some(u), None) => Some(args::unit_char(u)),
                    _ => {
                        return Err(FactoryError::illegal_argument(format!(
                            "tokenSeparator should be a char. \"{s}\" is invalid"
                        )))
                    }
                }
            }
        };
        let preserve_position_increments =
            args::get_boolean(args, "preservePositionIncrements", true);
        let max_graph_expansions = args::get_int(
            args,
            "maxGraphExpansions",
            misc::DEFAULT_MAX_GRAPH_EXPANSIONS,
        )?;
        args::reject_unknown(args)?;
        Ok(ConcatenateGraphFilterFactory {
            base,
            token_separator,
            preserve_position_increments,
            max_graph_expansions,
        })
    }
}

impl TokenFilterFactory for ConcatenateGraphFilterFactory {
    fn create(&self, input: Box<dyn TokenStream>) -> Result<Box<dyn TokenStream>, AnalysisError> {
        Ok(Box::new(misc::ConcatenateGraphFilter::with_options(
            input,
            self.token_separator,
            self.preserve_position_increments,
            self.max_graph_expansions,
        )))
    }
}

factory_struct! {
    /// `org.apache.lucene.analysis.miscellaneous.DateRecognizerFilterFactory`
    /// (`dateRecognizer`). Differs: only `Locale.ENGLISH` (`locale` absent or
    /// `en`); another well-formed locale is an `UnsupportedOperation` error,
    /// and a `datePattern` with the time zone letters `z`/`Z` an
    /// `IllegalArgument` one (see [`crate::miscellaneous::SimpleDateFormat`]).
    DateRecognizerFilterFactory { format: misc::SimpleDateFormat }
}
analysis_factory!(DateRecognizerFilterFactory);

/// `new Locale.Builder().setLanguageTag(tag).build()`, English only.
fn english_locale(tag: &str) -> Result<(), FactoryError> {
    let subtags: Vec<&str> = tag.split('-').collect();
    let well_formed = !subtags[0].is_empty()
        && (2..=8).contains(&subtags[0].len())
        && subtags[0].bytes().all(|b| b.is_ascii_alphabetic())
        && subtags[1..]
            .iter()
            .all(|s| (1..=8).contains(&s.len()) && s.bytes().all(|b| b.is_ascii_alphanumeric()));
    if !well_formed {
        return Err(FactoryError::new(
            JavaException::IllformedLocale,
            format!("Ill-formed language tag: {tag}"),
        ));
    }
    if tag.eq_ignore_ascii_case("en") {
        Ok(())
    } else {
        Err(FactoryError::new(
            JavaException::UnsupportedOperation,
            format!(
                "DateRecognizerFilterFactory: only Locale.ENGLISH's date formats are ported, not {tag}"
            ),
        ))
    }
}

impl FactoryClass for DateRecognizerFilterFactory {
    const NAME: &'static str = "dateRecognizer";
    const CLASS_NAME: &'static str =
        "org.apache.lucene.analysis.miscellaneous.DateRecognizerFilterFactory";
    fn from_args(args: &mut JavaArgs) -> Result<Self, FactoryError> {
        let base = FactoryBase::new(Self::CLASS_NAME, args)?;
        if let Some(tag) = args::get(args, "locale") {
            english_locale(&tag)?;
        }
        let format = match args::get(args, "datePattern") {
            Some(p) => misc::SimpleDateFormat::new(&p)?,
            None => misc::SimpleDateFormat::english_default(),
        };
        args::reject_unknown(args)?;
        Ok(DateRecognizerFilterFactory { base, format })
    }
}

impl TokenFilterFactory for DateRecognizerFilterFactory {
    fn create(&self, input: Box<dyn TokenStream>) -> Result<Box<dyn TokenStream>, AnalysisError> {
        Ok(Box::new(misc::DateRecognizerFilter::with_format(
            input,
            self.format.clone(),
        )))
    }
}

factory_struct! {
    /// `org.apache.lucene.analysis.miscellaneous.DelimitedTermFrequencyTokenFilterFactory`
    /// (`delimitedTermFrequency`).
    DelimitedTermFrequencyTokenFilterFactory { delimiter: u16 }
}
analysis_factory!(DelimitedTermFrequencyTokenFilterFactory);

impl FactoryClass for DelimitedTermFrequencyTokenFilterFactory {
    const NAME: &'static str = "delimitedTermFrequency";
    const CLASS_NAME: &'static str =
        "org.apache.lucene.analysis.miscellaneous.DelimitedTermFrequencyTokenFilterFactory";
    fn from_args(args: &mut JavaArgs) -> Result<Self, FactoryError> {
        let base = FactoryBase::new(Self::CLASS_NAME, args)?;
        let delimiter = args::get_char(
            args,
            "delimiter",
            misc::DEFAULT_TERM_FREQUENCY_DELIMITER as u16,
        )?;
        args::reject_unknown(args)?;
        Ok(DelimitedTermFrequencyTokenFilterFactory { base, delimiter })
    }
}

impl TokenFilterFactory for DelimitedTermFrequencyTokenFilterFactory {
    fn create(&self, input: Box<dyn TokenStream>) -> Result<Box<dyn TokenStream>, AnalysisError> {
        Ok(Box::new(
            misc::DelimitedTermFrequencyTokenFilter::with_delimiter(input, self.delimiter),
        ))
    }
}

factory_struct! {
    /// `org.apache.lucene.analysis.miscellaneous.DropIfFlaggedFilterFactory`
    /// (`dropIfFlagged`; Java does not check for unknown arguments).
    DropIfFlaggedFilterFactory { drop_flags: i32 }
}
analysis_factory!(DropIfFlaggedFilterFactory);

impl FactoryClass for DropIfFlaggedFilterFactory {
    const NAME: &'static str = "dropIfFlagged";
    const CLASS_NAME: &'static str =
        "org.apache.lucene.analysis.miscellaneous.DropIfFlaggedFilterFactory";
    fn from_args(args: &mut JavaArgs) -> Result<Self, FactoryError> {
        let base = FactoryBase::new(Self::CLASS_NAME, args)?;
        let drop_flags = args::get_int(args, "dropFlags", 2)?;
        Ok(DropIfFlaggedFilterFactory { base, drop_flags })
    }
}

impl TokenFilterFactory for DropIfFlaggedFilterFactory {
    fn create(&self, input: Box<dyn TokenStream>) -> Result<Box<dyn TokenStream>, AnalysisError> {
        Ok(Box::new(misc::DropIfFlaggedFilter::new(
            input,
            self.drop_flags,
        )))
    }
}

factory_struct! {
    /// `org.apache.lucene.analysis.miscellaneous.FingerprintFilterFactory` (`fingerprint`).
    FingerprintFilterFactory { max_output_token_size: i32, separator: char }
}
analysis_factory!(FingerprintFilterFactory);

impl FactoryClass for FingerprintFilterFactory {
    const NAME: &'static str = "fingerprint";
    const CLASS_NAME: &'static str =
        "org.apache.lucene.analysis.miscellaneous.FingerprintFilterFactory";
    fn from_args(args: &mut JavaArgs) -> Result<Self, FactoryError> {
        let base = FactoryBase::new(Self::CLASS_NAME, args)?;
        let max_output_token_size = args::get_int(
            args,
            "maxOutputTokenSize",
            misc::FINGERPRINT_DEFAULT_MAX_OUTPUT_TOKEN_SIZE,
        )?;
        let separator = args::get_char(
            args,
            "separator",
            misc::FINGERPRINT_DEFAULT_SEPARATOR as u16,
        )?;
        args::reject_unknown(args)?;
        Ok(FingerprintFilterFactory {
            base,
            max_output_token_size,
            separator: args::unit_char(separator),
        })
    }
}

impl TokenFilterFactory for FingerprintFilterFactory {
    fn create(&self, input: Box<dyn TokenStream>) -> Result<Box<dyn TokenStream>, AnalysisError> {
        Ok(Box::new(misc::FingerprintFilter::new(
            input,
            self.max_output_token_size,
            self.separator,
        )))
    }
}

/// Which `LimitToken*Filter` a [`LimitTokenFilterFactory`] builds.
#[derive(Debug, Clone, Copy)]
enum Limit {
    Count,
    Offset,
    Position,
}

/// `LimitTokenCountFilterFactory` (`limitTokenCount`),
/// `LimitTokenOffsetFilterFactory` (`limitTokenOffset`) and
/// `LimitTokenPositionFilterFactory` (`limitTokenPosition`): one required
/// limit and `consumeAllTokens`.
pub struct LimitTokenFilterFactory<const K: u8> {
    base: FactoryBase,
    max: i32,
    consume_all_tokens: bool,
}

/// `org.apache.lucene.analysis.miscellaneous.LimitTokenCountFilterFactory`.
pub type LimitTokenCountFilterFactory = LimitTokenFilterFactory<0>;
/// `org.apache.lucene.analysis.miscellaneous.LimitTokenOffsetFilterFactory`.
pub type LimitTokenOffsetFilterFactory = LimitTokenFilterFactory<1>;
/// `org.apache.lucene.analysis.miscellaneous.LimitTokenPositionFilterFactory`.
pub type LimitTokenPositionFilterFactory = LimitTokenFilterFactory<2>;

impl<const K: u8> LimitTokenFilterFactory<K> {
    const KIND: Limit = match K {
        0 => Limit::Count,
        1 => Limit::Offset,
        _ => Limit::Position,
    };
}

impl<const K: u8> super::AnalysisFactory for LimitTokenFilterFactory<K> {
    fn base(&self) -> &FactoryBase {
        &self.base
    }
    fn base_mut(&mut self) -> &mut FactoryBase {
        &mut self.base
    }
}

impl<const K: u8> FactoryClass for LimitTokenFilterFactory<K> {
    const NAME: &'static str = match K {
        0 => "limitTokenCount",
        1 => "limitTokenOffset",
        _ => "limitTokenPosition",
    };
    const CLASS_NAME: &'static str = match K {
        0 => "org.apache.lucene.analysis.miscellaneous.LimitTokenCountFilterFactory",
        1 => "org.apache.lucene.analysis.miscellaneous.LimitTokenOffsetFilterFactory",
        _ => "org.apache.lucene.analysis.miscellaneous.LimitTokenPositionFilterFactory",
    };
    fn from_args(args: &mut JavaArgs) -> Result<Self, FactoryError> {
        let base = FactoryBase::new(Self::CLASS_NAME, args)?;
        let key = match Self::KIND {
            Limit::Count => "maxTokenCount",
            Limit::Offset => "maxStartOffset",
            Limit::Position => "maxTokenPosition",
        };
        let max = args::require_int(args, key)?;
        let consume_all_tokens = args::get_boolean(args, "consumeAllTokens", false);
        args::reject_unknown(args)?;
        Ok(LimitTokenFilterFactory {
            base,
            max,
            consume_all_tokens,
        })
    }
}

impl<const K: u8> TokenFilterFactory for LimitTokenFilterFactory<K> {
    fn create(&self, input: Box<dyn TokenStream>) -> Result<Box<dyn TokenStream>, AnalysisError> {
        let (max, all) = (self.max, self.consume_all_tokens);
        Ok(match Self::KIND {
            Limit::Count => Box::new(misc::LimitTokenCountFilter::new(input, max, all)?),
            Limit::Offset => Box::new(misc::LimitTokenOffsetFilter::new(input, max, all)?),
            Limit::Position => Box::new(misc::LimitTokenPositionFilter::new(input, max, all)?),
        })
    }
}

/// How a [`TruncateTokenFilterFactory`] counts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Truncate {
    CodePoints,
    Chars,
}

factory_struct! {
    /// `org.apache.lucene.analysis.miscellaneous.TruncateTokenFilterFactory`
    /// (`truncate`): one of `truncateAfterCodePoints`, `truncateAfterChars`
    /// or the deprecated `prefixLength` (code points from 10.5.0, chars
    /// before). Differs: Java lists the three names in its error for more
    /// than one in an order that changes from JVM to JVM (`Map.of`); the port
    /// lists them in declaration order.
    TruncateTokenFilterFactory { truncate_after: i32, mode: Truncate }
}
analysis_factory!(TruncateTokenFilterFactory);

impl FactoryClass for TruncateTokenFilterFactory {
    const NAME: &'static str = "truncate";
    const CLASS_NAME: &'static str =
        "org.apache.lucene.analysis.miscellaneous.TruncateTokenFilterFactory";
    fn from_args(args: &mut JavaArgs) -> Result<Self, FactoryError> {
        let base = FactoryBase::new(Self::CLASS_NAME, args)?;
        const KEYS: [&str; 3] = [
            "truncateAfterCodePoints",
            "truncateAfterChars",
            "prefixLength",
        ];
        let avail: Vec<&str> = KEYS.into_iter().filter(|k| args.contains_key(k)).collect();
        if avail.len() > 1 {
            return Err(FactoryError::illegal_argument(format!(
                "Can only give one of the following parameters: [{}]",
                KEYS.join(", ")
            )));
        }
        let param = avail.first().copied().unwrap_or("prefixLength");
        let truncate_after = args::get_int(args, param, 5)?;
        let mode = match param {
            "truncateAfterCodePoints" => Truncate::CodePoints,
            "truncateAfterChars" => Truncate::Chars,
            _ if base
                .lucene_match_version()
                .on_or_after(Version::LUCENE_10_5_0) =>
            {
                Truncate::CodePoints
            }
            _ => Truncate::Chars,
        };
        if truncate_after < 1 {
            return Err(FactoryError::illegal_argument(format!(
                "{param} parameter must be a positive number: {truncate_after}"
            )));
        }
        args::reject_unknown_with(args, "Unknown parameter(s): ")?;
        Ok(TruncateTokenFilterFactory {
            base,
            truncate_after,
            mode,
        })
    }
}

impl TokenFilterFactory for TruncateTokenFilterFactory {
    fn create(&self, input: Box<dyn TokenStream>) -> Result<Box<dyn TokenStream>, AnalysisError> {
        Ok(Box::new(match self.mode {
            Truncate::CodePoints => {
                misc::TruncateTokenFilter::truncate_after_code_points(input, self.truncate_after)?
            }
            Truncate::Chars => {
                misc::TruncateTokenFilter::truncate_after_chars(input, self.truncate_after)?
            }
        }))
    }
}

factory_struct! {
    /// `org.apache.lucene.analysis.miscellaneous.TypeAsSynonymFilterFactory` (`typeAsSynonym`).
    TypeAsSynonymFilterFactory { prefix: Option<String>, ignore: Option<BTreeSet<String>>, syn_flags_mask: i32 }
}
analysis_factory!(TypeAsSynonymFilterFactory);

impl FactoryClass for TypeAsSynonymFilterFactory {
    const NAME: &'static str = "typeAsSynonym";
    const CLASS_NAME: &'static str =
        "org.apache.lucene.analysis.miscellaneous.TypeAsSynonymFilterFactory";
    fn from_args(args: &mut JavaArgs) -> Result<Self, FactoryError> {
        let base = FactoryBase::new(Self::CLASS_NAME, args)?;
        let prefix = args::get(args, "prefix");
        let ignore = args::get_set(args, "ignore");
        let syn_flags_mask = args::get_int(args, "synFlagsMask", !0)?;
        args::reject_unknown(args)?;
        Ok(TypeAsSynonymFilterFactory {
            base,
            prefix,
            ignore,
            syn_flags_mask,
        })
    }
}

impl TokenFilterFactory for TypeAsSynonymFilterFactory {
    fn create(&self, input: Box<dyn TokenStream>) -> Result<Box<dyn TokenStream>, AnalysisError> {
        let ignore: Option<HashSet<String>> =
            self.ignore.as_ref().map(|s| s.iter().cloned().collect());
        Ok(Box::new(misc::TypeAsSynonymFilter::new(
            input,
            self.prefix.as_deref(),
            ignore,
            self.syn_flags_mask,
        )))
    }
}

/// `org.apache.lucene.analysis.ngram.EdgeNGramFilterFactory` (`edgeNGram`)
/// and, with `EDGE` false, `NGramFilterFactory` (`nGram`).
pub struct GramFilterFactory<const EDGE: bool> {
    base: FactoryBase,
    min_gram_size: i32,
    max_gram_size: i32,
    preserve_original: bool,
}

/// `org.apache.lucene.analysis.ngram.EdgeNGramFilterFactory`.
pub type EdgeNGramFilterFactory = GramFilterFactory<true>;
/// `org.apache.lucene.analysis.ngram.NGramFilterFactory`.
pub type NGramFilterFactory = GramFilterFactory<false>;

impl<const EDGE: bool> super::AnalysisFactory for GramFilterFactory<EDGE> {
    fn base(&self) -> &FactoryBase {
        &self.base
    }
    fn base_mut(&mut self) -> &mut FactoryBase {
        &mut self.base
    }
}

impl<const EDGE: bool> FactoryClass for GramFilterFactory<EDGE> {
    const NAME: &'static str = if EDGE { "edgeNGram" } else { "nGram" };
    const CLASS_NAME: &'static str = if EDGE {
        "org.apache.lucene.analysis.ngram.EdgeNGramFilterFactory"
    } else {
        "org.apache.lucene.analysis.ngram.NGramFilterFactory"
    };
    fn from_args(args: &mut JavaArgs) -> Result<Self, FactoryError> {
        let base = FactoryBase::new(Self::CLASS_NAME, args)?;
        let min_gram_size = args::require_int(args, "minGramSize")?;
        let max_gram_size = args::require_int(args, "maxGramSize")?;
        // EdgeNGramTokenFilter / NGramTokenFilter.DEFAULT_PRESERVE_ORIGINAL
        let preserve_original = args::get_boolean(args, "preserveOriginal", false);
        args::reject_unknown(args)?;
        Ok(GramFilterFactory {
            base,
            min_gram_size,
            max_gram_size,
            preserve_original,
        })
    }
}

impl<const EDGE: bool> TokenFilterFactory for GramFilterFactory<EDGE> {
    fn create(&self, input: Box<dyn TokenStream>) -> Result<Box<dyn TokenStream>, AnalysisError> {
        let (min, max, keep) = (
            self.min_gram_size,
            self.max_gram_size,
            self.preserve_original,
        );
        Ok(if EDGE {
            Box::new(crate::ngram::EdgeNGramTokenFilter::new(
                input, min, max, keep,
            )?)
        } else {
            Box::new(crate::ngram::NGramTokenFilter::new(input, min, max, keep)?)
        })
    }
}

factory_struct! {
    /// `org.apache.lucene.analysis.pattern.PatternCaptureGroupFilterFactory`
    /// (`patternCaptureGroup`). As in Java, `preserve_original` is read but
    /// not consumed, and unknown arguments are not checked.
    PatternCaptureGroupFilterFactory { pattern: JavaPattern, preserve_original: bool }
}
analysis_factory!(PatternCaptureGroupFilterFactory);

impl FactoryClass for PatternCaptureGroupFilterFactory {
    const NAME: &'static str = "patternCaptureGroup";
    const CLASS_NAME: &'static str =
        "org.apache.lucene.analysis.pattern.PatternCaptureGroupFilterFactory";
    fn from_args(args: &mut JavaArgs) -> Result<Self, FactoryError> {
        let base = FactoryBase::new(Self::CLASS_NAME, args)?;
        let pattern = args::get_pattern(args, "pattern", "PatternCaptureGroupFilterFactory")?;
        let preserve_original = args
            .get("preserve_original")
            .is_none_or(args::parse_java_boolean);
        Ok(PatternCaptureGroupFilterFactory {
            base,
            pattern,
            preserve_original,
        })
    }
}

impl TokenFilterFactory for PatternCaptureGroupFilterFactory {
    fn create(&self, input: Box<dyn TokenStream>) -> Result<Box<dyn TokenStream>, AnalysisError> {
        Ok(Box::new(
            crate::pattern::PatternCaptureGroupTokenFilter::new(
                input,
                self.preserve_original,
                std::slice::from_ref(&self.pattern),
            ),
        ))
    }
}

factory_struct! {
    /// `org.apache.lucene.analysis.pattern.PatternReplaceFilterFactory` (`patternReplace`).
    PatternReplaceFilterFactory { pattern: JavaPattern, replacement: Option<String>, replace_all: bool }
}
analysis_factory!(PatternReplaceFilterFactory);

impl FactoryClass for PatternReplaceFilterFactory {
    const NAME: &'static str = "patternReplace";
    const CLASS_NAME: &'static str =
        "org.apache.lucene.analysis.pattern.PatternReplaceFilterFactory";
    fn from_args(args: &mut JavaArgs) -> Result<Self, FactoryError> {
        let base = FactoryBase::new(Self::CLASS_NAME, args)?;
        let pattern = args::get_pattern(args, "pattern", "PatternReplaceFilterFactory")?;
        let replacement = args::get(args, "replacement");
        let replace = args::get_one_of(args, "replace", &["all", "first"], Some("all"), true)?;
        args::reject_unknown(args)?;
        Ok(PatternReplaceFilterFactory {
            base,
            pattern,
            replacement,
            replace_all: replace.as_deref() == Some("all"),
        })
    }
}

impl TokenFilterFactory for PatternReplaceFilterFactory {
    fn create(&self, input: Box<dyn TokenStream>) -> Result<Box<dyn TokenStream>, AnalysisError> {
        Ok(Box::new(crate::pattern::PatternReplaceFilter::new(
            input,
            self.pattern.clone(),
            self.replacement.as_deref(),
            self.replace_all,
        )))
    }
}

factory_struct! {
    /// `org.apache.lucene.analysis.payloads.NumericPayloadTokenFilterFactory` (`numericPayload`).
    NumericPayloadTokenFilterFactory { payload: f32, type_match: String }
}
analysis_factory!(NumericPayloadTokenFilterFactory);

impl FactoryClass for NumericPayloadTokenFilterFactory {
    const NAME: &'static str = "numericPayload";
    const CLASS_NAME: &'static str =
        "org.apache.lucene.analysis.payloads.NumericPayloadTokenFilterFactory";
    fn from_args(args: &mut JavaArgs) -> Result<Self, FactoryError> {
        let base = FactoryBase::new(Self::CLASS_NAME, args)?;
        let payload = args::require_float(args, "payload")?;
        let type_match = args::require(args, "typeMatch")?;
        args::reject_unknown(args)?;
        Ok(NumericPayloadTokenFilterFactory {
            base,
            payload,
            type_match,
        })
    }
}

impl TokenFilterFactory for NumericPayloadTokenFilterFactory {
    fn create(&self, input: Box<dyn TokenStream>) -> Result<Box<dyn TokenStream>, AnalysisError> {
        Ok(Box::new(crate::payloads::NumericPayloadTokenFilter::new(
            input,
            self.payload,
            &self.type_match,
        )))
    }
}

factory_struct! {
    /// `org.apache.lucene.analysis.shingle.FixedShingleFilterFactory`
    /// (`fixedShingle`; Java does not check for unknown arguments).
    FixedShingleFilterFactory { shingle_size: i32, token_separator: String, filler_token: String }
}
analysis_factory!(FixedShingleFilterFactory);

impl FactoryClass for FixedShingleFilterFactory {
    const NAME: &'static str = "fixedShingle";
    const CLASS_NAME: &'static str = "org.apache.lucene.analysis.shingle.FixedShingleFilterFactory";
    fn from_args(args: &mut JavaArgs) -> Result<Self, FactoryError> {
        let base = FactoryBase::new(Self::CLASS_NAME, args)?;
        let shingle_size = args::get_int(args, "shingleSize", 2)?;
        let token_separator = args::get_or(args, "tokenSeparator", " ");
        let filler_token = args::get_or(args, "fillerToken", "_");
        Ok(FixedShingleFilterFactory {
            base,
            shingle_size,
            token_separator,
            filler_token,
        })
    }
}

impl TokenFilterFactory for FixedShingleFilterFactory {
    fn create(&self, input: Box<dyn TokenStream>) -> Result<Box<dyn TokenStream>, AnalysisError> {
        Ok(Box::new(
            crate::shingle::FixedShingleFilter::with_separator(
                input,
                self.shingle_size,
                &self.token_separator,
                &self.filler_token,
            )?,
        ))
    }
}

factory_struct! {
    /// `org.apache.lucene.analysis.shingle.ShingleFilterFactory` (`shingle`).
    ShingleFilterFactory {
        min_shingle_size: i32,
        max_shingle_size: i32,
        output_unigrams: bool,
        output_unigrams_if_no_shingles: bool,
        token_separator: String,
        filler_token: String,
    }
}
analysis_factory!(ShingleFilterFactory);

impl FactoryClass for ShingleFilterFactory {
    const NAME: &'static str = "shingle";
    const CLASS_NAME: &'static str = "org.apache.lucene.analysis.shingle.ShingleFilterFactory";
    fn from_args(args: &mut JavaArgs) -> Result<Self, FactoryError> {
        use crate::shingle::{
            DEFAULT_FILLER_TOKEN, DEFAULT_MAX_SHINGLE_SIZE, DEFAULT_MIN_SHINGLE_SIZE,
            DEFAULT_TOKEN_SEPARATOR,
        };
        let base = FactoryBase::new(Self::CLASS_NAME, args)?;
        let max_shingle_size = args::get_int(args, "maxShingleSize", DEFAULT_MAX_SHINGLE_SIZE)?;
        if max_shingle_size < 2 {
            return Err(FactoryError::illegal_argument(format!(
                "Invalid maxShingleSize ({max_shingle_size}) - must be at least 2"
            )));
        }
        let min_shingle_size = args::get_int(args, "minShingleSize", DEFAULT_MIN_SHINGLE_SIZE)?;
        if min_shingle_size < 2 {
            return Err(FactoryError::illegal_argument(format!(
                "Invalid minShingleSize ({min_shingle_size}) - must be at least 2"
            )));
        }
        if min_shingle_size > max_shingle_size {
            return Err(FactoryError::illegal_argument(format!(
                "Invalid minShingleSize ({min_shingle_size}) - must be no greater than maxShingleSize ({max_shingle_size})"
            )));
        }
        let output_unigrams = args::get_boolean(args, "outputUnigrams", true);
        let output_unigrams_if_no_shingles =
            args::get_boolean(args, "outputUnigramsIfNoShingles", false);
        let token_separator = args::get_or(args, "tokenSeparator", DEFAULT_TOKEN_SEPARATOR);
        let filler_token = args::get_or(args, "fillerToken", DEFAULT_FILLER_TOKEN);
        args::reject_unknown(args)?;
        Ok(ShingleFilterFactory {
            base,
            min_shingle_size,
            max_shingle_size,
            output_unigrams,
            output_unigrams_if_no_shingles,
            token_separator,
            filler_token,
        })
    }
}

impl TokenFilterFactory for ShingleFilterFactory {
    fn create(&self, input: Box<dyn TokenStream>) -> Result<Box<dyn TokenStream>, AnalysisError> {
        let mut f = crate::shingle::ShingleFilter::new(
            input,
            self.min_shingle_size,
            self.max_shingle_size,
        )?;
        f.set_output_unigrams(self.output_unigrams);
        f.set_output_unigrams_if_no_shingles(self.output_unigrams_if_no_shingles);
        f.set_token_separator(Some(&self.token_separator));
        f.set_filler_token(Some(&self.filler_token));
        Ok(Box::new(f))
    }
}
