//! The `ResourceLoaderAware` token filter factories: word files (`stop`,
//! `keepWord`, `commonGrams`, `commonGramsQuery`, `keywordMarker`,
//! `elision`), dictionaries (`dictionaryCompoundWord`,
//! `hyphenationCompoundWord`, `hunspellStem`, `stemmerOverride`), rule files
//! (`type`, `patternTyping`, the word delimiter `types`), named classes
//! (`delimitedPayload`'s encoder, `snowballPorter`'s stemmer), and the
//! synonym maps (`synonym`, `synonymGraph`).
//!
//! Each reads its arguments in its constructor and its resources in
//! `inform`, as in Java; `create` before `inform` builds what Java's would
//! with the fields still `null` (no filter where Java returns the input,
//! an `IllegalState` error where Java would throw a `NullPointerException`).

use std::sync::Arc;

use super::args::{self, JavaArgs};
use super::char_filters::parse_escaped;
use super::loader::{decode_utf8, get_lines, get_snowball_word_set, get_word_set, java_trim};
use super::{
    analysis_factory, factory_struct, spi, FactoryBase, FactoryClass, FactoryError, JavaException,
    ResourceLoader, TokenFilterFactory, TokenizerFactory,
};
use crate::compound::{CompoundSizes, CompoundWordTokenFilter, HyphenationTree};
use crate::hunspell::{Dictionary, HunspellError, HunspellStemFilter};
use crate::miscellaneous as misc;
use crate::synonym::{SolrSynonymParser, SynonymMap, SynonymParseError, WordnetSynonymParser};
use crate::token_stream::TokenStream;
use crate::util::java_regex::JavaMatcher;
use crate::util::JavaPattern;
use crate::{
    AnalysisError, Analyzer, AnalyzerDefinition, CharArraySet, TokenStreamComponents,
    ENGLISH_STOP_WORDS,
};

/// The `NullPointerException` Java throws from a factory used before
/// `inform`.
fn not_informed(class: &str) -> AnalysisError {
    AnalysisError::IllegalState(format!(
        "NullPointerException: {class} was not informed of its resources"
    ))
}

/// `String.split(regex)` with a literal separator: trailing empty items are
/// dropped (all of them, for a string of separators only).
pub(crate) fn java_split<'a>(s: &'a str, sep: &str) -> Vec<&'a str> {
    if s.is_empty() {
        return vec![""];
    }
    let mut parts: Vec<&str> = s.split(sep).collect();
    while parts.last().is_some_and(|p| p.is_empty()) {
        parts.pop();
    }
    parts
}

// ------------------------------------------------ AbstractWordsFileFilterFactory

/// What an `AbstractWordsFileFilterFactory` subclass builds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WordsKind {
    Stop,
    Keep,
    CommonGrams,
    CommonGramsQuery,
}

/// `org.apache.lucene.analysis.en.AbstractWordsFileFilterFactory` and its
/// four subclasses: `words` (comma-separated files), `format` (`wordset` or
/// `snowball`), `ignoreCase`.
pub struct WordsFileFilterFactory<const K: u8> {
    base: FactoryBase,
    words: Option<Arc<CharArraySet>>,
    word_files: Option<String>,
    format: Option<String>,
    ignore_case: bool,
}

/// `org.apache.lucene.analysis.core.StopFilterFactory` (`stop`).
pub type StopFilterFactory = WordsFileFilterFactory<0>;
/// `org.apache.lucene.analysis.miscellaneous.KeepWordFilterFactory` (`keepWord`).
pub type KeepWordFilterFactory = WordsFileFilterFactory<1>;
/// `org.apache.lucene.analysis.commongrams.CommonGramsFilterFactory` (`commonGrams`).
pub type CommonGramsFilterFactory = WordsFileFilterFactory<2>;
/// `org.apache.lucene.analysis.commongrams.CommonGramsQueryFilterFactory` (`commonGramsQuery`).
pub type CommonGramsQueryFilterFactory = WordsFileFilterFactory<3>;

impl<const K: u8> WordsFileFilterFactory<K> {
    const KIND: WordsKind = match K {
        0 => WordsKind::Stop,
        1 => WordsKind::Keep,
        2 => WordsKind::CommonGrams,
        _ => WordsKind::CommonGramsQuery,
    };

    /// `getWords()`.
    pub fn words(&self) -> Option<&Arc<CharArraySet>> {
        self.words.as_ref()
    }

    /// `isIgnoreCase()`.
    pub fn is_ignore_case(&self) -> bool {
        self.ignore_case
    }

    /// `createDefaultWords()`: the English stop words, or none for `keepWord`.
    fn create_default_words(&self) -> Option<CharArraySet> {
        match Self::KIND {
            WordsKind::Keep => None,
            _ => Some(CharArraySet::from_words(
                ENGLISH_STOP_WORDS,
                self.ignore_case,
            )),
        }
    }

    // Java: AbstractWordsFileFilterFactory.inform
    fn inform_impl(&mut self, loader: &dyn ResourceLoader) -> Result<(), FactoryError> {
        let words = match &self.word_files {
            Some(files) => {
                let format = self.format.as_deref().unwrap_or("");
                if args::java_equals_ignore_case(format, "wordset") {
                    get_word_set(loader, files, self.ignore_case)?
                } else if args::java_equals_ignore_case(format, "snowball") {
                    get_snowball_word_set(loader, files, self.ignore_case)?
                } else {
                    return Err(FactoryError::illegal_argument(format!(
                        "Unknown 'format' specified for 'words' file: {}",
                        self.format.as_deref().unwrap_or("null")
                    )));
                }
            }
            None => {
                if let Some(format) = &self.format {
                    return Err(FactoryError::illegal_argument(format!(
                        "'format' can not be specified w/o an explicit 'words' file: {format}"
                    )));
                }
                self.create_default_words()
            }
        };
        self.words = words.map(Arc::new);
        Ok(())
    }
}

impl<const K: u8> super::AnalysisFactory for WordsFileFilterFactory<K> {
    fn base(&self) -> &FactoryBase {
        &self.base
    }
    fn base_mut(&mut self) -> &mut FactoryBase {
        &mut self.base
    }
    fn is_resource_loader_aware(&self) -> bool {
        true
    }
    fn inform(&mut self, loader: &dyn ResourceLoader) -> Result<(), FactoryError> {
        self.inform_impl(loader)
    }
}

impl<const K: u8> FactoryClass for WordsFileFilterFactory<K> {
    const NAME: &'static str = match K {
        0 => "stop",
        1 => "keepWord",
        2 => "commonGrams",
        _ => "commonGramsQuery",
    };
    const CLASS_NAME: &'static str = match K {
        0 => "org.apache.lucene.analysis.core.StopFilterFactory",
        1 => "org.apache.lucene.analysis.miscellaneous.KeepWordFilterFactory",
        2 => "org.apache.lucene.analysis.commongrams.CommonGramsFilterFactory",
        _ => "org.apache.lucene.analysis.commongrams.CommonGramsQueryFilterFactory",
    };
    fn from_args(args: &mut JavaArgs) -> Result<Self, FactoryError> {
        let base = FactoryBase::new(Self::CLASS_NAME, args)?;
        let word_files = args::get(args, "words");
        let format = match args::get(args, "format") {
            Some(f) => Some(f),
            None => word_files.as_ref().map(|_| "wordset".to_string()),
        };
        let ignore_case = args::get_boolean(args, "ignoreCase", false);
        args::reject_unknown(args)?;
        Ok(WordsFileFilterFactory {
            base,
            words: None,
            word_files,
            format,
            ignore_case,
        })
    }
}

impl<const K: u8> TokenFilterFactory for WordsFileFilterFactory<K> {
    fn create(&self, input: Box<dyn TokenStream>) -> Result<Box<dyn TokenStream>, AnalysisError> {
        let words = self.words.clone();
        Ok(match Self::KIND {
            WordsKind::Stop => match words {
                Some(w) => Box::new(crate::StopFilter::new(input, w)),
                None => return Err(not_informed(Self::CLASS_NAME)),
            },
            WordsKind::Keep => match words {
                Some(w) => Box::new(misc::KeepWordFilter::new(input, w)),
                None => input,
            },
            WordsKind::CommonGrams => {
                Box::new(crate::commongrams::CommonGramsFilter::new(input, words))
            }
            WordsKind::CommonGramsQuery => {
                Box::new(crate::commongrams::CommonGramsQueryFilter::new(
                    crate::commongrams::CommonGramsFilter::new(input, words),
                ))
            }
        })
    }
}

// ------------------------------------------------------------- compounds

factory_struct! {
    /// `org.apache.lucene.analysis.compound.DictionaryCompoundWordTokenFilterFactory`
    /// (`dictionaryCompoundWord`).
    DictionaryCompoundWordTokenFilterFactory {
        dictionary: Option<Arc<CharArraySet>>,
        dict_file: String,
        sizes: CompoundSizes,
        only_longest_match_ignore_subwords: bool,
    }
}
analysis_factory!(DictionaryCompoundWordTokenFilterFactory, aware);

/// The three sizes, as `usize` (Java's negative-size errors come from the
/// filter's constructor, so a negative size is refused there).
fn compound_sizes(
    args: &mut JavaArgs,
    only_longest_match_default: bool,
) -> Result<(i32, i32, i32, bool), FactoryError> {
    let min_word_size = args::get_int(
        args,
        "minWordSize",
        crate::compound::DEFAULT_MIN_WORD_SIZE as i32,
    )?;
    let min_subword_size = args::get_int(
        args,
        "minSubwordSize",
        crate::compound::DEFAULT_MIN_SUBWORD_SIZE as i32,
    )?;
    let max_subword_size = args::get_int(
        args,
        "maxSubwordSize",
        crate::compound::DEFAULT_MAX_SUBWORD_SIZE as i32,
    )?;
    let only_longest_match =
        args::get_boolean(args, "onlyLongestMatch", only_longest_match_default);
    Ok((
        min_word_size,
        min_subword_size,
        max_subword_size,
        only_longest_match,
    ))
}

/// `CompoundWordTokenFilterBase`'s size checks.
fn checked_sizes(
    (min_word_size, min_subword_size, max_subword_size, only_longest_match): (i32, i32, i32, bool),
) -> Result<CompoundSizes, FactoryError> {
    let to_size = |v: i32, name: &str| {
        usize::try_from(v)
            .map_err(|_| FactoryError::illegal_argument(format!("{name} cannot be negative")))
    };
    Ok(CompoundSizes {
        min_word_size: to_size(min_word_size, "minWordSize")?,
        min_subword_size: to_size(min_subword_size, "minSubwordSize")?,
        max_subword_size: to_size(max_subword_size, "maxSubwordSize")?,
        only_longest_match,
    })
}

impl FactoryClass for DictionaryCompoundWordTokenFilterFactory {
    const NAME: &'static str = "dictionaryCompoundWord";
    const CLASS_NAME: &'static str =
        "org.apache.lucene.analysis.compound.DictionaryCompoundWordTokenFilterFactory";
    fn from_args(args: &mut JavaArgs) -> Result<Self, FactoryError> {
        let base = FactoryBase::new(Self::CLASS_NAME, args)?;
        let dict_file = args::require(args, "dictionary")?;
        let sizes = compound_sizes(args, true)?;
        let only_longest_match_ignore_subwords =
            args::get_boolean(args, "onlyLongestMatchIgnoreSubwords", true);
        args::reject_unknown(args)?;
        Ok(DictionaryCompoundWordTokenFilterFactory {
            base,
            dictionary: None,
            dict_file,
            sizes: checked_sizes(sizes)?,
            only_longest_match_ignore_subwords,
        })
    }
}

impl DictionaryCompoundWordTokenFilterFactory {
    // Java: DictionaryCompoundWordTokenFilterFactory.inform
    fn inform_impl(&mut self, loader: &dyn ResourceLoader) -> Result<(), FactoryError> {
        self.dictionary = get_word_set(loader, &self.dict_file, false)?.map(Arc::new);
        Ok(())
    }
}

impl TokenFilterFactory for DictionaryCompoundWordTokenFilterFactory {
    fn create(&self, input: Box<dyn TokenStream>) -> Result<Box<dyn TokenStream>, AnalysisError> {
        Ok(match &self.dictionary {
            None => input,
            Some(d) => Box::new(CompoundWordTokenFilter::dictionary_with(
                input,
                Arc::clone(d),
                self.sizes,
                self.only_longest_match_ignore_subwords,
            )),
        })
    }
}

factory_struct! {
    /// `org.apache.lucene.analysis.compound.HyphenationCompoundWordTokenFilterFactory`
    /// (`hyphenationCompoundWord`). The grammar's bytes are decoded as the
    /// JDK's SAX parser decodes them ([`super::xml_source`]): by the
    /// `encoding` argument when there is one, else by the byte-order mark
    /// or the XML declaration.
    HyphenationCompoundWordTokenFilterFactory {
        dictionary: Option<Arc<CharArraySet>>,
        hyphenator: Option<Arc<HyphenationTree>>,
        dict_file: Option<String>,
        encoding: Option<String>,
        hyp_file: String,
        sizes: CompoundSizes,
        no_sub_matches: bool,
        no_overlapping_matches: bool,
    }
}
analysis_factory!(HyphenationCompoundWordTokenFilterFactory, aware);

impl FactoryClass for HyphenationCompoundWordTokenFilterFactory {
    const NAME: &'static str = "hyphenationCompoundWord";
    const CLASS_NAME: &'static str =
        "org.apache.lucene.analysis.compound.HyphenationCompoundWordTokenFilterFactory";
    fn from_args(args: &mut JavaArgs) -> Result<Self, FactoryError> {
        let base = FactoryBase::new(Self::CLASS_NAME, args)?;
        let dict_file = args::get(args, "dictionary");
        let encoding = args::get(args, "encoding");
        let hyp_file = args::require(args, "hyphenator")?;
        let sizes = compound_sizes(args, false)?;
        let no_sub_matches = args::get_boolean(args, "noSubMatches", false);
        let no_overlapping_matches = args::get_boolean(args, "noOverlappingMatches", false);
        args::reject_unknown(args)?;
        Ok(HyphenationCompoundWordTokenFilterFactory {
            base,
            dictionary: None,
            hyphenator: None,
            dict_file,
            encoding,
            hyp_file,
            sizes: checked_sizes(sizes)?,
            no_sub_matches,
            no_overlapping_matches,
        })
    }
}

impl HyphenationCompoundWordTokenFilterFactory {
    // Java: HyphenationCompoundWordTokenFilterFactory.inform
    fn inform_impl(&mut self, loader: &dyn ResourceLoader) -> Result<(), FactoryError> {
        if let Some(dict_file) = &self.dict_file {
            self.dictionary = get_word_set(loader, dict_file, false)?.map(Arc::new);
        }
        let bytes = loader.open_resource(&self.hyp_file)?;
        let xml = super::xml_source::decode(&bytes, self.encoding.as_deref(), &self.hyp_file)?;
        let tree = HyphenationTree::from_xml(&xml)
            .map_err(|e| FactoryError::new(JavaException::Runtime, e.to_string()))?;
        self.hyphenator = Some(Arc::new(tree));
        Ok(())
    }
}

impl TokenFilterFactory for HyphenationCompoundWordTokenFilterFactory {
    fn create(&self, input: Box<dyn TokenStream>) -> Result<Box<dyn TokenStream>, AnalysisError> {
        let hyphenator = self
            .hyphenator
            .clone()
            .ok_or_else(|| not_informed(Self::CLASS_NAME))?;
        Ok(Box::new(CompoundWordTokenFilter::hyphenation(
            input,
            hyphenator,
            self.dictionary.clone(),
            self.sizes,
            self.no_sub_matches,
            self.no_overlapping_matches,
        )))
    }
}

// ------------------------------------------------------------------ type

factory_struct! {
    /// `org.apache.lucene.analysis.core.TypeTokenFilterFactory` (`type`).
    TypeTokenFilterFactory { use_whitelist: bool, stop_types_files: String, stop_types: Option<Vec<String>> }
}
analysis_factory!(TypeTokenFilterFactory, aware);

impl FactoryClass for TypeTokenFilterFactory {
    const NAME: &'static str = "type";
    const CLASS_NAME: &'static str = "org.apache.lucene.analysis.core.TypeTokenFilterFactory";
    fn from_args(args: &mut JavaArgs) -> Result<Self, FactoryError> {
        let base = FactoryBase::new(Self::CLASS_NAME, args)?;
        let stop_types_files = args::require(args, "types")?;
        let use_whitelist = args::get_boolean(args, "useWhitelist", false);
        args::reject_unknown(args)?;
        Ok(TypeTokenFilterFactory {
            base,
            use_whitelist,
            stop_types_files,
            stop_types: None,
        })
    }
}

impl TypeTokenFilterFactory {
    // Java: TypeTokenFilterFactory.inform
    fn inform_impl(&mut self, loader: &dyn ResourceLoader) -> Result<(), FactoryError> {
        let files = args::split_file_names(Some(&self.stop_types_files));
        if !files.is_empty() {
            let mut types = Vec::new();
            for file in &files {
                types.extend(get_lines(loader, java_trim(file))?);
            }
            self.stop_types = Some(types);
        }
        Ok(())
    }

    /// `getStopTypes()`.
    pub fn stop_types(&self) -> Option<&[String]> {
        self.stop_types.as_deref()
    }
}

impl TokenFilterFactory for TypeTokenFilterFactory {
    fn create(&self, input: Box<dyn TokenStream>) -> Result<Box<dyn TokenStream>, AnalysisError> {
        let types = self
            .stop_types
            .clone()
            .ok_or_else(|| not_informed(Self::CLASS_NAME))?;
        Ok(Box::new(crate::core_analysis::TypeTokenFilter::new(
            input,
            types,
            self.use_whitelist,
        )))
    }
}

// --------------------------------------------------------------- hunspell

factory_struct! {
    /// `org.apache.lucene.analysis.hunspell.HunspellStemFilterFactory`
    /// (`hunspellStem`): `dictionary` (comma-separated `.dic` files),
    /// `affix`, `ignoreCase`, `longestOnly`; `strictAffixParsing` and
    /// `recursionCap` are read and ignored, as in Java. Differs: a
    /// dictionary Lucene cannot parse is Java's `IOException`, whose message
    /// names the dictionaries by file name where Java prints its
    /// `InputStream` objects.
    HunspellStemFilterFactory {
        dictionary_files: String,
        affix_file: Option<String>,
        ignore_case: bool,
        longest_only: bool,
        dictionary: Option<Arc<Dictionary>>,
    }
}
analysis_factory!(HunspellStemFilterFactory, aware);

impl FactoryClass for HunspellStemFilterFactory {
    const NAME: &'static str = "hunspellStem";
    const CLASS_NAME: &'static str =
        "org.apache.lucene.analysis.hunspell.HunspellStemFilterFactory";
    fn from_args(args: &mut JavaArgs) -> Result<Self, FactoryError> {
        let base = FactoryBase::new(Self::CLASS_NAME, args)?;
        let dictionary_files = args::require(args, "dictionary")?;
        let affix_file = args::get(args, "affix");
        let ignore_case = args::get_boolean(args, "ignoreCase", false);
        let longest_only = args::get_boolean(args, "longestOnly", false);
        args::get_boolean(args, "strictAffixParsing", true);
        args::get_int(args, "recursionCap", 0)?;
        args::reject_unknown(args)?;
        Ok(HunspellStemFilterFactory {
            base,
            dictionary_files,
            affix_file,
            ignore_case,
            longest_only,
            dictionary: None,
        })
    }
}

/// A `Dictionary` constructor's exception, as the factory lets it through.
fn hunspell_error(e: HunspellError, dicts: &[&str], affix: &str) -> FactoryError {
    let kind = match &e {
        HunspellError::Parse { .. } => {
            return FactoryError::io(format!(
                "Unable to load hunspell data! [dictionary=[{}],affix={affix}]",
                dicts.join(", ")
            ))
        }
        HunspellError::IllegalArgument(_) => JavaException::IllegalArgument,
        HunspellError::NumberFormat(_) => JavaException::NumberFormat,
        HunspellError::IndexOutOfBounds(_) => JavaException::ArrayIndexOutOfBounds,
        HunspellError::UnsupportedCharset(_)
        | HunspellError::IllegalCharsetName(_)
        | HunspellError::Unsupported(_) => JavaException::UnsupportedOperation,
        HunspellError::UnmappableCharacter(_) => JavaException::Io,
        HunspellError::IllegalState(_) | HunspellError::NegativeArraySize(_) => {
            JavaException::IllegalState
        }
    };
    FactoryError::new(kind, e.to_string())
}

/// [`hunspell_error`] for the unit tests.
#[cfg(test)]
pub(crate) fn hunspell_error_for_tests(e: HunspellError) -> FactoryError {
    hunspell_error(e, &["d.dic"], "a.aff")
}

impl HunspellStemFilterFactory {
    // Java: HunspellStemFilterFactory.inform
    fn inform_impl(&mut self, loader: &dyn ResourceLoader) -> Result<(), FactoryError> {
        let dicts = java_split(&self.dictionary_files, ",");
        let mut dictionaries = Vec::with_capacity(dicts.len());
        for file in &dicts {
            dictionaries.push(loader.open_resource(file)?);
        }
        let affix_file = self.affix_file.as_deref().ok_or_else(|| {
            FactoryError::new(JavaException::NullPointer, "the affix file is null")
        })?;
        let affix = loader.open_resource(affix_file)?;
        let refs: Vec<&[u8]> = dictionaries.iter().map(Vec::as_slice).collect();
        let dictionary = Dictionary::new(&affix, &refs, self.ignore_case)
            .map_err(|e| hunspell_error(e, &dicts, affix_file))?;
        self.dictionary = Some(Arc::new(dictionary));
        Ok(())
    }
}

impl TokenFilterFactory for HunspellStemFilterFactory {
    fn create(&self, input: Box<dyn TokenStream>) -> Result<Box<dyn TokenStream>, AnalysisError> {
        let dictionary = self
            .dictionary
            .clone()
            .ok_or_else(|| not_informed(Self::CLASS_NAME))?;
        Ok(Box::new(HunspellStemFilter::new(
            input,
            dictionary,
            true,
            self.longest_only,
        )))
    }
}

// ---------------------------------------------------------- keyword marker

factory_struct! {
    /// `org.apache.lucene.analysis.miscellaneous.KeywordMarkerFilterFactory`
    /// (`keywordMarker`): `protected` files and/or a `pattern`
    /// (case-insensitive, Unicode-aware, under `ignoreCase`).
    KeywordMarkerFilterFactory {
        word_files: Option<String>,
        string_pattern: Option<String>,
        ignore_case: bool,
        pattern: Option<JavaPattern>,
        protected_words: Option<Arc<CharArraySet>>,
    }
}
analysis_factory!(KeywordMarkerFilterFactory, aware);

impl FactoryClass for KeywordMarkerFilterFactory {
    const NAME: &'static str = "keywordMarker";
    const CLASS_NAME: &'static str =
        "org.apache.lucene.analysis.miscellaneous.KeywordMarkerFilterFactory";
    fn from_args(args: &mut JavaArgs) -> Result<Self, FactoryError> {
        let base = FactoryBase::new(Self::CLASS_NAME, args)?;
        let word_files = args::get(args, "protected");
        let string_pattern = args::get(args, "pattern");
        let ignore_case = args::get_boolean(args, "ignoreCase", false);
        args::reject_unknown(args)?;
        Ok(KeywordMarkerFilterFactory {
            base,
            word_files,
            string_pattern,
            ignore_case,
            pattern: None,
            protected_words: None,
        })
    }
}

/// `Pattern.compile(p)`; a pattern Java refuses is its
/// `PatternSyntaxException`, one only the regex shim refuses an
/// `UnsupportedOperationException` (see [`args::unsupported_pattern`]).
fn compile_pattern(p: &str) -> Result<JavaPattern, FactoryError> {
    JavaPattern::compile(p).map_err(|e| {
        args::unsupported_pattern(&e)
            .unwrap_or_else(|| FactoryError::new(JavaException::PatternSyntax, e.to_string()))
    })
}

impl KeywordMarkerFilterFactory {
    // Java: KeywordMarkerFilterFactory.inform
    fn inform_impl(&mut self, loader: &dyn ResourceLoader) -> Result<(), FactoryError> {
        if let Some(files) = &self.word_files {
            self.protected_words = get_word_set(loader, files, self.ignore_case)?.map(Arc::new);
        }
        if let Some(p) = &self.string_pattern {
            // Pattern.CASE_INSENSITIVE | Pattern.UNICODE_CASE
            let p = if self.ignore_case {
                format!("(?iu){p}")
            } else {
                p.clone()
            };
            self.pattern = Some(compile_pattern(&p)?);
        }
        Ok(())
    }

    /// `isIgnoreCase()`.
    pub fn is_ignore_case(&self) -> bool {
        self.ignore_case
    }
}

impl TokenFilterFactory for KeywordMarkerFilterFactory {
    fn create(
        &self,
        mut input: Box<dyn TokenStream>,
    ) -> Result<Box<dyn TokenStream>, AnalysisError> {
        if let Some(p) = &self.pattern {
            input = Box::new(misc::PatternKeywordMarkerFilter::with_pattern(
                input,
                p.clone(),
            ));
        }
        if let Some(w) = &self.protected_words {
            input = Box::new(misc::SetKeywordMarkerFilter::new(input, Arc::clone(w)));
        }
        Ok(input)
    }
}

factory_struct! {
    /// `org.apache.lucene.analysis.miscellaneous.StemmerOverrideFilterFactory`
    /// (`stemmerOverride`): `word<TAB>stem` lines.
    StemmerOverrideFilterFactory {
        dictionary: Option<Arc<misc::StemmerOverrideMap>>,
        dictionary_files: Option<String>,
        ignore_case: bool,
    }
}
analysis_factory!(StemmerOverrideFilterFactory, aware);

impl FactoryClass for StemmerOverrideFilterFactory {
    const NAME: &'static str = "stemmerOverride";
    const CLASS_NAME: &'static str =
        "org.apache.lucene.analysis.miscellaneous.StemmerOverrideFilterFactory";
    fn from_args(args: &mut JavaArgs) -> Result<Self, FactoryError> {
        let base = FactoryBase::new(Self::CLASS_NAME, args)?;
        let dictionary_files = args::get(args, "dictionary");
        let ignore_case = args::get_boolean(args, "ignoreCase", false);
        args::reject_unknown(args)?;
        Ok(StemmerOverrideFilterFactory {
            base,
            dictionary: None,
            dictionary_files,
            ignore_case,
        })
    }
}

impl StemmerOverrideFilterFactory {
    // Java: StemmerOverrideFilterFactory.inform
    fn inform_impl(&mut self, loader: &dyn ResourceLoader) -> Result<(), FactoryError> {
        let Some(dictionary_files) = &self.dictionary_files else {
            return Ok(());
        };
        let files = args::split_file_names(Some(dictionary_files));
        if files.is_empty() {
            return Ok(());
        }
        let mut builder = misc::StemmerOverrideBuilder::new(self.ignore_case);
        for file in &files {
            for line in get_lines(loader, java_trim(file))? {
                // line.split("\t", 2): mapping[1] is out of bounds without a tab.
                let Some((word, stem)) = line.split_once('\t') else {
                    return Err(FactoryError::new(
                        JavaException::ArrayIndexOutOfBounds,
                        "Index 1 out of bounds for length 1",
                    ));
                };
                builder.add(word, stem);
            }
        }
        self.dictionary = Some(builder.build());
        Ok(())
    }

    /// `isIgnoreCase()`.
    pub fn is_ignore_case(&self) -> bool {
        self.ignore_case
    }
}

impl TokenFilterFactory for StemmerOverrideFilterFactory {
    fn create(&self, input: Box<dyn TokenStream>) -> Result<Box<dyn TokenStream>, AnalysisError> {
        Ok(match &self.dictionary {
            None => input,
            Some(d) => Box::new(misc::StemmerOverrideFilter::new(input, Arc::clone(d))),
        })
    }
}

// ---------------------------------------------------------- word delimiter

/// `WordDelimiter(Graph)FilterFactory.TYPE_PATTERN`.
const TYPE_PATTERN: &str = "(.*)\\s*=>\\s*(.*)\\s*$";

/// `parseType`: a type name's bits.
fn parse_type(s: &str) -> Option<u8> {
    // WordDelimiterIterator.LOWER, UPPER, ALPHA, DIGIT, ALPHANUM, SUBWORD_DELIM
    Some(match s {
        "LOWER" => 0x01,
        "UPPER" => 0x02,
        "ALPHA" => 0x03,
        "DIGIT" => 0x04,
        "ALPHANUM" => 0x07,
        "SUBWORD_DELIM" => 0x08,
        _ => return None,
    })
}

/// `parseTypes(rules)`: the default types (`WordDelimiterIterator.getType`)
/// of every char up to the highest one a rule names, then the rules.
fn parse_types(rules: &[String]) -> Result<Arc<[u8]>, FactoryError> {
    let pattern = JavaPattern::compile(TYPE_PATTERN)?;
    let mut type_map = std::collections::BTreeMap::new();
    for rule in rules {
        let mut m = JavaMatcher::new(&pattern, rule);
        if !m.find() {
            return Err(FactoryError::illegal_argument(format!(
                "Invalid Mapping Rule : [{rule}]"
            )));
        }
        let lhs = parse_escaped(java_trim(&m.group(1).unwrap_or_default()))?;
        let rhs = parse_type(java_trim(&m.group(2).unwrap_or_default()));
        if lhs.len() != 1 {
            return Err(FactoryError::illegal_argument(format!(
                "Invalid Mapping Rule : [{rule}]. Only a single character is allowed."
            )));
        }
        let Some(rhs) = rhs else {
            return Err(FactoryError::illegal_argument(format!(
                "Invalid Mapping Rule : [{rule}]. Illegal type."
            )));
        };
        type_map.insert(lhs[0], rhs);
    }
    let last = type_map
        .keys()
        .next_back()
        .copied()
        .ok_or_else(|| FactoryError::new(JavaException::NoSuchElement, "null"))?;
    let len = (usize::from(last) + 1).max(misc::DEFAULT_WORD_DELIM_TABLE.len());
    let mut types: Vec<u8> = (0..len)
        .map(|i| misc::word_delimiter_type(i as u32) as u8)
        .collect();
    for (ch, ty) in type_map {
        types[usize::from(ch)] = ty;
    }
    Ok(Arc::from(types))
}

/// The flags both word delimiter factories read, in Java's order.
fn word_delimiter_flags(args: &mut JavaArgs, graph: bool) -> Result<i32, FactoryError> {
    use misc::*;
    let mut flags = 0;
    let mut opts: Vec<(&str, i32, i32)> = vec![
        ("generateWordParts", 1, GENERATE_WORD_PARTS),
        ("generateNumberParts", 1, GENERATE_NUMBER_PARTS),
        ("catenateWords", 0, CATENATE_WORDS),
        ("catenateNumbers", 0, CATENATE_NUMBERS),
        ("catenateAll", 0, CATENATE_ALL),
        ("splitOnCaseChange", 1, SPLIT_ON_CASE_CHANGE),
        ("splitOnNumerics", 1, SPLIT_ON_NUMERICS),
        ("preserveOriginal", 0, PRESERVE_ORIGINAL),
        ("stemEnglishPossessive", 1, STEM_ENGLISH_POSSESSIVE),
    ];
    if graph {
        opts.push(("ignoreKeywords", 0, IGNORE_KEYWORDS));
    }
    for (name, default, flag) in opts {
        if args::get_int(args, name, default)? != 0 {
            flags |= flag;
        }
    }
    Ok(flags)
}

/// `org.apache.lucene.analysis.miscellaneous.WordDelimiterGraphFilterFactory`
/// (`wordDelimiterGraph`) and, with `GRAPH` false, the deprecated
/// `WordDelimiterFilterFactory` (`wordDelimiter`).
pub struct WordDelimiterFactory<const GRAPH: bool> {
    base: FactoryBase,
    word_files: Option<String>,
    types: Option<String>,
    flags: i32,
    adjust_offsets: bool,
    type_table: Option<Arc<[u8]>>,
    protected_words: Option<Arc<CharArraySet>>,
}

/// `org.apache.lucene.analysis.miscellaneous.WordDelimiterGraphFilterFactory`.
pub type WordDelimiterGraphFilterFactory = WordDelimiterFactory<true>;
/// `org.apache.lucene.analysis.miscellaneous.WordDelimiterFilterFactory`.
pub type WordDelimiterFilterFactory = WordDelimiterFactory<false>;

impl<const GRAPH: bool> super::AnalysisFactory for WordDelimiterFactory<GRAPH> {
    fn base(&self) -> &FactoryBase {
        &self.base
    }
    fn base_mut(&mut self) -> &mut FactoryBase {
        &mut self.base
    }
    fn is_resource_loader_aware(&self) -> bool {
        true
    }
    // Java: WordDelimiter(Graph)FilterFactory.inform
    fn inform(&mut self, loader: &dyn ResourceLoader) -> Result<(), FactoryError> {
        if let Some(files) = &self.word_files {
            self.protected_words = get_word_set(loader, files, false)?.map(Arc::new);
        }
        if let Some(types) = &self.types {
            let mut rules = Vec::new();
            for file in args::split_file_names(Some(types)) {
                rules.extend(get_lines(loader, java_trim(&file))?);
            }
            self.type_table = Some(parse_types(&rules)?);
        }
        Ok(())
    }
}

impl<const GRAPH: bool> FactoryClass for WordDelimiterFactory<GRAPH> {
    const NAME: &'static str = if GRAPH {
        "wordDelimiterGraph"
    } else {
        "wordDelimiter"
    };
    const CLASS_NAME: &'static str = if GRAPH {
        "org.apache.lucene.analysis.miscellaneous.WordDelimiterGraphFilterFactory"
    } else {
        "org.apache.lucene.analysis.miscellaneous.WordDelimiterFilterFactory"
    };
    fn from_args(args: &mut JavaArgs) -> Result<Self, FactoryError> {
        let base = FactoryBase::new(Self::CLASS_NAME, args)?;
        let flags = word_delimiter_flags(args, GRAPH)?;
        let word_files = args::get(args, "protected");
        let types = args::get(args, "types");
        let adjust_offsets = GRAPH && args::get_boolean(args, "adjustOffsets", true);
        args::reject_unknown(args)?;
        Ok(WordDelimiterFactory {
            base,
            word_files,
            types,
            flags,
            adjust_offsets,
            type_table: None,
            protected_words: None,
        })
    }
}

impl<const GRAPH: bool> TokenFilterFactory for WordDelimiterFactory<GRAPH> {
    fn create(&self, input: Box<dyn TokenStream>) -> Result<Box<dyn TokenStream>, AnalysisError> {
        let table = self
            .type_table
            .clone()
            .unwrap_or_else(|| Arc::from(&misc::DEFAULT_WORD_DELIM_TABLE[..]));
        let prot = self.protected_words.clone();
        Ok(if GRAPH {
            Box::new(misc::WordDelimiterGraphFilter::with_table(
                input,
                self.adjust_offsets,
                table,
                self.flags,
                prot,
            )?)
        } else {
            Box::new(misc::WordDelimiterFilter::with_table(
                input, table, self.flags, prot,
            ))
        })
    }
}

// ------------------------------------------------------------ pattern typing

factory_struct! {
    /// `org.apache.lucene.analysis.pattern.PatternTypingFilterFactory`
    /// (`patternTyping`): `flags pattern ::: typeTemplate` lines.
    PatternTypingFilterFactory { pattern_file: String, rules: Option<Vec<crate::pattern::PatternTypingRule>> }
}
analysis_factory!(PatternTypingFilterFactory, aware);

impl FactoryClass for PatternTypingFilterFactory {
    const NAME: &'static str = "patternTyping";
    const CLASS_NAME: &'static str =
        "org.apache.lucene.analysis.pattern.PatternTypingFilterFactory";
    fn from_args(args: &mut JavaArgs) -> Result<Self, FactoryError> {
        let base = FactoryBase::new(Self::CLASS_NAME, args)?;
        let pattern_file = args::require(args, "patternFile")?;
        args::reject_unknown(args)?;
        Ok(PatternTypingFilterFactory {
            base,
            pattern_file,
            rules: None,
        })
    }
}

impl PatternTypingFilterFactory {
    // Java: PatternTypingFilterFactory.inform
    fn inform_impl(&mut self, loader: &dyn ResourceLoader) -> Result<(), FactoryError> {
        let mut rules = Vec::new();
        for line in get_lines(loader, &self.pattern_file)? {
            let Some(first_space) = line.find(' ') else {
                let len = line.encode_utf16().count();
                return Err(FactoryError::new(
                    JavaException::StringIndexOutOfBounds,
                    format!("begin 0, end -1, length {len}"),
                ));
            };
            let flags = args::parse_java_int(&line[..first_space])?;
            let rest = &line[first_space + 1..];
            let split = java_split(rest, " ::: ");
            if split.len() != 2 {
                return Err(FactoryError::new(
                    JavaException::Runtime,
                    "The PatternTypingFilter: Always two there are, no more, no less, a pattern and a replacement (separated by ' ::: ' )",
                ));
            }
            rules.push(crate::pattern::PatternTypingRule {
                pattern: compile_pattern(split[0])?,
                flags,
                type_template: split[1].to_string(),
            });
        }
        self.rules = Some(rules);
        Ok(())
    }
}

impl TokenFilterFactory for PatternTypingFilterFactory {
    fn create(&self, input: Box<dyn TokenStream>) -> Result<Box<dyn TokenStream>, AnalysisError> {
        let rules = self
            .rules
            .clone()
            .ok_or_else(|| not_informed(Self::CLASS_NAME))?;
        Ok(Box::new(crate::pattern::PatternTypingFilter::new(
            input, rules,
        )))
    }
}

// --------------------------------------------------------- delimited payload

/// A `PayloadEncoder` named in configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Encoder {
    Float,
    Integer,
    Identity,
}

factory_struct! {
    /// `org.apache.lucene.analysis.payloads.DelimitedPayloadTokenFilterFactory`
    /// (`delimitedPayload`): `encoder` `float`, `integer`, `identity` or a
    /// `PayloadEncoder` class name (analysis-common's three).
    DelimitedPayloadTokenFilterFactory { encoder_class: String, delimiter: u16, encoder: Option<Encoder> }
}
analysis_factory!(DelimitedPayloadTokenFilterFactory, aware);

impl FactoryClass for DelimitedPayloadTokenFilterFactory {
    const NAME: &'static str = "delimitedPayload";
    const CLASS_NAME: &'static str =
        "org.apache.lucene.analysis.payloads.DelimitedPayloadTokenFilterFactory";
    fn from_args(args: &mut JavaArgs) -> Result<Self, FactoryError> {
        let base = FactoryBase::new(Self::CLASS_NAME, args)?;
        let encoder_class = args::require(args, "encoder")?;
        let delimiter = args::get_char(args, "delimiter", u16::from(b'|'))?;
        args::reject_unknown(args)?;
        Ok(DelimitedPayloadTokenFilterFactory {
            base,
            encoder_class,
            delimiter,
            encoder: None,
        })
    }
}

impl DelimitedPayloadTokenFilterFactory {
    // Java: DelimitedPayloadTokenFilterFactory.inform
    fn inform_impl(&mut self, _loader: &dyn ResourceLoader) -> Result<(), FactoryError> {
        self.encoder = Some(match self.encoder_class.as_str() {
            "float" | "org.apache.lucene.analysis.payloads.FloatEncoder" => Encoder::Float,
            "integer" | "org.apache.lucene.analysis.payloads.IntegerEncoder" => Encoder::Integer,
            "identity" | "org.apache.lucene.analysis.payloads.IdentityEncoder" => Encoder::Identity,
            "org.apache.lucene.analysis.payloads.AbstractEncoder" => {
                return Err(FactoryError::new(
                    JavaException::Runtime,
                    format!("Cannot create instance: {}", self.encoder_class),
                ))
            }
            other => return Err(spi::cannot_load_class(other)),
        });
        Ok(())
    }
}

impl TokenFilterFactory for DelimitedPayloadTokenFilterFactory {
    fn create(&self, input: Box<dyn TokenStream>) -> Result<Box<dyn TokenStream>, AnalysisError> {
        use crate::payloads::{
            DelimitedPayloadTokenFilter as F, FloatEncoder, IdentityEncoder, IntegerEncoder,
        };
        let d = self.delimiter;
        Ok(match self.encoder {
            None => return Err(not_informed(Self::CLASS_NAME)),
            Some(Encoder::Float) => Box::new(F::new(input, d, FloatEncoder)),
            Some(Encoder::Integer) => Box::new(F::new(input, d, IntegerEncoder)),
            Some(Encoder::Identity) => Box::new(F::new(input, d, IdentityEncoder)),
        })
    }
}

// ----------------------------------------------------------------- snowball

factory_struct! {
    /// `org.apache.lucene.analysis.snowball.SnowballPorterFilterFactory`
    /// (`snowballPorter`): `language` (a stemmer class name without
    /// `Stemmer`, `German2` read as `German`), `protected` files.
    SnowballPorterFilterFactory {
        language: String,
        word_files: Option<String>,
        stemmer: Option<&'static str>,
        protected_words: Option<Arc<CharArraySet>>,
    }
}
analysis_factory!(SnowballPorterFilterFactory, aware);

impl FactoryClass for SnowballPorterFilterFactory {
    const NAME: &'static str = "snowballPorter";
    const CLASS_NAME: &'static str =
        "org.apache.lucene.analysis.snowball.SnowballPorterFilterFactory";
    fn from_args(args: &mut JavaArgs) -> Result<Self, FactoryError> {
        let base = FactoryBase::new(Self::CLASS_NAME, args)?;
        let mut language = args::get_or(args, "language", "English");
        if language == "German2" {
            language = "German".into();
        }
        let word_files = args::get(args, "protected");
        args::reject_unknown(args)?;
        Ok(SnowballPorterFilterFactory {
            base,
            language,
            word_files,
            stemmer: None,
            protected_words: None,
        })
    }
}

impl SnowballPorterFilterFactory {
    // Java: SnowballPorterFilterFactory.inform
    fn inform_impl(&mut self, loader: &dyn ResourceLoader) -> Result<(), FactoryError> {
        let class_name = format!("org.tartarus.snowball.ext.{}Stemmer", self.language);
        let stemmer = crate::snowball::SnowballStemmer::for_name(&self.language)
            .ok_or_else(|| spi::cannot_load_class(&class_name))?;
        self.stemmer = Some(stemmer.name());
        if let Some(files) = &self.word_files {
            self.protected_words = get_word_set(loader, files, false)?.map(Arc::new);
        }
        Ok(())
    }
}

impl TokenFilterFactory for SnowballPorterFilterFactory {
    fn create(
        &self,
        mut input: Box<dyn TokenStream>,
    ) -> Result<Box<dyn TokenStream>, AnalysisError> {
        let stemmer = self
            .stemmer
            .and_then(crate::snowball::SnowballStemmer::for_name)
            .ok_or_else(|| not_informed(Self::CLASS_NAME))?;
        if let Some(w) = &self.protected_words {
            input = Box::new(misc::SetKeywordMarkerFilter::new(input, Arc::clone(w)));
        }
        Ok(Box::new(crate::snowball::SnowballFilter::new(
            input, stemmer,
        )))
    }
}

// ------------------------------------------------------------------ elision

factory_struct! {
    /// `org.apache.lucene.analysis.util.ElisionFilterFactory` (`elision`):
    /// `articles` files, else French's default articles.
    ElisionFilterFactory { articles_file: Option<String>, ignore_case: bool, articles: Option<Arc<CharArraySet>> }
}
analysis_factory!(ElisionFilterFactory, aware);

impl FactoryClass for ElisionFilterFactory {
    const NAME: &'static str = "elision";
    const CLASS_NAME: &'static str = "org.apache.lucene.analysis.util.ElisionFilterFactory";
    fn from_args(args: &mut JavaArgs) -> Result<Self, FactoryError> {
        let base = FactoryBase::new(Self::CLASS_NAME, args)?;
        let articles_file = args::get(args, "articles");
        let ignore_case = args::get_boolean(args, "ignoreCase", false);
        args::reject_unknown(args)?;
        Ok(ElisionFilterFactory {
            base,
            articles_file,
            ignore_case,
            articles: None,
        })
    }
}

impl ElisionFilterFactory {
    // Java: ElisionFilterFactory.inform
    fn inform_impl(&mut self, loader: &dyn ResourceLoader) -> Result<(), FactoryError> {
        self.articles = match &self.articles_file {
            None => Some(Arc::clone(&crate::lang::fr::DEFAULT_ARTICLES)),
            Some(files) => get_word_set(loader, files, self.ignore_case)?.map(Arc::new),
        };
        Ok(())
    }

    fn build(&self, input: Box<dyn TokenStream>) -> Result<Box<dyn TokenStream>, AnalysisError> {
        let articles = self
            .articles
            .clone()
            .ok_or_else(|| not_informed(Self::CLASS_NAME))?;
        Ok(Box::new(crate::util::ElisionFilter::new(input, articles)))
    }
}

impl TokenFilterFactory for ElisionFilterFactory {
    fn create(&self, input: Box<dyn TokenStream>) -> Result<Box<dyn TokenStream>, AnalysisError> {
        self.build(input)
    }

    fn normalize(&self, input: Box<dyn TokenStream>) -> Box<dyn TokenStream> {
        match &self.articles {
            Some(a) => Box::new(crate::util::ElisionFilter::new(input, Arc::clone(a))),
            None => input,
        }
    }
}

// ------------------------------------------------------------------ synonyms

/// The analyzer a synonym factory parses its rules with: the
/// `tokenizerFactory`'s tokenizer (else a `WhitespaceTokenizer`), lowercased
/// under `ignoreCase`.
struct SynonymRuleAnalyzer {
    tokenizer: Option<Arc<dyn TokenizerFactory>>,
    ignore_case: bool,
}

impl AnalyzerDefinition for SynonymRuleAnalyzer {
    fn create_components(&self, _field: &str) -> Result<TokenStreamComponents, AnalysisError> {
        let tokenizer: Box<dyn TokenStream> = match &self.tokenizer {
            Some(f) => f.create()?,
            None => Box::new(crate::util::WhitespaceTokenizer::new()),
        };
        Ok(if self.ignore_case {
            TokenStreamComponents::new(crate::LowerCaseFilter::new(tokenizer))
        } else {
            TokenStreamComponents::new(tokenizer)
        })
    }
}

/// `SynonymFilterFactory` (`synonym`, deprecated) and
/// `SynonymGraphFilterFactory` (`synonymGraph`): `synonyms` files,
/// `format` (`solr`, `wordnet` or a parser class name), `ignoreCase`,
/// `expand`, and either `analyzer` (an analyzer class name) or
/// `tokenizerFactory` (a tokenizer factory class name, whose arguments are
/// every other argument, `tokenizerFactory.` prefix stripped).
pub struct SynonymFactory<const GRAPH: bool> {
    base: FactoryBase,
    ignore_case: bool,
    tokenizer_factory: Option<String>,
    synonyms: String,
    format: Option<String>,
    expand: bool,
    analyzer_name: Option<String>,
    tok_args: JavaArgs,
    map: Option<Arc<SynonymMap>>,
}

/// `org.apache.lucene.analysis.synonym.SynonymGraphFilterFactory`.
pub type SynonymGraphFilterFactory = SynonymFactory<true>;
/// `org.apache.lucene.analysis.synonym.SynonymFilterFactory`.
pub type SynonymFilterFactory = SynonymFactory<false>;

impl<const GRAPH: bool> super::AnalysisFactory for SynonymFactory<GRAPH> {
    fn base(&self) -> &FactoryBase {
        &self.base
    }
    fn base_mut(&mut self) -> &mut FactoryBase {
        &mut self.base
    }
    fn is_resource_loader_aware(&self) -> bool {
        true
    }
    fn inform(&mut self, loader: &dyn ResourceLoader) -> Result<(), FactoryError> {
        self.inform_impl(loader)
    }
}

impl<const GRAPH: bool> FactoryClass for SynonymFactory<GRAPH> {
    const NAME: &'static str = if GRAPH { "synonymGraph" } else { "synonym" };
    const CLASS_NAME: &'static str = if GRAPH {
        "org.apache.lucene.analysis.synonym.SynonymGraphFilterFactory"
    } else {
        "org.apache.lucene.analysis.synonym.SynonymFilterFactory"
    };
    fn from_args(args: &mut JavaArgs) -> Result<Self, FactoryError> {
        let base = FactoryBase::new(Self::CLASS_NAME, args)?;
        let ignore_case = args::get_boolean(args, "ignoreCase", false);
        let synonyms = args::require(args, "synonyms")?;
        let format = args::get(args, "format");
        let expand = args::get_boolean(args, "expand", true);
        let analyzer_name = args::get(args, "analyzer");
        let tokenizer_factory = args::get(args, "tokenizerFactory");
        if let (Some(a), Some(t)) = (&analyzer_name, &tokenizer_factory) {
            return Err(FactoryError::illegal_argument(format!(
                "Analyzer and TokenizerFactory can't be specified both: {a} and {t}"
            )));
        }
        let mut tok_args = JavaArgs::new();
        if tokenizer_factory.is_some() {
            tok_args.put(
                super::LUCENE_MATCH_VERSION_PARAM,
                &base.lucene_match_version().to_string(),
            );
            for key in args.keys() {
                let value = args.remove(&key).unwrap_or_default();
                let stripped = key.strip_prefix("tokenizerFactory.").unwrap_or(&key);
                tok_args.put(stripped, &value);
            }
        }
        args::reject_unknown(args)?;
        Ok(SynonymFactory {
            base,
            ignore_case,
            tokenizer_factory,
            synonyms,
            format,
            expand,
            analyzer_name,
            tok_args,
            map: None,
        })
    }
}

/// `new RuntimeException(cause)`: the cause's `toString()`.
pub(crate) fn wrapped_runtime(cause: &FactoryError) -> FactoryError {
    FactoryError::new(
        JavaException::Runtime,
        format!("{}: {}", cause.kind.qualified_name(), cause.message),
    )
}

impl<const GRAPH: bool> SynonymFactory<GRAPH> {
    // Java: SynonymFilterFactory.loadTokenizerFactory
    fn load_tokenizer_factory(
        &self,
        loader: &dyn ResourceLoader,
        class_name: &str,
    ) -> Result<Arc<dyn TokenizerFactory>, FactoryError> {
        let ctor =
            spi::tokenizer_class(class_name).ok_or_else(|| spi::cannot_load_class(class_name))?;
        let mut tok_args = self.tok_args.clone();
        // clazz.getConstructor(Map.class).newInstance(tokArgs): a constructor's
        // exception arrives wrapped in an InvocationTargetException.
        let mut factory = ctor(&mut tok_args).map_err(|_| {
            FactoryError::new(
                JavaException::Runtime,
                "java.lang.reflect.InvocationTargetException",
            )
        })?;
        factory.inform(loader).map_err(|e| wrapped_runtime(&e))?;
        Ok(Arc::from(factory))
    }

    // Java: SynonymFilterFactory.inform
    fn inform_impl(&mut self, loader: &dyn ResourceLoader) -> Result<(), FactoryError> {
        let tokenizer = match &self.tokenizer_factory {
            Some(class_name) => Some(self.load_tokenizer_factory(loader, class_name)?),
            None => None,
        };
        let analyzer = match &self.analyzer_name {
            Some(class_name) => spi::new_analyzer(class_name)?,
            None => Analyzer::new(SynonymRuleAnalyzer {
                tokenizer,
                ignore_case: self.ignore_case,
            }),
        };
        let solr = match self.format.as_deref() {
            None | Some("solr") | Some("org.apache.lucene.analysis.synonym.SolrSynonymParser") => {
                true
            }
            Some("wordnet") | Some("org.apache.lucene.analysis.synonym.WordnetSynonymParser") => {
                false
            }
            Some(other) => return Err(spi::cannot_load_class(other)),
        };
        let parse_error = |e: SynonymParseError| match e {
            SynonymParseError::InvalidRule { .. } => {
                FactoryError::io("Error parsing synonyms file:")
            }
            SynonymParseError::Malformed { line } => FactoryError::new(
                JavaException::StringIndexOutOfBounds,
                format!("malformed WordNet line {line}"),
            ),
        };
        // loadSynonyms: each file decoded as strict UTF-8 and parsed.
        let mut texts = Vec::new();
        for file in args::split_file_names(Some(&self.synonyms)) {
            texts.push(decode_utf8(loader.open_resource(&file)?)?);
        }
        let map = if solr {
            let mut parser = SolrSynonymParser::new(true, self.expand, &analyzer);
            for text in &texts {
                parser.parse(text).map_err(parse_error)?;
            }
            parser.build()?
        } else {
            let mut parser = WordnetSynonymParser::new(true, self.expand, &analyzer);
            for text in &texts {
                parser.parse(text).map_err(parse_error)?;
            }
            parser.build()?
        };
        // Java: a map built from no rules has a null FST and the factory
        // returns its input.
        self.map = (!map.is_empty()).then(|| Arc::new(map));
        Ok(())
    }
}

impl<const GRAPH: bool> TokenFilterFactory for SynonymFactory<GRAPH> {
    fn create(&self, input: Box<dyn TokenStream>) -> Result<Box<dyn TokenStream>, AnalysisError> {
        let Some(map) = &self.map else {
            return Ok(input);
        };
        let map = Arc::clone(map);
        Ok(if GRAPH {
            Box::new(crate::synonym::SynonymGraphFilter::new(
                input,
                map,
                self.ignore_case,
            ))
        } else {
            Box::new(crate::synonym::SynonymFilter::new(
                input,
                map,
                self.ignore_case,
            ))
        })
    }
}
