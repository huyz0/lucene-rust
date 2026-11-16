//! The tokenizer factories: `StandardTokenizerFactory` (lucene-core) and
//! analysis-common's twelve, plus `ThaiTokenizerFactory`, whose tokenizer is
//! not supported (it parses its arguments as Java does; `create` is
//! `ThaiTokenizer`'s `UnsupportedOperationException`, see [`crate::lang::th`]).

use std::collections::HashSet;

use lucene_util::automaton::{operations, Automaton, RegExp, DEFAULT_DETERMINIZE_WORK_LIMIT};

use super::args::{self, JavaArgs};
use super::{
    analysis_factory, factory_struct, FactoryBase, FactoryClass, FactoryError, TokenizerFactory,
};
use crate::token_stream::TokenStream;
use crate::util::{JavaPattern, LetterTokenizer, UnicodeWhitespaceTokenizer, WhitespaceTokenizer};
use crate::{AnalysisError, KeywordTokenizer, StandardTokenizer};

/// `StandardTokenizer.MAX_TOKEN_LENGTH_LIMIT`.
const MAX_TOKEN_LENGTH_LIMIT: i32 = 1024 * 1024;
/// `StandardAnalyzer.DEFAULT_MAX_TOKEN_LENGTH`.
const DEFAULT_MAX_TOKEN_LENGTH: i32 = 255;
/// `KeywordTokenizer.DEFAULT_BUFFER_SIZE`.
const KEYWORD_DEFAULT_BUFFER_SIZE: i32 = 256;
/// `CharTokenizer.DEFAULT_MAX_WORD_LEN`.
const CHAR_TOKENIZER_DEFAULT_MAX_WORD_LEN: i32 = 255;

/// The `maxTokenLen` check of the keyword, letter and whitespace factories.
fn check_max_token_len(max_token_len: i32) -> Result<(), FactoryError> {
    if max_token_len > MAX_TOKEN_LENGTH_LIMIT || max_token_len <= 0 {
        return Err(FactoryError::illegal_argument(format!(
            "maxTokenLen must be greater than 0 and less than {MAX_TOKEN_LENGTH_LIMIT} passed: {max_token_len}"
        )));
    }
    Ok(())
}

factory_struct! {
    /// `org.apache.lucene.analysis.standard.StandardTokenizerFactory` (`standard`).
    StandardTokenizerFactory { max_token_length: i32 }
}
analysis_factory!(StandardTokenizerFactory);

impl FactoryClass for StandardTokenizerFactory {
    const NAME: &'static str = "standard";
    const CLASS_NAME: &'static str = "org.apache.lucene.analysis.standard.StandardTokenizerFactory";
    fn from_args(args: &mut JavaArgs) -> Result<Self, FactoryError> {
        let base = FactoryBase::new(Self::CLASS_NAME, args)?;
        let max_token_length = args::get_int(args, "maxTokenLength", DEFAULT_MAX_TOKEN_LENGTH)?;
        args::reject_unknown(args)?;
        Ok(StandardTokenizerFactory {
            base,
            max_token_length,
        })
    }
}

impl TokenizerFactory for StandardTokenizerFactory {
    fn create(&self) -> Result<Box<dyn TokenStream>, AnalysisError> {
        let mut t = StandardTokenizer::new();
        t.set_max_token_length(self.max_token_length)?;
        Ok(Box::new(t))
    }
}

factory_struct! {
    /// `org.apache.lucene.analysis.classic.ClassicTokenizerFactory` (`classic`).
    ClassicTokenizerFactory { max_token_length: i32 }
}
analysis_factory!(ClassicTokenizerFactory);

impl FactoryClass for ClassicTokenizerFactory {
    const NAME: &'static str = "classic";
    const CLASS_NAME: &'static str = "org.apache.lucene.analysis.classic.ClassicTokenizerFactory";
    fn from_args(args: &mut JavaArgs) -> Result<Self, FactoryError> {
        let base = FactoryBase::new(Self::CLASS_NAME, args)?;
        let max_token_length = args::get_int(args, "maxTokenLength", DEFAULT_MAX_TOKEN_LENGTH)?;
        args::reject_unknown(args)?;
        Ok(ClassicTokenizerFactory {
            base,
            max_token_length,
        })
    }
}

impl TokenizerFactory for ClassicTokenizerFactory {
    fn create(&self) -> Result<Box<dyn TokenStream>, AnalysisError> {
        let mut t = crate::classic::ClassicTokenizer::new();
        t.set_max_token_length(self.max_token_length)?;
        Ok(Box::new(t))
    }
}

factory_struct! {
    /// `org.apache.lucene.analysis.email.UAX29URLEmailTokenizerFactory` (`uax29UrlEmail`).
    UAX29URLEmailTokenizerFactory { max_token_length: i32 }
}
analysis_factory!(UAX29URLEmailTokenizerFactory);

impl FactoryClass for UAX29URLEmailTokenizerFactory {
    const NAME: &'static str = "uax29UrlEmail";
    const CLASS_NAME: &'static str =
        "org.apache.lucene.analysis.email.UAX29URLEmailTokenizerFactory";
    fn from_args(args: &mut JavaArgs) -> Result<Self, FactoryError> {
        let base = FactoryBase::new(Self::CLASS_NAME, args)?;
        let max_token_length = args::get_int(args, "maxTokenLength", DEFAULT_MAX_TOKEN_LENGTH)?;
        args::reject_unknown(args)?;
        Ok(UAX29URLEmailTokenizerFactory {
            base,
            max_token_length,
        })
    }
}

impl TokenizerFactory for UAX29URLEmailTokenizerFactory {
    fn create(&self) -> Result<Box<dyn TokenStream>, AnalysisError> {
        let mut t = crate::email::UAX29URLEmailTokenizer::new();
        t.set_max_token_length(self.max_token_length)?;
        Ok(Box::new(t))
    }
}

factory_struct! {
    /// `org.apache.lucene.analysis.core.KeywordTokenizerFactory` (`keyword`).
    /// `maxTokenLen` is only validated: Java's is the initial buffer size,
    /// and the token is the whole input either way.
    KeywordTokenizerFactory { max_token_len: i32 }
}
analysis_factory!(KeywordTokenizerFactory);

impl FactoryClass for KeywordTokenizerFactory {
    const NAME: &'static str = "keyword";
    const CLASS_NAME: &'static str = "org.apache.lucene.analysis.core.KeywordTokenizerFactory";
    fn from_args(args: &mut JavaArgs) -> Result<Self, FactoryError> {
        let base = FactoryBase::new(Self::CLASS_NAME, args)?;
        let max_token_len = args::get_int(args, "maxTokenLen", KEYWORD_DEFAULT_BUFFER_SIZE)?;
        check_max_token_len(max_token_len)?;
        args::reject_unknown(args)?;
        Ok(KeywordTokenizerFactory {
            base,
            max_token_len,
        })
    }
}

impl KeywordTokenizerFactory {
    /// The validated `maxTokenLen`.
    pub fn max_token_len(&self) -> i32 {
        self.max_token_len
    }
}

impl TokenizerFactory for KeywordTokenizerFactory {
    fn create(&self) -> Result<Box<dyn TokenStream>, AnalysisError> {
        Ok(Box::new(KeywordTokenizer::new()))
    }
}

factory_struct! {
    /// `org.apache.lucene.analysis.core.LetterTokenizerFactory` (`letter`).
    LetterTokenizerFactory { max_token_len: i32 }
}
analysis_factory!(LetterTokenizerFactory);

impl FactoryClass for LetterTokenizerFactory {
    const NAME: &'static str = "letter";
    const CLASS_NAME: &'static str = "org.apache.lucene.analysis.core.LetterTokenizerFactory";
    fn from_args(args: &mut JavaArgs) -> Result<Self, FactoryError> {
        let base = FactoryBase::new(Self::CLASS_NAME, args)?;
        let max_token_len =
            args::get_int(args, "maxTokenLen", CHAR_TOKENIZER_DEFAULT_MAX_WORD_LEN)?;
        check_max_token_len(max_token_len)?;
        args::reject_unknown(args)?;
        Ok(LetterTokenizerFactory {
            base,
            max_token_len,
        })
    }
}

impl TokenizerFactory for LetterTokenizerFactory {
    fn create(&self) -> Result<Box<dyn TokenStream>, AnalysisError> {
        // check_max_token_len made it positive.
        let len = usize::try_from(self.max_token_len).unwrap_or(1);
        Ok(Box::new(LetterTokenizer::with_max_token_len(len)?))
    }
}

/// `WhitespaceTokenizerFactory.RULE_NAMES`.
const RULE_NAMES: [&str; 2] = ["java", "unicode"];

factory_struct! {
    /// `org.apache.lucene.analysis.core.WhitespaceTokenizerFactory` (`whitespace`):
    /// `rule` `java` (`Character.isWhitespace`) or `unicode`.
    WhitespaceTokenizerFactory { unicode: bool, max_token_len: i32 }
}
analysis_factory!(WhitespaceTokenizerFactory);

impl FactoryClass for WhitespaceTokenizerFactory {
    const NAME: &'static str = "whitespace";
    const CLASS_NAME: &'static str = "org.apache.lucene.analysis.core.WhitespaceTokenizerFactory";
    fn from_args(args: &mut JavaArgs) -> Result<Self, FactoryError> {
        let base = FactoryBase::new(Self::CLASS_NAME, args)?;
        let rule = args::get_one_of(args, "rule", &RULE_NAMES, Some("java"), true)?;
        let max_token_len =
            args::get_int(args, "maxTokenLen", CHAR_TOKENIZER_DEFAULT_MAX_WORD_LEN)?;
        check_max_token_len(max_token_len)?;
        args::reject_unknown(args)?;
        Ok(WhitespaceTokenizerFactory {
            base,
            unicode: rule.as_deref() == Some("unicode"),
            max_token_len,
        })
    }
}

impl TokenizerFactory for WhitespaceTokenizerFactory {
    fn create(&self) -> Result<Box<dyn TokenStream>, AnalysisError> {
        let len = usize::try_from(self.max_token_len).unwrap_or(1);
        Ok(if self.unicode {
            Box::new(UnicodeWhitespaceTokenizer::with_max_token_len(len)?)
        } else {
            Box::new(WhitespaceTokenizer::with_max_token_len(len)?)
        })
    }
}

factory_struct! {
    /// `org.apache.lucene.analysis.ngram.EdgeNGramTokenizerFactory` (`edgeNGram`).
    EdgeNGramTokenizerFactory { min_gram_size: i32, max_gram_size: i32 }
}
analysis_factory!(EdgeNGramTokenizerFactory);

impl FactoryClass for EdgeNGramTokenizerFactory {
    const NAME: &'static str = "edgeNGram";
    const CLASS_NAME: &'static str = "org.apache.lucene.analysis.ngram.EdgeNGramTokenizerFactory";
    fn from_args(args: &mut JavaArgs) -> Result<Self, FactoryError> {
        let base = FactoryBase::new(Self::CLASS_NAME, args)?;
        // EdgeNGramTokenizer.DEFAULT_MIN_GRAM_SIZE / DEFAULT_MAX_GRAM_SIZE
        let min_gram_size = args::get_int(args, "minGramSize", 1)?;
        let max_gram_size = args::get_int(args, "maxGramSize", 1)?;
        args::reject_unknown(args)?;
        Ok(EdgeNGramTokenizerFactory {
            base,
            min_gram_size,
            max_gram_size,
        })
    }
}

impl TokenizerFactory for EdgeNGramTokenizerFactory {
    fn create(&self) -> Result<Box<dyn TokenStream>, AnalysisError> {
        Ok(Box::new(crate::ngram::NGramTokenizer::edge(
            self.min_gram_size,
            self.max_gram_size,
        )?))
    }
}

factory_struct! {
    /// `org.apache.lucene.analysis.ngram.NGramTokenizerFactory` (`nGram`).
    NGramTokenizerFactory { min_gram_size: i32, max_gram_size: i32 }
}
analysis_factory!(NGramTokenizerFactory);

impl FactoryClass for NGramTokenizerFactory {
    const NAME: &'static str = "nGram";
    const CLASS_NAME: &'static str = "org.apache.lucene.analysis.ngram.NGramTokenizerFactory";
    fn from_args(args: &mut JavaArgs) -> Result<Self, FactoryError> {
        let base = FactoryBase::new(Self::CLASS_NAME, args)?;
        let min_gram_size =
            args::get_int(args, "minGramSize", crate::ngram::DEFAULT_MIN_NGRAM_SIZE)?;
        let max_gram_size =
            args::get_int(args, "maxGramSize", crate::ngram::DEFAULT_MAX_NGRAM_SIZE)?;
        args::reject_unknown(args)?;
        Ok(NGramTokenizerFactory {
            base,
            min_gram_size,
            max_gram_size,
        })
    }
}

impl TokenizerFactory for NGramTokenizerFactory {
    fn create(&self) -> Result<Box<dyn TokenStream>, AnalysisError> {
        Ok(Box::new(crate::ngram::NGramTokenizer::new(
            self.min_gram_size,
            self.max_gram_size,
        )?))
    }
}

factory_struct! {
    /// `org.apache.lucene.analysis.path.PathHierarchyTokenizerFactory` (`pathHierarchy`).
    PathHierarchyTokenizerFactory { delimiter: char, replacement: char, reverse: bool, skip: i32 }
}
analysis_factory!(PathHierarchyTokenizerFactory);

impl FactoryClass for PathHierarchyTokenizerFactory {
    const NAME: &'static str = "pathHierarchy";
    const CLASS_NAME: &'static str =
        "org.apache.lucene.analysis.path.PathHierarchyTokenizerFactory";
    fn from_args(args: &mut JavaArgs) -> Result<Self, FactoryError> {
        let base = FactoryBase::new(Self::CLASS_NAME, args)?;
        let delimiter = args::get_char(args, "delimiter", crate::path::DEFAULT_DELIMITER as u16)?;
        let replacement = args::get_char(args, "replace", delimiter)?;
        let reverse = args::get_boolean(args, "reverse", false);
        let skip = args::get_int(args, "skip", crate::path::DEFAULT_SKIP)?;
        args::reject_unknown(args)?;
        Ok(PathHierarchyTokenizerFactory {
            base,
            delimiter: args::unit_char(delimiter),
            replacement: args::unit_char(replacement),
            reverse,
            skip,
        })
    }
}

impl TokenizerFactory for PathHierarchyTokenizerFactory {
    fn create(&self) -> Result<Box<dyn TokenStream>, AnalysisError> {
        Ok(if self.reverse {
            Box::new(crate::path::ReversePathHierarchyTokenizer::new(
                self.delimiter,
                self.replacement,
                self.skip,
            )?)
        } else {
            Box::new(crate::path::PathHierarchyTokenizer::new(
                self.delimiter,
                self.replacement,
                self.skip,
            )?)
        })
    }
}

factory_struct! {
    /// `org.apache.lucene.analysis.pattern.PatternTokenizerFactory` (`pattern`).
    PatternTokenizerFactory { pattern: JavaPattern, group: i32 }
}
analysis_factory!(PatternTokenizerFactory);

impl FactoryClass for PatternTokenizerFactory {
    const NAME: &'static str = "pattern";
    const CLASS_NAME: &'static str = "org.apache.lucene.analysis.pattern.PatternTokenizerFactory";
    fn from_args(args: &mut JavaArgs) -> Result<Self, FactoryError> {
        let base = FactoryBase::new(Self::CLASS_NAME, args)?;
        let pattern = args::get_pattern(args, "pattern", "PatternTokenizerFactory")?;
        let group = args::get_int(args, "group", -1)?;
        args::reject_unknown(args)?;
        Ok(PatternTokenizerFactory {
            base,
            pattern,
            group,
        })
    }
}

impl TokenizerFactory for PatternTokenizerFactory {
    fn create(&self) -> Result<Box<dyn TokenStream>, AnalysisError> {
        Ok(Box::new(crate::pattern::PatternTokenizer::new(
            &self.pattern,
            self.group,
        )?))
    }
}

/// `Operations.determinize(new RegExp(pattern).toAutomaton(), limit)`.
fn simple_pattern_dfa(args: &mut JavaArgs) -> Result<Automaton, FactoryError> {
    let limit = args::get_int(args, "determinizeWorkLimit", DEFAULT_DETERMINIZE_WORK_LIMIT)?;
    let pattern = args::require(args, "pattern")?;
    let a = RegExp::new(&pattern)?.to_automaton()?;
    Ok(operations::determinize(&a, limit).map_err(lucene_util::automaton::AutomatonError::from)?)
}

factory_struct! {
    /// `org.apache.lucene.analysis.pattern.SimplePatternTokenizerFactory` (`simplePattern`).
    SimplePatternTokenizerFactory { dfa: Automaton }
}
analysis_factory!(SimplePatternTokenizerFactory);

impl FactoryClass for SimplePatternTokenizerFactory {
    const NAME: &'static str = "simplePattern";
    const CLASS_NAME: &'static str =
        "org.apache.lucene.analysis.pattern.SimplePatternTokenizerFactory";
    fn from_args(args: &mut JavaArgs) -> Result<Self, FactoryError> {
        let base = FactoryBase::new(Self::CLASS_NAME, args)?;
        let dfa = simple_pattern_dfa(args)?;
        args::reject_unknown(args)?;
        Ok(SimplePatternTokenizerFactory { base, dfa })
    }
}

impl TokenizerFactory for SimplePatternTokenizerFactory {
    fn create(&self) -> Result<Box<dyn TokenStream>, AnalysisError> {
        Ok(Box::new(
            crate::pattern::SimplePatternTokenizer::from_automaton(&self.dfa)?,
        ))
    }
}

factory_struct! {
    /// `org.apache.lucene.analysis.pattern.SimplePatternSplitTokenizerFactory` (`simplePatternSplit`).
    SimplePatternSplitTokenizerFactory { dfa: Automaton }
}
analysis_factory!(SimplePatternSplitTokenizerFactory);

impl FactoryClass for SimplePatternSplitTokenizerFactory {
    const NAME: &'static str = "simplePatternSplit";
    const CLASS_NAME: &'static str =
        "org.apache.lucene.analysis.pattern.SimplePatternSplitTokenizerFactory";
    fn from_args(args: &mut JavaArgs) -> Result<Self, FactoryError> {
        let base = FactoryBase::new(Self::CLASS_NAME, args)?;
        let dfa = simple_pattern_dfa(args)?;
        args::reject_unknown(args)?;
        Ok(SimplePatternSplitTokenizerFactory { base, dfa })
    }
}

impl TokenizerFactory for SimplePatternSplitTokenizerFactory {
    fn create(&self) -> Result<Box<dyn TokenStream>, AnalysisError> {
        Ok(Box::new(
            crate::pattern::SimplePatternSplitTokenizer::from_automaton(&self.dfa)?,
        ))
    }
}

factory_struct! {
    /// `org.apache.lucene.analysis.th.ThaiTokenizerFactory` (`thai`): the
    /// arguments are Java's; without an installed Thai word iterator,
    /// creating the tokenizer is `ThaiTokenizer`'s
    /// `UnsupportedOperationException` (see [`crate::lang::th`]).
    ThaiTokenizerFactory { jvm_breaker: bool }
}
analysis_factory!(ThaiTokenizerFactory);

impl FactoryClass for ThaiTokenizerFactory {
    const NAME: &'static str = "thai";
    const CLASS_NAME: &'static str = "org.apache.lucene.analysis.th.ThaiTokenizerFactory";
    fn from_args(args: &mut JavaArgs) -> Result<Self, FactoryError> {
        let base = FactoryBase::new(Self::CLASS_NAME, args)?;
        // Ours: cut with the host JVM's word iterator, Lucene's own.
        let jvm_breaker = args::get_boolean(args, "jvmBreaker", false);
        args::reject_unknown(args)?;
        Ok(ThaiTokenizerFactory { base, jvm_breaker })
    }
}

impl TokenizerFactory for ThaiTokenizerFactory {
    fn create(&self) -> Result<Box<dyn TokenStream>, AnalysisError> {
        Ok(Box::new(if self.jvm_breaker {
            crate::lang::th::thai_tokenizer_jvm()?
        } else {
            crate::lang::th::thai_tokenizer()?
        }))
    }
}

factory_struct! {
    /// `org.apache.lucene.analysis.wikipedia.WikipediaTokenizerFactory` (`wikipedia`).
    WikipediaTokenizerFactory { token_output: i32, untokenized_types: HashSet<String> }
}
analysis_factory!(WikipediaTokenizerFactory);

impl FactoryClass for WikipediaTokenizerFactory {
    const NAME: &'static str = "wikipedia";
    const CLASS_NAME: &'static str =
        "org.apache.lucene.analysis.wikipedia.WikipediaTokenizerFactory";
    fn from_args(args: &mut JavaArgs) -> Result<Self, FactoryError> {
        let base = FactoryBase::new(Self::CLASS_NAME, args)?;
        let token_output = args::get_int(args, "tokenOutput", crate::wikipedia::TOKENS_ONLY)?;
        let untokenized_types = args::get_set(args, "untokenizedTypes")
            .unwrap_or_default()
            .into_iter()
            .collect();
        args::reject_unknown(args)?;
        Ok(WikipediaTokenizerFactory {
            base,
            token_output,
            untokenized_types,
        })
    }
}

impl TokenizerFactory for WikipediaTokenizerFactory {
    fn create(&self) -> Result<Box<dyn TokenStream>, AnalysisError> {
        Ok(Box::new(crate::wikipedia::WikipediaTokenizer::new(
            self.token_output,
            self.untokenized_types.clone(),
        )?))
    }
}

#[cfg(test)]
mod tests;
