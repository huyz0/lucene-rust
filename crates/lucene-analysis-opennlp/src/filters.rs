//! `OpenNLPPOSFilter`, `OpenNLPChunkerFilter` and `OpenNLPLemmatizerFilter`,
//! each reading a sentence at a time from its input through
//! `SentenceAttributeExtractor` (consecutive tokens of one sentence index),
//! running its OpenNLP tool over the sentence and replaying the tokens:
//! the POS tag or chunk tag as the token type, the lemma as the term of a
//! token that is not a keyword.

use lucene_analysis::{AnalysisError, AttributeSource, TokenFilter, TokenStream};

use crate::ops::LemmatizerOp;
use crate::tools::chunker::Chunker;
use crate::tools::postag::PosTagger;
use crate::tools::text::{self, JString};

/// `SentenceAttributeExtractor`.
#[derive(Default)]
struct SentenceAttributeExtractor {
    sentence_token_attrs: Vec<AttributeSource>,
    prev: Option<AttributeSource>,
    /// The last sentence's attribute copies, refilled for the next one
    /// (`clone_from` into their buffers rather than a new clone a token).
    spare: Vec<AttributeSource>,
    curr_sentence: i32,
    has_next_token: bool,
}

impl SentenceAttributeExtractor {
    fn new() -> Self {
        SentenceAttributeExtractor {
            has_next_token: true,
            ..Self::default()
        }
    }

    /// `extractSentenceAttributes()`: the next sentence's tokens (each the
    /// input's attributes cloned after it).
    fn extract<T: TokenStream>(&mut self, input: &mut T) -> Result<(), AnalysisError> {
        self.spare.append(&mut self.sentence_token_attrs);
        loop {
            self.has_next_token = input.increment_token()?;
            let sentence = input.attributes().sentence_index();
            let has_next = self.curr_sentence == sentence && self.has_next_token;
            self.curr_sentence = sentence;
            if let Some(prev) = self.prev.take() {
                self.sentence_token_attrs.push(prev);
            }
            let mut copy = self.spare.pop().unwrap_or_default();
            copy.clone_from(input.attributes());
            self.prev = Some(copy);
            if !has_next {
                return Ok(());
            }
        }
    }

    /// `allSentencesProcessed()`.
    fn all_sentences_processed(&self) -> bool {
        !self.has_next_token
    }

    /// `reset()`.
    fn reset(&mut self) {
        self.has_next_token = true;
        self.sentence_token_attrs.clear();
        self.curr_sentence = 0;
        self.prev = None;
    }
}

fn terms(attrs: &[AttributeSource]) -> Vec<JString> {
    attrs.iter().map(|a| text::units(a.term())).collect()
}

fn types(attrs: &[AttributeSource]) -> Vec<JString> {
    attrs.iter().map(|a| text::units(a.token_type())).collect()
}

/// A tagger the POS and chunker filters share the shape of: tags for a
/// sentence's tokens, from their attributes.
pub trait SentenceTagger: Send {
    /// The tags of the sentence whose tokens' attributes are `attrs`.
    fn tags(&mut self, attrs: &[AttributeSource]) -> Result<Vec<JString>, AnalysisError>;
}

impl SentenceTagger for PosTagger {
    fn tags(&mut self, attrs: &[AttributeSource]) -> Result<Vec<JString>, AnalysisError> {
        self.tag(&terms(attrs))
    }
}

/// The chunker, absent when the factory had no model (Java then fails on
/// the first sentence, dereferencing its null op).
impl SentenceTagger for Option<Chunker> {
    fn tags(&mut self, attrs: &[AttributeSource]) -> Result<Vec<JString>, AnalysisError> {
        match self {
            Some(c) => c.chunk(&terms(attrs), &types(attrs)),
            None => Err(AnalysisError::IllegalState(
                "NullPointerException: OpenNLPChunkerFilter has no chunker model".into(),
            )),
        }
    }
}

/// The POS and chunker filters: a sentence's tokens replayed with a tag
/// each as their type.
pub struct TaggingFilter<T: TokenStream, G> {
    input: T,
    tagger: G,
    extractor: SentenceAttributeExtractor,
    token_num: usize,
}

/// `OpenNLPPOSFilter`.
pub type OpenNlpPosFilter<T> = TaggingFilter<T, PosTagger>;

/// `OpenNLPChunkerFilter`.
pub type OpenNlpChunkerFilter<T> = TaggingFilter<T, Option<Chunker>>;

impl<T: TokenStream> OpenNlpPosFilter<T> {
    /// `new OpenNLPPOSFilter(input, posTaggerOp)`.
    pub fn new(input: T, tagger: PosTagger) -> Self {
        TaggingFilter {
            input,
            tagger,
            extractor: SentenceAttributeExtractor::new(),
            token_num: 0,
        }
    }
}

impl<T: TokenStream> OpenNlpChunkerFilter<T> {
    /// `new OpenNLPChunkerFilter(input, chunkerOp)`.
    pub fn new(input: T, chunker: Option<Chunker>) -> Self {
        TaggingFilter {
            input,
            tagger: chunker,
            extractor: SentenceAttributeExtractor::new(),
            token_num: 0,
        }
    }
}

impl<T: TokenStream, G: SentenceTagger> TaggingFilter<T, G> {
    /// `nextSentence()`: whether it has tokens.
    fn next_sentence(&mut self) -> Result<bool, AnalysisError> {
        self.token_num = 0;
        self.extractor.extract(&mut self.input)?;
        let tags = self.tagger.tags(&self.extractor.sentence_token_attrs)?;
        for (a, tag) in self.extractor.sentence_token_attrs.iter_mut().zip(tags) {
            a.set_token_type(text::to_string(&tag));
        }
        Ok(!self.extractor.sentence_token_attrs.is_empty())
    }
}

impl<T: TokenStream, G: SentenceTagger> TokenFilter for TaggingFilter<T, G> {
    type Input = T;

    fn input(&self) -> &T {
        &self.input
    }

    fn input_mut(&mut self) -> &mut T {
        &mut self.input
    }

    // ARITH: counters of a sentence's tokens, bounded by its length.
    #[allow(clippy::arithmetic_side_effects)]
    fn increment(&mut self) -> Result<bool, AnalysisError> {
        if self.token_num >= self.extractor.sentence_token_attrs.len()
            && (self.extractor.all_sentences_processed() || !self.next_sentence()?)
        {
            return Ok(false);
        }
        let saved = &self.extractor.sentence_token_attrs[self.token_num];
        self.input.attributes_mut().clone_from(saved);
        self.token_num += 1;
        Ok(true)
    }

    fn reset_filter(&mut self) -> Result<(), AnalysisError> {
        self.input.reset()?;
        self.extractor.reset();
        self.token_num = 0;
        Ok(())
    }
}

/// `OpenNLPLemmatizerFilter`: a sentence's tokens replayed, each that is
/// not a keyword with its lemma as the term.
pub struct OpenNlpLemmatizerFilter<T: TokenStream> {
    input: T,
    op: LemmatizerOp,
    extractor: SentenceAttributeExtractor,
    /// `sentenceTokenAttrsIter`'s position.
    next_attr: usize,
    lemmas: Vec<JString>,
    lemma_num: usize,
}

impl<T: TokenStream> OpenNlpLemmatizerFilter<T> {
    /// `new OpenNLPLemmatizerFilter(input, lemmatizerOp)`.
    pub fn new(input: T, op: LemmatizerOp) -> Self {
        OpenNlpLemmatizerFilter {
            input,
            op,
            extractor: SentenceAttributeExtractor::new(),
            next_attr: 0,
            lemmas: Vec::new(),
            lemma_num: 0,
        }
    }

    /// `nextSentence()`: whether it has tokens.
    fn next_sentence(&mut self) -> Result<bool, AnalysisError> {
        self.lemma_num = 0;
        self.extractor.extract(&mut self.input)?;
        let words: Vec<&AttributeSource> = self
            .extractor
            .sentence_token_attrs
            .iter()
            .filter(|a| !a.is_keyword())
            .collect();
        let tokens: Vec<JString> = words.iter().map(|a| text::units(a.term())).collect();
        let tags: Vec<JString> = words.iter().map(|a| text::units(a.token_type())).collect();
        self.lemmas = self.op.lemmatize(&tokens, &tags)?;
        self.next_attr = 0;
        Ok(!self.extractor.sentence_token_attrs.is_empty())
    }
}

impl<T: TokenStream> TokenFilter for OpenNlpLemmatizerFilter<T> {
    type Input = T;

    fn input(&self) -> &T {
        &self.input
    }

    fn input_mut(&mut self) -> &mut T {
        &mut self.input
    }

    // ARITH: counters of a sentence's tokens, bounded by its length.
    #[allow(clippy::arithmetic_side_effects)]
    fn increment(&mut self) -> Result<bool, AnalysisError> {
        if self.lemma_num >= self.lemmas.len()
            && (self.extractor.all_sentences_processed() || !self.next_sentence()?)
        {
            return Ok(false);
        }
        let Some(saved) = self.extractor.sentence_token_attrs.get(self.next_attr) else {
            return Err(AnalysisError::IllegalState(
                "NoSuchElementException: the sentence's tokens are exhausted".into(),
            ));
        };
        self.next_attr += 1;
        let atts = self.input.attributes_mut();
        atts.clone_from(saved);
        if !atts.is_keyword() {
            atts.set_term_utf16(&self.lemmas[self.lemma_num]);
            self.lemma_num += 1;
        }
        Ok(true)
    }

    fn reset_filter(&mut self) -> Result<(), AnalysisError> {
        self.input.reset()?;
        self.extractor.reset();
        self.next_attr = 0;
        self.lemmas.clear();
        self.lemma_num = 0;
        Ok(())
    }
}
