//! `org.apache.lucene.search.highlight` (`lucene-highlighter`): the classic
//! `Highlighter`. A token stream over the text is cut into fragments by a
//! [`Fragmenter`], each token scored by a [`Scorer`] ([`QueryScorer`] reads the
//! query's terms and, for phrases and span queries, their positions;
//! [`QueryTermScorer`] only the terms), adjacent matching tokens grouped
//! ([`TokenGroup`]) and marked up by a [`Formatter`], and the best fragments
//! returned.
//!
//! Offsets are UTF-16 code unit offsets, as Java's `String` indexes and the
//! analysis chain report them, so the text is held as UTF-16 while a fragment
//! is cut; a fragment boundary that splits a surrogate pair reads as U+FFFD
//! (Java would keep a lone surrogate).
//!
//! The unified highlighter is [`crate::highlighter`]; this is the older,
//! token-stream-driven one.

mod extractor;
mod formatters;
mod match_highlight;
mod term_vector;
#[cfg(test)]
mod tests;
pub mod unified;
pub mod vectorhighlight;
mod word_break;

pub use extractor::{
    DocFreqSource, PositionSpan, QueryScorer, QueryTermExtractor, QueryTermScorer,
    WeightedSpanTerm, WeightedSpanTermExtractor, WeightedTerm,
};
pub use formatters::{
    DefaultEncoder, GradientFormatter, NullFragmenter, SimpleFragmenter, SimpleHtmlEncoder,
    SimpleHtmlFormatter, SimpleSpanFragmenter, SpanGradientFormatter,
};

pub use match_highlight::{
    compute_offset_retrieval_strategies, default_scorer, BreakIteratorShrinkingAdjuster,
    DocHighlights, FieldValueHighlighter, FieldValueHighlighters, FieldValueProvider,
    MatchHighlighter, MatchRegionRetriever, OffsetRange, OffsetsFromMatchIterator,
    OffsetsFromPositions, OffsetsFromTokens, OffsetsFromValues, OffsetsRetrievalStrategy,
    OffsetsRetrievalStrategySupplier, OrHighlighter, Passage, PassageAdjuster, PassageFormatter,
    PassageSelector, QueryOffsetRange, StoredValues,
};
pub use term_vector::{TokenSources, TokenStreamFromTermVector};
pub use word_break::{Kind as BreakKind, RuleBreakIterator};

use std::sync::Arc;

use lucene_analysis::{AnalysisError, AttributeSource, TokenStream};

/// What the highlighter can fail with.
#[derive(Debug)]
pub enum Error {
    /// `InvalidTokenOffsetsException`: a token reaches past the text.
    InvalidTokenOffsets(String),
    /// The analysis chain failed.
    Analysis(AnalysisError),
    /// A query could not be run against the document.
    Search(crate::Error),
    /// `IllegalArgumentException`.
    IllegalArgument(String),
    /// `IOException`, with Java's message.
    Io(String),
    /// `RuntimeException`.
    Runtime(String),
}

impl Error {
    /// The Java exception's class name.
    pub fn class(&self) -> &'static str {
        match self {
            Error::InvalidTokenOffsets(_) => {
                "org.apache.lucene.search.highlight.InvalidTokenOffsetsException"
            }
            Error::Analysis(_) => "java.io.IOException",
            Error::Search(_) => "java.io.IOException",
            Error::IllegalArgument(_) => "java.lang.IllegalArgumentException",
            Error::Io(_) => "java.io.IOException",
            Error::Runtime(_) => "java.lang.RuntimeException",
        }
    }

    /// The exception's message.
    pub fn message(&self) -> String {
        match self {
            Error::InvalidTokenOffsets(m)
            | Error::IllegalArgument(m)
            | Error::Io(m)
            | Error::Runtime(m) => m.clone(),
            Error::Analysis(e) => e.to_string(),
            Error::Search(e) => e.to_string(),
        }
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.class(), self.message())
    }
}

impl std::error::Error for Error {}

impl From<AnalysisError> for Error {
    fn from(e: AnalysisError) -> Self {
        Error::Analysis(e)
    }
}

impl From<crate::Error> for Error {
    fn from(e: crate::Error) -> Self {
        Error::Search(e)
    }
}

/// The highlighter's result type.
pub type Result<T> = std::result::Result<T, Error>;

/// A token stream the highlighter owns, whatever it borrows (an analyzer's
/// pooled components, usually).
pub struct DynStream<'a>(Box<dyn TokenStream + 'a>);

impl<'a> DynStream<'a> {
    /// Wraps a stream.
    pub fn new(stream: impl TokenStream + 'a) -> Self {
        DynStream(Box::new(stream))
    }
}

impl TokenStream for DynStream<'_> {
    fn attributes(&self) -> &AttributeSource {
        (*self.0).attributes()
    }
    fn attributes_mut(&mut self) -> &mut AttributeSource {
        (*self.0).attributes_mut()
    }
    fn increment_token(&mut self) -> std::result::Result<bool, AnalysisError> {
        (*self.0).increment_token()
    }
    fn reset(&mut self) -> std::result::Result<(), AnalysisError> {
        (*self.0).reset()
    }
    fn end(&mut self) -> std::result::Result<(), AnalysisError> {
        (*self.0).end()
    }
    fn close(&mut self) -> std::result::Result<(), AnalysisError> {
        (*self.0).close()
    }
    fn as_tokenizer(&mut self) -> Option<&mut dyn lucene_analysis::Tokenizer> {
        (*self.0).as_tokenizer()
    }
    fn conditional_root(&mut self) -> Option<&mut dyn std::any::Any> {
        (*self.0).conditional_root()
    }
}

/// `Scorer`: scores the tokens of a text and so its fragments.
pub trait Scorer {
    /// `init(tokenStream)`: called before the text is read; may consume the
    /// stream, and returns the stream the highlighter then reads (the
    /// original, or a replayable copy of it).
    fn init<'a>(&mut self, stream: DynStream<'a>) -> Result<DynStream<'a>>;
    /// `startFragment(newFragment)`.
    fn start_fragment(&mut self);
    /// `getTokenScore()` for the current token.
    fn token_score(&mut self, token: &AttributeSource) -> f32;
    /// `getFragmentScore()`.
    fn fragment_score(&self) -> f32;
    /// `QueryScorer.setMaxDocCharsToAnalyze`, which the highlighter sets on a
    /// `QueryScorer`; any other scorer ignores it.
    fn set_max_doc_chars_to_analyze(&mut self, _max: i32) {}
}

/// `Fragmenter`: where one fragment ends and the next begins.
pub trait Fragmenter {
    /// `start(originalText, tokenStream)`.
    fn start(&mut self, original_text: &str);
    /// `isNewFragment()` after the current token.
    fn is_new_fragment(&mut self, token: &AttributeSource) -> bool;
}

/// `Formatter`: marks up a group of matching tokens.
pub trait Formatter {
    /// `highlightTerm(originalText, tokenGroup)`.
    fn highlight_term(&self, original_text: &str, token_group: &TokenGroup) -> String;
}

/// `Encoder`: escapes the text between and inside the marked-up terms.
pub trait Encoder {
    /// `encodeText(originalText)`.
    fn encode_text(&self, original_text: &str) -> String;
}

/// `TokenGroup`: tokens whose offsets overlap, which are marked up together.
#[derive(Debug, Clone)]
pub struct TokenGroup {
    scores: [f32; Self::MAX_NUM_TOKENS_PER_GROUP],
    num_tokens: usize,
    group_start: i32,
    group_end: i32,
    tot: f32,
    match_start_offset: i32,
    match_end_offset: i32,
}

impl Default for TokenGroup {
    fn default() -> Self {
        TokenGroup {
            scores: [0.0; Self::MAX_NUM_TOKENS_PER_GROUP],
            num_tokens: 0,
            group_start: 0,
            group_end: 0,
            tot: 0.0,
            match_start_offset: 0,
            match_end_offset: 0,
        }
    }
}

impl TokenGroup {
    const MAX_NUM_TOKENS_PER_GROUP: usize = 50;

    /// `new TokenGroup(tokenStream)`.
    pub fn new() -> Self {
        Self::default()
    }

    fn add_token(&mut self, score: f32, token: &AttributeSource) {
        if self.num_tokens < Self::MAX_NUM_TOKENS_PER_GROUP {
            let term_start = token.start_offset();
            let term_end = token.end_offset();
            if self.num_tokens == 0 {
                self.group_start = term_start;
                self.match_start_offset = term_start;
                self.group_end = term_end;
                self.match_end_offset = term_end;
                self.tot += score;
            } else {
                self.group_start = self.group_start.min(term_start);
                self.group_end = self.group_end.max(term_end);
                if score > 0.0 {
                    if self.tot == 0.0 {
                        self.match_start_offset = term_start;
                        self.match_end_offset = term_end;
                    } else {
                        self.match_start_offset = self.match_start_offset.min(term_start);
                        self.match_end_offset = self.match_end_offset.max(term_end);
                    }
                    self.tot += score;
                }
            }
            self.scores[self.num_tokens] = score;
            self.num_tokens += 1;
        }
    }

    fn is_distinct(&self, token: &AttributeSource) -> bool {
        token.start_offset() >= self.group_end
    }

    fn clear(&mut self) {
        self.num_tokens = 0;
        self.tot = 0.0;
    }

    /// `getScore(index)`.
    pub fn score(&self, index: usize) -> f32 {
        self.scores[index]
    }

    /// `getStartOffset()`: where the matching tokens start.
    pub fn start_offset(&self) -> i32 {
        self.match_start_offset
    }

    /// `getEndOffset()`: where the matching tokens end.
    pub fn end_offset(&self) -> i32 {
        self.match_end_offset
    }

    /// `getNumTokens()`.
    pub fn num_tokens(&self) -> usize {
        self.num_tokens
    }

    /// `getTotalScore()`.
    pub fn total_score(&self) -> f32 {
        self.tot
    }
}

/// `TextFragment`: a stretch of the marked-up text and its score.
#[derive(Debug, Clone)]
pub struct TextFragment {
    marked_up_text: Arc<Vec<u16>>,
    frag_num: usize,
    text_start_pos: usize,
    text_end_pos: usize,
    score: f32,
}

impl TextFragment {
    fn new(text_start_pos: usize, frag_num: usize) -> Self {
        TextFragment {
            marked_up_text: Arc::new(Vec::new()),
            frag_num,
            text_start_pos,
            text_end_pos: 0,
            score: 0.0,
        }
    }

    /// `getScore()`.
    pub fn score(&self) -> f32 {
        self.score
    }

    /// `getFragNum()`.
    pub fn frag_num(&self) -> usize {
        self.frag_num
    }

    /// `merge(frag2)`.
    pub fn merge(&mut self, other: &TextFragment) {
        self.text_end_pos = other.text_end_pos;
        self.score = self.score.max(other.score);
    }

    /// `follows(fragment)`.
    pub fn follows(&self, fragment: &TextFragment) -> bool {
        self.text_start_pos == fragment.text_end_pos
    }
}

impl std::fmt::Display for TextFragment {
    /// `toString()`: the marked-up text of this fragment.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let end = self.text_end_pos.min(self.marked_up_text.len());
        let start = self.text_start_pos.min(end);
        f.write_str(&String::from_utf16_lossy(&self.marked_up_text[start..end]))
    }
}

/// `Highlighter.DEFAULT_MAX_CHARS_TO_ANALYZE`.
pub const DEFAULT_MAX_CHARS_TO_ANALYZE: i32 = 50 * 1024;

/// `Highlighter`.
pub struct Highlighter {
    formatter: Box<dyn Formatter>,
    encoder: Box<dyn Encoder>,
    fragment_scorer: Box<dyn Scorer>,
    max_doc_chars_to_analyze: i32,
    text_fragmenter: Box<dyn Fragmenter>,
}

impl Highlighter {
    /// `new Highlighter(fragmentScorer)`: `<B>`/`</B>` marks.
    pub fn new(fragment_scorer: impl Scorer + 'static) -> Self {
        Self::with_formatter(SimpleHtmlFormatter::default(), fragment_scorer)
    }

    /// `new Highlighter(formatter, fragmentScorer)`.
    pub fn with_formatter(
        formatter: impl Formatter + 'static,
        fragment_scorer: impl Scorer + 'static,
    ) -> Self {
        Self::with_encoder(formatter, DefaultEncoder, fragment_scorer)
    }

    /// `new Highlighter(formatter, encoder, fragmentScorer)`.
    pub fn with_encoder(
        formatter: impl Formatter + 'static,
        encoder: impl Encoder + 'static,
        fragment_scorer: impl Scorer + 'static,
    ) -> Self {
        Highlighter {
            formatter: Box::new(formatter),
            encoder: Box::new(encoder),
            fragment_scorer: Box::new(fragment_scorer),
            max_doc_chars_to_analyze: DEFAULT_MAX_CHARS_TO_ANALYZE,
            text_fragmenter: Box::new(SimpleFragmenter::default()),
        }
    }

    /// `setTextFragmenter(fragmenter)`.
    pub fn set_text_fragmenter(&mut self, fragmenter: impl Fragmenter + 'static) {
        self.text_fragmenter = Box::new(fragmenter);
    }

    /// `setFragmentScorer(scorer)`.
    pub fn set_fragment_scorer(&mut self, scorer: impl Scorer + 'static) {
        self.fragment_scorer = Box::new(scorer);
    }

    /// `setEncoder(encoder)`.
    pub fn set_encoder(&mut self, encoder: impl Encoder + 'static) {
        self.encoder = Box::new(encoder);
    }

    /// `getMaxDocCharsToAnalyze()`.
    pub fn max_doc_chars_to_analyze(&self) -> i32 {
        self.max_doc_chars_to_analyze
    }

    /// `setMaxDocCharsToAnalyze(n)`.
    pub fn set_max_doc_chars_to_analyze(&mut self, max: i32) {
        self.max_doc_chars_to_analyze = max;
    }

    /// `getBestFragment(tokenStream, text)`.
    pub fn get_best_fragment(
        &mut self,
        token_stream: DynStream<'_>,
        text: &str,
    ) -> Result<Option<String>> {
        Ok(self
            .get_best_fragments(token_stream, text, 1)?
            .into_iter()
            .next())
    }

    /// `getBestFragments(tokenStream, text, maxNumFragments)`.
    pub fn get_best_fragments(
        &mut self,
        token_stream: DynStream<'_>,
        text: &str,
        max_num_fragments: usize,
    ) -> Result<Vec<String>> {
        let max_num_fragments = max_num_fragments.max(1);
        let frags = self.get_best_text_fragments(token_stream, text, true, max_num_fragments)?;
        Ok(frags
            .iter()
            .filter(|f| f.score() > 0.0)
            .map(ToString::to_string)
            .collect())
    }

    /// `getBestFragments(tokenStream, text, maxNumFragments, separator)`.
    pub fn get_best_fragments_joined(
        &mut self,
        token_stream: DynStream<'_>,
        text: &str,
        max_num_fragments: usize,
        separator: &str,
    ) -> Result<String> {
        Ok(self
            .get_best_fragments(token_stream, text, max_num_fragments)?
            .join(separator))
    }

    /// `getBestTextFragments(tokenStream, text, mergeContiguousFragments,
    /// maxNumFragments)`.
    pub fn get_best_text_fragments(
        &mut self,
        token_stream: DynStream<'_>,
        text: &str,
        merge_contiguous_fragments: bool,
        max_num_fragments: usize,
    ) -> Result<Vec<TextFragment>> {
        let units: Vec<u16> = text.encode_utf16().collect();
        let text_len = units.len();
        let mut new_text: Vec<u16> = Vec::new();
        let mut doc_frags: Vec<TextFragment> = vec![TextFragment::new(0, 0)];

        self.fragment_scorer
            .set_max_doc_chars_to_analyze(self.max_doc_chars_to_analyze);
        let mut stream = self.fragment_scorer.init(token_stream)?;
        self.fragment_scorer.start_fragment();

        let outcome = self.read_tokens(&mut stream, &units, &mut new_text, &mut doc_frags);
        // `finally`: end and close, ignoring what they throw
        let _ = stream.end();
        let _ = stream.close();
        outcome?;

        let _ = text_len;
        let marked_up = Arc::new(new_text);
        for f in &mut doc_frags {
            f.marked_up_text = Arc::clone(&marked_up);
        }
        // `FragmentQueue.insertWithOverflow`, popped least first into the array
        // from the back: best score first, the earlier fragment on a tie
        let mut ranked = doc_frags;
        ranked.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.frag_num.cmp(&b.frag_num))
        });
        ranked.truncate(max_num_fragments);
        if merge_contiguous_fragments {
            let mut slots: Vec<Option<TextFragment>> = ranked.into_iter().map(Some).collect();
            merge_contiguous(&mut slots);
            return Ok(slots
                .into_iter()
                .flatten()
                .filter(|f| f.score() > 0.0)
                .collect());
        }
        Ok(ranked)
    }

    /// The token loop of `getBestTextFragments`.
    fn read_tokens(
        &mut self,
        stream: &mut DynStream<'_>,
        units: &[u16],
        new_text: &mut Vec<u16>,
        doc_frags: &mut Vec<TextFragment>,
    ) -> Result<()> {
        let text_len = units.len();
        let slice = |from: i32, to: i32| -> &[u16] {
            let from = usize::try_from(from).unwrap_or(0).min(text_len);
            let to = usize::try_from(to).unwrap_or(0).min(text_len).max(from);
            &units[from..to]
        };
        let encode = |enc: &dyn Encoder, s: &[u16]| -> Vec<u16> {
            enc.encode_text(&String::from_utf16_lossy(s))
                .encode_utf16()
                .collect()
        };
        let mut last_end_offset = 0i32;
        let mut cur = doc_frags.len() - 1;
        self.text_fragmenter.start(&String::from_utf16_lossy(units));
        let mut token_group = TokenGroup::new();
        stream.reset()?;
        let mut next = stream.increment_token()?;
        while next && stream.attributes().start_offset() < self.max_doc_chars_to_analyze {
            let atts = stream.attributes();
            let text_len_i = i32::try_from(text_len).unwrap_or(i32::MAX);
            if atts.end_offset() > text_len_i || atts.start_offset() > text_len_i {
                return Err(Error::InvalidTokenOffsets(format!(
                    "Token {} exceeds length of provided text sized {}",
                    atts.term(),
                    text_len
                )));
            }
            if token_group.num_tokens() > 0 && token_group.is_distinct(atts) {
                let (start, end) = (token_group.start_offset(), token_group.end_offset());
                let token_text = String::from_utf16_lossy(slice(start, end));
                let marked = self
                    .formatter
                    .highlight_term(&self.encoder.encode_text(&token_text), &token_group);
                if start > last_end_offset {
                    new_text.extend(encode(self.encoder.as_ref(), slice(last_end_offset, start)));
                }
                new_text.extend(marked.encode_utf16());
                last_end_offset = end.max(last_end_offset);
                token_group.clear();
                if self.text_fragmenter.is_new_fragment(stream.attributes()) {
                    doc_frags[cur].score = self.fragment_scorer.fragment_score();
                    doc_frags[cur].text_end_pos = new_text.len();
                    doc_frags.push(TextFragment::new(new_text.len(), doc_frags.len()));
                    cur = doc_frags.len() - 1;
                    self.fragment_scorer.start_fragment();
                }
            }
            let score = self.fragment_scorer.token_score(stream.attributes());
            token_group.add_token(score, stream.attributes());
            next = stream.increment_token()?;
        }
        doc_frags[cur].score = self.fragment_scorer.fragment_score();
        if token_group.num_tokens() > 0 {
            let (start, end) = (token_group.start_offset(), token_group.end_offset());
            let token_text = String::from_utf16_lossy(slice(start, end));
            let marked = self
                .formatter
                .highlight_term(&self.encoder.encode_text(&token_text), &token_group);
            if start > last_end_offset {
                new_text.extend(encode(self.encoder.as_ref(), slice(last_end_offset, start)));
            }
            new_text.extend(marked.encode_utf16());
            last_end_offset = last_end_offset.max(end);
        }
        let text_len_i = i32::try_from(text_len).unwrap_or(i32::MAX);
        if last_end_offset < text_len_i && text_len_i <= self.max_doc_chars_to_analyze {
            new_text.extend(encode(
                self.encoder.as_ref(),
                slice(last_end_offset, text_len_i),
            ));
        }
        doc_frags[cur].text_end_pos = new_text.len();
        Ok(())
    }
}

/// `mergeContiguousFragments(frag)`.
fn merge_contiguous(frag: &mut [Option<TextFragment>]) {
    if frag.len() <= 1 {
        return;
    }
    loop {
        let mut merging_still_being_done = false;
        for i in 0..frag.len() {
            if frag[i].is_none() {
                continue;
            }
            for x in 0..frag.len() {
                if frag[x].is_none() {
                    continue;
                }
                if frag[i].is_none() {
                    break;
                }
                let (Some(fi), Some(fx)) = (&frag[i], &frag[x]) else {
                    continue;
                };
                // `frag1` is the earlier fragment, `frag2` the one that follows it
                let (frag1_num, frag2_num) = if fi.follows(fx) {
                    (x, i)
                } else if fx.follows(fi) {
                    (i, x)
                } else {
                    continue;
                };
                let (Some(frag1), Some(frag2)) = (frag[frag1_num].clone(), frag[frag2_num].clone())
                else {
                    continue;
                };
                let (best, worst) = if frag1.score() > frag2.score() {
                    (frag1_num, frag2_num)
                } else {
                    (frag2_num, frag1_num)
                };
                let mut merged = frag1;
                merged.merge(&frag2);
                frag[worst] = None;
                frag[best] = Some(merged);
                merging_still_being_done = true;
            }
        }
        if !merging_still_being_done {
            return;
        }
    }
}
