//! The small mock filters: `MockTokenFilter` (automaton stop filter),
//! `MockLowerCaseFilter`, `MockHoleInjectingTokenFilter`, and the payload
//! filters `MockFixedLengthPayloadFilter`, `MockVariableLengthPayloadFilter`,
//! `SimplePayloadFilter` and `MockPayloadFilter`.

use std::sync::{Arc, LazyLock, Mutex};

use lucene_analysis::java_character;
use lucene_analysis::{AnalysisError, TokenFilter, TokenStream};
use lucene_util::automaton::automata::{make_empty, make_string};
use lucene_util::automaton::operations::{determinize, union, DEFAULT_DETERMINIZE_WORK_LIMIT};
use lucene_util::automaton::CharacterRunAutomaton;
use lucene_util::java_random::JavaRandom;

use crate::util::test_util;

/// A `Function<CharSequence, Integer>`: the term frequency for a term.
pub type TermFreqFn = Arc<dyn Fn(&str) -> i32 + Send + Sync>;

/// A `Random` shared between an analyzer and the filters it creates.
pub type SharedRandom = Arc<Mutex<JavaRandom>>;

/// Locks a shared random (a poisoned lock still holds a usable stream).
pub(crate) fn lock(r: &SharedRandom) -> std::sync::MutexGuard<'_, JavaRandom> {
    r.lock().unwrap_or_else(|p| p.into_inner())
}

static EMPTY_STOPSET: LazyLock<Arc<CharacterRunAutomaton>> = LazyLock::new(|| {
    Arc::new(CharacterRunAutomaton::new(&make_empty()).expect("the empty automaton"))
});

/// `MockTokenFilter.ENGLISH_STOPSET`'s words.
const ENGLISH_STOPWORDS: [&str; 33] = [
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in", "into", "is", "it",
    "no", "not", "of", "on", "or", "such", "that", "the", "their", "then", "there", "these",
    "they", "this", "to", "was", "will", "with",
];

static ENGLISH_STOPSET: LazyLock<Arc<CharacterRunAutomaton>> = LazyLock::new(|| {
    let words: Vec<_> = ENGLISH_STOPWORDS.iter().map(|w| make_string(w)).collect();
    let refs: Vec<_> = words.iter().collect();
    let a = determinize(&union(&refs), DEFAULT_DETERMINIZE_WORK_LIMIT).expect("small automaton");
    Arc::new(CharacterRunAutomaton::new(&a).expect("determinized"))
});

/// `MockTokenFilter.EMPTY_STOPSET`: removes nothing.
pub fn empty_stopset() -> Arc<CharacterRunAutomaton> {
    Arc::clone(&EMPTY_STOPSET)
}

/// `MockTokenFilter.ENGLISH_STOPSET`: the classic English stop words.
pub fn english_stopset() -> Arc<CharacterRunAutomaton> {
    Arc::clone(&ENGLISH_STOPSET)
}

/// `org.apache.lucene.tests.analysis.MockTokenFilter`: removes the terms
/// `filter` accepts.
///
/// With a term-frequency function, it sets each kept token's frequency and
/// (as Java, which then has no position-increment attribute) does not carry
/// removed tokens' increments.
pub struct MockTokenFilter<I> {
    input: I,
    filter: Arc<CharacterRunAutomaton>,
    custom_term_freq: Option<TermFreqFn>,
    skipped_positions: i32,
}

impl<I: TokenStream> MockTokenFilter<I> {
    /// `MockTokenFilter(input, filter)`.
    pub fn new(input: I, filter: Arc<CharacterRunAutomaton>) -> Self {
        Self::with_term_freq(input, filter, None)
    }

    /// `MockTokenFilter(input, filter, customTermFreq)`.
    pub fn with_term_freq(
        input: I,
        filter: Arc<CharacterRunAutomaton>,
        custom_term_freq: Option<TermFreqFn>,
    ) -> Self {
        MockTokenFilter {
            input,
            filter,
            custom_term_freq,
            skipped_positions: 0,
        }
    }
}

impl<I: TokenStream> TokenFilter for MockTokenFilter<I> {
    type Input = I;

    fn input(&self) -> &I {
        &self.input
    }

    fn input_mut(&mut self) -> &mut I {
        &mut self.input
    }

    fn increment(&mut self) -> Result<bool, AnalysisError> {
        // return the first non-stop word found
        self.skipped_positions = 0;
        while self.input.increment_token()? {
            let atts = self.input.attributes_mut();
            let stop = self
                .filter
                .run_code_points(atts.term().chars().map(|c| c as i32));
            if !stop {
                match &self.custom_term_freq {
                    Some(f) => {
                        let freq = f(atts.term());
                        atts.set_term_frequency(freq)?;
                    }
                    None => {
                        let inc = atts.position_increment() + self.skipped_positions;
                        atts.set_position_increment(inc)?;
                    }
                }
                return Ok(true);
            }
            if self.custom_term_freq.is_none() {
                self.skipped_positions += atts.position_increment();
            }
        }
        // reached EOS -- return false
        Ok(false)
    }

    fn end_filter(&mut self) -> Result<(), AnalysisError> {
        self.input.end()?;
        if self.custom_term_freq.is_none() {
            let atts = self.input.attributes_mut();
            let inc = atts.position_increment() + self.skipped_positions;
            atts.set_position_increment(inc)?;
        }
        Ok(())
    }

    fn reset_filter(&mut self) -> Result<(), AnalysisError> {
        self.input.reset()?;
        self.skipped_positions = 0;
        Ok(())
    }
}

/// `org.apache.lucene.tests.analysis.MockLowerCaseFilter`: lower-cases each
/// code point (`CharacterUtils.toLowerCase`).
pub struct MockLowerCaseFilter<I> {
    input: I,
}

impl<I: TokenStream> MockLowerCaseFilter<I> {
    /// `MockLowerCaseFilter(in)`.
    pub fn new(input: I) -> Self {
        MockLowerCaseFilter { input }
    }
}

impl<I: TokenStream> TokenFilter for MockLowerCaseFilter<I> {
    type Input = I;

    fn input(&self) -> &I {
        &self.input
    }

    fn input_mut(&mut self) -> &mut I {
        &mut self.input
    }

    fn increment(&mut self) -> Result<bool, AnalysisError> {
        if !self.input.increment_token()? {
            return Ok(false);
        }
        let term = self.input.attributes_mut().term_mut();
        let lowered: String = term
            .chars()
            .map(|c| char::from_u32(java_character::to_lower_case(c as u32)).unwrap_or(c))
            .collect();
        *term = lowered;
        Ok(true)
    }
}

/// `org.apache.lucene.tests.analysis.MockHoleInjectingTokenFilter`: widens
/// some position increments, where that cannot break the graph.
pub struct MockHoleInjectingTokenFilter<I> {
    input: I,
    random_seed: i64,
    random: JavaRandom,
    max_pos: i32,
    pos: i32,
}

impl<I: TokenStream> MockHoleInjectingTokenFilter<I> {
    /// `MockHoleInjectingTokenFilter(random, in)`: draws its seed.
    pub fn new(random: &mut JavaRandom, input: I) -> Self {
        let random_seed = random.next_long();
        MockHoleInjectingTokenFilter {
            input,
            random_seed,
            random: JavaRandom::new(random_seed),
            max_pos: -1,
            pos: -1,
        }
    }
}

impl<I: TokenStream> TokenFilter for MockHoleInjectingTokenFilter<I> {
    type Input = I;

    fn input(&self) -> &I {
        &self.input
    }

    fn input_mut(&mut self) -> &mut I {
        &mut self.input
    }

    fn reset_filter(&mut self) -> Result<(), AnalysisError> {
        self.input.reset()?;
        self.random = JavaRandom::new(self.random_seed);
        self.max_pos = -1;
        self.pos = -1;
        Ok(())
    }

    fn increment(&mut self) -> Result<bool, AnalysisError> {
        if !self.input.increment_token()? {
            return Ok(false);
        }
        let atts = self.input.attributes_mut();
        let pos_inc = atts.position_increment();
        let mut next_pos = self.pos + pos_inc;
        // Carefully inject a hole only where it won't mess up the graph:
        if pos_inc > 0 && self.max_pos <= next_pos && self.random.next_int_bounded(5) == 3 {
            let hole_size = test_util::next_int(&mut self.random, 1, 5);
            atts.set_position_increment(pos_inc + hole_size)?;
            next_pos += hole_size;
        }
        self.pos = next_pos;
        self.max_pos = self.max_pos.max(self.pos + atts.position_length());
        Ok(true)
    }
}

/// `org.apache.lucene.tests.analysis.MockFixedLengthPayloadFilter`: a random
/// payload of a fixed length on every token.
pub struct MockFixedLengthPayloadFilter<I> {
    input: I,
    random: SharedRandom,
    bytes: Vec<u8>,
}

impl<I: TokenStream> MockFixedLengthPayloadFilter<I> {
    /// `MockFixedLengthPayloadFilter(random, in, length)`.
    ///
    /// # Errors
    /// `IllegalArgument` for a negative length.
    pub fn new(random: SharedRandom, input: I, length: i32) -> Result<Self, AnalysisError> {
        if length < 0 {
            return Err(AnalysisError::IllegalArgument(
                "length must be >= 0".to_string(),
            ));
        }
        Ok(MockFixedLengthPayloadFilter {
            input,
            random,
            bytes: vec![0; length as usize],
        })
    }
}

impl<I: TokenStream> TokenFilter for MockFixedLengthPayloadFilter<I> {
    type Input = I;

    fn input(&self) -> &I {
        &self.input
    }

    fn input_mut(&mut self) -> &mut I {
        &mut self.input
    }

    fn increment(&mut self) -> Result<bool, AnalysisError> {
        if !self.input.increment_token()? {
            return Ok(false);
        }
        lock(&self.random).next_bytes(&mut self.bytes);
        self.input
            .attributes_mut()
            .set_payload(Some(self.bytes.clone()));
        Ok(true)
    }
}

/// `MockVariableLengthPayloadFilter.MAXLENGTH`.
const MAX_PAYLOAD_LENGTH: usize = 129;

/// `org.apache.lucene.tests.analysis.MockVariableLengthPayloadFilter`: a
/// random payload of a random length (under 129) on every token.
pub struct MockVariableLengthPayloadFilter<I> {
    input: I,
    random: SharedRandom,
    bytes: [u8; MAX_PAYLOAD_LENGTH],
}

impl<I: TokenStream> MockVariableLengthPayloadFilter<I> {
    /// `MockVariableLengthPayloadFilter(random, in)`.
    pub fn new(random: SharedRandom, input: I) -> Self {
        MockVariableLengthPayloadFilter {
            input,
            random,
            bytes: [0; MAX_PAYLOAD_LENGTH],
        }
    }
}

impl<I: TokenStream> TokenFilter for MockVariableLengthPayloadFilter<I> {
    type Input = I;

    fn input(&self) -> &I {
        &self.input
    }

    fn input_mut(&mut self) -> &mut I {
        &mut self.input
    }

    fn increment(&mut self) -> Result<bool, AnalysisError> {
        if !self.input.increment_token()? {
            return Ok(false);
        }
        let len = {
            let mut r = lock(&self.random);
            r.next_bytes(&mut self.bytes);
            r.next_int_bounded(MAX_PAYLOAD_LENGTH as i32) as usize
        };
        self.input
            .attributes_mut()
            .set_payload(Some(self.bytes[..len].to_vec()));
        Ok(true)
    }
}

/// `org.apache.lucene.tests.analysis.SimplePayloadFilter`: the payload
/// `"pos: N"`, N counting tokens.
pub struct SimplePayloadFilter<I> {
    input: I,
    pos: i32,
}

impl<I: TokenStream> SimplePayloadFilter<I> {
    /// `SimplePayloadFilter(input)`.
    pub fn new(input: I) -> Self {
        SimplePayloadFilter { input, pos: 0 }
    }
}

impl<I: TokenStream> TokenFilter for SimplePayloadFilter<I> {
    type Input = I;

    fn input(&self) -> &I {
        &self.input
    }

    fn input_mut(&mut self) -> &mut I {
        &mut self.input
    }

    fn increment(&mut self) -> Result<bool, AnalysisError> {
        if !self.input.increment_token()? {
            return Ok(false);
        }
        self.input
            .attributes_mut()
            .set_payload(Some(format!("pos: {}", self.pos).into_bytes()));
        self.pos += 1;
        Ok(true)
    }

    fn reset_filter(&mut self) -> Result<(), AnalysisError> {
        self.input.reset()?;
        self.pos = 0;
        Ok(())
    }
}

/// `MockPayloadAnalyzer`'s package-private `MockPayloadFilter`: the payload
/// `"pos: N"`, and every second token after the first stacked on the
/// previous position.
pub struct MockPayloadFilter<I> {
    input: I,
    /// Java keeps the field name, unused.
    _field_name: String,
    pos: i32,
    i: i32,
}

impl<I: TokenStream> MockPayloadFilter<I> {
    /// `MockPayloadFilter(input, fieldName)`.
    pub fn new(input: I, field_name: &str) -> Self {
        MockPayloadFilter {
            input,
            _field_name: field_name.to_string(),
            pos: 0,
            i: 0,
        }
    }
}

impl<I: TokenStream> TokenFilter for MockPayloadFilter<I> {
    type Input = I;

    fn input(&self) -> &I {
        &self.input
    }

    fn input_mut(&mut self) -> &mut I {
        &mut self.input
    }

    fn increment(&mut self) -> Result<bool, AnalysisError> {
        if !self.input.increment_token()? {
            return Ok(false);
        }
        let atts = self.input.attributes_mut();
        atts.set_payload(Some(format!("pos: {}", self.pos).into_bytes()));
        let pos_incr = if self.pos == 0 || self.i % 2 == 1 {
            1
        } else {
            0
        };
        atts.set_position_increment(pos_incr)?;
        self.pos += pos_incr;
        self.i += 1;
        Ok(true)
    }

    fn reset_filter(&mut self) -> Result<(), AnalysisError> {
        self.input.reset()?;
        self.i = 0;
        self.pos = 0;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::canned::CannedTokenStream;
    use crate::analysis::mock_tokenizer::MockTokenizer;
    use crate::analysis::token::Token;
    use lucene_analysis::token_stream::consume;
    use lucene_analysis::Tokenizer;

    /// One token: term, position increment, start offset, payload.
    type Tok = (String, i32, i32, Option<Vec<u8>>);

    fn run(ts: &mut dyn TokenStream) -> (Vec<Tok>, i32) {
        let mut out = Vec::new();
        let end = consume(ts, |a| {
            out.push((
                a.term().to_string(),
                a.position_increment(),
                a.term_frequency(),
                a.payload().map(<[u8]>::to_vec),
            ))
        })
        .unwrap();
        (out, end.position_increment())
    }

    fn words(text: &str) -> MockTokenizer {
        MockTokenizer::default().with_text(text).unwrap()
    }

    #[test]
    fn stop_filter_carries_increments_into_the_next_token_and_end() {
        let mut f = MockTokenFilter::new(words("the fox is a"), english_stopset());
        let (t, end_inc) = run(&mut f);
        assert_eq!(t.len(), 1);
        assert_eq!((t[0].0.as_str(), t[0].1), ("fox", 2));
        assert_eq!(end_inc, 2);
        let mut f = MockTokenFilter::new(words("a b"), empty_stopset());
        assert_eq!(run(&mut f).0.len(), 2);
    }

    #[test]
    fn a_term_freq_function_sets_frequencies_and_keeps_increments() {
        let freq: TermFreqFn = Arc::new(|t: &str| t.len() as i32);
        let mut f =
            MockTokenFilter::with_term_freq(words("the quick"), english_stopset(), Some(freq));
        let (t, end_inc) = run(&mut f);
        assert_eq!((t[0].0.as_str(), t[0].1, t[0].2), ("quick", 1, 5));
        assert_eq!(end_inc, 0);
    }

    #[test]
    fn lower_case_filter_lowers_code_points() {
        let tok = MockTokenizer::new(crate::analysis::mock_tokenizer::whitespace(), false)
            .with_text("AbC \u{10400}")
            .unwrap();
        let (t, _) = run(&mut MockLowerCaseFilter::new(tok));
        assert_eq!(t[0].0, "abc");
        assert_eq!(t[1].0, "\u{10428}");
    }

    #[test]
    fn hole_injector_widens_some_increments_reproducibly() {
        let text = (0..60).map(|i| format!("w{i} ")).collect::<String>();
        let mut seed = JavaRandom::new(7);
        let mut f = MockHoleInjectingTokenFilter::new(&mut seed, words(&text));
        let (first, _) = run(&mut f);
        assert!(first.iter().any(|t| t.1 > 1), "some hole injected");
        f.input_mut()
            .set_reader(Box::new(lucene_analysis::StrReader::new(text.as_str())))
            .unwrap();
        let (second, _) = run(&mut f);
        assert_eq!(first, second, "reset replays the same holes");
        // a stacked token (posInc 0) never gets a hole
        let canned = CannedTokenStream::new(vec![
            Token::with_pos_inc("a", 1, 0, 1),
            Token::with_pos_inc("b", 0, 0, 1),
        ]);
        let mut f = MockHoleInjectingTokenFilter::new(&mut seed, canned);
        assert_eq!(run(&mut f).0[1].1, 0);
    }

    #[test]
    fn payload_filters() {
        let r: SharedRandom = Arc::new(Mutex::new(JavaRandom::new(1)));
        let mut f = MockFixedLengthPayloadFilter::new(r.clone(), words("a b"), 3).unwrap();
        let (t, _) = run(&mut f);
        assert!(t.iter().all(|t| t.3.as_ref().unwrap().len() == 3));
        assert!(MockFixedLengthPayloadFilter::new(r.clone(), words("a"), -1).is_err());
        let mut f = MockVariableLengthPayloadFilter::new(r, words("a b c"));
        let (t, _) = run(&mut f);
        assert!(t.iter().all(|t| t.3.as_ref().unwrap().len() < 129));
        let mut f = SimplePayloadFilter::new(words("a b"));
        let (t, _) = run(&mut f);
        assert_eq!(t[1].3.as_deref(), Some(&b"pos: 1"[..]));
        let mut f = MockPayloadFilter::new(words("a b c d"), "f");
        let (t, _) = run(&mut f);
        let incs: Vec<i32> = t.iter().map(|t| t.1).collect();
        assert_eq!(incs, [1, 1, 0, 1]);
        assert_eq!(t[3].3.as_deref(), Some(&b"pos: 2"[..]));
    }
}
