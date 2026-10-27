//! `BaseTokenStreamTestCase`: the assertions every analysis test uses --
//! [`assert_token_stream_contents`], [`assert_analyzes_to`],
//! [`check_random_data`] (random text through an analyzer, single- and
//! multi-threaded, compared with a reference pass) and its building block
//! [`check_analysis_consistency`].
//!
//! Java's many `assertTokenStreamContents`/`assertAnalyzesTo` overloads,
//! each a different subset of nullable arrays, are one [`Expected`] value
//! whose unset fields are Java's `null`s. JUnit assertion failures panic;
//! errors from the stream under test are returned.
//!
//! Differs:
//! - Every core attribute is always present in the port, so Java's
//!   `has no XAttribute` checks cannot fail and are not ported, and
//!   `check_analysis_consistency` always takes Java's "offset + pos +
//!   posLength + type" branch (the one a default-factory chain takes).
//! - `checkResetException`'s second half (a missing `close()` makes the
//!   next `tokenStream` throw) checks the port's contract instead: an
//!   [`lucene_analysis::AnalyzerTokenStream`] closes itself when dropped, so
//!   the next `tokenStream` after an unclosed one must work.
//! - `checkRandomData`'s indexing pass (Java's, when `rarely(random)`)
//!   hands the `RandomIndexWriter` each text pre-analyzed by the analyzer
//!   under test, through the same readers Java's `setReaderValue` gives it:
//!   the writer's own analyzer is a stand-in carrying the analyzer's
//!   position and offset gaps (the port's writer takes an owned
//!   `Arc<Analyzer>`), built off the writer's `Random(seed)`. The
//!   directory is a plain `FsDirectory` under a temporary directory, checked
//!   with `CheckIndex` after the writer closes, as Java's mock directory is
//!   on close.
//! - [`new_attribute_factory`] draws Java's `nextInt(3)` and names the
//!   factory Java would pick, but all three build the port's one fixed
//!   [`AttributeSource`] struct, so the choice changes no attribute.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::sync::{Arc, Mutex};

use lucene_analysis::attributes::{AttrValue, AttributeSource, CustomAttribute};
use lucene_analysis::{
    token_stream_to_automaton, AnalysisError, Analyzer, CharReader, StrReader, TokenStream,
    Tokenizer, POS_SEP,
};
use lucene_index::document::IndexOptions;
use lucene_index::document::{Document, Field, FieldToken, FieldTokens, FieldType, TextField};
use lucene_store::FsDirectory;
use lucene_util::java_random::JavaRandom;
use lucene_util::test_support::TempDir;

use super::mock_analyzer::MockAnalyzer;
use super::mock_char_filter::MockCharFilter;
use super::mock_reader_wrapper::MockReaderWrapper;
use super::mock_tokenizer::{self, MockTokenizer};
use crate::index::random_index_writer::{RandomIndexWriter, WriterEnv};
use crate::util::automaton_test_util::get_finite_strings_recursive;
use crate::util::{lucene_test_case, test_util};

/// `CheckClearAttributesAttributeImpl`: records whether the chain cleared
/// its attributes.
///
/// Java's `copyTo` calls `target.clear()`, so restoring a captured state
/// counts as a clear; here a clone (what `restoreState` copies in) is
/// "cleared" for the same reason.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct CheckClearAttributes {
    clear_called: bool,
}

impl Clone for CheckClearAttributes {
    fn clone(&self) -> Self {
        CheckClearAttributes { clear_called: true }
    }
}

impl CheckClearAttributes {
    /// `getAndResetClearCalled()`.
    pub fn get_and_reset_clear_called(&mut self) -> bool {
        std::mem::take(&mut self.clear_called)
    }
}

impl CustomAttribute for CheckClearAttributes {
    fn clear(&mut self) {
        self.clear_called = true;
    }

    fn reflect(&self, reflector: &mut dyn FnMut(&'static str, &'static str, AttrValue<'_>)) {
        reflector(
            "org.apache.lucene.tests.analysis.BaseTokenStreamTestCase$CheckClearAttributesAttribute",
            "clearCalled",
            AttrValue::Bool(self.clear_called),
        );
    }
}

/// What [`assert_token_stream_contents`] expects: the terms, and whichever
/// other attributes the test checks (`None` is Java's `null`, unchecked).
#[derive(Debug, Clone, PartialEq)]
pub struct Expected {
    /// The terms, in order.
    pub output: Vec<String>,
    /// Start offsets.
    pub start_offsets: Option<Vec<i32>>,
    /// End offsets.
    pub end_offsets: Option<Vec<i32>>,
    /// Types.
    pub types: Option<Vec<String>>,
    /// Position increments.
    pub pos_increments: Option<Vec<i32>>,
    /// Position lengths.
    pub pos_lengths: Option<Vec<i32>>,
    /// The end offset after `end()`.
    pub final_offset: Option<i32>,
    /// The position increment after `end()`.
    pub final_pos_inc: Option<i32>,
    /// Keyword flags.
    pub keyword_atts: Option<Vec<bool>>,
    /// Whether the graph's offsets must be consistent (Java's default
    /// `true`).
    pub graph_offsets_are_correct: bool,
    /// Payloads; a `None` entry is unchecked (as Java's `null` entry).
    pub payloads: Option<Vec<Option<Vec<u8>>>>,
    /// Flags.
    pub flags: Option<Vec<i32>>,
    /// Boosts (compared within 0.001).
    pub boost: Option<Vec<f32>>,
}

impl Expected {
    /// Only the terms.
    pub fn terms(output: &[&str]) -> Self {
        Expected {
            output: output.iter().map(|s| s.to_string()).collect(),
            start_offsets: None,
            end_offsets: None,
            types: None,
            pos_increments: None,
            pos_lengths: None,
            final_offset: None,
            final_pos_inc: None,
            keyword_atts: None,
            graph_offsets_are_correct: true,
            payloads: None,
            flags: None,
            boost: None,
        }
    }

    /// With start and end offsets.
    pub fn offsets(mut self, start: &[i32], end: &[i32]) -> Self {
        self.start_offsets = Some(start.to_vec());
        self.end_offsets = Some(end.to_vec());
        self
    }

    /// With types.
    pub fn types(mut self, types: &[&str]) -> Self {
        self.types = Some(types.iter().map(|s| s.to_string()).collect());
        self
    }

    /// With position increments.
    pub fn pos_incs(mut self, incs: &[i32]) -> Self {
        self.pos_increments = Some(incs.to_vec());
        self
    }

    /// With position lengths.
    pub fn pos_lens(mut self, lens: &[i32]) -> Self {
        self.pos_lengths = Some(lens.to_vec());
        self
    }

    /// With the final offset.
    pub fn final_offset(mut self, off: i32) -> Self {
        self.final_offset = Some(off);
        self
    }

    /// With the final position increment.
    pub fn final_pos_inc(mut self, inc: i32) -> Self {
        self.final_pos_inc = Some(inc);
        self
    }

    /// With keyword flags.
    pub fn keywords(mut self, k: &[bool]) -> Self {
        self.keyword_atts = Some(k.to_vec());
        self
    }

    /// With or without the graph offset check.
    pub fn graph_offsets(mut self, correct: bool) -> Self {
        self.graph_offsets_are_correct = correct;
        self
    }

    /// With payloads.
    pub fn payloads(mut self, p: Vec<Option<Vec<u8>>>) -> Self {
        self.payloads = Some(p);
        self
    }

    /// With flags.
    pub fn flags(mut self, f: &[i32]) -> Self {
        self.flags = Some(f.to_vec());
        self
    }

    /// With boosts.
    pub fn boosts(mut self, b: &[f32]) -> Self {
        self.boost = Some(b.to_vec());
        self
    }
}

const BOGUS_PAYLOAD: [u8; 5] = [0x00, 0xdf, 0x12, 0xbd, 0x24];

/// Which attributes the expectation checks (Java's non-null `xAtt`s).
struct Checked {
    offset: bool,
    types: bool,
    pos_inc: bool,
    pos_len: bool,
    keyword: bool,
    payload: bool,
    flags: bool,
    boost: bool,
}

impl Checked {
    fn of(e: &Expected) -> Self {
        Checked {
            offset: e.start_offsets.is_some()
                || e.end_offsets.is_some()
                || e.final_offset.is_some(),
            types: e.types.is_some(),
            pos_inc: e.pos_increments.is_some() || e.final_pos_inc.is_some(),
            pos_len: e.pos_lengths.is_some(),
            keyword: e.keyword_atts.is_some(),
            payload: e.payloads.is_some(),
            flags: e.flags.is_some(),
            boost: e.boost.is_some(),
        }
    }

    /// Java's "extra safety": clear, then bogus values in every checked
    /// attribute, so a stage that fails to set one is caught.
    fn bogus(&self, atts: &mut AttributeSource, keyword: bool) {
        atts.clear_attributes();
        atts.set_term("bogusTerm");
        if self.offset {
            atts.set_offset(14584724, 24683243).expect("valid offsets");
        }
        if self.types {
            atts.set_token_type("bogusType");
        }
        if self.pos_inc {
            atts.set_position_increment(45987657)
                .expect("valid increment");
        }
        if self.pos_len {
            atts.set_position_length(45987653).expect("valid length");
        }
        if self.keyword {
            atts.set_keyword(keyword);
        }
        if self.payload {
            atts.set_payload(Some(BOGUS_PAYLOAD.to_vec()));
        }
        if self.flags {
            atts.set_flags(!0);
        }
        if self.boost {
            atts.set_boost(-1.0);
        }
        // reset it, because we called clearAttributes() before
        atts.add_custom::<CheckClearAttributes>()
            .get_and_reset_clear_called();
    }
}

fn clear_called(atts: &mut AttributeSource) -> bool {
    atts.add_custom::<CheckClearAttributes>()
        .get_and_reset_clear_called()
}

/// `assertTokenStreamContents`: consumes `ts` (reset through close) and
/// asserts it produces `expected`.
///
/// # Panics
/// On any mismatch (Java's `AssertionError`).
///
/// # Errors
/// What the stream returns.
pub fn assert_token_stream_contents(
    ts: &mut dyn TokenStream,
    expected: &Expected,
) -> Result<(), AnalysisError> {
    let e = expected;
    let checked = Checked::of(e);
    ts.attributes_mut().add_custom::<CheckClearAttributes>();
    // Maps position to the start/end offset:
    let mut pos_to_start_offset: HashMap<i32, i32> = HashMap::new();
    let mut pos_to_end_offset: HashMap<i32, i32> = HashMap::new();
    ts.reset()?;
    let mut pos = -1;
    let mut last_start_offset = 0;
    for (i, want) in e.output.iter().enumerate() {
        // extra safety to enforce that the state is not preserved, and
        // also assign bogus values
        checked.bogus(ts.attributes_mut(), i & 1 == 0);
        assert!(ts.increment_token()?, "token {i} does not exist");
        assert!(
            clear_called(ts.attributes_mut()),
            "clearAttributes() was not called correctly in TokenStream chain at token {i}"
        );
        let atts = ts.attributes();
        let term = atts.term();
        assert_eq!(term, want, "term {i}");
        if let Some(s) = &e.start_offsets {
            assert_eq!(atts.start_offset(), s[i], "startOffset {i} term={term}");
        }
        if let Some(s) = &e.end_offsets {
            assert_eq!(atts.end_offset(), s[i], "endOffset {i} term={term}");
        }
        if let Some(t) = &e.types {
            assert_eq!(atts.token_type(), t[i], "type {i} term={term}");
        }
        if let Some(p) = &e.pos_increments {
            assert_eq!(
                atts.position_increment(),
                p[i],
                "posIncrement {i} term={term}"
            );
        }
        if let Some(p) = &e.pos_lengths {
            assert_eq!(atts.position_length(), p[i], "posLength {i} term={term}");
        }
        if let Some(k) = &e.keyword_atts {
            assert_eq!(atts.is_keyword(), k[i], "keywordAtt {i} term={term}");
        }
        if let Some(f) = &e.flags {
            assert_eq!(atts.flags(), f[i], "flagsAtt {i} term={term}");
        }
        if let Some(b) = &e.boost {
            assert!(
                (atts.boost() - b[i]).abs() <= 0.001,
                "boostAtt {i} term={term}: expected {} got {}",
                b[i],
                atts.boost()
            );
        }
        if let Some(p) = &e.payloads {
            if let Some(want) = &p[i] {
                assert_eq!(atts.payload(), Some(&want[..]), "payloads {i}");
            }
        }
        if checked.pos_inc {
            if i == 0 {
                assert!(
                    atts.position_increment() >= 1,
                    "first posIncrement must be >= 1"
                );
            } else {
                assert!(atts.position_increment() >= 0, "posIncrement must be >= 0");
            }
        }
        if checked.pos_len {
            assert!(
                atts.position_length() >= 1,
                "posLength must be >= 1; got: {}",
                atts.position_length()
            );
        }
        // we can enforce some basic things about a few attributes even if
        // the caller doesn't check:
        if checked.offset {
            let start_offset = atts.start_offset();
            let end_offset = atts.end_offset();
            if let Some(final_offset) = e.final_offset {
                assert!(
                    start_offset <= final_offset,
                    "startOffset (= {start_offset}) must be <= finalOffset (= {final_offset}) term={term}"
                );
                assert!(
                    end_offset <= final_offset,
                    "endOffset must be <= finalOffset: got endOffset={end_offset} vs finalOffset={final_offset} term={term}"
                );
            }
            assert!(
                start_offset >= last_start_offset,
                "offsets must not go backwards startOffset={start_offset} is < lastStartOffset={last_start_offset} term={term}"
            );
            last_start_offset = start_offset;
            if e.graph_offsets_are_correct && checked.pos_len && checked.pos_inc {
                // Validate offset consistency in the graph: all tokens
                // leaving from a certain pos have the same startOffset, and
                // all tokens arriving to a certain pos the same endOffset.
                pos += atts.position_increment();
                let pos_length = atts.position_length();
                let old = *pos_to_start_offset.entry(pos).or_insert(start_offset);
                assert_eq!(
                    old, start_offset,
                    "{i} inconsistent startOffset: pos={pos} posLen={pos_length} token={term}"
                );
                let end_pos = pos + pos_length;
                let old = *pos_to_end_offset.entry(end_pos).or_insert(end_offset);
                assert_eq!(
                    old, end_offset,
                    "inconsistent endOffset {i} pos={pos} posLen={pos_length} token={term}"
                );
            }
        }
    }
    if ts.increment_token()? {
        panic!(
            "TokenStream has more tokens than expected (expected count={}); extra token={}",
            e.output.len(),
            ts.attributes().term()
        );
    }
    // repeat our extra safety checks for end()
    checked.bogus(ts.attributes_mut(), true);
    ts.end()?;
    assert!(
        clear_called(ts.attributes_mut()),
        "super.end()/clearAttributes() was not called correctly in end()"
    );
    let atts = ts.attributes();
    if let Some(final_offset) = e.final_offset {
        assert_eq!(atts.end_offset(), final_offset, "finalOffset");
    }
    if checked.offset {
        assert!(atts.end_offset() >= 0, "finalOffset must be >= 0");
    }
    if let Some(final_pos_inc) = e.final_pos_inc {
        assert_eq!(atts.position_increment(), final_pos_inc, "finalPosInc");
    }
    ts.close()
}

/// Java's `String.length()`.
fn utf16_len(s: &str) -> i32 {
    s.encode_utf16().count() as i32
}

/// `assertAnalyzesTo`: `a`'s tokens for `input` in field "dummy" (final
/// offset the input's length), then [`check_reset_exception`] and
/// [`check_analysis_consistency`] (with a char filter) with `random`.
///
/// # Panics
/// On a mismatch.
///
/// # Errors
/// What the analyzer returns.
pub fn assert_analyzes_to(
    random: &mut JavaRandom,
    a: &Analyzer,
    input: &str,
    expected: &Expected,
) -> Result<(), AnalysisError> {
    let mut e = expected.clone();
    e.final_offset = Some(utf16_len(input));
    assert_token_stream_contents(&mut a.token_stream("dummy", input)?, &e)?;
    check_reset_exception(a, input)?;
    check_analysis_consistency(random, a, true, input, e.graph_offsets_are_correct)
}

/// `checkOneTerm`: `input` analyzes to the single term `expected`.
///
/// # Errors
/// What the analyzer returns.
pub fn check_one_term(
    random: &mut JavaRandom,
    a: &Analyzer,
    input: &str,
    expected: &str,
) -> Result<(), AnalysisError> {
    assert_analyzes_to(random, a, input, &Expected::terms(&[expected]))
}

/// `checkResetException`: `incrementToken` without `reset` must not return
/// a token (Java expects an `IllegalStateException` or `false`); then the
/// stream is consumed correctly.
///
/// # Panics
/// When a token comes back, or the error is not an `IllegalState` one.
///
/// # Errors
/// What the correct consumption returns.
pub fn check_reset_exception(a: &Analyzer, input: &str) -> Result<(), AnalysisError> {
    let mut ts = a.token_stream("bogus", input)?;
    match ts.increment_token() {
        Ok(true) => panic!("didn't get expected exception when reset() not called"),
        Ok(false) | Err(AnalysisError::IllegalState(_)) => {}
        Err(unexpected) => panic!("got wrong exception when reset() not called: {unexpected}"),
    }
    // consume correctly
    ts.reset()?;
    while ts.increment_token()? {}
    ts.end()?;
    ts.close()?;
    drop(ts);

    // check for a missing close(): Java's next tokenStream throws; a
    // dropped stream closes itself here, so the next one must work.
    let mut ts = a.token_stream("bogus", input)?;
    ts.reset()?;
    while ts.increment_token()? {}
    ts.end()?;
    drop(ts);
    let mut ts = a.token_stream("bogus", input)?;
    ts.reset()?;
    while ts.increment_token()? {}
    ts.end()?;
    ts.close()
}

/// The `AttributeFactory` [`new_attribute_factory`] picks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttributeFactoryKind {
    /// `TokenStream.DEFAULT_TOKEN_ATTRIBUTE_FACTORY`.
    TokenStreamDefault,
    /// `Token.TOKEN_ATTRIBUTE_FACTORY`.
    Token,
    /// `AttributeFactory.DEFAULT_ATTRIBUTE_FACTORY`.
    Default,
}

/// `newAttributeFactory(random)`: one `nextInt(3)` choosing among Java's
/// three factories. Each builds the same [`AttributeSource`] here (the
/// port's attributes are one fixed struct), so a caller uses the draw to
/// stay in step with a Java seed, not to vary attributes.
pub fn new_attribute_factory(random: &mut JavaRandom) -> AttributeFactoryKind {
    match random.next_int_bounded(3) {
        0 => AttributeFactoryKind::TokenStreamDefault,
        1 => AttributeFactoryKind::Token,
        _ => AttributeFactoryKind::Default,
    }
}

/// `assertStreamHasNumberOfTokens`.
///
/// # Errors
/// What the stream returns.
pub fn assert_stream_has_number_of_tokens(
    ts: &mut dyn TokenStream,
    expected_count: usize,
) -> Result<(), AnalysisError> {
    ts.reset()?;
    let mut count = 0;
    while ts.increment_token()? {
        count += 1;
    }
    ts.end()?;
    assert_eq!(count, expected_count, "wrong number of tokens");
    Ok(())
}

pub(crate) use crate::util::lucene_test_case::knobs;

/// `checkRandomData(random, a, iterations)`: max word length 20, not
/// simple, graph offsets checked.
///
/// # Errors
/// What the analyzer returns.
pub fn check_random_data(
    random: &mut JavaRandom,
    a: &Analyzer,
    iterations: i32,
) -> Result<(), AnalysisError> {
    check_random_data_with(random, a, iterations, 20, false, true)
}

/// `checkRandomData(random, a, iterations, maxWordLength, simple,
/// graphOffsetsAreCorrect)`: [`check_reset_exception`], then `iterations`
/// random strings through [`check_analysis_consistency`], once on this
/// thread and then again, the same strings, on 2-4 threads at once.
///
/// # Panics
/// On an assertion failure (on any thread).
///
/// # Errors
/// What the analyzer returns.
pub fn check_random_data_with(
    random: &mut JavaRandom,
    a: &Analyzer,
    iterations: i32,
    max_word_length: i32,
    simple: bool,
    graph_offsets_are_correct: bool,
) -> Result<(), AnalysisError> {
    check_reset_exception(a, "best effort")?;
    let seed = random.next_long();
    let use_char_filter = random.next_boolean();
    let (nightly, multiplier) = knobs();
    // Java indexes the text through a RandomIndexWriter when this holds
    // (`codecOk`: no SimpleText here, the port has no randomized codec).
    let codec_ok = i64::from(iterations) * i64::from(max_word_length) < 100_000;
    let index = lucene_test_case::rarely(random, nightly, multiplier) && codec_ok;
    let config = RandomRun {
        iterations,
        max_word_length,
        use_char_filter,
        simple,
        graph_offsets_are_correct,
    };
    let tmp = index.then(|| TempDir::new("bttc"));
    let dir = tmp.as_ref().map(|t| FsDirectory::open(t.path()));
    let iw = dir
        .as_ref()
        .map(|dir| Mutex::new(random_index_writer(seed, a, dir)));
    let run = || check_random_data_inner(&mut JavaRandom::new(seed), a, &config, iw.as_ref());
    run()?;
    // now test with multiple threads: each does the EXACT same thing, so
    // this should only fail from another thread on a real thread problem
    let num_threads = test_util::next_int(random, 2, 4);
    std::thread::scope(|s| {
        let handles: Vec<_> = (0..num_threads).map(|_| s.spawn(run)).collect();
        let mut result = Ok(());
        let mut failed = false;
        for h in handles {
            match h.join() {
                Ok(Ok(())) => {}
                Ok(Err(e)) => result = result.and(Err(e)),
                Err(_) => failed = true,
            }
        }
        assert!(!failed, "some thread(s) failed");
        result
    })?;
    if let (Some(iw), Some(dir)) = (iw, dir.as_ref()) {
        close_and_check(iw, dir);
    }
    Ok(())
}

/// `new RandomIndexWriter(new Random(seed), dir, a)`: `Random(seed)` makes
/// only the writer's draws, as Java's. The texts arrive pre-analyzed, so the
/// writer's analyzer contributes only `a`'s gaps on the `dummy` field; it is
/// a stand-in carrying them (an `Analyzer` is not shared), built without
/// touching the writer's stream.
fn random_index_writer<'d>(seed: i64, a: &Analyzer, dir: &'d FsDirectory) -> RandomIndexWriter<'d> {
    let stand_in = MockAnalyzer::new(&mut JavaRandom::new(seed))
        .build()
        .with_position_increment_gap(a.position_increment_gap_for_field("dummy"))
        .with_offset_gap(a.offset_gap_for_field("dummy"));
    RandomIndexWriter::with_analyzer(
        &mut JavaRandom::new(seed),
        dir,
        Arc::new(stand_in),
        WriterEnv::new(seed),
    )
    .expect("open the RandomIndexWriter")
}

/// `iw.close()`, then the `CheckIndex` Java's mock directory runs on close.
fn close_and_check(iw: Mutex<RandomIndexWriter<'_>>, dir: &FsDirectory) {
    iw.into_inner()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .close()
        .expect("close the RandomIndexWriter");
    test_util::check_index(dir).expect("CheckIndex over the indexed random text");
}

/// Prints Java's "TEST FAIL" line when an assertion unwinds through it.
struct FailNote<'a> {
    use_char_filter: bool,
    text: &'a str,
}

impl Drop for FailNote<'_> {
    fn drop(&mut self) {
        if std::thread::panicking() {
            eprintln!(
                "TEST FAIL: useCharFilter={} text='{}'",
                self.use_char_filter,
                escape(self.text)
            );
        }
    }
}

/// One `checkRandomData` run's settings, shared by its threads.
struct RandomRun {
    iterations: i32,
    max_word_length: i32,
    use_char_filter: bool,
    simple: bool,
    graph_offsets_are_correct: bool,
}

/// The `dummy` field's type Java draws when it indexes.
fn random_field_type(random: &mut JavaRandom) -> FieldType {
    let set = |r: lucene_index::document::Result<()>| r.expect("an unfrozen FieldType");
    let mut ft = FieldType::copy_of(&TextField::type_not_stored());
    if random.next_boolean() {
        set(ft.set_store_term_vectors(true));
        set(ft.set_store_term_vector_offsets(random.next_boolean()));
        set(ft.set_store_term_vector_positions(random.next_boolean()));
        if ft.store_term_vector_positions() {
            set(ft.set_store_term_vector_payloads(random.next_boolean()));
        }
    }
    if random.next_boolean() {
        set(ft.set_omit_norms(true));
    }
    set(ft.set_index_options(match random.next_int_bounded(4) {
        0 => IndexOptions::Docs,
        1 => IndexOptions::DocsAndFreqs,
        2 => IndexOptions::DocsAndFreqsAndPositions,
        _ => IndexOptions::DocsAndFreqsAndPositionsAndOffsets,
    }));
    ft
}

/// What `field.setReaderValue(..)` hands the indexing chain: `text` through
/// the spoon-feeding reader `Random(seed)` picks and the char filter, run
/// through `a` as `IndexingChain.invertTokenStream` consumes it.
fn index_tokens(
    a: &Analyzer,
    text: &str,
    seed: i64,
    use_char_filter: bool,
    remainder: i32,
) -> Result<FieldTokens, AnalysisError> {
    let mut feed = JavaRandom::new(seed);
    let mut reader: Box<dyn CharReader> = Box::new(StrReader::new(text));
    if feed.next_int_bounded(30) == 7 {
        reader = Box::new(MockReaderWrapper::new(Arc::new(Mutex::new(feed)), reader));
    }
    let reader = maybe_char_filter(reader, use_char_filter, remainder)?;
    let mut ts = a.token_stream_from_reader("dummy", reader)?;
    ts.reset()?;
    let mut tokens = Vec::new();
    while ts.increment_token()? {
        let atts = ts.attributes();
        tokens.push(FieldToken {
            term: atts.term_bytes().to_vec(),
            start_offset: atts.start_offset(),
            end_offset: atts.end_offset(),
            position_increment: atts.position_increment(),
            term_frequency: atts.term_frequency(),
            payload: atts.payload().map(<[u8]>::to_vec),
        });
    }
    ts.end()?;
    let end = ts.attributes().clone();
    ts.close()?;
    Ok(FieldTokens {
        tokens,
        final_position_increment: end.position_increment(),
        final_offset: end.end_offset(),
        end_attributes: Some(end),
    })
}

/// The document Java's `doc` holds: one `dummy` field per piled-up value.
fn dummy_document(ft: &FieldType, values: &[FieldTokens]) -> Document {
    let mut doc = Document::new();
    for v in values {
        doc.add(
            Field::from_token_stream("dummy", v.clone(), ft.clone())
                .expect("an indexed, tokenized, unstored type"),
        );
    }
    doc
}

fn check_random_data_inner(
    random: &mut JavaRandom,
    a: &Analyzer,
    c: &RandomRun,
    iw: Option<&Mutex<RandomIndexWriter<'_>>>,
) -> Result<(), AnalysisError> {
    // Java's `doc`: the field type, and each field's (reader) value; the
    // last is `currentField`, the one each text sets.
    let ft = iw.map(|_| random_field_type(random));
    let mut values = vec![FieldTokens::default()];
    for _ in 0..c.iterations {
        let text = test_util::random_analysis_string(random, c.max_word_length, c.simple);
        let note = FailNote {
            use_char_filter: c.use_char_filter,
            text: &text,
        };
        let (seed, remainder) = match consistency(
            random,
            a,
            c.use_char_filter,
            &text,
            c.graph_offsets_are_correct,
        ) {
            Ok(drawn) => drawn,
            Err(e) => {
                eprintln!(
                    "TEST FAIL: useCharFilter={} text='{}'",
                    c.use_char_filter,
                    escape(&text)
                );
                return Err(e);
            }
        };
        if let (Some(iw), Some(ft)) = (iw, ft.as_ref()) {
            let last = values.len() - 1;
            values[last] = index_tokens(a, &text, seed, c.use_char_filter, remainder)?;
            if random.next_int_bounded(7) == 0 {
                // pile up a multivalued field
                values.push(FieldTokens::default());
            } else {
                lock_writer(iw)
                    .add_document(&dummy_document(ft, &values))
                    .expect("index the random text");
                // back to 1 field
                values.truncate(1);
            }
        }
        drop(note);
    }
    Ok(())
}

/// The shared writer (Java's `RandomIndexWriter` is synchronized).
fn lock_writer<'m, 'd>(
    iw: &'m Mutex<RandomIndexWriter<'d>>,
) -> std::sync::MutexGuard<'m, RandomIndexWriter<'d>> {
    iw.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// `escape`: `s` with newlines, quotes, backslashes and non-ASCII UTF-16
/// units escaped Java-style (`\uXXXX`).
pub fn escape(s: &str) -> String {
    let mut sb = String::new();
    for c in s.encode_utf16() {
        match c {
            0xa => sb.push_str("\\n"),
            0xd => sb.push_str("\\r"),
            0x22 => sb.push_str("\\\""),
            0x5c => sb.push_str("\\\\"),
            0x20..=0x7f => sb.push(char::from(c as u8)),
            _ => sb.push_str(&format!("\\u{c:04x}")),
        }
    }
    sb
}

/// A reader over `reader`, through a [`MockCharFilter`] when asked.
fn maybe_char_filter(
    reader: Box<dyn CharReader>,
    use_char_filter: bool,
    remainder: i32,
) -> Result<Box<dyn CharReader>, AnalysisError> {
    Ok(if use_char_filter {
        Box::new(MockCharFilter::new(reader, remainder)?)
    } else {
        reader
    })
}

/// `end()` after an interrupted consumption: `MockTokenizer`'s complaint
/// about the state is expected and ignored.
fn end_ignoring_mock_anger(ts: &mut dyn TokenStream) -> Result<(), AnalysisError> {
    // SENTINEL-OK: `TokenStream::end` returns a `Result`, not a position.
    match ts.end() {
        Err(AnalysisError::IllegalState(m)) if m.contains("end() called in wrong state=") => Ok(()),
        other => other,
    }
}

/// `checkAnalysisConsistency`: analyzes `text` once to record the tokens,
/// maybe re-analyzes it with something evil (a reader that fails part way,
/// or a consumer that stops early), then asserts a clean re-analysis (maybe
/// through a spoon-feeding reader) matches the first, and normalizes the
/// text.
///
/// # Panics
/// On a mismatch.
///
/// # Errors
/// What the analyzer returns.
pub fn check_analysis_consistency(
    random: &mut JavaRandom,
    a: &Analyzer,
    use_char_filter: bool,
    text: &str,
    graph_offsets_are_correct: bool,
) -> Result<(), AnalysisError> {
    consistency(random, a, use_char_filter, text, graph_offsets_are_correct).map(drop)
}

/// [`check_analysis_consistency`], answering the final pass's reader seed
/// and the char filter's remainder: what Java's `field.setReaderValue`
/// rebuilds the reader from.
fn consistency(
    random: &mut JavaRandom,
    a: &Analyzer,
    use_char_filter: bool,
    text: &str,
    graph_offsets_are_correct: bool,
) -> Result<(i64, i32), AnalysisError> {
    let remainder = random.next_int_bounded(10);
    let reader = maybe_char_filter(Box::new(StrReader::new(text)), use_char_filter, remainder)?;
    let mut ts = a.token_stream_from_reader("dummy", reader)?;
    let mut tokens = Vec::new();
    let mut types = Vec::new();
    let mut positions = Vec::new();
    let mut position_lengths = Vec::new();
    let mut start_offsets = Vec::new();
    let mut end_offsets = Vec::new();
    ts.reset()?;
    // First pass: save away "correct" tokens
    while ts.increment_token()? {
        let atts = ts.attributes();
        tokens.push(atts.term().to_string());
        types.push(atts.token_type().to_string());
        positions.push(atts.position_increment());
        position_lengths.push(atts.position_length());
        start_offsets.push(atts.start_offset());
        end_offsets.push(atts.end_offset());
    }
    ts.end()?;
    ts.close()?;
    drop(ts);

    // verify reusing is "reproducible" and also get the normal token
    // stream sanity checks; KeywordTokenizer (for example) can produce a
    // token even when the input is empty
    if !tokens.is_empty() && !text.is_empty() {
        // (Optional) second pass: do something evil:
        let evilness = random.next_int_bounded(50);
        if evilness == 17 {
            // Throw an errant exception from the Reader; the wrapper
            // draws from the caller's random, as Java's shares it
            let shared = Arc::new(Mutex::new(random.clone()));
            let mut evil =
                MockReaderWrapper::new(Arc::clone(&shared), Box::new(StrReader::new(text)));
            let after = super::mock_filters::lock(&shared).next_int_bounded(utf16_len(text) + 1);
            evil.throw_exc_after_char(after);
            // some tokenizers read on setReader, so the stream is created
            // inside the "try"
            let result = maybe_char_filter(Box::new(evil), use_char_filter, remainder)
                .and_then(|r| a.token_stream_from_reader("dummy", r))
                .and_then(|mut ts| {
                    let consumed = ts.reset().and_then(|()| {
                        while ts.increment_token()? {}
                        Ok(())
                    });
                    match consumed {
                        Ok(()) => panic!("did not hit exception"),
                        Err(e) => {
                            assert!(MockReaderWrapper::is_my_evil_exception(&e), "{e}");
                        }
                    }
                    end_ignoring_mock_anger(&mut ts)?;
                    ts.close()
                });
            match result {
                Ok(()) => {}
                Err(e) => assert!(MockReaderWrapper::is_my_evil_exception(&e), "{e}"),
            }
            *random = super::mock_filters::lock(&shared).clone();
        } else if evilness == 7 {
            // Only consume a subset of the tokens:
            let num_tokens_to_read = random.next_int_bounded(tokens.len() as i32);
            let reader =
                maybe_char_filter(Box::new(StrReader::new(text)), use_char_filter, remainder)?;
            let mut ts = a.token_stream_from_reader("dummy", reader)?;
            ts.reset()?;
            for _ in 0..num_tokens_to_read {
                assert!(ts.increment_token()?);
            }
            end_ignoring_mock_anger(&mut ts)?;
            ts.close()?;
        }
    }

    // Final pass: verify clean tokenization matches results from the first
    let seed = random.next_long();
    let mut feed = JavaRandom::new(seed);
    let mut reader: Box<dyn CharReader> = Box::new(StrReader::new(text));
    if feed.next_int_bounded(30) == 7 {
        reader = Box::new(MockReaderWrapper::new(Arc::new(Mutex::new(feed)), reader));
    }
    let reader = maybe_char_filter(reader, use_char_filter, remainder)?;
    let mut ts = a.token_stream_from_reader("dummy", reader)?;
    // offset + pos + posLength + type: every attribute is present
    let expected = Expected {
        output: tokens,
        start_offsets: Some(start_offsets),
        end_offsets: Some(end_offsets),
        types: Some(types),
        pos_increments: Some(positions),
        pos_lengths: Some(position_lengths),
        final_offset: Some(utf16_len(text)),
        ..Expected::terms(&[]).graph_offsets(graph_offsets_are_correct)
    };
    assert_token_stream_contents(&mut ts, &expected)?;
    drop(ts);
    // TODO (as Java): what can we do besides testing this does not fail?
    a.normalize("dummy", text)?;
    Ok((seed, remainder))
}

/// `whitespaceMockTokenizer(input)`: not lowercased.
///
/// # Errors
/// What `set_reader` returns.
pub fn whitespace_mock_tokenizer(input: &str) -> Result<MockTokenizer, AnalysisError> {
    MockTokenizer::new(mock_tokenizer::whitespace(), false).with_text(input)
}

/// `whitespaceMockTokenizer(Reader)`.
///
/// # Errors
/// What `set_reader` returns.
pub fn whitespace_mock_tokenizer_from(
    input: Box<dyn CharReader>,
) -> Result<MockTokenizer, AnalysisError> {
    let mut t = MockTokenizer::new(mock_tokenizer::whitespace(), false);
    t.set_reader(input)?;
    Ok(t)
}

/// `keywordMockTokenizer(input)`.
///
/// # Errors
/// What `set_reader` returns.
pub fn keyword_mock_tokenizer(input: &str) -> Result<MockTokenizer, AnalysisError> {
    MockTokenizer::new(mock_tokenizer::keyword(), false).with_text(input)
}

/// `keywordMockTokenizer(Reader)`.
///
/// # Errors
/// What `set_reader` returns.
pub fn keyword_mock_tokenizer_from(
    input: Box<dyn CharReader>,
) -> Result<MockTokenizer, AnalysisError> {
    let mut t = MockTokenizer::new(mock_tokenizer::keyword(), false);
    t.set_reader(input)?;
    Ok(t)
}

/// `getGraphStrings(TokenStream)`: every path through the stream's token
/// graph, terms joined by spaces.
///
/// # Errors
/// What the stream returns.
pub fn get_graph_strings(ts: &mut dyn TokenStream) -> Result<HashSet<String>, AnalysisError> {
    let automaton = token_stream_to_automaton(ts)?;
    let sep = char::from_u32(POS_SEP as u32).expect("POS_SEP is a char");
    Ok(get_finite_strings_recursive(&automaton, -1)
        .into_iter()
        .map(|ir| {
            let bytes: Vec<u8> = ir.iter().map(|&b| b as u8).collect();
            String::from_utf8_lossy(&bytes).replace(sep, " ")
        })
        .collect())
}

/// `getGraphStrings(Analyzer, String)`.
///
/// # Errors
/// What the analyzer returns.
pub fn get_graph_strings_of(a: &Analyzer, text: &str) -> Result<HashSet<String>, AnalysisError> {
    get_graph_strings(&mut a.token_stream("dummy", text)?)
}

fn sorted_lines(strings: &HashSet<String>) -> String {
    let sorted: BTreeSet<_> = strings.iter().collect();
    sorted.iter().map(|s| format!("  {s}\n")).collect()
}

/// `assertGraphStrings(TokenStream, expected...)`: the stream's graph has
/// exactly these paths.
///
/// # Panics
/// On a missing or unexpected path.
///
/// # Errors
/// What the stream returns.
pub fn assert_graph_strings(
    ts: &mut dyn TokenStream,
    expected: &[&str],
) -> Result<(), AnalysisError> {
    let actual = get_graph_strings(ts)?;
    let expected: HashSet<String> = expected.iter().map(|s| s.to_string()).collect();
    let mut stray = actual
        .iter()
        .filter(|s| !expected.contains(*s))
        .chain(expected.iter().filter(|s| !actual.contains(*s)));
    if let Some(s) = stray.next() {
        panic!(
            "Analyzer created unexpected string path: {s}\nexpected:\n{}\nactual:\n{}",
            sorted_lines(&expected),
            sorted_lines(&actual)
        );
    }
    Ok(())
}

/// `assertGraphStrings(Analyzer, text, expected...)`: first
/// [`check_analysis_consistency`] (with a char filter), then the paths.
///
/// # Errors
/// What the analyzer returns.
pub fn assert_graph_strings_of(
    random: &mut JavaRandom,
    a: &Analyzer,
    text: &str,
    expected: &[&str],
) -> Result<(), AnalysisError> {
    check_analysis_consistency(random, a, true, text, true)?;
    assert_graph_strings(&mut a.token_stream("dummy", text)?, expected)
}

/// `toString(Analyzer, text)`: one line per token, `term at pos=P to
/// pos=Q offsets=S-E`.
///
/// # Errors
/// What the analyzer returns.
pub fn to_string(a: &Analyzer, text: &str) -> Result<String, AnalysisError> {
    let mut ts = a.token_stream("field", text)?;
    let mut b = String::new();
    ts.reset()?;
    let mut pos = -1;
    while ts.increment_token()? {
        let atts = ts.attributes();
        pos += atts.position_increment();
        b.push_str(&format!(
            "{} at pos={pos} to pos={} offsets={}-{}\n",
            atts.term(),
            pos + atts.position_length(),
            atts.start_offset(),
            atts.end_offset()
        ));
    }
    ts.end()?;
    Ok(b)
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use lucene_analysis::attributes::AttributeSource;
    use lucene_analysis::{AnalyzerDefinition, TokenStreamComponents, TokenizerInput};

    use super::*;
    use crate::analysis::mock_analyzer::{MockAnalyzer, MockSynonymAnalyzer};

    /// One scripted token.
    #[derive(Clone)]
    struct T {
        term: &'static str,
        start: i32,
        end: i32,
        inc: i32,
        len: i32,
        ty: &'static str,
        keyword: bool,
        flags: i32,
        boost: f32,
        payload: Option<Vec<u8>>,
    }

    fn t(term: &'static str, start: i32, end: i32, inc: i32, len: i32) -> T {
        T {
            term,
            start,
            end,
            inc,
            len,
            ty: "word",
            keyword: false,
            flags: 0,
            boost: 1.0,
            payload: None,
        }
    }

    /// What `incrementToken` does before `reset`.
    #[derive(Clone, Copy)]
    enum NoReset {
        Done,
        Token,
        WrongError,
        Contract,
    }

    /// A tokenizer replaying `toks`, with switches for each contract
    /// breach the assertions look for.
    struct Scripted {
        atts: AttributeSource,
        input: TokenizerInput,
        toks: Vec<T>,
        upto: usize,
        clear: bool,
        end_clear: bool,
        final_offset: i32,
        reset_called: bool,
        no_reset: NoReset,
        /// Resets so far, shared across the analyzer's streams.
        resets: Arc<AtomicUsize>,
        /// From this many resets on, `incrementToken` fails.
        fail_from_reset: usize,
        /// Suffix the term with the reset count (an inconsistent analyzer).
        count_terms: bool,
    }

    fn scripted(toks: Vec<T>, final_offset: i32) -> Scripted {
        Scripted {
            atts: AttributeSource::new(),
            input: TokenizerInput::new(),
            toks,
            upto: 0,
            clear: true,
            end_clear: true,
            final_offset,
            reset_called: false,
            no_reset: NoReset::Done,
            resets: Arc::new(AtomicUsize::new(0)),
            fail_from_reset: usize::MAX,
            count_terms: false,
        }
    }

    impl TokenStream for Scripted {
        fn attributes(&self) -> &AttributeSource {
            &self.atts
        }

        fn attributes_mut(&mut self) -> &mut AttributeSource {
            &mut self.atts
        }

        fn increment_token(&mut self) -> Result<bool, AnalysisError> {
            if !self.reset_called {
                match self.no_reset {
                    NoReset::Done => return Ok(false),
                    NoReset::Contract => return Err(AnalysisError::IllegalState("no".into())),
                    NoReset::WrongError => return Err(AnalysisError::Io("odd".into())),
                    NoReset::Token => {}
                }
            }
            if self.resets.load(Ordering::SeqCst) >= self.fail_from_reset {
                return Err(AnalysisError::Io("scripted failure".into()));
            }
            let Some(tok) = self.toks.get(self.upto).cloned() else {
                return Ok(false);
            };
            self.upto += 1;
            let a = &mut self.atts;
            if self.clear {
                a.clear_attributes();
            }
            if self.count_terms {
                a.set_term(&format!(
                    "{}{}",
                    tok.term,
                    self.resets.load(Ordering::SeqCst)
                ));
            } else {
                a.set_term(tok.term);
            }
            a.set_offset(tok.start, tok.end)?;
            a.set_position_increment(tok.inc)?;
            a.set_position_length(tok.len)?;
            a.set_token_type(tok.ty);
            a.set_keyword(tok.keyword);
            a.set_flags(tok.flags);
            a.set_boost(tok.boost);
            a.set_payload(tok.payload);
            Ok(true)
        }

        fn reset(&mut self) -> Result<(), AnalysisError> {
            self.input.reset();
            self.upto = 0;
            self.reset_called = true;
            self.resets.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }

        fn end(&mut self) -> Result<(), AnalysisError> {
            if self.end_clear {
                self.atts.end_attributes();
            }
            self.atts.set_offset(self.final_offset, self.final_offset)
        }

        fn close(&mut self) -> Result<(), AnalysisError> {
            self.reset_called = false;
            self.input.close()
        }

        fn as_tokenizer(&mut self) -> Option<&mut dyn Tokenizer> {
            Some(self)
        }

        fn conditional_root(&mut self) -> Option<&mut dyn std::any::Any> {
            None
        }
    }

    impl Tokenizer for Scripted {
        fn set_reader(&mut self, input: Box<dyn CharReader>) -> Result<(), AnalysisError> {
            self.input.set_reader(input)
        }
    }

    /// An analyzer whose every stream is `make()`.
    struct Def(Box<dyn Fn() -> Scripted + Send + Sync>);

    impl AnalyzerDefinition for Def {
        fn create_components(&self, _: &str) -> Result<TokenStreamComponents, AnalysisError> {
            Ok(TokenStreamComponents::new((self.0)()))
        }
    }

    fn analyzer(make: impl Fn() -> Scripted + Send + Sync + 'static) -> Analyzer {
        Analyzer::new(Def(Box::new(make)))
    }

    fn rich() -> Vec<T> {
        vec![
            T {
                ty: "alpha",
                keyword: true,
                flags: 3,
                boost: 2.0,
                payload: Some(vec![1]),
                ..t("a", 0, 1, 1, 1)
            },
            T {
                payload: Some(vec![2]),
                ..t("b", 2, 3, 1, 1)
            },
        ]
    }

    fn rich_expected() -> Expected {
        Expected::terms(&["a", "b"])
            .offsets(&[0, 2], &[1, 3])
            .types(&["alpha", "word"])
            .pos_incs(&[1, 1])
            .pos_lens(&[1, 1])
            .final_offset(3)
            .final_pos_inc(0)
            .keywords(&[true, false])
            .graph_offsets(true)
            .payloads(vec![Some(vec![1]), None])
            .flags(&[3, 0])
            .boosts(&[2.0, 1.0])
    }

    fn run(toks: Vec<T>, e: Expected) {
        let mut s = scripted(toks, 3);
        assert_token_stream_contents(&mut s, &e).unwrap();
    }

    fn run_rich(edit: impl FnOnce(&mut Expected)) {
        let mut e = rich_expected();
        edit(&mut e);
        run(rich(), e);
    }

    #[test]
    fn every_attribute_checked_and_matching() {
        run_rich(|_| {});
    }

    #[test]
    #[should_panic(expected = "term 0")]
    fn wrong_term() {
        run_rich(|e| e.output[0] = "x".into());
    }

    #[test]
    #[should_panic(expected = "startOffset 0")]
    fn wrong_start_offset() {
        run_rich(|e| e.start_offsets = Some(vec![1, 2]));
    }

    #[test]
    #[should_panic(expected = "endOffset 1")]
    fn wrong_end_offset() {
        run_rich(|e| e.end_offsets = Some(vec![1, 2]));
    }

    #[test]
    #[should_panic(expected = "type 1")]
    fn wrong_type() {
        run_rich(|e| e.types = Some(vec!["alpha".into(), "x".into()]));
    }

    #[test]
    #[should_panic(expected = "posIncrement 0")]
    fn wrong_position_increment() {
        run_rich(|e| e.pos_increments = Some(vec![2, 1]));
    }

    #[test]
    #[should_panic(expected = "posLength 0")]
    fn wrong_position_length() {
        run_rich(|e| e.pos_lengths = Some(vec![2, 1]));
    }

    #[test]
    #[should_panic(expected = "keywordAtt 0")]
    fn wrong_keyword() {
        run_rich(|e| e.keyword_atts = Some(vec![false, false]));
    }

    #[test]
    #[should_panic(expected = "flagsAtt 0")]
    fn wrong_flags() {
        run_rich(|e| e.flags = Some(vec![1, 0]));
    }

    #[test]
    #[should_panic(expected = "boostAtt 0")]
    fn wrong_boost() {
        run_rich(|e| e.boost = Some(vec![1.0, 1.0]));
    }

    #[test]
    #[should_panic(expected = "payloads 0")]
    fn wrong_payload() {
        run_rich(|e| e.payloads = Some(vec![Some(vec![9]), None]));
    }

    #[test]
    #[should_panic(expected = "finalOffset")]
    fn wrong_final_offset() {
        run_rich(|e| e.final_offset = Some(4));
    }

    #[test]
    #[should_panic(expected = "finalPosInc")]
    fn wrong_final_position_increment() {
        run_rich(|e| e.final_pos_inc = Some(1));
    }

    #[test]
    #[should_panic(expected = "first posIncrement must be >= 1")]
    fn first_increment_zero() {
        run(
            vec![t("a", 0, 1, 0, 1)],
            Expected::terms(&["a"]).pos_incs(&[0]),
        );
    }

    #[test]
    #[should_panic(expected = "startOffset (= 5) must be <= finalOffset (= 3)")]
    fn start_past_final_offset() {
        run(
            vec![t("a", 5, 6, 1, 1)],
            Expected::terms(&["a"]).final_offset(3),
        );
    }

    #[test]
    #[should_panic(expected = "endOffset must be <= finalOffset")]
    fn end_past_final_offset() {
        run(
            vec![t("a", 1, 5, 1, 1)],
            Expected::terms(&["a"]).final_offset(3),
        );
    }

    #[test]
    #[should_panic(expected = "offsets must not go backwards")]
    fn offsets_going_backwards() {
        run(
            vec![t("a", 2, 3, 1, 1), t("b", 0, 1, 1, 1)],
            Expected::terms(&["a", "b"]).offsets(&[2, 0], &[3, 1]),
        );
    }

    #[test]
    #[should_panic(expected = "inconsistent startOffset")]
    fn inconsistent_start_offsets() {
        run(
            vec![t("a", 0, 1, 1, 1), t("b", 1, 2, 0, 1)],
            Expected::terms(&["a", "b"])
                .offsets(&[0, 1], &[1, 2])
                .pos_incs(&[1, 0])
                .pos_lens(&[1, 1]),
        );
    }

    #[test]
    #[should_panic(expected = "inconsistent endOffset")]
    fn inconsistent_end_offsets() {
        run(
            vec![t("a", 0, 2, 1, 2), t("b", 1, 3, 1, 1)],
            Expected::terms(&["a", "b"])
                .offsets(&[0, 1], &[2, 3])
                .pos_incs(&[1, 1])
                .pos_lens(&[2, 1]),
        );
    }

    #[test]
    fn inconsistent_offsets_pass_when_the_graph_is_known_wrong() {
        run(
            vec![t("a", 0, 1, 1, 1), t("b", 1, 2, 0, 1)],
            Expected::terms(&["a", "b"])
                .offsets(&[0, 1], &[1, 2])
                .pos_incs(&[1, 0])
                .pos_lens(&[1, 1])
                .graph_offsets(false),
        );
    }

    #[test]
    #[should_panic(expected = "more tokens than expected (expected count=1); extra token=b")]
    fn extra_token() {
        run(rich(), Expected::terms(&["a"]));
    }

    #[test]
    #[should_panic(expected = "token 2 does not exist")]
    fn missing_token() {
        run(rich(), Expected::terms(&["a", "b", "c"]));
    }

    #[test]
    #[should_panic(
        expected = "clearAttributes() was not called correctly in TokenStream chain at token 0"
    )]
    fn token_without_clear() {
        let mut s = scripted(rich(), 3);
        s.clear = false;
        assert_token_stream_contents(&mut s, &Expected::terms(&["a", "b"])).unwrap();
    }

    #[test]
    #[should_panic(expected = "super.end()/clearAttributes() was not called correctly in end()")]
    fn end_without_clear() {
        let mut s = scripted(rich(), 3);
        s.end_clear = false;
        assert_token_stream_contents(&mut s, &Expected::terms(&["a", "b"])).unwrap();
    }

    #[test]
    fn reset_exception_accepts_false_or_illegal_state() {
        for mode in [NoReset::Done, NoReset::Contract] {
            let a = analyzer(move || Scripted {
                no_reset: mode,
                ..scripted(vec![t("a", 0, 1, 1, 1)], 1)
            });
            check_reset_exception(&a, "a").unwrap();
        }
    }

    #[test]
    #[should_panic(expected = "didn't get expected exception when reset() not called")]
    fn reset_exception_refuses_a_token() {
        let a = analyzer(|| Scripted {
            no_reset: NoReset::Token,
            ..scripted(vec![t("a", 0, 1, 1, 1)], 1)
        });
        check_reset_exception(&a, "a").unwrap();
    }

    #[test]
    #[should_panic(expected = "got wrong exception when reset() not called")]
    fn reset_exception_refuses_another_error() {
        let a = analyzer(|| Scripted {
            no_reset: NoReset::WrongError,
            ..scripted(vec![t("a", 0, 1, 1, 1)], 1)
        });
        check_reset_exception(&a, "a").unwrap();
    }

    #[test]
    fn attribute_factory_draws_like_java() {
        let mut r = JavaRandom::new(42);
        let mut twin = JavaRandom::new(42);
        let mut seen = HashSet::new();
        for _ in 0..30 {
            let kind = new_attribute_factory(&mut r);
            let expected = match twin.next_int_bounded(3) {
                0 => AttributeFactoryKind::TokenStreamDefault,
                1 => AttributeFactoryKind::Token,
                _ => AttributeFactoryKind::Default,
            };
            assert_eq!(kind, expected);
            seen.insert(format!("{kind:?}"));
        }
        assert_eq!(seen.len(), 3);
    }

    #[test]
    fn number_of_tokens() {
        assert_stream_has_number_of_tokens(&mut scripted(rich(), 3), 2).unwrap();
    }

    #[test]
    #[should_panic(expected = "wrong number of tokens")]
    fn wrong_number_of_tokens() {
        assert_stream_has_number_of_tokens(&mut scripted(rich(), 3), 3).unwrap();
    }

    #[test]
    fn random_data_returns_the_analyzers_error() {
        let resets = Arc::new(AtomicUsize::new(0));
        let a = analyzer(move || Scripted {
            resets: Arc::clone(&resets),
            // check_reset_exception's reset passes; the first random
            // string's fails
            fail_from_reset: 2,
            ..scripted(vec![t("a", 0, 0, 1, 1)], 0)
        });
        let err = check_random_data(&mut JavaRandom::new(1), &a, 3).unwrap_err();
        assert_eq!(err, AnalysisError::Io("scripted failure".into()));
    }

    #[test]
    #[should_panic(expected = "term 0")]
    fn random_data_catches_an_inconsistent_analyzer() {
        let resets = Arc::new(AtomicUsize::new(0));
        let a = analyzer(move || Scripted {
            resets: Arc::clone(&resets),
            count_terms: true,
            ..scripted(vec![t("a", 0, 0, 1, 1)], 0)
        });
        check_random_data_with(&mut JavaRandom::new(1), &a, 5, 20, true, true).unwrap();
    }

    /// Java's inner `checkRandomData`, replayed draw by draw with an index
    /// writer present: the field type's draws first, then per text the
    /// string, the consistency check's draws and `nextInt(7)`.
    fn replay_indexing(seed: i64, a: &Analyzer, iterations: i32) -> JavaRandom {
        let mut r = JavaRandom::new(seed);
        let term_vectors = r.next_boolean();
        if term_vectors {
            let _offsets = r.next_boolean();
            let positions = r.next_boolean();
            if positions {
                let _payloads = r.next_boolean();
            }
        }
        let _omit_norms = r.next_boolean();
        let _index_options = r.next_int_bounded(4);
        for _ in 0..iterations {
            let text = test_util::random_analysis_string(&mut r, 20, false);
            check_analysis_consistency(&mut r, a, false, &text, true).unwrap();
            let _pile_up = r.next_int_bounded(7);
        }
        r
    }

    #[test]
    fn random_data_makes_the_index_writers_draws() {
        let a = MockAnalyzer::new(&mut JavaRandom::new(4)).build();
        for seed in 0..8 {
            let run = |index: bool| {
                let mut r = JavaRandom::new(seed);
                let c = RandomRun {
                    iterations: 5,
                    max_word_length: 20,
                    use_char_filter: seed % 2 == 1,
                    simple: false,
                    graph_offsets_are_correct: true,
                };
                let tmp = TempDir::new("bttc-draws");
                let dir = FsDirectory::open(tmp.path());
                let iw = index.then(|| Mutex::new(random_index_writer(seed, &a, &dir)));
                check_random_data_inner(&mut r, &a, &c, iw.as_ref()).unwrap();
                if let Some(iw) = iw {
                    close_and_check(iw, &dir);
                }
                r.next_long()
            };
            let indexed = run(true);
            assert_eq!(
                indexed,
                replay_indexing(seed, &a, 5).next_long(),
                "seed {seed}"
            );
            assert_ne!(indexed, run(false), "seed {seed}: the draws were skipped");
        }
    }

    #[test]
    fn analysis_consistency_survives_both_evil_branches() {
        // evilness is nextInt(50) after nextInt(10): enough seeds reach 17
        // (a failing reader) and 7 (a consumer that stops early)
        let a = MockAnalyzer::new(&mut JavaRandom::new(3)).build();
        let (mut evil_reader, mut early_stop) = (0, 0);
        for seed in 0..300 {
            let mut probe = JavaRandom::new(seed);
            probe.next_int_bounded(10);
            match probe.next_int_bounded(50) {
                17 => evil_reader += 1,
                7 => early_stop += 1,
                _ => {}
            }
            let mut r = JavaRandom::new(seed);
            check_analysis_consistency(&mut r, &a, seed % 2 == 0, "some words to read", true)
                .unwrap();
        }
        assert!(
            evil_reader > 0 && early_stop > 0,
            "{evil_reader} {early_stop}"
        );
    }

    #[test]
    fn escape_is_javas() {
        assert_eq!(escape("a\n\r\"\\\u{e9}~"), "a\\n\\r\\\"\\\\\\u00e9~");
        assert_eq!(escape("\u{1f600}"), "\\ud83d\\ude00");
    }

    #[test]
    fn mock_tokenizer_helpers() {
        let e = Expected::terms(&["Ab", "c"]).offsets(&[0, 3], &[2, 4]);
        assert_token_stream_contents(&mut whitespace_mock_tokenizer("Ab c").unwrap(), &e).unwrap();
        assert_token_stream_contents(
            &mut whitespace_mock_tokenizer_from(Box::new(StrReader::new("Ab c"))).unwrap(),
            &e,
        )
        .unwrap();
        let k = Expected::terms(&["Ab c"]);
        assert_token_stream_contents(&mut keyword_mock_tokenizer("Ab c").unwrap(), &k).unwrap();
        assert_token_stream_contents(
            &mut keyword_mock_tokenizer_from(Box::new(StrReader::new("Ab c"))).unwrap(),
            &k,
        )
        .unwrap();
    }

    #[test]
    fn graph_strings() {
        let a = MockSynonymAnalyzer::analyzer();
        let want: HashSet<String> = ["guinea pig", "cavy"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(get_graph_strings_of(&a, "guinea pig").unwrap(), want);
        assert_graph_strings_of(
            &mut JavaRandom::new(5),
            &a,
            "guinea pig",
            &["cavy", "guinea pig"],
        )
        .unwrap();
    }

    #[test]
    #[should_panic(expected = "Analyzer created unexpected string path: ")]
    fn graph_strings_mismatch() {
        let a = MockSynonymAnalyzer::analyzer();
        assert_graph_strings(
            &mut a.token_stream("f", "guinea pig").unwrap(),
            &["guinea pig"],
        )
        .unwrap();
    }

    #[test]
    fn analyzer_to_string() {
        let a = MockSynonymAnalyzer::analyzer();
        assert_eq!(
            to_string(&a, "guinea pig").unwrap(),
            "guinea at pos=0 to pos=1 offsets=0-6\n\
             cavy at pos=0 to pos=2 offsets=0-10\n\
             pig at pos=1 to pos=2 offsets=7-10\n"
        );
    }

    #[test]
    fn check_one_term_and_the_clear_attribute() {
        let a = MockAnalyzer::new(&mut JavaRandom::new(1)).build();
        check_one_term(&mut JavaRandom::new(2), &a, "Word", "word").unwrap();
        let mut seen = Vec::new();
        let mut c = CheckClearAttributes::default();
        c.clear();
        c.reflect(&mut |class, key, value| seen.push(format!("{class}#{key}={value}")));
        assert_eq!(
            seen,
            ["org.apache.lucene.tests.analysis.BaseTokenStreamTestCase$CheckClearAttributesAttribute#clearCalled=true"]
        );
        assert!(c.get_and_reset_clear_called());
        assert!(!c.get_and_reset_clear_called());
        assert!(c.clone().get_and_reset_clear_called());
    }
}
