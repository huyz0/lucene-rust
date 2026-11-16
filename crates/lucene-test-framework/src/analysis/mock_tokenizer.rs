//! `MockTokenizer`: a tokenizer driven by a [`CharacterRunAutomaton`] (the
//! longest run of code points the automaton steps through is a token), with
//! a state machine that checks the consumer's `TokenStream` workflow.
//!
//! Differs: Java reads each character through one of four `Reader` entry
//! points chosen by a private `Random` drawn from the test context (and
//! never changing the tokens); a [`CharReader`] has one, so every character
//! is a one-unit `read` and no random is drawn. Java's constructor also
//! draws from the test context: `nextInt(3)` in `newAttributeFactory()` and
//! the `nextLong` seeding that private `Random`. Neither changes a token,
//! but they advance the test's `random()`, so a Rust test that builds
//! `MockTokenizer`s from the test's random does not replay a Java test seed
//! (a mock built from an explicit seed does: `GenTestFrameworkAnalysis`).

use std::sync::{Arc, LazyLock};

use lucene_analysis::attributes::AttributeSource;
use lucene_analysis::java_character;
use lucene_analysis::reader::{CharReader, StrReader};
use lucene_analysis::{AnalysisError, TokenStream, Tokenizer, TokenizerInput};
use lucene_util::automaton::operations::{determinize, DEFAULT_DETERMINIZE_WORK_LIMIT};
use lucene_util::automaton::{CharacterRunAutomaton, RegExp};

/// `CharTokenizer.DEFAULT_MAX_WORD_LEN`: the default token length limit.
pub const DEFAULT_MAX_TOKEN_LENGTH: i32 = 255;

fn run_automaton(regexp: &str) -> Arc<CharacterRunAutomaton> {
    let a = RegExp::new(regexp)
        .and_then(|r| r.to_automaton())
        .expect("a constant regexp parses");
    let a = determinize(&a, DEFAULT_DETERMINIZE_WORK_LIMIT).expect("small automaton");
    Arc::new(CharacterRunAutomaton::new(&a).expect("determinized"))
}

static WHITESPACE: LazyLock<Arc<CharacterRunAutomaton>> =
    LazyLock::new(|| run_automaton("[^ \t\r\n]+"));
static KEYWORD: LazyLock<Arc<CharacterRunAutomaton>> = LazyLock::new(|| run_automaton(".*"));
static SIMPLE: LazyLock<Arc<CharacterRunAutomaton>> = LazyLock::new(|| {
    run_automaton(
        "[A-Za-z\u{aa}\u{b5}\u{ba}\u{c0}-\u{d6}\u{d8}-\u{f6}\u{f8}-\u{2c1}\u{4e00}-\u{9fcc}]+",
    )
});

/// `MockTokenizer.WHITESPACE`: acts like `WhitespaceTokenizer`.
pub fn whitespace() -> Arc<CharacterRunAutomaton> {
    Arc::clone(&WHITESPACE)
}

/// `MockTokenizer.KEYWORD`: acts like `KeywordTokenizer` (but emits no token
/// for empty input).
pub fn keyword() -> Arc<CharacterRunAutomaton> {
    Arc::clone(&KEYWORD)
}

/// `MockTokenizer.SIMPLE`: acts like `LetterTokenizer` (Java's incomplete
/// Unicode 5.2 `[:Letter:]` class).
pub fn simple() -> Arc<CharacterRunAutomaton> {
    Arc::clone(&SIMPLE)
}

/// `MockTokenizer.State`: where the consumer is in the workflow.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamState {
    /// The consumer set a reader.
    SetReader,
    /// `reset()` was called.
    Reset,
    /// `incrementToken()` returned `true`.
    Increment,
    /// `incrementToken()` returned `false`.
    IncrementFalse,
    /// `end()` was called.
    End,
    /// `close()` was called.
    Close,
}

impl std::fmt::Display for StreamState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            StreamState::SetReader => "SETREADER",
            StreamState::Reset => "RESET",
            StreamState::Increment => "INCREMENT",
            StreamState::IncrementFalse => "INCREMENT_FALSE",
            StreamState::End => "END",
            StreamState::Close => "CLOSE",
        })
    }
}

/// `org.apache.lucene.tests.analysis.MockTokenizer`.
pub struct MockTokenizer {
    atts: AttributeSource,
    input: TokenizerInput,
    run_automaton: Arc<CharacterRunAutomaton>,
    lower_case: bool,
    max_token_length: i32,
    state: i32,
    off: i32,
    /// `-1`: empty.
    buffered_code_point: i32,
    buffered_off: i32,
    stream_state: StreamState,
    last_offset: i32,
    enable_checks: bool,
    /// The term's length in UTF-16 units (Java's `termAtt.length()`).
    term_len: i32,
}

impl Default for MockTokenizer {
    /// `MockTokenizer()`: `WHITESPACE`, lower-casing.
    fn default() -> Self {
        Self::new(whitespace(), true)
    }
}

impl MockTokenizer {
    /// `MockTokenizer(runAutomaton, lowerCase)`.
    pub fn new(run_automaton: Arc<CharacterRunAutomaton>, lower_case: bool) -> Self {
        Self::with_max_token_length(run_automaton, lower_case, DEFAULT_MAX_TOKEN_LENGTH)
    }

    /// `MockTokenizer(runAutomaton, lowerCase, maxTokenLength)`.
    pub fn with_max_token_length(
        run_automaton: Arc<CharacterRunAutomaton>,
        lower_case: bool,
        max_token_length: i32,
    ) -> Self {
        MockTokenizer {
            atts: AttributeSource::new(),
            input: TokenizerInput::new(),
            run_automaton,
            lower_case,
            max_token_length,
            state: 0,
            off: 0,
            buffered_code_point: -1,
            buffered_off: -1,
            stream_state: StreamState::Close,
            last_offset: 0,
            enable_checks: true,
            term_len: 0,
        }
    }

    /// A tokenizer already given `text` (the test framework's
    /// `whitespaceMockTokenizer(String)` and friends do this).
    pub fn with_text(mut self, text: &str) -> Result<Self, AnalysisError> {
        self.set_reader(Box::new(StrReader::new(text)))?;
        Ok(self)
    }

    /// `setEnableChecks`: when off, workflow violations are not errors.
    pub fn set_enable_checks(&mut self, enable_checks: bool) {
        self.enable_checks = enable_checks;
    }

    /// Where the consumer is in the workflow.
    pub fn stream_state(&self) -> StreamState {
        self.stream_state
    }

    /// `fail`: an `IllegalStateException` unless checks are off.
    fn fail(&self, message: String) -> Result<(), AnalysisError> {
        if self.enable_checks {
            Err(AnalysisError::IllegalState(message))
        } else {
            Ok(())
        }
    }

    fn fail_always(message: String) -> AnalysisError {
        AnalysisError::IllegalState(message)
    }

    /// `readChar()`: one UTF-16 unit, `-1` at the end.
    fn read_char(&mut self) -> Result<i32, AnalysisError> {
        let mut c = [0u16; 1];
        let n = self.input.reader()?.read(&mut c)?;
        Ok(if n == 0 { -1 } else { i32::from(c[0]) })
    }

    /// `readCodePoint()`.
    fn read_code_point(&mut self) -> Result<i32, AnalysisError> {
        let ch = self.read_char()?;
        if ch < 0 {
            return Ok(ch);
        }
        let unit = ch as u16;
        if java_character::is_low_surrogate(unit) {
            return Err(Self::fail_always(format!("unpaired low surrogate: {ch:x}")));
        }
        self.off += 1;
        if java_character::is_high_surrogate(unit) {
            let ch2 = self.read_char()?;
            if ch2 >= 0 {
                self.off += 1;
                if !java_character::is_low_surrogate(ch2 as u16) {
                    return Err(Self::fail_always(format!(
                        "unpaired high surrogate: {ch:x}, followed by: {ch2:x}"
                    )));
                }
                return Ok(0x10000 + ((ch - 0xD800) << 10) + (ch2 - 0xDC00));
            }
            return Err(Self::fail_always(format!(
                "stream ends with unpaired high surrogate: {ch:x}"
            )));
        }
        Ok(ch)
    }

    /// `isTokenChar(c)`: steps the automaton.
    fn is_token_char(&mut self, c: i32) -> bool {
        if self.state < 0 {
            self.state = 0;
        }
        self.state = self.run_automaton.0.step(self.state, c);
        self.state >= 0
    }

    /// `normalize(c)`.
    fn normalize(&self, c: i32) -> i32 {
        if self.lower_case {
            java_character::to_lower_case(c as u32) as i32
        } else {
            c
        }
    }

    fn append(&mut self, cp: i32) {
        let ch = char::from_u32(cp as u32).expect("a paired code point");
        self.atts.term_mut().push(ch);
        self.term_len += ch.len_utf16() as i32;
    }
}

impl TokenStream for MockTokenizer {
    fn attributes(&self) -> &AttributeSource {
        &self.atts
    }

    fn attributes_mut(&mut self) -> &mut AttributeSource {
        &mut self.atts
    }

    fn increment_token(&mut self) -> Result<bool, AnalysisError> {
        if self.stream_state != StreamState::Reset && self.stream_state != StreamState::Increment {
            self.fail(format!(
                "incrementToken() called while in wrong state: {}",
                self.stream_state
            ))?;
        }
        self.atts.clear_attributes();
        self.term_len = 0;
        loop {
            let start_offset;
            let mut cp;
            if self.buffered_code_point >= 0 {
                cp = self.buffered_code_point;
                start_offset = self.buffered_off;
                self.buffered_code_point = -1;
            } else {
                start_offset = self.off;
                cp = self.read_code_point()?;
            }
            if cp < 0 {
                break;
            } else if self.is_token_char(cp) {
                let mut end_offset;
                loop {
                    let n = self.normalize(cp);
                    self.append(n);
                    end_offset = self.off;
                    if self.term_len >= self.max_token_length {
                        break;
                    }
                    cp = self.read_code_point()?;
                    if !(cp >= 0 && self.is_token_char(cp)) {
                        break;
                    }
                }
                if self.term_len < self.max_token_length {
                    // buffer up, in case the "rejected" char can start a new word of its own
                    self.buffered_code_point = cp;
                    self.buffered_off = end_offset;
                } else {
                    // otherwise, it's because we hit term limit.
                    self.buffered_code_point = -1;
                }
                let corrected_start = self.input.correct_offset(start_offset);
                let corrected_end = self.input.correct_offset(end_offset);
                if corrected_start < 0 {
                    return Err(Self::fail_always(format!(
                        "invalid start offset: {corrected_start}, before correction: {start_offset}"
                    )));
                }
                if corrected_end < 0 {
                    return Err(Self::fail_always(format!(
                        "invalid end offset: {corrected_end}, before correction: {end_offset}"
                    )));
                }
                if corrected_start < self.last_offset {
                    return Err(Self::fail_always(format!(
                        "start offset went backwards: {corrected_start}, before correction: {start_offset}, lastOffset: {}",
                        self.last_offset
                    )));
                }
                self.last_offset = corrected_start;
                if corrected_end < corrected_start {
                    return Err(Self::fail_always(format!(
                        "end offset: {corrected_end} is before start offset: {corrected_start}"
                    )));
                }
                self.atts.set_offset(corrected_start, corrected_end)?;
                if self.state == -1 || self.run_automaton.0.is_accept(self.state) {
                    // either we hit a reject state (longest match), or end-of-text, but in an
                    // accept state
                    self.stream_state = StreamState::Increment;
                    return Ok(true);
                }
            }
        }
        self.stream_state = StreamState::IncrementFalse;
        Ok(false)
    }

    fn reset(&mut self) -> Result<(), AnalysisError> {
        self.input.reset();
        self.state = 0;
        self.last_offset = 0;
        self.off = 0;
        self.buffered_code_point = -1;
        let checked = if self.stream_state == StreamState::Reset {
            self.fail("double reset()".to_string())
        } else {
            Ok(())
        };
        self.stream_state = StreamState::Reset;
        checked
    }

    fn end(&mut self) -> Result<(), AnalysisError> {
        self.atts.end_attributes();
        let final_offset = self.input.correct_offset(self.off);
        let set = self.atts.set_offset(final_offset, final_offset);
        // some tokenizers, such as limiting tokenizers, call end() before incrementToken()
        // returns false.
        let checked = if self.stream_state != StreamState::IncrementFalse {
            // SENTINEL-OK: "end()" here is message text, not a call.
            self.fail(format!(
                "end() called in wrong state={}!",
                self.stream_state
            ))
        } else {
            Ok(())
        };
        self.stream_state = StreamState::End;
        set.and(checked)
    }

    fn close(&mut self) -> Result<(), AnalysisError> {
        let closed = self.input.close();
        // in some exceptional cases (e.g. TestIndexWriterExceptions) a test can prematurely
        // close(); these tests should disable this check.
        let checked = if !(self.stream_state == StreamState::End
            || self.stream_state == StreamState::Close)
        {
            self.fail(format!(
                "close() called in wrong state: {}",
                self.stream_state
            ))
        } else {
            Ok(())
        };
        self.stream_state = StreamState::Close;
        closed.and(checked)
    }

    fn as_tokenizer(&mut self) -> Option<&mut dyn Tokenizer> {
        Some(self)
    }

    fn conditional_root(&mut self) -> Option<&mut dyn std::any::Any> {
        None
    }
}

impl Tokenizer for MockTokenizer {
    fn set_reader(&mut self, input: Box<dyn CharReader>) -> Result<(), AnalysisError> {
        self.input.set_reader(input)?;
        // setReaderTestPoint
        let checked = if self.stream_state != StreamState::Close {
            self.fail(format!(
                "setReader() called in wrong state: {}",
                self.stream_state
            ))
        } else {
            Ok(())
        };
        self.stream_state = StreamState::SetReader;
        checked
    }
}

#[cfg(test)]
mod tests;
