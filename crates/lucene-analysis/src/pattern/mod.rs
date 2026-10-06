//! `org.apache.lucene.analysis.pattern`: the `java.util.regex` tokenizer and
//! filters (over [`JavaPattern`]/[`JavaMatcher`]) and the automaton-based
//! `SimplePatternTokenizer`/`SimplePatternSplitTokenizer` (over
//! `lucene-util`'s `RegExp` and `CharacterRunAutomaton`).

use lucene_util::automaton::{operations, Automaton, CharacterRunAutomaton, RegExp};

use crate::attributes::{AttributeSource, State};
use crate::charfilter::OffsetCorrections;
use crate::reader::{read_to_string, CharFilter, CharReader};
use crate::token_stream::{TokenFilter, TokenStream, Tokenizer, TokenizerInput};
use crate::util::java_regex::{append_replacement, JavaMatcher, JavaPattern};
use crate::{utf16_len, AnalysisError};

fn illegal(e: impl std::fmt::Display) -> AnalysisError {
    AnalysisError::IllegalArgument(e.to_string())
}

/// `org.apache.lucene.analysis.pattern.PatternTokenizer`: with `group < 0`
/// the text between matches, else every match's `group`.
pub struct PatternTokenizer {
    atts: AttributeSource,
    input: TokenizerInput,
    matcher: JavaMatcher,
    group: i32,
    /// `index`, in UTF-16 units (`i32::MAX` once exhausted).
    index: i32,
    /// `index` as a byte offset into the text.
    index_byte: usize,
    str_len: i32,
}

impl PatternTokenizer {
    /// `new PatternTokenizer(Pattern, int group)`.
    pub fn new(pattern: &JavaPattern, group: i32) -> Result<Self, AnalysisError> {
        let matcher = JavaMatcher::new(pattern, "");
        if group >= 0 && group as usize > matcher.group_count() {
            return Err(AnalysisError::IllegalArgument(format!(
                "invalid group specified: pattern only has: {} capturing groups",
                matcher.group_count()
            )));
        }
        Ok(PatternTokenizer {
            atts: AttributeSource::new(),
            input: TokenizerInput::new(),
            matcher,
            group,
            index: 0,
            index_byte: 0,
            str_len: 0,
        })
    }

    fn emit(&mut self, from: usize, to: usize, start: i32, end: i32) -> Result<(), AnalysisError> {
        self.atts.set_term(&self.matcher.text()[from..to]);
        let (s, e) = (
            self.input.correct_offset(start),
            self.input.correct_offset(end),
        );
        self.atts.set_offset(s, e)
    }
}

impl TokenStream for PatternTokenizer {
    fn attributes(&self) -> &AttributeSource {
        &self.atts
    }

    fn attributes_mut(&mut self) -> &mut AttributeSource {
        &mut self.atts
    }

    // Java: PatternTokenizer.incrementToken
    fn increment_token(&mut self) -> Result<bool, AnalysisError> {
        if self.index >= self.str_len {
            return Ok(false);
        }
        self.atts.clear_attributes();
        if self.group >= 0 {
            let g = self.group as usize;
            while self.matcher.find() {
                self.index = self.matcher.start(g);
                let end_index = self.matcher.end(g);
                if self.index == end_index {
                    continue;
                }
                let (bs, be) = self
                    .matcher
                    .group_bytes(g)
                    .expect("a non-empty group matched");
                self.emit(bs, be, self.index, end_index)?;
                return Ok(true);
            }
            self.index = i32::MAX;
            return Ok(false);
        }
        while self.matcher.find() {
            let (ms, me) = self.matcher.group_bytes(0).expect("group 0");
            let (start, end) = (self.matcher.start(0), self.matcher.end(0));
            if start - self.index > 0 {
                self.emit(self.index_byte, ms, self.index, start)?;
                self.index = end;
                self.index_byte = me;
                return Ok(true);
            }
            self.index = end;
            self.index_byte = me;
        }
        if self.str_len - self.index == 0 {
            self.index = i32::MAX;
            return Ok(false);
        }
        let len = self.matcher.text().len();
        self.emit(self.index_byte, len, self.index, self.str_len)?;
        self.index = i32::MAX;
        Ok(true)
    }

    // Java: PatternTokenizer.end
    fn end(&mut self) -> Result<(), AnalysisError> {
        self.atts.end_attributes();
        let ofs = self.input.correct_offset(self.str_len);
        self.atts.set_offset(ofs, ofs)
    }

    // Java: PatternTokenizer.reset (fillBuffer)
    fn reset(&mut self) -> Result<(), AnalysisError> {
        self.input.reset();
        let text = read_to_string(self.input.reader()?)?;
        self.str_len = utf16_len(&text) as i32;
        self.matcher.reset(&text);
        self.index = 0;
        self.index_byte = 0;
        Ok(())
    }

    fn close(&mut self) -> Result<(), AnalysisError> {
        self.matcher.reset("");
        self.input.close()
    }

    fn as_tokenizer(&mut self) -> Option<&mut dyn Tokenizer> {
        Some(self)
    }
}

impl Tokenizer for PatternTokenizer {
    fn set_reader(&mut self, input: Box<dyn CharReader>) -> Result<(), AnalysisError> {
        self.input.set_reader(input)
    }
}

/// `org.apache.lucene.analysis.pattern.PatternReplaceFilter`.
pub struct PatternReplaceFilter<I> {
    input: I,
    pattern: JavaPattern,
    replacement: String,
    all: bool,
}

impl<I: TokenStream> PatternReplaceFilter<I> {
    /// `new PatternReplaceFilter(TokenStream, Pattern, String replacement,
    /// boolean all)` (`None` replacement is `""`).
    pub fn new(input: I, pattern: JavaPattern, replacement: Option<&str>, all: bool) -> Self {
        PatternReplaceFilter {
            input,
            pattern,
            replacement: replacement.unwrap_or("").to_string(),
            all,
        }
    }
}

impl<I: TokenStream> TokenFilter for PatternReplaceFilter<I> {
    crate::filter_input!();
    // Java: PatternReplaceFilter.incrementToken
    fn increment(&mut self) -> Result<bool, AnalysisError> {
        if !self.input.increment_token()? {
            return Ok(false);
        }
        let a = self.input.attributes_mut();
        if self.pattern.regex().is_match(a.term()) {
            let t = self
                .pattern
                .replace(a.term(), &self.replacement, self.all)?;
            a.set_term(&t);
        }
        Ok(true)
    }
}

/// `org.apache.lucene.analysis.pattern.PatternCaptureGroupTokenFilter`: emits
/// every capture group of every match of every pattern, at the token's
/// position.
pub struct PatternCaptureGroupTokenFilter<I> {
    input: I,
    state: Option<State>,
    matchers: Vec<JavaMatcher>,
    group_counts: Vec<i32>,
    preserve_original: bool,
    current_group: Vec<i32>,
    current_matcher: i32,
    spare_len: i32,
}

impl<I: TokenStream> PatternCaptureGroupTokenFilter<I> {
    /// `new PatternCaptureGroupTokenFilter(TokenStream, boolean preserveOriginal, Pattern...)`.
    pub fn new(input: I, preserve_original: bool, patterns: &[JavaPattern]) -> Self {
        let matchers: Vec<JavaMatcher> = patterns.iter().map(|p| JavaMatcher::new(p, "")).collect();
        PatternCaptureGroupTokenFilter {
            input,
            state: None,
            group_counts: matchers.iter().map(|m| m.group_count() as i32).collect(),
            current_group: vec![-1; matchers.len()],
            matchers,
            preserve_original,
            current_matcher: -1,
            spare_len: 0,
        }
    }

    // Java: PatternCaptureGroupTokenFilter.nextCapture
    fn next_capture(&mut self) -> bool {
        let mut min_offset = i32::MAX;
        self.current_matcher = -1;
        let mut i = 0usize;
        while i < self.matchers.len() {
            if self.current_group[i] == -1 {
                self.current_group[i] = i32::from(self.matchers[i].find());
            }
            if self.current_group[i] != 0 {
                while self.current_group[i] < self.group_counts[i] + 1 {
                    let g = self.current_group[i] as usize;
                    let (start, end) = (self.matchers[i].start(g), self.matchers[i].end(g));
                    if start == end
                        || (self.preserve_original && start == 0 && self.spare_len == end)
                    {
                        self.current_group[i] += 1;
                        continue;
                    }
                    if start < min_offset {
                        min_offset = start;
                        self.current_matcher = i as i32;
                    }
                    break;
                }
                if self.current_group[i] == self.group_counts[i] + 1 {
                    self.current_group[i] = -1;
                    // Java: i-- (retry this matcher with its next match).
                    continue;
                }
            }
            i += 1;
        }
        self.current_matcher != -1
    }

    fn current_bytes(&self) -> (usize, usize) {
        let m = self.current_matcher as usize;
        self.matchers[m]
            .group_bytes(self.current_group[m] as usize)
            .expect("nextCapture picked a participating group")
    }
}

impl<I: TokenStream> TokenFilter for PatternCaptureGroupTokenFilter<I> {
    crate::filter_input!();

    // Java: PatternCaptureGroupTokenFilter.incrementToken
    fn increment(&mut self) -> Result<bool, AnalysisError> {
        if self.current_matcher != -1 && self.next_capture() {
            let (s, e) = self.current_bytes();
            let m = self.current_matcher as usize;
            let term = self.matchers[m].text()[s..e].to_string();
            let a = self.input.attributes_mut();
            a.clear_attributes();
            a.restore_state(self.state.as_ref().expect("a token was read"));
            a.set_position_increment(0)?;
            a.set_term(&term);
            self.current_group[m] += 1;
            return Ok(true);
        }
        if !self.input.increment_token()? {
            return Ok(false);
        }
        let a = self.input.attributes();
        let spare = a.term().to_string();
        self.spare_len = utf16_len(&spare) as i32;
        self.state = Some(a.capture_state());
        for (m, g) in self.matchers.iter_mut().zip(self.current_group.iter_mut()) {
            m.reset(&spare);
            *g = -1;
        }
        if self.preserve_original {
            self.current_matcher = 0;
        } else if self.next_capture() {
            let (s, e) = self.current_bytes();
            // Java: setLength(end) when the capture starts at 0, else copy.
            self.input.attributes_mut().set_term(&spare[s..e]);
            let m = self.current_matcher as usize;
            self.current_group[m] += 1;
        }
        Ok(true)
    }

    fn reset_filter(&mut self) -> Result<(), AnalysisError> {
        self.input.reset()?;
        self.state = None;
        self.current_matcher = -1;
        Ok(())
    }
}

/// `PatternTypingFilter.PatternTypingRule`.
#[derive(Debug, Clone)]
pub struct PatternTypingRule {
    /// `pattern()`.
    pub pattern: JavaPattern,
    /// `flags()`.
    pub flags: i32,
    /// `typeTemplate()`: a replacement string.
    pub type_template: String,
}

/// `org.apache.lucene.analysis.pattern.PatternTypingFilter`: the first rule
/// whose pattern is found in the term sets the type (its template, expanded
/// against the term) and the flags.
pub struct PatternTypingFilter<I> {
    input: I,
    rules: Vec<PatternTypingRule>,
}

impl<I: TokenStream> PatternTypingFilter<I> {
    /// `new PatternTypingFilter(TokenStream, PatternTypingRule...)`.
    pub fn new(input: I, rules: Vec<PatternTypingRule>) -> Self {
        PatternTypingFilter { input, rules }
    }
}

impl<I: TokenStream> TokenFilter for PatternTypingFilter<I> {
    crate::filter_input!();
    // Java: PatternTypingFilter.incrementToken
    fn increment(&mut self) -> Result<bool, AnalysisError> {
        if !self.input.increment_token()? {
            return Ok(false);
        }
        let a = self.input.attributes_mut();
        for rule in &self.rules {
            if rule.pattern.regex().is_match(a.term()) {
                // Java: matcher.replaceFirst(typeTemplate) over the term.
                let t = rule.pattern.replace(a.term(), &rule.type_template, false)?;
                a.set_token_type(t);
                a.set_flags(rule.flags);
                return Ok(true);
            }
        }
        Ok(true)
    }
}

/// `org.apache.lucene.analysis.pattern.PatternReplaceCharFilter`: reads the
/// whole input, replaces every match, and corrects offsets across each
/// replacement.
pub struct PatternReplaceCharFilter<R> {
    input: R,
    pattern: JavaPattern,
    replacement: String,
    transformed: Option<(Vec<u16>, usize)>,
    corrections: OffsetCorrections,
}

impl<R: CharReader> PatternReplaceCharFilter<R> {
    /// `new PatternReplaceCharFilter(Pattern, String replacement, Reader)`.
    pub fn new(pattern: JavaPattern, replacement: &str, input: R) -> Self {
        PatternReplaceCharFilter {
            input,
            pattern,
            replacement: replacement.to_string(),
            transformed: None,
            corrections: OffsetCorrections::default(),
        }
    }

    // Java: PatternReplaceCharFilter.processPattern
    fn process_pattern(&mut self, input: &str) -> Result<String, AnalysisError> {
        let mut out = String::with_capacity(input.len());
        let mut out_len = 0i32;
        let mut cumulative = 0i32;
        let mut last_match_end = 0i32;
        let mut last_byte = 0usize;
        for caps in self.pattern.regex().captures_iter(input) {
            let m = caps.get(0).expect("group 0");
            let start = utf16_len(&input[..m.start()]) as i32;
            let end = start + utf16_len(m.as_str()) as i32;
            let group_size = end - start;
            let skipped = &input[last_byte..m.start()];
            let skipped_size = start - last_match_end;
            last_match_end = end;
            let length_before_replacement = out_len + skipped_size;
            out.push_str(skipped);
            let mut rep = String::new();
            append_replacement(&mut rep, &caps, &self.replacement)?;
            out.push_str(&rep);
            let replacement_size = utf16_len(&rep) as i32;
            out_len = length_before_replacement + replacement_size;
            last_byte = m.end();
            if group_size != replacement_size {
                if replacement_size < group_size {
                    cumulative += group_size - replacement_size;
                    self.corrections
                        .add(length_before_replacement + replacement_size, cumulative);
                } else {
                    for i in group_size..replacement_size {
                        cumulative -= 1;
                        self.corrections
                            .add(length_before_replacement + i, cumulative);
                    }
                }
            }
        }
        out.push_str(&input[last_byte..]);
        Ok(out)
    }
}

impl<R: CharReader> CharFilter for PatternReplaceCharFilter<R> {
    fn input(&self) -> &dyn CharReader {
        &self.input
    }

    fn input_mut(&mut self) -> &mut dyn CharReader {
        &mut self.input
    }

    // Java: PatternReplaceCharFilter.read (fill on first use)
    fn read_filtered(&mut self, buf: &mut [u16]) -> Result<usize, AnalysisError> {
        if self.transformed.is_none() {
            let text = read_to_string(&mut self.input)?;
            let out = self.process_pattern(&text)?;
            self.transformed = Some((out.encode_utf16().collect(), 0));
        }
        let (units, pos) = self.transformed.as_mut().expect("filled above");
        let n = buf.len().min(units.len() - *pos);
        buf[..n].copy_from_slice(&units[*pos..*pos + n]);
        *pos += n;
        Ok(n)
    }

    // Java: PatternReplaceCharFilter.correct
    fn correct(&self, current_off: i32) -> i32 {
        self.corrections.correct(current_off).max(0)
    }
}

/// `Reader.read(char[])` one unit at a time with push-back, the I/O half
/// `SimplePatternTokenizer` and `SimplePatternSplitTokenizer` share.
#[derive(Default)]
struct UnitSource {
    pending: Vec<u16>,
    pending_limit: usize,
    pending_upto: usize,
    buffer: Vec<u16>,
    /// `None` is Java's `bufferLimit == -1`.
    buffer_limit: Option<usize>,
    buffer_next_read: usize,
    offset: i32,
}

impl UnitSource {
    fn reset(&mut self) {
        self.offset = 0;
        self.pending_upto = 0;
        self.pending_limit = 0;
        self.buffer_next_read = 0;
        self.buffer_limit = Some(0);
    }

    // Java: nextCodeUnit (the caller appends to the token)
    fn next_code_unit(&mut self, input: &mut TokenizerInput) -> Result<Option<u16>, AnalysisError> {
        if self.pending_upto < self.pending_limit {
            let r = self.pending[self.pending_upto];
            self.pending_upto += 1;
            if self.pending_upto == self.pending_limit {
                self.pending_upto = 0;
                self.pending_limit = 0;
            }
            self.offset += 1;
            return Ok(Some(r));
        }
        let Some(limit) = self.buffer_limit else {
            return Ok(None);
        };
        if self.buffer_next_read == limit {
            if self.buffer.is_empty() {
                self.buffer = vec![0; 1024];
            }
            let n = input.reader()?.read(&mut self.buffer)?;
            if n == 0 {
                self.buffer_limit = None;
                return Ok(None);
            }
            self.buffer_limit = Some(n);
            self.buffer_next_read = 0;
        }
        let r = self.buffer[self.buffer_next_read];
        self.buffer_next_read += 1;
        self.offset += 1;
        Ok(Some(r))
    }

    // Java: pushBack (`token` holds the units read for the token so far)
    fn push_back(&mut self, token: &[u16], count: usize) {
        if self.pending_limit == 0 {
            if self.buffer_limit.is_some() && self.buffer_next_read >= count {
                self.buffer_next_read -= count;
            } else {
                self.pending.clear();
                self.pending
                    .extend_from_slice(&token[token.len() - count..]);
                self.pending_limit = count;
            }
        } else {
            self.pending_upto -= count;
        }
        self.offset -= count as i32;
    }

    /// `end()`'s offset.
    fn end_offset(&self) -> i32 {
        self.offset + self.pending_limit as i32 - self.pending_upto as i32
    }
}

/// `nextCodePoint`: a unit, or a surrogate pair as one code point.
fn next_code_point(
    src: &mut UnitSource,
    input: &mut TokenizerInput,
    token: &mut Vec<u16>,
) -> Result<Option<i32>, AnalysisError> {
    let Some(ch) = src.next_code_unit(input)? else {
        return Ok(None);
    };
    token.push(ch);
    if crate::java_character::is_high_surrogate(ch) {
        // Java: Character.toCodePoint(ch, (char) nextCodeUnit()) -- at the
        // end of input the -1 becomes U+FFFF.
        let lo = match src.next_code_unit(input)? {
            Some(lo) => {
                token.push(lo);
                lo
            }
            None => 0xFFFF,
        };
        let cp = 0x10000 + ((i32::from(ch) - 0xD800) << 10) + (i32::from(lo) - 0xDC00);
        return Ok(Some(cp));
    }
    Ok(Some(i32::from(ch)))
}

fn run_automaton(a: &Automaton) -> Result<CharacterRunAutomaton, AnalysisError> {
    if !a.is_deterministic() {
        return Err(AnalysisError::IllegalArgument(
            "please determinize the incoming automaton first".into(),
        ));
    }
    CharacterRunAutomaton::new(a).map_err(illegal)
}

/// `org.apache.lucene.analysis.pattern.SimplePatternTokenizer`: the longest
/// match of a Lucene `RegExp` at each position is a token.
pub struct SimplePatternTokenizer {
    atts: AttributeSource,
    input: TokenizerInput,
    run_dfa: CharacterRunAutomaton,
    src: UnitSource,
    token: Vec<u16>,
}

impl SimplePatternTokenizer {
    /// `new SimplePatternTokenizer(String regexp)`.
    pub fn new(regexp: &str) -> Result<Self, AnalysisError> {
        let a = RegExp::new(regexp)
            .map_err(illegal)?
            .to_automaton()
            .map_err(illegal)?;
        Self::from_automaton(&a)
    }

    /// `new SimplePatternTokenizer(Automaton dfa)`.
    pub fn from_automaton(dfa: &Automaton) -> Result<Self, AnalysisError> {
        Ok(SimplePatternTokenizer {
            atts: AttributeSource::new(),
            input: TokenizerInput::new(),
            run_dfa: run_automaton(dfa)?,
            src: UnitSource::default(),
            token: Vec::new(),
        })
    }
}

impl TokenStream for SimplePatternTokenizer {
    fn attributes(&self) -> &AttributeSource {
        &self.atts
    }

    fn attributes_mut(&mut self) -> &mut AttributeSource {
        &mut self.atts
    }

    // Java: SimplePatternTokenizer.incrementToken
    fn increment_token(&mut self) -> Result<bool, AnalysisError> {
        self.atts.clear_attributes();
        self.token.clear();
        let dfa = &self.run_dfa.0;
        loop {
            let offset_start = self.src.offset;
            let Some(mut ch) = next_code_point(&mut self.src, &mut self.input, &mut self.token)?
            else {
                return Ok(false);
            };
            let mut state = dfa.step(0, ch);
            if state != -1 {
                let mut last_accept_length: i32 = -1;
                let mut at_end = false;
                loop {
                    if dfa.is_accept(state) {
                        last_accept_length = self.token.len() as i32;
                    }
                    match next_code_point(&mut self.src, &mut self.input, &mut self.token)? {
                        None => {
                            at_end = true;
                            break;
                        }
                        Some(c) => ch = c,
                    }
                    state = dfa.step(state, ch);
                    if state == -1 {
                        break;
                    }
                }
                if last_accept_length != -1 {
                    let extra = self.token.len() - last_accept_length as usize;
                    if extra != 0 {
                        self.src.push_back(&self.token, extra);
                    }
                    self.token.truncate(last_accept_length as usize);
                    self.atts.set_term_utf16(&self.token);
                    let s = self.input.correct_offset(offset_start);
                    let e = self.input.correct_offset(offset_start + last_accept_length);
                    self.atts.set_offset(s, e)?;
                    return Ok(true);
                } else if at_end {
                    return Ok(false);
                } else {
                    let n = self.token.len() - 1;
                    self.src.push_back(&self.token, n);
                    self.token.clear();
                }
            } else {
                self.token.clear();
            }
        }
    }

    fn end(&mut self) -> Result<(), AnalysisError> {
        self.atts.end_attributes();
        let ofs = self.input.correct_offset(self.src.end_offset());
        self.atts.set_offset(ofs, ofs)
    }

    fn reset(&mut self) -> Result<(), AnalysisError> {
        self.input.reset();
        self.src.reset();
        self.token.clear();
        Ok(())
    }

    fn close(&mut self) -> Result<(), AnalysisError> {
        self.input.close()
    }

    fn as_tokenizer(&mut self) -> Option<&mut dyn Tokenizer> {
        Some(self)
    }
}

impl Tokenizer for SimplePatternTokenizer {
    fn set_reader(&mut self, input: Box<dyn CharReader>) -> Result<(), AnalysisError> {
        self.input.set_reader(input)
    }
}

/// `org.apache.lucene.analysis.pattern.SimplePatternSplitTokenizer`: the
/// text between the longest matches of a Lucene `RegExp` are the tokens.
pub struct SimplePatternSplitTokenizer {
    atts: AttributeSource,
    input: TokenizerInput,
    run_dfa: CharacterRunAutomaton,
    src: UnitSource,
    token: Vec<u16>,
}

impl SimplePatternSplitTokenizer {
    /// `new SimplePatternSplitTokenizer(String regexp)`: determinized with
    /// `DEFAULT_DETERMINIZE_WORK_LIMIT`.
    pub fn new(regexp: &str) -> Result<Self, AnalysisError> {
        let a = RegExp::new(regexp)
            .map_err(illegal)?
            .to_automaton()
            .map_err(illegal)?;
        let a = operations::determinize(&a, operations::DEFAULT_DETERMINIZE_WORK_LIMIT)
            .map_err(illegal)?;
        Self::from_automaton(&a)
    }

    /// `new SimplePatternSplitTokenizer(Automaton dfa)`.
    pub fn from_automaton(dfa: &Automaton) -> Result<Self, AnalysisError> {
        Ok(SimplePatternSplitTokenizer {
            atts: AttributeSource::new(),
            input: TokenizerInput::new(),
            run_dfa: run_automaton(dfa)?,
            src: UnitSource::default(),
            token: Vec::new(),
        })
    }

    fn fill_token(&mut self, offset_start: i32) -> Result<bool, AnalysisError> {
        self.atts.set_term_utf16(&self.token);
        let s = self.input.correct_offset(offset_start);
        let e = self
            .input
            .correct_offset(offset_start + self.token.len() as i32);
        self.atts.set_offset(s, e)?;
        Ok(true)
    }
}

impl TokenStream for SimplePatternSplitTokenizer {
    fn attributes(&self) -> &AttributeSource {
        &self.atts
    }

    fn attributes_mut(&mut self) -> &mut AttributeSource {
        &mut self.atts
    }

    // Java: SimplePatternSplitTokenizer.incrementToken
    fn increment_token(&mut self) -> Result<bool, AnalysisError> {
        let mut offset_start = self.src.offset;
        self.atts.clear_attributes();
        self.token.clear();
        loop {
            // `sepUpto` counts the units of the separator candidate.
            let sep_start = self.token.len();
            let Some(mut ch) = next_code_point(&mut self.src, &mut self.input, &mut self.token)?
            else {
                if !self.token.is_empty() {
                    return self.fill_token(offset_start);
                }
                return Ok(false);
            };
            let dfa = &self.run_dfa.0;
            let mut state = dfa.step(0, ch);
            if state != -1 {
                let mut last_accept_length: i32 = -1;
                let mut at_end = false;
                loop {
                    if dfa.is_accept(state) {
                        last_accept_length = (self.token.len() - sep_start) as i32;
                    }
                    match next_code_point(&mut self.src, &mut self.input, &mut self.token)? {
                        None => {
                            at_end = true;
                            break;
                        }
                        Some(c) => ch = c,
                    }
                    state = dfa.step(state, ch);
                    if state == -1 {
                        break;
                    }
                }
                let sep_upto = self.token.len() - sep_start;
                if last_accept_length != -1 {
                    let extra = sep_upto - last_accept_length as usize;
                    if extra != 0 {
                        self.src.push_back(&self.token, extra);
                        self.token.truncate(self.token.len() - extra);
                    }
                    self.token
                        .truncate(self.token.len() - last_accept_length as usize);
                    if !self.token.is_empty() {
                        return self.fill_token(offset_start);
                    }
                    offset_start = self.src.offset;
                } else if at_end {
                    if !self.token.is_empty() {
                        return self.fill_token(offset_start);
                    }
                    return Ok(false);
                } else {
                    let n = sep_upto - 1;
                    self.src.push_back(&self.token, n);
                    self.token.truncate(self.token.len() - n);
                }
            }
        }
    }

    fn end(&mut self) -> Result<(), AnalysisError> {
        self.atts.end_attributes();
        let ofs = self.input.correct_offset(self.src.end_offset());
        self.atts.set_offset(ofs, ofs)
    }

    fn reset(&mut self) -> Result<(), AnalysisError> {
        self.input.reset();
        self.src.reset();
        self.token.clear();
        Ok(())
    }

    fn close(&mut self) -> Result<(), AnalysisError> {
        self.input.close()
    }

    fn as_tokenizer(&mut self) -> Option<&mut dyn Tokenizer> {
        Some(self)
    }
}

impl Tokenizer for SimplePatternSplitTokenizer {
    fn set_reader(&mut self, input: Box<dyn CharReader>) -> Result<(), AnalysisError> {
        self.input.set_reader(input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reader::StrReader;
    use crate::util::canned::{render, Canned};

    fn run(t: &mut dyn Tokenizer, text: &str) -> String {
        t.set_reader(Box::new(StrReader::new(text))).unwrap();
        render(t)
    }

    fn p(s: &str) -> JavaPattern {
        JavaPattern::compile(s).unwrap()
    }

    #[test]
    fn pattern_tokenizer() {
        let mut t = PatternTokenizer::new(&p("[ ,]+"), -1).unwrap();
        assert_eq!(run(&mut t, ",a, b😀,"), "a:1:2:1:1 b😀:4:7:1:1|8|0");
        assert_eq!(run(&mut t, "ab"), "ab:0:2:1:1|2|0");
        assert_eq!(run(&mut t, ""), "|0|0");
        let mut t = PatternTokenizer::new(&p("([a-z]+)([0-9]*)"), 1).unwrap();
        assert_eq!(run(&mut t, "ab12 C cd"), "ab:0:2:1:1 cd:7:9:1:1|9|0");
        let mut t = PatternTokenizer::new(&p("(x*)"), 1).unwrap();
        assert_eq!(run(&mut t, "ab"), "|2|0");
        assert!(PatternTokenizer::new(&p("(a)"), 2).is_err());
    }

    #[test]
    fn replace_capture_and_typing() {
        let mut f = PatternReplaceFilter::new(
            Canned::parse("hello:0:5:1:1 xyz:6:9:1:1|9|0"),
            p("[aeiou]"),
            Some("_"),
            true,
        );
        assert_eq!(render(&mut f), "h_ll_:0:5:1:1 xyz:6:9:1:1|9|0");
        let mut f =
            PatternReplaceFilter::new(Canned::parse("hello:0:5:1:1|5|0"), p("l"), None, false);
        assert_eq!(render(&mut f), "helo:0:5:1:1|5|0");
        let pats = [p("([A-Z][a-z]+)"), p("([0-9]+)")];
        let mut c = Canned::parse("x:0:1:1:1 y:2:3:1:1|3|0");
        c.set_terms(&["FooBar12", "zz"]);
        let mut f = PatternCaptureGroupTokenFilter::new(c, true, &pats);
        assert_eq!(
            render(&mut f),
            "FooBar12:0:1:1:1 Foo:0:1:0:1 Bar:0:1:0:1 12:0:1:0:1 zz:2:3:1:1|3|0"
        );
        let mut c = Canned::parse("x:0:1:1:1|1|0");
        c.set_terms(&["FooBar12"]);
        let mut f = PatternCaptureGroupTokenFilter::new(c, false, &pats);
        assert_eq!(render(&mut f), "Foo:0:1:1:1 Bar:0:1:0:1 12:0:1:0:1|1|0");
        let rules = vec![PatternTypingRule {
            pattern: p("^(\\d+)$"),
            flags: 2,
            type_template: "num_$1".into(),
        }];
        let mut f = PatternTypingFilter::new(Canned::parse("42:0:2:1:1 a:3:4:1:1|4|0"), rules);
        let mut out = Vec::new();
        crate::token_stream::consume(&mut f, |a| {
            out.push((a.token_type().to_string(), a.flags()))
        })
        .unwrap();
        assert_eq!(
            out,
            vec![("num_42".to_string(), 2), ("word".to_string(), 0)]
        );
        let rules = vec![PatternTypingRule {
            pattern: p("^([A-Z])"),
            flags: 1,
            type_template: "c_$1".into(),
        }];
        let mut c = Canned::parse("x:0:1:1:1");
        c.set_terms(&["Hello"]);
        let mut f = PatternTypingFilter::new(c, rules);
        let mut out = Vec::new();
        crate::token_stream::consume(&mut f, |a| out.push(a.token_type().to_string())).unwrap();
        assert_eq!(out, vec!["c_Hello"]);
    }

    #[test]
    fn replace_char_filter() {
        let mut f = PatternReplaceCharFilter::new(
            p("([a-z]+)-([a-z]+)"),
            "$2_$1x",
            StrReader::new("ab-cd e"),
        );
        let mut buf = [0u16; 32];
        let n = f.read(&mut buf).unwrap();
        assert_eq!(String::from_utf16(&buf[..n]).unwrap(), "cd_abx e");
        assert_eq!(f.read(&mut buf).unwrap(), 0);
        assert_eq!(
            (
                f.correct_offset(5),
                f.correct_offset(6),
                f.correct_offset(8)
            ),
            (4, 5, 7)
        );
        let mut f = PatternReplaceCharFilter::new(p("abc"), "z", StrReader::new("abc d"));
        let n = f.read(&mut buf).unwrap();
        assert_eq!(String::from_utf16(&buf[..n]).unwrap(), "z d");
        assert_eq!((f.correct_offset(1), f.correct_offset(2)), (3, 4));
    }

    #[test]
    fn simple_patterns() {
        let mut t = SimplePatternTokenizer::new("[a-z]+[0-9]*").unwrap();
        assert_eq!(
            run(&mut t, "ab12 Cd x😀y"),
            "ab12:0:4:1:1 d:6:7:1:1 x:8:9:1:1 y:11:12:1:1|12|0"
        );
        let mut t = SimplePatternTokenizer::new("abc").unwrap();
        assert_eq!(run(&mut t, "ababc ab"), "abc:2:5:1:1|8|0");
        let mut t = SimplePatternSplitTokenizer::new("[ ,]+").unwrap();
        assert_eq!(
            run(&mut t, ", a,b ,c😀"),
            "a:2:3:1:1 b:4:5:1:1 c😀:7:10:1:1|10|0"
        );
        let mut t = SimplePatternSplitTokenizer::new("ab").unwrap();
        assert_eq!(run(&mut t, "xaay"), "xaay:0:4:1:1|4|0");
        assert_eq!(run(&mut t, "ab"), "|2|0");
        assert!(SimplePatternTokenizer::new("(").is_err());
        let a = lucene_util::automaton::automata::make_string("ab");
        let b = lucene_util::automaton::automata::make_string("ac");
        let nfa = operations::union(&[&a, &b]);
        assert!(!nfa.is_deterministic());
        assert!(SimplePatternTokenizer::from_automaton(&nfa).is_err());
        assert!(SimplePatternSplitTokenizer::from_automaton(&nfa).is_err());
    }
}
