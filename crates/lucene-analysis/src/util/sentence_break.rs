//! `java.text.BreakIterator.getSentenceInstance()`: the JDK's legacy
//! rule-based sentence iterator, which `SegmentingTokenizerBase`'s
//! tokenizers (`HMMChineseTokenizer`, `ThaiTokenizer`) split their input
//! with. Not UAX #29, and not ICU's.
//!
//! The behaviour is re-specified here from black-box runs of the JDK (no JDK
//! code or data is copied, `docs/licences.md`), as `lang/final_sigma.rs`
//! re-specifies the word iterator. Every code point falls in one of thirteen
//! [`Class`]es -- from `Character.getType`, plus the JDK's own exceptions,
//! [`EXCEPTIONS`] -- and the first boundary after a position is the output
//! of a twelve-state scanner over those classes, learned from the JDK as a
//! Mealy machine (prefixes told apart by where every short continuation
//! breaks) and checked against it on every class string up to seven long
//! and on random texts:
//!
//! - The scan starts with a default boundary one code point ahead (two
//!   units for a supplementary one) and moves it: a state flagged *end*
//!   sets it after each character that enters it; a *lookahead* state
//!   records a pending boundary instead, which a later *lookahead-end*
//!   transition commits.
//! - A sentence runs to the first terminator (`!`, `?`, U+3002, U+FF01,
//!   U+FF1F), takes any terminators, periods, closing punctuation and
//!   quotes after it, then any spaces; or to a danda and its spaces; or
//!   through a paragraph separator (U+2029).
//! - After a period (`.`, U+FF0E) the boundary is pending: an upper-case or
//!   other letter (`Lu Lt Lm Lo`) after the period and its spaces, closing
//!   punctuation or quotes commits it; a lower-case letter does too, except
//!   after exactly one space; a digit, a second period, a letter right after
//!   the period, closing punctuation after a space, or a space after other
//!   punctuation resumes the sentence. Other characters are the JDK's
//!   corner: the scan can stop without committing, and the boundary falls
//!   back to the default one code point (`"a. #"` breaks as `a|.| #`).
//! - Non-spacing and enclosing marks and format characters do not move the
//!   scanner: they extend an *end* state's boundary, or a pending one, and
//!   at the start of a scan the default boundary.
//! - U+FFFF stops the scan as the text's end would, without the end's
//!   pending-boundary commit (it is the `CharacterIterator.DONE` the JDK's
//!   scanner tests for).
//!
//! Lone surrogates are `Other`; a scan never runs past the text it was
//! given (`SegmentingTokenizerBase` gives it the safe prefix of its buffer).
//!
//! The classes follow JDK 25 (Unicode 16.0, `java_character`), the JDK
//! OpenSearch 3.8 ships; [`EXCEPTIONS`] is printed by
//! `tools/GenSentenceBreakClasses.java` under it.

use std::sync::LazyLock;

use crate::java_character::{self as jc, char_count, code_point_at, get_type};

/// A code point's sentence-break class.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Class {
    /// Everything not below: symbols, most punctuation (opening included),
    /// spacing marks, controls, unassigned, private use, lone surrogates.
    Other = 0,
    /// `Zs`, `Zl`, `\t`, `\n`, `\f`, `\r`.
    Space,
    /// `!`, `?`, U+3002, U+FF01, U+FF1F.
    Term,
    /// `Pe`, `Pf`.
    Close,
    /// `"`, `'`.
    Quote,
    /// `Nd`, `Nl`, `No` and `,`.
    Digit,
    /// `.`, U+FF0E.
    Period,
    /// `Lu`, `Lt`, `Lm`, `Lo`.
    Upper,
    /// `Ll`.
    Lower,
    /// U+0964, U+0965.
    Danda,
    /// U+2029.
    Para,
    /// `Mn`, `Me`, `Cf`.
    Ignore,
    /// U+FFFF.
    Done,
}

/// Code points whose class is not the one [`class_of`]'s rule derives from
/// the general category, as JDK 25 classifies them: the last mark of some
/// supplementary runs reads as a digit, and some unassigned gaps between
/// supplementary ideograph blocks as letters. Printed by
/// `tools/GenSentenceBreakClasses.java`.
#[rustfmt::skip]
pub const EXCEPTIONS: [(u32, u32, Class); 128] = [
    (0x101FD, 0x101FD, Class::Digit),
    (0x102E0, 0x102E0, Class::Digit),
    (0x1037A, 0x1037A, Class::Digit),
    (0x10A03, 0x10A03, Class::Digit),
    (0x10A0F, 0x10A0F, Class::Digit),
    (0x10A3A, 0x10A3A, Class::Digit),
    (0x10A3F, 0x10A3F, Class::Digit),
    (0x10D27, 0x10D27, Class::Digit),
    (0x10D6D, 0x10D6D, Class::Digit),
    (0x10EFF, 0x10EFF, Class::Digit),
    (0x10F50, 0x10F50, Class::Digit),
    (0x10F85, 0x10F85, Class::Digit),
    (0x11001, 0x11001, Class::Digit),
    (0x11046, 0x11046, Class::Digit),
    (0x11070, 0x11070, Class::Digit),
    (0x11081, 0x11081, Class::Digit),
    (0x110B6, 0x110B6, Class::Digit),
    (0x110BD, 0x110BD, Class::Digit),
    (0x110C2, 0x110C2, Class::Digit),
    (0x110CD, 0x110CD, Class::Digit),
    (0x11102, 0x11102, Class::Digit),
    (0x1112B, 0x1112B, Class::Digit),
    (0x11134, 0x11134, Class::Digit),
    (0x11173, 0x11173, Class::Digit),
    (0x111BE, 0x111BE, Class::Digit),
    (0x111CC, 0x111CC, Class::Digit),
    (0x111CF, 0x111CF, Class::Digit),
    (0x11231, 0x11231, Class::Digit),
    (0x11234, 0x11234, Class::Digit),
    (0x1123E, 0x1123E, Class::Digit),
    (0x11241, 0x11241, Class::Digit),
    (0x112DF, 0x112DF, Class::Digit),
    (0x112EA, 0x112EA, Class::Digit),
    (0x11340, 0x11340, Class::Digit),
    (0x1136C, 0x1136C, Class::Digit),
    (0x11374, 0x11374, Class::Digit),
    (0x113C0, 0x113C0, Class::Digit),
    (0x113CE, 0x113CE, Class::Digit),
    (0x113D0, 0x113D0, Class::Digit),
    (0x113D2, 0x113D2, Class::Digit),
    (0x1143F, 0x1143F, Class::Digit),
    (0x11444, 0x11444, Class::Digit),
    (0x11446, 0x11446, Class::Digit),
    (0x1145E, 0x1145E, Class::Digit),
    (0x114B8, 0x114B8, Class::Digit),
    (0x114BA, 0x114BA, Class::Digit),
    (0x115B5, 0x115B5, Class::Digit),
    (0x1163A, 0x1163A, Class::Digit),
    (0x1163D, 0x1163D, Class::Digit),
    (0x116AB, 0x116AB, Class::Digit),
    (0x116AD, 0x116AD, Class::Digit),
    (0x116B5, 0x116B5, Class::Digit),
    (0x116B7, 0x116B7, Class::Digit),
    (0x1171D, 0x1171D, Class::Digit),
    (0x1171F, 0x1171F, Class::Digit),
    (0x11725, 0x11725, Class::Digit),
    (0x1172B, 0x1172B, Class::Digit),
    (0x11837, 0x11837, Class::Digit),
    (0x1193E, 0x1193E, Class::Digit),
    (0x11943, 0x11943, Class::Digit),
    (0x119D7, 0x119D7, Class::Digit),
    (0x119E0, 0x119E0, Class::Digit),
    (0x11A0A, 0x11A0A, Class::Digit),
    (0x11A38, 0x11A38, Class::Digit),
    (0x11A3E, 0x11A3E, Class::Digit),
    (0x11A47, 0x11A47, Class::Digit),
    (0x11A56, 0x11A56, Class::Digit),
    (0x11A5B, 0x11A5B, Class::Digit),
    (0x11A96, 0x11A96, Class::Digit),
    (0x11C36, 0x11C36, Class::Digit),
    (0x11C3D, 0x11C3D, Class::Digit),
    (0x11C3F, 0x11C3F, Class::Digit),
    (0x11CA7, 0x11CA7, Class::Digit),
    (0x11CB0, 0x11CB0, Class::Digit),
    (0x11D36, 0x11D36, Class::Digit),
    (0x11D3A, 0x11D3A, Class::Digit),
    (0x11D45, 0x11D45, Class::Digit),
    (0x11D47, 0x11D47, Class::Digit),
    (0x11D95, 0x11D95, Class::Digit),
    (0x11D97, 0x11D97, Class::Digit),
    (0x11F3A, 0x11F3A, Class::Digit),
    (0x11F40, 0x11F40, Class::Digit),
    (0x11F42, 0x11F42, Class::Digit),
    (0x11F5A, 0x11F5A, Class::Digit),
    (0x13440, 0x13440, Class::Digit),
    (0x13455, 0x13455, Class::Digit),
    (0x16129, 0x16129, Class::Digit),
    (0x1612F, 0x1612F, Class::Digit),
    (0x16AF4, 0x16AF4, Class::Digit),
    (0x16B36, 0x16B36, Class::Digit),
    (0x16F4F, 0x16F4F, Class::Digit),
    (0x16F92, 0x16F92, Class::Digit),
    (0x16FE4, 0x16FE4, Class::Digit),
    (0x1BCA3, 0x1BCA3, Class::Digit),
    (0x1CF2D, 0x1CF2D, Class::Digit),
    (0x1CF46, 0x1CF46, Class::Digit),
    (0x1D169, 0x1D169, Class::Digit),
    (0x1D182, 0x1D182, Class::Digit),
    (0x1D18B, 0x1D18B, Class::Digit),
    (0x1D1AD, 0x1D1AD, Class::Digit),
    (0x1D244, 0x1D244, Class::Digit),
    (0x1DA36, 0x1DA36, Class::Digit),
    (0x1DA6C, 0x1DA6C, Class::Digit),
    (0x1DA75, 0x1DA75, Class::Digit),
    (0x1DA84, 0x1DA84, Class::Digit),
    (0x1DA9F, 0x1DA9F, Class::Digit),
    (0x1DAAF, 0x1DAAF, Class::Digit),
    (0x1E006, 0x1E006, Class::Digit),
    (0x1E018, 0x1E018, Class::Digit),
    (0x1E021, 0x1E021, Class::Digit),
    (0x1E02A, 0x1E02A, Class::Digit),
    (0x1E08F, 0x1E08F, Class::Digit),
    (0x1E136, 0x1E136, Class::Digit),
    (0x1E2AE, 0x1E2AE, Class::Digit),
    (0x1E2EF, 0x1E2EF, Class::Digit),
    (0x1E4EF, 0x1E4EF, Class::Digit),
    (0x1E8D6, 0x1E8D6, Class::Digit),
    (0x1E94A, 0x1E94A, Class::Digit),
    (0x2A6E0, 0x2A6FF, Class::Upper),
    (0x2B73A, 0x2B73F, Class::Upper),
    (0x2B81E, 0x2B81F, Class::Upper),
    (0x2CEA2, 0x2CEAF, Class::Upper),
    (0x2EBE1, 0x2EBEF, Class::Upper),
    (0x2FA1E, 0x2FFFF, Class::Upper),
    (0x3134B, 0x3134F, Class::Upper),
    (0xE0001, 0xE0001, Class::Digit),
    (0xE007F, 0xE007F, Class::Digit),
    (0xE01EF, 0xE01EF, Class::Digit),
];

/// The class rule over the general category, before [`EXCEPTIONS`].
fn rule(cp: u32) -> Class {
    match cp {
        0xFFFF => return Class::Done,
        0x2029 => return Class::Para,
        0x0964 | 0x0965 => return Class::Danda,
        0x2E | 0xFF0E => return Class::Period,
        0x22 | 0x27 => return Class::Quote,
        0x21 | 0x3F | 0x3002 | 0xFF01 | 0xFF1F => return Class::Term,
        0x09 | 0x0A | 0x0C | 0x0D => return Class::Space,
        0x2C => return Class::Digit,
        _ => {}
    }
    match get_type(cp) {
        jc::SPACE_SEPARATOR | jc::LINE_SEPARATOR => Class::Space,
        jc::NON_SPACING_MARK | jc::ENCLOSING_MARK | jc::FORMAT => Class::Ignore,
        jc::END_PUNCTUATION | jc::FINAL_QUOTE_PUNCTUATION => Class::Close,
        jc::DECIMAL_DIGIT_NUMBER | jc::LETTER_NUMBER | jc::OTHER_NUMBER => Class::Digit,
        jc::LOWERCASE_LETTER => Class::Lower,
        jc::UPPERCASE_LETTER | jc::TITLECASE_LETTER | jc::MODIFIER_LETTER | jc::OTHER_LETTER => {
            Class::Upper
        }
        _ => Class::Other,
    }
}

/// The BMP's classes, the hot range, read once.
static BMP: LazyLock<Box<[Class]>> = LazyLock::new(|| {
    (0..0x10000u32)
        .map(rule)
        .collect::<Vec<_>>()
        .into_boxed_slice()
});

/// `cp`'s sentence-break class under JDK 25.
pub fn class_of(cp: u32) -> Class {
    if let Some(&c) = BMP.get(cp as usize) {
        return c;
    }
    let i = EXCEPTIONS.partition_point(|&(_, hi, _)| hi < cp);
    match EXCEPTIONS.get(i) {
        Some(&(lo, _, c)) if lo <= cp => c,
        _ => rule(cp),
    }
}

/// A scanner state (the states the JDK's machine is observed to have).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum State {
    /// Nothing read yet.
    Start,
    /// Inside a sentence. *End*.
    Body,
    /// After a terminator (and terminators, periods, closes, quotes). *End*.
    Term,
    /// After a terminator's spaces. *End*.
    TermSpace,
    /// After a danda (and its spaces). *End*.
    Danda,
    /// After a period, or a period and closes. *Lookahead*.
    Period,
    /// After a period and other punctuation, a quote or a danda.
    PeriodOther,
    /// After a period and one space. *Lookahead*.
    PeriodSpace,
    /// After a period and a quote. *Lookahead*.
    PeriodQuote,
    /// After a period and two or more spaces. *Lookahead*.
    PeriodSpaces,
    /// After a period, a quote and other punctuation.
    PeriodQuoteOther,
}

/// What a state does to the boundary when a character enters it.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Flag {
    None,
    /// The boundary moves after the character.
    End,
    /// The pending boundary moves after the character.
    Lookahead,
}

impl State {
    fn flag(self) -> Flag {
        match self {
            State::Body | State::Term | State::TermSpace | State::Danda => Flag::End,
            State::Period | State::PeriodSpace | State::PeriodQuote | State::PeriodSpaces => {
                Flag::Lookahead
            }
            State::Start | State::PeriodOther | State::PeriodQuoteOther => Flag::None,
        }
    }
}

/// A transition.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Step {
    Go(State),
    /// The scan ends before the character.
    Stop,
    /// The scan ends after the character (and the marks after it).
    StopAfter,
    /// The scan ends, committing the pending boundary.
    StopPending,
}

/// The transition from `s` on a class other than `Ignore` and `Done`.
fn step(s: State, c: Class) -> Step {
    use Class as C;
    use State as S;
    use Step::{Go, Stop, StopAfter, StopPending};
    match (s, c) {
        (_, C::Ignore | C::Done) => unreachable!("handled by the scan"),
        (S::Start | S::Body, C::Term) => Go(S::Term),
        (S::Start | S::Body, C::Period) => Go(S::Period),
        (S::Start | S::Body, C::Danda) => Go(S::Danda),
        (S::Start | S::Body, C::Para) => StopAfter,
        (S::Start | S::Body, _) => Go(S::Body),

        (S::Term, C::Term | C::Close | C::Quote | C::Period) => Go(S::Term),
        (S::Term | S::TermSpace, C::Space) => Go(S::TermSpace),
        (S::Term | S::TermSpace, C::Para) => StopAfter,
        (S::Term | S::TermSpace, _) => Stop,

        (S::Danda, C::Space) => Go(S::Danda),
        (S::Danda, _) => Stop,

        // After a period; the rows differ only where the JDK's do.
        (
            S::Period | S::PeriodOther | S::PeriodSpace | S::PeriodQuote | S::PeriodSpaces,
            C::Para,
        ) => StopAfter,
        (S::PeriodQuoteOther, C::Other | C::Quote | C::Danda) => Go(S::PeriodOther),
        (S::PeriodQuoteOther, C::Upper | C::Lower) => StopPending,
        (S::PeriodQuoteOther, _) => Stop,
        (_, C::Term) => Go(S::Term),
        (_, C::Period) => Go(S::Period),
        (_, C::Digit) => Go(S::Body),

        (S::Period, C::Other | C::Danda) => Go(S::PeriodOther),
        (S::Period, C::Space) => Go(S::PeriodSpace),
        (S::Period, C::Close) => Go(S::Period),
        (S::Period, C::Quote) => Go(S::PeriodQuote),
        (S::Period, _) => Go(S::Body),

        (S::PeriodOther, C::Other | C::Quote | C::Danda) => Go(S::PeriodOther),
        (S::PeriodOther, C::Space | C::Close) => Go(S::Body),
        (S::PeriodOther, _) => StopPending,

        (S::PeriodSpace, C::Other | C::Quote | C::Danda) => Go(S::PeriodOther),
        (S::PeriodSpace, C::Space) => Go(S::PeriodSpaces),
        (S::PeriodSpace, C::Close | C::Lower) => Go(S::Body),
        (S::PeriodSpace, _) => StopPending,

        (S::PeriodQuote, C::Other | C::Danda) => Go(S::PeriodQuoteOther),
        (S::PeriodQuote, C::Space) => Go(S::PeriodSpace),
        (S::PeriodQuote, C::Close) => Go(S::Period),
        (S::PeriodQuote, C::Quote) => Go(S::PeriodQuote),
        (S::PeriodQuote, _) => StopPending,

        (S::PeriodSpaces, C::Other | C::Quote | C::Danda) => Go(S::PeriodQuoteOther),
        (S::PeriodSpaces, C::Space) => Go(S::PeriodSpaces),
        (S::PeriodSpaces, C::Close) => Go(S::Body),
        (S::PeriodSpaces, _) => StopPending,
    }
}

/// How a scan that has stopped treats the marks after it.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Stopped {
    /// `StopAfter`: they extend the boundary.
    After,
    /// `StopPending`: they do not.
    Pending,
}

/// The first sentence boundary after `start` in `text`
/// (`BreakIterator.next()` from `start`), `None` at or past its end.
pub fn following_boundary(text: &[u16], start: usize) -> Option<usize> {
    let n = text.len();
    if start >= n {
        return None;
    }
    let mut result = start + char_count(code_point_at(text, start, n));
    let mut pending: Option<usize> = None;
    let mut state = State::Start;
    let mut stopped: Option<Stopped> = None;
    let mut i = start;
    while i < n {
        let cp = code_point_at(text, i, n);
        let p = i + char_count(cp);
        match class_of(cp) {
            Class::Done => return Some(result),
            Class::Ignore => match stopped {
                Some(Stopped::After) => result = p,
                Some(Stopped::Pending) => {}
                None => match state.flag() {
                    Flag::End => result = p,
                    Flag::Lookahead => pending = Some(p),
                    Flag::None if state == State::Start => result = p,
                    Flag::None => {}
                },
            },
            _ if stopped.is_some() => return Some(result),
            c => match step(state, c) {
                Step::Go(t) => {
                    match t.flag() {
                        Flag::End => result = p,
                        Flag::Lookahead => pending = Some(p),
                        Flag::None => {}
                    }
                    state = t;
                }
                Step::Stop => return Some(result),
                Step::StopAfter => {
                    result = p;
                    stopped = Some(Stopped::After);
                }
                Step::StopPending => {
                    result = pending.unwrap_or(result);
                    stopped = Some(Stopped::Pending);
                }
            },
        }
        i = p;
    }
    if stopped.is_none() && pending == Some(n) {
        return Some(n);
    }
    Some(result)
}

/// The `BreakIterator` contract `SegmentingTokenizerBase` drives: `setText`
/// over a window of its buffer, `current()`, `next()`.
pub trait BreakIterator {
    /// `setText(CharacterIterator)`: the text from now on; the position
    /// goes back to its start.
    fn set_text(&mut self, text: &[u16]);
    /// `current()`: the position, `0..=len`.
    fn current(&self) -> usize;
    /// `next()`: the next boundary, `None` (`DONE`) once at the end.
    fn next(&mut self) -> Option<usize>;
}

/// `BreakIterator.getSentenceInstance()` (every locale shares the JDK's
/// one set of sentence rules).
#[derive(Clone, Debug, Default)]
pub struct SentenceBreakIterator {
    text: Vec<u16>,
    pos: usize,
}

impl SentenceBreakIterator {
    pub fn new() -> Self {
        Self::default()
    }
}

impl BreakIterator for SentenceBreakIterator {
    fn set_text(&mut self, text: &[u16]) {
        self.text.clear();
        self.text.extend_from_slice(text);
        self.pos = 0;
    }

    fn current(&self) -> usize {
        self.pos
    }

    fn next(&mut self) -> Option<usize> {
        let b = following_boundary(&self.text, self.pos)?;
        self.pos = b;
        Some(b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn units(s: &str) -> Vec<u16> {
        s.encode_utf16().collect()
    }

    fn split(s: &str) -> Vec<String> {
        let u = units(s);
        let mut it = SentenceBreakIterator::new();
        it.set_text(&u);
        let mut out = Vec::new();
        let mut prev = it.current();
        while let Some(b) = it.next() {
            out.push(String::from_utf16_lossy(&u[prev..b]));
            prev = b;
        }
        assert_eq!(it.next(), None);
        assert_eq!(it.current(), u.len());
        out
    }

    #[test]
    fn sentences_as_the_jdk_splits_them() {
        // Each expectation is JDK 25's (and 21's) output.
        assert_eq!(
            split("Hello world. This is it. no more! Yes? ok."),
            ["Hello world. ", "This is it. no more! ", "Yes? ", "ok."]
        );
        assert_eq!(
            split("Mr. Smith went. e.g. this. 3.5 is. A.B. C"),
            ["Mr. ", "Smith went. e.g. this. 3.5 is. ", "A.B. ", "C"]
        );
        assert_eq!(
            split("He said \"Hi.\" Then. (Yes.) No."),
            ["He said \"Hi.\" ", "Then. ", "(Yes.) ", "No."]
        );
        assert_eq!(split("abc.def. ghi.  Jkl"), ["abc.def. ghi.  ", "Jkl"]);
        assert_eq!(split("x. 1 y. (z"), ["x. 1 y. ", "(z"]);
        // The default boundary: a scan that stops without committing.
        assert_eq!(split("a. #"), ["a", ".", " #"]);
        assert_eq!(split("a.#"), ["a", ".", "#"]);
        assert_eq!(split(".\u{964})#"), [".\u{964})#"]);
        // One space before a lower-case letter continues; two break.
        assert_eq!(split("a. b"), ["a. b"]);
        assert_eq!(split("a.  b"), ["a.  ", "b"]);
        // Paragraph separators, dandas, U+FFFF.
        assert_eq!(split("a.\u{2029}b"), ["a.\u{2029}", "b"]);
        assert_eq!(split("a\u{964}\u{2029}b"), ["a\u{964}", "\u{2029}", "b"]);
        assert_eq!(split("a\u{964}  b"), ["a\u{964}  ", "b"]);
        assert_eq!(split("a\u{FFFF}a"), ["a", "\u{FFFF}", "a"]);
        assert_eq!(split("a.\u{FFFF}a"), ["a", ".", "\u{FFFF}", "a"]);
        // Marks: transparent, extending the boundary.
        assert_eq!(split("a?\u{300} B"), ["a?\u{300} ", "B"]);
        assert_eq!(split("\u{300}\u{300}.#"), ["\u{300}\u{300}", ".", "#"]);
        assert_eq!(split("a\u{2029}\u{300}b"), ["a\u{2029}\u{300}", "b"]);
        // A supplementary default boundary is two units.
        assert_eq!(split("\u{1F600}.#"), ["\u{1F600}", ".", "#"]);
        assert_eq!(split(""), Vec::<String>::new());
    }

    #[test]
    fn classes_and_exceptions() {
        assert_eq!(class_of(u32::from(b'a')), Class::Lower);
        assert_eq!(class_of(0x4E2D), Class::Upper);
        assert_eq!(class_of(0x0903), Class::Other); // Mc
        assert_eq!(class_of(0x00AD), Class::Ignore);
        assert_eq!(class_of(0x00BB), Class::Close); // Pf
        assert_eq!(class_of(0x00AB), Class::Other); // Pi
        assert_eq!(class_of(0x0085), Class::Other);
        assert_eq!(class_of(0x2028), Class::Space);
        assert_eq!(class_of(0xD800), Class::Other);
        assert_eq!(class_of(0x1D167), Class::Ignore);
        assert_eq!(class_of(0x1D169), Class::Digit); // an exception
        assert_eq!(class_of(0x2A6E0), Class::Upper); // an exception
        assert_eq!(class_of(0x323B0), Class::Other);
        assert_eq!(class_of(0x10428), Class::Lower);
        assert_eq!(class_of(0x110000), Class::Other);
        // The table is sorted and disjoint.
        assert!(EXCEPTIONS.windows(2).all(|w| w[0].1 < w[1].0));
        assert!(EXCEPTIONS
            .iter()
            .all(|&(lo, hi, c)| lo <= hi && c != rule(lo)));
    }

    #[test]
    fn jdk25_only_classes() {
        // Unicode 16 (JDK 25): U+1171D, the mark before the changed U+1171E,
        // and the gap before CJK Extension I read differently under JDK 21.
        assert_eq!(class_of(0x1171D), Class::Digit);
        assert_eq!(class_of(0x2EBE1), Class::Upper);
        assert_eq!(class_of(0x18CD6), Class::Other);
    }
}
