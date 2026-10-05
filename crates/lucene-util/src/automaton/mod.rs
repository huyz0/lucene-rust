//! Lucene's public automaton API (`org.apache.lucene.util.automaton`), ported
//! class for class.
//!
//! This is the general, code-point-labelled [`Automaton`] with everything
//! Lucene builds on it: the [`automata`] factories, the [`operations`]
//! algebra (concatenate, union, intersection, minus, complement, determinize,
//! ...), [`RegExp`] with its 10.x flags, [`LevenshteinAutomata`] (with and
//! without transpositions), [`CompiledAutomaton`] and the run automata.
//! Algorithms follow the Java line by line so that state *numbering* -- not
//! just the accepted language -- matches Lucene; the differential test
//! `tests/automaton_fixtures.rs` holds it to Lucene 10.5.0's own output.
//!
//! The term-dictionary intersection in `lucene-codecs` keeps its own byte
//! DFA (`lucene-codecs/src/automaton.rs`), tuned for that one job; this
//! module is the API alongside it, not a replacement for it.
//!
//! Java-to-Rust conventions used throughout:
//!
//! - States and labels stay `i32`, as in Java, so `-1` keeps meaning "no
//!   state" wherever Lucene uses it (each such function says so).
//! - An exception Lucene throws becomes an [`AutomatonError`]; an
//!   `assert`/`checkIndex` on a programming error stays a panic.
//! - Identity checks (`a1 == a2` on objects) are [`std::ptr::eq`].
//! - Test-framework helpers Lucene 10 moved out of core
//!   (`AutomatonTestUtil.sameLanguage`, `subsetOf`, `isFinite`,
//!   `minimizeSimple`, `determinizeSimple`) live in [`operations`] too, since
//!   there is no test jar to put them in.

pub mod automata;
#[allow(clippy::module_inception)]
mod automaton;
mod case_folding;
mod compiled_automaton;
mod error;
mod finite_strings;
mod java_case_table;
mod lev_tables;
mod levenshtein;
mod nfa_run_automaton;
pub mod operations;
mod regexp;
mod run_automaton;
mod state_set;
mod strings_to_automaton;
mod utf32_to_utf8;

pub use automaton::{Automaton, Builder, StatePair, Transition, TransitionAccessor};
pub use case_folding::{expand as case_folding_expand, java_to_lower_case, java_to_upper_case};
pub use compiled_automaton::{AutomatonType, CompiledAutomaton};
pub use error::{AutomatonError, TooComplexToDeterminize};
pub use finite_strings::{FiniteStringsIterator, LimitedFiniteStringsIterator};
pub use levenshtein::{LevenshteinAutomata, MAXIMUM_SUPPORTED_DISTANCE};
pub use nfa_run_automaton::NfaRunAutomaton;
pub use operations::DEFAULT_DETERMINIZE_WORK_LIMIT;
pub use regexp::{AutomatonProvider, Kind, RegExp};
pub use run_automaton::{ByteRunAutomaton, ByteRunnable, CharacterRunAutomaton, RunAutomaton};
pub use state_set::{FrozenIntSet, IntSet, StateSet};
pub use strings_to_automaton::build as strings_to_automaton;
pub use utf32_to_utf8::Utf32ToUtf8;

/// `Character.MIN_CODE_POINT`.
pub const MIN_CODE_POINT: i32 = 0;
/// `Character.MAX_CODE_POINT`.
pub const MAX_CODE_POINT: i32 = 0x10FFFF;

/// `UnicodeUtil.newString(codePoints)` then `new BytesRef(CharSequence)`:
/// code points to UTF-16, then UTF-16 to UTF-8 with an unpaired surrogate
/// replaced by U+FFFD -- so two surrogate *code points* in a row that form a
/// valid pair encode as the supplementary character they spell, exactly as in
/// Java.
pub(crate) fn code_points_to_utf8(cps: &[i32]) -> Vec<u8> {
    let mut utf16: Vec<u32> = Vec::with_capacity(cps.len());
    for &cp in cps {
        let cp = cp as u32;
        if cp < 0x10000 {
            utf16.push(cp);
        } else {
            utf16.push(0xD7C0 + (cp >> 10));
            utf16.push(0xDC00 + (cp & 0x3FF));
        }
    }
    utf16_to_utf8(&utf16)
}

/// `UnicodeUtil.UTF16toUTF8` over UTF-16 code units.
pub(crate) fn utf16_to_utf8(units: &[u32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(units.len() * 3);
    let mut i = 0;
    while i < units.len() {
        let code = units[i];
        i += 1;
        if code < 0x80 {
            out.push(code as u8);
        } else if code < 0x800 {
            out.push((0xC0 | (code >> 6)) as u8);
            out.push((0x80 | (code & 0x3F)) as u8);
        } else if !(0xD800..=0xDFFF).contains(&code) {
            out.push((0xE0 | (code >> 12)) as u8);
            out.push((0x80 | ((code >> 6) & 0x3F)) as u8);
            out.push((0x80 | (code & 0x3F)) as u8);
        } else {
            if code < 0xDC00 && i < units.len() {
                let low = units[i];
                if (0xDC00..=0xDFFF).contains(&low) {
                    let utf32 = ((code - 0xD800) << 10) + (low - 0xDC00) + 0x10000;
                    i += 1;
                    out.push((0xF0 | (utf32 >> 18)) as u8);
                    out.push((0x80 | ((utf32 >> 12) & 0x3F)) as u8);
                    out.push((0x80 | ((utf32 >> 6) & 0x3F)) as u8);
                    out.push((0x80 | (utf32 & 0x3F)) as u8);
                    continue;
                }
            }
            out.extend_from_slice(&[0xEF, 0xBF, 0xBD]);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `UnicodeUtil.UTF16toUTF8`: well-formed text as UTF-8, a supplementary
    /// code point from its surrogate pair, and every unpaired surrogate as
    /// U+FFFD -- what a lossy decode of the same units encodes.
    #[test]
    fn utf16_and_code_points_encode_as_utf8() {
        let cps = [
            0x41, 0x7F, 0x80, 0x7FF, 0x800, 0xFFFF, 0x10000, 0x1F600, 0x10FFFF,
        ];
        let want: String = cps.iter().map(|&c| char::from_u32(c).unwrap()).collect();
        let cps: Vec<i32> = cps.iter().map(|&c| c as i32).collect();
        assert_eq!(code_points_to_utf8(&cps), want.as_bytes());
        for units in [
            &[0xD83D_u16, 0xDE00][..],
            &[0xD83D, 0x41],
            &[0x41, 0xD83D],
            &[0xDE00, 0x41],
            &[0xDE00, 0xD83D, 0xDE00],
            &[0xD800, 0xD800, 0xDC00],
        ] {
            let lossy = String::from_utf16_lossy(units);
            let wide: Vec<u32> = units.iter().map(|&u| u32::from(u)).collect();
            assert_eq!(utf16_to_utf8(&wide), lossy.as_bytes(), "{units:x?}");
        }
    }
}
