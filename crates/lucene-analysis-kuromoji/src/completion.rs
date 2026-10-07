//! `org.apache.lucene.analysis.ja.completion` (`CharSequenceUtils`,
//! `KatakanaRomanizer`) and `JapaneseCompletionFilter`: each token's surface
//! plus every romanization of its reading a user might be typing, for
//! input-method-aware suggestions.

use std::collections::HashMap;
use std::sync::LazyLock;

use lucene_analysis::{AnalysisError, TokenFilter, TokenStream};

use crate::attributes::ReadingAttribute;

/// `CharSequenceUtils`.
pub mod char_sequence_utils {
    fn is_hiragana(ch: u16) -> bool {
        (0x3040..=0x309f).contains(&ch)
    }
    fn is_katakana(ch: u16) -> bool {
        (0x30a0..=0x30ff).contains(&ch)
    }
    fn is_half_width_lowercase_alphabet(ch: u16) -> bool {
        (0x61..=0x7a).contains(&ch)
    }
    /// `isFullWidthLowercaseAlphabet(ch)`.
    pub fn is_full_width_lowercase_alphabet(ch: u16) -> bool {
        (0xff41..=0xff5a).contains(&ch)
    }
    /// `isLowercaseAlphabets(s)`.
    pub fn is_lowercase_alphabets(s: &[u16]) -> bool {
        s.iter()
            .all(|&c| is_half_width_lowercase_alphabet(c) || is_full_width_lowercase_alphabet(c))
    }
    /// `isKana(s)`.
    pub fn is_kana(s: &[u16]) -> bool {
        s.iter().all(|&c| is_hiragana(c) || is_katakana(c))
    }
    /// `isKatakanaOrHWAlphabets(s)`.
    pub fn is_katakana_or_hw_alphabets(s: &[u16]) -> bool {
        s.iter()
            .all(|&c| is_katakana(c) || is_half_width_lowercase_alphabet(c))
    }
    /// `toKatakana(s)`: ぁ..ゖ, ゝ and ゞ shifted to katakana.
    pub fn to_katakana(s: &[u16]) -> Vec<u16> {
        s.iter()
            .map(|&ch| {
                if (0x3041..=0x3096).contains(&ch) || ch == 0x309d || ch == 0x309e {
                    ch.wrapping_add(0x60)
                } else {
                    ch
                }
            })
            .collect()
    }
}

use char_sequence_utils as csu;

/// `KatakanaRomanizer`: `romaji_map.txt`'s keystrokes, by length, sorted.
pub struct KatakanaRomanizer {
    keystrokes: Vec<Vec<Vec<u16>>>,
    romaji_map: HashMap<Vec<u16>, Vec<Vec<u16>>>,
}

impl KatakanaRomanizer {
    /// `getInstance()`.
    pub fn instance() -> &'static KatakanaRomanizer {
        static INSTANCE: LazyLock<KatakanaRomanizer> =
            LazyLock::new(|| KatakanaRomanizer::parse(include_str!("resources/romaji_map.txt")));
        &INSTANCE
    }

    fn parse(text: &str) -> KatakanaRomanizer {
        let mut romaji_map: HashMap<Vec<u16>, Vec<Vec<u16>>> = HashMap::new();
        for line in crate::dict::user_dictionary::read_lines(text) {
            if line.starts_with('#') {
                continue;
            }
            let mut cols: Vec<&str> = lucene_analysis::factory::java_trim(line)
                .split(',')
                .collect();
            while cols.last().is_some_and(|c| c.is_empty()) {
                cols.pop();
            }
            if cols.len() < 2 {
                continue;
            }
            let prefix: Vec<u16> = cols[0].encode_utf16().collect();
            romaji_map.insert(
                prefix,
                cols[1..]
                    .iter()
                    .map(|c| c.encode_utf16().collect())
                    .collect(),
            );
        }
        let max = romaji_map.keys().map(Vec::len).max().unwrap_or(0);
        let mut keystrokes: Vec<Vec<Vec<u16>>> = (1..=max)
            .map(|l| {
                romaji_map
                    .keys()
                    .filter(|k| k.len() == l)
                    .cloned()
                    .collect()
            })
            .collect();
        for ks in &mut keystrokes {
            // keystroke array must be sorted in ascending order for binary
            // search.
            ks.sort();
        }
        KatakanaRomanizer {
            keystrokes,
            romaji_map,
        }
    }

    /// `romanize(input)`: every keystroke sequence a user may type for the
    /// katakana `input`.
    pub fn romanize(&self, input: &[u16]) -> Vec<Vec<u16>> {
        let mut pending: Vec<Vec<u16>> = Vec::new();
        let mut pos = 0usize;
        while pos < input.len() {
            // Greedily looks up the longest matched keystroke.
            let Some(key) = self.longest_keystroke_match(input, pos) else {
                break;
            };
            let candidates = &self.romaji_map[key];
            if pending.is_empty() {
                pending = candidates.clone();
            } else if candidates.len() == 1 {
                for p in &mut pending {
                    p.extend_from_slice(&candidates[0]);
                }
            } else {
                let mut outputs =
                    Vec::with_capacity(candidates.len().saturating_mul(pending.len()));
                for c in candidates {
                    for p in &pending {
                        let mut b = p.clone();
                        b.extend_from_slice(c);
                        outputs.push(b);
                    }
                }
                pending = outputs;
            }
            pos = pos.saturating_add(key.len());
        }
        if pos < input.len() {
            // add the remnants (that cannot be mapped to any romaji) as
            // suffix
            for p in &mut pending {
                p.extend_from_slice(&input[pos..]);
            }
        }
        pending
    }

    fn longest_keystroke_match(&self, input: &[u16], offset: usize) -> Option<&Vec<u16>> {
        let max = input
            .len()
            .saturating_sub(offset)
            .min(self.keystrokes.len());
        for len in (1..=max).rev() {
            let r = &input[offset..offset.saturating_add(len)];
            let ks = &self.keystrokes[len.saturating_sub(1)];
            if let Ok(i) = ks.binary_search_by(|k| k.as_slice().cmp(r)) {
                return Some(&ks[i]);
            }
        }
        // there's no matched keystroke
        None
    }
}

/// `JapaneseCompletionFilter.Mode`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CompletionMode {
    /// `INDEX` (`DEFAULT_MODE`).
    #[default]
    Index,
    /// `QUERY`: also joins a word split mid-IME-composition and runs of
    /// kana.
    Query,
}

impl CompletionMode {
    /// `Mode.valueOf(name)`.
    pub fn value_of(name: &str) -> Result<Self, AnalysisError> {
        match name {
            "INDEX" => Ok(CompletionMode::Index),
            "QUERY" => Ok(CompletionMode::Query),
            _ => Err(AnalysisError::IllegalArgument(format!(
                "No enum constant org.apache.lucene.analysis.ja.JapaneseCompletionFilter.Mode.{name}"
            ))),
        }
    }
}

/// `CompletionToken`.
#[derive(Debug, Clone)]
struct CompletionToken {
    term: Vec<u16>,
    is_first: bool,
    start_offset: i32,
    end_offset: i32,
}

/// `CompletionTokenGenerator`.
#[derive(Debug, Default)]
struct Generator {
    mode: CompletionMode,
    outputs: std::collections::VecDeque<CompletionToken>,
    /// `pdgSurface` and `pdgReading` (both set or both `None`).
    pending: Option<(Vec<u16>, Vec<u16>)>,
    pdg_start_offset: i32,
    pdg_end_offset: i32,
}

/// `CharsRefBuilder.append(CharSequence)`: `null` appends `"null"`.
fn append(b: &mut Vec<u16>, s: Option<&[u16]>) {
    match s {
        Some(s) => b.extend_from_slice(s),
        None => b.extend("null".encode_utf16()),
    }
}

impl Generator {
    fn reset(&mut self) {
        self.pending = None;
        self.outputs.clear();
    }

    fn add_token(&mut self, surface: &[u16], reading: Option<&[u16]>, start: i32, end: i32) {
        let Some((pdg_surface, pdg_reading)) = self.pending.as_mut() else {
            self.reset_pending_token(surface, reading, start, end);
            return;
        };
        if self.mode == CompletionMode::Query
            && !csu::is_lowercase_alphabets(pdg_surface)
            && csu::is_lowercase_alphabets(surface)
        {
            // words that are in mid-IME composition are split into two
            // tokens by JapaneseTokenizer; should be recovered when
            // querying (the surface stands in for the null reading).
            pdg_surface.extend_from_slice(surface);
            pdg_reading.extend_from_slice(surface);
            self.pdg_end_offset = end;
            self.generate_outputs();
            self.pending = None;
        } else if self.mode == CompletionMode::Query
            && csu::is_kana(pdg_surface)
            && csu::is_kana(surface)
        {
            // words that are all composed only of Katakana or Hiragana
            // should be concatenated when querying.
            pdg_surface.extend_from_slice(surface);
            append(pdg_reading, reading);
            self.pdg_end_offset = end;
        } else {
            self.generate_outputs();
            self.reset_pending_token(surface, reading, start, end);
        }
    }

    fn finish(&mut self) {
        self.generate_outputs();
        self.pending = None;
    }

    fn generate_outputs(&mut self) {
        let Some((surface, reading)) = &self.pending else {
            return;
        };
        // preserve original surface form as an output.
        self.outputs.push_back(CompletionToken {
            term: surface.clone(),
            is_first: true,
            start_offset: self.pdg_start_offset,
            end_offset: self.pdg_end_offset,
        });
        // skip readings that cannot be translated to romaji.
        if reading.is_empty() || !csu::is_katakana_or_hw_alphabets(reading) {
            return;
        }
        // translate the reading to romaji.
        for r in KatakanaRomanizer::instance().romanize(reading) {
            // set the same start/end offset as the original surface form
            // for romanized tokens.
            self.outputs.push_back(CompletionToken {
                term: r,
                is_first: false,
                start_offset: self.pdg_start_offset,
                end_offset: self.pdg_end_offset,
            });
        }
    }

    fn reset_pending_token(
        &mut self,
        surface: &[u16],
        reading: Option<&[u16]>,
        start: i32,
        end: i32,
    ) {
        let mut r = Vec::new();
        append(&mut r, reading);
        self.pending = Some((surface.to_vec(), r));
        self.pdg_start_offset = start;
        self.pdg_end_offset = end;
    }
}

/// `JapaneseCompletionFilter`.
pub struct JapaneseCompletionFilter<I> {
    input: I,
    generator: Generator,
    input_stream_consumed: bool,
}

impl<I: TokenStream> JapaneseCompletionFilter<I> {
    /// `new JapaneseCompletionFilter(input, mode)`.
    pub fn new(mut input: I, mode: CompletionMode) -> Self {
        input.attributes_mut().add_custom::<ReadingAttribute>();
        JapaneseCompletionFilter {
            input,
            generator: Generator {
                mode,
                ..Generator::default()
            },
            input_stream_consumed: false,
        }
    }

    // Java: mayIncrementToken
    fn may_increment_token(&mut self) -> Result<(), AnalysisError> {
        while self.generator.outputs.is_empty() {
            if !self.input_stream_consumed && self.input.increment_token()? {
                let atts = self.input.attributes();
                let surface: Vec<u16> = atts.term().encode_utf16().collect();
                let mut reading: Option<Vec<u16>> = atts
                    .custom::<ReadingAttribute>()
                    .and_then(ReadingAttribute::reading)
                    .map(|r| r.encode_utf16().collect());
                let (start, end) = (atts.start_offset(), atts.end_offset());
                if reading.is_none() && csu::is_kana(&surface) {
                    // use the surface form as reading when possible.
                    reading = Some(csu::to_katakana(&surface));
                }
                self.generator
                    .add_token(&surface, reading.as_deref(), start, end);
            } else {
                self.input_stream_consumed = true;
                if self.generator.pending.is_some() {
                    // a pending token remains.
                    self.generator.finish();
                } else {
                    // already consumed all tokens.
                    break;
                }
            }
        }
        Ok(())
    }
}

impl<I: TokenStream> TokenFilter for JapaneseCompletionFilter<I> {
    type Input = I;
    fn input(&self) -> &I {
        &self.input
    }
    fn input_mut(&mut self) -> &mut I {
        &mut self.input
    }
    fn increment(&mut self) -> Result<bool, AnalysisError> {
        self.may_increment_token()?;
        let Some(token) = self.generator.outputs.pop_front() else {
            return Ok(false);
        };
        let atts = self.input.attributes_mut();
        atts.clear_attributes();
        atts.set_term_utf16(&token.term);
        atts.set_position_increment(if token.is_first { 1 } else { 0 })?;
        atts.set_offset(token.start_offset, token.end_offset)?;
        Ok(true)
    }
    fn reset_filter(&mut self) -> Result<(), AnalysisError> {
        self.input.reset()?;
        self.generator.reset();
        self.input_stream_consumed = false;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn u(s: &str) -> Vec<u16> {
        s.encode_utf16().collect()
    }

    #[test]
    fn romanizes_every_keystroke_pattern() {
        let r = KatakanaRomanizer::instance();
        let out: Vec<String> = r
            .romanize(&u("シンブン"))
            .iter()
            .map(|v| String::from_utf16_lossy(v))
            .collect();
        assert!(out.contains(&"shinbun".to_string()), "{out:?}");
        assert!(out.contains(&"sinbun".to_string()), "{out:?}");
        assert!(r.romanize(&u("null")).is_empty());
        let out: Vec<String> = r
            .romanize(&u("アx"))
            .iter()
            .map(|v| String::from_utf16_lossy(v))
            .collect();
        assert_eq!(out, ["ax"]);
        assert!(csu::is_full_width_lowercase_alphabet(0xff41));
        assert_eq!(
            CompletionMode::value_of("QUERY").unwrap(),
            CompletionMode::Query
        );
        assert!(CompletionMode::value_of("x").is_err());
    }
}
