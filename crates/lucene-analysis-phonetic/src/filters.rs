//! The module's token filters: `PhoneticFilter`, `DoubleMetaphoneFilter`,
//! `BeiderMorseFilter` and `DaitchMokotoffSoundexFilter`
//! (`org.apache.lucene.analysis.phonetic`).

use std::collections::VecDeque;

use lucene_analysis::attributes::State;
use lucene_analysis::token_stream::{TokenFilter, TokenStream};
use lucene_analysis::AnalysisError;

use crate::bm::{CallerLanguages, PhoneticEngine};
use crate::daitch_mokotoff::DaitchMokotoffSoundex;
use crate::double_metaphone::DoubleMetaphone;
use crate::encoder::Encoder;
use crate::java::units;

macro_rules! input_accessors {
    () => {
        type Input = I;
        fn input(&self) -> &I {
            &self.input
        }
        fn input_mut(&mut self) -> &mut I {
            &mut self.input
        }
    };
}

/// `PhoneticFilter`: each term's code from a Commons Codec encoder, in
/// place of the term or (`inject`) as a token at the same position before
/// it. A term the encoder throws on, or codes to `""` or to itself, passes
/// unchanged.
pub struct PhoneticFilter<I> {
    input: I,
    inject: bool,
    encoder: Encoder,
    save: Option<State>,
    /// The term as UTF-16 units, reused across tokens.
    value: Vec<u16>,
}

impl<I: TokenStream> PhoneticFilter<I> {
    /// `new PhoneticFilter(in, encoder, inject)`.
    pub fn new(input: I, encoder: Encoder, inject: bool) -> Self {
        PhoneticFilter {
            input,
            inject,
            encoder,
            save: None,
            value: Vec::new(),
        }
    }
}

impl<I: TokenStream> TokenFilter for PhoneticFilter<I> {
    input_accessors!();

    // Java: PhoneticFilter.incrementToken
    fn increment(&mut self) -> Result<bool, AnalysisError> {
        if let Some(save) = self.save.take() {
            self.input.attributes_mut().restore_state(&save);
            return Ok(true);
        }
        if !self.input.increment_token()? {
            return Ok(false);
        }
        let attrs = self.input.attributes_mut();
        if attrs.term().is_empty() {
            return Ok(true);
        }
        self.value.clear();
        self.value.extend(attrs.term().encode_utf16());
        // Java: any exception from the encoder keeps the token as it is.
        let phonetic = match self.encoder.encode(&self.value) {
            Ok(v) if !v.is_empty() && v != self.value => v,
            _ => return Ok(true),
        };
        if !self.inject {
            attrs.set_term_utf16(&phonetic);
            return Ok(true);
        }
        let orig = attrs.position_increment();
        attrs.set_position_increment(0)?;
        self.save = Some(attrs.capture_state());
        attrs.set_position_increment(orig)?;
        attrs.set_term_utf16(&phonetic);
        Ok(true)
    }

    fn reset_filter(&mut self) -> Result<(), AnalysisError> {
        self.input.reset()?;
        self.save = None;
        Ok(())
    }
}

/// `DoubleMetaphoneFilter`: each term's primary and (when different)
/// alternate Double Metaphone codes, replacing the term or (`inject`)
/// after it at the same position.
pub struct DoubleMetaphoneFilter<I> {
    input: I,
    remaining: VecDeque<State>,
    encoder: DoubleMetaphone,
    inject: bool,
}

impl<I: TokenStream> DoubleMetaphoneFilter<I> {
    /// `new DoubleMetaphoneFilter(input, maxCodeLength, inject)`;
    /// `IllegalArgumentException` below 1.
    pub fn new(input: I, max_code_length: i32, inject: bool) -> Result<Self, AnalysisError> {
        if max_code_length < 1 {
            return Err(AnalysisError::IllegalArgument(
                "maxCodeLength must be >=1".into(),
            ));
        }
        let mut encoder = DoubleMetaphone::default();
        encoder.set_max_code_len(max_code_length);
        Ok(DoubleMetaphoneFilter {
            input,
            remaining: VecDeque::new(),
            encoder,
            inject,
        })
    }
}

impl<I: TokenStream> TokenFilter for DoubleMetaphoneFilter<I> {
    input_accessors!();

    // Java: DoubleMetaphoneFilter.incrementToken
    fn increment(&mut self) -> Result<bool, AnalysisError> {
        loop {
            if let Some(state) = self.remaining.pop_front() {
                self.input.attributes_mut().restore_state(&state);
                return Ok(true);
            }
            if !self.input.increment_token()? {
                return Ok(false);
            }
            let attrs = self.input.attributes_mut();
            if attrs.term().is_empty() {
                return Ok(true);
            }
            let mut first_alternative_increment = if self.inject {
                0
            } else {
                attrs.position_increment()
            };
            let v = units(attrs.term());
            let primary = self.encoder.double_metaphone(&v, false);
            let alternate = self.encoder.double_metaphone(&v, true);
            let mut save_state = self.inject;
            let primary_differs = primary.as_ref().is_some_and(|p| !p.is_empty() && *p != v);
            if let (true, Some(p)) = (primary_differs, &primary) {
                if save_state {
                    self.remaining.push_back(attrs.capture_state());
                }
                attrs.set_position_increment(first_alternative_increment)?;
                first_alternative_increment = 0;
                attrs.set_term_utf16(p);
                save_state = true;
            }
            if let (Some(a), Some(p)) = (&alternate, &primary) {
                if !a.is_empty() && a != p && *p != v {
                    if save_state {
                        // (Java also clears `saveState` here, then sets it.)
                        self.remaining.push_back(attrs.capture_state());
                    }
                    attrs.set_position_increment(first_alternative_increment)?;
                    attrs.set_term_utf16(a);
                    save_state = true;
                }
            }
            if self.remaining.is_empty() {
                return Ok(true);
            }
            if save_state {
                self.remaining.push_back(attrs.capture_state());
            }
        }
    }

    fn reset_filter(&mut self) -> Result<(), AnalysisError> {
        self.input.reset()?;
        self.remaining.clear();
        Ok(())
    }
}

/// `matcher.find()` over `encoded` from `*pos` for a run of units none of
/// which is in `stops` (Java's `([^...]+)`): the run, and `*pos` after it.
fn next_run(encoded: &[u16], pos: &mut usize, stops: &[u8]) -> Option<(usize, usize)> {
    let is_stop = |c: u16| stops.iter().any(|&s| u16::from(s) == c);
    let start = *pos + encoded.get(*pos..)?.iter().position(|&c| !is_stop(c))?;
    let end = encoded[start..]
        .iter()
        .position(|&c| is_stop(c))
        .map_or(encoded.len(), |e| start + e);
    *pos = end;
    Some((start, end))
}

/// `BeiderMorseFilter`: each term's Beider-Morse alternatives as tokens at
/// its position (the first one in place of the term).
pub struct BeiderMorseFilter<I> {
    input: I,
    engine: PhoneticEngine,
    languages: Option<CallerLanguages>,
    encoded: Vec<u16>,
    pos: usize,
    state: Option<State>,
}

/// `BeiderMorseFilter`'s matcher pattern `([^()|-]+)`.
const BM_STOPS: &[u8] = b"()|-";

impl<I: TokenStream> BeiderMorseFilter<I> {
    /// `new BeiderMorseFilter(input, engine)`: the languages guessed per term.
    pub fn new(input: I, engine: PhoneticEngine) -> Self {
        Self::with_languages(input, engine, None)
    }

    /// `new BeiderMorseFilter(input, engine, languages)`: `languages` names
    /// the term's languages (`None`: guess them).
    pub fn with_languages(input: I, engine: PhoneticEngine, languages: Option<&[String]>) -> Self {
        let languages = languages.map(|names| engine.languages(names));
        BeiderMorseFilter {
            input,
            engine,
            languages,
            encoded: Vec::new(),
            pos: 0,
            state: None,
        }
    }
}

impl<I: TokenStream> TokenFilter for BeiderMorseFilter<I> {
    input_accessors!();

    // Java: BeiderMorseFilter.incrementToken
    fn increment(&mut self) -> Result<bool, AnalysisError> {
        if let Some((s, e)) = next_run(&self.encoded, &mut self.pos, BM_STOPS) {
            let attrs = self.input.attributes_mut();
            if let Some(state) = &self.state {
                attrs.restore_state(state);
            }
            attrs.set_term_utf16(&self.encoded[s..e]);
            attrs.set_position_increment(0)?;
            return Ok(true);
        }
        if !self.input.increment_token()? {
            return Ok(false);
        }
        let attrs = self.input.attributes_mut();
        let term = units(attrs.term());
        self.encoded = match &self.languages {
            None => self.engine.encode(&term)?,
            Some(l) => self.engine.encode_with(&term, l)?,
        };
        self.state = Some(attrs.capture_state());
        self.pos = 0;
        if let Some((s, e)) = next_run(&self.encoded, &mut self.pos, BM_STOPS) {
            attrs.set_term_utf16(&self.encoded[s..e]);
        }
        Ok(true)
    }

    fn reset_filter(&mut self) -> Result<(), AnalysisError> {
        self.input.reset()?;
        self.encoded.clear();
        self.pos = 0;
        Ok(())
    }
}

/// `DaitchMokotoffSoundexFilter`: each term's Daitch-Mokotoff codes (every
/// branch) as tokens at its position, after the term (`inject`) or in its
/// place.
pub struct DaitchMokotoffSoundexFilter<I> {
    input: I,
    inject: bool,
    encoder: DaitchMokotoffSoundex,
    encoded: Vec<u16>,
    pos: usize,
    state: Option<State>,
}

impl<I: TokenStream> DaitchMokotoffSoundexFilter<I> {
    /// `new DaitchMokotoffSoundexFilter(in, inject)`.
    pub fn new(input: I, inject: bool) -> Self {
        DaitchMokotoffSoundexFilter {
            input,
            inject,
            encoder: DaitchMokotoffSoundex::default(),
            encoded: Vec::new(),
            pos: 0,
            state: None,
        }
    }
}

impl<I: TokenStream> TokenFilter for DaitchMokotoffSoundexFilter<I> {
    input_accessors!();

    // Java: DaitchMokotoffSoundexFilter.incrementToken
    fn increment(&mut self) -> Result<bool, AnalysisError> {
        if let Some((s, e)) = next_run(&self.encoded, &mut self.pos, b"|") {
            let attrs = self.input.attributes_mut();
            if let Some(state) = &self.state {
                attrs.restore_state(state);
            }
            attrs.set_term_utf16(&self.encoded[s..e]);
            attrs.set_position_increment(0)?;
            return Ok(true);
        }
        if !self.input.increment_token()? {
            return Ok(false);
        }
        let attrs = self.input.attributes_mut();
        if attrs.term().is_empty() {
            return Ok(true);
        }
        self.encoded = self.encoder.soundex(&units(attrs.term()));
        self.state = Some(attrs.capture_state());
        self.pos = 0;
        if !self.inject {
            if let Some((s, e)) = next_run(&self.encoded, &mut self.pos, b"|") {
                attrs.set_term_utf16(&self.encoded[s..e]);
            }
        }
        Ok(true)
    }

    fn reset_filter(&mut self) -> Result<(), AnalysisError> {
        self.input.reset()?;
        self.encoded.clear();
        self.pos = 0;
        self.state = None;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runs() {
        let e = units("(ab|c)-(d)");
        let mut pos = 0;
        assert_eq!(next_run(&e, &mut pos, BM_STOPS), Some((1, 3)));
        assert_eq!(next_run(&e, &mut pos, BM_STOPS), Some((4, 5)));
        assert_eq!(next_run(&e, &mut pos, BM_STOPS), Some((8, 9)));
        assert_eq!(next_run(&e, &mut pos, BM_STOPS), None);
        let mut pos = 0;
        assert_eq!(next_run(&[], &mut pos, b"|"), None);
        let mut pos = 5;
        assert_eq!(next_run(&units("a"), &mut pos, b"|"), None);
    }
}
