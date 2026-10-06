//! The `miscellaneous` filters that hold a token back or build one from
//! several: `TypeAsSynonymFilter`, `HyphenatedWordsFilter`,
//! `FixBrokenOffsetsFilter`, `FingerprintFilter`, and the streaming
//! `ASCIIFoldingFilter`.

use std::collections::HashSet;

use crate::attributes::State;
use crate::token_stream::{TokenFilter, TokenStream};
use crate::AnalysisError;

/// `org.apache.lucene.analysis.miscellaneous.TypeAsSynonymFilter`: after each
/// token, a token at the same position whose term is its type (prefixed).
pub struct TypeAsSynonymFilter<I> {
    input: I,
    prefix: Option<String>,
    ignore: Option<HashSet<String>>,
    syn_flags_mask: i32,
    saved_token: Option<State>,
}

impl<I: TokenStream> TypeAsSynonymFilter<I> {
    /// `new TypeAsSynonymFilter(TokenStream, String prefix, Set<String>
    /// ignore, int synFlagsMask)`; `TypeAsSynonymFilter(in)` is `(in, None,
    /// None, !0)`.
    pub fn new(
        input: I,
        prefix: Option<&str>,
        ignore: Option<HashSet<String>>,
        syn_flags_mask: i32,
    ) -> Self {
        TypeAsSynonymFilter {
            input,
            prefix: prefix.map(str::to_string),
            ignore,
            syn_flags_mask,
            saved_token: None,
        }
    }
}

impl<I: TokenStream> TokenFilter for TypeAsSynonymFilter<I> {
    crate::filter_input!();
    // Java: TypeAsSynonymFilter.incrementToken
    fn increment(&mut self) -> Result<bool, AnalysisError> {
        if let Some(saved) = self.saved_token.take() {
            let a = self.input.attributes_mut();
            a.restore_state(&saved);
            let mut term = self.prefix.clone().unwrap_or_default();
            term.push_str(a.token_type());
            a.set_term(&term);
            a.set_position_increment(0)?;
            a.set_flags(a.flags() & self.syn_flags_mask);
            return Ok(true);
        }
        if self.input.increment_token()? {
            let a = self.input.attributes();
            if self
                .ignore
                .as_ref()
                .is_none_or(|i| !i.contains(a.token_type()))
            {
                self.saved_token = Some(a.capture_state());
            }
            return Ok(true);
        }
        Ok(false)
    }

    fn reset_filter(&mut self) -> Result<(), AnalysisError> {
        self.input.reset()?;
        self.saved_token = None;
        Ok(())
    }
}

/// `org.apache.lucene.analysis.miscellaneous.HyphenatedWordsFilter`: joins a
/// token ending in `-` with the next one.
pub struct HyphenatedWordsFilter<I> {
    input: I,
    hyphenated: String,
    saved_state: Option<State>,
    exhausted: bool,
    last_end_offset: i32,
}

impl<I: TokenStream> HyphenatedWordsFilter<I> {
    /// `new HyphenatedWordsFilter(TokenStream)`.
    pub fn new(input: I) -> Self {
        HyphenatedWordsFilter {
            input,
            hyphenated: String::new(),
            saved_state: None,
            exhausted: false,
            last_end_offset: 0,
        }
    }

    // Java: HyphenatedWordsFilter.unhyphenate
    fn unhyphenate(&mut self) -> Result<(), AnalysisError> {
        let saved = self
            .saved_state
            .take()
            .expect("unhyphenate after a saved state");
        let a = self.input.attributes_mut();
        a.restore_state(&saved);
        a.set_term(&self.hyphenated);
        let start = a.start_offset();
        a.set_offset(start, self.last_end_offset)?;
        self.hyphenated.clear();
        Ok(())
    }
}

impl<I: TokenStream> TokenFilter for HyphenatedWordsFilter<I> {
    crate::filter_input!();
    // Java: HyphenatedWordsFilter.incrementToken
    fn increment(&mut self) -> Result<bool, AnalysisError> {
        while !self.exhausted && self.input.increment_token()? {
            let a = self.input.attributes();
            self.last_end_offset = a.end_offset();
            let term = a.term();
            if let Some(stem) = term.strip_suffix('-') {
                if self.saved_state.is_none() {
                    self.saved_state = Some(a.capture_state());
                }
                self.hyphenated.push_str(stem);
            } else if self.saved_state.is_none() {
                return Ok(true);
            } else {
                self.hyphenated.push_str(term);
                self.unhyphenate()?;
                return Ok(true);
            }
        }
        self.exhausted = true;
        if self.saved_state.is_some() {
            self.hyphenated.push('-');
            self.unhyphenate()?;
            return Ok(true);
        }
        Ok(false)
    }

    fn reset_filter(&mut self) -> Result<(), AnalysisError> {
        self.input.reset()?;
        self.hyphenated.clear();
        self.saved_state = None;
        self.exhausted = false;
        self.last_end_offset = 0;
        Ok(())
    }
}

/// `org.apache.lucene.analysis.miscellaneous.FixBrokenOffsetsFilter`
/// (deprecated in Java): clamps offsets to be non-decreasing.
pub struct FixBrokenOffsetsFilter<I> {
    input: I,
    last_start_offset: i32,
}

impl<I: TokenStream> FixBrokenOffsetsFilter<I> {
    /// `new FixBrokenOffsetsFilter(TokenStream)`.
    pub fn new(input: I) -> Self {
        FixBrokenOffsetsFilter {
            input,
            last_start_offset: 0,
        }
    }

    // Java: FixBrokenOffsetsFilter.fixOffsets
    fn fix_offsets(&mut self) -> Result<(), AnalysisError> {
        let a = self.input.attributes_mut();
        let start = a.start_offset().max(self.last_start_offset);
        let end = a.end_offset().max(start);
        a.set_offset(start, end)?;
        self.last_start_offset = start;
        Ok(())
    }
}

impl<I: TokenStream> TokenFilter for FixBrokenOffsetsFilter<I> {
    crate::filter_input!();
    fn increment(&mut self) -> Result<bool, AnalysisError> {
        if !self.input.increment_token()? {
            return Ok(false);
        }
        self.fix_offsets()?;
        Ok(true)
    }

    fn end_filter(&mut self) -> Result<(), AnalysisError> {
        self.input.end()?;
        self.fix_offsets()
    }

    fn reset_filter(&mut self) -> Result<(), AnalysisError> {
        self.input.reset()?;
        self.last_start_offset = 0;
        Ok(())
    }
}

/// `FingerprintFilter.DEFAULT_MAX_OUTPUT_TOKEN_SIZE`.
pub const FINGERPRINT_DEFAULT_MAX_OUTPUT_TOKEN_SIZE: i32 = 1024;
/// `FingerprintFilter.DEFAULT_SEPARATOR`.
pub const FINGERPRINT_DEFAULT_SEPARATOR: char = ' ';

/// `org.apache.lucene.analysis.miscellaneous.FingerprintFilter`: one token,
/// the sorted unique terms joined by the separator.
pub struct FingerprintFilter<I> {
    input: I,
    max_output_token_size: i32,
    separator: char,
    final_state: Option<State>,
    input_ended: bool,
}

impl<I: TokenStream> FingerprintFilter<I> {
    /// `new FingerprintFilter(TokenStream, int maxOutputTokenSize, char separator)`.
    pub fn new(input: I, max_output_token_size: i32, separator: char) -> Self {
        FingerprintFilter {
            input,
            max_output_token_size,
            separator,
            final_state: None,
            input_ended: false,
        }
    }

    // Java: FingerprintFilter.buildSingleOutputToken
    fn build_single_output_token(&mut self) -> Result<bool, AnalysisError> {
        self.input_ended = false;
        // Insertion order is irrelevant: the terms are sorted below.
        let mut unique: Vec<Vec<u16>> = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();
        let mut last_term = String::new();
        let mut output_token_size: i64 = 0;
        while self.input.increment_token()? {
            if output_token_size > i64::from(self.max_output_token_size) {
                continue;
            }
            let term = self.input.attributes().term();
            if !seen.contains(term) {
                if !seen.is_empty() {
                    output_token_size += 1;
                }
                seen.insert(term.to_string());
                unique.push(term.encode_utf16().collect());
                last_term = term.to_string();
                output_token_size += self.input.attributes().term_utf16_len() as i64;
            }
        }
        self.input.end()?;
        self.input_ended = true;
        let a = self.input.attributes_mut();
        let end = a.end_offset();
        a.set_offset(0, end)?;
        a.set_position_length(1)?;
        a.set_position_increment(1)?;
        a.set_token_type("fingerprint");
        if unique.is_empty() || output_token_size > i64::from(self.max_output_token_size) {
            a.set_term("");
            return Ok(false);
        }
        if unique.len() == 1 {
            a.set_term(&last_term);
            return Ok(true);
        }
        // Java sorts the char[]s by UTF-16 unit, then length.
        unique.sort();
        let mut sb: Vec<u16> = Vec::new();
        let mut sep = [0u16; 2];
        for item in &unique {
            if !sb.is_empty() {
                sb.extend_from_slice(self.separator.encode_utf16(&mut sep));
            }
            sb.extend_from_slice(item);
        }
        a.set_term_utf16(&sb);
        Ok(true)
    }
}

impl<I: TokenStream> TokenFilter for FingerprintFilter<I> {
    crate::filter_input!();
    // Java: FingerprintFilter.incrementToken
    fn increment(&mut self) -> Result<bool, AnalysisError> {
        if self.input_ended {
            return Ok(false);
        }
        let result = self.build_single_output_token()?;
        self.final_state = Some(self.input.attributes().capture_state());
        Ok(result)
    }

    // Java: FingerprintFilter.end
    fn end_filter(&mut self) -> Result<(), AnalysisError> {
        if !self.input_ended {
            self.input.end()?;
            self.input_ended = true;
        }
        if let Some(state) = &self.final_state {
            self.input.attributes_mut().restore_state(state);
        }
        Ok(())
    }

    fn reset_filter(&mut self) -> Result<(), AnalysisError> {
        self.input.reset()?;
        self.input_ended = false;
        Ok(())
    }
}

/// `org.apache.lucene.analysis.miscellaneous.ASCIIFoldingFilter`, streaming:
/// folds every term with [`crate::AsciiFoldingFilter`]'s table; with
/// `preserveOriginal`, a changed term is followed by the original at the same
/// position.
pub struct AsciiFoldingTokenFilter<I> {
    input: I,
    preserve_original: bool,
    state: Option<State>,
}

impl<I: TokenStream> AsciiFoldingTokenFilter<I> {
    /// `new ASCIIFoldingFilter(TokenStream, boolean preserveOriginal)`.
    pub fn new(input: I, preserve_original: bool) -> Self {
        AsciiFoldingTokenFilter {
            input,
            preserve_original,
            state: None,
        }
    }

    /// `isPreserveOriginal()`.
    pub fn is_preserve_original(&self) -> bool {
        self.preserve_original
    }
}

impl<I: TokenStream> TokenFilter for AsciiFoldingTokenFilter<I> {
    crate::filter_input!();
    // Java: ASCIIFoldingFilter.incrementToken
    fn increment(&mut self) -> Result<bool, AnalysisError> {
        if let Some(state) = self.state.take() {
            let a = self.input.attributes_mut();
            a.restore_state(&state);
            a.set_position_increment(0)?;
            return Ok(true);
        }
        if !self.input.increment_token()? {
            return Ok(false);
        }
        // Java folds only a term holding a char >= U+0080, and preserves
        // the original only when folding changed it.
        if let Some(folded) = crate::AsciiFoldingFilter::fold_term(self.input.attributes().term()) {
            if self.preserve_original {
                self.state = Some(self.input.attributes().capture_state());
            }
            self.input.attributes_mut().set_term(&folded);
        }
        Ok(true)
    }

    fn reset_filter(&mut self) -> Result<(), AnalysisError> {
        self.input.reset()?;
        self.state = None;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::canned::{render, Canned};

    #[test]
    fn type_as_synonym() {
        let mut c = Canned::parse("a:0:1:1:1 b:2:3:1:1");
        c.set_types(&["<ALPHANUM>", "<NUM>"]);
        c.set_flags(&[7, 1]);
        let ignore: HashSet<String> = ["<NUM>".to_string()].into();
        let mut f = TypeAsSynonymFilter::new(c, Some("_"), Some(ignore), 3);
        let mut out = Vec::new();
        crate::token_stream::consume(&mut f, |a| {
            out.push((a.term().to_string(), a.position_increment(), a.flags()))
        })
        .unwrap();
        assert_eq!(
            out,
            vec![
                ("a".into(), 1, 7),
                ("_<ALPHANUM>".into(), 0, 3),
                ("b".into(), 1, 1)
            ]
        );
    }

    #[test]
    fn hyphenated_words() {
        let mut f = HyphenatedWordsFilter::new(Canned::parse(
            "ecologi-:0:8:1:1 cal:9:12:1:1 x:13:14:1:1 dangling-:15:24:1:1|24|0",
        ));
        assert_eq!(
            render(&mut f),
            "ecological:0:12:1:1 x:13:14:1:1 dangling-:15:24:1:1|24|0"
        );
        let mut f = HyphenatedWordsFilter::new(Canned::parse("a-:0:2:1:1 b-:3:5:1:1|5|0"));
        assert_eq!(render(&mut f), "ab-:0:5:1:1|5|0");
    }

    #[test]
    fn broken_offsets_and_fingerprints() {
        let mut f = FixBrokenOffsetsFilter::new(Canned::parse("a:5:6:1:1 b:2:3:1:1 c:7:9:1:1|1|0"));
        assert_eq!(render(&mut f), "a:5:6:1:1 b:5:5:1:1 c:7:9:1:1|7|0");
        let mut f = FingerprintFilter::new(
            Canned::parse("b:0:1:1:1 a:2:3:1:1 b:4:5:1:1 \u{FB01}:6:7:1:1 😀:8:10:1:1|10|0"),
            1024,
            ' ',
        );
        assert_eq!(render(&mut f), "a b 😀 \u{FB01}:0:10:1:1|10|1");
        let mut f =
            FingerprintFilter::new(Canned::parse("only:0:4:1:1 only:5:9:1:1|9|0"), 1024, ' ');
        assert_eq!(render(&mut f), "only:0:9:1:1|9|1");
        let mut f = FingerprintFilter::new(
            Canned::parse("abc:0:3:1:1 de:4:6:1:1 f:7:8:1:1|8|0"),
            4,
            '_',
        );
        assert_eq!(render(&mut f), "|8|1");
        let mut f = FingerprintFilter::new(Canned::parse("|4|2"), 4, '_');
        assert_eq!(render(&mut f), "|4|1");
        let mut f = FingerprintFilter::new(Canned::parse("a:0:1:1:1|4|2"), 4, '_');
        f.reset().unwrap();
        f.end().unwrap();
        assert_eq!(f.attributes().end_offset(), 4);
    }

    #[test]
    fn ascii_folding_streams() {
        let mut f =
            AsciiFoldingTokenFilter::new(Canned::parse("café:0:4:1:1 abc:5:8:1:1|8|0"), true);
        assert!(f.is_preserve_original());
        assert_eq!(render(&mut f), "cafe:0:4:1:1 café:0:4:0:1 abc:5:8:1:1|8|0");
        let mut f = AsciiFoldingTokenFilter::new(Canned::parse("Æb:0:2:1:1 中:3:4:1:1|4|0"), false);
        assert_eq!(render(&mut f), "AEb:0:2:1:1 中:3:4:1:1|4|0");
    }
}
