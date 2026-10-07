//! `org.apache.lucene.analysis.compound`: decompounding filters that emit a
//! compound word's parts after it, at the same position --
//! `DictionaryCompoundWordTokenFilter` (dictionary subwords by brute force)
//! and `HyphenationCompoundWordTokenFilter` (subwords between the hyphenation
//! points of a [`HyphenationTree`], optionally checked against a
//! dictionary).
//!
//! `CompoundWordTokenFilterBase` is [`CompoundWordTokenFilter`] over a
//! [`Decompose`] strategy (Java's abstract `decompose()`); subwords are
//! UTF-16 slices of the term, as Java's `subSequence`.

pub mod hyphenation;

use std::collections::VecDeque;
use std::sync::Arc;

use crate::attributes::State;
use crate::token_stream::{TokenFilter, TokenStream};
use crate::{AnalysisError, CharArraySet};

pub use hyphenation::{HyphenationTree, PatternParser};

/// `CompoundWordTokenFilterBase.DEFAULT_MIN_WORD_SIZE`.
pub const DEFAULT_MIN_WORD_SIZE: usize = 5;
/// `CompoundWordTokenFilterBase.DEFAULT_MIN_SUBWORD_SIZE`.
pub const DEFAULT_MIN_SUBWORD_SIZE: usize = 2;
/// `CompoundWordTokenFilterBase.DEFAULT_MAX_SUBWORD_SIZE`.
pub const DEFAULT_MAX_SUBWORD_SIZE: usize = 15;

/// The base's settings (`minWordSize`, `minSubwordSize`, `maxSubwordSize`,
/// `onlyLongestMatch`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompoundSizes {
    /// Shorter terms are not decomposed.
    pub min_word_size: usize,
    /// The shortest subword.
    pub min_subword_size: usize,
    /// The longest subword.
    pub max_subword_size: usize,
    /// Only the longest dictionary match at each start.
    pub only_longest_match: bool,
}

impl Default for CompoundSizes {
    fn default() -> Self {
        CompoundSizes {
            min_word_size: DEFAULT_MIN_WORD_SIZE,
            min_subword_size: DEFAULT_MIN_SUBWORD_SIZE,
            max_subword_size: DEFAULT_MAX_SUBWORD_SIZE,
            only_longest_match: false,
        }
    }
}

/// `CompoundToken`: a subword, as its range of the decomposed term's UTF-16
/// units (Java copies it out; the term stays in `buf` until every subword
/// is emitted), and the whole token's offsets.
#[derive(Debug, Clone)]
struct CompoundToken {
    txt: std::ops::Range<usize>,
    start_offset: i32,
    end_offset: i32,
}

/// `CharArraySet.contains(char[], off, len)`.
fn dict_contains(dict: &CharArraySet, s: &[u16]) -> bool {
    dict.contains_utf16(s)
}

/// `CompoundWordTokenFilterBase.decompose()`: the subwords of `term`, as
/// `(start, length)` in UTF-16 units, in emission order.
pub trait Decompose: Send {
    /// Java's `decompose()`, over the term and the base's settings.
    fn decompose(&self, term: &[u16], sizes: &CompoundSizes, out: &mut Vec<(usize, usize)>);
}

/// `DictionaryCompoundWordTokenFilter.decompose`.
#[derive(Debug, Clone)]
pub struct DictionaryDecomposer {
    dictionary: Arc<CharArraySet>,
    only_longest_match_no_subwords: bool,
}

impl Decompose for DictionaryDecomposer {
    // Java: DictionaryCompoundWordTokenFilter.decompose
    fn decompose(&self, term: &[u16], sizes: &CompoundSizes, out: &mut Vec<(usize, usize)>) {
        let only_longest = sizes.only_longest_match || self.only_longest_match_no_subwords;
        let len = term.len();
        let mut i = 0usize;
        while i + sizes.min_subword_size <= len {
            let mut longest: Option<(usize, usize)> = None;
            for j in sizes.min_subword_size..=sizes.max_subword_size {
                if i + j > len {
                    break;
                }
                if dict_contains(&self.dictionary, &term[i..i + j]) {
                    if only_longest {
                        if longest.is_none_or(|(_, l)| l < j) {
                            longest = Some((i, j));
                        }
                    } else {
                        out.push((i, j));
                    }
                }
            }
            if let Some(m) = longest {
                out.push(m);
                if self.only_longest_match_no_subwords {
                    i += m.1 - 1;
                }
            }
            i += 1;
        }
    }
}

/// `HyphenationCompoundWordTokenFilter.decompose`.
#[derive(Debug, Clone)]
pub struct HyphenationDecomposer {
    hyphenator: Arc<HyphenationTree>,
    dictionary: Option<Arc<CharArraySet>>,
    no_sub_matches: bool,
    no_overlapping_matches: bool,
    calc_sub_matches: bool,
}

impl Decompose for HyphenationDecomposer {
    // Java: HyphenationCompoundWordTokenFilter.decompose
    fn decompose(&self, term: &[u16], sizes: &CompoundSizes, out: &mut Vec<(usize, usize)>) {
        let len = term.len();
        if let Some(d) = &self.dictionary {
            if !self.calc_sub_matches
                && (dict_contains(d, term) || len > 1 && dict_contains(d, &term[..len - 1]))
            {
                return; // the whole token is in the dictionary
            }
        }
        let Some(hyphens) = self.hyphenator.hyphenate(term, 1, 1) else {
            return;
        };
        let max_subword_size = sizes.max_subword_size.min(len.saturating_sub(1));
        let hyp = hyphens.hyphenation_points();
        let mut consumed: isize = -1;
        let mut i = 0usize;
        while i < hyp.len() {
            if self.no_overlapping_matches {
                i = i.max(consumed.max(0) as usize);
            }
            let start = hyp[i];
            let until = if self.no_sub_matches {
                consumed.max(i as isize)
            } else {
                i as isize
            };
            let mut j = hyp.len() as isize - 1;
            while j > until {
                let ju = j as usize;
                let part = hyp[ju] as isize - start as isize;
                if part > max_subword_size as isize {
                    j -= 1;
                    continue;
                }
                if part < sizes.min_subword_size as isize {
                    break;
                }
                let part = part as usize;
                let whole = self
                    .dictionary
                    .as_ref()
                    .is_none_or(|d| dict_contains(d, &term[start..start + part]));
                if whole {
                    out.push((start, part));
                    consumed = j;
                    if !self.calc_sub_matches {
                        break;
                    }
                } else if part > 0
                    && self
                        .dictionary
                        .as_ref()
                        .is_some_and(|d| dict_contains(d, &term[start..start + part - 1]))
                {
                    out.push((start, part - 1));
                    consumed = j;
                    if !self.calc_sub_matches {
                        break;
                    }
                }
                j -= 1;
            }
            i += 1;
        }
    }
}

/// `CompoundWordTokenFilterBase` over a [`Decompose`] strategy: each term at
/// least `min_word_size` long is followed by its subwords (position
/// increment 0, the term's offsets, every other attribute the term's).
pub struct CompoundWordTokenFilter<I, D> {
    input: I,
    decomposer: D,
    sizes: CompoundSizes,
    tokens: VecDeque<CompoundToken>,
    current: Option<State>,
    buf: Vec<u16>,
    parts: Vec<(usize, usize)>,
}

/// `DictionaryCompoundWordTokenFilter`.
pub type DictionaryCompoundWordTokenFilter<I> = CompoundWordTokenFilter<I, DictionaryDecomposer>;
/// `HyphenationCompoundWordTokenFilter`.
pub type HyphenationCompoundWordTokenFilter<I> = CompoundWordTokenFilter<I, HyphenationDecomposer>;

impl<I: TokenStream> CompoundWordTokenFilter<I, DictionaryDecomposer> {
    /// `new DictionaryCompoundWordTokenFilter(input, dictionary)`.
    pub fn dictionary(input: I, dictionary: Arc<CharArraySet>) -> Self {
        Self::dictionary_with(input, dictionary, CompoundSizes::default(), false)
    }

    /// `new DictionaryCompoundWordTokenFilter(input, dictionary,
    /// minWordSize, minSubwordSize, maxSubwordSize, onlyLongestMatch,
    /// onlyLongestMatchIgnoreSubwords)` (Java's negative-size
    /// `IllegalArgumentException`s cannot arise with `usize` sizes).
    pub fn dictionary_with(
        input: I,
        dictionary: Arc<CharArraySet>,
        sizes: CompoundSizes,
        only_longest_match_no_subwords: bool,
    ) -> Self {
        Self::with_decomposer(
            input,
            DictionaryDecomposer {
                dictionary,
                only_longest_match_no_subwords,
            },
            sizes,
        )
    }
}

impl<I: TokenStream> CompoundWordTokenFilter<I, HyphenationDecomposer> {
    /// `new HyphenationCompoundWordTokenFilter(input, hyphenator,
    /// dictionary, minWordSize, minSubwordSize, maxSubwordSize,
    /// onlyLongestMatch, noSubMatches, noOverlappingMatches)`; `dictionary`
    /// `None` takes every part between hyphenation points.
    pub fn hyphenation(
        input: I,
        hyphenator: Arc<HyphenationTree>,
        dictionary: Option<Arc<CharArraySet>>,
        sizes: CompoundSizes,
        no_sub_matches: bool,
        no_overlapping_matches: bool,
    ) -> Self {
        let calc_sub_matches =
            !sizes.only_longest_match && !no_sub_matches && !no_overlapping_matches;
        Self::with_decomposer(
            input,
            HyphenationDecomposer {
                hyphenator,
                dictionary,
                no_sub_matches,
                no_overlapping_matches,
                calc_sub_matches,
            },
            sizes,
        )
    }
}

impl<I: TokenStream, D: Decompose> CompoundWordTokenFilter<I, D> {
    /// The base over a decomposition strategy.
    pub fn with_decomposer(input: I, decomposer: D, sizes: CompoundSizes) -> Self {
        CompoundWordTokenFilter {
            input,
            decomposer,
            sizes,
            tokens: VecDeque::new(),
            current: None,
            buf: Vec::new(),
            parts: Vec::new(),
        }
    }
}

impl<I: TokenStream, D: Decompose> TokenFilter for CompoundWordTokenFilter<I, D> {
    crate::filter_input!();

    // Java: CompoundWordTokenFilterBase.incrementToken
    fn increment(&mut self) -> Result<bool, AnalysisError> {
        if let Some(token) = self.tokens.pop_front() {
            let a = self.input.attributes_mut();
            a.restore_state(self.current.as_ref().expect("a decomposed token"));
            a.set_term_utf16(&self.buf[token.txt]);
            a.set_offset(token.start_offset, token.end_offset)?;
            a.set_position_increment(0)?;
            return Ok(true);
        }
        self.current = None;
        if !self.input.increment_token()? {
            return Ok(false);
        }
        let a = self.input.attributes();
        if a.term_utf16_len() >= self.sizes.min_word_size {
            self.buf.clear();
            self.buf.extend(a.term().encode_utf16());
            self.parts.clear();
            self.decomposer
                .decompose(&self.buf, &self.sizes, &mut self.parts);
            let (start, end) = (a.start_offset(), a.end_offset());
            for &(off, len) in &self.parts {
                self.tokens.push_back(CompoundToken {
                    txt: off..off + len,
                    start_offset: start,
                    end_offset: end,
                });
            }
            if !self.tokens.is_empty() {
                self.current = Some(a.capture_state());
            }
        }
        Ok(true)
    }

    fn reset_filter(&mut self) -> Result<(), AnalysisError> {
        self.input.reset()?;
        self.tokens.clear();
        self.current = None;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::canned::{render, Canned};

    fn dict(words: &[&str]) -> Arc<CharArraySet> {
        Arc::new(CharArraySet::from_words(words, true))
    }

    fn run(mut f: impl TokenStream) -> String {
        render(&mut f)
    }

    // Expected outputs are Lucene 10.5.0's.
    #[test]
    fn dictionary_decomposition() {
        let d = dict(&["fuss", "ball", "fussball", "pumpe"]);
        let c = |t: &str| {
            let mut c = Canned::parse("x:0:13:1:1|13|0");
            c.set_terms(&[t]);
            c
        };
        assert_eq!(
            run(DictionaryCompoundWordTokenFilter::dictionary(c("fussballpumpe"), Arc::clone(&d))),
            "fussballpumpe:0:13:1:1 fuss:0:13:0:1 fussball:0:13:0:1 ball:0:13:0:1 pumpe:0:13:0:1|13|0"
        );
        let sizes = CompoundSizes {
            only_longest_match: true,
            ..CompoundSizes::default()
        };
        assert_eq!(
            run(DictionaryCompoundWordTokenFilter::dictionary_with(
                c("fussballpumpe"),
                Arc::clone(&d),
                sizes,
                false
            )),
            "fussballpumpe:0:13:1:1 fussball:0:13:0:1 ball:0:13:0:1 pumpe:0:13:0:1|13|0"
        );
        assert_eq!(
            run(DictionaryCompoundWordTokenFilter::dictionary_with(
                c("fussballpumpe"),
                Arc::clone(&d),
                CompoundSizes::default(),
                true
            )),
            "fussballpumpe:0:13:1:1 fussball:0:13:0:1 pumpe:0:13:0:1|13|0"
        );
        assert_eq!(
            run(DictionaryCompoundWordTokenFilter::dictionary(c("ball"), d)),
            "ball:0:13:1:1|13|0"
        );
    }
}
