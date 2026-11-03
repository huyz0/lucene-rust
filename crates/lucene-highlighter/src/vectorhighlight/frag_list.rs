//! `FieldFragList` and the `FragListBuilder`s that cut a document's phrases
//! into fragments.

use std::collections::HashSet;

use super::super::{Error, Result};
use super::field_phrase_list::{FieldPhraseList, Toffs, WeightedPhraseInfo};

/// `WeightedFragInfo.SubInfo`: one phrase of a fragment.
#[derive(Debug, Clone)]
pub struct SubInfo {
    /// The phrase's text.
    pub text: String,
    /// The ranges to mark up.
    pub terms_offsets: Vec<Toffs>,
    /// The phrase's number, which picks its tags.
    pub seqnum: i32,
    /// The phrase's weight.
    pub boost: f32,
}

/// `WeightedFragInfo`: a fragment, the phrases in it and their total weight.
#[derive(Debug, Clone)]
pub struct WeightedFragInfo {
    /// The phrases.
    pub sub_infos: Vec<SubInfo>,
    /// The fragment's score.
    pub total_boost: f32,
    /// Where the fragment starts.
    pub start_offset: i32,
    /// Where the fragment ends.
    pub end_offset: i32,
}

impl std::fmt::Display for WeightedFragInfo {
    /// `toString()`.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "subInfos=(")?;
        for si in &self.sub_infos {
            write!(f, "{}(", si.text)?;
            for to in &si.terms_offsets {
                write!(f, "{to}")?;
            }
            write!(f, ")")?;
        }
        write!(
            f,
            ")/{}({},{})",
            self.total_boost, self.start_offset, self.end_offset
        )
    }
}

/// How a fragment weighs its phrases.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FragListKind {
    /// `SimpleFieldFragList`: the sum of the phrases' boosts.
    Simple,
    /// `WeightedFieldFragList`: each distinct term's weight times its
    /// phrase's boost, normalised by the number of terms.
    Weighted,
}

/// `FieldFragList`.
#[derive(Debug, Clone)]
pub struct FieldFragList {
    kind: FragListKind,
    frag_infos: Vec<WeightedFragInfo>,
}

impl FieldFragList {
    /// An empty list weighing as `kind` does.
    pub fn new(kind: FragListKind) -> Self {
        Self {
            kind,
            frag_infos: Vec::new(),
        }
    }

    /// `add(startOffset, endOffset, phraseInfoList)`.
    pub fn add(&mut self, start_offset: i32, end_offset: i32, phrases: &[&WeightedPhraseInfo]) {
        match self.kind {
            FragListKind::Simple => {
                let mut total_boost = 0.0f32;
                let mut sub_infos = Vec::with_capacity(phrases.len());
                for p in phrases {
                    sub_infos.push(SubInfo {
                        text: p.text(),
                        terms_offsets: p.terms_offsets().to_vec(),
                        seqnum: p.seqnum(),
                        boost: p.boost(),
                    });
                    total_boost += p.boost();
                }
                self.frag_infos.push(WeightedFragInfo {
                    sub_infos,
                    total_boost,
                    start_offset,
                    end_offset,
                });
            }
            FragListKind::Weighted => {
                let mut temp = Vec::with_capacity(phrases.len());
                let mut distinct: HashSet<&str> = HashSet::new();
                let mut length = 0i32;
                for p in phrases {
                    let mut phrase_total_boost = 0.0f32;
                    for ti in p.terms_infos() {
                        if distinct.insert(ti.text()) {
                            phrase_total_boost += ti.weight() * p.boost();
                        }
                        length += 1;
                    }
                    temp.push(SubInfo {
                        text: p.text(),
                        terms_offsets: p.terms_offsets().to_vec(),
                        seqnum: p.seqnum(),
                        boost: phrase_total_boost,
                    });
                }
                // `length * (1 / (float) Math.sqrt(length))`
                let norm = length as f32 * (1.0 / f64::from(length).sqrt() as f32);
                let mut total_boost = 0.0f32;
                let mut sub_infos = Vec::with_capacity(temp.len());
                for mut s in temp {
                    s.boost *= norm;
                    total_boost += s.boost;
                    sub_infos.push(s);
                }
                self.frag_infos.push(WeightedFragInfo {
                    sub_infos,
                    total_boost,
                    start_offset,
                    end_offset,
                });
            }
        }
    }

    /// `getFragInfos()`.
    pub fn frag_infos(&self) -> &[WeightedFragInfo] {
        &self.frag_infos
    }

    /// The fragments, for a fragments builder that reorders and edits them.
    pub fn into_frag_infos(self) -> Vec<WeightedFragInfo> {
        self.frag_infos
    }
}

/// `FragListBuilder`: how a phrase list is cut into fragments.
pub trait FragListBuilder {
    /// `createFieldFragList(fieldPhraseList, fragCharSize)`.
    ///
    /// # Errors
    /// A `fragCharSize` the builder cannot work with.
    fn create_field_frag_list(
        &self,
        phrases: &FieldPhraseList,
        frag_char_size: i32,
    ) -> Result<FieldFragList>;
}

/// `BaseFragListBuilder.MARGIN_DEFAULT`.
pub const MARGIN_DEFAULT: i32 = 6;

/// `BaseFragListBuilder.MIN_FRAG_CHAR_SIZE_FACTOR`.
const MIN_FRAG_CHAR_SIZE_FACTOR: i32 = 3;

/// `BaseFragListBuilder`: fragments of about `fragCharSize` around the
/// phrases, `margin` characters before the first. [`SimpleFragListBuilder`]
/// and [`WeightedFragListBuilder`] differ in how the fragments weigh.
#[derive(Debug, Clone)]
pub struct BaseFragListBuilder {
    margin: i32,
    min_frag_char_size: i32,
    kind: FragListKind,
}

impl BaseFragListBuilder {
    fn with_margin(margin: i32, kind: FragListKind) -> Result<Self> {
        if margin < 0 {
            return Err(Error::IllegalArgument(format!(
                "margin({margin}) is too small. It must be 0 or higher."
            )));
        }
        Ok(Self {
            margin,
            min_frag_char_size: (margin.saturating_mul(MIN_FRAG_CHAR_SIZE_FACTOR)).max(1),
            kind,
        })
    }

    /// `acceptPhrase(info, matchLength, fragCharSize)`.
    fn accept_phrase(info: &WeightedPhraseInfo, match_length: i32, frag_char_size: i32) -> bool {
        info.terms_offsets().len() <= 1 || match_length <= frag_char_size
    }
}

impl FragListBuilder for BaseFragListBuilder {
    fn create_field_frag_list(
        &self,
        phrases: &FieldPhraseList,
        frag_char_size: i32,
    ) -> Result<FieldFragList> {
        if frag_char_size < self.min_frag_char_size {
            return Err(Error::IllegalArgument(format!(
                "fragCharSize({frag_char_size}) is too small. It must be {} or higher.",
                self.min_frag_char_size
            )));
        }
        let mut out = FieldFragList::new(self.kind);
        let list = phrases.phrase_list();
        let mut at = 0usize;
        let mut start_offset = 0i32;
        while let Some(phrase) = list.get(at) {
            if phrase.start_offset() < start_offset {
                at += 1;
                continue;
            }
            let mut wpil: Vec<&WeightedPhraseInfo> = Vec::new();
            let current_start = phrase.start_offset();
            let mut current_end = phrase.end_offset();
            let span_start = (current_start - self.margin).max(start_offset);
            let span_end = current_end.max(span_start.saturating_add(frag_char_size));
            at += 1;
            if Self::accept_phrase(phrase, current_end - current_start, frag_char_size) {
                wpil.push(phrase);
            }
            // pull until we crossed the current span end
            while let Some(next) = list.get(at) {
                if next.end_offset() > span_end {
                    break;
                }
                current_end = next.end_offset();
                at += 1;
                if Self::accept_phrase(next, current_end - current_start, frag_char_size) {
                    wpil.push(next);
                }
            }
            if wpil.is_empty() {
                continue;
            }
            let match_len = current_end - current_start;
            let new_margin = ((frag_char_size - match_len) / 2).max(0);
            let span_start = (current_start - new_margin).max(start_offset);
            let span_end = span_start + match_len.max(frag_char_size);
            start_offset = span_end;
            out.add(span_start, span_end, &wpil);
        }
        Ok(out)
    }
}

/// `SimpleFragListBuilder`.
pub struct SimpleFragListBuilder;

impl SimpleFragListBuilder {
    /// `new SimpleFragListBuilder()`: a base builder, as Java's subclass is.
    #[allow(clippy::new_ret_no_self)]
    pub fn new() -> BaseFragListBuilder {
        BaseFragListBuilder {
            margin: MARGIN_DEFAULT,
            min_frag_char_size: MARGIN_DEFAULT * MIN_FRAG_CHAR_SIZE_FACTOR,
            kind: FragListKind::Simple,
        }
    }

    /// `new SimpleFragListBuilder(margin)`.
    ///
    /// # Errors
    /// A negative margin.
    pub fn with_margin(margin: i32) -> Result<BaseFragListBuilder> {
        BaseFragListBuilder::with_margin(margin, FragListKind::Simple)
    }
}

/// `WeightedFragListBuilder`.
pub struct WeightedFragListBuilder;

impl WeightedFragListBuilder {
    /// `new WeightedFragListBuilder()`: a base builder, as Java's subclass is.
    #[allow(clippy::new_ret_no_self)]
    pub fn new() -> BaseFragListBuilder {
        BaseFragListBuilder {
            margin: MARGIN_DEFAULT,
            min_frag_char_size: MARGIN_DEFAULT * MIN_FRAG_CHAR_SIZE_FACTOR,
            kind: FragListKind::Weighted,
        }
    }

    /// `new WeightedFragListBuilder(margin)`.
    ///
    /// # Errors
    /// A negative margin.
    pub fn with_margin(margin: i32) -> Result<BaseFragListBuilder> {
        BaseFragListBuilder::with_margin(margin, FragListKind::Weighted)
    }
}

/// `SingleFragListBuilder`: one fragment holding the whole text.
#[derive(Debug, Default, Clone, Copy)]
pub struct SingleFragListBuilder;

impl FragListBuilder for SingleFragListBuilder {
    fn create_field_frag_list(
        &self,
        phrases: &FieldPhraseList,
        _frag_char_size: i32,
    ) -> Result<FieldFragList> {
        let mut out = FieldFragList::new(FragListKind::Simple);
        let all: Vec<&WeightedPhraseInfo> = phrases.phrase_list().iter().collect();
        if !all.is_empty() {
            out.add(0, i32::MAX, &all);
        }
        Ok(out)
    }
}
