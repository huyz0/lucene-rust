//! Native postings for `TermScorer` and `PhraseScorer`: binary blocks or SimpleText's line
//! reader and up to four impacts levels. No text-to-binary conversion.
//! Differs: text impacts are copied after a fallible positioning call so
//! the scorer's bound getters remain infallible.

use lucene_codecs::postings::{
    Impact, LazyDocsCursor, Level0Impacts, PositionsCursor, NO_MORE_DOCS,
};
use lucene_codecs::simpletext::fields_reader::SimpleTextImpactsEnum;

use crate::{blocktree, Result};

// The binary scorer already keeps LazyDocsCursor inline. Preserve that
// allocation and access pattern while introducing the text alternative.
#[allow(clippy::large_enum_variant)]
pub(crate) enum TermPostings<'a> {
    Binary(LazyDocsCursor<'a>),
    Text(Box<TextPostings<'a>>),
}

impl<'a> From<LazyDocsCursor<'a>> for TermPostings<'a> {
    fn from(cursor: LazyDocsCursor<'a>) -> Self {
        Self::Binary(cursor)
    }
}

pub(crate) struct TextPostings<'a> {
    cursor: SimpleTextImpactsEnum<'a>,
    has_freqs: bool,
    levels: Vec<(i32, Vec<Impact>)>,
}

fn text_error(error: lucene_codecs::simpletext::Error) -> crate::Error {
    crate::directory_reader::Error::SimpleText(error).into()
}

impl TextPostings<'_> {
    fn refresh(&mut self) -> Result<()> {
        let impacts = self.cursor.impacts().map_err(text_error)?;
        self.levels
            .resize_with(impacts.num_levels(), || (NO_MORE_DOCS, Vec::new()));
        for (level, (end, pairs)) in self.levels.iter_mut().enumerate() {
            *end = impacts.doc_id_up_to(level).expect("existing impacts level");
            pairs.clear();
            pairs.extend_from_slice(impacts.impacts(level).expect("existing impacts level"));
        }
        Ok(())
    }

    fn postings(&self) -> &lucene_codecs::simpletext::fields_reader::SimpleTextPostings<'_> {
        match &self.cursor {
            SimpleTextImpactsEnum::Slow(p) | SimpleTextImpactsEnum::Skip(p) => p,
        }
    }
}

impl<'a> TermPostings<'a> {
    pub(crate) fn text(cursor: SimpleTextImpactsEnum<'a>, has_freqs: bool) -> Result<Self> {
        let mut text = TextPostings {
            cursor,
            has_freqs,
            levels: Vec::new(),
        };
        text.refresh()?;
        Ok(Self::Text(Box::new(text)))
    }

    pub(crate) fn doc_id(&self) -> i32 {
        match self {
            Self::Binary(c) => c.doc_id(),
            Self::Text(c) => c.postings().doc_id(),
        }
    }

    pub(crate) fn has_freqs(&self) -> bool {
        match self {
            Self::Binary(c) => c.has_freqs(),
            Self::Text(c) => c.has_freqs,
        }
    }

    pub(crate) fn binary_docs_only_impacts(&self) -> bool {
        matches!(self, Self::Binary(_)) && !self.has_freqs()
    }

    pub(crate) fn freq(&self) -> Option<i32> {
        match self {
            Self::Binary(c) => c.freq(),
            Self::Text(c) => (c.postings().doc_id() >= 0 && c.postings().doc_id() != NO_MORE_DOCS)
                .then(|| c.postings().freq()),
        }
    }

    pub(crate) fn next_doc(&mut self) -> Result<i32> {
        match self {
            Self::Binary(c) => Ok(c.next_doc().map_err(blocktree::Error::Postings)?),
            Self::Text(c) => {
                if c.postings().doc_id() == NO_MORE_DOCS {
                    return Ok(NO_MORE_DOCS);
                }
                let doc = c.cursor.postings().next_doc().map_err(text_error)?;
                c.refresh()?;
                Ok(doc)
            }
        }
    }

    pub(crate) fn advance(&mut self, target: i32) -> Result<i32> {
        match self {
            Self::Binary(c) => Ok(c.advance(target).map_err(blocktree::Error::Postings)?),
            Self::Text(c) => {
                // Search's binary cursor permits a target at or behind its
                // current document; every format keeps that same contract.
                if c.postings().doc_id() >= target {
                    return Ok(c.postings().doc_id());
                }
                let doc = c.cursor.postings().advance(target).map_err(text_error)?;
                c.refresh()?;
                Ok(doc)
            }
        }
    }

    pub(crate) fn advance_shallow(&mut self, target: i32) -> Result<i32> {
        match self {
            Self::Binary(c) => Ok(c
                .advance_shallow(target)
                .map_err(blocktree::Error::Postings)?),
            Self::Text(c) => {
                c.cursor.advance_shallow(target).map_err(text_error)?;
                c.refresh()?;
                Ok(c.levels[0].0)
            }
        }
    }

    pub(crate) fn num_levels(&self) -> usize {
        match self {
            Self::Binary(c) => {
                if c.level1_last_doc_id() != NO_MORE_DOCS && !c.level1_impacts().is_empty() {
                    2
                } else {
                    1
                }
            }
            Self::Text(c) => c.levels.len(),
        }
    }

    pub(crate) fn doc_id_up_to(&self, level: usize) -> i32 {
        match self {
            Self::Binary(c) => match level {
                0 => c.level0_last_doc_id(),
                1 => c.level1_last_doc_id(),
                _ => NO_MORE_DOCS,
            },
            Self::Text(c) => c.levels.get(level).map_or(NO_MORE_DOCS, |l| l.0),
        }
    }

    pub(crate) fn level0_last_doc_id(&self) -> i32 {
        self.doc_id_up_to(0)
    }

    pub(crate) fn level0_impacts_view(&self) -> Level0Impacts<'_> {
        match self {
            Self::Binary(c) => c.level0_impacts_view(),
            Self::Text(c) => Level0Impacts::Decoded(&c.levels[0].1),
        }
    }

    pub(crate) fn impacts(&mut self, level: usize) -> &[Impact] {
        match self {
            Self::Binary(c) => match level {
                0 => c.level0_impacts(),
                1 => c.level1_impacts(),
                _ => &[],
            },
            Self::Text(c) => c.levels.get(level).map_or(&[], |l| l.1.as_slice()),
        }
    }

    pub(crate) fn level0_impacts(&mut self) -> &[Impact] {
        self.impacts(0)
    }

    pub(crate) fn advance_exact_in_bits(&mut self, target: i32) -> Option<bool> {
        match self {
            Self::Binary(c) => c.advance_exact_in_bits(target),
            Self::Text(_) => None,
        }
    }

    pub(crate) fn doc_id_run_end(&self) -> i32 {
        match self {
            Self::Binary(c) => c.doc_id_run_end(),
            // Java's default DocIdSetIterator.docIDRunEnd().
            Self::Text(c) => c.postings().doc_id().saturating_add(1),
        }
    }

    pub(crate) fn next_postings_at_most(
        &mut self,
        up_to: i32,
        max: usize,
        docs: &mut Vec<i32>,
        freqs: &mut Vec<i32>,
    ) -> Result<()> {
        match self {
            Self::Binary(c) => Ok(c
                .next_postings_at_most(up_to, max, docs, freqs)
                .map_err(blocktree::Error::Postings)?),
            Self::Text(c) => {
                // Java: PostingsEnum.nextPostings, at most 16 documents.
                docs.clear();
                freqs.clear();
                let p = c.cursor.postings();
                while p.doc_id() >= 0 && p.doc_id() < up_to && docs.len() < max.clamp(1, 16) {
                    docs.push(p.doc_id());
                    freqs.push(p.freq());
                    p.next_doc().map_err(text_error)?;
                }
                c.refresh()?;
                Ok(())
            }
        }
    }
}

/// Native positional postings for PhraseWeight: the binary cursor stays
/// inline, while text retains its line reader and all skip levels.
#[allow(clippy::large_enum_variant)]
pub(crate) enum PhrasePostings<'a> {
    Binary(PositionsCursor<'a>),
    Text(Box<TextPostings<'a>>),
}

impl<'a> From<PositionsCursor<'a>> for PhrasePostings<'a> {
    fn from(cursor: PositionsCursor<'a>) -> Self {
        Self::Binary(cursor)
    }
}

impl<'a> PhrasePostings<'a> {
    pub(crate) fn text(cursor: SimpleTextImpactsEnum<'a>) -> Result<Self> {
        let mut text = TextPostings {
            cursor,
            has_freqs: true,
            levels: Vec::new(),
        };
        text.refresh()?;
        Ok(Self::Text(Box::new(text)))
    }

    pub(crate) fn doc_id(&self) -> i32 {
        match self {
            Self::Binary(c) => c.doc_id(),
            Self::Text(c) => c.postings().doc_id(),
        }
    }

    pub(crate) fn freq(&self) -> i32 {
        match self {
            Self::Binary(c) => c.freq(),
            Self::Text(c) => c.postings().freq(),
        }
    }

    pub(crate) fn next_doc(&mut self) -> Result<i32> {
        match self {
            Self::Binary(c) => Ok(c.next_doc().map_err(blocktree::Error::Postings)?),
            Self::Text(c) => {
                if c.postings().doc_id() == NO_MORE_DOCS {
                    return Ok(NO_MORE_DOCS);
                }
                let doc = c.cursor.postings().next_doc().map_err(text_error)?;
                c.refresh()?;
                Ok(doc)
            }
        }
    }

    pub(crate) fn advance(&mut self, target: i32) -> Result<i32> {
        match self {
            Self::Binary(c) => Ok(c.advance(target).map_err(blocktree::Error::Postings)?),
            Self::Text(c) => {
                if c.postings().doc_id() >= target {
                    return Ok(c.postings().doc_id());
                }
                let doc = c.cursor.postings().advance(target).map_err(text_error)?;
                c.refresh()?;
                Ok(doc)
            }
        }
    }

    pub(crate) fn next_position(&mut self) -> Result<i32> {
        match self {
            Self::Binary(c) => Ok(c.next_position().map_err(blocktree::Error::Postings)?),
            Self::Text(c) => c.cursor.postings().next_position().map_err(text_error),
        }
    }

    pub(crate) fn positions_into(&mut self, out: &mut Vec<i32>) -> Result<()> {
        match self {
            Self::Binary(c) => Ok(c.positions_into(out).map_err(blocktree::Error::Postings)?),
            Self::Text(c) => {
                for _ in 0..c.postings().freq() {
                    out.push(c.cursor.postings().next_position().map_err(text_error)?);
                }
                Ok(())
            }
        }
    }

    pub(crate) fn advance_shallow(&mut self, target: i32) -> Result<()> {
        match self {
            Self::Binary(c) => {
                c.docs_mut()
                    .advance_shallow(target)
                    .map_err(blocktree::Error::Postings)?;
            }
            Self::Text(c) => {
                c.cursor.advance_shallow(target).map_err(text_error)?;
                c.refresh()?;
            }
        }
        Ok(())
    }

    pub(crate) fn num_levels(&self) -> usize {
        match self {
            Self::Binary(c) => {
                if !c.docs().has_freqs() || c.docs().level1_last_doc_id() == NO_MORE_DOCS {
                    1
                } else {
                    2
                }
            }
            Self::Text(c) => c.levels.len(),
        }
    }

    pub(crate) fn doc_id_up_to(&self, level: usize) -> i32 {
        match self {
            Self::Binary(c) => {
                if !c.docs().has_freqs() {
                    return NO_MORE_DOCS;
                }
                match level {
                    0 => c.docs().level0_last_doc_id(),
                    1 => c.docs().level1_last_doc_id(),
                    _ => NO_MORE_DOCS,
                }
            }
            Self::Text(c) => c.levels.get(level).map_or(NO_MORE_DOCS, |l| l.0),
        }
    }

    pub(crate) fn level_impacts(&mut self, level: usize, out: &mut Vec<Impact>) {
        out.clear();
        match self {
            Self::Binary(c) => {
                let c = c.docs_mut();
                if !c.has_freqs() {
                    out.push(Impact { freq: 1, norm: 1 });
                } else if level == 0 && c.level0_last_doc_id() != NO_MORE_DOCS {
                    out.extend_from_slice(c.level0_impacts());
                } else if level == 1 {
                    out.extend_from_slice(c.level1_impacts());
                } else {
                    out.push(Impact {
                        freq: i32::MAX,
                        norm: 1,
                    });
                }
            }
            Self::Text(c) => {
                if let Some((_, pairs)) = c.levels.get(level) {
                    out.extend_from_slice(pairs);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests;
