//! Port of `STBlockReader`, `STIntersectBlockReader` and
//! `STMergingBlockReader`: [`super::super`]'s terms enums over the shared
//! blocks, which skip the lines whose term the field lacks (the merging one
//! reads every field's).
//!
//! The overridden `readTermState`, `isBeyondLastTerm` and `nextTerm` live
//! in [`BlockReader`] (keyed on the terms' shared field infos and on
//! `shared_reader`); these wrap its `next`/`seekCeil`/`seekExact`.

use super::super::reader::field_postings;
use super::super::{
    BlockReader, Error, IntersectBlockReader, Result, SeekStatus, UniformSplitTermState,
    UniformSplitTerms,
};
use super::block_line::SharedFieldInfos;
use crate::postings::{DocInput, Postings, PostingsFlags};

/// `STBlockReader`: a field's `TermsEnum` over the shared blocks.
#[derive(Debug)]
pub struct StBlockReader<'a> {
    base: BlockReader<'a>,
}

impl<'a> StBlockReader<'a> {
    /// `new STBlockReader(...)` over `terms`, whose `shared` is set.
    pub(super) fn new(terms: &'a UniformSplitTerms) -> Self {
        let mut base = BlockReader::new(terms);
        base.shared_reader = true;
        StBlockReader { base }
    }

    /// `termOccursInField()`.
    fn term_occurs_in_field(&mut self) -> Result<bool> {
        Ok(self.base.read_term_state_opt()?.is_some())
    }

    /// `next()`: the next line whose term the field has.
    ///
    /// # Errors
    /// `Corrupted` for a malformed block.
    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> Result<Option<&[u8]>> {
        // Each turn reads a line or ends: the field's last block bounds it.
        loop {
            if self.base.next()?.is_none() {
                return Ok(None);
            }
            if self.term_occurs_in_field()? {
                return Ok(self.base.term());
            }
        }
    }

    /// `seekCeil(searchedTerm)`: on a term of another field, the field's
    /// next.
    ///
    /// # Errors
    /// `Corrupted` for a malformed block or dictionary.
    pub fn seek_ceil(&mut self, searched_term: &[u8]) -> Result<SeekStatus> {
        let mut seek_status = self.base.seek_ceil(searched_term)?;
        if seek_status != SeekStatus::End && !self.term_occurs_in_field()? {
            seek_status = if self.next()?.is_none() {
                SeekStatus::End
            } else {
                SeekStatus::NotFound
            };
        }
        Ok(seek_status)
    }

    /// `seekExact(searchedTerm)`: found, and the field's.
    ///
    /// # Errors
    /// `Corrupted` for a malformed block or dictionary.
    pub fn seek_exact(&mut self, searched_term: &[u8]) -> Result<bool> {
        Ok(self.base.seek_exact(searched_term)? && self.term_occurs_in_field()?)
    }

    /// `seekExact(term, state)`.
    pub fn seek_exact_state(&mut self, term: &[u8], state: &UniformSplitTermState) {
        self.base.seek_exact_state(term, state);
    }

    /// `term()`.
    pub fn term(&self) -> Option<&[u8]> {
        self.base.term()
    }

    /// `docFreq()`.
    ///
    /// # Errors
    /// See [`Self::term_state`].
    pub fn doc_freq(&mut self) -> Result<i32> {
        self.base.doc_freq()
    }

    /// `totalTermFreq()`.
    ///
    /// # Errors
    /// See [`Self::term_state`].
    pub fn total_term_freq(&mut self) -> Result<i64> {
        self.base.total_term_freq()
    }

    /// `termState()`.
    ///
    /// # Errors
    /// [`Error::Unsupported`] when not on a term of the field (Java: a
    /// `NullPointerException`); `Corrupted` for a malformed term state.
    pub fn term_state(&mut self) -> Result<UniformSplitTermState> {
        self.base.term_state()
    }

    /// `postings(reuse, flags)`.
    ///
    /// # Errors
    /// See [`Self::term_state`] and [`UniformSplitTerms::postings`].
    pub fn postings(&mut self, doc_in: &DocInput<'_>, flags: PostingsFlags) -> Result<Postings> {
        self.base.postings(doc_in, flags)
    }
}

/// `STIntersectBlockReader`: a field's terms an automaton accepts, over the
/// shared blocks.
#[derive(Debug)]
pub struct StIntersectBlockReader<'a> {
    inner: IntersectBlockReader<'a>,
}

impl<'a> StIntersectBlockReader<'a> {
    /// `new STIntersectBlockReader(...)` around the field's intersect.
    pub(super) fn new(inner: IntersectBlockReader<'a>) -> Self {
        StIntersectBlockReader { inner }
    }

    /// `next()`: the next accepted line whose term the field has.
    ///
    /// # Errors
    /// `Corrupted` for a malformed block or dictionary.
    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> Result<Option<&[u8]>> {
        // Each turn reads a line or ends: the field's last block bounds it.
        loop {
            if self.inner.next()?.is_none() {
                return Ok(None);
            }
            if self.inner.base.read_term_state_opt()?.is_some() {
                return Ok(self.inner.term());
            }
        }
    }

    /// `term()`.
    pub fn term(&self) -> Option<&[u8]> {
        self.inner.term()
    }

    /// `docFreq()`.
    ///
    /// # Errors
    /// See [`StBlockReader::term_state`].
    pub fn doc_freq(&mut self) -> Result<i32> {
        self.inner.doc_freq()
    }

    /// `totalTermFreq()`.
    ///
    /// # Errors
    /// See [`StBlockReader::term_state`].
    pub fn total_term_freq(&mut self) -> Result<i64> {
        self.inner.total_term_freq()
    }

    /// `termState()`.
    ///
    /// # Errors
    /// See [`StBlockReader::term_state`].
    pub fn term_state(&mut self) -> Result<UniformSplitTermState> {
        self.inner.term_state()
    }

    /// `postings(reuse, flags)`.
    ///
    /// # Errors
    /// See [`StBlockReader::postings`].
    pub fn postings(&mut self, doc_in: &DocInput<'_>, flags: PostingsFlags) -> Result<Postings> {
        self.inner.postings(doc_in, flags)
    }

    /// `seekCeil` and the `seekExact`s.
    ///
    /// # Errors
    /// Always [`Error::Unsupported`].
    pub fn seek_ceil(&mut self, term: &[u8]) -> Result<SeekStatus> {
        self.inner.seek_ceil(term)
    }
}

/// `STMergingBlockReader`: every line of the shared blocks, each with every
/// field's term state, over the union metadata.
#[derive(Debug)]
pub struct StMergingBlockReader<'a> {
    base: BlockReader<'a>,
}

impl<'a> StMergingBlockReader<'a> {
    /// `new STMergingBlockReader(...)` over the union `terms`.
    pub(super) fn new(union: &'a UniformSplitTerms) -> Self {
        let mut base = BlockReader::new(union);
        base.shared_reader = true;
        StMergingBlockReader { base }
    }

    /// `next()`: `nextTerm()`, every line up to the union's last term.
    ///
    /// # Errors
    /// `Corrupted` for a malformed block.
    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> Result<Option<&[u8]>> {
        self.base.next_term()
    }

    /// `term()`.
    pub fn term(&self) -> Option<&[u8]> {
        self.base.term()
    }

    /// `readFieldTermStatesMap(fieldTermStatesMap)`: the current line's
    /// `(fieldNumber, state)`s, in the line's (field-number) order; empty
    /// when not on a line.
    ///
    /// # Errors
    /// `Corrupted` for a malformed line or an unknown field number.
    pub fn read_field_term_states(
        &self,
        out: &mut Vec<(i32, UniformSplitTermState)>,
    ) -> Result<()> {
        self.base.read_field_term_states(out)
    }

    /// `postings(fieldName, termState, reuse, flags)`, by field number.
    ///
    /// # Errors
    /// `Corrupted` for a field number the segment lacks; as
    /// [`UniformSplitTerms::postings`].
    pub fn postings(
        &self,
        field_number: i32,
        state: &UniformSplitTermState,
        doc_in: &DocInput<'_>,
        flags: PostingsFlags,
    ) -> Result<Postings> {
        let infos: &SharedFieldInfos = self
            .base
            .terms
            .shared
            .as_deref()
            .ok_or(Error::Unsupported("not a shared-terms reader"))?;
        field_postings(infos.field(field_number)?, state, doc_in, flags)
    }

    /// `seekCeil`, `seekExact`, `readTermStateIfNotRead`: a merging reader
    /// only goes forward.
    ///
    /// # Errors
    /// Always [`Error::Unsupported`].
    pub fn seek_ceil(&mut self, _term: &[u8]) -> Result<SeekStatus> {
        Err(Error::Unsupported("a merging block reader does not seek"))
    }

    /// See [`Self::seek_ceil`].
    ///
    /// # Errors
    /// Always [`Error::Unsupported`].
    pub fn seek_exact(&mut self, _term: &[u8]) -> Result<bool> {
        Err(Error::Unsupported("a merging block reader does not seek"))
    }
}
