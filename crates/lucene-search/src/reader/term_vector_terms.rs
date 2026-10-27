//! A document's term vector field as a [`TermsEnum`] and its terms'
//! postings: `TermVectors.get(doc, field).iterator()`, Java's
//! `Lucene90CompressingTermVectorsReader.TVTermsEnum`/`TVPostingsEnum`, over
//! the [`TermVectorField`] value this port decodes a vector into.
//!
//! The enum walks the field's terms in order (`seekCeil` binary-searches
//! them); each term's postings are one document, `0`, with the term's
//! frequency, and its positions, offsets and payloads when the field stored
//! them -- whatever flags were asked for, as `TVPostingsEnum` does. A
//! position that was not stored reads `-1`, an offset not stored `-1`, and
//! reading past the last position fails (Java's `IllegalStateException`).
//! Java's enum reuse has no counterpart: every call hands out a new enum.

use lucene_codecs::blocktree::SeekStatus;
use lucene_codecs::term_vectors::{TermVectorField, TermVectorTerm};

use super::{DocIdSetIterator, PostingsEnum, PostingsFlags, TermsEnum, NO_MORE_DOCS};
use crate::{Error, Result};

/// `TVTermsEnum`: the terms of one term vector field.
pub struct TermVectorTermsEnum<'a> {
    field: &'a TermVectorField,
    /// The current term's index; `None` before the first, `len` once past
    /// the last.
    at: Option<usize>,
}

impl<'a> TermVectorTermsEnum<'a> {
    /// `terms.iterator()`.
    pub fn new(field: &'a TermVectorField) -> Self {
        Self { field, at: None }
    }

    fn at_term(&self) -> Option<&'a TermVectorTerm> {
        self.at.and_then(|i| self.field.terms.get(i))
    }

    fn positioned(&self) -> Result<&'a TermVectorTerm> {
        self.at_term()
            .ok_or_else(|| Error::Unsupported("the TermsEnum is not on a term".into()))
    }
}

impl TermsEnum for TermVectorTermsEnum<'_> {
    fn next(&mut self) -> Result<Option<&[u8]>> {
        let len = self.field.terms.len();
        self.at = Some(match self.at {
            None => 0,
            Some(i) => (i + 1).min(len),
        });
        Ok(self.at_term().map(|t| t.term.as_slice()))
    }

    fn term(&self) -> Option<&[u8]> {
        self.at_term().map(|t| t.term.as_slice())
    }

    fn try_seek_ceil(&mut self, target: &[u8]) -> Result<SeekStatus> {
        let terms = &self.field.terms;
        let i = terms.partition_point(|t| t.term.as_slice() < target);
        self.at = Some(i);
        Ok(match terms.get(i) {
            None => SeekStatus::End,
            Some(t) if t.term == target => SeekStatus::Found,
            Some(_) => SeekStatus::NotFound,
        })
    }

    /// `docFreq()`: a term vector is one document.
    fn doc_freq(&mut self) -> Result<i32> {
        self.positioned().map(|_| 1)
    }

    fn total_term_freq(&mut self) -> Result<i64> {
        self.positioned().map(|t| i64::from(t.freq))
    }

    fn postings(&mut self, _flags: PostingsFlags) -> Result<Box<dyn PostingsEnum>> {
        let term = self.positioned()?;
        Ok(Box::new(TermVectorPostings::new(term.clone())))
    }
}

/// `TVPostingsEnum`: one term's occurrences in its one document.
pub struct TermVectorPostings {
    term: TermVectorTerm,
    /// `-1` before `nextDoc`, `0` on the document, [`NO_MORE_DOCS`] after.
    doc: i32,
    /// Positions read in the current document.
    read: usize,
}

impl TermVectorPostings {
    /// The postings of `term`, unpositioned.
    pub fn new(term: TermVectorTerm) -> Self {
        Self {
            term,
            doc: -1,
            read: 0,
        }
    }

    /// The occurrence the last `nextPosition` read.
    fn last(&self) -> Option<usize> {
        if self.doc == 0 {
            self.read.checked_sub(1)
        } else {
            None
        }
    }
}

impl DocIdSetIterator for TermVectorPostings {
    fn doc_id(&self) -> i32 {
        self.doc
    }

    fn next_doc(&mut self) -> Result<i32> {
        self.doc = if self.doc == -1 { 0 } else { NO_MORE_DOCS };
        self.read = 0;
        Ok(self.doc)
    }

    fn advance(&mut self, target: i32) -> Result<i32> {
        if self.doc == -1 && target <= 0 {
            return self.next_doc();
        }
        self.doc = NO_MORE_DOCS;
        Ok(self.doc)
    }

    fn cost(&self) -> i64 {
        1
    }
}

impl PostingsEnum for TermVectorPostings {
    fn freq(&self) -> i32 {
        self.term.freq
    }

    fn next_position(&mut self) -> Result<i32> {
        if self.doc != 0 || self.read >= usize::try_from(self.term.freq).unwrap_or(0) {
            return Err(Error::Unsupported(
                "read past last position (IllegalStateException)".into(),
            ));
        }
        let k = self.read;
        self.read += 1;
        Ok(self
            .term
            .positions
            .as_ref()
            .and_then(|p| p.get(k).copied())
            .unwrap_or(-1))
    }

    fn start_offset(&self) -> i32 {
        self.last()
            .and_then(|k| self.term.start_offsets.as_ref()?.get(k).copied())
            .unwrap_or(-1)
    }

    fn end_offset(&self) -> i32 {
        self.last()
            .and_then(|k| self.term.end_offsets.as_ref()?.get(k).copied())
            .unwrap_or(-1)
    }

    fn payload(&self) -> Option<&[u8]> {
        let k = self.last()?;
        self.term
            .payloads
            .as_ref()?
            .get(k)
            .map(Vec::as_slice)
            .filter(|p| !p.is_empty())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn term(text: &str, positions: bool, offsets: bool, payloads: bool) -> TermVectorTerm {
        TermVectorTerm {
            term: text.as_bytes().to_vec(),
            freq: 2,
            positions: positions.then(|| vec![0, 1]),
            start_offsets: offsets.then(|| vec![0, 4]),
            end_offsets: offsets.then(|| vec![3, 7]),
            payloads: payloads.then(|| vec![b"pay1".to_vec(), Vec::new()]),
        }
    }

    fn field() -> TermVectorField {
        TermVectorField {
            field_number: 0,
            has_positions: true,
            has_offsets: true,
            has_payloads: true,
            terms: vec![term("bar", true, true, true), term("foo", true, true, true)],
        }
    }

    #[test]
    fn walks_and_seeks_the_terms() {
        let f = field();
        let mut te = TermVectorTermsEnum::new(&f);
        assert_eq!(te.term(), None);
        assert!(te.doc_freq().is_err());
        assert!(te.postings(PostingsFlags::None).is_err());
        assert_eq!(te.next().unwrap(), Some(&b"bar"[..]));
        assert_eq!(te.doc_freq().unwrap(), 1);
        assert_eq!(te.total_term_freq().unwrap(), 2);
        assert_eq!(te.next().unwrap(), Some(&b"foo"[..]));
        assert_eq!(te.next().unwrap(), None);
        assert_eq!(te.next().unwrap(), None);
        assert_eq!(te.try_seek_ceil(b"bar").unwrap(), SeekStatus::Found);
        assert_eq!(te.term(), Some(&b"bar"[..]));
        assert_eq!(te.try_seek_ceil(b"c").unwrap(), SeekStatus::NotFound);
        assert_eq!(te.term(), Some(&b"foo"[..]));
        assert_eq!(te.try_seek_ceil(b"zzz").unwrap(), SeekStatus::End);
        assert_eq!(te.term(), None);
        assert!(te.try_seek_exact(b"foo").unwrap());
        assert!(!te.try_seek_exact(b"baz").unwrap());
        assert!(te.ord().is_err());
    }

    #[test]
    fn postings_read_one_document_with_what_was_stored() {
        let mut p = TermVectorPostings::new(term("bar", true, true, true));
        assert_eq!(p.doc_id(), -1);
        assert_eq!(p.cost(), 1);
        assert_eq!(p.start_offset(), -1);
        assert!(p.next_position().is_err(), "unpositioned");
        assert_eq!(p.next_doc().unwrap(), 0);
        assert_eq!(p.freq(), 2);
        assert_eq!(p.payload(), None);
        assert_eq!(p.next_position().unwrap(), 0);
        assert_eq!((p.start_offset(), p.end_offset()), (0, 3));
        assert_eq!(p.payload(), Some(&b"pay1"[..]));
        assert_eq!(p.next_position().unwrap(), 1);
        assert_eq!((p.start_offset(), p.end_offset()), (4, 7));
        assert_eq!(p.payload(), None, "an empty payload is none");
        assert!(p.next_position().is_err(), "past the last position");
        assert_eq!(p.next_doc().unwrap(), NO_MORE_DOCS);
        assert_eq!(p.next_doc().unwrap(), NO_MORE_DOCS);

        let mut bare = TermVectorPostings::new(term("bar", false, false, false));
        assert_eq!(bare.advance(0).unwrap(), 0);
        assert_eq!(bare.next_position().unwrap(), -1);
        assert_eq!((bare.start_offset(), bare.end_offset()), (-1, -1));
        assert_eq!(bare.payload(), None);
        assert_eq!(bare.advance(1).unwrap(), NO_MORE_DOCS);
        let mut past = TermVectorPostings::new(term("bar", false, false, false));
        assert_eq!(past.advance(1).unwrap(), NO_MORE_DOCS);
    }
}
