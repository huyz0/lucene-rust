//! `SimpleTextTermVectorsReader`: the `.tvc` written by
//! [`super::term_vectors_writer`], walked once at open (`readIndex`) for
//! where each document starts, then parsed a document at a time.
//!
//! What differs from Java:
//! - The reader borrows the file's bytes. [`SimpleTextTermVectorsReader::get`]
//!   returns each field as a name and this crate's [`TermVectorField`]
//!   (`SimpleTVFields`/`SimpleTVTerms`), fields ordered by name in UTF-16
//!   code units (Java's `TreeMap<String>`) and terms by unsigned bytes
//!   (`TreeMap<BytesRef>`); a repeated name or term keeps its last value,
//!   as `put` does. A `null` payload is an empty one.
//!   The field number Java parses and drops is kept.
//!   Rebound readers share the immutable document index, as Java's clones do.
//!   [`SimpleTextTvTermsEnum`] and [`SimpleTextTvPostings`] are
//!   `SimpleTVTermsEnum` and its two postings enums in one.
//! - Java's `assert`s on each line's prefix and on the document count, its
//!   `NumberFormatException`s, its negative-array and index errors (a
//!   negative `freq` where the field has positions or offsets, more
//!   documents than `maxDoc`, a document outside it) are `Corrupted` (`IllegalArgument` for the document number). Arrays
//!   grow as lines are read rather than being sized from `freq`.

use std::cmp::Ordering;
use std::sync::Arc;

use super::term_vectors_writer::{
    DOC, END, ENDOFFSET, FIELD, FIELDNAME, FIELDOFFSETS, FIELDPAYLOADS, FIELDPOSITIONS,
    FIELDTERMCOUNT, NUMFIELDS, PAYLOAD, POSITION, STARTOFFSET, TERMFREQ, TERMTEXT,
};
use super::util::{self, parse_boolean, parse_int, read_line, read_value};
use super::{
    corrupt, feature_requested, Error, Result, SeekStatus, FREQS, NO_MORE_DOCS, POSITIONS,
};
use crate::term_vectors::{TermVectorField, TermVectorTerm};

/// `SimpleTextTermVectorsReader` over the whole `.tvc`.
#[derive(Debug, Clone)]
pub struct SimpleTextTermVectorsReader<'d> {
    data: &'d [u8],
    /// docid -> offset in the `.tvc`, just past its `doc` line.
    offsets: Arc<[usize]>,
}

/// Java's `String.compareTo`: UTF-16 code units.
fn cmp_utf16(a: &str, b: &str) -> Ordering {
    a.encode_utf16().cmp(b.encode_utf16())
}

impl<'d> SimpleTextTermVectorsReader<'d> {
    /// `new SimpleTextTermVectorsReader(directory, si, context)`:
    /// `readIndex(si.maxDoc())`.
    ///
    /// # Errors
    /// `Corrupted` for a document count other than `max_doc` or a bad
    /// checksum; `Eof` for a file without `END`.
    pub fn open(data: &'d [u8], max_doc: i32) -> Result<Self> {
        // Java: SimpleTextTermVectorsReader.readIndex
        let size = usize::try_from(max_doc)
            .map_err(|_| Error::IllegalArgument(format!("invalid maxDoc {max_doc}")))?;
        // ALLOC: offsets grow one per `doc N` line read, so never past the
        // file's line count.
        let mut offsets = Vec::new();
        let mut scratch = Vec::new();
        let mut pos = 0usize;
        while scratch != END {
            read_line(data, &mut pos, &mut scratch)?;
            if scratch.starts_with(DOC) {
                if offsets.len() == size {
                    return Err(corrupt(format!("more than maxDoc={max_doc} documents")));
                }
                offsets.push(pos);
            }
        }
        util::check_footer(data, 0, &mut pos)?;
        if offsets.len() != size {
            return Err(corrupt(format!(
                "{} documents, expected maxDoc={max_doc}",
                offsets.len()
            )));
        }
        Ok(SimpleTextTermVectorsReader {
            data,
            offsets: offsets.into(),
        })
    }

    /// Rebind the immutable document index to the same `.tvc` bytes.
    /// Clones share this index, as Java's cloned reader does.
    pub fn rebind<'n>(&self, data: &'n [u8]) -> SimpleTextTermVectorsReader<'n> {
        SimpleTextTermVectorsReader {
            data,
            offsets: self.offsets.clone(),
        }
    }

    /// The number of documents (`maxDoc`).
    pub fn max_doc(&self) -> usize {
        self.offsets.len()
    }

    /// `get(doc)`: `None` for a document without vectors (`numfields 0`),
    /// else its fields by name.
    ///
    /// # Errors
    /// [`Error::IllegalArgument`] for `doc` outside the segment; `Corrupted`
    /// for a malformed line; `Eof` for a short file.
    pub fn get(&self, doc: i32) -> Result<Option<Vec<(String, TermVectorField)>>> {
        // Java: SimpleTextTermVectorsReader.get
        let mut pos = usize::try_from(doc)
            .ok()
            .and_then(|d| self.offsets.get(d).copied())
            .ok_or_else(|| Error::IllegalArgument(format!("document {doc} out of bounds")))?;
        let data = self.data;
        let mut scratch = Vec::new();
        let num_fields = parse_int(read_value(data, &mut pos, &mut scratch, NUMFIELDS)?)?;
        if num_fields == 0 {
            return Ok(None);
        }
        let mut fields: Vec<(String, TermVectorField)> = Vec::new();
        for _ in 0..num_fields.max(0) {
            let field_number = parse_int(read_value(data, &mut pos, &mut scratch, FIELD)?)?;
            let name = util::read_string(data, &mut pos, &mut scratch, FIELDNAME)?;
            let positions =
                parse_boolean(read_value(data, &mut pos, &mut scratch, FIELDPOSITIONS)?);
            let offsets = parse_boolean(read_value(data, &mut pos, &mut scratch, FIELDOFFSETS)?);
            let payloads = parse_boolean(read_value(data, &mut pos, &mut scratch, FIELDPAYLOADS)?);
            let term_count = parse_int(read_value(data, &mut pos, &mut scratch, FIELDTERMCOUNT)?)?;
            let mut terms: Vec<TermVectorTerm> = Vec::new();
            for _ in 0..term_count.max(0) {
                let text = read_value(data, &mut pos, &mut scratch, TERMTEXT)?.to_vec();
                let freq = parse_int(read_value(data, &mut pos, &mut scratch, TERMFREQ)?)?;
                let term = read_postings(
                    data,
                    &mut pos,
                    &mut scratch,
                    text,
                    freq,
                    positions,
                    offsets,
                    payloads,
                )?;
                terms.push(term);
            }
            // `terms.put(term, postings)` into a TreeMap<BytesRef>: byte
            // order, a repeated term keeping its last postings.
            sort_keep_last(&mut terms, |a, b| a.term.cmp(&b.term));
            let field = TermVectorField {
                field_number,
                has_positions: positions,
                has_offsets: offsets,
                has_payloads: payloads,
                terms,
            };
            fields.push((name, field));
        }
        // `fields.put(fieldName, terms)` into a TreeMap<String>: UTF-16
        // order, a repeated name keeping its last field.
        sort_keep_last(&mut fields, |a, b| cmp_utf16(&a.0, &b.0));
        Ok(Some(fields))
    }
}

/// Sorts `v` by `cmp` and keeps, of each run of equal keys, the last one
/// pushed (a `TreeMap.put` per element, without its O(n^2) inserts here).
fn sort_keep_last<T>(v: &mut Vec<T>, cmp: impl Fn(&T, &T) -> Ordering) {
    // A stable sort keeps equal keys in push order; reversed, the last
    // pushed comes first in its run and `dedup_by` keeps it.
    v.sort_by(&cmp);
    v.reverse();
    v.dedup_by(|a, b| cmp(a, b) == Ordering::Equal);
    v.reverse();
}

/// One term's `freq` occurrences: positions (and payloads), offsets.
#[allow(clippy::too_many_arguments)]
fn read_postings(
    data: &[u8],
    pos: &mut usize,
    scratch: &mut Vec<u8>,
    term: Vec<u8>,
    freq: i32,
    positions: bool,
    offsets: bool,
    payloads: bool,
) -> Result<TermVectorTerm> {
    let mut t = TermVectorTerm {
        term,
        freq,
        positions: None,
        start_offsets: None,
        end_offsets: None,
        payloads: None,
    };
    if !(positions || offsets) {
        return Ok(t);
    }
    if freq < 0 {
        return Err(corrupt(format!("negative freq {freq}")));
    }
    let mut ps = Vec::new();
    let mut pays = Vec::new();
    let mut starts = Vec::new();
    let mut ends = Vec::new();
    for _ in 0..freq {
        if positions {
            ps.push(parse_int(read_value(data, pos, scratch, POSITION)?)?);
            if payloads {
                pays.push(read_value(data, pos, scratch, PAYLOAD)?.to_vec());
            }
        }
        if offsets {
            starts.push(parse_int(read_value(data, pos, scratch, STARTOFFSET)?)?);
            ends.push(parse_int(read_value(data, pos, scratch, ENDOFFSET)?)?);
        }
    }
    if positions {
        t.positions = Some(ps);
        if payloads {
            t.payloads = Some(pays);
        }
    }
    if offsets {
        t.start_offsets = Some(starts);
        t.end_offsets = Some(ends);
    }
    Ok(t)
}

/// `SimpleTVTermsEnum` over one field's terms.
#[derive(Debug, Clone)]
pub struct SimpleTextTvTermsEnum<'f> {
    terms: &'f [TermVectorTerm],
    /// The next entry `next()` returns.
    upto: usize,
    current: Option<&'f TermVectorTerm>,
}

impl<'f> SimpleTextTvTermsEnum<'f> {
    /// `terms.iterator()`.
    pub fn new(field: &'f TermVectorField) -> Self {
        SimpleTextTvTermsEnum {
            terms: &field.terms,
            upto: 0,
            current: None,
        }
    }

    /// `next()`.
    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> Option<&'f [u8]> {
        let t = self.terms.get(self.upto)?;
        self.upto = self.upto.saturating_add(1);
        self.current = Some(t);
        Some(&t.term)
    }

    /// `seekCeil(text)`: on the first term `>= text`.
    pub fn seek_ceil(&mut self, text: &[u8]) -> SeekStatus {
        // Java: SimpleTVTermsEnum.seekCeil (`terms.tailMap(text)`)
        self.upto = self.terms.partition_point(|t| t.term.as_slice() < text);
        match self.next() {
            None => SeekStatus::End,
            Some(t) if t == text => SeekStatus::Found,
            Some(_) => SeekStatus::NotFound,
        }
    }

    /// `term()`, `None` before `next`.
    pub fn term(&self) -> Option<&'f [u8]> {
        self.current.map(|t| t.term.as_slice())
    }

    /// `docFreq()`.
    pub fn doc_freq(&self) -> i32 {
        1
    }

    /// `totalTermFreq()`.
    ///
    /// # Errors
    /// [`Error::IllegalState`] before `next` (Java's `NullPointerException`).
    pub fn total_term_freq(&self) -> Result<i64> {
        self.current
            .map(|t| i64::from(t.freq))
            .ok_or(Error::IllegalState("no current term"))
    }

    /// `postings(reuse, flags)`.
    ///
    /// # Errors
    /// [`Error::IllegalState`] before `next`.
    pub fn postings(&self, flags: i32) -> Result<SimpleTextTvPostings<'f>> {
        // Java: SimpleTVTermsEnum.postings
        let t = self.current.ok_or(Error::IllegalState("no current term"))?;
        if feature_requested(flags, POSITIONS)
            && (t.positions.is_some() || t.start_offsets.is_some())
        {
            return Ok(SimpleTextTvPostings::Positions {
                term: t,
                doc: -1,
                next_pos: 0,
            });
        }
        let freq = if feature_requested(flags, FREQS) {
            t.freq
        } else {
            1
        };
        Ok(SimpleTextTvPostings::Docs { freq, doc: -1 })
    }
}

/// `SimpleTVDocsEnum` (`Docs`) and `SimpleTVPostingsEnum` (`Positions`):
/// one document, 0.
#[derive(Debug, Clone)]
pub enum SimpleTextTvPostings<'f> {
    /// Freqs only.
    Docs {
        /// `freq` (1 when freqs were not requested).
        freq: i32,
        /// `docID()`.
        doc: i32,
    },
    /// Positions and/or offsets.
    Positions {
        /// The term's occurrences.
        term: &'f TermVectorTerm,
        /// `docID()`.
        doc: i32,
        /// `nextPos`.
        next_pos: usize,
    },
}

impl SimpleTextTvPostings<'_> {
    fn doc_mut(&mut self) -> &mut i32 {
        match self {
            SimpleTextTvPostings::Docs { doc, .. }
            | SimpleTextTvPostings::Positions { doc, .. } => doc,
        }
    }

    /// `docID()`.
    pub fn doc_id(&self) -> i32 {
        match self {
            SimpleTextTvPostings::Docs { doc, .. }
            | SimpleTextTvPostings::Positions { doc, .. } => *doc,
        }
    }

    /// `nextDoc()`: 0, then `NO_MORE_DOCS`.
    pub fn next_doc(&mut self) -> i32 {
        let doc = self.doc_mut();
        *doc = if *doc == -1 { 0 } else { NO_MORE_DOCS };
        *doc
    }

    /// `advance(target)` (`slowAdvance`).
    pub fn advance(&mut self, target: i32) -> i32 {
        let mut doc = self.doc_id();
        while doc < target {
            doc = self.next_doc();
        }
        doc
    }

    /// `freq()`.
    pub fn freq(&self) -> i32 {
        match self {
            SimpleTextTvPostings::Docs { freq, .. } => *freq,
            SimpleTextTvPostings::Positions { term, .. } => {
                let n = term
                    .positions
                    .as_ref()
                    .or(term.start_offsets.as_ref())
                    .map_or(0, Vec::len);
                i32::try_from(n).unwrap_or(i32::MAX)
            }
        }
    }

    /// The occurrence `nextPosition` last returned.
    fn at(&self) -> Option<(&TermVectorTerm, usize)> {
        match self {
            SimpleTextTvPostings::Positions { term, next_pos, .. } => {
                next_pos.checked_sub(1).map(|i| (*term, i))
            }
            SimpleTextTvPostings::Docs { .. } => None,
        }
    }

    /// `nextPosition()`: -1 without positions.
    ///
    // SENTINEL: -1 when the field has no positions (Java's).
    ///
    /// # Errors
    /// [`Error::IllegalState`] past `freq()` occurrences (Java's `assert`).
    pub fn next_position(&mut self) -> Result<i32> {
        let freq = usize::try_from(self.freq()).unwrap_or(0);
        match self {
            SimpleTextTvPostings::Docs { .. } => Ok(-1),
            SimpleTextTvPostings::Positions { term, next_pos, .. } => {
                if *next_pos >= freq {
                    return Err(Error::IllegalState(
                        "nextPosition() called more than freq() times!",
                    ));
                }
                let p = term
                    .positions
                    .as_ref()
                    .and_then(|p| p.get(*next_pos))
                    .copied();
                *next_pos = next_pos.saturating_add(1);
                Ok(p.unwrap_or(-1))
            }
        }
    }

    /// `startOffset()`: -1 without offsets.
    ///
    // SENTINEL: -1 when the field has no offsets (Java's).
    pub fn start_offset(&self) -> i32 {
        self.at()
            .and_then(|(t, i)| t.start_offsets.as_ref()?.get(i).copied())
            .unwrap_or(-1)
    }

    /// `endOffset()`: -1 without offsets.
    ///
    // SENTINEL: -1 when the field has no offsets (Java's).
    pub fn end_offset(&self) -> i32 {
        self.at()
            .and_then(|(t, i)| t.end_offsets.as_ref()?.get(i).copied())
            .unwrap_or(-1)
    }

    /// `getPayload()`: `None` without payloads (or for an empty one).
    pub fn payload(&self) -> Option<&[u8]> {
        self.at()
            .and_then(|(t, i)| t.payloads.as_ref()?.get(i))
            .map(Vec::as_slice)
            .filter(|p| !p.is_empty())
    }

    /// `cost()`.
    pub fn cost(&self) -> i64 {
        1
    }
}

#[cfg(test)]
mod tests;
