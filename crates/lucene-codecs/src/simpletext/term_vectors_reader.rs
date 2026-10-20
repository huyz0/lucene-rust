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
//!   [`SimpleTextTvTermsEnum`] and [`SimpleTextTvPostings`] are
//!   `SimpleTVTermsEnum` and its two postings enums in one.
//! - Java's `assert`s on each line's prefix and on the document count, its
//!   `NumberFormatException`s, its negative-array and index errors (a
//!   negative `freq` where the field has positions or offsets, more
//!   documents than `maxDoc`, a document outside it) are `Corrupted` (`IllegalArgument` for the document number). Arrays
//!   grow as lines are read rather than being sized from `freq`.

use std::cmp::Ordering;

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
    offsets: Vec<usize>,
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
        Ok(SimpleTextTermVectorsReader { data, offsets })
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
mod tests {
    use super::super::term_vectors_writer::{
        term_vectors_file_name, write, SimpleTextTermVectorsWriter,
    };
    use super::*;
    use crate::field_infos::{FieldInfo, FieldInfos, IndexOptions};
    use crate::term_vectors::TermVectorsDocument;

    fn infos() -> FieldInfos {
        let f = |name: &str, n: i32| {
            FieldInfo::new(name, n)
                .with_index_options(IndexOptions::DocsAndFreqsAndPositionsAndOffsets)
                .with_store_term_vectors(true)
        };
        FieldInfos::new(vec![f("b", 0), f("a", 1)]).unwrap()
    }

    fn term(t: &[u8], pos: bool, off: bool, pay: bool) -> TermVectorTerm {
        TermVectorTerm {
            term: t.to_vec(),
            freq: 2,
            positions: pos.then(|| vec![1, 4]),
            start_offsets: off.then(|| vec![0, 9]),
            end_offsets: off.then(|| vec![3, 12]),
            payloads: pay.then(|| vec![vec![], b"\n\\".to_vec()]),
        }
    }

    fn field(
        n: i32,
        pos: bool,
        off: bool,
        pay: bool,
        terms: Vec<TermVectorTerm>,
    ) -> TermVectorField {
        TermVectorField {
            field_number: n,
            has_positions: pos,
            has_offsets: off,
            has_payloads: pay,
            terms,
        }
    }

    fn docs() -> Vec<TermVectorsDocument> {
        vec![
            TermVectorsDocument {
                fields: vec![
                    field(0, true, false, true, vec![term(b"x", true, false, true)]),
                    field(1, false, true, false, vec![term(b"m", false, true, false)]),
                ],
            },
            TermVectorsDocument::default(),
            TermVectorsDocument {
                fields: vec![field(
                    1,
                    false,
                    false,
                    false,
                    vec![
                        term(b"q", false, false, false),
                        term(b"z", false, false, false),
                    ],
                )],
            },
        ]
    }

    #[test]
    fn documents_read_back_with_fields_by_name() {
        let data = write(&docs(), &infos()).unwrap();
        let r = SimpleTextTermVectorsReader::open(&data, 3).unwrap();
        assert_eq!(r.max_doc(), 3);
        let d0 = r.get(0).unwrap().unwrap();
        assert_eq!(
            d0.iter().map(|f| f.0.as_str()).collect::<Vec<_>>(),
            ["a", "b"]
        );
        assert_eq!(d0[1].1, docs()[0].fields[0]);
        assert_eq!(d0[0].1, docs()[0].fields[1]);
        assert!(r.get(1).unwrap().is_none());
        assert!(r.get(3).is_err() && r.get(-1).is_err());
        assert_eq!(term_vectors_file_name("_0"), "_0.tvc");
    }

    #[test]
    fn terms_enum_and_postings() {
        let data = write(&docs(), &infos()).unwrap();
        let r = SimpleTextTermVectorsReader::open(&data, 3).unwrap();
        let d0 = r.get(0).unwrap().unwrap();
        // Positions and payloads, no offsets.
        let mut te = SimpleTextTvTermsEnum::new(&d0[1].1);
        assert!(
            te.term().is_none() && te.total_term_freq().is_err() && te.postings(POSITIONS).is_err()
        );
        assert_eq!(te.next(), Some(&b"x"[..]));
        assert_eq!((te.doc_freq(), te.total_term_freq().unwrap()), (1, 2));
        let mut p = te.postings(POSITIONS).unwrap();
        assert_eq!((p.doc_id(), p.cost()), (-1, 1));
        assert_eq!(p.payload(), None, "before the first position");
        assert_eq!(p.next_doc(), 0);
        assert_eq!(p.freq(), 2);
        assert_eq!(p.next_position().unwrap(), 1);
        assert_eq!(
            (p.start_offset(), p.end_offset(), p.payload()),
            (-1, -1, None)
        );
        assert_eq!(p.next_position().unwrap(), 4);
        assert_eq!(p.payload(), Some(&b"\n\\"[..]));
        assert!(p.next_position().is_err());
        assert_eq!(p.next_doc(), NO_MORE_DOCS);
        assert_eq!(te.next(), None);
        // Offsets only: positions read -1.
        let mut te = SimpleTextTvTermsEnum::new(&d0[0].1);
        assert_eq!(te.seek_ceil(b"m"), SeekStatus::Found);
        let mut p = te.postings(POSITIONS).unwrap();
        assert_eq!(p.advance(0), 0);
        assert_eq!(p.next_position().unwrap(), -1);
        assert_eq!((p.start_offset(), p.end_offset()), (0, 3));
        assert_eq!(te.seek_ceil(b"a"), SeekStatus::NotFound);
        assert_eq!(te.seek_ceil(b"n"), SeekStatus::End);
        // Freqs only; and freq 1 without FREQS.
        let d2 = r.get(2).unwrap().unwrap();
        let mut te = SimpleTextTvTermsEnum::new(&d2[0].1);
        assert_eq!(te.seek_ceil(b"r"), SeekStatus::NotFound);
        assert_eq!(te.term(), Some(&b"z"[..]));
        let mut p = te.postings(POSITIONS).unwrap();
        assert_eq!(p.freq(), 2);
        assert_eq!(p.next_position().unwrap(), -1);
        assert_eq!(
            (p.start_offset(), p.end_offset(), p.payload()),
            (-1, -1, None)
        );
        assert_eq!(p.advance(5), NO_MORE_DOCS);
        assert_eq!(te.postings(0).unwrap().freq(), 1);
        assert_eq!(te.postings(FREQS).unwrap().freq(), 2);
    }

    #[test]
    fn bad_files_and_counts_are_refused() {
        let data = write(&docs(), &infos()).unwrap();
        assert!(
            SimpleTextTermVectorsReader::open(&data, 2).is_err(),
            "more than maxDoc"
        );
        assert!(
            SimpleTextTermVectorsReader::open(&data, 4).is_err(),
            "fewer"
        );
        assert!(SimpleTextTermVectorsReader::open(&data, -1).is_err());
        assert!(
            SimpleTextTermVectorsReader::open(b"doc 0\n", 1).is_err(),
            "no END"
        );
        let mut damaged = data.clone();
        let at = damaged.iter().position(|&b| b == b'x').unwrap();
        damaged[at] = b'y';
        assert!(
            SimpleTextTermVectorsReader::open(&damaged, 3).is_err(),
            "checksum"
        );
        // A negative freq with positions.
        let mut w = SimpleTextTermVectorsWriter::new();
        let info = infos();
        w.start_document(1);
        w.start_field(info.field_by_number(0).unwrap(), 1, true, false, false);
        w.start_term(b"t", -1);
        let data = w.finish(1).unwrap();
        let r = SimpleTextTermVectorsReader::open(&data, 1).unwrap();
        assert!(r.get(0).is_err());
        // A malformed numfields line.
        let text = String::from_utf8(data)
            .unwrap()
            .replacen("numfields 1", "numfields x", 1);
        let mut resealed = text.as_bytes()[..text.rfind("checksum ").unwrap()].to_vec();
        util::write_checksum(&mut resealed);
        let r = SimpleTextTermVectorsReader::open(&resealed, 1).unwrap();
        assert!(r.get(0).is_err());
    }

    #[test]
    fn repeated_terms_and_fields_keep_the_last_value() {
        let info = infos();
        let mut w = SimpleTextTermVectorsWriter::new();
        w.start_document(2);
        w.start_field(info.field_by_number(0).unwrap(), 3, false, false, false);
        for (t, freq) in [(&b"b"[..], 1), (b"a", 2), (b"b", 3)] {
            w.start_term(t, freq);
        }
        w.start_field(info.field_by_number(0).unwrap(), 1, false, false, false);
        w.start_term(b"z", 4);
        let data = w.finish(1).unwrap();
        let r = SimpleTextTermVectorsReader::open(&data, 1).unwrap();
        let doc = r.get(0).unwrap().unwrap();
        assert_eq!(doc.len(), 1, "one field named b");
        let terms: Vec<_> = doc[0]
            .1
            .terms
            .iter()
            .map(|t| (t.term.clone(), t.freq))
            .collect();
        assert_eq!(terms, [(b"z".to_vec(), 4)], "the second `b` field wins");
        let mut w = SimpleTextTermVectorsWriter::new();
        w.start_document(1);
        w.start_field(info.field_by_number(0).unwrap(), 3, false, false, false);
        for (t, freq) in [(&b"b"[..], 1), (b"a", 2), (b"b", 3)] {
            w.start_term(t, freq);
        }
        let data = w.finish(1).unwrap();
        let r = SimpleTextTermVectorsReader::open(&data, 1).unwrap();
        let doc = r.get(0).unwrap().unwrap();
        let terms: Vec<_> = doc[0]
            .1
            .terms
            .iter()
            .map(|t| (t.term.clone(), t.freq))
            .collect();
        assert_eq!(terms, [(b"a".to_vec(), 2), (b"b".to_vec(), 3)]);
        let mut v = vec![(2, 'x'), (1, 'y'), (2, 'z'), (1, 'w')];
        sort_keep_last(&mut v, |a, b| a.0.cmp(&b.0));
        assert_eq!(v, [(1, 'w'), (2, 'z')]);
    }

    #[test]
    fn the_writer_refuses_bad_input() {
        let mut w = SimpleTextTermVectorsWriter::new();
        w.start_document(0);
        assert!(matches!(w.finish(2), Err(Error::Runtime(_))));
        let mut bad = docs();
        bad[0].fields[0].field_number = 9;
        assert!(write(&bad, &infos()).is_err(), "unknown field");
        let mut bad = docs();
        bad[0].fields[0].terms[0].positions = Some(vec![1]);
        assert!(write(&bad, &infos()).is_err(), "short occurrences");
    }
}
