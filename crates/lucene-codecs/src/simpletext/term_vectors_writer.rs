//! `SimpleTextTermVectorsWriter` (and `SimpleTextTermVectorsFormat`'s file
//! name): a segment's term vectors as one `.tvc` text file.
//!
//! ```text
//! doc <n>                          per document, 0..numDocs
//!   numfields <fields with vectors>
//!   field <field number>           per field, by name
//!     name <field name>
//!     positions <true|false>
//!     offsets   <true|false>
//!     payloads  <true|false>
//!     numterms <n>
//!     term <bytes, escaped>        per term, in byte order
//!       freq <n>
//!       position <p>               per occurrence, with positions
//!         payload <bytes>          with payloads (empty for none)
//!         startoffset <o>          with offsets
//!         endoffset <o>
//! END
//! checksum <20 digits>
//! ```
//!
//! What differs from Java: the writer builds the file in memory and
//! [`SimpleTextTermVectorsWriter::finish`] returns it; [`write()`] stands for
//! `TermVectorsWriter.addAllDocVectors` over this crate's
//! [`TermVectorsDocument`] model, whose empty payload is Java's `null`.

use super::util::{self, boolean_str, write_line};
use super::{Error, Result};
use crate::field_infos::{FieldInfo, FieldInfos};
use crate::term_vectors::TermVectorsDocument;

/// `VECTORS_EXTENSION`.
pub const VECTORS_EXTENSION: &str = "tvc";
/// `END`.
pub const END: &[u8] = b"END";
/// `DOC`.
pub const DOC: &[u8] = b"doc ";
/// `NUMFIELDS`.
pub const NUMFIELDS: &[u8] = b"  numfields ";
/// `FIELD`.
pub const FIELD: &[u8] = b"  field ";
/// `FIELDNAME`.
pub const FIELDNAME: &[u8] = b"    name ";
/// `FIELDPOSITIONS`.
pub const FIELDPOSITIONS: &[u8] = b"    positions ";
/// `FIELDOFFSETS`.
pub const FIELDOFFSETS: &[u8] = b"    offsets   ";
/// `FIELDPAYLOADS`.
pub const FIELDPAYLOADS: &[u8] = b"    payloads  ";
/// `FIELDTERMCOUNT`.
pub const FIELDTERMCOUNT: &[u8] = b"    numterms ";
/// `TERMTEXT`.
pub const TERMTEXT: &[u8] = b"    term ";
/// `TERMFREQ`.
pub const TERMFREQ: &[u8] = b"      freq ";
/// `POSITION`.
pub const POSITION: &[u8] = b"      position ";
/// `PAYLOAD`.
pub const PAYLOAD: &[u8] = b"        payload ";
/// `STARTOFFSET`.
pub const STARTOFFSET: &[u8] = b"        startoffset ";
/// `ENDOFFSET`.
pub const ENDOFFSET: &[u8] = b"        endoffset ";

/// `IndexFileNames.segmentFileName(segment, "", "tvc")`.
pub fn term_vectors_file_name(segment: &str) -> String {
    super::segment_file_name(segment, "", VECTORS_EXTENSION)
}

/// `SimpleTextTermVectorsWriter`.
#[derive(Debug, Default)]
pub struct SimpleTextTermVectorsWriter {
    out: Vec<u8>,
    num_docs_written: i32,
    offsets: bool,
    positions: bool,
    payloads: bool,
}

impl SimpleTextTermVectorsWriter {
    /// `new SimpleTextTermVectorsWriter(directory, segment, context)`.
    pub fn new() -> Self {
        Self::default()
    }

    fn line(&mut self, prefix: &[u8], value: &[u8]) {
        write_line(&mut self.out, prefix, value);
    }

    /// `startDocument(numVectorFields)`.
    pub fn start_document(&mut self, num_vector_fields: i32) {
        // Java: SimpleTextTermVectorsWriter.startDocument
        let n = self.num_docs_written.to_string();
        self.line(DOC, n.as_bytes());
        self.line(NUMFIELDS, num_vector_fields.to_string().as_bytes());
        self.num_docs_written = self.num_docs_written.saturating_add(1);
    }

    /// `startField(info, numTerms, positions, offsets, payloads)`.
    pub fn start_field(
        &mut self,
        info: &FieldInfo,
        num_terms: i32,
        positions: bool,
        offsets: bool,
        payloads: bool,
    ) {
        // Java: SimpleTextTermVectorsWriter.startField
        self.line(FIELD, info.number.to_string().as_bytes());
        self.line(FIELDNAME, info.name.as_bytes());
        self.line(FIELDPOSITIONS, boolean_str(positions));
        self.line(FIELDOFFSETS, boolean_str(offsets));
        self.line(FIELDPAYLOADS, boolean_str(payloads));
        self.line(FIELDTERMCOUNT, num_terms.to_string().as_bytes());
        self.positions = positions;
        self.offsets = offsets;
        self.payloads = payloads;
    }

    /// `startTerm(term, freq)`.
    pub fn start_term(&mut self, term: &[u8], freq: i32) {
        // Java: SimpleTextTermVectorsWriter.startTerm
        self.line(TERMTEXT, term);
        self.line(TERMFREQ, freq.to_string().as_bytes());
    }

    /// `addPosition(position, startOffset, endOffset, payload)`; `None` (or
    /// an empty payload) writes an empty `payload` line.
    pub fn add_position(
        &mut self,
        position: i32,
        start_offset: i32,
        end_offset: i32,
        payload: Option<&[u8]>,
    ) {
        // Java: SimpleTextTermVectorsWriter.addPosition
        if self.positions {
            self.line(POSITION, position.to_string().as_bytes());
            if self.payloads {
                self.line(PAYLOAD, payload.unwrap_or_default());
            }
        }
        if self.offsets {
            self.line(STARTOFFSET, start_offset.to_string().as_bytes());
            self.line(ENDOFFSET, end_offset.to_string().as_bytes());
        }
    }

    /// `finish(numDocs)`, then `close()`: the `.tvc`.
    ///
    /// # Errors
    /// [`Error::Runtime`] when `num_docs` documents were not started (Java's
    /// `RuntimeException`).
    pub fn finish(mut self, num_docs: i32) -> Result<Vec<u8>> {
        // Java: SimpleTextTermVectorsWriter.finish
        if self.num_docs_written != num_docs {
            return Err(Error::Runtime(format!(
                "mergeVectors produced an invalid result: mergedDocs is {num_docs} but vec numDocs is {}; now aborting this merge to prevent index corruption",
                self.num_docs_written
            )));
        }
        util::write(&mut self.out, END);
        util::write_newline(&mut self.out);
        util::write_checksum(&mut self.out);
        Ok(self.out)
    }
}

/// `TermVectorsWriter.addAllDocVectors` for every document, then `finish`:
/// each document's fields in the order given (Java's are by name), each
/// field's terms in the order given (Java's are in byte order).
///
/// # Errors
/// [`Error::IllegalArgument`] for a field number `field_infos` lacks, or an
/// occurrence list shorter than its term's `freq`.
pub fn write(docs: &[TermVectorsDocument], field_infos: &FieldInfos) -> Result<Vec<u8>> {
    // Java: TermVectorsWriter.addAllDocVectors
    let mut w = SimpleTextTermVectorsWriter::new();
    for doc in docs {
        w.start_document(i32::try_from(doc.fields.len()).unwrap_or(i32::MAX));
        for field in &doc.fields {
            let info = field_infos
                .field_by_number(field.field_number)
                .ok_or_else(|| {
                    Error::IllegalArgument(format!("no field number {}", field.field_number))
                })?;
            let num_terms = i32::try_from(field.terms.len()).unwrap_or(i32::MAX);
            w.start_field(
                info,
                num_terms,
                field.has_positions,
                field.has_offsets,
                field.has_payloads,
            );
            for term in &field.terms {
                w.start_term(&term.term, term.freq);
                if !(field.has_positions || field.has_offsets) {
                    continue;
                }
                let short = || {
                    Error::IllegalArgument(format!(
                        "field {}: fewer occurrences than freq {}",
                        info.name, term.freq
                    ))
                };
                for k in 0..usize::try_from(term.freq).unwrap_or(0) {
                    let at = |v: &Option<Vec<i32>>, absent: i32| -> Result<i32> {
                        match v {
                            Some(v) => v.get(k).copied().ok_or_else(short),
                            None => Ok(absent),
                        }
                    };
                    let position = at(&term.positions, -1)?;
                    let start = at(&term.start_offsets, -1)?;
                    let end = at(&term.end_offsets, -1)?;
                    let payload = match &term.payloads {
                        Some(p) => p.get(k).map(Vec::as_slice),
                        None => None,
                    };
                    w.add_position(position, start, end, payload);
                }
            }
        }
    }
    w.finish(i32::try_from(docs.len()).unwrap_or(i32::MAX))
}
