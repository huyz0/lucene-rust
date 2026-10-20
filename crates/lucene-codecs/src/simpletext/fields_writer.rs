//! `SimpleTextFieldsWriter`: a segment's postings as one `.pst` text file.

use super::skip::{SimpleTextSkipWriter, BLOCK_SIZE};
use super::util;
use super::{Error, Result};
use crate::postings_writer::{
    CompetitiveImpactAccumulator, FieldNorms, FieldPostingsInput, TermSource,
};
use crate::segment_state::SegmentWriteState;

/// `SimpleTextPostingsFormat.POSTINGS_EXTENSION`.
pub const POSTINGS_EXTENSION: &str = "pst";
/// The postings format's SPI name.
pub const FORMAT_NAME: &str = "SimpleText";

/// `SimpleTextFieldsWriter.END`.
pub const END: &[u8] = b"END";
/// `SimpleTextFieldsWriter.FIELD`.
pub const FIELD: &[u8] = b"field ";
/// `SimpleTextFieldsWriter.TERM`.
pub const TERM: &[u8] = b"  term ";
/// `SimpleTextFieldsWriter.DOC`.
pub const DOC: &[u8] = b"    doc ";
/// `SimpleTextFieldsWriter.FREQ`.
pub const FREQ: &[u8] = b"      freq ";
/// `SimpleTextFieldsWriter.POS`.
pub const POS: &[u8] = b"      pos ";
/// `SimpleTextFieldsWriter.START_OFFSET`.
pub const START_OFFSET: &[u8] = b"      startOffset ";
/// `SimpleTextFieldsWriter.END_OFFSET`.
pub const END_OFFSET: &[u8] = b"      endOffset ";
/// `SimpleTextFieldsWriter.PAYLOAD`.
pub const PAYLOAD: &[u8] = b"        payload ";

/// `IndexFileNames.segmentFileName(segment, suffix, "pst")`.
pub fn postings_file_name(segment: &str, suffix: &str) -> String {
    if suffix.is_empty() {
        format!("{segment}.{POSTINGS_EXTENSION}")
    } else {
        format!("{segment}_{suffix}.{POSTINGS_EXTENSION}")
    }
}

/// `SimpleTextPostingsFormat.fieldsConsumer(state).write(...)` then
/// `close()`: the whole `.pst`.
///
/// # Errors
/// What [`SimpleTextFieldsWriter::write`] refuses.
pub fn write<T: TermSource>(
    fields: &[FieldPostingsInput<'_, T>],
    norms: &[FieldNorms<'_>],
    state: &SegmentWriteState<'_>,
) -> Result<Vec<u8>> {
    let mut writer = SimpleTextFieldsWriter::new(state);
    writer.write(fields, norms)?;
    Ok(writer.close())
}

/// `SimpleTextFieldsWriter`.
#[derive(Debug)]
pub struct SimpleTextFieldsWriter<'s> {
    out: Vec<u8>,
    write_state: &'s SegmentWriteState<'s>,
    /// for write skip data.
    doc_count: i32,
    skip_writer: SimpleTextSkipWriter,
    competitive_impact_accumulator: CompetitiveImpactAccumulator,
    last_doc_file_pointer: i64,
}

/// `getNorm(doc, norms)`: 1 without norms or past the column.
fn get_norm(doc: i32, norms: Option<&[i64]>) -> i64 {
    let Some(values) = norms else { return 1 };
    usize::try_from(doc)
        .ok()
        .and_then(|d| values.get(d))
        .copied()
        .unwrap_or(1)
}

fn illegal(msg: String) -> Error {
    Error::IllegalArgument(msg)
}

impl<'s> SimpleTextFieldsWriter<'s> {
    /// `new SimpleTextFieldsWriter(writeState)`.
    pub fn new(write_state: &'s SegmentWriteState<'s>) -> Self {
        SimpleTextFieldsWriter {
            out: Vec::new(),
            write_state,
            doc_count: 0,
            skip_writer: SimpleTextSkipWriter::new(write_state.max_doc),
            competitive_impact_accumulator: CompetitiveImpactAccumulator::default(),
            last_doc_file_pointer: -1,
        }
    }

    fn file_pointer(&self) -> i64 {
        i64::try_from(self.out.len()).unwrap_or(i64::MAX)
    }

    fn line(&mut self, prefix: &[u8], value: &[u8]) {
        util::write(&mut self.out, prefix);
        util::write(&mut self.out, value);
        util::write_newline(&mut self.out);
    }

    /// `write(fieldInfos, fields, normsProducer)`: every field of `fields`,
    /// in the order given (a flush and a merge pass them in name order).
    ///
    /// # Errors
    /// [`Error::IllegalArgument`] for a field number the segment's
    /// `FieldInfos` lacks, or a term whose positions, offsets or payloads do
    /// not line up with its freqs.
    pub fn write<T: TermSource>(
        &mut self,
        fields: &[FieldPostingsInput<'_, T>],
        norms: &[FieldNorms<'_>],
    ) -> Result<()> {
        // for each field
        for field in fields {
            let field_info = self
                .write_state
                .field_info(field.field_number)
                .ok_or_else(|| illegal(format!("unknown field number {}", field.field_number)))?;
            let mut wrote_field = false;
            let options = field.index_options;
            let has_positions = options.subsumes_positions();
            let has_freqs = super::subsumes_freqs(options);
            let has_payloads = field_info.store_payloads;
            let has_offsets = options.subsumes_offsets();
            let field_norms = if field_info.omit_norms {
                None
            } else {
                norms
                    .iter()
                    .find(|n| n.field_number == field.field_number)
                    .map(|n| n.values)
            };

            // for each term in field
            for term in field.terms {
                self.doc_count = 0;
                self.skip_writer.reset_skip();
                self.competitive_impact_accumulator.clear();
                self.last_doc_file_pointer = -1;
                let docs = term.docs();
                if has_positions && term.position_lists() < docs.len() {
                    return Err(illegal(format!(
                        "{} positions lists for {} docs",
                        term.position_lists(),
                        docs.len()
                    )));
                }
                if has_offsets && term.offset_lists() < docs.len() {
                    return Err(illegal(format!(
                        "{} offsets lists for {} docs",
                        term.offset_lists(),
                        docs.len()
                    )));
                }
                let payload_lengths = term.payload_lengths();
                let payload_bytes = term.payload_bytes();
                let mut occurrence = 0usize;
                let mut payload_at = 0usize;
                let mut wrote_term = false;

                // for each doc in field+term
                for (d, &(doc, freq)) in docs.iter().enumerate() {
                    if !wrote_term {
                        if !wrote_field {
                            // we lazily do this, in case the field had no terms
                            self.line(FIELD, field_info.name.as_bytes());
                            wrote_field = true;
                        }
                        // we lazily do this, in case the term had zero docs
                        self.line(TERM, term.term());
                        wrote_term = true;
                    }
                    if self.last_doc_file_pointer == -1 {
                        self.last_doc_file_pointer = self.file_pointer();
                    }
                    self.line(DOC, doc.to_string().as_bytes());
                    if has_freqs {
                        self.line(FREQ, freq.to_string().as_bytes());
                        if has_positions {
                            let positions = term.positions(d);
                            let offsets = if has_offsets { term.offsets(d) } else { &[] };
                            let n = usize::try_from(freq).unwrap_or(0);
                            if positions.len() != n || (has_offsets && offsets.len() != n) {
                                return Err(illegal(format!(
                                    "doc {doc}: freq {freq}, {} positions, {} offsets",
                                    positions.len(),
                                    offsets.len()
                                )));
                            }
                            // for each pos in field+term+doc
                            for (i, &position) in positions.iter().enumerate() {
                                self.line(POS, position.to_string().as_bytes());
                                if has_offsets {
                                    let (start, end) = offsets[i];
                                    self.line(START_OFFSET, start.to_string().as_bytes());
                                    self.line(END_OFFSET, end.to_string().as_bytes());
                                }
                                if has_payloads {
                                    let len = payload_lengths
                                        .get(occurrence)
                                        .and_then(|&l| usize::try_from(l).ok())
                                        .ok_or_else(|| {
                                            illegal(format!("doc {doc}: missing payload length"))
                                        })?;
                                    let payload = payload_at
                                        .checked_add(len)
                                        .and_then(|end| payload_bytes.get(payload_at..end))
                                        .ok_or_else(|| {
                                            illegal(format!("doc {doc}: payload past its run"))
                                        })?;
                                    if !payload.is_empty() {
                                        self.line(PAYLOAD, payload);
                                    }
                                    // ARITH: `payload_at + len` was checked above.
                                    #[allow(clippy::arithmetic_side_effects)]
                                    {
                                        payload_at += len;
                                    }
                                }
                                // ARITH: one per occurrence of an in-memory list.
                                #[allow(clippy::arithmetic_side_effects)]
                                {
                                    occurrence += 1;
                                }
                            }
                        }
                        self.competitive_impact_accumulator
                            .add(freq, get_norm(doc, field_norms));
                    } else {
                        self.competitive_impact_accumulator
                            .add(1, get_norm(doc, field_norms));
                    }
                    // ARITH: a term's documents are an in-memory list of at
                    // most `maxDoc` (an `int`) entries.
                    #[allow(clippy::arithmetic_side_effects)]
                    {
                        self.doc_count += 1;
                    }
                    if self.doc_count % BLOCK_SIZE == 0 {
                        self.skip_writer.buffer_skip(
                            doc,
                            self.last_doc_file_pointer,
                            self.doc_count,
                            &self.competitive_impact_accumulator,
                        );
                        self.competitive_impact_accumulator.clear();
                        self.last_doc_file_pointer = -1;
                    }
                }
                if self.doc_count >= BLOCK_SIZE {
                    self.skip_writer.write_skip(&mut self.out);
                }
            }
        }
        Ok(())
    }

    /// `close()`: `END`, then the checksum line; the file's bytes.
    pub fn close(mut self) -> Vec<u8> {
        util::write(&mut self.out, END);
        util::write_newline(&mut self.out);
        util::write_checksum(&mut self.out);
        self.out
    }
}
