//! Port of `STUniformSplitTermsWriter`'s shared block output and
//! `STBlockWriter`; both flush and specialized merge feed this writer.
//!
//! Java pops the fields' terms off one queue by (term, field number); each
//! term's postings are written field by field, then the term goes into the
//! blocks as one line with every field's state. Here the postings writer
//! drives: [`StUniformSplitTermsWriter::add_term_states`] takes its term
//! states in that same order and gathers each term's into a line.

use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::sync::Arc;

use lucene_store::codec_util;
use lucene_util::fixed_bit_set::FixedBitSet;

use super::super::block_line::TermBytes;
use super::super::dictionary::FstDictionaryBuilder;
use super::super::field_metadata::FieldMetadata;
use super::super::writer::{write_fields_metadata, BlockSink, BlockWriter, PendingLine};
use super::super::{validate_settings, BlockEncoder, DeltaBaseTermStateSerializer, Error, Result};
use super::block_line::{write_line_term_states, FieldMetadataTermState};
use super::{NAME, VERSION_CURRENT};
use crate::postings_writer::{self, FieldTermStates};
use crate::segment_state::SegmentWriteState;

/// `STBlockWriter`'s state: the group's fields (`fieldMetadataList`) and
/// which of them the block being written holds (`fieldsInBlock`).
struct StBlockSink {
    fields: Vec<FieldMetadata>,
    fields_in_block: Vec<bool>,
}

impl BlockSink for StBlockSink {
    type States = Vec<FieldMetadataTermState>;

    /// `writeBlockLine`'s `writeLineTermStates` and `collectFields`.
    fn write_line_term_states(
        &mut self,
        out: &mut Vec<u8>,
        serializer: &mut DeltaBaseTermStateSerializer,
        states: &Vec<FieldMetadataTermState>,
    ) {
        write_line_term_states(out, serializer, states, &self.fields);
        for state in states {
            self.fields_in_block[state.field] = true;
        }
    }

    /// `updateFieldMetadata(blockStartFP)`: every field of the block.
    fn update_field_metadata(&mut self, block_start_fp: i64) {
        for (field, in_block) in self.fields.iter_mut().zip(&mut self.fields_in_block) {
            if std::mem::take(in_block) {
                if field.first_block_start_fp == -1 {
                    field.first_block_start_fp = block_start_fp;
                }
                field.last_block_start_fp = block_start_fp;
            }
        }
    }
}

/// `STUniformSplitTermsWriter`: one group's `.stustb` and `.stustd`.
pub struct StUniformSplitTermsWriter {
    block_writer: BlockWriter<Vec<FieldMetadataTermState>>,
    sink: StBlockSink,
    block_output: Vec<u8>,
    dictionary_output: Vec<u8>,
    dictionary: FstDictionaryBuilder,
    /// The line being gathered: its term and states (`termStates`).
    line_term: Vec<u8>,
    line_states: Vec<FieldMetadataTermState>,
}

impl std::fmt::Debug for StUniformSplitTermsWriter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StUniformSplitTermsWriter")
            .field("fields", &self.sink.fields.len())
            .field("block_output", &self.block_output.len())
            .finish()
    }
}

impl StUniformSplitTermsWriter {
    /// The constructor (both headers, then the postings writer's in the
    /// `.stustb`) and `createFieldMetadataList`: `fields` are `(number,
    /// docCount)` in the order the fields' metadata is written.
    ///
    /// # Errors
    /// [`Error::IllegalArgument`] for settings `validateSettings` refuses or
    /// a field number `state` lacks.
    pub fn new(
        state: &SegmentWriteState<'_>,
        target_num_block_lines: i32,
        delta_num_lines: i32,
        block_encoder: Option<Arc<dyn BlockEncoder>>,
        fields: &[(i32, i32)],
    ) -> Result<Self> {
        validate_settings(target_num_block_lines, delta_num_lines)?;
        let mut block_output = Vec::new();
        codec_util::write_index_header(
            &mut block_output,
            NAME,
            VERSION_CURRENT,
            state.segment_id,
            state.segment_suffix,
        );
        let mut dictionary_output = Vec::new();
        codec_util::write_index_header(
            &mut dictionary_output,
            NAME,
            VERSION_CURRENT,
            state.segment_id,
            state.segment_suffix,
        );
        postings_writer::write_postings_terms_header(
            &mut block_output,
            state.segment_id,
            state.segment_suffix,
        );
        let mut metadata = Vec::with_capacity(fields.len());
        for &(number, doc_count) in fields {
            let info = state
                .field_info(number)
                .ok_or_else(|| Error::IllegalArgument(format!("unknown field number {number}")))?;
            let mut field = FieldMetadata::new(info.clone(), doc_count);
            field.dictionary_start_fp = dictionary_output.len() as i64;
            metadata.push(field);
        }
        Ok(StUniformSplitTermsWriter {
            block_writer: BlockWriter::new(target_num_block_lines, delta_num_lines, block_encoder),
            sink: StBlockSink {
                fields_in_block: vec![false; metadata.len()],
                fields: metadata,
            },
            block_output,
            dictionary_output,
            dictionary: FstDictionaryBuilder::new(),
            line_term: Vec::new(),
            line_states: Vec::new(),
        })
    }

    /// `writePostingLines`' bookkeeping for the postings writer's term
    /// states of one field: each term joins the line being gathered, or
    /// (a later term) ends it -- `blockWriter.addLine` -- and starts the
    /// next. The calls come by term, then field number, as
    /// [`super::StUniformSplitPostingsFormat::write`] orders them.
    ///
    /// # Errors
    /// [`Error::IllegalArgument`] for a field not given to [`Self::new`] or
    /// a term out of that order; what the encoder or the dictionary fail
    /// with.
    pub fn add_term_states(&mut self, field: &FieldTermStates<'_>) -> Result<()> {
        let index = self
            .sink
            .fields
            .iter()
            .position(|f| f.field_info.number == field.field_number)
            .ok_or_else(|| {
                Error::IllegalArgument(format!("unknown field number {}", field.field_number))
            })?;
        for i in 0..field.len() {
            let term = field.term(i);
            let same_line = match self.line_states.last() {
                None => false,
                Some(last) => match term.cmp(self.line_term.as_slice()) {
                    Ordering::Greater => false,
                    Ordering::Equal
                        if self.sink.fields[last.field].field_info.number < field.field_number =>
                    {
                        true
                    }
                    _ => {
                        return Err(Error::IllegalArgument(format!(
                            "term states out of order for field number {}",
                            field.field_number
                        )))
                    }
                },
            };
            if !same_line {
                self.add_line()?;
                self.line_term.clear();
                self.line_term.extend_from_slice(term);
            }
            let (doc_freq, total_term_freq) = (field.doc_freq(i), field.total_term_freq(i));
            let metadata = &mut self.sink.fields[index];
            metadata.update_stats(doc_freq, total_term_freq);
            // `fieldMetadata.setLastTerm(term)`.
            metadata.last_term.clear();
            metadata.last_term.extend_from_slice(term);
            self.line_states.push(FieldMetadataTermState {
                field: index,
                doc_freq,
                total_term_freq,
                meta: field.term_metadata(i),
            });
        }
        Ok(())
    }

    /// `FieldMetadata.getDocCount`: count each merge field's seen documents
    /// once at metadata serialization, rather than scanning its bitset for
    /// every term. A flush already supplied its final counts to `new`.
    pub(crate) fn finish_merge(
        mut self,
        docs_seen: &BTreeMap<i32, FixedBitSet>,
    ) -> Result<(Vec<u8>, Vec<u8>)> {
        for field in &mut self.sink.fields {
            field.doc_count = docs_seen
                .get(&field.field_info.number)
                .map_or(0, |seen| seen.cardinality() as i32);
        }
        self.finish()
    }

    /// `STBlockWriter.addLine(term, termStates, dictionaryBuilder)` for the
    /// gathered line; nothing when there is none (Java: empty `termStates`).
    fn add_line(&mut self) -> Result<()> {
        if self.line_states.is_empty() {
            return Ok(());
        }
        let mdp_length =
            TermBytes::compute_mdp_length(self.block_writer.last_term.as_deref(), &self.line_term);
        let line = PendingLine {
            term: self.line_term.clone(),
            mdp_length,
            states: std::mem::take(&mut self.line_states),
        };
        self.block_writer.add_line(
            line,
            &mut self.block_output,
            &mut self.sink,
            &mut self.dictionary,
        )
    }

    /// `writeSegment`'s end: the last line and block, `writeFieldMetadataList`
    /// (the fields with terms) and `writeDictionary(fieldsNumber,
    /// dictionaryBuilder)`; the finished `.stustb` and `.stustd`.
    ///
    /// # Errors
    /// What the encoder or the dictionary fail with.
    pub fn finish(mut self) -> Result<(Vec<u8>, Vec<u8>)> {
        self.add_line()?;
        self.block_writer.finish_last_block(
            &mut self.block_output,
            &mut self.sink,
            &mut self.dictionary,
        )?;
        let mut fields_output = Vec::new();
        let mut fields_number = 0i32;
        for field in &self.sink.fields {
            if field.num_terms > 0 {
                field.write(&mut fields_output);
                // ARITH: at most the field count, an `i32`.
                #[allow(clippy::arithmetic_side_effects)]
                {
                    fields_number += 1;
                }
            }
        }
        let encoder = self.block_writer.block_encoder.as_deref();
        write_fields_metadata(
            &mut self.block_output,
            fields_number,
            &fields_output,
            encoder,
        )?;
        if fields_number > 0 {
            self.dictionary
                .build()?
                .write(&mut self.dictionary_output, encoder)?;
        }
        codec_util::write_footer(&mut self.dictionary_output);
        Ok((self.block_output, self.dictionary_output))
    }
}
