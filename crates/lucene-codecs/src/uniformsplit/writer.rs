//! Port of `UniformSplitTermsWriter` and `BlockWriter`.
//!
//! A field's terms are added one line at a time; once `target + delta`
//! lines wait, the block is cut at the line with the smallest `mdpLength`
//! (the last such) among the final `2 * delta` -- the line whose term is
//! most different from the one before, so the next block's key is short --
//! and the lines before it are written. Fewer than `target - delta` waiting
//! lines are written whole. Each block's key, its first term's minimal
//! distinguishing prefix, goes into the field's dictionary FST.

use std::sync::Arc;

use lucene_store::codec_util;
use lucene_store::data_output::DataOutput;

use super::block_line::{write_line, TermBytes};
use super::dictionary::FstDictionaryBuilder;
use super::field_metadata::FieldMetadata;
use super::{
    validate_settings, BlockEncoder, BlockHeader, DeltaBaseTermStateSerializer, Error, Result,
    NAME, VERSION_CURRENT,
};
use crate::field_infos::FieldInfo;
use crate::postings::TermMetadata;
use crate::postings_writer::{self, FieldTermStates};
use crate::segment_state::SegmentWriteState;

/// `BlockLine` on the write side: a term, its `mdpLength` against the term
/// before it, and its `BlockTermState`.
#[derive(Debug, Clone)]
struct PendingLine {
    term: Vec<u8>,
    mdp_length: usize,
    doc_freq: i32,
    total_term_freq: i64,
    meta: TermMetadata,
}

/// `BlockWriter`.
struct BlockWriter {
    target_num_block_lines: usize,
    delta_num_lines: usize,
    block_lines: Vec<PendingLine>,
    block_lines_write_buffer: Vec<u8>,
    term_states_write_buffer: Vec<u8>,
    block_write_buffer: Vec<u8>,
    term_state_serializer: DeltaBaseTermStateSerializer,
    block_encoder: Option<Arc<dyn BlockEncoder>>,
    /// `lastTerm`.
    last_term: Option<Vec<u8>>,
}

impl BlockWriter {
    fn new(
        target_num_block_lines: usize,
        delta_num_lines: usize,
        block_encoder: Option<Arc<dyn BlockEncoder>>,
    ) -> Self {
        BlockWriter {
            target_num_block_lines,
            delta_num_lines,
            block_lines: Vec::with_capacity(target_num_block_lines),
            block_lines_write_buffer: Vec::new(),
            term_states_write_buffer: Vec::new(),
            block_write_buffer: Vec::new(),
            term_state_serializer: DeltaBaseTermStateSerializer::default(),
            block_encoder,
            last_term: None,
        }
    }

    /// `addLine(term, blockTermState, dictionaryBuilder)`.
    // ARITH: `target + delta <= MAX_NUM_BLOCK_LINES` (`validateSettings`).
    #[allow(clippy::arithmetic_side_effects)]
    fn add_line(
        &mut self,
        line: PendingLine,
        block_output: &mut Vec<u8>,
        field: &mut FieldMetadata,
        dictionary: &mut FstDictionaryBuilder,
    ) -> Result<()> {
        self.last_term = Some(line.term.clone());
        self.block_lines.push(line);
        if self.block_lines.len() >= self.target_num_block_lines + self.delta_num_lines {
            self.split_and_write_block(block_output, field, dictionary)?;
        }
        Ok(())
    }

    /// `finishLastBlock(dictionaryBuilder)`.
    fn finish_last_block(
        &mut self,
        block_output: &mut Vec<u8>,
        field: &mut FieldMetadata,
        dictionary: &mut FstDictionaryBuilder,
    ) -> Result<()> {
        while !self.block_lines.is_empty() {
            self.split_and_write_block(block_output, field, dictionary)?;
        }
        self.last_term = None;
        Ok(())
    }

    /// `splitAndWriteBlock(dictionaryBuilder)`.
    ///
    /// # Errors
    /// [`Error::IllegalArgument`] where Java would write an empty block and
    /// fail on its key (`IndexOutOfBoundsException`; with assertions on,
    /// `deltaStart >= 1` fails first): only when the field's last lines
    /// number fewer than `2 * delta` and the first of them has the smallest
    /// `mdpLength`, which settings with `target >= 3 * delta + 1` never
    /// meet.
    fn split_and_write_block(
        &mut self,
        block_output: &mut Vec<u8>,
        field: &mut FieldMetadata,
        dictionary: &mut FstDictionaryBuilder,
    ) -> Result<()> {
        let num_lines = self.block_lines.len();
        // `delta < target` (`validateSettings`).
        if num_lines
            <= self
                .target_num_block_lines
                .saturating_sub(self.delta_num_lines)
        {
            let lines = std::mem::take(&mut self.block_lines);
            self.write_block(&lines, block_output, field, dictionary)?;
            self.block_lines = lines;
            self.block_lines.clear();
            return Ok(());
        }
        // Java's int `numLines - deltaNumLines * 2`: a negative start is
        // Java's `IndexOutOfBoundsException` on `blockLines.get`, refused
        // here as the split below is.
        let Some(delta_start) = num_lines.checked_sub(self.delta_num_lines.saturating_mul(2))
        else {
            return Err(Error::IllegalArgument(format!(
                "blockLines size: {num_lines} cannot be split with deltaNumLines={}",
                self.delta_num_lines
            )));
        };
        let mut min_mdp_length = usize::MAX;
        let mut min_mdp_end_index = 0;
        for (i, line) in self.block_lines.iter().enumerate().skip(delta_start) {
            if line.mdp_length <= min_mdp_length {
                min_mdp_length = line.mdp_length;
                min_mdp_end_index = i;
            }
        }
        if min_mdp_end_index == 0 {
            return Err(Error::IllegalArgument(format!(
                "blockLines size: {num_lines} cannot be split with deltaNumLines={}",
                self.delta_num_lines
            )));
        }
        let rest = self.block_lines.split_off(min_mdp_end_index);
        let lines = std::mem::replace(&mut self.block_lines, rest);
        self.write_block(&lines, block_output, field, dictionary)
    }

    /// `writeBlock(blockLines, dictionaryBuilder)`.
    fn write_block(
        &mut self,
        lines: &[PendingLine],
        block_output: &mut Vec<u8>,
        field: &mut FieldMetadata,
        dictionary: &mut FstDictionaryBuilder,
    ) -> Result<()> {
        let block_start_fp = block_output.len() as i64;
        // `addBlockKey`: the first term's minimal distinguishing prefix.
        let first = &lines[0];
        dictionary.add(&first.term[..first.mdp_length], block_start_fp)?;
        let middle = lines.len() >> 1;
        let mut middle_offset = -1;
        let mut previous: Option<&[u8]> = None;
        for (i, line) in lines.iter().enumerate() {
            let mut is_incremental_encoding_seed = i == 0;
            if i == middle {
                middle_offset = self.block_lines_write_buffer.len() as i32;
                is_incremental_encoding_seed = true;
            }
            self.write_block_line(
                is_incremental_encoding_seed,
                line,
                previous,
                field.field_info(),
            );
            previous = Some(&line.term);
        }
        let header = BlockHeader::new(
            lines.len() as i32,
            self.term_state_serializer.base_doc_start_fp,
            self.term_state_serializer.base_pos_start_fp,
            self.term_state_serializer.base_pay_start_fp,
            self.block_lines_write_buffer.len() as i32,
            middle_offset,
        );
        header.write(&mut self.block_write_buffer);
        self.block_write_buffer
            .extend_from_slice(&self.block_lines_write_buffer);
        self.block_write_buffer
            .extend_from_slice(&self.term_states_write_buffer);
        match &self.block_encoder {
            None => {
                block_output.write_vint(self.block_write_buffer.len() as i32);
                block_output.write_bytes(&self.block_write_buffer);
            }
            Some(encoder) => {
                let encoded = encoder.encode(&self.block_write_buffer)?;
                block_output.write_vint(encoded.len() as i32);
                block_output.write_bytes(&encoded);
            }
        }
        self.block_lines_write_buffer.clear();
        self.term_states_write_buffer.clear();
        self.block_write_buffer.clear();
        self.term_state_serializer.reset_base_start_fp();
        // `updateFieldMetadata(blockStartFP)`.
        if field.first_block_start_fp == -1 {
            field.first_block_start_fp = block_start_fp;
        }
        field.last_block_start_fp = block_start_fp;
        Ok(())
    }

    /// `writeBlockLine(isIncrementalEncodingSeed, line, previousLine)`.
    fn write_block_line(
        &mut self,
        is_incremental_encoding_seed: bool,
        line: &PendingLine,
        previous: Option<&[u8]>,
        field_info: &FieldInfo,
    ) {
        write_line(
            &mut self.block_lines_write_buffer,
            &line.term,
            line.mdp_length,
            previous,
            self.term_states_write_buffer.len() as i32,
            is_incremental_encoding_seed,
        );
        self.term_state_serializer.write_term_state(
            &mut self.term_states_write_buffer,
            field_info.index_options,
            field_info.store_payloads,
            line.doc_freq,
            line.total_term_freq,
            &line.meta,
        );
    }
}

/// `UniformSplitTermsWriter`: the `.ustb` and `.ustd` of one group, filled
/// field by field from the postings writer's term states.
pub struct UniformSplitTermsWriter<'s> {
    state: &'s SegmentWriteState<'s>,
    block_writer: BlockWriter,
    block_output: Vec<u8>,
    dictionary_output: Vec<u8>,
    fields_output: Vec<u8>,
    fields_number: i32,
}

impl<'s> UniformSplitTermsWriter<'s> {
    /// The constructor: both headers, then `postingsWriter.init`'s header in
    /// the `.ustb`.
    ///
    /// # Errors
    /// [`Error::IllegalArgument`] for settings `validateSettings` refuses.
    pub fn new(
        state: &'s SegmentWriteState<'s>,
        target_num_block_lines: i32,
        delta_num_lines: i32,
        block_encoder: Option<Arc<dyn BlockEncoder>>,
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
        // Both validated as `0 <= delta < target <= 1000`.
        let to_usize = |n: i32| usize::try_from(n).unwrap_or(0);
        Ok(UniformSplitTermsWriter {
            state,
            block_writer: BlockWriter::new(
                to_usize(target_num_block_lines),
                to_usize(delta_num_lines),
                block_encoder,
            ),
            block_output,
            dictionary_output,
            fields_output: Vec::new(),
            fields_number: 0,
        })
    }

    /// `writeFieldTerms(blockWriter, fieldsOutput, termsEnum, fieldInfo,
    /// normsProducer)` for one field's terms, as the postings writer laid
    /// them out.
    ///
    /// # Errors
    /// [`Error::IllegalArgument`] for a field number the segment's
    /// `FieldInfos` lacks; what the encoder or the dictionary fail with.
    pub fn write_field(&mut self, field: &FieldTermStates<'_>) -> Result<()> {
        let field_info = self
            .state
            .field_info(field.field_number)
            .ok_or_else(|| {
                Error::IllegalArgument(format!("unknown field number {}", field.field_number))
            })?
            .clone();
        let mut metadata = FieldMetadata::new(field_info, field.doc_count);
        metadata.dictionary_start_fp = self.dictionary_output.len() as i64;
        let mut dictionary = FstDictionaryBuilder::new();
        for i in 0..field.len() {
            let term = field.term(i);
            let (doc_freq, total_term_freq) = (field.doc_freq(i), field.total_term_freq(i));
            metadata.update_stats(doc_freq, total_term_freq);
            let mdp_length =
                TermBytes::compute_mdp_length(self.block_writer.last_term.as_deref(), term);
            let line = PendingLine {
                term: term.to_vec(),
                mdp_length,
                doc_freq,
                total_term_freq,
                meta: field.term_metadata(i),
            };
            self.block_writer.add_line(
                line,
                &mut self.block_output,
                &mut metadata,
                &mut dictionary,
            )?;
        }
        let last_term = self.block_writer.last_term.clone();
        self.block_writer.finish_last_block(
            &mut self.block_output,
            &mut metadata,
            &mut dictionary,
        )?;
        if metadata.num_terms > 0 {
            metadata.last_term = last_term.unwrap_or_default();
            metadata.write(&mut self.fields_output);
            // `writeDictionary(dictionaryBuilder)`.
            dictionary.build()?.write(
                &mut self.dictionary_output,
                self.block_writer.block_encoder.as_deref(),
            )?;
            // ARITH: one per field, at most the field count (an `i32`).
            #[allow(clippy::arithmetic_side_effects)]
            {
                self.fields_number += 1;
            }
        }
        Ok(())
    }

    /// `writeFieldsMetadata(fieldsNumber, fieldsOutput)` and the
    /// dictionary's footer: the finished `.ustb` and `.ustd`.
    ///
    /// # Errors
    /// What the encoder fails with.
    pub fn finish(mut self) -> Result<(Vec<u8>, Vec<u8>)> {
        let fields_start_position = self.block_output.len() as i64;
        self.block_output.write_vint(self.fields_number);
        match &self.block_writer.block_encoder {
            None => self.block_output.write_bytes(&self.fields_output),
            Some(encoder) => {
                let encoded = encoder.encode(&self.fields_output)?;
                self.block_output.write_vlong(encoded.len() as i64);
                self.block_output.write_bytes(&encoded);
            }
        }
        self.block_output.write_i64(fields_start_position);
        codec_util::write_footer(&mut self.block_output);
        codec_util::write_footer(&mut self.dictionary_output);
        Ok((self.block_output, self.dictionary_output))
    }
}
