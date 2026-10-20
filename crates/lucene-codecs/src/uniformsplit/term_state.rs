//! Port of `DeltaBaseTermStateSerializer`: a term's `IntBlockTermState`,
//! its file pointers written as deltas from the first non-zero one of its
//! block (the bases go in the block's header).

use lucene_store::data_output::DataOutput;

use super::{corrupt, read_vint, read_vlong, Result};
use crate::field_infos::IndexOptions;
use crate::postings::{TermMetadata, BLOCK_SIZE};

/// `IntBlockTermState` as this dictionary reads it: the statistics, the
/// postings pointers, and where the term's line is (`termBlockOrd`,
/// `blockFilePointer`), which `seekExact(term, state)` goes back to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UniformSplitTermState {
    /// `docFreq`.
    pub doc_freq: i32,
    /// `totalTermFreq` (`docFreq` for a field without frequencies).
    pub total_term_freq: i64,
    /// `docStartFP`, `singletonDocID`, `posStartFP`, `payStartFP`,
    /// `lastPosBlockOffset`.
    pub meta: TermMetadata,
    /// `termBlockOrd`: the line's 1-based index in its block.
    pub term_block_ord: i32,
    /// `blockFilePointer`: the start of the line's block in the `.ustb`.
    pub block_file_pointer: i64,
}

impl Default for UniformSplitTermState {
    /// `DeltaBaseTermStateSerializer.reset(termState)`.
    fn default() -> Self {
        UniformSplitTermState {
            doc_freq: 0,
            total_term_freq: 0,
            meta: TermMetadata::EMPTY,
            term_block_ord: 0,
            block_file_pointer: 0,
        }
    }
}

/// `DeltaBaseTermStateSerializer`: the running bases of the block being
/// written.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct DeltaBaseTermStateSerializer {
    /// `baseDocStartFP`.
    pub(crate) base_doc_start_fp: i64,
    /// `basePosStartFP`.
    pub(crate) base_pos_start_fp: i64,
    /// `basePayStartFP`.
    pub(crate) base_pay_start_fp: i64,
}

/// A file pointer off disk: Java adds a delta to its block's base without a
/// check and fails on the seek; this port refuses a negative one here.
fn file_pointer(fp: i64) -> Result<u64> {
    u64::try_from(fp).map_err(|_| corrupt(format!("negative file pointer {fp}")))
}

impl DeltaBaseTermStateSerializer {
    /// `resetBaseStartFP()`.
    pub(crate) fn reset_base_start_fp(&mut self) {
        *self = DeltaBaseTermStateSerializer::default();
    }

    /// `writeTermState(termStatesOutput, fieldInfo, termState)`.
    /// `total_term_freq` is `-1` for a field without frequencies, as
    /// `Lucene104PostingsWriter` leaves it (and never written then).
    pub(crate) fn write_term_state(
        &mut self,
        out: &mut Vec<u8>,
        index_options: IndexOptions,
        has_payloads: bool,
        doc_freq: i32,
        total_term_freq: i64,
        meta: &TermMetadata,
    ) {
        let has_freqs = index_options != IndexOptions::Docs;
        let has_positions = index_options.subsumes_positions();
        let has_offsets = index_options.subsumes_offsets();
        out.write_vint(doc_freq);
        if has_freqs {
            // Java asserts totalTermFreq >= docFreq; both are counts.
            out.write_vlong(total_term_freq.wrapping_sub(i64::from(doc_freq)));
        }
        // Lucene104's file pointers are below 2^63: `as i64` is exact.
        if meta.singleton_doc_id != -1 {
            out.write_vint(meta.singleton_doc_id);
        } else {
            if self.base_doc_start_fp == 0 {
                self.base_doc_start_fp = meta.doc_start_fp as i64;
            }
            out.write_vlong((meta.doc_start_fp as i64).wrapping_sub(self.base_doc_start_fp));
        }
        if has_positions {
            if self.base_pos_start_fp == 0 {
                self.base_pos_start_fp = meta.pos_start_fp as i64;
            }
            out.write_vlong((meta.pos_start_fp as i64).wrapping_sub(self.base_pos_start_fp));
            if has_payloads || has_offsets {
                if self.base_pay_start_fp == 0 {
                    self.base_pay_start_fp = meta.pay_start_fp as i64;
                }
                out.write_vlong((meta.pay_start_fp as i64).wrapping_sub(self.base_pay_start_fp));
            }
            if meta.last_pos_block_offset != -1 {
                out.write_vlong(meta.last_pos_block_offset);
            }
        }
    }

    /// `readTermState(baseDocStartFP, basePosStartFP, basePayStartFP,
    /// termStatesInput, fieldInfo, reuse)` at `*pos`.
    ///
    /// # Errors
    /// `Corrupted` for a term state past the block's end or a negative file
    /// pointer.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn read_term_state(
        base_doc_start_fp: i64,
        base_pos_start_fp: i64,
        base_pay_start_fp: i64,
        buf: &[u8],
        pos: &mut usize,
        index_options: IndexOptions,
        has_payloads: bool,
    ) -> Result<UniformSplitTermState> {
        let has_freqs = index_options != IndexOptions::Docs;
        let has_positions = index_options.subsumes_positions();
        let mut state = UniformSplitTermState {
            doc_freq: read_vint(buf, pos)?,
            ..UniformSplitTermState::default()
        };
        state.total_term_freq = if has_freqs {
            // Java's long addition.
            i64::from(state.doc_freq).wrapping_add(read_vlong(buf, pos)?)
        } else {
            i64::from(state.doc_freq)
        };
        if state.doc_freq == 1 {
            state.meta.singleton_doc_id = read_vint(buf, pos)?;
        } else {
            state.meta.doc_start_fp =
                file_pointer(base_doc_start_fp.wrapping_add(read_vlong(buf, pos)?))?;
        }
        if has_positions {
            state.meta.pos_start_fp =
                file_pointer(base_pos_start_fp.wrapping_add(read_vlong(buf, pos)?))?;
            if index_options.subsumes_offsets() || has_payloads {
                state.meta.pay_start_fp =
                    file_pointer(base_pay_start_fp.wrapping_add(read_vlong(buf, pos)?))?;
            }
            if state.total_term_freq > i64::from(BLOCK_SIZE) {
                state.meta.last_pos_block_offset = read_vlong(buf, pos)?;
            }
        }
        Ok(state)
    }
}
