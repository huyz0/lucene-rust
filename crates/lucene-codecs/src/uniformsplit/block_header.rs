//! Port of `BlockHeader` and `BlockHeader.Serializer`: a block's line
//! count, the term states' base file pointers, and where its term states and
//! middle line start.

use lucene_store::data_output::DataOutput;

use super::{corrupt, read_vint, read_vlong, Result, MAX_NUM_BLOCK_LINES};

/// `BlockHeader`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct BlockHeader {
    /// `linesCount`.
    pub(crate) lines_count: i32,
    /// `baseDocsFP`.
    pub(crate) base_docs_fp: i64,
    /// `basePositionsFP`.
    pub(crate) base_positions_fp: i64,
    /// `basePayloadsFP`.
    pub(crate) base_payloads_fp: i64,
    /// `termStatesBaseOffset`: the term states' start, after the lines.
    pub(crate) term_states_base_offset: i32,
    /// `middleLineIndex`: `linesCount >> 1`.
    pub(crate) middle_line_index: i32,
    /// `middleLineOffset`: the middle line's start, after the first line.
    pub(crate) middle_line_offset: i32,
}

impl BlockHeader {
    /// `new BlockHeader(...)` / `reset(...)`.
    pub(crate) fn new(
        lines_count: i32,
        base_docs_fp: i64,
        base_positions_fp: i64,
        base_payloads_fp: i64,
        term_states_base_offset: i32,
        middle_line_offset: i32,
    ) -> Self {
        BlockHeader {
            lines_count,
            base_docs_fp,
            base_positions_fp,
            base_payloads_fp,
            term_states_base_offset,
            middle_line_index: lines_count >> 1,
            middle_line_offset,
        }
    }

    /// `Serializer.write(output, blockHeader)`.
    pub(crate) fn write(&self, out: &mut Vec<u8>) {
        out.write_vint(self.lines_count);
        out.write_vlong(self.base_docs_fp);
        out.write_vlong(self.base_positions_fp);
        out.write_vlong(self.base_payloads_fp);
        out.write_vint(self.term_states_base_offset);
        out.write_vint(self.middle_line_offset);
    }

    /// `Serializer.read(input, reuse)` at `*pos`.
    ///
    /// # Errors
    /// `Corrupted` for a line count outside `1..=MAX_NUM_BLOCK_LINES`, a
    /// negative offset, or a header past the block's end.
    pub(crate) fn read(buf: &[u8], pos: &mut usize) -> Result<Self> {
        let lines_count = read_vint(buf, pos)?;
        if lines_count <= 0 || lines_count > MAX_NUM_BLOCK_LINES {
            return Err(corrupt(format!(
                "Illegal number of lines in block: {lines_count}"
            )));
        }
        let base_docs_fp = read_vlong(buf, pos)?;
        let base_positions_fp = read_vlong(buf, pos)?;
        let base_payloads_fp = read_vlong(buf, pos)?;
        let term_states_base_offset = read_vint(buf, pos)?;
        if term_states_base_offset < 0 {
            return Err(corrupt(format!(
                "Illegal termStatesBaseOffset= {term_states_base_offset}"
            )));
        }
        let middle_line_offset = read_vint(buf, pos)?;
        if middle_line_offset < 0 {
            return Err(corrupt(format!(
                "Illegal middleTermOffset= {middle_line_offset}"
            )));
        }
        Ok(BlockHeader::new(
            lines_count,
            base_docs_fp,
            base_positions_fp,
            base_payloads_fp,
            term_states_base_offset,
            middle_line_offset,
        ))
    }
}
