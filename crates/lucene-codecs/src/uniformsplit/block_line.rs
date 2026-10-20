//! Port of `TermBytes`, `BlockLine` and `BlockLine.Serializer`: one line of
//! a block -- where its term state starts, and its term, incrementally
//! encoded against the line before.

use lucene_store::data_output::DataOutput;

use super::{corrupt, length, read_slice, read_vint, read_vlong, Result};

/// `TermBytes`' `mdpLength` ("minimal distinguishing prefix") helpers; the
/// term itself lives in the caller's buffer (Java: a `BytesRef` field).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TermBytes;

impl TermBytes {
    /// `TermBytes.computeMdpLength(previousTerm, currentTerm)`: the length of
    /// the shortest prefix of `current` greater than `previous` (1 for the
    /// first term), at most `current`'s length.
    pub(crate) fn compute_mdp_length(previous: Option<&[u8]>, current: &[u8]) -> usize {
        let mdp = match previous {
            None => 1,
            // Terms come strictly increasing; equal terms (never written)
            // would have no distinguishing prefix, so the whole term.
            Some(prev) => {
                lucene_util::string_helper::sort_key_length(prev, current).unwrap_or(usize::MAX)
            }
        };
        mdp.min(current.len())
    }

    /// `getSuffixOffset()`: `max(mdpLength - 1, 0)`.
    pub(crate) fn suffix_offset(mdp_length: usize) -> usize {
        mdp_length.saturating_sub(1)
    }
}

/// `BlockLine.Serializer.numBitsToEncode(i)`: `32 -
/// Integer.numberOfLeadingZeros(i)`.
pub(crate) fn num_bits_to_encode(i: i32) -> u32 {
    // ARITH: `leading_zeros` of a `u32` is at most 32.
    #[allow(clippy::arithmetic_side_effects)]
    let bits = 32 - (i as u32).leading_zeros();
    bits
}

/// `BlockLine.Serializer.writeLine(blockOutput, line, previousLine,
/// termStateRelativeOffset, isIncrementalEncodingSeed)`, with
/// `writeIncrementallyEncodedTerm`.
pub(crate) fn write_line(
    out: &mut Vec<u8>,
    term: &[u8],
    mdp_length: usize,
    previous_term: Option<&[u8]>,
    term_state_relative_offset: i32,
    is_incremental_encoding_seed: bool,
) {
    out.write_vint(term_state_relative_offset);
    if is_incremental_encoding_seed {
        // Mdp length is always 1 for an incremental encoding seed.
        out.write_vlong(term.len() as i64);
        out.write_bytes(term);
        return;
    }
    if term.is_empty() {
        // Empty term.
        out.write_vlong(0);
        return;
    }
    // (mdpLength - 1) in the low bits(previous length) bits, the suffix
    // length above: (mdpLength - 1) <= previous length, as the previous term
    // is a prefix of the current one at worst.
    let num_mdp_bits = num_bits_to_encode(previous_term.map_or(0, <[u8]>::len) as i32);
    let suffix_offset = TermBytes::suffix_offset(mdp_length);
    // ARITH: a non-empty term has `1 <= mdpLength <= term.len()`, so the
    // suffix offset is within the term; a term's length is below 2^31 and
    // `num_mdp_bits <= 32`, so the shifted suffix length fits an i64.
    #[allow(clippy::arithmetic_side_effects)]
    let mdp_and_suffix_lengths =
        (((term.len() - suffix_offset) as i64) << num_mdp_bits) | (mdp_length as i64 - 1);
    out.write_vlong(mdp_and_suffix_lengths);
    out.write_bytes(&term[suffix_offset..]);
}

/// `BlockLine.Serializer` on the read side: `currentTerm`, the term of the
/// last line read, which the next line's suffix is appended to.
#[derive(Debug, Clone, Default)]
pub(crate) struct BlockLineReader {
    /// `currentTerm`.
    pub(crate) current_term: Vec<u8>,
}

impl BlockLineReader {
    /// `readLine(blockInput, isIncrementalEncodingSeed, reuse)` at `*pos`:
    /// the line's term state offset and `mdpLength`; its term is
    /// [`Self::current_term`].
    ///
    /// # Errors
    /// `Corrupted` for a negative offset or length, a shared prefix longer
    /// than the line before's term, or a line past the block's end.
    pub(crate) fn read_line(
        &mut self,
        buf: &[u8],
        pos: &mut usize,
        is_incremental_encoding_seed: bool,
    ) -> Result<(i32, usize)> {
        let term_state_relative_offset = read_vint(buf, pos)?;
        if term_state_relative_offset < 0 {
            return Err(corrupt(format!(
                "Illegal termStateRelativeOffset= {term_state_relative_offset}"
            )));
        }
        let mdp_length =
            self.read_incrementally_encoded_term(buf, pos, is_incremental_encoding_seed)?;
        Ok((term_state_relative_offset, mdp_length))
    }

    /// `readIncrementallyEncodedTerm(blockInput, isIncrementalEncodingSeed,
    /// reuse)`: the `mdpLength`.
    fn read_incrementally_encoded_term(
        &mut self,
        buf: &[u8],
        pos: &mut usize,
        is_incremental_encoding_seed: bool,
    ) -> Result<usize> {
        if is_incremental_encoding_seed {
            // Java's `(int) readVLong()`.
            let len = length(i64::from(read_vlong(buf, pos)? as i32))?;
            let bytes = read_slice(buf, pos, len)?;
            self.current_term.clear();
            self.current_term.extend_from_slice(bytes);
            return Ok(usize::from(len != 0));
        }
        let mdp_and_suffix_lengths = read_vlong(buf, pos)?;
        if mdp_and_suffix_lengths == 0 {
            // Empty term.
            self.current_term.clear();
            return Ok(0);
        }
        let num_mdp_bits = num_bits_to_encode(self.current_term.len() as i32);
        // Java's int arithmetic: `(1 << numMdpBits) - 1`, a shift by 32
        // being a shift by 0.
        let mask = 1i32.wrapping_shl(num_mdp_bits).wrapping_sub(1);
        let mdp_minus_one = length(i64::from((mdp_and_suffix_lengths & i64::from(mask)) as i32))?;
        let suffix_length = length(i64::from(
            ((mdp_and_suffix_lengths as u64) >> num_mdp_bits.min(63)) as i32,
        ))?;
        if mdp_minus_one > self.current_term.len() {
            // Java asserts it (and reads the missing prefix as zeros).
            return Err(corrupt(format!(
                "line shares {mdp_minus_one} bytes with a {}-byte term",
                self.current_term.len()
            )));
        }
        let suffix = read_slice(buf, pos, suffix_length)?;
        self.current_term.truncate(mdp_minus_one);
        self.current_term.extend_from_slice(suffix);
        // ARITH: `mdp_minus_one <= current_term.len()`, a buffer length.
        #[allow(clippy::arithmetic_side_effects)]
        let mdp_length = mdp_minus_one + 1;
        Ok(mdp_length)
    }
}
