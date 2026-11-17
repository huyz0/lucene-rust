//! Port of `org.apache.lucene.util.fst.FST` (read side) plus a from-scratch,
//! simplified construction path (`build_fst`) usable by this module's own
//! reader.
//!
//! This module implements enough of Lucene's FST (finite state transducer)
//! format to look up a byte-sequence key in an already-built, on-disk FST and
//! recover its accumulated output (`Util.get(fst, BytesRef)`). It does **not**
//! port real Lucene's *incremental* construction algorithm
//! (`FSTCompiler`/`Builder`'s node-freezing, streaming/memory-bounded build)
//! -- that remains a separate, much larger undertaking (see `docs/parity.md`).
//! Instead, `build_fst` (see the "FST construction" section near the bottom
//! of this file) builds a full in-memory trie upfront and serializes it
//! bottom-up to the same byte format `Fst::read`/`Fst::get` below already
//! parse -- correct and round-trip-verified against this module's own
//! unmodified reader. It **does** reproduce real `FSTCompiler`'s suffix
//! sharing / minimization (via a `NodeHash`-equivalent dedup table keyed on
//! each node's fully-resolved arc signature -- see `NodeHash`/`build_node`),
//! just via whole-tree hash-consing rather than incremental freezing; it does
//! not reproduce fixed-length arc nodes or output pushing. See that section's
//! doc comment for the precise list of what's reproduced and what's deferred.
//!
//! ## Scope of this slice
//!
//! - **On-heap only.** Real Lucene's `FST` can be backed by an off-heap
//!   (mmap'd) `OffHeapFSTStore` or an on-heap `OnHeapFSTStore`. Only the
//!   on-heap representation is ported: the FST body is a plain `Vec<u8>`
//!   (`OnHeapFSTStore` with `bytesArray != null` writes/reads the body as one
//!   contiguous forward byte array — see `OnHeapFSTStore.java`).
//! - **Single output type on the wire: `BytesRef`-shaped
//!   (`ByteSequenceOutputs`).** This is the output type real Lucene uses for
//!   the term index FST (`Lucene90BlockTreeTermsReader`'s `.tip` FST maps
//!   term-prefix byte sequences to concatenated block-pointer byte
//!   sequences), so it's the one needed to eventually navigate BlockTree.
//!   `Fst`/`build_fst`/`build_node` are hardcoded to `Vec<u8>` outputs and
//!   stay that way. Typed output values (`PositiveIntOutputs`,
//!   `ByteSequenceOutputs`, `PairOutputs<A, B>`) are layered *on top of* that
//!   single wire type via the `Outputs` trait and `build_fst_typed`/
//!   `Fst::get_typed`, which just encode/decode a typed value to/from the
//!   `Vec<u8>` this module already stores -- see the "Typed output values"
//!   section near the bottom of this file for why that's a faithful (not a
//!   corner-cut) port of `PairOutputs` given this builder never pushes
//!   output prefixes toward the root.
//! - **Variable-length ("list") arc nodes, plus all three fixed-length arc
//!   node encodings.** Real Lucene's `FSTCompiler` also emits fixed-length
//!   arc nodes for binary search (`ARCS_FOR_BINARY_SEARCH`), direct
//!   addressing (`ARCS_FOR_DIRECT_ADDRESSING`) and continuous ranges
//!   (`ARCS_FOR_CONTINUOUS`) once a node has "enough" arcs to make the
//!   space/speed tradeoff worth it. `ARCS_FOR_BINARY_SEARCH` is ported
//!   (`find_target_arc`'s sparse binary search over fixed-size arc slots,
//!   `read_arc`'s matching `BIT_TARGET_NEXT` target rule); `ARCS_FOR_DIRECT_ADDRESSING`
//!   is also ported (node header's label range + presence bitset ("bit
//!   table"), then fixed-size arc slots addressed only for present labels via
//!   `BitTableUtil`-equivalent bit counting -- see `bit_table_*` helpers below).
//!   `ARCS_FOR_CONTINUOUS` is ported too (`find_target_arc_continuous`):
//!   simpler than direct addressing since every label in the range is
//!   guaranteed present, so there's no presence bit-table and no per-arc
//!   label byte at all -- just `firstLabel` plus a direct
//!   `label - firstLabel` index into fixed-size arc slots.
//! - **Lookup (`get`) and full forward enumeration (`iter`), not seek.** `get`
//!   mirrors `Util.get(FST, BytesRef)`: walk arcs for a specific key and
//!   return whether it's accepted plus its output. `iter`/`FstEnum` mirrors
//!   `BytesRefFSTEnum`'s full ascending-order walk (`next()`) over every
//!   accepted key -- but not its `seekCeil`/`seekFloor`/`seekExact` (needed
//!   for prefix/range-bounded iteration, not a full walk); see `FstEnum`'s
//!   doc comment for the precise scope and `docs/parity.md` for what's
//!   deferred. `IntsRefFSTEnum` (the `IntsRef`-input analogue) is also not
//!   ported, since this module only has a `BytesRef`-shaped output type to
//!   begin with -- see the point above.
//!
//! ## Wire format
//!
//! FST metadata (`FST.FSTMetadata.save`/`FST.readMetadata`) is a normal codec
//! header (`CodecUtil.writeHeader`/`checkHeader`, name `"FST"`, versions 6..9)
//! followed by:
//!
//! ```text
//! AcceptsEmpty(byte)         --> 1 if the FST accepts the empty string, else 0
//! [EmptyOutputLen(vint), EmptyOutputBytesReversed]   -- only if AcceptsEmpty==1
//! InputType(byte)            --> 0=BYTE1, 1=BYTE2, 2=BYTE4
//! StartNode(vlong)
//! NumBytes(vlong)
//! ```
//!
//! `NumBytes` raw FST body bytes follow immediately (`OnHeapFSTStore`'s
//! forward `readBytes` call) -- in this port's fixtures, metadata and body
//! share one file, exactly like `FST.save(Path)` does (`save(out, out)`).
//!
//! The FST body itself is read via a *reverse* byte cursor
//! (`ReverseBytesReader`): `readByte()` returns `bytes[pos]` and then
//! decrements `pos`. Node/arc encoding is otherwise a conventional
//! byte-oriented record format (label, optional output, optional final
//! output, target) whose exact per-arc flag bits are ported verbatim from
//! `FST.java`'s `readArc`/`findTargetArc`/`readFirstRealTargetArc`.

use lucene_store::codec_util;
use lucene_store::data_input::{DataInput, SliceInput};
use lucene_store::error::Error as StoreError;
use lucene_util::vint::{write_vint, write_vlong};

/// Errors specific to FST decoding, layered on top of `lucene_store::Error`.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Store(#[from] StoreError),
    #[error("unsupported FST feature: {0}")]
    Unsupported(String),
    #[error("corrupt FST: {0}")]
    Corrupt(String),
}

pub type Result<T> = std::result::Result<T, Error>;

pub(crate) const FILE_FORMAT_NAME: &str = "FST";
const VERSION_START: i32 = 6;
/// `FST.VERSION_LITTLE_ENDIAN`: from this version on, `BYTE2` labels
/// (`FST.readLabel`'s `readShort()` branch) are stored little-endian and read
/// back as-is; versions before this one wrote/read them byte-swapped (real
/// Lucene's short-lived big-endian `DataOutput`/`DataInput` era). Needed here
/// because this port's supported version range (`VERSION_START..=
/// VERSION_CURRENT`, i.e. 6..=9) straddles this boundary.
pub(crate) const VERSION_LITTLE_ENDIAN: i32 = 8;
pub(crate) const VERSION_CONTINUOUS_ARCS: i32 = 9;
pub(crate) const VERSION_CURRENT: i32 = VERSION_CONTINUOUS_ARCS;

pub(crate) const BIT_FINAL_ARC: u8 = 1 << 0;
pub(crate) const BIT_LAST_ARC: u8 = 1 << 1;
pub(crate) const BIT_TARGET_NEXT: u8 = 1 << 2;
pub(crate) const BIT_STOP_NODE: u8 = 1 << 3;
pub(crate) const BIT_ARC_HAS_OUTPUT: u8 = 1 << 4;
pub(crate) const BIT_ARC_HAS_FINAL_OUTPUT: u8 = 1 << 5;

pub(crate) const ARCS_FOR_BINARY_SEARCH: u8 = BIT_ARC_HAS_FINAL_OUTPUT;
pub(crate) const ARCS_FOR_DIRECT_ADDRESSING: u8 = 1 << 6;
pub(crate) const ARCS_FOR_CONTINUOUS: u8 = ARCS_FOR_DIRECT_ADDRESSING + ARCS_FOR_BINARY_SEARCH;

pub(crate) const FINAL_END_NODE: i64 = -1;
pub(crate) const NON_FINAL_END_NODE: i64 = 0;

/// `FST.END_LABEL`: the label of the "fake" virtual arc `read_first_target_arc`
/// inserts to represent a node's own acceptance, before falling through to
/// its real outgoing arcs (if any). Only meaningful to enumeration
/// (`FstEnum`/`read_first_target_arc`/`read_next_arc`) -- `get`/
/// `find_target_arc` never see or produce this value (see
/// `find_target_arc`'s own doc comment on why `label_to_match` is always a
/// real byte there).
const END_LABEL: i32 = -1;

fn flag(flags: u8, bit: u8) -> bool {
    flags & bit != 0
}

/// `FST.INPUT_TYPE`: the width of input labels. The BlockTree term index
/// always uses `BYTE1` (raw term bytes); the others are ported for
/// completeness of metadata parsing only.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputType {
    Byte1,
    Byte2,
    Byte4,
}

/// A reverse byte-array cursor over the FST body, matching
/// `ReverseBytesReader`: `read_byte` returns the byte at the current position
/// and then moves the cursor *backwards*. Positions are absolute indices into
/// the backing array (never negative in valid usage, but kept as `i64` since
/// Java's `getPosition()`/`setPosition()` use `long`/int freely and a few call
/// sites subtract before checking).
struct BytesReader<'a> {
    bytes: &'a [u8],
    pos: i64,
}

impl<'a> BytesReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    fn get_position(&self) -> i64 {
        self.pos
    }

    fn set_position(&mut self, pos: i64) {
        self.pos = pos;
    }

    /// `ReverseBytesReader.skipBytes`. Java's `pos` is an `int` and the
    /// subtraction silently wraps; here it saturates instead, which is not a
    /// "make it fit" reflex but the only outcome that stays *wrong in the
    /// same direction*: `i64::MIN`/`i64::MAX` are both outside
    /// `0..bytes.len()`, so the very next `read_byte` reports corruption. A
    /// wrap could land the cursor back inside the body and decode a
    /// plausible arc from the wrong offset.
    fn skip_bytes(&mut self, count: i64) {
        self.pos = self.pos.saturating_sub(count);
    }

    fn read_byte(&mut self) -> Result<u8> {
        let idx = self.pos;
        if idx < 0 || idx as usize >= self.bytes.len() {
            return Err(Error::Corrupt(format!(
                "FST byte-reader position {idx} out of range (len={})",
                self.bytes.len()
            )));
        }
        // ARITH: the guard above established `0 <= idx < bytes.len()`, so
        // `pos - 1` is at worst -1 -- which the same guard rejects on the
        // next call.
        #[allow(clippy::arithmetic_side_effects)]
        {
            self.pos -= 1;
        }
        Ok(self.bytes[idx as usize])
    }

    fn read_bytes(&mut self, out: &mut [u8]) -> Result<()> {
        let run = self.take_below(out.len())?;
        for (slot, &b) in out.iter_mut().zip(run.iter().rev()) {
            *slot = b;
        }
        Ok(())
    }

    /// The next `len` bytes as the body slice they occupy (lowest address
    /// first, the reverse of reading order), with the cursor moved past
    /// them: one bounds check for the run instead of one per byte.
    fn take_below(&mut self, len: usize) -> Result<&'a [u8]> {
        if len == 0 {
            return Ok(&[]);
        }
        let end = usize::try_from(self.pos)
            .ok()
            .and_then(|p| p.checked_add(1))
            .filter(|&e| e <= self.bytes.len());
        match end.and_then(|e| Some((e.checked_sub(len)?, e))) {
            Some((start, end)) => {
                self.pos = self.pos.saturating_sub(len as i64);
                Ok(&self.bytes[start..end])
            }
            None => Err(Error::Corrupt(format!(
                "FST byte-reader read of {len} bytes at position {} out of range (len={})",
                self.pos,
                self.bytes.len()
            ))),
        }
    }

    /// `DataInput.readVInt`: identical varint algorithm regardless of the
    /// direction bytes are physically stored in, since it only ever calls
    /// `read_byte()` in sequence.
    // ARITH: `shift` is compared against 28 before every increment, so it
    // never leaves 7..=35.
    #[allow(clippy::arithmetic_side_effects)]
    fn read_vint(&mut self) -> Result<i32> {
        let mut b = self.read_byte()?;
        let mut v = (b & 0x7f) as i32;
        let mut shift = 7;
        while b & 0x80 != 0 {
            // `DataOutput.writeVInt` emits "between one and five bytes"
            // (its javadoc; the loop is `i >>>= 7` on an `int`, so it cannot
            // emit more), and this bound is exactly that -- so it cannot
            // reject a `vint` any writer produced.
            //
            // It is *stricter* than 10.5.0's reader, deliberately.
            // `DataInput.readVInt` there is a bare unchecked `for` loop with
            // no length limit at all (the "Invalid vInt detected (too many
            // bits)" exception is a later addition and is **not** in the
            // pinned tree -- `grep` it and you will only find `CHANGES.txt`).
            // Without a bound the loop keeps consuming continuation bytes off
            // a corrupt body, silently returning a value assembled from bits
            // no encoder wrote, and `shift` itself overflows on a long enough
            // run. `lucene_store::data_input` already makes the same choice.
            if shift > 28 {
                return Err(Error::Corrupt(
                    "invalid FST vInt detected (too many bits)".to_string(),
                ));
            }
            b = self.read_byte()?;
            v |= ((b & 0x7f) as i32).wrapping_shl(shift);
            shift += 7;
        }
        Ok(v)
    }

    // ARITH: `shift` is compared against 64 before every increment, so it
    // never leaves 7..=70.
    #[allow(clippy::arithmetic_side_effects)]
    fn read_vlong(&mut self) -> Result<i64> {
        let mut b = self.read_byte()?;
        let mut v = (b & 0x7f) as i64;
        let mut shift = 7;
        while b & 0x80 != 0 {
            // `DataOutput.writeVLong` emits "between one and nine bytes" and
            // rejects a negative value outright
            // (`IllegalArgumentException("cannot write negative vLong")`), so
            // nine bytes is the most any writer produces. This bound admits
            // ten, i.e. it is one byte *looser* than the writer -- which is
            // what makes it unable to reject a real file. It exists to stop
            // the `shift` accumulator and the unbounded scan, not to police
            // the encoding; see `read_vint` on why 10.5.0's own reader has no
            // such bound.
            if shift >= 64 {
                return Err(Error::Corrupt(
                    "invalid FST vLong detected (too many bits)".to_string(),
                ));
            }
            b = self.read_byte()?;
            v |= ((b & 0x7f) as i64).wrapping_shl(shift);
            shift += 7;
        }
        Ok(v)
    }

    fn skip_output(&mut self) -> Result<()> {
        let len = self.checked_output_len()?;
        if len != 0 {
            self.skip_bytes(len as i64);
        }
        Ok(())
    }

    fn read_output(&mut self) -> Result<Vec<u8>> {
        let len = self.checked_output_len()?;
        if len == 0 {
            return Ok(Vec::new());
        }
        let mut out = vec![0u8; len as usize];
        self.read_bytes(&mut out)?;
        Ok(out)
    }

    /// `ByteSequenceOutputs.read`'s `int len = in.readVInt()`, bounded before
    /// it reaches a `vec![0u8; len]`.
    ///
    /// A `vint` decodes to any `i32`, and the reverse cursor consumes the
    /// output's bytes *downward* from the current position -- so a valid
    /// length is at most `pos + 1`. Without that bound a negative length
    /// sign-extends through `as usize` to ~2^64 and a positive one reaches
    /// 2 GB: both are `Vec` allocations sized by a number just read off disk,
    /// which abort the process rather than returning an error `catch_unwind`
    /// could turn into a JVM exception.
    fn checked_output_len(&mut self) -> Result<i32> {
        let len = self.read_vint()?;
        if len < 0 || len as i64 > self.pos.saturating_add(1) {
            return Err(Error::Corrupt(format!(
                "FST output length {len} does not fit below cursor position {} (len={})",
                self.pos,
                self.bytes.len()
            )));
        }
        Ok(len)
    }
}

/// `FST.getNumPresenceBytes`: number of bytes needed to hold one presence bit
/// per label in a range of `label_range` labels.
fn num_presence_bytes(label_range: i32) -> i32 {
    // Java is `(labelRange + 7) >> 3` on an `int`, which overflows for a
    // `labelRange` off disk within 7 of `Integer.MAX_VALUE`. Widening to
    // `i64` for the one add makes the result exact for every `i32` input and
    // still fits back in an `i32` (the shift divides by 8).
    // ARITH: `label_range as i64 + 7` cannot overflow an `i64`, and the
    // result of `>> 3` on a value below 2^31 fits an `i32`.
    #[allow(clippy::arithmetic_side_effects)]
    let bytes = (label_range as i64 + 7) >> 3;
    bytes as i32
}

/// `BitTableUtil.isBitSet`, pre-positioned via `arc.bit_table_start`
/// (`FST.Arc.BitTable.isBitSet`). `arc.node_flags` must be
/// `ARCS_FOR_DIRECT_ADDRESSING`.
fn bit_table_is_bit_set(bit_index: i32, arc: &Arc, r: &mut BytesReader) -> Result<bool> {
    debug_assert_eq!(arc.node_flags, ARCS_FOR_DIRECT_ADDRESSING);
    r.set_position(arc.bit_table_start);
    r.skip_bytes((bit_index >> 3) as i64);
    let b = r.read_byte()?;
    Ok(b & (1u8 << (bit_index & 7)) != 0)
}

/// `BitTableUtil.countBits`, pre-positioned via `arc.bit_table_start`: the
/// total number of set bits in the whole bit-table, i.e. the number of arcs
/// actually present in this direct-addressing node.
fn bit_table_count_bits(arc: &Arc, r: &mut BytesReader) -> Result<i32> {
    debug_assert_eq!(arc.node_flags, ARCS_FOR_DIRECT_ADDRESSING);
    r.set_position(arc.bit_table_start);
    let num_bytes = num_presence_bytes(arc.num_arcs);
    // The accumulator is an `i64` because an `i32` one is *exactly* one short:
    // `num_arcs` is an unbounded `vint` for a `BYTE4` FST (`check_label_range`
    // only caps it at `i32::MAX` there), so `num_presence_bytes` reaches
    // 2^28 and eight bits per byte reaches 2^31 -- one past `i32::MAX`. The
    // `try_from` below is what a table that large would fail on; a real one is
    // at most `(0x110000 + 7) >> 3` bytes, since a node's labels are distinct
    // and drawn from the `INPUT_TYPE`'s alphabet.
    let table = r.take_below(usize::try_from(num_bytes).unwrap_or(usize::MAX))?;
    let count = i64::from(crate::fst_util::popcount_bytes(table));
    i32::try_from(count).map_err(|_| Error::Corrupt(format!("FST bit table too large: {count}")))
}

/// `BitTableUtil.countBitsUpTo`, pre-positioned via `arc.bit_table_start`:
/// the number of set bits strictly before `bit_index` -- i.e. `bit_index`'s
/// corresponding `presence_index` if its bit is set.
fn bit_table_count_bits_up_to(bit_index: i32, arc: &Arc, r: &mut BytesReader) -> Result<i32> {
    debug_assert_eq!(arc.node_flags, ARCS_FOR_DIRECT_ADDRESSING);
    r.set_position(arc.bit_table_start);
    let full_bytes = bit_index >> 3;
    // `i64`, for the same reason as `bit_table_count_bits`.
    let mut count = 0i64;
    // ARITH: `full_bytes` is `bit_index >> 3`, so at most 2^28, and each byte
    // contributes at most 8 -- `count` stays at or below 2^31. `rem_bits` is
    // `bit_index & 7`, i.e. 1..=7 inside the branch, so `1u8 << rem_bits` is
    // at most 128 and the `- 1` cannot underflow.
    #[allow(clippy::arithmetic_side_effects)]
    {
        let table = r.take_below(usize::try_from(full_bytes).unwrap_or(usize::MAX))?;
        count += i64::from(crate::fst_util::popcount_bytes(table));
        let rem_bits = bit_index & 7;
        if rem_bits != 0 {
            let b = r.read_byte()?;
            let mask = (1u8 << rem_bits) - 1;
            count += i64::from((b & mask).count_ones());
        }
    }
    i32::try_from(count).map_err(|_| Error::Corrupt(format!("FST bit table too large: {count}")))
}

/// `BitTableUtil.nextBitSet`, pre-positioned via `arc.bit_table_start`: the
/// index of the next set bit strictly after `bit_index` (which may be `-1`,
/// meaning "the first set bit"), or `-1` if none.
//
// SENTINEL: `-1` = "no next present arc". Unchecked, `-1` would flow on as
// an `arcIdx` and `read_arc` would derive
// `firstLabel - 1` -- for `firstLabel == 0`, exactly `END_LABEL`. A byte-flip
// sweep cannot find it: a plausible wrong label is a clean decode.
fn bit_table_next_bit_set(bit_index: i32, arc: &Arc, r: &mut BytesReader) -> Result<i32> {
    debug_assert_eq!(arc.node_flags, ARCS_FOR_DIRECT_ADDRESSING);
    r.set_position(arc.bit_table_start);
    let bit_table_bytes = num_presence_bytes(arc.num_arcs);
    // ARITH: taken at the boundary, because every term here sits near one.
    // `bit_index` is `-1` or an `arc_idx`, which `read_next_real_arc` bounds
    // to `0..num_arcs`, so `bit_index <= i32::MAX - 1` and `bit_index + 1`
    // cannot overflow. `bit_table_bytes = (num_arcs + 7) >> 3 <= 2^28`, and
    // the loop returns before `byte_index` reaches it, so
    // `byte_index <= 2^28 - 1`, `byte_index + 1 <= 2^28`, and
    // `byte_index << 3 <= 2^31 - 8`; adding `trailing_zeros() <= 7` gives at
    // most `i32::MAX` exactly. `shift` is `& 7`, so the `-1i32 << shift`
    // shift amount is 0..=7.
    #[allow(clippy::arithmetic_side_effects)]
    {
        let mut byte_index = bit_index / 8;
        let shift = ((bit_index + 1) & 7) as u32;
        let mask: i32 = -1i32 << shift;
        let mut i: i32;
        if mask == -1 && bit_index != -1 {
            r.skip_bytes((byte_index + 1) as i64);
            i = 0;
        } else {
            r.skip_bytes(byte_index as i64);
            let b = r.read_byte()? as i32;
            i = (b & 0xff) & mask;
        }
        while i == 0 {
            byte_index += 1;
            if byte_index >= bit_table_bytes {
                return Ok(-1);
            }
            i = (r.read_byte()? as i32) & 0xff;
        }
        let index = i.trailing_zeros() as i32 + (byte_index << 3);
        // The bit table is rounded up to whole bytes, so its last byte has up
        // to seven bits *past* the label range. `FSTCompiler` writes those as
        // zero; a set one is corruption, and returning it would hand
        // `read_arc` an `arcIdx` outside `0..numArcs` and therefore an implied
        // label outside the alphabet `check_label_range` just bounded.
        if index >= arc.num_arcs {
            return Ok(-1);
        }
        Ok(index)
    }
}

/// `BitTableUtil.previousBitSet`, pre-positioned via `arc.bit_table_start`:
/// the index of the next set bit strictly *before* `bit_index`, or `-1` if
/// none. Needed by `FstEnum`'s `seek_floor` support (`FSTEnum`'s
/// `findNextFloorArcDirectAddressing`/`doSeekFloorArrayDirectAddressing`) --
/// `bit_table_next_bit_set`'s mirror-image counterpart.
//
// SENTINEL: `-1` = "no previous present arc", the mirror of
// `bit_table_next_bit_set`'s and outside the domain of an arc index the same
// way.
fn bit_table_previous_bit_set(bit_index: i32, arc: &Arc, r: &mut BytesReader) -> Result<i32> {
    debug_assert_eq!(arc.node_flags, ARCS_FOR_DIRECT_ADDRESSING);
    debug_assert!(bit_index >= 0);
    r.set_position(arc.bit_table_start);
    // ARITH: again at the boundary. `bit_index >= 0`, so
    // `byte_index = bit_index >> 3 <= 2^28 - 1` and `byte_index << 3 <=
    // 2^31 - 8`; `byte_index` only decreases from there and stops at 0.
    // `i` is a single byte, so `1 <= i <= 255` when the loop exits and
    // `31 - leading_zeros()` is in `0..=7` -- the sum is at most `i32::MAX`.
    // `bit_index & 7` is 0..=7, so `1i32 << (bit_index & 7)` is at most 128
    // and the `- 1` cannot underflow.
    #[allow(clippy::arithmetic_side_effects)]
    {
        let mut byte_index = bit_index >> 3;
        r.skip_bytes(byte_index as i64);
        let mask = (1i32 << (bit_index & 7)) - 1;
        let mut i = (r.read_byte()? as i32) & mask;
        while i == 0 {
            if byte_index == 0 {
                return Ok(-1);
            }
            byte_index -= 1;
            r.skip_bytes(-2);
            i = r.read_byte()? as i32;
        }
        Ok(31 - i.leading_zeros() as i32 + (byte_index << 3))
    }
}

/// `ByteSequenceOutputs.add`: concatenate prefix output with an arc's own
/// output.
fn output_add(prefix: &[u8], output: &[u8]) -> Vec<u8> {
    if prefix.is_empty() {
        return output.to_vec();
    }
    if output.is_empty() {
        return prefix.to_vec();
    }
    // ARITH: two in-memory slice lengths, each at most `isize::MAX`; the sum
    // is what the `Vec` about to hold both of them needs, so an overflow here
    // would mean the concatenation itself cannot exist.
    #[allow(clippy::arithmetic_side_effects)]
    let mut out = Vec::with_capacity(prefix.len() + output.len());
    out.extend_from_slice(prefix);
    out.extend_from_slice(output);
    out
}

/// `FST.Arc<T>`, restricted to the fields needed for variable-length
/// ("list") arc nodes plus the three array-node encodings this port decodes
/// (`ARCS_FOR_BINARY_SEARCH`, `ARCS_FOR_DIRECT_ADDRESSING`,
/// `ARCS_FOR_CONTINUOUS` -- the latter two share the `first_label`/
/// `pos_arcs_start` fields, `ARCS_FOR_CONTINUOUS` just never needs
/// `bit_table_start`/`presence_index` since every label in its range is
/// present).
#[derive(Debug, Clone, Default)]
pub struct Arc {
    label: i32,
    output: Vec<u8>,
    target: i64,
    flags: u8,
    next_final_output: Vec<u8>,
    next_arc: i64,
    /// `FST.Arc.nodeFlags`: the node header's own flags byte (as opposed to
    /// `flags`, this arc's *own* per-arc flags byte). `0` for list-encoded
    /// nodes (never one of the two special constants below); one of
    /// `ARCS_FOR_BINARY_SEARCH`/`ARCS_FOR_DIRECT_ADDRESSING` for a fixed-
    /// length-arc node. Needed because both share a non-zero `bytes_per_arc`
    /// but decode differently (label storage, target-skip arithmetic).
    node_flags: u8,
    /// `FST.Arc.bytesPerArc`: 0 for list-encoded (variable length arc) nodes;
    /// non-zero for `ARCS_FOR_BINARY_SEARCH`/`ARCS_FOR_DIRECT_ADDRESSING`
    /// fixed-length-arc nodes.
    bytes_per_arc: i32,
    /// `FST.Arc.numArcs`: only meaningful when `bytes_per_arc != 0`. For
    /// `ARCS_FOR_BINARY_SEARCH` this is the arc-array size; for
    /// `ARCS_FOR_DIRECT_ADDRESSING` this is the label range (`numArcs` in
    /// `FST.java`'s own naming, despite actually holding a range width).
    num_arcs: i32,
    /// `FST.Arc.posArcsStart`: only meaningful when `bytes_per_arc != 0`.
    pos_arcs_start: i64,
    /// `FST.Arc.arcIdx`: this arc's index within its fixed-length-arc node's
    /// slots (only meaningful/maintained when `bytes_per_arc != 0`). Needed
    /// by `read_next_real_arc`'s fixed-length-arc branches to advance
    /// incrementally through a node's arcs during enumeration (`get`/
    /// `find_target_arc` never need this: they jump straight to one matched
    /// slot and stop).
    arc_idx: i32,
    /// `FST.Arc.bitTableStart`: start position of the presence bit-table for
    /// an `ARCS_FOR_DIRECT_ADDRESSING` node (only meaningful when
    /// `node_flags == ARCS_FOR_DIRECT_ADDRESSING`).
    bit_table_start: i64,
    /// `FST.Arc.firstLabel`: first (lowest) label of an
    /// `ARCS_FOR_DIRECT_ADDRESSING` node's label range (only meaningful when
    /// `node_flags == ARCS_FOR_DIRECT_ADDRESSING`).
    first_label: i32,
    /// `FST.Arc.presenceIndex`: this arc's index among only the *present*
    /// labels of an `ARCS_FOR_DIRECT_ADDRESSING` node (i.e. the count of set
    /// bits before `arc_idx` in the bit-table) -- only meaningful when
    /// `node_flags == ARCS_FOR_DIRECT_ADDRESSING`.
    presence_index: i32,
}

impl Arc {
    pub fn label(&self) -> i32 {
        self.label
    }

    pub fn output(&self) -> &[u8] {
        &self.output
    }

    pub fn target(&self) -> i64 {
        self.target
    }

    fn flag(&self, bit: u8) -> bool {
        flag(self.flags, bit)
    }

    pub fn is_last(&self) -> bool {
        self.flag(BIT_LAST_ARC)
    }

    pub fn is_final(&self) -> bool {
        self.flag(BIT_FINAL_ARC)
    }

    pub fn next_final_output(&self) -> &[u8] {
        &self.next_final_output
    }
}

fn target_has_arcs(arc: &Arc) -> bool {
    arc.target > 0
}

/// The invariants below all hold for any FST a `FSTCompiler` produced, and
/// Java states each of them as an `assert` -- i.e. a check that is *off* in
/// production. Ported as `debug_assert!` they were worse than that: a debug
/// build panics and a release build walks on with a nonsense arc, so the two
/// disagree on corrupt input and neither rejects it. They are typed errors
/// here instead. (The byte-flip sweep in
/// `tests/fst_byte_flip_sweep.rs` reaches five of them from a single flipped
/// byte.)
fn corrupt_fst<T>(what: &str) -> Result<T> {
    Err(Error::Corrupt(format!("malformed FST node: {what}")))
}

/// Parsed `FST.FSTMetadata`.
#[derive(Debug, Clone)]
pub struct FstMetadata {
    pub input_type: InputType,
    pub empty_output: Option<Vec<u8>>,
    pub start_node: i64,
    pub version: i32,
    pub num_bytes: i64,
}

/// The FST body ("`OnHeapFSTStore`'s bytes"), either owned by this `Fst` or
/// borrowed from a caller-owned byte slice.
///
/// ## On "off-heap" storage in this port
///
/// Real Lucene's `OffHeapFSTStore` backs the FST body with a slice into a
/// memory-mapped file, so opening a large FST never pays for a full extra
/// heap copy of its bytes -- the OS page cache is the only backing store,
/// and pages are faulted in lazily on access. This port's `Directory`
/// abstraction (`lucene_store::directory::MmapDirectory`) already provides
/// exactly that: `MmapDirectory::open` returns `Input::Mapped(memmap2::Mmap)`,
/// a real OS memory mapping that `Deref`s to `&[u8]` without ever copying the
/// file into a `Vec<u8>`.
///
/// What `FstBytes::Borrowed` adds on top is the missing piece for FSTs
/// specifically: `Fst::read` unconditionally allocates a fresh `Vec<u8>` and copies
/// `num_bytes` of body into it (see `Fst::read`'s `Owned` path below) --
/// *even if* the caller's `input` was itself already a zero-copy view over an
/// `Input::Mapped` mmap. That copy is real, avoidable off-heap-storage cost:
/// for a large `.tip` FST, it's a full second copy of already-resident,
/// already-addressable bytes. `Fst::read_borrowed` (below) instead slices the
/// body directly out of the caller's `'a`-lifetime buffer via
/// `SliceInput::slice`/`SliceInput::as_slice`, which for an `Input::Mapped`-
/// backed `SliceInput` is a sub-range of the mmap itself -- no allocation, no
/// copy, and (because `memmap2::Mmap` pages are only faulted in as touched)
/// genuinely OS-page-cache-backed the same way real Lucene's
/// `OffHeapFSTStore` is.
///
/// This is **not** a claim that this module itself performs `mmap(2)` --
/// it doesn't, and doesn't need to: that's `lucene-store`'s job, already
/// done. What's added here is the FST-level plumbing (a lifetime-generic
/// `Fst<'a>` and a body representation that can be either owned or borrowed)
/// so an `Fst` can be constructed *without* forcing a copy when the caller
/// already holds zero-copy, mmap-backed bytes (`MmapDirectory`, exercised by
/// `crates/lucene-store/src/directory.rs`'s own tests).
#[derive(Debug, Clone)]
enum FstBytes<'a> {
    Owned(Vec<u8>),
    Borrowed(&'a [u8]),
}

impl std::ops::Deref for FstBytes<'_> {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        match self {
            FstBytes::Owned(v) => v,
            FstBytes::Borrowed(s) => s,
        }
    }
}

/// A `ByteSequenceOutputs`-typed FST, read from bytes written by real
/// Lucene's `FST.save`.
///
/// `'a` is `'static` for `Fst::read`'s owned path (`Fst<'static>`, i.e.
/// plain `Fst` in the common case) and the lifetime of the caller's backing
/// buffer for `Fst::read_borrowed`'s zero-copy path -- see `FstBytes`'s doc
/// comment above for what "off-heap" honestly means in this port.
#[derive(Debug, Clone)]
pub struct Fst<'a> {
    metadata: FstMetadata,
    bytes: FstBytes<'a>,
}

/// Metadata common to both `Fst::read` (owned) and `Fst::read_borrowed`
/// (zero-copy) -- everything in the wire format up to, but not including,
/// the raw body bytes themselves, which each caller handles differently
/// (copy vs. borrow).
struct ParsedMetadata {
    input_type: InputType,
    empty_output: Option<Vec<u8>>,
    start_node: i64,
    version: i32,
    num_bytes: i64,
}

fn read_fst_metadata_prefix(input: &mut SliceInput) -> Result<ParsedMetadata> {
    let header = codec_util::check_header(input, FILE_FORMAT_NAME, VERSION_START, VERSION_CURRENT)?;
    let version = header.version;

    let empty_output = if input.read_byte()? == 1 {
        let num_bytes = input.read_vint()?;
        if num_bytes < 0 {
            return Err(Error::Corrupt(format!(
                "negative FST empty-output length {num_bytes}"
            )));
        }
        // The length is a `vint` off disk sizing a `Vec`: unbounded it is a
        // 2 GB reservation the process aborts on rather than an error the FFI
        // can report. Bound it by what is actually left in the file first.
        let remaining = input.len().saturating_sub(input.position());
        let num_bytes = num_bytes as usize;
        if num_bytes > remaining {
            return Err(Error::Corrupt(format!(
                "FST empty-output length {num_bytes} exceeds the {remaining} bytes left in the file"
            )));
        }
        let mut raw = vec![0u8; num_bytes];
        input.read_bytes(&mut raw)?;
        // `FSTMetadata.save` does *not* write the empty output's raw bytes
        // reversed: it first serializes the value through
        // `outputs.writeFinalOutput` -- for `ByteSequenceOutputs`, a `vint`
        // length followed by the payload -- and reverses *that* whole
        // buffer, so that a reverse `BytesReader` starting at its last byte
        // decodes it with exactly the same `readFinalOutput` call every arc
        // output goes through. So the length prefix is part of the reversed
        // blob and must be consumed, not kept: reversing the buffer and
        // taking it verbatim leaves a spurious leading length byte on every
        // empty output (see
        // `crates/lucene-codecs/tests/fst_empty_key_fixtures.rs`).
        //
        // `num_bytes == 0` is `NoOutputs`, whose `writeFinalOutput` emits
        // nothing at all; Java skips the `setPosition` in that case and its
        // `readFinalOutput` reads no bytes, so the decoded value is empty.
        if num_bytes == 0 {
            Some(Vec::new())
        } else {
            let mut r = BytesReader::new(&raw);
            // ARITH: the `num_bytes == 0` branch above returned already.
            #[allow(clippy::arithmetic_side_effects)]
            r.set_position(num_bytes as i64 - 1);
            Some(r.read_output()?)
        }
    } else {
        None
    };

    let input_type = match input.read_byte()? {
        0 => InputType::Byte1,
        1 => InputType::Byte2,
        2 => InputType::Byte4,
        other => return Err(Error::Corrupt(format!("invalid FST input type {other}"))),
    };
    let start_node = input.read_vlong()?;
    let num_bytes = input.read_vlong()?;
    if num_bytes < 0 {
        return Err(Error::Corrupt(format!("negative FST numBytes {num_bytes}")));
    }

    Ok(ParsedMetadata {
        input_type,
        empty_output,
        start_node,
        version,
        num_bytes,
    })
}

impl Fst<'static> {
    /// Port of `FST.readMetadata` + the `FST(FSTMetadata, DataInput)`
    /// constructor's `OnHeapFSTStore` body read, both operating on the same
    /// forward cursor (matching `FST.save(Path)`/`FST.read(Path, Outputs)`,
    /// which write/read metadata and body through one stream).
    ///
    /// Allocates a fresh `Vec<u8>` and copies the body into it (real
    /// Lucene's `OnHeapFSTStore` semantics). See `Fst::read_borrowed` for the
    /// zero-copy alternative when the caller's `input` is itself backed by
    /// mmap'd (or otherwise already-owned) bytes.
    pub fn read(input: &mut SliceInput) -> Result<Fst<'static>> {
        let meta = read_fst_metadata_prefix(input)?;
        // `numBytes` is a `vlong` off disk, validated non-negative but
        // otherwise unbounded, and it sizes a `Vec` -- so a corrupt FST
        // header reserves up to 2^63 bytes and *aborts* the process, which
        // `catch_unwind` cannot intercept and which therefore takes the JVM
        // with it. `read_borrowed` already bounds the same field through
        // `SliceInput::slice`; this is the owning path's equivalent.
        let remaining = input.len().saturating_sub(input.position());
        if meta.num_bytes as u64 > remaining as u64 {
            return Err(Error::Corrupt(format!(
                "FST numBytes {} exceeds the {remaining} bytes left in the file",
                meta.num_bytes
            )));
        }
        let mut bytes = vec![0u8; meta.num_bytes as usize];
        input.read_bytes(&mut bytes)?;

        Ok(Fst {
            metadata: FstMetadata {
                input_type: meta.input_type,
                empty_output: meta.empty_output,
                start_node: meta.start_node,
                version: meta.version,
                num_bytes: meta.num_bytes,
            },
            bytes: FstBytes::Owned(bytes),
        })
    }
}

impl<'a> Fst<'a> {
    /// Zero-copy equivalent of `Fst::read`: parses the same metadata but,
    /// instead of copying `num_bytes` of body into a new `Vec<u8>`, borrows
    /// them directly out of `input`'s own `'a`-lifetime backing buffer via
    /// `SliceInput::slice` -- no allocation for the body at all.
    ///
    /// This is real Lucene's `OffHeapFSTStore` distinction *iff* `input`
    /// itself is backed by a zero-copy source (an `Input::Mapped` mmap, via
    /// `lucene_store::directory::MmapDirectory` + `SliceInput::new` over its
    /// `Deref<Target = [u8]>`): in that case, the returned `Fst<'a>` never
    /// materializes a second full-size copy of the FST body, matching real
    /// Lucene's mmap'd-FST cost model. If `input` instead already wraps an
    /// owned `Vec<u8>` (e.g. from `Directory::open` on a non-mmap backend),
    /// this still avoids the *extra* copy `Fst::read` would have made, but
    /// obviously doesn't retroactively make the caller's own buffer
    /// OS-mmap'd -- see `FstBytes`'s doc comment for the precise claim.
    pub fn read_borrowed(input: &mut SliceInput<'a>) -> Result<Fst<'a>> {
        let meta = read_fst_metadata_prefix(input)?;
        let start = input.position();
        let end = start
            .checked_add(meta.num_bytes as usize)
            .ok_or_else(|| Error::Corrupt(format!("FST numBytes {} overflows", meta.num_bytes)))?;
        let bytes = input.slice(start, end)?;
        input.seek(end)?;

        Ok(Fst {
            metadata: FstMetadata {
                input_type: meta.input_type,
                empty_output: meta.empty_output,
                start_node: meta.start_node,
                version: meta.version,
                num_bytes: meta.num_bytes,
            },
            bytes: FstBytes::Borrowed(bytes),
        })
    }

    /// `FST.fromFSTReader(FST.readMetadata(metaIn, ..), new
    /// OffHeapFSTStore(indexIn, offset, metadata))`: the metadata read from
    /// one stream, the body borrowed from another file at `offset` --
    /// the split `Lucene90BlockTreeTermsReader` stores a field's term index
    /// in (metadata in `.tmd`, body in `.tip`).
    pub fn read_split(meta: &mut SliceInput, body_file: &'a [u8], offset: u64) -> Result<Fst<'a>> {
        let m = read_fst_metadata_prefix(meta)?;
        let start = usize::try_from(offset)
            .map_err(|_| Error::Corrupt(format!("FST offset {offset} out of range")))?;
        let bytes = start
            .checked_add(m.num_bytes as usize)
            .and_then(|end| body_file.get(start..end))
            .ok_or_else(|| {
                Error::Corrupt(format!(
                    "FST body [{start}, +{}) outside a {} byte file",
                    m.num_bytes,
                    body_file.len()
                ))
            })?;
        Ok(Fst {
            metadata: FstMetadata {
                input_type: m.input_type,
                empty_output: m.empty_output,
                start_node: m.start_node,
                version: m.version,
                num_bytes: m.num_bytes,
            },
            bytes: FstBytes::Borrowed(bytes),
        })
    }

    /// An FST over metadata already read and a body already located -- what
    /// `Lucene90BlockTreeTermsReader`'s `FieldReader` holds after its
    /// constructor, rebuilt per walk by
    /// the retired terms index's `FstTermsIndex` (a struct copy: the
    /// metadata it keeps has no empty output to clone).
    pub(crate) fn from_parts(metadata: FstMetadata, body: &'a [u8]) -> Fst<'a> {
        Fst {
            metadata,
            bytes: FstBytes::Borrowed(body),
        }
    }

    /// `FST.findTargetArc(label, follow, arc, in)` for one byte label over a
    /// fresh reader: the arc leaving `follow`'s target node with `label`,
    /// if any.
    pub(crate) fn find_target_arc_byte(&self, label: u8, follow: &Arc) -> Result<Option<Arc>> {
        let mut r = self.reader();
        self.find_target_arc(i32::from(label), follow, &mut r)
    }

    pub fn metadata(&self) -> &FstMetadata {
        &self.metadata
    }

    /// `true` if this `Fst`'s body is borrowed (zero-copy) rather than owned
    /// by this struct -- exposed for tests/diagnostics; not needed for
    /// lookup itself, which goes through `reader()`/`Deref` either way.
    pub fn is_borrowed(&self) -> bool {
        matches!(self.bytes, FstBytes::Borrowed(_))
    }

    fn reader(&self) -> BytesReader<'_> {
        BytesReader::new(&self.bytes)
    }

    /// `FST.readLabel`: decode one arc label per `metadata.input_type`.
    /// `BYTE1` (raw term bytes, used by the BlockTree term index) is an
    /// unsigned byte; `BYTE2` (UTF-16 code units, used by some suggesters) is
    /// an unsigned 16-bit value -- little-endian from `VERSION_LITTLE_ENDIAN`
    /// on, byte-swapped before that (matches `Short.reverseBytes` in
    /// `FST.java`'s own `metadata.version < VERSION_LITTLE_ENDIAN` branch);
    /// `BYTE4` (full Unicode code points) is a `vint`, *not* a fixed 4-byte
    /// value -- real Lucene picked a variable-length encoding there since
    /// most code points fit in 1-2 bytes despite the "4" in the name
    /// referring to the logical label width, not its wire encoding.
    fn read_label(&self, r: &mut BytesReader) -> Result<i32> {
        match self.metadata.input_type {
            InputType::Byte1 => Ok(r.read_byte()? as i32),
            InputType::Byte2 => {
                // `DataInput.readShort`: `b1` = first byte read, `b2` = second;
                // value = `(b2 << 8) | b1` (little-endian), regardless of
                // whether the underlying reader walks forward or backward --
                // see `BytesReader::read_vint`'s identical observation.
                let b1 = r.read_byte()?;
                let b2 = r.read_byte()?;
                let le = ((b2 as u16) << 8) | (b1 as u16);
                let v = if self.metadata.version < VERSION_LITTLE_ENDIAN {
                    le.swap_bytes()
                } else {
                    le
                };
                Ok(v as i32)
            }
            InputType::Byte4 => {
                // `FST.readLabel`'s BYTE4 branch is a bare `readVInt()`, which
                // decodes to any `i32`. Every label a `FSTCompiler` writes is
                // non-negative (labels come from an `IntsRef` of code points
                // or byte values, and `END_LABEL == -1` is synthetic and never
                // stored), so rejecting a negative one cannot reject a file
                // real Lucene wrote -- and it is what bounds
                // `first_label + arc_idx` and `target_label - first_label`
                // below to an `i32`.
                let label = r.read_vint()?;
                if label < 0 {
                    return Err(Error::Corrupt(format!("negative FST arc label {label}")));
                }
                Ok(label)
            }
        }
    }

    /// Reads and validates an array-node header's `numArcs`/`bytesPerArc`
    /// pair (`FST.readArcByIndex` and friends read both as plain `vint`s and
    /// then trust them).
    ///
    /// Both are `vint`s off disk, so both can arrive negative or absurd.
    /// `FSTCompiler` never emits an array node with fewer than one arc or a
    /// zero-width arc slot -- `ARCS_FOR_BINARY_SEARCH` needs at least
    /// `FIXED_LENGTH_ARC_SHALLOW_NUM_ARCS` arcs to be chosen at all -- so
    /// this rejects nothing a real writer produces, and it is the single
    /// bound that makes `numArcs - 1`, `numArcs - 2`, `firstLabel + arcIdx`
    /// and `getNumPresenceBytes(numArcs)` provably in range downstream.
    fn read_array_node_header(&self, r: &mut BytesReader) -> Result<(i32, i32)> {
        let num_arcs = r.read_vint()?;
        let bytes_per_arc = r.read_vint()?;
        if num_arcs < 1 || bytes_per_arc < 1 {
            return Err(Error::Corrupt(format!(
                "FST array node with numArcs={num_arcs}, bytesPerArc={bytes_per_arc}"
            )));
        }
        Ok((num_arcs, bytes_per_arc))
    }

    /// The fixed-size arc array of an `ARCS_FOR_BINARY_SEARCH` or
    /// `ARCS_FOR_CONTINUOUS` node occupies `bytesPerArc * numArcs` bytes
    /// ending at `pos_arcs_start`, and the reverse cursor walks it downward
    /// -- so it must physically fit below that position. (Direct addressing
    /// is deliberately excluded: its array holds only the *present* arcs,
    /// which is fewer than the `numArcs` label range. Its own bound is the
    /// presence bit-table's, applied in `read_presence_bytes`.)
    fn check_arc_array_fits(num_arcs: i32, bytes_per_arc: i32, pos_arcs_start: i64) -> Result<()> {
        // ARITH: both factors are validated `1..=i32::MAX` by
        // `read_array_node_header`, so the `i64` product is below 2^62;
        // `pos_arcs_start` is a cursor position below `bytes.len()`.
        #[allow(clippy::arithmetic_side_effects)]
        let (span, limit) = (
            bytes_per_arc as i64 * num_arcs as i64,
            pos_arcs_start.saturating_add(1),
        );
        if span > limit {
            return Err(Error::Corrupt(format!(
                "FST array node of {num_arcs} x {bytes_per_arc} bytes does not fit below \
                 position {pos_arcs_start}"
            )));
        }
        Ok(())
    }

    /// The label range of a direct-addressing or continuous node is
    /// `[firstLabel, firstLabel + numArcs)`, and those nodes do not store
    /// their labels at all -- `read_arc` derives each one as
    /// `firstLabel + arcIdx`. So the range must fit the alphabet the
    /// `INPUT_TYPE` declares: a `BYTE1` FST's labels are single bytes, a
    /// `BYTE2` FST's unsigned shorts. A real writer cannot emit a wider
    /// range (there are no such labels to put in it), and this bound plus
    /// `bit_table_next_bit_set`'s own is what makes `labels_to_bytes`'
    /// narrowing a proof rather than a runtime check on every emitted key.
    fn check_label_range(&self, first_label: i32, num_arcs: i32) -> Result<()> {
        // ARITH: `first_label >= 0` (`read_label`) and `num_arcs >= 1`
        // (`read_array_node_header`), both `i32`, so the `i64` sum is exact.
        #[allow(clippy::arithmetic_side_effects)]
        let end = first_label as i64 + num_arcs as i64;
        let limit = match self.metadata.input_type {
            InputType::Byte1 => 256,
            InputType::Byte2 => 65_536,
            InputType::Byte4 => i32::MAX as i64,
        };
        if end > limit {
            return Err(Error::Corrupt(format!(
                "FST label range [{first_label}, +{num_arcs}) is outside the {:?} alphabet",
                self.metadata.input_type
            )));
        }
        Ok(())
    }

    /// `FST.getFirstArc`: the virtual incoming arc to the start node.
    pub fn first_arc(&self) -> Arc {
        let (flags, next_final_output) = match &self.metadata.empty_output {
            Some(out) if !out.is_empty() => (
                BIT_FINAL_ARC | BIT_LAST_ARC | BIT_ARC_HAS_FINAL_OUTPUT,
                out.clone(),
            ),
            Some(_) => (BIT_FINAL_ARC | BIT_LAST_ARC, Vec::new()),
            None => (BIT_LAST_ARC, Vec::new()),
        };
        Arc {
            target: self.metadata.start_node,
            flags,
            next_final_output,
            ..Default::default()
        }
    }

    /// `FST.readArc`: decode one arc, assuming the flags byte has already
    /// been read into `arc.flags` and `r` is positioned right after it.
    fn read_arc(&self, arc: &mut Arc, r: &mut BytesReader) -> Result<()> {
        arc.label = if arc.node_flags == ARCS_FOR_DIRECT_ADDRESSING
            || arc.node_flags == ARCS_FOR_CONTINUOUS
        {
            // Direct-addressing and continuous nodes never store the label
            // explicitly -- it's implied by the arc's position in the label
            // range (`FST.readArc`'s `arc.label = arc.firstLabel() +
            // arc.arcIdx()` branch, shared by both encodings).
            // ARITH: `check_label_range` bounded `first_label + num_arcs` to
            // the `INPUT_TYPE`'s alphabet (at most `i32::MAX`) when this
            // node's header was read, and `arc_idx` is an index into
            // `0..num_arcs` -- `read_next_real_arc` range-checks it and
            // `bit_table_next_bit_set` refuses an index past `num_arcs`. So
            // the sum is at most `first_label + num_arcs - 1`.
            #[allow(clippy::arithmetic_side_effects)]
            let implied = arc.first_label + arc.arc_idx;
            implied
        } else {
            self.read_label(r)?
        };

        arc.output = if flag(arc.flags, BIT_ARC_HAS_OUTPUT) {
            r.read_output()?
        } else {
            Vec::new()
        };

        arc.next_final_output = if flag(arc.flags, BIT_ARC_HAS_FINAL_OUTPUT) {
            r.read_output()?
        } else {
            Vec::new()
        };

        if flag(arc.flags, BIT_STOP_NODE) {
            arc.target = if flag(arc.flags, BIT_FINAL_ARC) {
                FINAL_END_NODE
            } else {
                NON_FINAL_END_NODE
            };
            arc.next_arc = r.get_position();
        } else if flag(arc.flags, BIT_TARGET_NEXT) {
            arc.next_arc = r.get_position();
            if !flag(arc.flags, BIT_LAST_ARC) {
                if arc.bytes_per_arc == 0 {
                    // List-encoded node: must scan past the remaining
                    // sibling arcs to find this arc's implicit target.
                    self.seek_to_next_node(r)?;
                } else {
                    // Fixed-length-arc node (`ARCS_FOR_BINARY_SEARCH` or
                    // `ARCS_FOR_DIRECT_ADDRESSING`): the target is simply the
                    // position right before the fixed arcs array starts
                    // (`FST.readArc`'s `bytesPerArc() != 0` branch) -- no scan
                    // needed since every arc's on-disk size is known. For
                    // direct addressing, the array only holds *present* arcs,
                    // so its size is the bit-table's set-bit count, not the
                    // label range (`num_arcs`).
                    let num_arcs = if arc.node_flags == ARCS_FOR_DIRECT_ADDRESSING {
                        bit_table_count_bits(arc, r)?
                    } else {
                        arc.num_arcs
                    };
                    // ARITH: `bytes_per_arc` and `num_arcs` are both
                    // validated `1..=i32::MAX` (`read_array_node_header`), so
                    // the `i64` product is below 2^62 and the difference from
                    // a cursor position cannot overflow.
                    #[allow(clippy::arithmetic_side_effects)]
                    let arcs_end = arc.pos_arcs_start - arc.bytes_per_arc as i64 * num_arcs as i64;
                    r.set_position(arcs_end);
                }
            }
            arc.target = r.get_position();
        } else {
            arc.target = r.read_vlong()?;
            arc.next_arc = r.get_position();
        }
        Ok(())
    }

    /// `FST.seekToNextNode`: skip arcs (list encoding only) until the last
    /// arc of the current node has been consumed.
    fn seek_to_next_node(&self, r: &mut BytesReader) -> Result<()> {
        loop {
            let flags = r.read_byte()?;
            self.read_label(r)?;
            if flag(flags, BIT_ARC_HAS_OUTPUT) {
                r.skip_output()?;
            }
            if flag(flags, BIT_ARC_HAS_FINAL_OUTPUT) {
                r.skip_output()?;
            }
            if !flag(flags, BIT_STOP_NODE) && !flag(flags, BIT_TARGET_NEXT) {
                r.read_vlong()?;
            }
            if flag(flags, BIT_LAST_ARC) {
                return Ok(());
            }
        }
    }

    /// `FST.readPresenceBytes`: record the bit-table's start position and
    /// skip past its `getNumPresenceBytes(arc.numArcs)` bytes (the presence
    /// bits themselves are read lazily, on demand, via the `bit_table_*`
    /// helpers -- matching real Lucene's own "don't read them here, just
    /// skip them" comment).
    fn read_presence_bytes(&self, arc: &mut Arc, r: &mut BytesReader) -> Result<()> {
        debug_assert!(arc.bytes_per_arc > 0);
        debug_assert_eq!(arc.node_flags, ARCS_FOR_DIRECT_ADDRESSING);
        arc.bit_table_start = r.get_position();
        let table_bytes = num_presence_bytes(arc.num_arcs) as i64;
        // The presence bit-table is read downward from `bit_table_start`, so
        // it must physically fit below it. This is direct addressing's
        // counterpart to `check_arc_array_fits`, and it is what bounds
        // `num_arcs` (a *label range*, not an arc count) for every
        // `bit_table_*` helper.
        if table_bytes > arc.bit_table_start.saturating_add(1) {
            return Err(Error::Corrupt(format!(
                "FST presence table of {table_bytes} bytes does not fit below position {}",
                arc.bit_table_start
            )));
        }
        r.skip_bytes(table_bytes);
        Ok(())
    }

    /// `FST.readArcByDirectAddressing(Arc, BytesReader, int, int)`: seek to
    /// the arc slot for the (already known-present) `range_index`/
    /// `presence_index` pair and decode it via the shared `read_arc`.
    fn read_arc_by_direct_addressing(
        &self,
        arc: &mut Arc,
        r: &mut BytesReader,
        range_index: i32,
        presence_index: i32,
    ) -> Result<()> {
        // ARITH: `presence_index` is a set-bit count in a bit table bounded by
        // the body, minus at most one, and every caller has established it is
        // non-negative (`read_next_real_arc` and
        // `read_last_arc_by_direct_addressing` reject the `-1` sentinel;
        // `bit_table_count_bits_up_to` cannot return a negative). With
        // `bytes_per_arc` validated `1..=i32::MAX` the `i64` product is below
        // 2^62, and it is subtracted from `pos_arcs_start` rather than added
        // to it -- an unrejected `-1` would address *above* the arc array,
        // which `read_byte` would not catch.
        #[allow(clippy::arithmetic_side_effects)]
        let slot = arc.pos_arcs_start - presence_index as i64 * arc.bytes_per_arc as i64;
        r.set_position(slot);
        arc.arc_idx = range_index;
        arc.presence_index = presence_index;
        arc.flags = r.read_byte()?;
        self.read_arc(arc, r)
    }

    /// `FST.findTargetArc`'s `ARCS_FOR_BINARY_SEARCH` branch: `follow`'s
    /// target node has already been confirmed (by the caller) to be a fixed-
    /// length-arc, binary-searchable node, with `r` positioned right after
    /// the node header's `flags` byte. Ported field-for-field from
    /// `FST.java`'s `findTargetArc`: `numArcs`/`bytesPerArc` (both `vint`),
    /// then a sparse binary search over `numArcs` fixed-size arc slots, each
    /// addressed as `posArcsStart - bytesPerArc * idx` (slot `idx`'s flags
    /// byte sits one byte after that address, mirroring the `+1`/`-1`
    /// asymmetry in `FST.java`'s own address arithmetic between the
    /// mid-search label peek and the final `readNextRealArc` call).
    fn find_target_arc_binary_search(
        &self,
        label_to_match: i32,
        r: &mut BytesReader,
    ) -> Result<Option<Arc>> {
        let (num_arcs, bytes_per_arc) = self.read_array_node_header(r)?;
        let pos_arcs_start = r.get_position();
        Self::check_arc_array_fits(num_arcs, bytes_per_arc, pos_arcs_start)?;

        let mut low = 0i32;
        // ARITH: `num_arcs >= 1` (`read_array_node_header`), so `high >= 0`;
        // `mid` is in `low..=high`, so `mid + 1` and `mid - 1` stay inside
        // `-1..=num_arcs` and the `i64` slot product is below 2^62.
        #[allow(clippy::arithmetic_side_effects)]
        let mut high = num_arcs - 1;
        while low <= high {
            // `FST.findTargetArc` is `(low + high) >>> 1` -- an *unsigned*
            // shift, because `low + high` overflows an `int` once `numArcs`
            // (off disk, so unbounded) is near `Integer.MAX_VALUE`. A signed
            // `>>` there yields a negative midpoint and searches the wrong
            // side of the array; the `u32` round-trip below is Java's shift
            // bit for bit.
            let mid = (low.wrapping_add(high) as u32 >> 1) as i32;
            // +1 to skip over the mid arc's flags byte, matching
            // `FST.java`'s `posArcsStart - (bytesPerArc * mid + 1)`.
            // ARITH: `bytes_per_arc` and `mid` are `i32`s widened to `i64`, so the
            // product is below 2^62 and the `+ 1` and the subtraction from a
            // cursor position cannot overflow.
            #[allow(clippy::arithmetic_side_effects)]
            let mid_pos = pos_arcs_start - (bytes_per_arc as i64 * mid as i64 + 1);
            r.set_position(mid_pos);
            let mid_label = self.read_label(r)?;
            // ARITH: `mid` is in `low..=high` with `high <= num_arcs - 1`, so
            // `mid + 1` and `mid - 1` stay inside `-1..=num_arcs`.
            #[allow(clippy::arithmetic_side_effects)]
            match mid_label.cmp(&label_to_match) {
                std::cmp::Ordering::Less => low = mid + 1,
                std::cmp::Ordering::Greater => high = mid - 1,
                std::cmp::Ordering::Equal => {
                    // `arc_idx` must end up at `mid`: it is what
                    // `read_next_real_arc`/`read_next_arc_label` use to find
                    // the *following* slot, so a `find_target_arc` hit that
                    // left it at its default would make any subsequent
                    // enumeration from this arc (e.g. `FstEnum::seek_exact`
                    // then `next()`) resume from the wrong slot. Setting it
                    // to `mid - 1` and going through `read_next_real_arc` is
                    // `FST.findTargetArc`'s own `arc.arcIdx = mid - 1;
                    // return readNextRealArc(arc, in);`, which does the
                    // increment, the seek and the flags read in one place.
                    let mut arc = Arc {
                        node_flags: ARCS_FOR_BINARY_SEARCH,
                        bytes_per_arc,
                        num_arcs,
                        pos_arcs_start,
                        arc_idx: mid - 1,
                        ..Default::default()
                    };
                    self.read_next_real_arc(&mut arc, r)?;
                    return Ok(Some(arc));
                }
            }
        }
        Ok(None)
    }

    /// `FST.findTargetArc`'s `ARCS_FOR_DIRECT_ADDRESSING` branch: `follow`'s
    /// target node has already been confirmed (by the caller) to be a direct-
    /// addressing node, with `r` positioned right after the node header's
    /// `flags` byte. Ported field-for-field from `FST.java`: label range
    /// (`numArcs`, despite the name) + `bytesPerArc` (both `vint`), the
    /// presence bit-table (skipped, not read yet), `firstLabel`, then a
    /// direct `label_to_match - first_label` index into the label range --
    /// rejected outright if out of range or (if in range) not actually
    /// present in the bit-table, otherwise decoded via
    /// `read_arc_by_direct_addressing` at its bit-table-derived presence
    /// index.
    fn find_target_arc_direct_addressing(
        &self,
        label_to_match: i32,
        r: &mut BytesReader,
    ) -> Result<Option<Arc>> {
        // `num_arcs` is the label range here, not an arc count.
        let (num_arcs, bytes_per_arc) = self.read_array_node_header(r)?;
        let mut arc = Arc {
            node_flags: ARCS_FOR_DIRECT_ADDRESSING,
            num_arcs,
            bytes_per_arc,
            ..Default::default()
        };
        self.read_presence_bytes(&mut arc, r)?;
        arc.first_label = self.read_label(r)?;
        self.check_label_range(arc.first_label, num_arcs)?;
        arc.pos_arcs_start = r.get_position();

        // ARITH: `label_to_match` is a key label (`0..=i32::MAX`) or
        // `END_LABEL` (-1), and `first_label` is non-negative
        // (`read_label`), so the difference is in
        // `i32::MIN..=i32::MAX` -- `-1 - i32::MAX` is exactly `i32::MIN`.
        #[allow(clippy::arithmetic_side_effects)]
        let arc_index = label_to_match - arc.first_label;
        if arc_index < 0 || arc_index >= arc.num_arcs {
            return Ok(None); // Before or after the label range.
        }
        if !bit_table_is_bit_set(arc_index, &arc, r)? {
            return Ok(None); // Arc missing in the range.
        }
        let presence_index = bit_table_count_bits_up_to(arc_index, &arc, r)?;
        self.read_arc_by_direct_addressing(&mut arc, r, arc_index, presence_index)?;
        Ok(Some(arc))
    }

    /// `FST.findTargetArc`'s `ARCS_FOR_CONTINUOUS` branch: `follow`'s target
    /// node has already been confirmed (by the caller) to be a continuous
    /// node, with `r` positioned right after the node header's `flags` byte.
    /// Simpler than `ARCS_FOR_DIRECT_ADDRESSING`'s equivalent since every
    /// label in `[firstLabel, firstLabel + numArcs)` is guaranteed present --
    /// no presence bit-table at all, so a label range hit always resolves to
    /// a real arc. Ported field-for-field from `FST.java`'s `findTargetArc`:
    /// `numArcs`/`bytesPerArc` (both `vint`), then `firstLabel`, then a direct
    /// `label_to_match - first_label` index into the range, rejected if out
    /// of range, otherwise decoded via `read_next_real_arc` from `arc_idx =
    /// arc_index - 1` (mirroring `FST.java`'s own `arc.arcIdx = arcIndex - 1;
    /// return readNextRealArc(arc, in);` -- `read_next_real_arc`'s shared
    /// `bytes_per_arc != 0` branch increments `arc_idx` back to `arc_index`
    /// and seeks to that slot, reusing the exact arithmetic
    /// `ARCS_FOR_BINARY_SEARCH` already established).
    fn find_target_arc_continuous(
        &self,
        label_to_match: i32,
        r: &mut BytesReader,
    ) -> Result<Option<Arc>> {
        let (num_arcs, bytes_per_arc) = self.read_array_node_header(r)?;
        let first_label = self.read_label(r)?;
        self.check_label_range(first_label, num_arcs)?;
        let pos_arcs_start = r.get_position();
        Self::check_arc_array_fits(num_arcs, bytes_per_arc, pos_arcs_start)?;

        // ARITH: as in the direct-addressing branch above.
        #[allow(clippy::arithmetic_side_effects)]
        let arc_index = label_to_match - first_label;
        if arc_index < 0 || arc_index >= num_arcs {
            return Ok(None); // Before or after the label range.
        }

        // ARITH: `arc_index >= 0` was just established.
        #[allow(clippy::arithmetic_side_effects)]
        let start_idx = arc_index - 1;
        let mut arc = Arc {
            node_flags: ARCS_FOR_CONTINUOUS,
            num_arcs,
            bytes_per_arc,
            first_label,
            pos_arcs_start,
            arc_idx: start_idx,
            ..Default::default()
        };
        self.read_next_real_arc(&mut arc, r)?;
        Ok(Some(arc))
    }

    /// `FST.findTargetArc`: find the arc leaving `follow` labeled
    /// `label_to_match`, or `None` if there is none. List-encoded,
    /// `ARCS_FOR_BINARY_SEARCH`, `ARCS_FOR_DIRECT_ADDRESSING` and
    /// `ARCS_FOR_CONTINUOUS` nodes.
    ///
    /// `label_to_match == END_LABEL` is also supported (needed by
    /// `FstEnum::do_seek_exact`, which -- like real `FSTEnum.doSeekExact` --
    /// re-derives the "fake" accepting arc this way once the target key's
    /// bytes are exhausted): re-synthesizes the same synthetic arc
    /// `read_first_target_arc` would insert, without touching `r` at all.
    /// `Fst::get_labels`'s loop never does this (it calls this once per
    /// actual key label and then checks `arc.is_final()` directly instead),
    /// so for that caller `label_to_match` is always a real, non-negative
    /// label -- a raw byte (0..=255) for `BYTE1`, a UTF-16 code unit
    /// (0..=0xFFFF) for `BYTE2`, or a Unicode code point (0..=0x10FFFF) for
    /// `BYTE4`.
    fn find_target_arc(
        &self,
        label_to_match: i32,
        follow: &Arc,
        r: &mut BytesReader,
    ) -> Result<Option<Arc>> {
        if label_to_match == END_LABEL {
            return Ok(if follow.is_final() {
                let mut arc = Arc {
                    label: END_LABEL,
                    output: follow.next_final_output().to_vec(),
                    ..Default::default()
                };
                if follow.target() <= 0 {
                    arc.flags = BIT_LAST_ARC;
                } else {
                    // NOTE: next_arc is a node (not an address!) in this case.
                    arc.flags = 0;
                    arc.next_arc = follow.target();
                }
                arc.node_flags = arc.flags;
                Some(arc)
            } else {
                None
            });
        }
        debug_assert!(label_to_match >= 0);
        if !target_has_arcs(follow) {
            return Ok(None);
        }

        r.set_position(follow.target());
        let flags = r.read_byte()?;
        if flags == ARCS_FOR_BINARY_SEARCH {
            return self.find_target_arc_binary_search(label_to_match, r);
        }
        if flags == ARCS_FOR_DIRECT_ADDRESSING {
            return self.find_target_arc_direct_addressing(label_to_match, r);
        }
        if flags == ARCS_FOR_CONTINUOUS {
            return self.find_target_arc_continuous(label_to_match, r);
        }

        // Linear scan (list-encoded nodes).
        let mut arc = Arc {
            flags,
            ..Default::default()
        };
        loop {
            let flags = arc.flags;
            let pos = r.get_position();
            let label = self.read_label(r)?;
            if label == label_to_match {
                r.set_position(pos);
                self.read_arc(&mut arc, r)?;
                return Ok(Some(arc));
            } else if label > label_to_match || flag(flags, BIT_LAST_ARC) {
                // Either past the label in sorted order, or this was the
                // last arc of the node -- either way, no match.
                return Ok(None);
            } else {
                if flag(flags, BIT_ARC_HAS_OUTPUT) {
                    r.skip_output()?;
                }
                if flag(flags, BIT_ARC_HAS_FINAL_OUTPUT) {
                    r.skip_output()?;
                }
                if !flag(flags, BIT_STOP_NODE) && !flag(flags, BIT_TARGET_NEXT) {
                    r.read_vlong()?;
                }
                arc.flags = r.read_byte()?;
            }
        }
    }

    // --- Seek support helpers (`FSTEnum`'s array/direct/continuous "read
    // arc N", "read last arc" and "peek next label" primitives) -----------
    //
    // These are additional per-arc random-access primitives `find_target_arc`
    // and the ordered-enumeration methods above never needed (they either
    // jump straight to one matched arc and stop, or only ever advance
    // forward one arc at a time): reading an arbitrary arc by index within a
    // fixed-length-arc node, reading a node's *last* arc directly (without
    // scanning through every arc first), and peeking at the label of the
    // arc following the current one without mutating it. `FstEnum`'s
    // `seek_ceil`/`seek_floor`/`seek_exact` (below) are built on top of
    // these plus the existing `find_target_arc`/`read_first_real_target_arc`/
    // `read_next_real_arc`.

    /// `FST.readArcByIndex`: read the arc at (0-based) slot `idx` of an
    /// `ARCS_FOR_BINARY_SEARCH` node -- `arc` must already carry that node's
    /// header fields (`pos_arcs_start`/`bytes_per_arc`/`node_flags`), as set
    /// up by `find_target_arc_binary_search`/`read_first_real_target_arc`/
    /// `find_arc_binary_search`.
    fn read_arc_by_index(&self, arc: &mut Arc, r: &mut BytesReader, idx: i32) -> Result<()> {
        debug_assert!(arc.bytes_per_arc > 0);
        debug_assert_eq!(arc.node_flags, ARCS_FOR_BINARY_SEARCH);
        debug_assert!(idx >= 0 && idx < arc.num_arcs);
        // ARITH: two `i32`s widened to `i64`; the product is below 2^62.
        #[allow(clippy::arithmetic_side_effects)]
        let slot = arc.pos_arcs_start - idx as i64 * arc.bytes_per_arc as i64;
        r.set_position(slot);
        arc.arc_idx = idx;
        arc.flags = r.read_byte()?;
        self.read_arc(arc, r)
    }

    /// `FST.readArcByContinuous`: read the arc at (0-based) range-index
    /// `range_index` of an `ARCS_FOR_CONTINUOUS` node.
    fn read_arc_by_continuous(
        &self,
        arc: &mut Arc,
        r: &mut BytesReader,
        range_index: i32,
    ) -> Result<()> {
        debug_assert!(range_index >= 0 && range_index < arc.num_arcs);
        // ARITH: two `i32`s widened to `i64`; the product is below 2^62.
        #[allow(clippy::arithmetic_side_effects)]
        let slot = arc.pos_arcs_start - range_index as i64 * arc.bytes_per_arc as i64;
        r.set_position(slot);
        arc.arc_idx = range_index;
        arc.flags = r.read_byte()?;
        self.read_arc(arc, r)
    }

    /// `FST.readLastArcByDirectAddressing`: read an `ARCS_FOR_DIRECT_ADDRESSING`
    /// node's last (highest-labeled) present arc directly, without scanning
    /// through the whole bit-table arc by arc.
    fn read_last_arc_by_direct_addressing(&self, arc: &mut Arc, r: &mut BytesReader) -> Result<()> {
        // An all-zero bit table makes Java's
        // `BitTable.countBits(arc, in) - 1` produce `-1`, and a `-1`
        // `presence_index` addresses `pos_arcs_start + bytes_per_arc` -- which
        // is *above* the arc array's start and normally well inside the body,
        // so `read_byte` accepts it and decodes a garbage arc. It is the same
        // out-of-domain-sentinel shape as `bit_table_next_bit_set`'s `-1`, so
        // it is rejected here rather than propagated:
        // `FSTCompiler.writeNodeForDirectAddressing` only emits a
        // direct-addressing node for a run of arcs it has already written, and
        // every one of them sets its presence bit, so a real file never has a
        // node with no present arc at all.
        let present = bit_table_count_bits(arc, r)?;
        if present < 1 {
            return corrupt_fst("a direct-addressing node whose bit table has no set bit");
        }
        // ARITH: `present >= 1` was just established and `num_arcs >= 1` comes
        // from `read_array_node_header`, so neither subtraction underflows.
        #[allow(clippy::arithmetic_side_effects)]
        let (presence_index, range_index) = (present - 1, arc.num_arcs - 1);
        self.read_arc_by_direct_addressing(arc, r, range_index, presence_index)
    }

    /// `FST.readLastArcByContinuous`: read an `ARCS_FOR_CONTINUOUS` node's
    /// last arc.
    fn read_last_arc_by_continuous(&self, arc: &mut Arc, r: &mut BytesReader) -> Result<()> {
        // ARITH: `num_arcs >= 1` (`read_array_node_header`).
        #[allow(clippy::arithmetic_side_effects)]
        let range_index = arc.num_arcs - 1;
        self.read_arc_by_continuous(arc, r, range_index)
    }

    /// `FST.readLastTargetArc`: follow `follow` and read the *last* arc of
    /// its target node (as opposed to `read_first_real_target_arc`'s first
    /// arc) -- needed by `FstEnum::push_last`, `seek_floor`'s "beyond the
    /// last arc of this range" cases.
    fn read_last_target_arc(&self, follow: &Arc, r: &mut BytesReader) -> Result<Arc> {
        if !target_has_arcs(follow) {
            if !follow.is_final() {
                return corrupt_fst("an arc with no target that is not final");
            }
            return Ok(Arc {
                label: END_LABEL,
                target: FINAL_END_NODE,
                output: follow.next_final_output().to_vec(),
                flags: BIT_LAST_ARC,
                node_flags: BIT_LAST_ARC,
                ..Default::default()
            });
        }

        r.set_position(follow.target());
        let flags = r.read_byte()?;
        let mut arc = Arc {
            node_flags: flags,
            ..Default::default()
        };
        if flags == ARCS_FOR_BINARY_SEARCH
            || flags == ARCS_FOR_DIRECT_ADDRESSING
            || flags == ARCS_FOR_CONTINUOUS
        {
            let (num_arcs, bytes_per_arc) = self.read_array_node_header(r)?;
            arc.num_arcs = num_arcs;
            arc.bytes_per_arc = bytes_per_arc;
            if flags == ARCS_FOR_DIRECT_ADDRESSING {
                self.read_presence_bytes(&mut arc, r)?;
                arc.first_label = self.read_label(r)?;
                self.check_label_range(arc.first_label, num_arcs)?;
                arc.pos_arcs_start = r.get_position();
                self.read_last_arc_by_direct_addressing(&mut arc, r)?;
            } else if flags == ARCS_FOR_BINARY_SEARCH {
                // ARITH: `num_arcs >= 1`, so this is at worst -1 -- which is
                // what `read_next_real_arc`'s `arc_idx += 1` then turns into
                // slot 0, exactly as `FST.readLastTargetArc` does.
                #[allow(clippy::arithmetic_side_effects)]
                {
                    arc.arc_idx = arc.num_arcs - 2;
                }
                arc.pos_arcs_start = r.get_position();
                Self::check_arc_array_fits(num_arcs, bytes_per_arc, arc.pos_arcs_start)?;
                self.read_next_real_arc(&mut arc, r)?;
            } else {
                arc.first_label = self.read_label(r)?;
                self.check_label_range(arc.first_label, num_arcs)?;
                arc.pos_arcs_start = r.get_position();
                Self::check_arc_array_fits(num_arcs, bytes_per_arc, arc.pos_arcs_start)?;
                self.read_last_arc_by_continuous(&mut arc, r)?;
            }
        } else {
            arc.flags = flags;
            arc.bytes_per_arc = 0;
            while !flag(arc.flags, BIT_LAST_ARC) {
                self.read_label(r)?;
                if flag(arc.flags, BIT_ARC_HAS_OUTPUT) {
                    r.skip_output()?;
                }
                if flag(arc.flags, BIT_ARC_HAS_FINAL_OUTPUT) {
                    r.skip_output()?;
                }
                if !flag(arc.flags, BIT_STOP_NODE) && !flag(arc.flags, BIT_TARGET_NEXT) {
                    r.read_vlong()?;
                }
                arc.flags = r.read_byte()?;
            }
            // Undo the flags byte just read.
            r.skip_bytes(-1);
            arc.next_arc = r.get_position();
            self.read_next_real_arc(&mut arc, r)?;
        }
        if !arc.is_last() {
            return corrupt_fst("a node whose last arc is not flagged BIT_LAST_ARC");
        }
        Ok(arc)
    }

    /// `FST.readNextArcLabel`: peek at the label of the arc immediately
    /// following `arc` (which must not be the node's last arc) without
    /// mutating `arc` itself.
    fn read_next_arc_label(&self, arc: &Arc, r: &mut BytesReader) -> Result<i32> {
        if arc.is_last() {
            return corrupt_fst("no arc follows the node's last arc");
        }
        if arc.label == END_LABEL {
            r.set_position(arc.next_arc);
            let flags = r.read_byte()?;
            if flags == ARCS_FOR_BINARY_SEARCH
                || flags == ARCS_FOR_DIRECT_ADDRESSING
                || flags == ARCS_FOR_CONTINUOUS
            {
                let (num_arcs, _bytes_per_arc) = self.read_array_node_header(r)?;
                if flags == ARCS_FOR_BINARY_SEARCH {
                    r.read_byte()?; // Skip the arc's own flags byte.
                } else if flags == ARCS_FOR_DIRECT_ADDRESSING {
                    r.skip_bytes(num_presence_bytes(num_arcs) as i64);
                }
            }
        } else {
            match arc.node_flags {
                ARCS_FOR_BINARY_SEARCH => {
                    // ARITH: `arc_idx` is an arc index and `bytes_per_arc` is
                    // validated `1..=i32::MAX`, so `(1 + arc_idx)` fits an
                    // `i64` and the product is below 2^62.
                    #[allow(clippy::arithmetic_side_effects)]
                    let next_slot = arc.pos_arcs_start
                        - (1 + arc.arc_idx as i64) * arc.bytes_per_arc as i64
                        - 1;
                    r.set_position(next_slot);
                }
                ARCS_FOR_DIRECT_ADDRESSING => {
                    let next_index = bit_table_next_bit_set(arc.arc_idx, arc, r)?;
                    if next_index == -1 {
                        return corrupt_fst("no present arc follows this one in the bit table");
                    }
                    // ARITH: `check_label_range` bounded
                    // `first_label + num_arcs` to the `INPUT_TYPE`'s alphabet,
                    // and `bit_table_next_bit_set` returns an index below
                    // `num_arcs` or -1 (rejected above).
                    #[allow(clippy::arithmetic_side_effects)]
                    return Ok(arc.first_label + next_index);
                }
                ARCS_FOR_CONTINUOUS => {
                    // ARITH: as above, and this arm is only reached for an arc
                    // that is not its node's last (`read_next_arc_label`'s own
                    // precondition), so `arc_idx + 1 < num_arcs`.
                    #[allow(clippy::arithmetic_side_effects)]
                    return Ok(arc.first_label + arc.arc_idx + 1);
                }
                _ => {
                    debug_assert_eq!(arc.bytes_per_arc, 0);
                    // ARITH: `next_arc` is a cursor position below
                    // `bytes.len()`; `- 1` at worst makes it `-1`, which
                    // `read_byte` rejects.
                    #[allow(clippy::arithmetic_side_effects)]
                    let prev = arc.next_arc - 1;
                    r.set_position(prev);
                }
            }
        }
        self.read_label(r)
    }

    /// `Util.binarySearch`: binary search an `ARCS_FOR_BINARY_SEARCH` node's
    /// arcs for `target_label`, starting from `arc.arc_idx` (usually `0`, but
    /// `find_next_floor_arc_binary_search` resumes from a nonzero index).
    /// Returns the matching slot index if found, else `-1 - insertion_point`
    /// (matching `Collections.binarySearch`'s / `Util.binarySearch`'s own
    /// negative-encoding convention).
    fn find_arc_binary_search(
        &self,
        arc: &Arc,
        target_label: i32,
        r: &mut BytesReader,
    ) -> Result<i32> {
        debug_assert_eq!(arc.node_flags, ARCS_FOR_BINARY_SEARCH);
        let mut low = arc.arc_idx;
        // ARITH: `num_arcs >= 1` (`read_array_node_header`), so `high >= 0`;
        // `mid` stays in `low..=high` and the `i64` slot product is below
        // 2^62. `low <= num_arcs`, so `-1 - low` cannot underflow.
        #[allow(clippy::arithmetic_side_effects)]
        let mut high = arc.num_arcs - 1;
        while low <= high {
            // `Util.binarySearch` is `(low + high) >>> 1`; see
            // `find_target_arc_binary_search` for why the unsigned shift is
            // load-bearing.
            let mid = (low.wrapping_add(high) as u32 >> 1) as i32;
            r.set_position(arc.pos_arcs_start);
            // ARITH: as in `find_target_arc_binary_search` -- two `i32`s widened
            // to `i64`.
            #[allow(clippy::arithmetic_side_effects)]
            let skip = arc.bytes_per_arc as i64 * mid as i64 + 1;
            r.skip_bytes(skip);
            let mid_label = self.read_label(r)?;
            // ARITH: `mid` is in `low..=high`, so `mid + 1`/`mid - 1` stay inside
            // `-1..=num_arcs`.
            #[allow(clippy::arithmetic_side_effects)]
            match mid_label.cmp(&target_label) {
                std::cmp::Ordering::Less => low = mid + 1,
                std::cmp::Ordering::Greater => high = mid - 1,
                std::cmp::Ordering::Equal => return Ok(mid),
            }
        }
        // ARITH: `low` is in `0..=num_arcs`, so `-1 - low` is in
        // `-num_arcs - 1 ..= -1`.
        #[allow(clippy::arithmetic_side_effects)]
        Ok(-1 - low)
    }

    /// `FSTEnum.findNextFloorArcBinarySearch`: given `arc` positioned at its
    /// node's first arc, find and read the arc whose label is strictly less
    /// than `target_label` (skipping the first arc, which has already been
    /// read and is used as-is if it's already the floor arc).
    fn find_next_floor_arc_binary_search(
        &self,
        arc: &mut Arc,
        target_label: i32,
        r: &mut BytesReader,
    ) -> Result<()> {
        debug_assert_eq!(arc.node_flags, ARCS_FOR_BINARY_SEARCH);
        debug_assert_eq!(arc.arc_idx, 0);
        if arc.num_arcs > 1 {
            let idx = self.find_arc_binary_search(arc, target_label, r)?;
            if idx == -1 {
                return corrupt_fst("binary search returned an impossible insertion point");
            }
            // ARITH: `find_arc_binary_search` returns either a slot index in
            // `0..num_arcs` or `-1 - insertion_point` with the insertion
            // point in `0..=num_arcs`, i.e. a value in
            // `-num_arcs - 1 ..= num_arcs - 1`. `idx - 1` and `-2 - idx` are
            // therefore both within `i32` under the branch guards.
            #[allow(clippy::arithmetic_side_effects)]
            if idx > 1 {
                self.read_arc_by_index(arc, r, idx - 1)?;
            } else if idx < -2 {
                self.read_arc_by_index(arc, r, -2 - idx)?;
            }
        }
        Ok(())
    }

    /// `FSTEnum.findNextFloorArcDirectAddressing`: direct-addressing analogue
    /// of `find_next_floor_arc_binary_search`.
    fn find_next_floor_arc_direct_addressing(
        &self,
        arc: &mut Arc,
        target_label: i32,
        r: &mut BytesReader,
    ) -> Result<()> {
        debug_assert_eq!(arc.node_flags, ARCS_FOR_DIRECT_ADDRESSING);
        debug_assert_eq!(arc.label, arc.first_label);
        if arc.num_arcs > 1 {
            // ARITH: `first_label` is non-negative and `target_label` is a
            // key label or `END_LABEL`; see
            // `find_target_arc_direct_addressing`.
            #[allow(clippy::arithmetic_side_effects)]
            let target_index = target_label - arc.first_label;
            if target_index < 0 {
                return corrupt_fst("a label range starting above the label being sought");
            }
            if target_index >= arc.num_arcs {
                self.read_last_arc_by_direct_addressing(arc, r)?;
            } else {
                let floor_index = bit_table_previous_bit_set(target_index, arc, r)?;
                // SENTINEL-OK: `> 0`, not `>= 0`, is Java's own test
                // (`findNextFloorArcDirectAddressing`): `arc` is already
                // standing on index 0 (`arc.label() == arc.firstLabel()` is
                // asserted on entry), so both `0` (the floor *is* index 0) and
                // the `-1` sentinel mean "stay where we are". The sentinel
                // therefore never reaches `read_arc_by_direct_addressing`.
                if floor_index > 0 {
                    let presence_index = bit_table_count_bits_up_to(floor_index, arc, r)?;
                    self.read_arc_by_direct_addressing(arc, r, floor_index, presence_index)?;
                }
            }
        }
        Ok(())
    }

    /// `FSTEnum.findNextFloorArcContinuous`: continuous-node analogue of
    /// `find_next_floor_arc_binary_search`.
    fn find_next_floor_arc_continuous(
        &self,
        arc: &mut Arc,
        target_label: i32,
        r: &mut BytesReader,
    ) -> Result<()> {
        debug_assert_eq!(arc.node_flags, ARCS_FOR_CONTINUOUS);
        debug_assert_eq!(arc.label, arc.first_label);
        if arc.num_arcs > 1 {
            // ARITH: as in the direct-addressing sibling above; and
            // `target_index >= 0` on the branch that decrements it.
            #[allow(clippy::arithmetic_side_effects)]
            let target_index = target_label - arc.first_label;
            if target_index < 0 {
                return corrupt_fst("a label range starting above the label being sought");
            }
            if target_index >= arc.num_arcs {
                self.read_last_arc_by_continuous(arc, r)?;
            } else {
                // ARITH: this branch has `0 <= target_index < num_arcs`.
                #[allow(clippy::arithmetic_side_effects)]
                let floor_index = target_index - 1;
                self.read_arc_by_continuous(arc, r, floor_index)?;
            }
        }
        Ok(())
    }

    // --- Ordered enumeration (`BytesRefFSTEnum`) --------------------------
    //
    // The three methods below are `FST.java`'s `readFirstRealTargetArc`/
    // `readNextRealArc`/`readFirstTargetArc`/`readNextArc`, ported field-for-
    // field. Unlike `find_target_arc` (which only ever needs to *find* one
    // specific labeled arc and can stop as soon as it does), full ordered
    // enumeration needs to walk every arc of every node it visits in turn --
    // real Lucene's `readNextRealArc` is exactly that "read the arc after
    // this one" primitive, generic over list-encoded and all three
    // fixed-length-arc node encodings alike, which is what `find_target_arc`
    // never needed and so never implemented.

    /// `FST.readNextRealArc`: advance `arc` in-place to the arc immediately
    /// following it within the same node (never the virtual `END_LABEL`
    /// arc -- callers must route through `read_next_arc` for that). `arc`
    /// must already be positioned at a real arc of the node (as set up by
    /// `read_first_real_target_arc` or a previous call to this method).
    fn read_next_real_arc(&self, arc: &mut Arc, r: &mut BytesReader) -> Result<()> {
        if arc.node_flags == ARCS_FOR_DIRECT_ADDRESSING {
            // Direct-addressing node: advance to the next *present* label in
            // the range (the bit-table's next set bit after `arc_idx`, which
            // is `-1` on the very first call).
            let next_index = bit_table_next_bit_set(arc.arc_idx, arc, r)?;
            // `bit_table_next_bit_set` returns `-1` for "no next set bit", and
            // `-1` is as far outside `0..num_arcs` as an index past the end
            // is: passed on as an `arcIdx` it makes `read_arc` compute
            // `first_label - 1`, a decoded arc one label *below* the range the
            // node declared -- and, for `first_label == 0`, exactly
            // `END_LABEL`. A single flipped bit in an all-but-one-zero
            // presence byte reaches it, and the slot address it produces
            // (`pos_arcs_start`, since `presence_index` goes -1 -> 0) is
            // perfectly valid, so nothing downstream catches it.
            // `read_next_arc_label` already guards the same sentinel; Java
            // states it as `assert nextIndex != -1`.
            if next_index == -1 {
                return corrupt_fst("no present arc follows this one in the bit table");
            }
            // ARITH: `presence_index` counts set bits in a bit table whose
            // byte count is bounded by the body (`read_presence_bytes`), so
            // it is far below `i32::MAX`.
            #[allow(clippy::arithmetic_side_effects)]
            let next_presence = arc.presence_index + 1;
            return self.read_arc_by_direct_addressing(arc, r, next_index, next_presence);
        } else if arc.bytes_per_arc != 0 {
            // `ARCS_FOR_BINARY_SEARCH` or `ARCS_FOR_CONTINUOUS` fixed-length-arc
            // node: advance to the next fixed-size slot by index (identical
            // arithmetic for both -- `ARCS_FOR_CONTINUOUS` never needs to
            // consult a presence bit-table since every slot in its range is
            // present).
            // ARITH: `arc_idx` starts at -1 or -2 and advances by one per
            // arc of a node whose arc count is validated `>= 1` and bounded
            // by the body (`check_arc_array_fits`), so it stays far below
            // `i32::MAX`; the `i64` slot product is below 2^62.
            #[allow(clippy::arithmetic_side_effects)]
            {
                arc.arc_idx += 1;
            }
            if arc.arc_idx < 0 || arc.arc_idx >= arc.num_arcs {
                return corrupt_fst("an arc index past the end of its node's arc array");
            }
            // ARITH: `arc_idx` is an arc index bounded by the node's arc count
            // and `bytes_per_arc` is validated `1..=i32::MAX`, so the `i64`
            // product is below 2^62.
            #[allow(clippy::arithmetic_side_effects)]
            let slot = arc.pos_arcs_start - arc.bytes_per_arc as i64 * arc.arc_idx as i64;
            r.set_position(slot);
            arc.flags = r.read_byte()?;
        } else {
            // List-encoded node: `arc.next_arc` is the position `read_arc`
            // already computed as "where the next arc's flags byte lives".
            r.set_position(arc.next_arc);
            arc.flags = r.read_byte()?;
        }
        self.read_arc(arc, r)
    }

    /// `FST.readFirstRealTargetArc`: read the first real (non-virtual) arc
    /// of the node at `node_address`, handling list-encoded and all three
    /// fixed-length-arc node headers (mirrors `FST.readFirstArcInfo` followed
    /// by one `readNextRealArc` call).
    fn read_first_real_target_arc(&self, node_address: i64, r: &mut BytesReader) -> Result<Arc> {
        r.set_position(node_address);
        let flags = r.read_byte()?;
        let mut arc = Arc {
            node_flags: flags,
            ..Default::default()
        };
        if flags == ARCS_FOR_BINARY_SEARCH {
            let (num_arcs, bytes_per_arc) = self.read_array_node_header(r)?;
            arc.num_arcs = num_arcs;
            arc.bytes_per_arc = bytes_per_arc;
            arc.pos_arcs_start = r.get_position();
            Self::check_arc_array_fits(num_arcs, bytes_per_arc, arc.pos_arcs_start)?;
            arc.arc_idx = -1;
        } else if flags == ARCS_FOR_DIRECT_ADDRESSING {
            // `num_arcs` is the label range here, not an arc count.
            let (num_arcs, bytes_per_arc) = self.read_array_node_header(r)?;
            arc.num_arcs = num_arcs;
            arc.bytes_per_arc = bytes_per_arc;
            self.read_presence_bytes(&mut arc, r)?;
            arc.first_label = self.read_label(r)?;
            self.check_label_range(arc.first_label, num_arcs)?;
            arc.presence_index = -1;
            arc.pos_arcs_start = r.get_position();
            arc.arc_idx = -1;
        } else if flags == ARCS_FOR_CONTINUOUS {
            let (num_arcs, bytes_per_arc) = self.read_array_node_header(r)?;
            arc.num_arcs = num_arcs;
            arc.bytes_per_arc = bytes_per_arc;
            arc.first_label = self.read_label(r)?;
            self.check_label_range(arc.first_label, num_arcs)?;
            arc.pos_arcs_start = r.get_position();
            Self::check_arc_array_fits(num_arcs, bytes_per_arc, arc.pos_arcs_start)?;
            arc.arc_idx = -1;
        } else {
            // List-encoded node: `next_arc` re-anchors `read_next_real_arc`
            // back at this node's first arc (its flags byte gets re-read,
            // matching `FST.java`'s own re-read of the same byte here).
            arc.next_arc = node_address;
            arc.bytes_per_arc = 0;
            arc.node_flags = 0;
        }
        self.read_next_real_arc(&mut arc, r)?;
        Ok(arc)
    }

    /// `FST.readFirstTargetArc`: read the first arc leaving `follow`'s
    /// target node -- if `follow` is itself an accepting (final) arc, that
    /// "first arc" is a synthetic `END_LABEL` arc representing acceptance at
    /// this node, inserted *before* any of the node's real outgoing arcs
    /// (matching real Lucene's enumeration order: a node's own key, if any,
    /// sorts before all of its longer descendants).
    fn read_first_target_arc(&self, follow: &Arc, r: &mut BytesReader) -> Result<Arc> {
        if follow.is_final() {
            let mut arc = Arc {
                label: END_LABEL,
                output: follow.next_final_output().to_vec(),
                flags: BIT_FINAL_ARC,
                target: FINAL_END_NODE,
                ..Default::default()
            };
            if follow.target() <= 0 {
                arc.flags |= BIT_LAST_ARC;
            } else {
                // Not really an address here -- a node address to resume
                // real-arc enumeration from once this virtual arc is
                // exhausted (see `read_next_arc`).
                arc.next_arc = follow.target();
            }
            Ok(arc)
        } else {
            self.read_first_real_target_arc(follow.target(), r)
        }
    }

    /// `FST.readNextArc`: advance `arc` to the arc following it, transparently
    /// handling the case where `arc` was itself the synthetic `END_LABEL` arc
    /// (in which case "next" means the target node's first *real* arc).
    fn read_next_arc(&self, arc: &Arc, r: &mut BytesReader) -> Result<Arc> {
        if arc.label == END_LABEL {
            if arc.next_arc <= 0 {
                return Err(Error::Corrupt(
                    "read_next_arc called on the last arc of a node".to_string(),
                ));
            }
            self.read_first_real_target_arc(arc.next_arc, r)
        } else {
            let mut next = arc.clone();
            self.read_next_real_arc(&mut next, r)?;
            Ok(next)
        }
    }

    /// Port of `BytesRefFSTEnum`'s full forward walk (`FSTEnum.doNext`/
    /// `pushFirst`, no seek support -- see `Fst::iter`'s doc comment):
    /// returns an iterator over every `(key, output)` pair this FST accepts,
    /// in ascending key order.
    pub fn iter(&self) -> Result<FstEnum<'_, 'a>> {
        if self.metadata.input_type != InputType::Byte1 {
            return Err(Error::Unsupported(
                "Fst::iter only supports INPUT_TYPE.BYTE1 (byte-keyed `Vec<u8>` keys); \
                 use Fst::iter_labels for BYTE2/BYTE4 FSTs, whose labels don't fit in a byte"
                    .to_string(),
            ));
        }
        Ok(FstEnum::new(self))
    }

    /// Label-domain equivalent of `iter`, usable for **any** `INPUT_TYPE`:
    /// the same `FstEnum` engine (`BytesRefFSTEnum`'s full ascending walk),
    /// but its `next_labels`/`seek_*_labels` methods yield/accept `Vec<i32>`
    /// keys instead of `Vec<u8>` ones. `BYTE1` FSTs' labels are numerically
    /// identical to their byte values, so `iter_labels` (widened to `i32`)
    /// and `iter` (restricted to `u8`) walk a `BYTE1` FST identically, just
    /// with a wider key type -- but only `iter_labels` can walk a `BYTE2`/
    /// `BYTE4` FST, whose labels (UTF-16 code units / Unicode code points)
    /// generally don't fit in a `u8`.
    pub fn iter_labels(&self) -> FstEnum<'_, 'a> {
        FstEnum::new(self)
    }

    /// `IntsRefFSTEnum`'s constructor equivalent -- see `IntsRefFstEnum`'s
    /// type doc for exactly what this does and does not port.
    pub fn iter_ints(&self) -> Result<IntsRefFstEnum<'_, 'a>> {
        Ok(IntsRefFstEnum {
            inner: self.iter()?,
        })
    }

    /// Port of `Util.get(FST, BytesRef)`: look up `key` and return its
    /// accumulated output if the FST accepts it, else `None`. Only meaningful
    /// for `BYTE1` FSTs, whose labels are exactly the raw bytes of `key`; use
    /// `get_labels` for `BYTE2`/`BYTE4` FSTs, whose labels (UTF-16 code units
    /// / Unicode code points) aren't representable as a `u8` in general.
    pub fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        if self.metadata.input_type != InputType::Byte1 {
            return Err(Error::Unsupported(
                "Fst::get only supports INPUT_TYPE.BYTE1 (byte-keyed `&[u8]` keys); \
                 use Fst::get_labels for BYTE2/BYTE4 FSTs, whose labels don't fit in a byte"
                    .to_string(),
            ));
        }
        self.get_label_iter(key.iter().map(|&b| i32::from(b)))
    }

    /// Label-domain equivalent of `get`, mirroring `Util.get(FST, IntsRef)`:
    /// walks `find_target_arc` once per element of `labels` rather than once
    /// per raw byte, so it works for **any** `INPUT_TYPE` -- `BYTE1` FSTs'
    /// labels happen to be literal byte values (`get` delegates to this),
    /// `BYTE2` FSTs' labels are UTF-16 code units, and `BYTE4` FSTs' labels
    /// are full Unicode code points, neither representable in a `u8`, which
    /// is why `get` alone can't reach those.
    pub fn get_labels(&self, labels: &[i32]) -> Result<Option<Vec<u8>>> {
        self.get_label_iter(labels.iter().copied())
    }

    /// [`Self::get_labels`] over any label sequence. The output is
    /// accumulated in one buffer (`ByteSequenceOutputs.add` appends, so
    /// extending in place is the same concatenation).
    fn get_label_iter(&self, labels: impl IntoIterator<Item = i32>) -> Result<Option<Vec<u8>>> {
        let mut r = self.reader();
        let mut arc = self.first_arc();
        // The virtual first arc isn't itself the answer for an empty key --
        // Util.get walks every label and only then checks `arc.isFinal()`,
        // so an empty key's answer is the empty output.
        let mut output: Vec<u8> = Vec::new();

        for label in labels {
            match self.find_target_arc(label, &arc, &mut r)? {
                Some(next) => {
                    output.extend_from_slice(next.output());
                    arc = next;
                }
                None => return Ok(None),
            }
        }

        if arc.is_final() {
            output.extend_from_slice(arc.next_final_output());
            Ok(Some(output))
        } else {
            Ok(None)
        }
    }

    /// Port of `BytesRefFSTEnum.seekExact`: look up `key` and return its
    /// accumulated output if the FST accepts it, else `None`.
    ///
    /// This legitimately delegates to `Fst::get` rather than re-implementing
    /// an independent arc-by-arc descent: `get` *is* that descent (walk one
    /// arc per key byte via `find_target_arc`, never a linear scan over the
    /// FST's full contents), which is exactly what real `FSTEnum.doSeekExact`
    /// does too (its main loop is `fst.findTargetArc(targetLabel, arc,
    /// getArc(upto), fstReader)` called once per key byte) -- the only
    /// difference is `doSeekExact` additionally maintains `FSTEnum`'s
    /// persistent per-depth arc/output stack so a subsequent `next()` can
    /// resume enumeration from where the seek landed, which this
    /// non-iterator convenience method has no need for. See
    /// `FstEnum::seek_exact` for the stateful, stack-maintaining equivalent
    /// used when seeking is followed by continued enumeration.
    pub fn seek_exact(&self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        self.get(key)
    }

    /// Label-domain equivalent of `seek_exact`, delegating to `get_labels`
    /// for the same reason `seek_exact` delegates to `get` -- usable for any
    /// `INPUT_TYPE`, see `get_labels`.
    pub fn seek_exact_labels(&self, labels: &[i32]) -> Result<Option<Vec<u8>>> {
        self.get_labels(labels)
    }
}

/// Port of `BytesRefFSTEnum`, restricted to a full forward walk (real
/// Lucene's `next()`) -- **not** ported: `seekCeil`/`seekFloor`/`seekExact`
/// (`FSTEnum.doSeekCeil`/`doSeekFloor`/`doSeekExact`, `rewindPrefix`'s shared-
/// prefix-aware re-seek), which real Lucene needs for range/prefix queries
/// over a term dictionary but which is a substantially larger increment on
/// top of this (each seek variant has its own list/binary-search/direct-
/// addressing/continuous dispatch -- see `FSTEnum.java`). This type only
/// walks every accepted key in ascending order from the start, which is
/// already enough to enumerate any FST's full contents; see `docs/parity.md`
/// for the deferred seek support and `IntsRefFstEnum` (below) for the
/// `IntsRef`-keyed sibling of this type.
///
/// Constructed via `Fst::iter`. Implements `Iterator<Item = Result<(Vec<u8>,
/// Vec<u8>)>>`, yielding `(key, output)` pairs in ascending key order.
pub struct FstEnum<'f, 'a> {
    fst: &'f Fst<'a>,
    r: BytesReader<'f>,
    /// `FSTEnum.arcs`: `arcs[0]` is the virtual incoming arc to the start
    /// node (`Fst::first_arc`); `arcs[i]` (`i >= 1`) is the arc last read at
    /// depth `i`, potentially the synthetic `END_LABEL` arc representing
    /// acceptance at that depth.
    arcs: Vec<Arc>,
    /// `FSTEnum.output`: `outputs[i]` is the cumulative output through
    /// depth `i` (`outputs[0]` is always empty -- no output accumulated for
    /// the virtual root arc).
    outputs: Vec<Vec<u8>>,
    /// `BytesRefFSTEnum.current`: `labels[i]` (`i >= 1`) is the key label
    /// chosen at depth `i` -- i.e. the current key, once accepted, is
    /// `labels[1..upto]`. Stored as `i32` (the full arc-label domain,
    /// matching `Arc::label`'s own type) rather than `u8` so this single
    /// engine can walk `BYTE2`/`BYTE4` FSTs (UTF-16 code units / Unicode code
    /// points, both wider than a byte) as well as `BYTE1` ones -- the
    /// byte-keyed public API (`Iterator for FstEnum`, `seek_ceil`/
    /// `seek_floor`/`seek_exact`) narrows back to `u8` only at its own
    /// boundary, where `Fst::iter`'s `BYTE1`-only guard makes that lossless.
    labels: Vec<i32>,
    /// `FSTEnum.upto`.
    upto: usize,
    /// Set once enumeration has yielded every key, so a subsequent `next()`
    /// call keeps returning `None` (a fused iterator) instead of
    /// misreading `upto == 0`'s *other* meaning -- "not yet started" -- and
    /// restarting from the first key. `upto` alone can't distinguish these
    /// two states: both leave it at `0`.
    done: bool,
    /// `BytesRefFSTEnum.target`: the key most recently passed to
    /// `seek_ceil`/`seek_floor`/`seek_exact`, label-indexed from `0` (unlike
    /// `labels`, which is 1-indexed by depth) -- empty/unused until the
    /// first seek call. `i32`-typed for the same reason `labels` is.
    target: Vec<i32>,
    /// `FSTEnum.targetLength`: `target.len()`, cached since `get_target_label`
    /// (`BytesRefFSTEnum.getTargetLabel`) is called once per arc visited
    /// during a seek.
    target_length: usize,
}

impl<'f, 'a> FstEnum<'f, 'a> {
    fn new(fst: &'f Fst<'a>) -> Self {
        FstEnum {
            r: fst.reader(),
            fst,
            arcs: vec![fst.first_arc()],
            outputs: vec![Vec::new()],
            labels: vec![0i32],
            upto: 0,
            done: false,
            target: Vec::new(),
            target_length: 0,
        }
    }

    /// Guards the one loop in this enum that a corrupt body can make
    /// non-terminating: descending from an arc into its target node.
    ///
    /// `FSTCompiler` writes nodes bottom-up into a forward-growing byte
    /// store, so a node's address is always **greater** than every address
    /// its arcs point at -- an FST is a DAG and its arcs run strictly
    /// downward through the body. A body whose targets do not strictly
    /// decrease therefore contains a cycle, and a cycle turns `push_first`/
    /// `push_last`/`do_seek_exact` into an unbounded descent that grows
    /// `arcs`, `outputs` and `labels` one level per step until the process is
    /// killed by the OOM killer. That is worse than an abort: nothing catches
    /// it and no timeout fires.
    ///
    /// Not hypothetical -- a single flipped target byte in
    /// `tests/fst_byte_flip_sweep.rs` produces exactly that, which is how
    /// this was found.
    ///
    /// The check cannot reject anything a real writer produced: every FST
    /// fixture in `fixtures/data/fst*` (all four node encodings, both
    /// multi-level tries and array roots) walks fully with it in place.
    fn check_descent(&self, arc: &Arc) -> Result<()> {
        // `arcs[upto - 1].target()` is the address of the node `arc` was read
        // out of.
        let containing_node = match self.upto.checked_sub(1) {
            Some(i) => self.arcs[i].target(),
            None => return Ok(()),
        };
        if arc.target() >= containing_node {
            return Err(Error::Corrupt(format!(
                "FST arc target {} does not precede its own node at {containing_node}: the body \
                 is cyclic",
                arc.target()
            )));
        }
        Ok(())
    }

    fn set_arc(&mut self, idx: usize, arc: Arc) {
        if self.arcs.len() <= idx {
            // ARITH: `idx` is a depth into the enum's own stacks, bounded by
            // the FST's longest key; `idx + 1` cannot overflow a `usize`
            // without the `Vec` it sizes being impossible to allocate.
            #[allow(clippy::arithmetic_side_effects)]
            self.arcs.resize(idx + 1, Arc::default());
        }
        self.arcs[idx] = arc;
    }

    fn set_output(&mut self, idx: usize, output: Vec<u8>) {
        if self.outputs.len() <= idx {
            // ARITH: `idx` is a depth into the enum's own stacks, bounded by
            // the FST's longest key; `idx + 1` cannot overflow a `usize`
            // without the `Vec` it sizes being impossible to allocate.
            #[allow(clippy::arithmetic_side_effects)]
            self.outputs.resize(idx + 1, Vec::new());
        }
        self.outputs[idx] = output;
    }

    fn set_label(&mut self, idx: usize, label: i32) {
        if self.labels.len() <= idx {
            // ARITH: `idx` is a depth into the enum's own stacks, bounded by
            // the FST's longest key; `idx + 1` cannot overflow a `usize`
            // without the `Vec` it sizes being impossible to allocate.
            #[allow(clippy::arithmetic_side_effects)]
            self.labels.resize(idx + 1, 0i32);
        }
        self.labels[idx] = label;
    }

    /// `FSTEnum.pushFirst`: from `self.arcs[self.upto]`, repeatedly descend
    /// via `read_first_target_arc` until landing on an accepting (synthetic
    /// `END_LABEL`) arc -- the smallest-keyed accepted descendant of the
    /// node `self.upto` started at.
    // ARITH: every caller enters with `upto >= 1` (`advance` and
    // `rewind_prefix` set it to 1 before descending; the seek helpers only
    // reach here after `finish_seek_*` at depth >= 1), and `set_output` grows
    // `outputs` to `upto + 1` before `upto` is incremented -- so
    // `outputs[upto - 1]` is always in range. `upto` counts one level per arc
    // descended, so it is bounded by the FST's longest key.
    #[allow(clippy::arithmetic_side_effects)]
    fn push_first(&mut self) -> Result<()> {
        loop {
            let arc = self.arcs[self.upto].clone();
            let cum_output = output_add(&self.outputs[self.upto - 1], arc.output());
            self.set_output(self.upto, cum_output);
            if arc.label() == END_LABEL {
                break;
            }
            self.set_label(self.upto, arc.label());
            self.check_descent(&arc)?;
            self.upto += 1;
            let next = self.fst.read_first_target_arc(&arc, &mut self.r)?;
            self.set_arc(self.upto, next);
        }
        Ok(())
    }

    /// `FSTEnum.pushLast`: from `self.arcs[self.upto]`, repeatedly descend
    /// via `read_last_target_arc` until landing on an accepting (synthetic
    /// `END_LABEL`) arc -- the largest-keyed accepted descendant of the node
    /// `self.upto` started at. `push_first`'s mirror image, needed by
    /// `seek_floor`'s "target falls strictly between two sibling arcs, or
    /// past the last one" cases.
    // ARITH: same invariant as `push_first`.
    #[allow(clippy::arithmetic_side_effects)]
    fn push_last(&mut self) -> Result<()> {
        loop {
            let arc = self.arcs[self.upto].clone();
            self.set_label(self.upto, arc.label());
            let cum_output = output_add(&self.outputs[self.upto - 1], arc.output());
            self.set_output(self.upto, cum_output);
            if arc.label() == END_LABEL {
                break;
            }
            self.check_descent(&arc)?;
            self.upto += 1;
            let next = self.fst.read_last_target_arc(&arc, &mut self.r)?;
            self.set_arc(self.upto, next);
        }
        Ok(())
    }

    // --- getTargetLabel/getCurrentLabel/setCurrentLabel/incr -------------

    /// `BytesRefFSTEnum.getTargetLabel`.
    /// `BytesRefFSTEnum.getTargetLabel`.
    ///
    /// Java indexes `target.bytes[upto - 1]` behind the invariant
    /// `1 <= upto <= targetLength + 1`. Spelled here as a `checked_sub` plus
    /// a `get`, which is the same answer whenever that invariant holds
    /// (`upto - 1 == target_length` is exactly the out-of-range index) and a
    /// typed `END_LABEL` rather than an index panic if a future seek path
    /// ever breaks it.
    fn get_target_label(&self) -> i32 {
        match self.upto.checked_sub(1) {
            None => END_LABEL,
            Some(i) => self.target.get(i).copied().unwrap_or(END_LABEL),
        }
    }

    /// `BytesRefFSTEnum.getCurrentLabel`.
    fn get_current_label(&self) -> i32 {
        // `labels` is grown to `upto + 1` by `set_label` at every depth the
        // enum has visited; `get` rather than `[..]` so a depth it has not
        // cannot panic.
        self.labels.get(self.upto).copied().unwrap_or(END_LABEL)
    }

    /// `FSTEnum.rewindPrefix`: rewind `self.upto` back to the end of the
    /// shared prefix between the enum's current position and `self.target`,
    /// so `do_seek_ceil`/`do_seek_floor`/`do_seek_exact`'s main loops only
    /// need to (re)walk the target's differing suffix.
    fn rewind_prefix(&mut self) -> Result<()> {
        if self.upto == 0 {
            let root = self.arcs[0].clone();
            if !root.is_final() && !target_has_arcs(&root) {
                // Degenerate FST that accepts no keys at all (see the same
                // guard in `advance`) -- leave `self.upto` at `0` so the
                // seek drivers below treat this as "nothing to find" without
                // dereferencing a nonexistent node.
                return Ok(());
            }
            self.upto = 1;
            let first = self.fst.read_first_target_arc(&root, &mut self.r)?;
            self.set_arc(1, first);
            return Ok(());
        }

        let current_limit = self.upto;
        self.upto = 1;
        // ARITH: `upto` starts at 1 and only increases, so `upto - 1` is in
        // range; `target_length` is `target.len()`, so `+ 1` cannot overflow
        // a `usize`. Java's `int cmp = getCurrentLabel() - getTargetLabel();`
        // is an `int` subtraction that can overflow for a `BYTE4` label near
        // `Integer.MAX_VALUE` against `END_LABEL`; only its sign is ever
        // tested, so an `Ord::cmp` is the same answer with no subtraction at
        // all.
        #[allow(clippy::arithmetic_side_effects)]
        while self.upto < current_limit && self.upto <= self.target_length + 1 {
            match self.get_current_label().cmp(&self.get_target_label()) {
                std::cmp::Ordering::Less => {
                    // Seek forward: the shared prefix ends here.
                    break;
                }
                std::cmp::Ordering::Greater => {
                    // Seek backwards -- reset this arc to the first arc of its
                    // parent node.
                    let parent = self.arcs[self.upto - 1].clone();
                    let first = self.fst.read_first_target_arc(&parent, &mut self.r)?;
                    self.set_arc(self.upto, first);
                    break;
                }
                std::cmp::Ordering::Equal => self.upto += 1,
            }
        }
        Ok(())
    }

    /// `FSTEnum.rollbackToLastForkThenPush`: the current node's label range
    /// is exhausted with no ceiling arc found -- back up depth by depth
    /// until a node with an unvisited next-sibling arc is found, take that
    /// sibling, then `push_first` from there (the smallest-keyed accepted
    /// descendant of that sibling). If no such node exists (`self.upto`
    /// reaches `0`), there is no key `>=` the target anywhere in the FST.
    // ARITH: the guard below establishes `upto >= 1` before every decrement,
    // and the loop returns as soon as it reaches 0.
    #[allow(clippy::arithmetic_side_effects)]
    fn rollback_to_last_fork_then_push(&mut self) -> Result<()> {
        if self.upto == 0 {
            return Ok(());
        }
        self.upto -= 1;
        loop {
            if self.upto == 0 {
                return Ok(());
            }
            let prev = self.arcs[self.upto].clone();
            if !prev.is_last() {
                let next = self.fst.read_next_arc(&prev, &mut self.r)?;
                self.set_arc(self.upto, next);
                self.push_first()?;
                return Ok(());
            }
            self.upto -= 1;
        }
    }

    /// `FSTEnum.backtrackToFloorArc`: backtracks until it finds a node whose
    /// first arc is before the (possibly-updated, after backing up a level)
    /// target label, then on that node finds the arc just before the target
    /// label and `push_last`s from there. If `self.upto` reaches `0`, there
    /// is no key `<=` the target anywhere in the FST.
    // ARITH: every caller enters with `upto >= 1`, and the loop returns as
    // soon as the decrement reaches 0 -- so `upto - 1` never underflows.
    #[allow(clippy::arithmetic_side_effects)]
    fn backtrack_to_floor_arc(&mut self) -> Result<()> {
        loop {
            let target_label = self.get_target_label();
            let parent = self.arcs[self.upto - 1].clone();
            let mut arc = self.fst.read_first_target_arc(&parent, &mut self.r)?;
            if arc.label() < target_label {
                if !arc.is_last() {
                    if arc.bytes_per_arc != 0 && arc.label() != END_LABEL {
                        match arc.node_flags {
                            ARCS_FOR_BINARY_SEARCH => self.fst.find_next_floor_arc_binary_search(
                                &mut arc,
                                target_label,
                                &mut self.r,
                            )?,
                            ARCS_FOR_DIRECT_ADDRESSING => {
                                self.fst.find_next_floor_arc_direct_addressing(
                                    &mut arc,
                                    target_label,
                                    &mut self.r,
                                )?
                            }
                            _ => {
                                debug_assert_eq!(arc.node_flags, ARCS_FOR_CONTINUOUS);
                                self.fst.find_next_floor_arc_continuous(
                                    &mut arc,
                                    target_label,
                                    &mut self.r,
                                )?
                            }
                        }
                    } else {
                        while !arc.is_last()
                            && self.fst.read_next_arc_label(&arc, &mut self.r)? < target_label
                        {
                            arc = self.fst.read_next_arc(&arc, &mut self.r)?;
                        }
                    }
                }
                self.set_arc(self.upto, arc);
                self.push_last()?;
                return Ok(());
            }
            self.upto -= 1;
            if self.upto == 0 {
                return Ok(());
            }
        }
    }

    /// Shared tail of every "found an exact match at this depth" branch
    /// across `seek_ceil`'s/`seek_exact`'s per-encoding helpers: records the
    /// matched arc/output, then either stops (if it's the synthetic
    /// `END_LABEL` acceptance arc) or descends one level and continues the
    /// seek loop from the newly matched arc's target's first arc.
    // ARITH: reached only from a seek loop at depth `upto >= 1`, with
    // `set_output` having grown `outputs` past `upto - 1`.
    #[allow(clippy::arithmetic_side_effects)]
    fn finish_seek_match(&mut self, arc: Arc, target_label: i32) -> Result<bool> {
        let cum_output = output_add(&self.outputs[self.upto - 1], arc.output());
        self.set_output(self.upto, cum_output);
        self.set_arc(self.upto, arc.clone());
        if target_label == END_LABEL {
            return Ok(false);
        }
        self.set_label(self.upto, arc.label());
        self.check_descent(&arc)?;
        self.upto += 1;
        let next = self.fst.read_first_target_arc(&arc, &mut self.r)?;
        self.set_arc(self.upto, next);
        Ok(true)
    }

    /// Shared tail of every "ceiling arc found, but past an exact match" seek
    /// branch: record `arc` then `push_first` from it.
    fn finish_seek_push_first(&mut self, arc: Arc) -> Result<()> {
        self.set_arc(self.upto, arc);
        self.push_first()
    }

    /// Shared tail of every "floor arc found, but before an exact match" seek
    /// branch: record `arc` then `push_last` from it.
    fn finish_seek_push_last(&mut self, arc: Arc) -> Result<()> {
        self.set_arc(self.upto, arc);
        self.push_last()
    }

    // --- doSeekCeil ---------------------------------------------------------

    fn seek_ceil_list(&mut self, target_label: i32) -> Result<bool> {
        let arc = self.arcs[self.upto].clone();
        if arc.label() == target_label {
            self.finish_seek_match(arc, target_label)
        } else if arc.label() > target_label {
            self.finish_seek_push_first(arc)?;
            Ok(false)
        } else if arc.is_last() {
            self.rollback_to_last_fork_then_push()?;
            Ok(false)
        } else {
            let next = self.fst.read_next_arc(&arc, &mut self.r)?;
            self.set_arc(self.upto, next);
            Ok(true)
        }
    }

    fn seek_ceil_binary_search(&mut self, target_label: i32) -> Result<bool> {
        let mut arc = self.arcs[self.upto].clone();
        let idx = self
            .fst
            .find_arc_binary_search(&arc, target_label, &mut self.r)?;
        if idx >= 0 {
            self.fst.read_arc_by_index(&mut arc, &mut self.r, idx)?;
            self.finish_seek_match(arc, target_label)
        } else {
            // ARITH: `find_arc_binary_search` returns `-1 - insertion_point`
            // with the insertion point in `0..=num_arcs`, so `idx` is now in
            // `0..=num_arcs` and `idx - 1` is taken only when it equals
            // `num_arcs >= 1`.
            #[allow(clippy::arithmetic_side_effects)]
            let idx = -1 - idx;
            // ARITH: `idx` is now in `0..=num_arcs`, and this branch takes
            // `idx - 1` only when it equals `num_arcs >= 1`.
            #[allow(clippy::arithmetic_side_effects)]
            if idx == arc.num_arcs {
                // Dead end: target is after the last arc.
                self.fst.read_arc_by_index(&mut arc, &mut self.r, idx - 1)?;
                self.rollback_to_last_fork_then_push()?;
            } else {
                self.fst.read_arc_by_index(&mut arc, &mut self.r, idx)?;
                self.finish_seek_push_first(arc)?;
            }
            Ok(false)
        }
    }

    fn seek_ceil_direct_addressing(&mut self, target_label: i32) -> Result<bool> {
        let mut arc = self.arcs[self.upto].clone();
        // ARITH: `first_label` is non-negative (`read_label`) and
        // `target_label` is a key label or `END_LABEL` (-1), so the
        // difference is at worst `-1 - i32::MAX == i32::MIN`.
        #[allow(clippy::arithmetic_side_effects)]
        let target_index = target_label - arc.first_label;
        if target_index >= arc.num_arcs {
            self.rollback_to_last_fork_then_push()?;
            Ok(false)
        } else {
            let clamped_index = if target_index < 0 { -1 } else { target_index };
            if clamped_index >= 0 && bit_table_is_bit_set(clamped_index, &arc, &mut self.r)? {
                let presence_index = bit_table_count_bits_up_to(clamped_index, &arc, &mut self.r)?;
                self.fst.read_arc_by_direct_addressing(
                    &mut arc,
                    &mut self.r,
                    clamped_index,
                    presence_index,
                )?;
                self.finish_seek_match(arc, target_label)
            } else {
                let ceil_index = bit_table_next_bit_set(clamped_index, &arc, &mut self.r)?;
                if ceil_index == -1 {
                    return corrupt_fst("a direct-addressing node with no present arc at all");
                }
                let presence_index = bit_table_count_bits_up_to(ceil_index, &arc, &mut self.r)?;
                self.fst.read_arc_by_direct_addressing(
                    &mut arc,
                    &mut self.r,
                    ceil_index,
                    presence_index,
                )?;
                self.finish_seek_push_first(arc)?;
                Ok(false)
            }
        }
    }

    fn seek_ceil_continuous(&mut self, target_label: i32) -> Result<bool> {
        let mut arc = self.arcs[self.upto].clone();
        // ARITH: `first_label` is non-negative (`read_label`) and
        // `target_label` is a key label or `END_LABEL` (-1), so the
        // difference is at worst `-1 - i32::MAX == i32::MIN`.
        #[allow(clippy::arithmetic_side_effects)]
        let target_index = target_label - arc.first_label;
        if target_index >= arc.num_arcs {
            self.rollback_to_last_fork_then_push()?;
            Ok(false)
        } else if target_index < 0 {
            self.fst.read_arc_by_continuous(&mut arc, &mut self.r, 0)?;
            self.finish_seek_push_first(arc)?;
            Ok(false)
        } else {
            self.fst
                .read_arc_by_continuous(&mut arc, &mut self.r, target_index)?;
            self.finish_seek_match(arc, target_label)
        }
    }

    /// `FSTEnum.doSeekCeil`: seeks to the smallest accepted key `>=`
    /// `self.target`.
    fn do_seek_ceil(&mut self) -> Result<()> {
        self.rewind_prefix()?;
        if self.upto == 0 {
            // Degenerate, key-less FST -- see `rewind_prefix`'s guard.
            return Ok(());
        }
        loop {
            let target_label = self.get_target_label();
            let arc = self.arcs[self.upto].clone();
            let keep_going = if arc.bytes_per_arc != 0 && arc.label() != END_LABEL {
                match arc.node_flags {
                    ARCS_FOR_DIRECT_ADDRESSING => self.seek_ceil_direct_addressing(target_label)?,
                    ARCS_FOR_BINARY_SEARCH => self.seek_ceil_binary_search(target_label)?,
                    _ => {
                        debug_assert_eq!(arc.node_flags, ARCS_FOR_CONTINUOUS);
                        self.seek_ceil_continuous(target_label)?
                    }
                }
            } else {
                self.seek_ceil_list(target_label)?
            };
            if !keep_going {
                break;
            }
        }
        Ok(())
    }

    // --- doSeekFloor --------------------------------------------------------

    fn seek_floor_list(&mut self, target_label: i32) -> Result<bool> {
        let arc = self.arcs[self.upto].clone();
        if arc.label() == target_label {
            self.finish_seek_match(arc, target_label)
        } else if arc.label() > target_label {
            self.seek_floor_list_backtrack()?;
            Ok(false)
        } else if !arc.is_last() {
            if self.fst.read_next_arc_label(&arc, &mut self.r)? > target_label {
                self.finish_seek_push_last(arc)?;
                Ok(false)
            } else {
                let next = self.fst.read_next_arc(&arc, &mut self.r)?;
                self.set_arc(self.upto, next);
                Ok(true)
            }
        } else {
            self.finish_seek_push_last(arc)?;
            Ok(false)
        }
    }

    /// `doSeekFloorList`'s own inline backward walk (distinct from
    /// `backtrack_to_floor_arc`, which the array-encoding seek-floor variants
    /// use): correct for a rewound node of *any* encoding since it drives
    /// entirely through the generic `read_next_arc`/`read_next_arc_label`
    /// dispatchers rather than an encoding-specific fast path -- just
    /// potentially slower.
    // ARITH: same invariant as `backtrack_to_floor_arc`.
    #[allow(clippy::arithmetic_side_effects)]
    fn seek_floor_list_backtrack(&mut self) -> Result<()> {
        loop {
            let target_label = self.get_target_label();
            let parent = self.arcs[self.upto - 1].clone();
            let mut arc = self.fst.read_first_target_arc(&parent, &mut self.r)?;
            if arc.label() < target_label {
                while !arc.is_last()
                    && self.fst.read_next_arc_label(&arc, &mut self.r)? < target_label
                {
                    arc = self.fst.read_next_arc(&arc, &mut self.r)?;
                }
                self.set_arc(self.upto, arc);
                self.push_last()?;
                return Ok(());
            }
            self.upto -= 1;
            if self.upto == 0 {
                return Ok(());
            }
        }
    }

    fn seek_floor_binary_search(&mut self, target_label: i32) -> Result<bool> {
        let mut arc = self.arcs[self.upto].clone();
        let idx = self
            .fst
            .find_arc_binary_search(&arc, target_label, &mut self.r)?;
        if idx >= 0 {
            self.fst.read_arc_by_index(&mut arc, &mut self.r, idx)?;
            self.finish_seek_match(arc, target_label)
        } else if idx == -1 {
            self.backtrack_to_floor_arc()?;
            Ok(false)
        } else {
            // ARITH: this branch has `idx < -1`, i.e. `idx <= -2`, and
            // `find_arc_binary_search` never returns below `-num_arcs - 1`
            // -- so `-2 - idx` is in `0..=num_arcs - 1`.
            #[allow(clippy::arithmetic_side_effects)]
            let floor_slot = -2 - idx;
            self.fst
                .read_arc_by_index(&mut arc, &mut self.r, floor_slot)?;
            self.finish_seek_push_last(arc)?;
            Ok(false)
        }
    }

    fn seek_floor_direct_addressing(&mut self, target_label: i32) -> Result<bool> {
        let mut arc = self.arcs[self.upto].clone();
        // ARITH: `first_label` is non-negative (`read_label`) and
        // `target_label` is a key label or `END_LABEL` (-1), so the
        // difference is at worst `-1 - i32::MAX == i32::MIN`.
        #[allow(clippy::arithmetic_side_effects)]
        let target_index = target_label - arc.first_label;
        if target_index < 0 {
            self.backtrack_to_floor_arc()?;
            Ok(false)
        } else if target_index >= arc.num_arcs {
            self.fst
                .read_last_arc_by_direct_addressing(&mut arc, &mut self.r)?;
            self.finish_seek_push_last(arc)?;
            Ok(false)
        } else if bit_table_is_bit_set(target_index, &arc, &mut self.r)? {
            let presence_index = bit_table_count_bits_up_to(target_index, &arc, &mut self.r)?;
            self.fst.read_arc_by_direct_addressing(
                &mut arc,
                &mut self.r,
                target_index,
                presence_index,
            )?;
            self.finish_seek_match(arc, target_label)
        } else {
            let floor_index = bit_table_previous_bit_set(target_index, &arc, &mut self.r)?;
            // Java is `assert floorIndex != -1;` -- off in production, so
            // porting it as a `debug_assert!` would port the *absence* of the
            // check. It holds only for a well-formed bit table: bit 0 of a
            // direct-addressing node is `first_label` and is always present,
            // so a `target_index` that reached this branch is at least 1 and
            // has a set bit below it. A `.tip`/FST byte with that bit cleared
            // makes `previous_bit_set` answer `-1`, and `-1` here would be the
            // same sentinel misuse as in `read_next_real_arc`:
            // `read_arc_by_direct_addressing` derives `first_label - 1`, an
            // arc one label below the range the node declared, and for
            // `first_label == 0` exactly `END_LABEL` -- a *plausible wrong
            // answer*, which is why a byte-flip sweep passes over it.
            if floor_index == -1 {
                return corrupt_fst("no present arc precedes this label in the bit table");
            }
            let presence_index = bit_table_count_bits_up_to(floor_index, &arc, &mut self.r)?;
            self.fst.read_arc_by_direct_addressing(
                &mut arc,
                &mut self.r,
                floor_index,
                presence_index,
            )?;
            self.finish_seek_push_last(arc)?;
            Ok(false)
        }
    }

    fn seek_floor_continuous(&mut self, target_label: i32) -> Result<bool> {
        let mut arc = self.arcs[self.upto].clone();
        // ARITH: `first_label` is non-negative (`read_label`) and
        // `target_label` is a key label or `END_LABEL` (-1), so the
        // difference is at worst `-1 - i32::MAX == i32::MIN`.
        #[allow(clippy::arithmetic_side_effects)]
        let target_index = target_label - arc.first_label;
        if target_index < 0 {
            self.backtrack_to_floor_arc()?;
            Ok(false)
        } else if target_index >= arc.num_arcs {
            self.fst
                .read_last_arc_by_continuous(&mut arc, &mut self.r)?;
            self.finish_seek_push_last(arc)?;
            Ok(false)
        } else {
            self.fst
                .read_arc_by_continuous(&mut arc, &mut self.r, target_index)?;
            self.finish_seek_match(arc, target_label)
        }
    }

    /// `FSTEnum.doSeekFloor`: seeks to the largest accepted key `<=`
    /// `self.target`.
    fn do_seek_floor(&mut self) -> Result<()> {
        self.rewind_prefix()?;
        if self.upto == 0 {
            // Degenerate, key-less FST -- see `rewind_prefix`'s guard.
            return Ok(());
        }
        loop {
            let target_label = self.get_target_label();
            let arc = self.arcs[self.upto].clone();
            let keep_going = if arc.bytes_per_arc != 0 && arc.label() != END_LABEL {
                match arc.node_flags {
                    ARCS_FOR_DIRECT_ADDRESSING => {
                        self.seek_floor_direct_addressing(target_label)?
                    }
                    ARCS_FOR_BINARY_SEARCH => self.seek_floor_binary_search(target_label)?,
                    _ => {
                        debug_assert_eq!(arc.node_flags, ARCS_FOR_CONTINUOUS);
                        self.seek_floor_continuous(target_label)?
                    }
                }
            } else {
                self.seek_floor_list(target_label)?
            };
            if !keep_going {
                break;
            }
        }
        Ok(())
    }

    // --- doSeekExact ---------------------------------------------------------

    /// `FSTEnum.doSeekExact`: seeks to exactly `self.target`, returning
    /// whether it was found -- unlike `seek_ceil`/`seek_floor`, short-circuits
    /// as soon as a byte fails to match rather than continuing to search for
    /// a nearby key.
    // ARITH: `rewind_prefix` leaves `upto` at 0 (handled by the early return
    // below) or at least 1, and it only ever increases from there.
    #[allow(clippy::arithmetic_side_effects)]
    fn do_seek_exact(&mut self) -> Result<bool> {
        self.rewind_prefix()?;
        if self.upto == 0 {
            // Degenerate, key-less FST -- see `rewind_prefix`'s guard.
            return Ok(false);
        }
        let mut arc = self.arcs[self.upto - 1].clone();
        let mut target_label = self.get_target_label();
        loop {
            match self.fst.find_target_arc(target_label, &arc, &mut self.r)? {
                None => {
                    let first = self.fst.read_first_target_arc(&arc, &mut self.r)?;
                    self.set_arc(self.upto, first);
                    return Ok(false);
                }
                Some(next_arc) => {
                    self.set_arc(self.upto, next_arc.clone());
                    let cum_output = output_add(&self.outputs[self.upto - 1], next_arc.output());
                    self.set_output(self.upto, cum_output);
                    if target_label == END_LABEL {
                        return Ok(true);
                    }
                    self.set_label(self.upto, target_label);
                    self.check_descent(&next_arc)?;
                    self.upto += 1;
                    target_label = self.get_target_label();
                    arc = next_arc;
                }
            }
        }
    }

    /// Reads out `(key, output)` at the enum's current position, or `None`
    /// if the last seek found nothing (`self.upto == 0`) -- shared tail of
    /// `seek_ceil`/`seek_floor` (`BytesRefFSTEnum.setResult`). Label-domain
    /// (`Vec<i32>` keys); the byte-keyed public methods below narrow this to
    /// `Vec<u8>` at their own boundary.
    fn current_result(&self) -> Option<(Vec<i32>, Vec<u8>)> {
        if self.upto == 0 {
            None
        } else {
            Some((
                self.labels[1..self.upto].to_vec(),
                self.outputs[self.upto].clone(),
            ))
        }
    }

    /// `FSTEnum.doNext` + `BytesRefFSTEnum.setResult`: advance to the next
    /// accepted key in ascending order, or `Ok(None)` once every key has
    /// been visited. Label-domain; see `current_result`.
    fn advance(&mut self) -> Result<Option<(Vec<i32>, Vec<u8>)>> {
        if self.done {
            return Ok(None);
        }
        if self.upto == 0 {
            let root = self.arcs[0].clone();
            if !root.is_final() && !target_has_arcs(&root) {
                // Degenerate FST that accepts no keys at all (not even the
                // empty string) -- `read_first_target_arc` would otherwise
                // try to dereference a nonexistent node at address 0/below.
                // Real Lucene-written FSTs never hit this (every FST accepts
                // at least one key), but this port's `build_fst` allows
                // constructing one from an empty key set (see
                // `build_fst_empty_key_set_never_accepts_anything`).
                self.done = true;
                return Ok(None);
            }
            self.upto = 1;
            let first = self.fst.read_first_target_arc(&root, &mut self.r)?;
            self.set_arc(1, first);
        } else {
            // ARITH: see the comment inside the loop -- `upto >= 1` before every
            // decrement.
            #[allow(clippy::arithmetic_side_effects)]
            loop {
                // ARITH: `upto >= 1` on entry (the `upto == 0` branch above
                // returned), and the loop returns the moment it hits 0.
                if self.arcs[self.upto].is_last() {
                    self.upto -= 1;
                    if self.upto == 0 {
                        self.done = true;
                        return Ok(None);
                    }
                } else {
                    break;
                }
            }
            let cur = self.arcs[self.upto].clone();
            let next = self.fst.read_next_arc(&cur, &mut self.r)?;
            self.set_arc(self.upto, next);
        }
        self.push_first()?;
        let key = self.labels[1..self.upto].to_vec();
        let output = self.outputs[self.upto].clone();
        Ok(Some((key, output)))
    }

    // --- Label-domain public API (any `INPUT_TYPE`) -------------------------

    /// Label-domain equivalent of `seek_ceil`, usable for any `INPUT_TYPE`.
    ///
    /// **One deliberate divergence from `BytesRefFSTEnum`**, shared by
    /// `seek_floor_labels`/`seek_exact_labels`: a seek that finds nothing
    /// leaves this enum *exhausted*, so a following `next()` returns `None`.
    /// Java leaves `upto == 0` instead, which `doNext` reads as "not started
    /// yet" and so restarts enumeration from the very first key. Restarting
    /// is a surprising thing for a Rust `Iterator` to do after yielding
    /// `None`, and no caller here wants it; the fused behaviour is the
    /// intentional choice, not an oversight.
    pub fn seek_ceil_labels(&mut self, target: &[i32]) -> Result<Option<(Vec<i32>, Vec<u8>)>> {
        self.target = target.to_vec();
        self.target_length = target.len();
        self.do_seek_ceil()?;
        let result = self.current_result();
        self.done = result.is_none();
        Ok(result)
    }

    /// Label-domain equivalent of `seek_floor`, usable for any `INPUT_TYPE`.
    pub fn seek_floor_labels(&mut self, target: &[i32]) -> Result<Option<(Vec<i32>, Vec<u8>)>> {
        self.target = target.to_vec();
        self.target_length = target.len();
        self.do_seek_floor()?;
        let result = self.current_result();
        self.done = result.is_none();
        Ok(result)
    }

    /// Label-domain equivalent of `seek_exact`, usable for any `INPUT_TYPE`.
    pub fn seek_exact_labels(&mut self, target: &[i32]) -> Result<Option<(Vec<i32>, Vec<u8>)>> {
        self.target = target.to_vec();
        self.target_length = target.len();
        let found = self.do_seek_exact()?;
        if found {
            let result = self.current_result();
            self.done = result.is_none();
            Ok(result)
        } else {
            self.done = true;
            Ok(None)
        }
    }

    /// Label-domain equivalent of the `Iterator` impl below, usable for any
    /// `INPUT_TYPE`: advance to the next accepted key in ascending order, or
    /// `None` once every key has been visited (fused, like `Iterator`).
    #[allow(clippy::should_implement_trait)]
    pub fn next_labels(&mut self) -> Option<Result<(Vec<i32>, Vec<u8>)>> {
        match self.advance() {
            Ok(Some(pair)) => Some(Ok(pair)),
            Ok(None) => None,
            Err(e) => Some(Err(e)),
        }
    }

    // --- Byte-domain public API (`BYTE1` FSTs only -- see `Fst::iter`) ------

    /// Port of `BytesRefFSTEnum.seekCeil`: seeks to the smallest accepted
    /// key `>=` `target`, returning `(key, output)` if one exists. Only
    /// meaningful for `BYTE1` FSTs (see `seek_ceil_labels` for other
    /// `INPUT_TYPE`s); `Fst::iter`'s guard is what makes narrowing the
    /// label-domain result back to `u8` here lossless.
    pub fn seek_ceil(&mut self, target: &[u8]) -> Result<Option<(Vec<u8>, Vec<u8>)>> {
        let target: Vec<i32> = target.iter().map(|&b| b as i32).collect();
        Ok(self
            .seek_ceil_labels(&target)?
            .map(|(key, output)| (labels_to_bytes(&key), output)))
    }

    /// Port of `BytesRefFSTEnum.seekFloor`: seeks to the largest accepted key
    /// `<=` `target`, returning `(key, output)` if one exists. Only
    /// meaningful for `BYTE1` FSTs; see `seek_ceil`'s doc comment.
    pub fn seek_floor(&mut self, target: &[u8]) -> Result<Option<(Vec<u8>, Vec<u8>)>> {
        let target: Vec<i32> = target.iter().map(|&b| b as i32).collect();
        Ok(self
            .seek_floor_labels(&target)?
            .map(|(key, output)| (labels_to_bytes(&key), output)))
    }

    /// Port of `BytesRefFSTEnum.seekExact`: seeks to exactly `target`,
    /// returning `(target, output)` if the FST accepts it, else `None` --
    /// short-circuits as soon as a byte fails to match, unlike
    /// `seek_ceil`/`seek_floor`. Unlike `Fst::seek_exact` (a stateless
    /// convenience wrapper around `Fst::get`), this maintains `FstEnum`'s
    /// per-depth arc/output stack so a subsequent `next()` call can resume
    /// ordered enumeration from the found key. Only meaningful for `BYTE1`
    /// FSTs; see `seek_ceil`'s doc comment.
    pub fn seek_exact(&mut self, target: &[u8]) -> Result<Option<(Vec<u8>, Vec<u8>)>> {
        let target: Vec<i32> = target.iter().map(|&b| b as i32).collect();
        Ok(self
            .seek_exact_labels(&target)?
            .map(|(key, output)| (labels_to_bytes(&key), output)))
    }
}

/// Narrows a label-domain key back to bytes for the `BYTE1`-only public API
/// (`FstEnum::seek_ceil`/`seek_floor`/`seek_exact`, `Iterator for FstEnum`).
///
/// A `BYTE1` FST's *stored* labels are single bytes, but a direct-addressing
/// or continuous node's labels are **not stored** -- they are
/// `firstLabel + arcIdx` -- and a corrupt label range could exceed 255, which
/// an `as u8` would truncate (260 -> 4) into a plausible wrong key. The guard
/// is at the two places the label is formed rather than here:
/// `check_label_range` bounds `firstLabel + numArcs` by the
/// `INPUT_TYPE`'s alphabet, and `bit_table_next_bit_set` refuses an index
/// past `numArcs`. Every `arcIdx` is then in `0..numArcs`, so every label of a
/// `BYTE1` FST is in `0..=255` and this narrowing is total -- and free, which
/// a per-key check on this path is not (one measured at 7% of a full
/// enumeration).
fn labels_to_bytes(labels: &[i32]) -> Vec<u8> {
    debug_assert!(labels.iter().all(|&l| (0..=255).contains(&l)));
    labels.iter().map(|&l| l as u8).collect()
}

impl Iterator for FstEnum<'_, '_> {
    type Item = Result<(Vec<u8>, Vec<u8>)>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.advance() {
            Ok(Some((key, output))) => Some(Ok((labels_to_bytes(&key), output))),
            Ok(None) => None,
            Err(e) => Some(Err(e)),
        }
    }
}

// --- FST construction (simplified FSTCompiler) -----------------------------
//
// Real Lucene's `FSTCompiler` (`org.apache.lucene.util.fst.FSTCompiler`) is an
// *incremental*, memory-bounded builder: it consumes keys one at a time in
// sorted order, keeps only a stack of `UnCompiledNode`s for the current key's
// path, "freezes" (compiles to bytes) each node as soon as it's known no
// longer to change, and de-duplicates frozen nodes via a node hash table so
// that any suffix shared by two or more keys is written to the byte store
// exactly once (suffix sharing / minimization). That incremental,
// hash-consing algorithm is a large undertaking on its own.
//
// **What's implemented here instead** is a simpler, from-scratch construction
// that is *correct* (produces bytes `Fst::read`/`Fst::get` decode correctly)
// but *not* minimal:
//
// 1. Build an ordinary in-memory trie from the full sorted `(key, output)`
//    sequence (not incremental/streaming -- the whole key set is held in
//    memory as a trie before any bytes are written).
// 2. Serialize the trie bottom-up (post-order) into the exact list-encoded
//    ("variable length arc") node/arc byte format `Fst::read`'s
//    `read_arc`/`find_target_arc` already knows how to parse -- see the
//    module doc for the field layout and `tests::build_single_key_fst`
//    (which this generalizes to a full trie with branching) for the
//    address-ordering contract those functions rely on.
//
// Deliberately **not** done, all deferred to a future slice along with real
// `FSTCompiler` itself:
// - **Incremental/streaming construction.** Real `FSTCompiler` consumes keys
//   one at a time and only ever holds the current key's path of
//   not-yet-frozen nodes in memory, freezing (and dedup-checking) each node
//   as soon as no further key can extend it. This builder instead
//   materializes the entire trie in memory upfront and serializes it
//   bottom-up in one pass. **Suffix sharing / minimization itself is done**
//   (see `NodeHash`/`build_node` below): two keys that share a *suffix* (not
//   just a prefix) -- e.g. `"cat"` and `"bat"` sharing the final `"at"` --
//   get a single shared copy of that suffix's nodes in the byte store, same
//   as real `FSTCompiler`'s node hash table would produce, just discovered by
//   hash-consing the whole tree at once rather than incrementally.
// - **Fixed-length arc nodes** (`ARCS_FOR_BINARY_SEARCH`/`_DIRECT_ADDRESSING`/
//   `_CONTINUOUS`) and the `BIT_TARGET_NEXT` adjacent-node compaction: every
//   arc here writes an explicit `vlong` target rather than relying on
//   physical adjacency, and every node is list-encoded. Larger output, but
//   still within what `Fst::read` supports (which itself doesn't decode the
//   array-node encodings -- see the module doc).
// - **Output pushing down shared prefixes.** Real Lucene's outputs
//   (`ByteSequenceOutputs`) get pushed as far toward the root as possible so
//   that arcs shared by many keys carry a common output prefix once. This
//   builder instead puts each key's *entire* output on the single arc
//   leading into its accepting node (mirroring the existing hand-built
//   `tests::build_single_key_fst` shape), relying on `Fst::get`'s
//   `output_add` accumulation to still assemble the right final bytes.
// - Only `ByteSequenceOutputs`/`INPUT_TYPE.BYTE1`/on-heap, matching the
//   reader's own scope.
//
// A node in the trie: `children` keyed by the next input byte (a `BTreeMap`
// so iteration order is the ascending label order the wire format requires),
// plus whether this node itself is an accepting state for some key and, if
// so, that key's output.
#[derive(Debug, Default)]
struct TrieNode {
    children: std::collections::BTreeMap<u8, TrieNode>,
    is_final: bool,
    final_output: Vec<u8>,
}

impl TrieNode {
    fn insert(&mut self, key: &[u8], output: Vec<u8>) {
        let mut node = self;
        for &b in key {
            node = node.children.entry(b).or_default();
        }
        node.is_final = true;
        node.final_output = output;
    }
}

/// Appends one arc's bytes to `bytes` given its fields in logical (forward
/// read) order, returning that arc's address (the position `find_target_arc`
/// would `set_position` to land on this arc's flags byte). See the module
/// doc / `tests::append_arc_logical` for why the physical bytes are the
/// reverse of `logical`.
fn append_arc_logical(bytes: &mut Vec<u8>, logical: &[u8]) -> i64 {
    for &b in logical.iter().rev() {
        bytes.push(b);
    }
    // ARITH: `build_node` never appends an empty `logical` (every arc carries
    // at least a flags byte), so `bytes` is non-empty here.
    #[allow(clippy::arithmetic_side_effects)]
    let addr = (bytes.len() - 1) as i64;
    addr
}

/// One arc's structural identity for node-hash-consing purposes: the exact
/// fields that determine the bytes `build_node` would write for it (label,
/// finality + final output, and -- for non-stop arcs -- the *already
/// resolved* address of the target node). Two nodes whose full ordered arc
/// list compares equal are guaranteed to serialize to byte-identical arcs
/// (see `build_node`), so this doubles as the dedup key.
type ArcSignature = (u8, bool, Vec<u8>, Option<i64>);

/// Real `FSTCompiler`'s `NodeHash` maps a *frozen* node's arc signature to
/// the address it was already written at, so that a later node with an
/// identical signature (typically a shared suffix reached via a different
/// prefix) reuses that address instead of re-serializing a duplicate copy.
/// This is the same table, keyed on the whole-tree-at-once builder's
/// equivalent of "frozen node": one whose subtree has already been fully
/// recursively resolved to final addresses.
type NodeHash = std::collections::HashMap<Vec<ArcSignature>, i64>;

/// Serializes one trie node's children into the byte store, recursing into
/// any child that itself has further children first (post-order: a child
/// node's address must be known before the arc pointing at it can write its
/// `vlong` target). Returns this node's own address (the address of the arc
/// for its *smallest*-labeled child -- see the module doc's node/arc address
/// ordering contract). Panics only if called on a node with no children
/// (callers only recurse into/start from non-empty nodes).
///
/// Before writing any new bytes for this node, checks `node_hash` for an
/// already-frozen node with the exact same ordered arc signature (same
/// labels, same finality/final-outputs, same child target addresses) and, if
/// found, returns its existing address unchanged -- this is the suffix
/// sharing / minimization step: any suffix reachable from two or more
/// different paths in the trie collapses to a single copy in the byte store.
fn build_node(node: &TrieNode, bytes: &mut Vec<u8>, node_hash: &mut NodeHash) -> i64 {
    let labels: Vec<u8> = node.children.keys().copied().collect();
    assert!(!labels.is_empty(), "build_node requires at least one arc");

    let mut child_addr: std::collections::HashMap<u8, i64> = std::collections::HashMap::new();
    for &label in &labels {
        let child = &node.children[&label];
        if !child.children.is_empty() {
            child_addr.insert(label, build_node(child, bytes, node_hash));
        }
    }

    // Signature in ascending label order (independent of the descending
    // order arcs are physically appended in below) -- just a lookup key, so
    // canonical ordering only needs to be consistent across calls.
    let signature: Vec<ArcSignature> = labels
        .iter()
        .map(|&label| {
            let child = &node.children[&label];
            let has_children = !child.children.is_empty();
            (
                label,
                child.is_final,
                child.final_output.clone(),
                if has_children {
                    Some(child_addr[&label])
                } else {
                    None
                },
            )
        })
        .collect();

    if let Some(&existing_addr) = node_hash.get(&signature) {
        return existing_addr;
    }

    // Arcs must be appended in *descending* label order: the first one
    // appended lands at the lowest address and is read *last* when scanning
    // ascending from the node's start address, so it's the one carrying
    // BIT_LAST_ARC.
    let mut node_addr = 0i64;
    for (i, &label) in labels.iter().rev().enumerate() {
        let child = &node.children[&label];

        let mut flags = 0u8;
        if i == 0 {
            flags |= BIT_LAST_ARC;
        }
        if child.is_final {
            flags |= BIT_FINAL_ARC;
            if !child.final_output.is_empty() {
                flags |= BIT_ARC_HAS_FINAL_OUTPUT;
            }
        }

        let has_children = !child.children.is_empty();
        if !has_children {
            flags |= BIT_STOP_NODE;
        }

        let mut logical = Vec::new();
        logical.push(flags);
        logical.push(label);
        if flag(flags, BIT_ARC_HAS_FINAL_OUTPUT) {
            write_vint(&mut logical, child.final_output.len() as i32);
            logical.extend_from_slice(&child.final_output);
        }
        if has_children {
            write_vlong(&mut logical, child_addr[&label]);
        }

        node_addr = append_arc_logical(bytes, &logical);
    }

    node_hash.insert(signature, node_addr);
    node_addr
}

/// Errors specific to `build_fst`'s input contract.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum BuildError {
    #[error("build_fst requires keys in strictly ascending sorted order (duplicate or out-of-order key at index {index})")]
    NotSorted { index: usize },
}

/// A minimal from-scratch equivalent of real Lucene's `FSTCompiler` for the
/// `ByteSequenceOutputs`/`INPUT_TYPE.BYTE1`/on-heap slice this module already
/// reads -- see the section doc above for exactly which of real
/// `FSTCompiler`'s behaviors (suffix sharing, output pushing, fixed-length
/// arc nodes) are and aren't reproduced.
///
/// `entries` must already be sorted in strictly ascending key order (as real
/// `FSTCompiler.add` requires of its caller) -- checked, not silently
/// tolerated.
pub fn build_fst(entries: &[(Vec<u8>, Vec<u8>)]) -> std::result::Result<Fst<'static>, BuildError> {
    // ARITH: `i` starts at 1.
    #[allow(clippy::arithmetic_side_effects)]
    for i in 1..entries.len() {
        if entries[i - 1].0 >= entries[i].0 {
            return Err(BuildError::NotSorted { index: i });
        }
    }

    let mut empty_output: Option<Vec<u8>> = None;
    let mut root = TrieNode::default();
    for (key, output) in entries {
        if key.is_empty() {
            empty_output = Some(output.clone());
        } else {
            root.insert(key, output.clone());
        }
    }

    let mut bytes = Vec::new();
    let start_node = if root.children.is_empty() {
        // No non-empty keys at all: an empty body with a start node address
        // of 0 is never dereferenced by `Fst::get` since it only calls
        // `find_target_arc` when `target_has_arcs` (`target > 0`), and 0
        // fails that check.
        0
    } else {
        let mut node_hash = NodeHash::new();
        build_node(&root, &mut bytes, &mut node_hash)
    };

    Ok(Fst {
        metadata: FstMetadata {
            input_type: InputType::Byte1,
            empty_output,
            start_node,
            version: VERSION_CURRENT,
            num_bytes: bytes.len() as i64,
        },
        bytes: FstBytes::Owned(bytes),
    })
}

/// Serializes a built `Fst` back into the exact on-disk byte layout
/// `Fst::read` parses (codec header + `FSTMetadata.save` fields + raw body),
/// so a caller can round-trip through the real, unmodified `Fst::read` entry
/// point rather than only exercising the in-memory `Fst` this module's own
/// builder happens to construct directly.
pub fn write_fst(fst: &Fst<'_>) -> Vec<u8> {
    let mut out = Vec::new();
    codec_util::write_header(&mut out, FILE_FORMAT_NAME, VERSION_CURRENT);

    match &fst.metadata.empty_output {
        Some(output) => {
            out.push(1);
            // Mirror of `read_fst_metadata_prefix`'s decode (and of
            // `FSTMetadata.save`): serialize through `writeFinalOutput`'s
            // framing -- `vint(len)` then the payload -- reverse the whole
            // serialized buffer, then length-prefix *that*.
            // ARITH: an in-memory output plus the five bytes of its longest
            // possible `vint` length prefix.
            #[allow(clippy::arithmetic_side_effects)]
            let mut serialized = Vec::with_capacity(output.len() + 5);
            write_vint(&mut serialized, output.len() as i32);
            serialized.extend_from_slice(output);
            serialized.reverse();
            write_vint(&mut out, serialized.len() as i32);
            out.extend_from_slice(&serialized);
        }
        None => out.push(0),
    }

    out.push(match fst.metadata.input_type {
        InputType::Byte1 => 0,
        InputType::Byte2 => 1,
        InputType::Byte4 => 2,
    });
    write_vlong(&mut out, fst.metadata.start_node);
    write_vlong(&mut out, fst.metadata.num_bytes);
    out.extend_from_slice(&fst.bytes);
    out
}

// --- Typed output values (`PositiveIntOutputs`, `PairOutputs`) ------------
//
// Everything above this point stores exactly one output type on the wire:
// a raw byte sequence (`ByteSequenceOutputs`, matching `Fst::get`'s
// `Result<Option<Vec<u8>>>` and `build_fst`'s `(Vec<u8>, Vec<u8>)` entries).
// There is no generic `Outputs<T>`/arc-output abstraction underneath the
// reader or `build_node` -- both are hardcoded to `Vec<u8>` -- and this
// builder never pushes a shared output prefix up toward the root (see the
// "FST construction" section doc above: "Output pushing down shared
// prefixes ... deferred"). That absence of output-pushing is exactly what
// makes the two typed output values below simple to add correctly: each
// key's *entire* output value lives on the single arc leading to its
// accepting node (same as today), so combining "two arcs from different
// nodes get merged/shared" never has to reconcile an output already pushed
// partway down one path with a different one pushed down another -- there
// is no pushed output to reconcile. `build_node`'s existing `NodeHash`
// dedup keys on each node's exact arc signature, which includes the
// (encoded) final output bytes -- so two subtrees whose accepting arcs
// carry different typed output values naturally get different signatures
// and are never incorrectly merged; two subtrees whose typed values are
// equal (and whose deeper structure also matches) naturally *are* merged,
// which is the correct, space-saving behavior and is exercised by
// `tests::pair_outputs_shared_suffix_nodes_keep_distinct_first_components`
// below.
//
// Rather than threading a generic output type through `Fst`/`TrieNode`/
// `build_node` (a much larger change touching the reader too), typed
// support is layered *on top of* the existing byte-sequence machinery: an
// `Outputs` trait describes how to encode a typed value to/from the
// `Vec<u8>` this module already reads and writes, and `build_fst_typed`/
// `Fst::get_typed` are thin encode/decode wrappers around `build_fst`/
// `Fst::get`. This intentionally does not attempt real Lucene's separate
// `add`/`common`/`subtract` operations (`Outputs.add/common/subtract`) --
// those exist in real `FSTCompiler` purely to support output-pushing
// (finding/removing a shared output prefix so it can be hoisted onto a
// shared arc); since this builder never pushes outputs, there is no shared
// prefix to find or subtract, so those operations would have no caller and
// are omitted rather than stubbed out dishonestly.

/// A typed FST output value, encoded to/from the raw `Vec<u8>` this module's
/// `Fst`/`build_fst` already store on the wire.
///
/// **Scope, stated plainly: this layer is not wire-compatible with real
/// Lucene's `Outputs<T>` implementations, and is not meant to be.** Lucene
/// writes each arc's output through the `Outputs<T>` in force for that FST,
/// so a `PositiveIntOutputs` FST stores a bare `vlong` per arc and a
/// `PairOutputs<A, B>` FST stores `A`'s encoding immediately followed by
/// `B`'s. This port's reader is hardcoded to `ByteSequenceOutputs`' framing
/// (a `vint` length then the payload) at the arc level, and the codecs below
/// encode *inside* that payload -- so an FST built here with
/// `build_fst_typed::<PositiveIntOutputs>` is a valid `FST<BytesRef>`, not a
/// valid `FST<Long>`, and real Lucene would have to read it as the former.
/// That is fine for the only consumer, `suggest.rs`, which both writes and
/// reads its own FSTs; it would not be fine for reading a real Lucene
/// suggester file, which nothing here does. Making the reader generic over
/// `Outputs` is the prerequisite for that, and is deliberately not done
/// here.
///
/// `Self::Value` mirrors real
/// Lucene's `T` type parameter of `Outputs<T>`; `Self` itself is a
/// zero-sized marker type selecting *which* codec applies (matching real
/// Lucene's `Outputs<T>` being a singleton descriptor object distinct from
/// the values `T` it produces).
pub trait Outputs {
    /// The typed output value this codec encodes/decodes, e.g. `i64` for
    /// `PositiveIntOutputs` or `Vec<u8>` for `ByteSequenceOutputs`.
    type Value: Clone + PartialEq;

    /// `Outputs.getNoOutput()`: the identity/absent value (real Lucene's
    /// `NO_OUTPUT` sentinel), e.g. `0i64` or the empty byte sequence. Must
    /// encode to the empty `Vec<u8>` so it lines up with this module's
    /// existing "empty output bytes -> `BIT_ARC_HAS_(FINAL_)OUTPUT` unset"
    /// convention in `build_node`/`read_arc` -- i.e. a key whose typed
    /// output is `zero()` costs no extra output bytes on the wire, same as
    /// today's plain `Vec<u8>` empty-output keys.
    fn zero() -> Self::Value;

    /// Encode `value` to the byte sequence `build_fst`/`Fst::get` store and
    /// return respectively.
    fn encode(value: &Self::Value) -> Vec<u8>;

    /// Decode a byte sequence that `encode` is *supposed* to have produced.
    ///
    /// Fallible, because the bytes reach here off disk: `Fst::get_typed`
    /// hands over whatever payload the arc it walked carried, and a corrupt
    /// (or simply truncated) FST body can name a length or a continuation
    /// run that no `encode` would ever emit. Returning `Result` is what lets
    /// those become a typed `Error::Corrupt` instead of a slice-index or
    /// shift panic crossing the FFI boundary.
    fn decode(bytes: &[u8]) -> Result<Self::Value>;
}

/// Forward (non-reversed) `vint` reader over a plain slice -- the encode
/// direction's counterpart to `write_vint` (also forward) and distinct from
/// `BytesReader::read_vint`, which walks its cursor *backwards* to match the
/// on-disk FST body's own reverse-read convention. Typed output *values*
/// (this section) are encoded/decoded independently of that body layout, in
/// ordinary forward byte order, since they're just the payload `build_node`
/// copies verbatim into an arc's final-output bytes. Returns `(value,
/// bytes_consumed)`.
// ARITH: `idx` counts bytes consumed from `bytes` and is bounded by its
// length (every read goes through `get`); `shift` is compared against 28
// before every increment, so it never leaves 7..=35 and the `wrapping_shl`
// never sees an out-of-range amount.
#[allow(clippy::arithmetic_side_effects)]
fn read_vint_forward(bytes: &[u8]) -> Result<(i32, usize)> {
    let mut idx = 0usize;
    let mut b = *bytes.first().ok_or_else(|| malformed_output("vInt"))?;
    idx += 1;
    let mut v = (b & 0x7f) as i32;
    let mut shift = 7;
    while b & 0x80 != 0 {
        if shift > 28 {
            return Err(malformed_output("vInt"));
        }
        b = *bytes.get(idx).ok_or_else(|| malformed_output("vInt"))?;
        idx += 1;
        v |= ((b & 0x7f) as i32).wrapping_shl(shift);
        shift += 7;
    }
    Ok((v, idx))
}

/// The one error every typed-output decoder reports: the payload is not
/// something this module's `encode` could have produced.
fn malformed_output(kind: &str) -> Error {
    Error::Corrupt(format!("malformed {kind} in FST output payload"))
}

/// Forward `vlong` reader, `read_vint_forward`'s 64-bit counterpart (used by
/// `PositiveIntOutputs::decode`).
// ARITH: as `read_vint_forward`, with `shift` bounded at 64.
#[allow(clippy::arithmetic_side_effects)]
fn read_vlong_forward(bytes: &[u8]) -> Result<(i64, usize)> {
    let mut idx = 0usize;
    let mut b = *bytes.first().ok_or_else(|| malformed_output("vLong"))?;
    idx += 1;
    let mut v = (b & 0x7f) as i64;
    let mut shift = 7;
    while b & 0x80 != 0 {
        if shift >= 64 {
            return Err(malformed_output("vLong"));
        }
        b = *bytes.get(idx).ok_or_else(|| malformed_output("vLong"))?;
        idx += 1;
        v |= ((b & 0x7f) as i64).wrapping_shl(shift);
        shift += 7;
    }
    Ok((v, idx))
}

/// Port of `PositiveIntOutputs`: an FST output type whose value is a single
/// non-negative `i64` (real Lucene rejects negative values; this port
/// mirrors that with a `debug_assert`, since silently wrapping/rejecting in
/// release builds isn't this slice's concern and every current caller is a
/// test). `NO_OUTPUT` is `0`, matching real Lucene, which is why it encodes
/// to the empty byte sequence (see `Outputs::zero`'s doc comment).
pub struct PositiveIntOutputs;

impl Outputs for PositiveIntOutputs {
    type Value = i64;

    fn zero() -> i64 {
        0
    }

    fn encode(value: &i64) -> Vec<u8> {
        debug_assert!(
            *value >= 0,
            "PositiveIntOutputs requires non-negative values, got {value}"
        );
        if *value == 0 {
            return Vec::new();
        }
        let mut out = Vec::new();
        write_vlong(&mut out, *value);
        out
    }

    fn decode(bytes: &[u8]) -> Result<i64> {
        if bytes.is_empty() {
            return Ok(0);
        }
        let (value, consumed) = read_vlong_forward(bytes)?;
        if consumed != bytes.len() {
            return Err(malformed_output("vLong"));
        }
        Ok(value)
    }
}

/// Port of `ByteSequenceOutputs`, expressed as an `Outputs` impl: the
/// identity codec, matching what `build_fst`/`Fst::get` already do directly
/// for plain `(Vec<u8>, Vec<u8>)` callers. Exists so `PairOutputs<A, B>` can
/// use it as either component type (e.g. `PairOutputs<PositiveIntOutputs,
/// ByteSequenceOutputs>`, the shape real Lucene's synonym/suggest
/// infrastructure uses for `PairOutputs<Long, BytesRef>`).
pub struct ByteSequenceOutputs;

impl Outputs for ByteSequenceOutputs {
    type Value = Vec<u8>;

    fn zero() -> Vec<u8> {
        Vec::new()
    }

    fn encode(value: &Vec<u8>) -> Vec<u8> {
        value.clone()
    }

    fn decode(bytes: &[u8]) -> Result<Vec<u8>> {
        Ok(bytes.to_vec())
    }
}

/// Port of `PairOutputs.Pair<A, B>`: a value combining two independent
/// component output values.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Pair<A, B> {
    pub first: A,
    pub second: B,
}

/// Port of `PairOutputs<A, B>`: combines two independent `Outputs` codecs
/// (`A`, `B`) into one `Outputs` impl over `Pair<A::Value, B::Value>`, so a
/// single FST can map each key to both an `A`-typed and a `B`-typed output
/// simultaneously (e.g. a weight *and* a payload per key, as real Lucene's
/// synonym/suggest FSTs do with `PairOutputs<Long, BytesRef>`).
///
/// Encoding: if both components encode to the empty byte sequence (i.e. the
/// pair *is* `zero()`), the whole pair also encodes to the empty byte
/// sequence -- preserving the "no output bytes for a no-op key" convention
/// `Outputs::zero`'s doc comment describes, rather than always paying a
/// length-prefix byte even for an all-zero pair. Otherwise: the first
/// component's encoded length (`vint`), then its encoded bytes, then the
/// second component's encoded bytes filling the remainder (no length prefix
/// needed for the second component since it's always everything left).
pub struct PairOutputs<A, B>(std::marker::PhantomData<(A, B)>);

impl<A: Outputs, B: Outputs> Outputs for PairOutputs<A, B> {
    type Value = Pair<A::Value, B::Value>;

    fn zero() -> Self::Value {
        Pair {
            first: A::zero(),
            second: B::zero(),
        }
    }

    fn encode(value: &Self::Value) -> Vec<u8> {
        let first = A::encode(&value.first);
        let second = B::encode(&value.second);
        if first.is_empty() && second.is_empty() {
            return Vec::new();
        }
        let mut out = Vec::new();
        write_vint(&mut out, first.len() as i32);
        out.extend_from_slice(&first);
        out.extend_from_slice(&second);
        out
    }

    fn decode(bytes: &[u8]) -> Result<Self::Value> {
        if bytes.is_empty() {
            return Ok(Self::zero());
        }
        let (first_len, header_len) = read_vint_forward(bytes)?;
        // `first_len` is a `vint` off disk indexing the payload: negative
        // through `as usize` it becomes ~2^64, and positive it can simply run
        // past the end. Both are slice-index panics, so the split point is
        // formed with `checked_add` and taken with `get`.
        let split = usize::try_from(first_len)
            .ok()
            .and_then(|n| header_len.checked_add(n))
            .ok_or_else(|| malformed_output("pair output length"))?;
        let first_bytes = bytes
            .get(header_len..split)
            .ok_or_else(|| malformed_output("pair output length"))?;
        let second_bytes = bytes
            .get(split..)
            .ok_or_else(|| malformed_output("pair output length"))?;
        Ok(Pair {
            first: A::decode(first_bytes)?,
            second: B::decode(second_bytes)?,
        })
    }
}

/// Typed equivalent of `build_fst`: encodes each entry's `O::Value` output
/// via `O::encode` before delegating to the existing byte-sequence builder.
/// `entries` must be sorted in strictly ascending key order, exactly like
/// `build_fst`.
pub fn build_fst_typed<O: Outputs>(
    entries: &[(Vec<u8>, O::Value)],
) -> std::result::Result<Fst<'static>, BuildError> {
    let byte_entries: Vec<(Vec<u8>, Vec<u8>)> = entries
        .iter()
        .map(|(key, value)| (key.clone(), O::encode(value)))
        .collect();
    build_fst(&byte_entries)
}

impl<'a> Fst<'a> {
    /// Typed equivalent of `Fst::get`: looks up `key` and decodes its
    /// accumulated output bytes via `O::decode`.
    pub fn get_typed<O: Outputs>(&self, key: &[u8]) -> Result<Option<O::Value>> {
        match self.get(key)? {
            None => Ok(None),
            Some(bytes) => Ok(Some(O::decode(&bytes)?)),
        }
    }
}

// --- `IntsRefFSTEnum` (`IntsRef`-keyed ordered enumeration) ---------------
//
// Real Lucene's `IntsRefFSTEnum` is `BytesRefFSTEnum`'s sibling: both extend
// the same abstract `FSTEnum` base class (shared `pushFirst`/`pushLast`/
// `rewindPrefix`/`doSeekCeil`/`doSeekFloor`/`doSeekExact`/`doNext` algorithm),
// differing only in the *type* of one key element -- a `BytesRef`'s elements
// are bytes (0..=255), an `IntsRef`'s elements are full `int`s -- and in how
// a finished walk's accumulated `labels` array is turned back into the
// caller-facing key type (`BytesRef` vs `IntsRef`). Concretely,
// `IntsRefFSTEnum` is used over FSTs whose `FST.INPUT_TYPE` is `BYTE4`: each
// `int` key element is stored as its own arc label (a genuinely 32-bit-wide
// alphabet), which is how e.g. `SynonymMap`'s FST works -- token ordinals,
// not raw bytes, are the arc labels.
//
// This port's `Fst`/`Arc`/`read_label` et al. do carry labels as `i32`
// end-to-end (see `Arc::label`, `read_label`'s return type) -- so nothing
// forces bytes specifically at the *type* level. But the on-disk node/arc
// format this module's reader (`read_label`) and builder (`build_node`/
// `TrieNode`) actually implement is hardcoded to single-byte labels:
// `read_label` asserts `input_type == Byte1` and always does a plain
// `read_byte`, and `TrieNode::children` is keyed by `u8`. Real `BYTE4`/
// `BYTE2` FSTs (`FST.readLabel`'s other two branches, which decode a label
// as a 4-byte or 2-byte value with its own varint-like packing) are outside
// this reader's scope entirely (see the module doc and `Fst::get`'s guard).
// Building a genuinely `BYTE4`-keyed FST variant -- widening `TrieNode`,
// `build_node`'s arc-signature hashing, `read_label`/`find_target_arc`'s
// direct-addressing and continuous-node arithmetic, etc., all to accept a
// real 32-bit alphabet -- is substantially larger (it's effectively
// rebuilding the node/arc codec a second time for a wider label width).
//
// **What's implemented here instead**: `IntsRefFstEnum` presents the real
// `IntsRefFSTEnum` *API* (ordered enumeration + `seekCeil`/`seekFloor`/
// `seekExact` over `Vec<i32>` keys) but is a byte-keyed `BYTE1` FST wearing
// an int-keyed face -- each `i32` key element is encoded as 4 big-endian
// bytes (`encode_ints_key`/`decode_ints_key`) at the API boundary, so the
// underlying FST this type walks is an ordinary `Fst` built by
// `build_fst_from_ints`/`build_fst`, and `IntsRefFstEnum` itself is a thin
// wrapper around the existing `FstEnum` (composition, not a duplicated
// traversal algorithm) that decodes each 4-byte group back into an `i32` on
// the way out. Big-endian encoding is exactly what makes this a legitimate
// stand-in rather than a hack: unsigned-byte-lexicographic order over the
// 4-byte encoding matches numeric order over the original `i32`s, so
// ascending-key enumeration over the encoded bytes *is* ascending-key
// enumeration over the original ints -- **provided every key element is
// non-negative** (`i32::MIN..0`'s two's-complement bit pattern sorts as
// *larger*, unsigned, than any non-negative value, which would corrupt
// ordering). This mirrors real `IntsRefFSTEnum`'s actual usage: token
// ordinals (`SynonymMap` et al.) are always non-negative, so the restriction
// costs nothing in practice. `encode_ints_key`/`build_fst_from_ints` both
// document and enforce this rather than silently misordering negative
// inputs.
//
// This is **not** real Lucene's `IntsRefFSTEnum` (which needs no such
// restriction and walks a genuinely wider-alphabet FST) -- see `docs/parity.md`
// for this scoping call spelled out.

/// Encodes one `IntsRef`-equivalent key element (`i32`) as 4 big-endian
/// bytes, so a sequence of `i32`s becomes the byte-keyed `Fst` key
/// `IntsRefFstEnum`/`build_fst_from_ints` actually walk. Big-endian is what
/// makes unsigned-byte order over the encoding match numeric order over the
/// original ints -- see the section doc above for why that only holds for
/// non-negative inputs.
///
/// # Panics
///
/// Panics (in both debug and release builds, not compiled out) if any
/// element is negative -- a negative `i32`'s two's-complement bit pattern
/// would sort as *larger*, unsigned, than any non-negative value, silently
/// corrupting the ordering invariant the whole `IntsRefFstEnum` scoping
/// depends on. Rejected loudly instead.
pub fn encode_ints_key(key: &[i32]) -> Vec<u8> {
    // ARITH: four bytes per `i32` element of an in-memory key; the product
    // is the size of the `Vec` about to hold them, so an overflow here would
    // mean the encoded key cannot exist.
    #[allow(clippy::arithmetic_side_effects)]
    let mut out = Vec::with_capacity(key.len() * 4);
    for &v in key {
        assert!(
            v >= 0,
            "encode_ints_key requires non-negative elements, got {v}"
        );
        out.extend_from_slice(&v.to_be_bytes());
    }
    out
}

/// Inverse of `encode_ints_key`: splits a byte-keyed `Fst` key back into its
/// original `i32` sequence. `bytes.len()` must be a multiple of 4 (true of
/// any key produced by `encode_ints_key`, or read back out of an FST built by
/// `build_fst_from_ints`).
///
/// Fallible, because the key is assembled from arc labels walked out of an
/// FST body: enumerating a `BYTE1` FST (or a corrupt one) as `IntsRef` keys
/// produces lengths that are not multiples of four, which is rejected rather
/// than dropping the trailing one to three bytes into a plausible shorter key.
pub fn decode_ints_key(bytes: &[u8]) -> Result<Vec<i32>> {
    if !bytes.len().is_multiple_of(4) {
        return Err(Error::Corrupt(format!(
            "IntsRef key length {} is not a multiple of 4",
            bytes.len()
        )));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|c| i32::from_be_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

/// `IntsRef`-keyed equivalent of `build_fst_typed`: each entry's key is a
/// non-negative `i32` sequence, encoded via `encode_ints_key` before
/// delegating to the existing byte-keyed builder. `entries` must be sorted
/// in strictly ascending key order (lexicographic over the `i32` sequence,
/// which -- given non-negative elements -- is also the byte-encoded entries'
/// required ascending order).
pub fn build_fst_from_ints<O: Outputs>(
    entries: &[(Vec<i32>, O::Value)],
) -> std::result::Result<Fst<'static>, BuildError> {
    let byte_entries: Vec<(Vec<u8>, O::Value)> = entries
        .iter()
        .map(|(key, value)| (encode_ints_key(key), value.clone()))
        .collect();
    build_fst_typed::<O>(&byte_entries)
}

/// Port of `IntsRefFSTEnum`, restricted (like `FstEnum`/`BytesRefFSTEnum`
/// here) to a full forward walk plus `seekCeil`/`seekFloor`/`seekExact`. See
/// the section doc above this type for the byte-keyed-FST-wearing-an-int-API
/// scoping this implementation takes, and why. Constructed via
/// `Fst::iter_ints`. Implements `Iterator<Item = Result<(Vec<i32>, Vec<u8>)>>`,
/// yielding `(key, output)` pairs in ascending key order -- mirroring
/// `FstEnum`'s `(Vec<u8>, Vec<u8>)` shape, just with an `i32`-sequence key.
pub struct IntsRefFstEnum<'f, 'a> {
    inner: FstEnum<'f, 'a>,
}

impl<'f, 'a> IntsRefFstEnum<'f, 'a> {
    /// Port of `IntsRefFSTEnum.seekCeil` (inherited, unchanged, from
    /// `FSTEnum.doSeekCeil`): seeks to the smallest accepted key `>=`
    /// `target`, returning `(key, output)` if one exists.
    pub fn seek_ceil(&mut self, target: &[i32]) -> Result<Option<(Vec<i32>, Vec<u8>)>> {
        match self.inner.seek_ceil(&encode_ints_key(target))? {
            None => Ok(None),
            Some((key, output)) => Ok(Some((decode_ints_key(&key)?, output))),
        }
    }

    /// Port of `IntsRefFSTEnum.seekFloor`: seeks to the largest accepted key
    /// `<=` `target`, returning `(key, output)` if one exists.
    pub fn seek_floor(&mut self, target: &[i32]) -> Result<Option<(Vec<i32>, Vec<u8>)>> {
        match self.inner.seek_floor(&encode_ints_key(target))? {
            None => Ok(None),
            Some((key, output)) => Ok(Some((decode_ints_key(&key)?, output))),
        }
    }

    /// Port of `IntsRefFSTEnum.seekExact`: seeks to exactly `target`,
    /// returning `(target, output)` if the FST accepts it, else `None`.
    pub fn seek_exact(&mut self, target: &[i32]) -> Result<Option<(Vec<i32>, Vec<u8>)>> {
        match self.inner.seek_exact(&encode_ints_key(target))? {
            None => Ok(None),
            Some((key, output)) => Ok(Some((decode_ints_key(&key)?, output))),
        }
    }
}

impl Iterator for IntsRefFstEnum<'_, '_> {
    type Item = Result<(Vec<i32>, Vec<u8>)>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.inner.next()? {
            Ok((key, output)) => Some(decode_ints_key(&key).map(|k| (k, output))),
            Err(e) => Some(Err(e)),
        }
    }
}

#[cfg(test)]
mod tests;
