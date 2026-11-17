//! Decoders for Lucene's `DataInput` wire primitives (vint, vlong, zigzag,
//! group-varint, little-endian fixed-width), format-compatible with Lucene 10.x.
//!
//! Design notes (see PLAN.md §3.5): this is a trait with provided methods over a
//! minimal byte-cursor core, and `SliceInput` is the monomorphized zero-copy
//! implementation used for mmap'd regions — decode loops inline fully, no dyn.

use crate::error::{Error, Result};

/// [`DataInput::read_vint`]'s byte loop, shared by the trait default and the
/// slice fast path's multi-byte fallback so the two cannot drift apart.
// ARITH: `shift` is compared against 28 before every increment, so it
// never leaves 7..=35; `wrapping_shl` already carries the shift semantics.
#[allow(clippy::arithmetic_side_effects)]
#[inline]
fn read_vint_loop<R: DataInput + ?Sized>(r: &mut R) -> Result<i32> {
    let mut b = r.read_byte()?;
    let mut v = (b & 0x7f) as i32;
    let mut shift = 7;
    while b & 0x80 != 0 {
        if shift > 28 {
            return Err(Error::MalformedVarint);
        }
        b = r.read_byte()?;
        // The final (5th) byte contributes its low 4 bits into the sign
        // area; bits above that are shifted out, matching Java's
        // `(b & 0x7F) << 28` on an `int`.
        v |= ((b & 0x7f) as i32).wrapping_shl(shift);
        shift += 7;
    }
    Ok(v)
}

/// [`DataInput::read_vlong`]'s byte loop; see [`read_vint_loop`].
// ARITH: `shift` is compared against 64 before every increment.
#[allow(clippy::arithmetic_side_effects)]
#[inline]
fn read_vlong_loop<R: DataInput + ?Sized>(r: &mut R) -> Result<i64> {
    let mut b = r.read_byte()?;
    let mut v = (b & 0x7f) as i64;
    let mut shift = 7;
    while b & 0x80 != 0 {
        if shift >= 64 {
            return Err(Error::MalformedVarint);
        }
        b = r.read_byte()?;
        v |= ((b & 0x7f) as i64).wrapping_shl(shift);
        shift += 7;
    }
    Ok(v)
}

/// [`SliceInput`]'s varints that its in-slice decoders cannot finish: one
/// within a maximum length of the buffer's end, or a malformed one.
#[cold]
#[inline(never)]
fn slice_vint_slow(r: &mut SliceInput<'_>) -> Result<i32> {
    read_vint_loop(r)
}

/// See [`slice_vint_slow`].
#[cold]
#[inline(never)]
fn slice_vlong_slow(r: &mut SliceInput<'_>) -> Result<i64> {
    read_vlong_loop(r)
}

/// See [`slice_vint_slow`].
#[cold]
#[inline(never)]
fn slice_zlong_slow(r: &mut SliceInput<'_>) -> Result<i64> {
    read_zlong_loop(r)
}

/// [`DataInput::read_zlong`]'s byte loop, shared by the trait default and
/// the slice fast path's fallback.
// ARITH: `shift` is compared against 70 after every increment.
#[allow(clippy::arithmetic_side_effects)]
#[inline]
fn read_zlong_loop<R: DataInput + ?Sized>(r: &mut R) -> Result<i64> {
    // Same varint framing as vlong but must accept the 10-byte encoding of
    // values with the top bit set.
    let mut v: u64 = 0;
    let mut shift = 0u32;
    loop {
        let b = r.read_byte()?;
        v |= ((b & 0x7f) as u64).wrapping_shl(shift);
        if b & 0x80 == 0 {
            break;
        }
        shift += 7;
        if shift >= 70 {
            return Err(Error::MalformedVarint);
        }
    }
    Ok(lucene_util::zigzag::decode(v))
}

/// Sequential reader over Lucene-encoded bytes.
///
/// Implementors provide raw byte access; all wire-format decoding lives in the
/// provided methods so every backend decodes identically.
pub trait DataInput {
    fn read_byte(&mut self) -> Result<u8>;
    fn read_bytes(&mut self, out: &mut [u8]) -> Result<()>;

    /// Bytes remaining, used to pick the fast branchless path in group-varint.
    fn remaining(&self) -> usize;

    /// Absolute-position peek of 4 LE bytes without bounds concern beyond `remaining`.
    /// Default implementation goes through `read_bytes`; `SliceInput` overrides with
    /// an unaligned load.
    #[inline]
    fn read_u32_le(&mut self) -> Result<u32> {
        let mut b = [0u8; 4];
        self.read_bytes(&mut b)?;
        Ok(u32::from_le_bytes(b))
    }

    /// Lucene `DataInput.readInts(int[], int, int)`: fill `dst` with
    /// little-endian 32-bit words.
    ///
    /// The default is the naive loop, exactly as Lucene's default is; the point
    /// is that a backend able to do better can. `MemorySegmentIndexInput`
    /// overrides it with a bulk copy, and [`SliceInput`] does the same, which
    /// matters because `ForUtil`'s decode kernel reads up to 128 words per
    /// 256-value block and paid one bounds check and one `Result` per word
    /// before this existed.
    #[inline]
    fn read_u32s_le(&mut self, dst: &mut [u32]) -> Result<()> {
        for v in dst.iter_mut() {
            *v = self.read_u32_le()?;
        }
        Ok(())
    }

    /// Lucene `readVInt`. Negative Java ints occupy 5 bytes.
    ///
    /// Lucene 10.5.0's loop is `for (shift = 7; (b & 0x80) != 0; shift += 7)`
    /// — **unbounded**: it keeps consuming bytes for as long as the
    /// continuation bit is set, so a corrupt run of `0xFF` bytes reads on
    /// until EOF while `(b & 0x7F) << shift` silently wraps `shift` mod 32.
    /// (Lucene `main` later bounded it to `shift < 32`; this port predates
    /// neither and deliberately matches neither on that path.) On
    /// well-formed input all three agree byte for byte: `DataOutput.writeVInt`
    /// never sets the fifth byte's continuation bit (that byte is always
    /// `i >>> 28`, i.e. 4 bits). This port therefore stops at the fifth byte
    /// and reports [`Error::MalformedVarint`] rather than either
    /// half-decoding or running to EOF — a deliberate hardening of a path
    /// only corrupt input can reach.
    #[inline]
    fn read_vint(&mut self) -> Result<i32> {
        read_vint_loop(self)
    }

    /// Lucene `readZInt`: zigzag-decoded vint, the 32-bit counterpart of
    /// [`Self::read_zlong`] (stored fields' numeric ints, term-vector and
    /// doc-values deltas). Small negative values cost one byte here, versus
    /// the five a plain `readVInt` would spend.
    #[inline]
    fn read_zint(&mut self) -> Result<i32> {
        Ok(lucene_util::zigzag::decode_i32(self.read_vint()? as u32))
    }

    /// Lucene `readVLong` (non-negative on the wire; up to 9 bytes).
    #[inline]
    fn read_vlong(&mut self) -> Result<i64> {
        read_vlong_loop(self)
    }

    /// Lucene `readZLong`: zigzag-decoded vlong; full i64 range.
    #[inline]
    fn read_zlong(&mut self) -> Result<i64> {
        read_zlong_loop(self)
    }

    /// Lucene group-varint (`GroupVIntUtil.readGroupVInts`): full groups of 4
    /// values (1 flag byte + 1..4 LE bytes each), then a plain-vint tail.
    /// Values are unsigned 32-bit, widened to u64 like Lucene's `long[]` variant.
    fn read_group_vints(&mut self, dst: &mut [u64]) -> Result<()> {
        group_vints_default(self, dst)
    }

    /// Peek 4 LE bytes without advancing. Backends that can't peek may return
    /// `Eof` unconditionally: [`Self::read_group_vints`] probes this method
    /// once per group and takes its byte-at-a-time path when the probe fails.
    fn peek_u32_le(&mut self) -> Result<u32>;

    fn skip(&mut self, n: usize) -> Result<()>;

    /// Big-endian i32, as used by `CodecUtil.readBEInt` (header/footer magics only —
    /// everything else in the format is little-endian).
    #[inline]
    fn read_be_u32(&mut self) -> Result<u32> {
        let mut b = [0u8; 4];
        self.read_bytes(&mut b)?;
        Ok(u32::from_be_bytes(b))
    }

    /// Big-endian i64, as used by `CodecUtil.readBELong` (footer checksum).
    #[inline]
    fn read_be_u64(&mut self) -> Result<u64> {
        let hi = self.read_be_u32()? as u64;
        let lo = self.read_be_u32()? as u64;
        Ok((hi << 32) | lo)
    }

    /// Signed convenience wrapper: many callers of `CodecUtil.readBEInt` treat the
    /// result as a signed count (numSegments, delCount, ...).
    #[inline]
    fn read_be_i32(&mut self) -> Result<i32> {
        Ok(self.read_be_u32()? as i32)
    }

    /// Lucene `DataInput.readString`: vint byte-length prefix, UTF-8 payload.
    ///
    /// Two deliberate hardenings over Java, both about *corrupt* input only
    /// (valid input decodes identically):
    /// - a negative length is rejected rather than allocating; Java throws
    ///   `NegativeArraySizeException` from `new byte[length]`, whereas
    ///   `length as usize` here would be ~2^64 and abort the process on
    ///   allocation failure — an abort cannot be caught at the FFI boundary.
    /// - a length longer than the bytes actually remaining is rejected before
    ///   allocating, so a 5-byte corrupt header cannot make us reserve 2GB.
    /// - invalid UTF-8 is an error; Java's `new String(bytes, UTF_8)`
    ///   silently substitutes U+FFFD, which would let a corrupt segment name
    ///   or codec name compare unequal for a *different* reason later.
    fn read_string(&mut self) -> Result<String> {
        let len = self.read_length("string")?;
        let mut buf = vec![0u8; len];
        self.read_bytes(&mut buf)?;
        String::from_utf8(buf).map_err(|_| Error::Corrupted("invalid UTF-8 string".into()))
    }

    /// Reads a vint length/count and validates it against what the input can
    /// actually supply: negative is always corrupt, and a count larger than
    /// the remaining byte count can never be satisfied (every element costs
    /// at least one byte). Keeps corrupt input from driving an allocation.
    #[inline]
    fn read_length(&mut self, what: &str) -> Result<usize> {
        let len = self.read_vint()?;
        if len < 0 {
            return Err(Error::Corrupted(format!("negative {what} length: {len}")));
        }
        let len = len as usize;
        if len > self.remaining() {
            return Err(Error::Corrupted(format!(
                "{what} length {len} exceeds {} remaining bytes",
                self.remaining()
            )));
        }
        Ok(len)
    }

    /// Lucene `DataInput.readShort`: plain little-endian i16.
    #[inline]
    fn read_i16(&mut self) -> Result<i16> {
        let mut b = [0u8; 2];
        self.read_bytes(&mut b)?;
        Ok(i16::from_le_bytes(b))
    }

    /// `Short.toUnsignedInt(readShort())`: the same little-endian 2 bytes,
    /// widened as unsigned (used by block/doc-count headers, which are never
    /// negative even though Java stores them as `short`).
    #[inline]
    fn read_u16(&mut self) -> Result<u16> {
        let mut b = [0u8; 2];
        self.read_bytes(&mut b)?;
        Ok(u16::from_le_bytes(b))
    }

    /// Lucene `DataInput.readInt`: plain little-endian i32 (distinct from the
    /// header/footer's big-endian magics).
    #[inline]
    fn read_i32(&mut self) -> Result<i32> {
        Ok(self.read_u32_le()? as i32)
    }

    /// Lucene `DataInput.readLong`: plain little-endian i64.
    #[inline]
    fn read_i64(&mut self) -> Result<i64> {
        let mut b = [0u8; 8];
        self.read_bytes(&mut b)?;
        Ok(i64::from_le_bytes(b))
    }

    /// Lucene `DataInput.readLongs`: `count` consecutive little-endian i64s.
    ///
    /// Same story as [`Self::read_u32s_le`]: the default is Java's naive
    /// per-word loop, and backends that can do one bounds check for the whole
    /// run override it ([`SliceInput`] does, and `.liv` live-docs bitsets,
    /// BKD leaf blocks and stored-fields bitsets all read hundreds of words
    /// at a time through it).
    fn read_i64s(&mut self, dst: &mut [i64]) -> Result<()> {
        for slot in dst.iter_mut() {
            *slot = self.read_i64()?;
        }
        Ok(())
    }

    /// Lucene `DataInput.readMapOfStrings`: vint count, then `count` (key, value) string pairs.
    fn read_map_of_strings(&mut self) -> Result<Vec<(String, String)>> {
        // Each pair costs at least two bytes (two empty strings), so a count
        // above `remaining()` is corrupt by construction -- checked before
        // reserving, so a hand-crafted header can't request a huge Vec.
        let count = self.read_length("map-of-strings")?;
        let mut out = Vec::with_capacity(count);
        for _ in 0..count {
            let k = self.read_string()?;
            let v = self.read_string()?;
            out.push((k, v));
        }
        Ok(out)
    }

    /// Lucene `DataInput.readSetOfStrings`: vint count, then `count` strings.
    fn read_set_of_strings(&mut self) -> Result<Vec<String>> {
        let count = self.read_length("set-of-strings")?;
        let mut out = Vec::with_capacity(count);
        for _ in 0..count {
            out.push(self.read_string()?);
        }
        Ok(out)
    }
}

/// Zero-copy cursor over a byte slice (e.g. an mmap'd file region).
/// Cheap to `Clone` (a slice ref + a `usize`): cloning is how Lucene's
/// `IndexInput.clone()` maps to Rust.
#[derive(Clone, Debug)]
pub struct SliceInput<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> SliceInput<'a> {
    #[inline]
    pub fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    #[inline]
    pub fn position(&self) -> usize {
        self.pos
    }

    /// [`DataInput::read_group_vints`] into `i32`s, each holding its value's
    /// 32 bits: what a postings tail block decodes into its doc buffer
    /// (`GroupVIntUtil.readGroupVInts(in, int[], limit)`), with no `u64`
    /// staging buffer to zero and copy out of per block. Values, errors and
    /// their offsets are [`DataInput::read_group_vints`]'.
    // ARITH: as `read_group_vints` in `impl DataInput for SliceInput`: `pos
    // <= buf.len()`, so `pos + 17` cannot overflow; every other sum is
    // bounded by the 17-byte group or by `dst.len()`. Nothing is sized by a
    // value read off disk.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn read_group_vints_i32(&mut self, dst: &mut [i32]) -> Result<()> {
        const MASKS: [u32; 4] = [0xFF, 0xFFFF, 0xFF_FFFF, u32::MAX];
        let mut done = 0;
        for group in dst.chunks_exact_mut(4) {
            // `pos <= buf.len()`, so `pos + 17` cannot overflow a `usize`.
            let Some(bytes) = self.buf.get(self.pos..self.pos + 17) else {
                break;
            };
            let flag = bytes[0];
            let mut at = 1;
            for (j, slot) in group.iter_mut().enumerate() {
                // ARITH: `j < 4`, so the shift is at most 6; `at <= 13` and
                // the 2-bit length adds at most 4, inside the 17 bytes.
                let n_minus_1 = usize::from((flag >> (6 - 2 * j)) & 3);
                let word: [u8; 4] = bytes[at..at + 4].try_into().expect("four bytes");
                *slot = (u32::from_le_bytes(word) & MASKS[n_minus_1]) as i32;
                at += n_minus_1 + 1;
            }
            self.pos += at;
            done += 4;
        }
        if done < dst.len() {
            self.group_vints_i32_tail(&mut dst[done..])?;
        }
        Ok(())
    }

    /// [`Self::read_group_vints_i32`] for the groups too close to the end of
    /// the buffer for its 17-byte window -- a postings tail block's last
    /// group or two when the slice ends with the term, so once per rare term:
    /// not cold. A group whose own bytes are all there is decoded from them
    /// directly; the first that is cut short, and the plain-vint remainder,
    /// take the default decoder a group at a time, so errors and their
    /// offsets are its.
    #[inline(never)]
    // ARITH: `pos <= buf.len()` and a group spans at most 17 bytes, so `pos +
    // need` cannot overflow; `done` steps by 4 under `done + 4 <= dst.len()`.
    #[allow(clippy::arithmetic_side_effects)]
    fn group_vints_i32_tail(&mut self, dst: &mut [i32]) -> Result<()> {
        let mut done = 0;
        while done + 4 <= dst.len() {
            let Some(&flag) = self.buf.get(self.pos) else {
                break;
            };
            let lens =
                [flag >> 6, (flag >> 4) & 3, (flag >> 2) & 3, flag & 3].map(|l| usize::from(l) + 1);
            let need = 1 + lens.iter().sum::<usize>();
            let Some(bytes) = self.buf.get(self.pos..self.pos + need) else {
                break;
            };
            let mut at = 1;
            for (slot, &n) in dst[done..done + 4].iter_mut().zip(&lens) {
                let mut word = [0u8; 4];
                word[..n].copy_from_slice(&bytes[at..at + n]);
                *slot = u32::from_le_bytes(word) as i32;
                at += n;
            }
            self.pos += need;
            done += 4;
        }
        // Group-aligned: each chunk is a whole group, or the remainder of
        // fewer than four plain vints.
        let mut tmp = [0u64; 4];
        for chunk in dst[done..].chunks_mut(4) {
            let staged = &mut tmp[..chunk.len()];
            group_vints_default(self, staged)?;
            for (d, &v) in chunk.iter_mut().zip(staged.iter()) {
                *d = v as u32 as i32;
            }
        }
        Ok(())
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.buf.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }

    #[inline]
    pub fn seek(&mut self, pos: usize) -> Result<()> {
        if pos > self.buf.len() {
            return Err(self.eof());
        }
        self.pos = pos;
        Ok(())
    }

    /// A hint that the bytes at `pos` will be read soon: loads their cache line
    /// in the background ([`lucene_util::simd::prefetch`]) so a later read does
    /// not stall on it.
    /// No effect on the cursor or on correctness; a `pos` past the end, or a
    /// target without the instruction, does nothing.
    ///
    /// For a reader that learns where it will read next well before it gets
    /// there -- a postings cursor knows the next block's header offset as soon
    /// as it parses the current one, and has a block to decode in between.
    #[inline(always)]
    pub fn prefetch(&self, pos: usize) {
        lucene_util::simd::prefetch(self.buf, pos);
    }

    /// Zero-copy view of `[from..to)` of the underlying buffer, independent of the
    /// cursor position. Used by `codec_util` to compute the footer's CRC-32 over
    /// the exact byte range Lucene checksummed.
    #[inline]
    pub fn slice(&self, from: usize, to: usize) -> Result<&'a [u8]> {
        self.buf.get(from..to).ok_or_else(|| self.eof())
    }

    /// Port of `IndexInput.slice(sliceDescription, offset, length)`: returns a
    /// new, independent-file-pointer cursor over `[offset, offset+length)` of
    /// this input's own buffer, addressed from 0 as if it were its own file —
    /// exactly what a merge reading one source segment's sub-range (a stored
    /// field/doc-values/points block inside a shared `.cfs`, say) needs: a
    /// reader it can seek/advance freely without disturbing the parent or any
    /// other slice/clone of it.
    ///
    /// `description` mirrors Lucene's signature (useful context for future
    /// error messages) but isn't otherwise interpreted.
    ///
    /// Slicing a slice is supported: the result is itself a `SliceInput`, so
    /// calling `.slice_input(..)` again narrows further, offsets always being
    /// relative to *this* input's own `[0, len())`, not the original root
    /// buffer's addressing.
    ///
    /// Bounds are enforced up front: an out-of-range `offset`/`length` is
    /// rejected here rather than silently granting access to bytes outside
    /// the intended range (e.g. a neighboring sub-file's data).
    #[inline]
    pub fn slice_input(
        &self,
        description: &str,
        offset: u64,
        length: u64,
    ) -> Result<SliceInput<'a>> {
        let _ = description;
        let start = usize::try_from(offset).map_err(|_| self.eof())?;
        let len = usize::try_from(length).map_err(|_| self.eof())?;
        let end = start.checked_add(len).ok_or_else(|| self.eof())?;
        let buf = self.buf.get(start..end).ok_or_else(|| self.eof())?;
        Ok(SliceInput { buf, pos: 0 })
    }

    /// Remaining unread bytes `[pos, len())`, independent of `pos`'s effect on
    /// `read_bytes`/etc. Mostly a test/debug convenience (e.g. asserting a
    /// slice's full contents at once).
    #[inline]
    pub fn as_slice(&self) -> &'a [u8] {
        &self.buf[self.pos..]
    }

    #[cold]
    fn eof(&self) -> Error {
        Error::Eof { offset: self.pos }
    }
}

// ARITH: `SliceInput` maintains `pos <= buf.len()` as its type invariant --
// `seek` rejects anything past the end and every read advances `pos` only
// after a successful `get`. So `len() - pos` cannot wrap, `pos + 1`/`pos + 4`
// cannot overflow (`buf.len() <= isize::MAX`), and `pos += n` is guarded by
// `remaining() < n` first. No value read off disk sizes any of it.
//
// This allow covers the whole impl, which is wider than
// `docs/arithmetic-gate.md` prefers: **a method added here must re-verify the
// invariant above for itself, or carry its own narrower allow.**
#[allow(clippy::arithmetic_side_effects)]
impl DataInput for SliceInput<'_> {
    #[inline]
    fn read_byte(&mut self) -> Result<u8> {
        let b = *self.buf.get(self.pos).ok_or_else(|| self.eof())?;
        self.pos += 1;
        Ok(b)
    }

    /// Straight off the slice: with five bytes in view (a vint's maximum)
    /// the whole value is decoded without a bounds check or `Result` per
    /// byte, in a loop the compiler unrolls. Only a vint within five bytes of
    /// the end, or a malformed one, takes the shared byte loop -- so errors and
    /// their offsets are the default's.
    #[inline(always)]
    fn read_vint(&mut self) -> Result<i32> {
        if let Some(b) = self.buf.get(self.pos..self.pos + 5) {
            let mut v = 0i32;
            for (i, &x) in b.iter().enumerate() {
                // The fifth byte's high bits shift out of the `i32`, as
                // Java's `(b & 0x7F) << 28` does.
                v |= ((x & 0x7f) as i32).wrapping_shl(7 * i as u32);
                if x < 0x80 {
                    self.pos += i + 1;
                    return Ok(v);
                }
            }
        }
        slice_vint_slow(self)
    }

    /// See [`Self::read_vint`]; ten bytes cover every well-formed vlong.
    #[inline(always)]
    fn read_vlong(&mut self) -> Result<i64> {
        if let Some(b) = self.buf.get(self.pos..self.pos + 10) {
            let mut v = 0i64;
            for (i, &x) in b.iter().enumerate() {
                v |= ((x & 0x7f) as i64).wrapping_shl(7 * i as u32);
                if x < 0x80 {
                    self.pos += i + 1;
                    return Ok(v);
                }
            }
        }
        slice_vlong_slow(self)
    }

    /// See [`Self::read_vlong`], over the full ten-byte range a zigzag
    /// value can occupy; the default's byte loop for the rest.
    #[inline(always)]
    fn read_zlong(&mut self) -> Result<i64> {
        if let Some(b) = self.buf.get(self.pos..self.pos + 10) {
            let mut v = 0u64;
            for (i, &x) in b.iter().enumerate() {
                v |= ((x & 0x7f) as u64).wrapping_shl(7 * i as u32);
                if x < 0x80 {
                    self.pos += i + 1;
                    return Ok(lucene_util::zigzag::decode(v));
                }
            }
        }
        slice_zlong_slow(self)
    }

    #[inline]
    fn read_u32_le(&mut self) -> Result<u32> {
        let b = self
            .buf
            .get(self.pos..self.pos + 4)
            .ok_or_else(|| self.eof())?;
        let v = u32::from_le_bytes(b.try_into().unwrap());
        self.pos += 4;
        Ok(v)
    }

    #[inline]
    fn read_i32(&mut self) -> Result<i32> {
        Ok(self.read_u32_le()? as i32)
    }

    /// Validates the borrowed bytes in place instead of zero-filling a
    /// buffer and copying into it first.
    fn read_string(&mut self) -> Result<String> {
        let len = self.read_length("string")?;
        let bytes = &self.buf[self.pos..self.pos + len];
        // Consumed even when invalid, as the default's `read_bytes` does.
        self.pos += len;
        let s = std::str::from_utf8(bytes)
            .map_err(|_| Error::Corrupted("invalid UTF-8 string".into()))?;
        Ok(s.to_owned())
    }

    #[inline]
    fn read_i16(&mut self) -> Result<i16> {
        let b = self
            .buf
            .get(self.pos..self.pos + 2)
            .ok_or_else(|| self.eof())?;
        let v = i16::from_le_bytes([b[0], b[1]]);
        self.pos += 2;
        Ok(v)
    }

    #[inline]
    fn read_i64(&mut self) -> Result<i64> {
        let b = self
            .buf
            .get(self.pos..self.pos + 8)
            .ok_or_else(|| self.eof())?;
        let v = i64::from_le_bytes(b.try_into().unwrap());
        self.pos += 8;
        Ok(v)
    }

    #[inline]
    fn read_bytes(&mut self, out: &mut [u8]) -> Result<()> {
        let end = self.pos.checked_add(out.len()).ok_or_else(|| self.eof())?;
        let src = self.buf.get(self.pos..end).ok_or_else(|| self.eof())?;
        out.copy_from_slice(src);
        self.pos = end;
        Ok(())
    }

    #[inline]
    fn remaining(&self) -> usize {
        self.buf.len() - self.pos
    }

    #[inline]
    fn read_u32s_le(&mut self, dst: &mut [u32]) -> Result<()> {
        // One bounds check for the whole run, then a straight-line copy: on a
        // little-endian target LLVM lowers the `chunks_exact` loop to a bulk
        // move rather than per-word loads. `checked_*` because `len * 4` is
        // the one arithmetic here that a caller-supplied length could
        // overflow on a 32-bit target (where it would otherwise wrap to a
        // *smaller* end offset and read the wrong bytes).
        let end = dst
            .len()
            .checked_mul(4)
            .and_then(|n| self.pos.checked_add(n))
            .ok_or_else(|| self.eof())?;
        let src = self.buf.get(self.pos..end).ok_or_else(|| self.eof())?;
        for (d, c) in dst.iter_mut().zip(src.chunks_exact(4)) {
            *d = u32::from_le_bytes(c.try_into().unwrap());
        }
        self.pos = end;
        Ok(())
    }

    #[inline]
    fn read_i64s(&mut self, dst: &mut [i64]) -> Result<()> {
        let end = dst
            .len()
            .checked_mul(8)
            .and_then(|n| self.pos.checked_add(n))
            .ok_or_else(|| self.eof())?;
        let src = self.buf.get(self.pos..end).ok_or_else(|| self.eof())?;
        for (d, c) in dst.iter_mut().zip(src.chunks_exact(8)) {
            *d = i64::from_le_bytes(c.try_into().unwrap());
        }
        self.pos = end;
        Ok(())
    }

    #[inline]
    fn peek_u32_le(&mut self) -> Result<u32> {
        let src = self
            .buf
            .get(self.pos..self.pos + 4)
            .ok_or_else(|| self.eof())?;
        Ok(u32::from_le_bytes(src.try_into().unwrap()))
    }

    #[inline]
    fn skip(&mut self, n: usize) -> Result<()> {
        if self.remaining() < n {
            return Err(self.eof());
        }
        self.pos += n;
        Ok(())
    }

    /// Straight off the slice, as [`Self::read_vint`] is: a group whose flag
    /// and sixteen bytes are in view is decoded without a `Result` per
    /// value. The first group closer to the end than that, and everything
    /// after it, takes [`DataInput::read_group_vints`]' own path -- which is
    /// where Java's `length - pos >= 4 * Integer.BYTES` test sends it too --
    /// so values, errors and their offsets are the default's.
    fn read_group_vints(&mut self, dst: &mut [u64]) -> Result<()> {
        const MASKS: [u32; 4] = [0xFF, 0xFFFF, 0xFF_FFFF, u32::MAX];
        let mut done = 0;
        for group in dst.chunks_exact_mut(4) {
            // `pos <= buf.len()`, so `pos + 17` cannot overflow a `usize`.
            let Some(bytes) = self.buf.get(self.pos..self.pos + 17) else {
                break;
            };
            let flag = bytes[0];
            let mut at = 1;
            for (j, slot) in group.iter_mut().enumerate() {
                // ARITH: `j < 4`, so the shift is at most 6; `at <= 13` and
                // the 2-bit length adds at most 4, inside the 17 bytes.
                let n_minus_1 = usize::from((flag >> (6 - 2 * j)) & 3);
                let word: [u8; 4] = bytes[at..at + 4].try_into().expect("four bytes");
                *slot = u64::from(u32::from_le_bytes(word) & MASKS[n_minus_1]);
                at += n_minus_1 + 1;
            }
            self.pos += at;
            done += 4;
        }
        group_vints_default(self, &mut dst[done..])
    }
}

/// [`DataInput::read_group_vints`]' body, which an implementation that
/// overrides it (`SliceInput`) falls back to.
///
/// Java decides once per group, not once per value: the branchless path
/// over-reads up to 4 bytes for each of the four values, and the last one
/// starts at most 12 bytes into the group, so `length - pos >= 4 *
/// Integer.BYTES` is the single precondition that makes all four safe
/// (`GroupVIntUtil.readGroupVInt`). Only the very last group of a file can
/// fail it. A backend that cannot peek at all (no `RandomAccessInput` in
/// Java's terms) falls back the same way.
// ARITH: `i` walks `dst`'s own indices in steps of 4 under `i + 4 <=
// limit` / `i < limit`, and `n_minus_1` is a 2-bit field, so `i + j`,
// `i += 4` and `n_minus_1 + 1` are all bounded by `dst.len()` or by 4.
// Nothing here is sized by a value read off disk.
#[allow(clippy::arithmetic_side_effects)]
fn group_vints_default<D: DataInput + ?Sized>(input: &mut D, dst: &mut [u64]) -> Result<()> {
    const MASKS: [u32; 4] = [0xFF, 0xFFFF, 0xFF_FFFF, u32::MAX];
    let limit = dst.len();
    let mut i = 0;
    while i + 4 <= limit {
        let flag = input.read_byte()? as usize;
        let lens = [(flag >> 6) & 3, (flag >> 4) & 3, (flag >> 2) & 3, flag & 3];
        if input.remaining() >= 16 && input.peek_u32_le().is_ok() {
            for (j, &n_minus_1) in lens.iter().enumerate() {
                let v = input.peek_u32_le()? & MASKS[n_minus_1];
                input.skip(n_minus_1 + 1)?;
                dst[i + j] = v as u64;
            }
        } else {
            for (j, &n_minus_1) in lens.iter().enumerate() {
                let mut b = [0u8; 4];
                input.read_bytes(&mut b[..n_minus_1 + 1])?;
                dst[i + j] = u32::from_le_bytes(b) as u64;
            }
        }
        i += 4;
    }
    while i < limit {
        dst[i] = input.read_vint()? as u32 as u64;
        i += 1;
    }
    Ok(())
}

/// `ByteArrayDataInput.readVInt` at `*pos` of `buf`, advancing `*pos` past
/// it (left unchanged on error). The in-memory decoder is [`SliceInput`]'s.
#[inline]
pub fn read_vint_at(buf: &[u8], pos: &mut usize) -> Result<i32> {
    let mut input = SliceInput::new(buf);
    input.seek(*pos)?;
    let v = input.read_vint()?;
    *pos = input.position();
    Ok(v)
}

/// `ByteArrayDataInput.readVLong` at `*pos` of `buf`; see [`read_vint_at`].
#[inline]
pub fn read_vlong_at(buf: &[u8], pos: &mut usize) -> Result<i64> {
    let mut input = SliceInput::new(buf);
    input.seek(*pos)?;
    let v = input.read_vlong()?;
    *pos = input.position();
    Ok(v)
}

#[cfg(test)]
mod tests;
