//! Bit-packed values read with one unaligned 8-byte load each -- the shape of
//! Lucene's `DirectReader.DirectPackedReaderNN.get`, which is a single
//! `readInt`/`readLong` plus a shift and a mask.
//!
//! The load is bounds-checked **once**, when the reader is built: it works out
//! how many leading values have a whole 8-byte window inside the slice
//! (`DirectWriter` pads its output so that is all of them), and a read is then
//! one comparison against that count and an unchecked load. The unsafe code is
//! the load alone, and the comparison is what makes it sound; the public API is
//! safe and answers `None` for anything past the fast range, so a caller falls
//! back to its own checked path there.

/// A slice of little-endian bit-packed values, `bits` wide each.
#[derive(Debug, Clone, Copy)]
pub struct PackedLongs<'a> {
    bytes: &'a [u8],
    bits: u32,
    mask: u64,
    /// Every index below this has `byte(index) + 8 <= bytes.len()`.
    fast_limit: u64,
}

impl<'a> PackedLongs<'a> {
    /// `None` for a width a single 8-byte window cannot always hold: zero,
    /// above 64, or a non-byte-aligned width above 56 (whose value can start
    /// up to 7 bits into its first byte).
    pub fn new(bytes: &'a [u8], bits: u32) -> Option<Self> {
        if bits == 0 || bits > 64 || (!bits.is_multiple_of(8) && bits > 56) {
            return None;
        }
        let mask = if bits == 64 {
            u64::MAX
        } else {
            (1u64 << bits) - 1
        };
        // `floor(index * bits / 8) + 8 <= len` exactly when `index * bits <=
        // (len - 8) * 8 + 7`: the value may start anywhere in the last byte
        // the window can begin at.
        let fast_limit = match bytes.len().checked_sub(8) {
            Some(room) => (room as u64 * 8 + 7) / bits as u64 + 1,
            None => 0,
        };
        Some(PackedLongs {
            bytes,
            bits,
            mask,
            fast_limit,
        })
    }

    /// How many leading values [`Self::get`] answers.
    pub fn fast_limit(&self) -> u64 {
        self.fast_limit
    }

    /// This reader answering only the first `n` values (or fewer, if its fast
    /// range is shorter): a caller whose array holds `n` values folds that
    /// bound into the one comparison [`Self::get`] already makes, instead of
    /// making a second. Only ever lowers the limit, so it stays sound.
    pub fn limited_to(mut self, n: u64) -> Self {
        self.fast_limit = self.fast_limit.min(n);
        self
    }

    /// The value at `index`, or `None` past the range a whole 8-byte window
    /// covers (the caller's checked path answers those).
    #[inline]
    pub fn get(&self, index: u64) -> Option<u64> {
        if index >= self.fast_limit {
            return None;
        }
        // `index < fast_limit <= len * 8 / bits + 1`, so the product is at
        // most `len * 8 + bits` and cannot overflow a `u64`.
        let bit = index * self.bits as u64;
        let byte = (bit >> 3) as usize;
        // SAFETY: `index < fast_limit` gives `byte + 8 <= bytes.len()` (the
        // constructor's arithmetic), so the 8 bytes read lie inside the slice.
        // `read_unaligned` places no alignment requirement on the pointer.
        let word = unsafe { std::ptr::read_unaligned(self.bytes.as_ptr().add(byte) as *const u64) };
        Some((u64::from_le(word) >> (bit & 7)) & self.mask)
    }

    /// Values `start, start + 1, ...` into `out`, as many as fit and as the
    /// fast range covers; returns how many were written (0 when `start` is
    /// already past the fast range).
    ///
    /// A sequential scan's shape: one range check for the whole run instead
    /// of one per value, and a loop body with no branch at all, which is what
    /// lets it run at a value every cycle or two. The per-value width is
    /// dispatched once, outside the loop, so the common byte-aligned widths
    /// compile to a plain strided load.
    pub fn decode_range(&self, start: u64, out: &mut [i64]) -> usize {
        let Some(room) = self.fast_limit.checked_sub(start) else {
            return 0;
        };
        let n = out.len().min(usize::try_from(room).unwrap_or(usize::MAX));
        let out = &mut out[..n];
        match self.bits {
            8 => self.decode_run::<8>(start, out),
            16 => self.decode_run::<16>(start, out),
            32 => self.decode_run::<32>(start, out),
            64 => self.decode_run::<64>(start, out),
            _ => self.decode_run::<0>(start, out),
        }
        n
    }

    /// `decode_range`'s loop; `BITS == 0` means "read `self.bits`", any other
    /// value is that constant width.
    #[inline(always)]
    fn decode_run<const BITS: u32>(&self, start: u64, out: &mut [i64]) {
        let bits = if BITS == 0 {
            self.bits as u64
        } else {
            BITS as u64
        };
        let mask = self.mask;
        let base = self.bytes.as_ptr();
        for (i, slot) in out.iter_mut().enumerate() {
            let bit = (start + i as u64) * bits;
            // SAFETY: the caller trimmed `out` so every `start + i` is below
            // `fast_limit`, which (see `new`) puts the 8-byte window at
            // `bit / 8` inside `bytes`.
            let word =
                unsafe { std::ptr::read_unaligned(base.add((bit >> 3) as usize) as *const u64) };
            *slot = ((u64::from_le(word) >> (bit & 7)) & mask) as i64;
        }
    }
}

/// [`PackedLongs`] at a compile-time width `B`: the same single comparison
/// and unaligned load, with the multiply, shift and mask constants -- so a
/// byte-aligned width is a plain load, as Lucene's `DirectPackedReader8/16/
/// 32/64` are. For a loop that has dispatched on the width once, the way a
/// monomorphic Lucene call site sees one reader class.
#[derive(Debug, Clone, Copy)]
pub struct FixedPackedLongs<'a, const B: u32> {
    bytes: &'a [u8],
    /// As [`PackedLongs::fast_limit`]; 0 (so nothing is read) for a `B`
    /// [`PackedLongs::new`] rejects.
    fast_limit: u64,
}

impl<'a, const B: u32> FixedPackedLongs<'a, B> {
    pub fn new(bytes: &'a [u8]) -> Self {
        let fast_limit = PackedLongs::new(bytes, B).map_or(0, |p| p.fast_limit);
        FixedPackedLongs { bytes, fast_limit }
    }

    /// As [`PackedLongs::get`].
    #[inline(always)]
    pub fn get(&self, index: u64) -> Option<u64> {
        if index >= self.fast_limit {
            return None;
        }
        // As in `PackedLongs::get`: `index < fast_limit` bounds the product,
        // and `fast_limit` is non-zero only for a width `PackedLongs::new`
        // accepted, whose arithmetic it is.
        let bit = index * B as u64;
        // A byte-aligned width addresses its bytes directly: `index * 5`, one
        // `lea`, where `(index * 40) >> 3` is two dependent steps LLVM cannot
        // fold (it cannot rule out the multiply overflowing).
        let byte = if B.is_multiple_of(8) {
            (index * (B / 8) as u64) as usize
        } else {
            (bit >> 3) as usize
        };
        // The narrowest load that holds the value from its first bit -- as
        // Lucene's readers use `readShort`/`readInt` where those suffice: a
        // wide load at a narrow stride crosses cache lines more often (24-bit
        // reads were 20% slower through eight bytes). `shift + B` fits each: 1,
        // 2 and 4 bits never leave their byte, 12, 20 and 28 shift by at most
        // 4, and the byte-aligned widths do not shift.
        // SAFETY: `index < fast_limit` gives `byte + 8 <= bytes.len()`, as for
        // `PackedLongs::get` (the same constructor computed the limit, for
        // this same width), and no load below reads more than 8 bytes.
        let word = unsafe {
            let p = self.bytes.as_ptr().add(byte);
            match B {
                1 | 2 | 4 | 8 => u64::from(p.read()),
                12 | 16 => u64::from(u16::from_le(std::ptr::read_unaligned(p as *const u16))),
                20 | 24 | 28 | 32 => {
                    u64::from(u32::from_le(std::ptr::read_unaligned(p as *const u32)))
                }
                _ => u64::from_le(std::ptr::read_unaligned(p as *const u64)),
            }
        };
        let v = word >> (bit & 7);
        Some(v & u64::MAX.checked_shr(64u32.saturating_sub(B)).unwrap_or(0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The reference: bit-by-bit extraction.
    fn slow(bytes: &[u8], bits: u32, index: u64) -> u64 {
        let mut v = 0u64;
        for b in 0..bits as u64 {
            let bit = index * bits as u64 + b;
            let byte = bytes[(bit / 8) as usize];
            v |= (((byte >> (bit % 8)) & 1) as u64) << b;
        }
        v
    }

    #[test]
    fn matches_bit_by_bit_extraction_for_every_supported_width() {
        let mut s = 0x9e37_79b9_7f4a_7c15u64;
        let bytes: Vec<u8> = (0..517)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                s as u8
            })
            .collect();
        for bits in [1u32, 2, 4, 7, 8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64] {
            let p = PackedLongs::new(&bytes, bits).expect("supported width");
            let count = (bytes.len() as u64 * 8) / bits as u64;
            let mut fast = 0;
            for i in 0..count {
                match p.get(i) {
                    Some(v) => {
                        fast += 1;
                        assert_eq!(v, slow(&bytes, bits, i), "bits {bits} index {i}");
                    }
                    // Only the last few, whose window would run off the end.
                    None => assert!((i * bits as u64) / 8 + 8 > bytes.len() as u64),
                }
            }
            assert!(fast > 0);
        }
        assert!(PackedLongs::new(&bytes, 0).is_none());
        assert!(PackedLongs::new(&bytes, 57).is_none());
        assert!(PackedLongs::new(&bytes, 65).is_none());
        assert_eq!(PackedLongs::new(&[1, 2, 3], 8).unwrap().get(0), None);
    }

    /// `decode_range` is `get` over a run: same values, and it stops exactly
    /// where `get` starts answering `None`.
    /// `limited_to` only ever lowers the fast range: values below the new
    /// bound read as before, the rest go to the caller's checked path.
    #[test]
    fn limited_to_lowers_the_fast_range_and_never_raises_it() {
        let bytes = [0xABu8; 64];
        let p = PackedLongs::new(&bytes, 8).unwrap();
        let full = p.fast_limit();
        let q = p.limited_to(10);
        assert_eq!(q.fast_limit(), 10);
        assert_eq!(q.get(9), p.get(9));
        assert_eq!(q.get(10), None);
        assert_eq!(p.limited_to(u64::MAX).fast_limit(), full);
    }

    #[test]
    fn decode_range_agrees_with_get_and_stops_at_the_fast_limit() {
        let bytes: Vec<u8> = (0..300u32)
            .map(|i| (i.wrapping_mul(2_654_435_761) >> 13) as u8)
            .collect();
        for bits in [1u32, 3, 8, 12, 16, 24, 32, 48, 56, 64] {
            let p = PackedLongs::new(&bytes, bits).unwrap();
            for start in [
                0u64,
                1,
                5,
                63,
                p.fast_limit - 1,
                p.fast_limit,
                p.fast_limit + 3,
            ] {
                let mut out = [-1i64; 100];
                let n = p.decode_range(start, &mut out);
                for (i, &v) in out[..n].iter().enumerate() {
                    assert_eq!(
                        Some(v as u64),
                        p.get(start + i as u64),
                        "bits {bits} start {start}"
                    );
                }
                if n < out.len() {
                    assert_eq!(p.get(start + n as u64), None, "bits {bits} start {start}");
                }
                assert!(out[n..].iter().all(|&v| v == -1));
            }
        }
    }

    /// The const-width view answers exactly what the runtime-width reader does,
    /// including `None` past the fast range, and reads nothing for a width
    /// the runtime reader rejects.
    #[test]
    fn fixed_width_view_agrees_with_the_runtime_width_reader() {
        fn check<const B: u32>(bytes: &[u8]) {
            let fixed = FixedPackedLongs::<B>::new(bytes);
            match PackedLongs::new(bytes, B) {
                Some(p) => {
                    for i in 0..p.fast_limit + 4 {
                        assert_eq!(fixed.get(i), p.get(i), "bits {B} index {i}");
                    }
                    assert_eq!(fixed.get(u64::MAX), None);
                }
                None => assert_eq!(fixed.get(0), None, "bits {B}"),
            }
        }
        let bytes: Vec<u8> = (0..77u32)
            .map(|i| (i.wrapping_mul(2_654_435_761) >> 11) as u8)
            .collect();
        for len in [0usize, 7, 8, 9, 77] {
            let b = &bytes[..len];
            check::<0>(b);
            check::<1>(b);
            check::<2>(b);
            check::<4>(b);
            check::<8>(b);
            check::<12>(b);
            check::<16>(b);
            check::<28>(b);
            check::<32>(b);
            check::<40>(b);
            check::<56>(b);
            check::<57>(b);
            check::<64>(b);
        }
    }
}
