//! The value blocks of the `Adaptive90` doc-values format: a NUMERIC or
//! SORTED_NUMERIC value array cut into blocks of [`BLOCK_SIZE`] values, each
//! block encoded with whichever chain of transforms makes it smallest.
//!
//! `Lucene90DocValuesFormat` packs a whole field (or each 16K-value block) at
//! one width after one GCD and one offset. That is a poor fit for the columns
//! logs and metrics indexes hold: timestamps grow steadily (their deltas are
//! small but their range is wide), and decimal amounts indexed as `double`
//! keep a full 64-bit sortable long per value. This encoding picks, per
//! 128-value block, among:
//!
//! - **delta**: the first value, then the differences between neighbours;
//! - **offset**: subtract the block's minimum;
//! - **GCD**: divide by the greatest common divisor of the offsets;
//! - **decimal**: read each value as `NumericUtils.sortableLongToDouble` and
//!   store it as an integer `i` with `i / 10^e` equal to the double bit for
//!   bit (the idea of ALP, Afroozeh et al., SIGMOD 2024, without exceptions);
//!
//! then bit-packs what is left at the narrowest width (frame of reference).
//! A block whose values are all equal stores the value once.
//!
//! Every transform is lossless by construction: the encoder decodes each
//! candidate the way the reader will and keeps it only if every value comes
//! back exactly, so a column is never at risk from a misdetected type.
//!
//! # Block layout
//!
//! ```text
//! token: u8                 CONSTANT | DELTA | GCD | DECIMAL bits
//! CONSTANT: zlong value     (nothing else)
//! otherwise:
//!   bits_per_value: u8      0..=64
//!   [exponent: u8]          when DECIMAL
//!   [first: zlong]          when DELTA
//!   min: zlong
//!   [gcd: zlong]            when GCD
//!   packed: ceil(m * bits_per_value / 8) bytes, m = n (or n - 1 with DELTA)
//! ```
//!
//! `zlong` is a zig-zag LEB128 varint (at most 10 bytes). Packed values are a
//! little-endian bit stream: value `i` occupies bits `i * bpv .. (i+1) * bpv`.
//! The value region ends with [`PADDING`] zero bytes so a reader may always
//! load 16 bytes at a packed value's first byte.

use lucene_store::Error;
use lucene_util::numeric_utils::sortable_double_bits as flip;

/// log2 of [`BLOCK_SIZE`].
pub const BLOCK_SHIFT: u32 = 7;
/// Values per block.
pub const BLOCK_SIZE: usize = 1 << BLOCK_SHIFT;
/// Zero bytes after the last block, so every packed read can load 16 bytes.
pub const PADDING: usize = 16;

const CONSTANT: u8 = 0x08;
const DELTA: u8 = 0x01;
const GCD: u8 = 0x02;
const DECIMAL: u8 = 0x04;

/// Largest decimal exponent tried (`10^e` is exact as an `f64` up to 22).
const MAX_EXPONENT: u32 = 18;
/// Decimal mantissas stay below 2^53, so `i as f64` is exact.
const MAX_MANTISSA: i64 = 1 << 53;

const POW10: [f64; 23] = [
    1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15, 1e16,
    1e17, 1e18, 1e19, 1e20, 1e21, 1e22,
];

fn eof() -> Error {
    Error::Eof { offset: 0 }
}

/// `NumericUtils.doubleToSortableLong(i / 10^e)`: the decimal transform's
/// decode, used by the encoder to verify and by every reader.
#[inline]
fn decimal_to_sortable(mantissa: i64, exponent: u32) -> i64 {
    let pow = POW10.get(exponent as usize).copied().unwrap_or(1.0);
    flip((mantissa as f64 / pow).to_bits() as i64)
}

/// The mantissa that represents `sortable` at `exponent`, if one does.
#[inline]
fn decimal_mantissa(sortable: i64, exponent: u32) -> Option<i64> {
    let x = f64::from_bits(flip(sortable) as u64);
    let pow = POW10.get(exponent as usize).copied()?;
    let scaled = (x * pow).round();
    if !scaled.is_finite() || scaled.abs() >= MAX_MANTISSA as f64 {
        return None;
    }
    let m = scaled as i64;
    (decimal_to_sortable(m, exponent) == sortable).then_some(m)
}

fn bits_required(v: u64) -> u8 {
    // ARITH: `leading_zeros() <= 64`.
    #[allow(clippy::arithmetic_side_effects)]
    let bits = 64 - v.leading_zeros();
    bits as u8
}

fn gcd_u64(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        // ARITH: `b != 0` is the loop condition.
        #[allow(clippy::arithmetic_side_effects)]
        let r = a % b;
        a = b;
        b = r;
    }
    a
}

fn write_zlong(out: &mut Vec<u8>, v: i64) {
    let mut z = ((v << 1) ^ (v >> 63)) as u64;
    while z >= 0x80 {
        out.push((z as u8) | 0x80);
        z >>= 7;
    }
    out.push(z as u8);
}

fn read_zlong(buf: &[u8], pos: &mut usize) -> Result<i64, Error> {
    let mut z = 0u64;
    let mut shift = 0u32;
    loop {
        let Some(&b) = buf.get(*pos) else {
            return Err(eof());
        };
        *pos = pos.wrapping_add(1);
        if shift >= 64 {
            return Err(eof());
        }
        z |= u64::from(b & 0x7f) << shift;
        if b & 0x80 == 0 {
            break;
        }
        shift = shift.wrapping_add(7);
    }
    Ok(((z >> 1) as i64) ^ ((z & 1) as i64).wrapping_neg())
}

/// Frame of reference over one stream: its minimum, its GCD (1 when none
/// helps) and the width of `(x - min) / gcd`.
#[derive(Debug, Clone, Copy)]
struct Frame {
    min: i64,
    gcd: u64,
    bpv: u8,
}

impl Frame {
    fn of(stream: &[i64]) -> Frame {
        let min = stream.iter().copied().min().unwrap_or(0);
        let mut gcd = 0u64;
        let mut max = 0u64;
        for &x in stream {
            let d = x.wrapping_sub(min) as u64;
            max = max.max(d);
            if gcd != 1 {
                gcd = gcd_u64(gcd, d);
            }
        }
        let gcd = gcd.max(1);
        // ARITH: `gcd >= 1`.
        #[allow(clippy::arithmetic_side_effects)]
        let bpv = bits_required(max / gcd);
        Frame { min, gcd, bpv }
    }

    /// Bytes the frame's header fields and packed values take for `m`
    /// values (an estimate of the varints: exact enough to rank candidates).
    fn cost(&self, m: usize) -> usize {
        let packed = (m.saturating_mul(usize::from(self.bpv))).div_ceil(8);
        let gcd = if self.gcd > 1 { 9 } else { 0 };
        packed.saturating_add(10).saturating_add(gcd)
    }
}

/// One candidate encoding of a block: the transforms, and the stream that is
/// bit-packed.
struct Candidate {
    token: u8,
    exponent: u32,
    first: i64,
    frame: Frame,
    stream: Vec<i64>,
    cost: usize,
}

fn candidate(values: &[i64], decimal: Option<u32>, delta: bool) -> Candidate {
    let (token_base, exponent) = match decimal {
        Some(e) => (DECIMAL, e),
        None => (0, 0),
    };
    let (first, stream): (i64, Vec<i64>) = if delta {
        let first = values.first().copied().unwrap_or(0);
        (
            first,
            values.windows(2).map(|w| w[1].wrapping_sub(w[0])).collect(),
        )
    } else {
        (0, values.to_vec())
    };
    let frame = Frame::of(&stream);
    let mut token = token_base;
    if delta {
        token |= DELTA;
    }
    if frame.gcd > 1 {
        token |= GCD;
    }
    let extra = usize::from(decimal.is_some()).wrapping_add(if delta { 10 } else { 0 });
    let cost = frame.cost(stream.len()).saturating_add(extra);
    Candidate {
        token,
        exponent,
        first,
        frame,
        stream,
        cost,
    }
}

/// The smallest exponent at which every value of `values` is a decimal, and
/// the mantissas, or `None`.
fn decimals(values: &[i64]) -> Option<(u32, Vec<i64>)> {
    // The largest exponent any single value needs; then the block at it.
    let mut exponent = 0u32;
    for &v in values {
        let mut e = exponent;
        loop {
            if decimal_mantissa(v, e).is_some() {
                break;
            }
            if e >= MAX_EXPONENT {
                return None;
            }
            e = e.wrapping_add(1);
        }
        exponent = e;
    }
    let mantissas: Option<Vec<i64>> = values
        .iter()
        .map(|&v| decimal_mantissa(v, exponent))
        .collect();
    mantissas.map(|m| (exponent, m))
}

/// Appends one block of `values` (1..=[`BLOCK_SIZE`] of them) to `out`.
pub fn encode_block(values: &[i64], out: &mut Vec<u8>) {
    let Some(&first) = values.first() else {
        return;
    };
    if values.iter().all(|&v| v == first) {
        out.push(CONSTANT);
        write_zlong(out, first);
        return;
    }
    let mut best = candidate(values, None, false);
    let mut consider = |c: Candidate| {
        if c.cost < best.cost {
            best = c;
        }
    };
    if values.len() >= 2 {
        consider(candidate(values, None, true));
    }
    if let Some((exponent, mantissas)) = decimals(values) {
        consider(candidate(&mantissas, Some(exponent), false));
        if mantissas.len() >= 2 {
            consider(candidate(&mantissas, Some(exponent), true));
        }
    }
    out.push(best.token);
    out.push(best.frame.bpv);
    if best.token & DECIMAL != 0 {
        out.push(best.exponent as u8);
    }
    if best.token & DELTA != 0 {
        write_zlong(out, best.first);
    }
    write_zlong(out, best.frame.min);
    if best.token & GCD != 0 {
        write_zlong(out, best.frame.gcd as i64);
    }
    pack(&best.stream, best.frame, out);
}

fn pack(stream: &[i64], frame: Frame, out: &mut Vec<u8>) {
    let bpv = u32::from(frame.bpv);
    if bpv == 0 {
        return;
    }
    let total_bits = stream.len().saturating_mul(bpv as usize);
    let start = out.len();
    out.resize(start.saturating_add(total_bits.div_ceil(8)), 0);
    let packed = &mut out[start..];
    let mut acc: u128 = 0;
    let mut acc_bits = 0u32;
    let mut at = 0usize;
    for &x in stream {
        // ARITH: `frame.gcd >= 1`, and every `x - min` is the true
        // non-negative difference, so the quotient fits `bpv` bits.
        #[allow(clippy::arithmetic_side_effects)]
        let u = (x.wrapping_sub(frame.min) as u64) / frame.gcd;
        acc |= u128::from(u) << acc_bits;
        acc_bits = acc_bits.wrapping_add(bpv);
        while acc_bits >= 8 {
            if let Some(b) = packed.get_mut(at) {
                *b = acc as u8;
            }
            at = at.wrapping_add(1);
            acc >>= 8;
            acc_bits = acc_bits.wrapping_sub(8);
        }
    }
    if acc_bits > 0 {
        if let Some(b) = packed.get_mut(at) {
            *b = acc as u8;
        }
    }
}

/// Encodes `values` as consecutive blocks. Returns the encoded bytes
/// (padding included) and each block's start offset within them, plus the
/// end offset of the last block: `blocks + 1` entries, the address array the
/// format stores.
pub fn encode(values: &[i64]) -> (Vec<u8>, Vec<i64>) {
    let mut out = Vec::new();
    let mut addresses = Vec::with_capacity(values.len().div_ceil(BLOCK_SIZE).saturating_add(1));
    for block in values.chunks(BLOCK_SIZE) {
        addresses.push(out.len() as i64);
        encode_block(block, &mut out);
    }
    addresses.push(out.len() as i64);
    out.resize(out.len().saturating_add(PADDING), 0);
    (out, addresses)
}

/// A parsed block header: where its packed values start and how to turn
/// them back into values.
#[derive(Debug, Clone, Copy)]
struct Header {
    token: u8,
    bpv: u8,
    exponent: u32,
    first: i64,
    min: i64,
    gcd: i64,
    packed: usize,
}

fn header(block: &[u8]) -> Result<Header, Error> {
    let mut pos = 0usize;
    let token = *block.first().ok_or_else(eof)?;
    pos = pos.wrapping_add(1);
    if token & CONSTANT != 0 {
        let v = read_zlong(block, &mut pos)?;
        return Ok(Header {
            token,
            bpv: 0,
            exponent: 0,
            first: 0,
            min: v,
            gcd: 1,
            packed: pos,
        });
    }
    let bpv = *block.get(pos).ok_or_else(eof)?;
    pos = pos.wrapping_add(1);
    if bpv > 64 {
        return Err(eof());
    }
    let exponent = if token & DECIMAL != 0 {
        let e = *block.get(pos).ok_or_else(eof)?;
        pos = pos.wrapping_add(1);
        if u32::from(e) > MAX_EXPONENT {
            return Err(eof());
        }
        u32::from(e)
    } else {
        0
    };
    let first = if token & DELTA != 0 {
        read_zlong(block, &mut pos)?
    } else {
        0
    };
    let min = read_zlong(block, &mut pos)?;
    let gcd = if token & GCD != 0 {
        read_zlong(block, &mut pos)?
    } else {
        1
    };
    Ok(Header {
        token,
        bpv,
        exponent,
        first,
        min,
        gcd,
        packed: pos,
    })
}

/// The `index`-th packed value of a block's bit stream. `packed` must extend
/// [`PADDING`] bytes past the stream's end.
#[inline]
fn unpack_one(packed: &[u8], bpv: u8, index: usize) -> Result<u64, Error> {
    let bit = index.wrapping_mul(usize::from(bpv));
    let byte = bit >> 3;
    let shift = (bit & 7) as u32;
    let Some(window) = packed.get(byte..byte.wrapping_add(16)) else {
        return Err(eof());
    };
    let mut buf = [0u8; 16];
    buf.copy_from_slice(window);
    let w = u128::from_le_bytes(buf) >> shift;
    let mask = if bpv >= 64 {
        u64::MAX
    } else {
        (1u64 << bpv).wrapping_sub(1)
    };
    Ok((w as u64) & mask)
}

/// Fills `out` (the block's `out.len()` values, at most [`BLOCK_SIZE`]) from
/// the block starting at `block[0]`. `block` must run to the end of the value
/// region, padding included.
pub fn decode_block(block: &[u8], out: &mut [i64]) -> Result<(), Error> {
    let h = header(block)?;
    if h.token & CONSTANT != 0 {
        out.fill(h.min);
        return Ok(());
    }
    let packed = block.get(h.packed..).ok_or_else(eof)?;
    if h.token & DELTA != 0 {
        let Some((head, rest)) = out.split_first_mut() else {
            return Ok(());
        };
        *head = h.first;
        unpack_into(packed, h.bpv, rest)?;
        let mut acc = h.first;
        for v in rest.iter_mut() {
            acc = acc.wrapping_add(h.min.wrapping_add(v.wrapping_mul(h.gcd)));
            *v = acc;
        }
    } else {
        unpack_into(packed, h.bpv, out)?;
        for v in out.iter_mut() {
            *v = h.min.wrapping_add(v.wrapping_mul(h.gcd));
        }
    }
    if h.token & DECIMAL != 0 {
        for v in out.iter_mut() {
            *v = decimal_to_sortable(*v, h.exponent);
        }
    }
    Ok(())
}

/// The value at `index` of the block starting at `block[0]`, without
/// decoding the others. A delta block's value is its first value plus the
/// sum of the `index` deltas before it, so it unpacks those (but stores
/// none): reading a whole block in order is cheaper with [`decode_block`].
#[inline]
pub fn get_one(block: &[u8], index: usize) -> Result<i64, Error> {
    let h = header(block)?;
    if h.token & CONSTANT != 0 {
        return Ok(h.min);
    }
    let packed = block.get(h.packed..).ok_or_else(eof)?;
    let v = if h.token & DELTA != 0 {
        // value[i] = first + sum over k < i of (min + raw[k] * gcd).
        let mut raw = [0i64; BLOCK_SIZE];
        let raw = raw.get_mut(..index).ok_or_else(eof)?;
        unpack_into(packed, h.bpv, raw)?;
        let raw_sum = raw.iter().fold(0i64, |a, &r| a.wrapping_add(r));
        let steps = index as i64;
        h.first
            .wrapping_add(steps.wrapping_mul(h.min))
            .wrapping_add(raw_sum.wrapping_mul(h.gcd))
    } else {
        let raw = if h.bpv == 0 {
            0
        } else {
            unpack_one(packed, h.bpv, index)? as i64
        };
        h.min.wrapping_add(raw.wrapping_mul(h.gcd))
    };
    Ok(if h.token & DECIMAL != 0 {
        decimal_to_sortable(v, h.exponent)
    } else {
        v
    })
}

/// Unpacks `out.len()` values of `B` bits from `packed`, which holds
/// [`fixed_unpack_len`] bytes: eight values fill exactly `B` bytes, so with
/// `B` a constant each group of eight unrolls to fixed loads and shifts.
#[inline(always)]
fn unpack_fixed<const B: usize>(packed: &[u8], out: &mut [i64]) {
    let mask = (1u64 << B).wrapping_sub(1);
    let mut groups = out.chunks_exact_mut(8);
    let mut start = 0usize;
    for chunk in &mut groups {
        let window = &packed[start..start.wrapping_add(B).wrapping_add(8)];
        for (j, v) in chunk.iter_mut().enumerate() {
            let bit = j.wrapping_mul(B);
            let byte = bit >> 3;
            let mut buf = [0u8; 8];
            buf.copy_from_slice(&window[byte..byte.wrapping_add(8)]);
            *v = ((u64::from_le_bytes(buf) >> (bit & 7)) & mask) as i64;
        }
        start = start.wrapping_add(B);
    }
    let window = &packed[start..];
    for (j, v) in groups.into_remainder().iter_mut().enumerate() {
        let bit = j.wrapping_mul(B);
        let byte = bit >> 3;
        let mut buf = [0u8; 8];
        buf.copy_from_slice(&window[byte..byte.wrapping_add(8)]);
        *v = ((u64::from_le_bytes(buf) >> (bit & 7)) & mask) as i64;
    }
}

/// [`unpack_fixed`] for a run-time width of 1 to 56 bits.
fn seq_macro_unpack(packed: &[u8], bpv: u8, out: &mut [i64]) {
    macro_rules! widths {
        ($($b:literal)*) => {
            match bpv {
                $($b => unpack_fixed::<$b>(packed, out),)*
                _ => unreachable!("width {bpv} is not 1..=56"),
            }
        };
    }
    widths!(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28
        29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56)
}

/// The bytes [`unpack_fixed`] reads for `n` values of `bpv` bits: each full
/// group of eight loads up to 8 bytes past its `bpv` bytes, and the trailing
/// partial group's last value loads 8 bytes from its first byte. This stays
/// within the stream's end plus 8, so a final, partial block reads no further
/// than its [`PADDING`].
fn fixed_unpack_len(n: usize, bpv: usize) -> usize {
    let groups = n / 8;
    let rest = n % 8;
    let tail = if rest == 0 {
        8
    } else {
        (rest.wrapping_sub(1).wrapping_mul(bpv) >> 3).wrapping_add(8)
    };
    groups.wrapping_mul(bpv).wrapping_add(tail)
}

fn unpack_into(packed: &[u8], bpv: u8, out: &mut [i64]) -> Result<(), Error> {
    if bpv == 0 || out.is_empty() {
        out.fill(0);
        return Ok(());
    }
    if bpv <= 56 {
        let packed = packed
            .get(..fixed_unpack_len(out.len(), usize::from(bpv)))
            .ok_or_else(eof)?;
        seq_macro_unpack(packed, bpv, out);
    } else {
        for (i, v) in out.iter_mut().enumerate() {
            *v = unpack_one(packed, bpv, i)? as i64;
        }
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::arithmetic_side_effects)]
mod tests {
    use super::*;

    fn sortable(x: f64) -> i64 {
        flip(x.to_bits() as i64)
    }

    fn round_trip(values: &[i64]) -> usize {
        let (bytes, addresses) = encode(values);
        assert_eq!(addresses.len(), values.len().div_ceil(BLOCK_SIZE) + 1);
        for (b, chunk) in values.chunks(BLOCK_SIZE).enumerate() {
            let block = &bytes[addresses[b] as usize..];
            let mut out = vec![0i64; chunk.len()];
            decode_block(block, &mut out).unwrap();
            assert_eq!(out, chunk, "block {b}");
            for (i, &want) in chunk.iter().enumerate() {
                assert_eq!(get_one(block, i).unwrap(), want, "block {b} value {i}");
            }
        }
        bytes.len()
    }

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
    }

    #[test]
    fn constant_and_tiny_blocks() {
        assert_eq!(round_trip(&[]), PADDING);
        round_trip(&[7]);
        round_trip(&[i64::MIN]);
        round_trip(&[5; 300]);
        round_trip(&[1, 2]);
    }

    #[test]
    fn extremes_need_all_64_bits() {
        let v: Vec<i64> = (0..300)
            .map(|i| if i % 2 == 0 { i64::MIN } else { i64::MAX - i })
            .collect();
        round_trip(&v);
        let mut r = Rng(42);
        let v: Vec<i64> = (0..1000).map(|_| r.next() as i64).collect();
        round_trip(&v);
    }

    #[test]
    fn every_width_round_trips() {
        let mut r = Rng(7);
        for bpv in 0..=64u32 {
            let mask = if bpv == 64 {
                u64::MAX
            } else {
                (1u64 << bpv) - 1
            };
            let v: Vec<i64> = (0..257).map(|_| (r.next() & mask) as i64).collect();
            round_trip(&v);
        }
    }

    #[test]
    fn timestamps_take_the_delta_path() {
        let mut r = Rng(3);
        let mut t = 1_700_000_000_000i64;
        let v: Vec<i64> = (0..1280)
            .map(|_| {
                t += (r.next() % 50) as i64 * 1000;
                t
            })
            .collect();
        let size = round_trip(&v);
        // 50 distinct second steps: 6 bits a value plus headers.
        assert!(size < 1280, "{size}");
        assert_eq!(header(&encode(&v).0).unwrap().token & DELTA, DELTA);
    }

    #[test]
    fn gcd_is_found() {
        let v: Vec<i64> = (0..128).map(|i| 1000 + (i * 37 % 11) * 1000).collect();
        let (bytes, _) = encode(&v);
        let h = header(&bytes).unwrap();
        assert_eq!(h.token & GCD, GCD);
        assert_eq!(h.gcd % 1000, 0);
        round_trip(&v);
    }

    #[test]
    fn decimal_doubles_become_small_integers() {
        let mut r = Rng(11);
        let v: Vec<i64> = (0..1024)
            .map(|_| sortable((r.next() % 10000) as f64 / 100.0))
            .collect();
        let size = round_trip(&v);
        assert!(size < 1024 * 2 + 200, "{size}");
        assert_eq!(header(&encode(&v).0).unwrap().token & DECIMAL, DECIMAL);
    }

    #[test]
    fn awkward_doubles_stay_exact() {
        let specials = [
            -0.0,
            0.0,
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::MIN_POSITIVE,
            5e-324,
            0.1 + 0.2,
            1e300,
            -123.456,
            9007199254740993.0,
        ];
        let v: Vec<i64> = (0..256)
            .map(|i| sortable(specials[i % specials.len()]))
            .collect();
        round_trip(&v);
        // A block of decimals with one -0.0 cannot take the decimal path.
        let mut v: Vec<i64> = (0..128).map(|i| sortable(i as f64 / 10.0)).collect();
        v[5] = sortable(-0.0);
        round_trip(&v);
        assert_eq!(header(&encode(&v).0).unwrap().token & DECIMAL, 0);
    }

    #[test]
    fn negative_and_decreasing_runs() {
        let v: Vec<i64> = (0..500).map(|i| -10 * i).collect();
        round_trip(&v);
        let v: Vec<i64> = (0..500).map(|i| sortable(-(i as f64) * 0.25)).collect();
        round_trip(&v);
    }

    #[test]
    fn a_short_last_block_of_wide_values_decodes() {
        // A last block of a few wide values sits right before the padding:
        // decoding it must not read past it.
        let mut r = Rng(5);
        for tail in 1..BLOCK_SIZE {
            for bpv in [17u32, 40, 56] {
                let mut v = vec![5i64; BLOCK_SIZE];
                v.extend((0..tail).map(|_| (r.next() >> (64 - bpv)) as i64));
                round_trip(&v);
            }
        }
    }

    #[test]
    fn corrupt_blocks_are_errors() {
        assert!(decode_block(&[], &mut [0; 4]).is_err());
        assert!(decode_block(&[0, 65], &mut [0; 4]).is_err());
        assert!(decode_block(&[DECIMAL, 3, 40], &mut [0; 4]).is_err());
        // A varint that never ends.
        assert!(decode_block(
            &[CONSTANT, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff],
            &mut [0; 4]
        )
        .is_err());
        // Packed values past the end.
        assert!(decode_block(&[0, 8, 0], &mut [0; 4]).is_err());
        assert!(get_one(&[0, 60, 0], 3).is_err());
        assert_eq!(get_one(&[DELTA, 1, 0, 0, 0, 0], 0).unwrap(), 0);
        assert!(get_one(&[], 0).is_err());
        assert!(decode_block(&[DELTA, 1, 0, 0], &mut []).is_ok());
    }
}
