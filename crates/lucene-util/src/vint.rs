//! Variable-length integer encoding onto a growable byte vector
//! (`DataOutput.writeVInt` / `writeVLong` for in-memory builders).
//!
//! Seven value bits per byte, low group first, high bit set on every byte but
//! the last. Negative values keep shifting their sign bits out, so an `i32`
//! takes five bytes and an `i64` ten, exactly like Java's.

/// `DataOutput.writeVInt`: appends the 1..=5 byte encoding of `v`.
#[inline]
pub fn write_vint(out: &mut Vec<u8>, v: i32) {
    write_vlong_u64(out, u64::from(v as u32));
}

/// `DataOutput.writeVLong` of a non-negative `v` (a negative `v` writes the
/// ten-byte form, which `readVLong` decodes back).
#[inline]
pub fn write_vlong(out: &mut Vec<u8>, v: i64) {
    write_vlong_u64(out, v as u64);
}

/// The same encoding over an already-unsigned bit pattern.
#[inline]
pub fn write_vlong_u64(out: &mut Vec<u8>, mut v: u64) {
    while v & !0x7f != 0 {
        out.push((v & 0x7f) as u8 | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn enc_int(v: i32) -> Vec<u8> {
        let mut o = Vec::new();
        write_vint(&mut o, v);
        o
    }

    #[test]
    fn vint_boundaries() {
        assert_eq!(enc_int(0), [0]);
        assert_eq!(enc_int(127), [0x7f]);
        assert_eq!(enc_int(128), [0x80, 0x01]);
        assert_eq!(enc_int(16383), [0xff, 0x7f]);
        assert_eq!(enc_int(16384), [0x80, 0x80, 0x01]);
        assert_eq!(enc_int(i32::MAX), [0xff, 0xff, 0xff, 0xff, 0x07]);
        assert_eq!(enc_int(-1), [0xff, 0xff, 0xff, 0xff, 0x0f]);
        assert_eq!(enc_int(i32::MIN), [0x80, 0x80, 0x80, 0x80, 0x08]);
    }

    #[test]
    fn vlong_boundaries() {
        let mut o = Vec::new();
        write_vlong(&mut o, 0);
        write_vlong(&mut o, 128);
        write_vlong(&mut o, i64::MAX);
        assert_eq!(&o[..3], [0, 0x80, 0x01]);
        assert_eq!(o.len(), 3 + 9);
        let mut neg = Vec::new();
        write_vlong(&mut neg, -1);
        assert_eq!(neg.len(), 10);
        let mut u = Vec::new();
        write_vlong_u64(&mut u, u64::MAX);
        assert_eq!(u, neg);
    }
}
