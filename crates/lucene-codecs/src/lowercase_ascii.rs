//! Port of `org.apache.lucene.util.compress.LowercaseAsciiCompression.compress`:
//! packs mostly-lowercase-ASCII bytes into 6 bits each, with an exception list
//! for the bytes that do not fit. The decode side is
//! `crate::blocktree::decompress_lowercase_ascii`, where the block-tree reader
//! has always used it.

/// `isCompressible(b)`: `b + 1` falls in `0x20..=0x3f` or `0x60..=0x7f`
/// (digits, `.`, `-`, `_`, lowercase letters and their neighbours).
// ARITH: `b <= 255`.
#[allow(clippy::arithmetic_side_effects)]
fn is_compressible(b: u8) -> bool {
    let high3 = (u32::from(b) + 1) & !0x1f;
    high3 == 0x20 || high3 == 0x60
}

fn write_vint(out: &mut Vec<u8>, v: i32) {
    let mut v = v as u32;
    while v & !0x7f != 0 {
        out.push((v & 0x7f) as u8 | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

/// `LowercaseAsciiCompression.compress(in, len, tmp, out)`: appends the
/// compressed form of `input` to `out` and returns `true`, or returns
/// `false` (leaving `out` untouched) when `input` is shorter than 8 bytes or
/// has more than `len / 32` exceptions -- exactly Java's accept/reject rule,
/// including its counting of the extra exceptions that bridge a gap longer
/// than 255 bytes.
// ARITH: indices and counts bounded by `input.len()`; `i - previous` is
// taken only with `i >= previous`.
#[allow(clippy::arithmetic_side_effects)]
pub fn compress(input: &[u8], out: &mut Vec<u8>) -> bool {
    let len = input.len();
    if len < 8 {
        return false;
    }
    let max_exceptions = len >> 5;
    let mut previous = 0usize;
    let mut num_exceptions = 0usize;
    for (i, &b) in input.iter().enumerate() {
        if !is_compressible(b) {
            while i - previous > 0xff {
                num_exceptions += 1;
                previous += 0xff;
            }
            num_exceptions += 1;
            if num_exceptions > max_exceptions {
                return false;
            }
            previous = i;
        }
    }
    let compressed_len = len - (len >> 2);
    let mut tmp: Vec<u8> = input
        .iter()
        .map(|&b| {
            let b = u32::from(b) + 1;
            ((b & 0x1f) | ((b & 0x40) >> 1)) as u8
        })
        .collect();
    let mut o = 0;
    for shift in [(0x30u8, 2u32), (0x0c, 4), (0x03, 6)] {
        for i in compressed_len..len {
            tmp[o] |= (tmp[i] & shift.0) << shift.1;
            o += 1;
        }
    }
    out.extend_from_slice(&tmp[..compressed_len]);
    write_vint(out, num_exceptions as i32);
    if num_exceptions > 0 {
        let mut previous = 0usize;
        for (i, &b) in input.iter().enumerate() {
            if !is_compressible(b) {
                while i - previous > 0xff {
                    out.push(0xff);
                    previous += 0xff;
                    out.push(input[previous]);
                }
                out.push((i - previous) as u8);
                previous = i;
                out.push(b);
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use lucene_store::data_input::SliceInput;

    fn round_trip(input: &[u8]) -> Option<Vec<u8>> {
        let mut out = Vec::new();
        if !compress(input, &mut out) {
            assert!(out.is_empty());
            return None;
        }
        let mut back = vec![0u8; input.len()];
        crate::blocktree::decompress_lowercase_ascii(&mut SliceInput::new(&out), &mut back)
            .unwrap();
        assert_eq!(back, input);
        Some(out)
    }

    #[test]
    fn compresses_and_round_trips() {
        assert!(round_trip(b"short").is_none());
        let c = round_trip(b"abcdefghijklmnop").unwrap();
        assert_eq!(c.len(), 12 + 1);
        // One exception per 32 bytes is allowed; two are not.
        let mut s = b"lowercase-ascii_words.with.digits0123456789abc".to_vec();
        s[3] = b'Z';
        round_trip(&s).unwrap();
        s[20] = b'A';
        assert!(round_trip(&s).is_none());
        // A gap longer than 255 between exceptions costs bridging entries.
        let mut long = vec![b'q'; 700];
        long[0] = b'A';
        long[600] = b'B';
        round_trip(&long).unwrap();
        long[10] = b'C';
        long[300] = b'D';
        long[400] = 0xff;
        round_trip(&long).unwrap();
    }
}
