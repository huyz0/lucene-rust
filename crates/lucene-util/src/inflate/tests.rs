#![cfg(test)]
#![allow(clippy::arithmetic_side_effects)]

use super::*;

fn corpus(n: usize, seed: u64) -> Vec<u8> {
    let words = [
        "lucene",
        "segment",
        "merge",
        "the",
        "of",
        "{\"id\":",
        "\"body\":",
        "1234",
        "GET /a",
        "HTTP/1.1",
        "200",
        "stored",
        "fields",
    ];
    let mut x = seed | 1;
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        if x.is_multiple_of(9) {
            out.push((x >> 24) as u8);
        } else {
            out.extend_from_slice(words[(x >> 33) as usize % words.len()].as_bytes());
            out.push(b' ');
        }
    }
    out.truncate(n);
    out
}

fn miniz(raw: &[u8], level: u8) -> Vec<u8> {
    miniz_oxide::deflate::compress_to_vec(raw, level)
}

fn check(compressed: &[u8], raw: &[u8], dict: &[u8]) {
    let mut dest = dict.to_vec();
    dest.resize(dict.len() + raw.len(), 0xAA);
    Inflater::default()
        .inflate(compressed, &mut dest, dict.len(), raw.len())
        .unwrap();
    assert_eq!(&dest[dict.len()..], raw);
}

#[test]
fn decodes_every_block_type_and_level() {
    for n in [0, 1, 7, 100, 300, 5000, 70_000, 300_000] {
        let raw = corpus(n, n as u64 + 3);
        for level in [0, 1, 6, 9] {
            check(&miniz(&raw, level), &raw, &[]);
        }
    }
    let runs = vec![b'z'; 100_000];
    check(&miniz(&runs, 6), &runs, &[]);
    // Every short match period, inside text so the fast loop sees them.
    for period in 2..20 {
        let mut raw = corpus(3000, period as u64);
        for i in 0..2000 {
            raw.push(b'a' + (i % period) as u8);
        }
        raw.extend_from_slice(&corpus(3000, 99));
        check(&miniz(&raw, 6), &raw, &[]);
    }
}

#[test]
fn rejects_corruption_without_panicking() {
    let raw = corpus(50_000, 9);
    let good = miniz(&raw, 6);
    let mut x = 1u64;
    for _ in 0..3000 {
        x = x
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let mut bad = good.clone();
        let i = (x >> 33) as usize % bad.len();
        bad[i] ^= 1 << ((x >> 20) % 8);
        if x.is_multiple_of(3) {
            bad.truncate(i);
        }
        let mut dest = vec![0u8; raw.len()];
        let r = Inflater::default().inflate(&bad, &mut dest, 0, raw.len());
        if r.is_ok() {
            // A flip in a literal can still decode to the right length.
            assert_eq!(dest.len(), raw.len());
        }
    }
    // Wrong expected lengths.
    let mut dest = vec![0u8; raw.len() + 1];
    assert!(Inflater::default()
        .inflate(&good, &mut dest, 0, raw.len() + 1)
        .is_err());
    let mut dest = vec![0u8; raw.len() - 1];
    assert!(Inflater::default()
        .inflate(&good, &mut dest, 0, raw.len() - 1)
        .is_err());
    // Random garbage.
    for seed in 0..500u64 {
        let junk = corpus(64, seed);
        let mut dest = vec![0u8; 1000];
        let _ = Inflater::default().inflate(&junk, &mut dest, 0, 1000);
    }
}

#[test]
fn a_distance_past_the_start_is_corruption() {
    // A fixed block: literal 'a', then length 3 at distance 2 (only one
    // byte of history exists), then end.
    let mut w = BitWriter::default();
    w.put(1, 1);
    w.put(1, 2);
    w.put_rev(0x30 + b'a' as u32, 8);
    w.put_rev(1, 7); // length symbol 257 = 3
    w.put_rev(1, 5); // distance symbol 1 = 2
    w.put_rev(0, 7); // end of block
    let bytes = w.finish();
    let mut dest = vec![0u8; 4];
    assert!(Inflater::default()
        .inflate(&bytes, &mut dest, 0, 4)
        .is_err());
    // With one dictionary byte before the output, the same stream is fine.
    let mut dest = vec![b'x', 0, 0, 0, 0];
    Inflater::default()
        .inflate(&bytes, &mut dest, 1, 4)
        .unwrap();
    assert_eq!(&dest, b"xaxax");
}

#[derive(Default)]
struct BitWriter {
    out: Vec<u8>,
    acc: u64,
    n: u32,
}

impl BitWriter {
    fn put(&mut self, v: u32, n: u32) {
        self.acc |= (v as u64) << self.n;
        self.n += n;
        while self.n >= 8 {
            self.out.push(self.acc as u8);
            self.acc >>= 8;
            self.n -= 8;
        }
    }
    /// A Huffman code, most significant bit first.
    fn put_rev(&mut self, code: u32, len: u32) {
        self.put(code.reverse_bits() >> (32 - len), len);
    }
    fn finish(mut self) -> Vec<u8> {
        if self.n > 0 {
            self.out.push(self.acc as u8);
        }
        self.out
    }
}
