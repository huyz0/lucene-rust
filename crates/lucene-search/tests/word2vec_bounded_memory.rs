//! `read_dl4j_model` streams a zip as `ZipInputStream` does: an entry before
//! the model inflates through a fixed buffer, so a 256 MiB entry (1.7 MB
//! deflated) costs no more memory than an empty one. Alone in its test
//! binary, so the process's peak resident set is this test's.

use lucene_search::word2vec::read_dl4j_model;

/// LSB-first bit writer, as deflate packs its bits.
struct Bits {
    out: Vec<u8>,
    acc: u64,
    n: u32,
}

impl Bits {
    fn put(&mut self, v: u64, n: u32) {
        self.acc |= v << self.n;
        self.n += n;
        while self.n >= 8 {
            self.out.push(self.acc as u8);
            self.acc >>= 8;
            self.n -= 8;
        }
    }

    /// A Huffman code, which deflate packs most significant bit first.
    fn code(&mut self, code: u64, n: u32) {
        let rev = (0..n).fold(0, |r, i| (r << 1) | ((code >> i) & 1));
        self.put(rev, n);
    }
}

/// One fixed-Huffman block: a zero byte, then `copies` back-references of
/// 258 bytes at distance 1 -- `1 + 258 * copies` zero bytes.
fn zeros_deflated(copies: usize) -> Vec<u8> {
    let mut b = Bits {
        out: Vec::new(),
        acc: 0,
        n: 0,
    };
    b.put(1, 1); // BFINAL
    b.put(1, 2); // BTYPE = fixed Huffman
    b.code(0x30, 8); // literal 0
    for _ in 0..copies {
        b.code(0xC5, 8); // length code 285: 258 bytes
        b.code(0, 5); // distance code 0: 1 byte back
    }
    b.code(0, 7); // end of block
    b.put(0, 7);
    b.out
}

fn local_header(z: &mut Vec<u8>, name: &str, flags: u16, method: u16, size: u32) {
    z.extend_from_slice(&0x0403_4B50u32.to_le_bytes());
    z.extend_from_slice(&20u16.to_le_bytes());
    z.extend_from_slice(&flags.to_le_bytes());
    z.extend_from_slice(&method.to_le_bytes());
    z.extend_from_slice(&[0; 8]); // time, date, CRC-32
    z.extend_from_slice(&size.to_le_bytes());
    z.extend_from_slice(&size.to_le_bytes());
    z.extend_from_slice(&(name.len() as u16).to_le_bytes());
    z.extend_from_slice(&0u16.to_le_bytes());
    z.extend_from_slice(name.as_bytes());
}

/// `VmHWM`: the process's peak resident set, in KiB.
fn peak_kib() -> u64 {
    let status = std::fs::read_to_string("/proc/self/status").unwrap();
    let line = status.lines().find(|l| l.starts_with("VmHWM:")).unwrap();
    line.split_whitespace().nth(1).unwrap().parse().unwrap()
}

#[test]
#[cfg(target_os = "linux")]
fn a_large_entry_before_the_model_is_not_held() {
    const COPIES: usize = 1 << 20;
    let mut z = Vec::new();
    // The decoy: deflated, sizes in a data descriptor after it.
    local_header(&mut z, "decoy.bin", 8, 8, 0);
    z.extend_from_slice(&zeros_deflated(COPIES));
    z.extend_from_slice(&0x0807_4B50u32.to_le_bytes());
    z.extend_from_slice(&[0; 4]);
    z.extend_from_slice(&0u32.to_le_bytes());
    z.extend_from_slice(&((1 + 258 * COPIES) as u32).to_le_bytes());
    let model = b"2 2\nhello 1 0\nworld 0 1\n";
    local_header(&mut z, "syn0.txt", 0, 0, model.len() as u32);
    z.extend_from_slice(model);

    let before = peak_kib();
    let m = read_dl4j_model(&z).unwrap();
    let grew = peak_kib().saturating_sub(before);
    assert_eq!((m.term(0), m.term(1)), (&b"hello"[..], &b"world"[..]));
    // Holding the decoy would be 258 MiB.
    assert!(grew < 32 * 1024, "peak grew {grew} KiB");
}
