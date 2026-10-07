//! A `coredict.mem` that shares one array through many `TC_REFERENCE`s
//! costs memory in proportion to its bytes, as in Java (where every
//! reference is the same object), not to the arrays it names.
//!
//! One test in its own binary on purpose: it reads the process's peak
//! resident set (`VmHWM`), which nothing else may raise beside it. What it
//! cannot catch: an amplification below its 32 MiB margin.
#![cfg(target_os = "linux")]
#![allow(clippy::arithmetic_side_effects)] // test code: no value read off disk

use lucene_analysis_smartcn::hhmm::WordDictionary;

/// `VmHWM`, the peak resident set so far, in bytes.
fn peak_rss() -> usize {
    let status = std::fs::read_to_string("/proc/self/status").unwrap();
    let line = status.lines().find(|l| l.starts_with("VmHWM:")).unwrap();
    let kb: usize = line.split_whitespace().nth(1).unwrap().parse().unwrap();
    kb * 1024
}

/// The stream protocol's subset, numbering handles as `ObjectOutputStream`.
struct Stream {
    out: Vec<u8>,
    next: u32,
}

impl Stream {
    fn new() -> Self {
        Stream {
            out: vec![0xAC, 0xED, 0, 5],
            next: 0,
        }
    }

    /// `TC_ARRAY` with a new descriptor and `len`; the array's handle.
    fn array(&mut self, name: &str, len: usize) -> u32 {
        self.out
            .extend_from_slice(&[0x75, 0x72, 0, name.len() as u8]);
        self.out.extend_from_slice(name.as_bytes());
        self.out.extend_from_slice(&[0; 8]);
        self.out.extend_from_slice(&[2, 0, 0, 0x78, 0x70]);
        self.out.extend_from_slice(&(len as i32).to_be_bytes());
        self.next += 2;
        self.next - 1
    }

    fn reference(&mut self, handle: u32) {
        self.out.push(0x71);
        self.out
            .extend_from_slice(&(0x7E_0000 + handle).to_be_bytes());
    }

    /// The two hash tables, all empty.
    fn tables(&mut self) {
        self.array("[S", 12071);
        self.out.extend(std::iter::repeat_n(0, 2 * 12071));
        self.array("[C", 12071);
        self.out.extend(std::iter::repeat_n(0, 2 * 12071));
    }
}

#[test]
fn shared_arrays_are_not_copied_per_reference() {
    let base = peak_rss();
    let margin = 32 << 20;

    // One row of 20,001 words, every one the same 10,000-character `[C`:
    // 168 KB that a copy per reference turns into 400 MB.
    let mut s = Stream::new();
    s.tables();
    s.array("[[[C", 1);
    s.array("[[C", 20_001);
    let word = s.array("[C", 10_000);
    s.out.extend(std::iter::repeat_n(0x4E, 2 * 10_000));
    for _ in 0..20_000 {
        s.reference(word);
    }
    s.array("[[I", 0);
    assert!(s.out.len() < 170_000);
    let e = WordDictionary::from_mem(&s.out).err().unwrap().to_string();
    assert!(e.contains("differ in length"), "{e}");
    let grown = peak_rss().saturating_sub(base);
    assert!(grown < margin, "shared words: peak grew {grown} bytes");

    // 1,001 rows, every one the same row of 10,000 null words: 75 KB that a
    // copy per reference turns into 240 MB.
    let mut s = Stream::new();
    s.tables();
    s.array("[[[C", 1_001);
    let row = s.array("[[C", 10_000);
    s.out.extend(std::iter::repeat_n(0x70, 10_000));
    for _ in 0..1_000 {
        s.reference(row);
    }
    s.array("[[I", 0);
    let e = WordDictionary::from_mem(&s.out).err().unwrap().to_string();
    assert!(e.contains("differ in length"), "{e}");
    let grown = peak_rss().saturating_sub(base);
    assert!(grown < margin, "shared rows: peak grew {grown} bytes");
}
