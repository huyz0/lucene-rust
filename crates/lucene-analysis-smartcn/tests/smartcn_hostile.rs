//! Corrupt dictionaries never panic: truncations and byte flips of the
//! vendored `coredict.mem` and `bigramdict.mem` load as an error or as a
//! dictionary whose lookups still answer.
#![allow(clippy::arithmetic_side_effects)] // test code: no value read off disk

use lucene_analysis_smartcn::hhmm::{BigramDictionary, HHMMSegmenter, WordDictionary};
use std::sync::Arc;

fn inflate(z: &[u8]) -> Vec<u8> {
    miniz_oxide::inflate::decompress_to_vec_zlib(z).unwrap()
}

struct Lcg(u64);

impl Lcg {
    fn next(&mut self, n: usize) -> usize {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 33) % n as u64) as usize
    }
}

fn exercise(d: &WordDictionary) {
    let u: Vec<u16> = "中国人民我们的东西".encode_utf16().collect();
    for i in 0..u.len() {
        for j in i + 1..=u.len() {
            let w = &u[i..j];
            let p = d.get_prefix_match(w, 0);
            let _ = d.get_frequency(w);
            let _ = d.is_equal(w, p);
        }
    }
}

#[test]
fn corrupt_core_dictionaries_never_panic() {
    let bytes = inflate(include_bytes!("../src/resources/coredict.mem.z"));
    let mut r = Lcg(42);
    let (mut rejected, mut loaded) = (0, 0);
    for k in 0..120 {
        let mut b = bytes.clone();
        if k % 4 == 0 {
            b.truncate(r.next(b.len()));
        } else {
            // Bias toward the headers and the first rows, where structure is.
            let at = if k % 2 == 0 {
                r.next(4096)
            } else {
                r.next(b.len())
            };
            b[at] ^= 1 << r.next(8);
        }
        match WordDictionary::from_mem(&b) {
            Ok(d) => {
                exercise(&d);
                let seg = HHMMSegmenter::new(Arc::new(d), BigramDictionary::get_instance());
                let s: Vec<u16> = "我是中国人。abc 123".encode_utf16().collect();
                let _ = seg.process(&s);
                loaded += 1;
            }
            Err(_) => rejected += 1,
        }
    }
    assert!(
        rejected > 15 && loaded > 15,
        "rejected {rejected} loaded {loaded}"
    );
}

#[test]
fn corrupt_bigram_dictionaries_never_panic() {
    let bytes = inflate(include_bytes!("../src/resources/bigramdict.mem.z"));
    let mut r = Lcg(7);
    let mut rejected = 0;
    for k in 0..16 {
        let mut b = bytes.clone();
        if k % 2 == 0 {
            b.truncate(r.next(b.len()));
        } else {
            let at = r.next(64);
            b[at] ^= 1 << r.next(8);
        }
        match BigramDictionary::from_mem(&b) {
            Ok(d) => {
                let p: Vec<u16> = "中国@人民".encode_utf16().collect();
                let _ = d.get_frequency(&p);
            }
            Err(_) => rejected += 1,
        }
    }
    assert!(rejected > 6, "{rejected}");
}
