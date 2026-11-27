//! The front end's parser against Painless's own test suite
//! (`tests/data/corpus.bin`): every script Painless compiled parses, and the
//! parse failures are counted by their first words.
//!
//! `PAINLESS_FRONT_VERBOSE=1` prints each failure.

use std::collections::BTreeMap;

use lucene_painless::front::{parse, Limits};

struct Rd<'a> {
    b: &'a [u8],
    p: usize,
}

impl Rd<'_> {
    fn u8(&mut self) -> u8 {
        self.p += 1;
        self.b[self.p - 1]
    }
    fn i32(&mut self) -> i32 {
        self.p += 4;
        i32::from_le_bytes(self.b[self.p - 4..self.p].try_into().unwrap())
    }
    fn skip(&mut self, n: usize) {
        self.p += n;
    }
    fn bytes(&mut self) -> &[u8] {
        let n = self.i32() as usize;
        self.p += n;
        &self.b[self.p - n..self.p]
    }
    fn str(&mut self) -> String {
        String::from_utf8(self.bytes().to_vec()).unwrap()
    }
}

/// Steps over one encoded expected value.
fn skip_value(r: &mut Rd) {
    match r.u8() {
        0 => {}
        1 => r.skip(1),
        2..=5 | 7 => r.skip(4),
        6 | 8 => r.skip(8),
        9 => {
            r.bytes();
        }
        11 | 13 | 15 => {
            r.bytes();
            for _ in 0..r.i32() {
                skip_value(r);
            }
        }
        12 => {
            r.bytes();
            for _ in 0..r.i32() {
                skip_value(r);
                skip_value(r);
            }
        }
        14 => {
            r.bytes();
            r.bytes();
        }
        t => panic!("value tag {t}"),
    }
}

/// Every script, and whether Painless compiled it (statuses 0, 1, 3).
fn scripts() -> Vec<(String, bool)> {
    let data = std::fs::read(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/corpus.bin"
    ))
    .unwrap();
    assert_eq!(&data[..4], b"PCO2");
    let mut r = Rd { b: &data, p: 4 };
    let mut out = Vec::new();
    while r.p < data.len() {
        let src = r.str();
        match r.u8() {
            0 => {
                skip_value(&mut r);
                r.bytes();
                out.push((src, true));
            }
            1 => {
                r.bytes();
                if r.u8() == 1 {
                    r.bytes();
                }
                r.i32();
                r.bytes();
                out.push((src, true));
            }
            2 => out.push((src, false)),
            3 => {
                r.bytes();
                out.push((src, true));
            }
            s => panic!("status {s}"),
        }
    }
    out
}

#[test]
fn every_script_painless_compiled_parses() {
    let verbose = std::env::var("PAINLESS_FRONT_VERBOSE").is_ok();
    let limits = Limits::default();
    let (mut compiled, mut parsed) = (0, 0);
    let (mut refused, mut refused_parsed) = (0, 0);
    let mut failures: BTreeMap<String, usize> = BTreeMap::new();
    for (src, java_compiled) in scripts() {
        let result = parse(&src, &limits);
        if java_compiled {
            compiled += 1;
            match result {
                Ok(_) => parsed += 1,
                Err(e) => {
                    if verbose {
                        println!("{e}: {src}");
                    }
                    let key = e.message.split(',').next().unwrap_or("").to_string();
                    *failures.entry(key).or_default() += 1;
                }
            }
        } else {
            refused += 1;
            refused_parsed += result.is_ok() as usize;
        }
    }
    println!("compiled by Painless: {compiled}, parsed: {parsed}; failures: {failures:?}");
    println!("not compiled by Painless: {refused}, of which parse: {refused_parsed}");
    assert_eq!(
        parsed, compiled,
        "scripts Painless compiled that do not parse: {failures:?}"
    );
}

/// Cut anywhere, a script is parsed or refused: never a panic.
#[test]
fn truncated_scripts_never_panic() {
    let limits = Limits::default();
    for (src, _) in scripts() {
        let cuts: Vec<usize> = src.char_indices().map(|(i, _)| i).collect();
        for &i in cuts.iter().step_by((cuts.len() / 24).max(1)) {
            let _ = parse(&src[..i], &limits);
            let _ = parse(&src[i..], &limits);
        }
    }
}
