#![allow(clippy::arithmetic_side_effects)]
use lucene_codecs::lz4::{self, FastCompressionHashTable, HighCompressionHashTable};
use lucene_codecs::lz4_hc::Lz4HcCompressor;
use std::time::Instant;
fn chunks(path: &str, size: usize) -> Vec<Vec<u8>> {
    let data = std::fs::read(path).unwrap();
    let mut out = vec![]; let mut cur = vec![]; let mut n = 0;
    for l in data.split(|&b| b == b'\n') { cur.extend_from_slice(l); n += 1; if cur.len() >= size || n >= 1024 { out.push(std::mem::take(&mut cur)); n = 0; } }
    if !cur.is_empty() { out.push(cur) } out
}
// preset layout: dict = len/20 (<=64K), 10 sub-blocks
fn preset(c: &[u8], f: &mut dyn FnMut(&[u8], usize, usize, &mut Vec<u8>)) -> usize {
    let len = c.len(); let dl = (len / 20).min(1 << 16); let bl = (len - dl).div_ceil(10);
    let mut out = vec![]; let mut buf = vec![0u8; dl + bl];
    buf[..dl].copy_from_slice(&c[..dl]); f(&buf[..dl], 0, dl, &mut out);
    let mut s = dl; while s < len { let l = bl.min(len - s); buf[dl..dl + l].copy_from_slice(&c[s..s + l]); f(&buf[..dl + l], dl, l, &mut out); s += l; }
    out.len()
}
fn main() {
    for path in std::env::args().skip(1) {
        let ch = chunks(&path, 80 * 1024); let raw: usize = ch.iter().map(|c| c.len()).sum();
        let mut fast = FastCompressionHashTable::new(); let mut lhc = HighCompressionHashTable::new(); let mut hc = Lz4HcCompressor::default();
        let mut run = |name: &str, f: &mut dyn FnMut(&[u8], usize, usize, &mut Vec<u8>)| {
            let t = Instant::now(); let n: usize = ch.iter().map(|c| preset(c, f)).sum(); let s = t.elapsed().as_secs_f64();
            println!("{path} {name:12} ratio {:.3} {:.0} MB/s", raw as f64 / n as f64, raw as f64 / s / 1e6);
        };
        run("fast", &mut |b, d, l, o| lz4::compress_with_dictionary(b, 0, d, l, o, &mut fast));
        run("lucene-hc", &mut |b, d, l, o| lz4::compress_with_dictionary(b, 0, d, l, o, &mut lhc));
        run("hc9", &mut |b, d, l, o| hc.compress_with_dictionary(b, 0, d, l, o));
    }
}
