//! Rust side of the lucene-codecs benchmark pair (M14.g); the Java side is
//! `benchmarks/micro/java/CodecsMicro.java`, with the same case names, over
//! the indexes it builds (once) under each codec's subdirectory.
//!
//! `cx_bloom_seek`: `seekExact` through `BloomFilteringPostingsFormat`'s
//! filtered terms, for each of `bloom/targets.tsv`'s 200 000 ids (a quarter
//! present), over the files Java wrote.

use std::hint::black_box;
use std::time::Duration;

use lucene_codecs::bloom::BloomFilteredFieldsProducer;
use lucene_codecs::segment_state::SegmentReadState;

use super::measure;

/// FNV-1a over 64-bit words, identical to `CodecsMicro.Fnv`.
struct Fnv(u64);

impl Fnv {
    fn new() -> Self {
        Fnv(0xcbf2_9ce4_8422_2325)
    }
    fn add(&mut self, x: i64) {
        self.0 ^= x as u64;
        self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
    }
}

fn selected(name: &str) -> bool {
    std::env::var("MICRO_CASE").map_or(true, |only| only == name)
}

/// The one segment's name, id and `FieldInfos` in `dir`.
fn segment(dir: &str) -> (String, [u8; 16], lucene_codecs::field_infos::FieldInfos) {
    let si = std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .find(|n| n.ends_with(".si"))
        .expect("run CodecsMicro build first (scripts/bench-micro.sh --bench codecs)");
    let name = si.trim_end_matches(".si").to_string();
    let buf = std::fs::read(format!("{dir}/{si}")).unwrap();
    // magic, codec name, version, then the id.
    let at = 4 + 1 + usize::from(buf[4]) + 4;
    let id: [u8; 16] = buf[at..at + 16].try_into().unwrap();
    let fnm = std::fs::read(format!("{dir}/{name}.fnm")).unwrap();
    let fis = lucene_codecs::field_infos::parse(&fnm, &id, "").unwrap();
    (name, id, fis)
}

fn bench_bloom(w: Duration, m: Duration, dir: &str) {
    if !selected("cx_bloom_seek") {
        return;
    }
    let targets: Vec<Vec<u8>> = std::fs::read_to_string(format!("{dir}/targets.tsv"))
        .expect("run CodecsMicro build first (scripts/bench-micro.sh --bench codecs)")
        .lines()
        .map(|l| l.as_bytes().to_vec())
        .collect();
    let (name, id, fis) = segment(dir);
    let max_doc = 200_000;
    let suffix = "BloomFilter_0";
    let file = |ext: &str| std::fs::read(format!("{dir}/{name}_{suffix}.{ext}")).unwrap();
    let state = SegmentReadState {
        max_doc,
        segment_id: &id,
        segment_suffix: suffix,
        field_infos: &fis,
    };
    let producer = BloomFilteredFieldsProducer::open(
        &file("blm"),
        &file("tim"),
        &file("tip"),
        &file("tmd"),
        &state,
    )
    .unwrap();
    let terms = producer.terms("id").unwrap();
    let run = |f: &mut Fnv| {
        for t in &targets {
            f.add(match terms.try_seek_exact(t).unwrap() {
                Some(s) => i64::from(s.doc_freq),
                None => -1,
            });
        }
    };
    let mut f = Fnv::new();
    run(&mut f);
    println!("#check\tcx_bloom_seek\t{:016x}\t{}", f.0, targets.len());
    measure("cx_bloom_seek", w, m, || {
        let mut g = Fnv::new();
        run(black_box(&mut g));
        black_box(g.0);
        targets.len() as u64
    });
}

pub fn bench_codecs(w: Duration, m: Duration, dir: &str) {
    bench_bloom(w, m, &format!("{dir}/bloom"));
}
