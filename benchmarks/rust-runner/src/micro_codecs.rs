//! Rust side of the lucene-codecs benchmark pair (M14.g); the Java side is
//! `benchmarks/micro/java/CodecsMicro.java`, with the same case names, over
//! the indexes it builds (once) under each codec's subdirectory.
//!
//! `cx_bloom_seek`: `seekExact` through `BloomFilteringPostingsFormat`'s
//! filtered terms, for each of `bloom/targets.tsv`'s 200 000 ids (a quarter
//! present), over the files Java wrote.
//!
//! `cx_fst_seek`/`cx_direct_seek`: `seekExact` on the same kind of ids
//! (half present) through `FSTPostingsFormat` and `DirectPostingsFormat`,
//! over `memory/`'s files.
//!
//! `cx_bitvec_search`: a top-10 search of `HnswBitVectorsFormat`'s graph
//! for each of `bitvectors/queries.tsv`'s 2 000 random 256-bit queries.
//!
//! `cx_ords_seek`: per target of `ords/`, `seekExact` on the id (half
//! present) hashing its `ord()`, then `seekExact(ord)` on `ords.tsv`'s ord
//! hashing the term's length and last byte, through
//! `BlockTreeOrdsPostingsFormat`'s enum.
//!
//! `cx_usplit_seek`: `seekExact` on the same kind of ids (half present)
//! through `UniformSplitPostingsFormat`'s enum, over `usplit/`'s files.

use std::hint::black_box;
use std::time::Duration;

use lucene_codecs::bitvectors::{BitVectorsFiles, BitVectorsReader};
use lucene_codecs::blocktreeords::{BlockTreeOrdsPostingsFormat, OrdsFieldReader};
use lucene_codecs::bloom::BloomFilteredFieldsProducer;
use lucene_codecs::hnsw_vectors::SearchOptions;
use lucene_codecs::memory::{DirectFiles, DirectPostingsFormat, FstPostingsFormat};
use lucene_codecs::segment_state::SegmentReadState;
use lucene_codecs::uniformsplit::UniformSplitPostingsFormat;

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

fn read_targets(dir: &str) -> Vec<Vec<u8>> {
    std::fs::read_to_string(format!("{dir}/targets.tsv"))
        .expect("run CodecsMicro build first (scripts/bench-micro.sh --bench codecs)")
        .lines()
        .map(|l| l.as_bytes().to_vec())
        .collect()
}

/// Times `seek` (`Some(docFreq)` for a present term) over every target.
fn bench_seek(
    w: Duration,
    m: Duration,
    name: &str,
    targets: &[Vec<u8>],
    mut seek: impl FnMut(&[u8]) -> Option<i32>,
) {
    let mut run = |f: &mut Fnv| {
        for t in targets {
            f.add(seek(t).map_or(-1, i64::from));
        }
    };
    let mut f = Fnv::new();
    run(&mut f);
    println!("#check\t{name}\t{:016x}\t{}", f.0, targets.len());
    measure(name, w, m, || {
        let mut g = Fnv::new();
        run(black_box(&mut g));
        black_box(g.0);
        targets.len() as u64
    });
}

fn bench_memory(w: Duration, m: Duration, dir: &str) {
    if !selected("cx_fst_seek") && !selected("cx_direct_seek") {
        return;
    }
    let targets = read_targets(dir);
    let (name, id, fis) = segment(dir);
    let file =
        |suffix: &str, ext: &str| std::fs::read(format!("{dir}/{name}_{suffix}.{ext}")).unwrap();
    let state = |suffix| SegmentReadState {
        max_doc: 200_000,
        segment_id: &id,
        segment_suffix: suffix,
        field_infos: &fis,
    };
    if selected("cx_fst_seek") {
        let tfp = file("FST50_0", "tfp");
        let reader = FstPostingsFormat.open(&tfp, &state("FST50_0")).unwrap();
        let terms = reader.terms("fid").unwrap();
        // One enum, as Java's case reuses one `TermsEnum`.
        let mut e = terms.iter();
        bench_seek(w, m, "cx_fst_seek", &targets, |t| {
            e.seek_exact(t).unwrap().then(|| e.doc_freq())
        });
    }
    if selected("cx_direct_seek") {
        let suffix = "Direct_0";
        let (tim, tip, tmd, doc) = (
            file(suffix, "tim"),
            file(suffix, "tip"),
            file(suffix, "tmd"),
            file(suffix, "doc"),
        );
        let fields = DirectPostingsFormat::default()
            .open(
                &DirectFiles {
                    tim: &tim,
                    tip: &tip,
                    tmd: &tmd,
                    doc: &doc,
                    pos: None,
                    pay: None,
                },
                &state(suffix),
            )
            .unwrap();
        let terms = fields.terms("did").unwrap();
        let mut e = terms.iter();
        bench_seek(w, m, "cx_direct_seek", &targets, |t| {
            e.seek_exact(t).then(|| e.doc_freq())
        });
    }
}

fn bench_bitvectors(w: Duration, m: Duration, dir: &str) {
    if !selected("cx_bitvec_search") {
        return;
    }
    let queries: Vec<Vec<u8>> = std::fs::read_to_string(format!("{dir}/queries.tsv"))
        .expect("run CodecsMicro build first (scripts/bench-micro.sh --bench codecs)")
        .lines()
        .map(|l| {
            (0..l.len())
                .step_by(2)
                .map(|i| u8::from_str_radix(&l[i..i + 2], 16).unwrap())
                .collect()
        })
        .collect();
    let (name, id, fis) = segment(dir);
    let suffix = "HnswBitVectorsFormat_0";
    let file = |ext: &str| std::fs::read(format!("{dir}/{name}_{suffix}.{ext}")).unwrap();
    let files = BitVectorsFiles {
        vec: file("vec"),
        vemf: file("vemf"),
        vex: file("vex"),
        vem: file("vem"),
    };
    let reader = BitVectorsReader::open(&files, &id, suffix).unwrap();
    let field = fis.field_by_name("bv").unwrap().number;
    let run = |f: &mut Fnv| {
        for q in &queries {
            let (hits, _) = reader
                .search(field, q, 10, i32::MAX as u64, SearchOptions::default())
                .unwrap();
            for (doc, score) in hits {
                f.add(i64::from(doc));
                f.add(i64::from(score.to_bits() as i32));
            }
        }
    };
    let mut f = Fnv::new();
    run(&mut f);
    println!("#check\tcx_bitvec_search\t{:016x}\t{}", f.0, queries.len());
    measure("cx_bitvec_search", w, m, || {
        let mut g = Fnv::new();
        run(black_box(&mut g));
        black_box(g.0);
        queries.len() as u64
    });
}

/// One `cx_ords_seek` pass, on one enum as Java's `ordsPass`.
fn ords_pass(terms: &OrdsFieldReader, ids: &[Vec<u8>], ords: &[i64], f: &mut Fnv) {
    let mut e = terms.iter();
    for (id, &ord) in ids.iter().zip(ords) {
        f.add(if e.seek_exact(id).unwrap() {
            e.ord()
        } else {
            -1
        });
        e.seek_exact_ord(ord).unwrap();
        let t = e.term();
        f.add(((t.len() as i64) << 8) | i64::from(t[t.len() - 1]));
    }
}

fn bench_ords(w: Duration, m: Duration, dir: &str) {
    if !selected("cx_ords_seek") {
        return;
    }
    let ids = read_targets(dir);
    let ords: Vec<i64> = std::fs::read_to_string(format!("{dir}/ords.tsv"))
        .unwrap()
        .lines()
        .map(|l| l.parse().unwrap())
        .collect();
    let (name, id, fis) = segment(dir);
    let suffix = "BlockTreeOrds_0";
    let file = |ext: &str| std::fs::read(format!("{dir}/{name}_{suffix}.{ext}")).unwrap();
    let state = SegmentReadState {
        max_doc: 200_000,
        segment_id: &id,
        segment_suffix: suffix,
        field_infos: &fis,
    };
    let reader = BlockTreeOrdsPostingsFormat::default()
        .open(file("tio"), &file("tipo"), &state)
        .unwrap();
    let terms = reader.terms("oid").unwrap();
    let mut f = Fnv::new();
    ords_pass(terms, &ids, &ords, &mut f);
    println!("#check\tcx_ords_seek\t{:016x}\t{}", f.0, ids.len());
    measure("cx_ords_seek", w, m, || {
        let mut g = Fnv::new();
        ords_pass(terms, &ids, black_box(&ords), &mut g);
        black_box(g.0);
        ids.len() as u64
    });
}

fn bench_usplit(w: Duration, m: Duration, dir: &str) {
    if !selected("cx_usplit_seek") {
        return;
    }
    let targets = read_targets(dir);
    let (name, id, fis) = segment(dir);
    let suffix = "UniformSplit_0";
    let file = |ext: &str| std::fs::read(format!("{dir}/{name}_{suffix}.{ext}")).unwrap();
    let state = SegmentReadState {
        max_doc: 200_000,
        segment_id: &id,
        segment_suffix: suffix,
        field_infos: &fis,
    };
    let reader = UniformSplitPostingsFormat::default()
        .open(file("ustb"), &file("ustd"), &state)
        .unwrap();
    let terms = reader.terms("uid").unwrap();
    // One enum, as Java's case reuses one `TermsEnum` per pass.
    let mut e = terms.iter();
    bench_seek(w, m, "cx_usplit_seek", &targets, |t| {
        e.seek_exact(t).unwrap().then(|| e.doc_freq().unwrap())
    });
}

pub fn bench_codecs(w: Duration, m: Duration, dir: &str) {
    bench_bloom(w, m, &format!("{dir}/bloom"));
    bench_memory(w, m, &format!("{dir}/memory"));
    bench_bitvectors(w, m, &format!("{dir}/bitvectors"));
    bench_ords(w, m, &format!("{dir}/ords"));
    bench_usplit(w, m, &format!("{dir}/usplit"));
}
