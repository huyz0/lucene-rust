//! Segment-open, per-seek and term-intersection cost of the block-tree term
//! dictionary, measured against a **real Lucene 10.5.0-written** segment.
//!
//! This is the benchmark for finding A1 (`docs/sweep/m2/LEDGER.md`): this port
//! used to materialize every term of every field into a sorted array when a
//! segment was opened, where `Lucene103BlockTreeTermsReader` reads only each
//! field's `.tmd` record and lets `SegmentTermsEnum` load individual `.tim`
//! blocks lazily as a seek walks into them. M1.6 measured that at 52.7 ms
//! against Lucene's 0.34 ms for a whole `DirectoryReader.open`.
//!
//! The corpus is `benchmarks/.corpus/merged` (see `scripts/bench-corpus.sh`),
//! the same single-segment, force-merged, real-Lucene index
//! `scripts/bench-compare.sh` runs both engines over: ~4.7 MB `.tim`,
//! ~89 KB `.tip`, 579k distinct terms. Point `LUCENE_RUST_BENCH_INDEX` at
//! another index directory to use that instead.
//!
//! Every case is **skipped, not failed**, when the corpus is absent, because
//! it is a generated artifact and not checked in.
//!
//! Run with: `cargo bench -p lucene-codecs --bench blocktree_open`
// Test-support code opts out of the arithmetic gate at the file boundary:
// the gate exists for values read off disk in production decode paths, not
// for a fixture builder's own index arithmetic. See
// `docs/arithmetic-gate.md`.
#![allow(clippy::arithmetic_side_effects)]

use std::path::{Path, PathBuf};

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use lucene_codecs::regexp::RegexpPattern;
use lucene_codecs::{blocktree, field_infos};
use lucene_store::codec_util;
use lucene_store::data_input::{DataInput, SliceInput};

/// One real segment's term-dictionary bytes plus everything `blocktree::open`
/// needs to open them.
struct Corpus {
    tim: Vec<u8>,
    tip: Vec<u8>,
    tmd: Vec<u8>,
    field_infos: field_infos::FieldInfos,
    segment_id: [u8; 16],
    suffix: String,
}

fn corpus_dir() -> PathBuf {
    match std::env::var_os("LUCENE_RUST_BENCH_INDEX") {
        Some(p) => PathBuf::from(p),
        None => Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../benchmarks/.corpus/merged")
            .to_path_buf(),
    }
}

/// The `t0`..`t999999` dictionary the term-intersection cases run over --
/// `benchmarks/corpus/src/GenTermCorpus.java`, the shape
/// `docs/sweep/m2/b8-automata-analysis.md` measured the dead-prefix skip on.
fn terms_dir() -> PathBuf {
    match std::env::var_os("LUCENE_RUST_BENCH_TERMS_INDEX") {
        Some(p) => PathBuf::from(p),
        None => Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../benchmarks/.corpus/terms1m")
            .to_path_buf(),
    }
}

/// Finds the one file in `dir` ending in `ext`, or `None`.
fn find(dir: &Path, ext: &str) -> Option<PathBuf> {
    let mut hits: Vec<PathBuf> = std::fs::read_dir(dir)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.to_string_lossy().ends_with(ext))
        .collect();
    hits.sort();
    hits.into_iter().next()
}

fn load_corpus_at(dir: PathBuf) -> Option<Corpus> {
    let tim_path = find(&dir, ".tim")?;
    let tim = std::fs::read(&tim_path).ok()?;
    let tip = std::fs::read(find(&dir, ".tip")?).ok()?;
    let tmd = std::fs::read(find(&dir, ".tmd")?).ok()?;
    let fnm = std::fs::read(find(&dir, ".fnm")?).ok()?;

    // `_1z_Lucene104_0.tim` -> `Lucene104_0`; the codec suffix is part of every
    // index header this triple carries.
    let name = tim_path.file_name()?.to_string_lossy().to_string();
    let stem = name.strip_suffix(".tim")?;
    let suffix = stem
        .trim_start_matches('_')
        .split_once('_')
        .map(|(_, s)| s.to_string())?;

    // The segment id is the 16 bytes after the `IndexHeader`'s magic/codec
    // name/version -- read it straight off the `.tim` rather than parsing the
    // `.si`, which this crate has no reader for.
    let mut r = SliceInput::new(&tim);
    codec_util::check_header(&mut r, "BlockTreeTermsDict", 0, 0).ok()?;
    let mut segment_id = [0u8; 16];
    r.read_bytes(&mut segment_id).ok()?;

    let field_infos = field_infos::parse(&fnm, &segment_id, "").ok()?;
    Some(Corpus {
        tim,
        tip,
        tmd,
        field_infos,
        segment_id,
        suffix,
    })
}

impl Corpus {
    fn open(&self) -> blocktree::BlockTreeFields {
        blocktree::open(
            &self.tim,
            &self.tip,
            &self.tmd,
            &self.field_infos,
            &self.segment_id,
            &self.suffix,
            i32::MAX,
        )
        .expect("open the benchmark corpus' term dictionary")
    }
}

/// Deterministic pseudo-random ordering, so the seek benchmark measures cold
/// block loads rather than one hot block.
fn shuffle<T>(v: &mut [T]) {
    let mut state = 0x9E3779B97F4A7C15u64;
    for i in (1..v.len()).rev() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        v.swap(i, (state % (i as u64 + 1)) as usize);
    }
}

fn bench_blocktree(c: &mut Criterion) {
    let Some(corpus) = load_corpus_at(corpus_dir()) else {
        eprintln!(
            "blocktree_open: no benchmark corpus at {} -- skipping (run scripts/bench-corpus.sh)",
            corpus_dir().display()
        );
        return;
    };

    let mut group = c.benchmark_group("blocktree");
    group.sample_size(10);
    group.bench_function("open", |b| b.iter(|| black_box(corpus.open())));
    group.finish();

    let fields = corpus.open();
    // The widest field in the corpus is the analysed `body` text field; fall
    // back to whichever field has the most terms if this index has no `body`.
    let (field_name, field) = fields
        .iter_fields()
        .max_by_key(|(_, f)| f.num_terms)
        .expect("the corpus has at least one indexed field");
    eprintln!(
        "blocktree_open: field {field_name:?} has {} terms",
        field.num_terms
    );

    // A sample of real terms, in a shuffled order, plus their misses.
    let mut terms: Vec<Vec<u8>> = Vec::new();
    let mut it = field.iter();
    let mut n = 0usize;
    while let Some((t, _)) = it.next() {
        if n.is_multiple_of(97) {
            terms.push(t.to_vec());
        }
        n += 1;
    }
    shuffle(&mut terms);
    terms.truncate(2000);
    let misses: Vec<Vec<u8>> = terms
        .iter()
        .map(|t| {
            let mut m = t.clone();
            m.push(b'~');
            m
        })
        .collect();

    let mut group = c.benchmark_group("blocktree/seek_exact");
    group.bench_function("hit", |b| {
        b.iter(|| {
            let mut acc = 0i64;
            for t in &terms {
                acc += field
                    .seek_exact(black_box(t))
                    .map_or(0, |s| s.doc_freq as i64);
            }
            black_box(acc)
        })
    });
    group.bench_function("miss", |b| {
        b.iter(|| {
            let mut acc = 0i64;
            for t in &misses {
                acc += field
                    .seek_exact(black_box(t))
                    .map_or(0, |s| s.doc_freq as i64);
            }
            black_box(acc)
        })
    });
    group.finish();

    // Ordered enumeration of the whole field -- `TermsEnum.next()`.
    let mut group = c.benchmark_group("blocktree/next");
    group.sample_size(10);
    group.bench_function("whole_field", |b| {
        b.iter(|| {
            let mut it = field.iter();
            let mut n = 0u64;
            while let Some((t, s)) = it.next() {
                n += t.len() as u64 + s.doc_freq as u64;
            }
            black_box(n)
        })
    });
    group.finish();
}

/// `FieldReader.intersect`'s equivalent: how much the dead-prefix skip saves
/// now that skipping a prefix range also skips *loading* the `.tim` blocks
/// under it -- the win `docs/sweep/m2/b8-automata-analysis.md` could not get
/// while the whole dictionary was decoded at open.
///
/// `scan` is the same walk with the skip taken out: seek to the pattern's
/// literal prefix, then `next()` through the range testing every term.
fn bench_regexp_intersect(c: &mut Criterion) {
    let Some(corpus) = load_corpus_at(terms_dir()) else {
        eprintln!(
            "blocktree_open: no term corpus at {} -- skipping the intersect cases \
             (javac benchmarks/corpus/src/GenTermCorpus.java and run it)",
            terms_dir().display()
        );
        return;
    };
    let fields = corpus.open();
    let (_, field) = fields
        .iter_fields()
        .max_by_key(|(_, f)| f.num_terms)
        .expect("the term corpus has at least one indexed field");

    for src in ["t1[0-9]", "t1*z", "t[0-9]{4}", "t.*99"] {
        let pattern = RegexpPattern::new(src.as_bytes()).expect("valid pattern");
        let prefix = pattern.literal_prefix();

        // No-skip baseline: walk the whole literal-prefix range.
        let scan = || {
            let mut it = field.iter();
            let mut hits = 0usize;
            if it.seek_ceil(&prefix) == blocktree::SeekStatus::End {
                return hits;
            }
            while let Some((term, _)) = it.current() {
                if !term.starts_with(&prefix) {
                    break;
                }
                if pattern.matches(term) {
                    hits += 1;
                }
                if it.next().is_none() {
                    break;
                }
            }
            hits
        };
        let skip = || field.regexp_intersect(&pattern).count();
        assert_eq!(scan(), skip(), "{src}: the skip changed the match count");

        let mut group = c.benchmark_group(format!("blocktree/regexp_intersect/{src}"));
        group.sample_size(10);
        group.bench_function("scan", |b| b.iter(|| black_box(scan())));
        group.bench_function("skip", |b| b.iter(|| black_box(skip())));
        group.finish();
    }
}

criterion_group!(benches, bench_blocktree, bench_regexp_intersect);
criterion_main!(benches);
