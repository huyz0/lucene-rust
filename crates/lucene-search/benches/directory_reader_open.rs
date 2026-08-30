//! Settles c1's F-13: how much of `DirectoryReader::open`'s residual cost is
//! the `.tim`/`.tip` copy that `blocktree::open_shared` was meant to avoid?
//!
//! c1 took `blocktree::open` from 35.4 ms to 0.175 ms by making the term
//! dictionary lazy, and recorded that most of what remains is a 4.86 MB
//! `Arc::from(&tim[..])` copy out of the mapping. `open_shared` takes the
//! shared buffers directly and skips that copy -- but only if the caller can
//! *produce* an `Arc<[u8]>` without copying, which `lucene-store`'s `Input`
//! (an owned `Vec` or a `memmap2::Mmap`) cannot. This bench measures the three
//! quantities that decide the question:
//!
//! - `directory_reader_open` -- a whole `DirectoryReader::open`, the number a
//!   caller actually pays.
//! - `blocktree_open` / `blocktree_open_shared` -- the same term dictionary
//!   opened with and without the copy, from bytes already in memory. The
//!   difference between them *is* the copy.
//! - `arc_from_tim` -- the copy on its own, so the two above can be checked
//!   against it.
//!
//! Measured against `benchmarks/.corpus/merged` (`scripts/bench-corpus.sh`):
//! one force-merged, real-Lucene-written segment, ~4.7 MB `.tim`, ~89 KB
//! `.tip`, 579k terms. **The bench skips itself when that corpus is absent**,
//! since it is generated and deliberately not checked in.
//!
//! Run with `cargo bench -p lucene-search --bench directory_reader_open`.

use std::sync::Arc;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use lucene_codecs::{blocktree, field_infos};
use lucene_search::directory_reader::DirectoryReader;
use lucene_store::codec_util;
use lucene_store::data_input::{DataInput, SliceInput};
use lucene_store::{Directory, MmapDirectory};

fn corpus_dir() -> Option<std::path::PathBuf> {
    let dir = std::path::Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../benchmarks/.corpus/merged"
    ));
    dir.join("segments_1").exists().then(|| dir.to_path_buf())
}

/// The one segment's term-dictionary bytes, plus what `blocktree::open` needs.
struct Dict {
    tim: Vec<u8>,
    tip: Vec<u8>,
    tmd: Vec<u8>,
    field_infos: field_infos::FieldInfos,
    segment_id: [u8; 16],
    suffix: String,
}

/// Finds the one file in `dir` ending in `ext`.
fn find(dir: &std::path::Path, ext: &str) -> std::path::PathBuf {
    let mut hits: Vec<std::path::PathBuf> = std::fs::read_dir(dir)
        .expect("read corpus dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.to_string_lossy().ends_with(ext))
        .collect();
    hits.sort();
    hits.into_iter()
        .next()
        .unwrap_or_else(|| panic!("no {ext}"))
}

/// Same loader `lucene-codecs`' own `blocktree_open` bench uses: the segment
/// id is read straight off the `.tim` index header rather than parsed out of
/// the `.si`.
fn load_dict(dir: &std::path::Path) -> Dict {
    let tim_path = find(dir, ".tim");
    let tim = std::fs::read(&tim_path).expect("read .tim");
    let tip = std::fs::read(find(dir, ".tip")).expect("read .tip");
    let tmd = std::fs::read(find(dir, ".tmd")).expect("read .tmd");
    let fnm = std::fs::read(find(dir, ".fnm")).expect("read .fnm");

    let name = tim_path
        .file_name()
        .expect("file name")
        .to_string_lossy()
        .to_string();
    let stem = name.strip_suffix(".tim").expect(".tim suffix");
    let suffix = stem
        .trim_start_matches('_')
        .split_once('_')
        .map(|(_, s)| s.to_string())
        .expect("codec suffix");

    let mut r = SliceInput::new(&tim);
    codec_util::check_header(&mut r, "BlockTreeTermsDict", 0, 0).expect("tim header");
    let mut segment_id = [0u8; 16];
    r.read_bytes(&mut segment_id).expect("segment id");

    let field_infos = field_infos::parse(&fnm, &segment_id, "").expect("parse .fnm");
    Dict {
        tim,
        tip,
        tmd,
        field_infos,
        segment_id,
        suffix,
    }
}

fn bench(c: &mut Criterion) {
    let Some(dir_path) = corpus_dir() else {
        eprintln!(
            "directory_reader_open: benchmarks/.corpus/merged is absent -- \
             run scripts/bench-corpus.sh to measure this. Skipping."
        );
        return;
    };

    let mut group = c.benchmark_group("directory_reader_open");

    group.bench_function("directory_reader_open", |b| {
        b.iter(|| {
            let dir = MmapDirectory::open(dir_path.clone());
            black_box(DirectoryReader::open(&dir).expect("open index"));
        })
    });

    let d = load_dict(&dir_path);
    let max_doc = i32::MAX;

    group.bench_function("blocktree_open", |b| {
        b.iter(|| {
            black_box(
                blocktree::open(
                    &d.tim,
                    &d.tip,
                    &d.tmd,
                    &d.field_infos,
                    &d.segment_id,
                    &d.suffix,
                    max_doc,
                )
                .expect("open dict"),
            );
        })
    });

    let shared_tim: blocktree::SharedBytes = Arc::new(d.tim.clone());
    let shared_tip: blocktree::SharedBytes = Arc::new(d.tip.clone());
    group.bench_function("blocktree_open_shared", |b| {
        b.iter(|| {
            black_box(
                blocktree::open_shared(
                    blocktree::SharedBytes::clone(&shared_tim),
                    blocktree::SharedBytes::clone(&shared_tip),
                    &d.tmd,
                    &d.field_infos,
                    &d.segment_id,
                    &d.suffix,
                    max_doc,
                )
                .expect("open dict"),
            );
        })
    });

    // The shape `directory_reader` now uses: an `Arc<Input>` (a mapping)
    // handed straight to `open_shared`, with no copy anywhere.
    group.bench_function("blocktree_open_shared_from_mmap", |b| {
        let dir = MmapDirectory::open(dir_path.clone());
        let tim_name = find(&dir_path, ".tim");
        let tip_name = find(&dir_path, ".tip");
        let name = |p: &std::path::Path| p.file_name().unwrap().to_string_lossy().into_owned();
        b.iter(|| {
            let tim: blocktree::SharedBytes =
                Arc::new(dir.open(&name(&tim_name)).expect("map .tim"));
            let tip: blocktree::SharedBytes =
                Arc::new(dir.open(&name(&tip_name)).expect("map .tip"));
            black_box(
                blocktree::open_shared(
                    tim,
                    tip,
                    &d.tmd,
                    &d.field_infos,
                    &d.segment_id,
                    &d.suffix,
                    max_doc,
                )
                .expect("open dict"),
            );
        })
    });

    group.bench_function("arc_from_tim", |b| {
        b.iter(|| {
            let a: Arc<[u8]> = Arc::from(&d.tim[..]);
            black_box(a);
        })
    });

    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
