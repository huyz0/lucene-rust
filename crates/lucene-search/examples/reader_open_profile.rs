//! Where the time in `DirectoryReader::open` actually goes.
//!
//! Ledger item 14 has carried a 52.7 ms figure since `verdict-m1.6.md`, taken
//! before c1 made the term dictionary lazy and before c12's `open_shared`
//! stopped copying `.tim`/`.tip`. The ledger's own instruction is to
//! re-measure before planning against it, and a single end-to-end number
//! cannot say what is left -- so this replays `SegmentReader::open`'s phases
//! one at a time, over the same directory, in the same order, and prints the
//! breakdown next to the real `DirectoryReader::open` it is a replica of.
//!
//! The replica is checked, not assumed: the phase total is printed beside the
//! measured whole, and a replica that has drifted shows up as a gap between
//! them.
//!
//! Every figure is a **min of N alternating repetitions**, not a mean and not
//! criterion's estimate -- criterion reported 83/91/129 µs for identical code
//! on this host (`docs/sweep/m2/c24-arith-codecs.md`).
//!
//! ```text
//! cargo build -p lucene-search --release --example reader_open_profile
//! ./target/release/examples/reader_open_profile <index-dir> [reps]
//! ```
// A measurement harness's own arithmetic, as the other benches do. See
// `docs/arithmetic-gate.md`.
#![allow(clippy::arithmetic_side_effects)]

use std::hint::black_box;
use std::sync::Arc;
use std::time::Instant;

use lucene_codecs::postings::{DocInput, PayInput, PosInput};
use lucene_codecs::{blocktree, doc_values, field_infos, norms};
use lucene_index::{segment_info, segment_infos};
use lucene_search::directory_reader::DirectoryReader;
use lucene_store::{Directory, FsDirectory, Input, MmapDirectory};

/// Runs `op` `reps` times and keeps the minimum, which is the statistic that
/// survives a noisy host (see the module comment).
fn best(reps: usize, mut op: impl FnMut() -> u64) -> u128 {
    let mut min = u128::MAX;
    for _ in 0..reps {
        let t = Instant::now();
        let sink = op();
        min = min.min(t.elapsed().as_nanos());
        black_box(sink);
    }
    min
}

fn us(ns: u128) -> f64 {
    ns as f64 / 1000.0
}

fn main() {
    let mut args = std::env::args().skip(1);
    let index = args
        .next()
        .expect("usage: reader_open_profile <index-dir> [reps]");
    let reps: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(40);

    let dir = MmapDirectory::open(index.clone());
    // The A/B arms of the small-file change, both in this one process:
    // `map_everything` is what `MmapDirectory` did before
    // `SMALL_FILE_READ_THRESHOLD` existed.
    let map_everything = MmapDirectory::with_read_threshold(index.clone(), 0);

    // The whole thing, as `benchmarks/rust-runner`'s `reader_open` case runs
    // it: the reader plus the `.doc`/`.pos`/`.pay` opens a search needs.
    // Alternated arm by arm, so any drift over the run falls on both equally.
    let mut whole = u128::MAX;
    let mut whole_mapped = u128::MAX;
    let mut whole_with_segments = u128::MAX;
    let mut whole_with_segments_mapped = u128::MAX;
    // Means as well as minima. `benchmarks/rust-runner`'s `reader_open` case
    // reports a mean over a 1.5 s timed loop, and on this operation the two
    // statistics are far apart -- an open maps 1.6 GB of `.doc`/`.pos`/`.pay`
    // and unmaps it again, and that tail lands in the mean and not the min.
    // Printing both is what keeps the two harnesses' numbers comparable.
    let mut whole_sum = 0u128;
    let mut whole_mapped_sum = 0u128;
    for _ in 0..reps {
        let one = |d: &dyn Directory| {
            let t = Instant::now();
            let reader = DirectoryReader::open(d).expect("open");
            black_box(reader.segment_readers().len());
            t.elapsed().as_nanos()
        };
        let a = one(&dir);
        whole = whole.min(a);
        whole_sum += a;
        let b = one(&map_everything);
        whole_mapped = whole_mapped.min(b);
        whole_mapped_sum += b;

        let both = |d: &dyn Directory| {
            let t = Instant::now();
            let reader = DirectoryReader::open(d).expect("open");
            let opened = reader.open_segments().expect("open segments");
            black_box(opened.as_open_segments().len());
            t.elapsed().as_nanos()
        };
        whole_with_segments = whole_with_segments.min(both(&dir));
        whole_with_segments_mapped = whole_with_segments_mapped.min(both(&map_everything));
    }

    // Phase replica. Everything below is exactly what `SegmentReader::open`
    // does for a non-compound segment, called through the same public
    // functions, so a phase's cost here is that phase's cost there.
    let infos = segment_infos::read_latest(&dir).expect("segments_N");
    let commit = infos
        .segments
        .first()
        .expect("index has at least one segment");
    let name = commit.segment_name.clone();
    let id = commit.segment_id;

    let si_bytes = dir.open(&format!("{name}.si")).expect(".si");
    let si = segment_info::parse(&si_bytes, &id).expect("parse .si");
    assert!(
        !si.is_compound_file,
        "reader_open_profile replicates the non-compound path only"
    );

    let file = |ext: &str| -> Option<String> {
        si.files
            .iter()
            .find(|f| f.ends_with(ext))
            .map(ToString::to_string)
    };
    let suffix_of = |file_name: &str, ext: &str| -> String {
        if file_name == format!("{name}{ext}") {
            return String::new();
        }
        file_name
            .strip_prefix(&format!("{name}_"))
            .and_then(|s| s.strip_suffix(ext))
            .unwrap_or_default()
            .to_string()
    };

    let mut rows: Vec<(String, u128)> = Vec::new();
    let mut push = |label: &str, ns: u128| rows.push((label.to_string(), ns));

    push(
        "segment_infos::read_latest (list_all + parse segments_N)",
        best(reps, || {
            segment_infos::read_latest(&dir)
                .expect("segments_N")
                .segments
                .len() as u64
        }),
    );
    push(
        ".si open + parse",
        best(reps, || {
            let b = dir.open(&format!("{name}.si")).expect(".si");
            let si = segment_info::parse(&b, &id).expect("parse");
            si.doc_count as u64
        }),
    );

    let fnm = file(".fnm").expect(".fnm");
    push(
        ".fnm open + parse",
        best(reps, || {
            let b = dir.open(&fnm).expect(".fnm");
            let fi = field_infos::parse(&b, &id, "").expect("parse");
            fi.fields.len() as u64
        }),
    );

    // The three term-dictionary files, opened then handed to `open_shared`.
    let (tim, tip, tmd) = match (file(".tim"), file(".tip"), file(".tmd")) {
        (Some(a), Some(b), Some(c)) => (a, b, c),
        _ => {
            eprintln!("index has no term dictionary; nothing further to profile");
            return;
        }
    };
    let postings_suffix = suffix_of(&tim, ".tim");
    let fi = field_infos::parse(&dir.open(&fnm).expect(".fnm"), &id, "").expect("parse .fnm");

    push(
        ".tim/.tip/.tmd mmap only",
        best(reps, || {
            let a = dir.open(&tim).expect(".tim");
            let b = dir.open(&tip).expect(".tip");
            let c = dir.open(&tmd).expect(".tmd");
            (a.len() + b.len() + c.len()) as u64
        }),
    );
    let open_shared_only = {
        let a: Arc<Input> = Arc::new(dir.open(&tim).expect(".tim"));
        let b: Arc<Input> = Arc::new(dir.open(&tip).expect(".tip"));
        let c = dir.open(&tmd).expect(".tmd");
        best(reps, || {
            let f = blocktree::open_shared(
                Arc::clone(&a) as blocktree::SharedBytes,
                Arc::clone(&b) as blocktree::SharedBytes,
                &c,
                &fi,
                &id,
                &postings_suffix,
                si.doc_count,
            )
            .expect("open_shared");
            f.iter_fields().count() as u64
        })
    };
    push("blocktree::open_shared (mappings held)", open_shared_only);

    for ext in [".doc", ".pos", ".pay", ".kdm", ".kdi", ".kdd"] {
        let Some(f) = file(ext) else { continue };
        push(
            &format!("{ext} mmap"),
            best(reps, || dir.open(&f).expect(ext).len() as u64),
        );
    }

    if let (Some(dvm), Some(dvd)) = (file(".dvm"), file(".dvd")) {
        let dv_suffix = suffix_of(&dvm, ".dvm");
        push(
            ".dvm/.dvd open + parse_meta",
            best(reps, || {
                let m = dir.open(&dvm).expect(".dvm");
                let d = dir.open(&dvd).expect(".dvd");
                let (_, meta) = doc_values::parse_meta(&m, &id, &dv_suffix, &fi).expect("parse");
                (meta.numeric.len() + d.len()) as u64
            }),
        );
    }

    if let (Some(nvm), Some(nvd)) = (file(".nvm"), file(".nvd")) {
        let norms_suffix = suffix_of(&nvm, ".nvm");
        push(
            ".nvm/.nvd open + parse_meta + validate + footer",
            best(reps, || {
                let m = dir.open(&nvm).expect(".nvm");
                let d = dir.open(&nvd).expect(".nvd");
                let (_, meta) = norms::parse_meta(&m, &id, &norms_suffix).expect("parse");
                norms::validate_fields(&meta, &fi).expect("validate");
                norms::check_data_header_footer(&d, &id, &norms_suffix).expect("footer");
                d.len() as u64
            }),
        );
    }

    // How much of a small file's open is the mapping itself: the same file,
    // opened through the copying backend (`fs::read`) and the mapping one.
    // A `.si`/`.fnm`/`.dvm`/`.nvm` is a few hundred bytes -- one `read` --
    // where a mapping costs an `mmap`, a fault and a `munmap`. The `mmap`
    // arm has to bypass `MmapDirectory`'s own small-file threshold to be
    // measured at all, which is what `with_read_threshold(_, 0)` is for.
    let fsdir = FsDirectory::open(index.clone());
    let dir = &map_everything;
    for f in [
        file(".si"),
        Some(fnm.clone()),
        file(".dvm"),
        file(".nvm"),
        file(".tmd"),
    ]
    .into_iter()
    .flatten()
    {
        let len = dir.open(&f).expect("open").len();
        push(
            &format!("  [{f} {len}B] mmap open"),
            best(reps, || dir.open(&f).expect("open").len() as u64),
        );
        push(
            &format!("  [{f} {len}B] fs::read open"),
            best(reps, || fsdir.open(&f).expect("open").len() as u64),
        );
    }

    // `open_segments`: the three postings inputs, header + footer framing.
    let doc_buf = file(".doc").map(|f| dir.open(&f).expect(".doc"));
    let pos_buf = file(".pos").map(|f| dir.open(&f).expect(".pos"));
    let pay_buf = file(".pay").map(|f| dir.open(&f).expect(".pay"));
    push(
        "open_segments: Doc/Pos/PayInput::open (mappings held)",
        best(reps, || {
            let mut n = 0u64;
            if let Some(b) = &doc_buf {
                n += DocInput::open(b, &id, &postings_suffix).is_ok() as u64;
            }
            if let Some(b) = &pos_buf {
                n += PosInput::open(b, &id, &postings_suffix).is_ok() as u64;
            }
            if let Some(b) = &pay_buf {
                n += PayInput::open(b, &id, &postings_suffix).is_ok() as u64;
            }
            n
        }),
    );

    let width = rows.iter().map(|(l, _)| l.len()).max().unwrap_or(0);
    println!(
        "index: {index}  segments: {}  reps: {reps}",
        infos.segments.len()
    );
    println!("{:<width$}  {:>10}", "phase", "min us", width = width);
    let mut total = 0u128;
    for (label, ns) in &rows {
        if !label.starts_with("  [") {
            total += ns;
        }
        println!("{label:<width$}  {:>10.3}", us(*ns), width = width);
    }
    println!(
        "{:<width$}  {:>10.3}",
        "-- phase sum",
        us(total),
        width = width
    );
    for (label, ns) in [
        (
            "DirectoryReader::open MEAN (small files read)",
            whole_sum / reps as u128,
        ),
        (
            "DirectoryReader::open MEAN (everything mapped)",
            whole_mapped_sum / reps as u128,
        ),
        ("DirectoryReader::open (small files read)", whole),
        ("DirectoryReader::open (everything mapped)", whole_mapped),
        (
            "open + open_segments (small files read)",
            whole_with_segments,
        ),
        (
            "open + open_segments (everything mapped)",
            whole_with_segments_mapped,
        ),
    ] {
        println!("{label:<width$}  {:>10.3}", us(ns), width = width);
    }
}
