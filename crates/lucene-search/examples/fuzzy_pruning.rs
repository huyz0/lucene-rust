//! Alternating A/B measurement of `FuzzyTermsEnum`'s
//! `MaxNonCompetitiveBoostAttribute` feedback loop: the same corpus, the same
//! query and one build, expanded once with the loop and once without.
//!
//! What the loop does, in one sentence: `TopTermsRewrite.collectTerms`
//! publishes the worst boost still in its size-`maxExpansions` queue, and
//! `FuzzyTermsEnum.bottomChanged` drops `maxEdits` for as long as no term at
//! that distance could still compete, swapping the enumeration onto the
//! smaller automaton. This port has no automaton -- its fuzzy matcher is a
//! banded DP -- so the same signal narrows the band and tightens the
//! length filter instead, which is where the time goes.
//!
//! Criterion is not used: it reported 83/91/129 µs for identical code on this
//! project's machine (`docs/sweep/m2/c24-arith-codecs.md`), so every figure
//! here is a **min of N alternating repetitions**, the statistic that survives
//! a noisy host. Both arms run in one process from one build, so there is no
//! second binary and no rebuild between them -- only the `prune` argument.
//!
//! Corpus: `benchmarks/.corpus/terms1m` (`benchmarks/corpus/src/
//! GenTermCorpus.java` -- `t0`..`t999999`, one term per document), the same
//! dictionary `crates/lucene-codecs/benches/blocktree_open.rs` measures term
//! intersection on. **The example skips itself when that corpus is absent.**
//!
//! Run: `cargo run --release -p lucene-search --example fuzzy_pruning`
// Benchmark support code opts out of the arithmetic gate at the file
// boundary, as the fixture writers do. See `docs/arithmetic-gate.md`.
#![allow(clippy::arithmetic_side_effects)]

use lucene_codecs::blocktree::{self, BlockTreeFields};
use lucene_codecs::field_infos;
use lucene_search::{bench_only_fuzzy_expansion, FuzzyQuery};
use lucene_store::codec_util;
use lucene_store::data_input::{DataInput, SliceInput};
use std::hint::black_box;
use std::path::{Path, PathBuf};
use std::time::Instant;

fn terms_dir() -> PathBuf {
    match std::env::var_os("LUCENE_RUST_BENCH_TERMS_INDEX") {
        Some(p) => PathBuf::from(p),
        None => Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../benchmarks/.corpus/terms1m")
            .to_path_buf(),
    }
}

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

struct Corpus {
    tim: Vec<u8>,
    tip: Vec<u8>,
    tmd: Vec<u8>,
    field_infos: field_infos::FieldInfos,
    segment_id: [u8; 16],
    suffix: String,
}

impl Corpus {
    fn open(&self) -> BlockTreeFields {
        blocktree::open(
            &self.tim,
            &self.tip,
            &self.tmd,
            &self.field_infos,
            &self.segment_id,
            &self.suffix,
            i32::MAX,
        )
        .expect("open the term dictionary")
    }
}

/// Same loader as `crates/lucene-codecs/benches/blocktree_open.rs`: the
/// segment id is read off the `.tim`'s own index header rather than the `.si`,
/// which this crate has no reader for.
fn load(dir: PathBuf) -> Option<Corpus> {
    let tim_path = find(&dir, ".tim")?;
    let tim = std::fs::read(&tim_path).ok()?;
    let tip = std::fs::read(find(&dir, ".tip")?).ok()?;
    let tmd = std::fs::read(find(&dir, ".tmd")?).ok()?;
    let fnm = std::fs::read(find(&dir, ".fnm")?).ok()?;

    let name = tim_path.file_name()?.to_string_lossy().to_string();
    let stem = name.strip_suffix(".tim")?;
    let suffix = stem
        .trim_start_matches('_')
        .split_once('_')
        .map(|(_, s)| s.to_string())?;

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

fn main() {
    let Some(corpus) = load(terms_dir()) else {
        eprintln!(
            "fuzzy_pruning: no term corpus at {} -- skipping. \
             (javac benchmarks/corpus/src/GenTermCorpus.java and run it.)",
            terms_dir().display()
        );
        return;
    };
    let fields = corpus.open();
    let (field_name, field) = fields
        .iter_fields()
        .max_by_key(|(_, f)| f.num_terms)
        .expect("the corpus has at least one indexed field");
    let field_name = field_name.to_string();
    println!(
        "fuzzy_pruning: field {field_name:?}, {} terms",
        field.num_terms
    );
    println!();
    println!(
        "{:<34} {:>11} {:>11} {:>8} {:>7}",
        "case", "pruned", "whole", "speedup", "edits"
    );

    let reps = 40usize;
    // Query terms of different lengths, because `bottomChanged`'s threshold is
    // `1 - maxEdits / termLength`: the longer the query term, the higher the
    // boost of a distance-`k` match, so the sooner the budget can fall.
    let cases: [(&str, &[u8], u8, usize); 6] = [
        ("t100000 ed2 exp50", b"t100000", 2, 50),
        ("t100000 ed2 exp10", b"t100000", 2, 10),
        ("t100000 ed1 exp50", b"t100000", 1, 50),
        ("t12345 ed2 exp50", b"t12345", 2, 50),
        ("t1234 ed2 exp50", b"t1234", 2, 50),
        ("t100000 ed2 exp50 pfx2", b"t100000", 2, 50),
    ];

    for (i, (label, term, max_edits, max_expansions)) in cases.iter().enumerate() {
        let prefix_length = if i == 5 { 2 } else { 0 };
        let query = FuzzyQuery::new(field_name.clone(), term.to_vec())
            .with_max_edits(*max_edits)
            .with_prefix_length(prefix_length)
            .with_max_expansions(*max_expansions);

        // Correctness first: the two arms must agree before either timing
        // means anything.
        let (pruned_terms, pruned_df, final_edits) =
            bench_only_fuzzy_expansion(field, &query, true).expect("expand");
        let (whole_terms, whole_df, _) =
            bench_only_fuzzy_expansion(field, &query, false).expect("expand");
        assert_eq!(pruned_terms, whole_terms, "{label}: the arms disagree");
        assert_eq!(pruned_df, whole_df, "{label}: blended df disagrees");

        let mut pruned = u128::MAX;
        let mut whole = u128::MAX;
        let mut checksum = 0usize;
        for _ in 0..reps {
            let t = Instant::now();
            checksum ^= black_box(bench_only_fuzzy_expansion(field, &query, true))
                .expect("expand")
                .0
                .len();
            pruned = pruned.min(t.elapsed().as_nanos());

            let t = Instant::now();
            checksum ^= black_box(bench_only_fuzzy_expansion(field, &query, false))
                .expect("expand")
                .0
                .len();
            whole = whole.min(t.elapsed().as_nanos());
        }
        black_box(checksum);

        println!(
            "{label:<34} {:>9.1} µs {:>9.1} µs {:>7.2}x {:>4} -> {}",
            pruned as f64 / 1000.0,
            whole as f64 / 1000.0,
            whole as f64 / pruned as f64,
            max_edits,
            final_edits,
        );
    }
}
