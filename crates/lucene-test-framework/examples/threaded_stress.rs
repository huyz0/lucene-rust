//! M14.h h8's acceptance run: `ThreadedIndexingAndSearchingTestCase` over
//! a range of seeds, alternating a fresh NRT reader per search with a shared
//! reader reopened when changed. Each seed's run must pass every check the
//! test makes (deletes, packs in order, surviving documents, the document
//! count before and after a commit, `CheckIndex`).
//!
//! Usage: `threaded_stress [--seeds A..B | --seed N] [--multiplier M]`.

use std::time::Instant;

use lucene_test_framework::index::threaded_indexing_and_searching::{
    run_test, ReaderMode, ThreadedRunStats,
};
use lucene_test_framework::util::lucene_test_case::LuceneTestCase;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let value = |flag: &str| {
        args.iter()
            .position(|a| a == flag)
            .and_then(|i| args.get(i + 1))
            .map(String::as_str)
    };
    let (first, end): (i64, i64) = match (value("--seed"), value("--seeds")) {
        (Some(s), _) => {
            let s: i64 = s.parse().expect("--seed");
            (s, s + 1)
        }
        (None, Some(range)) => {
            let (a, b) = range.split_once("..").expect("--seeds A..B");
            (a.parse().expect("seed"), b.parse().expect("seed"))
        }
        (None, None) => (0, 20),
    };
    assert!(first < end, "--seeds A..B needs A < B");
    let multiplier: i32 = value("--multiplier").map_or(2, |m| m.parse().expect("--multiplier"));
    let started = Instant::now();
    let mut total = ThreadedRunStats::default();
    let mut runs = 0;
    for seed in first..end {
        let mode = if seed % 2 == 0 {
            ReaderMode::Nrt
        } else {
            ReaderMode::Reopen
        };
        let result = std::panic::catch_unwind(|| {
            let mut t = LuceneTestCase::with_config("threadedStress", seed, false, multiplier);
            run_test(&mut t, "threadedStress", mode)
        });
        match result {
            Ok(s) => {
                println!(
                    "seed {seed} ({mode:?}): ok -- {} adds, {} deletes, {} packs, {} searches, \
                     {} merged segment(s) seen",
                    s.adds, s.deletes, s.packs, s.searches, s.merged_segments_seen
                );
                total.adds += s.adds;
                total.deletes += s.deletes;
                total.packs += s.packs;
                total.searches += s.searches;
                total.term_queries += s.term_queries;
                total.merged_segments_seen += s.merged_segments_seen;
                total.warmed += s.warmed;
                runs += 1;
            }
            Err(_) => {
                println!("seed {seed} ({mode:?}): FAIL");
                println!("replay: threaded_stress --seed {seed} --multiplier {multiplier}");
                std::process::exit(1);
            }
        }
    }
    println!(
        "threaded_stress: {runs} run(s), {} adds, {} deletes, {} packs, {} searches, {} term \
         queries, {} merged segment(s) seen, {} warmed in {:.1}s",
        total.adds,
        total.deletes,
        total.packs,
        total.searches,
        total.term_queries,
        total.merged_segments_seen,
        total.warmed,
        started.elapsed().as_secs_f64()
    );
}
