//! `FieldTerms::compiled_terms` (`CompiledAutomaton.getTermsEnum(terms)`):
//! the block-tree walk stepping a `CompiledAutomaton`'s byte automaton, over
//! the real-Lucene 65 000-term dictionary in
//! `fixtures/data/regexp_intersect_index` (floor blocks, sub-blocks,
//! multi-byte UTF-8). Every byte the walk tests goes through the compiled
//! automaton's `step`, so this measures how the walk dispatches to it.
//!
//! Run with: `cargo bench -p lucene-codecs --bench compiled_intersect`
// Test-support code opts out of the arithmetic gate at the file boundary:
// see `docs/arithmetic-gate.md`.
#![allow(clippy::arithmetic_side_effects)]

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use lucene_codecs::{blocktree, field_infos};
use lucene_util::automaton::{operations, CompiledAutomaton, LevenshteinAutomata, RegExp};

fn dir() -> String {
    concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/data/regexp_intersect_index/"
    )
    .to_string()
}

fn open() -> blocktree::BlockTreeFields {
    let m: std::collections::HashMap<String, String> =
        std::fs::read_to_string(format!("{}manifest.properties", dir()))
            .expect("fixture manifest")
            .lines()
            .filter_map(|l| l.split_once('='))
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
    let read = |key: &str| std::fs::read(format!("{}{}", dir(), m[key])).unwrap();
    let mut id = [0u8; 16];
    for (i, slot) in id.iter_mut().enumerate() {
        *slot = u8::from_str_radix(&m["id_hex"][i * 2..i * 2 + 2], 16).unwrap();
    }
    let suffix = m["tim_file_name"]
        .trim_end_matches(".tim")
        .splitn(3, '_')
        .nth(2)
        .unwrap()
        .to_string();
    let fi = field_infos::parse(&read("fnm_file_name"), &id, "").expect("parse .fnm");
    blocktree::open(
        &read("tim_file_name"),
        &read("tip_file_name"),
        &read("tmd_file_name"),
        &fi,
        &id,
        &suffix,
        m["max_doc"].parse().unwrap(),
    )
    .expect("open blocktree")
}

fn bench_compiled_terms(c: &mut Criterion) {
    let fields = open();
    let field = fields.field("body").expect("body");
    let mut group = c.benchmark_group("compiled_terms");
    let det = |re: &str| {
        let a = RegExp::with_flags(re, RegExp::ALL, 0)
            .unwrap()
            .to_automaton()
            .unwrap();
        let a = operations::determinize(&a, operations::DEFAULT_DETERMINIZE_WORK_LIMIT).unwrap();
        CompiledAutomaton::with_options(&a, false, true, false).unwrap()
    };
    let mut cases: Vec<(String, CompiledAutomaton)> = [".*z", "t.*9", ".*1.*2.*", "t[^0-9].*"]
        .iter()
        .map(|re| (format!("dfa {re}"), det(re)))
        .collect();
    let lev = LevenshteinAutomata::new("t1234", true)
        .to_automaton(2)
        .unwrap();
    cases.push((
        "dfa lev2 t1234".into(),
        CompiledAutomaton::new(&lev).unwrap(),
    ));
    // Left non-deterministic: the `NFARunAutomaton` branch.
    let nfa = RegExp::with_flags(".*1.*2.*", RegExp::ALL, 0)
        .unwrap()
        .to_automaton()
        .unwrap();
    assert!(!nfa.is_deterministic());
    cases.push(("nfa .*1.*2.*".into(), CompiledAutomaton::new(&nfa).unwrap()));
    for (name, compiled) in &cases {
        group.bench_function(name.as_str(), |b| {
            b.iter(|| {
                let mut n = 0usize;
                for r in field.compiled_terms(compiled) {
                    n += r.unwrap().0.len();
                }
                black_box(n)
            })
        });
    }
    group.finish();
}

criterion_group!(benches, bench_compiled_terms);
criterion_main!(benches);
