//! Differential test for `org.apache.lucene.util.fst.Util` and typed FST arc
//! reading against Lucene 10.5.0: replays `fixtures/data/fst_util/*.txt`
//! (written by `fixtures/src/GenFstUtil.java`) over Lucene-built FSTs --
//! `Util.get`, `shortestPaths` from the root and from prefix nodes, a
//! `TopNSearcher` with an `acceptResult` filter, `readCeilArc`, and `toDot`'s
//! exact text.
#![allow(clippy::arithmetic_side_effects)]

use std::cmp::Ordering;

use lucene_codecs::fst::{ByteSequenceOutputs, Pair, PairOutputs, PositiveIntOutputs};
use lucene_codecs::fst_compiler::FstOutputs;
use lucene_codecs::fst_util::{self, TopNSearcher, TopResults, TypedFst};

fn dir() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/data/fst_util")
}

fn unhex(s: &str) -> Vec<u8> {
    if s == "-" {
        return Vec::new();
    }
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
        .collect()
}

fn hex(b: &[u8]) -> String {
    if b.is_empty() {
        return "-".into();
    }
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn fmt_results<V>(res: &TopResults<V>, fmt: &dyn Fn(&V) -> String) -> String {
    let mut s = format!("{} {}", res.is_complete, res.top_n.len());
    for r in &res.top_n {
        s.push_str(&format!(
            " {}:{}",
            hex(&fst_util::to_bytes_ref(&r.input)),
            fmt(&r.output)
        ));
    }
    s
}

/// Walks `prefix` from the root: the arc reached and the output so far.
fn walk<O: FstOutputs>(
    fst: &TypedFst<O>,
    prefix: &[u8],
) -> Option<(fst_util::Arc<O::Value>, O::Value)> {
    let mut arc = fst.first_arc();
    let mut out = O::no_output();
    let mut r = fst.bytes_reader();
    for &b in prefix {
        arc = fst.find_target_arc(i32::from(b), &arc, &mut r).unwrap()?;
        out = O::add(&out, arc.output());
    }
    Some((arc, out))
}

fn replay<O: FstOutputs>(
    name: &str,
    text: &str,
    cmp: impl Fn(&O::Value, &O::Value) -> Ordering + Clone,
    fmt: &dyn Fn(&O::Value) -> String,
) -> usize {
    let mut fst: Option<TypedFst<O>> = None;
    let mut n = 0;
    for (ln, line) in text.lines().enumerate() {
        let p: Vec<&str> = line.split(' ').collect();
        let ctx = || format!("{name}:{}: {}", ln + 1, &line[..line.len().min(120)]);
        match p[0] {
            "outputs" => {}
            "fst" => fst = Some(TypedFst::read(&unhex(p[1])).unwrap()),
            "get" => {
                let f = fst.as_ref().unwrap();
                let got = fst_util::get_bytes(f, &unhex(p[1])).unwrap();
                assert_eq!(got.map_or("null".into(), |v| fmt(&v)), p[2], "{}", ctx());
            }
            "top" => {
                let f = fst.as_ref().unwrap();
                let (arc, out) = walk(f, &unhex(p[1])).expect("prefix walk");
                let top_n: usize = p[2].parse().unwrap();
                let res =
                    fst_util::shortest_paths(f, &arc, out, cmp.clone(), top_n, p[3] == "true")
                        .unwrap();
                assert_eq!(fmt_results(&res, fmt), p[4..].join(" "), "{}", ctx());
            }
            "filtered" => {
                let f = fst.as_ref().unwrap();
                let (arc, out) = walk(f, &unhex(p[1])).expect("prefix walk");
                let mut s = TopNSearcher::new(f, 4, 12, cmp.clone());
                let mut calls = 0u32;
                s.accept_result = Some(Box::new(move |_: &[i32], _: &O::Value| {
                    let keep = calls & 1 == 0;
                    calls += 1;
                    keep
                }));
                s.add_start_paths(&arc, out, true, Vec::new()).unwrap();
                let res = s.search().unwrap();
                assert_eq!(fmt_results(&res, fmt), p[2..].join(" "), "{}", ctx());
            }
            "ceil" => {
                let f = fst.as_ref().unwrap();
                let (arc, _) = walk(f, &unhex(p[1])).expect("prefix walk");
                let label: i32 = p[2].parse().unwrap();
                let mut r = f.bytes_reader();
                let got = fst_util::read_ceil_arc(label, f, &arc, &mut r).unwrap();
                let got = match got {
                    None => "null".to_string(),
                    Some(c) => format!("{} {} {}", c.label(), fmt(c.output()), c.is_final()),
                };
                assert_eq!(got, p[3..].join(" "), "{}", ctx());
            }
            "dot1" | "dot2" => {
                let f = fst.as_ref().unwrap();
                let want = String::from_utf8(unhex(p[1])).unwrap();
                let same_rank = p[0] == "dot1";
                let got = fst_util::to_dot(f, same_rank, same_rank, fmt).unwrap();
                assert_eq!(got, want, "{}", ctx());
            }
            other => panic!("{}: unknown op {other}", ctx()),
        }
        n += 1;
    }
    n
}

#[test]
fn fst_util_matches_lucene() {
    let mut names: Vec<_> = std::fs::read_dir(dir())
        .expect("run scripts/gen-fixtures.sh --only GenFstUtil")
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .collect();
    names.sort();
    let mut total = 0;
    for name in &names {
        let text = std::fs::read_to_string(dir().join(name)).unwrap();
        if text.starts_with("outputs pair") {
            type P = PairOutputs<PositiveIntOutputs, ByteSequenceOutputs>;
            total += replay::<P>(
                name,
                &text,
                |a: &Pair<i64, Vec<u8>>, b: &Pair<i64, Vec<u8>>| a.first.cmp(&b.first),
                &|v: &Pair<i64, Vec<u8>>| format!("{}|{}", v.first, hex(&v.second)),
            );
        } else {
            total += replay::<PositiveIntOutputs>(
                name,
                &text,
                |a: &i64, b: &i64| a.cmp(b),
                &|v: &i64| v.to_string(),
            );
        }
    }
    assert!(names.len() >= 6, "{names:?}");
    assert!(total >= 1000, "only {total} checks");
}

/// The arc readers agree with each other over FSTs this port compiles with
/// every node encoding (dense label runs -- direct addressing or continuous
/// -- sparse fixed-length nodes searched by binary search, and
/// variable-length lists when fixed-length arcs are off) and every input
/// width: from every node, `readLastTargetArc` is the arc `readNextArc`
/// stops on, a binary-search node's `readArcByIndex(i)` is its `i`-th arc,
/// and `Util.get` finds every input's output.
#[test]
fn arc_readers_agree_over_every_node_encoding() {
    use lucene_codecs::fst::InputType;
    use lucene_codecs::fst_compiler::FstCompilerBuilder;
    // Dense runs (every label 0..40 under a few prefixes), sparse labels
    // (multiples of 19) and a few long chains.
    let mut inputs: Vec<Vec<i32>> = Vec::new();
    for p in 0..4 {
        for l in 0..40 {
            inputs.push(vec![p, l]);
        }
        for l in 0..12 {
            inputs.push(vec![p + 10, l * 19, l]);
        }
    }
    inputs.push(vec![50, 1, 2, 3, 4, 5]);
    inputs.push(vec![50, 1, 2, 9]);
    // Inputs that are prefixes of others: final arcs with arcs after them,
    // densely (direct addressing or continuous) and sparsely (binary
    // search) labelled.
    inputs.push(vec![50, 1, 2]);
    inputs.extend((10..40).map(|k| vec![50, 1, 2, k]));
    inputs.push(vec![51]);
    inputs.extend((0..12).map(|k| vec![51, k * 7]));
    inputs.sort();
    inputs.dedup();
    let mut shapes = std::collections::BTreeSet::new();
    let mut finals_with_arcs = 0;
    for input_type in [InputType::Byte1, InputType::Byte2, InputType::Byte4] {
        for fixed in [true, false] {
            let mut c = FstCompilerBuilder::new(input_type)
                .allow_fixed_length_arcs(fixed)
                .build::<PositiveIntOutputs>();
            for (i, input) in inputs.iter().enumerate() {
                c.add(input, i as i64 + 1).unwrap();
            }
            let compiled = c.compile().unwrap();
            // The saved form reads back to the same FST.
            let saved =
                TypedFst::<PositiveIntOutputs>::read(&compiled.save::<PositiveIntOutputs>())
                    .unwrap();
            let fst = TypedFst::<PositiveIntOutputs>::from_compiled(compiled);
            assert_eq!(fst.input_type(), input_type);
            assert_eq!(saved.input_type(), input_type);
            for (i, input) in inputs.iter().enumerate() {
                assert_eq!(fst_util::get(&saved, input).unwrap(), Some(i as i64 + 1));
            }
            assert!(fst.empty_output().is_none());
            assert!(format!("{fst:?}").contains("TypedFst"));
            for (i, input) in inputs.iter().enumerate() {
                assert_eq!(fst_util::get(&fst, input).unwrap(), Some(i as i64 + 1));
            }
            let mut r = fst.bytes_reader();
            let mut stack = vec![fst.first_arc()];
            while let Some(follow) = stack.pop() {
                if !fst_util::target_has_arcs(&follow) {
                    continue;
                }
                let last = fst.read_last_target_arc(&follow, &mut r).unwrap();
                let mut arc = follow.clone();
                fst.read_first_target_arc(&follow, &mut arc, &mut r)
                    .unwrap();
                let node_flags = arc.node_flags();
                shapes.insert((node_flags, fixed));
                let mut arcs = vec![arc.clone()];
                while !arc.is_last() {
                    fst.read_next_arc(&mut arc, &mut r).unwrap();
                    arcs.push(arc.clone());
                }
                // `readCeilArc`: the first arc at or past each label.
                let labels: Vec<i32> = arcs.iter().map(|a| a.label()).filter(|&l| l >= 0).collect();
                for l in 0..=labels.last().copied().unwrap_or(0) + 1 {
                    let want = labels.iter().copied().find(|&x| x >= l);
                    let got = fst_util::read_ceil_arc(l, &fst, &follow, &mut r)
                        .unwrap()
                        .map(|a| a.label());
                    assert_eq!(got, want, "ceil of {l} in {labels:?}");
                }
                // The label of the arc after each one, read without
                // moving to it.
                for w in arcs.windows(2) {
                    assert_eq!(
                        fst.read_next_arc_label(&w[0], &mut r).unwrap(),
                        w[1].label()
                    );
                }
                assert_eq!(
                    fst.is_expanded_target(&follow, &mut r).unwrap(),
                    arcs.iter()
                        .find(|a| a.label() >= 0)
                        .unwrap()
                        .bytes_per_arc()
                        > 0,
                    "{node_flags}"
                );
                for a in &arcs {
                    // A final arc ends an input: its end arc carries the
                    // final output; past a final node there is only that.
                    let end = TypedFst::<PositiveIntOutputs>::read_end_arc(a);
                    assert_eq!(end.is_some(), a.is_final());
                    // Looking for the end label finds that end arc.
                    let found = fst.find_target_arc(-1, a, &mut r).unwrap();
                    assert_eq!(found.map(|f| f.label()), end.as_ref().map(|e| e.label()));
                    let ceil = fst_util::read_ceil_arc(-1, &fst, a, &mut r).unwrap();
                    assert_eq!(ceil.is_some(), a.is_final());
                    if let (Some(mut end), true) = (end, fst_util::target_has_arcs(a)) {
                        // Ends here, or goes on: the end arc's next arc is
                        // the target node's first real one (its first
                        // target arc is the end arc itself, as in Lucene).
                        let mut first = a.clone();
                        fst.read_first_target_arc(a, &mut first, &mut r).unwrap();
                        assert_eq!(first.label(), -1);
                        fst.read_first_real_target_arc(a.target(), &mut first, &mut r)
                            .unwrap();
                        assert_eq!(
                            fst.read_next_arc_label(&end, &mut r).unwrap(),
                            first.label()
                        );
                        fst.read_next_arc(&mut end, &mut r).unwrap();
                        assert_eq!(end.label(), first.label());
                        finals_with_arcs += 1;
                    }
                    if !fst_util::target_has_arcs(a) {
                        let last = fst.read_last_target_arc(a, &mut r).unwrap();
                        assert_eq!(last.label(), -1);
                        assert!(!fst.is_expanded_target(a, &mut r).unwrap());
                    }
                }
                let end = arcs.last().unwrap();
                assert_eq!(
                    (last.label(), last.output(), last.target(), last.is_final()),
                    (end.label(), end.output(), end.target(), end.is_final()),
                    "last arc of the node {follow:?}"
                );
                // Binary-search nodes: every arc by its index.
                if node_flags == 32 {
                    assert_eq!(arcs.len() as i32, end.num_arcs());
                    assert!(end.bytes_per_arc() > 0);
                    for (idx, want) in arcs.iter().enumerate() {
                        let mut by_index = arc.clone();
                        fst.read_arc_by_index(&mut by_index, &mut r, idx as i32)
                            .unwrap();
                        assert_eq!(by_index.label(), want.label());
                        assert_eq!(by_index.arc_idx(), idx as i32);
                    }
                }
                let _ = (end.flags(), end.next_final_output());
                stack.extend(arcs.into_iter().filter(|a| a.label() >= 0));
            }
        }
    }
    // Every encoding showed up: plain lists, binary search, direct
    // addressing or continuous.
    let flags: std::collections::BTreeSet<u8> = shapes.iter().map(|&(f, _)| f).collect();
    assert!(flags.contains(&32), "{shapes:?}");
    assert!(flags.iter().any(|&f| f == 64 || f == 96), "{shapes:?}");
    assert!(shapes.iter().any(|&(_, fixed)| !fixed));
    assert!(finals_with_arcs > 0);
}

/// `Util.shortestPaths` when every path ties on output: the queue keeps the
/// lexicographically smallest inputs, comparing a tied candidate's input
/// against the queue's bottom before it enters.
#[test]
fn shortest_paths_break_ties_by_input() {
    use lucene_codecs::fst::InputType;
    use lucene_codecs::fst_compiler::FstCompilerBuilder;
    let mut c = FstCompilerBuilder::new(InputType::Byte1).build::<PositiveIntOutputs>();
    let inputs: Vec<Vec<i32>> = (0..20)
        .flat_map(|a| (0..3).map(move |b| vec![a, b]))
        .collect();
    for input in &inputs {
        c.add(input, 5).unwrap();
    }
    let fst = TypedFst::<PositiveIntOutputs>::from_compiled(c.compile().unwrap());
    for top_n in [1usize, 3, 7] {
        let res = fst_util::shortest_paths(
            &fst,
            &fst.first_arc(),
            0,
            |a: &i64, b: &i64| a.cmp(b),
            top_n,
            false,
        )
        .unwrap();
        assert!(res.is_complete);
        let got: Vec<Vec<i32>> = res.top_n.iter().map(|r| r.input.clone()).collect();
        assert_eq!(got, inputs[..top_n].to_vec(), "top {top_n}");
        assert!(res.top_n.iter().all(|r| r.output == 5));
    }
}
