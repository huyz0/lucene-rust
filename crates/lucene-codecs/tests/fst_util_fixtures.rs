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
