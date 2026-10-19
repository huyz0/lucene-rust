//! M14 T14.3 `misc`, differentially against Lucene 10.5.0:
//! `fixtures/src/GenMiscFst.java`'s `misc/fst/*.txt`.
//!
//! Each case adds keys (some more than once in a row, so the outputs'
//! `merge` runs) to an FST compiled with `UpToTwoPositiveIntOutputs` or
//! `ListOfOutputs`. The Rust compiler, given the same adds, must save the same
//! bytes with the same node and arc counts, and reading every key back must
//! give what `Util.get` gave.
#![allow(clippy::arithmetic_side_effects)]

use lucene_codecs::fst::{ByteSequenceOutputs, InputType, PositiveIntOutputs};
use lucene_codecs::fst_compiler::{FstCompilerBuilder, FstOutputs};
use lucene_codecs::fst_util::{get_bytes, TypedFst};
use lucene_search::misc::{ListOfOutputs, ListOutput, UpToTwoOutput, UpToTwoPositiveIntOutputs};

fn read_case(name: &str) -> String {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/data/misc/fst")
        .join(format!("{name}.txt"));
    std::fs::read_to_string(path).unwrap()
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

/// Replays case `name` with outputs `O`, `parse` reading an `L:`/`T:`/`M:`
/// payload into a value and `show` writing one back.
fn check<O: FstOutputs>(
    name: &str,
    parse: fn(&str) -> O::Value,
    show: fn(&O::Value) -> String,
) -> usize {
    let text = read_case(name);
    let mut c = FstCompilerBuilder::new(InputType::Byte1).build::<O>();
    let (mut want_stats, mut want_fst, mut gets) = (String::new(), String::new(), Vec::new());
    let mut adds = 0;
    for line in text.lines() {
        let (kind, rest) = line.split_once(' ').unwrap();
        match kind {
            "k" => {
                let (k, v) = rest.split_once(' ').unwrap();
                let key: Vec<i32> = unhex(k).into_iter().map(i32::from).collect();
                c.add(&key, parse(v)).unwrap();
                adds += 1;
            }
            "stats" => want_stats = rest.to_string(),
            "fst" => want_fst = rest.to_string(),
            "get" => {
                let (k, v) = rest.split_once(' ').unwrap();
                gets.push((unhex(k), v.to_string()));
            }
            other => panic!("{name}: unknown line {other}"),
        }
    }
    let fst = c.compile().unwrap();
    let stats = c.stats();
    assert_eq!(
        format!("{} {}", stats.node_count, stats.arc_count),
        want_stats,
        "{name}: node/arc counts"
    );
    let saved = fst.save::<O>();
    assert_eq!(hex(&saved), want_fst, "{name}: saved bytes");
    let typed = TypedFst::<O>::read(&saved).unwrap();
    assert!(!gets.is_empty());
    for (key, want) in &gets {
        let got = get_bytes(&typed, key).unwrap().expect("key is in the FST");
        assert_eq!(&show(&got), want, "{name}: get {}", hex(key));
    }
    adds
}

fn up_to_two(v: &str) -> UpToTwoOutput {
    UpToTwoOutput::of(v.strip_prefix("L:").unwrap().parse().unwrap())
}

fn show_up_to_two(v: &UpToTwoOutput) -> String {
    match v {
        UpToTwoOutput::One(v) => format!("L:{v}"),
        UpToTwoOutput::Two(a, b) => format!("T:{a},{b}"),
    }
}

#[test]
fn up_to_two_positive_int_outputs_match_lucene() {
    let shared =
        check::<UpToTwoPositiveIntOutputs<true>>("up_to_two_share", up_to_two, show_up_to_two);
    let unshared =
        check::<UpToTwoPositiveIntOutputs<false>>("up_to_two_noshare", up_to_two, show_up_to_two);
    assert!(shared > 120 && unshared > 120, "keys were added twice");
}

fn long_list(v: &str) -> ListOutput<i64> {
    ListOutput::One(v.strip_prefix("L:").unwrap().parse().unwrap())
}

fn show_long_list(v: &ListOutput<i64>) -> String {
    match v {
        ListOutput::One(v) => format!("L:{v}"),
        ListOutput::Many(list) => {
            let parts: Vec<String> = list.iter().map(i64::to_string).collect();
            format!("M:{}", parts.join(";"))
        }
    }
}

#[test]
fn list_of_longs_match_lucene() {
    let adds =
        check::<ListOfOutputs<PositiveIntOutputs>>("list_of_longs", long_list, show_long_list);
    assert!(adds > 120, "keys were added more than once");
}

fn byte_list(v: &str) -> ListOutput<Vec<u8>> {
    ListOutput::One(unhex(v.strip_prefix("L:").unwrap()))
}

fn show_byte_list(v: &ListOutput<Vec<u8>>) -> String {
    match v {
        ListOutput::One(v) => format!("L:{}", hex(v)),
        ListOutput::Many(list) => {
            let parts: Vec<String> = list.iter().map(|b| hex(b)).collect();
            format!("M:{}", parts.join(";"))
        }
    }
}

#[test]
fn list_of_bytes_match_lucene() {
    let adds =
        check::<ListOfOutputs<ByteSequenceOutputs>>("list_of_bytes", byte_list, show_byte_list);
    assert!(adds > 120, "keys were added more than once");
}
