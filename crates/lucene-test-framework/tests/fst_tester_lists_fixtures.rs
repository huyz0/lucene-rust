//! Actual Java FSTTester builds and verifies lists by repeated scalar additions.
use std::panic::{catch_unwind, AssertUnwindSafe};

use lucene_codecs::fst::PositiveIntOutputs;
use lucene_codecs::fst_util::IntsRefFstEnum;
use lucene_search::misc::{ListOfOutputs, ListOutput};
use lucene_test_framework::util::fst_tester::{FstTester, InputOutput};
use lucene_util::java_random::JavaRandom;

type Outputs = ListOfOutputs<PositiveIntOutputs>;
type Value = ListOutput<i64>;

fn input(text: &str) -> Vec<i32> {
    if text == "-" {
        Vec::new()
    } else {
        text.split(',').map(|v| v.parse().unwrap()).collect()
    }
}
fn output(text: &str) -> Value {
    let (kind, text) = text.split_once(':').unwrap();
    if kind == "S" {
        Value::One(text.parse().unwrap())
    } else {
        assert_eq!(kind, "M");
        Value::Many(if text.is_empty() {
            Vec::new()
        } else {
            text.split(',').map(|v| v.parse().unwrap()).collect()
        })
    }
}
fn unhex(text: &str) -> Vec<u8> {
    (0..text.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&text[i..i + 2], 16).unwrap())
        .collect()
}

#[test]
fn actual_java_tester_lists_match_bytes_enumeration_counts_failures_and_rng_tails() {
    let text = include_str!("../../../fixtures/testframework/data/fst_tester_lists.tsv");
    let mut cases = 0;
    for block in text.split("end\n").filter(|block| !block.is_empty()) {
        let mut rows = block.lines();
        let header: Vec<_> = rows.next().unwrap().split('\t').collect();
        assert_eq!(header[0], "case");
        let mode = header[2].parse().unwrap();
        let seed = header[3].parse().unwrap();
        let mut pairs = Vec::new();
        let mut expected = Vec::new();
        let mut saved = None;
        let mut result = None;
        for row in rows {
            let fields: Vec<_> = row.split('\t').collect();
            match fields[0] {
                "pair" => pairs.push(InputOutput {
                    input: input(fields[1]),
                    output: output(fields[2]),
                }),
                "enum" => expected.push((input(fields[1]), output(fields[2]))),
                "fst" => saved = Some(unhex(fields[1])),
                "result" => {
                    assert!(result.is_none());
                    result = Some((
                        fields[1],
                        fields[2].parse::<i64>().unwrap(),
                        fields[3].parse::<i64>().unwrap(),
                        fields[4].parse::<i32>().unwrap(),
                    ));
                }
                kind => panic!("unknown row {kind}"),
            }
        }
        let mut random = JavaRandom::new(seed);
        let mut tester = FstTester::<Outputs>::new(&mut random, 1, mode, pairs);
        let (status, fst) = match catch_unwind(AssertUnwindSafe(|| tester.do_test_lists())) {
            Ok(Ok(Some(fst))) => ("ok", Some(fst)),
            Ok(Ok(None)) => ("none", None),
            Ok(Err(error)) => panic!("unexpected decoder error for {header:?}: {error}"),
            Err(_) => ("assert", None),
        };
        let stats = (tester.node_count, tester.arc_count);
        drop(tester);
        let actual = (status, stats.0, stats.1, random.next_int());
        assert_eq!(actual, result.unwrap(), "{header:?}");
        if let Some(fst) = fst {
            assert_eq!(fst.save(), saved.unwrap(), "{header:?}: saved bytes");
            let mut enumeration = IntsRefFstEnum::new(&fst);
            let mut actual = Vec::new();
            while let Some((term, value)) = enumeration.next().unwrap() {
                actual.push((term.to_vec(), value.clone()));
            }
            assert_eq!(actual, expected, "{header:?}: complete outputs");
        } else {
            assert!(saved.is_none());
            assert!(expected.is_empty());
        }
        cases += 1;
    }
    assert_eq!(cases, 108);
}
