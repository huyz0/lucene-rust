//! Complete seeded FSTTester builds, random walks, seeks and list output digests.
use std::hint::black_box;
use std::time::{Duration, Instant};

use lucene_codecs::fst::PositiveIntOutputs;
use lucene_codecs::fst_util::IntsRefFstEnum;
use lucene_search::misc::{ListOfOutputs, ListOutput};
use lucene_test_framework::util::fst_tester::{FstTester, InputOutput};
use lucene_util::java_random::JavaRandom;

type Outputs = ListOfOutputs<PositiveIntOutputs>;
type Value = ListOutput<i64>;
struct Case {
    mode: i32,
    seed: i64,
    pairs: Vec<InputOutput<Value>>,
}
fn cases() -> Vec<Case> {
    let text = std::fs::read_to_string("fixtures/testframework/data/fst_tester_lists.tsv").unwrap();
    let mut cases = Vec::new();
    let mut current = None;
    for row in text.lines() {
        let fields: Vec<_> = row.split('\t').collect();
        match fields[0] {
            "case" => {
                current = (fields[1] == "mixed").then(|| Case {
                    mode: fields[2].parse().unwrap(),
                    seed: fields[3].parse().unwrap(),
                    pairs: Vec::new(),
                });
            }
            "pair" => {
                if let Some(case) = &mut current {
                    let input = if fields[1] == "-" {
                        Vec::new()
                    } else {
                        fields[1].split(',').map(|n| n.parse().unwrap()).collect()
                    };
                    let (kind, values) = fields[2].split_once(':').unwrap();
                    let output = if kind == "S" {
                        Value::One(values.parse().unwrap())
                    } else {
                        Value::Many(values.split(',').map(|n| n.parse().unwrap()).collect())
                    };
                    case.pairs.push(InputOutput { input, output });
                }
            }
            "end" => {
                if let Some(case) = current.take() {
                    cases.push(case);
                }
            }
            _ => {}
        }
    }
    assert_eq!(cases.len(), 12);
    cases
}
#[derive(Default)]
struct Digest(i64);
impl Digest {
    fn mix(&mut self, value: i64) {
        self.0 = self.0.wrapping_mul(31).wrapping_add(value);
    }
    fn bytes(&mut self, bytes: &[u8]) {
        self.mix(bytes.len() as i64);
        for &byte in bytes {
            self.mix(i64::from(byte));
        }
    }
    fn output(&mut self, value: &Value) {
        match value {
            Value::One(value) => {
                self.mix(1);
                self.mix(*value);
            }
            Value::Many(values) => {
                self.mix(2);
                self.mix(values.len() as i64);
                for &value in values {
                    self.mix(value);
                }
            }
        }
    }
}
fn batch(cases: &[Case], mode: i32) -> i64 {
    let mut digest = Digest::default();
    for case in cases.iter().filter(|case| case.mode == mode) {
        let mut random = JavaRandom::new(case.seed);
        // Both engines copy the pair, input and list containers during timing.
        let mut tester = FstTester::<Outputs>::new(&mut random, 1, mode, case.pairs.clone());
        let fst = tester.do_test_lists().unwrap().unwrap();
        digest.mix(tester.node_count);
        digest.mix(tester.arc_count);
        drop(tester);
        digest.mix(i64::from(random.next_int()));
        digest.bytes(&fst.save());
        let mut enumeration = IntsRefFstEnum::new(&fst);
        while let Some((input, output)) = enumeration.next().unwrap() {
            digest.mix(input.len() as i64);
            for &label in input {
                digest.mix(i64::from(label));
            }
            digest.output(output);
        }
    }
    black_box(digest.0)
}
fn run(cases: &[Case], mode: i32, ms: u64) -> (u64, Duration) {
    let expected = batch(cases, mode);
    let start = Instant::now();
    let mut count = 0;
    loop {
        assert_eq!(black_box(batch(cases, mode)), expected);
        count += 6;
        if start.elapsed() >= Duration::from_millis(ms) {
            return (count, start.elapsed());
        }
    }
}
fn input_graph_bytes(pairs: &Vec<InputOutput<Value>>) -> usize {
    let mut bytes = std::mem::size_of::<FstTester<'_, Outputs>>()
        + pairs.capacity() * std::mem::size_of::<InputOutput<Value>>();
    for pair in pairs {
        bytes += pair.input.capacity() * std::mem::size_of::<i32>();
        if let Value::Many(values) = &pair.output {
            bytes += values.capacity() * std::mem::size_of::<i64>();
        }
    }
    bytes
}
fn main() {
    let args: Vec<_> = std::env::args().collect();
    let warmup = args.get(1).map_or(1000, |s| s.parse().unwrap());
    let measure = args.get(2).map_or(1500, |s| s.parse().unwrap());
    let cases = cases();
    // Measure the clone layout actually handed to each timed tester.
    let sample = cases[0].pairs.clone();
    println!(
        "#memory\tfst_tester_lists_tester_input_graph\t{}",
        input_graph_bytes(&sample)
    );
    for mode in [0, 1] {
        let name = if mode == 0 { "utf8" } else { "utf32" };
        println!(
            "#check\tfst_tester_lists_{name}\t{}\t6",
            batch(&cases, mode)
        );
        run(&cases, mode, warmup);
        let (count, elapsed) = run(&cases, mode, measure);
        println!(
            "fst_tester_lists_{name}\t{:.3}\t{count}",
            elapsed.as_nanos() as f64 / count as f64
        );
    }
}
