//! Actual asserting impacts traversal over the same synthetic input/trace as Java.
use std::hint::black_box;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use lucene_search::reader::{DocIdSetIterator, Impact, ImpactsEnum, PostingsEnum, NO_MORE_DOCS};
use lucene_search::Result;
use lucene_test_framework::index::asserting_leaf_reader::AssertingImpactsEnum;

struct Input {
    docs: Vec<i32>,
    freqs: Vec<i32>,
    levels: Vec<(i32, Vec<Impact>)>,
    trace: Vec<(u8, i32)>,
    expected: u64,
}
impl Input {
    fn read(path: &Path) -> Self {
        let mut input = Self {
            docs: Vec::new(),
            freqs: Vec::new(),
            levels: Vec::new(),
            trace: Vec::new(),
            expected: 0,
        };
        for line in std::fs::read_to_string(path).unwrap().lines() {
            let row: Vec<_> = line.split('\t').collect();
            match row[0] {
                "D" => {
                    input.docs.push(row[1].parse().unwrap());
                    input.freqs.push(row[2].parse().unwrap());
                }
                "L" => {
                    let level: usize = row[1].parse().unwrap();
                    if level == input.levels.len() {
                        input.levels.push((row[2].parse().unwrap(), Vec::new()));
                    }
                    input.levels[level].1.push(Impact {
                        freq: row[3].parse().unwrap(),
                        norm: row[4].parse::<u64>().unwrap() as i64,
                    });
                }
                "T" => input
                    .trace
                    .push((row[1].parse().unwrap(), row[2].parse().unwrap())),
                "Z" => input.expected = row[1].parse().unwrap(),
                _ => panic!("unknown input row"),
            }
        }
        input
    }
}
struct Delegate {
    input: Arc<Input>,
    index: Option<usize>,
    position: i32,
    payload: [u8; 2],
}
impl DocIdSetIterator for Delegate {
    fn doc_id(&self) -> i32 {
        self.index.map_or(-1, |index| {
            self.input.docs.get(index).copied().unwrap_or(NO_MORE_DOCS)
        })
    }
    fn next_doc(&mut self) -> Result<i32> {
        self.index = Some(self.index.map_or(0, |index| index + 1));
        self.position = 0;
        Ok(self.doc_id())
    }
    fn advance(&mut self, target: i32) -> Result<i32> {
        let mut index = self.index.map_or(0, |index| index + 1);
        while index < self.input.docs.len() && self.input.docs[index] < target {
            index += 1;
        }
        self.index = Some(index);
        self.position = 0;
        Ok(self.doc_id())
    }
    fn cost(&self) -> i64 {
        self.input.docs.len() as i64
    }
}
impl PostingsEnum for Delegate {
    fn freq(&self) -> i32 {
        self.input.freqs[self.index.unwrap()]
    }
    fn next_position(&mut self) -> Result<i32> {
        let position = 1 + 3 * self.position;
        self.position += 1;
        self.payload = [self.doc_id() as u8, self.position as u8];
        Ok(position)
    }
    fn start_offset(&self) -> i32 {
        (1 + 3 * (self.position - 1)) * 5
    }
    fn end_offset(&self) -> i32 {
        self.start_offset() + 2
    }
    fn payload(&self) -> Option<&[u8]> {
        (self.position & 1 != 0).then_some(self.payload.as_slice())
    }
}
impl ImpactsEnum for Delegate {
    fn advance_shallow(&mut self, _target: i32) -> Result<i32> {
        Ok(self.input.levels[0].0)
    }
    fn impacts(&self) -> Vec<(i32, Vec<Impact>)> {
        self.input.levels.clone()
    }
}
fn fold(hash: u64, value: i64) -> u64 {
    hash.wrapping_mul(31).wrapping_add(value as u64)
}
fn traverse(iterator: &mut dyn ImpactsEnum, input: &Input) -> u64 {
    let mut hash = 0;
    for &(kind, arg) in &input.trace {
        hash = fold(hash, i64::from(kind));
        match kind {
            0 => {
                iterator.advance_shallow(arg).unwrap();
            }
            1 => hash = fold(hash, i64::from(iterator.next_doc().unwrap())),
            2 => hash = fold(hash, i64::from(iterator.advance(arg).unwrap())),
            3 => {
                let levels = iterator.impacts();
                hash = fold(hash, levels.len() as i64);
                for (bound, pairs) in levels {
                    hash = fold(hash, i64::from(bound));
                    hash = fold(hash, pairs.len() as i64);
                    for pair in pairs {
                        hash = fold(hash, i64::from(pair.freq));
                        hash = fold(hash, pair.norm);
                    }
                }
            }
            4 => hash = fold(hash, i64::from(iterator.freq())),
            5 => hash = fold(hash, i64::from(iterator.next_position().unwrap())),
            6 => {
                hash = fold(hash, i64::from(iterator.start_offset()));
                hash = fold(hash, i64::from(iterator.end_offset()));
            }
            7 => {
                let payload = iterator.payload();
                hash = fold(hash, payload.map_or(-1, |bytes| bytes.len() as i64));
                if let Some(bytes) = payload {
                    for &byte in bytes {
                        hash = fold(hash, i64::from(byte));
                    }
                }
            }
            8 => hash = fold(hash, i64::from(iterator.doc_id())),
            9 => hash = fold(hash, iterator.cost()),
            _ => panic!("unknown operation"),
        }
    }
    hash
}
fn run(input: &Arc<Input>, millis: u64) -> (u128, u64) {
    let wall = Instant::now();
    let mut nanos = 0;
    let mut count = 0;
    loop {
        let mut iterator = AssertingImpactsEnum::new(Box::new(Delegate {
            input: Arc::clone(input),
            index: None,
            position: 0,
            payload: [0; 2],
        }));
        let start = Instant::now();
        let actual = traverse(&mut iterator, input);
        nanos += start.elapsed().as_nanos();
        assert_eq!(actual, input.expected, "trace digest differs");
        black_box(actual);
        count += 1;
        drop(iterator);
        if wall.elapsed() >= Duration::from_millis(millis) {
            break;
        }
    }
    (nanos, count)
}
// Run in a separate process; normal wrapper traversal/setup stays unchanged.
fn run_raw(input: &Arc<Input>, millis: u64) -> (u128, u64) {
    let wall = Instant::now();
    let mut nanos = 0;
    let mut count = 0;
    loop {
        let mut iterator = Delegate {
            input: Arc::clone(input),
            index: None,
            position: 0,
            payload: [0; 2],
        };
        let start = Instant::now();
        let actual = traverse(&mut iterator, input);
        nanos += start.elapsed().as_nanos();
        assert_eq!(actual, input.expected, "trace digest differs");
        black_box(actual);
        count += 1;
        drop(iterator);
        if wall.elapsed() >= Duration::from_millis(millis) {
            break;
        }
    }
    (nanos, count)
}
fn main() {
    let args: Vec<_> = std::env::args().collect();
    let root = Path::new(&args[1]);
    let warmup = args[2].parse().unwrap();
    let measure = args[3].parse().unwrap();
    assert!(measure > 0);
    if args.get(4).is_some_and(|mode| mode == "raw") {
        let input = Arc::new(Input::read(&root.join("single.tsv")));
        run_raw(&input, warmup);
        let (nanos, count) = run_raw(&input, measure);
        println!(
            "asserting_impacts_single_raw\t{:.3}\t{count}",
            nanos as f64 / count as f64
        );
        println!(
            "#check\tasserting_impacts_single_raw\t{}\t{}\t{}",
            input.expected,
            input.trace.len(),
            input.docs.len()
        );
        return;
    }
    println!(
        "asserting_impacts_wrapper_shallow_bytes\t{}",
        std::mem::size_of::<AssertingImpactsEnum>()
    );
    for name in ["single", "multilevel", "positions"] {
        let input = Arc::new(Input::read(&root.join(format!("{name}.tsv"))));
        run(&input, warmup);
        let (nanos, count) = run(&input, measure);
        println!(
            "asserting_impacts_{name}\t{:.3}\t{count}",
            nanos as f64 / count as f64
        );
        println!(
            "#check\tasserting_impacts_{name}\t{}\t{}\t{}",
            input.expected,
            input.trace.len(),
            input.docs.len()
        );
    }
}
