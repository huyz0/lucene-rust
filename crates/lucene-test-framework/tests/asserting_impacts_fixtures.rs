//! Actual Java asserting-impacts outcomes, callback order and failure state.
//! Java's borrowed impacts-view invalidation guard has no counterpart in the
//! Rust trait's owned snapshots. Java CheckIndexException and AssertionError
//! remain distinct in the fixture; the infallible Rust impacts() reports both
//! through panic. Delegate I/O probes cover only the Rust Result methods.
use std::cell::RefCell;
use std::collections::BTreeSet;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::rc::Rc;

use lucene_codecs::postings::Impact;
use lucene_search::reader::{DocIdSetIterator, ImpactsEnum, PostingsEnum, NO_MORE_DOCS};
use lucene_search::Result;
use lucene_test_framework::index::asserting_leaf_reader::AssertingImpactsEnum;

const DATA: &str = include_str!("../../../fixtures/testframework/data/asserting_impacts.tsv");

fn unhex(text: &str) -> Vec<u8> {
    assert_eq!(text.len() % 2, 0);
    text.as_bytes()
        .chunks_exact(2)
        .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
        .collect()
}
fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}
fn text(encoded: &str) -> String {
    String::from_utf8(unhex(encoded)).unwrap()
}
fn ints(text: &str) -> Vec<i32> {
    text.split(',')
        .filter(|value| !value.is_empty())
        .map(|value| value.parse().unwrap())
        .collect()
}
type Levels = Vec<(i32, Vec<Impact>)>;
struct Config {
    freq: i32,
    docs: Vec<i32>,
    positions: Vec<i32>,
    starts: Vec<i32>,
    ends: Vec<i32>,
    payload: Option<Vec<u8>>,
    levels: Levels,
}
struct State {
    doc: i32,
    index: usize,
    position: usize,
    faults: BTreeSet<String>,
    events: Vec<String>,
}
struct Probe {
    config: Rc<Config>,
    state: Rc<RefCell<State>>,
}
impl Probe {
    fn operation(&self, name: &str, event: String) -> Result<()> {
        let mut state = self.state.borrow_mut();
        state.events.push(event);
        if state.faults.remove(name) {
            return Err(lucene_store::Error::Io(std::io::Error::other(format!(
                "fixture {name} failure"
            )))
            .into());
        }
        Ok(())
    }
    fn event(&self, name: &str) {
        self.state.borrow_mut().events.push(name.into());
    }
}
impl DocIdSetIterator for Probe {
    fn doc_id(&self) -> i32 {
        self.state.borrow().doc
    }
    fn next_doc(&mut self) -> Result<i32> {
        self.operation("next", "next".into())?;
        let mut state = self.state.borrow_mut();
        state.position = 0;
        state.doc = self
            .config
            .docs
            .get(state.index)
            .copied()
            .unwrap_or(NO_MORE_DOCS);
        state.index += 1;
        Ok(state.doc)
    }
    fn advance(&mut self, target: i32) -> Result<i32> {
        self.operation("advance", format!("advance:{target}"))?;
        let mut state = self.state.borrow_mut();
        state.position = 0;
        while self
            .config
            .docs
            .get(state.index)
            .is_some_and(|doc| *doc < target)
        {
            state.index += 1;
        }
        state.doc = self
            .config
            .docs
            .get(state.index)
            .copied()
            .unwrap_or(NO_MORE_DOCS);
        state.index += 1;
        Ok(state.doc)
    }
    fn cost(&self) -> i64 {
        self.event("cost");
        self.config.docs.len() as i64
    }
}
impl PostingsEnum for Probe {
    fn freq(&self) -> i32 {
        self.event("freq");
        self.config.freq
    }
    fn next_position(&mut self) -> Result<i32> {
        self.operation("position", "position".into())?;
        let mut state = self.state.borrow_mut();
        let position = self
            .config
            .positions
            .get(state.position)
            .copied()
            .unwrap_or(-1);
        state.position += 1;
        Ok(position)
    }
    fn start_offset(&self) -> i32 {
        self.event("start");
        let position = self
            .state
            .borrow()
            .position
            .checked_sub(1)
            .expect("positioned offset");
        self.config.starts.get(position).copied().unwrap_or(-1)
    }
    fn end_offset(&self) -> i32 {
        self.event("end");
        let position = self
            .state
            .borrow()
            .position
            .checked_sub(1)
            .expect("positioned offset");
        self.config.ends.get(position).copied().unwrap_or(-1)
    }
    fn payload(&self) -> Option<&[u8]> {
        self.event("payload");
        self.config.payload.as_deref()
    }
}
impl ImpactsEnum for Probe {
    fn advance_shallow(&mut self, target: i32) -> Result<i32> {
        self.operation("shallow", format!("shallow:{target}"))?;
        Ok(self
            .config
            .levels
            .first()
            .map_or(NO_MORE_DOCS, |level| level.0))
    }
    fn impacts(&self) -> Levels {
        self.event("impacts");
        self.config.levels.clone()
    }
}
fn snapshot(levels: &Levels) -> String {
    levels
        .iter()
        .map(|(upto, impacts)| {
            let pairs = impacts
                .iter()
                .map(|impact| format!("{}/{}", impact.freq, impact.norm as u64))
                .collect::<Vec<_>>()
                .join(",");
            format!("{upto}:{pairs}")
        })
        .collect::<Vec<_>>()
        .join(";")
}
fn action(
    wrapper: &mut AssertingImpactsEnum,
    state: &RefCell<State>,
    op: &str,
    argument: i32,
) -> Result<String> {
    Ok(match op {
        "D" => wrapper.doc_id().to_string(),
        "C" => wrapper.cost().to_string(),
        "N" => wrapper.next_doc()?.to_string(),
        "A" => wrapper.advance(argument)?.to_string(),
        // Java advanceShallow returns void; its Rust block-bound return is an
        // API adaptation rather than part of the shared observable contract.
        "S" => {
            wrapper.advance_shallow(argument)?;
            String::new()
        }
        "I" => snapshot(&wrapper.impacts()),
        "F" => wrapper.freq().to_string(),
        "P" => wrapper.next_position()?.to_string(),
        "O" => wrapper.start_offset().to_string(),
        "E" => wrapper.end_offset().to_string(),
        "Y" => wrapper.payload().map_or_else(|| "null".into(), hex),
        // Test setup, matching Java's delayed delegate-fault arming operation.
        "X" => {
            let fault = match argument {
                1 => "next",
                2 => "advance",
                _ => panic!("unknown fixture fault {argument}"),
            };
            state.borrow_mut().faults.insert(fault.into());
            String::new()
        }
        _ => panic!("unknown Java action {op}"),
    })
}
fn panic_message(error: Box<dyn std::any::Any + Send>) -> String {
    error
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| {
            error
                .downcast_ref::<&str>()
                .map(|message| (*message).into())
        })
        .expect("asserting wrapper must panic with a diagnostic")
}
fn run_case(lines: &[&str]) -> (String, usize) {
    let case: Vec<_> = lines[0].split('\t').collect();
    assert_eq!(case.len(), 10);
    let name = case[1];
    let mut config = Config {
        freq: case[3].parse().unwrap(),
        docs: ints(case[4]),
        positions: ints(case[5]),
        starts: ints(case[6]),
        ends: ints(case[7]),
        payload: (case[8] != "null").then(|| unhex(case[8])),
        levels: Vec::new(),
    };
    for line in lines.iter().filter(|line| line.starts_with("level\t")) {
        let level: Vec<_> = line.split('\t').collect();
        assert_eq!(level.len(), 4);
        assert_eq!(level[1], name);
        let impacts = level[3]
            .split(',')
            .filter(|pair| !pair.is_empty())
            .map(|pair| {
                let (freq, norm) = pair.split_once('/').unwrap();
                Impact {
                    freq: freq.parse().unwrap(),
                    norm: norm.parse::<u64>().unwrap() as i64,
                }
            })
            .collect();
        config.levels.push((level[2].parse().unwrap(), impacts));
    }
    let state = Rc::new(RefCell::new(State {
        doc: case[2].parse().unwrap(),
        index: 0,
        position: 0,
        faults: case[9]
            .split(',')
            .filter(|value| !value.is_empty())
            .map(str::to_owned)
            .collect(),
        events: Vec::new(),
    }));
    let mut wrapper = AssertingImpactsEnum::new(Box::new(Probe {
        config: Rc::new(config),
        state: state.clone(),
    }));
    let mut steps = 0;
    for line in lines.iter().filter(|line| line.starts_with("step\t")) {
        let step: Vec<_> = line.split('\t').collect();
        assert_eq!(step.len(), 11);
        assert_eq!(step[1], name);
        let op = step[2];
        state.borrow_mut().events.clear();
        let actual = catch_unwind(AssertUnwindSafe(|| {
            action(&mut wrapper, &state, op, step[3].parse().unwrap())
        }));
        let java_message = text(step[7]);
        match (step[4], actual) {
            ("ok", Ok(Ok(value))) => assert_eq!(value, text(step[6]), "{name}: {op}"),
            ("io", Ok(Err(error))) => assert!(
                error.to_string().contains(&java_message),
                "{name}: {op}: {error}"
            ),
            ("assert" | "check", Err(error)) => {
                let message = panic_message(error);
                if step[4] == "check" {
                    assert_eq!(step[5], "CheckIndexException");
                    assert_eq!(message, java_message, "{name}: {op}: CheckIndex diagnostic");
                } else if java_message.starts_with(" invalid docID()") {
                    let marker = java_message.replace(" in class GenAssertingImpacts$Probe", "");
                    assert_eq!(message, marker, "{name}: {op}: cached docID diagnostic");
                } else if java_message != "null" {
                    let marker = java_message
                        .strip_suffix(" ProbeImpacts")
                        .unwrap_or(&java_message);
                    assert!(
                        message.contains(marker),
                        "{name}: {op}: {message:?}, Java {java_message:?}"
                    );
                }
            }
            (expected, actual) => panic!("{name}: {op}: expected {expected}, got {actual:?}"),
        }
        let observed = state.borrow();
        assert_eq!(
            observed.events.join(","),
            step[8],
            "{name}: {op}: delegate calls"
        );
        assert_eq!(
            observed.doc,
            step[9].parse::<i32>().unwrap(),
            "{name}: {op}: delegate doc"
        );
        assert_eq!(
            observed.position,
            step[10].parse::<usize>().unwrap(),
            "{name}: {op}: delegate position"
        );
        steps += 1;
    }
    assert!(steps > 0);
    (name.into(), steps)
}

#[test]
fn impacts_matches_actual_java_checks_and_failure_transitions() {
    let mut group = Vec::new();
    let mut names = BTreeSet::new();
    let mut steps = 0;
    for line in DATA.lines() {
        if line.starts_with("case\t") && !group.is_empty() {
            let (name, count) = run_case(&group);
            assert!(names.insert(name));
            steps += count;
            group.clear();
        }
        group.push(line);
    }
    assert!(!group.is_empty());
    let (name, count) = run_case(&group);
    assert!(names.insert(name));
    steps += count;
    assert_eq!(names.len(), 56);
    assert_eq!(steps, 202);
    for required in [
        "shallow_error_retains_target",
        "equal_level_bounds_allowed",
        "next_error_retains_consumed_position",
        "advance_error_retains_consumed_position",
        "later_pairs_compare_only_first",
        "dominance_cursor_last_unchecked",
        "dominance_cursor_exhausted",
        "dominance_cursor_does_not_retry",
        "unsigned_norm_extremes",
        "dominance_unsigned_invalid",
    ] {
        assert!(
            names.contains(required),
            "missing semantic boundary {required}"
        );
    }
}
