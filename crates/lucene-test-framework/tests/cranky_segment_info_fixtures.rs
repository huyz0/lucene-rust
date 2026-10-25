//! Java's actual cranky format over real segment-info files and one shared RNG.
use std::cell::RefCell;
use std::rc::Rc;

use lucene_index::segment_info::{self, LuceneVersion, SegmentInfo};
use lucene_store::{ByteBuffersDirectory, Directory};
use lucene_test_framework::codecs::cranky_segment_info::{
    CrankySegmentInfoFormat, Lucene99SegmentInfoFormat, SegmentInfoFormatDelegate,
};
use lucene_util::java_random::JavaRandom;

fn info() -> SegmentInfo {
    SegmentInfo {
        id: *b"0123456789abcdef",
        version: LuceneVersion {
            major: 10,
            minor: 5,
            bugfix: 0,
        },
        min_version: None,
        doc_count: 7,
        is_compound_file: false,
        has_blocks: false,
        diagnostics: vec![],
        files: vec![],
        attributes: vec![],
        index_sort: None,
    }
}

#[derive(Default)]
struct State {
    reads: i32,
    writes: i32,
    fail_read: bool,
    fail_write: bool,
}
struct Recording {
    state: Rc<RefCell<State>>,
}
impl SegmentInfoFormatDelegate for Recording {
    fn read(
        &mut self,
        dir: &dyn Directory,
        name: &str,
        id: &[u8; 16],
    ) -> segment_info::Result<SegmentInfo> {
        let fail = {
            let mut state = self.state.borrow_mut();
            state.reads += 1;
            state.fail_read
        };
        if fail {
            return Err(
                lucene_store::Error::Io(std::io::Error::other("delegate read failure")).into(),
            );
        }
        Lucene99SegmentInfoFormat.read(dir, name, id)
    }
    fn write(
        &mut self,
        dir: &dyn Directory,
        name: &str,
        info: &mut SegmentInfo,
    ) -> segment_info::Result<()> {
        let fail = {
            let mut state = self.state.borrow_mut();
            state.writes += 1;
            state.fail_write
        };
        if fail {
            return Err(
                lucene_store::Error::Io(std::io::Error::other("delegate write failure")).into(),
            );
        }
        Lucene99SegmentInfoFormat.write(dir, name, info)
    }
}

#[test]
fn java_shared_random_and_real_segment_info_delegation() {
    let trace = include_str!("../../../fixtures/testframework/data/cranky_segment_info.tsv");
    let mut lines = trace.lines();
    let mut cases = 0;
    for seed in [0, 42, 777, i64::MIN] {
        for failures in [false, true] {
            let random = Rc::new(RefCell::new(JavaRandom::new(seed)));
            let state = Rc::new(RefCell::new(State::default()));
            let mut formats = [
                CrankySegmentInfoFormat::new(
                    Recording {
                        state: state.clone(),
                    },
                    random.clone(),
                ),
                CrankySegmentInfoFormat::new(
                    Recording {
                        state: state.clone(),
                    },
                    random.clone(),
                ),
            ];
            let dir = ByteBuffersDirectory::new();
            Lucene99SegmentInfoFormat
                .write(&dir, "_base", &mut info())
                .unwrap();
            let bytes = dir.open("_base.si").unwrap();
            assert_eq!(
                &*bytes,
                include_bytes!("../../../fixtures/testframework/data/cranky_segment_info.si")
            );
            for op in 0..256 {
                {
                    let mut state = state.borrow_mut();
                    state.fail_read = failures && op % 32 == 0;
                    state.fail_write = failures && op % 19 == 0;
                }
                let name = format!("_s{}", base36(op));
                let mut info = info();
                let read = op % 4 == 0;
                let (result, docs) = if read {
                    match formats[op % 2].read(&dir, "_base", &info.id) {
                        Ok(info) => ("ok".to_string(), info.doc_count.to_string()),
                        Err(err) => (err.to_string(), "-".to_string()),
                    }
                } else {
                    (
                        formats[op % 2]
                            .write(&dir, &name, &mut info)
                            .map_or_else(|e| e.to_string(), |_| "ok".to_string()),
                        "-".to_string(),
                    )
                };
                let state = state.borrow();
                let line = format!(
                    "op\t{seed}\t{failures}\t{op}\t{}\t{result}\t{}\t{}\t{}\t{docs}",
                    if read { "read" } else { "write" },
                    state.reads,
                    state.writes,
                    info.files.contains(&format!("{name}.si"))
                );
                assert_eq!(
                    Some(line.as_str()),
                    lines.next(),
                    "seed={seed} failures={failures} op={op}"
                );
                cases += 1;
            }
            let line = format!(
                "rng\t{seed}\t{failures}\t{}",
                random.borrow_mut().next_int()
            );
            assert_eq!(Some(line.as_str()), lines.next());
        }
    }
    assert_eq!(cases, 2048);
    assert!(lines.next().is_none());
}

fn base36(mut value: usize) -> String {
    if value == 0 {
        return "0".into();
    }
    let mut text = Vec::new();
    while value > 0 {
        text.push(b"0123456789abcdefghijklmnopqrstuvwxyz"[value % 36]);
        value /= 36;
    }
    text.reverse();
    String::from_utf8(text).unwrap()
}
