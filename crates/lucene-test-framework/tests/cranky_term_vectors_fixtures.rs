//! Actual Java wrapper traces: nested guards, input consumption, merge order and RNG tails.
use lucene_codecs::field_infos::{FieldInfo, FieldInfos};
use lucene_store::{ByteBuffersDirectory, DataInput, Directory, SliceInput};
use lucene_test_framework::codecs::cranky_term_vectors::*;
use lucene_util::java_random::JavaRandom;
use std::cell::RefCell;
use std::io;
use std::rc::Rc;
const POS: &[u8] = &[3, 2, 0, 10, 4, 1, 0];
const OFF: &[u8] = &[0, 2, 1, 3, 0, 0];
fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
struct Probe {
    log: Rc<RefCell<Vec<String>>>,
    fail: String,
    positions: usize,
}
impl Probe {
    fn call(&mut self, name: &str, detail: String) -> io::Result<()> {
        self.log.borrow_mut().push(format!("{name}{detail}"));
        if self.fail == name
            || (name == "position" && self.fail == format!("position#{}", self.positions))
        {
            Err(io::Error::other(format!("delegate {}", self.fail)))
        } else {
            Ok(())
        }
    }
}
impl TermVectorsWriter for Probe {
    fn start_document(&mut self, n: i32) -> io::Result<()> {
        self.call("startDoc", format!(":{n}"))
    }
    fn finish_document(&mut self) -> io::Result<()> {
        self.call("finishDoc", String::new())
    }
    fn start_field(&mut self, f: &FieldInfo, n: i32, p: bool, o: bool, y: bool) -> io::Result<()> {
        self.call(
            "startField",
            format!(":{}:{}:{n}:{p}:{o}:{y}", f.name, f.number),
        )
    }
    fn finish_field(&mut self) -> io::Result<()> {
        self.call("finishField", String::new())
    }
    fn start_term(&mut self, t: &[u8], n: i32) -> io::Result<()> {
        self.call("startTerm", format!(":{}:{n}", hex(t)))
    }
    fn finish_term(&mut self) -> io::Result<()> {
        self.call("finishTerm", String::new())
    }
    fn add_position(&mut self, p: i32, s: i32, e: i32, y: Option<&[u8]>) -> io::Result<()> {
        self.positions += 1;
        self.call(
            "position",
            format!(
                ":{p}:{s}:{e}:{}",
                y.map(|v| if v.is_empty() { "e".into() } else { hex(v) })
                    .unwrap_or_else(|| "-".into())
            ),
        )
    }
    fn finish(&mut self, n: i32) -> io::Result<()> {
        self.call("finish", format!(":{n}"))
    }
    fn close(&mut self) -> io::Result<()> {
        self.call("close", String::new())
    }
    fn ram_bytes_used(&self) -> usize {
        self.log.borrow_mut().push("ram".into());
        37
    }
    fn add_prox(
        &mut self,
        _: i32,
        _: Option<&mut dyn DataInput>,
        _: Option<&mut dyn DataInput>,
    ) -> io::Result<()> {
        panic!("delegate addProx")
    }
    fn merge(&mut self, _: &mut MergeState<'_>) -> io::Result<i32> {
        panic!("delegate merge")
    }
}
struct Format(Option<Probe>);
impl TermVectorsFormat for Format {
    type Writer = Probe;
    type Reader = Option<()>;
    fn vectors_writer(&mut self, _: &VectorState<'_>) -> io::Result<Probe> {
        let mut p = self.0.take().unwrap();
        p.call("open", String::new())?;
        Ok(p)
    }
    fn vectors_reader(&mut self, _: &VectorState<'_>) -> io::Result<Option<()>> {
        self.0.as_mut().unwrap().call("reader", String::new())?;
        Ok(None)
    }
}
struct Reader<'a> {
    name: &'a str,
    fail: &'a str,
    base: &'a SimpleTextVectorsReader,
    log: Rc<RefCell<Vec<String>>>,
}
impl Reader<'_> {
    fn call(&self, operation: String) -> io::Result<()> {
        self.log.borrow_mut().push(operation.clone());
        if operation == self.fail {
            Err(io::Error::other(format!("delegate {}", self.fail)))
        } else {
            Ok(())
        }
    }
}
impl TermVectorsReader for Reader<'_> {
    fn get(&self, doc: i32) -> io::Result<Option<VectorDocument>> {
        self.call(format!("get:{}:{doc}", self.name))?;
        self.base.get(doc)
    }
    fn check_integrity(&self) -> io::Result<()> {
        self.call(format!("integrity:{}", self.name))?;
        self.base.check_integrity()
    }
}
fn fixture(name: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/testframework/data")
        .join(name)
}
#[test]
fn java_traces() {
    let base = SimpleTextVectorsReader::open(
        std::fs::read(fixture("cranky_term_vectors.tvc")).unwrap(),
        2,
    )
    .unwrap();
    for line in std::fs::read_to_string(fixture("cranky_term_vectors.tsv"))
        .unwrap()
        .lines()
        .filter(|l| l.starts_with("case\t"))
    {
        let c: Vec<_> = line.split('\t').collect();
        let name = c[1];
        let seed = c[2].parse().unwrap();
        let fail = c[3];
        let random = Rc::new(RefCell::new(JavaRandom::new(seed)));
        let log = Rc::new(RefCell::new(Vec::new()));
        let probe = Probe {
            log: log.clone(),
            fail: fail.into(),
            positions: 0,
        };
        let mut writer = CrankyTermVectorsWriter::new(probe, random.clone());
        let mut p = None;
        let mut o = None;
        let mut merged = -1;
        let result: io::Result<()> = if let Some(operation) = name
            .strip_prefix("direct:")
            .and_then(|v| v.split(':').next())
        {
            let f = FieldInfo::new("f", 7);
            match operation {
                "reader" | "open" => {
                    let dir = ByteBuffersDirectory::new();
                    let state = VectorState {
                        directory: &dir,
                        segment: "_ctv",
                        max_doc: 0,
                    };
                    let probe = Probe {
                        log: log.clone(),
                        fail: fail.into(),
                        positions: 0,
                    };
                    let mut format =
                        CrankyTermVectorsFormat::new(Format(Some(probe)), random.clone());
                    if operation == "reader" {
                        format.vectors_reader(&state).map(|r| assert!(r.is_none()))
                    } else {
                        format.vectors_writer(&state).map(|_| ())
                    }
                }
                "startDoc" => writer.start_document(1),
                "finishDoc" => writer.finish_document(),
                "startField" => writer.start_field(&f, 1, true, true, true),
                "finishField" => writer.finish_field(),
                "startTerm" => writer.start_term(&[0, 10, 92, 255], 3),
                "finishTerm" => writer.finish_term(),
                "position" => writer.add_position(7, 10, 13, Some(&[])),
                "finish" => writer.finish(2),
                "close" => writer.close(),
                "resources" => {
                    assert_eq!(writer.ram_bytes_used(), 37);
                    assert!(std::ptr::eq(
                        writer.child_resources().next().unwrap(),
                        writer.delegate()
                    ));
                    Ok(())
                }
                _ => panic!("{name}"),
            }
        } else if let Some(v) = name.strip_prefix("prox:") {
            if v != "none" && v != "offsets" {
                p = Some(SliceInput::new(if v == "shortpos" { &[3] } else { POS }));
            }
            if v != "none" && v != "positions" {
                o = Some(SliceInput::new(if v == "shortoff" { &[0] } else { OFF }));
            }
            writer.add_prox(
                if v == "zero" {
                    0
                } else if v == "negative" {
                    -1
                } else {
                    3
                },
                p.as_mut().map(|i| i as &mut dyn DataInput),
                o.as_mut().map(|i| i as &mut dyn DataInput),
            )
        } else if let Some(v) = name.strip_prefix("merge:") {
            let a = Reader {
                name: "a",
                fail,
                base: &base,
                log: log.clone(),
            };
            let b = Reader {
                name: "b",
                fail,
                base: &base,
                log: log.clone(),
            };
            let maps = if v == "sorted" {
                vec![vec![2, -1], vec![0], vec![1, 3]]
            } else {
                vec![vec![0, -1], vec![1], vec![2, 3]]
            };
            let sources = [
                MergeSource {
                    reader: Some(&a),
                    doc_map: &maps[0],
                },
                MergeSource {
                    reader: None,
                    doc_map: &maps[1],
                },
                MergeSource {
                    reader: Some(&b),
                    doc_map: &maps[2],
                },
            ];
            let infos = FieldInfos::new(vec![FieldInfo::new("f", 7)]).unwrap();
            let mut abort = || Ok(());
            let mut state = MergeState {
                sources: &sources,
                field_infos: &infos,
                needs_index_sort: v == "sorted",
                check_aborted: &mut abort,
            };
            writer.merge(&mut state).map(|n| {
                merged = n;
            })
        } else {
            panic!("{name}")
        };
        assert_eq!(
            result
                .err()
                .map(|e| e.to_string())
                .unwrap_or_else(|| "ok".into()),
            c[4],
            "{name}"
        );
        let actual = log.borrow().join(";");
        assert_eq!(
            if actual.is_empty() { "-" } else { &actual },
            c[5],
            "{name}"
        );
        assert_eq!(
            random.borrow_mut().next_int(),
            c[6].parse::<i32>().unwrap(),
            "{name}"
        );
        assert_eq!(
            random.borrow_mut().next_long(),
            c[7].parse::<i64>().unwrap(),
            "{name}"
        );
        assert_eq!(
            p.as_ref().map(|i| i.position() as i32).unwrap_or(-1),
            c[8].parse::<i32>().unwrap(),
            "{name}"
        );
        assert_eq!(
            o.as_ref().map(|i| i.position() as i32).unwrap_or(-1),
            c[9].parse::<i32>().unwrap(),
            "{name}"
        );
        assert_eq!(merged, c[10].parse::<i32>().unwrap(), "{name}");
    }
}
fn document(writer: &mut impl TermVectorsWriter, rich: bool) -> io::Result<()> {
    writer.start_document(i32::from(rich))?;
    if rich {
        writer.start_field(&FieldInfo::new("f", 0), 1, true, true, true)?;
        writer.start_term(&[0, 10, 92, 255], 3)?;
        writer.add_prox(
            3,
            Some(&mut SliceInput::new(POS)),
            Some(&mut SliceInput::new(OFF)),
        )?;
        writer.finish_term()?;
        writer.finish_field()?;
    }
    writer.finish_document()
}
#[test]
fn real_bytes_and_reader() {
    let dir = ByteBuffersDirectory::new();
    let state = VectorState {
        directory: &dir,
        segment: "_ctv",
        max_doc: 2,
    };
    let mut format = CrankyTermVectorsFormat::new(
        SimpleTextVectorsFormat,
        Rc::new(RefCell::new(JavaRandom::new(42))),
    );
    let mut writer = format.vectors_writer(&state).unwrap();
    document(&mut writer, true).unwrap();
    document(&mut writer, false).unwrap();
    writer.finish(2).unwrap();
    writer.close().unwrap();
    writer.close().unwrap();
    assert_eq!(
        dir.open("_ctv.tvc").unwrap().as_ref(),
        std::fs::read(fixture("cranky_term_vectors.tvc")).unwrap()
    );
    let reader = format.vectors_reader(&state).unwrap();
    reader.check_integrity().unwrap();
    assert!(reader.get(1).unwrap().is_none());
    assert_eq!(
        reader.get(0).unwrap().unwrap()[0].1.terms[0].positions,
        Some(vec![1, 3, 3])
    );
    assert!(reader.get(-1).is_err());
    assert!(writer.start_document(0).is_err());
    assert!(writer.finish(2).is_err());
    let mut bytes = std::fs::read(fixture("cranky_term_vectors.tvc")).unwrap();
    bytes[0] ^= 1;
    assert!(SimpleTextVectorsReader::open(bytes, 2).is_err());
}
#[test]
fn aborted_merge_and_invalid_models() {
    let bytes = std::fs::read(fixture("cranky_term_vectors.tvc")).unwrap();
    let base = SimpleTextVectorsReader::open(bytes, 2).unwrap();
    let sources = [MergeSource {
        reader: Some(&base),
        doc_map: &[0],
    }];
    let infos = FieldInfos::new(vec![]).unwrap();
    let mut abort = || Err(io::Error::other("abort"));
    let mut state = MergeState {
        sources: &sources,
        field_infos: &infos,
        needs_index_sort: false,
        check_aborted: &mut abort,
    };
    let log = Rc::new(RefCell::new(Vec::new()));
    let mut p = Probe {
        log: log.clone(),
        fail: "-".into(),
        positions: 0,
    };
    assert_eq!(
        merge_default(&mut p, &mut state).unwrap_err().to_string(),
        "abort"
    );
    assert!(log.borrow().is_empty());
    let mut abort = || Ok(());
    state.check_aborted = &mut abort;
    assert_eq!(
        merge_default(&mut p, &mut state).unwrap_err().to_string(),
        "missing destination field"
    );
}
