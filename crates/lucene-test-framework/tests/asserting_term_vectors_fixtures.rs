//! Java -ea command programs prove assertion boundaries and successful-state updates.
use std::cell::RefCell;
use std::fmt;
use std::io;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::rc::Rc;

use lucene_codecs::field_infos::{FieldInfo, FieldInfos};
use lucene_store::{ByteBuffersDirectory, DataInput, Directory, SliceInput};
use lucene_test_framework::codecs::asserting_term_vectors::*;
use lucene_test_framework::codecs::cranky_term_vectors::*;

const POS: &[u8] = &[3, 2, 0, 10, 4, 1, 0];
const REAL_POS: &[u8] = &[3, 2, 0, 10, 4, 0];
const OFF: &[u8] = &[0, 2, 1, 3, 0, 0];
const BYTES: &[u8] =
    include_bytes!("../../../fixtures/testframework/data/asserting_term_vectors.tvc");
fn hex(value: &[u8]) -> String {
    value.iter().map(|v| format!("{v:02x}")).collect()
}
type Log = Rc<RefCell<Vec<String>>>;
struct Probe {
    log: Log,
    fail: String,
    closes: usize,
}
impl Probe {
    fn new(log: &Log) -> Self {
        Self {
            log: log.clone(),
            fail: "-".into(),
            closes: 0,
        }
    }
    fn call(&mut self, name: &str, detail: String) -> io::Result<()> {
        self.log.borrow_mut().push(format!("{name}{detail}"));
        if self.fail == name || self.fail == format!("{name}#{}", self.closes) {
            self.fail = "-".into();
            Err(io::Error::other(format!("delegate {name}")))
        } else {
            Ok(())
        }
    }
}
impl TermVectorsWriter for Probe {
    fn start_document(&mut self, n: i32) -> io::Result<()> {
        self.call("doc", format!(":{n}"))
    }
    fn finish_document(&mut self) -> io::Result<()> {
        self.call("enddoc", String::new())
    }
    fn start_field(&mut self, f: &FieldInfo, n: i32, p: bool, o: bool, y: bool) -> io::Result<()> {
        self.call("field", format!(":{}:{n}:{p}:{o}:{y}", f.number))
    }
    fn finish_field(&mut self) -> io::Result<()> {
        self.call("endfield", String::new())
    }
    fn start_term(&mut self, t: &[u8], n: i32) -> io::Result<()> {
        self.call("term", format!(":{}:{n}", hex(t)))
    }
    fn finish_term(&mut self) -> io::Result<()> {
        self.call("endterm", String::new())
    }
    fn add_position(&mut self, p: i32, s: i32, e: i32, y: Option<&[u8]>) -> io::Result<()> {
        let payload = y
            .map(|v| if v.is_empty() { "e".into() } else { hex(v) })
            .unwrap_or_else(|| "-".into());
        self.call("position", format!(":{p}:{s}:{e}:{payload}"))
    }
    fn finish(&mut self, n: i32) -> io::Result<()> {
        self.call("finish", format!(":{n}"))
    }
    fn close(&mut self) -> io::Result<()> {
        self.closes += 1;
        self.call("close", format!(":{}", self.closes))
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
        panic!("delegate prox called")
    }
    fn merge(&mut self, _: &mut MergeState<'_>) -> io::Result<i32> {
        panic!("delegate merge called")
    }
}
fn outcome(op: impl FnOnce() -> io::Result<()>) -> String {
    match catch_unwind(AssertUnwindSafe(op)) {
        Err(_) => "assert".into(),
        Ok(Err(error)) => format!("io:{error}"),
        Ok(Ok(())) => "ok".into(),
    }
}
fn command(w: &mut AssertingTermVectorsWriter<Probe>, token: &str) -> io::Result<()> {
    let a: Vec<_> = token.split(':').collect();
    let f = FieldInfo::new("f", 7);
    match a[0] {
        "fail" => {
            w.delegate_mut().fail = a[1].into();
            Ok(())
        }
        "doc" => w.start_document(a[1].parse().unwrap()),
        "enddoc" => w.finish_document(),
        "field" => w.start_field(
            &f,
            a[1].parse().unwrap(),
            a[2].contains('p'),
            a[2].contains('o'),
            a[2].contains('y'),
        ),
        "endfield" => w.finish_field(),
        "term" => w.start_term(&[0, 10, 92, 255], a[1].parse().unwrap()),
        "endterm" => w.finish_term(),
        "position" => w.add_position(7, 10, 13, Some(&[])),
        "finish" => w.finish(a[1].parse().unwrap()),
        "close" => w.close(),
        "resources" => {
            assert_eq!(w.ram_bytes_used(), 37);
            assert_eq!(w.child_resources().count(), 1);
            assert!(std::ptr::eq(
                w.child_resources().next().unwrap(),
                w.delegate()
            ));
            Ok(())
        }
        "prox" | "shortpos" | "shortoff" => {
            let mut p = SliceInput::new(if a[0] == "shortpos" { &[3] } else { POS });
            let mut o = SliceInput::new(if a[0] == "shortoff" { &[0] } else { OFF });
            let result = outcome(|| w.add_prox(3, Some(&mut p), Some(&mut o)));
            w.delegate()
                .log
                .borrow_mut()
                .push(format!("inputs:{}:{}", p.position(), o.position()));
            if result == "ok" {
                Ok(())
            } else {
                Err(io::Error::other(result))
            }
        }
        _ => panic!("unknown command {token}"),
    }
}
struct Reader {
    log: Log,
    fail: RefCell<String>,
    closes: usize,
}
impl Reader {
    fn new(log: &Log) -> Self {
        Self {
            log: log.clone(),
            fail: RefCell::new("-".into()),
            closes: 0,
        }
    }
    fn call(&self, name: &str) -> io::Result<()> {
        self.log.borrow_mut().push(name.into());
        if *self.fail.borrow() == name {
            *self.fail.borrow_mut() = "-".into();
            Err(io::Error::other(format!("delegate {name}")))
        } else {
            Ok(())
        }
    }
}
impl fmt::Display for Reader {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("ProbeReader")
    }
}
impl TermVectorsReader for Reader {
    fn get(&self, doc: i32) -> io::Result<Option<VectorDocument>> {
        self.call(&format!("get:{doc}"))?;
        Ok(None)
    }
    fn check_integrity(&self) -> io::Result<()> {
        self.call("integrity")
    }
}
impl OwnedVectorsReader for Reader {
    fn close(&mut self) -> io::Result<()> {
        self.closes += 1;
        self.call(&format!("close:{}", self.closes))
    }
    fn clone_reader(&self) -> io::Result<Self> {
        self.call("clone")?;
        Ok(Self::new(&self.log))
    }
    fn get_merge_instance(&self) -> io::Result<Self> {
        self.call("merge")?;
        Ok(Self::new(&self.log))
    }
}
#[test]
fn actual_java_assertion_and_callback_programs() {
    let base = OwnedSimpleTextReader::open(BYTES.to_vec(), 2).unwrap();
    for line in
        include_str!("../../../fixtures/testframework/data/asserting_term_vectors.tsv").lines()
    {
        let c: Vec<_> = line.split('\t').collect();
        let log = Log::default();
        let statuses = match c[0] {
            "writer" => {
                let mut w = AssertingTermVectorsWriter::new(Probe::new(&log));
                c[2].split(';')
                    .map(|token| outcome(|| command(&mut w, token)))
                    .collect::<Vec<_>>()
            }
            "reader" => {
                let p = Reader::new(&log);
                if let Some(fail) = c[1].strip_prefix("fail-") {
                    *p.fail.borrow_mut() = fail.into();
                }
                let mut r = AssertingTermVectorsReader::new(p);
                vec![outcome(|| match c[1] {
                    "clone" => r.clone_reader()?.close(),
                    "merge" => r.get_merge_instance()?.close(),
                    "close" | "fail-close:1" | "fail-close:2" => r.close(),
                    "integrity" | "fail-integrity" => r.check_integrity(),
                    "string" => {
                        assert_eq!(r.to_string(), "AssertingTermVectorsReader(ProbeReader)");
                        Ok(())
                    }
                    _ => {
                        assert!(r.get(0)?.is_none());
                        Ok(())
                    }
                })]
            }
            "merge" => {
                let mut w = AssertingTermVectorsWriter::new(Probe::new(&log));
                let sorted = c[1] == "sorted";
                let a = if sorted { [2, -1] } else { [0, -1] };
                let b = if sorted { [0] } else { [1] };
                let d = if sorted { [1, 3] } else { [2, 3] };
                let sources = [
                    MergeSource {
                        reader: Some(&base),
                        doc_map: &a,
                    },
                    MergeSource {
                        reader: None,
                        doc_map: &b,
                    },
                    MergeSource {
                        reader: Some(&base),
                        doc_map: &d,
                    },
                ];
                let infos = FieldInfos::new(vec![FieldInfo::new("f", 7)]).unwrap();
                let mut abort = || Ok(());
                let mut state = MergeState {
                    sources: &sources,
                    field_infos: &infos,
                    needs_index_sort: sorted,
                    check_aborted: &mut abort,
                };
                vec![outcome(|| {
                    let n = w.merge(&mut state)?;
                    log.borrow_mut().push(format!("merged:{n}"));
                    Ok(())
                })]
            }
            _ => panic!("unknown row {line}"),
        };
        assert_eq!(statuses.join(";"), c[3], "{} {}", c[0], c[1]);
        let actual = if log.borrow().is_empty() {
            "-".into()
        } else {
            log.borrow().join(";")
        };
        assert_eq!(actual, c[4], "{} {}", c[0], c[1]);
    }
}

#[test]
fn real_output_and_owned_reader_values() {
    let dir = ByteBuffersDirectory::new();
    let state = VectorState {
        directory: &dir,
        segment: "_atv",
        max_doc: 2,
    };
    let mut f = AssertingTermVectorsFormat::new(OwnedSimpleTextVectorsFormat);
    assert_eq!(std::mem::size_of_val(f.delegate()), 0);
    assert_eq!(std::mem::size_of_val(f.delegate_mut()), 0);
    let mut w = f.vectors_writer(&state).unwrap();
    w.start_document(1).unwrap();
    w.start_field(&FieldInfo::new("f", 0), 1, true, true, true)
        .unwrap();
    w.start_term(&[0, 10, 92, 255], 3).unwrap();
    w.add_prox(
        3,
        Some(&mut SliceInput::new(REAL_POS)),
        Some(&mut SliceInput::new(OFF)),
    )
    .unwrap();
    w.finish_term().unwrap();
    w.finish_field().unwrap();
    w.finish_document().unwrap();
    w.start_document(0).unwrap();
    w.finish_document().unwrap();
    w.finish(2).unwrap();
    w.close().unwrap();
    assert_eq!(dir.open("_atv.tvc").unwrap().as_ref(), BYTES);
    assert_eq!(w.ram_bytes_used(), BYTES.len());
    let mut r = f.vectors_reader(&state).unwrap();
    assert_eq!(r.delegate().to_string(), "SimpleTextTermVectorsReader");
    assert_eq!(r.delegate_mut().to_string(), "SimpleTextTermVectorsReader");
    r.check_integrity().unwrap();
    assert!(r.get(1).unwrap().is_none());
    let doc = r.get(0).unwrap().unwrap();
    assert_eq!(doc[0].0, "f");
    let t = &doc[0].1.terms[0];
    assert_eq!(t.term, [0, 10, 92, 255]);
    assert_eq!(t.freq, 3);
    assert_eq!(t.positions, Some(vec![1, 3, 3]));
    assert_eq!(t.start_offsets, Some(vec![0, 3, 6]));
    assert_eq!(t.end_offsets, Some(vec![2, 6, 6]));
    assert_eq!(t.payloads, Some(vec![vec![0, 10], vec![], vec![]]));
    assert!(r.get(-1).is_err());
    assert!(r.get(2).is_err());
    r.close().unwrap();
    assert!(r.get(0).is_err());
}

struct FailingFormat;
impl TermVectorsFormat for FailingFormat {
    type Writer = Probe;
    type Reader = Reader;
    fn vectors_writer(&mut self, _: &VectorState<'_>) -> io::Result<Probe> {
        Err(io::Error::other("open writer"))
    }
    fn vectors_reader(&mut self, _: &VectorState<'_>) -> io::Result<Reader> {
        Err(io::Error::other("open reader"))
    }
}
#[test]
fn open_errors_propagate_and_reader_clone_errors_propagate() {
    let dir = ByteBuffersDirectory::new();
    let s = VectorState {
        directory: &dir,
        segment: "missing",
        max_doc: 0,
    };
    let mut f = AssertingTermVectorsFormat::new(FailingFormat);
    assert_eq!(
        f.vectors_writer(&s).err().unwrap().to_string(),
        "open writer"
    );
    assert_eq!(
        f.vectors_reader(&s).err().unwrap().to_string(),
        "open reader"
    );
    assert!(OwnedSimpleTextVectorsFormat.vectors_reader(&s).is_err());
    let log = Log::default();
    let r = AssertingTermVectorsReader::new(Reader::new(&log));
    *r.delegate().fail.borrow_mut() = "clone".into();
    assert_eq!(
        r.clone_reader().err().unwrap().to_string(),
        "delegate clone"
    );
    *r.delegate().fail.borrow_mut() = "merge".into();
    assert_eq!(
        r.get_merge_instance().err().unwrap().to_string(),
        "delegate merge"
    );
}
