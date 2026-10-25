//! Actual Java cranky callbacks, base merge and real stored-fields triples.
use lucene_codecs::stored_fields::{
    self, Document, FieldValue, Mode, StoredFieldVisitor, VisitStatus,
};
use lucene_store::data_input::{DataInput, SliceInput};
use lucene_store::{ByteBuffersDirectory, Directory};
use lucene_test_framework::codecs::cranky_stored_fields::*;
use lucene_util::java_random::JavaRandom;
use std::cell::RefCell;
use std::rc::Rc;

const ID: [u8; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
fn state(directory: &dyn Directory, mode: Mode) -> StoredFieldsState<'_> {
    StoredFieldsState {
        directory,
        segment: "_real",
        id: &ID,
        suffix: "",
        mode,
    }
}
fn failure(message: &str) -> stored_fields::Error {
    lucene_store::Error::Io(std::io::Error::other(message.to_string())).into()
}
#[derive(Default)]
struct State {
    fail: bool,
    events: Vec<String>,
}
struct RecordingWriter {
    state: Rc<RefCell<State>>,
    children: Rc<Vec<ChildResource>>,
}
impl RecordingWriter {
    fn event(&self, event: String) -> stored_fields::Result<()> {
        let mut state = self.state.borrow_mut();
        state.events.push(event);
        if state.fail {
            Err(failure("delegate failure"))
        } else {
            Ok(())
        }
    }
}
impl StoredFieldsWriter for RecordingWriter {
    fn start_document(&mut self) -> stored_fields::Result<()> {
        self.event("start".into())
    }
    fn finish_document(&mut self) -> stored_fields::Result<()> {
        self.event("finishDoc".into())
    }
    fn finish(&mut self, count: i32) -> stored_fields::Result<()> {
        self.event(format!("finish:{count}"))
    }
    fn close(&mut self) -> stored_fields::Result<()> {
        self.event("close".into())
    }
    fn write_field(&mut self, field: i32, value: &FieldValue) -> stored_fields::Result<()> {
        let text = match value {
            FieldValue::Int(value) => format!("int:{field}:{value}"),
            FieldValue::Long(value) => format!("long:{field}:{value}"),
            FieldValue::Float(value) => format!("float:{field}:{}", value.to_bits()),
            FieldValue::Double(value) => format!("double:{field}:{}", value.to_bits()),
            FieldValue::Binary(value) => format!("binary:{field}:{}", hex(value)),
            FieldValue::String(value) => format!("string:{field}:{value}"),
        };
        self.event(text)
    }
    fn write_field_data_input(
        &mut self,
        field: i32,
        input: &mut dyn DataInput,
        length: usize,
    ) -> stored_fields::Result<()> {
        self.event(format!("input:{field}:{length}"))?;
        let mut bytes = vec![0; length];
        input.read_bytes(&mut bytes)?;
        assert_eq!(bytes, [9, 8]);
        Ok(())
    }
    fn ram_bytes_used(&self) -> usize {
        self.state.borrow_mut().events.push("ram".into());
        73
    }
    fn child_resources(&self) -> &[ChildResource] {
        self.state.borrow_mut().events.push("children".into());
        &self.children
    }
    fn merge(&mut self, _: &MergeState<'_>) -> stored_fields::Result<i32> {
        panic!("delegate merge called")
    }
}
struct RecordingFormat {
    state: Rc<RefCell<State>>,
    children: Rc<Vec<ChildResource>>,
}
impl StoredFieldsFormat for RecordingFormat {
    type Writer<'a> = RecordingWriter;
    type Reader<'a> = Rc<RefCell<State>>;
    fn fields_writer<'a>(
        &mut self,
        _: StoredFieldsState<'a>,
    ) -> stored_fields::Result<Self::Writer<'a>> {
        let writer = RecordingWriter {
            state: self.state.clone(),
            children: self.children.clone(),
        };
        writer.event("open".into())?;
        Ok(writer)
    }
    fn fields_reader<'a>(
        &mut self,
        _: StoredFieldsReadState<'a>,
    ) -> stored_fields::Result<Self::Reader<'a>> {
        let mut state = self.state.borrow_mut();
        state.events.push("read".into());
        if state.fail {
            Err(failure("delegate failure"))
        } else {
            Ok(self.state.clone())
        }
    }
}
fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}
fn first(bound: i32) -> i64 {
    (0..)
        .find(|&seed| JavaRandom::new(seed).next_int_bounded(bound) == 0)
        .unwrap()
}

struct Source {
    state: Rc<RefCell<State>>,
    source: i32,
    integrity: bool,
    visit: bool,
}
impl StoredFieldsSource for Source {
    fn check_integrity(&self) -> stored_fields::Result<()> {
        self.state
            .borrow_mut()
            .events
            .push(format!("integrity:{}", self.source));
        if self.integrity {
            Err(failure("source integrity"))
        } else {
            Ok(())
        }
    }
    fn visit_document(
        &self,
        doc: i32,
        visitor: &mut dyn StoredFieldVisitor,
    ) -> stored_fields::Result<()> {
        self.state
            .borrow_mut()
            .events
            .push(format!("visit:{}:{doc}", self.source));
        let field = 10 + self.source * 10;
        if visitor.needs_field(field)? == VisitStatus::Yes {
            visitor.int_field(field, self.source * 100 + doc)?;
        }
        if self.visit {
            return Err(failure("source visit"));
        }
        if visitor.needs_field(field + 1)? == VisitStatus::Yes {
            visitor.binary_field(field + 1, &[self.source as u8, doc as u8])?;
        }
        Ok(())
    }
}

#[test]
fn actual_java_callback_trace_and_super_merge() {
    let mut lines =
        include_str!("../../../fixtures/testframework/data/cranky_stored_fields.tsv").lines();
    let directory = ByteBuffersDirectory::new();
    let children = Rc::new(vec![ChildResource {
        label: "child".into(),
        bytes: 0,
    }]);
    let mut operations = 0;
    for seed in [
        0,
        1,
        42,
        -1,
        i64::MIN,
        i64::MAX,
        first(100),
        first(1000),
        first(10000),
    ] {
        let state = Rc::new(RefCell::new(State::default()));
        let random = Rc::new(RefCell::new(JavaRandom::new(seed)));
        let mut format = CrankyStoredFieldsFormat::new(
            RecordingFormat {
                state: state.clone(),
                children: children.clone(),
            },
            random.clone(),
        );
        let mut writer = CrankyStoredFieldsWriter::new(
            RecordingWriter {
                state: state.clone(),
                children: children.clone(),
            },
            random.clone(),
        );
        let empty_reader = Source {
            state: state.clone(),
            source: 0,
            integrity: false,
            visit: false,
        };
        let empty_sources = [MergeSource {
            reader: &empty_reader,
            doc_map: &[],
            field_map: &[],
        }];
        let empty = MergeState {
            sources: &empty_sources,
            needs_index_sort: false,
            check_aborted: &|| Ok(()),
        };
        let names = [
            "open",
            "read",
            "start",
            "finishDoc",
            "int",
            "long",
            "float",
            "double",
            "binary",
            "input",
            "string",
            "finish",
            "close",
            "ram",
            "children",
            "merge",
        ];
        for step in 0..2048 {
            {
                let mut state = state.borrow_mut();
                state.fail = step % 17 == 0;
                state.events.clear();
            }
            let op = names[step % names.len()];
            let mut input = SliceInput::new(&[9, 8]);
            let result = match op {
                "open" => format.fields_writer(crate_state(&directory)).map(|_| ()),
                "read" => format
                    .fields_reader(StoredFieldsReadState {
                        fdt: &[],
                        fdx: &[],
                        fdm: &[],
                        id: &ID,
                        suffix: "",
                    })
                    .map(|actual| assert!(Rc::ptr_eq(&actual, &state))),
                "start" => writer.start_document(),
                "finishDoc" => writer.finish_document(),
                "int" => writer.write_field(7, &FieldValue::Int(-3)),
                "long" => writer.write_field(7, &FieldValue::Long(i64::MIN)),
                "float" => writer.write_field(7, &FieldValue::Float(-0.0)),
                "double" => {
                    writer.write_field(7, &FieldValue::Double(f64::from_bits(0x7ff8000000000042)))
                }
                "binary" => writer.write_field(7, &FieldValue::Binary(vec![0, 1, 255])),
                "input" => writer.write_field_data_input(7, &mut input, 2),
                "string" => writer.write_field(7, &FieldValue::String("hello λ".into())),
                "finish" => writer.finish(3),
                "close" => writer.close(),
                "ram" => {
                    assert_eq!(writer.ram_bytes_used(), 73);
                    Ok(())
                }
                "children" => {
                    assert_eq!(writer.child_resources().as_ptr(), children.as_ptr());
                    Ok(())
                }
                "merge" => writer.merge(&empty).map(|count| assert_eq!(count, 0)),
                _ => unreachable!(),
            };
            let status = result.map_or_else(|error| error.to_string(), |_| "ok".into());
            let state = state.borrow();
            let line = format!(
                "op\t{seed}\t{step}\t{op}\t{}\t{status}\t{}\t{}",
                state.fail,
                state.events.join(","),
                input.position()
            );
            assert_eq!(
                Some(line.as_str()),
                lines.next(),
                "seed={seed}, step={step}"
            );
            operations += 1;
        }
        assert_eq!(
            Some(format!("rng\t{seed}\t{}", random.borrow_mut().next_int()).as_str()),
            lines.next()
        );
    }
    assert_eq!(operations, 18432);
    for sorted in [false, true] {
        for (integrity, visit) in [(false, false), (true, false), (false, true)] {
            let state = Rc::new(RefCell::new(State::default()));
            let random = Rc::new(RefCell::new(JavaRandom::new(42)));
            let mut writer = CrankyStoredFieldsWriter::new(
                RecordingWriter {
                    state: state.clone(),
                    children: children.clone(),
                },
                random.clone(),
            );
            let readers = [
                Source {
                    state: state.clone(),
                    source: 0,
                    integrity: false,
                    visit,
                },
                Source {
                    state: state.clone(),
                    source: 1,
                    integrity,
                    visit: false,
                },
            ];
            let maps = if sorted {
                [[0, -1, 2], [1, 3, -1]]
            } else {
                [[0, -1, 1], [2, 3, -1]]
            };
            let sources = [
                MergeSource {
                    reader: &readers[0],
                    doc_map: &maps[0],
                    field_map: &[(10, 0), (11, 1)],
                },
                MergeSource {
                    reader: &readers[1],
                    doc_map: &maps[1],
                    field_map: &[(20, 0), (21, 1)],
                },
            ];
            let result = writer.merge(&MergeState {
                sources: &sources,
                needs_index_sort: sorted,
                check_aborted: &|| Ok(()),
            });
            let (status, count) = match result {
                Ok(count) => ("ok".into(), count),
                Err(error) => (error.to_string(), -1),
            };
            let line = format!(
                "merge\t{sorted}\t{integrity}\t{visit}\t{status}\t{count}\t{}\t{}",
                state.borrow().events.join(","),
                random.borrow_mut().next_int()
            );
            assert_eq!(Some(line.as_str()), lines.next());
        }
    }
    assert!(lines.next().is_none());
}
fn crate_state(directory: &dyn Directory) -> StoredFieldsState<'_> {
    state(directory, Mode::BestSpeed)
}

fn real_values(doc: i32) -> Vec<FieldValue> {
    vec![
        FieldValue::Int(-3 + doc),
        FieldValue::Long(i64::MIN + i64::from(doc)),
        FieldValue::Float(-0.0),
        FieldValue::Double(f64::from_bits(0x7ff8000000000042)),
        FieldValue::Binary(vec![0, 1, 255]),
        FieldValue::Binary(vec![9, 8]),
        FieldValue::String(format!("hello λ {doc}")),
    ]
}
fn assert_document(document: Document, doc: i32) {
    assert_eq!(document.fields.len(), 7);
    for (number, (actual, expected)) in document.fields.iter().zip(real_values(doc)).enumerate() {
        assert_eq!(actual.field_number, number as i32);
        match (&actual.value, expected) {
            (FieldValue::Float(actual), FieldValue::Float(expected)) => {
                assert_eq!(actual.to_bits(), expected.to_bits())
            }
            (FieldValue::Double(actual), FieldValue::Double(_)) => {
                assert_eq!(actual.to_bits(), f64::NAN.to_bits())
            }
            (actual, expected) => assert_eq!(actual, &expected),
        }
    }
}
#[test]
fn real_java_and_native_triples_decode_all_types_and_nan_bits() {
    for (mode, fdt, fdx, fdm, rng) in [
        (
            Mode::BestSpeed,
            include_bytes!("../../../fixtures/testframework/data/cranky_stored_fields_speed.fdt")
                .as_slice(),
            include_bytes!("../../../fixtures/testframework/data/cranky_stored_fields_speed.fdx")
                .as_slice(),
            include_bytes!("../../../fixtures/testframework/data/cranky_stored_fields_speed.fdm")
                .as_slice(),
            include_str!("../../../fixtures/testframework/data/cranky_stored_fields_speed.rng"),
        ),
        (
            Mode::BestCompression,
            include_bytes!(
                "../../../fixtures/testframework/data/cranky_stored_fields_compression.fdt"
            )
            .as_slice(),
            include_bytes!(
                "../../../fixtures/testframework/data/cranky_stored_fields_compression.fdx"
            )
            .as_slice(),
            include_bytes!(
                "../../../fixtures/testframework/data/cranky_stored_fields_compression.fdm"
            )
            .as_slice(),
            include_str!(
                "../../../fixtures/testframework/data/cranky_stored_fields_compression.rng"
            ),
        ),
    ] {
        let directory = ByteBuffersDirectory::new();
        let random = Rc::new(RefCell::new(JavaRandom::new(0)));
        let mut format = CrankyStoredFieldsFormat::new(Lucene90StoredFieldsFormat, random.clone());
        let mut writer = format.fields_writer(state(&directory, mode)).unwrap();
        for doc in 0..3 {
            writer.start_document().unwrap();
            for (number, value) in real_values(doc).iter().enumerate() {
                if number == 5 {
                    writer
                        .write_field_data_input(5, &mut SliceInput::new(&[9, 8]), 2)
                        .unwrap();
                } else {
                    writer.write_field(number as i32, value).unwrap();
                }
            }
            writer.finish_document().unwrap();
        }
        assert!(writer.ram_bytes_used() > 0);
        assert!(writer.child_resources().is_empty());
        writer.finish(3).unwrap();
        writer.close().unwrap();
        assert_eq!(random.borrow_mut().next_int().to_string(), rng.trim());
        let native_fdt = directory.open("_real.fdt").unwrap();
        let native_fdx = directory.open("_real.fdx").unwrap();
        let native_fdm = directory.open("_real.fdm").unwrap();
        assert_eq!(
            &*native_fdt, fdt,
            "native fdt must equal actual Java wrapper output"
        );
        assert_eq!(
            &*native_fdx, fdx,
            "native fdx must equal actual Java wrapper output"
        );
        assert_eq!(
            &*native_fdm, fdm,
            "native fdm must equal actual Java wrapper output"
        );
        for (fdt, fdx, fdm) in [(fdt, fdx, fdm), (&*native_fdt, &*native_fdx, &*native_fdm)] {
            let reader = format
                .fields_reader(StoredFieldsReadState {
                    fdt,
                    fdx,
                    fdm,
                    id: &ID,
                    suffix: "",
                })
                .unwrap();
            reader.check_integrity().unwrap();
            assert_eq!(reader.max_doc(), 3);
            for doc in 0..3 {
                assert_document(reader.document(doc).unwrap(), doc);
            }
            let mut damaged = fdt.to_vec();
            let end = damaged.len() - 1;
            damaged[end] ^= 1;
            let reader = format
                .fields_reader(StoredFieldsReadState {
                    fdt: &damaged,
                    fdx,
                    fdm,
                    id: &ID,
                    suffix: "",
                })
                .unwrap();
            assert!(reader.check_integrity().is_err());
            assert!(format
                .fields_reader(StoredFieldsReadState {
                    fdt,
                    fdx,
                    fdm,
                    id: &[99; 16],
                    suffix: ""
                })
                .is_err());
            assert!(format
                .fields_reader(StoredFieldsReadState {
                    fdt: &fdt[..12],
                    fdx,
                    fdm,
                    id: &ID,
                    suffix: ""
                })
                .is_err());
        }
    }
}
#[test]
fn close_without_finish_matches_java_unfinished_files() {
    let dir = ByteBuffersDirectory::new();
    let random = Rc::new(RefCell::new(JavaRandom::new(0)));
    let mut format = CrankyStoredFieldsFormat::new(Lucene90StoredFieldsFormat, random.clone());
    let mut writer = format.fields_writer(crate_state(&dir)).unwrap();
    writer.close().unwrap();
    assert_eq!(
        &*dir.open("_real.fdt").unwrap(),
        include_bytes!("../../../fixtures/testframework/data/cranky_stored_fields_unfinished.fdt")
    );
    assert_eq!(
        &*dir.open("_real.fdm").unwrap(),
        include_bytes!("../../../fixtures/testframework/data/cranky_stored_fields_unfinished.fdm")
    );
    assert!(dir.open("_real.fdx").is_err());
    assert_eq!(
        random.borrow_mut().next_int().to_string(),
        include_str!("../../../fixtures/testframework/data/cranky_stored_fields_unfinished.rng")
            .trim()
    );
}

#[test]
fn native_reader_merge_remaps_all_types_skips_deletions_and_sorts() {
    let fdt = include_bytes!("../../../fixtures/testframework/data/cranky_stored_fields_speed.fdt");
    let fdx = include_bytes!("../../../fixtures/testframework/data/cranky_stored_fields_speed.fdx");
    let fdm = include_bytes!("../../../fixtures/testframework/data/cranky_stored_fields_speed.fdm");
    let reader = stored_fields::open(fdt, fdx, fdm, &ID, "").unwrap();
    for sorted in [false, true] {
        let directory = ByteBuffersDirectory::new();
        let random = Rc::new(RefCell::new(JavaRandom::new(0)));
        let mut format = CrankyStoredFieldsFormat::new(Lucene90StoredFieldsFormat, random);
        let mut writer = format.fields_writer(crate_state(&directory)).unwrap();
        let maps = if sorted {
            [[0, -1, 2], [1, 3, -1]]
        } else {
            [[0, -1, 1], [2, 3, -1]]
        };
        let remap = [(0, 6), (1, 5), (2, 4), (3, 3), (4, 2), (5, 1), (6, 0)];
        let sources = [
            MergeSource {
                reader: &reader,
                doc_map: &maps[0],
                field_map: &remap,
            },
            MergeSource {
                reader: &reader,
                doc_map: &maps[1],
                field_map: &remap,
            },
        ];
        assert_eq!(
            writer
                .merge(&MergeState {
                    sources: &sources,
                    needs_index_sort: sorted,
                    check_aborted: &|| Ok(())
                })
                .unwrap(),
            4
        );
        writer.close().unwrap();
        let output_fdt = directory.open("_real.fdt").unwrap();
        let output_fdx = directory.open("_real.fdx").unwrap();
        let output_fdm = directory.open("_real.fdm").unwrap();
        let merged = stored_fields::open(&output_fdt, &output_fdx, &output_fdm, &ID, "").unwrap();
        merged.check_integrity().unwrap();
        assert_eq!(merged.max_doc(), 4);
        let source_docs = if sorted { [0, 0, 2, 1] } else { [0, 2, 0, 1] };
        for (doc, source_doc) in source_docs.into_iter().enumerate() {
            let mut document = merged.document(doc as i32).unwrap();
            for field in &mut document.fields {
                field.field_number = 6 - field.field_number;
            }
            assert_document(document, source_doc);
        }
    }
}
struct FinishFailureDirectory {
    inner: ByteBuffersDirectory,
    failure: &'static str,
}
struct FailIndexClose(&'static str);
impl lucene_store::index_output::OutputFaults for FailIndexClose {
    fn before_write(&mut self, _: &str, _: usize) -> lucene_store::index_output::WriteVerdict {
        lucene_store::index_output::WriteVerdict::Write
    }
    fn on_close(&mut self, _: &str) -> lucene_store::index_output::CloseVerdict {
        lucene_store::index_output::CloseVerdict::FailAfterFlush(std::io::Error::other(self.0))
    }
}
impl Directory for FinishFailureDirectory {
    fn list_all(&self) -> lucene_store::Result<Vec<String>> {
        self.inner.list_all()
    }
    fn open(&self, name: &str) -> lucene_store::Result<lucene_store::directory::Input> {
        self.inner.open(name)
    }
    fn create_output(&self, name: &str) -> lucene_store::Result<lucene_store::FsIndexOutput> {
        if name.ends_with(".fdx") && self.failure == "create" {
            return Err(lucene_store::Error::Io(std::io::Error::other("fdx create")));
        }
        let mut output = self.inner.create_output(name)?;
        if name.ends_with(".fdx") && self.failure == "close" {
            output.set_faults(Box::new(FailIndexClose("fdx close")));
        }
        if name.ends_with(".fdt") && matches!(self.failure, "data_close" | "both_close") {
            output.set_faults(Box::new(FailIndexClose("fdt close")));
        }
        if name.ends_with(".fdm") && self.failure == "both_close" {
            output.set_faults(Box::new(FailIndexClose("fdm close")));
        }
        Ok(output)
    }
    fn sync(&self, names: &[String]) -> lucene_store::Result<()> {
        self.inner.sync(names)
    }
    fn rename(&self, source: &str, target: &str) -> lucene_store::Result<()> {
        self.inner.rename(source, target)
    }
    fn delete_file(&self, name: &str) -> lucene_store::Result<()> {
        self.inner.delete_file(name)
    }
    fn sync_meta_data(&self) -> lucene_store::Result<()> {
        self.inner.sync_meta_data()
    }
    fn obtain_lock(&self, name: &str) -> lucene_store::Result<Box<dyn lucene_store::lock::Lock>> {
        self.inner.obtain_lock(name)
    }
}
#[test]
fn finish_failures_preserve_actual_java_flush_and_index_metadata_order() {
    let base =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/testframework/data");
    for failure in [
        "count",
        "create",
        "close",
        "auto",
        "retry",
        "data_close",
        "both_close",
    ] {
        let directory = FinishFailureDirectory {
            inner: ByteBuffersDirectory::new(),
            failure,
        };
        let random = Rc::new(RefCell::new(JavaRandom::new(42)));
        let mut format = CrankyStoredFieldsFormat::new(Lucene90StoredFieldsFormat, random.clone());
        let mut writer = format.fields_writer(crate_state(&directory)).unwrap();
        writer.start_document().unwrap();
        writer
            .write_field(
                0,
                &if failure == "auto" {
                    FieldValue::String("x".repeat(180000))
                } else {
                    FieldValue::Int(17)
                },
            )
            .unwrap();
        writer.finish_document().unwrap();
        let message = if failure == "auto" {
            writer.close().unwrap();
            "unfinished automatic chunk".to_string()
        } else if matches!(failure, "data_close" | "both_close") {
            writer.finish(1).unwrap();
            writer.close().unwrap_err().to_string()
        } else {
            let message = writer
                .finish(if matches!(failure, "count" | "retry") {
                    2
                } else {
                    1
                })
                .unwrap_err()
                .to_string();
            if failure == "retry" {
                writer.finish(1).unwrap();
            }
            writer.close().unwrap();
            message
        };
        let expected = std::fs::read_to_string(
            base.join(format!("cranky_stored_fields_failure_{failure}.status")),
        )
        .unwrap();
        let mut expected = expected.lines();
        assert_eq!(message, expected.next().unwrap());
        assert_eq!(
            random.borrow_mut().next_int().to_string(),
            expected.next().unwrap()
        );
        for extension in ["fdt", "fdm"] {
            assert_eq!(
                directory
                    .open(&format!("_real.{extension}"))
                    .unwrap()
                    .as_ref(),
                std::fs::read(base.join(format!(
                    "cranky_stored_fields_failure_{failure}.{extension}"
                )))
                .unwrap(),
                "{failure}: {extension}"
            );
        }
        if matches!(failure, "close" | "retry" | "data_close" | "both_close") {
            assert_eq!(
                directory.open("_real.fdx").unwrap().as_ref(),
                std::fs::read(base.join(format!("cranky_stored_fields_failure_{failure}.fdx")))
                    .unwrap()
            );
        } else {
            assert!(directory.open("_real.fdx").is_err());
        }
    }
}
