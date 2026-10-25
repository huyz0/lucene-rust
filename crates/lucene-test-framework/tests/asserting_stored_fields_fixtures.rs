use lucene_codecs::stored_fields::{self, FieldValue, Mode, StoredFieldVisitor, VisitStatus};
use lucene_store::{ByteBuffersDirectory, DataInput, Directory, SliceInput};
use lucene_test_framework::codecs::asserting_stored_fields::*;
use lucene_test_framework::codecs::cranky_stored_fields::{MergeSource, MergeState};
use std::fmt;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{Arc, Mutex};

type Events = Arc<Mutex<Vec<String>>>;
fn failure(message: &str) -> stored_fields::Error {
    lucene_store::Error::Io(std::io::Error::other(message.to_owned())).into()
}
fn outcome(action: impl FnOnce() -> stored_fields::Result<()>) -> String {
    match catch_unwind(AssertUnwindSafe(action)) {
        Ok(Ok(())) => "ok".into(),
        Ok(Err(error)) => format!("io:{error}"),
        Err(_) => "assert".into(),
    }
}
struct ProbeWriter {
    events: Events,
    fail: bool,
    closes: usize,
    fail_close: usize,
    children: Vec<ChildResource>,
}
impl ProbeWriter {
    fn new(events: Events) -> Self {
        Self {
            events,
            fail: false,
            closes: 0,
            fail_close: 0,
            children: vec![ChildResource {
                label: "child".into(),
                bytes: 19,
            }],
        }
    }
    fn call(&self, operation: &str) -> stored_fields::Result<()> {
        self.events.lock().unwrap().push(operation.into());
        if self.fail {
            Err(failure("delegate failure"))
        } else {
            Ok(())
        }
    }
}
impl StoredFieldsWriter for ProbeWriter {
    fn start_document(&mut self) -> stored_fields::Result<()> {
        self.call("start")
    }
    fn finish_document(&mut self) -> stored_fields::Result<()> {
        self.call("end")
    }
    fn write_field(&mut self, number: i32, value: &FieldValue) -> stored_fields::Result<()> {
        assert_eq!(number, 7);
        self.call(match value {
            FieldValue::Int(value) => {
                assert_eq!(*value, -17);
                "int"
            }
            FieldValue::Long(value) => {
                assert_eq!(*value, i64::MIN);
                "long"
            }
            FieldValue::Float(value) => {
                assert_eq!(value.to_bits(), 0x80000000);
                "float"
            }
            FieldValue::Double(value) => {
                assert_eq!(value.to_bits(), 0x7ff8000000000042);
                "double"
            }
            FieldValue::Binary(value) => {
                assert_eq!(value, &[0, 1, 255]);
                "binary"
            }
            FieldValue::String(value) => {
                assert_eq!(value, "hello λ");
                "string"
            }
        })
    }
    fn write_field_data_input(
        &mut self,
        number: i32,
        input: &mut dyn DataInput,
        length: usize,
    ) -> stored_fields::Result<()> {
        self.call("input")?;
        assert_eq!(number, 7);
        assert_eq!(length, 2);
        let mut bytes = [0; 2];
        input.read_bytes(&mut bytes)?;
        assert_eq!(bytes, [9, 8]);
        Ok(())
    }
    fn finish(&mut self, count: i32) -> stored_fields::Result<()> {
        self.call(&format!("finish:{count}"))
    }
    fn close(&mut self) -> stored_fields::Result<()> {
        self.closes += 1;
        self.events
            .lock()
            .unwrap()
            .push(format!("close:{}", self.closes));
        if self.closes == self.fail_close {
            Err(failure("close failure"))
        } else {
            Ok(())
        }
    }
    fn ram_bytes_used(&self) -> usize {
        self.events.lock().unwrap().push("ram".into());
        73
    }
    fn child_resources(&self) -> &[ChildResource] {
        self.events.lock().unwrap().push("children".into());
        &self.children
    }
    fn merge(&mut self, _: &MergeState<'_>) -> stored_fields::Result<i32> {
        panic!("delegate merge called")
    }
}
fn writer_trace(sequence: &str) -> (String, String, usize) {
    let events = Events::default();
    let mut wrapped = AssertingStoredFieldsWriter::new(ProbeWriter::new(events.clone()));
    let mut statuses = Vec::new();
    let mut consumed = 0;
    for token in sequence.split(',') {
        let fail = token.ends_with('!');
        let op = token.trim_end_matches('!');
        wrapped.delegate_mut().fail = fail;
        if op.starts_with("close") {
            let delegate = wrapped.delegate_mut();
            delegate.fail = false;
            delegate.fail_close = if fail {
                delegate.closes + if op == "close2" { 2 } else { 1 }
            } else {
                0
            };
        }
        let mut input = SliceInput::new(&[9, 8]);
        statuses.push(outcome(|| match op {
            "start" => wrapped.start_document(),
            "end" => wrapped.finish_document(),
            "int" => wrapped.write_field(7, &FieldValue::Int(-17)),
            "long" => wrapped.write_field(7, &FieldValue::Long(i64::MIN)),
            "float" => wrapped.write_field(7, &FieldValue::Float(-0.0)),
            "double" => {
                wrapped.write_field(7, &FieldValue::Double(f64::from_bits(0x7ff8000000000042)))
            }
            "binary" => wrapped.write_field(7, &FieldValue::Binary(vec![0, 1, 255])),
            "string" => wrapped.write_field(7, &FieldValue::String("hello λ".into())),
            "input" => wrapped.write_field_data_input(7, &mut input, 2),
            "close" | "close2" => wrapped.close(),
            "resources" => {
                assert_eq!(wrapped.ram_bytes_used(), 73);
                assert!(std::ptr::eq(
                    wrapped.child_resources(),
                    wrapped.delegate().children.as_slice()
                ));
                Ok(())
            }
            _ => wrapped.finish(op.strip_prefix("finish").unwrap().parse().unwrap()),
        }));
        consumed += input.position();
    }
    let log = events.lock().unwrap().join(",");
    (statuses.join(";"), log, consumed)
}
struct ProbeReader {
    events: Events,
    name: String,
    fail: String,
    closes: usize,
    fail_close: usize,
}
impl ProbeReader {
    fn new(events: Events, name: &str) -> Self {
        Self {
            events,
            name: name.into(),
            fail: String::new(),
            closes: 0,
            fail_close: 0,
        }
    }
    fn call(&self, op: &str) -> stored_fields::Result<()> {
        self.events
            .lock()
            .unwrap()
            .push(format!("{}:{op}", self.name));
        if self.fail == op {
            Err(failure("reader failure"))
        } else {
            Ok(())
        }
    }
}
impl fmt::Display for ProbeReader {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("ProbeReader")
    }
}
impl StoredFieldsSource for ProbeReader {
    fn check_integrity(&self) -> stored_fields::Result<()> {
        self.call("integrity")
    }
    fn visit_document(
        &self,
        doc: i32,
        visitor: &mut dyn StoredFieldVisitor,
    ) -> stored_fields::Result<()> {
        self.call(&format!("doc:{doc}"))?;
        if visitor.needs_field(7)? == VisitStatus::Yes {
            visitor.int_field(7, 17)?;
        }
        Ok(())
    }
}
impl StoredFieldsReader for ProbeReader {
    fn max_doc(&self) -> i32 {
        2
    }
    fn close(&mut self) -> stored_fields::Result<()> {
        self.closes += 1;
        self.events
            .lock()
            .unwrap()
            .push(format!("{}:close:{}", self.name, self.closes));
        if self.closes == self.fail_close {
            Err(failure("close failure"))
        } else {
            Ok(())
        }
    }
    fn clone_reader(&self) -> Self {
        self.events
            .lock()
            .unwrap()
            .push(format!("{}:clone", self.name));
        Self::new(self.events.clone(), &format!("{}c", self.name))
    }
    fn get_merge_instance(&self) -> Self {
        self.events
            .lock()
            .unwrap()
            .push(format!("{}:merge", self.name));
        Self::new(self.events.clone(), &format!("{}m", self.name))
    }
}
struct Visitor {
    fail: bool,
}
impl StoredFieldVisitor for Visitor {
    fn needs_field(&mut self, _: i32) -> stored_fields::Result<VisitStatus> {
        if self.fail {
            Err(failure("visitor failure"))
        } else {
            Ok(VisitStatus::Yes)
        }
    }
    fn int_field(&mut self, number: i32, value: i32) -> stored_fields::Result<()> {
        assert_eq!((number, value), (7, 17));
        Ok(())
    }
}
fn reader_trace(name: &str) -> (String, String) {
    let events = Events::default();
    let mut wrapped = AssertingStoredFieldsReader::with_merging(
        ProbeReader::new(events.clone(), "r"),
        2,
        name == "merge-clone",
    );
    let mut visitor = Visitor { fail: false };
    let result = outcome(|| {
        match name {
            "valid" => {
                wrapped.visit_document(0, &mut visitor)?;
                wrapped.visit_document(1, &mut visitor)?;
                wrapped.check_integrity()?;
                wrapped.close()?;
            }
            "negative" => wrapped.visit_document(-1, &mut visitor)?,
            "past-end" => wrapped.visit_document(2, &mut visitor)?,
            "document-error" => {
                wrapped.delegate_mut().fail = "doc:0".into();
                wrapped.visit_document(0, &mut visitor)?;
            }
            "visitor-error" => wrapped.visit_document(0, &mut Visitor { fail: true })?,
            "integrity-error" => {
                wrapped.delegate_mut().fail = "integrity".into();
                wrapped.check_integrity()?;
            }
            "close-first" | "close-second" => {
                wrapped.delegate_mut().fail_close = if name == "close-first" { 1 } else { 2 };
                wrapped.close()?;
            }
            "clone" => {
                let mut copy = wrapped.clone_reader();
                wrapped.close()?;
                copy.visit_document(1, &mut visitor)?;
                copy.close()?;
            }
            "merge" => {
                let mut copy = wrapped.get_merge_instance();
                wrapped.close()?;
                copy.visit_document(1, &mut visitor)?;
                copy.close()?;
            }
            "merge-close" => {
                let mut copy = wrapped.get_merge_instance();
                copy.close()?;
                wrapped.visit_document(0, &mut visitor)?;
                wrapped.close()?;
            }
            "merge-clone" => {
                wrapped.clone_reader();
            }
            "merge-merge" => wrapped
                .get_merge_instance()
                .get_merge_instance()
                .visit_document(0, &mut visitor)?,
            "display" => assert_eq!(
                wrapped.to_string(),
                "AssertingStoredFieldsReader(ProbeReader)"
            ),
            "thread" | "merge-thread" | "thread-clone" => {
                let target = if name == "merge-thread" {
                    wrapped.get_merge_instance()
                } else {
                    wrapped
                };
                let clone = name == "thread-clone";
                let result = std::thread::spawn(move || {
                    outcome(|| {
                        if clone {
                            target
                                .clone_reader()
                                .visit_document(0, &mut Visitor { fail: false })
                        } else {
                            target.visit_document(0, &mut Visitor { fail: false })
                        }
                    })
                })
                .join()
                .unwrap();
                if result == "assert" {
                    panic!("expected Java wrong-thread assertion");
                }
                assert_eq!(result, "ok");
            }
            _ => panic!("{name}"),
        }
        Ok(())
    });
    let log = events.lock().unwrap().join(",");
    (result, log)
}
struct ProbeFormat {
    events: Events,
    fail: bool,
}
impl StoredFieldsFormat for ProbeFormat {
    type Writer<'a> = ProbeWriter;
    type Reader<'a> = ProbeReader;
    fn fields_writer<'a>(
        &mut self,
        _: StoredFieldsState<'a>,
    ) -> stored_fields::Result<Self::Writer<'a>> {
        self.events.lock().unwrap().push("open-writer".into());
        if self.fail {
            Err(failure("open failure"))
        } else {
            Ok(ProbeWriter::new(self.events.clone()))
        }
    }
    fn fields_reader<'a>(
        &mut self,
        _: StoredFieldsReadState<'a>,
    ) -> stored_fields::Result<Self::Reader<'a>> {
        self.events.lock().unwrap().push("open-reader".into());
        if self.fail {
            Err(failure("open failure"))
        } else {
            Ok(ProbeReader::new(self.events.clone(), "r"))
        }
    }
}
fn format_trace(name: &str) -> (String, String) {
    let events = Events::default();
    let mut format = AssertingStoredFieldsFormat::new(ProbeFormat {
        events: events.clone(),
        fail: false,
    });
    format.delegate_mut().fail = name.ends_with("error");
    assert!(Arc::ptr_eq(&format.delegate().events, &events));
    let directory = ByteBuffersDirectory::new();
    let status = outcome(|| {
        if name.starts_with("writer") {
            format.fields_writer(StoredFieldsState {
                directory: &directory,
                segment: "_asf",
                id: &[0; 16],
                suffix: "",
                mode: Mode::BestSpeed,
            })?;
        } else {
            let reader = format.fields_reader(StoredFieldsReadState {
                fdt: &[],
                fdx: &[],
                fdm: &[],
                id: &[0; 16],
                suffix: "",
            })?;
            assert_eq!(reader.max_doc(), 2);
            assert_eq!(reader.delegate().name, "r");
        }
        Ok(())
    });
    let log = events.lock().unwrap().join(",");
    (status, log)
}
struct MergeReader(ProbeReader);
impl StoredFieldsSource for MergeReader {
    fn check_integrity(&self) -> stored_fields::Result<()> {
        self.0.check_integrity()
    }
    fn visit_document(
        &self,
        doc: i32,
        visitor: &mut dyn StoredFieldVisitor,
    ) -> stored_fields::Result<()> {
        self.0.call(&format!("doc:{doc}"))?;
        if visitor.needs_field(7)? == VisitStatus::Yes {
            visitor.int_field(7, -17)?;
        }
        Ok(())
    }
}
fn merge_trace(name: &str) -> (String, String) {
    let events = Events::default();
    let a = MergeReader(ProbeReader::new(events.clone(), "a"));
    let b = MergeReader(ProbeReader::new(events.clone(), "b"));
    let maps = if name == "sorted" {
        [[1, -1], [0, 2]]
    } else {
        [[0, -1], [1, 2]]
    };
    let sources = [
        MergeSource {
            reader: &a,
            doc_map: &maps[0],
            field_map: &[],
        },
        MergeSource {
            reader: &b,
            doc_map: &maps[1],
            field_map: &[],
        },
    ];
    let mut writer = AssertingStoredFieldsWriter::new(ProbeWriter::new(events.clone()));
    let status = outcome(|| {
        assert_eq!(
            writer.merge(&MergeState {
                sources: &sources,
                needs_index_sort: name == "sorted",
                check_aborted: &|| Ok(())
            })?,
            3
        );
        Ok(())
    });
    let log = events.lock().unwrap().join(",");
    (status, log)
}
fn fixture(name: &str) -> Vec<u8> {
    std::fs::read(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../fixtures/testframework/data")
            .join(name),
    )
    .unwrap()
}
#[test]
fn java_state_and_reader_lifecycle_traces() {
    let trace = String::from_utf8(fixture("asserting_stored_fields.tsv")).unwrap();
    let mut counts = [0, 0, 0, 0];
    for line in trace.lines() {
        let columns: Vec<_> = line.split('\t').collect();
        match columns[0] {
            "writer" => {
                let (actual, events, input) = writer_trace(columns[1]);
                assert_eq!(
                    (actual.as_str(), events.as_str(), input),
                    (columns[2], columns[3], columns[4].parse().unwrap()),
                    "{}",
                    columns[1]
                );
                counts[0] += 1;
            }
            "reader" => {
                let (actual, events) = reader_trace(columns[1]);
                assert_eq!(
                    (actual.as_str(), events.as_str()),
                    (columns[2], columns[3]),
                    "{}",
                    columns[1]
                );
                counts[1] += 1;
            }
            "format" => {
                let (actual, events) = format_trace(columns[1]);
                assert_eq!(
                    (actual.as_str(), events.as_str()),
                    (columns[2], columns[3]),
                    "{}",
                    columns[1]
                );
                counts[2] += 1;
            }
            "merge" => {
                let (actual, events) = merge_trace(columns[1]);
                assert_eq!(
                    (actual.as_str(), events.as_str()),
                    (columns[2], columns[3]),
                    "{}",
                    columns[1]
                );
                counts[3] += 1;
            }
            _ => panic!("{line}"),
        }
    }
    assert_eq!(counts, [32, 17, 4, 2]);
}
const ID: [u8; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
fn values() -> Vec<FieldValue> {
    vec![
        FieldValue::Int(-17),
        FieldValue::Long(i64::MIN),
        FieldValue::Float(-0.0),
        FieldValue::Double(f64::from_bits(0x7ff8000000000042)),
        FieldValue::Binary(vec![0, 1, 255]),
        FieldValue::Binary(vec![9, 8]),
        FieldValue::String("hello λ".into()),
    ]
}
struct Collect {
    values: Vec<FieldValue>,
    status: VisitStatus,
}
impl StoredFieldVisitor for Collect {
    fn needs_field(&mut self, field: i32) -> stored_fields::Result<VisitStatus> {
        assert_eq!(field, 7);
        Ok(self.status)
    }
    fn int_field(&mut self, field: i32, value: i32) -> stored_fields::Result<()> {
        assert_eq!(field, 7);
        self.values.push(FieldValue::Int(value));
        Ok(())
    }
    fn long_field(&mut self, field: i32, value: i64) -> stored_fields::Result<()> {
        assert_eq!(field, 7);
        self.values.push(FieldValue::Long(value));
        Ok(())
    }
    fn float_field(&mut self, field: i32, value: f32) -> stored_fields::Result<()> {
        assert_eq!(field, 7);
        self.values.push(FieldValue::Float(value));
        Ok(())
    }
    fn double_field(&mut self, field: i32, value: f64) -> stored_fields::Result<()> {
        assert_eq!(field, 7);
        self.values.push(FieldValue::Double(value));
        Ok(())
    }
    fn binary_field(&mut self, field: i32, value: &[u8]) -> stored_fields::Result<()> {
        assert_eq!(field, 7);
        self.values.push(FieldValue::Binary(value.into()));
        Ok(())
    }
    fn string_field(&mut self, field: i32, value: &str) -> stored_fields::Result<()> {
        assert_eq!(field, 7);
        self.values.push(FieldValue::String(value.into()));
        Ok(())
    }
}
fn assert_values(reader: &impl StoredFieldsSource) {
    for status in [VisitStatus::Yes, VisitStatus::No, VisitStatus::Stop] {
        let mut visitor = Collect {
            values: Vec::new(),
            status,
        };
        reader.visit_document(0, &mut visitor).unwrap();
        if status != VisitStatus::Yes {
            assert!(visitor.values.is_empty());
            continue;
        }
        let expected = values();
        assert_eq!(visitor.values.len(), expected.len());
        for (actual, expected) in visitor.values.iter().zip(expected) {
            match (actual, expected) {
                (FieldValue::Float(a), FieldValue::Float(b)) => {
                    assert_eq!(a.to_bits(), b.to_bits())
                }
                (FieldValue::Double(a), FieldValue::Double(_)) => {
                    assert_eq!(a.to_bits(), 0x7ff8000000000000)
                }
                (a, b) => assert_eq!(a, &b),
            }
        }
    }
    let mut visitor = Collect {
        values: Vec::new(),
        status: VisitStatus::Yes,
    };
    reader.visit_document(1, &mut visitor).unwrap();
    assert!(visitor.values.is_empty());
}
#[test]
fn real_java_triples_and_independent_native_reader_handles() {
    for (stem, mode, empty) in [
        ("speed", Mode::BestSpeed, false),
        ("compression", Mode::BestCompression, false),
        ("empty", Mode::BestSpeed, true),
    ] {
        let directory = ByteBuffersDirectory::new();
        let mut format = AssertingStoredFieldsFormat::new(Lucene90StoredFieldsFormat);
        let mut writer = format
            .fields_writer(StoredFieldsState {
                directory: &directory,
                segment: "_asf",
                id: &ID,
                suffix: "",
                mode,
            })
            .unwrap();
        if !empty {
            writer.start_document().unwrap();
            for (index, value) in values().iter().enumerate() {
                if index == 5 {
                    writer
                        .write_field_data_input(7, &mut SliceInput::new(&[9, 8]), 2)
                        .unwrap();
                } else {
                    writer.write_field(7, value).unwrap();
                }
            }
            writer.finish_document().unwrap();
            writer.start_document().unwrap();
            writer.finish_document().unwrap();
        }
        writer.finish(if empty { 0 } else { 2 }).unwrap();
        writer.close().unwrap();
        writer.close().unwrap();
        for extension in ["fdt", "fdx", "fdm"] {
            assert_eq!(
                directory
                    .open(&format!("_asf.{extension}"))
                    .unwrap()
                    .as_ref(),
                fixture(&format!("asserting_stored_fields_{stem}.{extension}")),
                "{stem}.{extension}"
            );
        }
        let fdt = fixture(&format!("asserting_stored_fields_{stem}.fdt"));
        let fdx = fixture(&format!("asserting_stored_fields_{stem}.fdx"));
        let fdm = fixture(&format!("asserting_stored_fields_{stem}.fdm"));
        let mut reader = format
            .fields_reader(StoredFieldsReadState {
                fdt: &fdt,
                fdx: &fdx,
                fdm: &fdm,
                id: &ID,
                suffix: "",
            })
            .unwrap();
        reader.check_integrity().unwrap();
        assert_eq!(reader.max_doc(), if empty { 0 } else { 2 });
        assert!(reader.to_string().contains("Lucene90StoredFieldsReader"));
        let mut clone = reader.clone_reader();
        let mut merging = reader.get_merge_instance();
        reader.close().unwrap();
        assert!(reader.check_integrity().is_err());
        if !empty {
            assert_values(&clone);
            assert_values(&merging);
        }
        clone.check_integrity().unwrap();
        merging.check_integrity().unwrap();
        clone.close().unwrap();
        merging.close().unwrap();
        assert!(clone.check_integrity().is_err());
        assert!(merging.check_integrity().is_err());
    }
}
#[test]
fn corrupted_java_files_and_wrong_ids_fail_open_or_integrity() {
    for corruption in [
        "data-header",
        "index-header",
        "meta-truncated",
        "wrong-id",
        "checksum",
    ] {
        let mut fdt = fixture("asserting_stored_fields_speed.fdt");
        let mut fdx = fixture("asserting_stored_fields_speed.fdx");
        let mut fdm = fixture("asserting_stored_fields_speed.fdm");
        match corruption {
            "data-header" => fdt[0] ^= 1,
            "index-header" => fdx[0] ^= 1,
            "meta-truncated" => fdm.truncate(8),
            "checksum" => {
                let last = fdt.len() - 1;
                fdt[last] ^= 1;
            }
            _ => {}
        }
        let id = if corruption == "wrong-id" {
            [99; 16]
        } else {
            ID
        };
        let result = AssertingStoredFieldsFormat::new(Lucene90StoredFieldsFormat).fields_reader(
            StoredFieldsReadState {
                fdt: &fdt,
                fdx: &fdx,
                fdm: &fdm,
                id: &id,
                suffix: "",
            },
        );
        if corruption == "checksum" {
            assert!(result.unwrap().check_integrity().is_err());
        } else {
            assert!(result.is_err(), "{corruption}");
        }
    }
}
