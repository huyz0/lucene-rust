#![cfg(test)]
use std::cell::{Cell, RefCell};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::rc::Rc;
use std::sync::{Arc, Mutex};

use lucene_codecs::term_vectors::{TermVectorField, TermVectorTerm, TermVectorsDocument};
use lucene_codecs::term_vectors_writer::with_writer;
use lucene_store::{ByteBuffersDirectory, Directory};
use lucene_util::java_random::JavaRandom;

use crate::store::mock_directory_wrapper::{FailOn, FailureOp, MockDirectoryWrapper};

use super::*;

#[derive(Default)]
struct Calls {
    events: RefCell<Vec<String>>,
    fail: Cell<Option<&'static str>>,
}

struct Probe(Rc<Calls>);

impl Probe {
    fn call(
        &self,
        name: &'static str,
        details: impl std::fmt::Display,
    ) -> term_vectors::Result<()> {
        self.0.events.borrow_mut().push(format!("{name}{details}"));
        let second_close = self.0.fail.get() == Some("second_close")
            && name == "close"
            && self
                .0
                .events
                .borrow()
                .iter()
                .filter(|event| event.as_str() == "close")
                .count()
                == 2;
        if self.0.fail.get() == Some(name) || second_close {
            Err(term_vectors::Error::DocOutOfRange(7, 3))
        } else {
            Ok(())
        }
    }
}

impl TermVectorsWriteSink for Probe {
    fn start_document(&mut self, fields: i32) -> term_vectors::Result<()> {
        self.call("start_document", format_args!(":{fields}"))
    }
    fn finish_document(&mut self) -> term_vectors::Result<()> {
        self.call("finish_document", "")
    }
    fn start_field(
        &mut self,
        field: &FieldInfo,
        terms: i32,
        p: bool,
        o: bool,
        y: bool,
    ) -> term_vectors::Result<()> {
        self.call(
            "start_field",
            format_args!(":{}:{terms}:{p}:{o}:{y}", field.name),
        )
    }
    fn finish_field(&mut self) -> term_vectors::Result<()> {
        self.call("finish_field", "")
    }
    fn start_term(&mut self, term: &[u8], frequency: i32) -> term_vectors::Result<()> {
        self.call("start_term", format_args!(":{term:?}:{frequency}"))
    }
    fn finish_term(&mut self) -> term_vectors::Result<()> {
        self.call("finish_term", "")
    }
    fn add_position(
        &mut self,
        p: i32,
        s: i32,
        e: i32,
        y: Option<&[u8]>,
    ) -> term_vectors::Result<()> {
        self.call("add_position", format_args!(":{p}:{s}:{e}:{y:?}"))
    }
    fn finish(&mut self, documents: i32) -> term_vectors::Result<()> {
        self.call("finish", format_args!(":{documents}"))
    }
    fn close(&mut self) -> term_vectors::Result<()> {
        self.call("close", "")
    }
    fn ram_bytes_used(&self) -> usize {
        37
    }
}

fn retry(
    calls: &Calls,
    writer: &mut dyn TermVectorsWriteSink,
    name: &'static str,
    operation: impl Fn(&mut dyn TermVectorsWriteSink) -> term_vectors::Result<()>,
) {
    calls.fail.set(Some(name));
    assert!(matches!(
        operation(writer),
        Err(term_vectors::Error::DocOutOfRange(7, 3))
    ));
    calls.fail.set(None);
    operation(writer).unwrap();
}

#[test]
fn every_callback_preserves_arguments_and_only_advances_assertions_after_success() {
    let calls = Rc::new(Calls::default());
    let mut writer = AssertingSink::new(Probe(calls.clone()));
    let field = FieldInfo::new("body", 9);
    retry(&calls, &mut writer, "start_document", |w| {
        w.start_document(1)
    });
    retry(&calls, &mut writer, "start_field", |w| {
        w.start_field(&field, 1, true, true, true)
    });
    retry(&calls, &mut writer, "start_term", |w| {
        w.start_term(&[0, 255], 1)
    });
    retry(&calls, &mut writer, "add_position", |w| {
        w.add_position(2, 3, 5, Some(&[0, 255]))
    });
    retry(&calls, &mut writer, "finish_term", |w| w.finish_term());
    retry(&calls, &mut writer, "finish_field", |w| w.finish_field());
    retry(&calls, &mut writer, "finish_document", |w| {
        w.finish_document()
    });
    retry(&calls, &mut writer, "finish", |w| w.finish(1));
    assert_eq!(writer.ram_bytes_used(), 37);
    writer.close().unwrap();
    let one_pass = [
        "start_document:1",
        "start_field:body:1:true:true:true",
        "start_term:[0, 255]:1",
        "add_position:2:3:5:Some([0, 255])",
        "finish_term",
        "finish_field",
        "finish_document",
        "finish:1",
    ];
    let mut expected = Vec::new();
    for event in one_pass {
        expected.extend([event.to_owned(), event.to_owned()]);
    }
    expected.extend(["close".to_owned(), "close".to_owned()]);
    assert_eq!(*calls.events.borrow(), expected);
}

#[test]
fn close_calls_twice_after_success_and_stops_after_first_error() {
    let calls = Rc::new(Calls::default());
    let mut writer = AssertingSink::new(Probe(calls.clone()));
    calls.fail.set(Some("close"));
    assert!(matches!(
        writer.close(),
        Err(term_vectors::Error::DocOutOfRange(7, 3))
    ));
    assert_eq!(*calls.events.borrow(), ["close"]);
    calls.fail.set(None);
    writer.close().unwrap();
    assert_eq!(*calls.events.borrow(), ["close", "close", "close"]);

    let calls = Rc::new(Calls::default());
    let mut writer = AssertingSink::new(Probe(calls.clone()));
    calls.fail.set(Some("second_close"));
    assert!(matches!(
        writer.close(),
        Err(term_vectors::Error::DocOutOfRange(7, 3))
    ));
    assert_eq!(*calls.events.borrow(), ["close", "close"]);
}

#[test]
fn native_error_variants_and_unwrapped_io_errors_survive_conversion() {
    let error = native_error(framework_error(
        lucene_store::Error::Io(io::Error::new(
            io::ErrorKind::PermissionDenied,
            "native denied",
        ))
        .into(),
    ));
    let term_vectors::Error::Store(lucene_store::Error::Io(error)) = error else {
        panic!("native store error changed variant")
    };
    assert_eq!(error.kind(), io::ErrorKind::PermissionDenied);
    assert_eq!(error.to_string(), "native denied");
    let error = native_error(io::Error::new(
        io::ErrorKind::InvalidInput,
        "framework input",
    ));
    let term_vectors::Error::Store(lucene_store::Error::Io(error)) = error else {
        panic!("framework I/O error lost its type")
    };
    assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
    assert_eq!(error.to_string(), "framework input");
}

#[test]
fn installed_assertions_reject_incomplete_occurrences_and_finish_counts() {
    let calls = Rc::new(Calls::default());
    let mut writer = AssertingSink::new(Probe(calls));
    writer.start_document(1).unwrap();
    writer
        .start_field(&FieldInfo::new("body", 0), 1, false, true, false)
        .unwrap();
    writer.start_term(b"x", 1).unwrap();
    assert!(catch_unwind(AssertUnwindSafe(|| writer.finish_term())).is_err());
    writer.add_position(-1, 0, 1, None).unwrap();
    writer.finish_term().unwrap();
    writer.finish_field().unwrap();
    writer.finish_document().unwrap();
    assert!(catch_unwind(AssertUnwindSafe(|| writer.finish(2))).is_err());
    writer.finish(1).unwrap();
}

#[test]
fn native_close_attempts_every_output_and_preserves_the_first_failure() {
    let directory =
        MockDirectoryWrapper::new(&mut JavaRandom::new(73), ByteBuffersDirectory::new());
    let id = [64; 16];
    let mut writer = AssertingTermVectorsCodec
        .vectors_writer(TermVectorsWriteState {
            directory: &directory,
            segment: "_tv",
            id: &id,
            suffix: "",
            max_doc: 0,
            format: None,
        })
        .unwrap();
    writer.finish(0).unwrap();
    let observed = Arc::new(Mutex::new(Vec::new()));
    let recording = observed.clone();
    let failures = FailOn::new(move |context| {
        if context.op == FailureOp::CloseOutput {
            recording.lock().unwrap().push(context.name.to_owned());
            true
        } else {
            false
        }
    });
    directory.fail_on(failures.clone());
    let error = writer.close().unwrap_err();
    assert!(error.to_string().contains("CloseOutput _tv.tvm"), "{error}");
    assert_eq!(*observed.lock().unwrap(), ["_tv.tvm", "_tv.tvd", "_tv.tvx"]);
    assert_eq!(failures.fired(), 3);
    // Native close has consumed every output even when closing failed.
    // A later explicit close has no output to close again.
    writer.close().unwrap();
    assert_eq!(failures.fired(), 3);
}

#[test]
fn installed_codec_keeps_native_identity_and_binary_output_for_empty_and_populated_docs() {
    assert_eq!(AssertingTermVectorsCodec.name(), "Lucene104");
    assert!(AssertingTermVectorsCodec
        .stored_fields_writer_factory()
        .is_none());
    let codec = AssertingTermVectorsCodec;
    let factory = codec.term_vectors_writer_factory().unwrap();
    let id = [63; 16];
    let field = FieldInfo::new("body", 4);
    let doc = TermVectorsDocument {
        fields: vec![TermVectorField {
            field_number: 4,
            has_positions: true,
            has_offsets: true,
            has_payloads: true,
            terms: vec![TermVectorTerm {
                term: vec![0, 255],
                freq: 2,
                positions: Some(vec![0, 2]),
                start_offsets: Some(vec![0, 3]),
                end_offsets: Some(vec![2, 6]),
                payloads: Some(vec![vec![], vec![7, 0]]),
            }],
        }],
    };
    for populated in [false, true] {
        let directory = ByteBuffersDirectory::new();
        let docs = if populated {
            vec![TermVectorsDocument::default(), doc.clone()]
        } else {
            vec![]
        };
        let count = i32::try_from(docs.len()).unwrap();
        with_writer(
            factory,
            TermVectorsWriteState {
                directory: &directory,
                segment: "_tv",
                id: &id,
                suffix: "",
                max_doc: count,
                format: None,
            },
            |writer| -> term_vectors::Result<()> {
                if populated {
                    writer.start_document(0)?;
                    writer.finish_document()?;
                    writer.start_document(1)?;
                    writer.start_field(&field, 1, true, true, true)?;
                    writer.start_term(&[0, 255], 2)?;
                    writer.add_position(0, 0, 2, Some(&[]))?;
                    writer.add_position(2, 3, 6, Some(&[7, 0]))?;
                    writer.finish_term()?;
                    writer.finish_field()?;
                    writer.finish_document()?;
                }
                writer.finish(count)
            },
        )
        .unwrap();
        let expected = term_vectors::write_best_speed(&docs, &id, "");
        for (extension, bytes) in [
            ("tvd", expected.0),
            ("tvx", expected.1),
            ("tvm", expected.2),
        ] {
            assert_eq!(
                &*directory.open(&format!("_tv.{extension}")).unwrap(),
                bytes
            );
        }
    }
}
