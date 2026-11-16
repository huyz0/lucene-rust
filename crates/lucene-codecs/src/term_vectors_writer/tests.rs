#![cfg(test)]
use std::sync::{Arc, Mutex};

use lucene_store::ByteBuffersDirectory;

use super::*;

const ID: [u8; 16] = [87; 16];
fn state(directory: &dyn Directory) -> TermVectorsWriteState<'_> {
    TermVectorsWriteState {
        directory,
        segment: "_tv",
        id: &ID,
        suffix: "",
        max_doc: 1,
        format: None,
    }
}
fn field() -> FieldInfo {
    FieldInfo::new("f", 3)
}
fn document(writer: &mut dyn TermVectorsWriteSink, p: bool, o: bool, y: bool) {
    writer.start_document(1).unwrap();
    writer.start_field(&field(), 1, p, o, y).unwrap();
    writer.start_term(&[0, 255], 2).unwrap();
    if p || o {
        writer
            .add_position(
                if p { 1 } else { -1 },
                if o { 2 } else { -1 },
                if o { 5 } else { -1 },
                y.then_some(&[7, 8]),
            )
            .unwrap();
        writer
            .add_position(
                if p { 4 } else { -1 },
                if o { 6 } else { -1 },
                if o { 9 } else { -1 },
                None,
            )
            .unwrap();
    }
    writer.finish_term().unwrap();
    writer.finish_field().unwrap();
    writer.finish_document().unwrap();
}

#[test]
fn native_callbacks_preserve_flags_payloads_suffix_and_exact_encoder_bytes() {
    for (p, o, y) in [
        (false, false, false),
        (true, false, false),
        (false, true, false),
        (true, true, false),
        (true, false, true),
        (true, true, true),
    ] {
        for suffix in ["", "s"] {
            let directory = ByteBuffersDirectory::new();
            let mut state = state(&directory);
            state.suffix = suffix;
            let mut writer = NativeTermVectorsWriter::new(state).unwrap();
            let shallow = writer.ram_bytes_used();
            writer.start_document(1).unwrap();
            writer.start_field(&field(), 1, p, o, y).unwrap();
            writer.start_term(&[0, 255], 2).unwrap();
            if p || o {
                writer
                    .add_position(
                        if p { 1 } else { -1 },
                        if o { 2 } else { -1 },
                        if o { 5 } else { -1 },
                        y.then_some(&[7, 8]),
                    )
                    .unwrap();
                writer
                    .add_position(
                        if p { 4 } else { -1 },
                        if o { 6 } else { -1 },
                        if o { 9 } else { -1 },
                        None,
                    )
                    .unwrap();
            }
            assert!(writer.ram_bytes_used() > shallow);
            writer.finish_term().unwrap();
            assert!(writer.ram_bytes_used() > shallow);
            writer.finish_field().unwrap();
            assert!(writer.ram_bytes_used() > shallow);
            let expected_doc = writer.document.as_ref().unwrap().value.clone();
            writer.finish_document().unwrap();
            assert_eq!(writer.ram_bytes_used(), shallow);
            let mut expected = term_vectors::TermVectorsWriter::new(&ID, suffix);
            expected.add_document(&expected_doc);
            let expected = expected.finish();
            writer.finish(1).unwrap();
            assert_eq!(
                writer.finish(1).unwrap_err().to_string(),
                "already finished term-vector writer"
            );
            writer.close().unwrap();
            writer.close().unwrap();
            assert_eq!(
                writer.start_document(0).unwrap_err().to_string(),
                "closed term-vector writer"
            );
            let stem = if suffix.is_empty() { "_tv" } else { "_tv_s" };
            for (extension, bytes) in [
                ("tvd", expected.0),
                ("tvx", expected.1),
                ("tvm", expected.2),
            ] {
                assert_eq!(
                    &*directory.open(&format!("{stem}.{extension}")).unwrap(),
                    bytes
                );
            }
            let tvd = directory.open(&format!("{stem}.tvd")).unwrap();
            let tvx = directory.open(&format!("{stem}.tvx")).unwrap();
            let tvm = directory.open(&format!("{stem}.tvm")).unwrap();
            let reader = term_vectors::open(&tvd, &tvx, &tvm, &ID, suffix).unwrap();
            reader.check_integrity().unwrap();
            assert_eq!(reader.document(0).unwrap(), Some(expected_doc));
        }
    }
}

#[test]
fn empty_documents_empty_fields_and_chunk_boundaries_remain_native() {
    for count in [0, 1, 129] {
        let directory = ByteBuffersDirectory::new();
        let mut state = state(&directory);
        state.max_doc = count;
        let mut writer = NativeTermVectorsWriterFactory
            .vectors_writer(state)
            .unwrap();
        for doc in 0..count {
            writer
                .start_document(if doc % 2 == 0 { 0 } else { 1 })
                .unwrap();
            if doc % 2 != 0 {
                writer.start_field(&field(), 0, true, true, true).unwrap();
                writer.finish_field().unwrap();
            }
            writer.finish_document().unwrap();
        }
        writer.finish(count).unwrap();
        writer.close().unwrap();
        let tvd = directory.open("_tv.tvd").unwrap();
        let tvx = directory.open("_tv.tvx").unwrap();
        let tvm = directory.open("_tv.tvm").unwrap();
        let reader = term_vectors::open(&tvd, &tvx, &tvm, &ID, "").unwrap();
        reader.check_integrity().unwrap();
        for doc in 0..count {
            let actual = reader.document(doc).unwrap();
            if doc % 2 == 0 {
                assert!(actual.is_none());
            } else {
                assert_eq!(actual.unwrap().fields[0].terms.len(), 0);
            }
        }
    }
}

#[test]
fn malformed_callback_counts_scopes_and_flags_return_errors_without_panicking() {
    let directory = ByteBuffersDirectory::new();
    let mut negative = state(&directory);
    negative.max_doc = -1;
    assert!(NativeTermVectorsWriter::new(negative).is_err());
    for suffix in ["é".to_owned(), "x".repeat(256)] {
        let mut bad_suffix = state(&directory);
        bad_suffix.suffix = &suffix;
        assert!(NativeTermVectorsWriter::new(bad_suffix).is_err());
    }
    assert!(directory.list_all().unwrap().is_empty());
    let mut writer = NativeTermVectorsWriter::new(state(&directory)).unwrap();
    assert!(writer.finish_document().is_err());
    assert!(writer
        .start_field(&field(), 0, false, false, false)
        .is_err());
    assert!(writer.finish_field().is_err());
    assert!(writer.start_term(b"x", 1).is_err());
    assert!(writer.finish_term().is_err());
    assert!(writer.add_position(0, 0, 0, None).is_err());
    assert!(writer.finish(1).is_err());
    assert!(writer.start_document(-1).is_err());
    writer.start_document(1).unwrap();
    assert!(writer.start_document(0).is_err());
    assert!(writer.finish(1).is_err());
    assert!(writer.finish_document().is_err());
    assert!(writer
        .start_field(&field(), -1, false, false, false)
        .is_err());
    assert!(writer
        .start_field(&FieldInfo::new("invalid", -1), 0, false, false, false)
        .is_err());
    assert!(writer.start_field(&field(), 0, false, false, true).is_err());
    writer.start_field(&field(), 1, true, true, true).unwrap();
    assert!(writer.start_field(&field(), 1, true, true, true).is_err());
    assert!(writer.finish_document().is_err());
    assert!(writer.finish_field().is_err());
    assert!(writer.start_term(b"x", 0).is_err());
    assert!(writer.start_term(b"x", -1).is_err());
    writer.start_term(b"x", 1).unwrap();
    assert!(writer.start_term(b"x", 1).is_err());
    assert!(writer.finish_field().is_err());
    assert!(writer.finish_term().is_err());
    writer.add_position(0, 0, 1, None).unwrap();
    assert!(writer.add_position(0, 0, 1, None).is_err());
    writer.finish_term().unwrap();
    assert!(writer.start_term(b"x", 1).is_err());
    writer.finish_field().unwrap();
    assert!(writer
        .start_field(&field(), 0, false, false, false)
        .is_err());
    writer.finish_document().unwrap();
    assert!(writer.start_document(0).is_err());
    assert!(writer.finish(0).is_err());
    writer.finish(1).unwrap();
    writer.close().unwrap();
}

#[test]
fn occurrence_integer_values_retain_the_native_java_wrapping_semantics() {
    let directory = ByteBuffersDirectory::new();
    let mut writer = NativeTermVectorsWriter::new(state(&directory)).unwrap();
    writer.start_document(1).unwrap();
    writer.start_field(&field(), 1, true, true, false).unwrap();
    writer.start_term(b"t", 2).unwrap();
    writer
        .add_position(i32::MAX, i32::MIN, i32::MAX, None)
        .unwrap();
    writer
        .add_position(i32::MIN, i32::MAX, i32::MIN, None)
        .unwrap();
    writer.finish_term().unwrap();
    writer.finish_field().unwrap();
    let document = writer.document.as_ref().unwrap().value.clone();
    writer.finish_document().unwrap();
    writer.finish(1).unwrap();
    writer.close().unwrap();
    let mut encoder = term_vectors::TermVectorsWriter::new(&ID, "");
    encoder.add_document(&document);
    let expected = encoder.finish();
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

#[test]
fn payload_flags_and_nonpositional_occurrence_calls_are_checked() {
    for positions in [false, true] {
        let directory = ByteBuffersDirectory::new();
        let mut writer = NativeTermVectorsWriter::new(state(&directory)).unwrap();
        writer.start_document(1).unwrap();
        writer
            .start_field(&field(), 1, positions, false, false)
            .unwrap();
        writer.start_term(b"t", 1).unwrap();
        assert!(writer.add_position(0, -1, -1, Some(&[1])).is_err());
        if positions {
            writer.add_position(0, -1, -1, Some(&[])).unwrap();
        } else {
            assert!(writer.add_position(0, -1, -1, None).is_err());
        }
        writer.finish_term().unwrap();
        writer.finish_field().unwrap();
        writer.finish_document().unwrap();
        writer.finish(1).unwrap();
        writer.close().unwrap();
    }
}

#[test]
fn constructor_closes_successful_acquisitions_and_preserves_colliding_bytes() {
    for occupied in ["_tv.tvm", "_tv.tvd", "_tv.tvx"] {
        let directory = ByteBuffersDirectory::new();
        let mut output = directory.create_output(occupied).unwrap();
        output.write_byte(19);
        output.close().unwrap();
        let error = NativeTermVectorsWriter::new(state(&directory))
            .err()
            .unwrap();
        assert!(
            matches!(error, term_vectors::Error::Store(lucene_store::Error::Io(ref e)) if e.kind() == io::ErrorKind::AlreadyExists)
        );
        assert_eq!(&*directory.open(occupied).unwrap(), [19]);
        for name in directory.list_all().unwrap() {
            if name != occupied {
                assert_eq!(directory.file_length(&name).unwrap(), 0);
            }
        }
    }
}

#[derive(Debug, Default)]
struct Factory {
    events: Arc<Mutex<Vec<&'static str>>>,
    fail_open: bool,
    fail_close: bool,
}
struct Sink {
    events: Arc<Mutex<Vec<&'static str>>>,
    fail_close: bool,
}
fn failure(message: &str) -> term_vectors::Error {
    lucene_store::Error::Io(io::Error::other(message.to_owned())).into()
}
impl TermVectorsWriterFactory for Factory {
    fn vectors_writer<'a>(
        &self,
        state: TermVectorsWriteState<'a>,
    ) -> term_vectors::Result<Box<dyn TermVectorsWriteSink + 'a>> {
        assert_eq!(state.segment, "_tv");
        assert_eq!(state.id, &ID);
        assert_eq!(state.suffix, "");
        assert_eq!(state.max_doc, 1);
        self.events.lock().unwrap().push("open");
        if self.fail_open {
            return Err(failure("open"));
        }
        Ok(Box::new(Sink {
            events: self.events.clone(),
            fail_close: self.fail_close,
        }))
    }
}
impl TermVectorsWriteSink for Sink {
    fn start_document(&mut self, _: i32) -> term_vectors::Result<()> {
        self.events.lock().unwrap().push("start");
        Ok(())
    }
    fn finish_document(&mut self) -> term_vectors::Result<()> {
        Ok(())
    }
    fn start_field(
        &mut self,
        _: &FieldInfo,
        _: i32,
        _: bool,
        _: bool,
        _: bool,
    ) -> term_vectors::Result<()> {
        Ok(())
    }
    fn finish_field(&mut self) -> term_vectors::Result<()> {
        Ok(())
    }
    fn start_term(&mut self, _: &[u8], _: i32) -> term_vectors::Result<()> {
        Ok(())
    }
    fn finish_term(&mut self) -> term_vectors::Result<()> {
        Ok(())
    }
    fn add_position(
        &mut self,
        _: i32,
        _: i32,
        _: i32,
        _: Option<&[u8]>,
    ) -> term_vectors::Result<()> {
        Ok(())
    }
    fn finish(&mut self, _: i32) -> term_vectors::Result<()> {
        Ok(())
    }
    fn close(&mut self) -> term_vectors::Result<()> {
        self.events.lock().unwrap().push("close");
        if self.fail_close {
            Err(failure("close"))
        } else {
            Ok(())
        }
    }
    fn ram_bytes_used(&self) -> usize {
        0
    }
}
#[test]
fn helper_closes_after_success_or_error_and_preserves_first_error() {
    let directory = ByteBuffersDirectory::new();
    for fail_close in [false, true] {
        let factory = Factory {
            fail_close,
            ..Factory::default()
        };
        let result: term_vectors::Result<_> = with_writer(&factory, state(&directory), |sink| {
            sink.start_document(0)?;
            Ok(19)
        });
        if fail_close {
            assert_eq!(result.unwrap_err().to_string(), "close");
        } else {
            assert_eq!(result.unwrap(), 19);
        }
        assert_eq!(*factory.events.lock().unwrap(), ["open", "start", "close"]);
        factory.events.lock().unwrap().clear();
        let result: term_vectors::Result<()> =
            with_writer(&factory, state(&directory), |_| Err(failure("primary")));
        assert_eq!(result.unwrap_err().to_string(), "primary");
        assert_eq!(*factory.events.lock().unwrap(), ["open", "close"]);
    }
    let factory = Factory {
        fail_open: true,
        ..Factory::default()
    };
    let result: term_vectors::Result<()> =
        with_writer(&factory, state(&directory), |_| panic!("not acquired"));
    assert_eq!(result.unwrap_err().to_string(), "open");
    assert_eq!(*factory.events.lock().unwrap(), ["open"]);
}

#[test]
fn unfinished_close_drops_callback_buffers_without_finishing_a_segment() {
    let directory = ByteBuffersDirectory::new();
    let mut writer = NativeTermVectorsWriter::new(state(&directory)).unwrap();
    document(&mut writer, true, true, true);
    writer.close().unwrap();
    writer.close().unwrap();
    for name in directory.list_all().unwrap() {
        assert_eq!(directory.file_length(&name).unwrap(), 0);
    }
}
