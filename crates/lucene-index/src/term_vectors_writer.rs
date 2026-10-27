//! Installed native term-vector callbacks and operation-owned error cleanup.
//!
//! The existing owned document model resolves field metadata before replay
//! and validates occurrence shapes before term callbacks. Unlike Java's lazy
//! Fields/Terms/Postings graph, malformed owned input can therefore fail before
//! an earlier delegate callback. Valid inputs use Java's callback sequence.

use lucene_codecs::field_infos::FieldInfo;
use lucene_codecs::term_vectors::{self, TermVectorsDocument};
use lucene_codecs::term_vectors_writer::{
    self, TermVectorsWriteSink, TermVectorsWriteState, TermVectorsWriterFactory,
};

use crate::created_outputs::CreatedOutputs;

/// Java: `TermVectorsConsumer.flush` and `SegmentMerger.mergeTermVectors`.
/// Publication belongs to the caller. Only successful creations are removed
/// on returned errors; a failed create never owns a preexisting collision.
pub(crate) fn write_installed<E>(
    factory: &dyn TermVectorsWriterFactory,
    state: TermVectorsWriteState<'_>,
    operation: impl FnOnce(&mut dyn TermVectorsWriteSink) -> Result<(), E>,
) -> Result<Vec<String>, E>
where
    E: From<term_vectors::Error>,
{
    let outputs = CreatedOutputs::new(state.directory);
    let result = term_vectors_writer::with_writer(
        factory,
        TermVectorsWriteState {
            directory: &outputs,
            ..state
        },
        operation,
    );
    if let Err(error) = result {
        for name in outputs.files().iter() {
            let _ = state.directory.delete_file(name);
        }
        return Err(error);
    }
    let files = outputs.files().iter().cloned().collect();
    Ok(files)
}

fn invalid(message: impl Into<String>) -> term_vectors::Error {
    lucene_store::Error::Corrupted(message.into()).into()
}

/// Java: `TermVectorsWriter.addAllDocVectors`. The owned document model
/// replaces Java's lazy Fields/Terms/Postings graph; shape errors remain
/// returned errors before indexing occurrence arrays.
pub(crate) fn write_document(
    writer: &mut dyn TermVectorsWriteSink,
    document: &TermVectorsDocument,
    infos: &[FieldInfo],
) -> term_vectors::Result<()> {
    let mut fields = document
        .fields
        .iter()
        .map(|field| {
            let info = infos
                .iter()
                .find(|info| info.number == field.field_number)
                .ok_or_else(|| {
                    invalid(format!("unknown term-vector field {}", field.field_number))
                })?;
            Ok((info, field))
        })
        .collect::<term_vectors::Result<Vec<_>>>()?;
    // Java String.compareTo compares UTF-16 code units, including astral names.
    fields.sort_by(|(a, _), (b, _)| a.name.encode_utf16().cmp(b.name.encode_utf16()));
    let count = i32::try_from(fields.len()).map_err(|_| invalid("too many vector fields"))?;
    writer.start_document(count)?;
    for (info, field) in fields {
        let count =
            i32::try_from(field.terms.len()).map_err(|_| invalid("too many vector terms"))?;
        writer.start_field(
            info,
            count,
            field.has_positions,
            field.has_offsets,
            field.has_payloads,
        )?;
        for term in &field.terms {
            let frequency = usize::try_from(term.freq)
                .ok()
                .filter(|frequency| *frequency > 0)
                .ok_or_else(|| invalid("nonpositive term-vector frequency"))?;
            let positions = term.positions.as_deref();
            let starts = term.start_offsets.as_deref();
            let ends = term.end_offsets.as_deref();
            let payloads = term.payloads.as_deref();
            let matching = |enabled: bool, length: Option<usize>| {
                if enabled {
                    length == Some(frequency)
                } else {
                    length.is_none()
                }
            };
            if !matching(field.has_positions, positions.map(<[i32]>::len))
                || !matching(field.has_offsets, starts.map(<[i32]>::len))
                || !matching(field.has_offsets, ends.map(<[i32]>::len))
                || !matching(field.has_payloads, payloads.map(<[Vec<u8>]>::len))
                || field.has_payloads && !field.has_positions
            {
                return Err(invalid(
                    "term-vector occurrence arrays disagree with flags or frequency",
                ));
            }
            writer.start_term(&term.term, term.freq)?;
            if field.has_positions || field.has_offsets {
                for occurrence in 0..frequency {
                    // SENTINEL: Java passes -1 for each axis absent from this field.
                    // Enabled arrays have exactly frequency entries, checked above.
                    let position = positions.map_or(-1, |values| values[occurrence]);
                    let start = starts.map_or(-1, |values| values[occurrence]);
                    let end = ends.map_or(-1, |values| values[occurrence]);
                    let payload = payloads
                        .and_then(|values| values.get(occurrence))
                        .filter(|value| !value.is_empty())
                        .map(Vec::as_slice);
                    writer.add_position(position, start, end, payload)?;
                }
            }
            writer.finish_term()?;
        }
        writer.finish_field()?;
    }
    writer.finish_document()
}

pub(crate) fn flush(
    factory: &dyn TermVectorsWriterFactory,
    state: TermVectorsWriteState<'_>,
    infos: &[FieldInfo],
    documents: &[TermVectorsDocument],
) -> term_vectors::Result<Vec<String>> {
    write_installed(factory, state, |writer| {
        for document in documents {
            write_document(writer, document, infos)?;
        }
        writer.finish(state.max_doc)
    })
}

#[cfg(test)]
mod tests {
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };

    use lucene_codecs::term_vectors::{TermVectorField, TermVectorTerm};
    use lucene_codecs::term_vectors_writer::NativeTermVectorsWriterFactory;
    use lucene_store::{ByteBuffersDirectory, DataOutput, Directory};

    use super::*;

    #[derive(Debug, PartialEq, Eq)]
    enum Event {
        Document(i32),
        Field(String, i32, i32, bool, bool, bool),
        Term(Vec<u8>, i32),
        Position(i32, i32, i32, Option<Vec<u8>>),
        EndTerm,
        EndField,
        EndDocument,
        Finish(i32),
        Close,
    }

    #[derive(Default)]
    struct Recorder {
        events: Vec<Event>,
        fail_at: Option<&'static str>,
        closes: Option<Arc<AtomicUsize>>,
    }

    impl Recorder {
        fn record(&mut self, name: &'static str, event: Event) -> term_vectors::Result<()> {
            self.events.push(event);
            if self.fail_at == Some(name) {
                Err(term_vectors::Error::DocOutOfRange(23, 4))
            } else {
                Ok(())
            }
        }
    }

    impl TermVectorsWriteSink for Recorder {
        fn start_document(&mut self, fields: i32) -> term_vectors::Result<()> {
            self.record("document", Event::Document(fields))
        }
        fn finish_document(&mut self) -> term_vectors::Result<()> {
            self.record("end_document", Event::EndDocument)
        }
        fn start_field(
            &mut self,
            info: &FieldInfo,
            terms: i32,
            p: bool,
            o: bool,
            y: bool,
        ) -> term_vectors::Result<()> {
            self.record(
                "field",
                Event::Field(info.name.clone(), info.number, terms, p, o, y),
            )
        }
        fn finish_field(&mut self) -> term_vectors::Result<()> {
            self.record("end_field", Event::EndField)
        }
        fn start_term(&mut self, term: &[u8], frequency: i32) -> term_vectors::Result<()> {
            self.record("term", Event::Term(term.to_vec(), frequency))
        }
        fn finish_term(&mut self) -> term_vectors::Result<()> {
            self.record("end_term", Event::EndTerm)
        }
        fn add_position(
            &mut self,
            position: i32,
            start: i32,
            end: i32,
            payload: Option<&[u8]>,
        ) -> term_vectors::Result<()> {
            self.record(
                "position",
                Event::Position(position, start, end, payload.map(<[u8]>::to_vec)),
            )
        }
        fn finish(&mut self, documents: i32) -> term_vectors::Result<()> {
            self.record("finish", Event::Finish(documents))
        }
        fn close(&mut self) -> term_vectors::Result<()> {
            if let Some(closes) = &self.closes {
                closes.fetch_add(1, Ordering::SeqCst);
            }
            self.record("close", Event::Close)
        }
        fn ram_bytes_used(&self) -> usize {
            0
        }
    }

    fn term(bytes: &[u8], frequency: i32) -> TermVectorTerm {
        TermVectorTerm {
            term: bytes.to_vec(),
            freq: frequency,
            positions: None,
            start_offsets: None,
            end_offsets: None,
            payloads: None,
        }
    }

    fn field(number: i32, terms: Vec<TermVectorTerm>) -> TermVectorField {
        TermVectorField {
            field_number: number,
            has_positions: false,
            has_offsets: false,
            has_payloads: false,
            terms,
        }
    }

    #[test]
    fn callbacks_follow_utf16_field_order_keep_binary_terms_and_map_absent_axes() {
        let infos = [
            FieldInfo::new("\u{e000}", 2),
            FieldInfo::new("\u{10000}", 7),
            FieldInfo::new("z", 42),
        ];
        let mut offsets = field(42, vec![term(b"o", 2)]);
        offsets.has_offsets = true;
        offsets.terms[0].start_offsets = Some(vec![0, 7]);
        offsets.terms[0].end_offsets = Some(vec![3, 9]);
        let mut positions = field(7, vec![term(&[0, 255], 3)]);
        positions.has_positions = true;
        positions.has_payloads = true;
        positions.terms[0].positions = Some(vec![0, 3, 8]);
        positions.terms[0].payloads = Some(vec![vec![], vec![0, 255], vec![]]);
        let frequencies = field(2, vec![term(&[0, 255], 4), term(&[128], 1)]);
        let doc = TermVectorsDocument {
            fields: vec![frequencies, positions, offsets],
        };
        let mut writer = Recorder::default();
        write_document(&mut writer, &doc, &infos).unwrap();
        assert_eq!(
            writer.events,
            [
                Event::Document(3),
                Event::Field("z".into(), 42, 1, false, true, false),
                Event::Term(b"o".to_vec(), 2),
                Event::Position(-1, 0, 3, None),
                Event::Position(-1, 7, 9, None),
                Event::EndTerm,
                Event::EndField,
                // UTF-16 orders the surrogate pair before U+E000. UTF-8 byte
                // order and numeric field order both put these the other way.
                Event::Field("\u{10000}".into(), 7, 1, true, false, true),
                Event::Term(vec![0, 255], 3),
                Event::Position(0, -1, -1, None),
                Event::Position(3, -1, -1, Some(vec![0, 255])),
                Event::Position(8, -1, -1, None),
                Event::EndTerm,
                Event::EndField,
                Event::Field("\u{e000}".into(), 2, 2, false, false, false),
                Event::Term(vec![0, 255], 4),
                Event::EndTerm,
                Event::Term(vec![128], 1),
                Event::EndTerm,
                Event::EndField,
                Event::EndDocument,
            ]
        );
    }

    #[test]
    fn empty_documents_and_termless_fields_keep_their_callbacks() {
        let mut writer = Recorder::default();
        write_document(&mut writer, &TermVectorsDocument::default(), &[]).unwrap();
        assert_eq!(writer.events, [Event::Document(0), Event::EndDocument]);
        writer.events.clear();
        let infos = [FieldInfo::new("body", 0)];
        let mut empty = field(0, vec![]);
        empty.has_positions = true;
        empty.has_offsets = true;
        write_document(
            &mut writer,
            &TermVectorsDocument {
                fields: vec![empty],
            },
            &infos,
        )
        .unwrap();
        assert_eq!(
            writer.events,
            [
                Event::Document(1),
                Event::Field("body".into(), 0, 0, true, true, false),
                Event::EndField,
                Event::EndDocument,
            ]
        );
    }

    #[test]
    fn malformed_arrays_and_nonpositive_frequencies_fail_before_term_callbacks() {
        let infos = [FieldInfo::new("body", 0)];
        let mut complete = field(0, vec![term(b"x", 2)]);
        complete.has_positions = true;
        complete.has_offsets = true;
        complete.has_payloads = true;
        complete.terms[0].positions = Some(vec![0, 1]);
        complete.terms[0].start_offsets = Some(vec![0, 2]);
        complete.terms[0].end_offsets = Some(vec![1, 3]);
        complete.terms[0].payloads = Some(vec![vec![], vec![4]]);
        let mut cases = Vec::new();
        for frequency in [0, -1] {
            let mut bad = complete.clone();
            bad.terms[0].freq = frequency;
            cases.push((format!("frequency {frequency}"), bad, "nonpositive"));
        }
        for axis in 0..4 {
            for length in [None, Some(0), Some(1), Some(3)] {
                let mut bad = complete.clone();
                match axis {
                    0 => bad.terms[0].positions = length.map(|n| vec![0; n]),
                    1 => bad.terms[0].start_offsets = length.map(|n| vec![0; n]),
                    2 => bad.terms[0].end_offsets = length.map(|n| vec![0; n]),
                    _ => bad.terms[0].payloads = length.map(|n| vec![vec![]; n]),
                }
                cases.push((
                    format!("axis {axis} length {length:?}"),
                    bad,
                    "occurrence arrays",
                ));
            }
        }
        // Arrays are forbidden as well as required by their axis flags.
        for axis in 0..4 {
            let mut bad = field(0, vec![term(b"x", 2)]);
            match axis {
                0 => bad.terms[0].positions = Some(vec![0, 1]),
                1 => bad.terms[0].start_offsets = Some(vec![0, 2]),
                2 => bad.terms[0].end_offsets = Some(vec![1, 3]),
                _ => bad.terms[0].payloads = Some(vec![vec![], vec![]]),
            }
            cases.push((format!("disabled axis {axis}"), bad, "occurrence arrays"));
        }
        let mut payloads_without_positions = field(0, vec![term(b"x", 2)]);
        payloads_without_positions.has_payloads = true;
        payloads_without_positions.terms[0].payloads = Some(vec![vec![], vec![4]]);
        cases.push((
            "payloads without positions".into(),
            payloads_without_positions,
            "occurrence arrays",
        ));
        for (name, bad, message) in cases {
            let mut writer = Recorder::default();
            let error = write_document(
                &mut writer,
                &TermVectorsDocument { fields: vec![bad] },
                &infos,
            )
            .unwrap_err();
            assert!(
                matches!(error, term_vectors::Error::Store(lucene_store::Error::Corrupted(ref text)) if text.contains(message)),
                "{name}: {error}"
            );
            assert_eq!(
                writer.events.len(),
                2,
                "{name}: callbacks escaped pre-term validation"
            );
            assert!(
                matches!(
                    writer.events.as_slice(),
                    [Event::Document(1), Event::Field(..)]
                ),
                "{name}: {:?}",
                writer.events
            );
        }
    }

    #[test]
    fn unknown_field_fails_before_starting_a_document() {
        let mut writer = Recorder::default();
        let doc = TermVectorsDocument {
            fields: vec![field(17, vec![term(b"x", 1)])],
        };
        let error = write_document(&mut writer, &doc, &[FieldInfo::new("other", 3)]).unwrap_err();
        assert!(
            error.to_string().contains("unknown term-vector field 17"),
            "{error}"
        );
        assert!(writer.events.is_empty());
    }

    #[test]
    fn every_callback_error_keeps_its_native_variant_and_stops_traversal() {
        let infos = [FieldInfo::new("body", 0)];
        let mut input = field(0, vec![term(b"x", 1)]);
        input.has_positions = true;
        input.terms[0].positions = Some(vec![5]);
        let doc = TermVectorsDocument {
            fields: vec![input],
        };
        for (name, expected_calls) in [
            ("document", 1),
            ("field", 2),
            ("term", 3),
            ("position", 4),
            ("end_term", 5),
            ("end_field", 6),
            ("end_document", 7),
        ] {
            let mut writer = Recorder {
                fail_at: Some(name),
                ..Recorder::default()
            };
            assert!(
                matches!(
                    write_document(&mut writer, &doc, &infos),
                    Err(term_vectors::Error::DocOutOfRange(23, 4))
                ),
                "{name}"
            );
            assert_eq!(writer.events.len(), expected_calls, "{name}");
        }
    }

    #[derive(Debug)]
    struct OwnedOutputFactory(Arc<AtomicUsize>);

    impl TermVectorsWriterFactory for OwnedOutputFactory {
        fn vectors_writer<'a>(
            &self,
            state: TermVectorsWriteState<'a>,
        ) -> term_vectors::Result<Box<dyn TermVectorsWriteSink + 'a>> {
            let mut output = state
                .directory
                .create_output(&format!("{}.owned", state.segment))?;
            output.write_bytes(b"owned by this attempt");
            output.close()?;
            Ok(Box::new(Recorder {
                fail_at: Some("close"),
                closes: Some(self.0.clone()),
                ..Recorder::default()
            }))
        }
    }

    #[derive(Debug)]
    enum OperationError {
        Native(term_vectors::Error),
        Identity(Arc<()>),
    }

    impl From<term_vectors::Error> for OperationError {
        fn from(error: term_vectors::Error) -> Self {
            Self::Native(error)
        }
    }

    #[test]
    fn returned_errors_preserve_identity_close_once_and_remove_only_created_outputs() {
        let directory = ByteBuffersDirectory::new();
        let mut unrelated = directory.create_output("_foreign.owned").unwrap();
        unrelated.write_bytes(b"belongs to another operation");
        unrelated.close().unwrap();
        let closes = Arc::new(AtomicUsize::new(0));
        let factory = OwnedOutputFactory(closes.clone());
        let state = TermVectorsWriteState {
            directory: &directory,
            segment: "_candidate",
            id: &[75; 16],
            suffix: "",
            max_doc: 0,
        };
        let identity = Arc::new(());
        let error = write_installed(&factory, state, |_| {
            Err(OperationError::Identity(identity.clone()))
        })
        .unwrap_err();
        let OperationError::Identity(returned) = error else {
            panic!("close replaced the operation error")
        };
        assert!(Arc::ptr_eq(&returned, &identity));
        assert_eq!(closes.load(Ordering::SeqCst), 1);
        assert!(directory.open("_candidate.owned").is_err());
        assert_eq!(
            &*directory.open("_foreign.owned").unwrap(),
            b"belongs to another operation"
        );
        // An otherwise successful operation must return its close error.
        let error = write_installed(&factory, state, |_| Ok::<(), OperationError>(())).unwrap_err();
        assert!(matches!(
            error,
            OperationError::Native(term_vectors::Error::DocOutOfRange(23, 4))
        ));
        assert_eq!(closes.load(Ordering::SeqCst), 2);
        assert_eq!(directory.list_all().unwrap(), ["_foreign.owned"]);
    }

    #[test]
    fn flush_returns_created_native_files_and_keeps_empty_documents_in_binary_output() {
        let directory = ByteBuffersDirectory::new();
        let infos = [FieldInfo::new("body", 6)];
        let mut input = field(6, vec![term(&[0, 255], 2)]);
        input.has_offsets = true;
        input.terms[0].start_offsets = Some(vec![0, 4]);
        input.terms[0].end_offsets = Some(vec![2, 7]);
        let docs = [
            TermVectorsDocument::default(),
            TermVectorsDocument {
                fields: vec![input],
            },
        ];
        let id = [76; 16];
        let files = flush(
            &NativeTermVectorsWriterFactory,
            TermVectorsWriteState {
                directory: &directory,
                segment: "_tv",
                id: &id,
                suffix: "",
                max_doc: 2,
            },
            &infos,
            &docs,
        )
        .unwrap();
        assert_eq!(files, ["_tv.tvd", "_tv.tvm", "_tv.tvx"]);
        let data = directory.open("_tv.tvd").unwrap();
        let index = directory.open("_tv.tvx").unwrap();
        let meta = directory.open("_tv.tvm").unwrap();
        let expected = term_vectors::write_best_speed(&docs, &id, "");
        assert_eq!(&*data, expected.0);
        assert_eq!(&*index, expected.1);
        assert_eq!(&*meta, expected.2);
        let reader = term_vectors::open(&data, &index, &meta, &id, "").unwrap();
        reader.check_integrity().unwrap();
        assert_eq!(reader.document(0).unwrap(), None);
        assert_eq!(reader.document(1).unwrap(), Some(docs[1].clone()));
    }

    #[test]
    fn flush_propagates_document_and_finish_errors_and_allows_same_name_retry() {
        let directory = ByteBuffersDirectory::new();
        let infos = [FieldInfo::new("body", 0)];
        let id = [77; 16];
        let state = TermVectorsWriteState {
            directory: &directory,
            segment: "_retry",
            id: &id,
            suffix: "",
            max_doc: 1,
        };
        let bad = [TermVectorsDocument {
            fields: vec![field(0, vec![term(b"x", 0)])],
        }];
        let error = flush(&NativeTermVectorsWriterFactory, state, &infos, &bad).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("nonpositive term-vector frequency"),
            "{error}"
        );
        assert!(directory.list_all().unwrap().is_empty());
        let error = flush(&NativeTermVectorsWriterFactory, state, &infos, &[]).unwrap_err();
        assert!(
            error.to_string().contains("document count mismatch"),
            "{error}"
        );
        assert!(directory.list_all().unwrap().is_empty());
        let docs = [TermVectorsDocument {
            fields: vec![field(0, vec![term(b"x", 1)])],
        }];
        let files = flush(&NativeTermVectorsWriterFactory, state, &infos, &docs).unwrap();
        assert_eq!(files, ["_retry.tvd", "_retry.tvm", "_retry.tvx"]);
        let data = directory.open("_retry.tvd").unwrap();
        let index = directory.open("_retry.tvx").unwrap();
        let meta = directory.open("_retry.tvm").unwrap();
        let reader = term_vectors::open(&data, &index, &meta, &id, "").unwrap();
        assert_eq!(reader.document(0).unwrap(), Some(docs[0].clone()));
    }
}
