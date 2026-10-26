//! Boundary tests for stored-fields factory merges: pruning and mapping failures.

use std::sync::{Arc, Mutex};

use lucene_codecs::{
    codec::Lucene104Codec,
    field_infos::{
        DocValuesSkipIndexType, DocValuesType, FieldInfo, IndexOptions, VectorEncoding,
        VectorSimilarityFunction,
    },
    stored_fields::{self, Document, FieldValue, StoredField},
    stored_fields_writer::{
        StoredFieldsWriteSink, StoredFieldsWriteState, StoredFieldsWriterFactory,
    },
};
use lucene_index::{
    merge::{self, MergeOptions, MergeSource, PrunedField},
    segment_info::LuceneVersion,
    segment_infos,
};
use lucene_store::{ByteBuffersDirectory, DataInput, Directory};
use lucene_test_framework::codecs::asserting_stored_fields_writer::AssertingStoredFieldsCodec;
use lucene_util::fixed_bit_set::FixedBitSet;

const SOURCE_0: [u8; 16] = [1; 16];
const SOURCE_1: [u8; 16] = [2; 16];
const MERGED: [u8; 16] = [9; 16];

#[derive(Debug, Default)]
struct Probe {
    events: Mutex<Vec<String>>,
    fail_close: bool,
    fail_start: bool,
    fail_field: bool,
}
impl Probe {
    fn record(&self, event: impl Into<String>) {
        self.events.lock().unwrap().push(event.into());
    }
}
#[derive(Debug)]
struct Codec(Arc<Probe>);
impl Lucene104Codec for Codec {
    fn stored_fields_writer_factory(&self) -> Option<&dyn StoredFieldsWriterFactory> {
        Some(self)
    }
}
impl StoredFieldsWriterFactory for Codec {
    fn fields_writer<'a>(
        &self,
        state: StoredFieldsWriteState<'a>,
    ) -> stored_fields::Result<Box<dyn StoredFieldsWriteSink + 'a>> {
        self.0.record("open");
        Ok(Box::new(Sink {
            delegate: AssertingStoredFieldsCodec.fields_writer(state)?,
            probe: self.0.clone(),
        }))
    }
}
struct Sink<'a> {
    delegate: Box<dyn StoredFieldsWriteSink + 'a>,
    probe: Arc<Probe>,
}
impl StoredFieldsWriteSink for Sink<'_> {
    fn start_document(&mut self) -> stored_fields::Result<()> {
        self.probe.record("start");
        if self.probe.fail_start {
            return Err(lucene_store::Error::Io(std::io::Error::other(
                "primary merge start failure",
            ))
            .into());
        }
        self.delegate.start_document()
    }
    fn write_field(&mut self, number: i32, value: &FieldValue) -> stored_fields::Result<()> {
        self.probe.record(format!("field:{number}"));
        if self.probe.fail_field {
            return Err(lucene_store::Error::Io(std::io::Error::other(
                "primary merge field failure",
            ))
            .into());
        }
        self.delegate.write_field(number, value)
    }
    fn write_field_data_input(
        &mut self,
        number: i32,
        input: &mut dyn DataInput,
        length: usize,
    ) -> stored_fields::Result<()> {
        self.delegate.write_field_data_input(number, input, length)
    }
    fn finish_document(&mut self) -> stored_fields::Result<()> {
        self.delegate.finish_document()?;
        self.probe.record("end");
        Ok(())
    }
    fn finish(&mut self, num_docs: i32) -> stored_fields::Result<()> {
        self.delegate.finish(num_docs)?;
        self.probe.record(format!("finish:{num_docs}"));
        Ok(())
    }
    fn close(&mut self) -> stored_fields::Result<()> {
        self.probe.record("close");
        self.delegate.close()?;
        if self.probe.fail_close {
            Err(
                lucene_store::Error::Io(std::io::Error::other("secondary merge close failure"))
                    .into(),
            )
        } else {
            Ok(())
        }
    }
}

fn field(name: &str, number: i32) -> FieldInfo {
    FieldInfo {
        name: name.into(),
        number,
        store_term_vectors: false,
        omit_norms: false,
        store_payloads: false,
        soft_deletes_field: false,
        parent_field: false,
        index_options: IndexOptions::None,
        doc_values_type: DocValuesType::None,
        doc_values_skip_index_type: DocValuesSkipIndexType::None,
        doc_values_gen: -1,
        attributes: vec![],
        point_dimension_count: 0,
        point_index_dimension_count: 0,
        point_num_bytes: 0,
        vector_dimension: 0,
        vector_encoding: VectorEncoding::Float32,
        vector_similarity_function: VectorSimilarityFunction::Euclidean,
    }
}
fn document(values: &[(i32, &str)]) -> Document {
    Document {
        fields: values
            .iter()
            .map(|&(field_number, value)| StoredField {
                field_number,
                value: FieldValue::String(value.into()),
            })
            .collect(),
    }
}
fn version() -> LuceneVersion {
    LuceneVersion {
        major: 10,
        minor: 5,
        bugfix: 0,
    }
}
fn values(document: Document) -> Vec<(i32, String)> {
    document
        .fields
        .into_iter()
        .map(|field| {
            let FieldValue::String(value) = field.value else {
                panic!("unexpected field type")
            };
            (field.field_number, value)
        })
        .collect()
}

#[test]
fn pruning_uses_original_source_field_and_document_ids_before_remapping_and_reordering() {
    let fields0 = [field("keeper", 0), field("recovery", 1)];
    let fields1 = [field("recovery", 0), field("keeper", 1)];
    let bytes0 = stored_fields::write_best_speed(
        &[document(&[(0, "base"), (1, "base-recovery")])],
        &SOURCE_0,
        "",
    );
    let docs1 = [
        document(&[(0, "deleted-recovery"), (1, "deleted")]),
        document(&[(0, "drop-one"), (1, "one")]),
        document(&[(0, "retain-two"), (1, "two")]),
        document(&[(0, "drop-three"), (1, "three")]),
    ];
    let bytes1 = stored_fields::write_best_speed(&docs1, &SOURCE_1, "");
    let reader0 = stored_fields::open(&bytes0.0, &bytes0.1, &bytes0.2, &SOURCE_0, "").unwrap();
    let reader1 = stored_fields::open(&bytes1.0, &bytes1.1, &bytes1.2, &SOURCE_1, "").unwrap();
    let mut live = FixedBitSet::new(4);
    live.set(1);
    live.set(2);
    live.set(3);
    let mut keep = FixedBitSet::new(4);
    keep.set(2);
    let mut source1 = MergeSource::stored_only(&fields1, &reader1, Some(&live), Some(version()));
    source1.pruned_field = Some(PrunedField {
        field_number: 0,
        keep: Some(&keep),
    });
    let sources = [
        MergeSource::stored_only(&fields0, &reader0, None, Some(version())),
        source1,
    ];
    let probe = Arc::new(Probe::default());
    let options = MergeOptions {
        codec: Some(Arc::new(Codec(probe.clone()))),
        // Global source order: source0/doc0, source1/docs0..3. The deleted
        // doc participates in the permutation but cannot appear in output.
        reorder: Some(vec![3, 0, 1, 4, 2]),
        ..MergeOptions::default()
    };
    let directory = ByteBuffersDirectory::new();
    let merged = merge::merge_segments_mapped(
        &directory,
        &sources,
        None,
        &options,
        "_candidate",
        MERGED,
        "Lucene104",
        version(),
    )
    .unwrap();
    // SENTINEL: the deleted source document maps to -1, never an output doc.
    assert_eq!(merged.doc_id_maps, [vec![1], vec![-1, 3, 0, 2]]);
    let fdt = directory.open("_candidate.fdt").unwrap();
    let fdx = directory.open("_candidate.fdx").unwrap();
    let fdm = directory.open("_candidate.fdm").unwrap();
    let reader = stored_fields::open(&fdt, &fdx, &fdm, &MERGED, "").unwrap();
    reader.check_integrity().unwrap();
    assert_eq!(reader.max_doc(), 4);
    assert_eq!(
        values(reader.document(0).unwrap()),
        [(1, "retain-two".to_owned()), (0, "two".to_owned())]
    );
    assert_eq!(
        values(reader.document(1).unwrap()),
        [(0, "base".to_owned()), (1, "base-recovery".to_owned())]
    );
    assert_eq!(
        values(reader.document(2).unwrap()),
        [(0, "three".to_owned())]
    );
    assert_eq!(values(reader.document(3).unwrap()), [(0, "one".to_owned())]);
    assert_eq!(
        *probe.events.lock().unwrap(),
        [
            "open", "start", "field:1", "field:0", "end", "start", "field:0", "field:1", "end",
            "start", "field:0", "end", "start", "field:0", "end", "finish:4", "close",
        ]
    );
    assert!(directory.open("_candidate.si").is_ok());
}

#[test]
fn unknown_source_field_closes_the_writer_and_preserves_mapping_error_over_close_failure() {
    for fail_close in [false, true] {
        let fields = [field("known", 3)];
        // Valid native bytes with deliberately inconsistent source FieldInfos.
        // The known field maps first, then the absent number fails mid-document.
        let bytes = stored_fields::write_best_speed(
            &[document(&[(3, "known"), (7, "unregistered")])],
            &SOURCE_0,
            "",
        );
        let reader = stored_fields::open(&bytes.0, &bytes.1, &bytes.2, &SOURCE_0, "").unwrap();
        let source = MergeSource::stored_only(&fields, &reader, None, Some(version()));
        let probe = Arc::new(Probe {
            fail_close,
            ..Probe::default()
        });
        let options = MergeOptions {
            codec: Some(Arc::new(Codec(probe.clone()))),
            ..MergeOptions::default()
        };
        let directory = ByteBuffersDirectory::new();
        let result = merge::merge_segments_mapped(
            &directory,
            &[source],
            None,
            &options,
            "_candidate",
            MERGED,
            "Lucene104",
            version(),
        );
        assert!(matches!(
            result,
            Err(merge::Error::UnknownSourceFieldNumber { field_number: 7 })
        ));
        assert_eq!(
            *probe.events.lock().unwrap(),
            ["open", "start", "field:0", "close"]
        );
        // Returned merge errors close the writer and remove every output owned
        // by this attempt, while preserving the primary mapping error.
        assert!(directory.open("_candidate.fdm").is_err());
        assert!(directory.open("_candidate.fdt").is_err());
        assert!(directory.open("_candidate.fdx").is_err());
        assert!(directory.open("_candidate.fnm").is_err());
        assert!(directory.open("_candidate.si").is_err());
        assert!(directory.list_all().unwrap().is_empty());
        assert!(segment_infos::read_latest(&directory).is_err());
    }
}

#[test]
fn start_document_precedes_semantic_decode_and_its_error_wins_over_invalid_field_type() {
    // infoAndBits stores (field_number << 3) | type: field 0/type 6 is
    // reserved (valid tags are 0..=5). The opaque serialized-document API
    // preserves this byte while the real writer supplies valid compression,
    // chunk metadata and all footer checksums. No checksum repair is needed.
    let mut native =
        stored_fields::StoredFieldsWriter::new(stored_fields::Mode::BestSpeed, &SOURCE_0, "");
    native.add_serialized_document(1, &[6]);
    let bytes = native.finish();
    let reader = stored_fields::open(&bytes.0, &bytes.1, &bytes.2, &SOURCE_0, "").unwrap();
    reader.check_integrity().unwrap();
    let serialized = reader.serialized_document(0).unwrap();
    assert_eq!(serialized.num_stored_fields, 1);
    assert_eq!(serialized.bytes, [6]);
    assert!(matches!(
        reader.document(0),
        Err(stored_fields::Error::UnknownTypeTag(6))
    ));

    let fields = [field("value", 0)];
    for fail_start in [false, true] {
        let source = MergeSource::stored_only(&fields, &reader, None, Some(version()));
        let probe = Arc::new(Probe {
            fail_start,
            fail_close: true,
            ..Probe::default()
        });
        let options = MergeOptions {
            codec: Some(Arc::new(Codec(probe.clone()))),
            ..MergeOptions::default()
        };
        let directory = ByteBuffersDirectory::new();
        let result = merge::merge_segments_mapped(
            &directory,
            &[source],
            None,
            &options,
            "_candidate",
            MERGED,
            "Lucene104",
            version(),
        );
        if fail_start {
            let error = match result {
                Ok(_) => panic!("start failure must abort the merge before semantic decoding"),
                Err(error) => error,
            };
            assert_eq!(error.to_string(), "primary merge start failure");
        } else {
            assert!(matches!(
                result,
                Err(merge::Error::StoredFields(
                    stored_fields::Error::UnknownTypeTag(6)
                ))
            ));
        }
        // Both paths close despite a secondary close error. No field can be
        // delivered, no document can finish and no candidate can publish.
        assert_eq!(*probe.events.lock().unwrap(), ["open", "start", "close"]);
        assert!(directory.open("_candidate.fdm").is_err());
        assert!(directory.open("_candidate.fdt").is_err());
        assert!(directory.open("_candidate.fdx").is_err());
        assert!(directory.open("_candidate.fnm").is_err());
        assert!(directory.open("_candidate.si").is_err());
        assert!(directory.list_all().unwrap().is_empty());
        assert!(segment_infos::read_latest(&directory).is_err());
    }
}

#[test]
fn a_field_callback_precedes_later_decode_failure_and_its_error_keeps_precedence() {
    // The first infoAndBits byte 2 means field 0/int; zigzag byte 84 means
    // 42. The later infoAndBits byte 14 means field 1/type 6 (reserved).
    // Opaque native encoding keeps both field records while supplying real
    // compression, metadata and checksums, so integrity checking succeeds.
    let prefix = stored_fields::parse_document(1, &[2, 84]).unwrap();
    assert_eq!(prefix.fields.len(), 1);
    assert_eq!(prefix.fields[0].field_number, 0);
    assert_eq!(prefix.fields[0].value, FieldValue::Int(42));
    let mut native =
        stored_fields::StoredFieldsWriter::new(stored_fields::Mode::BestSpeed, &SOURCE_0, "");
    native.add_serialized_document(2, &[2, 84, 14]);
    let bytes = native.finish();
    let reader = stored_fields::open(&bytes.0, &bytes.1, &bytes.2, &SOURCE_0, "").unwrap();
    reader.check_integrity().unwrap();
    let serialized = reader.serialized_document(0).unwrap();
    assert_eq!(serialized.num_stored_fields, 2);
    assert_eq!(serialized.bytes, [2, 84, 14]);
    assert!(matches!(
        reader.document(0),
        Err(stored_fields::Error::UnknownTypeTag(6))
    ));

    let fields = [field("first", 0), field("later", 1)];
    for fail_field in [false, true] {
        let source = MergeSource::stored_only(&fields, &reader, None, Some(version()));
        let probe = Arc::new(Probe {
            fail_field,
            fail_close: true,
            ..Probe::default()
        });
        let options = MergeOptions {
            codec: Some(Arc::new(Codec(probe.clone()))),
            ..MergeOptions::default()
        };
        let directory = ByteBuffersDirectory::new();
        let result = merge::merge_segments_mapped(
            &directory,
            &[source],
            None,
            &options,
            "_candidate",
            MERGED,
            "Lucene104",
            version(),
        );
        if fail_field {
            let error = match result {
                Ok(_) => {
                    panic!("field failure must stop decoding before the malformed later field")
                }
                Err(error) => error,
            };
            assert_eq!(error.to_string(), "primary merge field failure");
        } else {
            assert!(matches!(
                result,
                Err(merge::Error::StoredFields(
                    stored_fields::Error::UnknownTypeTag(6)
                ))
            ));
        }
        assert_eq!(
            *probe.events.lock().unwrap(),
            ["open", "start", "field:0", "close"]
        );
        assert!(directory.open("_candidate.fdm").is_err());
        assert!(directory.open("_candidate.fdt").is_err());
        assert!(directory.open("_candidate.fdx").is_err());
        assert!(directory.open("_candidate.fnm").is_err());
        assert!(directory.open("_candidate.si").is_err());
        assert!(directory.list_all().unwrap().is_empty());
        assert!(segment_infos::read_latest(&directory).is_err());
    }
}

#[test]
fn a_mapping_error_precedes_later_decode_failure_and_keeps_its_typed_merge_variant() {
    // Field 0/int 42 is valid, but missing from the supplied source
    // FieldInfos. Field 1/type 6 is malformed and must never be decoded.
    let mut native =
        stored_fields::StoredFieldsWriter::new(stored_fields::Mode::BestSpeed, &SOURCE_0, "");
    native.add_serialized_document(2, &[2, 84, 14]);
    let bytes = native.finish();
    let reader = stored_fields::open(&bytes.0, &bytes.1, &bytes.2, &SOURCE_0, "").unwrap();
    reader.check_integrity().unwrap();
    let serialized = reader.serialized_document(0).unwrap();
    assert_eq!(serialized.num_stored_fields, 2);
    assert_eq!(serialized.bytes, [2, 84, 14]);
    let fields = [field("later", 1)];
    let source = MergeSource::stored_only(&fields, &reader, None, Some(version()));
    let probe = Arc::new(Probe {
        fail_close: true,
        ..Probe::default()
    });
    let options = MergeOptions {
        codec: Some(Arc::new(Codec(probe.clone()))),
        ..MergeOptions::default()
    };
    let directory = ByteBuffersDirectory::new();
    let result = merge::merge_segments_mapped(
        &directory,
        &[source],
        None,
        &options,
        "_candidate",
        MERGED,
        "Lucene104",
        version(),
    );
    // The typed mapping error must survive the stored-fields visitor bridge
    // and secondary close failure. No field can reach the output writer.
    assert!(matches!(
        result,
        Err(merge::Error::UnknownSourceFieldNumber { field_number: 0 })
    ));
    assert_eq!(*probe.events.lock().unwrap(), ["open", "start", "close"]);
    assert!(directory.open("_candidate.fdm").is_err());
    assert!(directory.open("_candidate.fdt").is_err());
    assert!(directory.open("_candidate.fdx").is_err());
    assert!(directory.open("_candidate.fnm").is_err());
    assert!(directory.open("_candidate.si").is_err());
    assert!(directory.list_all().unwrap().is_empty());
    assert!(segment_infos::read_latest(&directory).is_err());
}
