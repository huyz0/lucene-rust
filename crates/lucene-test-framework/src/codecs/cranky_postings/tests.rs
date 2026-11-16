#![cfg(test)]
use super::*;
use lucene_codecs::field_infos::{FieldInfo, FieldInfos};
const ID: [u8; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
fn infos() -> FieldInfos {
    lucene_codecs::field_infos::parse(
        include_bytes!("../../../../../fixtures/testframework/data/cranky_postings.fnm"),
        &ID,
        "",
    )
    .unwrap()
}
fn files() -> Files {
    [
        (
            "doc",
            include_bytes!("../../../../../fixtures/testframework/data/cranky_postings.doc")
                .as_slice(),
        ),
        (
            "psm",
            include_bytes!("../../../../../fixtures/testframework/data/cranky_postings.psm")
                .as_slice(),
        ),
        (
            "pos",
            include_bytes!("../../../../../fixtures/testframework/data/cranky_postings.pos")
                .as_slice(),
        ),
        (
            "pay",
            include_bytes!("../../../../../fixtures/testframework/data/cranky_postings.pay")
                .as_slice(),
        ),
        (
            "tim",
            include_bytes!("../../../../../fixtures/testframework/data/cranky_postings.tim")
                .as_slice(),
        ),
        (
            "tip",
            include_bytes!("../../../../../fixtures/testframework/data/cranky_postings.tip")
                .as_slice(),
        ),
        (
            "tmd",
            include_bytes!("../../../../../fixtures/testframework/data/cranky_postings.tmd")
                .as_slice(),
        ),
    ]
    .into_iter()
    .map(|(extension, bytes)| (extension.into(), bytes.to_vec()))
    .collect()
}
fn read<'a>(files: &'a Files, infos: &'a FieldInfos) -> ReadState<'a> {
    ReadState {
        files,
        segment: SegmentReadState {
            max_doc: 9,
            segment_id: &ID,
            segment_suffix: "s",
            field_infos: infos,
        },
    }
}
fn write(infos: &FieldInfos, max_doc: i32) -> SegmentWriteState<'_> {
    SegmentWriteState {
        max_doc,
        segment_id: &ID,
        segment_suffix: "s",
        field_infos: infos,
    }
}
fn one(name: &str, doc: i32) -> OwnedFields {
    OwnedFields(vec![FieldPostings {
        name: name.into(),
        terms: vec![Term {
            bytes: b"a".to_vec(),
            postings: vec![Posting {
                doc,
                freq: 1,
                positions: vec![],
            }],
        }],
    }])
}
fn reseal(bytes: &mut Vec<u8>) {
    bytes.truncate(bytes.len() - codec_util::FOOTER_LENGTH);
    codec_util::write_footer(bytes);
}
#[test]
fn native_metadata_headers_lengths_missing_files_and_integrity_corruption() {
    let infos = infos();
    let good = files();
    let mut format = Lucene104PostingsFormat;
    for extension in ["psm", "doc", "pos", "pay", "tim", "tip", "tmd"] {
        let mut missing = good.clone();
        missing.remove(extension);
        assert!(
            format.fields_producer(read(&missing, &infos)).is_err(),
            "missing {extension}"
        );
        let mut bad = good.clone();
        bad.get_mut(extension).unwrap()[0] ^= 1;
        assert!(
            format.fields_producer(read(&bad, &infos)).is_err(),
            "header {extension}"
        );
    }
    let at = codec_util::index_header_length("Lucene104PostingsWriterMeta", "s") + 16;
    for length in [-1i64, 1] {
        let mut bad = good.clone();
        let psm = bad.get_mut("psm").unwrap();
        psm[at..at + 8].copy_from_slice(&length.to_le_bytes());
        reseal(psm);
        let error = format.fields_producer(read(&bad, &infos)).err().unwrap();
        if length == -1 {
            assert_eq!(error.to_string(), "negative postings file length");
        }
    }
    let mut bad = good.clone();
    bad.get_mut("psm").unwrap().truncate(at + 2);
    assert!(format.fields_producer(read(&bad, &infos)).is_err());
    let mut wrong = read(&good, &infos);
    let wrong_id = [77; 16];
    wrong.segment.segment_id = &wrong_id;
    assert!(format.fields_producer(wrong).is_err());
    let mut reader = format.fields_producer(read(&good, &infos)).unwrap();
    reader.files.get_mut("doc").unwrap()[0] ^= 1;
    assert!(reader.check_integrity().is_err());
    reader.files.insert("broken".into(), vec![]);
    assert!(reader.check_integrity().is_err());
}
#[test]
fn buffer_lifecycle_empty_docs_only_and_recoverable_validation_errors() {
    let infos = FieldInfos::new(vec![
        FieldInfo::new("f", 0).with_index_options(IndexOptions::Docs)
    ])
    .unwrap();
    let mut format = Lucene104PostingsFormat;
    assert_eq!(
        format
            .fields_consumer(write(&infos, -1))
            .err()
            .unwrap()
            .to_string(),
        "negative maxDoc"
    );
    let mut consumer = format.fields_consumer(write(&infos, 2)).unwrap();
    assert!(consumer.output().is_none());
    assert_eq!(
        consumer
            .write(&one("missing", 0), None)
            .unwrap_err()
            .to_string(),
        "unrecognized postings field"
    );
    for doc in [-1, 2] {
        assert_eq!(
            consumer
                .write(&one("f", doc), None)
                .unwrap_err()
                .to_string(),
            "destination document out of bounds"
        );
    }
    consumer.write(&one("f", 0), None).unwrap();
    assert!(consumer.write(&one("f", 1), None).is_err());
    consumer.close().unwrap();
    consumer.close().unwrap();
    assert!(consumer.write(&one("f", 0), None).is_err());
    let output = output_files(consumer.output().unwrap());
    assert_eq!(output.len(), 5);
    let reader = format.fields_producer(read(&output, &infos)).unwrap();
    assert_eq!(reader.materialize().unwrap()[0].terms[0].postings[0].doc, 0);
    let mut empty = format.fields_consumer(write(&infos, 0)).unwrap();
    empty.close().unwrap();
    let output = output_files(empty.output().unwrap());
    let reader = format.fields_producer(read(&output, &infos)).unwrap();
    assert!(reader.materialize().unwrap().is_empty());
    reader.check_integrity().unwrap();
    let mut empty = format.fields_consumer(write(&infos, 0)).unwrap();
    empty
        .write(
            &OwnedFields(vec![FieldPostings {
                name: "f".into(),
                terms: vec![Term {
                    bytes: b"a".to_vec(),
                    postings: vec![],
                }],
            }]),
            None,
        )
        .unwrap();
    assert!(empty.output().is_some());
    let none = FieldInfos::new(vec![FieldInfo::new("f", 0)]).unwrap();
    let mut consumer = format.fields_consumer(write(&none, 2)).unwrap();
    assert_eq!(
        consumer.write(&one("f", 0), None).unwrap_err().to_string(),
        "field does not index postings"
    );
}
struct Producer {
    fields: OwnedFields,
    events: Rc<RefCell<Vec<&'static str>>>,
    integrity_error: bool,
    fields_error: bool,
}
impl Fields for Producer {
    fn materialize(&self) -> Result<Cow<'_, [FieldPostings]>> {
        self.events.borrow_mut().push("fields");
        if self.fields_error {
            Err(failure("fields failure"))
        } else {
            self.fields.materialize()
        }
    }
}
impl FieldsProducer for Producer {
    fn check_integrity(&self) -> Result<()> {
        self.events.borrow_mut().push("integrity");
        if self.integrity_error {
            Err(failure("integrity failure"))
        } else {
            Ok(())
        }
    }
    fn close(&mut self) -> Result<()> {
        self.events.borrow_mut().push("close");
        Ok(())
    }
}
#[test]
fn superclass_merge_orders_abort_integrity_guard_then_mapping_and_preserves_failures() {
    let infos = FieldInfos::new(vec![
        FieldInfo::new("f", 0).with_index_options(IndexOptions::Docs)
    ])
    .unwrap();
    let events = Rc::new(RefCell::new(vec![]));
    let mut producer = Producer {
        fields: one("f", 0),
        events: events.clone(),
        integrity_error: false,
        fields_error: true,
    };
    let mut format = Lucene104PostingsFormat;
    for fail_abort in [true, false] {
        let sources = [
            MergeSource {
                reader: None,
                max_doc: 0,
                doc_map: &[],
            },
            MergeSource {
                reader: Some(&producer),
                max_doc: 1,
                doc_map: &[0],
            },
        ];
        let check = || {
            events.borrow_mut().push("abort");
            if fail_abort {
                Err(failure("aborted"))
            } else {
                Ok(())
            }
        };
        let mut consumer = format.fields_consumer(write(&infos, 1)).unwrap();
        events.borrow_mut().clear();
        let error = consumer
            .merge(
                &MergeState {
                    sources: &sources,
                    check_aborted: &check,
                },
                None,
            )
            .unwrap_err();
        assert_eq!(
            error.to_string(),
            if fail_abort {
                "aborted"
            } else {
                "fields failure"
            }
        );
        assert_eq!(
            *events.borrow(),
            if fail_abort {
                vec!["abort"]
            } else {
                vec!["abort", "integrity", "fields"]
            }
        );
        assert!(consumer.output().is_none());
    }
    producer.integrity_error = true;
    let sources = [MergeSource {
        reader: Some(&producer),
        max_doc: 1,
        doc_map: &[0],
    }];
    let mut consumer = format.fields_consumer(write(&infos, 1)).unwrap();
    assert_eq!(
        consumer
            .merge(
                &MergeState {
                    sources: &sources,
                    check_aborted: &|| Ok(())
                },
                None
            )
            .unwrap_err()
            .to_string(),
        "integrity failure"
    );
    producer.integrity_error = false;
    let failing_seed = (0..)
        .find(|seed| JavaRandom::new(*seed).next_int_bounded(100) == 0)
        .unwrap();
    let native = format.fields_consumer(write(&infos, 1)).unwrap();
    let mut wrapper =
        CrankyFieldsConsumer::new(native, Rc::new(RefCell::new(JavaRandom::new(failing_seed))));
    let sources = [MergeSource {
        reader: Some(&producer),
        max_doc: 1,
        doc_map: &[0],
    }];
    events.borrow_mut().clear();
    assert_eq!(
        wrapper
            .merge(
                &MergeState {
                    sources: &sources,
                    check_aborted: &|| Ok(())
                },
                None
            )
            .unwrap_err()
            .to_string(),
        FAKE_WRITE_ERROR
    );
    assert_eq!(*events.borrow(), vec!["integrity"]);
    producer.fields_error = false;
    for (doc, max_doc, map, message) in [
        (-1, 1, vec![0], "source document out of bounds"),
        (1, 1, vec![0, 1], "source document out of bounds"),
        (0, 1, vec![], "document map out of bounds"),
        (0, 1, vec![-2], "invalid mapped document"),
    ] {
        producer.fields = one("f", doc);
        let sources = [MergeSource {
            reader: Some(&producer),
            max_doc,
            doc_map: &map,
        }];
        let mut consumer = format.fields_consumer(write(&infos, 2)).unwrap();
        assert_eq!(
            consumer
                .merge(
                    &MergeState {
                        sources: &sources,
                        check_aborted: &|| Ok(())
                    },
                    None
                )
                .unwrap_err()
                .to_string(),
            message
        );
    }
    producer.fields = one("f", 0);
    let sources = [MergeSource {
        reader: Some(&producer),
        max_doc: 1,
        doc_map: &[-1],
    }];
    let mut consumer = format.fields_consumer(write(&infos, 1)).unwrap();
    consumer
        .merge(
            &MergeState {
                sources: &sources,
                check_aborted: &|| Ok(()),
            },
            None,
        )
        .unwrap();
    let result = output_files(consumer.output().unwrap());
    assert!(format
        .fields_producer(read(&result, &infos))
        .unwrap()
        .materialize()
        .unwrap()
        .is_empty());
    producer.close().unwrap();
    assert_eq!(events.borrow().last(), Some(&"close"));
}
struct FailingNorms;
impl NormsProducer for FailingNorms {
    fn norms(&self, _: &FieldInfo) -> Result<Option<&[Option<i64>]>> {
        Err(failure("norms failure"))
    }
}
#[test]
fn norms_callbacks_optional_values_and_full_position_validation() {
    let infos = infos();
    let files = files();
    let mut format = Lucene104PostingsFormat;
    let input = format.fields_producer(read(&files, &infos)).unwrap();
    let mut consumer = format.fields_consumer(write(&infos, 9)).unwrap();
    assert_eq!(
        consumer
            .write(&input, Some(&FailingNorms))
            .unwrap_err()
            .to_string(),
        "norms failure"
    );
    assert!(consumer.output().is_none());
    let mut norms = OwnedNorms::default();
    assert!(norms.norms(&infos.fields[0]).unwrap().is_none());
    norms.0.insert("rich".into(), vec![None; 9]);
    consumer.write(&input, Some(&norms)).unwrap();
    consumer.close().unwrap();
    let mut invalid = input.materialize().unwrap().into_owned();
    let field = invalid
        .iter_mut()
        .find(|field| field.name == "rich")
        .unwrap();
    field.terms[0].postings[0].positions.clear();
    let mut consumer = format.fields_consumer(write(&infos, 9)).unwrap();
    assert!(consumer.write(&OwnedFields(invalid), None).is_err());
    let mut consumer = format.fields_consumer(write(&infos, 9)).unwrap();
    consumer
        .write(&input, Some(&OwnedNorms::default()))
        .unwrap();
    assert!(consumer.output().is_some());
}
