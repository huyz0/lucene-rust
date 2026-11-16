#![cfg(test)]
use super::*;

fn model() -> ModelReadTerms<'static> {
    ModelReadTerms {
        info: Cow::Owned(
            FieldInfo::new("f", 0)
                .with_index_options(IndexOptions::DocsAndFreqsAndPositionsAndOffsets),
        ),
        field: Cow::Owned(byte::FieldPostings {
            name: "f".into(),
            terms: vec![
                byte::Term {
                    bytes: b"a".to_vec(),
                    postings: vec![byte::Posting {
                        doc: 2,
                        freq: 2,
                        positions: vec![
                            Position {
                                position: 1,
                                start_offset: 3,
                                end_offset: 4,
                                payload: vec![9],
                            },
                            Position {
                                position: 2,
                                start_offset: 5,
                                end_offset: 7,
                                payload: vec![],
                            },
                        ],
                    }],
                },
                byte::Term {
                    bytes: b"c".to_vec(),
                    postings: vec![byte::Posting {
                        doc: 4,
                        freq: 1,
                        positions: vec![Position {
                            position: 0,
                            start_offset: 0,
                            end_offset: 1,
                            payload: vec![],
                        }],
                    }],
                },
            ],
        }),
    }
}
#[test]
fn owned_write_enums_report_unpositioned_and_missing_occurrences() {
    let model = model();
    let terms = ModelWriteTerms {
        field: model.field,
        info: model.info,
    };
    assert!(terms.has_positions());
    assert!(terms.has_offsets());
    assert!(!terms.has_payloads());
    let mut e = terms.iterator().unwrap();
    assert!(e.postings(None, POSITIONS).is_err());
    assert_eq!(e.next().unwrap(), Some(b"a".to_vec()));
    let mut docs = e.postings(None, POSITIONS).unwrap();
    assert!(docs.freq().is_err());
    assert!(docs.next_position().is_err());
    assert_eq!(docs.next_doc().unwrap(), 2);
    assert_eq!(docs.freq().unwrap(), 2);
    assert!(docs.start_offset().is_err());
    assert!(docs.end_offset().is_err());
    assert!(docs.payload().is_err());
    assert_eq!(docs.next_position().unwrap(), 1);
    assert_eq!(docs.start_offset().unwrap(), 3);
    assert_eq!(docs.end_offset().unwrap(), 4);
    assert_eq!(docs.payload().unwrap(), Some(vec![9]));
    assert_eq!(docs.next_position().unwrap(), 2);
    assert_eq!(docs.payload().unwrap(), None);
    assert!(docs.next_position().is_err());
    assert_eq!(docs.next_doc().unwrap(), NO_MORE_DOCS);
    assert!(docs.freq().is_err());
    assert_eq!(e.next().unwrap(), Some(b"c".to_vec()));
    assert!(e.next().unwrap().is_none());
    assert!(e.postings(None, 0).is_err());
}
#[test]
fn read_seeks_statistics_and_each_flag_preserve_only_requested_features() {
    let mut model = model();
    model.info.to_mut().store_payloads = true;
    assert_eq!(model.size(), 2);
    assert_eq!(model.sum_total_term_freq(), 3);
    assert_eq!(model.sum_doc_freq(), 2);
    assert_eq!(model.doc_count(), 2);
    assert!(model.has_freqs());
    assert!(model.has_positions());
    assert!(model.has_offsets());
    assert!(model.has_payloads());
    assert_eq!(model.min().unwrap(), Some(b"a".to_vec()));
    assert_eq!(model.max().unwrap(), Some(b"c".to_vec()));
    let mut e = model.iterator().unwrap();
    assert!(e.term().is_none());
    assert!(e.doc_freq().is_err());
    assert!(e.total_term_freq().is_err());
    assert!(e.postings(PostingsFlags::All).is_err());
    assert_eq!(e.try_seek_ceil(b"a").unwrap(), SeekStatus::Found);
    assert_eq!(e.term(), Some(b"a".as_slice()));
    assert_eq!(e.doc_freq().unwrap(), 1);
    assert_eq!(e.total_term_freq().unwrap(), 2);
    for flag in [
        PostingsFlags::None,
        PostingsFlags::Freqs,
        PostingsFlags::Positions,
        PostingsFlags::Offsets,
        PostingsFlags::Payloads,
        PostingsFlags::All,
    ] {
        let mut p = e.postings(flag).unwrap();
        assert_eq!(p.next_doc().unwrap(), 2);
        assert_eq!(p.freq(), if flag == PostingsFlags::None { 1 } else { 2 });
        if flag.wants_positions() {
            assert_eq!(p.next_position().unwrap(), 1);
            assert_eq!(
                p.start_offset(),
                if matches!(flag, PostingsFlags::Offsets | PostingsFlags::All) {
                    3
                } else {
                    -1
                }
            );
            assert_eq!(
                p.end_offset(),
                if matches!(flag, PostingsFlags::Offsets | PostingsFlags::All) {
                    4
                } else {
                    -1
                }
            );
            assert_eq!(
                p.payload(),
                matches!(flag, PostingsFlags::Payloads | PostingsFlags::All)
                    .then_some([9].as_slice())
            );
        }
    }
    assert_eq!(e.try_seek_ceil(b"b").unwrap(), SeekStatus::NotFound);
    assert_eq!(e.term(), Some(b"c".as_slice()));
    assert!(e.next().unwrap().is_none());
    assert_eq!(e.try_seek_ceil(b"z").unwrap(), SeekStatus::End);
    assert!(e.next().unwrap().is_none());
    drop(e);
    model.info.to_mut().index_options = IndexOptions::Docs;
    assert!(!model.has_freqs());
    assert!(!model.has_positions());
    assert!(!model.has_offsets());
    assert_eq!(model.sum_total_term_freq(), 2);
    let mut e = model.iterator().unwrap();
    assert!(e.next().unwrap().is_some());
    assert_eq!(e.total_term_freq().unwrap(), 1);
    let mut p = e.postings(PostingsFlags::All).unwrap();
    assert_eq!(p.next_doc().unwrap(), 2);
    assert_eq!(p.freq(), 1);
    let mut overlap = self::model();
    let posting = overlap.field.terms[0].postings[0].clone();
    overlap.field.to_mut().terms[1].postings.push(posting);
    assert_eq!(overlap.doc_count(), 2);
    let term = overlap.field.terms[0].clone();
    overlap.field.to_mut().terms = (0u16..129)
        .map(|n| {
            let mut term = term.clone();
            term.bytes = n.to_be_bytes().to_vec();
            term
        })
        .collect();
    assert_eq!(overlap.doc_count(), 1);
    overlap.field.to_mut().terms.pop();
    assert_eq!(overlap.doc_count(), 1);
}
struct BrokenFields;
impl ByteFields for BrokenFields {
    fn materialize(&self) -> Result<Cow<'_, [byte::FieldPostings]>> {
        Err(failure("fields failure"))
    }
}
struct Sink {
    closes: usize,
}
impl FieldsConsumer for Sink {
    fn write(&mut self, _: &dyn WriteFields, _: Option<&dyn NormsProducer>) -> Result<()> {
        Ok(())
    }
    fn close(&mut self) -> Result<()> {
        self.closes += 1;
        Ok(())
    }
}
#[test]
fn lazy_native_bridge_forwards_source_errors_and_missing_field_metadata() {
    let infos = FieldInfos::new(vec![]).unwrap();
    let broken = NativeFieldsView::new(&BrokenFields, &infos);
    assert_eq!(
        broken.field_names().unwrap_err().to_string(),
        "fields failure"
    );
    assert!(broken.terms("f").is_err());
    assert!(EncodingFields {
        source: &broken,
        infos: &infos
    }
    .materialize()
    .is_err());
    let fields = byte::OwnedFields(vec![model().field.into_owned()]);
    let view = NativeFieldsView::new(&fields, &infos);
    assert_eq!(view.field_names().unwrap(), vec!["f"]);
    assert!(view.terms("absent").unwrap().is_none());
    assert!(view.terms("f").is_err());
    let mut wrapped = AssertingFieldsConsumer::new(Sink { closes: 0 }, &infos);
    assert_eq!(wrapped.delegate().closes, 0);
    wrapped.delegate_mut().close().unwrap();
    assert_eq!(wrapped.delegate().closes, 1);
}
#[test]
fn native_acquisition_errors_and_producer_raw_accessors_remain_explicit() {
    let infos = FieldInfos::new(vec![]).unwrap();
    let id = [0; 16];
    let state = SegmentWriteState {
        max_doc: -1,
        segment_id: &id,
        segment_suffix: "",
        field_infos: &infos,
    };
    let mut format = AssertingPostingsFormat::new(Lucene104PostingsFormat);
    assert!(format.fields_consumer(state).is_err());
    let files = Files::new();
    assert!(format
        .fields_producer(ReadState {
            files: &files,
            segment: lucene_codecs::segment_state::SegmentReadState {
                max_doc: 0,
                segment_id: &id,
                segment_suffix: "",
                field_infos: &infos
            }
        })
        .is_err());
    let native = byte::PostingsFormat::fields_consumer(
        &mut byte::Lucene104PostingsFormat,
        SegmentWriteState {
            max_doc: 0,
            ..state
        },
    )
    .unwrap();
    let mut native = native;
    ByteConsumer::close(&mut native).unwrap();
    let files = byte::output_files(native.output().unwrap());
    let mut reader = format
        .fields_producer(ReadState {
            files: &files,
            segment: lucene_codecs::segment_state::SegmentReadState {
                max_doc: 0,
                segment_id: &id,
                segment_suffix: "",
                field_infos: &infos,
            },
        })
        .unwrap();
    assert_eq!(reader.delegate().size(), 0);
    let mut alias = reader.get_merge_instance();
    assert!(Rc::ptr_eq(
        reader.delegate().fields.as_ref().unwrap(),
        alias.delegate().fields.as_ref().unwrap()
    ));
    assert!(matches!(
        reader.delegate().materialize().unwrap(),
        Cow::Borrowed(_)
    ));
    reader
        .delegate()
        .shared
        .files
        .borrow_mut()
        .insert("short".into(), vec![]);
    assert!(reader.check_integrity().is_err());
    reader.delegate().shared.files.borrow_mut().remove("short");
    reader
        .delegate()
        .shared
        .files
        .borrow_mut()
        .get_mut("doc")
        .unwrap()[8] ^= 1;
    assert!(reader.check_integrity().is_err());
    reader.delegate_mut().close().unwrap();
    assert!(reader.delegate().fields.is_none());
    assert!(alias.delegate().materialize().is_err());
    assert!(alias.check_integrity().is_err());
    alias.close().unwrap();
    assert!(alias.delegate().fields.is_none());
    assert!(reader.terms("absent").unwrap().is_none());
}

struct TracedSource {
    label: &'static str,
    events: Rc<RefCell<Vec<String>>>,
    fields: byte::OwnedFields,
    calls: Cell<usize>,
    fail_at: Cell<usize>,
    integrity_fail: Cell<bool>,
}
impl ByteFields for TracedSource {
    fn materialize(&self) -> Result<Cow<'_, [byte::FieldPostings]>> {
        let call = self.calls.get() + 1;
        self.calls.set(call);
        self.events
            .borrow_mut()
            .push(format!("{}:{call}", self.label));
        if call == self.fail_at.get() {
            return Err(failure("later source failure"));
        }
        self.fields.materialize()
    }
}
impl byte::FieldsProducer for TracedSource {
    fn check_integrity(&self) -> Result<()> {
        self.events
            .borrow_mut()
            .push(format!("integrity:{}", self.label));
        if self.integrity_fail.get() {
            return Err(failure("integrity failure"));
        }
        Ok(())
    }
    fn close(&mut self) -> Result<()> {
        Ok(())
    }
}
fn traced(label: &'static str, events: Rc<RefCell<Vec<String>>>) -> TracedSource {
    TracedSource {
        label,
        events,
        calls: Cell::new(0),
        fail_at: Cell::new(usize::MAX),
        integrity_fail: Cell::new(false),
        fields: byte::OwnedFields(
            ["f", "g"]
                .into_iter()
                .enumerate()
                .map(|(doc, name)| byte::FieldPostings {
                    name: name.into(),
                    terms: vec![byte::Term {
                        bytes: b"x".to_vec(),
                        postings: vec![byte::Posting {
                            doc: doc as i32,
                            freq: 1,
                            positions: vec![],
                        }],
                    }],
                })
                .collect(),
        ),
    }
}
struct InspectMapped(Rc<RefCell<Vec<String>>>);
impl FieldsConsumer for InspectMapped {
    fn write(&mut self, fields: &dyn WriteFields, _: Option<&dyn NormsProducer>) -> Result<()> {
        self.0.borrow_mut().push("write".into());
        assert_eq!(fields.field_names()?, vec!["f", "g"]);
        let terms = fields.terms("f")?.unwrap();
        let mut e = terms.iterator()?;
        assert_eq!(e.next()?, Some(b"x".to_vec()));
        let mut docs = e.postings(None, FREQS)?;
        assert_eq!(docs.next_doc()?, 0);
        assert_eq!(docs.next_doc()?, 2);
        assert_eq!(docs.next_doc()?, NO_MORE_DOCS);
        assert!(fields.terms("f")?.is_some());
        assert!(fields.terms("absent")?.is_none());
        Ok(())
    }
    fn close(&mut self) -> Result<()> {
        Ok(())
    }
}
#[test]
fn selective_merge_repeats_every_source_callback_after_integrity_and_abort_checks() {
    let events = Rc::new(RefCell::new(Vec::new()));
    let first = traced("a", events.clone());
    let second = traced("b", events.clone());
    let infos = FieldInfos::new(
        ["f", "g"]
            .into_iter()
            .enumerate()
            .map(|(n, name)| {
                FieldInfo::new(name, n as i32).with_index_options(IndexOptions::DocsAndFreqs)
            })
            .collect(),
    )
    .unwrap();
    let sources = [
        byte::MergeSource {
            reader: Some(&first),
            max_doc: 2,
            doc_map: &[0, 1],
        },
        byte::MergeSource {
            reader: None,
            max_doc: 0,
            doc_map: &[],
        },
        byte::MergeSource {
            reader: Some(&second),
            max_doc: 2,
            doc_map: &[2, 3],
        },
    ];
    let check = || {
        events.borrow_mut().push("abort".into());
        Ok(())
    };
    let state = MergeState {
        bytes: byte::MergeState {
            sources: &sources,
            check_aborted: &check,
        },
        field_infos: &infos,
    };
    InspectMapped(events.clone()).merge(&state, None).unwrap();
    assert_eq!(
        *events.borrow(),
        [
            "abort",
            "integrity:a",
            "abort",
            "integrity:b",
            "write",
            "a:1",
            "b:1",
            "a:2",
            "b:2",
            "a:3",
            "b:3",
            "a:4",
            "b:4"
        ]
    );
    second.fail_at.set(5);
    let view = MappedWriteFields(&state);
    assert_eq!(
        view.terms("f").err().unwrap().to_string(),
        "later source failure"
    );
    assert_eq!(
        &events.borrow()[events.borrow().len() - 2..],
        ["a:5", "b:5"]
    );
    assert!(view.terms("f").unwrap().is_some());
    assert_eq!(first.calls.get(), 6);
    assert_eq!(second.calls.get(), 6);
    events.borrow_mut().clear();
    second.integrity_fail.set(true);
    assert_eq!(
        InspectMapped(events.clone())
            .merge(&state, None)
            .err()
            .unwrap()
            .to_string(),
        "integrity failure"
    );
    assert_eq!(
        *events.borrow(),
        ["abort", "integrity:a", "abort", "integrity:b"]
    );
    events.borrow_mut().clear();
    let abort = || {
        events.borrow_mut().push("abort-failure".into());
        Err(failure("abort failure"))
    };
    let state = MergeState {
        bytes: byte::MergeState {
            sources: &sources,
            check_aborted: &abort,
        },
        field_infos: &infos,
    };
    assert_eq!(
        InspectMapped(events.clone())
            .merge(&state, None)
            .err()
            .unwrap()
            .to_string(),
        "abort failure"
    );
    assert_eq!(*events.borrow(), ["abort-failure"]);
}
#[test]
fn selective_merge_checks_unrelated_fields_before_returning_requested_terms() {
    let infos = FieldInfos::new(vec![
        FieldInfo::new("f", 0).with_index_options(IndexOptions::DocsAndFreqs)
    ])
    .unwrap();
    let events = Rc::new(RefCell::new(Vec::new()));
    let mut producer = traced("a", events);
    for (max_doc, map, message) in [
        (2, &[0, -2][..], "invalid mapped document"),
        (2, &[0][..], "document map out of bounds"),
        (1, &[0, 1][..], "source document out of bounds"),
    ] {
        let sources = [byte::MergeSource {
            reader: Some(&producer),
            max_doc,
            doc_map: map,
        }];
        let state = MergeState {
            bytes: byte::MergeState {
                sources: &sources,
                check_aborted: &|| Ok(()),
            },
            field_infos: &infos,
        };
        let view = MappedWriteFields(&state);
        assert_eq!(view.field_names().unwrap_err().to_string(), message);
        assert_eq!(view.terms("f").err().unwrap().to_string(), message);
        assert_eq!(view.terms("absent").err().unwrap().to_string(), message);
    }
    producer.fields.0[1].terms[0].postings[0].doc = -1;
    let sources = [byte::MergeSource {
        reader: Some(&producer),
        max_doc: 2,
        doc_map: &[0, 1],
    }];
    let state = MergeState {
        bytes: byte::MergeState {
            sources: &sources,
            check_aborted: &|| Ok(()),
        },
        field_infos: &infos,
    };
    assert!(MappedWriteFields(&state).terms("f").is_err());
    producer.fields.0[1].terms[0].postings[0].doc = 1;
    let sources = [byte::MergeSource {
        reader: Some(&producer),
        max_doc: 2,
        doc_map: &[-1, -1],
    }];
    let state = MergeState {
        bytes: byte::MergeState {
            sources: &sources,
            check_aborted: &|| Ok(()),
        },
        field_infos: &infos,
    };
    let view = MappedWriteFields(&state);
    assert_eq!(view.field_names().unwrap(), vec!["f", "g"]);
    assert!(view
        .terms("f")
        .unwrap()
        .unwrap()
        .iterator()
        .unwrap()
        .next()
        .unwrap()
        .is_none());
    assert_eq!(
        view.terms("g").err().unwrap().to_string(),
        "unknown postings field"
    );
    assert!(view.terms("absent").unwrap().is_none());
}
