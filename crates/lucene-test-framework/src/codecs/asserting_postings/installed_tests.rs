#![cfg(test)]
use super::*;
use lucene_codecs::per_field_postings::Lucene104PostingsFormat as NativeFormat;
use lucene_codecs::postings_writer::TermPostings;
use lucene_store::{ByteBuffersDirectory, Directory};

fn term() -> TermPostings {
    TermPostings {
        term: b"term".to_vec(),
        docs: vec![(2, 2), (8, 1)],
        positions: vec![vec![1, 1], vec![6]],
        offsets: vec![vec![(3, 4), (3, 5)], vec![(9, 10)]],
        payload_lengths: vec![1, 0, 2],
        payload_bytes: vec![4, 7, 8],
    }
}
fn info() -> FieldInfo {
    let mut field = FieldInfo::new("f", 7)
        .with_index_options(IndexOptions::DocsAndFreqsAndPositionsAndOffsets)
        .with_omit_norms(true);
    field.store_payloads = true;
    field
}
fn input<'a>(terms: &'a [&'a dyn TermSource]) -> FieldPostingsInput<'a, &'a dyn TermSource> {
    FieldPostingsInput {
        field_number: 7,
        index_options: IndexOptions::DocsAndFreqsAndPositionsAndOffsets,
        doc_count: 2,
        has_payloads: true,
        terms,
    }
}
#[test]
fn term_bytes_borrow_native_storage_and_keep_owned_enumerator_compatibility() {
    let names: &[&[u8]] = &[b"", b"a", b"abbbbbbbbbbbbbbbbb", b"b", b"\x80", b"\xff"];
    let terms: Vec<_> = names
        .iter()
        .map(|name| TermPostings {
            term: name.to_vec(),
            ..term()
        })
        .collect();
    let refs: Vec<&dyn TermSource> = terms.iter().map(|term| term as &dyn TermSource).collect();
    let field = byte::FieldPostings {
        name: "f".into(),
        terms: names
            .iter()
            .map(|name| byte::Term {
                bytes: name.to_vec(),
                postings: Vec::new(),
            })
            .collect(),
    };
    for _ in 0..2 {
        let mut native = BorrowedNativeEnum {
            terms: &refs,
            next: 0,
            current: None,
        };
        let mut owned = ModelWriteEnum {
            field: &field,
            next: 0,
            current: None,
        };
        for (index, expected) in names.iter().enumerate() {
            {
                let bytes = native.next_bytes().unwrap().unwrap();
                assert!(matches!(&bytes, Cow::Borrowed(_)));
                assert_eq!(bytes.as_ref(), *expected);
                assert!(std::ptr::eq(bytes.as_ptr(), terms[index].term.as_ptr()));
            }
            // The mutable enum borrow ended; postings use this same term.
            let mut postings = native.postings(None, FREQS).unwrap();
            assert_eq!(postings.next_doc().unwrap(), 2);
            let bytes = owned.next_bytes().unwrap().unwrap();
            assert!(matches!(&bytes, Cow::Owned(_)));
            assert_eq!(bytes.as_ref(), *expected);
        }
        assert!(native.next_bytes().unwrap().is_none());
        assert!(native.next().unwrap().is_none());
        assert!(native.postings(None, 0).is_err());
        assert!(owned.next_bytes().unwrap().is_none());
        assert!(owned.postings(None, 0).is_err());
    }
    let mut native = BorrowedNativeEnum {
        terms: &refs,
        next: 0,
        current: None,
    };
    assert_eq!(native.next().unwrap(), Some(Vec::new()));
    assert_eq!(native.next_bytes().unwrap().unwrap().as_ref(), b"a");
    assert_eq!(native.next().unwrap(), Some(names[2].to_vec()));
}
#[test]
fn term_order_checks_cover_empty_growing_shrinking_and_unsigned_bytes_in_both_paths() {
    let names: &[&[u8]] = &[b"", b"a", b"abbbbbbbbbbbbbbbbb", b"b", b"\x80", b"\xff"];
    let check = |names: &[&[u8]], borrowed: bool| {
        let infos = FieldInfos::new(vec![info()]).unwrap();
        if borrowed {
            let terms: Vec<_> = names
                .iter()
                .map(|name| TermPostings {
                    term: name.to_vec(),
                    ..term()
                })
                .collect();
            let refs: Vec<&dyn TermSource> =
                terms.iter().map(|term| term as &dyn TermSource).collect();
            validate_native_inputs(&[input(&refs)], &infos.fields).unwrap();
        } else {
            let fields = byte::OwnedFields(vec![byte::FieldPostings {
                name: "f".into(),
                terms: names
                    .iter()
                    .map(|name| byte::Term {
                        bytes: name.to_vec(),
                        postings: Vec::new(),
                    })
                    .collect(),
            }]);
            validate_write_fields(&NativeFieldsView::new(&fields, &infos), &infos.fields).unwrap();
        }
    };
    for borrowed in [false, true] {
        for _ in 0..2 {
            check(names, borrowed);
            check(&[], borrowed);
        }
        for invalid in [
            &[b"".as_slice(), b"".as_slice()][..],
            &[b"b".as_slice(), b"a".as_slice()][..],
        ] {
            assert!(
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| check(invalid, borrowed)))
                    .is_err()
            );
        }
    }
}
#[test]
fn borrowed_views_reenumerate_without_copying_posting_runs() {
    let term = term();
    let refs = [&term as &dyn TermSource];
    let inputs = [input(&refs)];
    let infos = [info()];
    let fields = BorrowedNativeFields {
        inputs: &inputs,
        infos: &infos,
    };
    assert_eq!(fields.field_names().unwrap(), ["f"]);
    assert!(fields.terms("missing").unwrap().is_none());
    let unrelated = [FieldInfo::new("unrelated", 1)];
    assert!(BorrowedNativeFields {
        inputs: &inputs,
        infos: &unrelated
    }
    .terms("unrelated")
    .unwrap()
    .is_none());
    for _ in 0..2 {
        let terms = fields.terms("f").unwrap().unwrap();
        assert!(terms.has_positions());
        assert!(terms.has_offsets());
        assert!(terms.has_payloads());
        let mut iterator = terms.iterator().unwrap();
        assert!(iterator.postings(None, 0).is_err());
        assert_eq!(iterator.next().unwrap(), Some(b"term".to_vec()));
        let mut docs = iterator
            .postings(None, POSITIONS | OFFSETS | PAYLOADS)
            .unwrap();
        assert!(docs.freq().is_err());
        assert_eq!(docs.next_doc().unwrap(), 2);
        assert_eq!(docs.freq().unwrap(), 2);
        assert!(docs.payload().is_err());
        assert_eq!(docs.next_position().unwrap(), 1);
        assert_eq!(
            (docs.start_offset().unwrap(), docs.end_offset().unwrap()),
            (3, 4)
        );
        assert_eq!(docs.payload().unwrap(), Some(vec![4]));
        assert_eq!(docs.next_position().unwrap(), 1);
        assert_eq!(docs.payload().unwrap(), None);
        assert!(docs.next_position().is_err());
        assert_eq!(docs.next_doc().unwrap(), 8);
        assert_eq!(docs.next_position().unwrap(), 6);
        assert_eq!(docs.payload().unwrap(), Some(vec![7, 8]));
        assert_eq!(docs.next_doc().unwrap(), NO_MORE_DOCS);
        assert_eq!(docs.next_doc().unwrap(), NO_MORE_DOCS);
        assert!(docs.freq().is_err());
        assert!(iterator.next().unwrap().is_none());
        assert!(iterator.postings(Some(docs), 0).is_err());
    }
    validate_native_inputs(&inputs, &infos).unwrap();
    let probe = BorrowedNativePostings {
        term: &term,
        flags: 0,
        next: 0,
        current: None,
        position: 0,
        occurrence_base: 0,
    };
    assert!(std::ptr::eq(probe.term.docs().as_ptr(), term.docs.as_ptr()));
    let mut probe = probe;
    assert_eq!(probe.next_doc().unwrap(), 2);
    assert_eq!(probe.freq().unwrap(), 1);
    assert!(probe.next_position().is_err());
    assert_eq!(
        (probe.start_offset().unwrap(), probe.end_offset().unwrap()),
        (-1, -1)
    );
    assert_eq!(probe.payload().unwrap(), None);
}
#[test]
fn borrowed_views_report_invalid_optional_runs_and_post_delegate_errors() {
    let mut term = term();
    term.offsets.clear();
    term.payload_lengths.clear();
    let mut docs = BorrowedNativePostings {
        term: &term,
        flags: POSITIONS | OFFSETS | PAYLOADS,
        next: 0,
        current: None,
        position: 0,
        occurrence_base: 0,
    };
    assert_eq!(docs.next_doc().unwrap(), 2);
    assert!(docs.start_offset().is_err());
    assert_eq!(docs.next_position().unwrap(), 1);
    assert!(docs.start_offset().is_err());
    assert!(docs.payload().is_err());
    let refs = [&term as &dyn TermSource];
    assert!(validate_native_inputs(&[input(&refs)], &[info()]).is_err());
    term.payload_lengths = vec![10, 0, 2];
    let mut docs = BorrowedNativePostings {
        term: &term,
        flags: PAYLOADS,
        next: 0,
        current: Some(0),
        position: 1,
        occurrence_base: 0,
    };
    assert!(docs.payload().is_err());
    term.docs[0].1 = -1;
    let mut docs = BorrowedNativePostings {
        term: &term,
        flags: FREQS,
        next: 1,
        current: Some(0),
        position: 0,
        occurrence_base: 0,
    };
    assert!(docs.next_doc().is_err());
}

#[derive(Default)]
struct Delegate {
    writes: usize,
    merges: usize,
    closes: usize,
    fail_write: bool,
    fail_close: usize,
}
impl PostingsWriteSink for Delegate {
    fn write(
        &mut self,
        _: &[FieldPostingsInput<'_, &dyn TermSource>],
        _: &[FieldNorms<'_>],
    ) -> postings_writer::Result<()> {
        self.writes += 1;
        if self.fail_write {
            Err(lucene_store::Error::Io(std::io::Error::other("delegate write")).into())
        } else {
            Ok(())
        }
    }
    fn merge(
        &mut self,
        _: &[FieldPostingsInput<'_, &dyn TermSource>],
        _: &[FieldNorms<'_>],
    ) -> postings_writer::Result<()> {
        self.merges += 1;
        Ok(())
    }
    fn close(&mut self) -> postings_writer::Result<()> {
        self.closes += 1;
        if self.closes == self.fail_close {
            Err(lucene_store::Error::Io(std::io::Error::other("delegate close")).into())
        } else {
            Ok(())
        }
    }
}
#[test]
fn installed_assertions_follow_delegation_and_double_close_stops_at_first_error() {
    let infos = [info()];
    let mut invalid = term();
    invalid.offsets.clear();
    let refs = [&invalid as &dyn TermSource];
    let inputs = [input(&refs)];
    let mut wrapper = InstalledAssertingConsumer {
        infos: &infos,
        delegate: Delegate::default(),
    };
    assert!(wrapper
        .write(&inputs, &[])
        .unwrap_err()
        .to_string()
        .contains("missing offsets"));
    assert_eq!(wrapper.delegate.writes, 1);
    assert!(wrapper.merge(&inputs, &[]).is_err());
    assert_eq!(wrapper.delegate.merges, 1);
    wrapper.delegate.fail_write = true;
    assert_eq!(
        wrapper.write(&inputs, &[]).unwrap_err().to_string(),
        "delegate write"
    );
    for failed_call in [0, 1, 2] {
        let mut wrapper = InstalledAssertingConsumer {
            infos: &infos,
            delegate: Delegate {
                fail_close: failed_call,
                ..Delegate::default()
            },
        };
        assert_eq!(wrapper.close().is_err(), failed_call != 0);
        assert_eq!(
            wrapper.delegate.closes,
            if failed_call == 1 { 1 } else { 2 }
        );
    }
    let empty: [FieldPostingsInput<'_, &dyn TermSource>; 0] = [];
    validate_native_inputs(&empty, &infos).unwrap();
    // A successful native write can still violate Java's MAX_POSITION assertion.
    let mut term = term();
    term.positions[0][0] = MAX_POSITION + 1;
    let refs = [&term as &dyn TermSource];
    let mut wrapper = InstalledAssertingConsumer {
        infos: &infos,
        delegate: Delegate::default(),
    };
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        wrapper.write(&[input(&refs)], &[])
    }));
    assert!(result.is_err());
    assert_eq!(wrapper.delegate.writes, 1);
}
#[test]
fn installed_native_writer_reopens_through_existing_native_postings_reader() {
    let dir = ByteBuffersDirectory::new();
    let infos = FieldInfos::new(vec![info()]).unwrap();
    let id = [19; 16];
    let state = PostingsWriteState {
        directory: &dir,
        segment: "_0",
        id: &id,
        suffix: "Lucene104_0",
        max_doc: 9,
        field_infos: &infos.fields,
        format: NativeFormat::default(),
    };
    let term = term();
    let refs = [&term as &dyn TermSource];
    let inputs = [input(&refs)];
    assert_eq!(AssertingPostingsCodec.name(), "Lucene104");
    assert!(AssertingPostingsCodec.postings_writer_factory().is_some());
    let mut consumer = AssertingPostingsCodec.fields_consumer(state).unwrap();
    consumer.write(&inputs, &[]).unwrap();
    consumer.close().unwrap();
    consumer.close().unwrap();
    let files: Files = dir
        .list_all()
        .unwrap()
        .iter()
        .map(|name| {
            (
                name.rsplit('.').next().unwrap().to_owned(),
                dir.open(name).unwrap().to_vec(),
            )
        })
        .collect();
    assert_eq!(files.len(), 7);
    let reader = byte::PostingsFormat::fields_producer(
        &mut byte::Lucene104PostingsFormat,
        ReadState {
            files: &files,
            segment: lucene_codecs::segment_state::SegmentReadState {
                max_doc: 9,
                segment_id: &id,
                segment_suffix: "Lucene104_0",
                field_infos: &infos,
            },
        },
    )
    .unwrap();
    byte::FieldsProducer::check_integrity(&reader).unwrap();
    let decoded = reader.materialize().unwrap();
    assert_eq!(decoded[0].name, "f");
    assert_eq!(decoded[0].terms[0].postings[0].positions[1].end_offset, 5);
    assert_eq!(decoded[0].terms[0].postings[1].positions[0].payload, [7, 8]);
    drop(consumer);
    let mut merged = AssertingPostingsCodec
        .fields_consumer(PostingsWriteState {
            segment: "_1",
            ..state
        })
        .unwrap();
    merged.merge(&inputs, &[]).unwrap();
    merged.close().unwrap();
    assert_eq!(dir.list_all().unwrap().len(), 14);
}
