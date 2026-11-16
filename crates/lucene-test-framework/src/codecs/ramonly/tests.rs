#![cfg(test)]
use super::*;
use lucene_codecs::field_infos::{FieldInfos, IndexOptions};
use lucene_codecs::postings_writer::TermPostings;

fn infos(options: IndexOptions, payloads: bool) -> FieldInfos {
    let mut info = FieldInfo::new("body", 0).with_index_options(options);
    info.store_payloads = payloads;
    FieldInfos::new(vec![info]).unwrap()
}

fn write(
    infos: &FieldInfos,
    terms: &[TermPostings],
    max_doc: i32,
    number: i32,
) -> Result<RamPostings> {
    let format = RamOnlyPostingsFormat::default();
    let mut consumer = format.fields_consumer();
    consumer.write_materialized(
        &[FieldPostingsInput {
            field_number: number,
            index_options: infos.fields[0].index_options,
            doc_count: 1,
            has_payloads: infos.fields[0].store_payloads,
            terms,
        }],
        &SegmentWriteState {
            max_doc,
            segment_id: &[0; 16],
            segment_suffix: "",
            field_infos: infos,
        },
    )?;
    Ok(format.fields_producer(consumer.token())?.unwrap())
}

fn term() -> TermPostings {
    TermPostings {
        term: b"a".to_vec(),
        docs: vec![(0, 1)],
        positions: vec![vec![2]],
        ..TermPostings::default()
    }
}

#[test]
fn positional_fields_without_payloads_and_empty_fields() {
    let infos = infos(IndexOptions::DocsAndFreqsAndPositions, false);
    let mut repeated = term();
    repeated.docs[0].1 = 2;
    repeated.positions[0] = vec![2, 4];
    let producer = write(&infos, &[repeated], 1, 0).unwrap();
    let field = producer.terms("body").unwrap();
    assert!(!field.has_payloads());
    assert!(field.has_freqs());
    assert!(field.has_positions());
    assert!(!field.has_offsets());
    let mut e = field.iterator().unwrap();
    assert_eq!(e.next().unwrap(), Some(b"a".as_slice()));
    let mut p = e.postings(PostingsFlags::None).unwrap();
    assert_eq!(p.next_doc().unwrap(), 0);
    assert!(p.payload().is_none());
    assert_eq!(p.next_position().unwrap(), 2);
    assert_eq!(p.freq(), 2);
    assert_eq!(p.next_position().unwrap(), 4);
    assert!(p.payload().is_none());
    assert_eq!(p.next_doc().unwrap(), NO_MORE_DOCS);
    // Unlike Lucene104, RAMOnly's source enum ignores NONE and returns
    // the full supplied positions. The consumer must preserve that data.
    let format = RamOnlyPostingsFormat::default();
    let mut consumer = format.fields_consumer();
    let state = SegmentWriteState {
        max_doc: 1,
        segment_id: &[0; 16],
        segment_suffix: "",
        field_infos: &infos,
    };
    consumer.write(&[(0, field.as_ref())], &state).unwrap();
    let copied = format.fields_producer(consumer.token()).unwrap().unwrap();
    let copied_field = copied.terms("body").unwrap();
    let mut e = copied_field.iterator().unwrap();
    assert_eq!(e.next().unwrap(), Some(b"a".as_slice()));
    let mut p = e.postings(PostingsFlags::None).unwrap();
    assert_eq!(p.next_doc().unwrap(), 0);
    assert_eq!(p.freq(), 2);
    assert_eq!(p.next_position().unwrap(), 2);
    assert_eq!(p.next_position().unwrap(), 4);
    assert!(consumer.write(&[(99, field.as_ref())], &state).is_err());
    assert!(consumer
        .write(
            &[(0, field.as_ref())],
            &SegmentWriteState {
                max_doc: 0,
                ..state
            }
        )
        .is_err());
    let empty = write(&infos, &[], 0, 0).unwrap();
    let field = empty.terms("body").unwrap();
    assert_eq!(
        (
            field.size(),
            field.doc_count(),
            field.sum_doc_freq(),
            field.sum_total_term_freq()
        ),
        (0, 0, 0, 0)
    );
    let mut e = field.iterator().unwrap();
    assert!(e.next().unwrap().is_none());
    assert!(e.try_seek_ceil(b"x").is_err());
    assert_eq!(e.term(), Some(b"x".as_slice()));
}

#[test]
fn malformed_inputs_fail_instead_of_asserting_or_indexing_out_of_bounds() {
    let valid = infos(IndexOptions::DocsAndFreqsAndPositions, false);
    assert!(write(&valid, &[term()], 1, 99).is_err());
    assert!(write(&valid, &[term()], -1, 0).is_err());
    for option in [
        IndexOptions::Docs,
        IndexOptions::DocsAndFreqs,
        IndexOptions::DocsAndCustomFreqs,
        IndexOptions::DocsAndFreqsAndPositionsAndOffsets,
    ] {
        assert!(write(&infos(option, false), &[term()], 1, 0).is_err());
    }
    let mut bad = term();
    bad.term = vec![255];
    assert!(write(&valid, &[bad], 1, 0).is_err());
    let mut bad = term();
    bad.docs.clear();
    assert!(write(&valid, &[bad], 1, 0).is_err());
    for doc in [-1, 1] {
        let mut bad = term();
        bad.docs[0].0 = doc;
        assert!(write(&valid, &[bad], 1, 0).is_err());
    }
    let mut bad = term();
    bad.docs[0].1 = 0;
    assert!(write(&valid, &[bad], 1, 0).is_err());
    let mut bad = term();
    bad.positions[0].clear();
    assert!(write(&valid, &[bad], 1, 0).is_err());
    let payloads = infos(IndexOptions::DocsAndFreqsAndPositions, true);
    assert!(write(&payloads, &[term()], 1, 0).is_err());
    let mut bad = term();
    bad.payload_lengths = vec![2];
    bad.payload_bytes = vec![1];
    assert!(write(&payloads, &[bad], 1, 0).is_err());
    let mut bad = term();
    bad.payload_lengths = vec![0, 0];
    assert!(write(&payloads, &[bad], 1, 0).is_err());
    let mut bad = term();
    bad.payload_lengths = vec![0];
    bad.payload_bytes = vec![1];
    assert!(write(&payloads, &[bad], 1, 0).is_err());
}

#[test]
fn slots_are_atomic_and_live_for_the_formats_lifetime() {
    let format = RamOnlyPostingsFormat::default();
    let tokens = std::thread::scope(|scope| {
        let jobs: Vec<_> = (0..16)
            .map(|_| scope.spawn(|| format.fields_consumer().token().to_vec()))
            .collect();
        jobs.into_iter()
            .map(|j| j.join().unwrap())
            .collect::<Vec<_>>()
    });
    let mut distinct = tokens.clone();
    distinct.sort();
    distinct.dedup();
    assert_eq!(distinct.len(), 16);
    for token in tokens {
        assert_eq!(format.fields_producer(&token).unwrap().unwrap().size(), 0);
    }
    format.next_id.store(i32::MAX, Ordering::Relaxed);
    let high = format.fields_consumer();
    let negative = format.fields_consumer();
    assert_ne!(high.token(), negative.token());
    assert!(format.fields_producer(negative.token()).unwrap().is_some());
}
