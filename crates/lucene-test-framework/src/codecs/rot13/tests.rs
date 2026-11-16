#![cfg(test)]
use super::*;
use proptest::prelude::*;

proptest! {
    #[test]
    fn cypher_roundtrips_every_byte_sequence(bytes in prop::collection::vec(any::<u8>(), 0..8192)) {
        let encoded = Rot13Cypher::encode(&bytes).unwrap();
        prop_assert_eq!(&encoded[..7], &[0; 7]);
        prop_assert_eq!(Rot13Cypher::decode(&encoded).unwrap(), bytes);
    }
}

#[test]
fn malformed_blocks_and_nonzero_prefixes() {
    for length in 0..7 {
        assert!(matches!(
            Rot13Cypher::decode(&vec![0; length]),
            Err(Error::Store(_))
        ));
    }
    assert!(Rot13Cypher::decode(&[99; 7]).unwrap().is_empty());
    assert_eq!(
        Rot13Cypher::decode(&[99, 8, 7, 6, 5, 4, 3, 0, 255]).unwrap(),
        [243, 242]
    );
    assert_eq!(
        Rot13Cypher::encode(&[0, 127, 243, 255]).unwrap(),
        [0, 0, 0, 0, 0, 0, 0, 13, 140, 0, 12]
    );
}

#[test]
fn flags_optional_files_and_writer_errors() {
    use lucene_codecs::field_infos::{FieldInfo, FieldInfos, IndexOptions};
    use lucene_codecs::postings_writer::TermPostings;

    let normal = UniformSplitRot13PostingsFormat::default();
    assert!(!normal.dictionary_on_heap());
    assert!(UniformSplitRot13PostingsFormat::new(true).dictionary_on_heap());
    assert_eq!(normal.to_string(), "PostingsFormat(name=UniformSplitRot13)");
    let shared = StUniformSplitRot13PostingsFormat;
    assert_eq!(
        shared.to_string(),
        "PostingsFormat(name=STUniformSplitRot13)"
    );
    ENCODING_FLAGS.reset();
    assert_eq!(ENCODING_FLAGS.snapshot(), EncodingState::default());
    // No preceding block: the dictionary's hook must leave blocks false.
    assert_eq!(
        TrackedEncoder { shared: false }
            .encode_dictionary(&[255])
            .unwrap(),
        [0, 0, 0, 0, 0, 0, 0, 12]
    );
    assert_eq!(
        ENCODING_FLAGS.snapshot(),
        EncodingState {
            dictionary_encoded: true,
            ..EncodingState::default()
        }
    );
    assert!(TrackedDecoder.decode(&[]).is_err());
    assert!(ENCODING_FLAGS.snapshot().decoder_called);
    ENCODING_FLAGS.reset();
    assert_eq!(ENCODING_FLAGS.snapshot(), EncodingState::default());

    let infos = FieldInfos::new(vec![FieldInfo::new("body", 0)
        .with_index_options(IndexOptions::DocsAndFreqsAndPositionsAndOffsets)])
    .unwrap();
    let id = [7; 16];
    let state = SegmentWriteState {
        max_doc: 1,
        segment_id: &id,
        segment_suffix: "",
        field_infos: &infos,
    };
    let mut terms = vec![TermPostings {
        term: b"a".to_vec(),
        docs: vec![(0, 1)],
        positions: vec![vec![0]],
        offsets: vec![vec![(0, 1)]],
        ..TermPostings::default()
    }];
    for is_shared in [false, true] {
        let input = FieldPostingsInput {
            field_number: 0,
            index_options: IndexOptions::DocsAndFreqsAndPositionsAndOffsets,
            doc_count: 1,
            has_payloads: false,
            terms: &terms,
        };
        ENCODING_FLAGS.reset();
        let files = if is_shared {
            shared.fields_consumer(&[input], &[], &state)
        } else {
            normal.fields_consumer(&[input], &[], &state)
        }
        .unwrap();
        assert!(files.contains_key("pos"));
        assert!(files.contains_key("pay"));
        assert_eq!(
            ENCODING_FLAGS.snapshot(),
            EncodingState {
                blocks_encoded: true,
                dictionary_encoded: true,
                fields_metadata_encoded: true,
                ..EncodingState::default()
            }
        );
    }
    terms[0].docs[0].1 = 0;
    for is_shared in [false, true] {
        let input = FieldPostingsInput {
            field_number: 0,
            index_options: IndexOptions::DocsAndFreqsAndPositionsAndOffsets,
            doc_count: 1,
            has_payloads: false,
            terms: &terms,
        };
        let result = if is_shared {
            shared.fields_consumer(&[input], &[], &state)
        } else {
            normal.fields_consumer(&[input], &[], &state)
        };
        assert!(matches!(result, Err(Error::Write(_))));
    }
    terms[0].docs[0].1 = 1;
    terms[0].positions.clear();
    terms[0].offsets.clear();
    let infos = FieldInfos::new(vec![
        FieldInfo::new("body", 0).with_index_options(IndexOptions::DocsAndFreqs)
    ])
    .unwrap();
    let state = SegmentWriteState {
        field_infos: &infos,
        ..state
    };
    for is_shared in [false, true] {
        let input = FieldPostingsInput {
            field_number: 0,
            index_options: IndexOptions::DocsAndFreqs,
            doc_count: 1,
            has_payloads: false,
            terms: &terms,
        };
        let files = if is_shared {
            shared.fields_consumer(&[input], &[], &state)
        } else {
            normal.fields_consumer(&[input], &[], &state)
        }
        .unwrap();
        assert!(!files.contains_key("pos"));
        assert!(!files.contains_key("pay"));
    }
}
