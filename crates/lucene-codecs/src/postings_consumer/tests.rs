#![cfg(test)]
use super::*;
use crate::field_infos::IndexOptions;
use crate::postings_writer::{TermPostings, WriteOptions};
use lucene_store::ByteBuffersDirectory;

const ID: [u8; 16] = [63; 16];

fn state<'a>(directory: &'a dyn Directory, fields: &'a [FieldInfo]) -> PostingsWriteState<'a> {
    PostingsWriteState {
        directory,
        segment: "_p",
        id: &ID,
        suffix: "Lucene104_0",
        max_doc: 2,
        field_infos: fields,
        format: Lucene104PostingsFormat::default(),
    }
}

#[test]
fn native_bytes_and_block_settings_survive_borrowed_sources() {
    let mut field = FieldInfo::new("body", 0);
    field.index_options = IndexOptions::DocsAndFreqsAndPositionsAndOffsets;
    field.store_payloads = true;
    let fields = [field];
    let terms: Vec<_> = (0..12)
        .map(|i| TermPostings {
            term: format!("t{i:02}").into_bytes(),
            docs: vec![(0, 2), (1, 1)],
            positions: vec![vec![0, 0], vec![2]],
            offsets: vec![vec![(0, 1), (0, 2)], vec![(3, 4)]],
            payload_bytes: vec![7, 8, 9],
            payload_lengths: vec![1, 0, 2],
        })
        .collect();
    let references: Vec<&dyn TermSource> = terms.iter().map(|t| t as &dyn TermSource).collect();
    let inputs = [FieldPostingsInput {
        field_number: 0,
        index_options: fields[0].index_options,
        doc_count: 2,
        has_payloads: true,
        terms: references.as_slice(),
    }];
    let values = [2, 4];
    let norms = [FieldNorms {
        field_number: 0,
        values: &values,
    }];
    for suffix in ["", "Lucene104_0"] {
        for format in [
            Lucene104PostingsFormat::default(),
            Lucene104PostingsFormat::new(2, 3).unwrap(),
        ] {
            let directory = ByteBuffersDirectory::new();
            let state = PostingsWriteState {
                suffix,
                format,
                ..state(&directory, &fields)
            };
            let mut consumer = NativePostingsWriter::new(state).unwrap();
            consumer.write(&inputs, &norms).unwrap();
            assert_eq!(
                consumer.write(&inputs, &norms).unwrap_err().to_string(),
                "postings consumer is closed or already written"
            );
            consumer.close().unwrap();
            consumer.close().unwrap();
            let options = WriteOptions {
                has_prox: Some(true),
                has_payloads_or_offsets: Some(true),
                ..format.write_options()
            };
            let expected =
                postings_writer::write_fields_with_options(&inputs, &norms, &options, &ID, suffix)
                    .unwrap();
            let stem = if suffix.is_empty() {
                "_p"
            } else {
                "_p_Lucene104_0"
            };
            for (ext, bytes) in [
                ("doc", expected.doc),
                ("psm", expected.psm),
                ("pos", expected.pos),
                ("pay", expected.pay),
                ("tim", expected.tim),
                ("tip", expected.tip),
                ("tmd", expected.tmd),
            ] {
                assert_eq!(&*directory.open(&format!("{stem}.{ext}")).unwrap(), bytes);
            }
        }
    }
}

#[test]
fn empty_groups_keep_segment_wide_proximity_files_and_merge_finishes() {
    for options in [
        IndexOptions::Docs,
        IndexOptions::DocsAndFreqsAndPositions,
        IndexOptions::DocsAndFreqsAndPositionsAndOffsets,
    ] {
        let directory = ByteBuffersDirectory::new();
        let mut field = FieldInfo::new("empty", 0);
        field.index_options = options;
        let fields = [field];
        let state = state(&directory, &fields);
        let mut consumer = NativePostingsWriter::new(state).unwrap();
        consumer.merge(&[], &[]).unwrap();
        consumer.close().unwrap();
        let names = directory.list_all().unwrap();
        assert_eq!(
            names.len(),
            if options.subsumes_offsets() {
                7
            } else if options.subsumes_positions() {
                6
            } else {
                5
            }
        );
        assert_eq!(
            names.contains(&"_p_Lucene104_0.pos".into()),
            options.subsumes_positions()
        );
        assert_eq!(
            names.contains(&"_p_Lucene104_0.pay".into()),
            options.subsumes_offsets()
        );
        for name in names {
            assert!(directory.file_length(&name).unwrap() > 0);
        }
    }
}

#[test]
fn construction_preserves_collisions_and_failed_writes_are_terminal() {
    let mut field = FieldInfo::new("body", 0);
    field.index_options = IndexOptions::DocsAndFreqsAndPositionsAndOffsets;
    let fields = [field];
    let directory = ByteBuffersDirectory::new();
    for occupied in output_files(state(&directory, &fields)) {
        let directory = ByteBuffersDirectory::new();
        let mut output = directory.create_output(&occupied).unwrap();
        output.write_byte(19);
        output.close().unwrap();
        let error = NativePostingsWriter::new(state(&directory, &fields))
            .err()
            .unwrap();
        assert!(
            matches!(error, postings_writer::Error::Store(lucene_store::Error::Io(ref e)) if e.kind() == io::ErrorKind::AlreadyExists)
        );
        assert_eq!(&*directory.open(&occupied).unwrap(), [19]);
    }
    let directory = ByteBuffersDirectory::new();
    let mut consumer = NativePostingsWriter::new(state(&directory, &fields)).unwrap();
    consumer.close().unwrap();
    assert!(consumer.write(&[], &[]).is_err());
    let directory = ByteBuffersDirectory::new();
    let mut consumer = NativePostingsWriter::new(state(&directory, &fields)).unwrap();
    let invalid = TermPostings {
        term: b"a".to_vec(),
        docs: vec![(0, 0)],
        ..Default::default()
    };
    let terms = [&invalid as &dyn TermSource];
    let inputs = [FieldPostingsInput {
        field_number: 0,
        index_options: fields[0].index_options,
        doc_count: 1,
        has_payloads: false,
        terms: &terms,
    }];
    assert!(consumer.write(&inputs, &[]).is_err());
    assert!(consumer.write(&[], &[]).is_err());
    consumer.close().unwrap();
}
