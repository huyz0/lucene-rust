//! Native Lucene binary snapshots for installed term-vector writers.
//! The fixture records raw input tokens separately from native-reader output;
//! replay compares every live vector, including flags and occurrence bytes.

use std::{collections::BTreeMap, sync::Arc};

use lucene_codecs::{
    field_infos::{DocValuesSkipIndexType, DocValuesType, FieldInfo, IndexOptions},
    stored_fields::{self, FieldValue, StoredField},
};
use lucene_index::{
    buffered_updates::Term,
    document::{
        Document, Field, FieldToken, FieldTokens, FieldType, NumericDocValuesField, Store,
        StringField,
    },
    index_writer::{IndexWriter, DISABLE_AUTO_FLUSH_MB},
    merge_policy::log::LogDocMergePolicy,
    segment_info::{IndexSortField, LuceneVersion},
};
use lucene_search::{
    directory_reader::DirectoryReader,
    reader::{IndexReader, LeafReader},
};
use lucene_store::{ByteBuffersDirectory, Directory, FsDirectory};
use lucene_test_framework::codecs::asserting_term_vectors_writer::AssertingTermVectorsCodec;

#[derive(Clone)]
struct Definition {
    name: String,
    positions: bool,
    offsets: bool,
    payloads: bool,
}

#[derive(Default)]
struct Input {
    rank: i64,
    kind: String,
    fields: BTreeMap<String, Vec<FieldToken>>,
}

fn unhex(text: &str) -> Vec<u8> {
    if text == "-" {
        return Vec::new();
    }
    assert!(text.len().is_multiple_of(2));
    text.as_bytes()
        .chunks_exact(2)
        .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
        .collect()
}

fn hex(bytes: &[u8]) -> String {
    if bytes.is_empty() {
        return "-".into();
    }
    bytes.iter().map(|value| format!("{value:02x}")).collect()
}

fn name(text: &str) -> String {
    String::from_utf8(unhex(text)).unwrap()
}

fn definitions(fixture: &str, configured: bool) -> Vec<Definition> {
    fixture
        .lines()
        .filter_map(|line| {
            let row: Vec<_> = line.split('\t').collect();
            (row[0]
                == if configured {
                    "DEFINITION_CONFIG"
                } else {
                    "DEFINITION"
                })
            .then(|| Definition {
                name: name(row[1]),
                positions: row[2].parse().unwrap(),
                offsets: row[3].parse().unwrap(),
                payloads: row[4].parse().unwrap(),
            })
        })
        .collect()
}

fn inputs(rows: &[Vec<&str>]) -> BTreeMap<usize, Input> {
    let mut inputs = BTreeMap::<usize, Input>::new();
    for row in rows {
        match row[0] {
            "INPUT" => {
                assert!(inputs
                    .insert(
                        row[2].parse().unwrap(),
                        Input {
                            rank: row[3].parse().unwrap(),
                            kind: row[4].into(),
                            ..Input::default()
                        }
                    )
                    .is_none());
            }
            "INPUT_FIELD" => {
                let input = inputs.get_mut(&row[2].parse().unwrap()).unwrap();
                assert!(input.fields.insert(name(row[3]), Vec::new()).is_none());
            }
            "INPUT_TOKEN" => {
                let tokens = inputs
                    .get_mut(&row[2].parse().unwrap())
                    .unwrap()
                    .fields
                    .get_mut(&name(row[3]))
                    .unwrap();
                assert_eq!(tokens.len(), row[4].parse::<usize>().unwrap());
                tokens.push(FieldToken {
                    term: unhex(row[5]),
                    position_increment: row[6].parse().unwrap(),
                    start_offset: row[7].parse().unwrap(),
                    end_offset: row[8].parse().unwrap(),
                    payload: (row[9] != "null").then(|| unhex(row[9])),
                    term_frequency: 1,
                });
            }
            _ => {}
        }
    }
    for row in rows.iter().filter(|row| row[0] == "INPUT_FIELD") {
        let tokens = &inputs[&row[2].parse().unwrap()].fields[&name(row[3])];
        assert_eq!(tokens.len(), row[5].parse::<usize>().unwrap());
        assert_eq!(tokens.is_empty(), row[4] == "empty");
    }
    inputs
}

fn writer<'a>(
    directory: &'a dyn Directory,
    case: &str,
    configured: bool,
    definitions: &[Definition],
    inputs: &BTreeMap<usize, Input>,
) -> IndexWriter<'a> {
    let mut fields = Vec::new();
    if configured {
        fields.push(FieldInfo::new("id", 0));
        fields.push(FieldInfo::new("rank", 1).with_doc_values(
            DocValuesType::Numeric,
            DocValuesSkipIndexType::None,
            -1,
        ));
        for (index, definition) in definitions.iter().enumerate() {
            fields.push(
                FieldInfo::new(&definition.name, i32::try_from(index + 2).unwrap())
                    .with_index_options(if definition.offsets {
                        IndexOptions::DocsAndFreqsAndPositionsAndOffsets
                    } else {
                        IndexOptions::DocsAndFreqsAndPositions
                    })
                    .with_omit_norms(true)
                    .with_store_term_vectors(true)
                    .with_store_payloads(definition.payloads),
            );
        }
    }
    let mut writer = IndexWriter::open(
        directory,
        fields,
        "Lucene104",
        LuceneVersion {
            major: 10,
            minor: 5,
            bugfix: 0,
        },
    )
    .unwrap();
    if configured {
        writer.set_doc_values_field(Some("rank")).unwrap();
        for definition in definitions {
            writer.add_postings_field(&definition.name).unwrap();
            writer.add_term_vector_field(&definition.name).unwrap();
        }
        // The native stored-text facade uses StandardAnalyzer. The Java
        // configured corpus has exactly its `a β a` tokens/offsets, while
        // this source supplies the same per-occurrence payload attributes.
        let mut payloads = BTreeMap::new();
        for (&id, input) in inputs {
            for (name, tokens) in &input.fields {
                let mut position = -1;
                for token in tokens {
                    position += token.position_increment;
                    payloads.insert(
                        (i32::try_from(id).unwrap(), name.clone(), position),
                        token.payload.clone(),
                    );
                }
            }
        }
        writer
            .set_payload_source(Some(Box::new(move |context| {
                payloads
                    .get(&(context.doc_id, context.field.to_string(), context.position))
                    .cloned()
                    .flatten()
            })))
            .unwrap();
    } else {
        writer.enable_explicit_documents().unwrap();
    }
    if case.starts_with("asserting_") {
        writer
            .set_codec(Some(Arc::new(AssertingTermVectorsCodec)))
            .unwrap();
    }
    writer.set_use_compound_file(false);
    writer.set_max_buffered_docs(1000).unwrap();
    writer
        .set_ram_buffer_size_mb(DISABLE_AUTO_FLUSH_MB)
        .unwrap();
    writer.set_max_full_flush_merge_wait_millis(0);
    let mut policy = LogDocMergePolicy::new();
    policy.set_merge_factor(1000).unwrap();
    writer.set_pluggable_merge_policy(Some(Arc::new(policy)));
    if case.ends_with("_sorted") && !case.ends_with("_unsorted") {
        writer
            .set_index_sort(Some(&[IndexSortField::long("rank", false, None)]))
            .unwrap();
    }
    writer
}

fn add(
    writer: &mut IndexWriter<'_>,
    id: usize,
    input: &Input,
    definitions: &[Definition],
    field_order: usize,
    configured: bool,
) {
    if configured {
        let mut fields = vec![
            StoredField {
                field_number: 0,
                value: FieldValue::String(id.to_string()),
            },
            StoredField {
                field_number: 1,
                value: FieldValue::Long(input.rank),
            },
        ];
        for (index, definition) in definitions.iter().enumerate() {
            if let Some(tokens) = input.fields.get(&definition.name) {
                fields.push(StoredField {
                    field_number: i32::try_from(index + 2).unwrap(),
                    value: FieldValue::String(if tokens.is_empty() {
                        String::new()
                    } else {
                        "a β a".into()
                    }),
                });
            }
        }
        writer
            .add_document(stored_fields::Document { fields })
            .unwrap();
        return;
    }
    let mut document = Document::new();
    document.add(StringField::new("id", id.to_string(), Store::Yes));
    document.add(NumericDocValuesField::new("rank", input.rank));
    for index in 0..definitions.len() {
        let field = match field_order % 3 {
            1 => definitions.len() - 1 - index,
            2 => (index + 3) % definitions.len(),
            _ => index,
        };
        let definition = &definitions[field];
        let Some(tokens) = input.fields.get(&definition.name) else {
            continue;
        };
        let mut ty = FieldType::new();
        ty.set_tokenized(true).unwrap();
        ty.set_index_options(IndexOptions::DocsAndFreqsAndPositionsAndOffsets)
            .unwrap();
        ty.set_omit_norms(true).unwrap();
        ty.set_store_term_vectors(true).unwrap();
        ty.set_store_term_vector_positions(definition.positions)
            .unwrap();
        ty.set_store_term_vector_offsets(definition.offsets)
            .unwrap();
        ty.set_store_term_vector_payloads(definition.payloads)
            .unwrap();
        ty.freeze();
        document.add(
            Field::from_token_stream(
                &definition.name,
                FieldTokens {
                    tokens: tokens.clone(),
                    final_offset: tokens.last().map_or(0, |token| token.end_offset),
                    ..FieldTokens::default()
                },
                ty,
            )
            .unwrap(),
        );
    }
    writer.add_fields_document(&document).unwrap();
}

fn snapshot(
    directory: &dyn Directory,
    case: &str,
    phase: &str,
    inputs: &BTreeMap<usize, Input>,
    configured: bool,
) -> Vec<String> {
    let reader = DirectoryReader::open(directory).unwrap();
    let mut rows = Vec::new();
    let mut ordinal = 0;
    let mut base = 0;
    for (leaf, segment) in reader.segment_readers().iter().enumerate() {
        segment.check_integrity().unwrap();
        let fields = &segment.field_infos().fields;
        if !configured {
            for field in fields {
                rows.push(format!(
                    "FIELD\t{case}\t{phase}\t{leaf}\t{}\t{}\t{}",
                    hex(field.name.as_bytes()),
                    field.number,
                    field.store_term_vectors
                ));
            }
        }
        if let Some(vectors) = segment.term_vectors_reader().unwrap() {
            rows.push(format!(
                "CHUNKS\t{case}\t{phase}\t{leaf}\t{}\t{}\t{}",
                vectors.num_chunks(),
                vectors.num_dirty_chunks(),
                vectors.num_dirty_docs()
            ));
        }
        let id_field = fields
            .iter()
            .find(|field| field.name == "id")
            .unwrap()
            .number;
        let mut rank = segment.numeric_doc_values("rank").unwrap();
        for doc in 0..segment.max_doc() {
            if segment.live_docs().is_some_and(|bits| !bits.get_doc(doc)) {
                continue;
            }
            let stored = segment.stored_document(doc).unwrap().unwrap();
            let FieldValue::String(id) = &stored
                .fields
                .iter()
                .find(|field| field.field_number == id_field)
                .unwrap()
                .value
            else {
                panic!("stored id must be text")
            };
            let id: usize = id.parse().unwrap();
            let input = &inputs[&id];
            let rank = rank.as_mut().unwrap();
            assert!(rank.advance_exact(doc).unwrap());
            assert_eq!(rank.long_value(), input.rank);
            let vectors = segment.term_vectors(doc).unwrap();
            let count = vectors.as_ref().map_or(0, |vectors| vectors.fields.len());
            rows.push(format!(
                "DOC\t{case}\t{phase}\t{ordinal}\t{}\t{id}\t{}\t{count}\t{}",
                base + doc,
                input.kind,
                if vectors.is_some() { "fields" } else { "null" }
            ));
            if let Some(vectors) = vectors {
                let names: Vec<_> = vectors
                    .fields
                    .iter()
                    .map(|vector| {
                        &fields
                            .iter()
                            .find(|field| field.number == vector.field_number)
                            .unwrap()
                            .name
                    })
                    .collect();
                assert!(
                    names
                        .windows(2)
                        .all(|pair| pair[0].encode_utf16().cmp(pair[1].encode_utf16()).is_lt()),
                    "{case}/{phase}: native vector field order must follow UTF-16"
                );
                for vector in vectors.fields {
                    let field = fields
                        .iter()
                        .find(|field| field.number == vector.field_number)
                        .unwrap();
                    let name = hex(field.name.as_bytes());
                    rows.push(format!(
                        "VECTOR\t{case}\t{phase}\t{ordinal}\t{name}\t{}\t{}\t{}\t{}",
                        vector.terms.len(),
                        vector.has_positions,
                        vector.has_offsets,
                        vector.has_payloads
                    ));
                    assert!(vector
                        .terms
                        .windows(2)
                        .all(|pair| pair[0].term < pair[1].term));
                    for term in vector.terms {
                        let value = hex(&term.term);
                        rows.push(format!(
                            "TERM\t{case}\t{phase}\t{ordinal}\t{name}\t{value}\t{}",
                            term.freq
                        ));
                        if vector.has_positions || vector.has_offsets {
                            for occurrence in 0..usize::try_from(term.freq).unwrap() {
                                let position = term
                                    .positions
                                    .as_ref()
                                    .map_or(-1, |values| values[occurrence]);
                                let start = term
                                    .start_offsets
                                    .as_ref()
                                    .map_or(-1, |values| values[occurrence]);
                                let end = term
                                    .end_offsets
                                    .as_ref()
                                    .map_or(-1, |values| values[occurrence]);
                                let payload = term
                                    .payloads
                                    .as_ref()
                                    .map_or(&[][..], |values| values[occurrence].as_slice());
                                rows.push(format!("POSITION\t{case}\t{phase}\t{ordinal}\t{name}\t{value}\t{position}\t{start}\t{end}\t{}", hex(payload)));
                            }
                        }
                    }
                }
            }
            ordinal += 1;
        }
        base += segment.max_doc();
    }
    assert_eq!(ordinal, reader.num_docs());
    rows.sort();
    rows
}

fn assert_snapshot(
    rows: &[Vec<&str>],
    directory: &dyn Directory,
    case: &str,
    phase: &str,
    inputs: &BTreeMap<usize, Input>,
    configured: bool,
) {
    let kinds = if configured {
        &["DOC", "VECTOR", "TERM", "POSITION", "CHUNKS"][..]
    } else {
        &["FIELD", "DOC", "VECTOR", "TERM", "POSITION", "CHUNKS"][..]
    };
    let mut expected: Vec<_> = rows
        .iter()
        .filter(|row| kinds.contains(&row[0]) && row[2] == phase)
        .map(|row| row.join("\t"))
        .collect();
    if configured && case.ends_with("_configured_missing") {
        // The fixed-schema facade must retain hasVectors across segments.
        // Java discovers no field here; Rust instead writes one empty dirty
        // chunk. Check that deliberate geometry separately from Java's rows.
        assert_eq!(phase, "after");
        assert!(!expected.iter().any(|row| row.starts_with("CHUNKS\t")));
        expected.push(format!(
            "CHUNKS\t{case}\t{phase}\t0\t1\t1\t{}",
            inputs.len()
        ));
    }
    expected.sort();
    let actual = snapshot(directory, case, phase, inputs, configured);
    assert_eq!(actual, expected, "{case}/{phase}, configured={configured}");
}

fn replay(fixture: &str, case: &str, configured: bool) {
    if let Some(root) = std::env::var_os("LUCENE_TV_ROUTING_EXPORT") {
        let name = format!("{case}{}", if configured { "-configured" } else { "" });
        let path = std::path::PathBuf::from(root).join(name);
        assert!(
            !path.exists(),
            "export destination must be fresh: {}",
            path.display()
        );
        std::fs::create_dir_all(&path).unwrap();
        let directory = FsDirectory::open(&path);
        replay_directory(fixture, case, configured, &directory);
    } else {
        replay_directory(fixture, case, configured, &ByteBuffersDirectory::new());
    }
}

fn replay_directory(fixture: &str, case: &str, configured: bool, directory: &dyn Directory) {
    let rows: Vec<_> = fixture
        .lines()
        .map(|line| line.split('\t').collect::<Vec<_>>())
        .filter(|row| row.get(1) == Some(&case))
        .collect();
    let definitions = definitions(fixture, case.contains("_configured_"));
    let inputs = inputs(&rows);
    assert!(!inputs.is_empty(), "{case}: fixture has no inputs");
    let mut writer = writer(directory, case, configured, &definitions, &inputs);
    let merge = case.contains("_direct_") || case.contains("_imported_");
    if case.contains("_imported_") {
        for source in 0..3 {
            let input_directory = ByteBuffersDirectory::new();
            {
                let mut input_writer =
                    self::writer(&input_directory, case, false, &definitions, &inputs);
                for id in source * 4..(source + 1) * 4 {
                    add(
                        &mut input_writer,
                        id,
                        &inputs[&id],
                        &definitions,
                        source,
                        false,
                    );
                }
                input_writer.commit().unwrap();
            }
            writer.add_indexes(&[&input_directory]).unwrap();
        }
    } else {
        for (&id, input) in &inputs {
            add(&mut writer, id, input, &definitions, 0, configured);
            if merge && (id + 1).is_multiple_of(4) {
                writer.flush().unwrap();
            }
        }
    }
    if !merge && case.contains("_explicit_") {
        writer.flush().unwrap();
    }
    writer.commit().unwrap();
    if merge {
        assert_snapshot(&rows, directory, case, "initial", &inputs, configured);
        let deletes: Vec<_> = inputs
            .keys()
            .filter(|id| id.is_multiple_of(7))
            .map(|id| Term::new("id", id.to_string()))
            .collect();
        writer.delete_documents_by_term(&deletes).unwrap();
        writer.commit().unwrap();
        assert_snapshot(&rows, directory, case, "before", &inputs, configured);
        writer.force_merge(1).unwrap();
        writer.commit().unwrap();
    }
    assert_snapshot(&rows, directory, case, "after", &inputs, configured);
    if let Some(expected) = rows.iter().find(|row| row[0] == "FILES") {
        let files = directory
            .list_all()
            .unwrap()
            .iter()
            .filter(|file| {
                [".tvd", ".tvx", ".tvm"]
                    .iter()
                    .any(|suffix| file.ends_with(suffix))
            })
            .count();
        // Fixed-schema missing fields still need a native empty triple;
        // dynamically discovered fields retain Java's acquisition count.
        let expected_files = if configured && case.ends_with("_configured_missing") {
            assert_eq!(expected[2], "0");
            3
        } else {
            expected[2].parse::<usize>().unwrap()
        };
        assert_eq!(files, expected_files, "{case}: vector writer acquisition");
    }
}

#[test]
fn native_binary_term_vectors_match_java_writer_routing_snapshots() {
    let fixture = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/testframework/data/term_vectors_writer_routing.tsv"
    ))
    .unwrap();
    let mut cases: Vec<_> = fixture
        .lines()
        .filter_map(|line| {
            let row: Vec<_> = line.split('\t').collect();
            (row[0] == "INPUT").then(|| row[1])
        })
        .collect();
    cases.sort();
    cases.dedup();
    assert_eq!(
        cases.len(),
        44,
        "every generated flush/merge case must be replayed"
    );
    for case in cases {
        replay(&fixture, case, false);
        if case.contains("_configured_") {
            replay(&fixture, case, true);
        }
    }
}
