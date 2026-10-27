//! Native term-vector routing: prepared-document insertion + commit for flush;
//! forceMerge + commit for merges, on the same corpus as Java.
//! Setup, document preparation, accounting, readback and destruction are untimed.
use lucene_codecs::{field_infos::IndexOptions, stored_fields::FieldValue};
use lucene_index::{
    buffered_updates::Term,
    document::{
        Document, Field, FieldToken, FieldTokens, FieldType, NumericDocValuesField, Store,
        StringField,
    },
    index_writer::{IndexWriter, DISABLE_AUTO_FLUSH_MB},
    merge_policy::log::LogDocMergePolicy,
    segment_info::{IndexSortField, LuceneVersion},
    segment_infos,
};
use lucene_search::{directory_reader::DirectoryReader, reader::LeafReader};
use lucene_store::{ByteBuffersDirectory, Directory, FsDirectory};
use lucene_test_framework::codecs::asserting_term_vectors_writer::AssertingTermVectorsCodec;
use std::{collections::BTreeMap, hint::black_box, sync::Arc, time::Instant};

const DOCS: usize = 2048;
const SOURCES: usize = 4;
// Same insertion order as TermVectorsRoutingSupport.DEFINITIONS, deliberately
// different from UTF-16 field-name order and from UTF-8 scalar order.
const FIELDS: [(&str, bool, bool, bool); 8] = [
    ("z_full", true, true, true),
    ("\u{e000}", false, false, false),
    ("\u{10000}", true, false, false),
    ("a_offsets", false, true, false),
    ("b_payloads", true, false, true),
    ("m_sparse", true, true, false),
    ("x_empty", true, true, true),
    ("q_plain", false, false, false),
];

fn rank(doc: usize) -> i64 {
    ((doc % (DOCS / SOURCES)) * SOURCES + doc / (DOCS / SOURCES)) as i64
}
fn tokens(doc: usize, field: usize) -> Option<Vec<FieldToken>> {
    if doc.is_multiple_of(17) {
        return None;
    }
    if doc % 17 == 1 {
        return (field == 6).then(Vec::new);
    }
    if field == 6 {
        return Some(Vec::new());
    }
    if field == 5 && doc.is_multiple_of(3) {
        return None;
    }
    let mut values = vec![
        (vec![0], 1, 0, 2),
        (vec![255, 0, (doc % 13) as u8], 2, 4, 7),
        (vec![0], 0, 4, 7),
        (vec![97, 0, 128], 1, 10, 14),
        (vec![97, 0, 255], 1, 16, 21),
    ];
    for i in 0..doc % 3 {
        values.push((vec![0], 1, 24 + i as i32 * 4, 27 + i as i32 * 4));
    }
    Some(
        values
            .into_iter()
            .enumerate()
            .map(
                |(ordinal, (term, position_increment, start_offset, end_offset))| FieldToken {
                    term,
                    start_offset,
                    end_offset,
                    position_increment,
                    term_frequency: 1,
                    payload: if FIELDS[field].3 {
                        match (doc + ordinal) % 3 {
                            0 => None,
                            1 => Some(Vec::new()),
                            _ => Some(vec![doc as u8, ordinal as u8, 0, 255]),
                        }
                    } else {
                        None
                    },
                },
            )
            .collect(),
    )
}
fn document(doc: usize) -> Document {
    let mut result = Document::new();
    result.add(StringField::new("id", doc.to_string(), Store::Yes));
    result.add(NumericDocValuesField::new("rank", rank(doc)));
    for (field, &(name, positions, offsets, payloads)) in FIELDS.iter().enumerate() {
        let Some(tokens) = tokens(doc, field) else {
            continue;
        };
        let mut ty = FieldType::new();
        ty.set_tokenized(true).unwrap();
        ty.set_index_options(IndexOptions::DocsAndFreqsAndPositionsAndOffsets)
            .unwrap();
        ty.set_omit_norms(true).unwrap();
        ty.set_store_term_vectors(true).unwrap();
        ty.set_store_term_vector_positions(positions).unwrap();
        ty.set_store_term_vector_offsets(offsets).unwrap();
        ty.set_store_term_vector_payloads(payloads).unwrap();
        ty.freeze();
        let final_offset = tokens.last().map_or(0, |token| token.end_offset);
        result.add(
            Field::from_token_stream(
                name,
                FieldTokens {
                    tokens,
                    final_position_increment: 0,
                    final_offset,
                    end_attributes: None,
                },
                ty,
            )
            .unwrap(),
        );
    }
    result
}
fn prepare(directory: &dyn Directory, asserting: bool, mode: usize) -> IndexWriter<'_> {
    let mut writer = IndexWriter::open(
        directory,
        Vec::new(),
        "Lucene104",
        LuceneVersion {
            major: 10,
            minor: 5,
            bugfix: 0,
        },
    )
    .unwrap();
    writer.enable_explicit_documents().unwrap();
    if asserting {
        writer
            .set_codec(Some(Arc::new(AssertingTermVectorsCodec)))
            .unwrap();
    }
    writer.set_use_compound_file(false);
    writer.set_max_buffered_docs((DOCS * 2) as i32).unwrap();
    writer
        .set_ram_buffer_size_mb(DISABLE_AUTO_FLUSH_MB)
        .unwrap();
    writer.set_max_full_flush_merge_wait_millis(0);
    let mut policy = LogDocMergePolicy::new();
    policy.set_merge_factor(1000).unwrap();
    writer.set_pluggable_merge_policy(Some(Arc::new(policy)));
    if mode == 2 {
        writer
            .set_index_sort(Some(&[IndexSortField::long("rank", false, None)]))
            .unwrap();
    }
    if mode != 0 {
        for doc in 0..DOCS {
            writer.add_fields_document(&document(doc)).unwrap();
            if (doc + 1).is_multiple_of(DOCS / SOURCES) {
                writer.commit().unwrap();
            }
        }
        let deletes: Vec<_> = (0..DOCS)
            .filter(|doc| doc.is_multiple_of(7))
            .map(|doc| Term::new("id", doc.to_string()))
            .collect();
        writer.delete_documents_by_term(&deletes).unwrap();
        writer.commit().unwrap();
        assert_eq!(
            segment_infos::read_latest(directory)
                .unwrap()
                .segments
                .len(),
            SOURCES
        );
    }
    writer
}
fn prepared_documents(mode: usize) -> Vec<Document> {
    if mode == 0 {
        (0..DOCS).map(document).collect()
    } else {
        Vec::new()
    }
}
fn mix(digest: u64, value: u64) -> u64 {
    (digest ^ value).wrapping_mul(0x100_0000_01b3)
}
fn bytes(digest: &mut u64, values: &[u8]) {
    *digest = mix(*digest, values.len() as u64);
    for &value in values {
        *digest = mix(*digest, u64::from(value));
    }
}
struct Check {
    digest: u64,
    docs: usize,
    fields: usize,
    terms: usize,
    occurrences: usize,
    payload_bytes: usize,
    encoded_bytes: u64,
}
fn verify(directory: &dyn Directory, mode: usize) -> Check {
    let reader = DirectoryReader::open(directory).unwrap();
    let mut ids: Vec<_> = (0..DOCS)
        .filter(|doc| mode == 0 || !doc.is_multiple_of(7))
        .collect();
    if mode == 2 {
        ids.sort_by_key(|&doc| rank(doc));
    }
    assert_eq!(reader.max_doc(), ids.len() as i32);
    assert_eq!(reader.num_docs(), ids.len() as i32);
    assert_eq!(reader.segment_readers().len(), 1);
    let segment = &reader.segment_readers()[0];
    let mut ranks = segment.numeric_doc_values("rank").unwrap().unwrap();
    let mut digest = 0xcbf2_9ce4_8422_2325;
    let (mut fields, mut terms, mut occurrences, mut payload_bytes) = (0, 0, 0, 0);
    for (physical, id) in ids.iter().copied().enumerate() {
        let stored = segment.stored_document(physical as i32).unwrap().unwrap();
        let actual = stored
            .fields
            .iter()
            .find_map(|field| match &field.value {
                FieldValue::String(value) => Some(value.parse::<usize>().unwrap()),
                _ => None,
            })
            .unwrap();
        assert_eq!(actual, id);
        assert!(ranks.advance_exact(physical as i32).unwrap());
        assert_eq!(ranks.long_value(), rank(id));
        digest = mix(digest, id as u64);
        digest = mix(digest, rank(id) as u64);
        let mut expected_fields: Vec<_> = FIELDS
            .iter()
            .enumerate()
            .filter_map(|(field, &(name, p, o, y))| {
                tokens(id, field)
                    .filter(|tokens| !tokens.is_empty())
                    .map(|tokens| (name, p, o, y, tokens))
            })
            .collect();
        expected_fields.sort_by(|a, b| a.0.encode_utf16().cmp(b.0.encode_utf16()));
        digest = mix(digest, expected_fields.len() as u64);
        let actual = segment.term_vectors(physical as i32).unwrap();
        if expected_fields.is_empty() {
            assert!(
                actual.is_none(),
                "missing and empty input both read null vectors"
            );
            continue;
        }
        let actual = actual.unwrap();
        assert_eq!(actual.fields.len(), expected_fields.len());
        for (actual, (name, positions, offsets, payloads, tokens)) in
            actual.fields.iter().zip(expected_fields)
        {
            let field = segment
                .field_infos()
                .fields
                .iter()
                .find(|field| field.number == actual.field_number)
                .unwrap();
            assert_eq!(field.name, name, "UTF-16 vector field order");
            let has_payloads = payloads
                && tokens
                    .iter()
                    .any(|token| token.payload.as_ref().is_some_and(|p| !p.is_empty()));
            assert_eq!(actual.has_positions, positions);
            assert_eq!(actual.has_offsets, offsets);
            assert_eq!(actual.has_payloads, has_payloads);
            let mut expected_terms: BTreeMap<Vec<u8>, Vec<(i32, &FieldToken)>> = BTreeMap::new();
            let mut position = -1;
            for token in &tokens {
                position += token.position_increment;
                expected_terms
                    .entry(token.term.clone())
                    .or_default()
                    .push((position, token));
            }
            assert_eq!(actual.terms.len(), expected_terms.len());
            bytes(&mut digest, name.as_bytes());
            digest = mix(digest, u64::from(positions));
            digest = mix(digest, u64::from(offsets));
            digest = mix(digest, u64::from(has_payloads));
            digest = mix(digest, expected_terms.len() as u64);
            fields += 1;
            for (actual, (term, expected)) in actual.terms.iter().zip(expected_terms) {
                assert_eq!(actual.term, term);
                assert_eq!(actual.freq, expected.len() as i32);
                bytes(&mut digest, &term);
                digest = mix(digest, expected.len() as u64);
                terms += 1;
                occurrences += expected.len();
                if positions || offsets {
                    for (ordinal, &(position, token)) in expected.iter().enumerate() {
                        let p = if positions { position } else { -1 };
                        let start = if offsets { token.start_offset } else { -1 };
                        let end = if offsets { token.end_offset } else { -1 };
                        assert_eq!(actual.positions.as_ref().map_or(-1, |p| p[ordinal]), p);
                        assert_eq!(
                            actual.start_offsets.as_ref().map_or(-1, |p| p[ordinal]),
                            start
                        );
                        assert_eq!(actual.end_offsets.as_ref().map_or(-1, |p| p[ordinal]), end);
                        let expected_payload = if payloads {
                            token.payload.as_deref().unwrap_or_default()
                        } else {
                            &[]
                        };
                        let actual_payload = actual
                            .payloads
                            .as_ref()
                            .map_or(&[][..], |p| p[ordinal].as_slice());
                        assert_eq!(actual_payload, expected_payload);
                        digest = mix(digest, p as u64);
                        digest = mix(digest, start as u64);
                        digest = mix(digest, end as u64);
                        bytes(&mut digest, actual_payload);
                        payload_bytes += actual_payload.len();
                    }
                }
            }
        }
    }
    segment.check_integrity().unwrap();
    let encoded_bytes = directory
        .list_all()
        .unwrap()
        .iter()
        .filter(|name| {
            [".tvd", ".tvx", ".tvm"]
                .iter()
                .any(|ext| name.ends_with(ext))
        })
        .map(|name| directory.file_length(name).unwrap())
        .sum();
    Check {
        digest: black_box(digest),
        docs: ids.len(),
        fields,
        terms,
        occurrences,
        payload_bytes,
        encoded_bytes,
    }
}
fn name(asserting: bool, mode: usize) -> String {
    format!(
        "term_vectors_routing_{}_{}_commit",
        if asserting { "asserting" } else { "default" },
        ["flush", "merge", "sorted_merge"][mode]
    )
}
fn output(name: &str, ns: f64, iterations: u64, check: Check, accounted: usize) {
    println!(
        "#check\t{name}\t{}\t{}\t{}\t{}\t{}\t{}",
        check.digest, check.docs, check.fields, check.terms, check.occurrences, check.payload_bytes
    );
    println!("{name}\t{ns:.3}\t{iterations}");
    println!("{name}_encoded_bytes\t{}", check.encoded_bytes);
    println!("{name}_writer_accounted_bytes\t{accounted}");
    println!("{name}_logical_payload_bytes\t{}", check.payload_bytes);
}
fn run(asserting: bool, mode: usize, millis: u64, fixed: u64) -> (f64, u64, Check, usize) {
    let (mut nanos, mut iterations) = (0u128, 0u64);
    loop {
        let directory = ByteBuffersDirectory::new();
        let mut writer = prepare(&directory, asserting, mode);
        let documents = prepared_documents(mode);
        let insertion_nanos = if mode == 0 {
            // Include the callbacks Java performs during document insertion.
            let start = Instant::now();
            for document in &documents {
                writer.add_fields_document(document).unwrap();
            }
            start.elapsed().as_nanos()
        } else {
            0
        };
        let accounted = writer.ram_bytes_used();
        let start = Instant::now();
        if mode != 0 {
            writer.force_merge(1).unwrap();
        }
        writer.commit().unwrap();
        nanos += insertion_nanos + start.elapsed().as_nanos();
        iterations += 1;
        let check = verify(&directory, mode);
        if (fixed > 0 && iterations == fixed)
            || (fixed == 0 && nanos >= u128::from(millis) * 1_000_000)
        {
            return (
                nanos as f64 / iterations as f64,
                iterations,
                check,
                accounted,
            );
        }
    }
}
fn main() {
    let args: Vec<_> = std::env::args().collect();
    if args[1] == "--write-verify" {
        std::fs::create_dir_all(&args[2]).unwrap();
        let directory = FsDirectory::open(&args[2]);
        assert!(directory.list_all().unwrap().is_empty());
        let mode: usize = args[3].parse().unwrap();
        assert!(mode < 3);
        let mut writer = prepare(&directory, true, mode);
        for document in &prepared_documents(mode) {
            writer.add_fields_document(document).unwrap();
        }
        if mode != 0 {
            writer.force_merge(1).unwrap();
        }
        writer.commit().unwrap();
        let check = verify(&directory, mode);
        output(&name(true, mode), 1.0, 1, check, 0);
        return;
    }
    if args[1] == "--memory-case" {
        let case = &args[2];
        let fixed: u64 = args[3].parse().unwrap();
        assert!(fixed > 0);
        for mode in 0..3 {
            for asserting in [false, true] {
                let name = name(asserting, mode);
                if &name == case {
                    let (ns, n, check, accounted) = run(asserting, mode, 0, fixed);
                    output(&name, ns, n, check, accounted);
                    return;
                }
            }
        }
        panic!("unknown case {case}");
    }
    let warmup = args[1].parse().unwrap();
    let measure = args[2].parse().unwrap();
    assert!(warmup > 0 && measure > 0);
    for mode in 0..3 {
        for asserting in [false, true] {
            run(asserting, mode, warmup, 0);
            let (ns, n, check, accounted) = run(asserting, mode, measure, 0);
            output(&name(asserting, mode), ns, n, check, accounted);
        }
    }
}
