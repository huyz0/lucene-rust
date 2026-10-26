//! Installed native/asserting postings writers: whole flush and merge commits.
//! Corpus preparation and full semantic verification run outside each timer.
use lucene_codecs::{
    field_infos::IndexOptions,
    per_field_postings::{PER_FIELD_FORMAT_KEY, PER_FIELD_SUFFIX_KEY},
    stored_fields::FieldValue,
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
    segment_infos,
};
use lucene_search::{
    directory_reader::DirectoryReader,
    reader::{LeafReader, PostingsFlags, NO_MORE_DOCS},
};
use lucene_store::{ByteBuffersDirectory, Directory, FsDirectory};
use lucene_test_framework::codecs::asserting_postings::AssertingPostingsCodec;
use lucene_util::small_float::int_to_byte4;
use std::{collections::BTreeMap, hint::black_box, sync::Arc, time::Instant};
const DOCS: usize = 8192;
const SOURCES: usize = 4;
const FIELDS: [&str; 6] = ["docs", "freqs", "positions", "offsets", "sparse", "empty"];
fn options(field: usize) -> IndexOptions {
    match field {
        0 => IndexOptions::Docs,
        1 => IndexOptions::DocsAndFreqs,
        3 => IndexOptions::DocsAndFreqsAndPositionsAndOffsets,
        _ => IndexOptions::DocsAndFreqsAndPositions,
    }
}
fn rank(doc: usize) -> i64 {
    ((doc % (DOCS / SOURCES)) * SOURCES + doc / (DOCS / SOURCES)) as i64
}
fn tokens(doc: usize, field: usize) -> Vec<FieldToken> {
    if field == 5 || (field == 4 && doc.is_multiple_of(3)) {
        return Vec::new();
    }
    let mut values = vec![
        ("common".to_owned(), 1, 0, 6),
        ("common".to_owned(), 0, 10, 16),
        (format!("t{:03}", doc % 257), 2, 20, 24),
        (format!("u{doc:05}"), 1, 30, 36),
        (format!("bucket{:02}", doc % 11), 1, 40, 48),
    ];
    for i in 0..doc % 5 {
        values.push((
            "common".to_owned(),
            1,
            50 + 10 * i as i32,
            56 + 10 * i as i32,
        ));
    }
    values
        .into_iter()
        .enumerate()
        .map(
            |(ordinal, (term, position_increment, start_offset, end_offset))| FieldToken {
                term: term.into_bytes(),
                start_offset,
                end_offset,
                position_increment,
                term_frequency: 1,
                payload: if field == 3 && !(doc + ordinal).is_multiple_of(3) {
                    Some(vec![doc as u8, ordinal as u8, 255])
                } else {
                    None
                },
            },
        )
        .collect()
}
fn document(doc: usize) -> Document {
    let mut result = Document::new();
    result.add(StringField::new("id", doc.to_string(), Store::Yes));
    result.add(NumericDocValuesField::new("rank", rank(doc)));
    for (field, name) in FIELDS.iter().enumerate() {
        if field == 4 && doc.is_multiple_of(3) {
            continue;
        }
        let mut ty = FieldType::new();
        ty.set_tokenized(true).unwrap();
        ty.set_index_options(options(field)).unwrap();
        ty.set_omit_norms(field == 0).unwrap();
        ty.freeze();
        let tokens = tokens(doc, field);
        let final_offset = tokens.last().map_or(0, |token| token.end_offset);
        result.add(
            Field::from_token_stream(
                *name,
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
            .set_codec(Some(Arc::new(AssertingPostingsCodec)))
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
    for doc in 0..DOCS {
        writer.add_fields_document(&document(doc)).unwrap();
        if mode != 0 && (doc + 1).is_multiple_of(DOCS / SOURCES) {
            writer.commit().unwrap();
        }
    }
    if mode != 0 {
        let terms: Vec<_> = (0..DOCS)
            .filter(|doc| doc.is_multiple_of(7))
            .map(|doc| Term::new("id", doc.to_string()))
            .collect();
        writer.delete_documents_by_term(&terms).unwrap();
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
fn mix(digest: u64, value: u64) -> u64 {
    (digest ^ value).wrapping_mul(0x100_0000_01b3)
}
fn bytes(digest: &mut u64, value: &[u8]) {
    *digest = mix(*digest, value.len() as u64);
    for b in value {
        *digest = mix(*digest, u64::from(*b));
    }
}
#[derive(Clone, Copy)]
struct Check {
    digest: u64,
    docs: usize,
    terms: usize,
    postings: usize,
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
        ids.sort_by_key(|doc| rank(*doc));
    }
    assert_eq!(reader.max_doc(), ids.len() as i32);
    assert_eq!(reader.num_docs(), ids.len() as i32);
    assert_eq!(reader.segment_readers().len(), 1);
    let segment = &reader.segment_readers()[0];
    let mut digest = 0xcbf2_9ce4_8422_2325;
    let mut ranks = segment.numeric_doc_values("rank").unwrap().unwrap();
    for (physical, id) in ids.iter().enumerate() {
        let stored = segment.stored_document(physical as i32).unwrap().unwrap();
        let actual = stored
            .fields
            .iter()
            .find_map(|field| match &field.value {
                FieldValue::String(value) => Some(value.parse::<usize>().unwrap()),
                _ => None,
            })
            .unwrap();
        assert_eq!(actual, *id);
        digest = mix(digest, *id as u64);
        assert!(ranks.advance_exact(physical as i32).unwrap());
        assert_eq!(ranks.long_value(), rank(*id));
        digest = mix(digest, rank(*id) as u64);
    }
    let (mut term_count, mut posting_count, mut occurrences, mut payload_bytes) = (0, 0, 0, 0);
    for (field, name) in FIELDS.iter().enumerate() {
        let info = segment
            .field_infos()
            .fields
            .iter()
            .find(|info| info.name == *name)
            .unwrap();
        assert_eq!(info.index_options, options(field));
        assert_eq!(info.omit_norms, field == 0, "omit norms for {name}");
        assert_eq!(info.store_payloads, field == 3, "payload flag for {name}");
        let attribute = |key: &str| {
            info.attributes
                .iter()
                .find(|(name, _)| name == key)
                .map(|(_, value)| value.as_str())
        };
        assert_eq!(
            attribute(PER_FIELD_FORMAT_KEY),
            if field == 5 { None } else { Some("Lucene104") },
            "format for {name}"
        );
        assert_eq!(
            attribute(PER_FIELD_SUFFIX_KEY),
            if field == 5 { None } else { Some("0") },
            "suffix for {name}"
        );
        let mut norms = segment.norm_values(name).unwrap();
        assert_eq!(norms.is_some(), field != 0);
        type ExpectedPostings = BTreeMap<Vec<u8>, BTreeMap<i32, Vec<(i32, FieldToken)>>>;
        let mut expected = ExpectedPostings::new();
        let mut doc_count = 0;
        for (physical, id) in ids.iter().enumerate() {
            let values = tokens(*id, field);
            if !values.is_empty() {
                doc_count += 1;
            }
            if let Some(norms) = norms.as_mut() {
                let present = field != 4 || !id.is_multiple_of(3);
                assert_eq!(norms.advance_exact(physical as i32).unwrap(), present);
                if present {
                    let norm = int_to_byte4(
                        values.len().saturating_sub(usize::from(!values.is_empty())) as u32,
                    );
                    assert_eq!(norms.long_value(), i64::from(norm));
                    digest = mix(digest, u64::from(norm));
                }
            }
            let mut position = -1;
            for token in values {
                position += token.position_increment;
                expected
                    .entry(token.term.clone())
                    .or_default()
                    .entry(physical as i32)
                    .or_default()
                    .push((position, token));
            }
        }
        digest = mix(digest, field as u64);
        digest = mix(digest, expected.len() as u64);
        let actual = segment.terms(name).unwrap();
        if expected.is_empty() {
            assert!(actual.is_none());
            continue;
        }
        let actual = actual.unwrap();
        assert_eq!(actual.size(), expected.len() as i64);
        assert_eq!(actual.doc_count(), doc_count);
        assert_eq!(actual.has_freqs(), field != 0);
        assert_eq!(actual.has_positions(), field >= 2);
        assert_eq!(actual.has_offsets(), field == 3);
        assert_eq!(actual.has_payloads(), field == 3);
        let sum_docs: usize = expected.values().map(BTreeMap::len).sum();
        let sum_total: usize = expected
            .values()
            .flat_map(BTreeMap::values)
            .map(|values| if field == 0 { 1 } else { values.len() })
            .sum();
        assert_eq!(actual.sum_doc_freq(), sum_docs as i64);
        assert_eq!(actual.sum_total_term_freq(), sum_total as i64);
        digest = mix(digest, sum_docs as u64);
        digest = mix(digest, sum_total as u64);
        digest = mix(digest, doc_count as u64);
        let mut iterator = actual.iterator().unwrap();
        for (term, docs) in expected {
            assert_eq!(iterator.next().unwrap().unwrap(), term);
            bytes(&mut digest, &term);
            term_count += 1;
            assert_eq!(iterator.doc_freq().unwrap(), docs.len() as i32);
            let total: usize = docs
                .values()
                .map(|values| if field == 0 { 1 } else { values.len() })
                .sum();
            assert_eq!(iterator.total_term_freq().unwrap(), total as i64);
            let mut postings = iterator.postings(PostingsFlags::All).unwrap();
            for (doc, values) in docs {
                assert_eq!(postings.next_doc().unwrap(), doc);
                let freq = if field == 0 { 1 } else { values.len() as i32 };
                assert_eq!(postings.freq(), freq);
                digest = mix(digest, doc as u64);
                digest = mix(digest, freq as u64);
                posting_count += 1;
                occurrences += freq as usize;
                if field >= 2 {
                    for (position, token) in values {
                        assert_eq!(postings.next_position().unwrap(), position);
                        digest = mix(digest, position as u64);
                        let (start, end) = if field == 3 {
                            (token.start_offset, token.end_offset)
                        } else {
                            (-1, -1)
                        };
                        assert_eq!(postings.start_offset(), start);
                        assert_eq!(postings.end_offset(), end);
                        digest = mix(digest, start as u64);
                        digest = mix(digest, end as u64);
                        assert_eq!(postings.payload(), token.payload.as_deref());
                        let payload = token.payload.as_deref().unwrap_or_default();
                        bytes(&mut digest, payload);
                        payload_bytes += payload.len();
                    }
                }
            }
            assert_eq!(postings.next_doc().unwrap(), NO_MORE_DOCS);
        }
        assert!(iterator.next().unwrap().is_none());
    }
    segment.check_integrity().unwrap();
    let encoded_bytes = directory
        .list_all()
        .unwrap()
        .iter()
        .filter(|name| {
            ["doc", "psm", "pos", "pay", "tim", "tip", "tmd"]
                .iter()
                .any(|ext| name.ends_with(&format!(".{ext}")))
        })
        .map(|name| directory.file_length(name).unwrap())
        .sum();
    Check {
        digest: black_box(digest),
        docs: ids.len(),
        terms: term_count,
        postings: posting_count,
        occurrences,
        payload_bytes,
        encoded_bytes,
    }
}
fn name(asserting: bool, mode: usize) -> String {
    format!(
        "postings_routing_{}_{}_commit",
        if asserting { "asserting" } else { "default" },
        ["flush", "merge", "sorted_merge"][mode]
    )
}
fn output(name: &str, ns: f64, iterations: u64, check: Check, accounted: usize) {
    println!(
        "#check\t{name}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
        check.digest,
        check.docs,
        FIELDS.len(),
        check.terms,
        check.postings,
        check.occurrences,
        check.payload_bytes
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
        let accounted = writer.ram_bytes_used();
        let start = Instant::now();
        if mode != 0 {
            writer.force_merge(1).unwrap();
        }
        writer.commit().unwrap();
        nanos += start.elapsed().as_nanos();
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
        let mode = args[3].parse().unwrap();
        let mut writer = prepare(&directory, true, mode);
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
    assert!(measure > 0);
    for mode in 0..3 {
        for asserting in [false, true] {
            run(asserting, mode, warmup, 0);
            let (ns, n, check, accounted) = run(asserting, mode, measure, 0);
            output(&name(asserting, mode), ns, n, check, accounted);
        }
    }
}
