//! Whole IndexWriter flush/merge commits through native or installed asserting points.
//! Corpus preparation and complete decoded-result checks are outside each timer.
//! Rust buffers native point encoding until finish/merge; Java writes fields incrementally.
//! The timed interval includes both implementations' encoding, close and publication.
use lucene_codecs::stored_fields::FieldValue;
use lucene_index::{
    buffered_updates::Term,
    document::{Document, Field, FieldType, Store, StringField},
    index_writer::{IndexWriter, DISABLE_AUTO_FLUSH_MB},
    merge_policy::log::LogDocMergePolicy,
    segment_info::LuceneVersion,
    segment_infos,
};
use lucene_search::directory_reader::DirectoryReader;
use lucene_store::{ByteBuffersDirectory, Directory, FsDirectory};
use lucene_test_framework::codecs::asserting_points::AssertingPointsCodec;
use std::{hint::black_box, sync::Arc, time::Instant};
const DOCS: usize = 8192;
const SOURCES: usize = 4;
const FIELDS: [&str; 4] = ["one", "sparse", "three", "vanish"];
fn dimensions(field: usize) -> usize {
    if field == 2 {
        3
    } else {
        1
    }
}
fn index_dimensions(field: usize) -> usize {
    if field == 2 {
        2
    } else {
        1
    }
}
fn packed(values: &[i32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| ((*value as u32) ^ 0x8000_0000).to_be_bytes())
        .collect()
}
fn values(doc: usize, field: usize) -> Vec<Vec<u8>> {
    match field {
        0 => vec![packed(&[if doc == 0 {
            i32::MIN
        } else if doc == DOCS - 1 {
            i32::MAX
        } else {
            doc as i32 - 4096
        }])],
        1 if !doc.is_multiple_of(3) => vec![
            packed(&[7]),
            packed(&[7]),
            packed(&[(doc % 97) as i32 - 48]),
        ],
        2 => vec![
            packed(&[
                (doc % 17) as i32 - 8,
                ((doc * 13) % 103) as i32 - 51,
                i32::MIN,
            ]),
            packed(&[
                (doc % 17) as i32 - 8,
                ((doc * 13) % 103) as i32 - 51,
                i32::MAX,
            ]),
        ],
        3 if doc.is_multiple_of(7) => vec![packed(&[42])],
        _ => Vec::new(),
    }
}
fn document(doc: usize) -> Document {
    let mut result = Document::new();
    result.add(StringField::new("id", doc.to_string(), Store::Yes));
    for (field, name) in FIELDS.iter().enumerate() {
        let mut field_type = FieldType::new();
        field_type
            .set_dimensions_with_index(dimensions(field) as i32, index_dimensions(field) as i32, 4)
            .unwrap();
        field_type.freeze();
        for point in values(doc, field) {
            result.add(Field::from_bytes(*name, point, field_type.clone()).unwrap());
        }
    }
    result
}
fn writer<'a>(directory: &'a dyn Directory, asserting: bool) -> IndexWriter<'a> {
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
            .set_codec(Some(Arc::new(AssertingPointsCodec)))
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
    writer
}
fn prepare(directory: &dyn Directory, asserting: bool, merged: bool) -> IndexWriter<'_> {
    let mut writer = writer(directory, asserting);
    for doc in 0..DOCS {
        writer.add_fields_document(&document(doc)).unwrap();
        if merged && (doc + 1).is_multiple_of(DOCS / SOURCES) {
            writer.commit().unwrap();
        }
    }
    if merged {
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
    let expected_files = if merged { SOURCES * 3 } else { 0 };
    assert_eq!(
        directory
            .list_all()
            .unwrap()
            .iter()
            .filter(|name| name.ends_with(".kdm")
                || name.ends_with(".kdi")
                || name.ends_with(".kdd"))
            .count(),
        expected_files,
        "points outputs were created before the intended interval"
    );
    writer
}
fn mix(digest: u64, value: u64) -> u64 {
    (digest ^ value).wrapping_mul(0x100_0000_01b3)
}
#[derive(Clone, Copy)]
struct Check {
    digest: u64,
    docs: usize,
    encoded_bytes: u64,
    points: usize,
    packed_bytes: usize,
}
fn verify(directory: &dyn Directory, merged: bool) -> Check {
    let reader = DirectoryReader::open(directory).unwrap();
    let expected_docs: Vec<_> = (0..DOCS)
        .filter(|doc| !merged || !doc.is_multiple_of(7))
        .collect();
    assert_eq!(reader.max_doc(), expected_docs.len() as i32);
    assert_eq!(reader.num_docs(), expected_docs.len() as i32);
    assert_eq!(reader.segment_readers().len(), 1);
    let segment = &reader.segment_readers()[0];
    let points_reader = segment.points_reader().unwrap();
    let mut digest = 0xcbf2_9ce4_8422_2325;
    for (physical, expected) in expected_docs.iter().enumerate() {
        let stored = segment.stored_document(physical as i32).unwrap().unwrap();
        let id = stored
            .fields
            .iter()
            .find_map(|field| match &field.value {
                FieldValue::String(id) => Some(id.parse::<usize>().unwrap()),
                _ => None,
            })
            .unwrap();
        assert_eq!(id, *expected);
        digest = mix(digest, id as u64);
    }
    let mut point_count = 0;
    let mut packed_bytes = 0;
    for (field, name) in FIELDS.iter().enumerate() {
        let info = segment
            .field_infos()
            .fields
            .iter()
            .find(|info| info.name == *name)
            .unwrap();
        assert_eq!(info.point_dimension_count, dimensions(field) as i32);
        assert_eq!(
            info.point_index_dimension_count,
            index_dimensions(field) as i32
        );
        assert_eq!(info.point_num_bytes, 4);
        let mut expected: Vec<(i32, Vec<u8>)> = expected_docs
            .iter()
            .enumerate()
            .flat_map(|(physical, id)| {
                values(*id, field)
                    .into_iter()
                    .map(move |value| (physical as i32, value))
            })
            .collect();
        expected.sort();
        let actual = if let Some(meta) = points_reader.field(info.number) {
            assert!(!expected.is_empty());
            assert_eq!(meta.max_points_in_leaf_node, 512);
            assert!(meta.num_leaves > 1, "multi-leaf corpus required for {name}");
            assert_eq!(meta.point_count, expected.len() as i64);
            let expected_doc_count = expected_docs
                .iter()
                .filter(|id| !values(**id, field).is_empty())
                .count();
            assert_eq!(meta.doc_count, expected_doc_count as i32);
            let mut minimum = expected[0].1[..index_dimensions(field) * 4].to_vec();
            let mut maximum = minimum.clone();
            for (_, value) in &expected {
                for dim in 0..index_dimensions(field) {
                    let offset = dim * 4;
                    if value[offset..offset + 4] < minimum[offset..offset + 4] {
                        minimum[offset..offset + 4].copy_from_slice(&value[offset..offset + 4]);
                    }
                    if value[offset..offset + 4] > maximum[offset..offset + 4] {
                        maximum[offset..offset + 4].copy_from_slice(&value[offset..offset + 4]);
                    }
                }
            }
            assert_eq!(meta.min_packed_value, minimum);
            assert_eq!(meta.max_packed_value, maximum);
            let mut actual: Vec<_> = points_reader
                .decode_all_points(info.number)
                .unwrap()
                .into_iter()
                .map(|point| (point.doc_id, point.packed_value))
                .collect();
            actual.sort();
            actual
        } else {
            assert!(expected.is_empty(), "missing field {name}");
            Vec::new()
        };
        assert_eq!(actual, expected, "point multiset {name}");
        digest = mix(digest, field as u64);
        digest = mix(digest, actual.len() as u64);
        for (doc, value) in actual {
            point_count += 1;
            packed_bytes += value.len();
            digest = mix(digest, doc as u64);
            for byte in value {
                digest = mix(digest, u64::from(byte));
            }
        }
    }
    assert_eq!(point_count, if merged { 35_106 } else { 42_130 });
    assert_eq!(packed_bytes, if merged { 252_760 } else { 299_592 });
    segment.check_integrity().unwrap();
    let infos = segment_infos::read_latest(directory).unwrap();
    let encoded_bytes = infos
        .segments
        .iter()
        .flat_map(|info| {
            [
                format!("{}.kdm", info.segment_name),
                format!("{}.kdi", info.segment_name),
                format!("{}.kdd", info.segment_name),
            ]
        })
        .map(|name| directory.file_length(&name).unwrap())
        .sum();
    Check {
        digest: black_box(digest),
        docs: expected_docs.len(),
        encoded_bytes,
        points: point_count,
        packed_bytes,
    }
}

fn run(asserting: bool, merged: bool, millis: u64) -> (f64, u64, Check) {
    let mut nanos = 0;
    let mut iterations = 0;
    loop {
        let directory = ByteBuffersDirectory::new();
        let mut writer = prepare(&directory, asserting, merged);
        let started = Instant::now();
        if merged {
            writer.force_merge(1).unwrap();
        }
        writer.commit().unwrap();
        nanos += started.elapsed().as_nanos();
        iterations += 1;
        let check = verify(&directory, merged);
        if nanos >= u128::from(millis) * 1_000_000 {
            return (nanos as f64 / iterations as f64, iterations, check);
        }
    }
}
fn main() {
    let args: Vec<_> = std::env::args().collect();
    if args[1] == "--write-verify" {
        std::fs::create_dir_all(&args[2]).unwrap();
        let directory = FsDirectory::open(&args[2]);
        assert!(directory.list_all().unwrap().is_empty());
        let mut writer = prepare(&directory, true, true);
        writer.force_merge(1).unwrap();
        writer.commit().unwrap();
        let check = verify(&directory, true);
        println!(
            "#verify\t{}\t{}\t{}\t{}\t{}\t{}",
            check.digest,
            check.docs,
            FIELDS.len(),
            check.points,
            check.packed_bytes,
            check.encoded_bytes
        );
        return;
    }
    let warmup = args[1].parse().unwrap();
    let measure = args[2].parse().unwrap();
    assert!(measure > 0);
    println!("#memory_scope\tencoded points files, decoded packed value payload and shallow adapting codec only; BKD working buffers/writer/directory/allocator/RSS excluded");
    println!(
        "points_routing_asserting_codec_shallow_bytes\t{}",
        std::mem::size_of::<AssertingPointsCodec>()
    );
    for merged in [false, true] {
        for asserting in [false, true] {
            let name = format!(
                "points_routing_{}_{}_commit",
                if asserting { "asserting" } else { "default" },
                if merged { "merge" } else { "flush" }
            );
            run(asserting, merged, warmup);
            let (ns, iterations, check) = run(asserting, merged, measure);
            println!(
                "#check\t{name}\t{}\t{}\t{}\t{}\t{}",
                check.digest,
                check.docs,
                FIELDS.len(),
                check.points,
                check.packed_bytes
            );
            println!("{name}\t{ns:.3}\t{iterations}");
            println!("{name}_encoded_bytes\t{}", check.encoded_bytes);
            println!("{name}_packed_value_bytes\t{}", check.packed_bytes);
        }
    }
}
