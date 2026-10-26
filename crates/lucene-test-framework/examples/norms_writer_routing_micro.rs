//! Whole IndexWriter flush/merge commits through native or installed asserting norms.
//! Norm encoding, consumer close, sibling formats and publication are timed together.
use lucene_codecs::{norms, stored_fields::FieldValue};
use lucene_index::{
    buffered_updates::Term,
    document::{Document, Store, StringField, TextField},
    index_writer::{IndexWriter, DISABLE_AUTO_FLUSH_MB},
    merge_policy::log::LogDocMergePolicy,
    segment_info::LuceneVersion,
    segment_infos,
};
use lucene_search::directory_reader::DirectoryReader;
use lucene_store::{ByteBuffersDirectory, Directory, FsDirectory};
use lucene_test_framework::codecs::asserting_norms_writer::AssertingNormsCodec;
use std::{hint::black_box, sync::Arc, time::Instant};
const DOCS: usize = 1024;
const SOURCES: usize = 8;
const FIELDS: [&str; 4] = ["constant", "varying", "sparse", "vanish"];
fn value(doc: usize, field: usize) -> Option<i64> {
    match field {
        0 => Some(1),
        1 => Some((doc % 5) as i64),
        2 => (!doc.is_multiple_of(3)).then_some(if doc.is_multiple_of(5) { 0 } else { 3 }),
        3 => doc.is_multiple_of(7).then_some(2),
        _ => unreachable!(),
    }
}
fn document(doc: usize) -> Document {
    let mut result = Document::new();
    result.add(StringField::new("id", doc.to_string(), Store::Yes));
    for (field, name) in FIELDS.iter().enumerate() {
        if let Some(length) = value(doc, field) {
            let text = std::iter::repeat_n("x", length as usize)
                .collect::<Vec<_>>()
                .join(" ");
            result.add(TextField::new(*name, text, Store::No));
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
            .set_codec(Some(Arc::new(AssertingNormsCodec)))
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
    let expected_files = if merged { SOURCES * 2 } else { 0 };
    assert_eq!(
        directory
            .list_all()
            .unwrap()
            .iter()
            .filter(|name| name.ends_with(".nvm") || name.ends_with(".nvd"))
            .count(),
        expected_files,
        "norms outputs were created before the intended interval"
    );
    writer
}
fn mix(digest: u64, value: u64) -> u64 {
    (digest ^ value).wrapping_mul(0x100_0000_01b3)
}
fn verify(directory: &dyn Directory, merged: bool) -> (u64, usize, u64) {
    let reader = DirectoryReader::open(directory).unwrap();
    let expected_docs: Vec<_> = (0..DOCS)
        .filter(|doc| !merged || !doc.is_multiple_of(7))
        .collect();
    assert_eq!(reader.max_doc(), expected_docs.len() as i32);
    assert_eq!(reader.num_docs(), expected_docs.len() as i32);
    assert_eq!(reader.segment_readers().len(), 1);
    let segment = &reader.segment_readers()[0];
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
        for (field, name) in FIELDS.iter().enumerate() {
            let info = segment
                .field_infos()
                .fields
                .iter()
                .find(|info| info.name == *name)
                .unwrap();
            let actual = norms::norm_value(
                segment.norms_data().unwrap(),
                segment.norms_entry(info.number).unwrap(),
                physical as i32,
            )
            .unwrap();
            assert_eq!(actual, value(id, field), "norm {name} doc {id}");
            digest = mix(digest, u64::from(actual.is_some()));
            digest = mix(digest, actual.unwrap_or(0) as u64);
        }
    }
    segment.check_integrity().unwrap();
    let infos = segment_infos::read_latest(directory).unwrap();
    let bytes = infos
        .segments
        .iter()
        .flat_map(|info| {
            [
                format!("{}.nvm", info.segment_name),
                format!("{}.nvd", info.segment_name),
            ]
        })
        .map(|name| directory.file_length(&name).unwrap())
        .sum();
    (black_box(digest), expected_docs.len(), bytes)
}
fn run(asserting: bool, merged: bool, millis: u64) -> (f64, u64, u64, usize, u64) {
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
        let (digest, docs, bytes) = verify(&directory, merged);
        if nanos >= u128::from(millis) * 1_000_000 {
            return (
                nanos as f64 / iterations as f64,
                iterations,
                digest,
                docs,
                bytes,
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
        let mut writer = prepare(&directory, true, true);
        writer.force_merge(1).unwrap();
        writer.commit().unwrap();
        let (digest, docs, bytes) = verify(&directory, true);
        println!("#verify\t{digest}\t{docs}\t{}\t{bytes}", FIELDS.len());
        return;
    }
    let warmup = args[1].parse().unwrap();
    let measure = args[2].parse().unwrap();
    assert!(measure > 0);
    println!("#memory_scope\tcurrent encoded norms files and shallow adapting codec only; writer/directory/allocator/RSS excluded");
    println!(
        "norms_routing_asserting_codec_shallow_bytes\t{}",
        std::mem::size_of::<AssertingNormsCodec>()
    );
    for merged in [false, true] {
        for asserting in [false, true] {
            let name = format!(
                "norms_routing_{}_{}_commit",
                if asserting { "asserting" } else { "default" },
                if merged { "merge" } else { "flush" }
            );
            run(asserting, merged, warmup);
            let (ns, iterations, digest, docs, bytes) = run(asserting, merged, measure);
            println!(
                "#check\t{name}\t{digest}\t{docs}\t{}\t{bytes}",
                FIELDS.len()
            );
            println!("{name}\t{ns:.3}\t{iterations}");
            println!("{name}_encoded_bytes\t{bytes}");
        }
    }
}
