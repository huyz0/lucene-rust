//! Java twin: LiveDocsWriterRoutingMicro. Only deletion commit is timed.
use std::{hint::black_box, sync::Arc, time::Instant};

use lucene_index::{
    buffered_updates::Term,
    document::{Document, Store, StringField},
    index_writer::{IndexWriter, DISABLE_AUTO_FLUSH_MB},
    merge_policy::log::LogDocMergePolicy,
    segment_info::LuceneVersion,
};
use lucene_search::directory_reader::DirectoryReader;
use lucene_store::{ByteBuffersDirectory, Directory, FsDirectory};
use lucene_test_framework::codecs::asserting_live_docs_writer::AssertingLiveDocsCodec;

const DOCS: usize = 1024;
const SOURCES: usize = 8;

fn deleted(doc: usize, subsequent: bool) -> bool {
    doc.is_multiple_of(7) || (subsequent && doc.is_multiple_of(11))
}

fn expected(subsequent: bool) -> (u64, usize) {
    let mut digest = 0xcbf2_9ce4_8422_2325u64;
    let mut count = 0;
    for doc in 0..DOCS {
        if !deleted(doc, subsequent) {
            digest = (digest ^ doc as u64).wrapping_mul(0x100_0000_01b3);
            count += 1;
        }
    }
    (digest, count)
}

fn prepare(directory: &dyn Directory, asserting: bool, subsequent: bool) -> IndexWriter<'_> {
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
            .set_codec(Some(Arc::new(AssertingLiveDocsCodec)))
            .unwrap();
    }
    writer.set_use_compound_file(false);
    writer
        .set_max_buffered_docs((DOCS / SOURCES) as i32)
        .unwrap();
    writer
        .set_ram_buffer_size_mb(DISABLE_AUTO_FLUSH_MB)
        .unwrap();
    writer.set_max_full_flush_merge_wait_millis(0);
    let mut policy = LogDocMergePolicy::new();
    policy.set_merge_factor(1000).unwrap();
    writer.set_pluggable_merge_policy(Some(Arc::new(policy)));
    for doc in 0..DOCS {
        let mut document = Document::new();
        document.add(StringField::new("id", doc.to_string(), Store::No));
        writer.add_fields_document(&document).unwrap();
        if (doc + 1).is_multiple_of(DOCS / SOURCES) {
            writer.flush().unwrap();
        }
    }
    writer.commit().unwrap();
    if subsequent {
        queue_deletes(&mut writer, false);
        writer.commit().unwrap();
    }
    queue_deletes(&mut writer, subsequent);
    assert_candidate_absent(directory, subsequent);
    writer
}

fn assert_candidate_absent(directory: &dyn Directory, subsequent: bool) {
    let suffix = if subsequent { "_2.liv" } else { "_1.liv" };
    assert!(
        directory
            .list_all()
            .unwrap()
            .iter()
            .all(|file| !file.ends_with(suffix)),
        "pending deletion generation was written before timing"
    );
}

fn queue_deletes(writer: &mut IndexWriter<'_>, subsequent: bool) {
    let terms: Vec<_> = (0..DOCS)
        .filter(|doc| {
            if subsequent {
                doc.is_multiple_of(11) && !doc.is_multiple_of(7)
            } else {
                doc.is_multiple_of(7)
            }
        })
        .map(|doc| Term::new("id", doc.to_string()))
        .collect();
    writer.delete_documents_by_term(&terms).unwrap();
}

fn verify(directory: &dyn Directory, subsequent: bool) -> (u64, u64) {
    let reader = DirectoryReader::open(directory).unwrap();
    let (expected_digest, expected_count) = expected(subsequent);
    assert_eq!(reader.max_doc(), DOCS as i32);
    assert_eq!(reader.num_docs(), expected_count as i32);
    assert_eq!(reader.segment_readers().len(), SOURCES);
    let mut digest = 0xcbf2_9ce4_8422_2325u64;
    let mut base = 0;
    for segment in reader.segment_readers() {
        let live = segment.live_docs().unwrap();
        for doc in 0..segment.max_doc {
            let global = base + doc as usize;
            assert_eq!(live.get_doc(doc), !deleted(global, subsequent));
            if live.get_doc(doc) {
                digest = (digest ^ global as u64).wrapping_mul(0x100_0000_01b3);
            }
        }
        base += segment.max_doc as usize;
    }
    assert_eq!(digest, expected_digest);
    let files = directory.list_all().unwrap();
    // Count current live-docs output through each committed reader, independently
    // of stale files that IndexFileDeleter implementations may retain.
    let generation = if subsequent { "2" } else { "1" };
    let suffix = format!("_{generation}.liv");
    let liv_files: Vec<_> = files
        .iter()
        .filter(|file| file.ends_with(&suffix))
        .collect();
    assert_eq!(liv_files.len(), SOURCES);
    let encoded_bytes = liv_files
        .iter()
        .map(|file| directory.file_length(file).unwrap())
        .sum();
    (black_box(digest), encoded_bytes)
}

fn run(asserting: bool, subsequent: bool, millis: u64) -> (f64, u64, u64) {
    let mut elapsed = 0;
    let mut iterations = 0;
    let bytes = loop {
        let directory = ByteBuffersDirectory::new();
        let mut writer = prepare(&directory, asserting, subsequent);
        let started = Instant::now();
        writer.commit().unwrap();
        elapsed += started.elapsed().as_nanos();
        iterations += 1;
        let (_, bytes) = verify(&directory, subsequent);
        if elapsed >= u128::from(millis) * 1_000_000 {
            break bytes;
        }
    };
    (elapsed as f64 / iterations as f64, iterations, bytes)
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    if args[1] == "--write-verify" {
        std::fs::create_dir_all(&args[2]).unwrap();
        let directory = FsDirectory::open(&args[2]);
        assert!(directory.list_all().unwrap().is_empty());
        let mut writer = prepare(&directory, true, true);
        writer.commit().unwrap();
        let (digest, bytes) = verify(&directory, true);
        println!(
            "#verify\t{digest}\t{}\t{SOURCES}\t{bytes}",
            expected(true).1
        );
        return;
    }
    let warmup = args[1].parse().unwrap();
    let measure = args[2].parse().unwrap();
    assert!(measure > 0);
    println!(
        "#memory_scope\tlive-doc encoded bytes cover current .liv generations only; codec shallow size excludes native writer/directory/allocator/RSS"
    );
    println!(
        "liv_routing_asserting_codec_shallow_bytes\t{}",
        std::mem::size_of::<AssertingLiveDocsCodec>()
    );
    for subsequent in [false, true] {
        for asserting in [false, true] {
            let name = format!(
                "liv_routing_{}_{}_commit",
                if asserting { "asserting" } else { "default" },
                if subsequent { "subsequent" } else { "first" }
            );
            let (digest, count) = expected(subsequent);
            run(asserting, subsequent, warmup);
            let (ns, iterations, bytes) = run(asserting, subsequent, measure);
            println!("#check\t{name}\t{digest}\t{count}\t{SOURCES}\t{bytes}");
            println!("{name}\t{ns:.3}\t{iterations}");
            println!("{name}_live_docs_encoded_bytes\t{bytes}");
        }
    }
}
