//! Where each vector format's random-access data starts, on every way a
//! segment reaches disk: loose files or one compound file, flushed or merged.
//! A page read under memory pressure only touches one page per vector when
//! the data starts on a page, so a layout that loses the alignment on one of
//! these paths (as the compound file once did for `.vec`) fails here.

#![allow(clippy::arithmetic_side_effects)]

use lucene_codecs::compound_format;
use lucene_codecs::field_infos::{
    FieldInfo, IndexOptions, VectorEncoding, VectorSimilarityFunction,
};
use lucene_codecs::per_field_knn_vectors::{KnnVectorsFormat, IVF_QUANTIZERS};
use lucene_index::concurrent_writer::ConcurrentIndexWriter;
use lucene_index::index_writer::{
    DocumentVector, ExplicitDocument, IndexWriter, InvertedField, InvertedTerms,
};
use lucene_index::segment_info::LuceneVersion;
use lucene_search::directory_reader::DirectoryReader;
use lucene_store::directory::{Directory, FsDirectory};
use lucene_util::test_support::TempDir;

const DIM: usize = 32;
const PER_SEGMENT: usize = 1200;
const SEGMENTS: usize = 2;
const PAGE: usize = 4096;
const VERSION: LuceneVersion = LuceneVersion {
    major: 10,
    minor: 5,
    bugfix: 0,
};

fn vector(id: usize) -> Vec<f32> {
    let mut x = (id as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..DIM)
        .map(|_| {
            x ^= x >> 29;
            x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
            ((x >> 40) as f32 / (1u64 << 24) as f32) * 8.0 - 4.0 + (id % 17) as f32
        })
        .collect()
}

fn writer(dir: &FsDirectory, format: KnnVectorsFormat, compound: bool) -> IndexWriter<'_> {
    let mut w = IndexWriter::open(dir, Vec::new(), "Lucene104", VERSION).unwrap();
    w.enable_explicit_documents().unwrap();
    w.register_field(
        FieldInfo::new("t", 0)
            .with_index_options(IndexOptions::Docs)
            .with_omit_norms(true),
    )
    .unwrap();
    w.register_field(FieldInfo::new("v", 0).with_vectors(
        DIM as i32,
        VectorEncoding::Float32,
        VectorSimilarityFunction::Euclidean,
    ))
    .unwrap();
    w.set_knn_vectors_format_for_field("v", format);
    w.set_use_compound_file(compound);
    // A merged segment is compound too, whatever its share of the index.
    let mut settings = lucene_index::merge_policy::CompoundFileSettings::default();
    settings
        .set_no_cfs_ratio(if compound { 1.0 } else { 0.0 })
        .unwrap();
    w.set_merge_compound_file_settings(Some(settings));
    w.set_ram_buffer_size_mb(256.0).unwrap();
    w.set_merge_policy(None);
    w
}

fn write(dir: &FsDirectory, format: impl Fn() -> KnnVectorsFormat, compound: bool, merged: bool) {
    let w = ConcurrentIndexWriter::new(writer(dir, format(), compound), 1).unwrap();
    for s in 0..SEGMENTS {
        let (docs, vectors): (Vec<_>, Vec<_>) = (s * PER_SEGMENT..(s + 1) * PER_SEGMENT)
            .map(|id| {
                let mut d = ExplicitDocument::default();
                let mut terms = InvertedTerms::default();
                terms.push::<Vec<u8>>(b"x", 1, &[], &[], &[]);
                d.fields.inverted.push(InvertedField {
                    field_number: 0,
                    terms,
                    norm: None,
                });
                (d, vec![DocumentVector::float32("v", vector(id))])
            })
            .unzip();
        w.add_explicit_documents_with_vectors_in(None, docs, vectors)
            .unwrap();
        w.flush().unwrap();
    }
    w.commit().unwrap();
    w.into_writer().unwrap().close().unwrap();
    if merged {
        let mut w = writer(dir, format(), compound);
        w.force_merge(1).unwrap();
        w.commit().unwrap();
        w.close().unwrap();
    }
}

/// Each segment's `(file kind, offset of its data in the physical file)`:
/// the member's start in the `.cfs` (zero for a loose file) plus where the
/// format says the data starts in the member.
fn data_starts(dir: &FsDirectory, expect_compound: bool) -> Vec<(String, usize)> {
    let reader = DirectoryReader::open(dir).unwrap();
    let infos = lucene_index::segment_infos::read_latest(dir).unwrap();
    let mut out = Vec::new();
    for (seg, info) in reader.segment_readers().iter().zip(&infos.segments) {
        let member_start = |ext: &str| -> usize {
            let cfe = dir
                .list_all()
                .unwrap()
                .into_iter()
                .find(|f| f.ends_with(".cfe") && f.starts_with(&seg.segment_name));
            match cfe {
                Some(cfe) => {
                    assert!(expect_compound, "{cfe} in a loose run");
                    let entries =
                        compound_format::parse_entries(&dir.open(&cfe).unwrap(), &info.segment_id)
                            .unwrap();
                    let (_, e) = entries
                        .entries
                        .iter()
                        .find(|(n, _)| n.ends_with(ext))
                        .unwrap_or_else(|| panic!("no {ext} member"));
                    e.offset as usize
                }
                None => {
                    assert!(!expect_compound, "{} is not compound", seg.segment_name);
                    0
                }
            }
        };
        let number = seg
            .field_infos()
            .fields
            .iter()
            .find(|f| f.name == "v")
            .unwrap()
            .number;
        if let Some(flat) = seg.flat_vectors_reader_for(number).unwrap() {
            let entry = flat.field(number).unwrap();
            out.push((
                format!("{} .vec", seg.segment_name),
                member_start(".vec") + entry.vector_data_offset as usize,
            ));
        }
        if let Some(ivf) = seg.ivf_for(number).unwrap() {
            out.push((
                format!("{} .ivv", seg.segment_name),
                member_start(".ivv") + ivf.field(number).unwrap().vectors_offset(),
            ));
        }
        if let Some(disk) = seg.disk_hnsw_for(number).unwrap() {
            out.push((
                format!("{} .dhn", seg.segment_name),
                member_start(".dhn") + disk.field(number).unwrap().nodes_offset(),
            ));
        }
    }
    out
}

fn check(name: &str, format: impl Fn() -> KnnVectorsFormat, kinds: &[&str]) {
    for compound in [false, true] {
        for merged in [false, true] {
            let tmp = TempDir::new("vector-layout");
            let dir = FsDirectory::open(&tmp);
            write(&dir, &format, compound, merged);
            let starts = data_starts(&dir, compound);
            for kind in kinds {
                let of_kind: Vec<_> = starts.iter().filter(|(n, _)| n.ends_with(kind)).collect();
                assert!(
                    !of_kind.is_empty(),
                    "{name}: no {kind} (compound {compound}, merged {merged}): {starts:?}"
                );
                for (segment, at) in of_kind {
                    assert_eq!(
                        at % PAGE,
                        0,
                        "{name}: {segment} data at {at} (compound {compound}, merged {merged})"
                    );
                }
            }
        }
    }
}

#[test]
fn hnsw_flat_vectors_start_on_a_page_in_every_layout() {
    check(
        "hnsw",
        || KnnVectorsFormat::hnsw(16, 100).unwrap(),
        &[".vec"],
    );
}

#[test]
fn ivf_rerank_vectors_start_on_a_page_in_every_layout() {
    check("ivf", || KnnVectorsFormat::ivf(64, 0).unwrap(), &[".ivv"]);
}

#[test]
fn disk_hnsw_node_records_start_on_a_page_in_every_layout() {
    let bbq = IVF_QUANTIZERS.iter().position(|&n| n == "bbq").unwrap() as u8;
    check(
        "disk hnsw",
        || KnnVectorsFormat::hnsw_disk(bbq, 16, 100).unwrap(),
        &[".vec", ".dhn"],
    );
}

/// A compound segment read through the pager is read as its loose files
/// are: the vector data of its members is paged (faults are served, within
/// a budget) and searches return what the files read directly return.
#[test]
fn a_compound_segment_pages_its_vector_files_and_searches_the_same() {
    use lucene_search::knn_search::{knn_top_hits, KnnTarget};
    use lucene_store::{PagedFsConfig, PagedFsDirectory};

    let tmp = TempDir::new("vector-layout-paged");
    let dir = FsDirectory::open(&tmp);
    write(
        &dir,
        || KnnVectorsFormat::hnsw(16, 100).unwrap(),
        true,
        false,
    );
    let search = |reader: &DirectoryReader| {
        let opened = reader.open_segments().unwrap();
        let segments = opened.as_open_segments();
        (0..10)
            .map(|q| {
                knn_top_hits(
                    &segments,
                    "v",
                    &KnnTarget::Float(vector(5_000 + q)),
                    10,
                    50,
                    None,
                )
                .unwrap()
                .iter()
                .map(|h| (h.doc_id, h.score.to_bits()))
                .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
    };
    let direct = search(&DirectoryReader::open(&dir).unwrap());
    let paged = PagedFsDirectory::with_config(tmp.path(), PagedFsConfig::vectors_only());
    if !paged.is_paged() {
        return;
    }
    let pager = lucene_store::paged_input::Pager::global().unwrap();
    let before = pager.resident_bytes();
    let reader = DirectoryReader::open(&paged).unwrap();
    let through_pager = search(&reader);
    // The vectors themselves, read from the first segment.
    let seg = &reader.segment_readers()[0];
    let number = seg
        .field_infos()
        .fields
        .iter()
        .find(|f| f.name == "v")
        .unwrap()
        .number;
    let flat = seg.flat_vectors_reader_for(number).unwrap().unwrap();
    let values = flat.float_vector_values(number).unwrap();
    assert_eq!(values.vector(7).unwrap().len(), DIM);
    assert_eq!(direct, through_pager);
    assert!(
        pager.resident_bytes() > before,
        "the compound segment's vectors were not read through the pager"
    );
}
