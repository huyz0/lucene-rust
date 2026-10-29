//! Modelled on Lucene 10.5.2's `TestHnswGraphReuseDuringMerge`
//! (GITHUB#16400/#16403): merging segments written under the default codec,
//! whose vectors go through `PerFieldKnnVectorsFormat`, must reuse their HNSW
//! graphs instead of rebuilding the merged graph from scratch -- both when the
//! sources are deletion-free and when the base segment carries deletions
//! below `IncrementalHnswGraphMerger`'s 40% threshold.
//!
//! Java asserts on `MergingHnswGraphBuilder`'s InfoStream message; this port
//! has no HNSW InfoStream, so the merged segment's graph is compared arc for
//! arc with what `hnsw_vectors::merge_one_field` produces from the source
//! graphs (the reuse path), and checked to differ from a from-scratch build
//! over the same vectors.

// Test-support code opts out of the arithmetic gate at the file boundary:
// see `docs/arithmetic-gate.md`.
#![allow(clippy::arithmetic_side_effects)]

use lucene_codecs::field_infos::VectorSimilarityFunction;
use lucene_codecs::hnsw::{self, HnswGraphView, OnHeapHnswGraph};
use lucene_codecs::hnsw_vectors::{self, GraphMergeSource, HnswVectorsReader};
use lucene_codecs::vectors::FlatVectorsReader;
use lucene_index::buffered_updates::Term;
use lucene_index::document::{Document, KnnFloatVectorField, Store, StringField};
use lucene_index::index_writer::IndexWriter;
use lucene_index::segment_info::LuceneVersion;
use lucene_index::segment_infos;
use lucene_store::{Directory, FsDirectory};
use lucene_util::test_support::TempDir;

const DIM: usize = 8;
/// Above the tiny-segment cutoff of `shouldCreateGraph`, so graphs are built
/// at both flush and merge time.
const DOCS_PER_SEGMENT: usize = 1500;

/// One segment's vector files, read into memory before the merge deletes them.
struct SegmentVectors {
    id: [u8; 16],
    suffix: String,
    vec: Vec<u8>,
    vemf: Vec<u8>,
    vem: Vec<u8>,
    vex: Vec<u8>,
}

fn snapshot(dir: &dyn Directory) -> Vec<SegmentVectors> {
    let sis = segment_infos::read_latest(dir).unwrap();
    let files = dir.list_all().unwrap();
    sis.segments
        .iter()
        .map(|sci| {
            let prefix = format!("{}_", sci.segment_name);
            let vec_file = files
                .iter()
                .find(|f| f.starts_with(&prefix) && f.ends_with(".vec"))
                .unwrap_or_else(|| panic!("no .vec for {} in {files:?}", sci.segment_name));
            let stem = vec_file.trim_end_matches(".vec");
            let read = |ext: &str| dir.open(&format!("{stem}.{ext}")).unwrap().to_vec();
            SegmentVectors {
                id: sci.segment_id,
                suffix: stem[prefix.len()..].to_string(),
                vec: read("vec"),
                vemf: read("vemf"),
                vem: read("vem"),
                vex: read("vex"),
            }
        })
        .collect()
}

fn arcs<G: HnswGraphView>(graph: &G) -> Vec<(i32, i32, Vec<i32>)> {
    let mut out = Vec::new();
    let mut neighbors = Vec::new();
    for level in 0..graph.num_levels() {
        for node in graph.sorted_nodes_on_level(level).unwrap() {
            graph.neighbors_into(level, node, &mut neighbors).unwrap();
            let mut sorted = neighbors.clone();
            sorted.sort_unstable();
            out.push((level, node, sorted));
        }
    }
    out
}

fn add_vector_docs(w: &mut IndexWriter<'_>, num_docs: usize, id_prefix: Option<&str>, seed: u64) {
    let mut state = seed;
    for i in 0..num_docs {
        let v: Vec<f32> = (0..DIM)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                (state >> 40) as f32 / (1u64 << 24) as f32
            })
            .collect();
        let mut doc = Document::new();
        doc.add(KnnFloatVectorField::new("v", v, VectorSimilarityFunction::Euclidean).unwrap());
        if let Some(prefix) = id_prefix {
            doc.add(StringField::new("id", format!("{prefix}{i}"), Store::No));
        }
        w.add_fields_document(&doc).unwrap();
    }
}

fn open_writer(dir: &FsDirectory) -> IndexWriter<'_> {
    let version = LuceneVersion {
        major: 10,
        minor: 5,
        bugfix: 0,
    };
    IndexWriter::open(dir, Vec::new(), "Lucene104", version).unwrap()
}

/// Checks the single merged segment's graph against `merge_one_field` over
/// `sources` (reuse) and against a from-scratch build (no source graphs).
/// `deleted[i][doc]` says whether source `i`'s `doc` was deleted before the
/// merge. The merge policy picks the order the sources are merged in, so it
/// is recovered from the merged vectors: the order whose live vectors,
/// concatenated, are the merged segment's.
fn assert_merged_graph_reuses_sources(
    dir: &dyn Directory,
    sources: &[SegmentVectors],
    deleted: &[Vec<bool>],
) {
    let merged = snapshot(dir);
    assert_eq!(merged.len(), 1, "force_merge(1) leaves one segment");
    let merged = &merged[0];
    let flat =
        FlatVectorsReader::open(&merged.vemf, &merged.vec, &merged.id, &merged.suffix).unwrap();
    let field = flat.fields()[0].field_number;
    let values = flat.float_vector_values(field).unwrap();
    let merged_ord_to_doc: Vec<i32> = (0..values.size())
        .map(|ord| values.ord_to_doc(ord).unwrap())
        .collect();
    let written =
        HnswVectorsReader::open(&merged.vem, &merged.vex, &merged.id, &merged.suffix).unwrap();
    let written_graph = written.graph(field).unwrap().expect("merged graph");
    let m = written_graph.max_conn();

    let source_flats: Vec<FlatVectorsReader<'_>> = sources
        .iter()
        .map(|s| FlatVectorsReader::open(&s.vemf, &s.vec, &s.id, &s.suffix).unwrap())
        .collect();
    let source_readers: Vec<HnswVectorsReader<'_>> = sources
        .iter()
        .map(|s| HnswVectorsReader::open(&s.vem, &s.vex, &s.id, &s.suffix).unwrap())
        .collect();
    let source_graphs: Vec<_> = source_readers
        .iter()
        .map(|r| r.graph(field).unwrap().expect("source graph"))
        .collect();
    let ord_to_docs: Vec<Vec<i32>> = source_flats
        .iter()
        .map(|f| {
            let v = f.float_vector_values(field).unwrap();
            (0..v.size())
                .map(|ord| v.ord_to_doc(ord).unwrap())
                .collect()
        })
        .collect();
    // `MergeState.docMaps` for one merge order; `None` when the merged
    // vectors are not the sources' live vectors in that order.
    let doc_maps_for = |order: &[usize]| -> Option<Vec<Vec<i32>>> {
        let mut maps = vec![Vec::new(); sources.len()];
        let mut next = 0i32;
        for &i in order {
            let src = source_flats[i].float_vector_values(field).unwrap();
            for (doc, gone) in deleted[i].iter().enumerate() {
                if *gone {
                    maps[i].push(-1);
                    continue;
                }
                if next >= values.size()
                    || src.vector(doc as i32).unwrap() != values.vector(next).unwrap()
                {
                    return None;
                }
                maps[i].push(next);
                next += 1;
            }
        }
        (next == values.size()).then_some(maps)
    };
    let (order, doc_maps) = [vec![0usize, 1], vec![1, 0]]
        .into_iter()
        .find_map(|o| doc_maps_for(&o).map(|m| (o, m)))
        .expect("the merged vectors are the sources' live vectors in some order");
    let graph_sources = |with_graphs: bool| -> Vec<GraphMergeSource<'_, _>> {
        order
            .iter()
            .map(|&i| GraphMergeSource {
                graph: with_graphs.then_some(&source_graphs[i]),
                ord_to_doc: &ord_to_docs[i],
                doc_map: &doc_maps[i],
            })
            .collect()
    };
    let build = |with_graphs: bool| -> OnHeapHnswGraph {
        hnsw_vectors::merge_one_field(
            values.ord_scorer(),
            m,
            hnsw::DEFAULT_BEAM_WIDTH,
            hnsw::DEFAULT_RAND_SEED,
            &merged_ord_to_doc,
            &graph_sources(with_graphs),
        )
        .unwrap()
        .expect("above the graph threshold")
    };
    let reused = build(true);
    let rebuilt = build(false);
    assert_ne!(
        arcs(&reused),
        arcs(&rebuilt),
        "reuse and rebuild must be distinguishable for this test to mean anything"
    );
    assert!(
        arcs(&written_graph) == arcs(&reused),
        "the merged graph was not built by joining the source graphs"
    );
}

#[test]
fn merge_reuses_graphs_under_the_default_codec() {
    let tmp = TempDir::new("hnsw-graph-reuse");
    let dir = FsDirectory::open(&tmp);
    let mut w = open_writer(&dir);
    for s in 0..2u64 {
        add_vector_docs(&mut w, DOCS_PER_SEGMENT, None, 17 + s);
        w.commit().unwrap();
    }
    let sources = snapshot(&dir);
    assert_eq!(sources.len(), 2);
    assert!(
        sources[0].suffix.starts_with("Lucene99HnswVectorsFormat"),
        "routed through PerFieldKnnVectorsFormat: {}",
        sources[0].suffix
    );
    w.force_merge(1).unwrap();
    w.commit().unwrap();
    let none = vec![false; DOCS_PER_SEGMENT];
    assert_merged_graph_reuses_sources(&dir, &sources, &[none.clone(), none]);
}

/// A base graph with deletions below the delete-percentage threshold is still
/// chosen as the base, so the merged graph joins both source graphs.
#[test]
fn merge_reuses_a_base_graph_with_deletions() {
    // Enough docs that the base segment's live count after the deletions
    // still exceeds the other segment's, keeping it the base graph.
    let base_docs = 2000usize;
    let tmp = TempDir::new("hnsw-graph-reuse-deletes");
    let dir = FsDirectory::open(&tmp);
    let mut w = open_writer(&dir);
    add_vector_docs(&mut w, base_docs, Some("d"), 5);
    w.commit().unwrap();
    add_vector_docs(&mut w, DOCS_PER_SEGMENT, None, 6);
    w.commit().unwrap();
    let sources = snapshot(&dir);
    assert_eq!(sources.len(), 2);

    // Delete 20% of the base segment, below the 40% threshold.
    let terms: Vec<Term> = (0..base_docs)
        .step_by(5)
        .map(|i| Term::new("id", format!("d{i}")))
        .collect();
    w.delete_documents_by_term(&terms).unwrap();
    // Committed before merging so the deletions are applied to the base
    // segment the merge reads (Java's `forceMerge` applies them by flushing;
    // this writer's `force_merge` resolves still-buffered deletions against
    // the merged segment instead, which would merge a deletion-free base).
    w.commit().unwrap();
    w.force_merge(1).unwrap();
    w.commit().unwrap();

    let base_deleted: Vec<bool> = (0..base_docs).map(|i| i % 5 == 0).collect();
    assert_eq!(
        base_deleted.iter().filter(|d| !**d).count(),
        base_docs - base_docs / 5
    );
    assert_merged_graph_reuses_sources(
        &dir,
        &sources,
        &[base_deleted, vec![false; DOCS_PER_SEGMENT]],
    );
}
