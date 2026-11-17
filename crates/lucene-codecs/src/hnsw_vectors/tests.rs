#![cfg(test)]
// The arithmetic gate is about values read off disk; a test's `i + 1` is
// not one. See docs/arithmetic-gate.md.
#![allow(clippy::arithmetic_side_effects)]
use super::*;
use crate::hnsw::{HnswGraphBuilder, KnnCollector, UpdateableVectorScorer};

const ID: [u8; ID_LENGTH] = *b"hnswvectorstest1";

/// A one-dimensional scorer, so "nearest" is readable off the page.
#[derive(Debug, Clone)]
struct Line {
    values: Vec<f32>,
    query: f32,
}

impl VectorScorer for Line {
    fn score(&mut self, node: i32) -> Result<f32> {
        let d = self.values[node as usize] - self.query;
        Ok(1.0 / (1.0 + d * d))
    }

    fn max_ord(&self) -> i32 {
        self.values.len() as i32
    }
}

impl UpdateableVectorScorer for Line {
    fn set_scoring_ordinal(&mut self, ord: i32) -> Result<()> {
        self.query = self.values[ord as usize];
        Ok(())
    }
}

/// The unfiltered, unseeded defaults -- Java's plain
/// `KnnFloatVectorQuery` arguments.
fn opts<'a>() -> SearchOptions<'a> {
    SearchOptions::default()
}

fn line(n: usize) -> Line {
    Line {
        values: (0..n).map(|i| i as f32).collect(),
        query: 0.0,
    }
}

fn build(n: usize) -> OnHeapHnswGraph {
    HnswGraphBuilder::new(line(n), 8, 32, crate::hnsw::DEFAULT_RAND_SEED)
        .unwrap()
        .build(n as i32)
        .unwrap()
}

fn field<'g>(graph: Option<&'g OnHeapHnswGraph>, count: i32) -> HnswVectorsField<'g> {
    HnswVectorsField {
        field_number: 0,
        encoding: VectorEncoding::Float32,
        similarity: VectorSimilarityFunction::Euclidean,
        dimension: 1,
        count,
        graph,
        m: 8,
    }
}

fn repack(buf: &[u8], patch: impl FnOnce(&mut Vec<u8>)) -> Vec<u8> {
    let mut body = buf[..buf.len() - codec_util::FOOTER_LENGTH].to_vec();
    patch(&mut body);
    codec_util::write_footer(&mut body);
    body
}

fn open(meta: &[u8], index: &[u8]) -> Result<HnswVectorsReader<'static>> {
    let index: &'static [u8] = Box::leak(index.to_vec().into_boxed_slice());
    HnswVectorsReader::open(meta, index, &ID, "")
}

#[test]
fn a_graph_round_trips_arc_for_arc() {
    let graph = build(300);
    assert!(graph.num_levels() >= 2);
    let (vex, vem) = write_hnsw_vectors(&[field(Some(&graph), 300)], &ID, "").unwrap();
    let reader = HnswVectorsReader::open(&vem, &vex, &ID, "").unwrap();
    assert_eq!(reader.version(), VERSION_CURRENT);
    assert_eq!(reader.fields().len(), 1);
    assert!(reader.field(9).is_none());
    let entry = reader.field(0).unwrap();
    assert_eq!(entry.dimension, 1);
    assert_eq!(entry.size, 300);
    assert_eq!(entry.m, 8);
    assert_eq!(entry.encoding, VectorEncoding::Float32);
    assert_eq!(entry.similarity, VectorSimilarityFunction::Euclidean);

    let off = reader.graph(0).unwrap().unwrap();
    assert_eq!(off.size(), graph.size());
    assert_eq!(off.num_levels(), graph.num_levels());
    assert_eq!(off.entry_node(), graph.entry_node());
    assert_eq!(off.max_conn(), graph.max_conn());
    let mut want = Vec::new();
    let mut got = Vec::new();
    for level in 0..graph.num_levels() {
        assert_eq!(
            off.sorted_nodes_on_level(level).unwrap(),
            graph.sorted_nodes_on_level(level).unwrap()
        );
        for node in graph.sorted_nodes_on_level(level).unwrap() {
            graph.neighbors_into(level, node, &mut want).unwrap();
            want.sort_unstable();
            want.dedup();
            off.neighbors_into(level, node, &mut got).unwrap();
            assert_eq!(got, want, "level {level} node {node}");
        }
    }
}

/// The prefetch hints walk off no edge: every node, out-of-range ones,
/// upper levels, and a graph whose offsets point past its `.vex`.
#[test]
fn neighbor_prefetch_ignores_any_bad_target() {
    let graph = build(300);
    let (vex, vem) = write_hnsw_vectors(&[field(Some(&graph), 300)], &ID, "").unwrap();
    let reader = HnswVectorsReader::open(&vem, &vex, &ID, "").unwrap();
    let off = reader.graph(0).unwrap().unwrap();
    for level in -1..=off.num_levels() {
        for node in [-1, 0, 1, 150, 299, 300, i32::MAX, i32::MIN] {
            off.prefetch_neighbors(level, node);
            graph.prefetch_neighbors(level, node);
        }
    }
    // The same graph over a `.vex` cut short: offsets now point past it.
    let entry = reader.field(0).unwrap();
    let cut = &vex[..entry.vector_index_offset as usize + 8];
    if let Ok(short) = OffHeapHnswGraph::new(entry, cut, reader.version()) {
        for node in 0..300 {
            short.prefetch_neighbors(0, node);
        }
    }
}

/// A field Lucene decided was too small for a graph: `numLevels == 0`, a
/// zero-length `.vex` region, and a reader that must not go looking for
/// node offsets that were never written.
#[test]
fn a_field_without_a_graph_round_trips() {
    let (vex, vem) = write_hnsw_vectors(&[field(None, 5)], &ID, "").unwrap();
    let reader = HnswVectorsReader::open(&vem, &vex, &ID, "").unwrap();
    let entry = reader.field(0).unwrap();
    assert_eq!(entry.num_levels, 0);
    assert_eq!(entry.vector_index_length, 0);
    assert_eq!(entry.m, 8);
    assert!(reader.graph(0).unwrap().is_none());
    assert!(matches!(reader.graph(7), Err(Error::UnknownField(7))));
}

#[test]
fn several_fields_share_one_file_pair() {
    let a = build(200);
    let b = build(150);
    let mut fa = field(Some(&a), 200);
    let mut fb = field(Some(&b), 150);
    fb.field_number = 4;
    fb.encoding = VectorEncoding::Byte;
    fb.similarity = VectorSimilarityFunction::Cosine;
    fa.field_number = 2;
    let (vex, vem) = write_hnsw_vectors(&[fa, fb], &ID, "").unwrap();
    let reader = HnswVectorsReader::open(&vem, &vex, &ID, "").unwrap();
    assert_eq!(reader.fields().len(), 2);
    assert_eq!(reader.field(4).unwrap().encoding, VectorEncoding::Byte);
    let ga = reader.graph(2).unwrap().unwrap();
    let gb = reader.graph(4).unwrap().unwrap();
    assert_eq!(ga.size(), 200);
    assert_eq!(gb.size(), 150);
    let mut out = Vec::new();
    // The second field's region starts partway into `.vex`; a wrong
    // `vectorIndexOffset` decodes the first field's neighbours here.
    gb.neighbors_into(0, 0, &mut out).unwrap();
    let mut want = Vec::new();
    b.neighbors_into(0, 0, &mut want).unwrap();
    want.sort_unstable();
    want.dedup();
    assert_eq!(out, want);
}

#[test]
fn writer_rejects_a_level_zero_that_is_not_every_node() {
    let graph = build(200);
    // A `count` that disagrees with the graph is the writer-side shape of
    // Java's `assert nodesOnLevel.size() == count`.
    assert!(matches!(
        write_hnsw_vectors(&[field(Some(&graph), 199)], &ID, ""),
        Err(Error::CorruptMeta(_))
    ));
}

/// The writer must not be able to emit a `.vem` its own reader rejects --
/// a `M` past `MAXIMUM_MAX_CONN` builds a graph fine and then makes the
/// segment unopenable by this port *and* by Lucene, whose
/// `OffHeapHnswGraph` allocates `int[M * 2]` on the value.
#[test]
fn writer_rejects_parameters_its_own_reader_would_refuse() {
    let graph = build(200);
    let mut bad_dim = field(Some(&graph), 200);
    bad_dim.dimension = 0;
    assert!(matches!(
        write_hnsw_vectors(&[bad_dim], &ID, ""),
        Err(Error::InvalidGraphParameter(_))
    ));

    let mut bad_count = field(None, -1);
    bad_count.count = -1;
    assert!(matches!(
        write_hnsw_vectors(&[bad_count], &ID, ""),
        Err(Error::InvalidGraphParameter(_))
    ));

    // No graph, so `M` comes from the field rather than the graph.
    let mut bad_m = field(None, 5);
    bad_m.m = crate::hnsw::MAXIMUM_MAX_CONN + 1;
    assert!(matches!(
        write_hnsw_vectors(&[bad_m], &ID, ""),
        Err(Error::InvalidGraphParameter(_))
    ));
    let mut zero_m = field(None, 5);
    zero_m.m = 0;
    assert!(matches!(
        write_hnsw_vectors(&[zero_m], &ID, ""),
        Err(Error::InvalidGraphParameter(_))
    ));
}

#[test]
fn seeking_a_node_that_is_not_on_the_level_is_rejected() {
    let graph = build(300);
    let (vex, vem) = write_hnsw_vectors(&[field(Some(&graph), 300)], &ID, "").unwrap();
    let reader = HnswVectorsReader::open(&vem, &vex, &ID, "").unwrap();
    let off = reader.graph(0).unwrap().unwrap();
    let top = off.num_levels() - 1;
    let on_top = off.sorted_nodes_on_level(top).unwrap();
    let missing = (0..300).find(|n| !on_top.contains(n)).unwrap();
    let mut out = Vec::new();
    assert!(matches!(
        off.neighbors_into(top, missing, &mut out),
        Err(Error::CorruptMeta(_))
    ));
    assert!(matches!(
        off.neighbors_into(top + 5, 0, &mut out),
        Err(Error::CorruptMeta(_))
    ));
    assert!(matches!(
        off.sorted_nodes_on_level(top + 5),
        Err(Error::CorruptMeta(_))
    ));
}

// ---------------- search ----------------

#[test]
fn search_over_a_graph_finds_the_nearest_neighbours() {
    let graph = build(300);
    let (vex, vem) = write_hnsw_vectors(&[field(Some(&graph), 300)], &ID, "").unwrap();
    let reader = HnswVectorsReader::open(&vem, &vex, &ID, "").unwrap();
    let off = reader.graph(0).unwrap().unwrap();
    let mut scorer = line(300);
    scorer.query = 150.4;
    let (hits, _) = search(&mut scorer, Some(&off), 3, u64::MAX, opts()).unwrap();
    let got: Vec<i32> = hits.into_iter().map(|(n, _)| n).collect();
    assert_eq!(got, vec![150, 151, 149]);
}

#[test]
fn search_falls_back_to_an_exhaustive_scan_without_a_graph() {
    let mut scorer = line(300);
    scorer.query = 7.2;
    // `graph: None` is the tiny-segment case.
    let (hits, _) = search::<OnHeapHnswGraph, _>(&mut scorer, None, 3, u64::MAX, opts()).unwrap();
    let got: Vec<i32> = hits.into_iter().map(|(n, _)| n).collect();
    assert_eq!(got, vec![7, 8, 6]);
}

/// Java takes the exhaustive branch whenever a graph walk would visit at
/// least as many nodes as the field holds -- so a large `k` over a small
/// field never touches the graph, and must still be exact.
#[test]
fn a_large_k_takes_the_exhaustive_branch_and_is_exact() {
    let graph = build(120);
    let (vex, vem) = write_hnsw_vectors(&[field(Some(&graph), 120)], &ID, "").unwrap();
    let reader = HnswVectorsReader::open(&vem, &vex, &ID, "").unwrap();
    let off = reader.graph(0).unwrap().unwrap();
    assert!(expected_visited_nodes(60, 120) >= 120);
    let mut scorer = line(120);
    scorer.query = 60.0;
    let (hits, _) = search(&mut scorer, Some(&off), 60, u64::MAX, opts()).unwrap();
    assert_eq!(hits.len(), 60);
    assert_eq!(hits[0].0, 60);
}

/// The two arguments a caller derives from something outside this
/// segment, and the two that would therefore turn a caller mistake into a
/// panic rather than an error: an accept set shorter than the field
/// (`FixedBitSet::get` past the end is an out-of-bounds index) and a
/// negative "cardinality". Java has neither check -- its `Bits` is
/// unbounded by construction and its `filteredDocCount` is an `assert`.
#[test]
fn a_short_accept_set_or_a_negative_filtered_doc_count_is_an_error_not_a_panic() {
    let mut scorer = line(300);
    let short = FixedBitSet::new(299);
    let e = search::<OnHeapHnswGraph, _>(
        &mut scorer,
        None,
        3,
        u64::MAX,
        SearchOptions {
            accept_ords: Some(&short),
            ..opts()
        },
    )
    .unwrap_err();
    assert!(e.to_string().contains("covers 299 ordinals"), "{e}");

    // Exactly the field's size is accepted, which is what every caller in
    // this workspace builds.
    let exact = FixedBitSet::new(300);
    let (hits, _) = search::<OnHeapHnswGraph, _>(
        &mut scorer,
        None,
        3,
        u64::MAX,
        SearchOptions {
            accept_ords: Some(&exact),
            ..opts()
        },
    )
    .unwrap();
    assert!(hits.is_empty(), "an all-zero accept set accepts nothing");

    let e = search::<OnHeapHnswGraph, _>(
        &mut scorer,
        None,
        3,
        u64::MAX,
        SearchOptions {
            filtered_doc_count: Some(-1),
            ..opts()
        },
    )
    .unwrap_err();
    assert!(e.to_string().contains("is negative"), "{e}");

    // And the bound is the *graph's* ordinal space when that is wider
    // than the field's -- the case the exhaustive branch cannot reach but
    // the graph walk can, since the arcs name whatever the `.vem` says.
    let wide = OnHeapHnswGraph::with_size(4, 400);
    let e = search(
        &mut scorer,
        Some(&wide),
        3,
        u64::MAX,
        SearchOptions {
            accept_ords: Some(&exact),
            ..opts()
        },
    )
    .unwrap_err();
    assert!(e.to_string().contains("short of the 400"), "{e}");
}

/// `filteredDocCount` is what decides the graph walk against the
/// exhaustive scan, and it is the *filter's* cardinality, not the field's:
/// a filter selective enough that a walk would visit more nodes than there
/// are acceptable documents takes the scan, which is the whole point of
/// Java's `unfilteredVisit >= filteredDocCount` test. The scan must still
/// honour the accept set -- and its early-termination check sits *inside*
/// the accept test, so a run of rejected ordinals cannot end it.
#[test]
fn a_selective_filtered_doc_count_drops_the_search_to_an_accept_aware_scan() {
    let graph = build(300);
    let (vex, vem) = write_hnsw_vectors(&[field(Some(&graph), 300)], &ID, "").unwrap();
    let reader = HnswVectorsReader::open(&vem, &vex, &ID, "").unwrap();
    let off = reader.graph(0).unwrap().unwrap();
    let mut accept = FixedBitSet::new(300);
    // Everything from 200 up, so the accepted set starts well past the
    // ordinals a scan would reach first.
    for ord in 200..300 {
        accept.set(ord);
    }
    let mut scorer = line(300);
    scorer.query = 150.4;
    // `expected_visited_nodes(3, 300) == 17`, above a cardinality of 10,
    // so this one takes the scan.
    let mut tiny = FixedBitSet::new(300);
    tiny.set(250);
    tiny.set(251);
    let (hits, _) = search(
        &mut scorer,
        Some(&off),
        3,
        u64::MAX,
        SearchOptions {
            accept_ords: Some(&tiny),
            filtered_doc_count: Some(2),
            ..opts()
        },
    )
    .unwrap();
    assert_eq!(
        hits.iter().map(|(n, _)| *n).collect::<Vec<_>>(),
        vec![250, 251]
    );

    // And with a cardinality large enough to keep the graph walk, the
    // accept set still gates what is collected.
    let (hits, _) = search(
        &mut scorer,
        Some(&off),
        3,
        u64::MAX,
        SearchOptions {
            accept_ords: Some(&accept),
            filtered_doc_count: Some(100),
            ..opts()
        },
    )
    .unwrap();
    assert_eq!(
        hits.iter().map(|(n, _)| *n).collect::<Vec<_>>(),
        vec![200, 201, 202]
    );
}

/// Seeding reaches the graph walk and nothing else: with the seeds set,
/// `findBestEntryPoint`'s descent is skipped, and the exhaustive branch
/// ignores them exactly as Java's does (nothing there consults the search
/// strategy).
#[test]
fn seed_ordinals_reach_the_graph_walk_and_are_ignored_by_the_scan() {
    let graph = build(300);
    let (vex, vem) = write_hnsw_vectors(&[field(Some(&graph), 300)], &ID, "").unwrap();
    let reader = HnswVectorsReader::open(&vem, &vex, &ID, "").unwrap();
    let off = reader.graph(0).unwrap().unwrap();
    let mut scorer = line(300);
    scorer.query = 150.4;
    let seeded = SearchOptions {
        seed_ords: Some(&[149, 150, 151]),
        ..opts()
    };
    let (hits, _) = search(&mut scorer, Some(&off), 3, u64::MAX, seeded).unwrap();
    assert_eq!(
        hits.iter().map(|(n, _)| *n).collect::<Vec<_>>(),
        vec![150, 151, 149]
    );

    // `k >= numVectors` takes the scan, where the seeds are irrelevant
    // and the answer is exact regardless.
    let (hits, _) = search(&mut scorer, Some(&off), 300, u64::MAX, seeded).unwrap();
    assert_eq!(hits.len(), 300);
    assert_eq!(hits[0].0, 150);
}

#[test]
fn search_of_an_empty_field_or_zero_k_collects_nothing() {
    let mut empty = Line {
        values: Vec::new(),
        query: 0.0,
    };
    assert!(
        search::<OnHeapHnswGraph, _>(&mut empty, None, 5, u64::MAX, opts())
            .unwrap()
            .0
            .is_empty()
    );
    let mut scorer = line(10);
    assert!(
        search::<OnHeapHnswGraph, _>(&mut scorer, None, 0, u64::MAX, opts())
            .unwrap()
            .0
            .is_empty()
    );
}

#[test]
fn the_exhaustive_branch_honours_the_visit_limit() {
    let mut scorer = line(500);
    scorer.query = 400.0;
    // Fewer than one bulk block, so the loop stops mid-batch.
    let (hits, early) = search::<OnHeapHnswGraph, _>(&mut scorer, None, 5, 10, opts()).unwrap();
    assert!(early, "the visit limit is what stopped the scan");
    assert!(hits.len() <= 5);
    // The first `EXHAUSTIVE_BULK_SCORE_ORDS` ordinals are 0..64, none of
    // which is near the query, so an honoured limit shows up as a wrong
    // (but bounded) answer rather than the exact one.
    assert!(hits.iter().all(|(n, _)| *n < 64));
}

#[test]
fn a_collector_reports_the_nodes_it_kept_in_descending_score_order() {
    let graph = build(300);
    let (vex, vem) = write_hnsw_vectors(&[field(Some(&graph), 300)], &ID, "").unwrap();
    let reader = HnswVectorsReader::open(&vem, &vex, &ID, "").unwrap();
    let off = reader.graph(0).unwrap().unwrap();
    let mut scorer = line(300);
    scorer.query = 10.0;
    let mut collector = KnnCollector::new(4, u64::MAX);
    let mut searcher = HnswGraphSearcher::new(4, off.size());
    searcher
        .search(&mut collector, &mut scorer, &off, None)
        .unwrap();
    let hits = collector.top_docs();
    for pair in hits.windows(2) {
        assert!(pair[0].1 >= pair[1].1);
    }
    assert_eq!(hits[0].0, 10);
}

// ---------------- corrupt metadata ----------------

fn valid_pair() -> (Vec<u8>, Vec<u8>) {
    let graph = build(300);
    write_hnsw_vectors(&[field(Some(&graph), 300)], &ID, "").unwrap()
}

/// Byte offset of the first field's `dimension` vint in `.vem`, found by
/// decoding rather than counting.
fn meta_cursor(meta: &[u8]) -> SliceInput<'_> {
    let mut input = SliceInput::new(meta);
    codec_util::check_index_header(
        &mut input,
        META_CODEC,
        VERSION_START,
        VERSION_CURRENT,
        &ID,
        "",
    )
    .unwrap();
    input.read_i32().unwrap(); // field number
    input.read_i32().unwrap(); // encoding
    input.read_i32().unwrap(); // similarity
    input.read_vlong().unwrap(); // index offset
    input.read_vlong().unwrap(); // index length
    input
}

#[test]
fn an_illegal_dimension_count_or_max_conn_is_rejected() {
    let (vex, vem) = valid_pair();
    let mut cur = meta_cursor(&vem);
    let dim_at = cur.position();
    assert_eq!(vem[dim_at], 1);
    assert!(matches!(
        open(&repack(&vem, |b| b[dim_at] = 0), &vex),
        Err(Error::CorruptMeta(_))
    ));
    cur.read_vint().unwrap();
    let count_at = cur.position();
    assert!(matches!(
        open(
            &repack(&vem, |b| b[count_at..count_at + 4]
                .copy_from_slice(&(-1i32).to_le_bytes())),
            &vex
        ),
        Err(Error::CorruptMeta(_))
    ));
    cur.read_i32().unwrap();
    let m_at = cur.position();
    assert!(matches!(
        open(&repack(&vem, |b| b[m_at] = 0), &vex),
        Err(Error::CorruptMeta(_))
    ));
    // `MAXIMUM_MAX_CONN` is 512, so a two-byte vint above it is rejected.
    assert!(matches!(
        open(
            &repack(&vem, |b| {
                b[m_at] = 0x80;
                b[m_at + 1] = 0x40;
            }),
            &vex
        ),
        Err(Error::CorruptMeta(_))
    ));
}

#[test]
fn an_upper_level_node_count_out_of_range_is_rejected() {
    let (vex, vem) = valid_pair();
    let mut cur = meta_cursor(&vem);
    cur.read_vint().unwrap(); // dimension
    cur.read_i32().unwrap(); // count
    cur.read_vint().unwrap(); // M
    cur.read_vint().unwrap(); // numLevels
    let level1_count_at = cur.position();
    // Zero nodes on an upper level is impossible: the level exists because
    // something is on it.
    assert!(matches!(
        open(&repack(&vem, |b| b[level1_count_at] = 0), &vex),
        Err(Error::CorruptMeta(_))
    ));
}

/// The level count is a vint, so a single flipped byte can name
/// `Integer.MAX_VALUE` levels. Java answers that with
/// `new int[numLevels][]`, an `OutOfMemoryError`; allocating
/// `vec![Vec::new(); 2^31 - 1]` instead would be **51 GB, an abort**, which
/// `catch_unwind` cannot turn into a JVM exception.
#[test]
fn an_absurd_level_count_is_a_decode_error_not_an_allocation() {
    let (vex, vem) = valid_pair();
    let mut cur = meta_cursor(&vem);
    cur.read_vint().unwrap(); // dimension
    cur.read_i32().unwrap(); // count
    cur.read_vint().unwrap(); // M
    let num_levels_at = cur.position();
    // A five-byte vint for `i32::MAX`, in place of the one-byte real one.
    let mut body = vem[..num_levels_at].to_vec();
    body.write_vint(i32::MAX);
    body.extend_from_slice(&vem[num_levels_at + 1..vem.len() - codec_util::FOOTER_LENGTH]);
    codec_util::write_footer(&mut body);
    assert!(open(&body, &vex).is_err());
}

/// The per-level node count is bounded by `size`, which is a raw `.vem`
/// `int`: together they used to size a `Vec::with_capacity` of up to
/// `i32::MAX` ordinals -- 8.6 GB, the same abort shape. The file itself is
/// the bound: one vint per node.
#[test]
fn an_upper_level_node_count_larger_than_the_file_is_a_decode_error() {
    let (vex, vem) = valid_pair();
    let mut cur = meta_cursor(&vem);
    cur.read_vint().unwrap(); // dimension
    let count_at = cur.position();
    cur.read_i32().unwrap(); // count
    cur.read_vint().unwrap(); // M
    cur.read_vint().unwrap(); // numLevels
    let level1_count_at = cur.position();
    let mut body = vem[..count_at].to_vec();
    body.write_i32(i32::MAX); // size, so the `<= size` check passes
    body.extend_from_slice(&vem[count_at + 4..level1_count_at]);
    body.write_vint(i32::MAX); // nodes on level 1
    body.extend_from_slice(&vem[level1_count_at + 1..vem.len() - codec_util::FOOTER_LENGTH]);
    codec_util::write_footer(&mut body);
    let err = open(&body, &vex).unwrap_err();
    assert!(
        format!("{err:?}").contains("bytes left in the .vem"),
        "expected the node count to be rejected against the file length, got {err:?}"
    );
}

/// Flip bit 0 and bit 7 of every `.vem` and `.vex` body byte, re-sign the
/// footer so only a semantic invariant can reject the file, and require a
/// typed error or a clean decode from `open`, `graph`, a full walk of
/// every node on every level, and a search. Never a panic, never an abort.
///
/// The fixture is the 300-vector, multi-level graph: a single-level or
/// single-node graph would leave the per-level node lists, the
/// `DirectMonotonic` node-offset array and the group-varint neighbour
/// decoder untouched, which is measuring the fixture rather than the
/// decoder.
#[test]
fn every_resigned_single_byte_vem_and_vex_corruption_is_an_error_or_a_clean_decode() {
    let (vex, vem) = valid_pair();
    let graph = build(300);
    assert!(graph.num_levels() >= 2, "fixture must be multi-level");
    let mut flipped = 0usize;
    let mut rejected = 0usize;
    for which in 0..2 {
        let original: &[u8] = if which == 0 { &vem } else { &vex };
        let body_len = original.len() - codec_util::FOOTER_LENGTH;
        for at in 0..body_len {
            for bit in [0u8, 7] {
                let patched = repack(original, |b| b[at] ^= 1 << bit);
                let (m, x) = if which == 0 {
                    (patched.clone(), vex.clone())
                } else {
                    (vem.clone(), patched.clone())
                };
                flipped += 1;
                if walk_everything(&m, &x).is_err() {
                    rejected += 1;
                }
            }
        }
    }
    assert_eq!(
        flipped,
        2 * (vem.len() + vex.len() - 2 * codec_util::FOOTER_LENGTH)
    );
    assert!(
        rejected > flipped / 4,
        "only {rejected} of {flipped} flips rejected -- suspiciously few"
    );
    eprintln!(".vem+.vex byte-flip sweep: {rejected}/{flipped} rejected");
}

/// Every read a query performs, so a flip anywhere in either file has to
/// surface as a typed error rather than a panic.
fn walk_everything(meta: &[u8], index: &[u8]) -> Result<()> {
    let index: &'static [u8] = Box::leak(index.to_vec().into_boxed_slice());
    let reader = HnswVectorsReader::open(meta, index, &ID, "")?;
    for entry in reader.fields() {
        let number = entry.field_number;
        let Some(off) = reader.graph(number)? else {
            continue;
        };
        let mut out = Vec::new();
        for level in 0..off.num_levels() {
            for node in off.sorted_nodes_on_level(level)? {
                off.neighbors_into(level, node, &mut out)?;
            }
        }
        let mut scorer = line(off.size().max(1) as usize);
        search(&mut scorer, Some(&off), 10, u64::MAX, opts())?;
    }
    Ok(())
}

#[test]
fn a_graph_region_past_the_end_of_the_index_file_is_rejected() {
    let (vex, vem) = valid_pair();
    let mut input = SliceInput::new(&vem);
    codec_util::check_index_header(
        &mut input,
        META_CODEC,
        VERSION_START,
        VERSION_CURRENT,
        &ID,
        "",
    )
    .unwrap();
    input.read_i32().unwrap();
    input.read_i32().unwrap();
    input.read_i32().unwrap();
    let offset_at = input.position();
    assert!(matches!(
        open(
            &repack(&vem, |b| {
                b[offset_at] = 0xFF;
                b[offset_at + 1] = 0x7F;
            }),
            &vex
        ),
        Err(Error::CorruptMeta(_))
    ));
}

#[test]
fn a_negative_field_number_is_rejected() {
    let (vex, vem) = valid_pair();
    let at = codec_util::index_header_length(META_CODEC, "");
    assert!(matches!(
        open(
            &repack(&vem, |b| b[at..at + 4]
                .copy_from_slice(&(-5i32).to_le_bytes())),
            &vex
        ),
        Err(Error::CorruptMeta(_))
    ));
}

#[test]
fn a_truncated_meta_file_is_rejected() {
    let (vex, vem) = valid_pair();
    assert!(open(&vem[..vem.len() - 1], &vex).is_err());
    assert!(open(&vem, &vex[..vex.len() - 1]).is_err());
}

#[test]
fn an_out_of_range_neighbour_ordinal_is_rejected() {
    // A hand-built one-node graph whose only neighbour points past the end
    // of the graph: the delta decoder must refuse it rather than hand a
    // wild ordinal to the scorer.
    let mut graph = OnHeapHnswGraph::new(8);
    graph.add_node(0, 0);
    graph.try_set_new_entry_node(0, 0);
    graph.neighbors_mut(0, 0).add_in_order(0, 1.0);
    let (vex, vem) = write_hnsw_vectors(&[field(Some(&graph), 1)], &ID, "").unwrap();
    // Bump the single neighbour ordinal from 0 to 5. `writeGroupVInts`
    // emits whole groups of four behind a flag byte and the remainder as
    // plain vints, so a lone neighbour is one byte right after the vint
    // arc count.
    let payload_at = codec_util::index_header_length(INDEX_CODEC, "") + 1;
    let broken = repack(&vex, |b| b[payload_at] = 5);
    let reader = open(&vem, &broken).unwrap();
    let off = reader.graph(0).unwrap().unwrap();
    let mut out = Vec::new();
    assert!(matches!(
        off.neighbors_into(0, 0, &mut out),
        Err(Error::CorruptMeta(_))
    ));
}
// ---------------- merge ----------------

/// A `Line` scorer that counts how many similarity computations it is
/// asked for -- the unit the merge is supposed to save.
#[derive(Debug, Clone)]
struct CountingLine {
    inner: Line,
    scores: std::rc::Rc<std::cell::Cell<u64>>,
}

impl VectorScorer for CountingLine {
    fn score(&mut self, node: i32) -> Result<f32> {
        self.scores.set(self.scores.get() + 1);
        self.inner.score(node)
    }

    fn max_ord(&self) -> i32 {
        self.inner.max_ord()
    }
}

impl UpdateableVectorScorer for CountingLine {
    fn set_scoring_ordinal(&mut self, ord: i32) -> Result<()> {
        self.inner.set_scoring_ordinal(ord)
    }
}

fn counting(n: usize) -> (CountingLine, std::rc::Rc<std::cell::Cell<u64>>) {
    let scores = std::rc::Rc::new(std::cell::Cell::new(0u64));
    (
        CountingLine {
            inner: line(n),
            scores: scores.clone(),
        },
        scores,
    )
}

fn identity(n: usize) -> Vec<i32> {
    (0..n as i32).collect()
}

/// The claim the incremental merger exists to make: merging a big segment
/// with a small one costs far less than rebuilding the whole graph. Counted
/// in similarity computations, which is what dominates both.
#[test]
fn merging_reuses_the_largest_graph_instead_of_rebuilding() {
    let big = 1200usize;
    let small = 120usize;
    let total = big + small;
    let big_graph = build(big);
    let small_graph = build(small);

    // Source 0's docs land first in the merged segment, source 1's after.
    let big_docs = identity(big);
    let small_docs = identity(small);
    let big_map = identity(big);
    let small_map: Vec<i32> = (0..small as i32).map(|d| d + big as i32).collect();
    let merged_ord_to_doc = identity(total);
    let sources = [
        GraphMergeSource {
            graph: Some(&big_graph),
            ord_to_doc: &big_docs,
            doc_map: &big_map,
        },
        GraphMergeSource {
            graph: Some(&small_graph),
            ord_to_doc: &small_docs,
            doc_map: &small_map,
        },
    ];

    let (scorer, merge_scores) = counting(total);
    let merged = merge_one_field(
        scorer,
        8,
        32,
        crate::hnsw::DEFAULT_RAND_SEED,
        &merged_ord_to_doc,
        &sources,
    )
    .unwrap()
    .expect("a 1320-vector segment is well past the graph threshold");

    let (rebuild_scorer, rebuild_scores) = counting(total);
    let rebuilt = HnswGraphBuilder::new(rebuild_scorer, 8, 32, crate::hnsw::DEFAULT_RAND_SEED)
        .unwrap()
        .build(total as i32)
        .unwrap();

    assert_eq!(merged.sorted_nodes_on_level(0).unwrap().len(), total);
    assert!(
        merge_scores.get() * 3 < rebuild_scores.get(),
        "the incremental merge scored {} times against a rebuild's {} -- it is supposed to \
             cost a fraction, not a comparable amount",
        merge_scores.get(),
        rebuild_scores.get()
    );
    let _ = rebuilt;

    // And it must still be searchable: a query anywhere in either source's
    // range finds its own vector.
    let mut searcher = HnswGraphSearcher::new(10, merged.size());
    for target in [0usize, 5, big - 1, big, total - 1] {
        let mut collector = KnnCollector::new(10, u64::MAX);
        let mut scorer = Line {
            values: (0..total).map(|i| i as f32).collect(),
            query: target as f32,
        };
        searcher
            .search(&mut collector, &mut scorer, &merged, None)
            .unwrap();
        let found: Vec<i32> = collector.top_docs().into_iter().map(|(n, _)| n).collect();
        assert!(
            found.contains(&(target as i32)),
            "merged graph lost ordinal {target}: {found:?}"
        );
    }
}

/// The **mixed** case, and the only one that reaches `new_ord_mapping`'s
/// `initialized` bitset: one source has a reusable graph and one does not,
/// so `order.len() != sources.len()` and Java's
/// `graphReaders.size() == numReaders ? null : new FixedBitSet(maxOrd)`
/// allocates the set that records which merged ordinals a reused graph
/// already covers. Every ordinal the graphless source contributes must be
/// left for insertion from scratch, and every ordinal the reused graph
/// covers must be marked -- so the merged graph has to end up complete,
/// with both sources' vectors reachable.
#[test]
fn a_source_without_a_graph_leaves_its_ordinals_to_be_inserted_from_scratch() {
    let reusable = 600usize;
    let graphless = 300usize;
    let total = reusable + graphless;
    let big_graph = build(reusable);
    let docs_a = identity(reusable);
    let docs_b = identity(graphless);
    let map_a = identity(reusable);
    let map_b: Vec<i32> = (0..graphless as i32).map(|d| d + reusable as i32).collect();
    let merged_ord_to_doc = identity(total);
    let sources = [
        GraphMergeSource {
            graph: Some(&big_graph),
            ord_to_doc: &docs_a,
            doc_map: &map_a,
        },
        GraphMergeSource {
            graph: None,
            ord_to_doc: &docs_b,
            doc_map: &map_b,
        },
    ];
    let merged = merge_one_field(
        line(total),
        8,
        32,
        crate::hnsw::DEFAULT_RAND_SEED,
        &merged_ord_to_doc,
        &sources,
    )
    .unwrap()
    .expect("900 vectors is past the graph threshold");

    // Completeness is the property the `initialized` bookkeeping exists
    // for: an ordinal marked when it should not be is never inserted and
    // vanishes from the graph; one left unmarked is inserted twice.
    assert_eq!(merged.sorted_nodes_on_level(0).unwrap().len(), total);
    let mut seen = std::collections::HashSet::new();
    for node in merged.sorted_nodes_on_level(0).unwrap() {
        assert!(seen.insert(node), "ordinal {node} appears twice on level 0");
    }
    assert_eq!(seen.len(), total);
    // The reused graph's own ordinals and the graphless source's are both
    // present, which is what says the bitset covered the right half.
    assert!(seen.contains(&0));
    assert!(seen.contains(&(reusable as i32 - 1)));
    assert!(seen.contains(&(reusable as i32)));
    assert!(seen.contains(&(total as i32 - 1)));
}

/// No source has a graph (every segment was below the threshold), so
/// there is nothing to reuse and Lucene rebuilds. The result must still be
/// a complete graph.
#[test]
fn merging_rebuilds_when_no_source_has_a_graph() {
    let total = 800usize;
    let docs_a = identity(400);
    let docs_b = identity(400);
    let map_a = identity(400);
    let map_b: Vec<i32> = (0..400).map(|d| d + 400).collect();
    let merged_ord_to_doc = identity(total);
    let sources: [GraphMergeSource<'_, OnHeapHnswGraph>; 2] = [
        GraphMergeSource {
            graph: None,
            ord_to_doc: &docs_a,
            doc_map: &map_a,
        },
        GraphMergeSource {
            graph: None,
            ord_to_doc: &docs_b,
            doc_map: &map_b,
        },
    ];
    let merged = merge_one_field(
        line(total),
        8,
        32,
        crate::hnsw::DEFAULT_RAND_SEED,
        &merged_ord_to_doc,
        &sources,
    )
    .unwrap()
    .unwrap();
    assert_eq!(merged.sorted_nodes_on_level(0).unwrap().len(), total);
    // A rebuild is exactly `HnswGraphBuilder::build`, so it must be
    // arc-for-arc that graph.
    let rebuilt = build(total);
    assert_eq!(merged.entry_node(), rebuilt.entry_node());
    assert_eq!(merged.num_levels(), rebuilt.num_levels());
    let mut a = Vec::new();
    let mut b = Vec::new();
    for level in 0..merged.num_levels() {
        for node in merged.sorted_nodes_on_level(level).unwrap() {
            merged.neighbors_into(level, node, &mut a).unwrap();
            rebuilt.neighbors_into(level, node, &mut b).unwrap();
            assert_eq!(a, b, "level {level} node {node}");
        }
    }
}

/// Below `HNSW_GRAPH_THRESHOLD` Lucene writes no graph at all, on merge
/// exactly as on flush -- a merged segment of 40 vectors must not suddenly
/// acquire one.
#[test]
fn merging_writes_no_graph_below_the_threshold() {
    let docs = identity(40);
    let map = identity(40);
    let merged_ord_to_doc = identity(40);
    let graph = build(40);
    let sources = [GraphMergeSource {
        graph: Some(&graph),
        ord_to_doc: &docs,
        doc_map: &map,
    }];
    assert!(merge_one_field(
        line(40),
        8,
        32,
        crate::hnsw::DEFAULT_RAND_SEED,
        &merged_ord_to_doc,
        &sources,
    )
    .unwrap()
    .is_none());
    // ...and an empty merged field is `None` too.
    assert!(merge_one_field(
        line(1),
        8,
        32,
        crate::hnsw::DEFAULT_RAND_SEED,
        &[],
        &sources,
    )
    .unwrap()
    .is_none());
}

/// A graph more than `DELETE_PCT_THRESHOLD` deleted is not used as the
/// base: its connectivity has degraded, so Lucene prefers a rebuild.
#[test]
fn a_heavily_deleted_graph_is_not_used_as_the_base() {
    let n = 3000usize;
    let graph = build(n);
    let docs = identity(n);
    // Drop 60% -- past the 40% threshold.
    let mut map = vec![-1i32; n];
    let mut kept = 0i32;
    for (d, slot) in map.iter_mut().enumerate() {
        if d % 5 < 2 {
            *slot = kept;
            kept += 1;
        }
    }
    let merged_ord_to_doc = identity(kept as usize);
    let sources = [GraphMergeSource {
        graph: Some(&graph),
        ord_to_doc: &docs,
        doc_map: &map,
    }];
    let merged = merge_one_field(
        line(kept as usize),
        8,
        32,
        crate::hnsw::DEFAULT_RAND_SEED,
        &merged_ord_to_doc,
        &sources,
    )
    .unwrap()
    .unwrap();
    // A rebuild, so it is arc-for-arc `HnswGraphBuilder::build(kept)`.
    let rebuilt = build(kept as usize);
    assert_eq!(merged.entry_node(), rebuilt.entry_node());
    let mut a = Vec::new();
    let mut b = Vec::new();
    for node in 0..kept {
        merged.neighbors_into(0, node, &mut a).unwrap();
        rebuilt.neighbors_into(0, node, &mut b).unwrap();
        assert_eq!(a, b, "node {node}");
    }
}

/// `M`/`beamWidth` out of range must be an error, not a panic -- the merge
/// entry point allocates a graph sized by `M` before any builder sees it.
#[test]
fn merge_rejects_out_of_range_graph_parameters() {
    let graph = build(1400);
    let docs = identity(1400);
    let map = identity(1400);
    let sources = [GraphMergeSource {
        graph: Some(&graph),
        ord_to_doc: &docs,
        doc_map: &map,
    }];
    for (m, beam) in [(0, 32), (513, 32), (8, 0), (8, 3201)] {
        assert!(
            matches!(
                merge_one_field(
                    line(1400),
                    m,
                    beam,
                    crate::hnsw::DEFAULT_RAND_SEED,
                    &identity(1400),
                    &sources
                ),
                Err(Error::InvalidGraphParameter(_))
            ),
            "M={m} beamWidth={beam} must be rejected"
        );
    }
}

/// A source whose graph size disagrees with its vector count is a caller
/// bug that would otherwise index the ordinal map out of bounds.
#[test]
fn merge_rejects_a_source_whose_graph_and_vectors_disagree() {
    let graph = build(1500);
    let docs = identity(1400);
    let map = identity(1400);
    let merged_ord_to_doc = identity(1400);
    let sources = [GraphMergeSource {
        graph: Some(&graph),
        ord_to_doc: &docs,
        doc_map: &map,
    }];
    assert!(matches!(
        merge_one_field(
            line(1400),
            8,
            32,
            crate::hnsw::DEFAULT_RAND_SEED,
            &merged_ord_to_doc,
            &sources
        ),
        Err(Error::InvalidGraphParameter(_))
    ));

    // And a doc map with no entry for a doc the source's vectors name.
    let graph = build(1400);
    let docs = identity(1400);
    let short = identity(3);
    let sources = [GraphMergeSource {
        graph: Some(&graph),
        ord_to_doc: &docs,
        doc_map: &short,
    }];
    assert!(matches!(
        merge_one_field(
            line(1400),
            8,
            32,
            crate::hnsw::DEFAULT_RAND_SEED,
            &identity(1400),
            &sources
        ),
        Err(Error::InvalidGraphParameter(_))
    ));
}

/// The whole merge, end to end at the codec level: two segments' `.vec`/
/// `.vemf`/`.vem`/`.vex` in, one segment's out, reopened and searched.
#[test]
fn a_merged_segment_round_trips_through_both_readers() {
    use crate::vectors::{
        FieldVectorData, FlatVectorMergeSource, FlatVectorsField, FlatVectorsReader,
        FlatVectorsWriter, MergeSourceValues, MergedFlatVectorField,
    };

    const DIM: i32 = 4;
    // Two source segments whose vectors interleave in value space, so the
    // merged graph has to link regions neither source graph did.
    let make = |offset: f32, count: i32| -> (Vec<u8>, Vec<u8>) {
        let values: Vec<f32> = (0..count)
            .flat_map(|i| (0..DIM).map(move |d| offset + i as f32 * 2.0 + d as f32 * 0.01))
            .collect();
        crate::vectors::write_flat_vectors(
            &[FlatVectorsField {
                field_number: 0,
                similarity: VectorSimilarityFunction::Euclidean,
                dimension: DIM,
                docs: (0..count).collect(),
                values: FieldVectorData::Float32(values),
            }],
            count,
            &ID,
            "",
        )
        .unwrap()
    };
    let (a_vec, a_meta) = make(0.0, 900);
    let (b_vec, b_meta) = make(1.0, 700);
    let a = FlatVectorsReader::open(&a_meta, &a_vec, &ID, "").unwrap();
    let b = FlatVectorsReader::open(&b_meta, &b_vec, &ID, "").unwrap();

    let a_map = identity(900);
    let b_map: Vec<i32> = (0..700).map(|d| d + 900).collect();
    let flat_sources = vec![
        FlatVectorMergeSource {
            values: MergeSourceValues::Float32(a.float_vector_values(0).unwrap()),
            doc_map: &a_map,
        },
        FlatVectorMergeSource {
            values: MergeSourceValues::Float32(b.float_vector_values(0).unwrap()),
            doc_map: &b_map,
        },
    ];
    let mut writer = FlatVectorsWriter::new(1600, &ID, "");
    writer
        .merge_one_flat_vector_field(&MergedFlatVectorField {
            field_number: 0,
            encoding: VectorEncoding::Float32,
            similarity: VectorSimilarityFunction::Euclidean,
            dimension: DIM,
            sources: &flat_sources,
        })
        .unwrap();
    let (merged_vec, merged_vemf) = writer.finish();

    // Both source graphs, then the merged graph over the merged flat file.
    let a_graph = HnswGraphBuilder::new(
        a.float_vector_values(0).unwrap().ord_scorer(),
        8,
        32,
        crate::hnsw::DEFAULT_RAND_SEED,
    )
    .unwrap()
    .build(900)
    .unwrap();
    let b_graph = HnswGraphBuilder::new(
        b.float_vector_values(0).unwrap().ord_scorer(),
        8,
        32,
        crate::hnsw::DEFAULT_RAND_SEED,
    )
    .unwrap()
    .build(700)
    .unwrap();

    let merged_flat = FlatVectorsReader::open(&merged_vemf, &merged_vec, &ID, "").unwrap();
    let merged_values = merged_flat.float_vector_values(0).unwrap();
    let merged_ord_to_doc: Vec<i32> = (0..merged_values.size())
        .map(|o| merged_values.ord_to_doc(o).unwrap())
        .collect();
    let a_docs = identity(900);
    let b_docs = identity(700);
    let graph_sources = [
        GraphMergeSource {
            graph: Some(&a_graph),
            ord_to_doc: &a_docs,
            doc_map: &a_map,
        },
        GraphMergeSource {
            graph: Some(&b_graph),
            ord_to_doc: &b_docs,
            doc_map: &b_map,
        },
    ];
    let merged_graph = merge_one_field(
        merged_values.ord_scorer(),
        8,
        32,
        crate::hnsw::DEFAULT_RAND_SEED,
        &merged_ord_to_doc,
        &graph_sources,
    )
    .unwrap()
    .unwrap();

    let (vex, vem) = write_hnsw_vectors(
        &[HnswVectorsField {
            field_number: 0,
            encoding: VectorEncoding::Float32,
            similarity: VectorSimilarityFunction::Euclidean,
            dimension: DIM,
            count: 1600,
            graph: Some(&merged_graph),
            m: 8,
        }],
        &ID,
        "",
    )
    .unwrap();

    // Reopen the written bytes and search them, comparing against the
    // exhaustive answer over the same merged file.
    let reader = HnswVectorsReader::open(&vem, &vex, &ID, "").unwrap();
    let graph = reader.graph(0).unwrap().unwrap();
    assert_eq!(graph.size(), 1600);
    let mut hits = 0usize;
    let mut total = 0usize;
    for q in 0..10 {
        let query: Vec<f32> = (0..DIM)
            .map(|d| q as f32 * 137.0 + d as f32 * 0.01)
            .collect();
        let exact: Vec<i32> = merged_values
            .exhaustive_search(&query, 10)
            .unwrap()
            .into_iter()
            .map(|(d, _)| d)
            .collect();
        let mut scorer = merged_values.scorer(&query).unwrap();
        let approx: Vec<i32> = search(&mut scorer, Some(&graph), 10, u64::MAX, opts())
            .unwrap()
            .0
            .into_iter()
            .map(|(ord, _)| merged_values.ord_to_doc(ord).unwrap())
            .collect();
        total += exact.len();
        hits += approx.iter().filter(|d| exact.contains(d)).count();
    }
    let recall = hits as f64 / total as f64;
    assert!(
        recall >= 0.9,
        "search over the merged, serialized graph recalled {recall}"
    );

    // `ConcurrentHnswMerger`: the base graph copied, everything else
    // inserted by four workers in small batches so they interleave. The
    // result depends on scheduling (as in Java), so it is held to the same
    // recall bar as the sequential merge, not to its arcs.
    let concurrent = merge_one_field_concurrent(
        || Ok(merged_values.ord_scorer()),
        8,
        32,
        crate::hnsw::DEFAULT_RAND_SEED,
        &merged_ord_to_doc,
        &graph_sources,
        4,
        64,
    )
    .unwrap()
    .unwrap();
    assert_eq!(concurrent.size(), 1600);
    let mut hits = 0usize;
    let mut total = 0usize;
    for q in 0..10 {
        let query: Vec<f32> = (0..DIM)
            .map(|d| q as f32 * 137.0 + d as f32 * 0.01)
            .collect();
        let exact: Vec<i32> = merged_values
            .exhaustive_search(&query, 10)
            .unwrap()
            .into_iter()
            .map(|(d, _)| d)
            .collect();
        let mut scorer = merged_values.scorer(&query).unwrap();
        let approx: Vec<i32> = search(&mut scorer, Some(&concurrent), 10, u64::MAX, opts())
            .unwrap()
            .0
            .into_iter()
            .map(|(ord, _)| merged_values.ord_to_doc(ord).unwrap())
            .collect();
        total += exact.len();
        hits += approx.iter().filter(|d| exact.contains(d)).count();
    }
    let recall = hits as f64 / total as f64;
    assert!(
        recall >= 0.9,
        "the concurrently merged graph recalled {recall}"
    );
    // No usable base graph (every source dropped): the whole graph is
    // built concurrently.
    let none: [GraphMergeSource<'_, OnHeapHnswGraph>; 0] = [];
    let rebuilt = merge_one_field_concurrent(
        || Ok(merged_values.ord_scorer()),
        8,
        32,
        crate::hnsw::DEFAULT_RAND_SEED,
        &merged_ord_to_doc,
        &none,
        3,
        100,
    )
    .unwrap()
    .unwrap();
    assert_eq!(rebuilt.size(), 1600);
    assert!(merge_one_field_concurrent(
        || Ok(merged_values.ord_scorer()),
        8,
        32,
        1,
        &merged_ord_to_doc,
        &none,
        0,
        100,
    )
    .is_err());
    assert!(merge_one_field_concurrent(
        || Ok(merged_values.ord_scorer()),
        8,
        32,
        1,
        &[],
        &none,
        2,
        100,
    )
    .unwrap()
    .is_none());
}

/// A graph over 2-d-ish points on a line for the filtered-search tests.
fn filtered_graph(n: usize) -> (crate::hnsw::OnHeapHnswGraph, Vec<f32>) {
    let points: Vec<f32> = (0..n).map(|i| ((i * 7919) % 1000) as f32 / 10.0).collect();
    let g = HnswGraphBuilder::new(
        Line {
            values: points.clone(),
            query: 0.0,
        },
        8,
        50,
        crate::hnsw::DEFAULT_RAND_SEED,
    )
    .unwrap()
    .build(n as i32)
    .unwrap();
    (g, points)
}

/// `KnnSearchStrategy.Hnsw(60)` with a filter passing 5%: the
/// `FilteredHnswGraphSearcher` walk collects only accepted nodes and
/// finds the exact best among them; threshold 0 (Lucene's default) and
/// no filter take the ordinary walk.
#[test]
fn filtered_strategy_finds_accepted_neighbours() {
    let n = 2000;
    let (g, points) = filtered_graph(n);
    let mut accept = FixedBitSet::new(n);
    for i in (0..n).step_by(20) {
        accept.set(i);
    }
    let mut scorer = Line {
        values: points.clone(),
        query: 50.0,
    };
    let with = |threshold: i32| SearchOptions {
        accept_ords: Some(&accept),
        filtered_doc_count: Some(100),
        seed_ords: None,
        filtered_search_threshold: threshold,
    };
    let (hits, _) = search(&mut scorer, Some(&g), 5, u64::MAX, with(60)).unwrap();
    // A filtered walk may collect fewer than k (Lucene then falls back to
    // an exact search in `AbstractKnnVectorQuery`).
    assert!(!hits.is_empty() && hits.len() <= 5, "{hits:?}");
    assert!(hits.iter().all(|&(o, _)| o % 20 == 0));
    let mut exact: Vec<(i32, f32)> = (0..n)
        .step_by(20)
        .map(|i| (i as i32, scorer.clone().score(i as i32).unwrap()))
        .collect();
    exact.sort_by(|a, b| b.1.total_cmp(&a.1));
    assert!(
        hits[0].1 >= exact[0].1 * 0.99,
        "{:?} vs {:?}",
        hits[0],
        exact[0]
    );
    let (plain, _) = search(&mut scorer, Some(&g), 5, u64::MAX, with(0)).unwrap();
    assert!(plain.iter().all(|&(o, _)| o % 20 == 0));
    let (all, _) = search(&mut scorer, Some(&g), 5, u64::MAX, opts()).unwrap();
    assert_eq!(all.len(), 5);
}

/// `FilteredHnswGraphSearcher.create`'s `0 < filterSize < graph size`
/// check, and the accept-set bound.
#[test]
fn filtered_searcher_validates_filter_size_and_accept_set() {
    let (g, points) = filtered_graph(300);
    let mut scorer = Line {
        values: points,
        query: 5.0,
    };
    let accept = FixedBitSet::new(300);
    for size in [0, 300] {
        let mut searcher = HnswGraphSearcher::new(5, g.size());
        let mut collector = KnnCollector::new(5, u64::MAX);
        assert!(searcher
            .search_filtered(&mut collector, &mut scorer, &g, &accept, size, None)
            .is_err());
    }
    let mut searcher = HnswGraphSearcher::new(5, g.size());
    let mut collector = KnnCollector::new(5, u64::MAX);
    let short = FixedBitSet::new(10);
    assert!(searcher
        .search_filtered(&mut collector, &mut scorer, &g, &short, 5, None)
        .is_err());
    let mut searcher = HnswGraphSearcher::new(5, g.size());
    let mut collector = KnnCollector::new(5, u64::MAX);
    assert!(searcher
        .search_filtered(&mut collector, &mut scorer, &g, &accept, 30, Some(&[]))
        .is_err());
}

/// The base graph is the one with the most live vectors, whatever the reader
/// order (Lucene 10.6, #16739): a graph with deletions does not stay the base
/// over a larger deletion-free one.
#[test]
fn the_base_graph_is_picked_by_live_count() {
    let with_deletes = build(3000);
    let clean = build(2500);
    let docs_a = identity(3000);
    let docs_b = identity(2500);
    // Delete 30% of the first graph's documents: 2100 live of 3000.
    let map_a: Vec<i32> = (0..3000i32)
        .map(|d| if d % 10 < 3 { -1 } else { d })
        .collect();
    let map_b = identity(2500);
    let sources = [
        GraphMergeSource {
            graph: Some(&with_deletes),
            ord_to_doc: &docs_a,
            doc_map: &map_a,
        },
        GraphMergeSource {
            graph: Some(&clean),
            ord_to_doc: &docs_b,
            doc_map: &map_b,
        },
    ];
    let (_, largest) = add_readers(&sources).unwrap();
    assert_eq!(largest, Some(1));
}
