//! Differential tests for `lucene_search::vector_query` against real Lucene
//! 10.5.0 output.
//!
//! Two fixtures, both written by `fixtures/src/GenVectors.java` through a
//! real `IndexWriter` and queried through a real `IndexSearcher`:
//!
//! - `fixtures/data/vectors_index/` — one 4000-document segment, five vector
//!   fields (dense/sparse FLOAT32 with three float similarities, a BYTE
//!   field, and a below-threshold field Lucene writes no graph for). Its
//!   manifest records, per query, exactly what
//!   `KnnFloatVectorQuery`/`KnnByteVectorQuery` returned.
//! - `fixtures/data/vectors_multi_index/` — the same documents split across
//!   four segments, plus a `bucket` `StringField` used as a KNN **filter**.
//!   Its manifest records the multi-segment `KnnFloatVectorQuery` results
//!   (global doc ids), the same queries with a selective and a permissive
//!   filter, and the exact (brute-force) top-k among each filter's accepted
//!   documents.
//!
//! Every assertion here is doc-for-doc and score-for-score against Lucene's
//! own answer. None of them is a recall threshold: c5 established that recall
//! does *not* discriminate for this subsystem (mutating the diversity rule
//! took graph agreement to 1/4273 while recall rose).

use std::collections::HashMap;

use lucene_codecs::field_infos::{self, FieldInfos};
use lucene_codecs::hnsw_vectors::HnswVectorsReader;
use lucene_codecs::vectors::FlatVectorsReader;
use lucene_search::vector_query::{
    search_knn_byte_vector_query, search_knn_byte_vector_query_multi_segment,
    search_knn_byte_vector_query_multi_segment_concurrent, search_knn_float_vector_query,
    search_knn_float_vector_query_multi_segment,
    search_knn_float_vector_query_multi_segment_concurrent, KnnByteVectorQuery,
    KnnFloatVectorQuery, KnnSegment, VectorsInput,
};
use lucene_search::ScoreDoc;
use lucene_util::fixed_bit_set::FixedBitSet;

// ---------------------------------------------------------------------------
// Manifest plumbing
// ---------------------------------------------------------------------------

struct Manifest {
    dir: String,
    kv: HashMap<String, String>,
}

impl Manifest {
    fn load(name: &str) -> Self {
        let dir = format!("{}/../../fixtures/data/{name}/", env!("CARGO_MANIFEST_DIR"));
        let text = std::fs::read_to_string(format!("{dir}manifest.properties"))
            .expect("run scripts/gen-fixtures.sh first (GenVectors)");
        Manifest {
            dir,
            kv: text
                .lines()
                .filter(|l| !l.is_empty() && !l.starts_with('#'))
                .filter_map(|l| l.split_once('='))
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
        }
    }

    fn get(&self, key: &str) -> &str {
        self.kv
            .get(key)
            .unwrap_or_else(|| panic!("manifest key {key} missing"))
    }

    fn opt(&self, key: &str) -> Option<&str> {
        self.kv.get(key).map(|s| s.as_str())
    }

    fn int(&self, key: &str) -> i32 {
        self.get(key).parse().unwrap()
    }

    fn file(&self, name: &str) -> Vec<u8> {
        std::fs::read(format!("{}{name}", self.dir)).expect("fixture file")
    }
}

fn segment_id(hex: &str) -> [u8; 16] {
    let mut id = [0u8; 16];
    for (i, slot) in id.iter_mut().enumerate() {
        *slot = u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16).expect("hex byte");
    }
    id
}

/// `doc:scoreBits;doc:scoreBits;…` — scores are `Float.floatToIntBits`, so
/// the fixture pins the exact float and not a rounded decimal.
fn parse_hits(spec: &str) -> Vec<(i32, f32)> {
    spec.split(';')
        .filter(|s| !s.is_empty())
        .map(|pair| {
            let (d, s) = pair.split_once(':').expect("doc:score");
            (
                d.parse().unwrap(),
                f32::from_bits(s.parse::<i32>().unwrap() as u32),
            )
        })
        .collect()
}

fn float_vec(spec: &str) -> Vec<f32> {
    spec.split(',')
        .map(|s| f32::from_bits(s.parse::<i32>().unwrap() as u32))
        .collect()
}

fn byte_vec(spec: &str) -> Vec<u8> {
    spec.split(',')
        .map(|s| s.parse::<i32>().unwrap() as i8 as u8)
        .collect()
}

fn doc_list(spec: &str) -> Vec<i32> {
    spec.split(',')
        .filter(|s| !s.is_empty())
        .map(|s| s.parse().unwrap())
        .collect()
}

/// Lucene's scores are `float` arithmetic in a different lane order than
/// ours (c5 finding 15), so an exact bit comparison would be dishonest;
/// 1e-6 relative is the tolerance c5 and c13 both use. The **doc ids and
/// their order** are compared exactly, which is the half that discriminates.
fn assert_hits_match(got: &[ScoreDoc], expected: &[(i32, f32)], what: &str) {
    let got_docs: Vec<i32> = got.iter().map(|h| h.doc_id).collect();
    let want_docs: Vec<i32> = expected.iter().map(|(d, _)| *d).collect();
    assert_eq!(got_docs, want_docs, "{what}: doc ids");
    for (i, (g, (_, es))) in got.iter().zip(expected).enumerate() {
        assert!(
            (g.score - es).abs() <= 1e-6 * es.abs().max(1.0),
            "{what}: score at rank {i}: {} vs {es}",
            g.score
        );
    }
}

// ---------------------------------------------------------------------------
// Opening a fixture segment
// ---------------------------------------------------------------------------

/// The byte buffers one segment's `VectorsInput` borrows from.
struct SegmentBytes {
    fnm: Vec<u8>,
    vemf: Vec<u8>,
    vec: Vec<u8>,
    vem: Vec<u8>,
    vex: Vec<u8>,
    id: [u8; 16],
    suffix: String,
    max_doc: i32,
    doc_base: i32,
}

impl SegmentBytes {
    /// The single-segment fixture, whose manifest keys are unprefixed.
    fn single(m: &Manifest) -> Self {
        Self::read(m, "", 0)
    }

    fn read(m: &Manifest, prefix: &str, doc_base: i32) -> Self {
        let name = m.get(&format!("{prefix}segment_name")).to_string();
        SegmentBytes {
            fnm: m.file(&format!("{name}.fnm")),
            vemf: m.file(m.get(&format!("{prefix}vemf_file"))),
            vec: m.file(m.get(&format!("{prefix}vec_file"))),
            vem: m.file(m.get(&format!("{prefix}vem_file"))),
            vex: m.file(m.get(&format!("{prefix}vex_file"))),
            id: segment_id(m.get(&format!("{prefix}id_hex"))),
            suffix: m.get(&format!("{prefix}segment_suffix")).to_string(),
            max_doc: m.int(&format!("{prefix}max_doc")),
            doc_base,
        }
    }

    fn field_infos(&self) -> FieldInfos {
        field_infos::parse(&self.fnm, &self.id, "").expect(".fnm")
    }

    fn input<'a>(
        &'a self,
        infos: &'a FieldInfos,
        with_graph: bool,
        live_docs: Option<&'a FixedBitSet>,
        filter: Option<&'a FixedBitSet>,
    ) -> VectorsInput<'a> {
        VectorsInput {
            flat: FlatVectorsReader::open(&self.vemf, &self.vec, &self.id, &self.suffix)
                .expect(".vemf/.vec"),
            hnsw: with_graph.then(|| {
                HnswVectorsReader::open(&self.vem, &self.vex, &self.id, &self.suffix)
                    .expect(".vem/.vex")
            }),
            field_infos: infos,
            live_docs,
            filter,
            max_doc: self.max_doc,
        }
    }
}

// ---------------------------------------------------------------------------
// Single segment
// ---------------------------------------------------------------------------

/// The headline single-segment differential: every FLOAT32 and BYTE query the
/// fixture records must come back with the same doc ids in the same order and
/// the same scores real Lucene's `KnnFloatVectorQuery`/`KnnByteVectorQuery`
/// produced.
#[test]
fn knn_queries_reproduce_lucene_over_one_segment() {
    let m = Manifest::load("vectors_index");
    let seg = SegmentBytes::single(&m);
    let infos = seg.field_infos();
    let input = seg.input(&infos, true, None, None);
    let mut checked = 0;
    for f in 0..m.int("field_count") {
        let fk = format!("f{f}");
        let Some(count) = m.opt(&format!("q.{fk}.count")) else {
            continue;
        };
        let field = m.get(&format!("{fk}.name")).to_string();
        let float = m.get(&format!("{fk}.encoding")) == "FLOAT32";
        for q in 0..count.parse::<i32>().unwrap() {
            let qk = format!("q.{fk}.{q}");
            let expected = parse_hits(m.get(&format!("{qk}.hnsw")));
            let got = if float {
                let query =
                    KnnFloatVectorQuery::new(&field, float_vec(m.get(&format!("{qk}.vec"))), 10)
                        .unwrap();
                search_knn_float_vector_query(&input, &query).unwrap()
            } else {
                let query =
                    KnnByteVectorQuery::new(&field, byte_vec(m.get(&format!("{qk}.vec"))), 10)
                        .unwrap();
                search_knn_byte_vector_query(&input, &query).unwrap()
            };
            assert_hits_match(&got, &expected, &qk);
            checked += 1;
        }
    }
    assert!(checked >= 60, "expected several fields' worth of queries");
}

/// With no `.vem`/`.vex` opened the dispatch takes its exhaustive branch,
/// which is *exact* — so it must reproduce the fixture's brute-force
/// expectations rather than the graph's.
#[test]
fn without_a_graph_the_search_is_lucenes_exact_brute_force() {
    let m = Manifest::load("vectors_index");
    let seg = SegmentBytes::single(&m);
    let infos = seg.field_infos();
    let input = seg.input(&infos, false, None, None);
    let field = m.get("f0.name").to_string();
    for q in 0..3 {
        let qk = format!("q.f0.{q}");
        let expected = parse_hits(m.get(&format!("{qk}.exact")));
        let query =
            KnnFloatVectorQuery::new(&field, float_vec(m.get(&format!("{qk}.vec"))), 10).unwrap();
        assert_hits_match(
            &search_knn_float_vector_query(&input, &query).unwrap(),
            &expected,
            &qk,
        );
    }
}

/// A sparse field's ordinals are not its doc ids: at least one returned doc
/// id is past the field's ordinal range, so an ordinal-for-doc-id bug could
/// not pass this.
#[test]
fn a_sparse_field_returns_doc_ids_not_ordinals() {
    let m = Manifest::load("vectors_index");
    let seg = SegmentBytes::single(&m);
    let infos = seg.field_infos();
    let input = seg.input(&infos, true, None, None);
    let field = m.get("f1.name").to_string();
    let query = KnnFloatVectorQuery::new(&field, float_vec(m.get("q.f1.0.vec")), 10).unwrap();
    let hits = search_knn_float_vector_query(&input, &query).unwrap();
    assert_hits_match(&hits, &parse_hits(m.get("q.f1.0.hnsw")), "q.f1.0");
    let count = m.int("f1.count");
    assert!(
        hits.iter().any(|h| h.doc_id >= count),
        "expected a doc id past the ordinal range"
    );
}

/// Deletions are filtered **inside** the graph walk (Java's
/// `KnnVectorValues.getAcceptOrds`), so no deleted document can come back and
/// `k` is still filled — and the surviving prefix of the undeleted top-10 is
/// unchanged and in order.
#[test]
fn deleted_documents_never_come_back() {
    let m = Manifest::load("vectors_index");
    let seg = SegmentBytes::single(&m);
    let infos = seg.field_infos();
    let field = m.get("f0.name").to_string();
    let query = KnnFloatVectorQuery::new(&field, float_vec(m.get("q.f0.0.vec")), 10).unwrap();

    let before = {
        let input = seg.input(&infos, true, None, None);
        search_knn_float_vector_query(&input, &query).unwrap()
    };
    let deleted: Vec<i32> = before.iter().take(3).map(|h| h.doc_id).collect();
    let mut live = FixedBitSet::new(seg.max_doc as usize);
    for d in 0..seg.max_doc {
        if !deleted.contains(&d) {
            live.set(d as usize);
        }
    }

    let input = seg.input(&infos, true, Some(&live), None);
    let after = search_knn_float_vector_query(&input, &query).unwrap();
    for h in &after {
        assert!(
            !deleted.contains(&h.doc_id),
            "deleted doc {} returned",
            h.doc_id
        );
    }
    assert_eq!(after.len(), 10, "k is still filled");
    let survivors: Vec<i32> = before
        .iter()
        .map(|h| h.doc_id)
        .filter(|d| !deleted.contains(d))
        .collect();
    let got: Vec<i32> = after.iter().map(|h| h.doc_id).collect();
    assert_eq!(&got[..survivors.len()], &survivors[..]);
}

/// A wider beam is allowed to be *better* than `KnnFloatVectorQuery`'s
/// `k`-wide one, never worse.
#[test]
fn a_wider_ef_search_never_returns_worse_hits() {
    let m = Manifest::load("vectors_index");
    let seg = SegmentBytes::single(&m);
    let infos = seg.field_infos();
    let input = seg.input(&infos, true, None, None);
    let field = m.get("f0.name").to_string();
    let target = float_vec(m.get("q.f0.0.vec"));
    let narrow = search_knn_float_vector_query(
        &input,
        &KnnFloatVectorQuery::new(&field, target.clone(), 10).unwrap(),
    )
    .unwrap();
    let wide = search_knn_float_vector_query(
        &input,
        &KnnFloatVectorQuery::new(&field, target, 10)
            .unwrap()
            .with_ef_search(200),
    )
    .unwrap();
    assert_eq!(wide.len(), 10);
    for (i, (n, w)) in narrow.iter().zip(&wide).enumerate() {
        assert!(
            w.score >= n.score - 1e-6,
            "rank {i}: wider beam scored worse"
        );
    }
}

// ---------------------------------------------------------------------------
// Multi-segment
// ---------------------------------------------------------------------------

struct MultiFixture {
    m: Manifest,
    segments: Vec<SegmentBytes>,
}

impl MultiFixture {
    fn load() -> Self {
        Self::load_named("vectors_multi_index")
    }

    fn load_named(name: &str) -> Self {
        let m = Manifest::load(name);
        let mut segments = Vec::new();
        let mut doc_base = 0;
        for s in 0..m.int("segment_count") {
            let seg = SegmentBytes::read(&m, &format!("s{s}."), doc_base);
            doc_base += seg.max_doc;
            segments.push(seg);
        }
        MultiFixture { m, segments }
    }

    fn infos(&self) -> Vec<FieldInfos> {
        self.segments.iter().map(|s| s.field_infos()).collect()
    }

    fn filter_bitsets(&self, key: &str) -> Vec<FixedBitSet> {
        self.segments
            .iter()
            .enumerate()
            .map(|(s, seg)| {
                let spec = self.m.get(&format!("s{s}.{key}"));
                lucene_search::accept_bitset(doc_list(spec), seg.max_doc)
            })
            .collect()
    }

    fn leaves<'a>(
        &'a self,
        infos: &'a [FieldInfos],
        filters: Option<&'a [FixedBitSet]>,
    ) -> Vec<KnnSegment<'a>> {
        self.segments
            .iter()
            .enumerate()
            .map(|(i, seg)| KnnSegment {
                vectors: seg.input(&infos[i], true, None, filters.map(|f| &f[i])),
                doc_base: seg.doc_base,
            })
            .collect()
    }
}

/// The multi-segment fan-out must reproduce
/// `IndexSearcher.search(new KnnFloatVectorQuery(field, target, k), k)` over
/// a four-segment index **doc for doc in global doc-id space**, which is what
/// pins the pro-rata `perLeafTopK` sizing and the re-entrant second pass:
/// searching each leaf for `k` instead returns a different answer.
#[test]
fn multi_segment_knn_reproduces_lucene() {
    let f = MultiFixture::load();
    let infos = f.infos();
    let leaves = f.leaves(&infos, None);
    let k = f.m.int("k") as usize;
    let mut checked = 0;
    for fk in ["f0", "f1", "f2"] {
        let Some(count) = f.m.opt(&format!("q.{fk}.count")) else {
            continue;
        };
        let field = f.m.get(&format!("{fk}.name")).to_string();
        for q in 0..count.parse::<i32>().unwrap() {
            let qk = format!("q.{fk}.{q}");
            let query =
                KnnFloatVectorQuery::new(&field, float_vec(f.m.get(&format!("{qk}.vec"))), k)
                    .unwrap();
            let got = search_knn_float_vector_query_multi_segment(&leaves, &query).unwrap();
            assert_hits_match(&got, &parse_hits(f.m.get(&format!("{qk}.hnsw"))), &qk);
            checked += 1;
        }
    }
    assert!(checked >= 20, "expected the fixture's full query set");
}

/// The BYTE encoding takes the same fan-out.
#[test]
fn multi_segment_byte_knn_reproduces_lucene() {
    let f = MultiFixture::load();
    let infos = f.infos();
    let leaves = f.leaves(&infos, None);
    let k = f.m.int("k") as usize;
    let field = f.m.get("f3.name").to_string();
    let count: i32 = f.m.get("q.f3.count").parse().unwrap();
    for q in 0..count {
        let qk = format!("q.f3.{q}");
        let query =
            KnnByteVectorQuery::new(&field, byte_vec(f.m.get(&format!("{qk}.vec"))), k).unwrap();
        let got = search_knn_byte_vector_query_multi_segment(&leaves, &query).unwrap();
        assert_hits_match(&got, &parse_hits(f.m.get(&format!("{qk}.hnsw"))), &qk);
    }
}

/// The concurrent fan-out is not "usually the same" as the sequential one —
/// it is the same, because only the per-leaf searches move onto rayon's pool
/// and the merge stays sequential and in segment order.
#[test]
fn the_concurrent_fan_out_returns_exactly_the_sequential_answer() {
    let f = MultiFixture::load();
    let infos = f.infos();
    let leaves = f.leaves(&infos, None);
    let k = f.m.int("k") as usize;
    let field = f.m.get("f0.name").to_string();
    let count: i32 = f.m.get("q.f0.count").parse().unwrap();
    for q in 0..count {
        let qk = format!("q.f0.{q}");
        let query =
            KnnFloatVectorQuery::new(&field, float_vec(f.m.get(&format!("{qk}.vec"))), k).unwrap();
        let seq = search_knn_float_vector_query_multi_segment(&leaves, &query).unwrap();
        let par = search_knn_float_vector_query_multi_segment_concurrent(&leaves, &query).unwrap();
        assert_eq!(seq, par, "{qk}");
        assert_hits_match(&par, &parse_hits(f.m.get(&format!("{qk}.hnsw"))), &qk);
    }
}

/// A **selective** filter (fewer accepted documents per leaf than
/// `perLeafTopK`) takes Java's exact-search short circuit on every leaf, so
/// the answer is not approximate at all and must equal Lucene's
/// `KnnFloatVectorQuery(field, target, k, filter)` exactly.
#[test]
fn a_selective_filter_matches_lucenes_exact_fallback() {
    let f = MultiFixture::load();
    let infos = f.infos();
    let filters = f.filter_bitsets("selective_docs");
    let leaves = f.leaves(&infos, Some(&filters));
    let k = f.m.int("k") as usize;
    let field = f.m.get("f0.name").to_string();
    let count: i32 = f.m.get("q.f0.count").parse().unwrap();
    for q in 0..count {
        let qk = format!("q.f0.{q}");
        let query =
            KnnFloatVectorQuery::new(&field, float_vec(f.m.get(&format!("{qk}.vec"))), k).unwrap();
        let got = search_knn_float_vector_query_multi_segment(&leaves, &query).unwrap();
        assert_hits_match(
            &got,
            &parse_hits(f.m.get(&format!("{qk}.selective"))),
            &format!("{qk} (selective filter)"),
        );
    }
}

/// A **permissive** filter goes down the approximate path with
/// `acceptOrds` inside the graph walk and `visitedLimit = cost + 1`, and must
/// still agree with Lucene doc for doc.
#[test]
fn a_permissive_filter_matches_lucenes_filtered_graph_walk() {
    let f = MultiFixture::load();
    let infos = f.infos();
    let filters = f.filter_bitsets("permissive_docs");
    let leaves = f.leaves(&infos, Some(&filters));
    let k = f.m.int("k") as usize;
    let field = f.m.get("f0.name").to_string();
    let count: i32 = f.m.get("q.f0.count").parse().unwrap();
    for q in 0..count {
        let qk = format!("q.f0.{q}");
        let query =
            KnnFloatVectorQuery::new(&field, float_vec(f.m.get(&format!("{qk}.vec"))), k).unwrap();
        let got = search_knn_float_vector_query_multi_segment(&leaves, &query).unwrap();
        for h in &got {
            let seg = f
                .segments
                .iter()
                .position(|s| h.doc_id >= s.doc_base && h.doc_id < s.doc_base + s.max_doc)
                .expect("a hit inside some segment");
            assert!(
                filters[seg].get((h.doc_id - f.segments[seg].doc_base) as usize),
                "{qk}: doc {} is not in the filter",
                h.doc_id
            );
        }
        assert_hits_match(
            &got,
            &parse_hits(f.m.get(&format!("{qk}.permissive"))),
            &format!("{qk} (permissive filter)"),
        );
    }
}

/// A filter that accepts nothing returns nothing — Java's
/// `cost <= perLeafTopK` short circuit over an empty accept set, not a graph
/// walk that quietly ignores the filter.
#[test]
fn an_empty_filter_returns_no_hits() {
    let f = MultiFixture::load();
    let infos = f.infos();
    let filters: Vec<FixedBitSet> = f
        .segments
        .iter()
        .map(|s| FixedBitSet::new(s.max_doc as usize))
        .collect();
    let leaves = f.leaves(&infos, Some(&filters));
    let field = f.m.get("f0.name").to_string();
    let query = KnnFloatVectorQuery::new(&field, float_vec(f.m.get("q.f0.0.vec")), 10).unwrap();
    assert!(search_knn_float_vector_query_multi_segment(&leaves, &query)
        .unwrap()
        .is_empty());
}

/// The re-entry pass is not decorative on this fixture, and this is the test
/// that says so: the 40-document leaf's `perLeafTopK` is **5** against
/// `k = 10`, so any query whose Lucene answer draws more than five hits from
/// that leaf can only be reproduced by searching it a second time with a
/// full-`k` collector. Five of the twenty dense queries do exactly that.
///
/// Without this, `multi_segment_knn_reproduces_lucene` would still pass on a
/// port that skipped phase 2 *if* no query happened to need it -- which is
/// precisely the kind of silent gap the fixture is built to close.
#[test]
fn the_optimistic_reentry_pass_is_what_fills_k_from_a_small_leaf() {
    let f = MultiFixture::load();
    let small = f.segments.last().expect("four segments");
    let proportion = small.max_doc as f32 / f.m.int("index_max_doc") as f32;
    let per_leaf = lucene_search::per_leaf_top_k(f.m.int("k") as usize, proportion);
    assert_eq!(per_leaf, 5, "the small leaf's phase-1 collector");

    let infos = f.infos();
    let leaves = f.leaves(&infos, None);
    let k = f.m.int("k") as usize;
    let field = f.m.get("f0.name").to_string();
    let count: i32 = f.m.get("q.f0.count").parse().unwrap();
    let mut needed_reentry = 0;
    for q in 0..count {
        let qk = format!("q.f0.{q}");
        let query =
            KnnFloatVectorQuery::new(&field, float_vec(f.m.get(&format!("{qk}.vec"))), k).unwrap();
        let got = search_knn_float_vector_query_multi_segment(&leaves, &query).unwrap();
        let from_small = got.iter().filter(|h| h.doc_id >= small.doc_base).count();
        if from_small > per_leaf {
            needed_reentry += 1;
        }
    }
    assert!(
        needed_reentry >= 5,
        "expected several queries the small leaf can only fill via phase 2, got {needed_reentry}"
    );
}

/// The BYTE fan-out has a concurrent entry point too, and it is the same
/// answer -- and still Lucene's.
#[test]
fn the_concurrent_byte_fan_out_returns_exactly_the_sequential_answer() {
    let f = MultiFixture::load();
    let infos = f.infos();
    let leaves = f.leaves(&infos, None);
    let k = f.m.int("k") as usize;
    let field = f.m.get("f3.name").to_string();
    let count: i32 = f.m.get("q.f3.count").parse().unwrap();
    for q in 0..count {
        let qk = format!("q.f3.{q}");
        let query =
            KnnByteVectorQuery::new(&field, byte_vec(f.m.get(&format!("{qk}.vec"))), k).unwrap();
        let seq = search_knn_byte_vector_query_multi_segment(&leaves, &query).unwrap();
        let par = search_knn_byte_vector_query_multi_segment_concurrent(&leaves, &query).unwrap();
        assert_eq!(seq, par, "{qk}");
        assert_hits_match(&par, &parse_hits(f.m.get(&format!("{qk}.hnsw"))), &qk);
    }
}

/// Deletions on a **sparse** field exercise the ordinal translation the dense
/// case skips: `acceptOrds` there is a fresh bitset built by walking
/// `ordToDoc`, so an off-by-one in it would return documents the caller
/// deleted, or drop live ones.
#[test]
fn deletions_on_a_sparse_field_are_translated_into_ordinal_space() {
    let m = Manifest::load("vectors_index");
    let seg = SegmentBytes::single(&m);
    let infos = seg.field_infos();
    let field = m.get("f1.name").to_string();
    let query = KnnFloatVectorQuery::new(&field, float_vec(m.get("q.f1.0.vec")), 10).unwrap();

    let before = {
        let input = seg.input(&infos, true, None, None);
        search_knn_float_vector_query(&input, &query).unwrap()
    };
    // The sparse field's doc ids are not its ordinals — that is the point.
    assert!(before.iter().any(|h| h.doc_id >= m.int("f1.count")));

    let deleted: Vec<i32> = before.iter().take(4).map(|h| h.doc_id).collect();
    let mut live = FixedBitSet::new(seg.max_doc as usize);
    for d in 0..seg.max_doc {
        if !deleted.contains(&d) {
            live.set(d as usize);
        }
    }
    let input = seg.input(&infos, true, Some(&live), None);
    let after = search_knn_float_vector_query(&input, &query).unwrap();
    for h in &after {
        assert!(
            !deleted.contains(&h.doc_id),
            "deleted doc {} returned",
            h.doc_id
        );
    }
    assert_eq!(after.len(), 10);
    let survivors: Vec<i32> = before
        .iter()
        .map(|h| h.doc_id)
        .filter(|d| !deleted.contains(d))
        .collect();
    let got: Vec<i32> = after.iter().map(|h| h.doc_id).collect();
    assert_eq!(&got[..survivors.len()], &survivors[..]);
}

/// `visited_limit` early-terminates the exhaustive scan the way Java's
/// `KnnCollector.earlyTerminated()` does: a limit below the field's size cuts
/// the scan short and returns whatever was collected, rather than running to
/// completion or failing.
#[test]
fn a_tight_visited_limit_stops_the_exhaustive_scan_early() {
    let m = Manifest::load("vectors_index");
    let seg = SegmentBytes::single(&m);
    let infos = seg.field_infos();
    // No graph opened, so every search takes the exhaustive branch.
    let input = seg.input(&infos, false, None, None);
    let field = m.get("f0.name").to_string();
    let target = float_vec(m.get("q.f0.0.vec"));
    let full = search_knn_float_vector_query(
        &input,
        &KnnFloatVectorQuery::new(&field, target.clone(), 10).unwrap(),
    )
    .unwrap();
    let capped = search_knn_float_vector_query(
        &input,
        &KnnFloatVectorQuery::new(&field, target, 10)
            .unwrap()
            .with_visited_limit(64),
    )
    .unwrap();
    assert_eq!(full.len(), 10);
    assert_eq!(capped.len(), 10, "the first batch already fills k");
    // It really did stop early: the capped scan only ever saw the first
    // batch of ordinals, so its worst hit cannot beat the full scan's.
    assert!(capped[9].score <= full[9].score);
    assert_ne!(
        capped.iter().map(|h| h.doc_id).collect::<Vec<_>>(),
        full.iter().map(|h| h.doc_id).collect::<Vec<_>>(),
        "a limit of 64 over 4000 vectors must not reproduce the full scan"
    );
}

/// `ef_search` widens the **collector** and nothing else. Java's two cost
/// tests (`cost <= perLeafTopK` and `scoreDocs.length >= perLeafTopK`) compare
/// against the un-widened `perLeafTopK`, so a wider beam cannot flip a leaf
/// onto the exact-search branch Java would have walked the graph for — it is
/// documented as buying recall, and taking a different *branch* is a different
/// kind of answer, not a better one.
///
/// What is observable from outside is the contract: the filter is still
/// honoured, `k` is still filled, and no hit is worse than the narrow beam's
/// at the same rank.
#[test]
fn a_wider_beam_still_honours_the_filter_and_never_scores_worse() {
    let f = MultiFixture::load();
    let infos = f.infos();
    let filters = f.filter_bitsets("permissive_docs");
    let leaves = f.leaves(&infos, Some(&filters));
    let k = f.m.int("k") as usize;
    let field = f.m.get("f0.name").to_string();
    let count: i32 = f.m.get("q.f0.count").parse().unwrap();
    for q in 0..count {
        let qk = format!("q.f0.{q}");
        let target = float_vec(f.m.get(&format!("{qk}.vec")));
        let narrow = search_knn_float_vector_query_multi_segment(
            &leaves,
            &KnnFloatVectorQuery::new(&field, target.clone(), k).unwrap(),
        )
        .unwrap();
        let wide = search_knn_float_vector_query_multi_segment(
            &leaves,
            &KnnFloatVectorQuery::new(&field, target, k)
                .unwrap()
                .with_ef_search(500),
        )
        .unwrap();
        assert_eq!(wide.len(), narrow.len(), "{qk}");
        for (i, (n, w)) in narrow.iter().zip(&wide).enumerate() {
            assert!(w.score >= n.score - 1e-6, "{qk}: rank {i} scored worse");
            let seg = f
                .segments
                .iter()
                .position(|s| w.doc_id >= s.doc_base && w.doc_id < s.doc_base + s.max_doc)
                .expect("a hit inside some segment");
            assert!(
                filters[seg].get((w.doc_id - f.segments[seg].doc_base) as usize),
                "{qk}: doc {} is not in the filter",
                w.doc_id
            );
        }
    }
}

// ---------------------------------------------------------------------------
// The seeded re-entry pass (`fixtures/data/vectors_seeded_index`)
// ---------------------------------------------------------------------------

/// The `vectors_multi_index` fixture reaches the optimistic re-entry pass,
/// but only on its 40-document leaf — which is below `shouldCreateGraph` and
/// therefore takes the exhaustive branch, where Java ignores the search
/// strategy entirely. So none of the 80 queries above says anything about
/// seeding.
///
/// `vectors_seeded_index` exists to close that: 1400/700/700/40 documents
/// queried at `k = 100`, with the second 700-document leaf holding a tight
/// cluster every target sits next to. Its `perLeafTopK` is 93 — below `k`,
/// which is what makes a leaf re-enterable at all — and 700 vectors is above
/// `shouldCreateGraph`, so the leaf Java re-enters really does have a graph
/// and really is searched by `SeededHnswGraphSearcher`.
///
/// This test asserts the whole thing doc-for-doc and score-for-score against
/// `IndexSearcher.search(new KnnFloatVectorQuery(field, target, 100), 100)`.
#[test]
fn the_seeded_reentry_pass_reproduces_lucene_over_a_graph_bearing_leaf() {
    let f = MultiFixture::load_named("vectors_seeded_index");
    let infos = f.infos();
    let leaves = f.leaves(&infos, None);
    let k = f.m.int("k") as usize;
    let field = f.m.get("f0.name").to_string();
    let count: i32 = f.m.get("q.f0.count").parse().unwrap();
    for q in 0..count {
        let qk = format!("q.f0.{q}");
        let query =
            KnnFloatVectorQuery::new(&field, float_vec(f.m.get(&format!("{qk}.vec"))), k).unwrap();
        let got = search_knn_float_vector_query_multi_segment(&leaves, &query).unwrap();
        assert_hits_match(&got, &parse_hits(f.m.get(&format!("{qk}.hnsw"))), &qk);
    }
}

/// The same index at `k = 10`, the **control** for the test above: the answer
/// is reproduced with no leaf re-entered at all, so a failure in one and not
/// the other localises the fault.
///
/// "No leaf re-entered" is asserted, not assumed, and from Lucene's own
/// recorded answer rather than from a hook into this port: one pass collects
/// at most `perLeafTopK` hits from a leaf, so a recorded answer taking more
/// than that from any leaf would mean a second pass ran. Asserting it matters
/// because the exclusion here is **by score**, not structural -- at `k = 10`
/// the four `perLeafTopK` values are 30, 24, 24 and 6, and the 40-document
/// leaf's 6 is below `k`. Its uniform, cluster-free vectors are what keep it
/// uncompetitive, and vectors can shift.
#[test]
fn the_seeded_fixture_also_reproduces_lucene_without_any_reentry() {
    let f = MultiFixture::load_named("vectors_seeded_index");
    let infos = f.infos();
    let leaves = f.leaves(&infos, None);
    let k = f.m.int("small_k") as usize;
    let field = f.m.get("f0.name").to_string();
    let count: i32 = f.m.get("q.f0.count").parse().unwrap();
    let index_max_doc: i32 = f.segments.iter().map(|s| s.max_doc).sum();
    let caps: Vec<usize> = f
        .segments
        .iter()
        .map(|seg| {
            lucene_search::vector_query::per_leaf_top_k(
                k,
                seg.max_doc as f32 / index_max_doc as f32,
            )
        })
        .collect();
    for q in 0..count {
        let qk = format!("q.f0.{q}");
        let query =
            KnnFloatVectorQuery::new(&field, float_vec(f.m.get(&format!("{qk}.vec"))), k).unwrap();
        let got = search_knn_float_vector_query_multi_segment(&leaves, &query).unwrap();
        let want = parse_hits(f.m.get(&format!("{qk}.hnsw_small_k")));
        assert_hits_match(&got, &want, &qk);
        for (s, seg) in f.segments.iter().enumerate() {
            let end = seg.doc_base + seg.max_doc;
            let from_leaf = want
                .iter()
                .filter(|(d, _)| *d >= seg.doc_base && *d < end)
                .count();
            assert!(
                from_leaf <= caps[s],
                "{qk}: leaf {s} contributed {from_leaf} hits with perLeafTopK = {}, so Lucene \
                 re-entered it and this is no longer a no-re-entry control",
                caps[s]
            );
        }
    }
}

/// A fixture that quietly stops reaching the branch it exists for proves
/// nothing, so the branch's preconditions are asserted directly rather than
/// assumed:
///
/// 1. the clustered leaf's `perLeafTopK` is **below** `k`, without which no
///    leaf can satisfy `perLeaf.scoreDocs[len-1].score >= minTopKScore`
///    except through a score tie. (The recorded `perLeafTopK` values are
///    **re-derived** in the generator, because
///    `AbstractKnnVectorQuery.perLeafTopKCalculation` is private -- so
///    assertion 1 compares this port's formula against a hand-copy of the
///    same formula and cannot catch a shared misreading. It is a tripwire on
///    the fixture's shape; assertion 3 is the one with real weight.)
/// 2. that leaf carries an HNSW graph, without which the seeded search is
///    never consulted (the exhaustive branch ignores the strategy);
/// 3. and Lucene's own answer takes **more** than `perLeafTopK` hits from
///    that leaf — which phase 1 cannot produce, since its collector holds
///    exactly `perLeafTopK`. That third one is the assertion that fails if
///    the re-entry pass stops firing.
#[test]
fn the_seeded_fixture_still_reaches_the_reentry_pass_on_a_leaf_with_a_graph() {
    let f = MultiFixture::load_named("vectors_seeded_index");
    let k = f.m.int("k") as usize;
    let clustered = f.m.int("clustered_segment") as usize;
    let index_max_doc: i32 = f.segments.iter().map(|s| s.max_doc).sum();
    assert_eq!(index_max_doc, f.m.int("index_max_doc"));

    // 1. The pro-rata collector sizes Lucene recorded, reproduced here.
    for (s, seg) in f.segments.iter().enumerate() {
        let want = f.m.int(&format!("s{s}.per_leaf_top_k")) as usize;
        let got = lucene_search::vector_query::per_leaf_top_k(
            k,
            seg.max_doc as f32 / index_max_doc as f32,
        );
        assert_eq!(got, want, "leaf {s}");
    }
    let per_leaf_top_k = f.m.int(&format!("s{clustered}.per_leaf_top_k")) as usize;
    assert!(
        per_leaf_top_k < k,
        "leaf {clustered} collects {per_leaf_top_k} of k = {k}: it cannot be re-entered"
    );

    // 2. It has a graph, so the seeded searcher is what runs on the re-entry.
    let infos = f.infos();
    let seg = &f.segments[clustered];
    let input = seg.input(&infos[clustered], true, None, None);
    let number = infos[clustered]
        .fields
        .iter()
        .find(|fi| fi.name == f.m.get("f0.name"))
        .expect("the vector field")
        .number;
    assert!(
        input
            .hnsw
            .as_ref()
            .unwrap()
            .graph(number)
            .unwrap()
            .is_some(),
        "leaf {clustered} was written without a graph, so seeding cannot reach it"
    );

    // 3. Lucene's own answer takes more from that leaf than one pass could.
    let base = seg.doc_base;
    let end = base + seg.max_doc;
    let count: i32 = f.m.get("q.f0.count").parse().unwrap();
    let mut seen_beyond_phase_one = 0;
    for q in 0..count {
        let hits = parse_hits(f.m.get(&format!("q.f0.{q}.hnsw")));
        let from_leaf = hits.iter().filter(|(d, _)| *d >= base && *d < end).count();
        if from_leaf > per_leaf_top_k {
            seen_beyond_phase_one += 1;
        }
    }
    assert!(
        seen_beyond_phase_one > 0,
        "no recorded query takes more than {per_leaf_top_k} hits from leaf {clustered}, so the \
         re-entry pass never fires and this fixture no longer covers seeding"
    );
}

// ---------------------------------------------------------------------------
// Single-segment filtered KNN (`fixtures/data/vectors_filter_index`)
// ---------------------------------------------------------------------------

/// A one-leaf index, where `leafProportion == 1` makes `perLeafTopK == k` and
/// `perLeafResults.size() > 1` false — so `IndexSearcher.search(query, k)` is
/// exactly [`search_knn_float_vector_query`] with no pro-rata sizing and no
/// re-entry. That is the ground truth `lucene-ffi` needs, since it opens one
/// segment per handle.
struct FilterFixture {
    m: Manifest,
    seg: SegmentBytes,
}

impl FilterFixture {
    fn load() -> Self {
        let m = Manifest::load("vectors_filter_index");
        let seg = SegmentBytes::single(&m);
        FilterFixture { m, seg }
    }

    fn filter(&self, key: &str) -> FixedBitSet {
        lucene_search::accept_bitset(doc_list(self.m.get(key)), self.seg.max_doc)
    }
}

/// Both of Java's filtered branches over one leaf, plus the unfiltered
/// control, for both encodings — doc for doc and score for score.
///
/// `selective` (6 accepted documents against `k = 10`) is the
/// `cost <= perLeafTopK` short circuit into `exactSearch`; `permissive` (a
/// quarter of the index) is the graph walk with `acceptOrds` and
/// `visitedLimit = cost + 1`.
#[test]
fn single_segment_filtered_knn_reproduces_lucene() {
    let f = FilterFixture::load();
    let infos = f.seg.field_infos();
    let k = f.m.int("k") as usize;
    let selective = f.filter("selective_docs");
    let permissive = f.filter("permissive_docs");
    assert_eq!(selective.cardinality(), 6, "the exact-search branch's cost");
    assert!(permissive.cardinality() > k, "the graph-walk branch's cost");

    let mut checked = 0;
    for (fk, byte) in [("f0", false), ("f1", true)] {
        let field = f.m.get(&format!("{fk}.name")).to_string();
        let count: i32 = f.m.get(&format!("q.{fk}.count")).parse().unwrap();
        for q in 0..count {
            let qk = format!("q.{fk}.{q}");
            let spec = f.m.get(&format!("{qk}.vec")).to_string();
            for (key, filter) in [
                ("hnsw", None),
                ("selective", Some(&selective)),
                ("permissive", Some(&permissive)),
            ] {
                let input = f.seg.input(&infos, true, None, filter);
                let got = if byte {
                    let query = KnnByteVectorQuery::new(&field, byte_vec(&spec), k).unwrap();
                    search_knn_byte_vector_query(&input, &query).unwrap()
                } else {
                    let query = KnnFloatVectorQuery::new(&field, float_vec(&spec), k).unwrap();
                    search_knn_float_vector_query(&input, &query).unwrap()
                };
                let want = parse_hits(f.m.get(&format!("{qk}.{key}")));
                assert_hits_match(&got, &want, &format!("{qk}.{key}"));
                checked += 1;
            }
        }
    }
    assert_eq!(checked, 120, "the fixture's whole query set");
}

/// Every hit a filtered search returns is in the filter — the assertion a
/// graph walk that quietly ignored `acceptOrds` would fail even while its
/// doc-for-doc comparison happened to pass on an unselective filter.
#[test]
fn a_single_segment_filtered_hit_is_always_in_the_filter() {
    let f = FilterFixture::load();
    let infos = f.seg.field_infos();
    let k = f.m.int("k") as usize;
    let field = f.m.get("f0.name").to_string();
    for key in ["selective_docs", "permissive_docs"] {
        let filter = f.filter(key);
        let input = f.seg.input(&infos, true, None, Some(&filter));
        for q in 0..f.m.get("q.f0.count").parse::<i32>().unwrap() {
            let spec = f.m.get(&format!("q.f0.{q}.vec")).to_string();
            let query = KnnFloatVectorQuery::new(&field, float_vec(&spec), k).unwrap();
            let got = search_knn_float_vector_query(&input, &query).unwrap();
            assert!(!got.is_empty());
            for hit in &got {
                assert!(
                    filter.get(hit.doc_id as usize),
                    "{key}: doc {} is not in the filter",
                    hit.doc_id
                );
            }
        }
    }
}
