//! What the KNN fan-out and the filtered path actually cost.
//!
//! Two questions this bench exists to answer, both raised by the
//! `c16-knn-query` sweep batch:
//!
//! 1. **How does the multi-segment fan-out scale?** Not "linearly in the
//!    number of segments" -- Java's pro-rata `perLeafTopK` makes each leaf's
//!    collector *larger* than `k` when there are several leaves (for four
//!    leaves and `k = 10` it is 30, 24, 24 and 5), and a wider collector is a
//!    wider beam, so four segments cost visibly more than one segment of the
//!    same total size even before any concurrency. The sequential and
//!    concurrent rows say what rayon buys back on a 4-leaf index.
//! 2. **What does a filter cost?** Java takes one of two paths depending on
//!    how many documents the filter accepts: an exact scan when the accepted
//!    set is smaller than `perLeafTopK`, and a graph walk with `acceptOrds`
//!    otherwise. Those have very different shapes and both are measured.
//!
//! `cargo bench -p lucene-search --bench knn_multi_segment`

use std::collections::HashMap;

use criterion::{criterion_group, criterion_main, BatchSize, Criterion};

use lucene_codecs::field_infos::{self, FieldInfos};
use lucene_codecs::hnsw_vectors::HnswVectorsReader;
use lucene_codecs::vectors::FlatVectorsReader;
use lucene_search::vector_query::{
    search_knn_float_vector_query, search_knn_float_vector_query_multi_segment,
    search_knn_float_vector_query_multi_segment_concurrent, KnnFloatVectorQuery, KnnSegment,
    VectorsInput,
};
use lucene_util::fixed_bit_set::FixedBitSet;

struct Manifest {
    dir: String,
    kv: HashMap<String, String>,
}

impl Manifest {
    fn load(name: &str) -> Self {
        let dir = format!("{}/../../fixtures/data/{name}/", env!("CARGO_MANIFEST_DIR"));
        let text = std::fs::read_to_string(format!("{dir}manifest.properties"))
            .expect("run scripts/gen-fixtures.sh first");
        Manifest {
            dir,
            kv: text
                .lines()
                .filter_map(|l| l.split_once('='))
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
        }
    }
    fn get(&self, key: &str) -> &str {
        self.kv.get(key).expect("manifest key")
    }
    fn file(&self, name: &str) -> Vec<u8> {
        std::fs::read(format!("{}{name}", self.dir)).expect("fixture file")
    }
}

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
    fn read(m: &Manifest, prefix: &str, doc_base: i32) -> Self {
        let name = m.get(&format!("{prefix}segment_name")).to_string();
        let hex = m.get(&format!("{prefix}id_hex"));
        let mut id = [0u8; 16];
        for (i, slot) in id.iter_mut().enumerate() {
            *slot = u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16).unwrap();
        }
        SegmentBytes {
            fnm: m.file(&format!("{name}.fnm")),
            vemf: m.file(m.get(&format!("{prefix}vemf_file"))),
            vec: m.file(m.get(&format!("{prefix}vec_file"))),
            vem: m.file(m.get(&format!("{prefix}vem_file"))),
            vex: m.file(m.get(&format!("{prefix}vex_file"))),
            id,
            suffix: m.get(&format!("{prefix}segment_suffix")).to_string(),
            max_doc: m.get(&format!("{prefix}max_doc")).parse().unwrap(),
            doc_base,
        }
    }

    fn field_infos(&self) -> FieldInfos {
        field_infos::parse(&self.fnm, &self.id, "").expect(".fnm")
    }

    fn input<'a>(
        &'a self,
        infos: &'a FieldInfos,
        filter: Option<&'a FixedBitSet>,
    ) -> VectorsInput<'a> {
        VectorsInput {
            flat: FlatVectorsReader::open(&self.vemf, &self.vec, &self.id, &self.suffix).unwrap(),
            hnsw: Some(
                HnswVectorsReader::open(&self.vem, &self.vex, &self.id, &self.suffix).unwrap(),
            ),
            field_infos: infos,
            live_docs: None,
            filter,
            max_doc: self.max_doc,
        }
    }
}

fn float_vec(spec: &str) -> Vec<f32> {
    spec.split(',')
        .map(|s| f32::from_bits(s.parse::<i32>().unwrap() as u32))
        .collect()
}

fn doc_list(spec: &str) -> Vec<i32> {
    spec.split(',')
        .filter(|s| !s.is_empty())
        .map(|s| s.parse().unwrap())
        .collect()
}

fn bench(c: &mut Criterion) {
    // One 4000-document segment: the "same data, one leaf" baseline.
    let single_m = Manifest::load("vectors_index");
    let single = SegmentBytes::read(&single_m, "", 0);
    let single_infos = single.field_infos();
    let single_field = single_m.get("f0.name").to_string();
    let single_targets: Vec<Vec<f32>> = (0..20)
        .map(|q| float_vec(single_m.get(&format!("q.f0.{q}.vec"))))
        .collect();

    // The same 4000 documents split across four unequal segments.
    let multi_m = Manifest::load("vectors_multi_index");
    let segment_count: i32 = multi_m.get("segment_count").parse().unwrap();
    let mut segments = Vec::new();
    let mut doc_base = 0;
    for s in 0..segment_count {
        let seg = SegmentBytes::read(&multi_m, &format!("s{s}."), doc_base);
        doc_base += seg.max_doc;
        segments.push(seg);
    }
    let multi_infos: Vec<FieldInfos> = segments.iter().map(|s| s.field_infos()).collect();
    let multi_field = multi_m.get("f0.name").to_string();
    let multi_targets: Vec<Vec<f32>> = (0..20)
        .map(|q| float_vec(multi_m.get(&format!("q.f0.{q}.vec"))))
        .collect();

    let selective: Vec<FixedBitSet> = segments
        .iter()
        .enumerate()
        .map(|(s, seg)| {
            lucene_search::accept_bitset(
                doc_list(multi_m.get(&format!("s{s}.selective_docs"))),
                seg.max_doc,
            )
        })
        .collect();
    let permissive: Vec<FixedBitSet> = segments
        .iter()
        .enumerate()
        .map(|(s, seg)| {
            lucene_search::accept_bitset(
                doc_list(multi_m.get(&format!("s{s}.permissive_docs"))),
                seg.max_doc,
            )
        })
        .collect();

    fn leaves<'a>(
        segments: &'a [SegmentBytes],
        infos: &'a [FieldInfos],
        filters: Option<&'a [FixedBitSet]>,
    ) -> Vec<KnnSegment<'a>> {
        segments
            .iter()
            .enumerate()
            .map(|(i, seg)| KnnSegment {
                vectors: seg.input(&infos[i], filters.map(|f| &f[i])),
                doc_base: seg.doc_base,
            })
            .collect()
    }

    let mut group = c.benchmark_group("knn");
    // 20 queries per iteration, so one sample is a stable batch rather than
    // one lucky graph walk.
    group.bench_function("one_segment_4000_docs", |b| {
        let input = single.input(&single_infos, None);
        b.iter_batched(
            || (),
            |()| {
                for t in &single_targets {
                    let q = KnnFloatVectorQuery::new(&single_field, t.clone(), 10).unwrap();
                    std::hint::black_box(search_knn_float_vector_query(&input, &q).unwrap());
                }
            },
            BatchSize::SmallInput,
        );
    });

    group.bench_function("four_segments_sequential", |b| {
        let ls = leaves(&segments, &multi_infos, None);
        b.iter(|| {
            for t in &multi_targets {
                let q = KnnFloatVectorQuery::new(&multi_field, t.clone(), 10).unwrap();
                std::hint::black_box(search_knn_float_vector_query_multi_segment(&ls, &q).unwrap());
            }
        });
    });

    group.bench_function("four_segments_concurrent", |b| {
        let ls = leaves(&segments, &multi_infos, None);
        b.iter(|| {
            for t in &multi_targets {
                let q = KnnFloatVectorQuery::new(&multi_field, t.clone(), 10).unwrap();
                std::hint::black_box(
                    search_knn_float_vector_query_multi_segment_concurrent(&ls, &q).unwrap(),
                );
            }
        });
    });

    // The crossover pair: `k = 100` makes each leaf's collector (and so its
    // beam) roughly ten times wider, which is what tells rayon's per-task
    // dispatch cost apart from the search itself.
    group.bench_function("four_segments_sequential_k100", |b| {
        let ls = leaves(&segments, &multi_infos, None);
        b.iter(|| {
            for t in &multi_targets {
                let q = KnnFloatVectorQuery::new(&multi_field, t.clone(), 100).unwrap();
                std::hint::black_box(search_knn_float_vector_query_multi_segment(&ls, &q).unwrap());
            }
        });
    });

    group.bench_function("four_segments_concurrent_k100", |b| {
        let ls = leaves(&segments, &multi_infos, None);
        b.iter(|| {
            for t in &multi_targets {
                let q = KnnFloatVectorQuery::new(&multi_field, t.clone(), 100).unwrap();
                std::hint::black_box(
                    search_knn_float_vector_query_multi_segment_concurrent(&ls, &q).unwrap(),
                );
            }
        });
    });

    group.bench_function("four_segments_filter_selective_20_docs", |b| {
        let ls = leaves(&segments, &multi_infos, Some(&selective));
        b.iter(|| {
            for t in &multi_targets {
                let q = KnnFloatVectorQuery::new(&multi_field, t.clone(), 10).unwrap();
                std::hint::black_box(search_knn_float_vector_query_multi_segment(&ls, &q).unwrap());
            }
        });
    });

    group.bench_function("four_segments_filter_permissive_1000_docs", |b| {
        let ls = leaves(&segments, &multi_infos, Some(&permissive));
        b.iter(|| {
            for t in &multi_targets {
                let q = KnnFloatVectorQuery::new(&multi_field, t.clone(), 10).unwrap();
                std::hint::black_box(search_knn_float_vector_query_multi_segment(&ls, &q).unwrap());
            }
        });
    });

    // The seeded re-entry pass, which the `vectors_multi_index` rows above do
    // *not* measure: the only leaf they re-enter is the 40-document one,
    // which has no graph and so ignores the search strategy.
    // `vectors_seeded_index` is built so that a 700-document leaf **with** a
    // graph is re-entered at `k = 100` (its `perLeafTopK` is 93), which is
    // where `SeededHnswGraphSearcher` earns its keep: phase 2 restarts level
    // 0's beam from phase 1's own 93 hits instead of descending the graph
    // again.
    let seeded_m = Manifest::load("vectors_seeded_index");
    let seeded_count: i32 = seeded_m.get("segment_count").parse().unwrap();
    let mut seeded_segments = Vec::new();
    let mut base = 0;
    for s in 0..seeded_count {
        let seg = SegmentBytes::read(&seeded_m, &format!("s{s}."), base);
        base += seg.max_doc;
        seeded_segments.push(seg);
    }
    let seeded_infos: Vec<FieldInfos> = seeded_segments.iter().map(|s| s.field_infos()).collect();
    let seeded_field = seeded_m.get("f0.name").to_string();
    let seeded_targets: Vec<Vec<f32>> = (0..20)
        .map(|q| float_vec(seeded_m.get(&format!("q.f0.{q}.vec"))))
        .collect();

    group.bench_function("seeded_reentry_four_segments_k100", |b| {
        let ls = leaves(&seeded_segments, &seeded_infos, None);
        b.iter(|| {
            for t in &seeded_targets {
                let q = KnnFloatVectorQuery::new(&seeded_field, t.clone(), 100).unwrap();
                std::hint::black_box(search_knn_float_vector_query_multi_segment(&ls, &q).unwrap());
            }
        });
    });

    // The same index at a `k` no leaf can be re-entered at: the phase-1-only
    // cost the row above is measured against.
    group.bench_function("seeded_index_four_segments_k10_no_reentry", |b| {
        let ls = leaves(&seeded_segments, &seeded_infos, None);
        b.iter(|| {
            for t in &seeded_targets {
                let q = KnnFloatVectorQuery::new(&seeded_field, t.clone(), 10).unwrap();
                std::hint::black_box(search_knn_float_vector_query_multi_segment(&ls, &q).unwrap());
            }
        });
    });

    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
