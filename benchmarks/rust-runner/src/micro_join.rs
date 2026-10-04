//! The block-join benchmark pair (M10 T10.2), against
//! `benchmarks/micro/java/JoinMicro.java`: `ToParentBlockJoinQuery` in each
//! score mode, alone and inside a boolean, `ToChildBlockJoinQuery`,
//! `ParentsChildrenBlockJoinQuery`, `ParentChildrenBlockJoinQuery` over 200
//! parents, a `ToParentBlockJoinSortField` sort and
//! `DiversifyingChildrenFloatKnnVectorQuery` (unfiltered and with a child
//! filter), each the top 10 through `IndexSearcher.search`, over the 60 000-block index the Java side builds
//! (`JoinMicro build <dir>`). Each case prints a `#check` digest of its hits
//! the report compares before it shows a ratio.

use std::hint::black_box;
use std::sync::Arc;
use std::time::Duration;

use lucene_codecs::hnsw_vectors::HnswVectorsReader;
use lucene_codecs::vectors::FlatVectorsReader;
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::index_searcher::{IndexSearcher, SegmentNorms};
use lucene_search::join::{
    BitSetProducer, DiversifyingChildrenFloatKnnVectorQuery, JoinSortType,
    ParentChildrenBlockJoinQuery, ParentsChildrenBlockJoinQuery, QueryBitSetProducer, ScoreMode,
    ToChildBlockJoinQuery, ToParentBlockJoinQuery, ToParentBlockJoinSortField,
};
use lucene_search::query::{BooleanQuery, Clause, TermQuery};
use lucene_search::vector_query::{filter_bitsets, KnnFloatVectorQuery, KnnSegment, VectorsInput};
use lucene_store::FsDirectory;

use super::measure;

/// FNV-1a over 64-bit words, identical to `JoinMicro.Fnv`.
struct Fnv(u64);

impl Fnv {
    fn new() -> Self {
        Fnv(0xcbf2_9ce4_8422_2325)
    }
    fn add(&mut self, x: i64) {
        self.0 ^= x as u64;
        self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
    }
}

fn check(case: &str, d: &Fnv, n: u64) {
    println!("#check\t{case}\t{:016x}\t{n}", d.0);
}

fn term(field: &str, value: &str) -> Clause {
    Clause::Term(TermQuery::new(field, value.as_bytes().to_vec()))
}

fn one(c: Clause) -> BooleanQuery {
    BooleanQuery {
        must: vec![c],
        ..Default::default()
    }
}

/// `JoinMicro.level`: `+body:word #type:type`.
fn level(ty: &str, word: &str) -> Clause {
    Clause::Boolean(Box::new(BooleanQuery {
        filter: vec![term("type", ty)],
        must: vec![term("body", word)],
        ..Default::default()
    }))
}

fn cases(name: &str, s: &IndexSearcher<'_, '_>, qs: &[BooleanQuery], w: Duration, m: Duration) {
    let mut f = Fnv::new();
    for q in qs {
        for h in s.search(q, 10).unwrap().score_docs {
            f.add(i64::from(h.doc));
            f.add(i64::from(h.score.to_bits() as i32));
        }
    }
    check(name, &f, qs.len() as u64);
    measure(name, w, m, || {
        for q in black_box(qs) {
            black_box(s.search(q, 10).unwrap().total_hits.value);
        }
        qs.len() as u64
    });
}

pub fn bench_join(w: Duration, m: Duration, dir: &str) {
    let text = std::fs::read_to_string(format!("{dir}/join-words.tsv"))
        .expect("run JoinMicro build first (scripts/bench-micro.sh --bench join)");
    let words: Vec<(String, String)> = text
        .lines()
        .map(|l| {
            let (a, b) = l.split_once('\t').unwrap();
            (a.to_string(), b.to_string())
        })
        .collect();
    let reader = DirectoryReader::open(&FsDirectory::open(std::path::Path::new(dir))).unwrap();
    let opened = reader.open_segments().unwrap();
    let segments = opened.as_open_segments();
    let owned = reader.field_norms_by_field(&["body".to_string()]);
    let norms: Vec<SegmentNorms<'_, '_>> = owned.iter().map(Some).collect();
    let s = IndexSearcher::new(&segments, &norms).unwrap();
    let parents: Arc<dyn BitSetProducer> =
        Arc::new(QueryBitSetProducer::new(one(term("type", "parent"))));
    let children: Arc<dyn BitSetProducer> =
        Arc::new(QueryBitSetProducer::new(one(term("type", "child"))));

    for (name, mode) in [
        ("none", ScoreMode::None),
        ("avg", ScoreMode::Avg),
        ("max", ScoreMode::Max),
        ("total", ScoreMode::Total),
        ("min", ScoreMode::Min),
    ] {
        let qs: Vec<BooleanQuery> = words
            .iter()
            .map(|(a, _)| {
                one(
                    ToParentBlockJoinQuery::new(level("child", a), Arc::clone(&parents), mode)
                        .into(),
                )
            })
            .collect();
        cases(&format!("join_to_parent_{name}"), &s, &qs, w, m);
    }
    let nested: Vec<BooleanQuery> = words
        .iter()
        .map(|(a, b)| BooleanQuery {
            must: vec![ToParentBlockJoinQuery::new(
                level("child", a),
                Arc::clone(&parents),
                ScoreMode::Avg,
            )
            .into()],
            should: vec![term("body", b)],
            ..Default::default()
        })
        .collect();
    cases("join_to_parent_in_bool", &s, &nested, w, m);
    let to_child: Vec<BooleanQuery> = words
        .iter()
        .map(|(a, _)| {
            one(ToChildBlockJoinQuery::new(level("parent", a), Arc::clone(&parents)).into())
        })
        .collect();
    cases("join_to_child", &s, &to_child, w, m);
    let parents_children: Vec<BooleanQuery> = words
        .iter()
        .map(|(a, b)| {
            one(ParentsChildrenBlockJoinQuery::new(
                Arc::clone(&parents),
                level("parent", a),
                level("child", b),
                3,
            )
            .unwrap()
            .into())
        })
        .collect();
    cases("join_parents_children", &s, &parents_children, w, m);
    // The children of 200 parents, every 250th from the first.
    let bits = parents.bit_set(&segments[0]).unwrap().unwrap();
    let mut parent = bits.next_set_bit(0).unwrap();
    let mut parent_children = Vec::new();
    for i in 0..200 {
        parent_children.push(one(ParentChildrenBlockJoinQuery::new(
            Arc::clone(&parents),
            level("child", &words[i % words.len()].0),
            parent as i32,
        )
        .into()));
        for _ in 0..250 {
            if parent + 1 >= reader.max_doc() as usize {
                break;
            }
            parent = bits.next_set_bit(parent + 1).unwrap();
        }
    }
    cases("join_parent_children", &s, &parent_children, w, m);
    // Parents by their children's lowest price.
    let sort_qs: Vec<BooleanQuery> = words[..8]
        .iter()
        .map(|(a, _)| one(level("parent", a)))
        .collect();
    let sort = [ToParentBlockJoinSortField::new(
        "price",
        JoinSortType::Long,
        false,
        false,
        None,
        None,
        Arc::clone(&parents),
        Arc::clone(&children),
    )
    .unwrap()
    .sort_field()];
    let readers = reader.segment_readers();
    let mut f = Fnv::new();
    for q in &sort_qs {
        for h in s.search_sorted(readers, q, 10, &sort, None).unwrap().hits {
            f.add(i64::from(h.doc));
            f.add(h.values[0]);
        }
    }
    check("join_sort", &f, sort_qs.len() as u64);
    measure("join_sort", w, m, || {
        for q in black_box(&sort_qs) {
            black_box(
                s.search_sorted(readers, q, 10, &sort, None)
                    .unwrap()
                    .hits
                    .len(),
            );
        }
        sort_qs.len() as u64
    });

    // The nearest child of each of the ten nearest parents, unfiltered and
    // filtered (the filter's bit set built per search, as Java's rewrite does).
    let vectors: Vec<Vec<f32>> = std::fs::read_to_string(format!("{dir}/join-vectors.tsv"))
        .unwrap()
        .lines()
        .map(|l| {
            l.split('\t')
                .map(|h| f32::from_bits(u32::from_str_radix(h, 16).unwrap()))
                .collect()
        })
        .collect();
    let seg = &readers[0];
    const SUFFIX: &str = "Lucene99HnswVectorsFormat_0";
    let file =
        |ext: &str| std::fs::read(format!("{dir}/{}_{SUFFIX}.{ext}", seg.segment_name)).unwrap();
    let (vemf, vec, vem, vex) = (file("vemf"), file("vec"), file("vem"), file("vex"));
    let id = seg.segment_id();
    let flat = FlatVectorsReader::open(&vemf, &vec, &id, SUFFIX).unwrap();
    let hnsw = HnswVectorsReader::open(&vem, &vex, &id, SUFFIX).unwrap();
    let queries: Vec<DiversifyingChildrenFloatKnnVectorQuery> = vectors
        .iter()
        .map(|v| {
            DiversifyingChildrenFloatKnnVectorQuery::new(
                KnnFloatVectorQuery::new("vec", v.clone(), 10).unwrap(),
                Arc::clone(&parents),
            )
        })
        .collect();
    let filters: Vec<Clause> = (0..queries.len())
        .map(|i| level("child", &words[i % words.len()].0))
        .collect();
    let search = |i: usize, filtered: bool| {
        let bits = filtered.then(|| filter_bitsets(&segments, &filters[i]).unwrap());
        let knn = [KnnSegment {
            vectors: VectorsInput {
                flat: flat.clone(),
                hnsw: Some(hnsw.clone().into()),
                field_infos: seg.field_infos(),
                live_docs: seg.live_docs(),
                filter: bits.as_ref().map(|b| &b[0]),
                max_doc: seg.max_doc,
            },
            doc_base: seg.doc_base,
        }];
        queries[i].search(&segments, &knn).unwrap()
    };
    for (name, filtered) in [
        ("join_knn_diversify", false),
        ("join_knn_diversify_filtered", true),
    ] {
        let mut f = Fnv::new();
        for i in 0..queries.len() {
            for h in search(i, filtered) {
                f.add(i64::from(h.doc_id));
                f.add(i64::from(h.score.to_bits() as i32));
            }
        }
        check(name, &f, queries.len() as u64);
        measure(name, w, m, || {
            for i in 0..queries.len() {
                black_box(search(black_box(i), filtered).len());
            }
            queries.len() as u64
        });
    }
}
