//! The query-time join benchmark pair (M10 T10.3), against
//! `benchmarks/micro/java/QueryJoinMicro.java`: per word, `JoinUtil`'s
//! `createJoinQuery` over `+type:from +body:word` (collecting the from side)
//! and the top 10 of the join it returns -- the terms join in each score mode
//! (single-valued) and multi-valued, the numeric join, and the
//! global-ordinals join -- over the 200 000-document, four-segment index the
//! Java side builds (`QueryJoinMicro build <dir>`). Each case prints a
//! `#check` digest of its hits the report compares before it shows a ratio.

use std::hint::black_box;
use std::sync::Arc;
use std::time::Duration;

use lucene_search::directory_reader::DirectoryReader;
use lucene_search::index_searcher::{IndexSearcher, SegmentNorms};
use lucene_search::join::{
    create_global_ordinals_join_query, create_join_query, create_numeric_join_query, ordinal_map,
    NumericType, ScoreMode,
};
use lucene_search::query::{BooleanQuery, Clause, TermQuery};
use lucene_store::FsDirectory;

use super::measure;

/// FNV-1a over 64-bit words, identical to `QueryJoinMicro.Fnv`.
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

fn term(field: &str, value: &str) -> Clause {
    Clause::Term(TermQuery::new(field, value.as_bytes().to_vec()))
}

fn one(c: Clause) -> BooleanQuery {
    BooleanQuery {
        must: vec![c],
        ..Default::default()
    }
}

/// `QueryJoinMicro.from`: `+body:word #type:from`.
fn from(word: &str) -> Clause {
    Clause::Boolean(Box::new(BooleanQuery {
        filter: vec![term("type", "from")],
        must: vec![term("body", word)],
        ..Default::default()
    }))
}

fn cases(
    name: &str,
    s: &IndexSearcher<'_, '_>,
    words: &[String],
    join: &dyn Fn(&str) -> Clause,
    w: Duration,
    m: Duration,
) {
    let mut f = Fnv::new();
    for word in words {
        for h in s.search(&one(join(word)), 10).unwrap().score_docs {
            f.add(i64::from(h.doc));
            f.add(i64::from(h.score.to_bits() as i32));
        }
    }
    println!("#check\t{name}\t{:016x}\t{}", f.0, words.len());
    measure(name, w, m, || {
        for word in black_box(words) {
            black_box(s.search(&one(join(word)), 10).unwrap().total_hits.value);
        }
        words.len() as u64
    });
}

pub fn bench_query_join(w: Duration, m: Duration, dir: &str) {
    let words: Vec<String> = std::fs::read_to_string(format!("{dir}/qjoin-words.tsv"))
        .expect("run QueryJoinMicro build first (scripts/bench-micro.sh --bench query_join)")
        .lines()
        .map(str::to_string)
        .collect();
    let reader = DirectoryReader::open(&FsDirectory::open(std::path::Path::new(dir))).unwrap();
    let mut opened = reader.open_segments().unwrap();
    opened.open_points().unwrap();
    let segments = opened.as_open_segments();
    let owned = reader.field_norms_by_field(&["body".to_string()]);
    let norms: Vec<SegmentNorms<'_, '_>> = owned.iter().map(Some).collect();
    let s = IndexSearcher::new(&segments, &norms).unwrap();

    for (name, mode) in [
        ("none", ScoreMode::None),
        ("avg", ScoreMode::Avg),
        ("max", ScoreMode::Max),
        ("total", ScoreMode::Total),
        ("min", ScoreMode::Min),
    ] {
        cases(
            &format!("qjoin_terms_{name}"),
            &s,
            &words,
            &|word| create_join_query("fk", false, "pk", &from(word), &s, mode).unwrap(),
            w,
            m,
        );
    }
    cases(
        "qjoin_terms_mv_max",
        &s,
        &words,
        &|word| create_join_query("fkm", true, "pk", &from(word), &s, ScoreMode::Max).unwrap(),
        w,
        m,
    );
    for (name, mode) in [("none", ScoreMode::None), ("max", ScoreMode::Max)] {
        cases(
            &format!("qjoin_numeric_{name}"),
            &s,
            &words,
            &|word| {
                create_numeric_join_query(
                    "nfk",
                    false,
                    "npk",
                    NumericType::Long,
                    &from(word),
                    &s,
                    mode,
                )
                .unwrap()
            },
            w,
            m,
        );
    }
    let map = ordinal_map(&s, "gj").unwrap();
    let to = term("type", "to");
    for (name, mode, min, max) in [
        ("qjoin_gord_none", ScoreMode::None, 0, i32::MAX),
        ("qjoin_gord_max", ScoreMode::Max, 0, i32::MAX),
        ("qjoin_gord_avg_minmax", ScoreMode::Avg, 2, 10),
    ] {
        cases(
            name,
            &s,
            &words,
            &|word| {
                create_global_ordinals_join_query(
                    "gj",
                    &from(word),
                    &to,
                    &s,
                    mode,
                    Some(Arc::clone(&map)),
                    min,
                    max,
                )
                .unwrap()
            },
            w,
            m,
        );
    }
}
