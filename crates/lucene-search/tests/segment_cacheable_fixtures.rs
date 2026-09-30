//! **`SegmentCacheable.isCacheable` against real Lucene.**
//!
//! `fixtures/src/GenSegmentCacheable.java` records, over
//! `doc_values_updates_index` (whose `val` and `tag` fields carry doc-values
//! updates and whose `keep` field does not), whether each of eleven weights
//! -- field-exists over updated and plain fields, a term, booleans and
//! dis-maxes (with an uncacheable clause, and of 17 clauses), constant score
//! and boost -- and three values sources per field are cacheable on the
//! segment.

use lucene_search::directory_reader::DirectoryReader;
use lucene_search::query::{
    BooleanQuery, BoostQuery, Clause, ConstantScoreQuery, DisjunctionMaxQuery, FieldExistsQuery,
    TermQuery,
};
use lucene_search::segment_cacheable::is_cacheable;
use lucene_search::values_source::{self as vs, ValuesContext};
use lucene_store::FsDirectory;

fn fixture(name: &str) -> String {
    format!("{}/../../fixtures/data/{name}", env!("CARGO_MANIFEST_DIR"))
}

fn exists(f: &str) -> Clause {
    Clause::Exists(FieldExistsQuery::new(f))
}

fn id(v: &str) -> Clause {
    Clause::Term(TermQuery::new("id", v.as_bytes().to_vec()))
}

#[test]
fn cacheability_matches_real_lucene() {
    let cases = std::fs::read_to_string(fixture("segment_cacheable/cases.txt"))
        .expect("run scripts/gen-fixtures.sh --only GenSegmentCacheable");
    let want = |key: &str| -> String {
        cases
            .lines()
            .find_map(|l| l.strip_prefix(&format!("{key}=")))
            .unwrap_or_else(|| panic!("{key} missing"))
            .to_string()
    };
    let reader = DirectoryReader::open(&FsDirectory::open(fixture("doc_values_updates_index")))
        .expect("open");
    let segs = reader.segment_readers();
    let got = |c: &Clause| -> String {
        segs.iter()
            .map(|r| if is_cacheable(c, Some(r)) { '1' } else { '0' })
            .collect()
    };

    let filter = |f: &str| {
        let mut b = BooleanQuery::new();
        b.must.push(id("7"));
        b.filter.push(exists(f));
        Clause::Boolean(Box::new(b))
    };
    let mut big = BooleanQuery::new();
    big.should = (0..17).map(|i| id(&i.to_string())).collect();
    let queries: Vec<(&str, Clause)> = vec![
        ("exists_val", exists("val")),
        ("exists_tag", exists("tag")),
        ("exists_keep", exists("keep")),
        ("term", id("7")),
        ("bool_filter_val", filter("val")),
        ("bool_filter_keep", filter("keep")),
        ("bool_17", Clause::Boolean(Box::new(big))),
        (
            "dismax_17",
            Clause::DisjunctionMax(Box::new(DisjunctionMaxQuery::new(
                (0..17).map(|i| id(&i.to_string())),
                0.0,
            ))),
        ),
        (
            "dismax_tag",
            Clause::DisjunctionMax(Box::new(DisjunctionMaxQuery::new(
                [id("1"), exists("tag")],
                0.0,
            ))),
        ),
        (
            "const_val",
            Clause::ConstantScore(Box::new(ConstantScoreQuery::new(exists("val"), 1.0))),
        ),
        (
            "boost_keep",
            Clause::Boost(Box::new(BoostQuery::new(exists("keep"), 2.0))),
        ),
    ];
    for (name, q) in &queries {
        assert_eq!(got(q), want(&format!("q.{name}")), "{name}");
    }

    for f in ["val", "tag", "keep", "missing"] {
        let s: String = segs
            .iter()
            .flat_map(|r| {
                let ctx = ValuesContext::for_reader(r);
                [
                    vs::from_long_field(f).is_cacheable(&ctx, 0),
                    vs::long_from_long_field(f).is_cacheable(&ctx, 0),
                    vs::to_long_values_source(vs::from_long_field(f)).is_cacheable(&ctx, 0),
                ]
            })
            .map(|b| if b { '1' } else { '0' })
            .collect();
        assert_eq!(s, want(&format!("vs.{f}")), "values source over {f}");
    }
}
