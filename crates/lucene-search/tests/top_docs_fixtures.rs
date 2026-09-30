#![allow(clippy::arithmetic_side_effects)]
//! **`TopDocs.merge` and `TopDocs.rrf` against real Lucene.**
//!
//! `fixtures/src/GenTopDocs.java` draws synthetic shard results from a seeded
//! random source -- scores from a small set so ties within and across shards
//! are common, shard indices set or unset, empty shards, `start` offsets past
//! the hits -- and records what Lucene 10.5.0 returns for each: `merge` by
//! score under three tie-breakers, `merge` by eight sorts (numeric, score,
//! document and keyword keys, reversed, missing first and last), and `rrf`
//! for four `k`s, plus the argument errors. Every returned hit is compared by
//! document, score bits, shard index and sort values.

use std::collections::HashMap;

use lucene_search::collector::{TotalHits, TotalHitsRelation};
use lucene_search::top_docs::{
    merge, merge_field_docs, rrf, ShardFieldDoc, ShardScoreDoc, ShardTopFieldDocs, TieBreaker,
    TopDocs,
};
use lucene_search::top_field::{FieldDoc, SortField, SortType};

fn cases() -> HashMap<String, String> {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/data/top_docs/cases.txt"
    );
    std::fs::read_to_string(path)
        .expect("run scripts/gen-fixtures.sh --only GenTopDocs")
        .lines()
        .filter_map(|l| l.split_once('='))
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}

fn total(s: &str) -> TotalHits {
    let (v, rel) = s.split_once(':').unwrap();
    TotalHits {
        value: v.parse().unwrap(),
        relation: if rel == "eq" {
            TotalHitsRelation::EqualTo
        } else {
            TotalHitsRelation::GreaterThanOrEqualTo
        },
    }
}

fn score_docs(s: &str) -> TopDocs {
    let mut parts = s.split(';');
    let total_hits = total(parts.next().unwrap());
    let score_docs = parts
        .map(|h| {
            let f: Vec<&str> = h.split(':').collect();
            ShardScoreDoc {
                doc: f[0].parse().unwrap(),
                score: f32::from_bits(f[1].parse::<i32>().unwrap() as u32),
                shard_index: f[2].parse().unwrap(),
            }
        })
        .collect();
    TopDocs {
        total_hits,
        score_docs,
    }
}

fn sort_of(spec: &str) -> Vec<(String, SortField)> {
    spec.split(',')
        .enumerate()
        .map(|(i, k)| {
            let (ty, rev) = k.split_once(':').unwrap();
            let reverse = rev == "true";
            let field = format!("f{i}");
            let sf = match ty {
                "long" => SortField::numeric(&field, SortType::Long, reverse),
                "int" => SortField::numeric(&field, SortType::Int, reverse),
                "double" => SortField::numeric(&field, SortType::Double, reverse),
                "float" => SortField::numeric(&field, SortType::Float, reverse),
                "score" => {
                    let mut s = SortField::score();
                    s.reverse = reverse;
                    s
                }
                "doc" => {
                    let mut s = SortField::doc();
                    s.reverse = reverse;
                    s
                }
                "string_first" => SortField::string(&field, reverse),
                "string_last" => {
                    let mut s = SortField::string(&field, reverse);
                    s.missing = 1;
                    s
                }
                other => panic!("sort type {other}"),
            };
            (ty.to_string(), sf)
        })
        .collect()
}

/// `NumericUtils.doubleToSortableLong`.
fn double_sortable(bits: i64) -> i64 {
    bits ^ ((bits >> 63) & 0x7fff_ffff_ffff_ffff)
}

fn field_docs(s: &str, types: &[String]) -> ShardTopFieldDocs {
    let mut parts = s.split(';');
    let total_hits = total(parts.next().unwrap());
    let hits = parts
        .map(|h| {
            let f: Vec<&str> = h.split(':').collect();
            let mut values = Vec::new();
            let mut terms = Vec::new();
            let mut any_term = false;
            for (k, ty) in types.iter().enumerate() {
                let raw = f[3 + k];
                let (v, t) = match ty.as_str() {
                    "long" | "int" | "doc" => (raw.parse::<i64>().unwrap(), None),
                    "double" => (double_sortable(raw.parse::<i64>().unwrap()), None),
                    "float" => (
                        i64::from(lucene_util::numeric_utils::sortable_float_bits(
                            raw.parse::<i32>().unwrap(),
                        )),
                        None,
                    ),
                    "score" => (i64::from(raw.parse::<i32>().unwrap() as u32), None),
                    _ => {
                        any_term = true;
                        if raw == "-" {
                            (0, None)
                        } else {
                            let hex = &raw[1..];
                            let bytes = (0..hex.len())
                                .step_by(2)
                                .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap())
                                .collect::<Vec<u8>>();
                            (0, Some(bytes))
                        }
                    }
                };
                values.push(v);
                terms.push(t);
            }
            if !any_term {
                terms.clear();
            }
            ShardFieldDoc {
                fields: FieldDoc {
                    doc: f[0].parse().unwrap(),
                    values,
                    terms,
                },
                score: f32::from_bits(f[1].parse::<i32>().unwrap() as u32),
                shard_index: f[2].parse().unwrap(),
            }
        })
        .collect();
    ShardTopFieldDocs { total_hits, hits }
}

fn tie(name: &str) -> TieBreaker<'static> {
    match name {
        "doc" => TieBreaker::DocId,
        "shard" => TieBreaker::ShardIndex,
        _ => TieBreaker::Default,
    }
}

fn assert_score_docs(case: &str, got: &TopDocs, want: &TopDocs) {
    assert_eq!(got.total_hits, want.total_hits, "{case}: total");
    let g: Vec<(i32, u32, i32)> = got
        .score_docs
        .iter()
        .map(|d| (d.doc, d.score.to_bits(), d.shard_index))
        .collect();
    let w: Vec<(i32, u32, i32)> = want
        .score_docs
        .iter()
        .map(|d| (d.doc, d.score.to_bits(), d.shard_index))
        .collect();
    assert_eq!(g, w, "{case}: hits");
}

#[test]
fn top_docs_merge_and_rrf_match_real_lucene() {
    let c = cases();
    let n: usize = c["case_count"].parse().unwrap();
    assert!(n > 150, "the fixture records every case");
    let (mut score_cases, mut field_cases, mut rrf_cases, mut errors) = (0, 0, 0, 0);
    for i in 0..n {
        let k = |f: &str| c[&format!("case.{i}.{f}")].clone();
        let case = format!("case.{i}");
        let shards: usize = k("shards").parse().unwrap();
        let out = k("out");
        match k("kind").as_str() {
            "score" => {
                score_cases += 1;
                let input: Vec<TopDocs> = (0..shards)
                    .map(|s| score_docs(&k(&format!("shard.{s}"))))
                    .collect();
                let got = merge(
                    k("start").parse().unwrap(),
                    k("top_n").parse().unwrap(),
                    &input,
                    &tie(&k("tie")),
                );
                if out == "error" {
                    errors += 1;
                    assert!(got.is_err(), "{case}: Lucene threw");
                } else {
                    assert_score_docs(&case, &got.unwrap(), &score_docs(&out));
                }
            }
            "field" => {
                field_cases += 1;
                let spec = sort_of(&k("sort"));
                let types: Vec<String> = spec.iter().map(|(t, _)| t.clone()).collect();
                let sort: Vec<SortField> = spec.into_iter().map(|(_, s)| s).collect();
                let input: Vec<ShardTopFieldDocs> = (0..shards)
                    .map(|s| field_docs(&k(&format!("shard.{s}")), &types))
                    .collect();
                let got = merge_field_docs(
                    &sort,
                    k("start").parse().unwrap(),
                    k("top_n").parse().unwrap(),
                    &input,
                    &tie(&k("tie")),
                );
                if out == "error" {
                    errors += 1;
                    assert!(got.is_err(), "{case}: Lucene threw");
                } else {
                    let got = got.unwrap();
                    let want = field_docs(&out, &types);
                    assert_eq!(got.total_hits, want.total_hits, "{case}: total");
                    assert_eq!(got.hits, want.hits, "{case}: hits");
                }
            }
            "rrf" => {
                rrf_cases += 1;
                let input: Vec<TopDocs> = (0..shards)
                    .map(|s| score_docs(&k(&format!("shard.{s}"))))
                    .collect();
                let got = rrf(k("top_n").parse().unwrap(), k("k").parse().unwrap(), &input);
                if out == "error" {
                    errors += 1;
                    assert!(got.is_err(), "{case}: Lucene threw");
                } else {
                    assert_score_docs(&case, &got.unwrap(), &score_docs(&out));
                }
            }
            other => panic!("kind {other}"),
        }
    }
    assert!(score_cases > 50 && field_cases > 50 && rrf_cases > 30);
    assert!(errors >= 3, "the argument errors are recorded");
}
