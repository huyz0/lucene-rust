#![allow(clippy::arithmetic_side_effects)]
//! **A view that masks keyword fields shows masked values -- and only those.**
//!
//! The security plugin's field masking keeps a keyword field's ordinals and sort order and
//! rewrites what `lookupOrd` hands out. `SegmentReader::with_security_view` does the same: the
//! `terms` aggregation over a masked view lists, for the same buckets in the same order with
//! the same counts, each term's masked bytes, and no raw term of the field is among the values
//! it emits. Everything that is not masked is untouched.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use lucene_aggs::terms_agg::terms;
use lucene_codecs::value_mask::{SharedMask, ValueMask};
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::query::MatchAllDocsQuery;
use lucene_search::{BooleanQuery, Clause, TermQuery};
use lucene_store::FsDirectory;

/// `m:` and the bytes reversed: never equal to the raw value, and recognisably a function of it.
#[derive(Debug)]
struct Reverse;

impl ValueMask for Reverse {
    fn apply(&self, raw: &[u8]) -> lucene_store::Result<Vec<u8>> {
        let mut out = b"m:".to_vec();
        out.extend(raw.iter().rev());
        Ok(out)
    }
}

fn mask_of(raw: &[u8]) -> Vec<u8> {
    Reverse.apply(raw).unwrap()
}

fn all() -> BooleanQuery {
    let mut b = BooleanQuery::new();
    b.must.push(Clause::MatchAllDocs(MatchAllDocsQuery::new(0)));
    b
}

fn reader() -> DirectoryReader {
    let dir = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/data/terms_aggs_index"
    );
    DirectoryReader::open(&FsDirectory::open(dir)).expect("open")
}

fn masked_view(reader: &DirectoryReader, fields: &[&str]) -> DirectoryReader {
    let mask: SharedMask = Arc::new(Reverse);
    let masks: HashMap<String, SharedMask> = fields
        .iter()
        .map(|f| ((*f).to_string(), Arc::clone(&mask)))
        .collect();
    let hidden = vec![HashSet::new(); reader.segment_readers().len()];
    reader.with_security_view(&hidden, &masks).expect("view")
}

fn terms_of(
    reader: &DirectoryReader,
    field: &str,
    shard_size: usize,
) -> lucene_aggs::terms_agg::TermsResult {
    let mut opened = reader.open_segments().expect("open postings");
    opened.open_points().expect("open points");
    let segments = opened.as_open_segments();
    terms(
        &segments,
        reader.segment_readers(),
        &all(),
        field,
        shard_size,
    )
    .unwrap_or_else(|e| panic!("{field}: {e}"))
}

/// Single-valued sorted-set, multi-valued sorted-set, sorted and raw-bytes keyword fields.
const FIELDS: [&str; 4] = ["kw", "mkw", "sk", "bk"];

#[test]
fn terms_aggregations_list_the_masked_values_of_the_same_buckets() {
    let base = reader();
    let view = masked_view(&base, &FIELDS);
    let mut checked = 0;
    for field in FIELDS {
        for shard_size in [1usize, 3, 25, 1000] {
            let raw = terms_of(&base, field, shard_size);
            let shown = terms_of(&view, field, shard_size);
            assert_eq!(
                shown.other_doc_count, raw.other_doc_count,
                "{field} {shard_size}"
            );
            let want: Vec<(Vec<u8>, u64)> =
                raw.buckets.iter().map(|(t, n)| (mask_of(t), *n)).collect();
            assert_eq!(shown.buckets, want, "{field} {shard_size}");
            checked += raw.buckets.len();
        }
    }
    assert!(checked > 50, "{checked}");
}

#[test]
fn no_raw_value_of_a_masked_field_is_emitted() {
    let base = reader();
    let view = masked_view(&base, &FIELDS);
    for field in FIELDS {
        let raw: HashSet<Vec<u8>> = terms_of(&base, field, 1000)
            .buckets
            .into_iter()
            .map(|(t, _)| t)
            .collect();
        assert!(raw.len() > 3, "{field}: the fixture has values");
        for shard_size in [1usize, 3, 25, 1000] {
            for (term, _) in terms_of(&view, field, shard_size).buckets {
                assert!(
                    !raw.contains(&term),
                    "{field}: the raw value {term:?} came out of a masked view"
                );
                assert!(term.starts_with(b"m:"), "{field}: {term:?}");
            }
        }
    }
}

#[test]
fn a_field_that_is_not_masked_is_shown_as_it_is() {
    let base = reader();
    let view = masked_view(&base, &["kw"]);
    for field in ["mkw", "sk", "bk", "hk"] {
        assert_eq!(
            terms_of(&view, field, 1000),
            terms_of(&base, field, 1000),
            "{field}"
        );
    }
}

#[test]
fn the_postings_of_a_masked_field_are_not_there() {
    // `pk` is indexed as well as kept in doc values.
    let base = reader();
    let view = masked_view(&base, &["pk"]);
    let raw_terms = terms_of(&base, "pk", 1000).buckets;
    let (some_raw, _) = raw_terms.first().unwrap();
    let by_term = |reader: &DirectoryReader| {
        let mut opened = reader.open_segments().expect("open postings");
        opened.open_points().expect("open points");
        let segments = opened.as_open_segments();
        let mut b = BooleanQuery::new();
        b.must
            .push(Clause::Term(TermQuery::new("pk", some_raw.clone())));
        terms(&segments, reader.segment_readers(), &b, "kw", 1000).unwrap()
    };
    assert!(
        !by_term(&base).buckets.is_empty(),
        "the unmasked field is searchable by its term"
    );
    assert!(
        by_term(&view).buckets.is_empty(),
        "the masked field has no postings: its term matches nothing"
    );
}
