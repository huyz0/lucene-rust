//! Shared fixtures for the facets / grouping / join benches: one synthetic
//! multi-segment index built in memory through `IndexWriter`.
#![allow(dead_code, clippy::arithmetic_side_effects)]

use lucene_index::document::{
    Document, LongPoint, NumericDocValuesField, SortedDocValuesField, SortedNumericDocValuesField,
    SortedSetDocValuesField, Store, StringField,
};
use lucene_index::index_writer::IndexWriter;
use lucene_index::merge_policy::upgrade::LATEST;
use lucene_search::facets::{FacetEntry, FacetsConfig, SortedSetDocValuesFacetField};
use lucene_store::ByteBuffersDirectory;

/// A tiny deterministic PRNG (xorshift64*), so every run indexes the same
/// documents.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed.max(1))
    }
    pub fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
    pub fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

/// The facets configuration of [`build_index`]'s `$facets` field.
pub fn facets_config() -> FacetsConfig {
    let mut c = FacetsConfig::new();
    c.set_multi_valued("Tag", true);
    c
}

/// `docs` documents in `segments` segments (a flush every `docs / segments`
/// documents). Fields:
/// - `body` (StringField): `all` on every document, `t30` on 30% of them;
/// - `tags` (SORTED_SET, 20% of documents, 1-3 of 1,000 terms);
/// - `num` (NUMERIC, 20% of documents, 0..1,000,000) and `dense` (NUMERIC,
///   every document, 0..1,000,000);
/// - `nums` (SORTED_NUMERIC, 50% of documents, 1-3 values in 0..1,000,000);
/// - `$facets`: `Author` (one of 100) and `Tag` (multi-valued, 2 of 2,000);
/// - `group` (SORTED, one of `groups` 24-byte terms);
/// - `join_to` (an indexed term), `join_key` (SORTED), `join_keys`
///   (SORTED_SET) and `join_num` (NUMERIC): one of `docs / 10` keys.
pub fn build_index(docs: usize, segments: usize, groups: usize) -> ByteBuffersDirectory {
    let dir = ByteBuffersDirectory::new();
    let config = facets_config();
    let per_segment = docs.div_ceil(segments.max(1));
    let keys = (docs / 10).max(1) as u64;
    let mut rng = Rng::new(0x5eed);
    {
        let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", LATEST).unwrap();
        w.set_ram_buffer_size_mb(4096.0).unwrap();
        for i in 0..docs {
            let mut d = Document::new();
            d.add(StringField::new("body", "all", Store::No));
            if rng.below(10) < 3 {
                d.add(StringField::new("body", "t30", Store::No));
            }
            if rng.below(5) == 0 {
                for _ in 0..1 + rng.below(3) {
                    d.add(SortedSetDocValuesField::new(
                        "tags",
                        format!("tag{:04}", rng.below(1000)).into_bytes(),
                    ));
                }
                d.add(NumericDocValuesField::new(
                    "num",
                    rng.below(1_000_000) as i64,
                ));
            }
            d.add(NumericDocValuesField::new(
                "dense",
                rng.below(1_000_000) as i64,
            ));
            if rng.below(2) == 0 {
                for _ in 0..1 + rng.below(3) {
                    d.add(SortedNumericDocValuesField::new(
                        "nums",
                        rng.below(1_000_000) as i64,
                    ));
                }
            }
            let g = rng.below(groups.max(1) as u64);
            d.add(SortedDocValuesField::new(
                "group",
                format!("group-value-{g:012}").into_bytes(),
            ));
            let key = rng.below(keys);
            let key_term = format!("k{key:09}");
            d.add(StringField::new("join_to", key_term.clone(), Store::No));
            d.add(SortedDocValuesField::new(
                "join_key",
                key_term.clone().into_bytes(),
            ));
            d.add(SortedSetDocValuesField::new(
                "join_keys",
                key_term.into_bytes(),
            ));
            d.add(NumericDocValuesField::new("join_num", key as i64));
            d.add(LongPoint::new("join_pt", &[key as i64]).unwrap());
            let facets: Vec<FacetEntry> = vec![
                SortedSetDocValuesFacetField::new("Author", &[&format!("a{:03}", rng.below(100))])
                    .unwrap()
                    .into(),
                SortedSetDocValuesFacetField::new("Tag", &[&format!("t{:04}", rng.below(1000))])
                    .unwrap()
                    .into(),
                SortedSetDocValuesFacetField::new("Tag", &[&format!("u{:04}", rng.below(1000))])
                    .unwrap()
                    .into(),
            ];
            let built = config.build(None, &facets, d).unwrap();
            w.add_fields_document(&built).unwrap();
            if (i + 1) % per_segment == 0 {
                w.flush().unwrap();
            }
        }
        w.commit().unwrap();
    }
    dir
}
