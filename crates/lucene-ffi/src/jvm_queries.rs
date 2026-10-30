//! The query-tree nodes that take the last Lucene queries OpenSearch builds
//! off the Java fallback (the plugin's Track A): multi-term queries over any
//! automaton, the doc-values-only queries, `CoveringQuery` and the two
//! distance-feature queries.
//!
//! | kind | layout after the kind byte | Lucene query (OpenSearch DSL) |
//! |---|---|---|
//! | `27` multi-term | `field`, `rewrite: u8` (`0` `CONSTANT_SCORE_BLENDED_REWRITE`, `1` `CONSTANT_SCORE_REWRITE`, `2` `DOC_VALUES_REWRITE`), the `CompiledAutomaton` its terms enumeration runs ([`decode_compiled_automaton`]) | any `AutomatonQuery` (`wildcard`, `regexp` with flags or `case_insensitive`, `prefix` on `index_prefixes`), `FuzzyQuery`, under a constant-score or doc-values rewrite |
//! | `28` doc-values term set | `field`, `count: i32`, the terms | `TermInSetQuery` under `DOC_VALUES_REWRITE` (`terms` on a doc-values-only `keyword`) |
//! | `29` doc-values numeric range | `field`, `lower: i64`, `upper: i64` (inclusive) | `SortedNumericDocValuesRangeQuery` (`range`/`term` on a doc-values-only number or date) |
//! | `30` doc-values numeric set | `field`, `count: i32`, the `i64`s | `SortedNumericDocValuesSetQuery` (`terms` on a doc-values-only number) |
//! | `31` doc-values ordinal range | `field`, `flags: u8` (as `NODE_TERM_RANGE`'s), the bounds it says are present | `SortedSetDocValuesRangeQuery` (`range`/`term` on a doc-values-only `keyword` or `ip`) |
//! | `32` doc-values multi-range | `field`, `count: i32`, each `lower`, `upper` (inclusive) | `SortedSetDocValuesMultiRangeQuery` (`terms` on a doc-values-only `ip`) |
//! | `33` covering | `count: i32`, the sub-queries (nodes), the minimum: `0` + `i64` constant, or `1` + `field` (the field's first sorted-numeric value) | `CoveringQuery` (`terms_set`) |
//! | `34` long distance feature | `field`, `origin: i64`, `pivot: i64` | `LongDistanceFeatureQuery` (`distance_feature` on a `date`) |
//! | `35` lat/lon distance feature | `field`, `lat: f64`, `lon: f64`, `pivot: f64` (metres) | `LatLonPointDistanceFeatureQuery` (`distance_feature` on a `geo_point`) |
//! | `36` synonym | `field`, `count: i32`, each `term: bytes`, `boost: f32` bits as `i32` | `SynonymQuery` (a `match` whose analyzer stacks tokens: `hunspell`, `synonym_graph`, `keyword_repeat`) |
//!
//! The doc-values and distance-feature nodes become
//! [`DocumentClause`]s keyed by their own wire bytes, as the geo nodes do.
//! Every node is built through the constructor that checks what Java's
//! checks; a count is bounded by the node limit or by the blob before it
//! sizes anything.

use std::sync::Arc;

use lucene_search::document::geo::LatLonPointDistanceFeatureQuery;
use lucene_search::document::{
    DocValuesTermInSetQuery, DocumentQuery, LongDistanceFeatureQuery,
    SortedNumericDocValuesRangeQuery, SortedNumericDocValuesSetQuery, SortedSetDocValuesRangeQuery,
};
use lucene_search::extended_query::{
    AutomatonQuery, DocumentClause, ExtendedQuery, MultiTermQuery, MultiTermSource, RewriteMethod,
    SynonymQuery,
};
use lucene_search::query::Clause;
use lucene_search::reader::doc_values::get_sorted_numeric;
use lucene_search::reader::SortedNumericDocValues;
use lucene_search::sandbox::search::{ByteRange, CoveringQuery, SortedSetDocValuesMultiRangeQuery};
use lucene_search::values_source::{
    long_constant, BoxDoubleValues, BoxLongValues, LongValues, LongValuesSource, ValuesContext,
};

use crate::error::FfiStatus;
use crate::jvm_nodes::{decode_compiled_automaton, invalid};
use crate::jvm_reader::{hex, Cursor};
use crate::query::check_clause_count;

pub(crate) const NODE_MULTI_TERM: u8 = 27;
pub(crate) const NODE_DV_TERM_SET: u8 = 28;
pub(crate) const NODE_DV_NUMERIC_RANGE: u8 = 29;
pub(crate) const NODE_DV_NUMERIC_SET: u8 = 30;
pub(crate) const NODE_DV_ORD_RANGE: u8 = 31;
pub(crate) const NODE_DV_MULTI_RANGE: u8 = 32;
pub(crate) const NODE_COVERING: u8 = 33;
pub(crate) const NODE_LONG_DISTANCE_FEATURE: u8 = 34;
pub(crate) const NODE_GEO_DISTANCE_FEATURE: u8 = 35;
/// `SynonymQuery`: `field`, `n: i32`, then `n` of `term: bytes`, `boost: f32`.
pub(crate) const NODE_SYNONYM: u8 = 36;
/// The first and last kinds this module decodes.
pub(crate) const NODE_FIRST: u8 = NODE_MULTI_TERM;
pub(crate) const NODE_LAST: u8 = NODE_SYNONYM;

/// The multi-term node's rewrite methods.
const REWRITE_CONSTANT_SCORE_BLENDED: u8 = 0;
const REWRITE_CONSTANT_SCORE: u8 = 1;
const REWRITE_DOC_VALUES: u8 = 2;

/// The covering node's minimum sources.
const MINIMUM_CONSTANT: u8 = 0;
const MINIMUM_FIELD: u8 = 1;

/// `NODE_TERM_RANGE`'s flags, which the ordinal range shares.
const RANGE_HAS_LOWER: u8 = 1;
const RANGE_HAS_UPPER: u8 = 2;
const RANGE_INCLUDE_LOWER: u8 = 4;
const RANGE_INCLUDE_UPPER: u8 = 8;

fn utf8(b: &[u8]) -> Result<&str, FfiStatus> {
    std::str::from_utf8(b).map_err(|_| FfiStatus::InvalidUtf8)
}

/// A node of kinds 27-35, its kind byte read; `start` is the node's offset.
pub(crate) fn decode(
    c: &mut Cursor<'_>,
    kind: u8,
    start: usize,
    depth: usize,
    nodes: &mut usize,
) -> Result<Clause, FfiStatus> {
    let field =
        |c: &mut Cursor<'_>| -> Result<String, FfiStatus> { Ok(utf8(c.bytes()?)?.to_string()) };
    let built = |what: &str, e: lucene_search::Error| invalid(format!("query tree: {what}: {e}"));
    let document: Arc<dyn DocumentQuery> = match kind {
        NODE_MULTI_TERM => {
            let field = field(c)?;
            let rewrite = match c.u8()? {
                REWRITE_CONSTANT_SCORE_BLENDED => RewriteMethod::ConstantScoreBlended,
                REWRITE_CONSTANT_SCORE => RewriteMethod::ConstantScore,
                REWRITE_DOC_VALUES => RewriteMethod::DocValues,
                other => {
                    return Err(invalid(format!(
                        "query tree: unknown multi-term rewrite {other} (expected 0..=2)"
                    )))
                }
            };
            // `CompiledAutomaton`'s byte automaton: what `getTermsEnum`
            // intersects the terms with, whatever the query class built it.
            let automaton = decode_compiled_automaton(c)?;
            let source = MultiTermSource::Automaton(AutomatonQuery::new(field, automaton, true));
            return Ok(Clause::Extended(Box::new(ExtendedQuery::MultiTerm(
                MultiTermQuery::new(source, rewrite),
            ))));
        }
        NODE_DV_TERM_SET => {
            let field = field(c)?;
            let count = c.len()?;
            *nodes = nodes.saturating_add(count);
            check_clause_count(*nodes)?;
            let mut terms = Vec::new();
            for _ in 0..count {
                terms.push(c.bytes()?.to_vec());
            }
            Arc::new(DocValuesTermInSetQuery::new(field, terms))
        }
        NODE_DV_NUMERIC_RANGE => {
            let field = field(c)?;
            let (lower, upper) = (c.i64()?, c.i64()?);
            Arc::new(SortedNumericDocValuesRangeQuery::new(field, lower, upper))
        }
        NODE_DV_NUMERIC_SET => {
            // One leaf to Lucene's clause visitor, however many numbers; each
            // is eight bytes of the blob.
            let field = field(c)?;
            let count = c.len()?;
            let mut numbers = Vec::new();
            for _ in 0..count {
                numbers.push(c.i64()?);
            }
            Arc::new(SortedNumericDocValuesSetQuery::new(field, numbers))
        }
        NODE_DV_ORD_RANGE => {
            let field = field(c)?;
            let flags = c.u8()?;
            if flags
                & !(RANGE_HAS_LOWER | RANGE_HAS_UPPER | RANGE_INCLUDE_LOWER | RANGE_INCLUDE_UPPER)
                != 0
            {
                return Err(invalid(format!(
                    "query tree: doc-values range flags {flags:#x}"
                )));
            }
            let lower = if flags & RANGE_HAS_LOWER != 0 {
                Some(c.bytes()?.to_vec())
            } else {
                None
            };
            let upper = if flags & RANGE_HAS_UPPER != 0 {
                Some(c.bytes()?.to_vec())
            } else {
                None
            };
            Arc::new(SortedSetDocValuesRangeQuery::new(
                field,
                lower,
                upper,
                flags & RANGE_INCLUDE_LOWER != 0,
                flags & RANGE_INCLUDE_UPPER != 0,
            ))
        }
        NODE_DV_MULTI_RANGE => {
            let field = field(c)?;
            let count = c.len()?;
            if count == 0 {
                return Err(invalid(
                    "query tree: a doc-values multi-range with no ranges",
                ));
            }
            let mut ranges = Vec::new();
            for _ in 0..count {
                let lower = c.bytes()?.to_vec();
                let upper = c.bytes()?.to_vec();
                ranges.push(ByteRange::new(lower, upper));
            }
            Arc::new(SortedSetDocValuesMultiRangeQuery::new(field, ranges))
        }
        NODE_COVERING => {
            let count = c.len()?;
            check_clause_count(nodes.saturating_add(count))?;
            let mut queries = Vec::new();
            for _ in 0..count {
                queries.push(crate::jvm_reader::decode_node(c, depth + 1, nodes)?);
            }
            let minimum: Arc<dyn LongValuesSource> = match c.u8()? {
                MINIMUM_CONSTANT => long_constant(c.i64()?),
                MINIMUM_FIELD => Arc::new(FirstValueSource(field(c)?)),
                other => {
                    return Err(invalid(format!(
                        "query tree: unknown covering minimum {other} (expected 0..=1)"
                    )))
                }
            };
            let q = CoveringQuery::new(queries, minimum).map_err(|e| built("covering query", e))?;
            return Ok(Clause::Extended(Box::new(ExtendedQuery::Covering(q))));
        }
        NODE_LONG_DISTANCE_FEATURE => {
            let field = field(c)?;
            let (origin, pivot) = (c.i64()?, c.i64()?);
            Arc::new(
                LongDistanceFeatureQuery::new(field, origin, pivot)
                    .map_err(|e| built("distance feature", e))?,
            )
        }
        NODE_SYNONYM => {
            let field = field(c)?;
            let count = c.len()?;
            *nodes = nodes.saturating_add(count);
            check_clause_count(*nodes)?;
            let mut terms = Vec::with_capacity(count); // alloc-ok: count is within the clause limit
            for _ in 0..count {
                let term = c.bytes()?.to_vec();
                terms.push((term, f32::from_bits(c.i32()? as u32)));
            }
            let q = SynonymQuery::new(field, terms).map_err(|e| built("synonym query", e))?;
            return Ok(Clause::Extended(Box::new(ExtendedQuery::Synonym(q))));
        }
        NODE_GEO_DISTANCE_FEATURE => {
            let field = field(c)?;
            let (lat, lon, pivot) = (c.f64()?, c.f64()?, c.f64()?);
            Arc::new(
                LatLonPointDistanceFeatureQuery::new(field, lat, lon, pivot)
                    .map_err(|e| built("distance feature", e))?,
            )
        }
        other => {
            return Err(invalid(format!(
                "query tree: unknown node kind {other} (expected {NODE_FIRST}..={NODE_LAST})"
            )))
        }
    };
    let key = hex(c.since(start));
    Ok(DocumentClause::new(key, document).into())
}

/// OpenSearch's `TermsSetQueryBuilder.FieldValuesSource` over plain
/// sorted-numeric field data: a document's first (smallest) value, none
/// for a document without one (which `CoveringQuery` never matches).
struct FirstValueSource(String);

struct FirstValues<'c> {
    values: Box<dyn SortedNumericDocValues + 'c>,
    current: i64,
}

impl LongValues for FirstValues<'_> {
    fn advance_exact(&mut self, doc: i32) -> lucene_search::Result<bool> {
        if !self.values.advance_exact(doc)? {
            return Ok(false);
        }
        self.current = self.values.next_value()?;
        Ok(true)
    }

    fn long_value(&mut self) -> lucene_search::Result<i64> {
        Ok(self.current)
    }
}

impl LongValuesSource for FirstValueSource {
    fn get_values<'c>(
        &self,
        ctx: &ValuesContext<'c>,
        leaf: usize,
        _scores: Option<BoxDoubleValues<'c>>,
    ) -> lucene_search::Result<BoxLongValues<'c>> {
        Ok(Box::new(FirstValues {
            values: get_sorted_numeric(ctx.leaf_reader(leaf)?, &self.0)?,
            current: -1,
        }))
    }

    fn needs_scores(&self) -> bool {
        false
    }

    /// `FieldValuesSource.isCacheable`: always.
    fn is_cacheable(&self, _ctx: &ValuesContext<'_>, _leaf: usize) -> bool {
        true
    }

    fn describe(&self) -> String {
        format!("long({})", self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jvm_nodes::tests::{fails, open, search, Blob, BODIES};
    use crate::jvm_reader::ffi_close_jvm_reader;
    use lucene_index::document::{
        Document, LatLonDocValuesField, LatLonPoint, LongField, SortedNumericDocValuesField,
        SortedSetDocValuesField, Store, StringField, TextField,
    };
    use lucene_index::index_writer::IndexWriter;
    use lucene_index::segment_info::LuceneVersion;
    use lucene_store::FsDirectory;
    use lucene_util::test_support::TempDir;

    /// Twelve documents in two segments (global doc `i` is document `i`):
    /// `body` (`BODIES[i % 6]`), and doc values only -- `kw` (`k{i % 4}`),
    /// `n` (`i`), `msm` (`i % 3`) -- plus `date` (points and doc values,
    /// `i * 1000`) and `loc` (latitude `i`, longitude 0).
    fn index(tag: &str) -> TempDir {
        let tmp = TempDir::new(tag);
        let dir = FsDirectory::open(tmp.path());
        let version = LuceneVersion {
            major: 10,
            minor: 5,
            bugfix: 0,
        };
        let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", version).unwrap();
        for i in 0..12i64 {
            let mut d = Document::new();
            d.add(StringField::new("id", i.to_string(), Store::Yes));
            d.add(TextField::new("body", BODIES[(i % 6) as usize], Store::No));
            d.add(SortedSetDocValuesField::new(
                "kw",
                format!("k{}", i % 4).into_bytes(),
            ));
            d.add(SortedNumericDocValuesField::new("n", i));
            d.add(SortedNumericDocValuesField::new("msm", i % 3));
            d.add(LongField::new("date", i * 1000, Store::No));
            d.add(LatLonPoint::new("loc", i as f64, 0.0).unwrap());
            d.add(LatLonDocValuesField::new("loc", i as f64, 0.0).unwrap());
            w.add_fields_document(&d).unwrap();
            if i == 5 {
                w.commit().unwrap();
            }
        }
        w.commit().unwrap();
        tmp
    }

    fn docs(h: u64, blob: &Blob) -> Vec<i32> {
        let mut d: Vec<i32> = search(h, blob).0.into_iter().map(|(d, _)| d).collect();
        d.sort_unstable();
        d
    }

    fn matching(f: impl Fn(i32) -> bool) -> Vec<i32> {
        (0..12).filter(|&i| f(i)).collect()
    }

    /// `k0` or `k2`: `k`, then `0` or `2`, as a normal byte automaton.
    fn k0_or_k2() -> Blob {
        Blob::default()
            .u8(3)
            .i32(3)
            // State 0: not accepting, `k` to 1.
            .u8(0)
            .i32(1)
            .i32(1)
            .i32(i32::from(b'k'))
            .i32(i32::from(b'k'))
            // State 1: `0` or `2` to 2.
            .u8(0)
            .i32(2)
            .i32(2)
            .i32(i32::from(b'0'))
            .i32(i32::from(b'0'))
            .i32(2)
            .i32(i32::from(b'2'))
            .i32(i32::from(b'2'))
            // State 2: accepting.
            .u8(1)
            .i32(0)
    }

    fn multi_term(field: &str, rewrite: u8, automaton: &Blob) -> Blob {
        Blob::tree()
            .u8(NODE_MULTI_TERM)
            .str(field)
            .u8(rewrite)
            .raw(automaton)
    }

    #[test]
    fn a_multi_term_node_matches_its_automaton_under_every_rewrite() {
        let tmp = index("jvm-queries-multi-term");
        let h = open(&tmp);
        let alpha = matching(|i| BODIES[(i % 6) as usize].contains("alpha"));
        for rewrite in [REWRITE_CONSTANT_SCORE_BLENDED, REWRITE_CONSTANT_SCORE] {
            let single = multi_term("body", rewrite, &Blob::default().u8(2).str("alpha"));
            let (hits, total) = search(h, &single);
            assert_eq!(total, alpha.len() as i64, "rewrite {rewrite}");
            assert!(hits.iter().all(|&(_, s)| s == 1.0), "constant scores");
            assert_eq!(docs(h, &single), alpha);
            // Every term, and none.
            assert_eq!(
                docs(h, &multi_term("body", rewrite, &Blob::default().u8(1))).len(),
                12
            );
            assert!(docs(h, &multi_term("body", rewrite, &Blob::default().u8(0))).is_empty());
        }
        // A doc-values-only keyword, through its ordinals.
        let even = matching(|i| i % 2 == 0);
        assert_eq!(
            docs(h, &multi_term("kw", REWRITE_DOC_VALUES, &k0_or_k2())),
            even
        );
        // Under a boost, scored as it.
        let boosted = Blob::tree()
            .u8(3)
            .f32(2.5)
            .u8(NODE_MULTI_TERM)
            .str("kw")
            .u8(REWRITE_DOC_VALUES)
            .raw(&k0_or_k2());
        assert!(search(h, &boosted).0.iter().all(|&(_, s)| s == 2.5));
        let (rc, e) = fails(h, &multi_term("kw", 9, &k0_or_k2()));
        assert_ne!(rc, 0);
        assert!(e.contains("multi-term rewrite 9"), "{e}");
        assert_eq!(ffi_close_jvm_reader(h), 0);
    }

    fn synonym(terms: &[(&str, f32)]) -> Blob {
        let mut b = Blob::tree()
            .u8(NODE_SYNONYM)
            .str("body")
            .i32(terms.len() as i32);
        for (t, boost) in terms {
            b = b.str(t).f32(*boost);
        }
        b
    }

    #[test]
    fn a_synonym_node_scores_its_terms_as_one() {
        let tmp = index("jvm-queries-synonym");
        let h = open(&tmp);
        let either = matching(|i| {
            let b = BODIES[(i % 6) as usize];
            b.contains("delta") || b.contains("epsilon")
        });
        let q = synonym(&[("epsilon", 1.0), ("delta", 0.5)]);
        assert_eq!(docs(h, &q), either);
        let (hits, _) = search(h, &q);
        // One pseudo-term: a document's score depends on its summed weighted
        // frequency and its length only, so the two `delta beta` documents tie.
        let score = |d: i32| hits.iter().find(|&&(x, _)| x == d).unwrap().1;
        assert_eq!(score(3), score(9));
        assert!(hits.iter().all(|&(_, s)| s > 0.0));
        // No term matches nothing; a boost outside (0, 1] is Java's builder error.
        assert!(docs(h, &synonym(&[])).is_empty());
        let (rc, e) = fails(h, &synonym(&[("delta", 2.0)]));
        assert_ne!(rc, 0);
        assert!(e.contains("synonym query"), "{e}");
        assert_eq!(ffi_close_jvm_reader(h), 0);
    }

    #[test]
    fn the_doc_values_nodes_match_what_the_doc_values_hold() {
        let tmp = index("jvm-queries-doc-values");
        let h = open(&tmp);
        let set = Blob::tree()
            .u8(NODE_DV_TERM_SET)
            .str("kw")
            .i32(3)
            .str("k3")
            .str("k1")
            .str("absent");
        assert_eq!(docs(h, &set), matching(|i| i % 2 == 1));
        let range = |lo: i64, hi: i64| {
            let b = Blob::tree().u8(NODE_DV_NUMERIC_RANGE).str("n");
            Blob([b.0, lo.to_le_bytes().to_vec(), hi.to_le_bytes().to_vec()].concat())
        };
        assert_eq!(docs(h, &range(3, 7)), matching(|i| (3..=7).contains(&i)));
        assert!(docs(h, &range(7, 3)).is_empty());
        let numbers = [1i64, 5, 100, i64::MIN];
        let mut numeric_set = Blob::tree()
            .u8(NODE_DV_NUMERIC_SET)
            .str("n")
            .i32(numbers.len() as i32);
        for n in numbers {
            numeric_set.0.extend_from_slice(&n.to_le_bytes());
        }
        assert_eq!(docs(h, &numeric_set), vec![1, 5]);
        // [k1, k2): k1; (-inf, k1]: k0 and k1.
        let ords = Blob::tree()
            .u8(NODE_DV_ORD_RANGE)
            .str("kw")
            .u8(RANGE_HAS_LOWER | RANGE_HAS_UPPER | RANGE_INCLUDE_LOWER)
            .str("k1")
            .str("k2");
        assert_eq!(docs(h, &ords), matching(|i| i % 4 == 1));
        let open_lower = Blob::tree()
            .u8(NODE_DV_ORD_RANGE)
            .str("kw")
            .u8(RANGE_HAS_UPPER | RANGE_INCLUDE_UPPER)
            .str("k1");
        assert_eq!(docs(h, &open_lower), matching(|i| i % 4 <= 1));
        let (_, e) = fails(h, &Blob::tree().u8(NODE_DV_ORD_RANGE).str("kw").u8(0x10));
        assert!(e.contains("doc-values range flags"), "{e}");
        let multi = Blob::tree()
            .u8(NODE_DV_MULTI_RANGE)
            .str("kw")
            .i32(2)
            .str("k3")
            .str("k3")
            .str("k0")
            .str("k0");
        assert_eq!(docs(h, &multi), matching(|i| i % 4 == 0 || i % 4 == 3));
        let (_, e) = fails(h, &Blob::tree().u8(NODE_DV_MULTI_RANGE).str("kw").i32(0));
        assert!(e.contains("no ranges"), "{e}");
        assert_eq!(ffi_close_jvm_reader(h), 0);
    }

    #[test]
    fn a_covering_node_needs_each_documents_minimum_of_its_queries() {
        let tmp = index("jvm-queries-covering");
        let h = open(&tmp);
        let words = ["alpha", "beta", "gamma"];
        let count = |i: i32| {
            words
                .iter()
                .filter(|w| BODIES[(i % 6) as usize].split(' ').any(|t| t == **w))
                .count() as i64
        };
        let covering = |minimum: Blob| {
            let mut b = Blob::tree().u8(NODE_COVERING).i32(3);
            for w in words {
                b = b.term("body", w);
            }
            b.raw(&minimum)
        };
        // `msm` is `i % 3`; below 1 counts as 1.
        let by_field = covering(Blob::default().u8(MINIMUM_FIELD).str("msm"));
        let (hits, total) = search(h, &by_field);
        let want = matching(|i| count(i) >= i64::from(i % 3).max(1));
        assert_eq!(total, want.len() as i64);
        assert_eq!(docs(h, &by_field), want);
        assert!(
            hits.iter().all(|&(_, s)| s > 0.0),
            "scored by the sum of the matches"
        );
        // A missing field: no document has a minimum, so none matches.
        assert!(docs(
            h,
            &covering(Blob::default().u8(MINIMUM_FIELD).str("absent"))
        )
        .is_empty());
        let mut constant = Blob::default().u8(MINIMUM_CONSTANT);
        constant.0.extend_from_slice(&2i64.to_le_bytes());
        assert_eq!(docs(h, &covering(constant)), matching(|i| count(i) >= 2));
        // A covering query scores a document by the sum of its matching
        // queries, as the same SHOULD disjunction does.
        let mut boolean = Blob::tree().u8(1).i32(1).i32(3);
        for w in words {
            boolean = boolean.u8(2).term("body", w);
        }
        let (or, _) = search(h, &boolean);
        for (doc, score) in hits {
            let want = or
                .iter()
                .find(|&&(d, _)| d == doc)
                .expect("in the disjunction")
                .1;
            assert_eq!(score, want, "doc {doc}");
        }
        let (_, e) = fails(h, &covering(Blob::default().u8(7)));
        assert!(e.contains("covering minimum 7"), "{e}");
        assert_eq!(ffi_close_jvm_reader(h), 0);
    }

    #[test]
    fn the_distance_feature_nodes_score_by_closeness_to_the_origin() {
        let tmp = index("jvm-queries-distance");
        let h = open(&tmp);
        let mut long = Blob::tree().u8(NODE_LONG_DISTANCE_FEATURE).str("date");
        long.0.extend_from_slice(&5000i64.to_le_bytes());
        long.0.extend_from_slice(&1000i64.to_le_bytes());
        let (hits, total) = search(h, &long);
        assert_eq!(total, 12);
        assert_eq!(hits[0], (5, 1.0), "{hits:?}");
        assert_eq!(hits[1].1, 0.5);
        let geo = Blob::tree()
            .u8(NODE_GEO_DISTANCE_FEATURE)
            .str("loc")
            .f64(5.0)
            .f64(0.0)
            .f64(1000.0);
        let (hits, total) = search(h, &geo);
        assert_eq!(total, 12);
        assert_eq!(hits[0].0, 5);
        let mut bad = Blob::tree().u8(NODE_LONG_DISTANCE_FEATURE).str("date");
        bad.0.extend_from_slice(&0i64.to_le_bytes());
        bad.0.extend_from_slice(&0i64.to_le_bytes());
        let (_, e) = fails(h, &bad);
        assert!(e.contains("pivotDistance must be > 0"), "{e}");
        let (_, e) = fails(
            h,
            &Blob::tree()
                .u8(NODE_GEO_DISTANCE_FEATURE)
                .str("loc")
                .f64(91.0)
                .f64(0.0)
                .f64(1.0),
        );
        assert!(e.contains("distance feature"), "{e}");
        assert_eq!(ffi_close_jvm_reader(h), 0);
    }
}
