#![allow(clippy::arithmetic_side_effects)]
//! **`STRING_VAL`, `BinarySortField` and the `MIDDLE_MIN`/`MIDDLE_MAX`
//! selectors against real Lucene.**
//!
//! `fixtures/src/GenSortVal.java` writes two segments (the first with
//! deletions) with a sparse `BINARY` column (empty values included), a
//! multi-valued `SORTED_SET` column and a numeric tie-breaker, and records
//! Lucene's sorted top hits for three queries under ten sorts -- bytes keys
//! (`STRING_VAL` and `BinarySortField`, ascending and reversed, missing first
//! and last), `SortedSetSortField` with `MIDDLE_MIN`/`MIDDLE_MAX` (and `MIN`),
//! mixed with numeric, score and other keys -- at two `topN`s and two
//! total-hits thresholds, plus a `searchAfter` page from the tenth hit. Every
//! hit is compared by document and sort values, with the total and its
//! relation.

mod m7support;

use std::collections::HashMap;

use lucene_search::collector::TotalHitsRelation;
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::field_norms::FieldNorms;
use lucene_search::top_field::{search_sorted, FieldDoc, Selector, SortField, SortType};
use lucene_store::FsDirectory;
use m7support::{fixture, Grammar, Manifest};

const GRAMMAR: Grammar = Grammar {
    text: "body",
    range: "r",
};

fn sort_of(spec: &str) -> (Vec<SortField>, Vec<String>) {
    let mut sort = Vec::new();
    let mut kinds = Vec::new();
    for k in spec.split(',') {
        let p: Vec<&str> = k.split(':').collect();
        let reverse = p[2] == "true";
        let last = p[3] == "last";
        let sf = match p[1] {
            "val" => {
                let mut s = SortField::string_val(p[0], reverse);
                s.missing = i64::from(last);
                s
            }
            "binary" => SortField::binary(p[0], reverse, last),
            "min" | "middle_min" | "middle_max" => {
                let mut s = SortField::string(p[0], reverse);
                s.missing = i64::from(last);
                s.selector = match p[1] {
                    "min" => Selector::Min,
                    "middle_min" => Selector::MiddleMin,
                    _ => Selector::MiddleMax,
                };
                s
            }
            "long" => SortField::numeric(p[0], SortType::Long, reverse),
            "custom_mod3" => SortField::custom(p[0], custom::ids().0, reverse),
            "custom_len" => SortField::custom(p[0], custom::ids().1, reverse),
            "score" => {
                let mut s = SortField::score();
                s.reverse = reverse;
                s
            }
            other => panic!("{other}"),
        };
        sort.push(sf);
        kinds.push(p[1].to_string());
    }
    (sort, kinds)
}

/// The fixture's two `FieldComparatorSource`s, as `GenSortVal`'s `Mod3` and
/// `Length`.
mod custom {
    use std::cmp::Ordering;
    use std::sync::{Arc, OnceLock};

    use lucene_codecs::doc_values::{BinaryReader, NumericReader};
    use lucene_search::top_field::{
        register_comparator_source, CustomSortId, FieldComparator, FieldComparatorSource, LeafCtx,
        LeafFieldComparator, SortValue,
    };
    use lucene_search::Result;

    pub fn ids() -> (CustomSortId, CustomSortId) {
        static IDS: OnceLock<(CustomSortId, CustomSortId)> = OnceLock::new();
        *IDS.get_or_init(|| {
            (
                register_comparator_source(Arc::new(Mod3Source)),
                register_comparator_source(Arc::new(LengthSource)),
            )
        })
    }

    struct Mod3Source;
    struct Mod3(String);
    struct Mod3Leaf<'a>(Option<NumericReader<'a>>);

    impl FieldComparatorSource for Mod3Source {
        fn new_comparator(&self, field: &str, _n: usize, _r: bool) -> Box<dyn FieldComparator> {
            Box::new(Mod3(field.to_string()))
        }
    }

    impl FieldComparator for Mod3 {
        fn leaf<'a>(&self, ctx: LeafCtx<'a>) -> Result<Box<dyn LeafFieldComparator + 'a>> {
            let r = ctx.reader;
            let col = r.field_infos().field_by_name(&self.0).and_then(|i| {
                let (meta, data) = r.doc_values_for_field(i.number)?;
                meta.numeric_entry(i.number)
                    .map(|e| NumericReader::new(data, e))
            });
            Ok(Box::new(Mod3Leaf(col)))
        }
        fn compare_values(&self, a: &SortValue, b: &SortValue) -> Ordering {
            let (SortValue::Long(a), SortValue::Long(b)) = (a, b) else {
                return Ordering::Equal;
            };
            a.rem_euclid(3).cmp(&b.rem_euclid(3)).then(a.cmp(b))
        }
    }

    impl LeafFieldComparator for Mod3Leaf<'_> {
        fn value(&mut self, doc: i32, _score: f32) -> Result<SortValue> {
            Ok(SortValue::Long(match self.0.as_mut() {
                Some(c) => c.value(doc)?.unwrap_or(0),
                None => 0,
            }))
        }
    }

    struct LengthSource;
    struct Length(String);
    struct LengthLeaf<'a>(Option<BinaryReader<'a>>);

    impl FieldComparatorSource for LengthSource {
        fn new_comparator(&self, field: &str, _n: usize, _r: bool) -> Box<dyn FieldComparator> {
            Box::new(Length(field.to_string()))
        }
    }

    impl FieldComparator for Length {
        fn leaf<'a>(&self, ctx: LeafCtx<'a>) -> Result<Box<dyn LeafFieldComparator + 'a>> {
            let r = ctx.reader;
            let col = r.field_infos().field_by_name(&self.0).and_then(|i| {
                let (meta, data) = r.doc_values_for_field(i.number)?;
                meta.binary_entry(i.number)
                    .map(|e| BinaryReader::new(data, e))
            });
            Ok(Box::new(LengthLeaf(col)))
        }
        fn compare_values(&self, a: &SortValue, b: &SortValue) -> Ordering {
            let (SortValue::Bytes(a), SortValue::Bytes(b)) = (a, b) else {
                return Ordering::Equal;
            };
            match (a, b) {
                (None, None) => Ordering::Equal,
                (None, Some(_)) => Ordering::Less,
                (Some(_), None) => Ordering::Greater,
                (Some(x), Some(y)) => x.len().cmp(&y.len()).then(x.cmp(y)),
            }
        }
        fn values_are_bytes(&self) -> bool {
            true
        }
    }

    impl LeafFieldComparator for LengthLeaf<'_> {
        fn value(&mut self, doc: i32, _score: f32) -> Result<SortValue> {
            Ok(SortValue::Bytes(match self.0.as_mut() {
                Some(c) => c.value(doc)?.map(<[u8]>::to_vec),
                None => None,
            }))
        }
    }
}

/// One of Lucene's hits as this port's [`FieldDoc`].
fn field_doc(s: &str, kinds: &[String]) -> FieldDoc {
    let p: Vec<&str> = s.split(':').collect();
    let mut values = Vec::new();
    let mut terms = Vec::new();
    for (k, kind) in kinds.iter().enumerate() {
        let raw = p[1 + k];
        match kind.as_str() {
            "long" | "custom_mod3" => {
                values.push(raw.parse().unwrap());
                terms.push(None);
            }
            "score" => {
                values.push(i64::from(raw.parse::<i32>().unwrap() as u32));
                terms.push(None);
            }
            _ => {
                values.push(0);
                terms.push(if raw == "-" {
                    None
                } else {
                    let hex = &raw[1..];
                    Some(
                        (0..hex.len())
                            .step_by(2)
                            .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap())
                            .collect(),
                    )
                });
            }
        }
    }
    // A sort without a bytes key carries no terms at all.
    if terms.iter().all(Option::is_none)
        && kinds
            .iter()
            .all(|k| matches!(k.as_str(), "long" | "custom_mod3" | "score"))
    {
        terms.clear();
    }
    FieldDoc {
        doc: p[0].parse().unwrap(),
        values,
        terms,
    }
}

fn hits(s: &str, kinds: &[String]) -> Vec<FieldDoc> {
    if s.is_empty() {
        return Vec::new();
    }
    s.split(',').map(|h| field_doc(h, kinds)).collect()
}

#[test]
fn string_val_and_middle_selector_sorts_match_real_lucene() {
    let dir = fixture("sort_val_index");
    let m = Manifest::load(&format!("{dir}/manifest.properties"));
    let reader = DirectoryReader::open(&FsDirectory::open(&dir)).expect("open reader");
    let mut opened = reader.open_segments().expect("open postings");
    opened.open_points().expect("open points");
    let segments = opened.as_open_segments();
    let owned: Vec<HashMap<String, FieldNorms<'_>>> = reader
        .field_norms("body")
        .into_iter()
        .map(|n| n.into_iter().map(|n| ("body".to_string(), n)).collect())
        .collect();
    let norms: Vec<Option<&HashMap<String, FieldNorms<'_>>>> = owned.iter().map(Some).collect();
    let readers = reader.segment_readers();

    let runs: usize = m.get("run_count").parse().unwrap();
    let mut pages = 0;
    let mut failures = Vec::new();
    for r in 0..runs {
        let k = |f: &str| m.get(&format!("run.{r}.{f}")).to_string();
        let q = GRAMMAR.query(&k("query"));
        let (sort, kinds) = sort_of(&k("sort"));
        let top_n: usize = k("top_n").parse().unwrap();
        let threshold = match k("threshold").as_str() {
            "max" => u64::MAX,
            t => t.parse().unwrap(),
        };
        let got = search_sorted(
            &segments, readers, &q, &norms, &sort, top_n, threshold, None,
        )
        .unwrap();
        let want = hits(&k("hits"), &kinds);
        let want_total: u64 = k("total").parse().unwrap();
        let gte = k("relation") == "gte";
        let total_ok = if gte {
            got.total.relation == TotalHitsRelation::GreaterThanOrEqualTo
                && got.total.value > threshold
        } else {
            got.total.relation == TotalHitsRelation::EqualTo && got.total.value == want_total
        };
        if got.hits != want || !total_ok {
            failures.push(format!(
                "run {r} {} {}: got {:?} total {:?}\n want {:?} total {want_total} {gte}",
                k("query"),
                k("sort"),
                got.hits.iter().take(4).collect::<Vec<_>>(),
                got.total,
                want.iter().take(4).collect::<Vec<_>>()
            ));
            continue;
        }
        if let Some(after) = m.opt(&format!("run.{r}.after")) {
            pages += 1;
            let after = field_doc(after, &kinds);
            let page = search_sorted(
                &segments,
                readers,
                &q,
                &norms,
                &sort,
                10,
                u64::MAX,
                Some(&after),
            )
            .unwrap();
            let want = hits(m.get(&format!("run.{r}.page")), &kinds);
            if page.hits != want {
                failures.push(format!(
                    "run {r} {} {} page: got {:?}\n want {:?}",
                    k("query"),
                    k("sort"),
                    page.hits.iter().take(4).collect::<Vec<_>>(),
                    want.iter().take(4).collect::<Vec<_>>()
                ));
            }
        }
    }
    assert!(runs >= 100 && pages >= 20, "runs {runs}, pages {pages}");
    assert!(
        failures.is_empty(),
        "{} of {runs} runs disagree:\n{}",
        failures.len(),
        failures
            .iter()
            .take(8)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    );

    // A `SortRescorer` over every match orders the documents as the sorted
    // search does -- custom and bytes keys included.
    let searcher = lucene_search::index_searcher::IndexSearcher::new(&segments, &norms).unwrap();
    let ctx = lucene_search::values_source::ValuesContext::new(&searcher);
    let all = GRAMMAR.query("(t w0)");
    let first = searcher.search(&all, 10_000).unwrap();
    for spec in [
        "n:custom_mod3:true:first,bv:custom_len:false:first",
        "bv:val:true:last",
        "ss:middle_min:false:last,n:long:false:first",
    ] {
        let (sort, kinds) = sort_of(spec);
        let want =
            search_sorted(&segments, readers, &all, &norms, &sort, 15, u64::MAX, None).unwrap();
        let got = lucene_search::rescorer::SortRescorer::new(sort)
            .rescore_field_docs(&ctx, &first, 15)
            .unwrap();
        let docs: Vec<i32> = got.hits.iter().map(|h| h.fields.doc).collect();
        let want_docs: Vec<i32> = want.hits.iter().map(|h| h.doc).collect();
        assert_eq!(docs, want_docs, "rescore by {spec}");
        for (g, w) in got.hits.iter().zip(&want.hits) {
            assert_eq!(g.fields.terms.len(), w.terms.len(), "{spec} {kinds:?}");
            for (k, kind) in kinds.iter().enumerate() {
                if kind != "score" {
                    assert_eq!(g.fields.values[k], w.values[k], "{spec}");
                    assert_eq!(g.fields.terms.get(k), w.terms.get(k), "{spec}");
                }
            }
        }
    }

    // A bytes key over a column of another type is refused, as
    // `DocValues.getBinary` refuses it.
    let bad = [SortField::string_val("ss", false)];
    assert!(search_sorted(
        &segments,
        readers,
        &GRAMMAR.query("(all)"),
        &norms,
        &bad,
        5,
        u64::MAX,
        None
    )
    .is_err());
}
