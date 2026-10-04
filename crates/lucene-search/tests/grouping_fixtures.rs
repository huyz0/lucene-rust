//! M10 T10.4, differentially against Lucene 10.5.0:
//! `fixtures/src/GenGrouping.java`.
//!
//! Every recorded search -- `GroupingSearch` by a term, long-range,
//! double-range or score-range selector (group and within-group sorts,
//! offsets and limits, max scores, all groups, group heads, documents
//! without a group, caching), the collector managers over one slice and two
//! (first pass, top groups with each score merge mode, distinct values, all
//! groups, group heads), `GroupingSearch` by blocks, and
//! `TermGroupFacetCollector` (single- and multi-valued, prefixes, sizes,
//! minimum counts, orders, offsets) -- is run here over the same index and
//! printed as the generator prints it: the same groups, documents, score
//! bits and sort values, or an error where Lucene threw.

// Test fixtures' own arithmetic -- see `docs/arithmetic-gate.md`'s "Test code".
#![allow(clippy::arithmetic_side_effects)]

use std::fmt::Debug;
use std::hash::Hash;

use lucene_search::directory_reader::DirectoryReader;
use lucene_search::grouping::{
    search_manager, AllGroupHeadsCollectorManager, AllGroupsCollectorManager,
    DistinctValuesCollectorManager, DoubleRange, DoubleRangeFactory, DoubleRangeGroupSelector,
    FirstPassGroupingCollectorManager, GroupSelector, GroupSortValue, GroupingSearch, LongRange,
    LongRangeFactory, LongRangeGroupSelector, ScoreMergeMode, SearchGroup, Sort,
    TermGroupFacetCollector, TermGroupSelector, TopGroups, TopGroupsCollectorManager,
};
use lucene_search::index_searcher::{IndexSearcher, SegmentNorms};
use lucene_search::leaf_collector::search_segments;
use lucene_search::query::{BooleanQuery, Clause, MatchAllDocsQuery, TermQuery};
use lucene_search::top_field::{Selector, SortField, SortType};
use lucene_search::values_source;
use lucene_store::FsDirectory;

fn data() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/data/grouping")
}

fn term(field: &str, value: &str) -> Clause {
    Clause::Term(TermQuery::new(field, value.as_bytes().to_vec()))
}

/// The generator's query spec.
fn query(spec: &str) -> BooleanQuery {
    let (kind, words) = spec.split_once(':').unwrap_or((spec, ""));
    let two = || {
        let (a, b) = words.split_once(',').unwrap();
        (term("body", a), term("body", b))
    };
    match kind {
        "all" => BooleanQuery {
            must: vec![Clause::MatchAllDocs(MatchAllDocsQuery::new(0))],
            ..Default::default()
        },
        "t" => BooleanQuery {
            must: vec![term("body", words)],
            ..Default::default()
        },
        "or" => {
            let (a, b) = two();
            BooleanQuery {
                should: vec![a, b],
                ..Default::default()
            }
        }
        "not" => {
            let (a, b) = two();
            BooleanQuery {
                must: vec![a],
                must_not: vec![b],
                ..Default::default()
            }
        }
        k => panic!("query {k}"),
    }
}

/// The generator's sort spec.
fn sort(spec: &str) -> Sort {
    match spec {
        "REL" => return Sort::relevance(),
        "IDX" => return Sort::index_order(),
        _ => {}
    }
    Sort::new(
        spec.split(',')
            .map(|k| {
                let (name, rev) = match k.strip_suffix(":r") {
                    Some(n) => (n, true),
                    None => (k, false),
                };
                let mut f = match name {
                    "score" => SortField::score(),
                    "doc" => SortField::doc(),
                    "s" | "slast" => SortField::string("s", false),
                    "g" => SortField::string("g", false),
                    "fmv" => SortField::string("fmv", false),
                    "n" | "nm" => SortField::numeric("n", SortType::Long, false),
                    "ni" => SortField::numeric("ni", SortType::Int, false),
                    "ds" => SortField::numeric("ds", SortType::Double, false),
                    "fs" => SortField::numeric("fs", SortType::Float, false),
                    "mvmax" => SortField::numeric("mv", SortType::Long, false),
                    n => panic!("sort key {n}"),
                };
                f.reverse = rev;
                match name {
                    "slast" => f.missing = 1,
                    "nm" => f.missing = 7,
                    "mvmax" => f.selector = Selector::Max,
                    _ => {}
                }
                f
            })
            .collect(),
    )
}

/// `key=value;...`.
fn params(spec: &str) -> std::collections::HashMap<&str, &str> {
    spec.split(';')
        .map(|kv| kv.split_once('=').unwrap())
        .collect()
}

fn hex32(f: f32) -> String {
    format!("{:x}", if f.is_nan() { 0x7fc0_0000 } else { f.to_bits() })
}

fn hex64(d: f64) -> String {
    format!(
        "{:x}",
        if d.is_nan() {
            0x7ff8_0000_0000_0000
        } else {
            d.to_bits()
        }
    )
}

fn sort_value(v: &GroupSortValue) -> String {
    match v {
        GroupSortValue::Float(f) => format!("f:{}", hex32(*f)),
        GroupSortValue::Double(d) => format!("d:{}", hex64(*d)),
        GroupSortValue::Long(l) => format!("l:{l}"),
        GroupSortValue::Int(i) => format!("i:{i}"),
        GroupSortValue::Bytes(None) => "null".into(),
        GroupSortValue::Bytes(Some(b)) => format!("b:{}", String::from_utf8_lossy(b)),
    }
}

fn sort_values(vs: &[GroupSortValue]) -> String {
    vs.iter().map(sort_value).collect::<Vec<_>>().join(",")
}

/// How a group value prints.
trait Value {
    fn fmt(&self) -> String;
}

impl Value for Vec<u8> {
    fn fmt(&self) -> String {
        format!("b:{}", String::from_utf8_lossy(self))
    }
}

impl Value for LongRange {
    fn fmt(&self) -> String {
        format!("L({},{})", self.min, self.max)
    }
}

impl Value for DoubleRange {
    fn fmt(&self) -> String {
        format!("D({},{})", hex64(self.min), hex64(self.max))
    }
}

impl Value for () {
    fn fmt(&self) -> String {
        "null".into()
    }
}

fn val<V: Value>(v: Option<&V>) -> String {
    v.map_or_else(|| "null".into(), Value::fmt)
}

fn sorted_vals<V: Value>(vs: &[Option<V>]) -> String {
    let mut l: Vec<String> = vs.iter().map(|v| val(v.as_ref())).collect();
    l.sort();
    l.join(",")
}

fn top_groups<V: Value>(tg: Option<&TopGroups<V>>) -> String {
    let Some(tg) = tg else {
        return "null".into();
    };
    let mut b = format!(
        "thc={} tghc={} tgc={} ms={} gsn={} wsn={}",
        tg.total_hit_count,
        tg.total_grouped_hit_count,
        tg.total_group_count
            .map_or_else(|| "null".into(), |n| n.to_string()),
        hex32(tg.max_score),
        tg.group_sort.len(),
        tg.within_group_sort.len()
    );
    for g in &tg.groups {
        b.push_str(&format!(
            " |gv={} sv={} s={} ms={} th={} docs=",
            val(g.group_value.as_ref()),
            sort_values(&g.group_sort_values),
            hex32(g.score),
            hex32(g.max_score),
            g.total_hits.value
        ));
        let docs: Vec<String> = g
            .score_docs
            .iter()
            .map(|d| {
                let mut s = format!("{}:{}", d.doc, hex32(d.score));
                if let Some(f) = &d.fields {
                    s.push(':');
                    s.push_str(&sort_values(f));
                }
                s
            })
            .collect();
        b.push_str(&docs.join(";"));
    }
    b
}

fn search_groups<V: Value>(groups: &[SearchGroup<V>]) -> String {
    groups
        .iter()
        .map(|g| {
            format!(
                "{}={}",
                val(g.group_value.as_ref()),
                sort_values(&g.sort_values)
            )
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn bits(b: &lucene_util::fixed_bit_set::FixedBitSet) -> String {
    (0..b.len())
        .filter(|&i| b.get(i))
        .map(|i| i.to_string())
        .collect::<Vec<_>>()
        .join(",")
}

/// A `GroupingSearch` by a selector, printed.
fn grouping_search<'a, S, F>(
    searcher: &IndexSearcher<'_, 'a>,
    p: &std::collections::HashMap<&str, &str>,
    factory: F,
) -> lucene_search::Result<String>
where
    S: GroupSelector<'a>,
    S::Value: Value,
    F: Fn() -> S,
{
    let n = |k: &str| p[k].parse::<usize>().unwrap();
    let flag = |k: &str| p[k] == "1";
    let mut g = GroupingSearch::new(factory)
        .set_group_sort(sort(p["gs"]))
        .set_sort_within_group(sort(p["ws"]))
        .set_group_docs_offset(n("gdo"))
        .set_group_docs_limit(n("gdl"))
        .set_include_max_score(flag("ms"))
        .set_all_groups(flag("ag"))
        .set_all_group_heads(flag("ah"))
        .set_ignore_docs_without_group_field(flag("ign"));
    let cache: Vec<&str> = p["cache"].split(':').collect();
    match cache[0] {
        "docs" => g = g.set_caching(cache[1].parse().unwrap(), cache[2] == "1"),
        "mb" => g = g.set_caching_in_mb(cache[1].parse().unwrap(), cache[2] == "1"),
        _ => {}
    }
    let r = g.search(searcher, &query(p["q"]), n("go"), n("gl"))?;
    Ok(format!(
        "{} #groups={} #heads={}",
        top_groups(Some(&r.top_groups)),
        sorted_vals(&r.matching_groups),
        bits(&r.matching_group_heads)
    ))
}

/// The collector managers, printed.
fn managers<'a, S, F>(
    searcher: &IndexSearcher<'_, 'a>,
    p: &std::collections::HashMap<&str, &str>,
    factory: F,
) -> lucene_search::Result<String>
where
    S: GroupSelector<'a>,
    S::Value: Value + Clone + Eq + Hash + Debug,
    F: Fn() -> S + Copy,
{
    let n = |k: &str| p[k].parse::<usize>().unwrap();
    let flag = |k: &str| p[k] == "1";
    let q = query(p["q"]);
    let (gs, ws) = (sort(p["gs"]), sort(p["ws"]));
    let first = search_manager(
        searcher,
        &q,
        &FirstPassGroupingCollectorManager::new(
            factory,
            gs.clone(),
            n("go"),
            n("gl"),
            flag("ign"),
        )?,
    )?;
    let mut b = format!("first={}", search_groups(&first));
    if !first.is_empty() {
        let smm = match p["smm"] {
            "None" => ScoreMergeMode::None,
            "Total" => ScoreMergeMode::Total,
            "Avg" => ScoreMergeMode::Avg,
            m => panic!("{m}"),
        };
        let top = search_manager(
            searcher,
            &q,
            &TopGroupsCollectorManager::new(
                factory,
                first.clone(),
                gs.clone(),
                ws.clone(),
                n("gdo"),
                n("gdl"),
                flag("ms"),
                smm,
            ),
        )?;
        b.push_str(&format!(" #top={}", top_groups(top.as_ref())));
        let distinct = search_manager(
            searcher,
            &q,
            &DistinctValuesCollectorManager::new(factory, first, || TermGroupSelector::new("g2")),
        )?;
        b.push_str(" #distinct=");
        for gc in &distinct {
            b.push_str(&format!(
                "{}{{{}}}",
                val(gc.group_value.as_ref()),
                sorted_vals(&gc.unique_values)
            ));
        }
    }
    // A range selector these two never hand a scorer has no values.
    match search_manager(searcher, &q, &AllGroupsCollectorManager::new(factory)) {
        Ok(all) => b.push_str(&format!(" #all={}", sorted_vals(&all))),
        Err(_) => b.push_str(" #all=ERR"),
    }
    match search_manager(
        searcher,
        &q,
        &AllGroupHeadsCollectorManager::new(factory, ws),
    ) {
        Ok(heads) => {
            let mut h = heads.retrieve_group_heads().to_vec();
            h.sort_unstable();
            b.push_str(&format!(
                " #heads=[{}]",
                h.iter()
                    .map(|d| d.to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            ));
        }
        Err(_) => b.push_str(" #heads=ERR"),
    }
    Ok(b)
}

/// Runs `$body` with `$factory` bound to the selector `$spec` names.
macro_rules! with_selector {
    ($spec:expr, |$factory:ident| $body:expr) => {{
        let parts: Vec<&str> = $spec.split(':').collect();
        match parts[0] {
            "term" => {
                let field = parts[1].to_string();
                let $factory = || TermGroupSelector::new(&field);
                $body
            }
            "long" => {
                let f = LongRangeFactory {
                    min: parts[2].parse().unwrap(),
                    width: parts[3].parse().unwrap(),
                    max: parts[4].parse().unwrap(),
                };
                let $factory =
                    || LongRangeGroupSelector::new(values_source::long_from_long_field("n"), f);
                $body
            }
            "double" => {
                let f = DoubleRangeFactory {
                    min: parts[2].parse().unwrap(),
                    width: parts[3].parse().unwrap(),
                    max: parts[4].parse().unwrap(),
                };
                let $factory =
                    || DoubleRangeGroupSelector::new(values_source::from_double_field("d"), f);
                $body
            }
            "dscore" => {
                let f = DoubleRangeFactory {
                    min: parts[1].parse().unwrap(),
                    width: parts[2].parse().unwrap(),
                    max: parts[3].parse().unwrap(),
                };
                let $factory = || DoubleRangeGroupSelector::new(values_source::scores(), f);
                $body
            }
            s => panic!("selector {s}"),
        }
    }};
}

fn block<'a>(
    searcher: &IndexSearcher<'_, 'a>,
    p: &std::collections::HashMap<&str, &str>,
) -> lucene_search::Result<String> {
    let n = |k: &str| p[k].parse::<usize>().unwrap();
    let end = BooleanQuery {
        must: vec![term("end", "x")],
        ..Default::default()
    };
    let tg = GroupingSearch::by_blocks(end)
        .set_group_sort(sort(p["gs"]))
        .set_sort_within_group(sort(p["ws"]))
        .set_group_docs_offset(n("gdo"))
        .set_group_docs_limit(n("gdl"))
        .search_blocks(searcher, &query(p["q"]), n("go"), n("gl"))?;
    Ok(top_groups(Some(&tg)))
}

fn facet<'a>(
    searcher: &IndexSearcher<'_, 'a>,
    p: &std::collections::HashMap<&str, &str>,
) -> lucene_search::Result<String> {
    let n = |k: &str| p[k].parse::<usize>().unwrap();
    let prefix = (p["prefix"] != "-").then(|| p["prefix"].as_bytes().to_vec());
    let mut c = TermGroupFacetCollector::new(p["group"], p["field"], p["mv"] == "1", prefix);
    search_segments(searcher, &query(p["q"]), &mut c)?;
    let res = c.merge_segment_results(n("size"), p["min"].parse().unwrap(), p["bycount"] == "1")?;
    let mut b = format!(
        "total={} missing={}",
        res.total_count(),
        res.total_missing_count()
    );
    for e in res.facet_entries(n("offset"), n("limit")) {
        b.push_str(&format!(
            " {}:{}",
            String::from_utf8_lossy(&e.value),
            e.count
        ));
    }
    Ok(b)
}

#[test]
fn grouping_matches_lucene() {
    let dir = data();
    let reader = DirectoryReader::open(&FsDirectory::open(dir.join("index"))).unwrap();
    let opened = reader.open_segments().unwrap();
    let segments = opened.as_open_segments();
    let owned = reader.field_norms_by_field(&["body".to_string()]);
    let norms: Vec<SegmentNorms<'_, '_>> = owned.iter().map(Some).collect();
    let searcher = IndexSearcher::new(&segments, &norms).unwrap();
    let mut sliced = IndexSearcher::new(&segments, &norms).unwrap();
    sliced.set_slices(vec![vec![0, 2], vec![1, 3]]).unwrap();

    let breader = DirectoryReader::open(&FsDirectory::open(dir.join("blocks"))).unwrap();
    let bopened = breader.open_segments().unwrap();
    let bsegments = bopened.as_open_segments();
    let bowned = breader.field_norms_by_field(&["body".to_string()]);
    let bnorms: Vec<SegmentNorms<'_, '_>> = bowned.iter().map(Some).collect();
    let bsearcher = IndexSearcher::new(&bsegments, &bnorms).unwrap();
    let mut bsliced = IndexSearcher::new(&bsegments, &bnorms).unwrap();
    bsliced.set_slices(vec![vec![0], vec![1, 2]]).unwrap();

    let text = std::fs::read_to_string(dir.join("searches.tsv")).unwrap();
    let (mut checked, mut errors) = (0, 0);
    let mut kinds = std::collections::HashMap::new();
    for line in text.lines() {
        let mut cols = line.splitn(3, '\t');
        let (kind, spec, want) = (
            cols.next().unwrap(),
            cols.next().unwrap(),
            cols.next().unwrap(),
        );
        let p = params(spec);
        let got = match kind {
            "gs" => with_selector!(p["sel"], |f| grouping_search(&searcher, &p, f)),
            "mgr" => {
                let s = if p["sliced"] == "1" {
                    &sliced
                } else {
                    &searcher
                };
                with_selector!(p["sel"], |f| managers(s, &p, f))
            }
            "block" => block(
                if p["sliced"] == "1" {
                    &bsliced
                } else {
                    &bsearcher
                },
                &p,
            ),
            "facet" => facet(&searcher, &p),
            k => panic!("kind {k}"),
        };
        if want.starts_with("ERR ") {
            assert!(
                got.is_err(),
                "{kind} {spec}: Lucene threw {want}, got {got:?}"
            );
            errors += 1;
        } else {
            let got = got.unwrap_or_else(|e| panic!("{kind} {spec}: {e}"));
            assert_eq!(got, want, "{kind} {spec}");
        }
        *kinds.entry(kind).or_insert(0) += 1;
        checked += 1;
    }
    assert!(checked > 600, "{checked}");
    assert!(errors >= 3, "{errors}");
    assert!(kinds.len() == 4, "{kinds:?}");
}

/// What the recorded searches do not reach: the collectors driven directly,
/// their accessors, and their refusals.
#[test]
fn grouping_edges() {
    use lucene_search::grouping::{
        AllGroupHeadsCollector, BlockGroupingCollector, DistinctValuesCollector,
        FirstPassGroupingCollector, TopGroupsCollector,
    };
    use lucene_search::leaf_collector::SegmentCollector;

    let dir = data();
    let reader = DirectoryReader::open(&FsDirectory::open(dir.join("index"))).unwrap();
    let opened = reader.open_segments().unwrap();
    let segments = opened.as_open_segments();
    let owned = reader.field_norms_by_field(&["body".to_string()]);
    let norms: Vec<SegmentNorms<'_, '_>> = owned.iter().map(Some).collect();
    let searcher = IndexSearcher::new(&segments, &norms).unwrap();
    let all = query("all");

    // The two passes by hand, a group given twice.
    let mut first =
        FirstPassGroupingCollector::new(TermGroupSelector::new("g"), Sort::index_order(), 3, true)
            .unwrap();
    search_segments(&searcher, &all, &mut first).unwrap();
    let mut groups = first.top_groups(0).unwrap();
    assert_eq!(groups.len(), 3);
    assert!(groups.iter().all(|g| g.group_value.is_some()), "ignored");
    groups.push(groups[0].clone());
    let mut second = TopGroupsCollector::new(
        TermGroupSelector::new("g"),
        groups.clone(),
        Sort::index_order(),
        Sort::relevance(),
        2,
        false,
    )
    .unwrap();
    search_segments(&searcher, &all, &mut second).unwrap();
    let pass = second.second_pass();
    assert_eq!(pass.groups().len(), 4);
    assert!(pass.total_hit_count() >= pass.total_grouped_hit_count());
    assert!(pass.group_selector().current_value().is_some());
    let _ = pass.reducer();
    let tg = second.top_groups(0);
    assert_eq!(tg.groups[0].score_docs, tg.groups[3].score_docs);
    let mut distinct = DistinctValuesCollector::new(
        TermGroupSelector::new("g"),
        groups,
        TermGroupSelector::new("g2"),
    )
    .unwrap();
    search_segments(&searcher, &all, &mut distinct).unwrap();
    let counts = distinct.groups();
    assert_eq!(counts.len(), 4);
    assert_eq!(counts[0], counts[3]);
    assert!(distinct.second_pass().total_hit_count() > 0);
    let mut heads = AllGroupHeadsCollector::new(TermGroupSelector::new("g"), Sort::index_order());
    search_segments(&searcher, &all, &mut heads).unwrap();
    assert_eq!(heads.group_heads().len(), heads.group_heads_size());
    assert_eq!(
        heads.retrieve_group_heads_bits(1).cardinality(),
        1,
        "only doc 0 fits"
    );

    // A key grouping does not sort by, and a segment without a reader.
    let mut by_val = SortField::string("s", false);
    by_val.ty = SortType::StringVal;
    let mut bad = FirstPassGroupingCollector::new(
        TermGroupSelector::new("g"),
        Sort::new(vec![by_val]),
        3,
        false,
    )
    .unwrap();
    assert!(search_segments(&searcher, &all, &mut bad).is_err());
    let bare: Vec<_> = segments
        .iter()
        .map(|s| lucene_search::multi_segment::OpenSegment { reader: None, ..*s })
        .collect();
    for sort in [sort("n"), sort("s")] {
        let mut c =
            FirstPassGroupingCollector::new(TermGroupSelector::new("g"), sort, 3, false).unwrap();
        assert!(c.set_next_reader(0, &bare[0]).is_err());
    }
    // Selectors before a segment: none, or no values to read.
    let mut t = TermGroupSelector::new("g");
    assert_eq!(
        t.advance_to(0, 1.0).unwrap(),
        lucene_search::grouping::GroupState::Skip
    );
    let mut lr = LongRangeGroupSelector::new(
        values_source::long_from_long_field("n"),
        LongRangeFactory {
            min: 0,
            width: 1,
            max: 9,
        },
    );
    assert!(lr.set_scorer().is_err());
    assert!(lr.advance_to(0, 1.0).is_err());

    // GroupingSearch asked for the grouping it was not built for.
    let by_field = GroupingSearch::new(|| TermGroupSelector::new("g")).disable_caching();
    assert!(by_field.search_blocks(&searcher, &all, 0, 1).is_err());
    let end = BooleanQuery {
        must: vec![term("end", "x")],
        ..Default::default()
    };
    let by_blocks = GroupingSearch::by_blocks(end.clone());
    assert!(
        by_blocks.search_blocks(&searcher, &all, 0, 1).is_err(),
        "no group ends"
    );

    // Blocks sorted by relevance within the group need scores.
    let breader = DirectoryReader::open(&FsDirectory::open(dir.join("blocks"))).unwrap();
    let bopened = breader.open_segments().unwrap();
    let bsegments = bopened.as_open_segments();
    let bnorms: Vec<SegmentNorms<'_, '_>> = bsegments.iter().map(|_| None).collect();
    let bsearcher = IndexSearcher::new(&bsegments, &bnorms).unwrap();
    let mut block = BlockGroupingCollector::new(Sort::index_order(), 3, false, end).unwrap();
    search_segments(&bsearcher, &all, &mut block).unwrap();
    assert!(block.top_groups(&Sort::relevance(), 0, 0, 1).is_err());
    let tg = block
        .top_groups(&Sort::index_order(), 1, 0, 1)
        .unwrap()
        .unwrap();
    assert_eq!(tg.groups.len(), 2);
    assert!(tg.max_score.is_nan());
    assert!(block
        .top_groups(&Sort::index_order(), 3, 0, 1)
        .unwrap()
        .is_none());
}
