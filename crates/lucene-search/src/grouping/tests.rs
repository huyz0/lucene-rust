//! Unit tests of the grouping module's own boundaries: the merges, the
//! ordered set, the range factories, the facet result, and the argument
//! checks. The collectors themselves are proven against Lucene by
//! `tests/grouping_fixtures.rs`.

use super::*;
use crate::grouping::sort::GroupSortValue as V;
use crate::top_field::SortType;

fn sg(value: Option<&str>, sort: f32) -> SearchGroup<Vec<u8>> {
    SearchGroup {
        group_value: value.map(|v| v.as_bytes().to_vec()),
        sort_values: vec![V::Float(sort)],
    }
}

fn hit(doc: i32, score: f32) -> GroupScoreDoc {
    GroupScoreDoc {
        doc,
        score,
        fields: None,
        shard_index: -1,
    }
}

fn docs(value: Option<&str>, hits: Vec<GroupScoreDoc>, score: f32) -> GroupDocs<Vec<u8>> {
    GroupDocs {
        score,
        max_score: hits.first().map_or(f32::NAN, |h| h.score),
        total_hits: TotalHits {
            value: hits.len() as u64,
            relation: TotalHitsRelation::EqualTo,
        },
        score_docs: hits,
        group_value: value.map(|v| v.as_bytes().to_vec()),
        group_sort_values: vec![V::Float(1.0)],
    }
}

fn tg(groups: Vec<GroupDocs<Vec<u8>>>, total_group_count: Option<i32>) -> TopGroups<Vec<u8>> {
    TopGroups {
        total_hit_count: 10,
        total_grouped_hit_count: 8,
        total_group_count,
        groups,
        group_sort: vec![SortField::score()],
        within_group_sort: vec![SortField::score()],
        max_score: 1.0,
    }
}

#[test]
fn the_tree_set_keeps_order_and_drops_equals() {
    let mut set: TreeSet<i32> = TreeSet::new();
    let mut cmp = |a: &i32, b: &i32| a.cmp(b);
    for x in [5, 1, 9, 5] {
        set.add(x, &mut cmp);
    }
    assert_eq!(set.len(), 3);
    assert_eq!(set.iter().copied().collect::<Vec<_>>(), vec![1, 5, 9]);
    assert_eq!((set.first(), set.last()), (Some(1), Some(9)));
    assert!(set.remove(&5, &mut cmp));
    assert!(!set.remove(&5, &mut cmp));
    assert_eq!(set.poll_first(), Some(1));
    assert_eq!(set.poll_last(), Some(9));
    assert_eq!(set.poll_first(), None);
    assert_eq!(set.poll_last(), None);
}

#[test]
fn the_group_index_holds_one_null_key() {
    let mut ix: GroupIndex<String> = GroupIndex::default();
    ix.insert(None, 3);
    ix.insert(Some("a".into()), 1);
    assert_eq!(ix.len(), 2);
    assert_eq!(ix.get(None), Some(3));
    assert_eq!(ix.get(Some(&"a".to_string())), Some(1));
    ix.remove(None);
    ix.remove(Some(&"a".to_string()));
    assert_eq!(ix.len(), 0);
    assert_eq!(ix.get(None), None);
}

#[test]
fn non_nan_max_skips_nan() {
    assert_eq!(non_nan_max(f32::NAN, 2.0), 2.0);
    assert_eq!(non_nan_max(3.0, f32::NAN), 3.0);
    assert_eq!(non_nan_max(1.0, 2.0), 2.0);
    assert!(non_nan_max(f32::NAN, f32::NAN).is_nan());
}

#[test]
fn search_groups_merge_by_their_best_shard() {
    let sort = Sort::relevance();
    let shards = vec![
        vec![sg(Some("a"), 5.0), sg(Some("b"), 3.0), sg(None, 1.0)],
        vec![sg(Some("b"), 4.0), sg(Some("c"), 3.5), sg(Some("a"), 0.5)],
        vec![],
    ];
    let merged = SearchGroup::merge(&shards, 0, 10, &sort);
    let names: Vec<_> = merged
        .iter()
        .map(|g| g.group_value.clone().map(|v| String::from_utf8(v).unwrap()))
        .collect();
    assert_eq!(
        names,
        vec![Some("a".into()), Some("b".into()), Some("c".into()), None]
    );
    // `b`'s values are its best shard's.
    assert_eq!(merged[1].sort_values, vec![V::Float(4.0)]);
    // Offsets and limits.
    let page = SearchGroup::merge(&shards, 1, 2, &sort);
    assert_eq!(page.len(), 2);
    assert_eq!(page[0].group_value.as_deref(), Some(&b"b"[..]));
    assert!(SearchGroup::merge(&shards, 9, 2, &sort).is_empty());
    // A tie goes to the lower shard.
    let ties = vec![vec![sg(Some("x"), 1.0)], vec![sg(Some("y"), 1.0)]];
    let merged = SearchGroup::merge(&ties, 0, 2, &sort);
    assert_eq!(merged[0].group_value.as_deref(), Some(&b"x"[..]));
    // The same group, equal in a later shard, keeps the lower shard.
    let same = vec![vec![sg(Some("x"), 1.0)], vec![sg(Some("x"), 1.0)]];
    assert_eq!(SearchGroup::merge(&same, 0, 2, &sort).len(), 1);
}

#[test]
fn top_groups_merge_checks_its_shards_and_combines_scores() {
    let sort = Sort::relevance();
    assert!(
        TopGroups::<Vec<u8>>::merge(&[], &sort, &sort, 0, 5, ScoreMergeMode::None)
            .unwrap()
            .is_none()
    );
    let a = tg(
        vec![docs(Some("g"), vec![hit(1, 3.0), hit(4, 1.0)], 2.0)],
        Some(2),
    );
    let b = tg(vec![docs(Some("g"), vec![hit(9, 2.0)], 1.0)], None);
    let total = TopGroups::merge(
        &[a.clone(), b.clone()],
        &sort,
        &sort,
        0,
        5,
        ScoreMergeMode::Total,
    )
    .unwrap()
    .unwrap();
    assert_eq!(total.total_hit_count, 20);
    assert_eq!(total.total_group_count, Some(2));
    let g = &total.groups[0];
    assert_eq!(g.score, 3.0);
    assert_eq!(g.total_hits.value, 3);
    assert_eq!(
        g.score_docs
            .iter()
            .map(|h| (h.doc, h.shard_index))
            .collect::<Vec<_>>(),
        vec![(1, 0), (9, 1), (4, 0)]
    );
    assert_eq!(g.max_score, 3.0);
    let avg = TopGroups::merge(
        &[a.clone(), b.clone()],
        &sort,
        &sort,
        1,
        1,
        ScoreMergeMode::Avg,
    )
    .unwrap()
    .unwrap();
    assert_eq!(avg.groups[0].score, 1.0);
    assert_eq!(avg.groups[0].score_docs.len(), 1);
    assert_eq!(avg.groups[0].score_docs[0].doc, 9);
    let none = TopGroups::merge(
        std::slice::from_ref(&a),
        &sort,
        &sort,
        0,
        5,
        ScoreMergeMode::None,
    )
    .unwrap()
    .unwrap();
    assert!(none.groups[0].score.is_nan());
    // An empty group under `Avg` has no score.
    let empty = tg(vec![docs(Some("g"), vec![], 0.0)], None);
    let r = TopGroups::merge(&[empty], &sort, &sort, 0, 5, ScoreMergeMode::Avg)
        .unwrap()
        .unwrap();
    assert!(r.groups[0].score.is_nan());
    // By a field: each hit's sort values; the max scores combined.
    let by_doc = Sort::index_order();
    let mut fa = a.clone();
    for h in &mut fa.groups[0].score_docs {
        h.fields = Some(vec![V::Int(h.doc)]);
    }
    let mut fb = b.clone();
    fb.groups[0].score_docs[0].fields = Some(vec![V::Int(9)]);
    let r = TopGroups::merge(&[fb, fa], &sort, &by_doc, 0, 2, ScoreMergeMode::None)
        .unwrap()
        .unwrap();
    assert_eq!(
        r.groups[0]
            .score_docs
            .iter()
            .map(|h| h.doc)
            .collect::<Vec<_>>(),
        vec![1, 4]
    );
    assert_eq!(r.groups[0].max_score, 3.0);
    // Shards that disagree.
    let other = tg(vec![docs(Some("h"), vec![], 0.0)], None);
    let err = TopGroups::merge(
        &[a.clone(), other],
        &sort,
        &sort,
        0,
        5,
        ScoreMergeMode::None,
    );
    assert!(err.unwrap_err().to_string().contains("group values differ"));
    let null = tg(vec![docs(None, vec![], 0.0)], None);
    assert!(
        TopGroups::merge(&[null, a.clone()], &sort, &sort, 0, 5, ScoreMergeMode::None).is_err()
    );
    let two = tg(
        vec![docs(Some("g"), vec![], 0.0), docs(Some("h"), vec![], 0.0)],
        None,
    );
    let err = TopGroups::merge(&[a, two], &sort, &sort, 0, 5, ScoreMergeMode::None);
    assert!(err
        .unwrap_err()
        .to_string()
        .contains("number of groups differs"));
}

#[test]
fn block_groups_merge_by_their_sort_values() {
    let sort = Sort::new(vec![SortField::numeric("n", SortType::Long, false)]);
    let mk = |vals: &[i64], max: f32, count: Option<i32>| {
        let groups = vals
            .iter()
            .map(|&v| {
                let mut d = docs(None, vec![hit(v as i32, 1.0)], f32::NAN);
                d.group_sort_values = vec![V::Long(v)];
                d
            })
            .collect();
        let mut t = tg(groups, count);
        t.max_score = max;
        t
    };
    let shards = vec![
        mk(&[1, 4], 2.0, Some(2)),
        mk(&[], 9.0, None),
        mk(&[2, 4], 3.0, Some(3)),
    ];
    let r = TopGroups::merge_block_groups(&shards, &sort, 1, 2, &sort);
    assert_eq!(
        r.groups
            .iter()
            .map(|g| g.group_sort_values.clone())
            .collect::<Vec<_>>(),
        vec![vec![V::Long(2)], vec![V::Long(4)]]
    );
    // Ties to the lower slice: the first `4` is the first shard's.
    assert_eq!(r.groups[1].score_docs[0].doc, 4);
    assert_eq!(r.total_group_count, Some(5));
    assert_eq!(r.total_hit_count, 30);
    assert_eq!(r.total_grouped_hit_count, 2);
    // A slice without groups adds no max score.
    assert_eq!(r.max_score, 3.0);
    // By relevance the max score is the first group's shard's.
    let rel = TopGroups::merge_block_groups(&shards[..1], &Sort::relevance(), 0, 5, &sort);
    assert_eq!(rel.max_score, 2.0);
    let none = TopGroups::<()>::merge_block_groups(&[], &sort, 0, 5, &sort);
    assert!(none.groups.is_empty() && none.max_score.is_nan() && none.total_group_count.is_none());
    let tc = tg(vec![], None).with_total_group_count(Some(7));
    assert_eq!(tc.total_group_count, Some(7));
}

#[test]
fn range_factories_bucket_as_java_does() {
    let f = LongRangeFactory {
        min: 0,
        width: 10,
        max: 50,
    };
    assert_eq!(
        f.get_range(-1).unwrap(),
        LongRange {
            min: i64::MIN,
            max: 0
        }
    );
    assert_eq!(
        f.get_range(50).unwrap(),
        LongRange {
            min: 50,
            max: i64::MAX
        }
    );
    assert_eq!(f.get_range(27).unwrap(), LongRange { min: 20, max: 30 });
    let zero = LongRangeFactory {
        min: 0,
        width: 0,
        max: 50,
    };
    assert!(zero.get_range(5).is_err());
    assert!(zero.get_range(-5).is_ok());
    let d = DoubleRangeFactory {
        min: 0.0,
        width: 0.5,
        max: 2.0,
    };
    assert_eq!(
        d.get_range(-1.0),
        DoubleRange {
            min: f64::from_bits(1),
            max: 0.0
        }
    );
    assert_eq!(
        d.get_range(2.0),
        DoubleRange {
            min: 2.0,
            max: f64::MAX
        }
    );
    assert_eq!(d.get_range(1.2), DoubleRange { min: 1.0, max: 1.5 });
    let nan = d.get_range(f64::NAN);
    assert!(nan.min.is_nan() && nan == nan);
    assert_ne!(
        DoubleRange { min: 0.0, max: 1.0 },
        DoubleRange {
            min: -0.0,
            max: 1.0
        }
    );
    let mut set = std::collections::HashSet::new();
    set.insert(nan);
    assert!(set.contains(&d.get_range(f64::NAN)));
}

#[test]
fn the_facet_result_keeps_the_top_entries() {
    let mut by_count = GroupedFacetResult::new(2, 1, true, 10, 3);
    by_count.add_facet_count(b"a".to_vec(), 3);
    by_count.add_facet_count(b"b".to_vec(), 5);
    by_count.add_facet_count(b"z".to_vec(), 0);
    // Full: only an entry sorting before the last gets in.
    by_count.add_facet_count(b"c".to_vec(), 3);
    by_count.add_facet_count(b"0".to_vec(), 4);
    let e = by_count.facet_entries(0, 10);
    assert_eq!(
        e.iter()
            .map(|e| (e.value.clone(), e.count))
            .collect::<Vec<_>>(),
        vec![(b"b".to_vec(), 5), (b"0".to_vec(), 4)]
    );
    assert_eq!(by_count.facet_entries(1, 1).len(), 1);
    assert!(by_count.facet_entries(5, 1).is_empty());
    assert_eq!(
        (by_count.total_count(), by_count.total_missing_count()),
        (10, 3)
    );
    let mut by_value = GroupedFacetResult::new(3, 0, false, 0, 0);
    for (v, c) in [("m", 1), ("a", 9), ("z", 2), ("b", 1)] {
        by_value.add_facet_count(v.as_bytes().to_vec(), c);
    }
    assert_eq!(
        by_value
            .facet_entries(0, 9)
            .iter()
            .map(|e| e.value.clone())
            .collect::<Vec<_>>(),
        vec![b"a".to_vec(), b"m".to_vec(), b"z".to_vec()],
        "once full, the last entry's count is the minimum, by value too"
    );
}

#[test]
fn collectors_check_their_arguments() {
    let sort = Sort::relevance();
    assert!(
        FirstPassGroupingCollector::new(TermGroupSelector::new("g"), sort.clone(), 0, false)
            .is_err()
    );
    assert!(FirstPassGroupingCollectorManager::new(
        || TermGroupSelector::new("g"),
        sort.clone(),
        0,
        0,
        false
    )
    .is_err());
    assert!(TopGroupsCollector::new(
        TermGroupSelector::new("g"),
        vec![],
        sort.clone(),
        sort.clone(),
        1,
        true
    )
    .is_err());
    for by_score in [true, false] {
        let Err(e) = GroupTopDocs::new(by_score, &sort, 0, false) else {
            panic!("no hits accepted");
        };
        assert!(e.to_string().contains("numHits must be > 0"));
    }
    let end = BooleanQuery::default();
    assert!(BlockGroupingCollector::new(sort.clone(), 0, true, end.clone()).is_err());
    assert!(BlockGroupingCollectorManager::new(
        sort.clone(),
        0,
        0,
        true,
        end.clone(),
        sort.clone(),
        0,
        1
    )
    .is_err());
    assert!(BlockGroupingCollectorManager::new(
        sort.clone(),
        0,
        1,
        true,
        end.clone(),
        sort.clone(),
        0,
        0
    )
    .is_err());
    // A block collector that saw nothing has no groups.
    let c = BlockGroupingCollector::new(sort.clone(), 2, true, end).unwrap();
    assert!(c.top_groups(&sort, 0, 0, 1).unwrap().is_none());
    let mut first =
        FirstPassGroupingCollector::new(TermGroupSelector::new("g"), sort.clone(), 2, false)
            .unwrap();
    assert!(first.top_groups(0).is_none());
    assert_eq!(first.group_selector().current_value(), None);
    let _ = first.into_group_selector();
    let all = AllGroupsCollector::new(TermGroupSelector::new("g"));
    assert_eq!(all.group_count(), 0);
    let heads = AllGroupHeadsCollector::new(TermGroupSelector::new("g"), sort);
    assert_eq!(heads.group_heads_size(), 0);
    assert!(heads.retrieve_group_heads().is_empty());
    assert_eq!(heads.retrieve_group_heads_bits(4).cardinality(), 0);
}

#[test]
fn group_top_docs_keep_the_best_hits() {
    let sort = Sort::relevance();
    let mut c = GroupTopDocs::new(true, &sort, 2, false).unwrap();
    let leaf_free = |c: &mut GroupTopDocs<'_>| {
        for (doc, score) in [(0, 1.0), (1, 3.0), (2, 3.0), (3, 2.0), (4, f32::NAN)] {
            c.collect(doc, score).unwrap();
        }
    };
    leaf_free(&mut c);
    assert_eq!(
        c.all().iter().map(|h| h.doc).collect::<Vec<_>>(),
        vec![1, 2],
        "ties keep the earlier document"
    );
    assert_eq!(c.total_hits().value, 5);
    assert_eq!(c.top_docs(1, 5).len(), 1);
    assert!(c.top_docs(2, 5).is_empty());
    assert!(c.top_docs(0, 0).is_empty());
    assert!(c.max_score().is_none());
    // A field sort needs its segment's keys before it collects.
    let mut f = GroupTopDocs::new(false, &Sort::index_order(), 2, true).unwrap();
    assert!(f.max_score().unwrap().is_nan());
    assert!(f.collect(0, 1.0).is_err());
    assert!(!f.sorted_by_score());
}

#[test]
fn heads_results_set_their_bits() {
    let r = GroupHeadsResult {
        group_heads: vec![1, 3, 99],
    };
    assert_eq!(r.retrieve_group_heads(), &[1, 3, 99]);
    let bits = r.retrieve_group_heads_bits(4);
    assert!(bits.get(1) && bits.get(3) && bits.cardinality() == 2);
}

#[test]
fn a_queued_group_moves_up_when_a_later_shard_beats_it() {
    let sort = Sort::relevance();
    // `x` and `y` lead their shards; once they are taken, `b` is queued at
    // 7 behind `z` (7.8), and shard 1's 7.5 for `b` beats it.
    let shards = vec![
        vec![sg(Some("x"), 9.0), sg(Some("b"), 7.0)],
        vec![sg(Some("y"), 8.0), sg(Some("b"), 7.5)],
        vec![sg(Some("z"), 7.8)],
    ];
    let merged = SearchGroup::merge(&shards, 0, 4, &sort);
    let names: Vec<_> = merged
        .iter()
        .map(|g| (g.group_value.clone().unwrap(), g.sort_values.clone()))
        .collect();
    assert_eq!(
        names,
        vec![
            (b"x".to_vec(), vec![V::Float(9.0)]),
            (b"y".to_vec(), vec![V::Float(8.0)]),
            (b"z".to_vec(), vec![V::Float(7.8)]),
            (b"b".to_vec(), vec![V::Float(7.5)]),
        ]
    );
}

#[test]
fn doubles_compare_with_one_nan() {
    let d = SortField::numeric("d", SortType::Double, false);
    assert_eq!(
        sort::compare_values(&d, &V::Double(f64::NAN), &V::Double(-f64::NAN)),
        Ordering::Equal
    );
    assert_eq!(
        sort::compare_values(&d, &V::Double(f64::NAN), &V::Double(f64::INFINITY)),
        Ordering::Greater
    );
}
