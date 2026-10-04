//! The grouping benchmark pair (M10 T10.4), against
//! `benchmarks/micro/java/GroupingMicro.java`: per word, one grouping search
//! over `body:word` -- `GroupingSearch` by term (by relevance; by a field
//! with a within-group sort; with all groups and group heads; cached), by
//! long range and by blocks, the first-pass and distinct-values managers,
//! and the grouped facets -- over the 200 000-document, four-segment index
//! the Java side builds (`GroupingMicro build <dir>`). Each case prints a
//! `#check` digest of its result the report compares before it shows a
//! ratio.

use std::hint::black_box;
use std::time::Duration;

use lucene_search::directory_reader::DirectoryReader;
use lucene_search::grouping::{
    search_manager, DistinctValuesCollectorManager, FirstPassGroupingCollectorManager,
    GroupingSearch, LongRangeFactory, LongRangeGroupSelector, Sort, TermGroupFacetCollector,
    TermGroupSelector, TopGroups,
};
use lucene_search::index_searcher::{IndexSearcher, SegmentNorms};
use lucene_search::leaf_collector::search_segments;
use lucene_search::query::{BooleanQuery, Clause, TermQuery};
use lucene_search::top_field::{SortField, SortType};
use lucene_search::values_source;
use lucene_store::FsDirectory;

use super::measure;

/// FNV-1a over 64-bit words, identical to `GroupingMicro.Fnv`.
struct Fnv(u64);

impl Fnv {
    fn new() -> Self {
        Fnv(0xcbf2_9ce4_8422_2325)
    }
    fn add(&mut self, x: i64) {
        self.0 ^= x as u64;
        self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
    }
    fn bytes(&mut self, b: Option<&[u8]>) {
        match b {
            None => self.add(-1),
            Some(b) => {
                for &x in b {
                    self.add(i64::from(x as i8));
                }
            }
        }
    }
}

fn top_groups<V>(f: &mut Fnv, tg: &TopGroups<V>) {
    f.add(i64::from(tg.total_hit_count));
    f.add(i64::from(tg.total_grouped_hit_count));
    f.add(tg.total_group_count.map_or(-1, i64::from));
    for g in &tg.groups {
        f.add(g.total_hits.value as i64);
        for d in &g.score_docs {
            f.add(i64::from(d.doc));
            let bits = if d.score.is_nan() {
                0x7fc0_0000
            } else {
                d.score.to_bits()
            };
            f.add(i64::from(bits as i32));
        }
    }
}

fn q(word: &str) -> BooleanQuery {
    BooleanQuery {
        must: vec![Clause::Term(TermQuery::new(
            "body",
            word.as_bytes().to_vec(),
        ))],
        ..Default::default()
    }
}

fn cases(name: &str, words: &[String], w: Duration, m: Duration, run: &dyn Fn(&str, &mut Fnv)) {
    let mut f = Fnv::new();
    for word in words {
        run(word, &mut f);
    }
    println!("#check\t{name}\t{:016x}\t{}", f.0, words.len());
    measure(name, w, m, || {
        let mut g = Fnv::new();
        for word in black_box(words) {
            run(word, &mut g);
        }
        black_box(g.0);
        words.len() as u64
    });
}

pub fn bench_grouping(w: Duration, m: Duration, dir: &str) {
    let words: Vec<String> = std::fs::read_to_string(format!("{dir}/group-words.tsv"))
        .expect("run GroupingMicro build first (scripts/bench-micro.sh --bench grouping)")
        .lines()
        .map(str::to_string)
        .collect();
    let reader = DirectoryReader::open(&FsDirectory::open(std::path::Path::new(dir))).unwrap();
    let opened = reader.open_segments().unwrap();
    let segments = opened.as_open_segments();
    let owned = reader.field_norms_by_field(&["body".to_string()]);
    let norms: Vec<SegmentNorms<'_, '_>> = owned.iter().map(Some).collect();
    let s = IndexSearcher::new(&segments, &norms).unwrap();
    let by_n = Sort::new(vec![SortField::numeric("n", SortType::Long, true)]);
    let by_s = Sort::new(vec![SortField::string("s", false)]);

    cases("grp_term_rel", &words, w, m, &|word, f| {
        let g = GroupingSearch::new(|| TermGroupSelector::new("g")).set_group_docs_limit(3);
        top_groups(f, &g.search(&s, &q(word), 0, 10).unwrap().top_groups);
    });
    cases("grp_term_sorted", &words, w, m, &|word, f| {
        let g = GroupingSearch::new(|| TermGroupSelector::new("g"))
            .set_group_sort(by_n.clone())
            .set_sort_within_group(by_s.clone())
            .set_group_docs_limit(3);
        top_groups(f, &g.search(&s, &q(word), 5, 10).unwrap().top_groups);
    });
    cases("grp_term_all", &words, w, m, &|word, f| {
        let g = GroupingSearch::new(|| TermGroupSelector::new("g"))
            .set_all_groups(true)
            .set_all_group_heads(true);
        let r = g.search(&s, &q(word), 0, 10).unwrap();
        top_groups(f, &r.top_groups);
        f.add(r.matching_groups.len() as i64);
    });
    cases("grp_term_cached", &words, w, m, &|word, f| {
        let g = GroupingSearch::new(|| TermGroupSelector::new("g"))
            .set_caching(1_000_000, true)
            .set_group_docs_limit(3);
        top_groups(f, &g.search(&s, &q(word), 0, 10).unwrap().top_groups);
    });
    cases("grp_long_range", &words, w, m, &|word, f| {
        let factory = LongRangeFactory {
            min: 0,
            width: 250,
            max: 10_000,
        };
        let g = GroupingSearch::new(|| {
            LongRangeGroupSelector::new(values_source::long_from_long_field("n"), factory)
        })
        .set_group_docs_limit(2);
        top_groups(f, &g.search(&s, &q(word), 0, 10).unwrap().top_groups);
    });
    cases("grp_blocks", &words, w, m, &|word, f| {
        let end = BooleanQuery {
            must: vec![Clause::Term(TermQuery::new("end", b"x".to_vec()))],
            ..Default::default()
        };
        let g = GroupingSearch::by_blocks(end).set_group_docs_limit(2);
        top_groups(f, &g.search_blocks(&s, &q(word), 0, 10).unwrap());
    });
    cases("grp_distinct", &words, w, m, &|word, f| {
        let first = search_manager(
            &s,
            &q(word),
            &FirstPassGroupingCollectorManager::new(
                || TermGroupSelector::new("g"),
                Sort::relevance(),
                0,
                20,
                false,
            )
            .unwrap(),
        )
        .unwrap();
        f.add(first.len() as i64);
        if first.is_empty() {
            return;
        }
        let counts = search_manager(
            &s,
            &q(word),
            &DistinctValuesCollectorManager::new(
                || TermGroupSelector::new("g"),
                first,
                || TermGroupSelector::new("v"),
            ),
        )
        .unwrap();
        for c in &counts {
            f.bytes(c.group_value.as_deref());
            f.add(c.unique_values.len() as i64);
        }
    });
    for mv in [false, true] {
        let name = if mv { "grp_facet_mv" } else { "grp_facet_sv" };
        cases(name, &words, w, m, &|word, f| {
            let mut c = TermGroupFacetCollector::new("g", if mv { "fmv" } else { "fsv" }, mv, None);
            search_segments(&s, &q(word), &mut c).unwrap();
            let res = c.merge_segment_results(10, 0, true).unwrap();
            f.add(i64::from(res.total_count()));
            f.add(i64::from(res.total_missing_count()));
            for e in res.facet_entries(0, 10) {
                f.bytes(Some(&e.value));
                f.add(i64::from(e.count));
            }
        });
    }
}
