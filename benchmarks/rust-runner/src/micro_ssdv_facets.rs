//! M13 T13.4d2's sorted-set facet and sampling pair (the Java twin is
//! `benchmarks/micro/java/SsdvFacetsMicro.java`), with the same case names,
//! over the 200,000-document, four-segment index `SsdvFacetsMicro build
//! <dir>` writes: `SortedSetDocValuesFacetField`s for a flat `Author`
//! (2,000 values), a multi-valued `Tag` (1-3 of 500) and a hierarchical
//! `Date` (year/month), and `q:rare` on every 100th document. Cases (the hits
//! collected beforehand unless named otherwise): `ssdv_state`, the
//! `DefaultSortedSetDocValuesReaderState` built (the ordinal map across the
//! segments), units: labels; `ssdv_count_all` and `ssdv_count_rare`,
//! `SortedSetDocValuesFacetCounts` over every hit (segment counts migrated)
//! and over `q:rare` (1%: each ordinal remapped as it comes), then
//! `getTopChildren(10, "Author")`, units: hits; `ssdv_count_reader`, the
//! counts of every live document, no collector, units: documents;
//! `ssdv_count_concurrent`, `ConcurrentSortedSetDocValuesFacetCounts` over
//! every hit on rayon's global pool (one thread per core, as the Java side's
//! pool), units: hits; `ssdv_top_children`, four `getTopChildren(10, ..)`
//! calls over the match-all counts, units: calls; `sampling_collect`, a
//! `RandomSamplingFacetsCollector` search (sample size 10,000, seed 42) over
//! every document and its sample drawn, units: hits;
//! `sampling_count_amortize`, the counts over that sample and
//! `amortizeFacetCounts` of `Author`, units: sampled hits. Each prints a
//! `#check` digest of its result first.

use std::cell::RefCell;
use std::collections::HashMap;
use std::path::Path;
use std::time::Duration;

use lucene_search::directory_reader::DirectoryReader;
use lucene_search::facets::{
    ConcurrentSortedSetDocValuesFacetCounts, DefaultSortedSetDocValuesReaderState, Facets,
    FacetsCollector, FacetsCollectorManager, FacetsConfig, RandomSamplingFacetsCollectorManager,
    SortedSetDocValuesFacetCounts,
};
use lucene_search::field_norms::FieldNorms;
use lucene_search::index_searcher::{IndexSearcher, SegmentNorms};
use lucene_search::query::{BooleanQuery, Clause, MatchAllDocsQuery, TermQuery};
use lucene_search::reader::{DocIdSetIterator, NO_MORE_DOCS};
use lucene_store::FsDirectory;

use super::micro_facet_counts::{cases, Fnv};

/// `SsdvFacetsMicro.config()`.
fn config() -> FacetsConfig {
    let mut c = FacetsConfig::new();
    c.set_hierarchical("Date", true);
    c.set_multi_valued("Tag", true);
    c
}

fn hits(fc: &FacetsCollector<'_>) -> u64 {
    fc.matching_docs().iter().map(|m| m.total_hits as u64).sum()
}

/// `SsdvFacetsMicro.sampled`: the documents of the sample, iterated.
fn sampled(fc: &FacetsCollector<'_>) -> u64 {
    let mut n = 0u64;
    for m in fc.matching_docs() {
        let mut it = m.bits.iterator();
        while it.next_doc().unwrap() != NO_MORE_DOCS {
            n += 1;
        }
    }
    n
}

fn tops(f: &mut Fnv, facets: &dyn Facets) {
    f.result(facets.get_top_children(10, "Author", &[]).unwrap());
    f.result(facets.get_top_children(10, "Tag", &[]).unwrap());
    f.result(facets.get_top_children(10, "Date", &[]).unwrap());
    f.result(facets.get_top_children(10, "Date", &["y2003"]).unwrap());
}

pub fn bench_ssdv_facets(w: Duration, m: Duration, dir: &str) {
    let dir = Path::new(dir);
    assert!(
        dir.join("built").exists(),
        "run SsdvFacetsMicro build first (scripts/bench-micro.sh --bench ssdv_facets)"
    );
    let reader = DirectoryReader::open(&FsDirectory::open(dir.join("index"))).unwrap();
    let opened = reader.open_segments().unwrap();
    let segments = opened.as_open_segments();
    let owned = reader.field_norms_by_field(&["q".to_string()]);
    let norms: Vec<SegmentNorms<'_, '_>> = owned
        .iter()
        .map(|m: &HashMap<String, FieldNorms<'_>>| Some(m))
        .collect();
    let searcher = IndexSearcher::new(&segments, &norms).unwrap();
    let config = config();
    let all = BooleanQuery {
        must: vec![Clause::MatchAllDocs(MatchAllDocsQuery::new(
            reader.max_doc(),
        ))],
        ..Default::default()
    };
    let rare = BooleanQuery {
        must: vec![Clause::Term(TermQuery::new("q", "rare"))],
        ..Default::default()
    };
    let fcm = FacetsCollectorManager::new(false);
    let fc_all = fcm.collect(&searcher, &all).unwrap();
    let fc_rare = fcm.collect(&searcher, &rare).unwrap();
    let (hits_all, hits_rare) = (hits(&fc_all), hits(&fc_rare));
    let state = DefaultSortedSetDocValuesReaderState::new(&reader, &config).unwrap();

    cases("ssdv_state", w, m, &|f| {
        let s = DefaultSortedSetDocValuesReaderState::new(&reader, &config).unwrap();
        f.add(i64::from(s.size()));
        s.size() as u64
    });
    cases("ssdv_count_all", w, m, &|f| {
        let c = SortedSetDocValuesFacetCounts::with_hits(&state, &fc_all).unwrap();
        f.result(c.get_top_children(10, "Author", &[]).unwrap());
        hits_all
    });
    cases("ssdv_count_rare", w, m, &|f| {
        let c = SortedSetDocValuesFacetCounts::with_hits(&state, &fc_rare).unwrap();
        f.result(c.get_top_children(10, "Author", &[]).unwrap());
        hits_rare
    });
    cases("ssdv_count_reader", w, m, &|f| {
        let c = SortedSetDocValuesFacetCounts::new(&state).unwrap();
        f.result(c.get_top_children(10, "Author", &[]).unwrap());
        reader.num_docs() as u64
    });
    cases("ssdv_count_concurrent", w, m, &|f| {
        let c = ConcurrentSortedSetDocValuesFacetCounts::with_hits(&state, &fc_all).unwrap();
        f.result(c.get_top_children(10, "Author", &[]).unwrap());
        hits_all
    });
    let counts = SortedSetDocValuesFacetCounts::with_hits(&state, &fc_all).unwrap();
    cases("ssdv_top_children", w, m, &|f| {
        tops(f, &counts);
        4
    });
    let sampling = RandomSamplingFacetsCollectorManager::new(10_000, 42);
    cases("sampling_collect", w, m, &|f| {
        let mut c = sampling.collect(&searcher, &all).unwrap();
        f.add(sampled(c.matching_docs().unwrap()) as i64);
        f.add(c.sampling_rate().to_bits() as i64);
        hits_all
    });
    let sampler = RefCell::new(sampling.collect(&searcher, &all).unwrap());
    let units = sampled(sampler.borrow_mut().matching_docs().unwrap());
    cases("sampling_count_amortize", w, m, &|f| {
        let mut s = sampler.borrow_mut();
        let r = {
            let sample = s.matching_docs().unwrap();
            let c = SortedSetDocValuesFacetCounts::with_hits(&state, sample).unwrap();
            c.get_top_children(10, "Author", &[]).unwrap()
        };
        f.result(s.amortize_facet_counts(r, &config, &reader).unwrap());
        units
    });
}
