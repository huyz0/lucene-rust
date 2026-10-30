//! SimpleTextPhraseMicro's twin, measuring the native phrase scorer before
//! general codec routing is enabled. Only one segment and ClassicSimilarity:
//! its statistics are local and its scoring goes through the scorer tree.
use std::hint::black_box;
use std::time::Duration;

use lucene_codecs::blocktree::BlockTreeFields;
use lucene_search::collector::{ScoringCollector, TopDocsCollector};
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::index_searcher::IndexSearcher;
use lucene_search::multi_segment::OpenSegment;
use lucene_search::query::{BooleanQuery, Clause, PhraseQuery};
use lucene_search::similarities::ClassicSimilarity;
use lucene_store::FsDirectory;

const INITIAL_HASH: u64 = 0xcbf2_9ce4_8422_2325;

fn add(hash: u64, value: u64) -> u64 {
    (hash ^ value).wrapping_mul(0x0000_0100_0000_01b3)
}

struct Digest {
    hash: u64,
    hits: u64,
}

impl ScoringCollector for Digest {
    fn collect(&mut self, doc: i32, score: f32) {
        self.hash = add(self.hash, doc as u64);
        self.hash = add(self.hash, u64::from(score.to_bits()));
        self.hits += 1;
    }
}

fn run(searcher: &IndexSearcher<'_, '_>, query: &BooleanQuery, top: bool) -> u64 {
    if !top {
        let mut collector = Digest {
            hash: INITIAL_HASH,
            hits: 0,
        };
        searcher.search_collector(query, &mut collector).unwrap();
        return add(collector.hash, collector.hits);
    }
    let mut collector = TopDocsCollector::with_total_hits_threshold(10, 0);
    searcher.search_collector(query, &mut collector).unwrap();
    let mut hash = INITIAL_HASH;
    for hit in collector.top_docs() {
        hash = add(hash, hit.doc_id as u64);
        hash = add(hash, u64::from(hit.score.to_bits()));
    }
    add(hash, collector.top_docs().len() as u64)
}

pub fn bench(warmup: Duration, measure: Duration, root: &str) {
    let reader = DirectoryReader::open(&FsDirectory::open(root)).unwrap();
    assert_eq!(reader.segment_readers().len(), 1, "one segment required");
    let segment = &reader.segment_readers()[0];
    let empty = BlockTreeFields::empty();
    // This bounded harness supplies the native reader to the production
    // scorer tree directly. DirectoryReader's guard remains until every
    // query family and reader-wide statistics supports text formats.
    let segments = [OpenSegment {
        fields: &empty,
        doc_in: None,
        pos_in: None,
        pay_in: None,
        live_docs: segment.live_docs(),
        doc_base: 0,
        max_doc: Some(segment.max_doc),
        cache: None,
        points: None,
        reader: Some(segment),
        index_sort_prefix: false,
        similarity: None,
    }];
    let norms = reader.field_norms_by_field(&["body".into()]).unwrap();
    let norm_refs = [Some(&norms[0])];
    let similarity = ClassicSimilarity::default();
    let mut searcher = IndexSearcher::new(&segments, &norm_refs).unwrap();
    searcher.set_similarity(&similarity);
    for (name, slop, words) in [
        ("exact", 0, ["alpha", "beta"]),
        ("sloppy", 2, ["beta", "alpha"]),
        ("repeat", 3, ["alpha", "alpha"]),
    ] {
        let query = BooleanQuery {
            must: vec![Clause::Phrase(
                PhraseQuery::new("body", words.into_iter().map(|s| s.as_bytes().to_vec()))
                    .with_slop(slop),
            )],
            ..Default::default()
        };
        for top in [false, true] {
            let name = format!("cx_st_phrase_{name}_{}", if top { "top" } else { "all" });
            if std::env::var("MICRO_CASE").is_ok_and(|only| only != name) {
                continue;
            }
            println!("#check\t{name}\t{:016x}\t1", run(&searcher, &query, top));
            super::measure(&name, warmup, measure, || {
                black_box(run(&searcher, black_box(&query), top));
                1
            });
        }
    }
}
