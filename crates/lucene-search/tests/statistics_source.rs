#![allow(clippy::arithmetic_side_effects)]
//! **`IndexSearcher.termStatistics`/`collectionStatistics` overridden.**
//!
//! Java's `ShardSearchingTestBase.ShardIndexSearcher` scores one node's
//! reader under statistics summed over every node, so its scores are those
//! of one searcher over all of them. Here each segment of
//! `fixtures/src/GenMinScore.java`'s three-segment index is a "node": a
//! searcher over that segment alone, with a [`StatisticsSource`] answering
//! with the whole reader's statistics, must score every hit exactly as the
//! searcher over all three segments does (scores bit for bit, under BM25 and
//! a non-default similarity), and the plain per-segment searcher must not.

mod m7support;

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};

use lucene_search::directory_reader::DirectoryReader;
use lucene_search::field_norms::FieldNorms;
use lucene_search::index_searcher::{IndexSearcher, StatisticsSource};
use lucene_search::multi_segment::OpenSegment;
use lucene_search::similarities::{
    ClassicSimilarity, CollectionStatistics, Similarity, TermStatistics,
};
use lucene_search::top_field::SortField;
use lucene_search::Result;
use lucene_store::FsDirectory;
use m7support::{fixture, Grammar};

/// Every term's and field's statistics as the whole reader has them.
struct Global {
    terms: HashMap<(String, Vec<u8>), TermStatistics>,
    fields: HashMap<String, CollectionStatistics>,
    asked: AtomicUsize,
}

impl StatisticsSource for Global {
    fn term_statistics(
        &self,
        field: &str,
        term: &[u8],
        local: TermStatistics,
    ) -> Result<TermStatistics> {
        self.asked.fetch_add(1, Ordering::Relaxed);
        assert!(local.doc_freq > 0, "asked about a term the shard lacks");
        Ok(self.terms[&(field.to_string(), term.to_vec())].clone())
    }

    fn collection_statistics(
        &self,
        field: &str,
        local: Option<CollectionStatistics>,
    ) -> Result<Option<CollectionStatistics>> {
        assert!(local.is_some());
        Ok(self.fields.get(field).copied())
    }
}

const QUERIES: [&str; 6] = [
    "(t w0)",
    "(t w30)",
    "(b 0 (? (t w3)) (? (t w4)) (? (t w5)))",
    "(b 2 (? (t w1)) (? (t w6)) (? (t w7)))",
    "(b 0 (+ (t w1)) (- (t w2)))",
    "(b 0 (+ (t w0)) (? (p w1 w2)))",
];

#[test]
fn a_shard_under_global_statistics_scores_as_the_whole_reader() {
    let dir = fixture("min_score_index");
    let reader = DirectoryReader::open(&FsDirectory::open(&dir)).expect("open reader");
    let opened = reader.open_segments().expect("open postings");
    let segments = opened.as_open_segments();
    let owned: Vec<HashMap<String, FieldNorms<'_>>> = reader
        .field_norms("body")
        .unwrap()
        .into_iter()
        .map(|n| n.into_iter().map(|n| ("body".to_string(), n)).collect())
        .collect();
    let norms: Vec<Option<&HashMap<String, FieldNorms<'_>>>> = owned.iter().map(Some).collect();
    let full = IndexSearcher::new(&segments, &norms).unwrap();
    assert!(full.statistics().is_none());

    let mut global = Global {
        terms: HashMap::new(),
        fields: HashMap::new(),
        asked: AtomicUsize::new(0),
    };
    let body = full.collection_statistics("body").unwrap().unwrap();
    assert_eq!(body.max_doc, 9000);
    assert!(full.collection_statistics("nosuchfield").unwrap().is_none());
    global.fields.insert("body".into(), body);
    for i in 0..40 {
        let t = format!("w{i}");
        if let Some(s) = full.term_statistics("body", t.as_bytes()).unwrap() {
            global.terms.insert(("body".into(), t.into_bytes()), s);
        }
    }
    assert!(full
        .term_statistics("body", b"nosuchterm")
        .unwrap()
        .is_none());

    let grammar = Grammar {
        text: "body",
        range: "r",
    };
    let classic = ClassicSimilarity::default();
    let sims: [Option<&dyn Similarity>; 2] = [None, Some(&classic)];
    let mut differed = false;
    for sim in sims {
        let mut full = IndexSearcher::new(&segments, &norms).unwrap();
        if let Some(sim) = sim {
            full.set_similarity(sim);
        }
        for (s, seg) in segments.iter().enumerate() {
            let base = seg.doc_base;
            let alone = [OpenSegment {
                doc_base: 0,
                ..*seg
            }];
            let alone_norms = [norms[s]];
            let mut shard = IndexSearcher::new(&alone, &alone_norms).unwrap();
            if let Some(sim) = sim {
                shard.set_similarity(sim);
            }
            let plain_body = shard.collection_statistics("body").unwrap().unwrap();
            assert!(plain_body.doc_count < body.doc_count);
            let mut plain = IndexSearcher::new(&alone, &alone_norms).unwrap();
            if let Some(sim) = sim {
                plain.set_similarity(sim);
            }
            shard.set_statistics(Some(&global));
            assert_eq!(shard.collection_statistics("body").unwrap(), Some(body));
            assert_eq!(
                shard.term_statistics("body", b"w0").unwrap(),
                full.term_statistics("body", b"w0").unwrap()
            );
            for text in QUERIES {
                let q = grammar.query(text);
                let n = seg.max_doc.unwrap() as usize;
                let want: Vec<(i32, u32)> = full
                    .search(&q, 9000)
                    .unwrap()
                    .score_docs
                    .iter()
                    .filter(|h| h.doc >= base && h.doc - base < n as i32)
                    .map(|h| (h.doc - base, h.score.to_bits()))
                    .collect();
                assert!(!want.is_empty(), "{text}: no hits in segment {s}");
                let got: Vec<(i32, u32)> = shard
                    .search(&q, n)
                    .unwrap()
                    .score_docs
                    .iter()
                    .map(|h| (h.doc, h.score.to_bits()))
                    .collect();
                assert_eq!(got, want, "{text}: segment {s}");
                let local: Vec<(i32, u32)> = plain
                    .search(&q, n)
                    .unwrap()
                    .score_docs
                    .iter()
                    .map(|h| (h.doc, h.score.to_bits()))
                    .collect();
                differed |= local != want;

                // a score-sorted search reads the same statistics
                let readers = &reader.segment_readers()[s..=s];
                let sorted = shard
                    .search_sorted(readers, &q, 5, &[SortField::score()], None)
                    .unwrap();
                let top: Vec<i32> = sorted.hits.iter().map(|h| h.doc).collect();
                let first: Vec<i32> = want.iter().take(5).map(|h| h.0).collect();
                assert_eq!(top, first, "{text}: sorted {s}");

                // and explain
                let (doc, score) = want[0];
                let e = shard.explain(&q, doc).unwrap();
                assert_eq!(e.value.to_bits(), score, "{text}: explain {s}\n{e}");
            }
        }
    }
    assert!(differed, "the shard's own statistics never changed a score");
    assert!(global.asked.load(Ordering::Relaxed) > 0);
}
