//! A multi-vector field written by the index writer, read back through the
//! searcher and rescored: every encoding ranks the hits as exact `MaxSim`
//! does, and the `float` encoding is the field Lucene's own
//! `LateInteractionRescorer` reads, ranking the same.

#![allow(clippy::arithmetic_side_effects)]

use std::collections::HashMap;

use lucene_codecs::field_infos::VectorSimilarityFunction;
use lucene_index::document::{BinaryDocValuesField, Document, Store, StringField};
use lucene_index::index_writer::IndexWriter;
use lucene_index::segment_info::LuceneVersion;
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::field_norms::FieldNorms;
use lucene_search::index_searcher::IndexSearcher;
use lucene_search::rescorer::{late_interaction_rescorer, Rescorer};
use lucene_search::top_docs::{ShardScoreDoc, TopDocs};
use lucene_search::values_source::ValuesContext;
use lucene_search::{TotalHits, TotalHitsRelation};
use lucene_store::ByteBuffersDirectory;
use lucene_tensor::{encode, max_sim_rescorer, MaxSimQuery, Missing, TokenEncoding};

const DIM: usize = 8;

fn unit(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    let mut v: Vec<f32> = (0..n * DIM)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32) - 0.5
        })
        .collect();
    for t in v.chunks_exact_mut(DIM) {
        let norm = t.iter().map(|x| x * x).sum::<f32>().sqrt();
        t.iter_mut().for_each(|x| *x /= norm);
    }
    v
}

fn exact(q: &[f32], d: &[f32]) -> f32 {
    q.chunks(DIM)
        .map(|qt| {
            d.chunks(DIM)
                .map(|dt| qt.iter().zip(dt).map(|(a, b)| a * b).sum::<f32>())
                .fold(f32::NEG_INFINITY, f32::max)
        })
        .sum()
}

#[test]
fn every_encoding_reranks_hits_by_max_sim() {
    let dir = ByteBuffersDirectory::new();
    let mut w = IndexWriter::open(
        &dir,
        Vec::new(),
        "Lucene104",
        LuceneVersion {
            major: 10,
            minor: 5,
            bugfix: 0,
        },
    )
    .unwrap();
    let docs: Vec<Vec<f32>> = (0..60)
        .map(|i| unit(i + 100, 1 + (i as usize * 7) % 40))
        .collect();
    for (i, d) in docs.iter().enumerate() {
        let mut doc = Document::new();
        doc.add(StringField::new("id", i.to_string(), Store::Yes));
        // every fifth document has no multi-vector
        if i % 5 != 4 {
            for enc in [
                TokenEncoding::Float32,
                TokenEncoding::Float16,
                TokenEncoding::Int8,
            ] {
                doc.add(BinaryDocValuesField::new(
                    enc.name(),
                    encode(d, DIM, enc).unwrap(),
                ));
            }
        }
        w.add_fields_document(&doc).unwrap();
        if i == 29 {
            w.commit().unwrap();
        }
    }
    w.commit().unwrap();
    w.close().unwrap();

    let reader = DirectoryReader::open(&dir).unwrap();
    let opened = reader.open_segments().unwrap();
    let segments = opened.as_open_segments();
    assert_eq!(segments.len(), 2);
    let norms: Vec<Option<&HashMap<String, FieldNorms<'_>>>> = vec![None; segments.len()];
    let searcher = IndexSearcher::new(&segments, &norms).unwrap();
    let ctx = ValuesContext::new(&searcher);

    let q = unit(7, 5);
    let rows: Vec<Vec<f32>> = q.chunks(DIM).map(<[f32]>::to_vec).collect();
    let first = TopDocs {
        total_hits: TotalHits {
            value: 60,
            relation: TotalHitsRelation::EqualTo,
        },
        score_docs: (0..60)
            .map(|d| ShardScoreDoc::new(d, 1.0 + d as f32))
            .collect(),
    };
    let mut want: Vec<(i32, f32)> = (0..60)
        .filter(|i| i % 5 != 4)
        .map(|i| (i, exact(&q, &docs[i as usize])))
        .collect();
    want.sort_by(|a, b| b.1.total_cmp(&a.1));
    let want_ids: Vec<i32> = want.iter().take(10).map(|x| x.0).collect();

    for enc in [
        TokenEncoding::Float32,
        TokenEncoding::Float16,
        TokenEncoding::Int8,
    ] {
        let r = max_sim_rescorer(enc.name(), MaxSimQuery::new(&rows).unwrap(), Missing::Zero);
        let got = r.rescore(&ctx, &first, 10).unwrap();
        let ids: Vec<i32> = got.score_docs.iter().map(|h| h.doc).collect();
        let exact_of: HashMap<i32, f32> = want.iter().copied().collect();
        for h in &got.score_docs {
            let e = exact_of[&h.doc];
            assert!(
                (h.score - e).abs() < 0.03,
                "{enc:?} doc {} {} {e}",
                h.doc,
                h.score
            );
        }
        if enc == TokenEncoding::Int8 {
            // lossy: a near tie at the cut may swap, nothing more
            let kept = ids.iter().filter(|d| want_ids.contains(d)).count();
            assert!(kept >= 9, "{enc:?} {ids:?} {want_ids:?}");
        } else {
            assert_eq!(ids, want_ids, "{enc:?}");
        }
        // a hit without a value keeps its first-pass score under FirstPassScore
        let r = max_sim_rescorer(
            enc.name(),
            MaxSimQuery::new(&rows).unwrap(),
            Missing::FirstPassScore,
        );
        let all = r.rescore(&ctx, &first, 60).unwrap();
        let d4 = all.score_docs.iter().find(|h| h.doc == 4).unwrap();
        assert_eq!(d4.score, 5.0);
    }

    // Lucene's rescorer over the same `float` field: DOT_PRODUCT scores each
    // token (1 + dot) / 2, a monotone map, so the order is the same.
    let lucene =
        late_interaction_rescorer("float", rows, VectorSimilarityFunction::DotProduct).unwrap();
    let got = lucene.rescore(&ctx, &first, 10).unwrap();
    let ids: Vec<i32> = got.score_docs.iter().map(|h| h.doc).collect();
    assert_eq!(ids, want_ids);
}

#[test]
fn a_field_without_values_or_with_bad_ones_is_handled() {
    let dir = ByteBuffersDirectory::new();
    let mut w = IndexWriter::open(
        &dir,
        Vec::new(),
        "Lucene104",
        LuceneVersion {
            major: 10,
            minor: 5,
            bugfix: 0,
        },
    )
    .unwrap();
    for i in 0..3 {
        let mut doc = Document::new();
        doc.add(StringField::new("id", i.to_string(), Store::Yes));
        doc.add(BinaryDocValuesField::new("bad", vec![1, 2, 3]));
        w.add_fields_document(&doc).unwrap();
    }
    w.commit().unwrap();
    w.close().unwrap();
    let reader = DirectoryReader::open(&dir).unwrap();
    let opened = reader.open_segments().unwrap();
    let segments = opened.as_open_segments();
    let norms: Vec<Option<&HashMap<String, FieldNorms<'_>>>> = vec![None; segments.len()];
    let searcher = IndexSearcher::new(&segments, &norms).unwrap();
    let ctx = ValuesContext::new(&searcher);
    let first = TopDocs {
        total_hits: TotalHits {
            value: 3,
            relation: TotalHitsRelation::EqualTo,
        },
        score_docs: (0..3).map(|d| ShardScoreDoc::new(d, 2.0)).collect(),
    };
    let q = MaxSimQuery::new(&[vec![1.0, 0.0]]).unwrap();
    // no such field: every hit is missing
    let r = max_sim_rescorer("absent", q.clone(), Missing::Zero);
    let got = r.rescore(&ctx, &first, 3).unwrap();
    assert!(got.score_docs.iter().all(|h| h.score == 0.0));
    let r = max_sim_rescorer("id", q.clone(), Missing::FirstPassScore);
    let got = r.rescore(&ctx, &first, 3).unwrap();
    assert!(got.score_docs.iter().all(|h| h.score == 2.0));
    // a value that is not a multi-vector is an error, not a score
    let r = max_sim_rescorer("bad", q.clone(), Missing::Zero);
    let err = r.rescore(&ctx, &first, 3).unwrap_err();
    assert!(err.to_string().contains("not a multi-vector"), "{err}");

    use lucene_search::values_source::DoubleValuesSource;
    let source = lucene_tensor::MaxSimValuesSource::new("bad", q);
    assert_eq!(
        source.describe(),
        "MaxSimValuesSource(field=bad queryTokens=1 dim=2)"
    );
    assert!(source.equality_key().starts_with(&source.describe()));
    assert!(!source.needs_scores());
    assert!(source.is_cacheable(&ctx, 0));
}
