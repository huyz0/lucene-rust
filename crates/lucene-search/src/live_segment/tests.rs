//! A [`LiveSegment`] holding a subset of terms must search like the flushed
//! segment of the same documents.

use std::collections::BTreeMap;

use lucene_codecs::field_infos::{FieldInfo, IndexOptions};
use lucene_codecs::postings_writer::TermPostings;
use lucene_index::index_writer::{
    ExplicitDocument, ExplicitFields, IndexWriter, InvertedField, InvertedTerm,
};
use lucene_index::merge_policy::upgrade::LATEST;
use lucene_store::byte_buffers_directory::ByteBuffersDirectory;

use super::{LiveField, LiveSegment};
use crate::directory_reader::DirectoryReader;
use crate::index_searcher::{IndexSearcher, SegmentNorms};
use crate::query::{BooleanQuery, Clause, PhraseQuery, TermQuery};

/// One document: per field, its tokens in order.
type Doc = Vec<(&'static str, Vec<String>)>;

fn infos() -> Vec<FieldInfo> {
    vec![
        FieldInfo::new("body", 0).with_index_options(IndexOptions::DocsAndFreqsAndPositions),
        FieldInfo::new("tag", 1).with_index_options(IndexOptions::Docs),
    ]
}

fn corpus(n: usize) -> Vec<Doc> {
    let mut x: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut next = move || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x
    };
    (0..n)
        .map(|_| {
            let len = 3 + (next() % 40) as usize;
            // A skewed vocabulary: low words are frequent, so some terms span
            // several 128-doc blocks.
            let body = (0..len)
                .map(|_| {
                    let r = next() % 1000;
                    format!("w{}", (r * r) / 20_000)
                })
                .collect();
            let tag = vec![format!("t{}", next() % 5)];
            vec![("body", body), ("tag", tag)]
        })
        .collect()
}

/// A field's norm: its token count, as a small positive byte.
fn norm(tokens: &[String]) -> i64 {
    tokens.len().min(127) as i64
}

fn inverted(number: i32, tokens: &[String], positions: bool) -> InvertedField {
    let mut by_term: BTreeMap<&str, Vec<i32>> = BTreeMap::new();
    for (p, t) in tokens.iter().enumerate() {
        by_term.entry(t).or_default().push(p as i32);
    }
    InvertedField {
        field_number: number,
        terms: by_term
            .into_iter()
            .map(|(t, ps)| InvertedTerm {
                term: t.as_bytes().to_vec(),
                freq: if positions { ps.len() as i32 } else { 1 },
                positions: if positions { ps } else { Vec::new() },
                offsets: Vec::new(),
                payloads: Vec::new(),
            })
            .collect(),
        norm: Some(norm(tokens)),
    }
}

fn flushed(docs: &[Doc]) -> (ByteBuffersDirectory, DirectoryReader) {
    let dir = ByteBuffersDirectory::new();
    {
        let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", LATEST).unwrap();
        w.set_use_compound_file(false);
        w.enable_explicit_documents().unwrap();
        for info in infos() {
            w.register_field(info).unwrap();
        }
        let explicit: Vec<ExplicitDocument> = docs
            .iter()
            .map(|d| ExplicitDocument {
                stored: Vec::new(),
                fields: ExplicitFields {
                    inverted: d
                        .iter()
                        .enumerate()
                        .map(|(i, (_, toks))| inverted(i as i32, toks, i == 0))
                        .collect(),
                    ..ExplicitFields::default()
                },
            })
            .collect();
        w.add_explicit_documents(explicit).unwrap();
        w.commit().unwrap();
    }
    let reader = DirectoryReader::open(&dir).unwrap();
    (dir, reader)
}

/// The live fields for `docs`, keeping only `wanted` terms (all for `None`).
fn live_fields(docs: &[Doc], wanted: Option<&[&str]>) -> Vec<LiveField> {
    infos()
        .into_iter()
        .enumerate()
        .map(|(fi, info)| {
            let positions = fi == 0;
            let mut terms: BTreeMap<String, TermPostings> = BTreeMap::new();
            let mut norms = vec![1i64; docs.len()];
            let (mut sum_ttf, mut sum_df, mut doc_count) = (0i64, 0i64, 0i32);
            for (d, doc) in docs.iter().enumerate() {
                let toks = &doc[fi].1;
                doc_count += 1;
                norms[d] = norm(toks);
                let mut by_term: BTreeMap<&String, Vec<i32>> = BTreeMap::new();
                for (p, t) in toks.iter().enumerate() {
                    by_term.entry(t).or_default().push(p as i32);
                }
                for (t, ps) in by_term {
                    let freq = if positions { ps.len() as i32 } else { 1 };
                    sum_ttf += i64::from(freq);
                    sum_df += 1;
                    let tp = terms.entry(t.clone()).or_insert_with(|| TermPostings {
                        term: t.as_bytes().to_vec(),
                        ..TermPostings::default()
                    });
                    tp.docs.push((d as i32, freq));
                    if positions {
                        tp.positions.push(ps);
                    }
                }
            }
            if !positions {
                // A DOCS field's sumTotalTermFreq is its sumDocFreq.
                sum_ttf = sum_df;
            }
            let num_terms = terms.len() as i64;
            let kept = terms
                .into_iter()
                .filter(|(t, _)| wanted.is_none_or(|w| w.contains(&t.as_str())))
                .map(|(_, p)| p)
                .collect();
            LiveField {
                info,
                terms: kept,
                norms: Some(norms),
                num_terms,
                sum_total_term_freq: sum_ttf,
                sum_doc_freq: sum_df,
                doc_count,
            }
        })
        .collect()
}

fn term(f: &str, t: &str) -> Clause {
    Clause::Term(TermQuery::new(f, t.as_bytes().to_vec()))
}

fn queries() -> Vec<(BooleanQuery, Vec<&'static str>)> {
    let mut q = Vec::new();
    let mut b = BooleanQuery::new();
    b.should.push(term("body", "w0"));
    q.push((b, vec!["w0"]));
    let mut b = BooleanQuery::new();
    b.should.push(term("body", "w3"));
    b.should.push(term("body", "w20"));
    b.should.push(term("body", "w41"));
    q.push((b, vec!["w3", "w20", "w41"]));
    let mut b = BooleanQuery::new();
    b.must.push(term("body", "w1"));
    b.should.push(term("body", "w7"));
    b.filter.push(term("tag", "t2"));
    q.push((b, vec!["w1", "w7", "t2"]));
    let mut b = BooleanQuery::new();
    b.must
        .push(Clause::Phrase(PhraseQuery::new("body", ["w0", "w1"])));
    q.push((b, vec!["w0", "w1"]));
    let mut b = BooleanQuery::new();
    b.should.push(term("body", "w2"));
    b.must_not.push(term("tag", "t0"));
    q.push((b, vec!["w2", "t0"]));
    q
}

fn hits(s: &IndexSearcher<'_, '_>, q: &BooleanQuery) -> (u64, Vec<(i32, f32)>) {
    let top = s.search(q, 25).unwrap();
    (
        top.total_hits.value,
        top.score_docs.iter().map(|h| (h.doc, h.score)).collect(),
    )
}

fn live_hits(seg: &LiveSegment, q: &BooleanQuery) -> (u64, Vec<(i32, f32)>) {
    let inputs = seg.inputs().unwrap();
    let segments = [seg.segment(&inputs, 0)];
    let norms = seg.norms();
    let segment_norms: [SegmentNorms<'_, '_>; 1] = [Some(&norms)];
    let s = IndexSearcher::new(&segments, &segment_norms).unwrap();
    hits(&s, q)
}

#[test]
fn a_subset_of_terms_searches_like_the_flushed_segment() {
    let docs = corpus(700);
    let (_dir, reader) = flushed(&docs);
    let mut opened = reader.open_segments().unwrap();
    opened.open_points().unwrap();
    let segments = opened.as_open_segments();
    let owned = reader
        .field_norms_by_field(&["body".to_string(), "tag".to_string()])
        .unwrap();
    let norms: Vec<SegmentNorms<'_, '_>> = owned.iter().map(Some).collect();
    let s = IndexSearcher::new(&segments, &norms).unwrap();
    for (q, wanted) in queries() {
        let expected = hits(&s, &q);
        assert!(expected.0 > 0, "{q:?} matches nothing");
        let live =
            LiveSegment::build(&live_fields(&docs, Some(&wanted)), docs.len() as i32, &[]).unwrap();
        assert_eq!(live_hits(&live, &q), expected, "{q:?}");
    }
}

#[test]
fn deleted_documents_do_not_match() {
    let docs = corpus(300);
    let all = LiveSegment::build(&live_fields(&docs, None), 300, &[]).unwrap();
    let deleted: Vec<i32> = (0..300).step_by(3).collect();
    let some = LiveSegment::build(&live_fields(&docs, None), 300, &deleted).unwrap();
    let mut q = BooleanQuery::new();
    q.should.push(term("body", "w0"));
    let (n_all, _) = live_hits(&all, &q);
    let (n_some, top) = live_hits(&some, &q);
    assert!(n_some < n_all);
    assert!(top.iter().all(|(d, _)| d % 3 != 0));
}

#[test]
fn a_field_with_no_wanted_term_still_builds() {
    let docs = corpus(50);
    let seg = LiveSegment::build(&live_fields(&docs, Some(&["absent"])), 50, &[]).unwrap();
    let mut q = BooleanQuery::new();
    q.should.push(term("body", "w0"));
    assert_eq!(live_hits(&seg, &q).0, 0);
}

#[test]
fn malformed_input_is_refused() {
    let docs = corpus(10);
    let mut fields = live_fields(&docs, None);
    fields[0].terms.reverse();
    assert!(LiveSegment::build(&fields, 10, &[]).is_err());
    let fields = live_fields(&docs, None);
    assert!(LiveSegment::build(&fields, 9, &[]).is_err());
}

/// Build cost of a search's view (`--ignored --nocapture`, release).
#[test]
#[ignore]
fn build_cost() {
    for n in [10_000usize, 100_000] {
        let docs = corpus(n);
        for wanted in [&["w0"][..], &["w40"], &["w0", "w3", "w20", "w41"]] {
            let mut fields = live_fields(&docs, Some(wanted));
            if std::env::var("NO_NORMS").is_ok() {
                for f in &mut fields {
                    f.norms = None;
                }
            }
            if std::env::var("NO_POS").is_ok() {
                for f in &mut fields {
                    if f.info.index_options == IndexOptions::DocsAndFreqsAndPositions {
                        f.info.index_options = IndexOptions::DocsAndFreqs;
                        for t in &mut f.terms {
                            t.positions.clear();
                        }
                    }
                }
            }
            if std::env::var("ONE_FIELD").is_ok() {
                fields.truncate(1);
            }
            let postings: usize = fields
                .iter()
                .flat_map(|f| &f.terms)
                .map(|t| t.docs.len())
                .sum();
            let t0 = std::time::Instant::now();
            let rounds = 20;
            for _ in 0..rounds {
                std::hint::black_box(LiveSegment::build(&fields, n as i32, &[]).unwrap());
            }
            let us = t0.elapsed().as_micros() as f64 / f64::from(rounds);
            println!("docs={n} terms={wanted:?} postings={postings} build_us={us:.0} ns_per_posting={:.1}", us * 1000.0 / postings as f64);
        }
    }
}

/// The flushed segments of `docs`, one per `split` chunk.
fn flushed_in(docs: &[Doc], split: usize) -> (ByteBuffersDirectory, DirectoryReader) {
    let dir = ByteBuffersDirectory::new();
    {
        let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", LATEST).unwrap();
        w.set_use_compound_file(false);
        w.enable_explicit_documents().unwrap();
        for info in infos() {
            w.register_field(info).unwrap();
        }
        for chunk in docs.chunks(split) {
            let explicit: Vec<ExplicitDocument> = chunk
                .iter()
                .map(|d| ExplicitDocument {
                    stored: Vec::new(),
                    fields: ExplicitFields {
                        inverted: d
                            .iter()
                            .enumerate()
                            .map(|(i, (_, toks))| inverted(i as i32, toks, i == 0))
                            .collect(),
                        ..ExplicitFields::default()
                    },
                })
                .collect();
            w.add_explicit_documents(explicit).unwrap();
            w.commit().unwrap();
        }
    }
    let reader = DirectoryReader::open(&dir).unwrap();
    (dir, reader)
}

#[test]
fn searched_beside_flushed_segments_it_scores_as_once_flushed() {
    let docs = corpus(700);
    // All 700 flushed, as two segments of 400 and 300.
    let (_all_dir, all) = flushed_in(&docs, 400);
    let mut all_opened = all.open_segments().unwrap();
    all_opened.open_points().unwrap();
    let all_segments = all_opened.as_open_segments();
    assert_eq!(all_segments.len(), 2);
    let all_owned = all
        .field_norms_by_field(&["body".to_string(), "tag".to_string()])
        .unwrap();
    let all_norms: Vec<SegmentNorms<'_, '_>> = all_owned.iter().map(Some).collect();
    let expected_searcher = IndexSearcher::new(&all_segments, &all_norms).unwrap();
    // The first 400 flushed, the last 300 in the buffer.
    let (_dir, flushed) = flushed_in(&docs[..400], 400);
    let mut opened = flushed.open_segments().unwrap();
    opened.open_points().unwrap();
    for (q, wanted) in queries() {
        let mut segments = opened.as_open_segments();
        let expected = hits(&expected_searcher, &q);
        let live = LiveSegment::build(&live_fields(&docs[400..], Some(&wanted)), 300, &[]).unwrap();
        let inputs = live.inputs().unwrap();
        segments.push(live.segment(&inputs, 400));
        // One average length over flushed and buffered documents, for every
        // segment.
        let average = |field: &str| {
            let flushed = segments[0].fields.field(field)?;
            let [_, ttf, _, dc] = live.field_stats(field)?;
            Some(crate::field_norms::avg_field_length(
                flushed.sum_total_term_freq + ttf,
                i64::from(flushed.doc_count) + dc,
            ))
        };
        let flushed_owned = flushed
            .field_norms_by_field(&["body".to_string(), "tag".to_string()])
            .unwrap();
        let flushed_norms: std::collections::HashMap<String, crate::field_norms::FieldNorms<'_>> =
            flushed_owned[0]
                .iter()
                .map(|(f, n)| {
                    (
                        f.clone(),
                        crate::field_norms::FieldNorms::with_avg_field_length(
                            n.data,
                            n.entry,
                            average(f).unwrap(),
                        ),
                    )
                })
                .collect();
        let live_norms = live.norms_with_average(&average);
        let norms: [SegmentNorms<'_, '_>; 2] = [Some(&flushed_norms), Some(&live_norms)];
        let s = IndexSearcher::new(&segments, &norms).unwrap();
        assert_eq!(hits(&s, &q), expected, "{q:?}");
    }
}

#[test]
fn a_field_without_positions_or_norms_searches_like_the_flushed_segment() {
    let docs = corpus(400);
    let (_dir, reader) = flushed(&docs);
    let mut opened = reader.open_segments().unwrap();
    opened.open_points().unwrap();
    let segments = opened.as_open_segments();
    let owned = reader
        .field_norms_by_field(&["body".to_string(), "tag".to_string()])
        .unwrap();
    let norms: Vec<SegmentNorms<'_, '_>> = owned.iter().map(Some).collect();
    let s = IndexSearcher::new(&segments, &norms).unwrap();
    // Scored clauses on the body without positions; the tag only filters,
    // so its norms are not read.
    let mut q = BooleanQuery::new();
    q.should.push(term("body", "w3"));
    q.should.push(term("body", "w9"));
    q.filter.push(term("tag", "t1"));
    let mut fields = live_fields(&docs, Some(&["w3", "w9", "t1"]));
    fields[0].info.index_options = IndexOptions::DocsAndFreqs;
    for t in &mut fields[0].terms {
        t.positions.clear();
    }
    fields[1].norms = None;
    let live = LiveSegment::build(&fields, 400, &[]).unwrap();
    assert_eq!(live_hits(&live, &q), hits(&s, &q));
}

#[test]
fn deletes_and_norms_outside_the_contract_are_refused() {
    let docs = corpus(10);
    assert!(LiveSegment::build(&live_fields(&docs, None), 10, &[10]).is_err());
    assert!(LiveSegment::build(&live_fields(&docs, None), 10, &[-1]).is_err());
    let mut fields = live_fields(&docs, None);
    fields[1].info.omit_norms = true;
    assert!(LiveSegment::build(&fields, 10, &[]).is_err());
}

/// The writer's buffer, read through a live snapshot, searches like the
/// committed segment of the same documents and deletes.
#[test]
fn the_writers_buffer_searches_like_its_committed_segment() {
    use lucene_index::buffered_updates::Term;
    use lucene_index::concurrent_writer::{ConcurrentIndexWriter, LiveFieldRequest, LiveRequest};

    let docs = corpus(700);
    let explicit = |d: &Doc| ExplicitDocument {
        stored: Vec::new(),
        fields: ExplicitFields {
            inverted: d
                .iter()
                .enumerate()
                .map(|(i, (_, toks))| inverted(i as i32, toks, i == 0))
                .collect(),
            ..ExplicitFields::default()
        },
    };
    let gone = Term::new("tag", b"t4".to_vec());
    fn open(dir: &ByteBuffersDirectory) -> IndexWriter<'_> {
        let mut w = IndexWriter::open(dir, Vec::new(), "Lucene104", LATEST).unwrap();
        w.set_use_compound_file(false);
        w.enable_explicit_documents().unwrap();
        for info in infos() {
            w.register_field(info).unwrap();
        }
        w
    }

    let committed = ByteBuffersDirectory::new();
    {
        let mut w = open(&committed);
        for d in &docs {
            w.add_explicit_documents(vec![explicit(d)]).unwrap();
        }
        w.delete_documents_by_term(std::slice::from_ref(&gone))
            .unwrap();
        w.commit().unwrap();
    }
    let reader = DirectoryReader::open(&committed).unwrap();
    let mut opened = reader.open_segments().unwrap();
    opened.open_points().unwrap();
    let segments = opened.as_open_segments();
    let owned = reader
        .field_norms_by_field(&["body".to_string(), "tag".to_string()])
        .unwrap();
    let norms: Vec<SegmentNorms<'_, '_>> = owned.iter().map(Some).collect();
    let s = IndexSearcher::new(&segments, &norms).unwrap();

    let buffered = ByteBuffersDirectory::new();
    let w = ConcurrentIndexWriter::new(open(&buffered), 1)
        .unwrap()
        .with_live_search();
    for d in &docs {
        w.add_explicit_documents(vec![explicit(d)]).unwrap();
    }
    w.delete_documents_by_term(&[gone]).unwrap();

    for (q, wanted) in queries() {
        let request = LiveRequest {
            fields: ["body", "tag"]
                .iter()
                .map(|f| LiveFieldRequest {
                    field: f.to_string(),
                    terms: wanted.iter().map(|t| t.as_bytes().to_vec()).collect(),
                    positions: true,
                })
                .collect(),
        };
        let snap = w.live_snapshot(&request).unwrap();
        assert_eq!(snap.buffers.len(), 1);
        let b = snap.buffers.into_iter().next().unwrap();
        assert!(!b.deleted.is_empty());
        let fields: Vec<LiveField> = b
            .fields
            .into_iter()
            .map(|f| LiveField {
                info: f.info,
                terms: f.terms,
                norms: f.norms,
                num_terms: f.num_terms,
                sum_total_term_freq: f.sum_total_term_freq,
                sum_doc_freq: f.sum_doc_freq,
                doc_count: f.doc_count,
            })
            .collect();
        let live = LiveSegment::build(&fields, b.max_doc, &b.deleted).unwrap();
        let expected = hits(&s, &q);
        assert!(expected.0 > 0, "{q:?} matches nothing");
        assert_eq!(live_hits(&live, &q), expected, "{q:?}");
    }
}
