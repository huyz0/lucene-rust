//! What c15's offsets-carrying postings API buys the unified highlighter's
//! `POSTINGS` offset source: highlighting **one** document of a term whose
//! postings list is long.
//!
//! The A/B is in one build. `positions` is what `offsets_from_postings` called
//! before (c12 §3.4): the only offset-carrying accessor this port had, which
//! decodes every document's positions and offsets and builds a
//! `Vec<Position>` per document in the term. `occurrences_for_doc` is Java's
//! `PostingsOffsetStrategy` shape -- `advance(doc)`, then walk that one
//! document -- over c15's `read_occurrences_for_docs`.
//!
//! Three documents per term: the first in the postings list, one in the
//! middle, and the last. The new path's cost depends on where the document
//! sits (everything after it is never read); the old path's does not.
//!
//! Measured against `benchmarks/.corpus/merged` (`scripts/bench-corpus.sh`);
//! **the bench skips itself when that corpus is absent**.
//!
//! Run with `cargo bench -p lucene-search --bench highlight_offsets`.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use lucene_search::directory_reader::DirectoryReader;
use lucene_store::MmapDirectory;

fn corpus_dir() -> Option<String> {
    let dir = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../benchmarks/.corpus/merged"
    );
    std::path::Path::new(dir)
        .join("segments_1")
        .exists()
        .then(|| dir.to_string())
}

fn bench(c: &mut Criterion) {
    let Some(dir_path) = corpus_dir() else {
        eprintln!(
            "highlight_offsets: benchmarks/.corpus/merged is absent -- \
             run scripts/bench-corpus.sh to measure this. Skipping."
        );
        return;
    };

    let dir = MmapDirectory::open(dir_path);
    let reader = DirectoryReader::open(&dir).expect("open index");
    let opened = reader.open_segments().expect("open segments");
    let segments = opened.as_open_segments();
    let seg = &segments[0];
    let field_terms = seg.fields.field("body").expect("a body field");
    let pos_in = seg.pos_in.as_ref().expect("a .pos file");

    let mut group = c.benchmark_group("highlight_offsets");
    group.sample_size(10);

    for term in ["t0", "t500", "t999"] {
        let Some(stats) = field_terms.seek_exact(term.as_bytes()) else {
            continue;
        };
        let Some(postings) = field_terms
            .postings(term.as_bytes(), seg.doc_in)
            .expect("postings")
        else {
            continue;
        };
        let df = postings.docs.len();
        eprintln!(
            "{term}: docFreq={df} totalTermFreq={}",
            stats.total_term_freq
        );
        for (label, doc_id) in [
            ("first", postings.docs[0]),
            ("middle", postings.docs[df / 2]),
            ("last", postings.docs[df - 1]),
        ] {
            group.bench_function(format!("{term}_df{df}/{label}/whole_term"), |b| {
                b.iter(|| {
                    let per_doc = field_terms
                        .positions(term.as_bytes(), seg.doc_in, pos_in, seg.pay_in)
                        .expect("positions")
                        .expect("term present");
                    let index = postings.docs.iter().position(|&d| d == doc_id).unwrap();
                    black_box(per_doc[index].len());
                })
            });
            // c15's shape, kept as an A/B arm: `advance(doc)` over an
            // already-materialized doc list, with `.pos` addressed by a
            // running sum over *every* preceding document's frequency. This
            // is what `occurrences_for_doc` did before c20 wired up `.doc`'s
            // `.pos`/`.pay` skip pointers, and the doc-list decode plus that
            // frequency sum is the whole of the residual c15 recorded.
            group.bench_function(format!("{term}_df{df}/{label}/one_doc_freq_sum"), |b| {
                b.iter(|| {
                    let postings = field_terms
                        .postings(term.as_bytes(), seg.doc_in)
                        .expect("postings")
                        .expect("term present");
                    let index = postings.docs.binary_search(&doc_id).expect("doc present");
                    let (occurrences, _starts) = field_terms
                        .occurrences_for_docs(
                            term.as_bytes(),
                            seg.doc_in,
                            pos_in,
                            seg.pay_in,
                            &postings.freqs,
                            stats.total_term_freq,
                            &[index],
                        )
                        .expect("occurrences");
                    black_box(occurrences.len());
                })
            });
            group.bench_function(format!("{term}_df{df}/{label}/one_doc"), |b| {
                b.iter(|| {
                    let occurrences = field_terms
                        .occurrences_for_doc(
                            term.as_bytes(),
                            seg.doc_in,
                            pos_in,
                            seg.pay_in,
                            doc_id,
                        )
                        .expect("occurrences")
                        .expect("doc present");
                    black_box(occurrences.len());
                })
            });
        }
    }
    group.finish();

    // The same defect class one function over: `search_phrase_query`
    // (unscored) fetched every position of every term before intersecting the
    // doc lists, where the scored path already intersected first and asked
    // only for the intersection's positions. This measures exactly that
    // difference on a two-term phrase -- the position fetch, not the matching
    // that follows it, which is unchanged.
    let mut group = c.benchmark_group("phrase_positions");
    group.sample_size(10);
    let terms: [&[u8]; 2] = [b"t0", b"t1"];
    if terms.iter().all(|t| field_terms.seek_exact(t).is_some()) {
        group.bench_function("t0_t1/all_positions_then_intersect", |b| {
            b.iter(|| {
                let mut total = 0usize;
                for term in terms {
                    let (postings, positions, _starts) = field_terms
                        .positions_flat(term, seg.doc_in, pos_in, seg.pay_in)
                        .expect("positions")
                        .expect("term present");
                    total += postings.docs.len() + positions.len();
                }
                black_box(total);
            })
        });
        group.bench_function("t0_t1/intersect_then_positions", |b| {
            b.iter(|| {
                let mut docs: Vec<Vec<i32>> = Vec::new();
                let mut freqs: Vec<Vec<i32>> = Vec::new();
                for term in terms {
                    let postings = field_terms
                        .postings(term, seg.doc_in)
                        .expect("postings")
                        .expect("term present");
                    docs.push(postings.docs);
                    freqs.push(postings.freqs);
                }
                // The intersection, without pulling in the crate's private
                // `Conjunction`: both lists are ascending.
                let mut candidates: Vec<i32> = Vec::new();
                let (mut i, mut j) = (0usize, 0usize);
                while i < docs[0].len() && j < docs[1].len() {
                    match docs[0][i].cmp(&docs[1][j]) {
                        std::cmp::Ordering::Less => i += 1,
                        std::cmp::Ordering::Greater => j += 1,
                        std::cmp::Ordering::Equal => {
                            candidates.push(docs[0][i]);
                            i += 1;
                            j += 1;
                        }
                    }
                }
                let mut total = 0usize;
                for (t, term) in terms.iter().enumerate() {
                    let mut wanted = Vec::with_capacity(candidates.len());
                    let mut cursor = 0usize;
                    for &doc_id in &candidates {
                        while cursor < docs[t].len() && docs[t][cursor] < doc_id {
                            cursor += 1;
                        }
                        wanted.push(cursor);
                    }
                    let stats = field_terms.seek_exact(term).expect("term present");
                    let (positions, _starts) = field_terms
                        .positions_for_docs(
                            term,
                            seg.doc_in,
                            pos_in,
                            seg.pay_in,
                            &freqs[t],
                            stats.total_term_freq,
                            &wanted,
                        )
                        .expect("positions");
                    total += positions.len();
                }
                black_box(total + candidates.len());
            })
        });
    }
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
