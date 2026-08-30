//! **The impacts this port writes must never bound a document below its real
//! score.** An impact that is too *high* costs pruning; one that is too *low*
//! makes MAXSCORE skip a block containing a document that belonged in the
//! top-`n` -- a missing hit, silently, with no error anywhere.
//!
//! Ledger item 18 was "impacts are computed against norm 1": every level-0 and
//! level-1 impact was `(maxFreq, 1)`, which is sound (norm 1 is the shortest
//! field and so the highest-scoring one) but far too loose to prune with. With
//! real norms the bound gets tight, and tightness is exactly where soundness
//! can be lost -- so this file checks the property that matters directly,
//! against postings **this port wrote**, read back through the real decoder:
//!
//! 1. `bounds_never_fall_below_a_real_document_score` -- for every level-0
//!    block and every level-1 span, `max_score_for_impacts` is `>=` the BM25
//!    score of every document that block/span covers. This is the invariant a
//!    skipped document violates, checked at its source.
//! 2. `pruned_top_n_equals_brute_force_top_n` -- the end-to-end consequence:
//!    a MAXSCORE-shaped block skip driven by those bounds returns the same
//!    top-`n` as scoring every document.
//! 3. `real_norms_make_the_bound_tighter_than_norm_one` -- the win, measured:
//!    the same postings written without norms give a strictly higher (looser)
//!    bound for a block of long documents.
//!
//! Verified to fail against the unfixed code by making `competitive_impacts`
//! emit `(maxFreq, maxNorm)` -- the plausible-looking shortcut that is *not*
//! the Pareto frontier -- which trips (1) and (2) at once.
// Test-support code opts out of the arithmetic gate at the file boundary, as
// the other fixture builders do. See `docs/arithmetic-gate.md`.
#![allow(clippy::arithmetic_side_effects)]

use lucene_codecs::blocktree;
use lucene_codecs::field_infos::{
    DocValuesSkipIndexType, DocValuesType, FieldInfo, FieldInfos, IndexOptions, VectorEncoding,
    VectorSimilarityFunction,
};
use lucene_codecs::postings::{DocInput, Postings};
use lucene_codecs::postings_writer::{
    write_single_field, write_single_field_with_norms, FieldNorms, FieldPostingsInput, TermPostings,
};
use lucene_search::collector::{ScoringCollector, TopDocsCollector};
use lucene_search::similarity::{decode_norm, max_score_for_impacts, score};
use lucene_store::codec_util::ID_LENGTH;

const SEG_ID: [u8; ID_LENGTH] = [77u8; ID_LENGTH];
const SUFFIX: &str = "";
const TERM: &[u8] = b"alpha";

/// Enough documents for two full level-1 spans plus a partial one, so both
/// impact levels and the trailing full blocks are all written.
const NUM_DOCS: i32 = 8192 + 2048;
/// The *collection*'s document count, deliberately much larger than the
/// term's `docFreq`, so `idf` is a real number rather than the ~0 a term
/// present in every document produces.
const DOC_COUNT: i64 = 1_000_000;
const AVG_FIELD_LENGTH: f32 = 40.0;

fn field_info() -> FieldInfo {
    FieldInfo {
        name: "body".to_string(),
        number: 0,
        store_term_vectors: false,
        omit_norms: false,
        store_payloads: false,
        soft_deletes_field: false,
        parent_field: false,
        index_options: IndexOptions::DocsAndFreqs,
        doc_values_type: DocValuesType::None,
        doc_values_skip_index_type: DocValuesSkipIndexType::None,
        doc_values_gen: -1,
        attributes: Vec::new(),
        point_dimension_count: 0,
        point_index_dimension_count: 0,
        point_num_bytes: 0,
        vector_dimension: 0,
        vector_encoding: VectorEncoding::Float32,
        vector_similarity_function: VectorSimilarityFunction::Euclidean,
    }
}

/// One term over every document, with freq and norm deliberately
/// **anti-correlated within a block**: the highest-frequency document of a
/// block is not the shortest one.
///
/// That is the shape the naive `(maxFreq, minNorm)` bound gets right (it is
/// an over-estimate) and the equally naive `(maxFreq, maxNorm)` shortcut gets
/// *wrong*, and it is why the accumulator has to compute a frontier rather
/// than a corner: a real block's competitive set is several `(freq, norm)`
/// pairs, none of which dominates the others.
fn docs_and_norms() -> (Vec<(i32, i32)>, Vec<i64>) {
    let mut docs = Vec::with_capacity(NUM_DOCS as usize);
    let mut norms = Vec::with_capacity(NUM_DOCS as usize);
    for doc in 0..NUM_DOCS {
        let block = doc / 256;
        let within = doc % 256;
        // The first four blocks hold the competitive documents -- the term is
        // frequent there and the fields are short. Everything after is a
        // long-field, low-frequency tail, which is what a top-`n` collector
        // gets to prune once its queue is full.
        let freq = if block < 4 {
            8 + (within % 5)
        } else {
            1 + (within % 3)
        };
        // The *encoded* norm byte a real `.nvd` stores, sign-extended the way
        // `norms::norm_value` hands it back. Never 0: `Lucene104PostingsWriter`
        // asserts `norm != 0` for a document that has a posting, and this port
        // only writes 0 for a document that carries the field with no tokens
        // -- which by construction has no posting to bound.
        //
        // The tail's bytes run past 127, so they sign-extend to *negative*
        // `i64`s and the accumulator's unsigned norm comparison is exercised
        // rather than assumed. Within a block the byte still varies, so each
        // block's competitive set is a real frontier of several pairs and not
        // a single corner.
        let norm_byte: u32 = if block < 4 {
            1 + (within as u32 % 4)
        } else {
            60 + ((block as u32 * 4) % 190) + (within as u32 % 3) * 2
        };
        docs.push((doc, freq));
        norms.push(i64::from(norm_byte as u8 as i8));
    }
    (docs, norms)
}

/// Writes the term and reads its [`Postings`] (impacts included) back through
/// the real decoder. `with_norms` selects the arm.
fn written_postings(with_norms: bool) -> (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>) {
    let (docs, norms) = docs_and_norms();
    let terms = vec![TermPostings {
        term: TERM.to_vec(),
        docs,
        ..Default::default()
    }];
    let input = FieldPostingsInput {
        field_number: 0,
        index_options: IndexOptions::DocsAndFreqs,
        doc_count: NUM_DOCS,
        has_payloads: false,
        terms: &terms,
    };
    let out = if with_norms {
        let field_norms = [FieldNorms {
            field_number: 0,
            values: &norms,
        }];
        write_single_field_with_norms(&input, &field_norms, &SEG_ID, SUFFIX).expect("write")
    } else {
        write_single_field(&input, &SEG_ID, SUFFIX).expect("write")
    };
    (out.tim, out.tip, out.tmd, out.doc)
}

/// Opens `written_postings`' bytes and returns the term's decoded postings.
fn postings_of(tim: &[u8], tip: &[u8], tmd: &[u8], doc: &[u8]) -> Postings {
    let field_infos = FieldInfos {
        fields: vec![field_info()],
    };
    let fields = blocktree::open(tim, tip, tmd, &field_infos, &SEG_ID, SUFFIX, NUM_DOCS)
        .expect("open terms");
    let doc_in = DocInput::open(doc, &SEG_ID, SUFFIX).expect("open .doc");
    fields
        .field("body")
        .expect("field")
        .postings(TERM, Some(&doc_in))
        .expect("postings")
        .expect("term present")
}

/// This port's BM25 score for one document of the term under test.
fn doc_score(freq: i32, norm: i64, doc_freq: i64) -> f32 {
    score(
        doc_freq,
        DOC_COUNT,
        freq as f32,
        decode_norm(norm),
        AVG_FIELD_LENGTH,
    )
}

/// The invariant a skipped document violates: **no block's bound may fall
/// below the score of any document that block covers.**
///
/// Checked for level 0 and level 1 alike -- a level-1 span's impacts have to
/// bound every one of its 32 blocks, and getting that wrong skips 8 192
/// documents at a time rather than 256.
#[test]
fn bounds_never_fall_below_a_real_document_score() {
    let (tim, tip, tmd, doc) = written_postings(true);
    let postings = postings_of(&tim, &tip, &tmd, &doc);
    let (_, norms) = docs_and_norms();
    let doc_freq = postings.docs.len() as i64;

    assert!(
        !postings.level0_impacts.is_empty(),
        "the fixture must produce level-0 impacts, or this proves nothing"
    );
    assert!(
        !postings.level1_impacts.is_empty(),
        "the fixture must produce level-1 impacts, or the level-1 half proves nothing"
    );

    for (label, column) in [
        ("level 0", &postings.level0_impacts),
        ("level 1", &postings.level1_impacts),
    ] {
        let mut covered_from = 0i32;
        for (last_doc, impacts) in column {
            assert!(
                !impacts.is_empty(),
                "{label}: an empty impacts list is rejected outright by real Lucene"
            );
            let bound = max_score_for_impacts(impacts, doc_freq, DOC_COUNT, AVG_FIELD_LENGTH);
            for (i, &d) in postings.docs.iter().enumerate() {
                if d < covered_from || d > *last_doc {
                    continue;
                }
                let actual = doc_score(postings.freqs[i], norms[d as usize], doc_freq);
                assert!(
                    actual <= bound,
                    "{label}: doc {d} scores {actual}, above its block's bound {bound} \
                     -- MAXSCORE would skip it"
                );
            }
            covered_from = *last_doc + 1;
        }
    }
}

/// The end-to-end consequence of the invariant above: a MAXSCORE-shaped block
/// skip driven by the written impacts must return the same top-`n` as scoring
/// every document.
///
/// The skip is written out here rather than borrowed from `lucene-search`'s
/// own `#[cfg(test)]` helper (which an integration test cannot see), which is
/// the point: the two are independent implementations of the same rule.
#[test]
fn pruned_top_n_equals_brute_force_top_n() {
    let (tim, tip, tmd, doc) = written_postings(true);
    let postings = postings_of(&tim, &tip, &tmd, &doc);
    let (_, norms) = docs_and_norms();
    let doc_freq = postings.docs.len() as i64;

    for top_n in [1usize, 10, 100] {
        let mut brute = TopDocsCollector::new(top_n);
        for (i, &d) in postings.docs.iter().enumerate() {
            brute.collect(d, doc_score(postings.freqs[i], norms[d as usize], doc_freq));
        }

        let mut pruned = TopDocsCollector::new(top_n);
        let mut skipped_blocks = 0usize;
        let mut i = 0usize;
        while i < postings.docs.len() {
            let d = postings.docs[i];
            let block = postings
                .level0_impacts
                .iter()
                .find(|(last, _)| d <= *last)
                .map(|(last, impacts)| (*last, impacts.as_slice()));
            if let Some((last, impacts)) = block {
                if pruned.top_docs().len() >= top_n {
                    let bound =
                        max_score_for_impacts(impacts, doc_freq, DOC_COUNT, AVG_FIELD_LENGTH);
                    let worst = pruned.top_docs().last().map(|h| h.score);
                    if worst.is_some_and(|w| bound <= w) {
                        skipped_blocks += 1;
                        while i < postings.docs.len() && postings.docs[i] <= last {
                            i += 1;
                        }
                        continue;
                    }
                }
            }
            pruned.collect(d, doc_score(postings.freqs[i], norms[d as usize], doc_freq));
            i += 1;
        }

        assert_eq!(
            brute.top_docs(),
            pruned.top_docs(),
            "top-{top_n}: pruning by the written impacts changed the result"
        );
        assert!(
            skipped_blocks > 0,
            "top-{top_n}: no block was ever skipped, so this proves nothing about pruning"
        );
    }
}

/// The win, measured: with real norms the bound is *strictly lower* -- and
/// therefore prunes more -- than the `(maxFreq, 1)` bound a normless writer
/// can prove, for every block whose documents are longer than the shortest
/// possible field.
#[test]
fn real_norms_make_the_bound_tighter_than_norm_one() {
    let (tim, tip, tmd, doc) = written_postings(true);
    let with = postings_of(&tim, &tip, &tmd, &doc);
    let (tim, tip, tmd, doc) = written_postings(false);
    let without = postings_of(&tim, &tip, &tmd, &doc);

    assert_eq!(
        with.docs, without.docs,
        "the two arms must differ only in their impacts"
    );
    assert_eq!(with.freqs, without.freqs);
    assert_eq!(with.level0_impacts.len(), without.level0_impacts.len());

    let doc_freq = with.docs.len() as i64;
    let mut tighter = 0usize;
    for ((last_a, a), (last_b, b)) in with.level0_impacts.iter().zip(&without.level0_impacts) {
        assert_eq!(last_a, last_b);
        assert_eq!(
            b.as_slice().len(),
            1,
            "the normless arm still writes exactly one (maxFreq, 1) impact"
        );
        assert_eq!(b[0].norm, 1);
        let bound_with = max_score_for_impacts(a, doc_freq, DOC_COUNT, AVG_FIELD_LENGTH);
        let bound_without = max_score_for_impacts(b, doc_freq, DOC_COUNT, AVG_FIELD_LENGTH);
        assert!(
            bound_with <= bound_without,
            "real norms produced a *higher* bound ({bound_with} > {bound_without}) -- \
             that would be a loosening, not a tightening"
        );
        if bound_with < bound_without {
            tighter += 1;
        }
    }
    assert!(
        tighter > 0,
        "no block's bound got tighter, so the norms are not reaching the impacts"
    );
}
