//! Differential test against real `Lucene104PostingsWriter` bytes for a
//! positions-indexing term long enough to carry **`.pos`/`.pay` skip data**.
//!
//! `blocktree_index`'s positions field ("pos") has `docFreq = 3` /
//! `totalTermFreq = 4`: every occurrence lives in the vint tail, no `.doc`
//! full block exists, and so not one byte of the level-0/level-1 pos/pay skip
//! records is present in it. `postings_skip_index` (`GenPostingsSkip.java`)
//! is a term in all 8 500 documents with 25 500 occurrences, offsets and
//! payloads -- one level-1 entry, 33 level-0 block headers, and a
//! group-varint tail, each skip record carrying the `.pos`/`.pay` pointer its
//! documents' occurrences begin at.
//!
//! Ground truth is Java's own `PostingsEnum.advance(doc)` +
//! `nextPosition()`/`startOffset()`/`endOffset()`/`getPayload()`, taken with
//! a **fresh enum per document** so that every sampled document is reached
//! through the skip data rather than by sequential iteration -- which is
//! exactly the shape `postings::read_occurrences_for_doc` implements.
//!
//! Two properties of the fixture are load-bearing enough to be asserted
//! rather than assumed, because without them a reader that ignored the skip
//! data entirely would still pass every check here:
//!
//! - the level-1 entry's `posBufferUpto` must be **non-zero**. Per-document
//!   frequencies cycle on a period coprime with 256 to make it so; with the
//!   period-4 cycle this fixture was first generated with, the
//!   8 192-document level-1 boundary landed exactly on a `.pos` block
//!   boundary and every level-1 `posBufferUpto` was `0`.
//! - a second, sparser term (`gapterm`, in 40% of the documents) must be
//!   present, because the dense term is in *every* document and so takes
//!   Lucene's degenerate `docRange == BLOCK_SIZE` doc-delta encoding in every
//!   one of its blocks. The sparse term is what covers a skip-driven
//!   `advance` into a packed-FOR or bit-set block, and an `advance` whose
//!   target the term does not contain.
//!
//! Regenerate with `fixtures/src/GenPostingsSkip.java`.
// Test-support code opts out of the arithmetic gate at the file boundary:
// the gate exists for values read off disk in production decode paths, not
// for a test's own index arithmetic. See `docs/arithmetic-gate.md`.
#![allow(clippy::arithmetic_side_effects)]

use lucene_codecs::blocktree;
use lucene_codecs::field_infos;
use lucene_codecs::postings::{self, Position};

fn dir() -> String {
    concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/data/postings_skip_index/"
    )
    .to_string()
}

struct Manifest {
    kv: Vec<(String, String)>,
}

impl Manifest {
    fn load() -> Self {
        let text = std::fs::read_to_string(format!("{}manifest.properties", dir()))
            .expect("run fixtures generator first (GenPostingsSkip)");
        let kv = text
            .lines()
            .filter_map(|l| l.split_once('='))
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        Manifest { kv }
    }

    fn get(&self, key: &str) -> &str {
        self.kv
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
            .unwrap_or_else(|| panic!("manifest key {key} missing"))
    }
}

fn id_from_hex(hex: &str) -> [u8; 16] {
    let mut id = [0u8; 16];
    for (i, slot) in id.iter_mut().enumerate() {
        *slot = u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16).unwrap();
    }
    id
}

fn read_raw(name: &str) -> Vec<u8> {
    std::fs::read(format!("{}{}.raw", dir(), name)).unwrap_or_else(|_| panic!("missing {name}.raw"))
}

/// One `pos,startOffset,endOffset,payloadHex|NONE` triple-plus-payload as the
/// manifest writes it.
fn parse_occurrences(spec: &str) -> Vec<Position> {
    if spec.is_empty() {
        return Vec::new();
    }
    spec.split(';')
        .map(|occ| {
            let parts: Vec<&str> = occ.split(',').collect();
            assert_eq!(parts.len(), 4, "malformed occurrence {occ:?}");
            let payload = if parts[3] == "NONE" {
                Vec::new()
            } else {
                (0..parts[3].len() / 2)
                    .map(|i| u8::from_str_radix(&parts[3][i * 2..i * 2 + 2], 16).unwrap())
                    .collect()
            };
            Position {
                position: parts[0].parse().unwrap(),
                start_offset: parts[1].parse().unwrap(),
                end_offset: parts[2].parse().unwrap(),
                payload,
            }
        })
        .collect()
}

struct Fixture {
    manifest: Manifest,
    fnm: Vec<u8>,
    tim: Vec<u8>,
    tip: Vec<u8>,
    tmd: Vec<u8>,
    doc: Vec<u8>,
    pos: Vec<u8>,
    pay: Vec<u8>,
}

impl Fixture {
    fn load() -> Self {
        let manifest = Manifest::load();
        Fixture {
            fnm: read_raw(manifest.get("fnm_file_name")),
            tim: read_raw(manifest.get("tim_file_name")),
            tip: read_raw(manifest.get("tip_file_name")),
            tmd: read_raw(manifest.get("tmd_file_name")),
            doc: read_raw(manifest.get("doc_file_name")),
            pos: read_raw(manifest.get("pos_file_name")),
            pay: read_raw(manifest.get("pay_file_name")),
            manifest,
        }
    }

    fn open(
        &self,
    ) -> (
        blocktree::BlockTreeFields,
        postings::DocInput<'_>,
        postings::PosInput<'_>,
        postings::PayInput<'_>,
    ) {
        let m = &self.manifest;
        let id = id_from_hex(m.get("id_hex"));
        let suffix = m.get("segment_suffix");
        let max_doc: i32 = m.get("max_doc").parse().unwrap();
        let field_infos = field_infos::parse(&self.fnm, &id, "").expect("parse .fnm");
        let fields = blocktree::open(
            &self.tim,
            &self.tip,
            &self.tmd,
            &field_infos,
            &id,
            suffix,
            max_doc,
        )
        .expect("open blocktree");
        (
            fields,
            postings::DocInput::open(&self.doc, &id, suffix).expect("open .doc"),
            postings::PosInput::open(&self.pos, &id, suffix).expect("open .pos"),
            postings::PayInput::open(&self.pay, &id, suffix).expect("open .pay"),
        )
    }
}

/// The fixture really does contain the structure this file exists to test:
/// past `LEVEL1_NUM_DOCS` documents (so a level-1 entry exists) and past
/// `BLOCK_SIZE` occurrences (so `.pos` has full blocks and a vint tail).
#[test]
fn the_fixture_term_actually_carries_skip_data() {
    let fx = Fixture::load();
    let (fields, _doc_in, _pos_in, _pay_in) = fx.open();
    let field = fields.field("pskip").expect("pskip field");
    let stats = field
        .seek_exact(fx.manifest.get("term").as_bytes())
        .expect("term present");
    let doc_freq: i32 = fx.manifest.get("docFreq").parse().unwrap();
    let total_term_freq: i64 = fx.manifest.get("totalTermFreq").parse().unwrap();
    assert_eq!(stats.doc_freq, doc_freq);
    assert_eq!(stats.total_term_freq, total_term_freq);
    assert!(
        doc_freq > 32 * 256,
        "docFreq {doc_freq} must exceed LEVEL1_NUM_DOCS for a level-1 entry to exist"
    );
    assert!(
        total_term_freq % 256 != 0 && total_term_freq > 256,
        "totalTermFreq {total_term_freq} must span full .pos blocks and a vint tail"
    );

    // The level-1 entry's own `posBufferUpto`, as the generator derived it
    // from the frequencies it wrote. Zero here would make the level-1
    // `.pos`/`.pay` pointers indistinguishable from a reader that ignores
    // them -- see this file's module doc.
    let level1_pos_buffer_upto: i64 = fx.manifest.get("level1_pos_buffer_upto").parse().unwrap();
    assert!(
        level1_pos_buffer_upto != 0,
        "the level-1 span boundary must fall *inside* a .pos block, not on one: \
         with posBufferUpto == 0 a reader that never reads the field passes"
    );

    // And the sparse term, whose `.doc` blocks are not all-consecutive.
    let sparse_doc_freq: i32 = fx.manifest.get("sparse_docFreq").parse().unwrap();
    assert!(
        sparse_doc_freq > 256 && sparse_doc_freq < doc_freq,
        "the sparse term must span several .doc blocks without filling them"
    );

    // The term metadata must locate the vint tail: `lastPosBlockOffset` is
    // what tells the skip-driven walk a full block from the tail once it has
    // jumped into the middle of `.pos` (b5 F4 wrote this as a constant 0).
    let meta = field
        .term_metadata(fx.manifest.get("term").as_bytes())
        .expect("term metadata")
        .expect("term present");
    assert!(
        meta.last_pos_block_offset > 0,
        "real Lucene records where the vint position tail begins"
    );
}

/// The headline property: for every document Java sampled, the skip-driven
/// single-document walk returns exactly the occurrences Java's own
/// `advance(doc)` + `nextPosition()` produced.
#[test]
fn advance_then_walk_matches_real_lucene_for_every_sampled_document() {
    let fx = Fixture::load();
    let (fields, doc_in, pos_in, pay_in) = fx.open();
    let field = fields.field("pskip").expect("pskip field");
    let term = fx.manifest.get("term").as_bytes();

    let sampled: Vec<i32> = fx
        .manifest
        .get("sampled_docs")
        .split(',')
        .map(|d| d.parse().unwrap())
        .collect();
    assert!(sampled.len() > 40, "the fixture samples every boundary");

    for doc_id in sampled {
        let expected = parse_occurrences(fx.manifest.get(&format!("doc.{doc_id}.occurrences")));
        let expected_freq: usize = fx
            .manifest
            .get(&format!("doc.{doc_id}.freq"))
            .parse()
            .unwrap();
        assert_eq!(expected.len(), expected_freq);
        let got = field
            .occurrences_for_doc(term, Some(&doc_in), &pos_in, Some(&pay_in), doc_id)
            .expect("occurrences_for_doc")
            .unwrap_or_else(|| panic!("doc {doc_id} is in the term's postings"));
        assert_eq!(got, expected, "doc {doc_id}");
    }
}

/// The whole-term reader -- which addresses `.pos` by a running frequency sum
/// and reads no skip pointer at all -- must agree with real Lucene on the
/// same documents, and therefore with the skip-driven walk.
///
/// This is what makes the test above conclusive rather than circular: the two
/// readers share the per-block wire decode but nothing of how they locate a
/// document's occurrence window, so a shared misunderstanding of the skip
/// records cannot make both of them agree with Java.
#[test]
fn the_whole_term_reader_agrees_with_real_lucene_on_the_same_documents() {
    let fx = Fixture::load();
    let (fields, doc_in, pos_in, pay_in) = fx.open();
    let field = fields.field("pskip").expect("pskip field");
    let term = fx.manifest.get("term").as_bytes();

    let per_doc = field
        .positions(term, Some(&doc_in), &pos_in, Some(&pay_in))
        .expect("positions")
        .expect("term present");
    let doc_freq: usize = fx.manifest.get("docFreq").parse().unwrap();
    assert_eq!(per_doc.len(), doc_freq);

    // Every document of this fixture contains the term, so doc id == index.
    for spec in fx.manifest.get("sampled_docs").split(',') {
        let doc_id: usize = spec.parse().unwrap();
        let expected = parse_occurrences(fx.manifest.get(&format!("doc.{doc_id}.occurrences")));
        assert_eq!(per_doc[doc_id], expected, "doc {doc_id}");
    }
}

/// The sparse term, and the two things only it can cover.
///
/// Every document contains `skipterm`, so all 33 of its level-0 blocks take
/// Lucene's `docRange == BLOCK_SIZE` degenerate doc-delta encoding -- the one
/// shape that carries no per-document information. `gapterm` is in 40% of the
/// documents, so its blocks are packed-FOR or unary bit sets, and most of the
/// sampled document ids are *not* in it.
///
/// So this covers two things the dense term cannot: a skip-driven walk whose
/// `.doc` block actually has to be bit-unpacked to find the target, and
/// `advance(doc)` for a document the term does not contain, which must report
/// "not here" rather than the next document's occurrences. Java's own
/// `advance` result is the ground truth for the second.
#[test]
fn the_sparse_term_covers_real_doc_delta_blocks_and_absent_documents() {
    let fx = Fixture::load();
    let (fields, doc_in, pos_in, pay_in) = fx.open();
    let field = fields.field("pskip").expect("pskip field");
    let term = fx.manifest.get("sparse_term").as_bytes();

    let mut present = 0usize;
    let mut absent = 0usize;
    for spec in fx.manifest.get("sampled_docs").split(',') {
        let doc_id: i32 = spec.parse().unwrap();
        let landed: i32 = fx
            .manifest
            .get(&format!("sparse.{doc_id}.advance"))
            .parse()
            .unwrap();
        let got = field
            .occurrences_for_doc(term, Some(&doc_in), &pos_in, Some(&pay_in), doc_id)
            .expect("occurrences_for_doc");
        if landed == doc_id {
            let expected =
                parse_occurrences(fx.manifest.get(&format!("sparse.{doc_id}.occurrences")));
            assert_eq!(
                got.unwrap_or_else(|| panic!("doc {doc_id} is in gapterm's postings")),
                expected,
                "doc {doc_id}"
            );
            present += 1;
        } else {
            // Java advanced past `doc_id`, so the term is not in it.
            assert!(
                got.is_none(),
                "doc {doc_id} is not in gapterm's postings (Java landed on {landed})"
            );
            absent += 1;
        }
    }
    assert!(
        present > 10 && absent > 10,
        "the sample must contain both kinds ({present} present, {absent} absent)"
    );
}

/// The whole-term reader's positions for `term` -- pinned against real
/// Lucene by this file's other tests -- as one `Vec<i32>` per document, in the
/// term's own document order, plus the documents themselves.
fn expected_positions(
    field: &blocktree::FieldTerms,
    term: &[u8],
    doc_in: &postings::DocInput<'_>,
    pos_in: &postings::PosInput<'_>,
    pay_in: &postings::PayInput<'_>,
) -> (Vec<i32>, Vec<Vec<i32>>) {
    let docs = field
        .postings(term, Some(doc_in))
        .expect("postings")
        .expect("term present")
        .docs;
    let positions = field
        .positions(term, Some(doc_in), pos_in, Some(pay_in))
        .expect("positions")
        .expect("term present")
        .into_iter()
        .map(|occ| occ.into_iter().map(|p| p.position).collect())
        .collect();
    (docs, positions)
}

/// `PositionsCursor` walked document by document, reading every position:
/// the sequential `nextDoc()` + `nextPosition()` shape, across 33 `.doc`
/// blocks, a level-1 boundary, full `.pos` blocks and the vint tail.
#[test]
fn lazy_positions_sequential_walk_matches_the_whole_term_reader() {
    let fx = Fixture::load();
    let (fields, doc_in, pos_in, pay_in) = fx.open();
    let field = fields.field("pskip").expect("pskip field");
    for term in [fx.manifest.get("term"), fx.manifest.get("sparse_term")] {
        let term = term.as_bytes();
        let (docs, expected) = expected_positions(field, term, &doc_in, &pos_in, &pay_in);
        let mut c = field
            .lazy_positions(term, &doc_in, &pos_in)
            .expect("lazy_positions")
            .expect("term present");
        let mut i = 0;
        loop {
            let doc = c.next_doc().expect("next_doc");
            if doc == postings::NO_MORE_DOCS {
                break;
            }
            assert_eq!(doc, docs[i]);
            assert_eq!(c.freq() as usize, expected[i].len(), "freq of doc {doc}");
            let got: Vec<i32> = (0..c.freq())
                .map(|_| c.next_position().expect("next_position"))
                .collect();
            assert_eq!(got, expected[i], "positions of doc {doc}");
            i += 1;
        }
        assert_eq!(i, docs.len());
    }
}

/// Reading only *some* documents' positions -- and only some of a document's
/// positions -- must not disturb the ones read later: every unread position
/// is exactly what `accumulatePendingPositions`/`skipPositions` step over.
#[test]
fn lazy_positions_skipping_documents_and_partial_reads_stay_aligned() {
    let fx = Fixture::load();
    let (fields, doc_in, pos_in, pay_in) = fx.open();
    let field = fields.field("pskip").expect("pskip field");
    let term = fx.manifest.get("term").as_bytes();
    let (docs, expected) = expected_positions(field, term, &doc_in, &pos_in, &pay_in);
    // Several visiting patterns: every 3rd doc, every 300th (skips whole
    // `.pos` blocks between reads), and reading just the first position of
    // every 2nd doc.
    for (stride, partial) in [(3usize, false), (300, false), (2, true), (1, true)] {
        let mut c = field
            .lazy_positions(term, &doc_in, &pos_in)
            .expect("lazy_positions")
            .expect("term present");
        let mut i = 0;
        while c.next_doc().expect("next_doc") != postings::NO_MORE_DOCS {
            if i % stride == 0 {
                let n = if partial { 1 } else { c.freq() as usize };
                let got: Vec<i32> = (0..n)
                    .map(|_| c.next_position().expect("next_position"))
                    .collect();
                assert_eq!(
                    got,
                    expected[i][..n],
                    "stride {stride} partial {partial}: doc {}",
                    docs[i]
                );
            }
            i += 1;
        }
    }
}

/// `advance` through the skip data (level-1 and level-0 jumps, the `.pos`
/// origin they carry) and then read positions: the phrase path's shape. Checked
/// against Java's own per-document occurrences from the manifest.
#[test]
fn lazy_positions_after_advance_match_real_lucene() {
    let fx = Fixture::load();
    let (fields, doc_in, pos_in, _pay_in) = fx.open();
    let field = fields.field("pskip").expect("pskip field");
    let term = fx.manifest.get("term").as_bytes();
    let sampled: Vec<i32> = fx
        .manifest
        .get("sampled_docs")
        .split(',')
        .map(|d| d.parse().unwrap())
        .collect();
    // One cursor advanced through every sampled document in order, and a
    // fresh cursor per document, so both the incremental and the cold path
    // are covered.
    let mut shared = field
        .lazy_positions(term, &doc_in, &pos_in)
        .expect("lazy_positions")
        .expect("term present");
    for &target in &sampled {
        let want: Vec<i32> =
            parse_occurrences(fx.manifest.get(&format!("doc.{target}.occurrences")))
                .into_iter()
                .map(|p| p.position)
                .collect();
        for fresh in [false, true] {
            let mut own;
            let c = if fresh {
                own = field
                    .lazy_positions(term, &doc_in, &pos_in)
                    .expect("lazy_positions")
                    .expect("term present");
                &mut own
            } else {
                &mut shared
            };
            assert_eq!(c.advance(target).expect("advance"), target);
            let got: Vec<i32> = (0..c.freq())
                .map(|_| c.next_position().expect("next_position"))
                .collect();
            assert_eq!(got, want, "doc {target} (fresh cursor: {fresh})");
        }
    }
}

/// Asking for more positions than the document has is a caller bug; it must
/// be an error, not a read of the next document's positions.
#[test]
fn lazy_positions_refuses_to_read_past_the_documents_frequency() {
    let fx = Fixture::load();
    let (fields, doc_in, pos_in, _pay_in) = fx.open();
    let field = fields.field("pskip").expect("pskip field");
    let term = fx.manifest.get("term").as_bytes();
    let mut c = field
        .lazy_positions(term, &doc_in, &pos_in)
        .expect("lazy_positions")
        .expect("term present");
    // Before the first document there is nothing to read.
    assert!(c.next_position().is_err());
    c.next_doc().expect("next_doc");
    for _ in 0..c.freq() {
        c.next_position().expect("within freq");
    }
    assert!(c.next_position().is_err());
    // An absent term is `None`, and a docs-only field is refused.
    assert!(field
        .lazy_positions(b"no-such-term", &doc_in, &pos_in)
        .expect("lookup")
        .is_none());
}

/// The lazy document cursor on `gapterm` -- Java-written blocks that are
/// packed deltas or unary bit sets -- under both `DocsOnly` (Lucene's
/// `PostingsEnum.NONE`: a bit-set `advance` that keeps no rank, the fast
/// header walk) and `Freqs` (the rank from cumulative popcounts, so `freq()`
/// must still be this document's). Java's own `advance` landing is the
/// ground truth, from a fresh cursor and from one that keeps moving forward.
#[test]
fn lazy_cursor_advance_on_the_sparse_term_matches_real_lucene() {
    let fx = Fixture::load();
    let (fields, doc_in, _, _) = fx.open();
    let field = fields.field("pskip").expect("pskip field");
    let term = fx.manifest.get("sparse_term").as_bytes();
    let mut samples: Vec<i32> = fx
        .manifest
        .get("sampled_docs")
        .split(',')
        .map(|s| s.parse().unwrap())
        .collect();
    samples.sort_unstable();
    samples.dedup();
    let landed = |doc: i32| -> i32 {
        fx.manifest
            .get(&format!("sparse.{doc}.advance"))
            .parse()
            .unwrap()
    };
    let freq_of = |doc: i32| -> i32 {
        parse_occurrences(fx.manifest.get(&format!("sparse.{doc}.occurrences"))).len() as i32
    };
    // Every `gapterm` document has one occurrence, so `freq()` cannot show a
    // wrong rank; the document `next_doc` returns after a bit-set `advance`
    // can, since the expansion restarts from the rank. Its truth is the whole
    // term's document list, which this file pins to Lucene above.
    let all_docs = field
        .postings(term, Some(&doc_in))
        .unwrap()
        .expect("gapterm")
        .docs;
    for flags in [
        postings::PostingsFlags::DocsOnly,
        postings::PostingsFlags::Freqs,
    ] {
        let open = || {
            field
                .lazy_postings_with_flags(term, &doc_in, flags)
                .unwrap()
                .expect("gapterm")
        };
        let mut walking = open();
        for &doc in &samples {
            let want = landed(doc);
            let got = open().advance(doc).unwrap();
            assert_eq!(got, want, "{flags:?} fresh advance({doc})");
            if walking.doc_id() < doc {
                assert_eq!(
                    walking.advance(doc).unwrap(),
                    want,
                    "{flags:?} advance({doc})"
                );
            }
            if flags == postings::PostingsFlags::Freqs && want == doc {
                assert_eq!(walking.freq(), Some(freq_of(doc)), "freq of {doc}");
            }
            if walking.doc_id() == want && want != postings::NO_MORE_DOCS {
                let next = all_docs
                    .iter()
                    .copied()
                    .find(|&d| d > want)
                    .unwrap_or(postings::NO_MORE_DOCS);
                assert_eq!(
                    walking.next_doc().unwrap(),
                    next,
                    "{flags:?} next_doc after {want}"
                );
            }
        }
    }
}
