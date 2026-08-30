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
