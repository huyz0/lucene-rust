//! Every postings read path, on the retired postings formats of the M8
//! backward-codecs corpus (`fixtures/data/bwc/<version>/`, written by that
//! Lucene release): `Lucene90` (9.0.0), `Lucene99` (9.11.1), `Lucene912`
//! (9.12.2) and `Lucene101` (10.2.2), plus the current `Lucene104` (10.4.0)
//! as the control.
//!
//! `crates/lucene-search/tests/bwc_fixtures.rs` proves the whole-term decode
//! (`TermsEnum` + `read_postings` + `read_positions`) against the digests
//! Lucene wrote. This file holds every *other* entry point to that decoded
//! term -- the lazy docs cursor's `next_doc`/`advance`, the positions cursor,
//! and the wanted-documents and one-document occurrence readers the phrase
//! scorer and the highlighter use -- to the same answer, term by term, so a
//! retired-format term cannot take a path that disagrees with the one the
//! digests check.

// The arithmetic gate is about values read off disk; a test's index
// arithmetic is not one (`docs/arithmetic-gate.md`, "Test code").
#![allow(clippy::arithmetic_side_effects)]

use std::path::PathBuf;

use lucene_codecs::blocktree::{self, BlockTreeFields};
use lucene_codecs::field_infos::{self, IndexOptions};
use lucene_codecs::postings::{DocInput, PayInput, PosInput, PostingsFormat, NO_MORE_DOCS};

struct Segment {
    fields: BlockTreeFields,
    field_infos: field_infos::FieldInfos,
    doc: Vec<u8>,
    pos: Vec<u8>,
    pay: Vec<u8>,
    id: [u8; 16],
    suffix: String,
}

fn open(version: &str) -> Segment {
    open_in("bwc", version, 3000)
}

/// `_0` of `fixtures/data/<corpus>/<version>/`; a `.pay` the segment does
/// not have (no offsets or payloads) reads as empty.
fn open_in(corpus: &str, version: &str, max_doc: i32) -> Segment {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/data")
        .join(corpus)
        .join(version);
    let names: Vec<String> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .collect();
    let file = |ext: &str| {
        let name = names
            .iter()
            .find(|n| n.starts_with("_0_") && n.ends_with(ext))
            .unwrap_or_else(|| panic!("{version}: no _0 {ext}"));
        (name.clone(), std::fs::read(dir.join(name)).unwrap())
    };
    let optional = |ext: &str| {
        names
            .iter()
            .find(|n| n.starts_with("_0_") && n.ends_with(ext))
            .map(|n| std::fs::read(dir.join(n)).unwrap())
            .unwrap_or_default()
    };
    let (tim_name, tim) = file(".tim");
    let suffix = tim_name
        .strip_prefix("_0_")
        .unwrap()
        .strip_suffix(".tim")
        .unwrap()
        .to_string();
    // The segment id is the one every file's index header carries.
    let name_len = tim[4] as usize;
    let mut id = [0u8; 16];
    id.copy_from_slice(&tim[9 + name_len..25 + name_len]);
    let fnm = std::fs::read(dir.join("_0.fnm")).unwrap();
    let field_infos = field_infos::parse(&fnm, &id, "").unwrap();
    let fields = blocktree::open(
        &tim,
        &file(".tip").1,
        &file(".tmd").1,
        &field_infos,
        &id,
        &suffix,
        max_doc,
    )
    .unwrap();
    Segment {
        fields,
        field_infos,
        doc: file(".doc").1,
        pos: file(".pos").1,
        pay: optional(".pay"),
        id,
        suffix,
    }
}

fn check_version(version: &str, format: PostingsFormat) {
    let seg = open(version);
    let doc_in = DocInput::open(&seg.doc, &seg.id, &seg.suffix).unwrap();
    let pos_in = PosInput::open(&seg.pos, &seg.id, &seg.suffix).unwrap();
    let pay_in = PayInput::open(&seg.pay, &seg.id, &seg.suffix).unwrap();
    assert_eq!(doc_in.format(), format);
    assert_eq!(pos_in.format(), format);
    let mut checked_terms = 0;
    for fi in &seg.field_infos.fields {
        if fi.index_options == IndexOptions::None || fi.name == "id" {
            continue;
        }
        let terms = seg.fields.field(&fi.name).unwrap();
        assert_eq!(terms.postings_format(), format);
        let has_pos = fi.index_options.subsumes_positions();
        let mut te = terms.iter();
        let mut all_terms = Vec::new();
        while let Some((t, stats)) = te.try_next().unwrap() {
            all_terms.push((t.to_vec(), stats));
        }
        for (term, stats) in all_terms {
            let ctx = format!("{version} {}:{}", fi.name, String::from_utf8_lossy(&term));
            let eager = terms.postings(&term, Some(&doc_in)).unwrap().unwrap();
            assert_eq!(eager.docs.len() as i32, stats.doc_freq, "{ctx}");

            // The lazy docs cursor, stepped and advanced.
            if stats.doc_freq > 1 {
                let mut c = terms.lazy_postings(&term, &doc_in).unwrap().unwrap();
                for (i, &d) in eager.docs.iter().enumerate() {
                    let got = c.next_doc().unwrap_or_else(|e| {
                        panic!("{ctx} next_doc #{i} (df {}): {e}", stats.doc_freq)
                    });
                    assert_eq!(got, d, "{ctx} next_doc #{i}");
                    assert_eq!(c.freq().unwrap_or(1), eager.freqs[i], "{ctx} freq #{i}");
                }
                assert_eq!(c.next_doc().unwrap(), NO_MORE_DOCS, "{ctx}");
                // Documents only: the frequency blocks stepped over.
                let mut c = terms
                    .lazy_postings_with_flags(
                        &term,
                        &doc_in,
                        lucene_codecs::postings::PostingsFlags::DocsOnly,
                    )
                    .unwrap()
                    .unwrap();
                for (i, &d) in eager.docs.iter().enumerate() {
                    assert_eq!(c.next_doc().unwrap(), d, "{ctx} docs-only #{i}");
                }
                assert_eq!(c.next_doc().unwrap(), NO_MORE_DOCS, "{ctx}");
                let mut c = terms.lazy_postings(&term, &doc_in).unwrap().unwrap();
                for &d in eager.docs.iter().step_by(37) {
                    let got = c.advance(d).unwrap_or_else(|e| {
                        panic!("{ctx} advance({d}) (df {}): {e}", stats.doc_freq)
                    });
                    assert_eq!(got, d, "{ctx} advance({d})");
                    // One past it lands on the next document.
                    let next = eager
                        .docs
                        .iter()
                        .copied()
                        .find(|&x| x > d)
                        .unwrap_or(NO_MORE_DOCS);
                    assert_eq!(c.advance(d + 1).unwrap(), next, "{ctx} advance({})", d + 1);
                    if next == NO_MORE_DOCS {
                        break;
                    }
                }
            }
            if !has_pos {
                continue;
            }
            let positions = terms
                .positions(&term, Some(&doc_in), &pos_in, Some(&pay_in))
                .unwrap()
                .unwrap();

            // The positions cursor.
            let mut pc = terms
                .lazy_positions(&term, &doc_in, &pos_in)
                .unwrap()
                .unwrap();
            for (i, &d) in eager.docs.iter().enumerate() {
                if i % 3 == 1 {
                    continue; // positions of a skipped document are never read
                }
                assert_eq!(pc.advance(d).unwrap(), d, "{ctx}");
                let mut got = Vec::new();
                pc.positions_into(&mut got).unwrap();
                let want: Vec<i32> = positions[i].iter().map(|p| p.position).collect();
                assert_eq!(got, want, "{ctx} positions of doc {d}");
            }

            // Wanted-documents readers: every other document.
            let wanted: Vec<usize> = (0..eager.docs.len()).step_by(2).collect();
            let (flat, starts) = terms
                .positions_for_docs(
                    &term,
                    Some(&doc_in),
                    &pos_in,
                    Some(&pay_in),
                    &eager.freqs,
                    stats.total_term_freq,
                    &wanted,
                )
                .unwrap();
            let (occ, occ_starts) = terms
                .occurrences_for_docs(
                    &term,
                    Some(&doc_in),
                    &pos_in,
                    Some(&pay_in),
                    &eager.freqs,
                    stats.total_term_freq,
                    &wanted,
                )
                .unwrap();
            for (w, &i) in wanted.iter().enumerate() {
                let want: Vec<i32> = positions[i].iter().map(|p| p.position).collect();
                assert_eq!(
                    &flat[starts[w] as usize..starts[w + 1] as usize],
                    &want[..],
                    "{ctx} positions_for_docs #{i}"
                );
                assert_eq!(
                    &occ[occ_starts[w] as usize..occ_starts[w + 1] as usize],
                    &positions[i][..],
                    "{ctx} occurrences_for_docs #{i}"
                );
            }
            // One document's occurrences, and a document the term lacks.
            for (i, &d) in eager.docs.iter().enumerate().step_by(11) {
                let one = terms
                    .occurrences_for_doc(&term, Some(&doc_in), &pos_in, Some(&pay_in), d)
                    .unwrap();
                assert_eq!(one.as_deref(), Some(&positions[i][..]), "{ctx} doc {d}");
            }
            if let Some(absent) = (0..3000).find(|x| eager.docs.binary_search(x).is_err()) {
                let none = terms
                    .occurrences_for_doc(&term, Some(&doc_in), &pos_in, Some(&pay_in), absent)
                    .unwrap();
                assert!(none.is_none(), "{ctx} doc {absent} is not in the postings");
            }
            checked_terms += 1;
        }
    }
    assert!(
        checked_terms > 50,
        "{version}: only {checked_terms} positional terms"
    );
}

#[test]
fn lucene90_postings_agree_on_every_read_path() {
    check_version("9.0.0", PostingsFormat::Lucene90);
}

#[test]
fn lucene99_postings_agree_on_every_read_path() {
    check_version("9.11.1", PostingsFormat::Lucene99);
}

#[test]
fn lucene912_postings_agree_on_every_read_path() {
    check_version("9.12.2", PostingsFormat::Lucene912);
}

#[test]
fn lucene101_postings_agree_on_every_read_path() {
    check_version("10.2.2", PostingsFormat::Lucene101);
}

#[test]
fn lucene104_postings_agree_on_every_read_path() {
    check_version("10.4.0", PostingsFormat::Lucene104);
}

/// The skip data `BwcWrite`'s 3,000-document segments never reach:
/// `fixtures/data/bwc-big/<version>/` (`fixtures/bwc/BwcBig.java`) is one
/// 20,000-document segment whose `all`/`half`/`run` terms span every level of
/// the trailing multi-level skip list (`Lucene90`/`Lucene99`: entries every
/// 128, 1,024 and 8,192 documents) and the inline level-1 entries
/// (`Lucene912`/`Lucene101`: every 4,096). For each term, on both fields:
///
/// - the lazy cursor's `next_doc` and freqs, and `advance` at strides that
///   land inside blocks, on block boundaries, and across whole level-1 spans,
///   against the eager whole-term decode;
/// - `advance_shallow` then `advance`: the block it reports covers the
///   target, and its level-0 impacts -- and the level-1 impacts, over the
///   span `level1_last_doc_id` reports -- bound every frequency in it (the
///   soundness a max-score skip relies on);
/// - positions through the lazy positions cursor, skipping documents.
fn check_big(version: &str, format: PostingsFormat) {
    const MAX_DOC: i32 = 20_000;
    let seg = open_in("bwc-big", version, MAX_DOC);
    let doc_in = DocInput::open(&seg.doc, &seg.id, &seg.suffix).unwrap();
    let pos_in = PosInput::open(&seg.pos, &seg.id, &seg.suffix).unwrap();
    assert_eq!(doc_in.format(), format);
    let mut level1_checks = 0;
    let mut shallow_checks = 0;
    for field in ["f", "d"] {
        let terms = seg.fields.field(field).unwrap();
        let has_pos = field == "f";
        for term in ["all", "half", "third", "rare", "run", "peak"] {
            let Some(eager) = terms.postings(term.as_bytes(), Some(&doc_in)).unwrap() else {
                assert_eq!(field, "d", "{version} f:{term} missing");
                continue;
            };
            let ctx = format!("{version} {field}:{term}");
            let docs = &eager.docs;
            let next_geq = |t: i32| {
                let i = docs.partition_point(|&x| x < t);
                (i, docs.get(i).copied().unwrap_or(NO_MORE_DOCS))
            };

            let mut c = terms
                .lazy_postings(term.as_bytes(), &doc_in)
                .unwrap()
                .unwrap();
            for (i, &d) in docs.iter().enumerate() {
                assert_eq!(c.next_doc().unwrap(), d, "{ctx} next_doc #{i}");
                if has_pos {
                    assert_eq!(c.freq().unwrap(), eager.freqs[i], "{ctx} freq #{i}");
                }
            }
            assert_eq!(c.next_doc().unwrap(), NO_MORE_DOCS, "{ctx}");

            for stride in [1, 127, 128, 129, 1000, 1024, 4097, 8193] {
                let mut c = terms
                    .lazy_postings(term.as_bytes(), &doc_in)
                    .unwrap()
                    .unwrap();
                let mut t = 0;
                loop {
                    let (_, want) = next_geq(t);
                    assert_eq!(
                        c.advance(t).unwrap(),
                        want,
                        "{ctx} stride {stride} advance({t})"
                    );
                    if want == NO_MORE_DOCS {
                        break;
                    }
                    t = want + stride;
                }
            }

            let mut c = terms
                .lazy_postings(term.as_bytes(), &doc_in)
                .unwrap()
                .unwrap();
            let mut t = 0;
            while t < MAX_DOC {
                let upto = c.advance_shallow(t).unwrap();
                let (from, want) = next_geq(t);
                if upto != NO_MORE_DOCS && has_pos {
                    assert!(upto >= t, "{ctx} advance_shallow({t}) = {upto}");
                    let to = docs.partition_point(|&x| x <= upto);
                    let block_max = eager.freqs[from..to].iter().copied().max();
                    let bound = c.level0_impacts().iter().map(|i| i.freq).max();
                    if let Some(m) = block_max {
                        assert!(
                            bound.is_some_and(|b| b >= m),
                            "{ctx} level-0 impacts {bound:?} under freq {m} in {t}..={upto}"
                        );
                        shallow_checks += 1;
                    }
                    let l1 = c.level1_last_doc_id();
                    if l1 != NO_MORE_DOCS && !c.level1_impacts().is_empty() {
                        let to1 = docs.partition_point(|&x| x <= l1);
                        if let Some(m) = eager.freqs[from..to1].iter().copied().max() {
                            let b = c.level1_impacts().iter().map(|i| i.freq).max().unwrap();
                            assert!(
                                b >= m,
                                "{ctx} level-1 impacts {b} under freq {m} in {t}..={l1}"
                            );
                            level1_checks += 1;
                        }
                    }
                }
                let got = c.advance(t).unwrap();
                assert_eq!(got, want, "{ctx} advance({t}) after advance_shallow");
                if want == NO_MORE_DOCS {
                    break;
                }
                t = want + 333;
            }

            if has_pos {
                let positions = terms
                    .positions(term.as_bytes(), Some(&doc_in), &pos_in, None)
                    .unwrap()
                    .unwrap();
                let mut pc = terms
                    .lazy_positions(term.as_bytes(), &doc_in, &pos_in)
                    .unwrap()
                    .unwrap();
                for (i, &d) in docs.iter().enumerate() {
                    if i % 200 >= 150 {
                        continue; // runs of skipped documents cross whole blocks
                    }
                    assert_eq!(pc.advance(d).unwrap(), d, "{ctx}");
                    let mut got = Vec::new();
                    pc.positions_into(&mut got).unwrap();
                    let want: Vec<i32> = positions[i].iter().map(|p| p.position).collect();
                    assert_eq!(got, want, "{ctx} positions of doc {d}");
                }
            }
        }
    }
    assert!(
        shallow_checks > 50,
        "{version}: {shallow_checks} shallow checks"
    );
    assert!(
        level1_checks > 0,
        "{version}: no level-1 impacts were checked"
    );
}

#[test]
fn lucene90_skip_data_at_every_level() {
    check_big("9.0.0", PostingsFormat::Lucene90);
}

#[test]
fn lucene99_skip_data_at_every_level() {
    check_big("9.11.1", PostingsFormat::Lucene99);
}

#[test]
fn lucene912_skip_data_at_every_level() {
    check_big("9.12.2", PostingsFormat::Lucene912);
}

#[test]
fn lucene101_skip_data_at_every_level() {
    check_big("10.2.2", PostingsFormat::Lucene101);
}
