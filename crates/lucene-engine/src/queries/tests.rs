#![cfg(test)]
//! Decoding the query-tree nodes of kinds 46-57: what each reads, and the
//! blobs it refuses.
use super::*;
use crate::query_blob::test_support::Blob;
use crate::query_blob::Cursor;

fn decode_blob(blob: &Blob) -> std::result::Result<Clause, Status> {
    let mut c = Cursor::new(&blob.0);
    let mut nodes = 0;
    crate::query_blob::decode_node(&mut c, 0, &mut nodes)
}

fn i64s(blob: Blob, v: i64) -> Blob {
    blob.raw(&Blob(v.to_le_bytes().to_vec()))
}

fn phrase_positions(slop: i32, terms: &[(i32, &str)]) -> Blob {
    let mut b = Blob::default()
        .u8(NODE_PHRASE_POSITIONS)
        .str("f")
        .i32(slop)
        .i32(terms.len() as i32);
    for (position, term) in terms {
        b = b.i32(*position).str(term);
    }
    b
}

fn multi_phrase(prefix: bool, slop: i32, positions: &[(i32, &[&str])]) -> Blob {
    let mut b = Blob::default()
        .u8(if prefix {
            NODE_MULTI_PHRASE_PREFIX
        } else {
            NODE_MULTI_PHRASE
        })
        .str("f")
        .i32(slop);
    if prefix {
        b = b.i32(50);
    }
    b = b.i32(positions.len() as i32);
    for (position, terms) in positions {
        b = b.i32(*position).i32(terms.len() as i32);
        for t in *terms {
            b = b.str(t);
        }
    }
    b
}

#[test]
fn phrases_with_positions() {
    let ok = decode_blob(&phrase_positions(1, &[(0, "a"), (2, "b")])).unwrap();
    assert!(matches!(ok, Clause::Phrase(_)));
    assert!(decode_blob(&phrase_positions(-1, &[(0, "a")])).is_err());
    assert!(decode_blob(&phrase_positions(0, &[])).is_err());
    // A truncated blob is refused, not read past.
    let mut cut = phrase_positions(0, &[(0, "a"), (1, "b")]);
    cut.0.truncate(cut.0.len() - 2);
    assert!(decode_blob(&cut).is_err());
}

#[test]
fn multi_phrases() {
    let ok = decode_blob(&multi_phrase(false, 2, &[(0, &["a", "b"]), (1, &["c"])])).unwrap();
    assert!(matches!(ok, Clause::MultiPhrase(_)));
    let prefix = decode_blob(&multi_phrase(true, 0, &[(0, &["a"]), (1, &["c"])])).unwrap();
    assert!(matches!(prefix, Clause::Extended(_)));
    for prefix in [false, true] {
        assert!(decode_blob(&multi_phrase(prefix, -1, &[(0, &["a"])])).is_err());
        assert!(decode_blob(&multi_phrase(prefix, 0, &[])).is_err());
        assert!(decode_blob(&multi_phrase(prefix, 0, &[(0, &[])])).is_err());
    }
}

#[test]
fn fuzzy_nodes() {
    let node = |edits: u8, prefix: i32| {
        Blob::default()
            .u8(NODE_FUZZY)
            .str("f")
            .str("term")
            .u8(edits)
            .i32(prefix)
            .u8(1)
    };
    for edits in 0..=2 {
        assert!(matches!(
            decode_blob(&node(edits, 1)).unwrap(),
            Clause::Fuzzy(_)
        ));
    }
    assert!(decode_blob(&node(3, 0)).is_err());
    assert!(decode_blob(&node(1, -1)).is_err());
}

#[test]
fn substring_nodes() {
    let node = |kind: u8, case: u8, pattern: &[u8]| {
        Blob::default()
            .u8(NODE_SUBSTRING)
            .str("f")
            .u8(kind)
            .u8(case)
            .bytes(pattern)
    };
    for kind in 0..=3 {
        for case in 0..=1 {
            assert!(
                matches!(
                    decode_blob(&node(kind, case, b"ab*")).unwrap(),
                    Clause::Extended(_)
                ),
                "kind {kind} case {case}"
            );
        }
    }
    assert!(decode_blob(&node(4, 0, b"ab")).is_err());
    assert!(decode_blob(&node(0, 2, b"ab")).is_err());
    assert_eq!(
        decode_blob(&node(0, 0, &[0xff, 0xfe])).err(),
        Some(Status::InvalidUtf8)
    );
}

#[test]
fn point_range_or_doc_values() {
    let node = i64s(
        i64s(Blob::default().u8(NODE_POINT_RANGE_OR_DV).str("n"), 5),
        9,
    );
    assert!(matches!(decode_blob(&node).unwrap(), Clause::Extended(_)));
    let short = Blob::default().u8(NODE_POINT_RANGE_OR_DV).str("n");
    assert!(decode_blob(&short).is_err());
}

#[test]
fn more_like_this_nodes() {
    let node = |limit: i32| {
        Blob::default()
            .u8(NODE_MORE_LIKE_THIS)
            .u8(1)
            .f32(2.0)
            .i32(1)
            .i32(2)
            .str("body")
            .i32(limit)
            .i32(2)
            .str("cat")
            .f32(1.5)
            .str("dog")
            .f32(0.5)
            .str("title")
            .i32(1)
            .i32(0)
    };
    assert!(matches!(
        decode_blob(&node(3)).unwrap(),
        Clause::Extended(_)
    ));
    assert!(decode_blob(&node(-1)).is_err());
}

#[test]
fn visit_wrappers() {
    let inner = Blob::default().term("f", "x");
    let only = Blob::default().u8(NODE_VISIT_ONLY).raw(&inner);
    assert!(matches!(decode_blob(&only).unwrap(), Clause::Extended(_)));
    let nested = Blob::default().u8(NODE_VISIT_ONLY).raw(&only);
    assert!(matches!(decode_blob(&nested).unwrap(), Clause::Extended(_)));
}

#[test]
fn join_nodes() {
    let node = |mode: u8, min: i32, max: i32| {
        Blob::default()
            .u8(NODE_JOIN)
            .str("join_f")
            .u8(mode)
            .i32(min)
            .i32(max)
            .term("to", "a")
            .term("from", "b")
    };
    for mode in 0..=4 {
        assert!(matches!(
            decode_blob(&node(mode, 0, 5)).unwrap(),
            Clause::Extended(_)
        ));
    }
    assert!(decode_blob(&node(5, 0, 5)).is_err());
    assert!(decode_blob(&node(1, -1, 5)).is_err());
    assert!(decode_blob(&node(1, 3, 2)).is_err());
}

#[test]
fn doc_and_score_bases_must_ascend() {
    let bad = Blob::default().u8(NODE_DOC_AND_SCORE).i32(2).i32(10).i32(5);
    assert!(decode_blob(&bad).is_err());
    let negative = Blob::default().u8(NODE_DOC_AND_SCORE).i32(1).i32(-1);
    assert!(decode_blob(&negative).is_err());
}
