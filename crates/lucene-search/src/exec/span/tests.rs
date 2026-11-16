#![cfg(test)]
use super::*;

fn doc(entries: &[(&str, &[i32])]) -> DocPositions {
    entries
        .iter()
        .map(|(t, ps)| (("f".to_string(), t.as_bytes().to_vec()), ps.to_vec()))
        .collect()
}

fn term(t: &str) -> SpanQuery {
    SpanQuery::span_term("f", t)
}

/// The fused in-order near of terms scores as `emit` + `freq_of` do,
/// bit for bit: random documents, two to five clauses (a repeated term
/// among them), slops from 0 to 6, and empty clauses.
#[test]
fn the_fused_ordered_near_of_terms_scores_as_the_spans_do() {
    let mut seed = 0x2545_f491_4f6c_dd1du64;
    let mut next = move |n: u64| {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed % n
    };
    let names = ["a", "b", "c", "d"];
    let mut fused_hits = 0;
    for round in 0..3000 {
        let mut entries: Vec<(&str, Vec<i32>)> = Vec::new();
        for name in names {
            let mut ps: Vec<i32> = (0..next(12)).map(|_| next(40) as i32).collect();
            ps.sort_unstable();
            ps.dedup();
            entries.push((name, ps));
        }
        let d: DocPositions = entries
            .iter()
            .map(|(t, ps)| (("f".to_string(), t.as_bytes().to_vec()), ps.clone()))
            .collect();
        // up to ten clauses: past the eight the walks keep on the stack
        let n = if round % 50 == 0 {
            9 + round % 2
        } else {
            2 + (round % 4)
        };
        let clauses: Vec<SpanQuery> = (0..n).map(|_| term(names[next(4) as usize])).collect();
        let slop = next(7) as u32;
        let q = SpanQuery::span_near(clauses, slop, true);
        let mut leaves = Vec::new();
        crate::collect_span_leaves(&q, &mut leaves);
        leaves.sort_unstable();
        leaves.dedup();
        let compiled = compile(&q, &leaves);
        let pos: Vec<Vec<i32>> = leaves
            .iter()
            .map(|k| d.get(k).cloned().unwrap_or_default())
            .collect();
        let (fused, s) = ordered_term_leaves(&compiled).expect("an in-order near of terms");
        let got = ordered_terms_freq(&fused, s, &pos);
        assert_eq!(got.to_bits(), sloppy_freq(&q, &d).to_bits(), "{q:?} {d:?}");
        if got > 0.0 {
            fused_hits += 1;
        }
    }
    assert!(fused_hits > 300, "{fused_hits}");
    // only an in-order near of two or more terms is fused
    let leaves = vec![
        ("f".to_string(), b"a".to_vec()),
        ("f".to_string(), b"b".to_vec()),
    ];
    let unordered = compile(
        &SpanQuery::span_near([term("a"), term("b")], 1, false),
        &leaves,
    );
    assert!(ordered_term_leaves(&unordered).is_none());
    let one = compile(&SpanQuery::span_near([term("a")], 1, true), &leaves);
    assert!(ordered_term_leaves(&one).is_none());
    let nested = compile(
        &SpanQuery::span_near([term("a"), SpanQuery::span_or([term("a")])], 1, true),
        &leaves,
    );
    assert!(ordered_term_leaves(&nested).is_none());
    assert!(ordered_term_leaves(&Compiled::Term(0)).is_none());
    // an absent leaf is on no document; a near of no clauses matches none
    let mut absent = [LeafPositions::Absent];
    assert_eq!(absent[0].advance(0).unwrap(), NO_MORE_DOCS);
    assert_eq!(absent[0].freq_at(0), 0);
    let mut out = vec![];
    absent[0].positions_at(0, &mut out).unwrap();
    assert!(out.is_empty());
    let empty = Compiled::Near {
        clauses: vec![],
        slop: 0,
        in_order: true,
    };
    assert_eq!(approximate(&empty, 0, &mut absent).unwrap(), NO_MORE_DOCS);
}

#[test]
fn a_term_span_has_width_zero_and_counts_every_position() {
    let d = doc(&[("a", &[0, 3, 7])]);
    assert_eq!(
        emissions(&term("a"), &d),
        vec![(0, 1, 0), (3, 4, 0), (7, 8, 0)]
    );
    assert_eq!(sloppy_freq(&term("a"), &d), 3.0);
    assert_eq!(sloppy_freq(&term("b"), &d), 0.0);
}

#[test]
fn an_or_keeps_repeats_in_start_end_order() {
    let d = doc(&[("a", &[2, 5]), ("b", &[1, 5])]);
    let q = SpanQuery::span_or([term("a"), term("b"), term("a")]);
    assert_eq!(
        emissions(&q, &d),
        vec![
            (1, 2, 0),
            (2, 3, 0),
            (2, 3, 0),
            (5, 6, 0),
            (5, 6, 0),
            (5, 6, 0)
        ]
    );
    assert_eq!(sloppy_freq(&q, &d), 6.0);
}

#[test]
fn near_widths_are_the_gaps_in_order_and_the_extent_out_of_order() {
    // a@0 b@2: in order, one gap of 1 -> width 1, freq 1/2.
    let d = doc(&[("a", &[0]), ("b", &[2])]);
    let ordered = SpanQuery::span_near([term("a"), term("b")], 1, true);
    assert_eq!(emissions(&ordered, &d), vec![(0, 3, 1)]);
    assert_eq!(sloppy_freq(&ordered, &d), 0.5);
    // Out of order: `maxEnd - start` = 3 -> freq 1/4.
    let unordered = SpanQuery::span_near([term("b"), term("a")], 1, false);
    assert_eq!(emissions(&unordered, &d), vec![(0, 3, 3)]);
    assert_eq!(sloppy_freq(&unordered, &d), 0.25);
    // Too far apart for slop 0; an empty clause list or a missing clause
    // matches nothing.
    let tight = SpanQuery::span_near([term("a"), term("b")], 0, true);
    assert!(emissions(&tight, &d).is_empty());
    assert!(emissions(&SpanQuery::span_near(std::iter::empty(), 0, true), &d).is_empty());
    let missing = SpanQuery::span_near([term("a"), term("z")], 5, false);
    assert!(emissions(&missing, &d).is_empty());
}

#[test]
fn the_freq_sum_rounds_to_float_at_every_step() {
    // Three in-order spans with a gap of 2 (1/3 each): summed in `double` and rounded
    // to `float` each time, as `freq += 1.0 / (1.0 + width)` does.
    let d = doc(&[("a", &[0, 10, 20]), ("b", &[3, 13, 23])]);
    let q = SpanQuery::span_near([term("a"), term("b")], 2, true);
    let mut want = 0.0f32;
    for _ in 0..3 {
        want = (f64::from(want) + 1.0 / 3.0) as f32;
    }
    assert_eq!(sloppy_freq(&q, &d).to_bits(), want.to_bits());
}

#[test]
fn the_query_field_is_the_first_leafs() {
    assert_eq!(span_field(&term("a")), Some("f"));
    let or = SpanQuery::span_or([SpanQuery::span_term("g", "x"), term("a")]);
    assert_eq!(span_field(&or), Some("g"));
    assert_eq!(span_field(&SpanQuery::span_or(std::iter::empty())), None);
}
/// A segment without the query's field and without a `.pos` file (an
/// OpenSearch tombstone segment) matches nothing, although the
/// reader-wide statistics say other segments have the terms.
#[test]
fn a_segment_without_the_field_matches_nothing_without_pos_input() {
    let fields = lucene_codecs::blocktree::BlockTreeFields::empty();
    let mut global = crate::GlobalStats::new();
    let stats = crate::CollectionStats {
        doc_freq: 3,
        doc_count: 5,
        total_term_freq: 4,
        max_doc: 9,
        sum_total_term_freq: 20,
        sum_doc_freq: 12,
    };
    global.insert_term("f", "a", stats);
    global.insert_term("f", "b", stats);
    let q = SpanQuery::span_near([term("a"), term("b")], 1, true);
    let hits = resolve_span(&fields, None, None, None, None, None, Some(&global), &q).unwrap();
    assert!(hits.is_empty());
}

/// A leaf's positions, payloads and occurrences, over the spans
/// fixture's real postings: a lazy cursor's, a pulsed singleton's and an
/// absent term's.
#[test]
fn leaf_positions_read_positions_payloads_and_occurrences() {
    let dir = std::path::Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/data/spans/index"
    ));
    let reader =
        crate::directory_reader::DirectoryReader::open(&lucene_store::FsDirectory::open(dir))
            .unwrap();
    let opened = reader.open_segments().unwrap();
    let segments = opened.as_open_segments();
    for seg in &segments {
        let ctx = LeafContext {
            fields: seg.fields,
            doc_in: seg.doc_in,
            pos_in: seg.pos_in,
            pay_in: seg.pay_in,
            live_docs: seg.live_docs,
            points: None,
            norms: None,
            global: None,
            max_doc: seg.max_doc,
            cache: None,
            reader: None,
            similarity: seg.similarity,
        };
        let pos_in = ctx.pos_in.unwrap();
        let ft = ctx.fields.field("pay").unwrap();
        for term in [b"apple".as_slice(), b"zeta", b"nosuch"] {
            let mut leaf = LeafPositions::open(&ctx, pos_in, "pay", term).unwrap();
            let mut streamed = LeafPositions::open(&ctx, pos_in, "pay", term).unwrap();
            let streams = streamed.stream_payloads(&ctx);
            assert_eq!(streams, leaf.is_lazy(), "{term:?}");
            let mut doc = leaf.next_doc(-1).unwrap();
            let mut out = Vec::new();
            while doc != NO_MORE_DOCS {
                assert_eq!(streamed.advance(doc).unwrap(), doc);
                leaf.occurrences_at(&ctx, ft, term, doc, &mut out).unwrap();
                let want = ft
                    .occurrences_for_doc(term, ctx.doc_in, pos_in, ctx.pay_in, doc)
                    .unwrap()
                    .unwrap();
                assert_eq!(out, want, "{term:?} doc {doc}");
                if streams {
                    for o in &want {
                        let (p, payload) = streamed.next_position_with_payload().unwrap();
                        assert_eq!(p, o.position);
                        assert_eq!(payload.unwrap_or_default(), o.payload.as_slice());
                    }
                } else {
                    assert!(streamed.next_position().is_err());
                    assert!(streamed.next_position_with_payload().is_err());
                }
                // Not on the document asked: nothing.
                leaf.occurrences_at(&ctx, ft, term, doc + 1, &mut out)
                    .unwrap();
                assert!(out.is_empty());
                doc = leaf.next_doc(doc).unwrap();
            }
        }
    }
}
