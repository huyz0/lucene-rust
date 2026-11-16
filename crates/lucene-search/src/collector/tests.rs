#![cfg(test)]

/// A sorted list turns into a heap on the first insert into a full list
/// that lands more than `SHIFT_TO_HEAP` places from the end, and stays a
/// list while inserts land closer; queues up to `ALWAYS_SORTED` stay a
/// list whatever the inserts.
#[test]
fn sorted_list_turns_into_heap_only_for_far_inserts() {
    let top_n = 50;
    // Scores 1000, 999, ..., 951 for docs 0..50.
    let full = || {
        let mut c = TopDocsCollector::new(top_n);
        for doc in 0..50 {
            ScoringCollector::collect(&mut c, doc, 1000.0 - doc as f32);
        }
        assert!(!c.heap_ordered());
        c
    };
    // After the worst (951) leaves, 959.5 lands at index 41 of 50:
    // `SHIFT_TO_HEAP` places from the end.
    let mut at_limit = full();
    ScoringCollector::collect(&mut at_limit, 50, 959.5);
    assert!(!at_limit.heap_ordered());
    // 960.5 lands one place further.
    let mut past_limit = full();
    ScoringCollector::collect(&mut past_limit, 50, 960.5);
    assert!(past_limit.heap_ordered());
    for c in [&at_limit, &past_limit] {
        let got = c.top_docs();
        assert_eq!(got.len(), top_n);
        assert!(got.windows(2).all(|w| rank_order(&w[0], &w[1]).is_gt()));
        assert!(got.iter().any(|h| h.doc_id == 50));
        assert!(got.iter().all(|h| h.doc_id != 49));
    }

    // Hits that each beat the worst by a little land at the end every
    // time: the list stays a list.
    let mut near_end = full();
    for k in 0..500 {
        ScoringCollector::collect(&mut near_end, 50 + k, 951.0 + (k + 1) as f32 / 1000.0);
    }
    assert!(!near_end.heap_ordered());
    assert_eq!(near_end.top_docs()[top_n - 1].doc_id, 549);

    let mut rising = TopDocsCollector::new(top_n);
    for doc in 0..1000 {
        ScoringCollector::collect(&mut rising, doc, doc as f32);
    }
    assert!(rising.heap_ordered());
    let got: Vec<i32> = rising.top_docs().iter().map(|h| h.doc_id).collect();
    let want: Vec<i32> = (950..1000).rev().collect();
    assert_eq!(got, want);

    // Rising scores into a small queue stay a list.
    let mut small = TopDocsCollector::new(ALWAYS_SORTED);
    for doc in 0..1000 {
        ScoringCollector::collect(&mut small, doc, doc as f32);
    }
    assert!(!small.heap_ordered());
    assert_eq!(small.top_docs()[0].doc_id, 999);
}

/// The collector keeps the same hits, in the same order, as a full sort of
/// the stream, as a sorted list and as a heap; with score ties, `-inf`,
/// NaN (heap sizes) and out-of-order doc ids, and read between inserts.
#[test]
fn heap_ordered_queue_keeps_the_sorted_top_n() {
    let mut x: u64 = 0x9e37_79b9_7f4a_7c15;
    let mut next = || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x
    };
    for &top_n in &[
        1,
        2,
        10,
        ALWAYS_SORTED,
        ALWAYS_SORTED + 1,
        50,
        SORTED_MAX,
        SORTED_MAX + 1,
        100,
        1000,
    ] {
        let mut stream: Vec<ScoreDoc> = (0..5000)
            .map(|doc_id| {
                let r = next();
                // A small queue full of NaN keeps the last NaN, not a
                // sorted top (Java's quirk, tested on its own), so the
                // sorted-list sizes see no NaN.
                let score = match r % 97 {
                    0 if top_n > SORTED_MAX => f32::NAN,
                    1 => f32::NEG_INFINITY,
                    _ => (r % 300) as f32 / 7.0,
                };
                ScoreDoc { doc_id, score }
            })
            .collect();
        // Out of order too: the tie-break is by doc id, not arrival.
        for i in (1..stream.len()).rev() {
            let j = (next() % (i as u64 + 1)) as usize;
            stream.swap(i, j);
        }
        let mut c = TopDocsCollector::new(top_n);
        for (i, h) in stream.iter().enumerate() {
            ScoringCollector::collect(&mut c, h.doc_id, h.score);
            if i == 2500 {
                assert!(c
                    .top_docs()
                    .windows(2)
                    .all(|w| rank_order(&w[0], &w[1]).is_gt()));
            }
        }
        let mut want: Vec<ScoreDoc> = stream
            .iter()
            .filter(|h| h.score != f32::NEG_INFINITY)
            .map(|h| ScoreDoc {
                doc_id: h.doc_id,
                score: canonical_score(h.score),
            })
            .collect();
        want.sort_by(|a, b| rank_order(b, a));
        want.truncate(top_n);
        let got = c.top_docs();
        assert_eq!(got.len(), want.len());
        for (g, w) in got.iter().zip(&want) {
            assert_eq!(g.doc_id, w.doc_id);
            assert_eq!(g.score.to_bits(), w.score.to_bits());
        }
    }
}

/// `collect_range`'s default is a `collect` per document, and a
/// `LeafCollector` hands its range on shifted by the doc base.
#[test]
fn a_range_is_a_collect_per_document() {
    struct Docs(Vec<(i32, f32)>);
    impl ScoringCollector for Docs {
        fn collect(&mut self, doc_id: i32, score: f32) {
            self.0.push((doc_id, score));
        }
    }
    let mut docs = Docs(Vec::new());
    LeafCollector::new(&mut docs, 10).collect_range(0, 3, 0.5);
    docs.collect_range(4, 4, 1.0);
    assert_eq!(docs.0, vec![(10, 0.5), (11, 0.5), (12, 0.5)]);
}

/// `collect_many` against `collect` one document at a time, from the same
/// state: the kept hits, the total and its relation, and how far the
/// batch went (the rest collected one by one). Streams with ties, a NaN,
/// a queue filling mid-batch, thresholds at, below and past the batch, a
/// `searchAfter` page and a shared bar (declined).
#[test]
fn collect_many_is_collect_per_document() {
    use std::sync::Arc;
    // First a negative NaN -- x86's run-time NaN -- which Java
    // canonicalizes to the top of the order: kept while the queue fills,
    // and while it is the only hit it is the worst kept one, which every
    // hit after it replaces.
    let mut scores = vec![f32::from_bits(0xFFC0_0000)];
    scores.extend([1.0f32, 3.0, 2.0, 2.0, 5.0, 0.5, 2.0, f32::NAN, 4.0, 1.0]);
    scores.extend((0..30).map(|i| (i % 7) as f32 * 0.5));
    // Long runs past a full queue, as a keyword's equal scores arrive:
    // losers, a NaN and a winner deep inside a run of them.
    scores.extend(std::iter::repeat_n(1.0, 60));
    scores.push(f32::NAN);
    scores.extend(std::iter::repeat_n(1.0, 13));
    scores.push(9.0);
    scores.extend(std::iter::repeat_n(1.0, 20));

    let docs: Vec<i32> = (0..scores.len() as i32).map(|d| d * 3).collect();
    let bases = [0, 1000];
    let mut makers: Vec<Box<dyn Fn() -> TopDocsCollector>> = Vec::new();
    for top_n in [0usize, 1, 3, 10, 50] {
        for threshold in [0u64, 2, 5, 10, 39, 40, u64::MAX] {
            makers.push(Box::new(move || {
                TopDocsCollector::with_total_hits_threshold(top_n, threshold)
            }));
        }
        makers.push(Box::new(move || {
            TopDocsCollector::with_total_hits_threshold(top_n, 20).with_after(ScoreDoc {
                doc_id: 30,
                score: 2.0,
            })
        }));
        makers.push(Box::new(move || {
            TopDocsCollector::with_total_hits_threshold(top_n, 20)
                .with_shared_max_score(Arc::new(MaxScoreAccumulator::new()), 0)
        }));
    }
    let mut batched = 0;
    for make in &makers {
        for &base in &bases {
            for split in [0, 1, 4, docs.len()] {
                // A prefix one by one first, so batches start from a
                // partly or wholly filled queue.
                let (mut one, mut many) = (make(), make());
                for c in [&mut one, &mut many] {
                    for i in 0..split {
                        c.collect(base + docs[i], scores[i]);
                    }
                }
                for i in split..docs.len() {
                    one.collect(base + docs[i], scores[i]);
                }
                let n = many.collect_many(&docs[split..], &scores[split..], base);
                batched += n;
                for i in split + n..docs.len() {
                    many.collect(base + docs[i], scores[i]);
                }
                let key = |c: &TopDocsCollector| {
                    let hits: Vec<(i32, u32)> = c
                        .top_docs()
                        .iter()
                        .map(|h| (h.doc_id, h.score.to_bits()))
                        .collect();
                    let t = c.total_hits();
                    (hits, t.value, t.relation)
                };
                assert_eq!(
                    key(&many),
                    key(&one),
                    "split {split} base {base} batched {n}"
                );
            }
        }
    }
    assert!(batched > 0, "some batch went through collect_many");
}
use super::*;

fn field_docs(v: &[(i32, i64)]) -> Vec<FieldValueDoc> {
    v.iter()
        .map(|&(doc_id, value)| FieldValueDoc { doc_id, value })
        .collect()
}

#[test]
fn top_field_collector_empty_input_yields_no_hits() {
    let c = TopFieldCollector::new(3, SortDirection::Ascending);
    assert!(c.top_docs().is_empty());
}

#[test]
fn top_field_collector_top_n_zero_keeps_nothing() {
    let mut c = TopFieldCollector::new(0, SortDirection::Ascending);
    c.offer(1, 5);
    c.offer(2, 9);
    assert!(c.top_docs().is_empty());
}

#[test]
fn top_field_collector_ascending_orders_smallest_first() {
    let mut c = TopFieldCollector::new(5, SortDirection::Ascending);
    c.offer(1, 30);
    c.offer(2, 10);
    c.offer(3, 20);
    assert_eq!(
        c.top_docs().to_vec(),
        field_docs(&[(2, 10), (3, 20), (1, 30)])
    );
}

#[test]
fn top_field_collector_descending_orders_largest_first() {
    let mut c = TopFieldCollector::new(5, SortDirection::Descending);
    c.offer(1, 30);
    c.offer(2, 10);
    c.offer(3, 20);
    assert_eq!(
        c.top_docs().to_vec(),
        field_docs(&[(1, 30), (3, 20), (2, 10)])
    );
}

#[test]
fn top_field_collector_truncates_to_top_n_ascending() {
    let mut c = TopFieldCollector::new(2, SortDirection::Ascending);
    c.offer(1, 30);
    c.offer(2, 10);
    c.offer(3, 20);
    // Worst (doc 1, value 30) must be evicted, keeping the two smallest.
    assert_eq!(c.top_docs().to_vec(), field_docs(&[(2, 10), (3, 20)]));
}

#[test]
fn top_field_collector_truncates_to_top_n_descending() {
    let mut c = TopFieldCollector::new(2, SortDirection::Descending);
    c.offer(1, 30);
    c.offer(2, 10);
    c.offer(3, 20);
    // Worst (doc 2, value 10) must be evicted, keeping the two largest.
    assert_eq!(c.top_docs().to_vec(), field_docs(&[(1, 30), (3, 20)]));
}

#[test]
fn top_field_collector_tie_break_prefers_lower_doc_id() {
    let mut c = TopFieldCollector::new(2, SortDirection::Ascending);
    c.offer(5, 2);
    c.offer(2, 2);
    c.offer(9, 2);
    assert_eq!(c.top_docs().to_vec(), field_docs(&[(2, 2), (5, 2)]));
}

#[test]
fn field_rank_order_ties_break_by_ascending_doc_id() {
    let a = FieldValueDoc {
        doc_id: 1,
        value: 5,
    };
    let b = FieldValueDoc {
        doc_id: 2,
        value: 5,
    };
    assert_eq!(
        field_rank_order(&a, &b, SortDirection::Ascending),
        std::cmp::Ordering::Greater
    );
    assert_eq!(
        field_rank_order(&a, &b, SortDirection::Descending),
        std::cmp::Ordering::Greater
    );
}

#[test]
fn vec_collector_collects_in_call_order() {
    let mut c = VecCollector::default();
    c.collect(3);
    c.collect(7);
    assert_eq!(c.docs, vec![3, 7]);
}

#[test]
fn count_collector_counts_calls_not_values() {
    let mut c = CountCollector::default();
    c.collect(0);
    c.collect(0);
    c.collect(5);
    assert_eq!(c.count, 3);
}

fn score_docs(v: &[(i32, f32)]) -> Vec<ScoreDoc> {
    v.iter()
        .map(|&(doc_id, score)| ScoreDoc { doc_id, score })
        .collect()
}

#[test]
fn top_docs_collector_empty_input_yields_no_hits() {
    let c = TopDocsCollector::new(3);
    assert!(c.top_docs().is_empty());
}

#[test]
fn top_docs_collector_top_n_zero_keeps_nothing() {
    let mut c = TopDocsCollector::new(0);
    c.collect(1, 5.0);
    c.collect(2, 9.0);
    assert!(c.top_docs().is_empty());
}

#[test]
fn top_docs_collector_fewer_than_n_keeps_all_sorted_by_score_desc() {
    let mut c = TopDocsCollector::new(5);
    c.collect(1, 1.0);
    c.collect(2, 3.0);
    c.collect(3, 2.0);
    assert_eq!(
        c.top_docs().to_vec(),
        score_docs(&[(2, 3.0), (3, 2.0), (1, 1.0)])
    );
}

#[test]
fn top_docs_collector_exactly_n_keeps_all_sorted() {
    let mut c = TopDocsCollector::new(3);
    c.collect(1, 1.0);
    c.collect(2, 3.0);
    c.collect(3, 2.0);
    assert_eq!(
        c.top_docs().to_vec(),
        score_docs(&[(2, 3.0), (3, 2.0), (1, 1.0)])
    );
}

#[test]
fn top_docs_collector_more_than_n_evicts_the_worst() {
    let mut c = TopDocsCollector::new(2);
    c.collect(1, 1.0);
    c.collect(2, 3.0);
    c.collect(3, 2.0);
    // 1.0 (doc 1) is the worst score and gets evicted once a better candidate
    // (doc 3, score 2.0) arrives.
    assert_eq!(c.top_docs().to_vec(), score_docs(&[(2, 3.0), (3, 2.0)]));
}

#[test]
fn top_docs_collector_candidate_worse_than_all_kept_hits_is_dropped() {
    let mut c = TopDocsCollector::new(2);
    c.collect(1, 5.0);
    c.collect(2, 4.0);
    c.collect(3, 1.0); // worse than both kept hits -- must not be kept.
    assert_eq!(c.top_docs().to_vec(), score_docs(&[(1, 5.0), (2, 4.0)]));
}

#[test]
fn top_docs_collector_tie_break_prefers_lower_doc_id() {
    let mut c = TopDocsCollector::new(2);
    c.collect(5, 2.0);
    c.collect(2, 2.0);
    c.collect(9, 2.0);
    // All tied at score 2.0 -- lowest doc IDs (2, 5) must win over doc 9.
    assert_eq!(c.top_docs().to_vec(), score_docs(&[(2, 2.0), (5, 2.0)]));
}

#[test]
fn top_docs_collector_tie_break_eviction_prefers_lower_doc_id() {
    let mut c = TopDocsCollector::new(1);
    c.collect(9, 3.0);
    c.collect(2, 3.0); // ties doc 9 on score; lower doc id must win.
    assert_eq!(c.top_docs().to_vec(), score_docs(&[(2, 3.0)]));
}

/// The kept hits as `(doc, score bits)`, best first.
fn kept(c: &TopDocsCollector) -> Vec<(i32, u32)> {
    c.top_docs()
        .iter()
        .map(|h| (h.doc_id, h.score.to_bits()))
        .collect()
}

/// `TopScoreDocCollector.collect`'s `if (score <= topScore)` reject,
/// negated, is what decides a NaN: a NaN hit always enters (it ranks above
/// `+inf` once `floatToIntBits` has canonicalized it), and a queue whose
/// worst hit is a NaN lets every hit in, so with one slot the *last* NaN
/// document wins. No threshold is pushed for a NaN (`Math.nextUp(NaN) >
/// minCompetitiveScore` is false), so the count stays exact.
#[test]
fn a_nan_worst_hit_is_replaced_by_every_later_hit_as_in_java() {
    let neg_nan = f32::from_bits(0xFFC0_0000);
    let mut one = TopDocsCollector::new(1);
    for doc in 0..5 {
        one.collect(doc, if doc % 2 == 0 { f32::NAN } else { neg_nan });
    }
    assert_eq!(kept(&one), vec![(4, 0x7fc0_0000)]);
    assert_eq!(one.total_hits().relation, TotalHitsRelation::EqualTo);
    // No bar was ever pushed: `-inf`, which prunes nothing.
    assert_eq!(one.min_competitive_score(), Some(f32::NEG_INFINITY));

    // Three slots: the first two NaN hits stay, the last slot churns.
    let mut three = TopDocsCollector::new(3);
    for doc in 0..6 {
        three.collect(doc, neg_nan);
    }
    assert_eq!(
        kept(&three),
        vec![(0, 0x7fc0_0000), (1, 0x7fc0_0000), (5, 0x7fc0_0000)]
    );
    assert_eq!(three.total_hits().relation, TotalHitsRelation::EqualTo);
    // A finite hit after a queue full of NaN replaces the worst NaN and
    // becomes the worst kept hit -- now a real threshold.
    three.collect(6, 1.5);
    assert_eq!(
        kept(&three),
        vec![(0, 0x7fc0_0000), (1, 0x7fc0_0000), (6, 1.5f32.to_bits())]
    );
    assert_eq!(three.min_competitive_score(), Some(1.5));
    assert_eq!(
        three.total_hits().relation,
        TotalHitsRelation::GreaterThanOrEqualTo
    );
    // And a NaN hit then evicts it again; the bar already pushed stays.
    three.collect(7, f32::NAN);
    assert_eq!(
        kept(&three),
        vec![(0, 0x7fc0_0000), (1, 0x7fc0_0000), (7, 0x7fc0_0000)]
    );
    assert_eq!(three.min_competitive_score(), Some(1.5));
    assert_eq!(three.total_hits().value, 8);
}

/// A finite queue: a NaN evicts the worst finite hit and ranks first.
#[test]
fn a_nan_hit_evicts_the_worst_finite_hit_and_ranks_first() {
    let mut c = TopDocsCollector::new(2);
    c.collect(0, 3.0);
    c.collect(1, 2.0);
    c.collect(2, f32::from_bits(0xFFC0_0000));
    assert_eq!(kept(&c), vec![(2, 0x7fc0_0000), (0, 3.0f32.to_bits())]);
    c.collect(3, f32::INFINITY);
    assert_eq!(
        kept(&c),
        vec![(2, 0x7fc0_0000), (3, f32::INFINITY.to_bits())]
    );
}

/// While the queue is not full its top is `LEAST_COMPETITIVE_CODE`, whose
/// score is `-inf`, and `-inf <= -inf`: Java never keeps a `-inf` hit,
/// though it counts it.
#[test]
fn a_negative_infinity_hit_is_counted_but_never_kept() {
    let mut c = TopDocsCollector::new(3);
    c.collect(0, f32::NEG_INFINITY);
    c.collect(1, 1.0);
    c.collect(2, f32::NEG_INFINITY);
    assert_eq!(kept(&c), vec![(1, 1.0f32.to_bits())]);
    assert_eq!(c.total_hits().value, 3);
    assert_eq!(c.total_hits().relation, TotalHitsRelation::EqualTo);
}

/// `count_losing_hits` declines a NaN on either side: none of those hits
/// would lose under `score <= topScore`.
#[test]
fn count_losing_hits_declines_a_nan() {
    let mut c = TopDocsCollector::new(1);
    c.collect(0, 2.0);
    assert!(!ScoringCollector::count_losing_hits(&mut c, 4, f32::NAN));
    assert!(ScoringCollector::count_losing_hits(&mut c, 4, 2.0));
    assert_eq!(c.total_hits().value, 5);
    let mut n = TopDocsCollector::new(1);
    n.collect(0, f32::NAN);
    assert!(!ScoringCollector::count_losing_hits(&mut n, 4, 1.0));
    assert_eq!(n.total_hits().value, 1);
}

/// `DocScoreEncoder.encode` goes through `floatToIntBits`: both NaN signs
/// encode alike, above `+inf`.
#[test]
fn doc_score_encoder_canonicalizes_nan() {
    let neg = doc_score_encoder::encode(3, f32::from_bits(0xFFC0_0000));
    let pos = doc_score_encoder::encode(3, f32::NAN);
    assert_eq!(neg, pos);
    assert!(pos > doc_score_encoder::encode(3, f32::INFINITY));
    assert_eq!(doc_score_encoder::to_score(pos).to_bits(), 0x7fc0_0000);
    assert_eq!(canonical_score(-0.0).to_bits(), (-0.0f32).to_bits());
}

#[test]
fn rank_order_orders_by_score_desc_then_doc_id_asc() {
    let a = ScoreDoc {
        doc_id: 1,
        score: 5.0,
    };
    let b = ScoreDoc {
        doc_id: 2,
        score: 5.0,
    };
    assert_eq!(rank_order(&a, &b), std::cmp::Ordering::Greater);
    assert_eq!(rank_order(&b, &a), std::cmp::Ordering::Less);
    let c = ScoreDoc {
        doc_id: 3,
        score: 6.0,
    };
    assert_eq!(rank_order(&c, &a), std::cmp::Ordering::Greater);
}

// --- CollapsingCollector ---

#[test]
fn collapsing_collector_empty_input_yields_no_hits() {
    let c = CollapsingCollector::new(3);
    assert!(c.top_docs().is_empty());
}

#[test]
fn collapsing_collector_top_n_zero_keeps_nothing() {
    let mut c = CollapsingCollector::new(0);
    c.offer(1, 5.0, Some(1));
    c.offer(2, 9.0, Some(2));
    assert!(c.top_docs().is_empty());
}

#[test]
fn collapsing_collector_keeps_only_highest_scoring_doc_per_key() {
    let mut c = CollapsingCollector::new(10);
    // Three docs share collapse-key 1; only the highest score (doc 2)
    // should survive.
    c.offer(1, 1.0, Some(1));
    c.offer(2, 5.0, Some(1));
    c.offer(3, 3.0, Some(1));
    let docs = c.top_docs();
    assert_eq!(
        docs,
        vec![ScoreDoc {
            doc_id: 2,
            score: 5.0
        }]
    );
}

#[test]
fn collapsing_collector_distinct_keys_all_appear_subject_to_top_n() {
    let mut c = CollapsingCollector::new(10);
    c.offer(1, 1.0, Some(1));
    c.offer(2, 2.0, Some(2));
    c.offer(3, 3.0, Some(3));
    let docs = c.top_docs();
    assert_eq!(
        docs,
        vec![
            ScoreDoc {
                doc_id: 3,
                score: 3.0
            },
            ScoreDoc {
                doc_id: 2,
                score: 2.0
            },
            ScoreDoc {
                doc_id: 1,
                score: 1.0
            },
        ]
    );
}

#[test]
fn collapsing_collector_truncates_group_winners_to_top_n() {
    let mut c = CollapsingCollector::new(2);
    c.offer(1, 1.0, Some(1));
    c.offer(2, 2.0, Some(2));
    c.offer(3, 3.0, Some(3));
    let docs = c.top_docs();
    assert_eq!(
        docs,
        vec![
            ScoreDoc {
                doc_id: 3,
                score: 3.0
            },
            ScoreDoc {
                doc_id: 2,
                score: 2.0
            },
        ]
    );
}

#[test]
fn collapsing_collector_missing_key_docs_each_survive_as_own_group() {
    let mut c = CollapsingCollector::new(10);
    // Two docs with no collapse-key value -- both must survive
    // independently (this port's EXPAND-equivalent null-group policy),
    // not collapse together.
    c.offer(1, 1.0, None);
    c.offer(2, 2.0, None);
    c.offer(3, 3.0, Some(9));
    let mut docs = c.top_docs();
    docs.sort_by_key(|d| d.doc_id);
    assert_eq!(
        docs,
        vec![
            ScoreDoc {
                doc_id: 1,
                score: 1.0
            },
            ScoreDoc {
                doc_id: 2,
                score: 2.0
            },
            ScoreDoc {
                doc_id: 3,
                score: 3.0
            },
        ]
    );
}

#[test]
fn collapsing_collector_replaces_group_winner_on_better_candidate() {
    let mut c = CollapsingCollector::new(10);
    c.offer(1, 5.0, Some(1));
    c.offer(2, 1.0, Some(1)); // worse -- must not replace doc 1.
    let docs = c.top_docs();
    assert_eq!(
        docs,
        vec![ScoreDoc {
            doc_id: 1,
            score: 5.0
        }]
    );
}

#[test]
fn collapsing_collector_tie_break_prefers_lower_doc_id_within_group() {
    let mut c = CollapsingCollector::new(10);
    c.offer(9, 3.0, Some(1));
    c.offer(2, 3.0, Some(1)); // ties on score -- lower doc id wins.
    let docs = c.top_docs();
    assert_eq!(
        docs,
        vec![ScoreDoc {
            doc_id: 2,
            score: 3.0
        }]
    );
}

// ---- `ScoreMode` / `TotalHits` (`org.apache.lucene.search.ScoreMode`,
// `TotalHits`, `TopScoreDocCollector.totalHitsThreshold`) ----

#[test]
fn score_mode_predicates_match_javas_enum_table() {
    // `ScoreMode`'s Java constructor arguments are `(isExhaustive,
    // needsScores)`: COMPLETE(true,true), COMPLETE_NO_SCORES(true,false),
    // TOP_SCORES(false,true), TOP_DOCS(false,false),
    // TOP_DOCS_WITH_SCORES(false,true). Asserted as a table so a future
    // edit to either predicate has to disagree with Java on purpose.
    let table = [
        (ScoreMode::Complete, true, true),
        (ScoreMode::CompleteNoScores, true, false),
        (ScoreMode::TopScores, false, true),
        (ScoreMode::TopDocs, false, false),
        (ScoreMode::TopDocsWithScores, false, true),
    ];
    for (mode, exhaustive, needs_scores) in table {
        assert_eq!(mode.is_exhaustive(), exhaustive, "{mode:?}");
        assert_eq!(mode.needs_scores(), needs_scores, "{mode:?}");
    }
}

#[test]
fn total_hits_counts_every_collected_doc_not_just_the_kept_ones() {
    let mut c = TopDocsCollector::new(2);
    for doc in 0..10 {
        c.collect(doc, doc as f32);
    }
    assert_eq!(c.top_docs().len(), 2);
    assert_eq!(c.total_hits().value, 10);
}

#[test]
fn total_hits_is_exact_until_a_competitive_threshold_is_published() {
    // Threshold 0 (this port's default): the queue fills at the second
    // document, a threshold is published, and the count stops being exact.
    let mut c = TopDocsCollector::new(2);
    c.collect(0, 1.0);
    assert_eq!(c.total_hits().relation, TotalHitsRelation::EqualTo);
    assert_eq!(c.min_competitive_score(), None, "queue not full yet");
    c.collect(1, 2.0);
    assert_eq!(c.min_competitive_score(), Some(1.0));
    assert_eq!(
        c.total_hits().relation,
        TotalHitsRelation::GreaterThanOrEqualTo
    );
}

#[test]
fn a_total_hits_threshold_keeps_the_count_exact_and_delays_pruning() {
    // `TopScoreDocCollectorManager(numHits, totalHitsThreshold)`: no
    // threshold is published while `totalHits <= totalHitsThreshold`, so
    // the first five counts are exact even though the queue filled at two.
    let mut c = TopDocsCollector::with_total_hits_threshold(2, 5);
    for doc in 0..5 {
        c.collect(doc, doc as f32);
        assert_eq!(c.min_competitive_score(), None, "doc {doc}");
        assert_eq!(c.total_hits().relation, TotalHitsRelation::EqualTo);
    }
    c.collect(5, 5.0);
    assert!(c.min_competitive_score().is_some());
    assert_eq!(
        c.total_hits().relation,
        TotalHitsRelation::GreaterThanOrEqualTo
    );
    assert_eq!(c.total_hits().value, 6);
}

#[test]
fn an_infinite_threshold_is_complete_mode_and_never_prunes() {
    // Java: `totalHitsThreshold == Integer.MAX_VALUE ? COMPLETE :
    // TOP_SCORES`. In COMPLETE mode no threshold may ever be published, so
    // `totalHits` stays exact no matter how many documents arrive.
    let mut c = TopDocsCollector::with_total_hits_threshold(2, u64::MAX);
    assert_eq!(c.score_mode(), ScoreMode::Complete);
    assert!(c.score_mode().is_exhaustive());
    for doc in 0..50 {
        c.collect(doc, doc as f32);
    }
    assert_eq!(c.min_competitive_score(), None);
    assert_eq!(c.total_hits().value, 50);
    assert_eq!(c.total_hits().relation, TotalHitsRelation::EqualTo);
    assert_eq!(c.top_docs().len(), 2);
}

#[test]
fn default_collector_reports_top_scores_mode() {
    let c = TopDocsCollector::new(3);
    assert_eq!(c.score_mode(), ScoreMode::TopScores);
    assert!(!c.score_mode().is_exhaustive(), "pruning must be legal");
    assert!(c.score_mode().needs_scores());
    // Reached through the trait too, since that is what the scorers use.
    assert_eq!(ScoringCollector::score_mode(&c), ScoreMode::TopScores);
}

#[test]
fn a_collector_that_says_nothing_gets_the_conservative_default() {
    struct Bare(u32);
    impl ScoringCollector for Bare {
        fn collect(&mut self, _doc_id: i32, _score: f32) {
            self.0 += 1;
        }
    }
    let c = Bare(0);
    assert_eq!(c.score_mode(), ScoreMode::Complete);
    assert!(
        c.score_mode().is_exhaustive(),
        "no pruning without opting in"
    );
    assert_eq!(c.min_competitive_score(), None);
}

#[test]
fn pruning_threshold_is_min_competitive_score_gated_on_the_score_mode() {
    // Non-exhaustive: the two agree.
    let mut top = TopDocsCollector::new(2);
    top.collect(0, 1.0);
    top.collect(1, 2.0);
    assert_eq!(top.min_competitive_score(), Some(1.0));
    assert_eq!(ScoringCollector::pruning_threshold(&top), Some(1.0));

    // Exhaustive: a threshold may not be handed out even if one exists.
    // (`TopDocsCollector` also refuses to compute one in this mode, so
    // force the interesting case with a collector that does.)
    struct AlwaysCompetitive;
    impl ScoringCollector for AlwaysCompetitive {
        fn collect(&mut self, _doc_id: i32, _score: f32) {}
        fn min_competitive_score(&self) -> Option<f32> {
            Some(42.0)
        }
        fn score_mode(&self) -> ScoreMode {
            ScoreMode::Complete
        }
    }
    let c = AlwaysCompetitive;
    assert_eq!(c.min_competitive_score(), Some(42.0));
    assert_eq!(
        c.pruning_threshold(),
        None,
        "an exhaustive collector must not authorize pruning, whatever its \
             min competitive score says"
    );

    struct TopScoresMode;
    impl ScoringCollector for TopScoresMode {
        fn collect(&mut self, _doc_id: i32, _score: f32) {}
        fn min_competitive_score(&self) -> Option<f32> {
            Some(7.0)
        }
        fn score_mode(&self) -> ScoreMode {
            ScoreMode::TopScores
        }
    }
    assert_eq!(TopScoresMode.pruning_threshold(), Some(7.0));
}

#[test]
fn total_hits_displays_like_javas_to_string() {
    assert_eq!(
        TotalHits {
            value: 12,
            relation: TotalHitsRelation::EqualTo
        }
        .to_string(),
        "12 hits"
    );
    assert_eq!(
        TotalHits {
            value: 1000,
            relation: TotalHitsRelation::GreaterThanOrEqualTo
        }
        .to_string(),
        "1000+ hits"
    );
}

#[test]
fn a_zero_top_n_collector_still_counts_every_hit_exactly() {
    let mut c = TopDocsCollector::new(0);
    for doc in 0..7 {
        c.collect(doc, 1.0);
    }
    assert!(c.top_docs().is_empty());
    assert_eq!(c.total_hits().value, 7);
    assert_eq!(c.total_hits().relation, TotalHitsRelation::EqualTo);
}

// ---- `searchAfter` -------------------------------------------------

/// A page walk over a queue that is deliberately full of score ties, so
/// the boundary is decided by the doc-id half of the rule.
#[test]
fn after_pages_walk_the_ranking_without_repeating_or_losing_a_hit() {
    let hits: Vec<(i32, f32)> = (0..9).map(|d| (d, if d < 4 { 2.0 } else { 1.0 })).collect();
    let mut pages: Vec<Vec<ScoreDoc>> = Vec::new();
    let mut after: Option<ScoreDoc> = None;
    for _ in 0..3 {
        let mut c = TopDocsCollector::new(3);
        if let Some(a) = after {
            c = c.with_after(a);
        }
        for &(doc, score) in &hits {
            c.collect(doc, score);
        }
        let page = c.top_docs().to_vec();
        after = page.last().copied();
        pages.push(page);
    }
    let walked: Vec<i32> = pages.iter().flatten().map(|h| h.doc_id).collect();
    assert_eq!(walked, (0..9).collect::<Vec<_>>());
    // The score run boundary falls inside page 2, which is the case a
    // score-only comparison gets wrong.
    assert_eq!(
        pages[1].iter().map(|h| h.score).collect::<Vec<_>>(),
        vec![2.0, 1.0, 1.0]
    );
}

#[test]
fn a_paged_hit_is_still_counted_in_total_hits() {
    // `TopScoreDocCollector` increments `totalHits` *before* the `after`
    // test, so page 2 reports the same total page 1 did.
    let mut c = TopDocsCollector::new(2).with_after(ScoreDoc {
        doc_id: 1,
        score: 5.0,
    });
    for doc in 0..6 {
        c.collect(doc, 5.0);
    }
    assert_eq!(c.total_hits().value, 6);
    assert_eq!(
        c.top_docs().iter().map(|h| h.doc_id).collect::<Vec<_>>(),
        vec![2, 3]
    );
}

#[test]
fn after_rejects_a_tie_on_a_lower_doc_id_and_keeps_a_higher_one() {
    // The exact boundary: `score == afterScore && doc <= afterDoc`.
    let mut c = TopDocsCollector::new(5).with_after(ScoreDoc {
        doc_id: 3,
        score: 1.0,
    });
    c.collect(3, 1.0); // the `after` hit itself
    c.collect(2, 1.0); // ranked above it (tie, lower doc id)
    c.collect(4, 1.0); // ranked below it
    c.collect(0, 2.0); // outranks it on score
    c.collect(9, 0.5); // below it on score
    assert_eq!(
        c.top_docs().iter().map(|h| h.doc_id).collect::<Vec<_>>(),
        vec![4, 9]
    );
}

// ---- `MaxScoreAccumulator` -----------------------------------------

#[test]
fn the_doc_score_code_orders_exactly_as_the_hit_queue_does() {
    // The claim the whole single-atomic design rests on: integer order over
    // the code is `rank_order` over the pair.
    let pairs = [
        (0i32, f32::NEG_INFINITY),
        (0, 0.0f32),
        (5, 0.0),
        (i32::MAX, 0.0),
        (0, 1.0),
        (7, 1.0),
        (3, 12.5),
        (0, f32::MAX),
    ];
    for &(da, sa) in &pairs {
        for &(db, sb) in &pairs {
            let code_order =
                doc_score_encoder::encode(da, sa).cmp(&doc_score_encoder::encode(db, sb));
            let rank = rank_order(
                &ScoreDoc {
                    doc_id: da,
                    score: sa,
                },
                &ScoreDoc {
                    doc_id: db,
                    score: sb,
                },
            );
            assert_eq!(code_order, rank, "({da},{sa}) vs ({db},{sb})");
        }
    }
}

#[test]
fn the_doc_score_code_round_trips_both_halves() {
    for &(doc, score) in &[
        (0i32, 0.0f32),
        (1, 1.0),
        (123_456, 0.25),
        (i32::MAX, f32::MAX),
        (0, f32::NEG_INFINITY),
    ] {
        let code = doc_score_encoder::encode(doc, score);
        assert_eq!(doc_score_encoder::doc_id(code), doc, "doc {doc}");
        assert_eq!(
            doc_score_encoder::to_score(code).to_bits(),
            score.to_bits(),
            "score {score}"
        );
    }
}

#[test]
fn an_empty_accumulator_publishes_no_threshold() {
    let acc = MaxScoreAccumulator::new();
    assert_eq!(acc.threshold_for(0), None);
    assert_eq!(MaxScoreAccumulator::default().threshold_for(0), None);
}

#[test]
fn the_accumulator_keeps_the_best_pair_and_tie_breaks_on_the_doc_base() {
    let acc = MaxScoreAccumulator::new();
    acc.accumulate(100, 1.0);
    acc.accumulate(50, 0.5); // worse score: ignored
                             // Same score, higher doc id: also worse.
    acc.accumulate(200, 1.0);
    // A leaf starting at or after doc 100 cannot win a tie on 1.0, so it
    // may demand strictly more -- expressed under this crate's
    // `bound <= threshold` rule as the score itself.
    assert_eq!(acc.threshold_for(100), Some(1.0));
    assert_eq!(acc.threshold_for(1_000), Some(1.0));
    // A leaf that may hold smaller doc ids could still win a tie, so its
    // threshold is one ULP lower.
    assert_eq!(acc.threshold_for(0), Some(1.0f32.next_down()));
    assert_eq!(acc.threshold_for(99), Some(1.0f32.next_down()));
}

#[test]
fn a_shared_accumulator_raises_a_leafs_threshold_above_its_own_queue() {
    let acc = std::sync::Arc::new(MaxScoreAccumulator::new());
    // Another leaf has already proved 10.0 is competitive, at a doc id
    // below this leaf's base.
    acc.accumulate(0, 10.0);
    let mut c = TopDocsCollector::new(2).with_shared_max_score(std::sync::Arc::clone(&acc), 50);
    // Nothing collected yet: the local half has no answer, the shared half
    // does.
    assert_eq!(c.min_competitive_score(), Some(10.0));
    c.collect(0, 1.0);
    c.collect(1, 2.0);
    // The local half now says 1.0; the shared half still wins.
    assert_eq!(c.min_competitive_score(), Some(10.0));
    // Once this leaf beats it, its own queue takes over -- and it has
    // published its worst kept hit into the accumulator, at global doc ids.
    c.collect(2, 20.0);
    c.collect(3, 30.0);
    assert_eq!(c.min_competitive_score(), Some(20.0));
    assert_eq!(acc.threshold_for(0), Some(20.0f32.next_down()));
    assert_eq!(
        doc_score_encoder::doc_id(acc.acc.load(std::sync::atomic::Ordering::Relaxed)),
        52,
        "the accumulated doc id is global: doc 2 plus doc_base 50"
    );
}

#[test]
fn the_accumulator_is_written_only_when_this_leafs_own_bar_rises() {
    // Java gates `minScoreAcc.accumulate` on `localMinScore >
    // minCompetitiveScore`; without that gate the atomic read-modify-write
    // is on the per-document path. What must not happen is the gate
    // *suppressing* a publish the accumulator needed.
    let acc = std::sync::Arc::new(MaxScoreAccumulator::new());
    let mut c = TopDocsCollector::new(2).with_shared_max_score(std::sync::Arc::clone(&acc), 0);
    // Descending scores: the queue fills, then every later hit loses, so the
    // bar rises exactly once.
    for (doc, score) in [(0i32, 9.0f32), (1, 8.0), (2, 7.0), (3, 6.0), (4, 5.0)] {
        c.collect(doc, score);
    }
    assert_eq!(c.min_competitive_score(), Some(8.0));
    assert_eq!(acc.threshold_for(1), Some(8.0));

    // Ascending scores: the bar rises on nearly every hit, and the
    // accumulator must end on the last one, not the first.
    let acc = std::sync::Arc::new(MaxScoreAccumulator::new());
    let mut c = TopDocsCollector::new(2).with_shared_max_score(std::sync::Arc::clone(&acc), 0);
    for (doc, score) in [(0i32, 1.0f32), (1, 2.0), (2, 3.0), (3, 4.0), (4, 5.0)] {
        c.collect(doc, score);
    }
    assert_eq!(c.min_competitive_score(), Some(4.0));
    assert_eq!(acc.threshold_for(4), Some(4.0));
}

#[test]
fn a_shared_threshold_alone_makes_the_hit_count_a_lower_bound() {
    // The collector's own queue is not full, so it has published nothing --
    // but another leaf has, and a scorer is allowed to skip against it, so
    // `totalHits` can no longer claim to be exact. Java's
    // `updateGlobalMinCompetitiveScore` sets the relation for exactly this
    // reason.
    let acc = std::sync::Arc::new(MaxScoreAccumulator::new());
    acc.accumulate(0, 100.0);
    let mut c = TopDocsCollector::new(5).with_shared_max_score(acc, 10);
    c.collect(0, 1.0);
    assert!(c.top_docs().len() < 5, "the queue is deliberately not full");
    assert_eq!(c.local_min_competitive_score(), None);
    assert_eq!(c.min_competitive_score(), Some(100.0));
    assert_eq!(
        c.total_hits().relation,
        TotalHitsRelation::GreaterThanOrEqualTo
    );
}

#[test]
fn an_exhaustive_collector_publishes_nothing_into_the_accumulator() {
    // `total_hits_threshold == u64::MAX` is `ScoreMode::COMPLETE`, and a
    // collector promising an exact count must not let any leaf skip.
    let acc = std::sync::Arc::new(MaxScoreAccumulator::new());
    let mut c = TopDocsCollector::with_total_hits_threshold(1, u64::MAX)
        .with_shared_max_score(std::sync::Arc::clone(&acc), 0);
    for doc in 0..5 {
        c.collect(doc, 1.0);
    }
    assert_eq!(acc.threshold_for(0), None);
    assert_eq!(c.pruning_threshold(), None);
    assert_eq!(c.total_hits().relation, TotalHitsRelation::EqualTo);
}
