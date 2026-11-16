#![cfg(test)]
use super::*;

fn t(field: &str, bytes: &str) -> Term {
    Term::new(field, bytes.as_bytes())
}

#[test]
fn sequence_numbers_start_at_one_and_increase_by_one() {
    let mut q = DeleteQueue::new();
    assert_eq!(q.next_sequence_number(), 1);
    assert_eq!(q.next_sequence_number(), 2);
    assert_eq!(q.add_term_deletes(&[t("id", "a")], 0), 3);
    assert_eq!(q.add_query_deletes(&[DeleteQuery::MatchAll], 0), 4);
    assert_eq!(q.last_sequence_number(), 4);
}

#[test]
fn skip_sequence_numbers_leaves_a_gap() {
    let mut q = DeleteQueue::new();
    assert_eq!(q.next_sequence_number(), 1);
    q.skip_sequence_numbers(10);
    assert_eq!(q.next_sequence_number(), 12);
}

/// A sequence number is only meaningful as an *order*, so a jump that
/// rewinds the counter hands the same number out twice, and one that
/// overflows it hands out a negative that sorts below every number already
/// issued. Java's `nextSeqNo.addAndGet(jump)` does both silently; the bare
/// `+=` here panicked on the second in a debug build.
#[test]
fn a_backwards_or_overflowing_jump_never_reissues_a_sequence_number() {
    let mut q = DeleteQueue::new();
    assert_eq!(q.next_sequence_number(), 1);
    q.skip_sequence_numbers(-100);
    assert_eq!(
        q.next_sequence_number(),
        2,
        "a negative jump must not rewind"
    );

    let mut q = DeleteQueue::new();
    q.skip_sequence_numbers(i64::MAX);
    q.skip_sequence_numbers(i64::MAX);
    assert_eq!(q.last_sequence_number(), MAX_SEQ_NO - 1);
    // ...and the counter is still far enough from the top that the very
    // next operation cannot overflow it.
    assert_eq!(q.next_sequence_number(), MAX_SEQ_NO);
    assert_eq!(q.next_sequence_number(), MAX_SEQ_NO + 1);
}

#[test]
fn a_repeated_delete_term_keeps_the_highest_doc_id_upto() {
    // Java's `BufferedUpdates.addTerm`: the higher limit wins, so two
    // `updateDocument(t, ...)` calls in a row delete both of the earlier
    // documents rather than only the first.
    let mut b = BufferedUpdates::default();
    b.add_term(t("id", "a"), 3);
    b.add_term(t("id", "a"), 1);
    assert_eq!(b.delete_terms[&t("id", "a")], 3);
    b.add_term(t("id", "a"), 7);
    assert_eq!(b.delete_terms[&t("id", "a")], 7);
}

#[test]
fn global_entries_are_unlimited_and_private_entries_are_bounded() {
    let mut q = DeleteQueue::new();
    q.add_term_deletes(&[t("id", "a")], 5);
    assert_eq!(q.private_updates().delete_terms[&t("id", "a")], 5);
    let global = q.freeze_global_buffer().expect("global packet");
    assert_eq!(
        global.delete_terms,
        vec![(Arc::new(t("id", "a")), MAX_DOC_ID_UPTO)]
    );
}

#[test]
fn freeze_global_buffer_is_empty_the_second_time() {
    let mut q = DeleteQueue::new();
    q.add_term_deletes(&[t("id", "a")], 0);
    assert!(q.freeze_global_buffer().is_some());
    assert!(q.freeze_global_buffer().is_none());
}

#[test]
fn a_private_packet_carries_the_buffers_limits_and_names_its_segment() {
    let mut q = DeleteQueue::new();
    q.add_term_deletes(&[t("id", "a")], 2);
    q.add_query_deletes(&[DeleteQuery::Term(t("id", "b"))], 3);
    let packet = q.freeze_private_buffer("_0").expect("private packet");
    assert_eq!(packet.delete_terms, vec![(Arc::new(t("id", "a")), 2)]);
    assert_eq!(packet.delete_queries.len(), 1);
    assert_eq!(packet.delete_queries[0].1, 3);
    assert_eq!(packet.private_segment.as_deref(), Some("_0"));
}

#[test]
fn freezing_the_private_buffer_resets_it_for_the_next_segment() {
    // Every `docIDUpto` in the buffer indexes the document buffer the
    // flush just emptied; carrying one across would re-target it at the
    // next segment's first documents.
    let mut q = DeleteQueue::new();
    q.add_term_deletes(&[t("id", "a")], 2);
    assert!(q.freeze_private_buffer("_0").is_some());
    assert!(!q.private_updates().any());
    assert!(q.freeze_private_buffer("_1").is_none());
}

#[test]
fn push_stamps_ascending_generations_starting_at_one() {
    let mut stream = BufferedUpdatesStream::new();
    let mut b = BufferedUpdates::default();
    b.add_term(t("id", "a"), MAX_DOC_ID_UPTO);
    let first = stream.push(FrozenBufferedUpdates::new(&b, None));
    let second = stream.push(FrozenBufferedUpdates::new(&b, None));
    assert_eq!((first, second), (1, 2));
    assert_eq!(stream.next_gen(), 3);
    assert_eq!(stream.pending().len(), 2);
}

#[test]
fn a_packet_applies_to_older_segments_and_not_to_newer_ones() {
    let mut stream = BufferedUpdatesStream::new();
    let mut b = BufferedUpdates::default();
    b.add_term(t("id", "a"), MAX_DOC_ID_UPTO);
    stream.push(FrozenBufferedUpdates::new(&b, None));
    let packet = &stream.pending()[0];
    assert_eq!(packet.del_gen(), 1);
    // A segment flushed before this packet (generation 0) gets it...
    assert!(packet.applies_to(0));
    // ...and one flushed after (generation 2) does not.
    assert!(!packet.applies_to(2));
}

#[test]
fn only_a_segments_own_private_packet_honours_the_doc_id_upto_limit() {
    let mut stream = BufferedUpdatesStream::new();
    let mut b = BufferedUpdates::default();
    b.add_query(DeleteQuery::MatchAll, 4);
    let gen = stream.push(FrozenBufferedUpdates::new(&b, Some("_0".into())));
    let packet = &stream.pending()[0];
    // Its own segment: the limit is enforced.
    assert_eq!(packet.limit_for(gen, 4), 4);
    // An older segment: no limit, every doc predates the delete.
    assert_eq!(packet.limit_for(gen - 1, 4), MAX_DOC_ID_UPTO);
}

#[test]
fn doc_values_updates_record_value_reset_and_arrival_order() {
    let mut b = BufferedUpdates::default();
    b.add_doc_values_update(
        &DocValuesUpdate::Numeric {
            term: t("id", "a"),
            field: "n".into(),
            value: Some(7),
        },
        10,
    );
    b.add_doc_values_update(
        &DocValuesUpdate::Numeric {
            term: t("id", "a"),
            field: "n".into(),
            value: None,
        },
        11,
    );
    let buffer = &b.field_updates["n"];
    assert!(buffer.is_numeric);
    assert_eq!(buffer.updates[0].value, UpdateValue::Numeric(7));
    // A null-valued field is `reset(doc)`, not "set to 0".
    assert_eq!(buffer.updates[1].value, UpdateValue::None);
    assert_eq!(b.num_field_updates, 2);
}

#[test]
fn binary_updates_are_kept_separate_from_numeric_ones() {
    let mut b = BufferedUpdates::default();
    b.add_doc_values_update(
        &DocValuesUpdate::Binary {
            term: t("id", "a"),
            field: "bin".into(),
            value: Some(b"hello".to_vec()),
        },
        0,
    );
    let buffer = &b.field_updates["bin"];
    assert!(!buffer.is_numeric);
    assert_eq!(
        buffer.updates[0].value,
        UpdateValue::Binary(b"hello".to_vec())
    );
}

#[test]
fn clear_resets_every_bucket_but_not_the_sequence_number() {
    let mut q = DeleteQueue::new();
    q.add_term_deletes(&[t("id", "a")], 0);
    assert!(q.any_changes());
    q.clear();
    assert!(!q.any_changes());
    // Java's rollback builds a *new* queue that continues the seqNo
    // space; a caller must never see a seqNo twice.
    assert_eq!(q.next_sequence_number(), 2);
}

#[test]
fn stream_clear_drops_the_packets_but_never_rewinds_the_generation() {
    // The bug this pins: `rollback()` restores committed segments that
    // still carry the `buffered_deletes_gen` their flush stamped on them.
    // Rewinding the counter to 1 (as Java's `clear()` does, safely, because
    // Java then closes the writer) would make every later packet sort
    // *below* those segments, and `applies_to` would reject the delete for
    // all but the oldest -- a silent, partial delete.
    let mut stream = BufferedUpdatesStream::new();
    let mut b = BufferedUpdates::default();
    b.add_term(t("id", "a"), MAX_DOC_ID_UPTO);
    assert_eq!(stream.push(FrozenBufferedUpdates::new(&b, None)), 1);
    assert_eq!(stream.push(FrozenBufferedUpdates::new(&b, None)), 2);
    assert!(stream.any());

    stream.clear();
    assert!(!stream.any());
    assert_eq!(
        stream.next_gen(),
        3,
        "the counter must continue past every generation already stamped \
             on a segment"
    );
}

#[test]
fn take_pending_hands_over_the_packets_and_empties_the_stream() {
    let mut stream = BufferedUpdatesStream::new();
    let mut b = BufferedUpdates::default();
    b.add_term(t("id", "a"), MAX_DOC_ID_UPTO);
    stream.push(FrozenBufferedUpdates::new(&b, None));
    stream.push(FrozenBufferedUpdates::new(&b, None));
    let taken = stream.take_pending();
    assert_eq!(taken.len(), 2);
    assert_eq!(taken[0].del_gen(), 1);
    assert_eq!(taken[1].del_gen(), 2);
    assert!(!stream.any());
    // ...and the counter still moves forward.
    assert_eq!(stream.next_gen(), 3);
}

#[test]
fn frozen_packet_sorts_its_terms() {
    let mut b = BufferedUpdates::default();
    b.add_term(t("id", "c"), 1);
    b.add_term(t("id", "a"), 2);
    b.add_term(t("body", "z"), 3);
    let packet = FrozenBufferedUpdates::new(&b, None);
    let terms: Vec<&Term> = packet.delete_terms.iter().map(|(t, _)| &**t).collect();
    assert_eq!(terms[0], &t("body", "z"));
    assert_eq!(terms[1], &t("id", "a"));
    assert_eq!(terms[2], &t("id", "c"));
}

#[test]
fn an_empty_buffer_freezes_to_a_packet_that_reports_nothing() {
    let b = BufferedUpdates::default();
    let packet = FrozenBufferedUpdates::new(&b, None);
    assert!(!packet.any());
}
