use super::*;

fn control(slots: &[(usize, usize)]) -> FlushControl {
    let mut c = FlushControl::new(slots.len());
    for (i, &(docs, ram)) in slots.iter().enumerate() {
        c.commit_slot_bytes(i, docs, ram);
    }
    c
}

const POLICY: FlushByRamOrCountsPolicy = FlushByRamOrCountsPolicy {
    max_buffered_docs: Some(10),
    ram_buffer_bytes: Some(1000),
};

#[test]
fn the_document_count_marks_the_buffer_that_reached_it() {
    let mut c = control(&[(10, 5), (3, 900)]);
    POLICY.on_change(&mut c, Some(0));
    assert!(c.is_flush_pending(0));
    assert!(!c.is_flush_pending(1));
    assert_eq!((c.active_bytes(), c.flush_bytes()), (900, 5));
    // A count below the limit marks nothing while RAM is under it.
    let mut c = control(&[(9, 5)]);
    POLICY.on_change(&mut c, Some(0));
    assert_eq!(c.num_pending(), 0);
}

#[test]
fn ram_marks_the_largest_unmarked_buffer_not_the_one_that_changed() {
    let mut c = control(&[(1, 300), (1, 600), (0, 0)]);
    POLICY.on_change(&mut c, Some(0));
    assert_eq!(c.num_pending(), 0, "900 < 1000");
    c.commit_slot_bytes(0, 2, 400);
    POLICY.on_change(&mut c, Some(0));
    assert!(c.is_flush_pending(1) && !c.is_flush_pending(0));
    assert_eq!((c.active_bytes(), c.flush_bytes()), (400, 600));
    // Marked RAM no longer counts toward the trigger.
    POLICY.on_change(&mut c, Some(0));
    assert!(!c.is_flush_pending(0));
    assert!(!c.get_and_reset_apply_all_deletes());
}

#[test]
fn deletes_count_toward_the_trigger_and_alone_apply_themselves() {
    let mut c = control(&[(1, 500)]);
    c.set_delete_bytes_used(500);
    POLICY.on_change(&mut c, Some(0));
    assert!(c.is_flush_pending(0), "active + deletes reach the limit");
    assert!(!c.get_and_reset_apply_all_deletes());

    let mut c = control(&[(1, 10)]);
    c.set_delete_bytes_used(1000);
    POLICY.on_change(&mut c, None);
    assert!(c.get_and_reset_apply_all_deletes());
    assert!(!c.get_and_reset_apply_all_deletes(), "reset");
    assert_eq!(c.num_pending(), 0, "only deletes over the limit");

    let mut c = control(&[(1, 1000)]);
    c.set_delete_bytes_used(1000);
    POLICY.on_change(&mut c, Some(0));
    assert!(c.get_and_reset_apply_all_deletes() && c.is_flush_pending(0));

    // After a delete (no buffer), buffers are not marked.
    let mut c = control(&[(1, 900)]);
    c.set_delete_bytes_used(200);
    POLICY.on_change(&mut c, None);
    assert_eq!(c.num_pending(), 0);
    assert!(!c.get_and_reset_apply_all_deletes());
}

#[test]
fn disabled_limits_mark_nothing() {
    let off = FlushByRamOrCountsPolicy::default();
    let mut c = control(&[(1_000_000, usize::MAX / 2)]);
    c.set_delete_bytes_used(usize::MAX / 2);
    off.on_change(&mut c, Some(0));
    assert_eq!(c.num_pending(), 0);
    assert!(!c.get_and_reset_apply_all_deletes());
}

#[test]
fn the_largest_writer_is_the_first_of_equals_and_skips_empty_and_marked() {
    let mut c = control(&[(0, 0), (1, 0), (2, 7), (1, 7)]);
    assert_eq!(c.find_largest_non_pending_writer(), Some(2));
    c.set_flush_pending(2);
    assert_eq!(c.find_largest_non_pending_writer(), Some(3));
    // A buffer whose RAM is not counted yet can still be chosen.
    c.set_flush_pending(3);
    assert_eq!(c.find_largest_non_pending_writer(), Some(1));
    c.set_flush_pending(1);
    assert_eq!(c.find_largest_non_pending_writer(), None);
    // An empty buffer is never marked; marking twice changes nothing.
    c.set_flush_pending(0);
    c.set_flush_pending(2);
    assert_eq!(c.num_pending(), 3);
    assert_eq!(c.flush_bytes(), 14);
    // Out-of-range slots are ignored.
    c.set_flush_pending(9);
    c.commit_slot_bytes(9, 1, 1);
    assert_eq!(c.checkout_for_flush(9), 0);
    assert!(!c.is_flush_pending(9));
    assert_eq!(c.slots().len(), 4);
}

#[test]
fn checkout_moves_bytes_once_and_flush_done_releases_them() {
    let mut c = control(&[(1, 100), (1, 50)]);
    c.set_flush_pending(0);
    // A marked buffer that grows keeps counting toward flush bytes.
    c.commit_slot_bytes(0, 2, 120);
    assert_eq!((c.active_bytes(), c.flush_bytes()), (50, 120));
    assert_eq!(c.checkout_for_flush(0), 120);
    assert_eq!((c.active_bytes(), c.flush_bytes()), (50, 120));
    assert_eq!(c.slots()[0], SlotState::default());
    assert_eq!(c.checkout_for_flush(1), 50);
    assert_eq!((c.active_bytes(), c.flush_bytes()), (0, 170));
    c.flush_done(120);
    c.flush_done(50);
    assert_eq!(c.flush_bytes(), 0);
    assert_eq!(c.delete_bytes_used(), 0);
}
