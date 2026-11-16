#![cfg(test)]
use super::*;

fn leaf() -> FieldExistsLeaf {
    FieldExistsLeaf {
        source: Some(FieldExistsSource::DocValues),
        max_doc: 10,
        num_docs: 10,
        has_doc_values_skip_index: false,
        has_points: false,
        is_indexed: false,
        terms_doc_count: None,
        vector_size: None,
        points_doc_count: None,
        skipper_doc_count: None,
    }
}

/// A doc-values field that also declares a skip index -- the arm Java takes
/// first.
fn skipped(doc_count: i32) -> FieldExistsLeaf {
    FieldExistsLeaf {
        has_doc_values_skip_index: true,
        skipper_doc_count: Some(doc_count),
        ..leaf()
    }
}

#[test]
fn match_all_docs_counts_live_documents_only() {
    assert_eq!(count_match_all_docs(7, None), 7);
    let mut bits = FixedBitSet::new(7);
    for i in [0usize, 2, 5] {
        bits.set(i);
    }
    assert_eq!(count_match_all_docs(7, Some(&bits)), 3);
    // A negative `max_doc` is not a value a segment can carry, but it is a
    // value a caller can pass; clamp rather than produce a negative count.
    assert_eq!(count_match_all_docs(-1, None), 0);
}

#[test]
fn a_field_absent_from_the_leaf_counts_zero_and_blocks_the_rewrite() {
    let absent = FieldExistsLeaf {
        source: None,
        ..leaf()
    };
    assert_eq!(count_field_exists_leaf(&absent), Some(0));
    assert!(!field_exists_rewrites_to_match_all_docs(
        &[absent],
        Some(10),
        10
    ));
}

#[test]
fn the_norms_branch_shortcuts_only_when_the_field_is_complete() {
    let complete = FieldExistsLeaf {
        source: Some(FieldExistsSource::Norms),
        terms_doc_count: Some(10),
        num_docs: 8,
        ..leaf()
    };
    // Every document has the field; two are deleted, so the answer is
    // `numDocs()`, not `maxDoc()`.
    assert_eq!(count_field_exists_leaf(&complete), Some(8));
    let partial = FieldExistsLeaf {
        terms_doc_count: Some(6),
        ..complete
    };
    // Java's norms branch has no count of its own for a partial field.
    assert_eq!(count_field_exists_leaf(&partial), None);
}

#[test]
fn a_complete_doc_values_field_answers_num_docs_and_a_partial_one_answers_the_count() {
    let complete = FieldExistsLeaf {
        num_docs: 9,
        ..skipped(10)
    };
    assert_eq!(count_field_exists_leaf(&complete), Some(9));

    assert_eq!(count_field_exists_leaf(&skipped(4)), Some(4));

    // Partial *and* deleted: the two sets have to be intersected, which
    // only a scan can do.
    let partial_with_deletions = FieldExistsLeaf {
        num_docs: 9,
        ..skipped(4)
    };
    assert_eq!(count_field_exists_leaf(&partial_with_deletions), None);

    // Zero short-circuits before either of those.
    let empty = FieldExistsLeaf {
        num_docs: 9,
        ..skipped(0)
    };
    assert_eq!(count_field_exists_leaf(&empty), Some(0));
}

#[test]
fn the_doc_values_arm_is_selected_by_field_info_flags_not_by_what_resolves() {
    // Every proxy present at a different value, so which one Java's ladder
    // takes is visible in the answer.
    let all = FieldExistsLeaf {
        skipper_doc_count: Some(2),
        points_doc_count: Some(3),
        terms_doc_count: Some(4),
        ..leaf()
    };
    assert_eq!(
        count_field_exists_leaf(&FieldExistsLeaf {
            has_doc_values_skip_index: true,
            ..all
        }),
        Some(2)
    );
    assert_eq!(
        count_field_exists_leaf(&FieldExistsLeaf {
            has_points: true,
            ..all
        }),
        Some(3)
    );
    assert_eq!(
        count_field_exists_leaf(&FieldExistsLeaf {
            is_indexed: true,
            ..all
        }),
        Some(4)
    );
    // Points wins over terms when both flags are set, as Java's `else if`
    // chain does.
    assert_eq!(
        count_field_exists_leaf(&FieldExistsLeaf {
            has_points: true,
            is_indexed: true,
            ..all
        }),
        Some(3)
    );
    // A declared skip index whose skipper could not be read counts **0**,
    // Java's `skipper == null ? 0 : docCount()`. It does *not* fall through
    // to the points or terms proxies, which is exactly what a presence-based
    // `.or` chain would have done -- and would have answered 3 here.
    assert_eq!(
        count_field_exists_leaf(&FieldExistsLeaf {
            has_doc_values_skip_index: true,
            skipper_doc_count: None,
            ..all
        }),
        Some(0)
    );
    // No flag set at all is Java's `count == -1` falling through to
    // `super.count`.
    assert_eq!(count_field_exists_leaf(&all), None);
}

#[test]
fn the_points_and_terms_proxies_are_refused_when_the_leaf_has_deletions() {
    // Java gates both on `reader.hasDeletions() == false`: a *doc* count
    // says nothing about which of those documents are live, so with
    // deletions there is nothing to shortcut from. The skipper arm is not
    // gated, because the reconciliation below it already is.
    let deleted = FieldExistsLeaf {
        num_docs: 9,
        points_doc_count: Some(4),
        terms_doc_count: Some(4),
        has_points: true,
        is_indexed: true,
        ..leaf()
    };
    assert_eq!(count_field_exists_leaf(&deleted), None);
    // The same leaf without deletions answers from the proxy.
    assert_eq!(
        count_field_exists_leaf(&FieldExistsLeaf {
            num_docs: 10,
            ..deleted
        }),
        Some(4)
    );
    // And a *complete* field still shortcuts to `numDocs()` with deletions,
    // because that path goes through the skipper arm.
    assert_eq!(
        count_field_exists_leaf(&FieldExistsLeaf {
            num_docs: 9,
            ..skipped(10)
        }),
        Some(9)
    );
}

#[test]
fn the_rewrite_needs_every_leaf_and_reads_the_top_level_reader_for_norms() {
    let dv_complete = FieldExistsLeaf {
        skipper_doc_count: Some(10),
        ..leaf()
    };
    let dv_partial = FieldExistsLeaf {
        skipper_doc_count: Some(3),
        ..leaf()
    };
    assert!(field_exists_rewrites_to_match_all_docs(
        &[dv_complete, dv_complete],
        None,
        20
    ));
    assert!(!field_exists_rewrites_to_match_all_docs(
        &[dv_complete, dv_partial],
        None,
        20
    ));
    // Java's own asymmetry: for a norms-sourced field the per-leaf counts
    // are never read -- both leaves below say 10 of 10, and the decision
    // still comes from the reader-wide pair.
    let norms = FieldExistsLeaf {
        source: Some(FieldExistsSource::Norms),
        terms_doc_count: Some(10),
        ..leaf()
    };
    assert!(field_exists_rewrites_to_match_all_docs(
        &[norms, norms],
        Some(20),
        20
    ));
    assert!(!field_exists_rewrites_to_match_all_docs(
        &[norms, norms],
        Some(19),
        20
    ));
    // A reader with no leaves has no documents, so the two queries agree.
    assert!(field_exists_rewrites_to_match_all_docs(&[], None, 0));
}
