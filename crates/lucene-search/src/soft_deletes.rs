//! Task #48: soft-delete **visibility** — real Lucene's `IndexWriterConfig
//! .setSoftDeletesField` convention, where a document is invisible to search
//! if *either* its hard-delete bit is cleared in `.liv` *or* its
//! soft-deletes-field doc-values value is present for that doc.
//!
//! # Scope: what this is
//!
//! Real Lucene's soft-delete check
//! (`SoftDeletesDirectoryReaderWrapper`/`PendingSoftDeletes`) is exactly a
//! `DocValuesFieldExistsQuery` over the configured soft-deletes field: a doc
//! is soft-deleted if that field has *any* value for it in the current
//! (newest) version of the document — not a specific sentinel value, mere
//! presence. [`is_soft_deleted`]/[`effective_live_docs`] use that same
//! presence-based rule against an already-opened
//! [`lucene_codecs::doc_values::NumericEntry`] (the read side this port
//! already has — see `crates/lucene-codecs/src/doc_values.rs`, task #4/#21).
//!
//! # Scope: what this deliberately is not
//!
//! **Incremental marking: now available via task #54's update overlay.**
//! Real Lucene's soft-delete *write* path (`IndexWriter.softUpdateDocument`)
//! relies on `NumericDocValuesFieldUpdates`: an existing segment's
//! doc-values file gets a small delta record appended (a per-doc-values-
//! generation "diff" file), not rewritten from scratch, so marking one doc
//! soft-deleted is cheap even on a huge segment.
//! [`lucene_codecs::doc_values_updates`] (task #54) now provides exactly
//! this primitive in single-generation form: [`mark_soft_deleted_via_overlay`]
//! writes a tiny standalone overlay file recording just the one doc's
//! soft-deletes-field value (any value — presence is all that matters, see
//! [`is_soft_deleted`]'s doc comment) without touching the base
//! `.dvd`/`.dvm` bytes at all, and [`is_soft_deleted_with_overlay`]/
//! [`effective_live_docs_with_overlay`] check that overlay before falling
//! back to the base decode. This is a **single overlay round** (one file, one
//! batch of updates) — task #54 explicitly does not implement stacking many
//! sequential update generations with newest-wins semantics across rounds,
//! nor `SegmentCommitInfo`/`.si` `docValuesGen` metadata wiring (see that
//! module's own doc comment for the full scope statement). A caller needing
//! a second independent update round today has to compose a fresh merged
//! overlay itself rather than relying on multi-generation stacking.
//!
//! The plain (non-overlay) [`is_soft_deleted`]/[`effective_live_docs`]
//! functions below are unchanged and remain the right choice whenever no
//! overlay is in play (the common case — most segments never get an
//! incremental doc-values update at all).
//!
//! **No cross-segment orchestration.** Like [`crate::directory_reader`],
//! this operates on one already-opened segment's already-decoded
//! `NumericEntry` + `.dvd` bytes; a caller wires it in per segment.
//!
//! # Why a plain [`FixedBitSet`] output, not a new parameter everywhere
//!
//! Every scored-query function in `crate::lib`/`crate::multi_segment` already
//! takes `live_docs: Option<&FixedBitSet>` and treats `None` as "every doc is
//! live" (see e.g. `search_term_query`). [`effective_live_docs`] computes a
//! single combined bitset — hard-live AND NOT soft-deleted — that a caller
//! can pass into any of those *existing* parameters unchanged: no query
//! function's signature needs to grow an extra "and also check this
//! doc-values field" parameter, and every existing single-hard-delete-only
//! caller is completely unaffected (they simply never call
//! [`effective_live_docs`] and keep passing their own `live_docs` straight
//! through, exactly as before).

use std::collections::HashMap;

use lucene_codecs::doc_values::{self, NumericEntry};
use lucene_codecs::doc_values_updates;
use lucene_store::codec_util::ID_LENGTH;
use lucene_util::fixed_bit_set::FixedBitSet;

pub type Error = doc_values::Error;
pub type Result<T> = std::result::Result<T, Error>;

/// An already-opened numeric doc-values field configured as a segment's
/// soft-deletes field (`FieldInfo.soft_deletes_field == true`,
/// `doc_values_type == Numeric`): the decoded [`NumericEntry`] plus its
/// `.dvd` data bytes, exactly the two pieces
/// [`lucene_codecs::doc_values::numeric_value`] needs.
#[derive(Debug, Clone, Copy)]
pub struct SoftDeletesField<'a> {
    pub data: &'a [u8],
    pub entry: &'a NumericEntry,
}

/// A single doc's soft-delete state: real Lucene's rule is *presence*, not
/// value equality — `DocValuesFieldExistsQuery`, not a marker-value compare.
/// A doc the field has no value for at all (sparse encoding, `numeric_value`
/// returning `None`) is not soft-deleted; any value present — including `0`
/// — means it is.
pub fn is_soft_deleted(field: &SoftDeletesField<'_>, doc: i32) -> Result<bool> {
    Ok(doc_values::numeric_value(field.data, field.entry, doc)?.is_some())
}

/// The single per-doc visibility check this task exists to get right: NOT
/// live if hard-deleted (`live_docs` bit cleared) OR soft-deleted (soft-
/// deletes field has a value for this doc) — an OR of "is invisible", i.e.
/// an AND of "is visible": `live(doc) == live_docs.get(doc) &&
/// !soft_deleted(doc)`.
///
/// `soft_deletes: None` means this segment has no soft-deletes field
/// configured at all (the common case for every fixture/test this port had
/// before this task) — falls back to pure hard-delete visibility, unchanged
/// from every existing call site's behavior.
pub fn is_live(
    live_docs: Option<&FixedBitSet>,
    soft_deletes: Option<&SoftDeletesField<'_>>,
    doc: i32,
) -> Result<bool> {
    let hard_live = live_docs.is_none_or(|bits| bits.get(doc as usize));
    if !hard_live {
        return Ok(false);
    }
    if let Some(field) = soft_deletes {
        if is_soft_deleted(field, doc)? {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Computes a single combined live-docs bitset — hard-live AND NOT
/// soft-deleted, for every doc `0..max_doc` — suitable to pass directly into
/// any existing `live_docs: Option<&FixedBitSet>` parameter in this crate
/// (see this module's doc comment for why that's the chosen integration
/// point rather than a new parameter on every query function).
///
/// Returns `live_docs` unchanged (cloned) when `soft_deletes` is `None` — no
/// new allocation-shape surprises for the (still overwhelmingly common)
/// case of a segment with no soft-deletes field configured at all.
pub fn effective_live_docs(
    live_docs: Option<&FixedBitSet>,
    soft_deletes: Option<&SoftDeletesField<'_>>,
    max_doc: usize,
) -> Result<Option<FixedBitSet>> {
    let Some(field) = soft_deletes else {
        return Ok(live_docs.cloned());
    };

    let mut bits = match live_docs {
        Some(existing) => existing.clone(),
        None => {
            let mut all_live = FixedBitSet::new(max_doc);
            for i in 0..max_doc {
                all_live.set(i);
            }
            all_live
        }
    };

    for doc in 0..max_doc {
        if bits.get(doc) && is_soft_deleted(field, doc as i32)? {
            bits.clear(doc);
        }
    }

    Ok(Some(bits))
}

/// Task #54's concrete unblocking of this module's write-side gap: marks a
/// single document soft-deleted by writing **only** a tiny update-overlay
/// file (`lucene_codecs::doc_values_updates::write_numeric_updates`) — the
/// base segment's `.dvd`/`.dvm` bytes are never read, decoded, or rewritten
/// by this function at all. The overlay records `(doc, marker_value)`; per
/// [`is_soft_deleted`]'s presence-based rule, `marker_value`'s actual number
/// is irrelevant (any value in the overlay means "soft-deleted"), so `0` is
/// used here as an arbitrary constant. Composing this with any *other*
/// already-buffered updates for the same segment (multiple docs marked
/// before one flush) is the caller's responsibility — pass every
/// `(doc, value)` pair accumulated so far; this is still exactly one overlay
/// generation (task #54's documented single-round MVP scope), not a
/// stacked-generations API.
pub fn mark_soft_deleted_via_overlay(
    docs: &[i32],
    segment_id: &[u8; ID_LENGTH],
    segment_suffix: &str,
) -> Vec<u8> {
    let updates: Vec<(i32, i64)> = docs.iter().map(|&doc| (doc, 0i64)).collect();
    doc_values_updates::write_numeric_updates(&updates, segment_id, segment_suffix)
}

/// The overlay-aware counterpart to [`is_soft_deleted`]: a doc is
/// soft-deleted if it has a value in the overlay (marked via
/// [`mark_soft_deleted_via_overlay`] or any other overlay write) OR the base
/// doc-values field has a value for it — the same presence-based OR
/// [`is_soft_deleted`] already implements, just with an extra source of
/// presence to check first.
pub fn is_soft_deleted_with_overlay(
    field: &SoftDeletesField<'_>,
    overlay: &HashMap<i32, i64>,
    doc: i32,
) -> Result<bool> {
    if overlay.contains_key(&doc) {
        return Ok(true);
    }
    is_soft_deleted(field, doc)
}

/// The overlay-aware counterpart to [`effective_live_docs`]: identical
/// hard-live-AND-NOT-soft-deleted computation, except a doc's soft-delete
/// state is resolved through the overlay first (see
/// [`is_soft_deleted_with_overlay`]) rather than the base doc-values decode
/// alone — this is what lets a doc marked soft-deleted purely via
/// [`mark_soft_deleted_via_overlay`] (no base rewrite) actually become
/// invisible to search.
pub fn effective_live_docs_with_overlay(
    live_docs: Option<&FixedBitSet>,
    soft_deletes: Option<&SoftDeletesField<'_>>,
    overlay: &HashMap<i32, i64>,
    max_doc: usize,
) -> Result<Option<FixedBitSet>> {
    let Some(field) = soft_deletes else {
        if overlay.is_empty() {
            return Ok(live_docs.cloned());
        }
        // No base soft-deletes field configured, but the overlay itself
        // carries soft-delete marks (e.g. a soft-deletes field introduced
        // only via updates, never present in the base segment) -- still
        // apply them.
        let mut bits = match live_docs {
            Some(existing) => existing.clone(),
            None => {
                let mut all_live = FixedBitSet::new(max_doc);
                for i in 0..max_doc {
                    all_live.set(i);
                }
                all_live
            }
        };
        for doc in 0..max_doc {
            if bits.get(doc) && overlay.contains_key(&(doc as i32)) {
                bits.clear(doc);
            }
        }
        return Ok(Some(bits));
    };

    let mut bits = match live_docs {
        Some(existing) => existing.clone(),
        None => {
            let mut all_live = FixedBitSet::new(max_doc);
            for i in 0..max_doc {
                all_live.set(i);
            }
            all_live
        }
    };

    for doc in 0..max_doc {
        if bits.get(doc) && is_soft_deleted_with_overlay(field, overlay, doc as i32)? {
            bits.clear(doc);
        }
    }

    Ok(Some(bits))
}

#[cfg(test)]
mod tests {
    use super::*;
    use lucene_store::codec_util::ID_LENGTH;

    /// Reuses the real checked-in `doc_values_index` fixture's "sparse"
    /// numeric field (real Java-`IndexWriter`-written `.dvm`/`.dvd`/`.fnm`
    /// bytes, task #4/#21 -- see `crates/lucene-codecs/tests/
    /// doc_values_fixtures.rs`) rather than a hand-built or synthetically
    /// encoded field: this is genuinely presence-shaped (`IndexedDISI`, not
    /// dense) doc-values, exactly the encoding real Lucene's actual
    /// soft-deletes field convention produces (a fresh version of a document
    /// gets a value; older, still-live documents don't touch the field at
    /// all and stay sparse-absent) -- a dense field (the only shape this
    /// port's write side currently supports, see
    /// [`doc_values::write_single_dense_numeric_field`]'s doc comment) can't
    /// represent "doc has no value" for any doc at all, so it can't stand in
    /// for a real soft-deletes field's presence semantics.
    ///
    /// Per the fixture's manifest (`field.sparse.values=5,NONE,15,NONE,25`):
    /// docs `0`, `2`, `4` have a value (soft-deleted); docs `1`, `3` don't
    /// (not soft-deleted). `max_doc` for this field is `5`.
    struct SparseFixture {
        data: Vec<u8>,
        entry: NumericEntry,
    }

    fn load_sparse_fixture() -> SparseFixture {
        let dir = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../fixtures/data/doc_values_index/"
        );
        let manifest = std::fs::read_to_string(format!("{dir}manifest.properties"))
            .expect("run fixtures generator first (GenDocValues)");
        let get = |key: &str| -> String {
            manifest
                .lines()
                .find_map(|l| l.strip_prefix(&format!("{key}=")))
                .unwrap_or_else(|| panic!("manifest key {key} missing"))
                .to_string()
        };
        let id_hex = get("id_hex");
        let mut segment_id = [0u8; ID_LENGTH];
        for (i, slot) in segment_id.iter_mut().enumerate() {
            *slot = u8::from_str_radix(&id_hex[i * 2..i * 2 + 2], 16).unwrap();
        }
        let segment_name = get("segment_name");
        let dvm_name = get("dvm_file_name");
        // `PerFieldDocValuesFormat` suffix, e.g. `_0_Lucene90_0.dvm` (segment
        // `_0`) -> `Lucene90_0` (see `doc_values_fixtures.rs`'s `dv_suffix`).
        let suffix = dvm_name
            .strip_prefix(&format!("{segment_name}_"))
            .and_then(|s| s.strip_suffix(".dvm"))
            .unwrap_or_else(|| panic!("unexpected dvm file name shape: {dvm_name}"))
            .to_string();
        let fnm = std::fs::read(format!("{dir}{}.raw", get("fnm_file_name"))).unwrap();
        let field_infos =
            lucene_codecs::field_infos::parse(&fnm, &segment_id, "").expect("parse .fnm");
        let meta = std::fs::read(format!("{dir}{dvm_name}.raw")).unwrap();
        let data = std::fs::read(format!("{dir}{}.raw", get("dvd_file_name"))).unwrap();
        let field_number = get("field_numbers")
            .split(',')
            .find_map(|kv| {
                let (name, num) = kv.split_once(':').unwrap();
                (name == "sparse").then(|| num.parse::<i32>().unwrap())
            })
            .expect("sparse field missing from field_numbers");

        let (_, parsed) =
            doc_values::parse_meta(&meta, &segment_id, &suffix, &field_infos).expect("parse .dvm");
        let entry = parsed.numeric_entry(field_number).unwrap().clone();
        SparseFixture { data, entry }
    }

    impl SparseFixture {
        fn field(&self) -> SoftDeletesField<'_> {
            SoftDeletesField {
                data: &self.data,
                entry: &self.entry,
            }
        }
    }

    // --- is_soft_deleted / is_live: the four visibility combinations ---

    #[test]
    fn neither_hard_nor_soft_deleted_is_live() {
        let fx = load_sparse_fixture();
        let field = fx.field();
        // Doc 1 has no value in the real fixture (sparse-absent).
        assert!(!is_soft_deleted(&field, 1).unwrap());
        assert!(is_live(None, Some(&field), 1).unwrap());
    }

    #[test]
    fn hard_deleted_only_is_invisible() {
        let fx = load_sparse_fixture();
        let field = fx.field();
        let mut live = FixedBitSet::new(5);
        for i in 0..5 {
            live.set(i);
        }
        live.clear(3); // doc 3 hard-deleted (also sparse-absent: not soft-deleted)
        assert!(!is_soft_deleted(&field, 3).unwrap());
        assert!(!is_live(Some(&live), Some(&field), 3).unwrap());
        // Sanity: an untouched, sparse-absent doc is still live.
        assert!(is_live(Some(&live), Some(&field), 1).unwrap());
    }

    #[test]
    fn soft_deleted_only_is_invisible() {
        let fx = load_sparse_fixture();
        let field = fx.field();
        let mut live = FixedBitSet::new(5);
        for i in 0..5 {
            live.set(i);
        }
        // Doc 0 has a real value in the fixture -> soft-deleted, hard-live.
        assert!(is_soft_deleted(&field, 0).unwrap());
        assert!(!is_live(Some(&live), Some(&field), 0).unwrap());
        // Sanity: a sparse-absent doc under the same all-live bitset is live.
        assert!(is_live(Some(&live), Some(&field), 1).unwrap());
    }

    #[test]
    fn both_hard_and_soft_deleted_is_invisible_not_contradictory() {
        let fx = load_sparse_fixture();
        let field = fx.field();
        let mut live = FixedBitSet::new(5);
        for i in 0..5 {
            live.set(i);
        }
        live.clear(0); // doc 0 hard-deleted -- and it's also soft-deleted (present)
        assert!(is_soft_deleted(&field, 0).unwrap());
        assert!(!is_live(Some(&live), Some(&field), 0).unwrap());
    }

    #[test]
    fn no_soft_deletes_field_falls_back_to_pure_hard_delete_behavior() {
        let mut live = FixedBitSet::new(2);
        live.set(0);
        assert!(is_live(Some(&live), None, 0).unwrap());
        assert!(!is_live(Some(&live), None, 1).unwrap());
    }

    // --- effective_live_docs ---

    #[test]
    fn effective_live_docs_with_no_soft_deletes_field_passes_through_unchanged() {
        let mut live = FixedBitSet::new(3);
        live.set(0);
        live.set(2);
        let effective = effective_live_docs(Some(&live), None, 3).unwrap().unwrap();
        assert_eq!(effective.cardinality(), 2);
        assert!(effective.get(0) && !effective.get(1) && effective.get(2));
    }

    #[test]
    fn effective_live_docs_with_no_hard_deletes_and_no_soft_field_is_none() {
        assert!(effective_live_docs(None, None, 5).unwrap().is_none());
    }

    #[test]
    fn effective_live_docs_ors_hard_and_soft_deletes_from_all_live_state() {
        let fx = load_sparse_fixture();
        let field = fx.field();
        // No hard deletes at all (`None`); soft-deletes present at docs 0, 2, 4.
        let effective = effective_live_docs(None, Some(&field), 5).unwrap().unwrap();
        assert_eq!(effective.cardinality(), 2);
        assert!(!effective.get(0) && effective.get(1) && !effective.get(2));
        assert!(effective.get(3) && !effective.get(4));
    }

    #[test]
    fn effective_live_docs_combines_existing_hard_deletes_with_soft_deletes() {
        let fx = load_sparse_fixture();
        let field = fx.field();
        let mut live = FixedBitSet::new(5);
        for i in 0..5 {
            live.set(i);
        }
        live.clear(1); // doc 1 hard-deleted (sparse-absent, so soft-live)
                       // Soft-deletes: docs 0, 2, 4 (from the fixture).
        let effective = effective_live_docs(Some(&live), Some(&field), 5)
            .unwrap()
            .unwrap();
        // Union of hard-deleted {1} and soft-deleted {0, 2, 4}: only doc 3
        // survives both checks.
        assert_eq!(effective.cardinality(), 1);
        assert!(!effective.get(0)); // soft-deleted
        assert!(!effective.get(1)); // hard-deleted
        assert!(!effective.get(2)); // soft-deleted
        assert!(effective.get(3)); // neither
        assert!(!effective.get(4)); // soft-deleted
    }

    #[test]
    fn effective_live_docs_doc_hard_and_soft_deleted_together_still_just_zero() {
        let fx = load_sparse_fixture();
        let field = fx.field();
        let mut live = FixedBitSet::new(5);
        for i in 0..5 {
            live.set(i);
        }
        live.clear(0); // doc 0 hard-deleted -- and ALSO soft-deleted (present)
        let effective = effective_live_docs(Some(&live), Some(&field), 5)
            .unwrap()
            .unwrap();
        // Not double-counted/contradictory: doc 0 is invisible for exactly
        // one reason as far as the bitset is concerned (bit cleared once),
        // and docs 2/4 (soft-deleted only) are independently still cleared
        // too -- only the two sparse-absent, hard-live docs (1, 3) survive.
        assert_eq!(effective.cardinality(), 2);
        assert!(!effective.get(0));
        assert!(effective.get(1));
        assert!(!effective.get(2));
        assert!(effective.get(3));
        assert!(!effective.get(4));
    }

    // --- end-to-end: compose with a real scored query over a real fixture ---

    /// Reuses the real checked-in `blocktree_index` fixture
    /// [`crate::term_delete`]'s own tests use (field `body`, term `bird` ->
    /// docs `[1, 4]`, real postings/`.doc` bytes) *and* the real
    /// `doc_values_index` fixture's sparse numeric field (docs `0`, `2`, `4`
    /// present -> soft-deleted) to prove [`effective_live_docs`] genuinely
    /// composes with an existing scored query end-to-end, against real
    /// decoded doc-values bytes from an actual Java-written segment (not a
    /// hand-built/synthetic bitset): doc `4` (soft-deleted per the
    /// doc-values fixture) must be excluded from `"bird"`'s results, leaving
    /// only doc `1`.
    ///
    /// The two fixtures come from different, unrelated Java-written
    /// segments (this port has no single checked-in fixture combining real
    /// postings with a real soft-deletes doc-values field from the same
    /// `IndexWriter` run) -- but every byte consulted here, from both
    /// fixtures, was written by a real Lucene writer and round-tripped
    /// through this port's real decoders, so the composition genuinely
    /// exercises real bytes end-to-end, just from two independent sources.
    #[test]
    fn composes_with_a_real_term_query_end_to_end() {
        use crate::query::TermQuery;
        use lucene_codecs::blocktree;
        use lucene_codecs::field_infos as fi;
        use lucene_codecs::postings::DocInput;

        let dir = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../fixtures/data/blocktree_index/"
        );
        let manifest = std::fs::read_to_string(format!("{dir}manifest.properties"))
            .expect("run fixtures generator first (GenBlockTree)");
        let get = |key: &str| -> String {
            manifest
                .lines()
                .find_map(|l| l.strip_prefix(&format!("{key}=")))
                .unwrap_or_else(|| panic!("manifest key {key} missing"))
                .to_string()
        };
        let id_hex = get("id_hex");
        let mut segment_id = [0u8; ID_LENGTH];
        for (i, slot) in segment_id.iter_mut().enumerate() {
            *slot = u8::from_str_radix(&id_hex[i * 2..i * 2 + 2], 16).unwrap();
        }
        let suffix = get("segment_suffix");
        let max_doc: i32 = get("max_doc").parse().unwrap();
        let read_raw = |name: &str| -> Vec<u8> {
            std::fs::read(format!("{dir}{name}.raw")).unwrap_or_else(|_| panic!("missing {name}"))
        };
        let fnm = read_raw(&get("fnm_file_name"));
        let field_infos = fi::parse(&fnm, &segment_id, "").expect("parse .fnm");
        let tim = read_raw(&get("tim_file_name"));
        let tip = read_raw(&get("tip_file_name"));
        let tmd = read_raw(&get("tmd_file_name"));
        let fields = blocktree::open(
            &tim,
            &tip,
            &tmd,
            &field_infos,
            &segment_id,
            &suffix,
            max_doc,
        )
        .expect("open blocktree");
        let doc_bytes = read_raw(&get("doc_file_name"));
        let doc_in = DocInput::open(&doc_bytes, &segment_id, &suffix).expect("open .doc");

        // Real sparse soft-deletes field: docs 0, 2, 4 present -> soft-deleted.
        // Its own real domain is only max_doc=5 (see load_sparse_fixture's
        // doc comment); it's evaluated here over blocktree_index's much
        // larger max_doc (8958) purely because effective_live_docs needs one
        // shared doc-id space to combine with. For every doc id >= 5, the
        // sparse field's IndexedDISI simply has no entry, so
        // is_soft_deleted returns false there -- those docs are correctly
        // "not soft-deleted" (never touched by the fixture), not silently
        // miscounted. This test only draws conclusions about docs 1 and 4,
        // both inside the sparse field's real 0..5 domain, where "bird"'s
        // real matches happen to fall -- it does not claim the two
        // fixtures' domains genuinely overlap beyond that.
        let soft_fx = load_sparse_fixture();
        let soft_field = soft_fx.field();
        let effective = effective_live_docs(None, Some(&soft_field), max_doc as usize)
            .unwrap()
            .unwrap();

        let query = TermQuery::new("body", "bird");
        let mut collector = crate::collector::VecCollector::default();
        crate::search_term_query(
            &fields,
            Some(&doc_in),
            Some(&effective),
            &query,
            &mut collector,
        )
        .unwrap();
        // Real fixture: "bird" matches docs [1, 4]; doc 4 is soft-deleted per
        // the doc-values fixture, so only doc 1 should come back.
        assert_eq!(collector.docs, vec![1]);

        // Confirm this really is the OR of hard+soft: additionally
        // hard-deleting doc 1 leaves nothing live for "bird" at all.
        let mut hard_live = FixedBitSet::new(max_doc as usize);
        for i in 0..max_doc as usize {
            hard_live.set(i);
        }
        hard_live.clear(1);
        let effective_both =
            effective_live_docs(Some(&hard_live), Some(&soft_field), max_doc as usize)
                .unwrap()
                .unwrap();
        let mut collector2 = crate::collector::VecCollector::default();
        crate::search_term_query(
            &fields,
            Some(&doc_in),
            Some(&effective_both),
            &query,
            &mut collector2,
        )
        .unwrap();
        assert!(collector2.docs.is_empty());
    }

    // --- task #54: overlay-based incremental soft-delete marking ---

    const SEG_ID: [u8; ID_LENGTH] = [3u8; ID_LENGTH];

    #[test]
    fn mark_soft_deleted_via_overlay_marks_only_the_given_doc_no_base_touch() {
        // The real sparse fixture already has docs 0, 2, 4 soft-deleted and
        // 1, 3 not. Mark doc 3 soft-deleted via ONLY an overlay write -- no
        // base .dvd/.dvm rewrite -- and confirm the overlay-aware check now
        // reports it soft-deleted, while the plain (non-overlay) base check
        // still correctly reports doc 3 as not soft-deleted (proving the
        // base bytes were genuinely never touched).
        let fx = load_sparse_fixture();
        let field = fx.field();
        assert!(!is_soft_deleted(&field, 3).unwrap());

        let overlay_bytes = mark_soft_deleted_via_overlay(&[3], &SEG_ID, "");
        let overlay =
            doc_values_updates::read_numeric_updates(&overlay_bytes, &SEG_ID, "").unwrap();

        assert!(is_soft_deleted_with_overlay(&field, &overlay, 3).unwrap());
        // Base-only check is unaffected: no base rewrite happened.
        assert!(!is_soft_deleted(&field, 3).unwrap());
        // Docs already soft-deleted in the base are still soft-deleted
        // through the overlay-aware path too (overlay doesn't erase base
        // presence).
        assert!(is_soft_deleted_with_overlay(&field, &overlay, 0).unwrap());
        // An untouched, base-absent doc stays not-soft-deleted.
        assert!(!is_soft_deleted_with_overlay(&field, &overlay, 1).unwrap());
    }

    #[test]
    fn effective_live_docs_with_overlay_excludes_overlay_marked_doc() {
        let fx = load_sparse_fixture();
        let field = fx.field();
        let mut live = FixedBitSet::new(5);
        for i in 0..5 {
            live.set(i);
        }
        // Base soft-deletes: docs 0, 2, 4. Mark doc 3 soft-deleted purely via
        // overlay (doc 1 stays untouched -> still live).
        let overlay_bytes = mark_soft_deleted_via_overlay(&[3], &SEG_ID, "");
        let overlay =
            doc_values_updates::read_numeric_updates(&overlay_bytes, &SEG_ID, "").unwrap();

        let effective = effective_live_docs_with_overlay(Some(&live), Some(&field), &overlay, 5)
            .unwrap()
            .unwrap();
        // Only doc 1 survives: 0,2,4 base-soft-deleted, 3 overlay-soft-deleted.
        assert_eq!(effective.cardinality(), 1);
        assert!(effective.get(1));
        assert!(!effective.get(0));
        assert!(!effective.get(2));
        assert!(!effective.get(3));
        assert!(!effective.get(4));

        // Without the overlay, doc 3 would have stayed live.
        let effective_no_overlay = effective_live_docs(Some(&live), Some(&field), 5)
            .unwrap()
            .unwrap();
        assert!(effective_no_overlay.get(3));
    }

    #[test]
    fn effective_live_docs_with_overlay_empty_overlay_matches_plain_version() {
        let fx = load_sparse_fixture();
        let field = fx.field();
        let mut live = FixedBitSet::new(5);
        for i in 0..5 {
            live.set(i);
        }
        let empty_overlay = HashMap::new();
        let a = effective_live_docs_with_overlay(Some(&live), Some(&field), &empty_overlay, 5)
            .unwrap()
            .unwrap();
        let b = effective_live_docs(Some(&live), Some(&field), 5)
            .unwrap()
            .unwrap();
        assert_eq!(a.cardinality(), b.cardinality());
        for i in 0..5 {
            assert_eq!(a.get(i), b.get(i));
        }
    }

    #[test]
    fn effective_live_docs_with_overlay_and_no_soft_deletes_field_still_applies_overlay() {
        // No base soft-deletes field at all, but the overlay itself carries
        // marks -- covers the `soft_deletes: None` branch inside
        // effective_live_docs_with_overlay.
        let mut live = FixedBitSet::new(3);
        for i in 0..3 {
            live.set(i);
        }
        let overlay_bytes = mark_soft_deleted_via_overlay(&[1], &SEG_ID, "");
        let overlay =
            doc_values_updates::read_numeric_updates(&overlay_bytes, &SEG_ID, "").unwrap();
        let effective = effective_live_docs_with_overlay(Some(&live), None, &overlay, 3)
            .unwrap()
            .unwrap();
        assert!(effective.get(0));
        assert!(!effective.get(1));
        assert!(effective.get(2));
    }

    #[test]
    fn effective_live_docs_with_overlay_no_field_no_overlay_no_hard_deletes_is_none() {
        let empty_overlay = HashMap::new();
        assert!(
            effective_live_docs_with_overlay(None, None, &empty_overlay, 5)
                .unwrap()
                .is_none()
        );
    }
}
