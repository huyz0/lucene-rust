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
//! **No cheap incremental marking.** Real Lucene's soft-delete *write* path
//! (`IndexWriter.softUpdateDocument`) relies on
//! `NumericDocValuesFieldUpdates`: an existing segment's doc-values file gets
//! a small delta record appended (a per-doc-values-generation "diff" file),
//! not rewritten from scratch, so marking one doc soft-deleted is cheap even
//! on a huge segment. This port's doc-values write side
//! ([`lucene_codecs::doc_values::write_single_dense_numeric_field`]) only
//! ever writes a brand-new, complete `.dvm`/`.dvd`/`.dvs` triple for a single
//! dense field — there is no incremental-update format/codec here at all
//! (see that function's own doc comment: "Deliberately not attempted here…
//! sparse fields (`IndexedDISI`)… table compression… multiple fields").
//! Building a fake "cheap update" shim on top of a full-rewrite primitive
//! would misrepresent what actually happens on disk, so this port does not
//! pretend to have it. **Marking** a document soft-deleted is therefore left
//! to whatever wrote the segment's doc-values in the first place (a fresh
//! flush, or [`crate::update_document`]-style new-segment flow once a
//! soft-deletes field is wired into it) — this module only ever *reads*
//! that state, which is a complete, real, and independently useful half.
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

use lucene_codecs::doc_values::{self, NumericEntry};
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
}
