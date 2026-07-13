//! Single-segment delete-by-term resolution: "which live doc IDs does
//! `deleteDocuments(new Term(field, bytes))` actually name, in one
//! already-opened segment" — the resolve half of real Lucene's two-phase
//! `BufferedUpdates` (resolve) + `ReadersAndUpdates.writeLiveDocs` (apply)
//! delete flow. [`deletes.rs`](crate::deletes) is the apply half; this module
//! is the resolve half, scoped to one segment.
//!
//! # Why this lives in `lucene-index`, not `lucene-search`
//!
//! The dependency graph is strictly downward: `util ← store ← codecs ← index
//! ← search ← core ← ffi` (see the `architecture` skill and `PLAN.md` §1).
//! `crates/lucene-search/Cargo.toml` depends on `lucene-index`, so
//! `lucene-index` depending back on `lucene-search` would invert that edge
//! into a cycle. `lucene-search`'s own `term_doc_ids` (in
//! `crates/lucene-search/src/lib.rs`) already does exactly this lookup —
//! `field.postings(term, doc_in)` filtered by `live_docs` — but every
//! primitive it calls (`lucene_codecs::blocktree::BlockTreeFields`,
//! `lucene_codecs::postings::DocInput`) lives in `lucene-codecs`, which
//! `lucene-index` already depends on directly. So rather than duplicate a
//! shared primitive one layer down (which would need a `lucene-codecs`
//! change) or depend upward on `lucene-search` (which would invert the DAG),
//! this module reimplements the same handful of lines directly against
//! `lucene-codecs`: it is a real primitive already living at the right layer,
//! not something that needs promoting or duplicating.
//!
//! # Scope
//!
//! Resolution is scoped to **one already-opened segment** — same shape
//! `lucene-search`'s functions take (`BlockTreeFields` + opened `.doc` file).
//! A real `IndexWriter.deleteDocuments(Term)` resolves against *every*
//! currently-open segment via `BufferedUpdates`/`ReaderPool`; this port has
//! no multi-segment reader/writer orchestration, so that part is still
//! deferred (see `docs/parity.md`). Delete-by-query beyond a single exact
//! term, and real `updateDocument`, are also out of scope here — see
//! `deletes.rs`'s module doc for why.

use lucene_codecs::blocktree::BlockTreeFields;
use lucene_codecs::postings::DocInput;
use lucene_util::fixed_bit_set::FixedBitSet;

use lucene_store::directory::Directory;

use crate::deletes;
use crate::segment_infos::SegmentCommitInfo;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    BlockTree(#[from] lucene_codecs::blocktree::Error),
    #[error(transparent)]
    Deletes(#[from] deletes::Error),
}

pub type Result<T> = std::result::Result<T, Error>;

/// Resolves `(field, term)` to every **live** doc ID matching it in this
/// segment, ascending (same order `Postings::docs` and
/// `lucene-search::term_doc_ids` both already guarantee).
///
/// Mirrors `lucene-search::term_doc_ids`'s exact semantics: an unknown
/// `field` or a `term` not present in that field's dictionary yields an
/// empty `Vec`, not an error (matches `TermQuery.createWeight`'s `null`
/// `Scorer` "no matches" outcome — a delete for a term nothing currently
/// matches is a legitimate no-op, not a caller bug).
///
/// `doc_in` is `None` for segments with no `.doc` file opened; only actually
/// consulted when the term's `docFreq > 1` (see
/// `BlockTreeFields::field`/`FieldTerms::postings`) — passing `None` for a
/// term that turns out to need it surfaces as an [`Error`].
///
/// `live_docs` is the segment's current `.liv` bitset (`None` means "no
/// deletions yet, every doc is live"), same convention `deletes::apply_deletes`
/// and `lucene-search::search_term_query` both already use. Only live docs
/// are returned — a term match on an already-deleted doc contributes nothing
/// new to delete.
pub fn resolve_term_doc_ids(
    fields: &BlockTreeFields,
    doc_in: Option<&DocInput<'_>>,
    live_docs: Option<&FixedBitSet>,
    field: &str,
    term: &[u8],
) -> Result<Vec<i32>> {
    let Some(field_terms) = fields.field(field) else {
        return Ok(Vec::new());
    };
    let Some(postings) = field_terms.postings(term, doc_in)? else {
        return Ok(Vec::new());
    };
    Ok(postings
        .docs
        .iter()
        .copied()
        .filter(|&doc_id| live_docs.is_none_or(|bits| bits.get(doc_id as usize)))
        .collect())
}

/// The full single-segment "resolve then apply" delete-by-term flow: resolves
/// `(field, term)` to its matching live doc IDs via [`resolve_term_doc_ids`],
/// then applies them via [`deletes::apply_deletes`] exactly as if a caller had
/// hand-picked those doc IDs — closing the gap between "here's a term to
/// delete" and "here's the actual `mark_deleted`/`apply_deletes` call" for one
/// segment. Matches real Lucene's per-segment resolve-then-apply shape;
/// resolving across an `IndexWriter`'s whole open segment set is still out of
/// scope (see this module's doc comment).
///
/// Parameters are the union of [`resolve_term_doc_ids`]'s and
/// [`deletes::apply_deletes`]'s: `fields`/`doc_in` to resolve the term,
/// `current_live_docs`/`max_doc` for both resolution (live-doc filtering) and
/// application (bitset sizing/bounds-checking), `dir`/`sci` to write the
/// updated `.liv` file and produce the next `SegmentCommitInfo`.
#[allow(clippy::too_many_arguments)]
pub fn resolve_and_apply_term_delete(
    dir: &dyn Directory,
    sci: &SegmentCommitInfo,
    fields: &BlockTreeFields,
    doc_in: Option<&DocInput<'_>>,
    current_live_docs: Option<&FixedBitSet>,
    max_doc: usize,
    field: &str,
    term: &[u8],
) -> Result<SegmentCommitInfo> {
    let doc_ids = resolve_term_doc_ids(fields, doc_in, current_live_docs, field, term)?;
    Ok(deletes::apply_deletes(
        dir,
        sci,
        current_live_docs,
        max_doc,
        doc_ids,
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use lucene_codecs::blocktree;
    use lucene_codecs::field_infos;
    use lucene_store::codec_util::ID_LENGTH;
    use lucene_store::directory::FsDirectory;

    // Reuses the same checked-in real-Lucene fixture
    // (`fixtures/data/blocktree_index/`) `lucene-search`'s own unit tests
    // open -- see that crate's `open_fixture` helper. A real fixture beats a
    // hand-built one wherever one is already available (`test-coverage`
    // skill). Known contents (see `manifest.properties`): field `body`, term
    // `cat` -> docs [0, 2] (docFreq 2, needs `.doc`); term `dog` -> docs [0,
    // 1]; term `bird` -> docs [1, 4]; field `id`, term `id0` -> doc [0]
    // (docFreq 1, singleton, no `.doc` needed). `max_doc` = 8959.
    struct Fixture {
        fields: BlockTreeFields,
        doc_bytes: Vec<u8>,
        segment_id: [u8; ID_LENGTH],
        suffix: String,
        max_doc: usize,
    }

    impl Fixture {
        fn doc_in(&self) -> DocInput<'_> {
            DocInput::open(&self.doc_bytes, &self.segment_id, &self.suffix).expect("open .doc")
        }
    }

    fn open_fixture() -> Fixture {
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
        let field_infos = field_infos::parse(&fnm, &segment_id, "").expect("parse .fnm");
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

        Fixture {
            fields,
            doc_bytes,
            segment_id,
            suffix,
            max_doc: max_doc as usize,
        }
    }

    fn sci(segment_name: &str, del_gen: i64, del_count: i32) -> SegmentCommitInfo {
        SegmentCommitInfo {
            segment_name: segment_name.to_string(),
            segment_id: [7u8; ID_LENGTH],
            codec_name: "Lucene104".to_string(),
            del_gen,
            del_count,
            field_infos_gen: -1,
            doc_values_gen: -1,
            soft_del_count: 0,
            sci_id: None,
            field_infos_files: vec![],
            dv_update_files: vec![],
        }
    }

    fn tempdir() -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "lucene-rust-term-delete-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    // --- resolve_term_doc_ids ---

    #[test]
    fn term_with_multiple_matching_docs() {
        let fx = open_fixture();
        let doc_in = fx.doc_in();
        let docs = resolve_term_doc_ids(&fx.fields, Some(&doc_in), None, "body", b"cat").unwrap();
        assert_eq!(docs, vec![0, 2]);
    }

    #[test]
    fn term_with_single_match_no_doc_file_needed() {
        let fx = open_fixture();
        // Singleton (docFreq == 1) postings never touch `.doc` -- `None` is
        // fine here, matching `lucene-search`'s own singleton-path coverage.
        let docs = resolve_term_doc_ids(&fx.fields, None, None, "id", b"id0").unwrap();
        assert_eq!(docs, vec![0]);
    }

    #[test]
    fn term_with_no_matches_is_empty_not_an_error() {
        let fx = open_fixture();
        let doc_in = fx.doc_in();
        let docs =
            resolve_term_doc_ids(&fx.fields, Some(&doc_in), None, "body", b"zzz-missing").unwrap();
        assert!(docs.is_empty());
    }

    #[test]
    fn missing_field_is_empty_not_an_error() {
        let fx = open_fixture();
        let doc_in = fx.doc_in();
        let docs =
            resolve_term_doc_ids(&fx.fields, Some(&doc_in), None, "nonexistent", b"cat").unwrap();
        assert!(docs.is_empty());
    }

    #[test]
    fn live_docs_filter_excludes_already_deleted_docs() {
        let fx = open_fixture();
        let doc_in = fx.doc_in();
        let mut live = FixedBitSet::new(fx.max_doc);
        for i in 0..fx.max_doc {
            live.set(i);
        }
        live.clear(0); // doc 0 already deleted
        let docs =
            resolve_term_doc_ids(&fx.fields, Some(&doc_in), Some(&live), "body", b"cat").unwrap();
        assert_eq!(docs, vec![2]); // doc 0 filtered out, doc 2 remains
    }

    // --- resolve_and_apply_term_delete: real Directory I/O ---

    #[test]
    fn resolves_and_applies_against_a_real_flushed_segment() {
        let fx = open_fixture();
        let doc_in = fx.doc_in();
        let tmp = tempdir();
        let dir = FsDirectory::open(&tmp);
        let info = sci("_0", -1, 0);

        let updated = resolve_and_apply_term_delete(
            &dir,
            &info,
            &fx.fields,
            Some(&doc_in),
            None,
            fx.max_doc,
            "body",
            b"dog", // docs [0, 1]
        )
        .unwrap();

        assert_eq!(updated.del_gen, 1);
        assert_eq!(updated.del_count, 2);

        let bytes = std::fs::read(tmp.join("_0_1.liv")).unwrap();
        let parsed =
            lucene_codecs::live_docs::parse(&bytes, &info.segment_id, 1, fx.max_doc, 2).unwrap();
        assert!(!parsed.get(0));
        assert!(!parsed.get(1));
        assert!(parsed.get(2));
        assert!(parsed.get(4));
    }

    #[test]
    fn deleting_the_same_term_twice_increments_gen_without_double_counting() {
        let fx = open_fixture();
        let doc_in = fx.doc_in();
        let tmp = tempdir();
        let dir = FsDirectory::open(&tmp);
        let info = sci("_0", -1, 0);

        let after_first = resolve_and_apply_term_delete(
            &dir,
            &info,
            &fx.fields,
            Some(&doc_in),
            None,
            fx.max_doc,
            "body",
            b"dog", // docs [0, 1]
        )
        .unwrap();
        assert_eq!(after_first.del_gen, 1);
        assert_eq!(after_first.del_count, 2);

        let first_liv = std::fs::read(tmp.join("_0_1.liv")).unwrap();
        let first_bits =
            lucene_codecs::live_docs::parse(&first_liv, &info.segment_id, 1, fx.max_doc, 2)
                .unwrap();

        // Second round deletes the same term again: docs 0 and 1 are already
        // gone, so resolution now only sees them as no-longer-live and the
        // round should be a no-op del_count-wise, still bumping del_gen.
        let after_second = resolve_and_apply_term_delete(
            &dir,
            &after_first,
            &fx.fields,
            Some(&doc_in),
            Some(&first_bits),
            fx.max_doc,
            "body",
            b"dog",
        )
        .unwrap();
        assert_eq!(after_second.del_gen, 2);
        assert_eq!(after_second.del_count, 2); // not double-counted

        // A different term ("cat" -> docs [0, 2]) whose doc 0 overlaps with
        // the first round: only doc 2 is newly deleted.
        let after_third = resolve_and_apply_term_delete(
            &dir,
            &after_second,
            &fx.fields,
            Some(&doc_in),
            Some(&first_bits),
            fx.max_doc,
            "body",
            b"cat",
        )
        .unwrap();
        assert_eq!(after_third.del_gen, 3);
        assert_eq!(after_third.del_count, 3); // only doc 2 newly deleted
    }

    #[test]
    fn missing_term_delete_is_a_no_op_round_that_still_bumps_gen() {
        let fx = open_fixture();
        let doc_in = fx.doc_in();
        let tmp = tempdir();
        let dir = FsDirectory::open(&tmp);
        let info = sci("_0", -1, 0);

        let updated = resolve_and_apply_term_delete(
            &dir,
            &info,
            &fx.fields,
            Some(&doc_in),
            None,
            fx.max_doc,
            "body",
            b"zzz-missing",
        )
        .unwrap();
        assert_eq!(updated.del_gen, 1);
        assert_eq!(updated.del_count, 0);
    }
}
