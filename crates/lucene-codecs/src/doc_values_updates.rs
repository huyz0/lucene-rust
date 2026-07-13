//! Task #54: a numeric doc-values **update overlay** -- the incremental
//! mechanism real Lucene's `NumericDocValuesFieldUpdates` provides so that
//! marking a handful of docs' doc-values field with a new value doesn't
//! require rewriting a whole segment's `.dvd`/`.dvm` triple.
//!
//! # What real Lucene does
//!
//! `IndexWriter.updateNumericDocValue` buffers `(docId, newValue)` pairs in a
//! `NumericDocValuesFieldUpdates`; when they're flushed
//! (`DocValuesFieldUpdates.Container`/`FieldUpdatesWriter`), Lucene appends a
//! small new "generation" file recording just the sparse `(docId, value)`
//! deltas, bumps `SegmentCommitInfo.docValuesGen`, and records the new file
//! against that generation in `SegmentCommitInfo.getDocValuesUpdatesFiles()`.
//! A reader opening the segment applies every generation's deltas as an
//! overlay on top of the base `.dvd`/`.dvm` triple, with later generations
//! winning ties for the same doc. Crucially, the base doc-values file itself
//! is never rewritten for this -- only new (small) overlay files accumulate.
//!
//! # Scope of this port: single-generation MVP
//!
//! This module implements exactly **one** overlay round: write a sparse
//! `(docId -> newValue)` map to a small standalone file, and read a base
//! numeric doc-values value *through* that overlay (overlay wins if present,
//! else fall back to the base `.dvd` decode). This is the real property the
//! task needs -- "update without a full rewrite" -- proven end to end.
//!
//! **Explicitly not implemented** (future work, not silently assumed):
//! - **Multiple sequential generations.** Real Lucene supports arbitrarily
//!   many update rounds, each producing its own generation file, with
//!   newest-generation-wins semantics when the same doc is touched more than
//!   once across rounds. This port has exactly one overlay layer; composing
//!   two overlays (or re-deriving a merged overlay from many rounds) is not
//!   implemented. A caller who needs a second update round today would have
//!   to build a fresh overlay from a merged map themselves.
//! - **`SegmentCommitInfo`/`.si` `docValuesGen` wiring.** This module does
//!   not touch segment metadata / commit generation counters at all; it's a
//!   standalone read/write primitive a caller (e.g. task #37/#48's future
//!   commit-lifecycle code) can adopt once that wiring exists.
//!
//! # Byte format: this port's own invention
//!
//! There is no real Lucene fixture for a `NumericDocValuesFieldUpdates`
//! generation file checked into this repo (same honest situation as task
//! #49/#52's index-sort format), and unlike the base `.dvm`/`.dvd` format
//! there is also no plan to derive one, since the MVP scope here
//! deliberately stops short of the full generation/`FieldInfos`-versioning
//! machinery that produces real Lucene's actual on-disk bytes for this
//! format. The encoding below is therefore **not** a port of any specific
//! real Lucene byte layout -- it's a simple, self-consistent, documented
//! encoding invented for this port, reusing this crate's existing
//! `codec_util` header/footer/CRC machinery for structural integrity (so it
//! composes with the same corruption-detection conventions every other
//! format in this crate uses), but the field layout in between the header
//! and footer is specific to this module.
//!
//! Format: `codec_util` index header (codec name
//! [`NUMERIC_UPDATES_CODEC`], version [`VERSION_CURRENT`], a segment id +
//! suffix exactly like every other per-segment file in this crate), then a
//! `vint` count of entries, then that many `(i32 doc_id, i64 new_value)`
//! pairs in ascending `doc_id` order (ascending order is enforced on write
//! and validated on read, matching this crate's other sorted-array formats),
//! then a `codec_util` footer (CRC32 checksum).

use std::collections::HashMap;

use lucene_store::codec_util::{self, ID_LENGTH};
use lucene_store::data_input::{DataInput, SliceInput};
use lucene_store::DataOutput;

use crate::doc_values::{self, NumericEntry};

const NUMERIC_UPDATES_CODEC: &str = "LuceneRustNumericDocValuesUpdates";
const VERSION_START: i32 = 0;
const VERSION_CURRENT: i32 = VERSION_START;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Store(#[from] lucene_store::Error),
    #[error("doc ids must be written in strictly ascending order: {prev} then {next}")]
    OutOfOrder { prev: i32, next: i32 },
    #[error(transparent)]
    DocValues(#[from] doc_values::Error),
}

pub type Result<T> = std::result::Result<T, Error>;

/// Writes a sparse `(docId, newValue)` overlay to its own small standalone
/// file -- the alternative to rewriting a whole segment's `.dvd`/`.dvm`
/// triple just to change a handful of docs' values. `updates` need not be
/// pre-sorted; this function sorts (and de-duplicates, keeping the *last*
/// occurrence of a repeated doc id, matching "last write wins" semantics for
/// a single buffered update batch) before writing.
///
/// `segment_id`/`segment_suffix` are threaded through exactly like every
/// other per-segment file in this crate, so the overlay is tied to the same
/// segment identity as the base doc-values file it augments.
pub fn write_numeric_updates(
    updates: &[(i32, i64)],
    segment_id: &[u8; ID_LENGTH],
    segment_suffix: &str,
) -> Vec<u8> {
    let mut sorted: Vec<(i32, i64)> = updates.to_vec();
    // Stable sort by doc id keeps later-in-input entries after earlier ones
    // for equal keys; then keep only the last entry per doc id below.
    sorted.sort_by_key(|&(doc, _)| doc);
    let mut deduped: Vec<(i32, i64)> = Vec::with_capacity(sorted.len());
    for (doc, value) in sorted {
        if deduped.last().is_some_and(|&(last_doc, _)| last_doc == doc) {
            *deduped.last_mut().unwrap() = (doc, value);
        } else {
            deduped.push((doc, value));
        }
    }

    let mut out: Vec<u8> = Vec::new();
    codec_util::write_index_header(
        &mut out,
        NUMERIC_UPDATES_CODEC,
        VERSION_CURRENT,
        segment_id,
        segment_suffix,
    );
    out.write_vint(deduped.len() as i32);
    for (doc, value) in &deduped {
        out.write_i32(*doc);
        out.write_i64(*value);
    }
    codec_util::write_footer(&mut out);
    out
}

/// Reads back an overlay file written by [`write_numeric_updates`] into a
/// `docId -> newValue` map (a `HashMap` composes directly with the overlay
/// lookup in [`numeric_value_with_updates`]; ordering on disk was only ever
/// needed for the strictly-ascending validation below, not for the returned
/// shape).
pub fn read_numeric_updates(
    bytes: &[u8],
    segment_id: &[u8; ID_LENGTH],
    segment_suffix: &str,
) -> Result<HashMap<i32, i64>> {
    let mut input = SliceInput::new(bytes);
    codec_util::check_index_header(
        &mut input,
        NUMERIC_UPDATES_CODEC,
        VERSION_START,
        VERSION_CURRENT,
        segment_id,
        segment_suffix,
    )?;

    let count = input.read_vint()?;
    let mut map = HashMap::with_capacity(count.max(0) as usize);
    let mut prev: Option<i32> = None;
    for _ in 0..count {
        let doc = input.read_i32()?;
        let value = input.read_i64()?;
        if let Some(p) = prev {
            if doc <= p {
                return Err(Error::OutOfOrder { prev: p, next: doc });
            }
        }
        prev = Some(doc);
        map.insert(doc, value);
    }

    codec_util::check_footer(&mut input, bytes.len())?;
    Ok(map)
}

/// The overlay-aware numeric doc-values read: checks `updates` first (the
/// incremental overlay), falling back to the existing full
/// [`doc_values::numeric_value`] base decode when `doc` isn't present in the
/// overlay. This is the "read through the update" half of the mechanism --
/// composing an already-open base doc-values entry with an already-decoded
/// overlay map, no file I/O of its own.
///
/// `Ok(None)` means `doc` legitimately has no value in either the overlay or
/// the base (matching [`doc_values::numeric_value`]'s own `None` meaning).
pub fn numeric_value_with_updates(
    base_entry: &NumericEntry,
    base_data: &[u8],
    updates: &HashMap<i32, i64>,
    doc_id: i32,
) -> doc_values::Result<Option<i64>> {
    if let Some(&value) = updates.get(&doc_id) {
        return Ok(Some(value));
    }
    doc_values::numeric_value(base_data, base_entry, doc_id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use lucene_store::codec_util::ID_LENGTH;

    const SEG_ID: [u8; ID_LENGTH] = [7u8; ID_LENGTH];

    #[test]
    fn overlay_round_trip() {
        let updates = [(0, 100i64), (5, 200), (3, 300)];
        let bytes = write_numeric_updates(&updates, &SEG_ID, "");
        let map = read_numeric_updates(&bytes, &SEG_ID, "").unwrap();
        assert_eq!(map.len(), 3);
        assert_eq!(map.get(&0), Some(&100));
        assert_eq!(map.get(&5), Some(&200));
        assert_eq!(map.get(&3), Some(&300));
    }

    #[test]
    fn overlay_round_trip_unsorted_input_and_duplicate_doc_keeps_last() {
        // Doc 2 appears twice; the later entry (value 99) should win, matching
        // last-write-wins semantics for a single buffered update batch.
        let updates = [(5, 1i64), (2, 42), (2, 99)];
        let bytes = write_numeric_updates(&updates, &SEG_ID, "");
        let map = read_numeric_updates(&bytes, &SEG_ID, "").unwrap();
        assert_eq!(map.len(), 2);
        assert_eq!(map.get(&2), Some(&99));
        assert_eq!(map.get(&5), Some(&1));
    }

    #[test]
    fn empty_overlay_round_trips_to_empty_map() {
        let bytes = write_numeric_updates(&[], &SEG_ID, "");
        let map = read_numeric_updates(&bytes, &SEG_ID, "").unwrap();
        assert!(map.is_empty());
    }

    #[test]
    fn wrong_segment_id_rejected() {
        let bytes = write_numeric_updates(&[(0, 1)], &SEG_ID, "");
        let other_id = [9u8; ID_LENGTH];
        assert!(read_numeric_updates(&bytes, &other_id, "").is_err());
    }

    #[test]
    fn truncated_file_rejected() {
        let bytes = write_numeric_updates(&[(0, 1), (1, 2)], &SEG_ID, "");
        let truncated = &bytes[..bytes.len() - 4];
        assert!(read_numeric_updates(truncated, &SEG_ID, "").is_err());
    }

    #[test]
    fn hand_built_out_of_order_doc_ids_rejected() {
        // Hand-build a file with doc ids [5, 3] (not ascending) to exercise
        // the OutOfOrder error path -- write_numeric_updates itself always
        // sorts, so this scenario can only be reached via a malformed file.
        let mut out: Vec<u8> = Vec::new();
        codec_util::write_index_header(
            &mut out,
            NUMERIC_UPDATES_CODEC,
            VERSION_CURRENT,
            &SEG_ID,
            "",
        );
        out.write_vint(2);
        out.write_i32(5);
        out.write_i64(1);
        out.write_i32(3);
        out.write_i64(2);
        codec_util::write_footer(&mut out);
        assert!(matches!(
            read_numeric_updates(&out, &SEG_ID, ""),
            Err(Error::OutOfOrder { prev: 5, next: 3 })
        ));
    }

    // --- numeric_value_with_updates ---

    fn dense_entry_and_data() -> (NumericEntry, Vec<u8>) {
        // A trivially simple dense field: 4 docs, values [10, 20, 30, 40],
        // plain (no table, gcd=1, min=0), built via the real writer so this
        // stays honest about the base format rather than hand-rolling one.
        let (meta, data, _skip) =
            doc_values::write_single_dense_numeric_field(0, &[10, 20, 30, 40], 4, &SEG_ID, "")
                .unwrap();
        let field_infos = crate::field_infos::FieldInfos {
            fields: vec![crate::field_infos::FieldInfo {
                name: "f".to_string(),
                number: 0,
                store_term_vectors: false,
                omit_norms: false,
                store_payloads: false,
                soft_deletes_field: false,
                parent_field: false,
                index_options: crate::field_infos::IndexOptions::None,
                doc_values_type: crate::field_infos::DocValuesType::Numeric,
                doc_values_skip_index_type: crate::field_infos::DocValuesSkipIndexType::None,
                doc_values_gen: -1,
                attributes: Vec::new(),
                point_dimension_count: 0,
                point_index_dimension_count: 0,
                point_num_bytes: 0,
                vector_dimension: 0,
                vector_encoding: crate::field_infos::VectorEncoding::Float32,
                vector_similarity_function: crate::field_infos::VectorSimilarityFunction::Euclidean,
            }],
        };
        let (_, parsed) = doc_values::parse_meta(&meta, &SEG_ID, "", &field_infos).unwrap();
        let entry = parsed.numeric_entry(0).unwrap().clone();
        (entry, data)
    }

    #[test]
    fn overlay_value_overrides_base_value_for_doc_present_in_both() {
        let (entry, data) = dense_entry_and_data();
        assert_eq!(
            doc_values::numeric_value(&data, &entry, 1).unwrap(),
            Some(20)
        );
        let mut updates = HashMap::new();
        updates.insert(1, 999i64);
        let result = numeric_value_with_updates(&entry, &data, &updates, 1).unwrap();
        assert_eq!(result, Some(999));
    }

    #[test]
    fn doc_absent_from_overlay_falls_back_to_base_value() {
        let (entry, data) = dense_entry_and_data();
        let mut updates = HashMap::new();
        updates.insert(1, 999i64);
        // Doc 2 isn't in the overlay -> falls back to its base value (30).
        let result = numeric_value_with_updates(&entry, &data, &updates, 2).unwrap();
        assert_eq!(result, Some(30));
    }

    #[test]
    fn empty_overlay_is_a_no_op_fallback_to_base_for_every_doc() {
        let (entry, data) = dense_entry_and_data();
        let updates = HashMap::new();
        for (doc, expected) in [(0, 10), (1, 20), (2, 30), (3, 40)] {
            let result = numeric_value_with_updates(&entry, &data, &updates, doc).unwrap();
            assert_eq!(result, Some(expected));
        }
    }
}
