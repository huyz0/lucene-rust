//! Doc-values **field updates** -- the `IndexWriter.updateNumericDocValue` /
//! `updateBinaryDocValue` mechanism, which changes a handful of documents'
//! doc-values without reindexing them.
//!
//! # What real Lucene does
//!
//! `IndexWriter.updateNumericDocValue` buffers `(Term, field, value)` triples
//! in a `DocValuesUpdate`; a flush resolves them to doc ids
//! (`NumericDocValuesFieldUpdates`/`BinaryDocValuesFieldUpdates`) and
//! `ReadersAndUpdates.writeFieldUpdates` writes them out. What it writes is
//! **a new generation of ordinary doc-values files for the updated field** --
//! the field's whole column, merged from the reader's current values and the
//! resolved updates, through `Lucene90DocValuesConsumer`. The base
//! `.dvm`/`.dvd` are left untouched but superseded for that field;
//! `SegmentCommitInfo.docValuesGen` and a per-field `dvUpdatesFiles` map
//! record the new files, and a matching `FieldInfos` generation records the
//! field's new `FieldInfo.docValuesGen` so a reader knows which generation to
//! resolve (`SegmentDocValuesProducer`).
//!
//! A generation can also *remove* a doc's value, not just change it --
//! `DocValuesFieldUpdates.reset(docId)`, which `IndexWriter.updateDocValues`
//! reaches whenever the caller passes a field with a null value. That is why
//! every value here is an `Option`: `Some(v)` sets the doc's value to `v`,
//! `None` makes the doc read back as having no value at all. Java encodes the
//! same distinction as a `hasValue` bit packed alongside each buffered doc id
//! (`DocValuesFieldUpdates.HAS_VALUE_MASK`); on disk it becomes the merged
//! column's `IndexedDISI` docs-with-field structure, or the `meta[-2, 0]`
//! "no documents with values" marker when a generation resets every value.
//!
//! # Two representations, one meaning
//!
//! **The index's representation is Lucene's own.** When
//! `IndexWriter.updateNumericDocValue`/`updateBinaryDocValue` are flushed,
//! `ReadersAndUpdates.writeFieldUpdates` does *not* write a delta file: it
//! rewrites the updated field's **whole column** through
//! `Lucene90DocValuesConsumer` into a new generation of ordinary doc-values
//! files, `_<segment>_<base36 gen>_Lucene90_0.{dvm,dvd,dvs}`, records them in
//! `SegmentCommitInfo.getDocValuesUpdatesFiles()` keyed by field number, and
//! writes a new `FieldInfos` generation (`_<segment>_<base36 gen>.fnm`)
//! carrying the field's new `FieldInfo.docValuesGen`. A reader then resolves,
//! per field, the generation named by that `docValuesGen`
//! (`SegmentDocValuesProducer`). [`merge_numeric_column`]/
//! [`merge_binary_column`] and [`write_numeric_generation`]/
//! [`write_binary_generation`] below are that path; `lucene_index::field_updates`
//! drives them and owns the file naming and the `SegmentCommitInfo`
//! bookkeeping.
//!
//! **The delta encoding below is not an index format.** The
//! `LuceneRustNumericDocValuesUpdates`/`LuceneRustBinaryDocValuesUpdates`
//! files [`write_numeric_updates`]/[`write_binary_updates`] produce are this
//! port's own invention and **nothing in an index references them any more**:
//! no `segments_N`, no `.si`, no `SegmentCommitInfo::files`. They survive as a
//! standalone serialization for `lucene_search::soft_deletes`'
//! `mark_soft_deleted_via_overlay`, which uses them to carry a soft-delete
//! marking outside an index. Do not reach for them when writing a segment:
//! a segment that references them is a segment real Lucene cannot open, which
//! is precisely the defect `docs/sweep/m2/c14-dv-updates-format.md` closes.
//!
//! Format of that standalone encoding: `codec_util` index header (codec name
//! [`NUMERIC_UPDATES_CODEC`], version [`VERSION_CURRENT`], a segment id +
//! suffix exactly like every other per-segment file in this crate), then a
//! `vint` count of entries, then that many entries in ascending `doc_id`
//! order (ascending order is enforced on write and validated on read,
//! matching this crate's other sorted-array formats), then a `codec_util`
//! footer (CRC32 checksum). An entry is `(i32 doc_id, u8 has_value)` followed
//! by an `i64 new_value` only when `has_value != 0` -- a `has_value` of `0`
//! is Java's `reset(doc)`, "this generation removes the doc's value".
//!
//! [`VERSION_START`] files predate the `has_value` byte and are still read:
//! every entry in one is a plain `(i32, i64)` value assignment.

use std::collections::HashMap;

use lucene_store::codec_util::{self, ID_LENGTH};
use lucene_store::data_input::{DataInput, SliceInput};
use lucene_store::DataOutput;

use crate::doc_values::{self, NumericEntry};

const NUMERIC_UPDATES_CODEC: &str = "LuceneRustNumericDocValuesUpdates";
/// The binary counterpart's codec name -- a distinct name so a binary overlay
/// file handed to [`read_numeric_updates`] (or the reverse) fails the header
/// check instead of decoding garbage.
const BINARY_UPDATES_CODEC: &str = "LuceneRustBinaryDocValuesUpdates";
const VERSION_START: i32 = 0;
/// Adds the per-entry `has_value` byte, so a generation can *remove* a doc's
/// value (`DocValuesFieldUpdates.reset`) and not only overwrite it.
const VERSION_HAS_VALUE: i32 = 1;
const VERSION_CURRENT: i32 = VERSION_HAS_VALUE;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Store(#[from] lucene_store::Error),
    #[error("doc ids must be written in strictly ascending order: {prev} then {next}")]
    OutOfOrder { prev: i32, next: i32 },
    #[error("binary update value length {0} is negative")]
    NegativeLength(i32),
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
    updates: &[(i32, Option<i64>)],
    segment_id: &[u8; ID_LENGTH],
    segment_suffix: &str,
) -> Vec<u8> {
    let mut sorted: Vec<(i32, Option<i64>)> = updates.to_vec();
    // Stable sort by doc id keeps later-in-input entries after earlier ones
    // for equal keys; then keep only the last entry per doc id below.
    sorted.sort_by_key(|&(doc, _)| doc);
    let mut deduped: Vec<(i32, Option<i64>)> = Vec::with_capacity(sorted.len());
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
        match value {
            Some(v) => {
                out.push(1);
                out.write_i64(*v);
            }
            // `reset(doc)`: the doc's value is removed by this generation, so
            // there is nothing to store for it.
            None => out.push(0),
        }
    }
    codec_util::write_footer(&mut out);
    out
}

/// Reads back an overlay file written by [`write_numeric_updates`] into a
/// `docId -> Option<newValue>` map (a `HashMap` composes directly with the
/// overlay lookup in [`numeric_value_with_updates`]; ordering on disk was
/// only ever needed for the strictly-ascending validation below, not for the
/// returned shape). A `None` value is Java's `reset(doc)` -- this generation
/// removes the doc's value rather than setting one.
pub fn read_numeric_updates(
    bytes: &[u8],
    segment_id: &[u8; ID_LENGTH],
    segment_suffix: &str,
) -> Result<HashMap<i32, Option<i64>>> {
    let mut input = SliceInput::new(bytes);
    let header = codec_util::check_index_header(
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
        // Files at `VERSION_START` predate the `has_value` byte; every entry
        // in one sets a value.
        let value = if header.version >= VERSION_HAS_VALUE {
            match input.read_byte()? {
                0 => None,
                _ => Some(input.read_i64()?),
            }
        } else {
            Some(input.read_i64()?)
        };
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
/// `Ok(None)` means `doc` legitimately has no value: either the newest
/// generation that touched it removed it (`reset`), or neither the overlay nor
/// the base ever had one (matching [`doc_values::numeric_value`]'s own `None`
/// meaning).
pub fn numeric_value_with_updates(
    base_entry: &NumericEntry,
    base_data: &[u8],
    updates: &HashMap<i32, Option<i64>>,
    doc_id: i32,
) -> doc_values::Result<Option<i64>> {
    if let Some(&value) = updates.get(&doc_id) {
        // Present with `None` is a `reset(doc)`: the doc has no value any
        // more, which is *not* the same as being absent from the overlay.
        return Ok(value);
    }
    doc_values::numeric_value(base_data, base_entry, doc_id)
}

/// The overlay-aware numeric doc-values read for **any number** of chained
/// update generations: checks `generations` from newest to oldest (later
/// entries in the slice win), falling back to the existing full
/// [`doc_values::numeric_value`] base decode when `doc` isn't present in any
/// generation. This is [`numeric_value_with_updates`] generalized from one
/// overlay layer to a whole ordered chain, matching real Lucene's
/// newest-generation-wins semantics when the same doc is touched more than
/// once across sequential update rounds.
///
/// `generations` must be in **ascending generation order** (oldest first --
/// generation 1 at index 0, generation 2 at index 1, and so on), the same
/// order real Lucene's `SegmentCommitInfo.docValuesGen` counter assigns as
/// updates accumulate. An empty slice degenerates to a plain base decode,
/// identical to [`numeric_value_with_updates`] with an empty map.
///
/// `Ok(None)` means `doc` legitimately has no value: either the newest
/// generation that touched it removed it (`reset`), or no generation and the
/// base ever had one (matching [`doc_values::numeric_value`]'s own `None`
/// meaning).
pub fn numeric_value_with_generations(
    base_entry: &NumericEntry,
    base_data: &[u8],
    generations: &[HashMap<i32, Option<i64>>],
    doc_id: i32,
) -> doc_values::Result<Option<i64>> {
    for generation in generations.iter().rev() {
        if let Some(&value) = generation.get(&doc_id) {
            // A `reset(doc)` in a newer generation wins over an older
            // generation's value and over the base, exactly like a value does.
            return Ok(value);
        }
    }
    doc_values::numeric_value(base_data, base_entry, doc_id)
}

/// The BINARY counterpart of [`write_numeric_updates`] -- real Lucene's
/// `BinaryDocValuesFieldUpdates`, reached from
/// `IndexWriter.updateBinaryDocValue(Term, String, BytesRef)` and from
/// `updateDocValues` with a `BinaryDocValuesField`.
///
/// Java runs both update types through the same `DocValuesFieldUpdates` base
/// class and differs only in how a value is buffered (`BytesRefBuilder` +
/// `PagedMutable` offsets/lengths instead of a `PagedMutable` of longs), so
/// the semantics ported here are identical to the numeric side's: ascending
/// by doc, the last write for a doc wins within one generation, and a `None`
/// value is `DocValuesFieldUpdates.reset(doc)` -- the doc's value is
/// *removed*, which is not the same as an empty `BytesRef`.
///
/// Format: the same `codec_util` index header/footer envelope
/// [`write_numeric_updates`] uses (codec [`BINARY_UPDATES_CODEC`]), a `vint`
/// entry count, then per entry `(i32 doc_id, u8 has_value)` followed, when
/// `has_value != 0`, by a `vint` byte length and that many bytes. There is no
/// [`VERSION_START`]-era binary format to stay compatible with: the binary
/// side did not exist before [`VERSION_HAS_VALUE`], so its `has_value` byte is
/// unconditional.
pub fn write_binary_updates(
    updates: &[(i32, Option<Vec<u8>>)],
    segment_id: &[u8; ID_LENGTH],
    segment_suffix: &str,
) -> Vec<u8> {
    let mut sorted: Vec<(i32, Option<Vec<u8>>)> = updates.to_vec();
    // Stable sort + keep-last is exactly the numeric side's dedup, and for the
    // same reason: `DocValuesFieldUpdates.finish()` stable-sorts by doc so the
    // last `add` for a doc is the one an iterator sees.
    sorted.sort_by_key(|(doc, _)| *doc);
    let mut deduped: Vec<(i32, Option<Vec<u8>>)> = Vec::with_capacity(sorted.len());
    for (doc, value) in sorted {
        if deduped.last().is_some_and(|(last_doc, _)| *last_doc == doc) {
            *deduped.last_mut().unwrap() = (doc, value);
        } else {
            deduped.push((doc, value));
        }
    }

    let mut out: Vec<u8> = Vec::new();
    codec_util::write_index_header(
        &mut out,
        BINARY_UPDATES_CODEC,
        VERSION_CURRENT,
        segment_id,
        segment_suffix,
    );
    out.write_vint(deduped.len() as i32);
    for (doc, value) in &deduped {
        out.write_i32(*doc);
        match value {
            Some(bytes) => {
                out.push(1);
                out.write_vint(bytes.len() as i32);
                out.write_bytes(bytes);
            }
            None => out.push(0),
        }
    }
    codec_util::write_footer(&mut out);
    out
}

/// Reads back an overlay written by [`write_binary_updates`]. A `None` value
/// is Java's `reset(doc)`; `Some(vec![])` is a genuine empty value, and the
/// two are deliberately distinguishable (an empty `BytesRef` is a legal
/// `BinaryDocValues` value).
pub fn read_binary_updates(
    bytes: &[u8],
    segment_id: &[u8; ID_LENGTH],
    segment_suffix: &str,
) -> Result<HashMap<i32, Option<Vec<u8>>>> {
    let mut input = SliceInput::new(bytes);
    codec_util::check_index_header(
        &mut input,
        BINARY_UPDATES_CODEC,
        VERSION_HAS_VALUE,
        VERSION_CURRENT,
        segment_id,
        segment_suffix,
    )?;

    let count = input.read_vint()?;
    let mut map = HashMap::with_capacity(count.max(0) as usize);
    let mut prev: Option<i32> = None;
    for _ in 0..count {
        let doc = input.read_i32()?;
        let value = match input.read_byte()? {
            0 => None,
            _ => {
                let len = input.read_vint()?;
                if len < 0 {
                    return Err(Error::NegativeLength(len));
                }
                let mut buf = vec![0u8; len as usize];
                input.read_bytes(&mut buf)?;
                Some(buf)
            }
        };
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

/// [`numeric_value_with_updates`] for BINARY doc values: the overlay wins over
/// the base decode, and an overlay entry of `None` is a `reset(doc)` that
/// shadows the base rather than falling through to it.
pub fn binary_value_with_updates<'d>(
    base_entry: &doc_values::BinaryEntry,
    base_data: &'d [u8],
    updates: &'d HashMap<i32, Option<Vec<u8>>>,
    doc_id: i32,
) -> doc_values::Result<Option<&'d [u8]>> {
    if let Some(value) = updates.get(&doc_id) {
        return Ok(value.as_deref());
    }
    doc_values::binary_value(base_data, base_entry, doc_id)
}

/// [`numeric_value_with_generations`] for BINARY doc values: `generations`
/// must be in ascending generation order (oldest first), and the newest
/// generation that touched `doc_id` wins -- whether it sets a value or resets
/// it.
pub fn binary_value_with_generations<'d>(
    base_entry: &doc_values::BinaryEntry,
    base_data: &'d [u8],
    generations: &'d [HashMap<i32, Option<Vec<u8>>>],
    doc_id: i32,
) -> doc_values::Result<Option<&'d [u8]>> {
    for generation in generations.iter().rev() {
        if let Some(value) = generation.get(&doc_id) {
            return Ok(value.as_deref());
        }
    }
    doc_values::binary_value(base_data, base_entry, doc_id)
}

// ---------------------------------------------------------------------------
// Lucene's actual on-disk representation of a doc-values update
// ---------------------------------------------------------------------------

/// `ReadersAndUpdates.handleDVUpdates`' merge step for one NUMERIC field:
/// materialises the field's **whole** column as it will look after `updates`
/// are applied, so the caller can hand it to
/// [`write_numeric_generation`] and get the `.dvm`/`.dvd`/`.dvs` triple real
/// Lucene writes for a doc-values update generation.
///
/// This is Java's `MergedDocValues` over `reader.getNumericDocValues(field)`
/// and `DocValuesFieldUpdates.mergedIterator(subs)`, collapsed into an array
/// because the whole column is being rewritten anyway: Java's merge-sort of
/// two iterators exists to avoid materialising, not to change the answer.
///
/// - `base` is the field's current column (`None` when the segment carries no
///   doc values for this field at all, which is Java's
///   `FieldInfos.FieldNumbers.constructFieldInfo` case -- the field exists
///   globally but never had a value in this segment).
/// - `updates` is every resolved `(doc, value)` pair, in the order the packets
///   produced them. A later entry for the same doc wins (Java's
///   `mergedIterator` breaks ties on `delGen`, and the caller appends in
///   ascending packet order), and `None` is
///   `DocValuesFieldUpdates.reset(doc)` -- the doc's value is *removed*, which
///   is why the result is `Option<i64>` per doc rather than `i64`.
///
/// The returned vector always has exactly `max_doc` entries.
pub fn merge_numeric_column(
    base: Option<(&NumericEntry, &[u8])>,
    updates: &[(i32, Option<i64>)],
    max_doc: i32,
) -> doc_values::Result<Vec<Option<i64>>> {
    let mut column: Vec<Option<i64>> = match base {
        Some((entry, data)) => {
            // A **cursor**, not `numeric_value` per document: the free
            // function re-derives everything on every call, so a sparse
            // column would cost one walk of the `IndexedDISI` block headers
            // per document -- `O(maxDoc x blocks)` where Java's
            // `MergedDocValues` makes one forward pass. `doc_values.rs`'s own
            // module doc calls this out as the rule for any multi-lookup
            // caller, and a whole-column rewrite is the largest one there is.
            let mut reader = doc_values::NumericReader::new(data, entry);
            (0..max_doc)
                .map(|doc| reader.value(doc))
                .collect::<doc_values::Result<Vec<_>>>()?
        }
        None => vec![None; max_doc.max(0) as usize],
    };
    for &(doc, value) in updates {
        if doc < 0 || doc >= max_doc {
            return Err(doc_values::Error::DocOutOfRange(doc, max_doc as i64));
        }
        column[doc as usize] = value;
    }
    Ok(column)
}

/// [`merge_numeric_column`] for BINARY doc values -- same contract, same
/// `None`-means-`reset(doc)` rule, and the same "an empty `Vec<u8>` is a
/// present value" distinction the overlay side already draws.
pub fn merge_binary_column(
    base: Option<(&doc_values::BinaryEntry, &[u8])>,
    updates: &[(i32, Option<Vec<u8>>)],
    max_doc: i32,
) -> doc_values::Result<Vec<Option<Vec<u8>>>> {
    let mut column: Vec<Option<Vec<u8>>> = match base {
        Some((entry, data)) => {
            // Same rule as the numeric side: one forward cursor, not one
            // `binary_value` re-derivation per document.
            let mut reader = doc_values::BinaryReader::new(data, entry);
            (0..max_doc)
                .map(|doc| reader.value(doc).map(|v| v.map(<[u8]>::to_vec)))
                .collect::<doc_values::Result<Vec<_>>>()?
        }
        None => vec![None; max_doc.max(0) as usize],
    };
    for (doc, value) in updates {
        if *doc < 0 || *doc >= max_doc {
            return Err(doc_values::Error::DocOutOfRange(*doc, max_doc as i64));
        }
        column[*doc as usize] = value.clone();
    }
    Ok(column)
}

/// Writes one NUMERIC doc-values **update generation**: the field's whole
/// merged column as a standalone `Lucene90DocValuesFormat`
/// `.dvm`/`.dvd`/`.dvs` triple, which is exactly what
/// `Lucene90DocValuesConsumer.addNumericField` produces for
/// `ReadersAndUpdates.handleDVUpdates`.
///
/// The caller supplies `segment_suffix`; Java builds it as
/// `PerFieldDocValuesFormat.getFullSegmentSuffix(Long.toString(gen, 36),
/// "Lucene90_0")`, i.e. `"<base36 gen>_Lucene90_0"`, and uses the same string
/// both in the file names (`_0_1_Lucene90_0.dvd`) and inside each file's index
/// header. Getting the two out of step is silent: the file exists, and
/// `checkIndexHeader` rejects it.
///
/// The dense/sparse/empty choice is `writeValues`', keyed on how many docs the
/// merged column leaves with a value -- a generation that only *resets* values
/// legitimately yields the empty (`meta[-2, 0]`) shape.
pub fn write_numeric_generation(
    field_number: i32,
    column: &[Option<i64>],
    segment_id: &[u8; ID_LENGTH],
    segment_suffix: &str,
) -> doc_values::WriteResult<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    let max_doc = column.len() as i32;
    if column.iter().all(Option::is_some) {
        let dense: Vec<i64> = column
            .iter()
            .map(|v| v.expect("checked all-some"))
            .collect();
        doc_values::write_single_dense_numeric_field(
            field_number,
            &dense,
            max_doc,
            segment_id,
            segment_suffix,
        )
    } else {
        let sparse: Vec<(i32, i64)> = column
            .iter()
            .enumerate()
            .filter_map(|(doc, v)| v.map(|v| (doc as i32, v)))
            .collect();
        doc_values::write_single_sparse_numeric_field(
            field_number,
            &sparse,
            max_doc,
            segment_id,
            segment_suffix,
        )
    }
}

/// [`write_numeric_generation`] for BINARY doc values --
/// `Lucene90DocValuesConsumer.addBinaryField` over the merged column.
pub fn write_binary_generation(
    field_number: i32,
    column: &[Option<Vec<u8>>],
    segment_id: &[u8; ID_LENGTH],
    segment_suffix: &str,
) -> doc_values::WriteResult<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    let max_doc = column.len() as i32;
    if column.iter().all(Option::is_some) {
        let dense: Vec<Vec<u8>> = column
            .iter()
            .map(|v| v.clone().expect("checked all-some"))
            .collect();
        doc_values::write_single_dense_binary_field(
            field_number,
            &dense,
            max_doc,
            segment_id,
            segment_suffix,
        )
    } else {
        let sparse: Vec<(i32, Vec<u8>)> = column
            .iter()
            .enumerate()
            .filter_map(|(doc, v)| v.clone().map(|v| (doc as i32, v)))
            .collect();
        doc_values::write_single_sparse_binary_field(
            field_number,
            &sparse,
            max_doc,
            segment_id,
            segment_suffix,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lucene_store::codec_util::ID_LENGTH;

    const SEG_ID: [u8; ID_LENGTH] = [7u8; ID_LENGTH];

    #[test]
    fn overlay_round_trip() {
        let updates = [(0, Some(100i64)), (5, Some(200)), (3, Some(300))];
        let bytes = write_numeric_updates(&updates, &SEG_ID, "");
        let map = read_numeric_updates(&bytes, &SEG_ID, "").unwrap();
        assert_eq!(map.len(), 3);
        assert_eq!(map.get(&0), Some(&Some(100)));
        assert_eq!(map.get(&5), Some(&Some(200)));
        assert_eq!(map.get(&3), Some(&Some(300)));
    }

    #[test]
    fn overlay_round_trip_unsorted_input_and_duplicate_doc_keeps_last() {
        // Doc 2 appears twice; the later entry (value 99) should win, matching
        // last-write-wins semantics for a single buffered update batch.
        let updates = [(5, Some(1i64)), (2, Some(42)), (2, Some(99))];
        let bytes = write_numeric_updates(&updates, &SEG_ID, "");
        let map = read_numeric_updates(&bytes, &SEG_ID, "").unwrap();
        assert_eq!(map.len(), 2);
        assert_eq!(map.get(&2), Some(&Some(99)));
        assert_eq!(map.get(&5), Some(&Some(1)));
    }

    #[test]
    fn empty_overlay_round_trips_to_empty_map() {
        let bytes = write_numeric_updates(&[], &SEG_ID, "");
        let map = read_numeric_updates(&bytes, &SEG_ID, "").unwrap();
        assert!(map.is_empty());
    }

    #[test]
    fn wrong_segment_id_rejected() {
        let bytes = write_numeric_updates(&[(0, Some(1))], &SEG_ID, "");
        let other_id = [9u8; ID_LENGTH];
        assert!(read_numeric_updates(&bytes, &other_id, "").is_err());
    }

    #[test]
    fn truncated_file_rejected() {
        let bytes = write_numeric_updates(&[(0, Some(1)), (1, Some(2))], &SEG_ID, "");
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
        out.push(1);
        out.write_i64(1);
        out.write_i32(3);
        out.push(1);
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
        updates.insert(1, Some(999i64));
        let result = numeric_value_with_updates(&entry, &data, &updates, 1).unwrap();
        assert_eq!(result, Some(999));
    }

    #[test]
    fn doc_absent_from_overlay_falls_back_to_base_value() {
        let (entry, data) = dense_entry_and_data();
        let mut updates = HashMap::new();
        updates.insert(1, Some(999i64));
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

    // --- numeric_value_with_generations (multi-generation overlay chain) ---

    #[test]
    fn three_generations_newest_wins_for_a_doc_touched_by_all_three() {
        let (entry, data) = dense_entry_and_data();
        // Doc 1 (base value 20) gets updated at generation 1, then again at
        // generation 2, then again at generation 3 -- generation 3's value
        // must win, matching newest-generation-wins semantics.
        let gen1 = HashMap::from([(1, Some(1_001i64))]);
        let gen2 = HashMap::from([(1, Some(1_002i64))]);
        let gen3 = HashMap::from([(1, Some(1_003i64))]);
        let generations = [gen1, gen2, gen3];
        let result = numeric_value_with_generations(&entry, &data, &generations, 1).unwrap();
        assert_eq!(result, Some(1_003));
    }

    #[test]
    fn overlapping_doc_sets_across_generations_each_doc_takes_its_own_newest_write() {
        let (entry, data) = dense_entry_and_data();
        // gen1 touches docs 0 and 1; gen2 touches docs 1 and 2 (overlapping
        // on doc 1, where gen2 must win); gen3 touches only doc 2 again.
        let gen1 = HashMap::from([(0, Some(100i64)), (1, Some(101))]);
        let gen2 = HashMap::from([(1, Some(201i64)), (2, Some(202))]);
        let gen3 = HashMap::from([(2, Some(302i64))]);
        let generations = [gen1, gen2, gen3];

        // Doc 0: only gen1 touched it -> gen1's value.
        assert_eq!(
            numeric_value_with_generations(&entry, &data, &generations, 0).unwrap(),
            Some(100)
        );
        // Doc 1: gen1 then gen2 touched it -> gen2 (newer) wins.
        assert_eq!(
            numeric_value_with_generations(&entry, &data, &generations, 1).unwrap(),
            Some(201)
        );
        // Doc 2: gen2 then gen3 touched it -> gen3 (newest) wins.
        assert_eq!(
            numeric_value_with_generations(&entry, &data, &generations, 2).unwrap(),
            Some(302)
        );
    }

    #[test]
    fn doc_untouched_by_any_generation_falls_back_to_base() {
        let (entry, data) = dense_entry_and_data();
        let gen1 = HashMap::from([(0, Some(900i64))]);
        let gen2 = HashMap::from([(1, Some(901i64))]);
        let generations = [gen1, gen2];
        // Doc 3 (base value 40) isn't in either generation.
        let result = numeric_value_with_generations(&entry, &data, &generations, 3).unwrap();
        assert_eq!(result, Some(40));
    }

    #[test]
    fn empty_generation_chain_degenerates_to_plain_base_decode() {
        let (entry, data) = dense_entry_and_data();
        let generations: [HashMap<i32, Option<i64>>; 0] = [];
        for (doc, expected) in [(0, 10), (1, 20), (2, 30), (3, 40)] {
            let result = numeric_value_with_generations(&entry, &data, &generations, doc).unwrap();
            assert_eq!(result, Some(expected));
        }
    }

    #[test]
    fn a_generation_that_reverts_to_an_earlier_generations_untouched_state_still_falls_through() {
        let (entry, data) = dense_entry_and_data();
        // gen1 touches doc 2; gen2 touches a disjoint doc (0) only, so for
        // doc 2 the chain must fall through past gen2 to gen1's write.
        let gen1 = HashMap::from([(2, Some(555i64))]);
        let gen2 = HashMap::from([(0, Some(777i64))]);
        let generations = [gen1, gen2];
        let result = numeric_value_with_generations(&entry, &data, &generations, 2).unwrap();
        assert_eq!(result, Some(555));
    }

    #[test]
    fn generations_can_be_written_and_read_back_via_existing_single_generation_io_then_chained() {
        // Proves the chain composes with the *unmodified* per-generation
        // write_numeric_updates/read_numeric_updates I/O -- each generation
        // really is just a standalone file, as the module doc comment says.
        let (entry, data) = dense_entry_and_data();
        let gen1_bytes = write_numeric_updates(&[(1, Some(111i64))], &SEG_ID, "");
        let gen2_bytes = write_numeric_updates(&[(1, Some(222i64)), (2, Some(322))], &SEG_ID, "");
        let gen1 = read_numeric_updates(&gen1_bytes, &SEG_ID, "").unwrap();
        let gen2 = read_numeric_updates(&gen2_bytes, &SEG_ID, "").unwrap();
        let generations = [gen1, gen2];

        // Doc 1: both generations touched it -> gen2 (newer) wins.
        assert_eq!(
            numeric_value_with_generations(&entry, &data, &generations, 1).unwrap(),
            Some(222)
        );
        // Doc 2: only gen2 touched it.
        assert_eq!(
            numeric_value_with_generations(&entry, &data, &generations, 2).unwrap(),
            Some(322)
        );
        // Doc 0: untouched by either generation -> base value 10.
        assert_eq!(
            numeric_value_with_generations(&entry, &data, &generations, 0).unwrap(),
            Some(10)
        );
    }

    // --- reset(doc): a generation that removes a doc's value ---

    #[test]
    fn reset_entry_round_trips_as_a_none_value() {
        let bytes = write_numeric_updates(&[(1, None), (2, Some(5i64))], &SEG_ID, "");
        let map = read_numeric_updates(&bytes, &SEG_ID, "").unwrap();
        assert_eq!(map.get(&1), Some(&None));
        assert_eq!(map.get(&2), Some(&Some(5)));
        // "reset" is not the same as "absent from the overlay".
        assert!(map.contains_key(&1));
        assert_eq!(map.get(&3), None);
    }

    /// `DocValuesFieldUpdates.reset(doc)` makes the doc read back as having no
    /// value at all, shadowing whatever the base `.dvd` holds -- unlike a doc
    /// the overlay never touched, which falls through to the base.
    #[test]
    fn reset_shadows_the_base_value() {
        let (entry, data) = dense_entry_and_data();
        let updates = HashMap::from([(1, None)]);
        assert_eq!(
            doc_values::numeric_value(&data, &entry, 1).unwrap(),
            Some(20)
        );
        assert_eq!(
            numeric_value_with_updates(&entry, &data, &updates, 1).unwrap(),
            None
        );
        // Doc 2 is untouched: still the base value.
        assert_eq!(
            numeric_value_with_updates(&entry, &data, &updates, 2).unwrap(),
            Some(30)
        );
    }

    /// Across generations the newest write wins whether it is a value or a
    /// reset, in both directions.
    #[test]
    fn newest_generation_wins_whether_it_sets_or_resets() {
        let (entry, data) = dense_entry_and_data();
        // Doc 0: set at gen1, reset at gen2 -> no value.
        // Doc 1: reset at gen1, set at gen2 -> gen2's value.
        let gen1 = HashMap::from([(0, Some(11i64)), (1, None)]);
        let gen2 = HashMap::from([(0, None), (1, Some(22i64))]);
        let generations = [gen1, gen2];
        assert_eq!(
            numeric_value_with_generations(&entry, &data, &generations, 0).unwrap(),
            None
        );
        assert_eq!(
            numeric_value_with_generations(&entry, &data, &generations, 1).unwrap(),
            Some(22)
        );
    }

    /// A reset in an older generation still shadows the base when no newer
    /// generation touches the doc.
    #[test]
    fn an_older_generations_reset_survives_a_disjoint_newer_generation() {
        let (entry, data) = dense_entry_and_data();
        let gen1 = HashMap::from([(2, None)]);
        let gen2 = HashMap::from([(0, Some(7i64))]);
        let generations = [gen1, gen2];
        assert_eq!(
            numeric_value_with_generations(&entry, &data, &generations, 2).unwrap(),
            None
        );
    }

    /// A `VERSION_START` file predates the per-entry `has_value` byte; every
    /// entry in one is a plain value assignment and must still decode.
    #[test]
    fn version_start_file_without_has_value_bytes_still_reads() {
        let mut out: Vec<u8> = Vec::new();
        codec_util::write_index_header(&mut out, NUMERIC_UPDATES_CODEC, VERSION_START, &SEG_ID, "");
        out.write_vint(2);
        out.write_i32(1);
        out.write_i64(111);
        out.write_i32(4);
        out.write_i64(444);
        codec_util::write_footer(&mut out);

        let map = read_numeric_updates(&out, &SEG_ID, "").unwrap();
        assert_eq!(map.get(&1), Some(&Some(111)));
        assert_eq!(map.get(&4), Some(&Some(444)));
    }

    // --- BINARY updates (`BinaryDocValuesFieldUpdates`) ---

    fn bin(bytes: &[u8]) -> Option<Vec<u8>> {
        Some(bytes.to_vec())
    }

    #[test]
    fn binary_overlay_round_trips_variable_length_values() {
        let updates = [
            (0, bin(b"alpha")),
            (5, bin(b"")),
            (3, bin(b"a much longer value than the others")),
        ];
        let bytes = write_binary_updates(&updates, &SEG_ID, "");
        let map = read_binary_updates(&bytes, &SEG_ID, "").unwrap();
        assert_eq!(map.len(), 3);
        assert_eq!(map[&0].as_deref(), Some(&b"alpha"[..]));
        // An empty value is a *value*, not an absence.
        assert_eq!(map[&5].as_deref(), Some(&b""[..]));
        assert_eq!(
            map[&3].as_deref(),
            Some(&b"a much longer value than the others"[..])
        );
    }

    #[test]
    fn binary_reset_round_trips_as_a_none_value_distinct_from_an_empty_one() {
        let updates = [(0, None), (1, bin(b""))];
        let bytes = write_binary_updates(&updates, &SEG_ID, "");
        let map = read_binary_updates(&bytes, &SEG_ID, "").unwrap();
        assert_eq!(map[&0], None);
        assert_eq!(map[&1], Some(Vec::new()));
    }

    #[test]
    fn binary_overlay_unsorted_input_and_duplicate_doc_keeps_last() {
        let updates = [(2, bin(b"first")), (0, bin(b"zero")), (2, bin(b"second"))];
        let bytes = write_binary_updates(&updates, &SEG_ID, "");
        let map = read_binary_updates(&bytes, &SEG_ID, "").unwrap();
        assert_eq!(map.len(), 2);
        assert_eq!(map[&2].as_deref(), Some(&b"second"[..]));
    }

    #[test]
    fn binary_overlay_rejects_a_numeric_overlay_file() {
        // Distinct codec names: handing a numeric overlay to the binary reader
        // must fail the header check rather than decode nonsense.
        let numeric = write_numeric_updates(&[(0, Some(1))], &SEG_ID, "");
        assert!(read_binary_updates(&numeric, &SEG_ID, "").is_err());
        let binary = write_binary_updates(&[(0, bin(b"x"))], &SEG_ID, "");
        assert!(read_numeric_updates(&binary, &SEG_ID, "").is_err());
    }

    #[test]
    fn binary_overlay_rejects_a_wrong_segment_id_and_a_corrupt_footer() {
        let bytes = write_binary_updates(&[(0, bin(b"x"))], &SEG_ID, "");
        assert!(read_binary_updates(&bytes, &[9u8; ID_LENGTH], "").is_err());
        let mut corrupt = bytes.clone();
        let last = corrupt.len() - 1;
        corrupt[last] ^= 0xFF;
        assert!(read_binary_updates(&corrupt, &SEG_ID, "").is_err());
    }

    #[test]
    fn binary_overlay_rejects_out_of_order_doc_ids_on_read() {
        let mut out: Vec<u8> = Vec::new();
        codec_util::write_index_header(
            &mut out,
            BINARY_UPDATES_CODEC,
            VERSION_CURRENT,
            &SEG_ID,
            "",
        );
        out.write_vint(2);
        out.write_i32(5);
        out.push(0);
        out.write_i32(2);
        out.push(0);
        codec_util::write_footer(&mut out);
        assert!(matches!(
            read_binary_updates(&out, &SEG_ID, ""),
            Err(Error::OutOfOrder { prev: 5, next: 2 })
        ));
    }

    #[test]
    fn binary_overlay_rejects_a_negative_value_length() {
        let mut out: Vec<u8> = Vec::new();
        codec_util::write_index_header(
            &mut out,
            BINARY_UPDATES_CODEC,
            VERSION_CURRENT,
            &SEG_ID,
            "",
        );
        out.write_vint(1);
        out.write_i32(0);
        out.push(1);
        out.write_vint(-1);
        codec_util::write_footer(&mut out);
        assert!(matches!(
            read_binary_updates(&out, &SEG_ID, ""),
            Err(Error::NegativeLength(-1))
        ));
    }

    /// The overlay-read helpers against an *empty* base entry: a binary field
    /// with no values at all, so every answer comes from the overlay chain.
    /// `BinaryEntry::is_empty_field()` short-circuits the base decode, which
    /// is what lets this test exercise resolution without a real `.dvd`.
    fn empty_binary_entry() -> doc_values::BinaryEntry {
        doc_values::BinaryEntry {
            field_number: 0,
            // `DOCS_WITH_FIELD_EMPTY` (-2): `is_empty_field()` is true, so
            // `binary_value` returns `None` without touching `base_data`.
            docs_with_field_offset: -2,
            docs_with_field_length: 0,
            jump_table_entry_count: 0,
            dense_rank_power: 0,
            num_docs_with_field: 0,
            min_length: 0,
            max_length: 0,
            data_offset: 0,
            data_length: 0,
            addresses: None,
        }
    }

    #[test]
    fn binary_overlay_read_prefers_the_overlay_then_falls_back_to_the_base() {
        let entry = empty_binary_entry();
        let mut updates = HashMap::new();
        updates.insert(0, bin(b"set"));
        updates.insert(1, None);
        assert_eq!(
            binary_value_with_updates(&entry, &[], &updates, 0).unwrap(),
            Some(&b"set"[..])
        );
        // A `reset` shadows the base rather than falling through to it.
        assert_eq!(
            binary_value_with_updates(&entry, &[], &updates, 1).unwrap(),
            None
        );
        // Untouched doc: the base decode (an empty field) answers.
        assert_eq!(
            binary_value_with_updates(&entry, &[], &updates, 2).unwrap(),
            None
        );
    }

    #[test]
    fn binary_newest_generation_wins_whether_it_sets_or_resets() {
        let entry = empty_binary_entry();
        let mut older = HashMap::new();
        older.insert(0, bin(b"old"));
        older.insert(1, bin(b"kept"));
        let mut newer = HashMap::new();
        newer.insert(0, bin(b"new"));
        newer.insert(2, None);
        let generations = vec![older, newer];

        assert_eq!(
            binary_value_with_generations(&entry, &[], &generations, 0).unwrap(),
            Some(&b"new"[..])
        );
        // A doc the newer generation did not touch falls through to the older.
        assert_eq!(
            binary_value_with_generations(&entry, &[], &generations, 1).unwrap(),
            Some(&b"kept"[..])
        );
        assert_eq!(
            binary_value_with_generations(&entry, &[], &generations, 2).unwrap(),
            None
        );
    }

    #[test]
    fn binary_an_older_generations_reset_survives_a_disjoint_newer_generation() {
        let entry = empty_binary_entry();
        let mut older = HashMap::new();
        older.insert(0, None);
        let mut newer = HashMap::new();
        newer.insert(1, bin(b"other"));
        let generations = vec![older, newer];
        assert_eq!(
            binary_value_with_generations(&entry, &[], &generations, 0).unwrap(),
            None
        );
    }

    #[test]
    fn binary_no_generations_degenerates_to_the_base_decode() {
        let entry = empty_binary_entry();
        assert_eq!(
            binary_value_with_generations(&entry, &[], &[], 0).unwrap(),
            None
        );
    }

    // -----------------------------------------------------------------
    // The generational (real-Lucene) representation
    // -----------------------------------------------------------------

    /// Writes a base NUMERIC column and reads its entry back, so the merge
    /// tests below start from a genuine `.dvd`/`.dvm` rather than a
    /// hand-built entry.
    fn base_numeric(values: &[Option<i64>]) -> (NumericEntry, Vec<u8>) {
        let (meta, data, _) =
            write_numeric_generation(0, values, &SEG_ID, "").expect("write base column");
        let fis = numeric_field_infos();
        let (_, parsed) = doc_values::parse_meta(&meta, &SEG_ID, "", &fis).expect("parse base");
        (
            parsed.numeric_entry(0).expect("numeric entry").clone(),
            data,
        )
    }

    fn numeric_field_infos() -> crate::field_infos::FieldInfos {
        crate::field_infos::FieldInfos {
            fields: vec![dv_field(crate::field_infos::DocValuesType::Numeric)],
        }
    }

    fn binary_field_infos() -> crate::field_infos::FieldInfos {
        crate::field_infos::FieldInfos {
            fields: vec![dv_field(crate::field_infos::DocValuesType::Binary)],
        }
    }

    fn dv_field(ty: crate::field_infos::DocValuesType) -> crate::field_infos::FieldInfo {
        crate::field_infos::FieldInfo {
            name: "f".to_string(),
            number: 0,
            store_term_vectors: false,
            omit_norms: true,
            store_payloads: false,
            soft_deletes_field: false,
            parent_field: false,
            index_options: crate::field_infos::IndexOptions::None,
            doc_values_type: ty,
            doc_values_skip_index_type: crate::field_infos::DocValuesSkipIndexType::None,
            doc_values_gen: -1,
            attributes: vec![],
            point_dimension_count: 0,
            point_index_dimension_count: 0,
            point_num_bytes: 0,
            vector_dimension: 0,
            vector_encoding: crate::field_infos::VectorEncoding::Float32,
            vector_similarity_function: crate::field_infos::VectorSimilarityFunction::Euclidean,
        }
    }

    fn read_numeric_column(meta: &[u8], data: &[u8], max_doc: i32) -> Vec<Option<i64>> {
        let fis = numeric_field_infos();
        let (_, parsed) = doc_values::parse_meta(meta, &SEG_ID, "", &fis).unwrap();
        let entry = parsed.numeric_entry(0).unwrap();
        (0..max_doc)
            .map(|d| doc_values::numeric_value(data, entry, d).unwrap())
            .collect()
    }

    fn read_binary_column(meta: &[u8], data: &[u8], max_doc: i32) -> Vec<Option<Vec<u8>>> {
        let fis = binary_field_infos();
        let (_, parsed) = doc_values::parse_meta(meta, &SEG_ID, "", &fis).unwrap();
        let entry = parsed.binary_entry(0).unwrap();
        (0..max_doc)
            .map(|d| {
                doc_values::binary_value(data, entry, d)
                    .unwrap()
                    .map(<[u8]>::to_vec)
            })
            .collect()
    }

    #[test]
    fn a_merged_numeric_column_keeps_every_document_the_update_did_not_touch() {
        // The single easiest thing to lose in a full-column rewrite, and the
        // one nothing structural catches: a generation that only holds the
        // updated documents reads back as "no value" for all the rest.
        let (entry, data) = base_numeric(&[Some(10), Some(20), Some(30), Some(40)]);
        let column =
            merge_numeric_column(Some((&entry, &data)), &[(1, Some(99)), (3, None)], 4).unwrap();
        assert_eq!(column, vec![Some(10), Some(99), Some(30), None]);
    }

    #[test]
    fn a_later_update_for_the_same_doc_wins() {
        let (entry, data) = base_numeric(&[Some(1), Some(2)]);
        let column = merge_numeric_column(
            Some((&entry, &data)),
            &[(0, Some(5)), (0, None), (0, Some(7))],
            2,
        )
        .unwrap();
        assert_eq!(column[0], Some(7));
    }

    #[test]
    fn a_field_with_no_base_column_starts_from_all_absent() {
        // Java's `FieldInfos.FieldNumbers.constructFieldInfo` case: the field
        // exists globally but never had a value in this segment.
        let column = merge_numeric_column(None, &[(2, Some(4))], 4).unwrap();
        assert_eq!(column, vec![None, None, Some(4), None]);
    }

    #[test]
    fn an_update_outside_the_segments_doc_range_is_an_error_not_a_silent_drop() {
        // A resolver that produced an out-of-range doc id has a bug, and
        // "that update silently didn't happen" is the worst way to learn it.
        // `doc_values::write_single_sparse_numeric_field` already errors on
        // the same condition; these two must agree.
        assert!(merge_numeric_column(None, &[(9, Some(2))], 3).is_err());
        assert!(merge_numeric_column(None, &[(-1, Some(1))], 3).is_err());
        assert!(merge_binary_column(None, &[(3, Some(vec![1]))], 3).is_err());
    }

    #[test]
    fn a_fully_dense_merged_column_round_trips_through_the_dense_writer() {
        let (meta, data, _) =
            write_numeric_generation(0, &[Some(3), Some(4), Some(5)], &SEG_ID, "").unwrap();
        assert_eq!(
            read_numeric_column(&meta, &data, 3),
            vec![Some(3), Some(4), Some(5)]
        );
    }

    #[test]
    fn a_partly_reset_merged_column_round_trips_through_the_sparse_writer() {
        let (meta, data, _) =
            write_numeric_generation(0, &[Some(3), None, Some(5)], &SEG_ID, "").unwrap();
        assert_eq!(
            read_numeric_column(&meta, &data, 3),
            vec![Some(3), None, Some(5)]
        );
    }

    #[test]
    fn a_generation_that_reset_every_value_round_trips_as_an_empty_column() {
        // `updateDocValues(term, field with a null value)` matching every
        // document. Java writes the `meta[-2, 0]` "no documents with values"
        // shape here, not a zero-length DISI.
        let (meta, data, _) = write_numeric_generation(0, &[None, None], &SEG_ID, "").unwrap();
        assert_eq!(read_numeric_column(&meta, &data, 2), vec![None, None]);
    }

    #[test]
    fn a_merged_binary_column_distinguishes_an_empty_value_from_a_removed_one() {
        let (meta, data, _) = write_binary_generation(
            0,
            &[Some(Vec::new()), None, Some(b"x".to_vec())],
            &SEG_ID,
            "",
        )
        .unwrap();
        assert_eq!(
            read_binary_column(&meta, &data, 3),
            vec![Some(Vec::new()), None, Some(b"x".to_vec())]
        );
    }

    #[test]
    fn a_binary_update_reads_the_base_column_for_the_docs_it_does_not_touch() {
        let base_values = vec![
            Some(b"aa".to_vec()),
            Some(b"bb".to_vec()),
            Some(b"cc".to_vec()),
        ];
        let (meta, data, _) = write_binary_generation(0, &base_values, &SEG_ID, "").unwrap();
        let fis = binary_field_infos();
        let (_, parsed) = doc_values::parse_meta(&meta, &SEG_ID, "", &fis).unwrap();
        let entry = parsed.binary_entry(0).unwrap();

        let column = merge_binary_column(
            Some((entry, &data)),
            &[(1, Some(b"updated".to_vec())), (2, None)],
            3,
        )
        .unwrap();
        assert_eq!(
            column,
            vec![Some(b"aa".to_vec()), Some(b"updated".to_vec()), None]
        );
    }

    #[test]
    fn a_generation_written_with_a_generation_suffix_only_opens_with_that_suffix() {
        // The suffix is in the file *name* and in the file's index header, and
        // the two must be the same string -- a mismatch is a file that exists
        // and fails `checkIndexHeader`.
        let (meta, _data, _) =
            write_numeric_generation(0, &[Some(1)], &SEG_ID, "1_Lucene90_0").unwrap();
        let fis = numeric_field_infos();
        assert!(doc_values::parse_meta(&meta, &SEG_ID, "1_Lucene90_0", &fis).is_ok());
        assert!(doc_values::parse_meta(&meta, &SEG_ID, "", &fis).is_err());
        assert!(doc_values::parse_meta(&meta, &SEG_ID, "2_Lucene90_0", &fis).is_err());
    }
}
