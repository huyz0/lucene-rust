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
//! `NumericDocValuesUpdates`/`BinaryDocValuesUpdates`
//! files [`write_numeric_updates`]/[`write_binary_updates`] produce are this
//! port's own invention and **nothing in an index references them any more**:
//! no `segments_N`, no `.si`, no `SegmentCommitInfo::files`. They survive as a
//! standalone serialization for `lucene_search::soft_deletes`'
//! `mark_soft_deleted_via_overlay`, which uses them to carry a soft-delete
//! marking outside an index. Do not reach for them when writing a segment:
//! a segment that references them is a segment real Lucene cannot open.
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

const NUMERIC_UPDATES_CODEC: &str = "NumericDocValuesUpdates";
/// The binary counterpart's codec name -- a distinct name so a binary overlay
/// file handed to [`read_numeric_updates`] (or the reverse) fails the header
/// check instead of decoding garbage.
const BINARY_UPDATES_CODEC: &str = "BinaryDocValuesUpdates";
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
    // Sized by the count only as far as the bytes left could back it (each
    // entry takes at least 5): a corrupt count fails on a short read, not in
    // the allocator.
    let mut map = HashMap::with_capacity((count.max(0) as usize).min(input.remaining() / 5));
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
    // Sized by the count only as far as the bytes left could back it (each
    // entry takes at least 5): a corrupt count fails on a short read, not in
    // the allocator.
    let mut map = HashMap::with_capacity((count.max(0) as usize).min(input.remaining() / 5));
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
                if len as usize > input.remaining() {
                    return Err(lucene_store::Error::Eof {
                        offset: input.position(),
                    }
                    .into());
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
                .map(|doc| reader.value_ref(doc).map(|v| v.map(<[u8]>::to_vec)))
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

/// [`write_numeric_generation`] for a field with a doc-values skip index:
/// the new generation carries the field's skip index too, as
/// `Lucene90DocValuesConsumer.addNumericField` writes one for every field
/// whose `FieldInfo` asks for it.
pub fn write_numeric_generation_with_skip_index(
    field_number: i32,
    column: &[Option<i64>],
    segment_id: &[u8; ID_LENGTH],
    segment_suffix: &str,
) -> doc_values::WriteResult<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    let sparse: Vec<(i32, i64)> = column
        .iter()
        .enumerate()
        .filter_map(|(doc, v)| v.map(|v| (doc as i32, v)))
        .collect();
    doc_values::write_fields_with_skip_indexes(
        &[doc_values::DenseField::SparseNumeric(field_number, &sparse)],
        &[field_number],
        column.len() as i32,
        segment_id,
        segment_suffix,
    )
}

/// [`write_numeric_generation`] for BINARY doc values --
/// `Lucene90DocValuesConsumer.addBinaryField` over the merged column. The
/// column is consumed: its values move into the writer's input rather than
/// being copied one allocation per document.
pub fn write_binary_generation(
    field_number: i32,
    column: Vec<Option<Vec<u8>>>,
    segment_id: &[u8; ID_LENGTH],
    segment_suffix: &str,
) -> doc_values::WriteResult<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    let max_doc = column.len() as i32;
    if column.iter().all(Option::is_some) {
        let dense: Vec<Vec<u8>> = column.into_iter().flatten().collect();
        doc_values::write_single_dense_binary_field(
            field_number,
            &dense,
            max_doc,
            segment_id,
            segment_suffix,
        )
    } else {
        let sparse: Vec<(i32, Vec<u8>)> = column
            .into_iter()
            .enumerate()
            .filter_map(|(doc, v)| v.map(|v| (doc as i32, v)))
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
mod tests;
