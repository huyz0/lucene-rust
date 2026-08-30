//! Port of `org.apache.lucene.codecs.lucene90.Lucene90NormsFormat` (`.nvm`
//! metadata + `.nvd` data) — read-only.
//!
//! Norms are a per-field, per-doc score-normalization value (one integer of
//! 0/1/2/4/8 bytes, depending on the range needed). Three shapes exist,
//! selected by `docs_with_field_offset`:
//! - **empty** (`-2`): no document has this field indexed at all.
//! - **dense** (`-1`): every doc up to `maxDoc` has a value — a flat array.
//! - **sparse** (`>= 0`): only some docs have a value, addressed through an
//!   `IndexedDISI` bitset (see [`crate::indexed_disi`]) giving each present
//!   doc's ordinal, which indexes the same flat value array dense fields use.
//!
//! Wire format, `.nvm` (little-endian throughout — no vints, unlike most
//! other formats; header/footer per `codec_util`):
//! ```text
//! IndexHeader(codec="Lucene90NormsMetadata", version=0, id, suffix)
//! per field (terminated by FieldNumber == -1):
//!   FieldNumber          --> i32
//!   DocsWithFieldOffset  --> i64  (-2 empty, -1 dense, >=0 sparse offset into .nvd)
//!   DocsWithFieldLength  --> i64  (sparse bitset length in .nvd, meaningless if not sparse)
//!   JumpTableEntryCount  --> i16
//!   DenseRankPower       --> u8
//!   NumDocsWithField     --> i32
//!   BytesPerNorm         --> u8  (must be one of 0, 1, 2, 4, 8)
//!   NormsOffset          --> i64  (offset into .nvd, or the single constant
//!                            value itself when BytesPerNorm == 0)
//! Footer
//! ```
//!
//! `.nvd` is just `IndexHeader, <raw bytes>, Footer`; dense values for a
//! field live at `NormsOffset + doc * BytesPerNorm`, little-endian,
//! sign-extended to i64 (matching `RandomAccessInput.readByte/Short/Int/Long`).

use lucene_store::codec_util::{self, ID_LENGTH};
use lucene_store::data_input::{DataInput, SliceInput};
use lucene_store::data_output::DataOutput;

use crate::field_infos::{FieldInfos, IndexOptions};
use crate::indexed_disi;

const DATA_CODEC: &str = "Lucene90NormsData";
const METADATA_CODEC: &str = "Lucene90NormsMetadata";
const VERSION_START: i32 = 0;
const VERSION_CURRENT: i32 = 0;

const DOCS_WITH_FIELD_EMPTY: i64 = -2;
const DOCS_WITH_FIELD_DENSE: i64 = -1;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Store(#[from] lucene_store::Error),
    #[error("invalid bytesPerValue: {0}, field number {1}")]
    InvalidBytesPerNorm(u8, i32),
    #[error("doc {0} is out of range (numDocsWithField={1})")]
    DocOutOfRange(i32, i32),
    #[error("invalid field number: {0} (.nvm names a field the .fnm does not have)")]
    UnknownFieldNumber(i32),
    #[error("invalid field: {0} (.nvm carries norms for a field whose .fnm says it has none)")]
    FieldHasNoNorms(String),
}

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, thiserror::Error)]
pub enum WriteError {
    #[error(
        "write_single_dense_field requires values.len() == max_doc (every doc must have a value); got {values} values for max_doc={max_doc}"
    )]
    NotDense { values: usize, max_doc: i32 },
    #[error("sparse norms write requires strictly ascending doc ids; doc {0} is out of order or duplicated")]
    DocIdsNotAscending(i32),
    #[error("sparse norms write requires doc {0} < max_doc={1}")]
    DocIdOutOfRange(i32, i32),
    #[error("write_fields requires at least one field")]
    EmptyFieldList,
    #[error("write_fields requires distinct field numbers; field {0} appears more than once")]
    DuplicateFieldNumber(i32),
}

pub type WriteResult<T> = std::result::Result<T, WriteError>;

#[derive(Debug, Clone, Copy)]
pub struct NormsEntry {
    pub field_number: i32,
    pub docs_with_field_offset: i64,
    pub docs_with_field_length: i64,
    pub jump_table_entry_count: i16,
    pub dense_rank_power: u8,
    pub num_docs_with_field: i32,
    pub bytes_per_norm: u8,
    pub norms_offset: i64,
}

impl NormsEntry {
    pub fn is_empty_field(&self) -> bool {
        self.docs_with_field_offset == DOCS_WITH_FIELD_EMPTY
    }

    pub fn is_dense(&self) -> bool {
        self.docs_with_field_offset == DOCS_WITH_FIELD_DENSE
    }
}

#[derive(Debug, Clone)]
pub struct Norms {
    pub entries: Vec<NormsEntry>,
}

impl Norms {
    pub fn entry(&self, field_number: i32) -> Option<&NormsEntry> {
        self.entries.iter().find(|e| e.field_number == field_number)
    }
}

/// Parses a whole `.nvm` metadata file already read into memory.
pub fn parse_meta(
    buf: &[u8],
    segment_id: &[u8; ID_LENGTH],
    segment_suffix: &str,
) -> Result<(i32, Norms)> {
    let mut input = SliceInput::new(buf);
    let header = codec_util::check_index_header(
        &mut input,
        METADATA_CODEC,
        VERSION_START,
        VERSION_CURRENT,
        segment_id,
        segment_suffix,
    )?;

    let mut entries = Vec::new();
    loop {
        let field_number = input.read_i32()?;
        if field_number == -1 {
            break;
        }
        let docs_with_field_offset = input.read_i64()?;
        let docs_with_field_length = input.read_i64()?;
        let jump_table_entry_count = input.read_i16()?;
        let dense_rank_power = input.read_byte()?;
        let num_docs_with_field = input.read_i32()?;
        let bytes_per_norm = input.read_byte()?;
        if !matches!(bytes_per_norm, 0 | 1 | 2 | 4 | 8) {
            return Err(Error::InvalidBytesPerNorm(bytes_per_norm, field_number));
        }
        let norms_offset = input.read_i64()?;

        entries.push(NormsEntry {
            field_number,
            docs_with_field_offset,
            docs_with_field_length,
            jump_table_entry_count,
            dense_rank_power,
            num_docs_with_field,
            bytes_per_norm,
            norms_offset,
        });
    }

    codec_util::check_footer(&mut input, buf.len())?;

    Ok((header.version, entries_to_norms(entries)))
}

fn entries_to_norms(entries: Vec<NormsEntry>) -> Norms {
    Norms { entries }
}

/// The half of `Lucene90NormsProducer.readFields` that needs the segment's
/// `FieldInfos`: every `.nvm` entry must name a field that exists **and**
/// that actually has norms.
///
/// Java does this inside `readFields`, which takes `FieldInfos`; this port's
/// [`parse_meta`] does not, so it is a separate call. That split is
/// deliberate, not an oversight: `parse_meta` has 23 call sites across four
/// crates, several of them in files other batches own, and threading a
/// `&FieldInfos` through all of them buys nothing at the ones that are
/// hand-built round-trip tests. The two call sites where the diagnostic
/// actually matters -- `check_index`'s `norms.*` checks and the segment
/// reader's open -- call this. See `docs/sweep/m2/c15-postings-api.md` for
/// the decision (b6 #4, c7 F-23).
///
/// What it catches: a `.nvm` naming a field number the `.fnm` does not have
/// (the entry is then unreachable, so every norm lookup for the *real* field
/// silently returns "no norms" and every score is computed with a default
/// norm), and a `.nvm` carrying norms for a field whose `FieldInfo` says it
/// has none (`indexOptions == NONE` or `omitNorms`), which means the two
/// files disagree about what the segment contains.
pub fn validate_fields(norms: &Norms, field_infos: &FieldInfos) -> Result<()> {
    for entry in &norms.entries {
        let Some(info) = field_infos
            .fields
            .iter()
            .find(|f| f.number == entry.field_number)
        else {
            return Err(Error::UnknownFieldNumber(entry.field_number));
        };
        // `FieldInfo.hasNorms()`.
        if info.index_options == IndexOptions::None || info.omit_norms {
            return Err(Error::FieldHasNoNorms(info.name.clone()));
        }
    }
    Ok(())
}

/// Validates a whole `.nvd` data file's header/footer (does not decode the
/// per-field regions, which are addressed by absolute offset from `.nvm`
/// entries and have no self-describing structure of their own beyond that).
/// Returns the format version for cross-checking against the meta file's.
pub fn check_data_header_footer(
    buf: &[u8],
    segment_id: &[u8; ID_LENGTH],
    segment_suffix: &str,
) -> Result<i32> {
    let mut input = SliceInput::new(buf);
    let header = codec_util::check_index_header(
        &mut input,
        DATA_CODEC,
        VERSION_START,
        VERSION_CURRENT,
        segment_id,
        segment_suffix,
    )?;
    // Norms data files are only checksum-validated structurally on open in
    // Lucene (full-file CRC is too costly for a forward-only read pattern);
    // mirror that by only requiring the footer to *exist* and be
    // well-formed, not that we've read every byte up to it.
    codec_util::retrieve_checksum(buf)?;
    Ok(header.version)
}

/// Reads the norm value for `doc`, handling all three shapes (empty, dense,
/// sparse). `data` is the whole `.nvd` file's bytes. Returns `Ok(None)` when
/// `doc` legitimately has no norm (an empty field, or a doc a sparse field
/// skips) — that is normal, not an error; only a truly out-of-range `doc` or
/// a decode failure is `Err`.
pub fn norm_value(data: &[u8], entry: &NormsEntry, doc: i32) -> Result<Option<i64>> {
    if doc < 0 {
        return Err(Error::DocOutOfRange(doc, entry.num_docs_with_field));
    }
    if entry.is_empty_field() {
        return Ok(None);
    }
    if entry.is_dense() {
        if doc >= entry.num_docs_with_field {
            return Err(Error::DocOutOfRange(doc, entry.num_docs_with_field));
        }
        return Ok(Some(read_value_at_ordinal(data, entry, doc as i64)?));
    }

    // Sparse: docs_with_field_offset/length address an IndexedDISI region.
    let region = sparse_region(data, entry)?;
    // One forward-only pass over the block headers rather than decoding the
    // whole region: `DisiCursor` reads one 4-byte header per 65,536 documents
    // and then resolves the ordinal inside a single block (a DENSE block via
    // its rank table, when the metadata says one is there). It allocates
    // nothing, where `decode_doc_ids` built a `Vec<i32>` of every present doc.
    //
    // A caller doing more than one lookup on the same sparse field should hold
    // its own `DisiCursor` and walk it forward, which is what this function
    // cannot do across calls -- see `doc_values::NumericReader` for the shape.
    match indexed_disi::DisiCursor::new(region, entry.dense_rank_power).advance_exact(doc)? {
        Some(ordinal) => Ok(Some(read_value_at_ordinal(data, entry, ordinal as i64)?)),
        None => Ok(None),
    }
}

/// The `IndexedDISI` region a sparse entry's `docsWithFieldOffset`/`Length`
/// address inside `.nvd`. Public because `check_index` needs the same
/// range with the same bounds, and two copies of this rule is one too many.
///
/// Both halves are `i64`s read straight off `.nvm`, so their **sum** is as
/// untrusted as either one: `offset + length` on two values a corrupt file
/// chose overflows before `data.get` ever sees a range, and an overflow is a
/// panic rather than the decode error the caller is prepared for. Java never
/// has this exposure because `IndexInput.slice(offset, length)` takes the two
/// separately and range-checks each against the file.
pub fn sparse_region<'d>(data: &'d [u8], entry: &NormsEntry) -> Result<&'d [u8]> {
    let start = usize::try_from(entry.docs_with_field_offset)
        .map_err(|_| lucene_store::Error::Eof { offset: 0 })?;
    let end = entry
        .docs_with_field_offset
        .checked_add(entry.docs_with_field_length)
        .and_then(|e| usize::try_from(e).ok())
        .ok_or(lucene_store::Error::Eof { offset: 0 })?;
    data.get(start..end)
        .ok_or(lucene_store::Error::Eof { offset: 0 }.into())
}

/// Reads the norm value at `ordinal` (either the doc id itself for a dense
/// field, or the doc's rank among docs-with-a-value for a sparse one).
///
/// Public so a caller holding a sparse field's doc-id list -- decoded once
/// rather than per lookup, see [`norm_value`]'s own note -- can finish the
/// lookup itself — both
/// index the same flat `NormsOffset + ordinal * BytesPerNorm` array shape.
pub fn read_value_at_ordinal(data: &[u8], entry: &NormsEntry, ordinal: i64) -> Result<i64> {
    if entry.bytes_per_norm == 0 {
        // A single constant value for every doc, encoded directly in the
        // offset field rather than a separate array.
        return Ok(entry.norms_offset);
    }

    // `norms_offset` is an unconstrained `i64` read straight off `.nvm` and
    // `ordinal` is a doc id or a DISI rank, so neither the product nor the
    // sum is bounded by anything this port controls: a corrupt `.nvm` makes
    // `norms_offset + ordinal * bytesPerNorm` overflow, which is a panic in a
    // debug build and a wrap to a plausible-looking in-range offset in a
    // release one. Fold the bound into the arithmetic and report it as the
    // corruption it is. `offset as usize` afterwards is safe because the
    // `try_from` rejects a negative offset, which would otherwise become a
    // huge `usize` and merely look like an EOF.
    let offset = ordinal
        .checked_mul(entry.bytes_per_norm as i64)
        .and_then(|scaled| entry.norms_offset.checked_add(scaled))
        .and_then(|off| usize::try_from(off).ok())
        .ok_or_else(|| {
            lucene_store::Error::Corrupted(format!(
                "norms offset out of range: normsOffset={} ordinal={ordinal} bytesPerNorm={}",
                entry.norms_offset, entry.bytes_per_norm
            ))
        })?;
    let mut input = SliceInput::new(data);
    input.seek(offset)?;
    let value = match entry.bytes_per_norm {
        1 => input.read_byte()? as i8 as i64,
        2 => input.read_i16()? as i64,
        4 => input.read_i32()? as i64,
        8 => input.read_i64()?,
        // Already validated in `parse_meta`.
        _ => unreachable!("bytesPerNorm validated to be one of 0,1,2,4,8"),
    };
    Ok(value)
}

/// One norms field's per-doc values, as accepted by [`write_fields`].
///
/// `Dense` is `addNormsField`'s `numDocsWithValue == maxDoc` branch (every doc
/// `0..max_doc` has a norm, no `IndexedDISI` structure); `Sparse` is its
/// `else` branch (only the listed docs have one, addressed through an
/// [`indexed_disi`] bitset). A field with no values at all is `Sparse` with an
/// empty list, which writes `addNormsField`'s `numDocsWithValue == 0` branch.
#[derive(Debug, Clone, Copy)]
pub enum NormsField<'a> {
    Dense(i32, &'a [i64]),
    /// `(field number, (doc id, norm) pairs)`. Doc ids need not be sorted;
    /// [`write_fields`] sorts a clone. Each must be unique and `< max_doc`.
    Sparse(i32, &'a [(i32, i64)]),
}

impl NormsField<'_> {
    fn field_number(&self) -> i32 {
        match self {
            NormsField::Dense(n, _) | NormsField::Sparse(n, _) => *n,
        }
    }
}

/// Port of `Lucene90NormsConsumer.numBytesPerValue`: the narrowest of 0/1/2/4/8
/// bytes that can hold every value in `[min, max]`. `0` means "every doc has
/// the same value", which is stored once in the meta entry's `normsOffset`
/// slot instead of a per-doc array.
fn num_bytes_per_value(min: i64, max: i64) -> u8 {
    if min >= max {
        0
    } else if min >= i8::MIN as i64 && max <= i8::MAX as i64 {
        1
    } else if min >= i16::MIN as i64 && max <= i16::MAX as i64 {
        2
    } else if min >= i32::MIN as i64 && max <= i32::MAX as i64 {
        4
    } else {
        8
    }
}

/// Writes a whole `.nvm`/`.nvd` pair holding **one or more** norms fields --
/// the multi-field analogue of [`write_single_dense_field`] and
/// [`write_single_sparse_field`], both of which are thin one-element-slice
/// wrappers over it (same precedent as
/// [`crate::doc_values::write_dense_fields`]). Real Lucene's
/// `Lucene90NormsConsumer` always puts every norms field of a segment into the
/// same pair, one `addNormsField` call each, which is exactly what this does.
///
/// Every field's per-value width is chosen independently by
/// [`num_bytes_per_value`], covering all five of Java's cases (constant, 1, 2,
/// 4, 8 bytes per doc).
///
/// Returns `(meta_bytes, data_bytes)` matching the real writer's two
/// `IndexOutput`s; unlike doc values, norms have no third (`.dvs`-style) file.
pub fn write_fields(
    fields: &[NormsField<'_>],
    max_doc: i32,
    segment_id: &[u8; ID_LENGTH],
    segment_suffix: &str,
) -> WriteResult<(Vec<u8>, Vec<u8>)> {
    if fields.is_empty() {
        return Err(WriteError::EmptyFieldList);
    }
    // ARITH: `i` and `j` index `fields`, whose length is bounded by
    // `isize::MAX`, so `i + 1` cannot overflow.
    #[allow(clippy::arithmetic_side_effects)]
    for i in 0..fields.len() {
        for j in (i + 1)..fields.len() {
            if fields[i].field_number() == fields[j].field_number() {
                return Err(WriteError::DuplicateFieldNumber(fields[i].field_number()));
            }
        }
    }

    // Normalise every field to `(doc ids or None for dense, values)` and
    // validate before touching either buffer, same "fail before writing"
    // order `doc_values`'s writers use.
    let mut prepared: Vec<(i32, Option<Vec<i32>>, Vec<i64>)> = Vec::with_capacity(fields.len());
    for field in fields {
        match field {
            NormsField::Dense(number, values) => {
                if values.len() != max_doc as usize {
                    return Err(WriteError::NotDense {
                        values: values.len(),
                        max_doc,
                    });
                }
                prepared.push((*number, None, values.to_vec()));
            }
            NormsField::Sparse(number, doc_values) => {
                let mut sorted: Vec<(i32, i64)> = doc_values.to_vec();
                sorted.sort_unstable_by_key(|&(doc, _)| doc);
                // ARITH: the range starts at 1, so `i - 1` is in bounds.
                #[allow(clippy::arithmetic_side_effects)]
                for i in 1..sorted.len() {
                    if sorted[i - 1].0 == sorted[i].0 {
                        return Err(WriteError::DocIdsNotAscending(sorted[i].0));
                    }
                }
                for &(doc, _) in &sorted {
                    if doc < 0 || doc >= max_doc {
                        return Err(WriteError::DocIdOutOfRange(doc, max_doc));
                    }
                }
                // A sparse list that happens to cover every doc *is* the dense
                // case on disk: `addNormsField` branches on
                // `numDocsWithValue == maxDoc`, not on how the caller phrased
                // it, and the reader's `is_dense()` depends on that marker.
                let dense = sorted.len() == max_doc as usize;
                let docs = if dense {
                    None
                } else {
                    Some(sorted.iter().map(|&(doc, _)| doc).collect())
                };
                prepared.push((*number, docs, sorted.iter().map(|&(_, v)| v).collect()));
            }
        }
    }

    let mut meta: Vec<u8> = Vec::new();
    codec_util::write_index_header(
        &mut meta,
        METADATA_CODEC,
        VERSION_CURRENT,
        segment_id,
        segment_suffix,
    );

    let mut data: Vec<u8> = Vec::new();
    codec_util::write_index_header(
        &mut data,
        DATA_CODEC,
        VERSION_CURRENT,
        segment_id,
        segment_suffix,
    );

    for (field_number, docs, values) in &prepared {
        meta.write_i32(*field_number);

        match docs {
            // `numDocsWithValue == 0`: meta[-2, 0]. `numDocsWithValue ==
            // maxDoc`: meta[-1, 0]. Neither writes an IndexedDISI structure.
            None if values.is_empty() => {
                meta.write_i64(DOCS_WITH_FIELD_EMPTY);
                meta.write_i64(0);
                meta.write_i16(-1); // jumpTableEntryCount
                meta.push(0xFF); // denseRankPower (-1 as u8)
            }
            None => {
                meta.write_i64(DOCS_WITH_FIELD_DENSE);
                meta.write_i64(0);
                meta.write_i16(-1);
                meta.push(0xFF);
            }
            Some(doc_ids) if doc_ids.is_empty() => {
                meta.write_i64(DOCS_WITH_FIELD_EMPTY);
                meta.write_i64(0);
                meta.write_i16(-1);
                meta.push(0xFF);
            }
            Some(doc_ids) => {
                // `IndexedDISI.writeBitSet(it, data)`, which is what
                // `Lucene90NormsConsumer.addNormsField` calls, defaults to
                // `DEFAULT_DENSE_RANK_POWER`: 256 bytes per DENSE block for a
                // ~26x faster cold lookup inside one.
                let disi_bytes = indexed_disi::write_with_dense_rank_power(
                    doc_ids,
                    indexed_disi::DEFAULT_DENSE_RANK_POWER,
                );
                let offset = data.len() as i64;
                data.extend_from_slice(&disi_bytes);
                meta.write_i64(offset);
                // ARITH: `offset` was `data.len()` immediately before the
                // `extend_from_slice` above and a `Vec` only grows there.
                #[allow(clippy::arithmetic_side_effects)]
                let disi_len = data.len() as i64 - offset;
                meta.write_i64(disi_len);
                meta.write_i16(-1); // jumpTableEntryCount: no jump table written
                meta.push(indexed_disi::DEFAULT_DENSE_RANK_POWER); // denseRankPower
            }
        }

        meta.write_i32(values.len() as i32); // numDocsWithValue

        let min = values.iter().copied().min().unwrap_or(i64::MAX);
        let max = values.iter().copied().max().unwrap_or(i64::MIN);
        let bytes_per_value = num_bytes_per_value(min, max);
        meta.push(bytes_per_value);
        if bytes_per_value == 0 {
            // Java writes `min`, which for an empty field is
            // `Long.MAX_VALUE`; nothing ever reads it, since an empty field
            // never resolves a doc to an ordinal.
            meta.write_i64(min);
        } else {
            meta.write_i64(data.len() as i64); // normsOffset
            for &v in values {
                match bytes_per_value {
                    1 => data.push(v as i8 as u8),
                    2 => data.write_i16(v as i16),
                    4 => data.write_i32(v as i32),
                    _ => data.write_i64(v),
                }
            }
        }
    }

    meta.write_i32(-1); // field list terminator
    codec_util::write_footer(&mut meta);
    codec_util::write_footer(&mut data);

    Ok((meta, data))
}

/// Port of `Lucene90NormsConsumer.addNormsField` for **a single norms field,
/// DENSE** (every doc from `0` to `max_doc - 1` has a value) -- the
/// `numDocsWithValue == maxDoc` branch. A thin wrapper over
/// [`write_fields`]; see [`write_single_sparse_field`] for the sparse
/// (`IndexedDISI`) shape and [`write_fields`] for more than one field in one
/// `.nvm`/`.nvd` pair.
///
/// All five of Java's per-value widths are supported (constant, 1, 2, 4 and 8
/// bytes per doc); the narrowest one that fits the value range is chosen, by
/// [`num_bytes_per_value`].
///
/// Returns `(meta_bytes, data_bytes)` matching the real writer's two
/// `IndexOutput`s; unlike doc values, norms have no third (`.dvs`-style) file.
pub fn write_single_dense_field(
    field_number: i32,
    values: &[i64],
    max_doc: i32,
    segment_id: &[u8; ID_LENGTH],
    segment_suffix: &str,
) -> WriteResult<(Vec<u8>, Vec<u8>)> {
    write_fields(
        &[NormsField::Dense(field_number, values)],
        max_doc,
        segment_id,
        segment_suffix,
    )
}

/// Port of `Lucene90NormsConsumer.addNormsField`'s **sparse** branch
/// (`numDocsWithValue != maxDoc`): only the docs in `doc_values` have a norm,
/// recorded as an [`indexed_disi`] bitset in `.nvd`, and the per-doc value
/// array is indexed by *rank among present docs* rather than by doc id --
/// exactly what [`norm_value`]'s sparse branch reads back.
///
/// `doc_values` need not be sorted; this function sorts a clone. Each doc id
/// must be unique and `< max_doc`. Two shapes degenerate on purpose, matching
/// `addNormsField`'s own branching rather than the caller's phrasing: an empty
/// list writes the "no document has this field" marker (`-2`), and a list that
/// covers all `max_doc` docs writes the dense marker (`-1`).
pub fn write_single_sparse_field(
    field_number: i32,
    doc_values: &[(i32, i64)],
    max_doc: i32,
    segment_id: &[u8; ID_LENGTH],
    segment_suffix: &str,
) -> WriteResult<(Vec<u8>, Vec<u8>)> {
    write_fields(
        &[NormsField::Sparse(field_number, doc_values)],
        max_doc,
        segment_id,
        segment_suffix,
    )
}

#[cfg(test)]
mod tests {
    // The arithmetic gate is about values read off disk; a test's `i + 1` is
    // not one. See docs/arithmetic-gate.md.
    #![allow(clippy::arithmetic_side_effects)]

    use super::*;

    /// `Lucene90NormsProducer.readFields`' `FieldInfos` half, which
    /// [`validate_fields`] carries in this port.
    mod validate_fields_tests {
        use super::*;
        use crate::field_infos::{
            DocValuesSkipIndexType, DocValuesType, FieldInfo, VectorEncoding,
            VectorSimilarityFunction,
        };

        fn field(
            number: i32,
            name: &str,
            index_options: IndexOptions,
            omit_norms: bool,
        ) -> FieldInfo {
            FieldInfo {
                name: name.to_string(),
                number,
                store_term_vectors: false,
                omit_norms,
                store_payloads: false,
                soft_deletes_field: false,
                parent_field: false,
                index_options,
                doc_values_type: DocValuesType::None,
                doc_values_skip_index_type: DocValuesSkipIndexType::None,
                doc_values_gen: -1,
                attributes: Vec::new(),
                point_dimension_count: 0,
                point_index_dimension_count: 0,
                point_num_bytes: 0,
                vector_dimension: 0,
                vector_encoding: VectorEncoding::Float32,
                vector_similarity_function: VectorSimilarityFunction::Euclidean,
            }
        }

        fn norms_for(field_numbers: &[i32]) -> Norms {
            Norms {
                entries: field_numbers
                    .iter()
                    .map(|&field_number| NormsEntry {
                        field_number,
                        docs_with_field_offset: DOCS_WITH_FIELD_DENSE,
                        docs_with_field_length: 0,
                        jump_table_entry_count: -1,
                        dense_rank_power: 0xFF,
                        num_docs_with_field: 1,
                        bytes_per_norm: 1,
                        norms_offset: 0,
                    })
                    .collect(),
            }
        }

        #[test]
        fn entries_naming_real_norms_fields_are_accepted() {
            let infos = FieldInfos {
                fields: vec![
                    field(0, "body", IndexOptions::DocsAndFreqs, false),
                    field(3, "title", IndexOptions::Docs, false),
                ],
            };
            assert!(validate_fields(&norms_for(&[0, 3]), &infos).is_ok());
        }

        #[test]
        fn an_entry_for_a_field_the_fnm_does_not_have_is_rejected() {
            let infos = FieldInfos {
                fields: vec![field(0, "body", IndexOptions::DocsAndFreqs, false)],
            };
            assert!(matches!(
                validate_fields(&norms_for(&[0, 7]), &infos),
                Err(Error::UnknownFieldNumber(7))
            ));
        }

        #[test]
        fn an_entry_for_a_field_with_no_norms_is_rejected() {
            // `FieldInfo.hasNorms()` is `indexOptions != NONE && !omitNorms`;
            // both halves must reject.
            let omitted = FieldInfos {
                fields: vec![field(0, "body", IndexOptions::DocsAndFreqs, true)],
            };
            assert!(matches!(
                validate_fields(&norms_for(&[0]), &omitted),
                Err(Error::FieldHasNoNorms(name)) if name == "body"
            ));

            let unindexed = FieldInfos {
                fields: vec![field(0, "body", IndexOptions::None, false)],
            };
            assert!(matches!(
                validate_fields(&norms_for(&[0]), &unindexed),
                Err(Error::FieldHasNoNorms(name)) if name == "body"
            ));
        }
    }

    /// Test-only `.nvm`/`.nvd` byte builder, independent of the Java fixture
    /// under `tests/norms_fixtures.rs` (which exercises a real IndexWriter's
    /// output): this covers error/edge paths — invalid bytesPerNorm, empty/
    /// sparse fields, out-of-range docs, and each of the four nonzero byte
    /// widths — that a single realistic fixture doesn't naturally hit all of.
    struct EntryBuilder {
        field_number: i32,
        docs_with_field_offset: i64,
        docs_with_field_length: i64,
        jump_table_entry_count: i16,
        dense_rank_power: u8,
        num_docs_with_field: i32,
        bytes_per_norm: u8,
        norms_offset: i64,
    }

    impl EntryBuilder {
        fn dense(field_number: i32, bytes_per_norm: u8, num_docs: i32, norms_offset: i64) -> Self {
            Self {
                field_number,
                docs_with_field_offset: DOCS_WITH_FIELD_DENSE,
                docs_with_field_length: 0,
                jump_table_entry_count: 0,
                dense_rank_power: indexed_disi::NO_RANK,
                num_docs_with_field: num_docs,
                bytes_per_norm,
                norms_offset,
            }
        }

        fn build(&self, out: &mut Vec<u8>) {
            out.extend_from_slice(&self.field_number.to_le_bytes());
            out.extend_from_slice(&self.docs_with_field_offset.to_le_bytes());
            out.extend_from_slice(&self.docs_with_field_length.to_le_bytes());
            out.extend_from_slice(&self.jump_table_entry_count.to_le_bytes());
            out.push(self.dense_rank_power);
            out.extend_from_slice(&self.num_docs_with_field.to_le_bytes());
            out.push(self.bytes_per_norm);
            out.extend_from_slice(&self.norms_offset.to_le_bytes());
        }
    }

    fn build_nvm(id: &[u8; ID_LENGTH], entries: &[EntryBuilder]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&codec_util::CODEC_MAGIC.to_be_bytes());
        write_string(&mut out, METADATA_CODEC);
        out.extend_from_slice(&(VERSION_CURRENT as u32).to_be_bytes());
        out.extend_from_slice(id);
        out.push(0); // empty suffix
        for e in entries {
            e.build(&mut out);
        }
        out.extend_from_slice(&(-1i32).to_le_bytes()); // terminator
        out.extend_from_slice(&codec_util::FOOTER_MAGIC.to_be_bytes());
        out.extend_from_slice(&0u32.to_be_bytes());
        let checksum = crc32fast::hash(&out) as u64;
        out.extend_from_slice(&checksum.to_be_bytes());
        out
    }

    fn build_nvd(id: &[u8; ID_LENGTH], payload: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&codec_util::CODEC_MAGIC.to_be_bytes());
        write_string(&mut out, DATA_CODEC);
        out.extend_from_slice(&(VERSION_CURRENT as u32).to_be_bytes());
        out.extend_from_slice(id);
        out.push(0);
        out.extend_from_slice(payload);
        out.extend_from_slice(&codec_util::FOOTER_MAGIC.to_be_bytes());
        out.extend_from_slice(&0u32.to_be_bytes());
        let checksum = crc32fast::hash(&out) as u64;
        out.extend_from_slice(&checksum.to_be_bytes());
        out
    }

    fn write_vint(out: &mut Vec<u8>, mut v: i32) {
        loop {
            let mut b = (v & 0x7f) as u8;
            v = ((v as u32) >> 7) as i32;
            if v != 0 {
                b |= 0x80;
                out.push(b);
            } else {
                out.push(b);
                break;
            }
        }
    }

    fn write_string(out: &mut Vec<u8>, s: &str) {
        write_vint(out, s.len() as i32);
        out.extend_from_slice(s.as_bytes());
    }

    /// Length of the index header a `.nvm`/`.nvd` file starts with, for a
    /// given codec name and this module's fixed version/id/empty-suffix
    /// shape — used to compute absolute offsets for hand-built `.nvd` bytes.
    fn nvm_header_len(codec: &str) -> usize {
        4 + 1 + codec.len() + 4 + ID_LENGTH + 1 // magic + vint-len + name + version + id + suffix-len
    }

    #[test]
    fn empty_meta_parses_no_fields() {
        let id = [1u8; ID_LENGTH];
        let buf = build_nvm(&id, &[]);
        let (version, norms) = parse_meta(&buf, &id, "").unwrap();
        assert_eq!(version, 0);
        assert_eq!(norms.entries.len(), 0);
    }

    #[test]
    fn invalid_bytes_per_norm_rejected() {
        let id = [1u8; ID_LENGTH];
        let mut e = EntryBuilder::dense(0, 3, 5, 0); // 3 is not a valid width
        e.bytes_per_norm = 3;
        let buf = build_nvm(&id, &[e]);
        assert!(matches!(
            parse_meta(&buf, &id, ""),
            Err(Error::InvalidBytesPerNorm(3, 0))
        ));
    }

    #[test]
    fn empty_field_has_no_value_anywhere() {
        let id = [1u8; ID_LENGTH];
        let mut e = EntryBuilder::dense(0, 1, 0, 0);
        e.docs_with_field_offset = DOCS_WITH_FIELD_EMPTY;
        let buf = build_nvm(&id, &[e]);
        let (_, norms) = parse_meta(&buf, &id, "").unwrap();
        let entry = norms.entry(0).unwrap();
        assert!(entry.is_empty_field());
        assert_eq!(norm_value(&[], entry, 0).unwrap(), None);
    }

    /// A sparse `.nvm` entry whose `docsWithFieldOffset + docsWithFieldLength`
    /// overflows `i64`.
    ///
    /// Both halves are read straight off `.nvm` with no relationship between
    /// them on the wire, so a corrupt file can pick any pair. The sum used to
    /// be a plain `+` feeding a slice range: in a debug build that is
    /// `attempt to add with overflow`, i.e. a panic where the caller (and,
    /// through the FFI, the JVM) expected a decode error. Java is not exposed
    /// to it because `IndexInput.slice(offset, length)` takes the two
    /// separately and range-checks each.
    #[test]
    fn a_sparse_entry_whose_offset_plus_length_overflows_is_rejected() {
        let mut e = EntryBuilder::dense(0, 1, 3, 0);
        e.docs_with_field_offset = i64::MAX;
        e.docs_with_field_length = 1;
        let entry = to_entry(&e);
        assert!(!entry.is_dense() && !entry.is_empty_field());
        let got = norm_value(&[0, 0, 0], &entry, 0);
        assert!(got.is_err(), "{got:?}");

        // A negative offset that is neither of the two sentinels
        // (`DOCS_WITH_FIELD_DENSE` = -1, `DOCS_WITH_FIELD_EMPTY` = -2) takes
        // the same route rather than sign-extending into a ~2^64 slice start.
        let mut e = EntryBuilder::dense(0, 1, 3, 0);
        e.docs_with_field_offset = -3;
        e.docs_with_field_length = 4;
        let entry = to_entry(&e);
        let got = norm_value(&[0, 0, 0], &entry, 0);
        assert!(got.is_err(), "{got:?}");
    }

    #[test]
    fn doc_out_of_range_rejected() {
        let id = [1u8; ID_LENGTH];
        let e = EntryBuilder::dense(0, 1, 3, 0);
        let buf = build_nvm(&id, &[e]);
        let (_, norms) = parse_meta(&buf, &id, "").unwrap();
        let entry = norms.entry(0).unwrap();
        assert!(matches!(
            norm_value(&[0, 0, 0], entry, 3),
            Err(Error::DocOutOfRange(3, 3))
        ));
        assert!(matches!(
            norm_value(&[0, 0, 0], entry, -1),
            Err(Error::DocOutOfRange(-1, 3))
        ));
    }

    /// A `.nvm` entry's `normsOffset` is an unconstrained `i64` off disk.
    /// `normsOffset + ordinal * bytesPerNorm` overflowed before it was
    /// bounded: a panic in a debug build (dead JVM through the FFI) and, in a
    /// release build, a wrap to a small in-range offset that reads a
    /// *plausible* norm out of the wrong place in the `.nvd`.
    #[test]
    fn corrupt_norms_offset_is_a_decode_error_not_an_overflow() {
        let id = [1u8; ID_LENGTH];
        let e = EntryBuilder::dense(0, 1, 5, i64::MAX);
        let buf = build_nvm(&id, &[e]);
        let (_, norms) = parse_meta(&buf, &id, "").unwrap();
        let entry = norms.entry(0).unwrap();
        // doc 0 needs no scaling and merely lands past EOF; doc 1 onwards
        // overflows the `i64` addition itself.
        for doc in 1..5 {
            assert!(
                matches!(
                    norm_value(&[0u8; 8], entry, doc),
                    Err(Error::Store(lucene_store::Error::Corrupted(_)))
                ),
                "doc {doc}"
            );
        }
    }

    /// The same shape from the other side: a negative `normsOffset` used to
    /// become a huge `usize` through an `as` cast and merely look like EOF.
    #[test]
    fn negative_norms_offset_is_a_decode_error() {
        let id = [1u8; ID_LENGTH];
        let e = EntryBuilder::dense(0, 2, 5, -8);
        let buf = build_nvm(&id, &[e]);
        let (_, norms) = parse_meta(&buf, &id, "").unwrap();
        let entry = norms.entry(0).unwrap();
        assert!(matches!(
            norm_value(&[0u8; 16], entry, 0),
            Err(Error::Store(lucene_store::Error::Corrupted(_)))
        ));
    }

    #[test]
    fn constant_value_when_bytes_per_norm_zero() {
        let id = [1u8; ID_LENGTH];
        let e = EntryBuilder::dense(0, 0, 5, 7); // constant value 7 for all docs
        let buf = build_nvm(&id, &[e]);
        let (_, norms) = parse_meta(&buf, &id, "").unwrap();
        let entry = norms.entry(0).unwrap();
        for doc in 0..5 {
            assert_eq!(norm_value(&[], entry, doc).unwrap(), Some(7));
        }
    }

    #[test]
    fn every_nonzero_byte_width_decodes_correctly() {
        let id = [1u8; ID_LENGTH];

        // width 1: value -5 at doc 0
        let payload1 = vec![(-5i8) as u8];
        let data = build_nvd(&id, &payload1);
        let header_len = nvm_header_len(DATA_CODEC);
        let e = EntryBuilder::dense(0, 1, 1, header_len as i64);
        assert_eq!(
            norm_value(&data, &to_entry(&e), 0).unwrap(),
            Some(-5),
            "width 1"
        );

        // width 2: value -300
        let mut payload2 = Vec::new();
        payload2.extend_from_slice(&(-300i16).to_le_bytes());
        let data = build_nvd(&id, &payload2);
        let e = EntryBuilder::dense(0, 2, 1, header_len as i64);
        assert_eq!(
            norm_value(&data, &to_entry(&e), 0).unwrap(),
            Some(-300),
            "width 2"
        );

        // width 4: value -70000
        let mut payload4 = Vec::new();
        payload4.extend_from_slice(&(-70000i32).to_le_bytes());
        let data = build_nvd(&id, &payload4);
        let e = EntryBuilder::dense(0, 4, 1, header_len as i64);
        assert_eq!(
            norm_value(&data, &to_entry(&e), 0).unwrap(),
            Some(-70000),
            "width 4"
        );

        // width 8: value i64::MIN
        let mut payload8 = Vec::new();
        payload8.extend_from_slice(&i64::MIN.to_le_bytes());
        let data = build_nvd(&id, &payload8);
        let e = EntryBuilder::dense(0, 8, 1, header_len as i64);
        assert_eq!(
            norm_value(&data, &to_entry(&e), 0).unwrap(),
            Some(i64::MIN),
            "width 8"
        );
    }

    #[test]
    fn sparse_field_returns_value_for_present_doc_and_none_for_absent() {
        // IndexedDISI region: a single SPARSE block covering docs [0,65536)
        // with docs 1 and 3 present, then the mandatory sentinel block.
        let mut disi_bytes = Vec::new();
        disi_bytes.extend_from_slice(&0u16.to_le_bytes()); // block 0
        disi_bytes.extend_from_slice(&1u16.to_le_bytes()); // numValues-1 = 1 (2 values)
        disi_bytes.extend_from_slice(&1u16.to_le_bytes()); // doc 1
        disi_bytes.extend_from_slice(&3u16.to_le_bytes()); // doc 3
        disi_bytes.extend_from_slice(&((i32::MAX >> 16) as u16).to_le_bytes());
        disi_bytes.extend_from_slice(&0u16.to_le_bytes()); // numValues-1 = 0 (1 value)
        disi_bytes.extend_from_slice(&((i32::MAX & 0xFFFF) as u16).to_le_bytes());

        // .nvd layout: [ disi_bytes ][ values: byte per present doc, in doc order ]
        let disi_offset = 0i64;
        let disi_length = disi_bytes.len() as i64;
        let mut data = disi_bytes.clone();
        data.push(11); // value for doc 1 (ordinal 0)
        data.push(33); // value for doc 3 (ordinal 1)

        let mut e = EntryBuilder::dense(0, 1, 2, disi_length); // norms right after the DISI region
        e.docs_with_field_offset = disi_offset;
        e.docs_with_field_length = disi_length;
        let entry = to_entry(&e);

        assert_eq!(norm_value(&data, &entry, 1).unwrap(), Some(11));
        assert_eq!(norm_value(&data, &entry, 3).unwrap(), Some(33));
        assert_eq!(norm_value(&data, &entry, 2).unwrap(), None);
    }

    fn to_entry(e: &EntryBuilder) -> NormsEntry {
        NormsEntry {
            field_number: e.field_number,
            docs_with_field_offset: e.docs_with_field_offset,
            docs_with_field_length: e.docs_with_field_length,
            jump_table_entry_count: e.jump_table_entry_count,
            dense_rank_power: e.dense_rank_power,
            num_docs_with_field: e.num_docs_with_field,
            bytes_per_norm: e.bytes_per_norm,
            norms_offset: e.norms_offset,
        }
    }

    #[test]
    fn check_data_header_footer_valid() {
        let id = [2u8; ID_LENGTH];
        let data = build_nvd(&id, b"payload-bytes");
        let version = check_data_header_footer(&data, &id, "").unwrap();
        assert_eq!(version, 0);
    }

    #[test]
    fn check_data_header_footer_wrong_id_rejected() {
        let id = [2u8; ID_LENGTH];
        let data = build_nvd(&id, b"payload-bytes");
        let wrong_id = [3u8; ID_LENGTH];
        assert!(check_data_header_footer(&data, &wrong_id, "").is_err());
    }

    #[test]
    fn wrong_id_rejected_on_meta() {
        let id = [1u8; ID_LENGTH];
        let buf = build_nvm(&id, &[]);
        let wrong_id = [9u8; ID_LENGTH];
        assert!(matches!(
            parse_meta(&buf, &wrong_id, ""),
            Err(Error::Store(_))
        ));
    }

    #[test]
    fn write_single_dense_field_round_trips_through_own_reader() {
        let id = [7u8; ID_LENGTH];
        let values = vec![5i64, -100, 0, 127, -128];
        let (meta_bytes, data_bytes) =
            write_single_dense_field(0, &values, values.len() as i32, &id, "").unwrap();

        let version = check_data_header_footer(&data_bytes, &id, "").unwrap();
        assert_eq!(version, VERSION_CURRENT);

        let (meta_version, norms) = parse_meta(&meta_bytes, &id, "").unwrap();
        assert_eq!(meta_version, VERSION_CURRENT);
        let entry = norms.entry(0).unwrap();
        assert!(entry.is_dense());
        assert_eq!(entry.bytes_per_norm, 1);
        for (doc, &want) in values.iter().enumerate() {
            assert_eq!(
                norm_value(&data_bytes, entry, doc as i32).unwrap(),
                Some(want)
            );
        }
    }

    #[test]
    fn write_single_dense_field_constant_values_uses_zero_byte_encoding() {
        let id = [8u8; ID_LENGTH];
        let values = vec![3i64; 4];
        let (meta_bytes, data_bytes) =
            write_single_dense_field(1, &values, values.len() as i32, &id, "").unwrap();

        let (_, norms) = parse_meta(&meta_bytes, &id, "").unwrap();
        let entry = norms.entry(1).unwrap();
        assert_eq!(entry.bytes_per_norm, 0);
        for doc in 0..values.len() as i32 {
            assert_eq!(norm_value(&data_bytes, entry, doc).unwrap(), Some(3));
        }
        // No per-doc array is written for the constant case.
        assert!(data_bytes.len() < nvm_header_len(DATA_CODEC) + values.len() + 16);
    }

    #[test]
    fn write_single_dense_field_rejects_non_dense_input() {
        let id = [9u8; ID_LENGTH];
        let values = vec![1i64, 2];
        assert!(matches!(
            write_single_dense_field(0, &values, 3, &id, ""),
            Err(WriteError::NotDense {
                values: 2,
                max_doc: 3
            })
        ));
    }

    /// `Lucene90NormsConsumer.numBytesPerValue` picks the narrowest of
    /// 0/1/2/4/8 bytes that fits the value range. Every width must round-trip
    /// through this module's own reader, including the negative values a
    /// custom `Similarity` can produce (the reader sign-extends).
    #[test]
    fn write_single_dense_field_uses_every_per_value_width() {
        let id = [10u8; ID_LENGTH];
        let cases: [(&[i64], u8); 5] = [
            (&[3, 3, 3], 0),
            (&[-128, 0, 127], 1),
            (&[-129, 0, 300], 2),
            (&[i16::MIN as i64 - 1, 0, 70_000], 4),
            (&[i32::MIN as i64 - 1, 0, 5_000_000_000], 8),
        ];
        for (values, want_width) in cases {
            let (meta_bytes, data_bytes) =
                write_single_dense_field(1, values, values.len() as i32, &id, "").unwrap();
            let (_, norms) = parse_meta(&meta_bytes, &id, "").unwrap();
            let entry = norms.entry(1).unwrap();
            assert_eq!(entry.bytes_per_norm, want_width, "values {values:?}");
            assert!(entry.is_dense());
            check_data_header_footer(&data_bytes, &id, "").unwrap();
            for (doc, &want) in values.iter().enumerate() {
                assert_eq!(
                    norm_value(&data_bytes, entry, doc as i32).unwrap(),
                    Some(want),
                    "values {values:?} doc {doc}"
                );
            }
        }
    }

    /// The sparse branch (`numDocsWithValue != maxDoc`): only the listed docs
    /// get a norm, the value array is indexed by rank rather than doc id, and
    /// every other doc reads back as `None`.
    #[test]
    fn write_single_sparse_field_round_trips_and_skips_absent_docs() {
        let id = [11u8; ID_LENGTH];
        let max_doc = 10;
        // Deliberately unsorted input, first and last doc missing.
        let doc_values = [(7i32, 400i64), (1, 100), (4, -300), (6, 350)];
        let (meta_bytes, data_bytes) =
            write_single_sparse_field(2, &doc_values, max_doc, &id, "").unwrap();
        let (_, norms) = parse_meta(&meta_bytes, &id, "").unwrap();
        let entry = norms.entry(2).unwrap();
        assert!(!entry.is_dense() && !entry.is_empty_field());
        assert_eq!(entry.num_docs_with_field, 4);
        // -300..=400 needs 2 bytes per value.
        assert_eq!(entry.bytes_per_norm, 2);
        check_data_header_footer(&data_bytes, &id, "").unwrap();

        let mut want = std::collections::HashMap::new();
        for (doc, v) in doc_values {
            want.insert(doc, v);
        }
        for doc in 0..max_doc {
            assert_eq!(
                norm_value(&data_bytes, entry, doc).unwrap(),
                want.get(&doc).copied(),
                "doc {doc}"
            );
        }
    }

    /// `addNormsField` branches on the *count*, not on how the caller phrased
    /// it: a sparse list covering every doc is written as dense (`-1`), and an
    /// empty one as "no document has this field" (`-2`).
    #[test]
    fn write_single_sparse_field_degenerates_to_the_dense_and_empty_markers() {
        let id = [12u8; ID_LENGTH];
        let all: Vec<(i32, i64)> = (0..4).map(|d| (d, d as i64 + 1)).collect();
        let (meta_bytes, data_bytes) = write_single_sparse_field(0, &all, 4, &id, "").unwrap();
        let (_, norms) = parse_meta(&meta_bytes, &id, "").unwrap();
        let entry = norms.entry(0).unwrap();
        assert!(entry.is_dense());
        for (doc, v) in &all {
            assert_eq!(norm_value(&data_bytes, entry, *doc).unwrap(), Some(*v));
        }

        let (meta_bytes, data_bytes) = write_single_sparse_field(0, &[], 4, &id, "").unwrap();
        let (_, norms) = parse_meta(&meta_bytes, &id, "").unwrap();
        let entry = norms.entry(0).unwrap();
        assert!(entry.is_empty_field());
        assert_eq!(entry.num_docs_with_field, 0);
        for doc in 0..4 {
            assert_eq!(norm_value(&data_bytes, entry, doc).unwrap(), None);
        }
    }

    #[test]
    fn write_single_sparse_field_rejects_duplicate_and_out_of_range_doc_ids() {
        let id = [13u8; ID_LENGTH];
        assert!(matches!(
            write_single_sparse_field(0, &[(1, 1), (1, 2)], 4, &id, ""),
            Err(WriteError::DocIdsNotAscending(1))
        ));
        assert!(matches!(
            write_single_sparse_field(0, &[(4, 1)], 4, &id, ""),
            Err(WriteError::DocIdOutOfRange(4, 4))
        ));
        assert!(matches!(
            write_single_sparse_field(0, &[(-1, 1)], 4, &id, ""),
            Err(WriteError::DocIdOutOfRange(-1, 4))
        ));
    }

    /// `Lucene90NormsConsumer` puts every norms field of a segment into one
    /// `.nvm`/`.nvd` pair; entries must not cross-contaminate, and each field
    /// picks its own per-value width.
    #[test]
    fn write_fields_interleaves_several_fields_in_one_pair() {
        let id = [14u8; ID_LENGTH];
        let max_doc = 6;
        let dense: Vec<i64> = (0..max_doc as i64).map(|d| d * 1000).collect();
        let sparse = [(0i32, 5i64), (5, 9)];
        let constant = vec![42i64; max_doc as usize];
        let (meta_bytes, data_bytes) = write_fields(
            &[
                NormsField::Dense(1, &dense),
                NormsField::Sparse(2, &sparse),
                NormsField::Dense(3, &constant),
            ],
            max_doc,
            &id,
            "",
        )
        .unwrap();
        let (_, norms) = parse_meta(&meta_bytes, &id, "").unwrap();
        assert_eq!(norms.entries.len(), 3);
        check_data_header_footer(&data_bytes, &id, "").unwrap();

        let f1 = norms.entry(1).unwrap();
        assert_eq!(f1.bytes_per_norm, 2);
        for (doc, &want) in dense.iter().enumerate() {
            assert_eq!(norm_value(&data_bytes, f1, doc as i32).unwrap(), Some(want));
        }

        let f2 = norms.entry(2).unwrap();
        assert_eq!(f2.num_docs_with_field, 2);
        assert_eq!(norm_value(&data_bytes, f2, 0).unwrap(), Some(5));
        assert_eq!(norm_value(&data_bytes, f2, 3).unwrap(), None);
        assert_eq!(norm_value(&data_bytes, f2, 5).unwrap(), Some(9));

        let f3 = norms.entry(3).unwrap();
        assert_eq!(f3.bytes_per_norm, 0);
        for doc in 0..max_doc {
            assert_eq!(norm_value(&data_bytes, f3, doc).unwrap(), Some(42));
        }
    }

    #[test]
    fn write_fields_rejects_an_empty_or_duplicated_field_list() {
        let id = [15u8; ID_LENGTH];
        assert!(matches!(
            write_fields(&[], 1, &id, ""),
            Err(WriteError::EmptyFieldList)
        ));
        assert!(matches!(
            write_fields(
                &[NormsField::Dense(0, &[1]), NormsField::Dense(0, &[2])],
                1,
                &id,
                ""
            ),
            Err(WriteError::DuplicateFieldNumber(0))
        ));
    }
}
