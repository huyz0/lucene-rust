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

    /// A dense entry giving every one of `max_doc` documents the norm
    /// `value` -- `Lucene90NormsConsumer`'s constant encoding
    /// (`bytesPerNorm == 0`, the value in `normsOffset`), which reads back
    /// through [`norm_value`] without touching any `.nvd` bytes.
    pub fn constant(field_number: i32, max_doc: i32, value: i64) -> Self {
        Self {
            field_number,
            docs_with_field_offset: DOCS_WITH_FIELD_DENSE,
            docs_with_field_length: 0,
            jump_table_entry_count: -1,
            dense_rank_power: 0xFF,
            num_docs_with_field: max_doc,
            bytes_per_norm: 0,
            norms_offset: value,
        }
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
/// [`parse_meta`] does not, so it is a separate call. **The split is
/// deliberate.** Three things decide it (four production call sites; the
/// rest are `#[cfg(test)]` round-trips and integration tests):
///
/// 1. **Norms do not need `FieldInfos` to parse.** Every `.nvm` entry is
///    fixed-shape. `doc_values::parse_meta` *does* take a `&FieldInfos`, but
///    for a structural reason norms have no analogue of: a doc-values entry
///    carries a skip-index sub-record only when the field declares one, so
///    the byte stream cannot be walked without the field's `FieldInfo`. The
///    two signatures differ because the two formats do.
/// 2. **One of the four production call sites cannot take a folded form.**
///    `check_index::check_field_norms` needs the `.nvm` to parse *and then*
///    reports a separately-named `norms.entries_name_real_norms_fields`
///    check, so that a `.fnm`/`.nvm` disagreement is a named failure and the
///    rest of the norms pass (`norms.entry_present`, the per-field norm
///    values) still runs. Folding validation into `parse_meta` would abort
///    that pass at the first problem and would still need a second,
///    non-validating entry point -- it moves the split rather than removing
///    it.
/// 3. **Parse-then-validate is this port's shape, not a shortfall.**
///    `field_infos::parse` sits behind the validating `FieldInfos::new`
///    constructor for the same reason.
///
/// The *behaviour* gap Java's `readFields` covers is closed: the two places
/// its diagnostic fires -- the segment reader's open
/// (`lucene_search::directory_reader`) and `check_index` -- both call this.
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
    match indexed_disi::DisiCursor::new(
        region,
        entry.dense_rank_power,
        entry.jump_table_entry_count,
    )
    .advance_exact(doc)?
    {
        Some(ordinal) => Ok(Some(read_value_at_ordinal(data, entry, ordinal as i64)?)),
        None => Ok(None),
    }
}

/// One field's norms, for many lookups -- `Lucene90NormsProducer.getNorms`'s
/// `NumericDocValues`, whose dense one-byte case is a single `readByte` at
/// `normsOffset + doc`.
///
/// [`norm_value`] re-derives everything per call: the field's shape, its
/// offset, a `SliceInput` to seek. Measured against Lucene's per-document
/// norm read that was 1.9 ns to 0.5 ns. This resolves the dense one-byte
/// array once, so a lookup is a bounds check and a load; a sparse field keeps
/// one forward `IndexedDISI` cursor for the whole scan (rewinding for a doc
/// behind it), as `NumericReader` does; anything else falls through to
/// [`norm_value`], which owns every error case. The values returned are
/// [`norm_value`]'s, document for document.
#[derive(Debug)]
pub struct NormsReader<'a> {
    data: &'a [u8],
    entry: &'a NormsEntry,
    dense_bytes: Option<&'a [u8]>,
    docs: Option<indexed_disi::DisiCursor<'a>>,
}

impl<'a> NormsReader<'a> {
    pub fn new(data: &'a [u8], entry: &'a NormsEntry) -> Self {
        let dense_bytes =
            (!entry.is_empty_field() && entry.is_dense() && entry.bytes_per_norm == 1)
                .then(|| {
                    let start = usize::try_from(entry.norms_offset).ok()?;
                    let len = usize::try_from(entry.num_docs_with_field).ok()?;
                    data.get(start..start.checked_add(len)?)
                })
                .flatten();
        let docs = (!entry.is_empty_field() && !entry.is_dense())
            .then(|| sparse_region(data, entry).ok())
            .flatten()
            .map(|region| {
                indexed_disi::DisiCursor::new(
                    region,
                    entry.dense_rank_power,
                    entry.jump_table_entry_count,
                )
            });
        Self {
            data,
            entry,
            dense_bytes,
            docs,
        }
    }

    /// `doc`'s norm, or `None` when the field has none for it -- exactly
    /// [`norm_value`]'s answer.
    #[inline]
    pub fn value(&mut self, doc: i32) -> Result<Option<i64>> {
        if let Some(bytes) = self.dense_bytes {
            if let Some(&b) = usize::try_from(doc).ok().and_then(|d| bytes.get(d)) {
                return Ok(Some(b as i8 as i64));
            }
        }
        if let Some(cursor) = self.docs.as_mut() {
            if doc >= 0 {
                if doc < cursor.doc_id() {
                    cursor.reset();
                }
                return Ok(match cursor.advance_exact(doc)? {
                    Some(ordinal) => Some(read_value_at_ordinal(
                        self.data,
                        self.entry,
                        ordinal as i64,
                    )?),
                    None => None,
                });
            }
        }
        norm_value(self.data, self.entry, doc)
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
                let (disi_bytes, jump_table_entry_count) =
                    indexed_disi::write_with_dense_rank_power(
                        doc_ids,
                        indexed_disi::DEFAULT_DENSE_RANK_POWER,
                    );
                let offset = data.len() as i64;
                data.extend_from_slice(&disi_bytes);
                meta.write_i64(offset);
                // ARITH: `offset` was `data.len()` immediately before the
                // `extend_from_slice` above and a `Vec` only grows there.
                // `docsWithFieldLength` spans the jump table too -- it is what
                // the reader subtracts the table's bytes from.
                #[allow(clippy::arithmetic_side_effects)]
                let disi_len = data.len() as i64 - offset;
                meta.write_i64(disi_len);
                meta.write_i16(jump_table_entry_count);
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
mod tests;
