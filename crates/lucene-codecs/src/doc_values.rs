//! Port of `org.apache.lucene.codecs.lucene90.Lucene90DocValuesFormat`
//! (`.dvm` metadata + `.dvd` data + `.dvs` skip index) — read and write. All five
//! doc-values types are supported (NUMERIC, BINARY, SORTED, SORTED_SET,
//! SORTED_NUMERIC).
//!
//! A field with a non-`None` `doc_values_skip_index_type` carries an extra
//! [`DocValuesSkipperMeta`] record in `.dvm` (`Lucene90DocValuesProducer
//! .readDocValueSkipperMeta`) pointing at a run of per-interval min/max/doc-
//! count summaries in `.dvs`, decoded eagerly here by [`parse_skip_index`]
//! into a [`DocValuesSkipIndex`] -- see that function's doc comment for the
//! on-disk level structure. Range queries (`lucene-search`'s
//! `SkipBlockRange`) and sorts (the numeric and keyword competitive
//! iterators) skip whole intervals with it.
//!
//! `SORTED` fields store a per-doc ordinal (reusing the exact NUMERIC entry
//! layout — see [`NumericEntry`]) into a terms dictionary
//! ([`crate::terms_dict`]) mapping each ordinal to its term bytes.
//! `SORTED_NUMERIC`/`SORTED_SET` are the multi-valued forms: zero or more
//! numbers/ordinals per doc, addressed through a [`crate::direct_monotonic`]
//! array of `(start, end)` ranges into a flat values array — unless every
//! doc-with-a-value happens to have exactly one, in which case Java (and
//! this port) collapses back to the single-valued shape with no addresses
//! array at all. A `SORTED_SET` field storing zero-or-one ordinal per doc
//! is written as a plain `SORTED` field with a multi-valued flag byte
//! (see [`SortedSetEntry`]), not as a degenerate `SORTED_NUMERIC`.
//!
//! A NUMERIC field's values can also be split into "varying bits-per-value"
//! blocks (`Lucene90DocValuesConsumer.writeValues`'s `doBlocks` path): rather
//! than one uniform width for the whole field, each `2^blockShift`-value
//! block (16384 in real Lucene) is independently bit-packed, whenever doing
//! so saves at least 10% versus the whole-field width. On disk this is
//! `tableSize < -1` in the meta (`blockShift = -2 - tableSize`); see
//! [`NumericEntry::block_shift`] and [`decode_value_varying_bpv`] for the
//! block layout this decodes.
//!
//! Three encodings for a NUMERIC field's per-doc values (`bitsPerValue` in
//! the meta):
//! - **constant** (`bitsPerValue == 0`): every doc with a value has the same
//!   one, stored directly as `minValue`.
//! - **table-compressed**: a small (`<= 256` entry) lookup table of distinct
//!   values; each doc stores a bit-packed ordinal into it.
//! - **delta/GCD-compressed**: each doc stores a bit-packed `(value - min) /
//!   gcd`; `gcd == 1 && min == 0` is the common case (e.g. plain ordinals).
//!
//! BINARY fields are simpler: a flat concatenated byte blob, addressed either
//! directly (`doc * length`, fixed-width) or through a [`crate::direct_monotonic`]
//! array of end offsets (variable-width).
//!
//! As with [`crate::norms`], docs-with-a-value is one of empty/dense/sparse
//! (dense: implicit by doc id; sparse: via [`crate::indexed_disi`]).
//!
//! The `Adaptive90` format ([`Lucene90DocValuesFormat::rust_adaptive`])
//! writes these same files, except that a NUMERIC or SORTED_NUMERIC value
//! array may instead be [`crate::adaptive_values`] blocks where those are
//! smaller (see [`AdaptiveEntry`]). Stock Lucene does not know the format
//! name, so it refuses such a field rather than misreading it. The ordinals
//! of SORTED and SORTED_SET fields may be adaptive blocks too (see
//! `ORDS_SLACK_PERCENT`).

use std::collections::{HashMap, HashSet};
use std::hash::BuildHasherDefault;

use lucene_store::codec_util::{self, ID_LENGTH};
use lucene_store::data_input::{DataInput, SliceInput};
use lucene_store::DataOutput;

use crate::automaton::SetHasher;
use crate::direct_monotonic;
use crate::direct_reader;
use crate::field_infos::FieldInfos;
use crate::indexed_disi;
use crate::lz4;
use crate::terms_dict::{self, TermsDictEntry};
mod numeric_body;
use numeric_body::write_numeric_values_body;
mod write_terms;
use write_terms::write_terms_dict;

const SKIP_INDEX_META_CODEC: &str = "Lucene90DocValuesSkipIndex";
const DATA_META_CODEC: &str = "Lucene90DocValuesData";

const META_CODEC: &str = "Lucene90DocValuesMetadata";
const DISI_CODEC: &str = "Lucene90DocValuesDISI";
/// The `tableSize` that marks a value array of [`crate::adaptive_values`]
/// blocks, which only the `Adaptive90` format writes: below every
/// `-2 - blockShift` Lucene90 writes. The rest of such an entry keeps
/// Lucene90's shape, so Lucene's own `.dvm` parser walks past it; the
/// format's name in the field's `PerFieldDocValuesFormat.format` attribute is
/// what keeps a stock reader from opening the field.
const ADAPTIVE_TABLE_SIZE: i32 = -1000;
/// The `tableSize` of an [`ADAPTIVE_TABLE_SIZE`] value array whose
/// descriptor is followed by [`crate::adaptive_zones`] statistics: written
/// for a value array whose values are the documents in order.
const ADAPTIVE_ZONES_TABLE_SIZE: i32 = -1001;
/// `DirectMonotonicWriter` block shift of an adaptive value array's block
/// addresses.
const ADAPTIVE_ADDRESS_BLOCK_SHIFT: u32 = 16;
/// How much larger, in percent, an `Adaptive90` ordinal array with
/// statistics may be than Lucene90's packing of the same ordinals.
const ORDS_SLACK_PERCENT: usize = 3;
const VERSION_START: i32 = 0;
const VERSION_SKIPPER_SEPARATE_FILE: i32 = 1;
const VERSION_SKIPPER_MAX_VALUE_COUNT: i32 = 2;
/// Lucene 10.6: a sparse field's `IndexedDISI` moves from `.dvd` into its own
/// `.dvp` file (`Lucene90DocValuesFormat.VERSION_DISI_SEPARATE_FILE`).
const VERSION_DISI_SEPARATE_FILE: i32 = 3;
const VERSION_CURRENT: i32 = VERSION_DISI_SEPARATE_FILE;
/// The version a writer stamps by default: the last one stock Lucene 10.5.x
/// can read.
const VERSION_WRITE_DEFAULT: i32 = VERSION_SKIPPER_MAX_VALUE_COUNT;

const DOC_VALUES_TYPE_NUMERIC: u8 = 0;
const DOC_VALUES_TYPE_BINARY: u8 = 1;
const DOC_VALUES_TYPE_SORTED: u8 = 2;
const DOC_VALUES_TYPE_SORTED_SET: u8 = 3;
const DOC_VALUES_TYPE_SORTED_NUMERIC: u8 = 4;

const DOCS_WITH_FIELD_EMPTY: i64 = -2;
const DOCS_WITH_FIELD_DENSE: i64 = -1;

/// `Lucene90DocValuesFormat.TERMS_DICT_BLOCK_LZ4_SHIFT`/`_MASK`: 64 terms per
/// LZ4-compressed block (matches [`crate::terms_dict`]'s `BLOCK_SIZE`).
const TERMS_DICT_BLOCK_LZ4_SHIFT: i64 = 6;
const TERMS_DICT_BLOCK_LZ4_MASK: i64 = (1 << TERMS_DICT_BLOCK_LZ4_SHIFT) - 1;
/// `Lucene90DocValuesFormat.TERMS_DICT_REVERSE_INDEX_SHIFT`/`_MASK`: one
/// sampled term address every 1024 ordinals, for the coarse reverse index.
const TERMS_DICT_REVERSE_INDEX_SHIFT: u32 = 10;
const TERMS_DICT_REVERSE_INDEX_MASK: i64 = (1i64 << TERMS_DICT_REVERSE_INDEX_SHIFT) - 1;
/// `Lucene90DocValuesFormat.DIRECT_MONOTONIC_BLOCK_SHIFT`: every address
/// array this format writes -- BINARY's variable-length end offsets,
/// SORTED_NUMERIC/SORTED_SET's per-doc value ranges, and the terms dict's
/// block and reverse-index addresses -- is bit-packed in blocks of `2^16`
/// values, with one `(min, avgInc, offset, bitsPerValue)` record per block in
/// `.dvm`.
///
/// A shift of `0` (one block per value) would round-trip, since the shift is
/// stored per array, but the *metadata* would grow linearly with the value
/// count: 21 bytes of `.dvm` and one eagerly-materialised
/// [`direct_monotonic::Meta`] block per address, against ~336 bytes and 16
/// blocks for a 1M-doc variable-length BINARY field. See
/// `direct_monotonic_block_shift_keeps_meta_size_constant`.
const DIRECT_MONOTONIC_BLOCK_SHIFT: u32 = 16;

/// `DirectMonotonicWriter` block shift for both the terms-address and
/// reverse-index address arrays. Java's `addTermsDict`/`writeTermsIndex` both
/// pass [`DIRECT_MONOTONIC_BLOCK_SHIFT`], and so does this port.
const TERMS_DICT_DIRECT_MONOTONIC_BLOCK_SHIFT: u32 = DIRECT_MONOTONIC_BLOCK_SHIFT;

/// `Lucene90DocValuesFormat.SKIP_INDEX_MAX_LEVEL`: at most 4 levels in the
/// multi-level skip structure written per doc-values field.
const SKIP_INDEX_MAX_LEVEL: u8 = 4;

/// `Lucene90DocValuesFormat.SKIP_INDEX_LEVEL_SHIFT`: each skip level groups
/// `1 << 3 == 8` intervals of the level below it.
const SKIP_INDEX_LEVEL_SHIFT: u32 = 3;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Store(#[from] lucene_store::Error),
    #[error("unknown field number: {0}")]
    UnknownFieldNumber(i32),
    #[error("field {0} has unknown doc values type byte {1}")]
    UnsupportedFieldType(i32, u8),
    #[error("invalid table size: {0}")]
    InvalidTableSize(i32),
    #[error("invalid bitsPerValue: {0}, field number {1}")]
    InvalidBitsPerValue(u8, i32),
    #[error("doc {0} is out of range (numValues={1})")]
    DocOutOfRange(i32, i64),
    #[error("field {0} has invalid multiValued flag: {1}")]
    InvalidMultiValuedFlag(i32, u8),
    #[error("field {field_number}'s address array names values [{start}, {end}) outside its {num_values} value(s)")]
    CorruptAddressRange {
        field_number: i32,
        start: i64,
        end: i64,
        num_values: i64,
    },
    #[error("doc-values skip index level count {0} is out of range (must be 1..={SKIP_INDEX_MAX_LEVEL})")]
    InvalidSkipIndexLevelCount(u8),
    #[error(
        "doc-values skip index offset/length ({0}, {1}) is out of range for a {2}-byte .dvs slice"
    )]
    SkipIndexOutOfRange(i64, i64, usize),
    /// A `Substring90` entry's values are compressed blocks, read through
    /// [`BinaryReader::value_ref`] rather than sliced in place.
    #[error("field {0}'s values are compressed blocks; read them with BinaryReader::value_ref")]
    CompressedBinary(i32),
}

pub type Result<T> = std::result::Result<T, Error>;

/// The bytes of a segment's `.dvp` file, shared by every entry whose sparse
/// `IndexedDISI` lives there.
#[derive(Clone)]
pub struct DisiBytes(std::sync::Arc<dyn AsRef<[u8]> + Send + Sync>);

impl DisiBytes {
    pub fn new(bytes: std::sync::Arc<dyn AsRef<[u8]> + Send + Sync>) -> Self {
        Self(bytes)
    }
}

impl std::fmt::Debug for DisiBytes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "DisiBytes({} bytes)", (*self.0).as_ref().len())
    }
}

/// Where an entry's docs-with-field region (`IndexedDISI`) is: in `.dvd`
/// beside the values (format versions before 3), or in `.dvp` (from version
/// 3, Lucene 10.6). A version-3 entry whose `.dvp` was never attached reads
/// as an empty region, so a sparse lookup fails instead of misreading `.dvd`.
#[derive(Debug, Clone, Default)]
pub struct DisiLoc {
    separate: bool,
    bytes: Option<DisiBytes>,
}

impl DisiLoc {
    /// The entry's DISI lives in `.dvp`, held by `bytes`.
    pub fn in_separate_file(bytes: DisiBytes) -> Self {
        Self {
            separate: true,
            bytes: Some(bytes),
        }
    }

    /// Whether the region is in `.dvp` rather than `.dvd`.
    pub fn is_separate(&self) -> bool {
        self.separate
    }

    /// The bytes the entry's `docs_with_field_offset` indexes: `.dvp` for a
    /// version-3 entry, otherwise `data` (the `.dvd`).
    pub fn backing<'a>(&'a self, data: &'a [u8]) -> &'a [u8] {
        if !self.separate {
            return data;
        }
        self.bytes.as_ref().map_or(&[], |b| (*b.0).as_ref())
    }
}

#[derive(Debug, Clone)]
pub struct NumericEntry {
    pub field_number: i32,
    /// Where the docs-with-field region is ([`DisiLoc`]).
    pub disi: DisiLoc,
    pub docs_with_field_offset: i64,
    pub docs_with_field_length: i64,
    pub jump_table_entry_count: i16,
    pub dense_rank_power: u8,
    pub num_values: i64,
    pub table: Option<Vec<i64>>,
    pub bits_per_value: u8,
    pub min_value: i64,
    pub gcd: i64,
    pub values_offset: i64,
    pub values_length: i64,
    /// `Some(shift)` when this field's values are split into independently
    /// bit-packed blocks of `2^shift` values each (`Lucene90DocValuesConsumer
    /// .writeValues`'s `doBlocks` path, on disk as `tableSize < -1` ->
    /// `blockShift = -2 - tableSize`) rather than one uniform width for the
    /// whole field. `bits_per_value` is then a meaningless `0xFF` sentinel;
    /// each block carries its own width and its own `min`-equivalent delta,
    /// read via [`value_jump_table_offset`](Self::value_jump_table_offset)
    /// -- see [`decode_value_varying_bpv`].
    pub block_shift: Option<u32>,
    /// Absolute byte offset into the whole `.dvd` file of this field's
    /// per-block offset table (one `i64` absolute block-start offset per
    /// block, plus one trailing self-referential entry) -- only meaningful
    /// when [`block_shift`](Self::block_shift) is `Some`.
    pub value_jump_table_offset: i64,
    /// `Some` when the values are [`crate::adaptive_values`] blocks (only in
    /// a `Adaptive90` file); `bits_per_value`, `min_value`, `gcd` and
    /// `table` then describe nothing.
    pub adaptive: Option<Box<AdaptiveEntry>>,
}

/// Where a `Adaptive90` value array's blocks are. The blocks fill the
/// entry's `values_offset`/`values_length` region (padding included); the
/// entry's `valueJumpTableOffset` points at a descriptor in `.dvd` --
/// `vint blockShift`, the `DirectMonotonic` metadata, `i64 offset` and
/// `i64 length` -- of each block's start within that region, with one
/// trailing end offset. The descriptor is read once, on first use.
#[derive(Debug, Clone)]
pub struct AdaptiveEntry {
    /// Offset in `.dvd` of the block-address descriptor.
    pub descriptor_offset: i64,
    /// Whether [`crate::adaptive_zones`] statistics follow the descriptor
    /// (an [`ADAPTIVE_ZONES_TABLE_SIZE`] entry).
    pub zones: bool,
    addresses: std::sync::OnceLock<AdaptiveAddresses>,
}

/// An adaptive value array's block addresses, decoded from its descriptor.
#[derive(Debug, Clone)]
pub struct AdaptiveAddresses {
    pub offset: i64,
    pub length: i64,
    pub meta: direct_monotonic::Meta,
    /// The per-window statistics, when the entry has them.
    pub zones: Option<crate::adaptive_zones::Zones>,
}

impl AdaptiveEntry {
    /// An entry whose descriptor sits at `descriptor_offset` in `.dvd`.
    pub fn new(descriptor_offset: i64) -> Self {
        AdaptiveEntry {
            descriptor_offset,
            zones: false,
            addresses: std::sync::OnceLock::new(),
        }
    }

    /// The block addresses of a value array of `num_values` values, read
    /// from `data` (the whole `.dvd` file) the first time.
    pub fn addresses(&self, data: &[u8], num_values: i64) -> Result<&AdaptiveAddresses> {
        if let Some(a) = self.addresses.get() {
            return Ok(a);
        }
        let start = usize::try_from(self.descriptor_offset)
            .map_err(|_| lucene_store::Error::Eof { offset: 0 })?;
        let mut input = SliceInput::new(data);
        input.seek(start)?;
        let block_shift = input.read_vint()?;
        if num_values < 0 {
            return Err(Error::InvalidTableSize(ADAPTIVE_TABLE_SIZE));
        }
        let num_blocks = num_values
            .saturating_add(crate::adaptive_values::BLOCK_SIZE as i64)
            .saturating_sub(1)
            >> crate::adaptive_values::BLOCK_SHIFT;
        let meta = direct_monotonic::load_meta(
            &mut input,
            num_blocks.saturating_add(1),
            u32::try_from(block_shift).map_err(|_| Error::InvalidTableSize(ADAPTIVE_TABLE_SIZE))?,
        )?;
        let offset = input.read_i64()?;
        let length = input.read_i64()?;
        let zones = if self.zones {
            Some(crate::adaptive_zones::Zones::parse(
                data,
                input.position(),
                num_values,
            )?)
        } else {
            None
        };
        Ok(self.addresses.get_or_init(|| AdaptiveAddresses {
            offset,
            length,
            meta,
            zones,
        }))
    }
}

impl NumericEntry {
    /// The docs-with-field (`IndexedDISI`) region of a sparse field, out of
    /// `data` (the `.dvd`) or the `.dvp` this entry was opened with.
    pub fn docs_region<'a>(&'a self, data: &'a [u8]) -> Result<&'a [u8]> {
        region(
            self.disi.backing(data),
            self.docs_with_field_offset,
            self.docs_with_field_length,
        )
    }

    pub fn is_empty_field(&self) -> bool {
        self.docs_with_field_offset == DOCS_WITH_FIELD_EMPTY
    }

    pub fn is_dense(&self) -> bool {
        self.docs_with_field_offset == DOCS_WITH_FIELD_DENSE
    }
}

#[derive(Debug, Clone)]
pub struct BinaryEntry {
    pub field_number: i32,
    /// Where the docs-with-field region is ([`DisiLoc`]).
    pub disi: DisiLoc,
    pub docs_with_field_offset: i64,
    pub docs_with_field_length: i64,
    pub jump_table_entry_count: i16,
    pub dense_rank_power: u8,
    pub num_docs_with_field: i32,
    pub min_length: i32,
    pub max_length: i32,
    pub data_offset: i64,
    pub data_length: i64,
    /// `None` for fixed-length fields (`min_length == max_length`), where an
    /// entry's offset is computed directly as `ordinal * max_length`.
    pub addresses: Option<BinaryAddresses>,
    /// The `Substring90` format's blocks and gram index, in place of one
    /// byte run ([`crate::substring`]).
    pub substring: Option<Box<crate::substring::SubstringEntry>>,
    /// The mask the values are shown through ([`crate::value_mask`]), `None` for a field shown
    /// as it is: [`BinaryReader::value_ref`] applies it and the paths that cannot refuse.
    pub mask: Option<crate::value_mask::SharedMask>,
}

#[derive(Debug, Clone)]
pub struct BinaryAddresses {
    pub offset: i64,
    pub length: i64,
    pub meta: direct_monotonic::Meta,
}

impl BinaryEntry {
    /// [`NumericEntry::docs_region`] for a BINARY field.
    pub fn docs_region<'a>(&'a self, data: &'a [u8]) -> Result<&'a [u8]> {
        region(
            self.disi.backing(data),
            self.docs_with_field_offset,
            self.docs_with_field_length,
        )
    }

    pub fn is_empty_field(&self) -> bool {
        self.docs_with_field_offset == DOCS_WITH_FIELD_EMPTY
    }

    pub fn is_dense(&self) -> bool {
        self.docs_with_field_offset == DOCS_WITH_FIELD_DENSE
    }

    pub fn is_fixed_length(&self) -> bool {
        self.min_length == self.max_length
    }
}

/// A single-valued SORTED field: a per-doc ordinal ([`NumericEntry`],
/// exactly the same layout NUMERIC fields use) into a terms dictionary.
#[derive(Debug, Clone)]
pub struct SortedEntry {
    pub field_number: i32,
    pub ords: NumericEntry,
    pub terms: TermsDictEntry,
}

/// A `(start, end)` address range array into a flat multi-valued array --
/// present only when a doc-with-a-value can have more than one value; if
/// every doc-with-a-value has exactly one, Java (and this port) collapses
/// to the single-valued shape (`addresses: None`) with no address array at
/// all, since each value's ordinal is then just its doc's rank.
#[derive(Debug, Clone)]
pub struct MultiValueAddresses {
    pub offset: i64,
    pub length: i64,
    pub meta: direct_monotonic::Meta,
}

/// A multi-valued (zero or more per doc) numeric field, or the ordinal
/// half of a multi-valued SORTED_SET field -- both share this exact entry
/// shape (`Lucene90DocValuesProducer.readSortedNumeric`).
#[derive(Debug, Clone)]
pub struct SortedNumericEntry {
    pub field_number: i32,
    pub numeric: NumericEntry,
    pub num_docs_with_field: i32,
    pub addresses: Option<MultiValueAddresses>,
}

/// A SORTED_SET field: either written as a plain single-valued [`SortedEntry`]
/// (every doc has zero or one ordinal) or a true multi-valued form (ordinals
/// via [`SortedNumericEntry`], resolved against the same terms dictionary).
#[derive(Debug, Clone)]
pub enum SortedSetKind {
    Single(SortedEntry),
    Multi {
        ords: SortedNumericEntry,
        terms: TermsDictEntry,
    },
}

#[derive(Debug, Clone)]
pub struct SortedSetEntry {
    pub field_number: i32,
    pub kind: SortedSetKind,
}

/// `Lucene90DocValuesProducer.DocValuesSkipperEntry`: the fixed-size summary
/// recorded in `.dvm` for a field with a doc-values skip index, pointing at
/// its per-interval detail in `.dvs` (see [`DocValuesSkipIndex`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DocValuesSkipperMeta {
    /// Byte offset of this field's skip data within `.dvs`.
    pub offset: i64,
    /// Byte length of this field's skip data within `.dvs`.
    pub length: i64,
    pub min_value: i64,
    pub max_value: i64,
    pub doc_count: i32,
    pub max_doc_id: i32,
    /// Most values per doc seen across the whole field, or `-1` if unknown
    /// (pre-`VERSION_SKIPPER_MAX_VALUE_COUNT` files never recorded it and
    /// `doc_count == 0` is the only case this port can still infer as `0`).
    pub max_value_count: i32,
}

/// One level's summary within a single skip interval:
/// `Lucene90DocValuesConsumer.SkipAccumulator`, as written by `writeLevels`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SkipIndexLevelInterval {
    pub min_doc_id: i32,
    pub max_doc_id: i32,
    pub min_value: i64,
    pub max_value: i64,
    pub doc_count: i32,
}

/// One skip interval, one entry per level it was written at.
/// `levels[0]` is the finest (base, every-`skipIndexIntervalSize`-docs)
/// interval; `levels[levels.len() - 1]` is the coarsest. On disk the levels
/// are written coarsest-first (so a lazy/streaming reader can bail out
/// after the first uncompetitive level without reading the rest) but this
/// port reorders them to level-ascending on the way in, since it decodes
/// every interval eagerly and ascending is the more natural iteration order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SkipIndexInterval {
    pub levels: Vec<SkipIndexLevelInterval>,
}

/// A field's whole decoded doc-values skip index: the `.dvm` summary plus
/// every interval decoded out of its `.dvs` slice. See [`parse_skip_index`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DocValuesSkipIndex {
    pub min_value: i64,
    pub max_value: i64,
    pub doc_count: i32,
    pub max_doc_id: i32,
    pub max_value_count: i32,
    pub intervals: Vec<SkipIndexInterval>,
}

#[derive(Debug, Clone, Default)]
pub struct DocValuesMeta {
    pub numeric: Vec<NumericEntry>,
    pub binary: Vec<BinaryEntry>,
    pub sorted: Vec<SortedEntry>,
    pub sorted_numeric: Vec<SortedNumericEntry>,
    pub sorted_set: Vec<SortedSetEntry>,
    /// Keyed by field number, present only for fields whose `FieldInfo` has
    /// a non-`None` `doc_values_skip_index_type` (mirrors the Java reader's
    /// own `skippers` map -- a skip index is orthogonal to the field's
    /// doc-values *type*, so it isn't folded into any of the `Vec`s above).
    pub skippers: std::collections::HashMap<i32, DocValuesSkipperMeta>,
}

impl DocValuesMeta {
    /// Gives every entry whose `IndexedDISI` is in `.dvp` the file's bytes.
    pub fn attach_disi(&mut self, bytes: DisiBytes) {
        self.for_each_disi_mut(|loc| {
            if loc.separate {
                loc.bytes = Some(bytes.clone());
            }
        });
    }

    fn for_each_disi_mut(&mut self, mut f: impl FnMut(&mut DisiLoc)) {
        for e in &mut self.numeric {
            f(&mut e.disi);
        }
        for e in &mut self.binary {
            f(&mut e.disi);
        }
        for e in &mut self.sorted {
            f(&mut e.ords.disi);
        }
        for e in &mut self.sorted_numeric {
            f(&mut e.numeric.disi);
        }
        for e in &mut self.sorted_set {
            match &mut e.kind {
                SortedSetKind::Single(sorted) => f(&mut sorted.ords.disi),
                SortedSetKind::Multi { ords, .. } => f(&mut ords.numeric.disi),
            }
        }
    }

    pub fn numeric_entry(&self, field_number: i32) -> Option<&NumericEntry> {
        self.numeric.iter().find(|e| e.field_number == field_number)
    }

    pub fn binary_entry(&self, field_number: i32) -> Option<&BinaryEntry> {
        self.binary.iter().find(|e| e.field_number == field_number)
    }

    pub fn sorted_entry(&self, field_number: i32) -> Option<&SortedEntry> {
        self.sorted.iter().find(|e| e.field_number == field_number)
    }

    pub fn sorted_numeric_entry(&self, field_number: i32) -> Option<&SortedNumericEntry> {
        self.sorted_numeric
            .iter()
            .find(|e| e.field_number == field_number)
    }

    pub fn sorted_set_entry(&self, field_number: i32) -> Option<&SortedSetEntry> {
        self.sorted_set
            .iter()
            .find(|e| e.field_number == field_number)
    }

    /// The `.dvm` skip-index summary for `field_number`, if that field has
    /// one. Pass it to [`parse_skip_index`] along with the segment's `.dvs`
    /// bytes to decode the full per-interval structure.
    pub fn skipper_meta(&self, field_number: i32) -> Option<&DocValuesSkipperMeta> {
        self.skippers.get(&field_number)
    }
}

/// Parses a whole `.dvm` metadata file already read into memory. `field_infos`
/// must be the segment's already-parsed `.fnm` (used to reject unknown field
/// numbers and to detect unsupported per-field doc-values skip indexes).
/// Whether doc-values files of format `version` (from [`parse_meta`]) keep
/// sparse fields' `IndexedDISI` in a `.dvp` file, which the caller then
/// opens and hands to [`DocValuesMeta::attach_disi`].
pub fn has_disi_file(version: i32) -> bool {
    version >= VERSION_DISI_SEPARATE_FILE
}

pub fn parse_meta(
    buf: &[u8],
    segment_id: &[u8; ID_LENGTH],
    segment_suffix: &str,
    field_infos: &FieldInfos,
) -> Result<(i32, DocValuesMeta)> {
    let mut input = SliceInput::new(buf);
    let header = codec_util::check_index_header(
        &mut input,
        META_CODEC,
        VERSION_START,
        VERSION_CURRENT,
        segment_id,
        segment_suffix,
    )?;

    let mut meta = DocValuesMeta::default();
    loop {
        let field_number = input.read_i32()?;
        if field_number == -1 {
            break;
        }
        let field = field_infos
            .field_by_number(field_number)
            .ok_or(Error::UnknownFieldNumber(field_number))?;

        let ty = input.read_byte()?;
        if field.doc_values_skip_index_type != crate::field_infos::DocValuesSkipIndexType::None {
            let skipper = read_skipper_meta(&mut input, header.version)?;
            meta.skippers.insert(field_number, skipper);
        }
        match ty {
            DOC_VALUES_TYPE_NUMERIC => {
                meta.numeric
                    .push(read_numeric_entry(&mut input, field_number)?);
            }
            DOC_VALUES_TYPE_BINARY => {
                meta.binary
                    .push(read_binary_entry(&mut input, field_number)?);
            }
            DOC_VALUES_TYPE_SORTED => {
                meta.sorted
                    .push(read_sorted_entry(&mut input, field_number)?);
            }
            DOC_VALUES_TYPE_SORTED_NUMERIC => {
                meta.sorted_numeric
                    .push(read_sorted_numeric_entry(&mut input, field_number)?);
            }
            DOC_VALUES_TYPE_SORTED_SET => {
                meta.sorted_set
                    .push(read_sorted_set_entry(&mut input, field_number)?);
            }
            other => return Err(Error::UnsupportedFieldType(field_number, other)),
        }
    }

    if header.version < VERSION_SKIPPER_MAX_VALUE_COUNT {
        infer_max_value_counts(&mut meta, field_infos);
    }
    if header.version >= VERSION_DISI_SEPARATE_FILE {
        meta.for_each_disi_mut(|loc| loc.separate = true);
    }

    codec_util::check_footer(&mut input, buf.len())?;

    Ok((header.version, meta))
}

/// Port of `Lucene90DocValuesProducer.inferMaxValueCounts`, run only for
/// `.dvm` files older than [`VERSION_SKIPPER_MAX_VALUE_COUNT`], which never
/// stored `maxValueCount` and so leave [`read_skipper_meta`] with the `-1`
/// "unknown" sentinel.
///
/// For a single-valued doc-values type the answer is known without reading
/// anything: NUMERIC and SORTED store exactly one value per doc-with-a-value.
/// SORTED_NUMERIC and SORTED_SET only *sometimes* are -- they carry one value
/// per doc precisely when the field collapsed to the no-address-array shape
/// (`numValues == numDocsWithField`), which is the same equality
/// [`read_sorted_numeric_entry`] already uses to decide whether an address
/// array is present. Anything else (a genuinely multi-valued field, or a
/// BINARY field, whose skip index Java also leaves alone) keeps `-1`.
fn infer_max_value_counts(meta: &mut DocValuesMeta, field_infos: &FieldInfos) {
    use crate::field_infos::DocValuesType;

    let inferred: Vec<(i32, i32)> = meta
        .skippers
        .iter()
        .filter(|(_, e)| e.max_value_count == -1 && e.doc_count != 0)
        .filter_map(|(&field_number, _)| {
            let ty = field_infos.field_by_number(field_number)?.doc_values_type;
            let single = match ty {
                DocValuesType::Numeric | DocValuesType::Sorted => true,
                DocValuesType::SortedNumeric => meta
                    .sorted_numeric_entry(field_number)
                    .is_some_and(|e| e.numeric.num_values == e.num_docs_with_field as i64),
                DocValuesType::SortedSet => {
                    meta.sorted_set_entry(field_number)
                        .is_some_and(|e| match &e.kind {
                            SortedSetKind::Single(_) => true,
                            SortedSetKind::Multi { ords, .. } => {
                                ords.numeric.num_values == ords.num_docs_with_field as i64
                            }
                        })
                }
                DocValuesType::None | DocValuesType::Binary => false,
            };
            single.then_some((field_number, 1))
        })
        .collect();

    for (field_number, max_value_count) in inferred {
        if let Some(entry) = meta.skippers.get_mut(&field_number) {
            entry.max_value_count = max_value_count;
        }
    }
}

fn read_binary_entry(input: &mut SliceInput, field_number: i32) -> Result<BinaryEntry> {
    let data_offset = input.read_i64()?;
    let data_length = input.read_i64()?;
    let docs_with_field_offset = input.read_i64()?;
    let docs_with_field_length = input.read_i64()?;
    let jump_table_entry_count = input.read_i16()?;
    let dense_rank_power = input.read_byte()?;
    let num_docs_with_field = input.read_i32()?;
    let min_length = input.read_i32()?;
    let max_length = input.read_i32()?;

    if min_length == crate::substring::MIN_LENGTH_MARKER {
        let substring = crate::substring::read_meta(input)?;
        return Ok(BinaryEntry {
            mask: None,
            disi: DisiLoc::default(),
            field_number,
            docs_with_field_offset,
            docs_with_field_length,
            jump_table_entry_count,
            dense_rank_power,
            num_docs_with_field,
            min_length,
            max_length,
            data_offset,
            data_length,
            addresses: None,
            substring: Some(Box::new(substring)),
        });
    }
    let addresses = if min_length < max_length {
        let addresses_offset = input.read_i64()?;
        // ARITH: `num_docs_with_field` is an `i32` widened to `i64`, so the
        // `+ 1` is nowhere near an `i64` boundary. `load_meta` is what bounds
        // the value itself (it rejects a negative count and caps
        // the reservation by the bytes left in `.dvm`).
        #[allow(clippy::arithmetic_side_effects)]
        let num_addresses = num_docs_with_field as i64 + 1;
        let block_shift = input.read_vint()?;
        let addr_meta = direct_monotonic::load_meta(input, num_addresses, block_shift as u32)?;
        let addresses_length = input.read_i64()?;
        Some(BinaryAddresses {
            offset: addresses_offset,
            length: addresses_length,
            meta: addr_meta,
        })
    } else {
        None
    };

    let _ = field_number;
    Ok(BinaryEntry {
        mask: None,
        disi: DisiLoc::default(),
        field_number,
        docs_with_field_offset,
        docs_with_field_length,
        jump_table_entry_count,
        dense_rank_power,
        num_docs_with_field,
        min_length,
        max_length,
        data_offset,
        data_length,
        addresses,
        substring: None,
    })
}

fn read_sorted_entry(input: &mut SliceInput, field_number: i32) -> Result<SortedEntry> {
    let ords = read_numeric_entry(input, field_number)?;
    let terms = terms_dict::read_term_dict_entry(input)?;
    Ok(SortedEntry {
        field_number,
        ords,
        terms,
    })
}

/// Port of `Lucene90DocValuesProducer.readSortedNumeric`: a NUMERIC entry
/// (`numValues` = total value count across all docs) plus, only when a
/// doc-with-a-value can hold more than one value, an address array mapping
/// each doc's rank to a `[start, end)` range in that flat value array.
fn read_sorted_numeric_entry(
    input: &mut SliceInput,
    field_number: i32,
) -> Result<SortedNumericEntry> {
    let numeric = read_numeric_entry(input, field_number)?;
    let num_docs_with_field = input.read_i32()?;
    let addresses = if num_docs_with_field as i64 != numeric.num_values {
        let addresses_offset = input.read_i64()?;
        let block_shift = input.read_vint()?;
        // ARITH: as in `read_binary_entry` -- an `i32` widened to `i64`.
        #[allow(clippy::arithmetic_side_effects)]
        let num_addresses = num_docs_with_field as i64 + 1;
        let meta = direct_monotonic::load_meta(input, num_addresses, block_shift as u32)?;
        let addresses_length = input.read_i64()?;
        Some(MultiValueAddresses {
            offset: addresses_offset,
            length: addresses_length,
            meta,
        })
    } else {
        None
    };
    Ok(SortedNumericEntry {
        field_number,
        numeric,
        num_docs_with_field,
        addresses,
    })
}

/// Port of `Lucene90DocValuesProducer.readSortedSet`.
fn read_sorted_set_entry(input: &mut SliceInput, field_number: i32) -> Result<SortedSetEntry> {
    let multi_valued = input.read_byte()?;
    let kind = match multi_valued {
        0 => SortedSetKind::Single(read_sorted_entry(input, field_number)?),
        1 => {
            let ords = read_sorted_numeric_entry(input, field_number)?;
            let terms = terms_dict::read_term_dict_entry(input)?;
            SortedSetKind::Multi { ords, terms }
        }
        other => return Err(Error::InvalidMultiValuedFlag(field_number, other)),
    };
    Ok(SortedSetEntry { field_number, kind })
}

fn read_numeric_entry(input: &mut SliceInput, field_number: i32) -> Result<NumericEntry> {
    let docs_with_field_offset = input.read_i64()?;
    let docs_with_field_length = input.read_i64()?;
    let jump_table_entry_count = input.read_i16()?;
    let dense_rank_power = input.read_byte()?;
    let num_values = input.read_i64()?;

    let table_size = input.read_i32()?;
    if table_size > 256 {
        return Err(Error::InvalidTableSize(table_size));
    }
    if table_size == ADAPTIVE_TABLE_SIZE || table_size == ADAPTIVE_ZONES_TABLE_SIZE {
        return read_adaptive_numeric_entry(
            input,
            field_number,
            docs_with_field_offset,
            docs_with_field_length,
            jump_table_entry_count,
            dense_rank_power,
            num_values,
            table_size == ADAPTIVE_ZONES_TABLE_SIZE,
        );
    }
    // `tableSize < -1` means this field's values are split into varying-
    // bits-per-value blocks (`blockShift = -2 - tableSize`) rather than a
    // lookup table; `bits_per_value` below is then a meaningless `0xFF`
    // sentinel (real Lucene writes `numBitsPerValue = 0xFF` in this case),
    // so skip the normal table/enum reads for it.
    // ARITH: `table_size` is an `i32`, so `-2 - table_size` as `i64`
    // spans at most `-2 - i32::MIN`, three orders of magnitude inside `i64`.
    #[allow(clippy::arithmetic_side_effects)]
    let (table, block_shift) = if table_size < -1 {
        let shift = -2i64 - table_size as i64;
        if !(0..=63).contains(&shift) {
            return Err(Error::InvalidTableSize(table_size));
        }
        (None, Some(shift as u32))
    } else if table_size >= 0 {
        let mut t = Vec::with_capacity(table_size as usize);
        for _ in 0..table_size {
            t.push(input.read_i64()?);
        }
        (Some(t), None)
    } else {
        (None, None)
    };

    let bits_per_value = input.read_byte()?;
    if block_shift.is_none()
        && !matches!(
            bits_per_value,
            0 | 1 | 2 | 4 | 8 | 12 | 16 | 20 | 24 | 28 | 32 | 40 | 48 | 56 | 64
        )
    {
        return Err(Error::InvalidBitsPerValue(bits_per_value, field_number));
    }
    let min_value = input.read_i64()?;
    let gcd = input.read_i64()?;
    let values_offset = input.read_i64()?;
    let values_length = input.read_i64()?;
    let value_jump_table_offset = input.read_i64()?; // only meaningful for varying-bpv blocks

    Ok(NumericEntry {
        disi: DisiLoc::default(),
        field_number,
        docs_with_field_offset,
        docs_with_field_length,
        jump_table_entry_count,
        dense_rank_power,
        num_values,
        table,
        bits_per_value,
        min_value,
        gcd,
        values_offset,
        values_length,
        block_shift,
        value_jump_table_offset,
        adaptive: None,
    })
}

/// The rest of a NUMERIC entry whose `tableSize` is
/// [`ADAPTIVE_TABLE_SIZE`]: Lucene90's fields (`valueJumpTableOffset` holding
/// the block addresses' offset), then the addresses' `DirectMonotonic`
/// metadata and length.
#[allow(clippy::too_many_arguments)]
fn read_adaptive_numeric_entry(
    input: &mut SliceInput,
    field_number: i32,
    docs_with_field_offset: i64,
    docs_with_field_length: i64,
    jump_table_entry_count: i16,
    dense_rank_power: u8,
    num_values: i64,
    zones: bool,
) -> Result<NumericEntry> {
    let bits_per_value = input.read_byte()?;
    let min_value = input.read_i64()?;
    let gcd = input.read_i64()?;
    let values_offset = input.read_i64()?;
    let values_length = input.read_i64()?;
    let descriptor_offset = input.read_i64()?;
    Ok(NumericEntry {
        disi: DisiLoc::default(),
        field_number,
        docs_with_field_offset,
        docs_with_field_length,
        jump_table_entry_count,
        dense_rank_power,
        num_values,
        table: None,
        bits_per_value,
        min_value,
        gcd,
        values_offset,
        values_length,
        block_shift: None,
        value_jump_table_offset: descriptor_offset,
        adaptive: Some(Box::new(AdaptiveEntry {
            zones,
            ..AdaptiveEntry::new(descriptor_offset)
        })),
    })
}

/// Validates a whole `.dvd` data file's header/footer, mirroring
/// [`crate::norms::check_data_header_footer`] (structural-only footer check,
/// no full-file CRC — same rationale: forward-only doc-values access makes a
/// full recompute too costly to do on every open).
pub fn check_data_header_footer(
    buf: &[u8],
    segment_id: &[u8; ID_LENGTH],
    segment_suffix: &str,
) -> Result<i32> {
    let mut input = SliceInput::new(buf);
    let header = codec_util::check_index_header(
        &mut input,
        DATA_META_CODEC,
        VERSION_START,
        VERSION_CURRENT,
        segment_id,
        segment_suffix,
    )?;
    codec_util::retrieve_checksum(buf)?;
    Ok(header.version)
}

/// Validates a `.dvp` file's header and footer against the `.dvm`'s format
/// `meta_version` (`Lucene90DocValuesProducer`'s `disiVersion` check).
pub fn check_disi_header_footer(
    buf: &[u8],
    segment_id: &[u8; ID_LENGTH],
    segment_suffix: &str,
    meta_version: i32,
) -> Result<()> {
    let mut input = SliceInput::new(buf);
    let header = codec_util::check_index_header(
        &mut input,
        DISI_CODEC,
        VERSION_DISI_SEPARATE_FILE,
        VERSION_CURRENT,
        segment_id,
        segment_suffix,
    )?;
    if header.version != meta_version {
        return Err(lucene_store::Error::Corrupted(format!(
            "Format versions mismatch: meta={meta_version}, disi={}",
            header.version
        ))
        .into());
    }
    codec_util::retrieve_checksum(buf)?;
    Ok(())
}

/// Reads one `Lucene90DocValuesProducer.DocValuesSkipperEntry` out of
/// `.dvm` -- called from [`parse_meta`] right after the field's type byte,
/// for any field whose `FieldInfo` carries a non-`None`
/// `doc_values_skip_index_type`. `meta_version` is the `.dvm` header's own
/// version (not `.dvs`'s), matching Java's `readDocValueSkipperMeta`, which
/// gates the trailing `maxValueCount` field on it.
fn read_skipper_meta(input: &mut SliceInput, meta_version: i32) -> Result<DocValuesSkipperMeta> {
    let offset = input.read_i64()?;
    let length = input.read_i64()?;
    let max_value = input.read_i64()?;
    let min_value = input.read_i64()?;
    let doc_count = input.read_i32()?;
    let max_doc_id = input.read_i32()?;
    let max_value_count = if meta_version >= VERSION_SKIPPER_MAX_VALUE_COUNT {
        input.read_i32()?
    } else if doc_count == 0 {
        0
    } else {
        -1
    };
    Ok(DocValuesSkipperMeta {
        offset,
        length,
        min_value,
        max_value,
        doc_count,
        max_doc_id,
        max_value_count,
    })
}

/// Validates a `.dvs` skip-index file's header/footer -- same structural,
/// no-full-checksum shape as [`check_data_header_footer`], but against
/// `.dvs`'s own codec name.
pub fn check_skip_index_header_footer(
    buf: &[u8],
    segment_id: &[u8; ID_LENGTH],
    segment_suffix: &str,
) -> Result<i32> {
    let mut input = SliceInput::new(buf);
    let header = codec_util::check_index_header(
        &mut input,
        SKIP_INDEX_META_CODEC,
        // A `.dvs` file only exists from `VERSION_SKIPPER_SEPARATE_FILE`
        // onwards -- before that the skip data lived inline in `.dvd` -- so a
        // `.dvs` claiming version 0 is corrupt, exactly as Java's own
        // `checkIndexHeader(skipIn, ..., VERSION_SKIPPER_SEPARATE_FILE, ...)`
        // rejects it.
        VERSION_SKIPPER_SEPARATE_FILE,
        VERSION_CURRENT,
        segment_id,
        segment_suffix,
    )?;
    codec_util::retrieve_checksum(buf)?;
    Ok(header.version)
}

/// Decodes the full per-interval doc-values skip index
/// (`Lucene90DocValuesConsumer.writeLevels`'s inverse) for one field, given
/// its `.dvm` summary (`skipper`) and the segment's whole `.dvs` buffer.
///
/// Real Lucene reads this lazily, interval-by-interval, while `advance`-ing
/// a query scan (`Lucene90DocValuesProducer.getSkipper`); this port instead
/// decodes every interval in `[skipper.offset, skipper.offset +
/// skipper.length)` up front into a `Vec`, which is simpler to reason about
/// and to test, at the cost of not being useful yet for actually skipping
/// (see the module doc comment's TODO on wiring this into a range-query
/// scan).
///
/// Each interval is a run of 1..=[`SKIP_INDEX_MAX_LEVEL`] levels, written
/// coarsest-first:
/// - one `u8` level count,
/// - then, per level from coarsest down to level 0: `maxDocID: i32`,
///   `minDocID: i32`, `maxValue: i64`, `minValue: i64`, `docCount: i32`
///   (28 bytes/level).
///
/// This function reads sequential intervals until the byte window is fully
/// consumed (real Lucene never needs to do this -- it stops once it finds a
/// competitive interval -- so there's no equivalent Java loop to mirror
/// beyond `writeLevels`'s write order).
pub fn parse_skip_index(
    dvs_buf: &[u8],
    segment_id: &[u8; ID_LENGTH],
    segment_suffix: &str,
    skipper: &DocValuesSkipperMeta,
) -> Result<DocValuesSkipIndex> {
    check_skip_index_header_footer(dvs_buf, segment_id, segment_suffix)?;

    let offset = usize::try_from(skipper.offset)
        .map_err(|_| Error::SkipIndexOutOfRange(skipper.offset, skipper.length, dvs_buf.len()))?;
    let length = usize::try_from(skipper.length)
        .map_err(|_| Error::SkipIndexOutOfRange(skipper.offset, skipper.length, dvs_buf.len()))?;
    let end = offset
        .checked_add(length)
        .filter(|&e| e <= dvs_buf.len())
        .ok_or(Error::SkipIndexOutOfRange(
            skipper.offset,
            skipper.length,
            dvs_buf.len(),
        ))?;

    let mut input = SliceInput::new(&dvs_buf[offset..end]);
    let mut intervals = Vec::new();
    while input.position() < length {
        let levels = input.read_byte()?;
        if levels == 0 || levels > SKIP_INDEX_MAX_LEVEL {
            return Err(Error::InvalidSkipIndexLevelCount(levels));
        }
        let mut by_level: Vec<Option<SkipIndexLevelInterval>> = vec![None; levels as usize];
        for level in (0..levels as usize).rev() {
            let max_doc_id = input.read_i32()?;
            let min_doc_id = input.read_i32()?;
            let max_value = input.read_i64()?;
            let min_value = input.read_i64()?;
            let doc_count = input.read_i32()?;
            by_level[level] = Some(SkipIndexLevelInterval {
                min_doc_id,
                max_doc_id,
                min_value,
                max_value,
                doc_count,
            });
        }
        intervals.push(SkipIndexInterval {
            levels: by_level
                .into_iter()
                .map(|l| l.expect("every level index was written above"))
                .collect(),
        });
    }

    Ok(DocValuesSkipIndex {
        min_value: skipper.min_value,
        max_value: skipper.max_value,
        doc_count: skipper.doc_count,
        max_doc_id: skipper.max_doc_id,
        max_value_count: skipper.max_value_count,
        intervals,
    })
}

/// `DocIdSetIterator.NO_MORE_DOCS`: the sentinel every level's min/max doc id
/// takes once a [`DocValuesSkipper`] runs past its field's last interval.
pub const NO_MORE_DOCS: i32 = i32::MAX;

/// A cursor over one field's decoded [`DocValuesSkipIndex`], with exactly the
/// semantics of Java's `DocValuesSkipper` as
/// `Lucene90DocValuesProducer.getSkipper` implements it.
///
/// The position only ever moves forward, via [`advance`](Self::advance); the
/// interval it lands on is described per level by
/// [`min_doc_id`](Self::min_doc_id)/[`max_doc_id`](Self::max_doc_id)/
/// [`min_value`](Self::min_value)/[`max_value`](Self::max_value)/
/// [`doc_count`](Self::doc_count), level 0 being the finest
/// (`skipIndexIntervalSize` docs) and each higher level covering `8^level`
/// level-0 intervals.
///
/// Three behaviours are easy to get wrong and are reproduced deliberately:
///
/// - **Levels persist across intervals.** Java stores the per-level arrays
///   outside the read loop and only overwrites the levels the *current*
///   interval record carries. A record for a plain level-0 interval leaves
///   levels 1..3 holding the coarse summaries read from the last interval that
///   had them -- which is exactly what makes the
///   `while (levels < MAX && maxDocID[levels] >= target) levels++` widening at
///   the end of `advance` meaningful.
/// - **A level's `maxDocID` is stored before it is tested.** Java writes
///   `if ((maxDocID[level] = input.readInt()) < target)`, so even the level
///   whose bound *failed* leaves its `maxDocID` behind; the widening loop above
///   depends on that too.
/// - **Skipping is by whole subtrees.** Bailing out at level `L` jumps
///   `SKIP_INDEX_JUMP_LENGTH_PER_LEVEL[L]` bytes, which is exactly the byte
///   size of the `8^L` level-0 intervals that level covers (`writeLevels`
///   gives a level-`L` record only to a complete group). Over this port's
///   already-decoded `Vec<SkipIndexInterval>` that is a `+= 8^L` on the
///   interval cursor.
///
/// Unlike Java's, this cursor never does I/O -- [`parse_skip_index`] decoded
/// the whole `.dvs` slice up front -- so [`advance`](Self::advance) is
/// infallible.
#[derive(Debug, Clone)]
pub struct DocValuesSkipper<'a> {
    index: &'a DocValuesSkipIndex,
    /// Index of the next interval to read, i.e. Java's file pointer into the
    /// field's `.dvs` slice.
    cursor: usize,
    levels: usize,
    min_doc_id: [i32; SKIP_INDEX_MAX_LEVEL as usize],
    max_doc_id: [i32; SKIP_INDEX_MAX_LEVEL as usize],
    min_value: [i64; SKIP_INDEX_MAX_LEVEL as usize],
    max_value: [i64; SKIP_INDEX_MAX_LEVEL as usize],
    doc_count: [i32; SKIP_INDEX_MAX_LEVEL as usize],
}

impl<'a> DocValuesSkipper<'a> {
    /// A skipper positioned before the first interval: every level's
    /// `min_doc_id`/`max_doc_id` is `-1` and [`num_levels`](Self::num_levels)
    /// is 1, matching the instance initializer in Java's `getSkipper`.
    pub fn new(index: &'a DocValuesSkipIndex) -> Self {
        Self {
            index,
            cursor: 0,
            levels: 1,
            min_doc_id: [-1; SKIP_INDEX_MAX_LEVEL as usize],
            max_doc_id: [-1; SKIP_INDEX_MAX_LEVEL as usize],
            min_value: [0; SKIP_INDEX_MAX_LEVEL as usize],
            max_value: [0; SKIP_INDEX_MAX_LEVEL as usize],
            doc_count: [0; SKIP_INDEX_MAX_LEVEL as usize],
        }
    }

    /// Advances so that every level describes the first interval that can
    /// still contain a document `>= target`.
    ///
    /// `target` must be greater than the current `max_doc_id(0)` (Java
    /// documents the behaviour as undefined otherwise, and asserts it); this
    /// port simply keeps scanning forward from where it is, which for a
    /// backwards `target` leaves the skipper where it already was.
    ///
    /// Past the field's last document every level reports [`NO_MORE_DOCS`].
    /// ARITH: (whole body) `level` is bounded by `SKIP_INDEX_MAX_LEVEL` (4),
    /// so `SKIP_INDEX_LEVEL_SHIFT * level` is at most 9 and the jump is at
    /// most `1 << 9`. `cursor` only ever advances while
    /// `self.index.intervals.get(cursor)` is `Some`, so it stays within
    /// `intervals.len() + 512` -- a `Vec` length plus a constant. `levels` is
    /// compared against `SKIP_INDEX_MAX_LEVEL` before every increment.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn advance(&mut self, target: i32) {
        if target > self.index.max_doc_id {
            self.min_doc_id = [NO_MORE_DOCS; SKIP_INDEX_MAX_LEVEL as usize];
            self.max_doc_id = [NO_MORE_DOCS; SKIP_INDEX_MAX_LEVEL as usize];
            return;
        }
        while let Some(interval) = self.index.intervals.get(self.cursor) {
            let mut valid = true;
            for level in (0..interval.levels.len().min(SKIP_INDEX_MAX_LEVEL as usize)).rev() {
                let l = interval.levels[level];
                self.max_doc_id[level] = l.max_doc_id;
                if l.max_doc_id < target {
                    // Jump over the whole `8^level` subtree this level covers.
                    self.cursor += 1usize << (SKIP_INDEX_LEVEL_SHIFT * level as u32);
                    valid = false;
                    break;
                }
                self.min_doc_id[level] = l.min_doc_id;
                self.max_value[level] = l.max_value;
                self.min_value[level] = l.min_value;
                self.doc_count[level] = l.doc_count;
            }
            if valid {
                self.levels = interval.levels.len().min(SKIP_INDEX_MAX_LEVEL as usize);
                while self.levels < SKIP_INDEX_MAX_LEVEL as usize
                    && self.max_doc_id[self.levels] >= target
                {
                    self.levels += 1;
                }
                self.cursor += 1;
                return;
            }
        }
        // Only reachable on a truncated/corrupt skip index, where `max_doc_id`
        // promised an interval the `.dvs` slice does not actually hold. Report
        // exhaustion rather than spinning or panicking.
        self.min_doc_id = [NO_MORE_DOCS; SKIP_INDEX_MAX_LEVEL as usize];
        self.max_doc_id = [NO_MORE_DOCS; SKIP_INDEX_MAX_LEVEL as usize];
    }

    /// Port of `DocValuesSkipper.advance(long minValue, long maxValue)`:
    /// advances until every level's value range intersects
    /// `[min_value, max_value]`, using the coarsest non-intersecting level to
    /// skip as many documents at a time as possible. Leaves the skipper
    /// exhausted if no interval intersects.
    pub fn advance_range(&mut self, min_value: i64, max_value: i64) {
        if self.min_doc_id(0) == -1 {
            self.advance(0);
        }
        while self.min_doc_id(0) != NO_MORE_DOCS
            && (self.min_value(0) > max_value || self.max_value(0) < min_value)
        {
            let mut max_doc_id = self.max_doc_id(0);
            let mut next_level = 1;
            while next_level < self.num_levels()
                && (self.min_value(next_level) > max_value
                    || self.max_value(next_level) < min_value)
            {
                max_doc_id = self.max_doc_id(next_level);
                // ARITH: bounded by `num_levels() <= SKIP_INDEX_MAX_LEVEL`.
                #[allow(clippy::arithmetic_side_effects)]
                {
                    next_level += 1;
                }
            }
            if max_doc_id == NO_MORE_DOCS {
                self.advance(NO_MORE_DOCS);
                return;
            }
            // ARITH: `NO_MORE_DOCS` *is* `i32::MAX` and the branch above
            // returned for it, so `max_doc_id <= i32::MAX - 1` here.
            // `next_level` is compared against `num_levels()`, which
            // `advance` keeps at or below `SKIP_INDEX_MAX_LEVEL`.
            #[allow(clippy::arithmetic_side_effects)]
            self.advance(max_doc_id + 1);
        }
    }

    /// Number of levels describing the current position. Changes as the
    /// skipper moves between intervals.
    pub fn num_levels(&self) -> usize {
        self.levels
    }

    /// Inclusive minimum doc id of the current interval at `level`; `-1`
    /// before the first [`advance`](Self::advance), [`NO_MORE_DOCS`] once
    /// exhausted.
    pub fn min_doc_id(&self, level: usize) -> i32 {
        self.min_doc_id[level]
    }

    /// Inclusive maximum doc id of the current interval at `level`; `-1`
    /// before the first [`advance`](Self::advance), [`NO_MORE_DOCS`] once
    /// exhausted.
    pub fn max_doc_id(&self, level: usize) -> i32 {
        self.max_doc_id[level]
    }

    /// Inclusive lower bound on the values in the current interval at `level`.
    pub fn min_value(&self, level: usize) -> i64 {
        self.min_value[level]
    }

    /// Inclusive upper bound on the values in the current interval at `level`.
    pub fn max_value(&self, level: usize) -> i64 {
        self.max_value[level]
    }

    /// Number of documents with a value in the current interval at `level`.
    pub fn doc_count(&self, level: usize) -> i32 {
        self.doc_count[level]
    }

    /// Field-wide minimum value (`DocValuesSkipper.minValue()`).
    pub fn global_min_value(&self) -> i64 {
        self.index.min_value
    }

    /// Field-wide maximum value (`DocValuesSkipper.maxValue()`).
    pub fn global_max_value(&self) -> i64 {
        self.index.max_value
    }

    /// Field-wide count of documents that have a value
    /// (`DocValuesSkipper.docCount()`).
    pub fn global_doc_count(&self) -> i32 {
        self.index.doc_count
    }

    /// Field-wide maximum number of values on any one document
    /// (`DocValuesSkipper.maxValueCount()`), or `-1` when the `.dvm` was
    /// written by a codec too old to record it and it could not be inferred
    /// (see [`infer_max_value_counts`]).
    pub fn max_value_count(&self) -> i32 {
        self.index.max_value_count
    }
}

/// The `[offset, offset + length)` region of `.dvd` (or `.dvs`) a `.dvm`
/// entry names.
///
/// Both halves are independent `i64`s read straight off the metadata file,
/// so `offset + length` is exactly the shape `docs/arithmetic-gate.md`
/// exists for: it panics in a debug build and, in a release build, wraps to
/// a *plausible in-range* offset that slices valid-looking doc values out of
/// the wrong part of the file -- the silent-wrong-answer half of the class.
/// Folding the
/// widening, the add and the bounds check into one place also means there is
/// one parser of this shape rather than eight.
fn region(data: &[u8], offset: i64, length: i64) -> Result<&[u8]> {
    offset
        .checked_add(length)
        .and_then(|end| Some(usize::try_from(offset).ok()?..usize::try_from(end).ok()?))
        .and_then(|r| data.get(r))
        .ok_or_else(|| lucene_store::Error::Eof { offset: 0 }.into())
}

/// [`region`] for a `[start, end)` pair rather than an offset and a length.
fn slice_at(data: &[u8], start: i64, end: i64) -> Result<&[u8]> {
    usize::try_from(start)
        .ok()
        .zip(usize::try_from(end).ok())
        .and_then(|(start, end)| data.get(start..end))
        .ok_or_else(|| lucene_store::Error::Eof { offset: 0 }.into())
}

/// Reads the numeric doc-values value for `doc`, handling all three
/// docs-with-a-value shapes (empty/dense/sparse) and all three per-value
/// encodings (constant/table/GCD-delta). `data` is the whole `.dvd` file's
/// bytes. `Ok(None)` means `doc` legitimately has no value.
pub fn numeric_value(data: &[u8], entry: &NumericEntry, doc: i32) -> Result<Option<i64>> {
    if doc < 0 {
        return Err(Error::DocOutOfRange(doc, entry.num_values));
    }
    if entry.is_empty_field() {
        return Ok(None);
    }
    if entry.is_dense() {
        if doc as i64 >= entry.num_values {
            return Err(Error::DocOutOfRange(doc, entry.num_values));
        }
        return Ok(Some(decode_value(data, entry, doc as i64)?));
    }

    let region = entry.docs_region(data)?;
    // One forward-only pass over the block headers rather than decoding the
    // whole region: `DisiCursor` reads one 4-byte header per 65,536 documents
    // and then resolves the ordinal inside a single block. It allocates
    // nothing. A caller making more than one lookup should hold a
    // `NumericReader`, which keeps the cursor between calls.
    match indexed_disi::DisiCursor::new(
        region,
        entry.dense_rank_power,
        entry.jump_table_entry_count,
    )
    .advance_exact(doc)?
    {
        Some(ordinal) => Ok(Some(decode_value(data, entry, ordinal as i64)?)),
        None => Ok(None),
    }
}

/// A cursor over one NUMERIC field's values, holding the state Lucene's
/// per-field readers hold.
///
/// [`numeric_value`] is a free function and so re-derives everything on every
/// call: for a sparse field it walks the `IndexedDISI` block headers from the
/// start of the region, and for a varying-bits-per-value field it re-reads the
/// jump table and the block header even when the previous call was in the same
/// block. Lucene's `Lucene90DocValuesProducer` does neither: `getNumeric`
/// composes an `IndexedDISI` **cursor** with a `LongValues`
/// (`SparseNumericDocValues`), and `VaryingBPVReader.getLongValue` opens with
/// `if (this.block != block)`.
///
/// Anything reading more than one value from a field -- a sort, a facet count,
/// a range scan -- should hold one of these. `numeric_value` stays for
/// single-lookup callers and for tests.
///
/// **Memory is O(1) in the field's cardinality**, which is the whole point of
/// composing a cursor rather than decoding the sparse field's `IndexedDISI`
/// region into a doc-id array.
///
/// Both caches are transparent: the values returned are identical to
/// [`numeric_value`]'s, which is asserted document-for-document in this
/// module's tests.
#[derive(Debug, Clone)]
pub struct NumericReader<'a> {
    data: &'a [u8],
    entry: &'a NumericEntry,
    /// Forward-only cursor over the docs-with-field region of a sparse field;
    /// `None` for a dense or empty one, and `None` when the region's byte range
    /// is not inside `data` (a lookup then falls through to [`numeric_value`],
    /// which raises the error).
    docs: Option<indexed_disi::DisiCursor<'a>>,
    /// The currently decoded varying-bits-per-value block, if any:
    /// `(block index, bits per value, delta, values slice range)`.
    block: Option<VaryingBlock>,
    /// The last decoded adaptive delta block, if any.
    adaptive_block: Option<Box<AdaptiveBlock>>,
    /// A dense single-width field, resolved once -- see [`FastDense`].
    fast: Option<FastDense<'a>>,
    /// A sparse single-width field's values, by ordinal.
    sparse_fast: Option<FastDense<'a>>,
}

/// A dense numeric field with one bit width for every value, resolved once:
/// `Lucene90DocValuesProducer`'s `getNumeric` hands back a
/// `DirectReader.getInstance(slice, bitsPerValue)` specialized to the width,
/// and a per-document read is then one `readLong`, a shift and a mask. This is
/// that, for [`NumericReader`]: the values region, width, mask and the
/// GCD/table transform are looked up when the reader is built instead of on
/// every call.
#[derive(Debug, Clone, Copy)]
enum FastDense<'a> {
    /// `bitsPerValue == 0`: every document has `min_value`.
    Constant { num_values: i64, value: i64 },
    Packed {
        /// One unaligned load per value, bounds-checked once at construction;
        /// its limit is lowered to `num_values` (`PackedLongs::limited_to`), so a
        /// lookup is one unsigned compare (a negative doc wraps above it).
        packed: lucene_util::packed_longs::PackedLongs<'a>,
        num_values: i64,
        gcd: i64,
        min: i64,
        table: Option<&'a [i64]>,
    },
}

impl<'a> FastDense<'a> {
    fn new(data: &'a [u8], entry: &'a NumericEntry) -> Option<Self> {
        if !entry.is_dense() {
            return None;
        }
        Self::values(data, entry)
    }

    /// The same reader over the values array alone, indexed by value
    /// ordinal: what a sparse field's reads use once its `IndexedDISI` has
    /// turned the document into an ordinal.
    fn values(data: &'a [u8], entry: &'a NumericEntry) -> Option<Self> {
        if entry.is_empty_field() || entry.block_shift.is_some() || entry.adaptive.is_some() {
            return None;
        }
        if entry.bits_per_value == 0 {
            return Some(FastDense::Constant {
                num_values: entry.num_values,
                value: entry.min_value,
            });
        }
        if !direct_reader::is_supported_bits(entry.bits_per_value) {
            return None;
        }
        let packed = lucene_util::packed_longs::PackedLongs::new(
            region(data, entry.values_offset, entry.values_length).ok()?,
            entry.bits_per_value as u32,
        )?;
        // The document bound folded into the packed reader's own fast
        // limit: a lookup is then one compare, not two.
        let packed = packed.limited_to(u64::try_from(entry.num_values).unwrap_or(0));
        Some(FastDense::Packed {
            packed,
            num_values: entry.num_values,
            gcd: entry.gcd,
            min: entry.min_value,
            table: entry.table.as_deref(),
        })
    }

    /// `doc`'s value, or `None` when this fast path cannot answer (an
    /// out-of-range doc, the last few values that `DirectWriter`'s padding
    /// does not cover with a whole 8-byte window, a table index past the
    /// table) -- the general path then answers, including with the error.
    #[inline]
    fn get(&self, doc: i32) -> Option<i64> {
        match *self {
            FastDense::Constant { num_values, value } => {
                (doc >= 0 && (doc as i64) < num_values).then_some(value)
            }
            FastDense::Packed {
                packed,
                gcd,
                min,
                table,
                ..
            } => {
                // A negative doc wraps above every limit; `packed` already
                // stops at `num_values` (see `new`).
                let raw = packed.get(doc as u32 as u64)?;
                Some(match table {
                    Some(t) => *t.get(raw as usize)?,
                    None => gcd.wrapping_mul(raw as i64).wrapping_add(min),
                })
            }
        }
    }

    /// [`Self::get`] for the run of documents starting at `start`: fills a
    /// prefix of `out` and returns its length, stopping wherever `get` would
    /// have answered `None` (0 when it already would for `start`).
    fn fill(&self, start: i32, out: &mut [i64]) -> usize {
        if start < 0 {
            return 0;
        }
        match *self {
            FastDense::Constant { num_values, value } => {
                let room = usize::try_from(num_values.saturating_sub(start as i64)).unwrap_or(0);
                let n = out.len().min(room);
                out[..n].fill(value);
                n
            }
            FastDense::Packed {
                packed,
                num_values,
                gcd,
                min,
                table,
                ..
            } => {
                let room = usize::try_from(num_values.saturating_sub(start as i64)).unwrap_or(0);
                let len = out.len().min(room);
                let n = packed.decode_range(start as u64, &mut out[..len]);
                match table {
                    Some(t) => {
                        for (i, v) in out[..n].iter_mut().enumerate() {
                            match t.get(*v as usize) {
                                Some(&mapped) => *v = mapped,
                                // A table index past the table: stop here and
                                // let the general path report it.
                                None => return i,
                            }
                        }
                    }
                    None => {
                        for v in &mut out[..n] {
                            *v = gcd.wrapping_mul(*v).wrapping_add(min);
                        }
                    }
                }
                n
            }
        }
    }
}
/// One decoded varying-bits-per-value block header, cached by
/// [`NumericReader`] so consecutive values in the same block cost one
/// `direct_reader::get` and nothing else.
#[derive(Debug, Clone, Copy)]
struct VaryingBlock {
    index: i64,
    bits_per_value: u8,
    delta: i64,
    values_start: usize,
    values_end: usize,
}

impl<'a> NumericReader<'a> {
    /// Opens a cursor over `entry`'s values in `data` (the whole `.dvd` file).
    ///
    /// Allocates nothing and reads nothing: a sparse field's docs-with-field
    /// region is walked lazily, one block header at a time, as lookups reach
    /// into it.
    pub fn new(data: &'a [u8], entry: &'a NumericEntry) -> Self {
        let docs = if entry.is_empty_field() || entry.is_dense() {
            None
        } else {
            entry.docs_region(data).ok().map(|region| {
                indexed_disi::DisiCursor::new(
                    region,
                    entry.dense_rank_power,
                    entry.jump_table_entry_count,
                )
            })
        };
        Self {
            data,
            entry,
            docs,
            block: None,
            adaptive_block: None,
            fast: FastDense::new(data, entry),
            sparse_fast: if entry.is_dense() {
                None
            } else {
                FastDense::values(data, entry)
            },
        }
    }

    /// The [`crate::adaptive_zones`] statistics of documents
    /// `start..start + ZONE_SIZE` (fewer in the last zone), and how many
    /// documents that zone holds: `Some` only for a dense field written with
    /// zones and a `start` on a zone boundary.
    ///
    /// # Errors
    ///
    /// A block-address descriptor or zone section that runs past `.dvd`.
    pub fn window_stats(
        &self,
        start: i32,
    ) -> Result<Option<(crate::adaptive_zones::ZoneStats, usize)>> {
        use crate::adaptive_zones::ZONE_SHIFT;
        let Some(adaptive) = &self.entry.adaptive else {
            return Ok(None);
        };
        if !adaptive.zones || !self.entry.is_dense() {
            return Ok(None);
        }
        let Ok(start) = usize::try_from(start) else {
            return Ok(None);
        };
        if start.trailing_zeros() < ZONE_SHIFT && start != 0 {
            return Ok(None);
        }
        let addresses = adaptive.addresses(self.data, self.entry.num_values)?;
        let Some(zones) = &addresses.zones else {
            return Ok(None);
        };
        let zone = start >> ZONE_SHIFT;
        Ok(zones
            .get(self.data, zone)
            .map(|stats| (stats, zones.count(zone))))
    }

    /// Whether [`Self::window_stats`] can answer: a dense field written with
    /// [`crate::adaptive_zones`] statistics.
    pub fn has_window_stats(&self) -> bool {
        self.entry.is_dense() && self.entry.adaptive.as_ref().is_some_and(|a| a.zones)
    }

    /// Whether [`Self::dense_value`] answers every in-range document.
    pub fn has_dense_fast(&self) -> bool {
        self.fast.is_some()
    }

    /// [`Self::value`] when its dense single-width fast path can answer:
    /// `doc`'s value, or `None` for "ask [`Self::value`]" (never "no value").
    #[inline]
    pub fn dense_value(&self, doc: i32) -> Option<i64> {
        self.fast.as_ref().and_then(|f| f.get(doc))
    }

    /// [`Self::value`] for a dense varying-bits-per-value field when `doc`
    /// falls in the block the last read decoded: one packed read, inline.
    /// `None` means "ask [`Self::value`]" (never "no value"), as for
    /// [`Self::dense_value`].
    #[inline]
    pub fn block_value(&self, doc: i32) -> Option<i64> {
        let b = self.block.as_ref()?;
        let shift = self.entry.block_shift?;
        if !self.entry.is_dense() || self.entry.adaptive.is_some() {
            return None;
        }
        let ordinal = i64::from(doc);
        if ordinal < 0 || ordinal >= self.entry.num_values || ordinal >> shift != b.index {
            return None;
        }
        if b.bits_per_value == 0 {
            return Some(b.delta);
        }
        let values = self.data.get(b.values_start..b.values_end)?;
        let raw =
            direct_reader::get_fast(values, b.bits_per_value, ordinal & low_bits_mask(shift))?;
        Some(self.entry.gcd.wrapping_mul(raw).wrapping_add(b.delta))
    }

    /// This document's value, or `None` when it legitimately has none.
    /// Identical to `numeric_value(data, entry, doc)`, just without the
    /// per-call rederivation.
    ///
    /// Documents may be asked for in any order. Ascending order -- what a sort,
    /// a facet count or a range scan does, and the only order Lucene's
    /// `SparseNumericDocValues` supports at all -- walks the docs-with-field
    /// region once for the whole scan. Going backwards rewinds the cursor
    /// (`DisiCursor::reset`), which costs a re-walk of the block headers and
    /// still allocates nothing.
    ///
    /// The dense single-width lookup is this inlined function; every other
    /// shape is [`Self::value_slow`], kept out of line so this stays small
    /// enough to inline into a caller's loop.
    /// The field's entry: `numValues` is the iterator's `cost()`.
    pub fn entry(&self) -> &'a NumericEntry {
        self.entry
    }

    #[inline(always)]
    pub fn value(&mut self, doc: i32) -> Result<Option<i64>> {
        if let Some(v) = self.fast.as_ref().and_then(|f| f.get(doc)) {
            return Ok(Some(v));
        }
        // A sparse field read forward inside the current `IndexedDISI`
        // block: no header to read, only the ordinal's value to decode.
        if let Some(cursor) = self.docs.as_mut() {
            match cursor.advance_exact_in_block(doc) {
                Some(None) => return Ok(None),
                Some(Some(ordinal)) => {
                    let fast = self.sparse_fast.as_ref();
                    if let Some(v) = fast.and_then(|f| f.get(i32::try_from(ordinal).ok()?)) {
                        return Ok(Some(v));
                    }
                    return self.decode_at(ordinal as i64).map(Some);
                }
                None => {}
            }
        }
        self.value_slow(doc)
    }

    /// The value of the `index`-th document that has one (its position
    /// among them, which a caller that already found the document's rank in
    /// the field's `IndexedDISI` knows): no second walk of the
    /// `IndexedDISI`.
    pub fn value_at_index(&mut self, index: i64) -> Result<i64> {
        if let Some(v) = self
            .sparse_fast
            .as_ref()
            .and_then(|f| f.get(i32::try_from(index).ok()?))
        {
            return Ok(v);
        }
        if index < 0 || index >= self.entry.num_values {
            return Err(Error::DocOutOfRange(
                i32::try_from(index).unwrap_or(i32::MAX),
                self.entry.num_values,
            ));
        }
        self.decode_at(index)
    }

    /// The value at `ordinal` in the values array.
    fn decode_at(&mut self, ordinal: i64) -> Result<i64> {
        if let Some(adaptive) = &self.entry.adaptive {
            return adaptive_value(
                self.data,
                self.entry,
                adaptive,
                ordinal,
                &mut self.adaptive_block,
            );
        }
        match self.entry.block_shift {
            Some(shift) => self.decode_varying(shift, ordinal),
            None => decode_value(self.data, self.entry, ordinal),
        }
    }

    /// [`Self::value`] for every shape but the dense single-width one.
    #[inline(never)]
    fn value_slow(&mut self, doc: i32) -> Result<Option<i64>> {
        if doc < 0 {
            return Err(Error::DocOutOfRange(doc, self.entry.num_values));
        }
        if self.entry.is_empty_field() {
            return Ok(None);
        }
        let ordinal = if self.entry.is_dense() {
            if doc as i64 >= self.entry.num_values {
                return Err(Error::DocOutOfRange(doc, self.entry.num_values));
            }
            doc as i64
        } else {
            let Some(cursor) = self.docs.as_mut() else {
                // Unreadable region: let the general path raise the error.
                return numeric_value(self.data, self.entry, doc);
            };
            if doc < cursor.doc_id() {
                cursor.reset();
            }
            match cursor.advance_exact(doc)? {
                Some(ordinal) => {
                    let fast = self.sparse_fast.as_ref();
                    if let Some(v) = fast.and_then(|f| f.get(i32::try_from(ordinal).ok()?)) {
                        return Ok(Some(v));
                    }
                    ordinal as i64
                }
                None => return Ok(None),
            }
        };
        self.decode_at(ordinal).map(Some)
    }

    /// Calls `f(doc, value)` for every document in `start..end` that has a
    /// value, in ascending order -- [`Self::value`] over a range, which is how
    /// every sequential consumer (a range filter, a sort, a facet count) reads
    /// a column.
    ///
    /// On a dense single-width field (the shape [`FastDense`] resolves) values
    /// are decoded a chunk at a time by [`PackedLongs::decode_range`] -- one
    /// range check per chunk and a branch-free loop -- instead of one checked,
    /// `Result`-returning call per document. That is this port's answer to
    /// what Lucene gets from the JIT specializing and inlining
    /// `DirectReader`'s per-width `get`; per document, the chunked loop costs a
    /// fraction of either. Everything else goes through [`Self::value`].
    ///
    /// [`PackedLongs::decode_range`]: lucene_util::packed_longs::PackedLongs::decode_range
    pub fn for_each_value(
        &mut self,
        start: i32,
        end: i32,
        mut f: impl FnMut(i32, i64),
    ) -> Result<()> {
        const CHUNK: usize = 256;
        let mut buf = [0i64; CHUNK];
        let mut doc = start.max(0);
        while doc < end {
            // ARITH: `doc < end`, so the difference is positive and fits.
            #[allow(clippy::arithmetic_side_effects)]
            let want = ((end - doc) as usize).min(CHUNK);
            let n = match (&self.fast, &self.entry.adaptive) {
                (Some(fast), _) => fast.fill(doc, &mut buf[..want]),
                (None, Some(adaptive)) if self.entry.is_dense() => {
                    let have =
                        usize::try_from(self.entry.num_values.saturating_sub(i64::from(doc)))
                            .unwrap_or(0)
                            .min(want);
                    adaptive_run(
                        self.data,
                        self.entry,
                        adaptive,
                        i64::from(doc),
                        &mut buf[..have],
                        &mut self.adaptive_block,
                    )?;
                    have
                }
                (None, None) if self.entry.is_dense() && self.entry.block_shift.is_some() => {
                    let have =
                        usize::try_from(self.entry.num_values.saturating_sub(i64::from(doc)))
                            .unwrap_or(0)
                            .min(want);
                    let shift = self.entry.block_shift.unwrap_or_default();
                    self.decode_varying_run(shift, i64::from(doc), &mut buf[..have])?;
                    have
                }
                _ => 0,
            };
            if n == 0 {
                if let Some(v) = self.value(doc)? {
                    f(doc, v);
                }
                // ARITH: `doc < end <= i32::MAX`.
                #[allow(clippy::arithmetic_side_effects)]
                {
                    doc += 1;
                }
                continue;
            }
            // ARITH: `i < n <= end - doc`, so `doc + i < end`, and `doc + n <=
            // end`.
            #[allow(clippy::arithmetic_side_effects)]
            {
                for (i, &v) in buf[..n].iter().enumerate() {
                    f(doc + i as i32, v);
                }
                doc += n as i32;
            }
        }
        Ok(())
    }

    /// The values of the documents `start..start + values.len()` in one
    /// forward pass: `values[i]` is document `start + i`'s value when bit `i`
    /// of `present` is set (`present` must hold at least
    /// `values.len().div_ceil(64)` words; the others are left as they are).
    /// Each is what [`Self::value`] answers, which the tests check document
    /// for document.
    ///
    /// For a window-at-a-time consumer (an aggregation over every document of
    /// a segment): a dense single-width column is decoded a chunk at a time
    /// ([`FastDense::fill`]), and a sparse one walks its `IndexedDISI` once
    /// and decodes the window's values -- consecutive ordinals -- in one
    /// batch, instead of a checked lookup through every layer per document.
    pub fn fill_window(
        &mut self,
        start: i32,
        values: &mut [i64],
        present: &mut [u64],
    ) -> Result<()> {
        let len = values.len();
        let words = len.div_ceil(64);
        let Some(present) = present.get_mut(..words) else {
            return Err(Error::DocOutOfRange(start, len as i64));
        };
        present.fill(0);
        if start < 0 || i32::try_from(len).is_err() || start.checked_add(len as i32).is_none() {
            return Err(Error::DocOutOfRange(start, len as i64));
        }
        // ARITH: `start + len` fits an `i32` (checked above), so every
        // `start + i` below does, and `i >> 6 < words`.
        #[allow(clippy::arithmetic_side_effects)]
        {
            let mut done = 0usize;
            if let Some(fast) = self.fast {
                while done < len {
                    let n = fast.fill(start + done as i32, &mut values[done..]);
                    if n == 0 {
                        break;
                    }
                    done += n;
                }
                set_first_bits(present, done);
            } else if let (true, Some(adaptive)) = (self.entry.is_dense(), &self.entry.adaptive) {
                let have = usize::try_from(self.entry.num_values.saturating_sub(i64::from(start)))
                    .unwrap_or(0)
                    .min(len);
                adaptive_run(
                    self.data,
                    self.entry,
                    adaptive,
                    i64::from(start),
                    &mut values[..have],
                    &mut self.adaptive_block,
                )?;
                set_first_bits(present, have);
                return Ok(());
            } else if let (true, Some(shift)) = (self.entry.is_dense(), self.entry.block_shift) {
                // Dense, varying bits per value: the documents are the
                // ordinals.
                let have = usize::try_from(self.entry.num_values.saturating_sub(i64::from(start)))
                    .unwrap_or(0)
                    .min(len);
                self.decode_varying_run(shift, i64::from(start), &mut values[..have])?;
                set_first_bits(present, have);
                return Ok(());
            } else if let (true, Some(cursor)) = (
                self.sparse_fast.is_some()
                    || self.entry.block_shift.is_some()
                    || self.entry.adaptive.is_some(),
                self.docs.as_mut(),
            ) {
                let sparse = self.sparse_fast;
                if start < cursor.doc_id() {
                    cursor.reset();
                }
                // Walk the window: which documents have a value, and the
                // first one's ordinal; the rest follow it one by one. A whole
                // block's worth of words at once where the encoding allows.
                let mut first = None;
                let mut count = 0usize;
                let whole = if len.is_multiple_of(64) {
                    cursor
                        .window_bits(start, &mut present[..len / 64])?
                        .and_then(|o| usize::try_from(o).ok())
                } else {
                    None
                };
                match whole {
                    Some(o) => {
                        count = present.iter().map(|w| w.count_ones() as usize).sum();
                        first = (count > 0).then_some(o);
                    }
                    // Document by document, from a clean slate.
                    None => present.fill(0),
                }
                for i in (0..len).filter(|_| whole.is_none()) {
                    let doc = start + i as i32;
                    let ord = match cursor.advance_exact_in_block(doc) {
                        Some(o) => o,
                        None => cursor.advance_exact(doc)?,
                    };
                    if let Some(o) = ord {
                        if first.is_none() {
                            first = Some(o);
                        }
                        present[i >> 6] |= 1 << (i & 63);
                        count += 1;
                    }
                }
                if let Some(first) = first {
                    // The values, packed at the front, then moved to their
                    // documents from the last down (each moves up or stays).
                    let mut n = 0;
                    if let Some(adaptive) = &self.entry.adaptive {
                        adaptive_run(
                            self.data,
                            self.entry,
                            adaptive,
                            first as i64,
                            &mut values[..count],
                            &mut self.adaptive_block,
                        )?;
                        n = count;
                    } else if let (None, Some(shift)) = (sparse, self.entry.block_shift) {
                        self.decode_varying_run(shift, first as i64, &mut values[..count])?;
                        n = count;
                    }
                    while n < count {
                        let got = match (sparse, i32::try_from(first + n)) {
                            (Some(sparse), Ok(o)) => sparse.fill(o, &mut values[n..count]),
                            _ => 0,
                        };
                        if got == 0 {
                            values[n] = self.decode_at((first + n) as i64)?;
                            n += 1;
                        } else {
                            n += got;
                        }
                    }
                    // Slot by slot from the top, without a branch per value:
                    // each slot takes the next value down, which only a set
                    // bit consumes. In place, since a slot is never below the
                    // values still to move (`k <= slot + 1` throughout), and
                    // a slot without a value is left holding a copy that
                    // `present` says to ignore.
                    let mut k = count;
                    for w in (0..words).rev() {
                        let bits = present[w];
                        if bits == 0 {
                            continue;
                        }
                        let top = (64 - bits.leading_zeros() as usize).min(len - w * 64);
                        for b in (0..top).rev() {
                            let slot = w * 64 + b;
                            values[slot] = values[k.saturating_sub(1)];
                            k -= (bits >> b & 1) as usize;
                        }
                    }
                }
                return Ok(());
            }
            for i in done..len {
                if let Some(v) = self.value(start + i as i32)? {
                    values[i] = v;
                    present[i >> 6] |= 1 << (i & 63);
                }
            }
        }
        Ok(())
    }

    /// [`decode_value_varying_bpv`] with the block header kept between calls --
    /// `VaryingBPVReader.getLongValue`'s `if (this.block != block)`.
    fn decode_varying(&mut self, shift: u32, ordinal: i64) -> Result<i64> {
        decode_varying_cached(self.data, self.entry, shift, ordinal, &mut self.block)
    }

    /// The values at ordinals `first..first + out.len()` of a
    /// varying-bits-per-value field, each what [`Self::decode_varying`]
    /// answers: a block's header read once and its packed run decoded in one
    /// pass (`PackedLongs::decode_range`), the rest of a run the packed reader
    /// cannot reach read value by value.
    fn decode_varying_run(&mut self, shift: u32, first: i64, out: &mut [i64]) -> Result<()> {
        let mask = low_bits_mask(shift);
        let mut done = 0usize;
        // ARITH: `done < out.len()`, `n <= out.len() - done`, `first + done`
        // is an ordinal below the field's value count (an `i64` read off
        // disk, checked by the block lookup), and `in_block >= 1`.
        #[allow(clippy::arithmetic_side_effects)]
        while done < out.len() {
            let ordinal = first + done as i64;
            let within = ordinal & mask;
            let in_block = usize::try_from(mask - within + 1).unwrap_or(usize::MAX);
            let n = (out.len() - done).min(in_block);
            let index = ordinal >> shift;
            let block = match self.block {
                Some(b) if b.index == index => b,
                _ => {
                    let b = read_varying_block(self.data, self.entry, index)?;
                    self.block = Some(b);
                    b
                }
            };
            let run = &mut out[done..done + n];
            if block.bits_per_value == 0 {
                run.fill(block.delta);
            } else {
                let Some(values) = self.data.get(block.values_start..block.values_end) else {
                    return Err(lucene_store::Error::Eof { offset: 0 }.into());
                };
                let decoded = if direct_reader::is_supported_bits(block.bits_per_value) {
                    lucene_util::packed_longs::PackedLongs::new(
                        values,
                        u32::from(block.bits_per_value),
                    )
                    .map_or(0, |p| p.decode_range(within as u64, run))
                } else {
                    0
                };
                for (k, v) in run.iter_mut().enumerate().skip(decoded) {
                    *v = direct_reader::get(values, block.bits_per_value, within + k as i64)?;
                }
                let (gcd, delta) = (self.entry.gcd, block.delta);
                for v in run.iter_mut() {
                    *v = gcd.wrapping_mul(*v).wrapping_add(delta);
                }
            }
            done += n;
        }
        Ok(())
    }
}

/// [`decode_value_varying_bpv`] with the block header kept in `cache`
/// between calls -- `VaryingBPVReader.getLongValue`'s
/// `if (this.block != block)`.
fn decode_varying_cached(
    data: &[u8],
    entry: &NumericEntry,
    shift: u32,
    ordinal: i64,
    cache: &mut Option<VaryingBlock>,
) -> Result<i64> {
    let index = ordinal >> shift;
    let block = match *cache {
        Some(b) if b.index == index => b,
        _ => {
            let b = read_varying_block(data, entry, index)?;
            *cache = Some(b);
            b
        }
    };
    if block.bits_per_value == 0 {
        return Ok(block.delta);
    }
    // `let ... else`, not `ok_or`: the error is built only when it is
    // raised, not on every value (it was, per document, in an
    // aggregation's profile).
    let Some(values) = data.get(block.values_start..block.values_end) else {
        return Err(lucene_store::Error::Eof { offset: 0 }.into());
    };
    let mask = low_bits_mask(shift);
    let raw = direct_reader::get(values, block.bits_per_value, ordinal & mask)?;
    Ok(entry.gcd.wrapping_mul(raw).wrapping_add(block.delta))
}

/// [`decode_value`], with a varying-bits-per-value field's block header kept
/// in `cache` so consecutive ordinals in one block skip the jump-table read.
#[inline]
fn decode_value_cached(
    data: &[u8],
    entry: &NumericEntry,
    ordinal: i64,
    cache: &mut Option<VaryingBlock>,
    adaptive_cache: &mut Option<Box<AdaptiveBlock>>,
) -> Result<i64> {
    if let Some(adaptive) = &entry.adaptive {
        return adaptive_value(data, entry, adaptive, ordinal, adaptive_cache);
    }
    match entry.block_shift {
        Some(shift) => decode_varying_cached(data, entry, shift, ordinal, cache),
        None => decode_value(data, entry, ordinal),
    }
}

/// `(1 << shift) - 1` -- the low-`shift`-bits mask a varying-bits-per-value
/// field uses to turn an ordinal into an index within its block.
///
/// Computed in `u64` rather than `i64` on purpose. `read_numeric_entry`
/// accepts `shift` in `0..=63` (Java's `blockShift` is 14, but the field is
/// `-2 - tableSize` off `.dvm`, so a `tableSize` of `-65` reaches 63), and
/// `1i64 << 63` is `i64::MIN`, whose `- 1` **overflows** -- a debug-build
/// panic at the one value the range check lets through. In `u64` the
/// subtraction is exact for every accepted `shift` and the reinterpretation
/// back to `i64` is the bit pattern Java's `long` arithmetic produces.
fn low_bits_mask(shift: u32) -> i64 {
    debug_assert!(
        shift <= 63,
        "read_numeric_entry bounds blockShift to 0..=63"
    );
    // ARITH: `shift <= 63` (rejected in `read_numeric_entry`), so
    // `1u64 << shift` is at most `1 << 63` and the `- 1` on an unsigned value
    // that is at least 1 cannot underflow.
    #[allow(clippy::arithmetic_side_effects)]
    let mask = (1u64 << shift) - 1;
    mask as i64
}

/// Reads one varying-bits-per-value block's header via the per-field jump
/// table. Shared by [`decode_value_varying_bpv`] and [`NumericReader`], so
/// there is one parser of this header rather than two to keep in step.
fn read_varying_block(data: &[u8], entry: &NumericEntry, index: i64) -> Result<VaryingBlock> {
    let jump_table_pos = entry
        .value_jump_table_offset
        .checked_add(
            index
                .checked_mul(8)
                .ok_or(lucene_store::Error::Eof { offset: 0 })?,
        )
        .ok_or(lucene_store::Error::Eof { offset: 0 })?;
    let jump_table_pos =
        usize::try_from(jump_table_pos).map_err(|_| lucene_store::Error::Eof { offset: 0 })?;
    let mut input = SliceInput::new(data);
    input.seek(jump_table_pos)?;
    let block_start =
        usize::try_from(input.read_i64()?).map_err(|_| lucene_store::Error::Eof { offset: 0 })?;

    input.seek(block_start)?;
    let bits_per_value = input.read_byte()?;
    let delta = input.read_i64()?;
    if bits_per_value == 0 {
        return Ok(VaryingBlock {
            index,
            bits_per_value,
            delta,
            values_start: 0,
            values_end: 0,
        });
    }
    let length = input.read_i32()?;
    let length = usize::try_from(length).map_err(|_| lucene_store::Error::Eof { offset: 0 })?;
    let values_start = input.position();
    let values_end = values_start
        .checked_add(length)
        .ok_or(lucene_store::Error::Eof { offset: 0 })?;
    Ok(VaryingBlock {
        index,
        bits_per_value,
        delta,
        values_start,
        values_end,
    })
}

/// One decoded [`crate::adaptive_values`] block, kept by a reader so the
/// other values of the block cost nothing, and the block it read last.
#[derive(Debug, Clone)]
struct AdaptiveBlock {
    index: i64,
    last: i64,
    values: [i64; crate::adaptive_values::BLOCK_SIZE],
}

/// Block `block` of an adaptive value array: its bytes (to the end of the
/// region, padding included) and how many values it holds.
fn adaptive_block<'d>(
    data: &'d [u8],
    entry: &NumericEntry,
    adaptive: &AdaptiveEntry,
    block: i64,
) -> Result<(&'d [u8], usize)> {
    use crate::adaptive_values::{BLOCK_SHIFT, BLOCK_SIZE};
    let first = block
        .checked_shl(BLOCK_SHIFT)
        .filter(|&f| f >= 0 && f < entry.num_values)
        .ok_or(Error::DocOutOfRange(i32::MAX, entry.num_values))?;
    let len = usize::try_from(entry.num_values.wrapping_sub(first))
        .unwrap_or(BLOCK_SIZE)
        .min(BLOCK_SIZE);
    let values = region(data, entry.values_offset, entry.values_length)?;
    let resolved = adaptive.addresses(data, entry.num_values)?;
    let addresses = region(data, resolved.offset, resolved.length)?;
    let start = direct_monotonic::get(addresses, &resolved.meta, block)?;
    let Some(bytes) = usize::try_from(start).ok().and_then(|s| values.get(s..)) else {
        return Err(lucene_store::Error::Eof { offset: 0 }.into());
    };
    Ok((bytes, len))
}

/// The value at `ordinal` of an adaptive value array. A reader walking
/// forward (its first read, a second read in one block, or a read in the
/// block after the last) decodes the block whole into `cache` and answers
/// from it while it stays there; a jump elsewhere answers the one value
/// directly.
fn adaptive_value(
    data: &[u8],
    entry: &NumericEntry,
    adaptive: &AdaptiveEntry,
    ordinal: i64,
    cache: &mut Option<Box<AdaptiveBlock>>,
) -> Result<i64> {
    use crate::adaptive_values::{self as av, BLOCK_SHIFT};
    if ordinal < 0 || ordinal >= entry.num_values {
        return Err(Error::DocOutOfRange(
            i32::try_from(ordinal).unwrap_or(i32::MAX),
            entry.num_values,
        ));
    }
    let block = ordinal >> BLOCK_SHIFT;
    let within = (ordinal & low_bits_mask(BLOCK_SHIFT)) as usize;
    // A second read in the block last read, or a read in the block after it,
    // is a reader walking forward.
    let sequential = match cache.as_deref_mut() {
        Some(c) if c.index == block => return Ok(c.values[within]),
        Some(c) => {
            let forward = c.last == block || c.last.wrapping_add(1) == block;
            c.last = block;
            forward
        }
        None => true,
    };
    let (bytes, len) = adaptive_block(data, entry, adaptive, block)?;
    if within >= len {
        return Err(Error::DocOutOfRange(
            i32::try_from(ordinal).unwrap_or(i32::MAX),
            entry.num_values,
        ));
    }
    if !sequential {
        return Ok(av::get_one(bytes, within)?);
    }
    let c = cache.get_or_insert_with(|| {
        Box::new(AdaptiveBlock {
            index: -1,
            last: -1,
            values: [0; av::BLOCK_SIZE],
        })
    });
    c.index = -1;
    av::decode_block(bytes, &mut c.values[..len])?;
    c.index = block;
    c.last = block;
    Ok(c.values[within])
}

/// The values at ordinals `first..first + out.len()` of an adaptive value
/// array, a block at a time.
fn adaptive_run(
    data: &[u8],
    entry: &NumericEntry,
    adaptive: &AdaptiveEntry,
    first: i64,
    out: &mut [i64],
    cache: &mut Option<Box<AdaptiveBlock>>,
) -> Result<()> {
    use crate::adaptive_values::{self as av, BLOCK_SHIFT};
    let mut done = 0usize;
    while done < out.len() {
        let ordinal = first.saturating_add(done as i64);
        let block = ordinal >> BLOCK_SHIFT;
        let within = (ordinal & low_bits_mask(BLOCK_SHIFT)) as usize;
        let cached = cache.as_deref().filter(|c| c.index == block);
        let (src, len): (&[i64], usize) = match cached {
            Some(c) => {
                let len = usize::try_from(entry.num_values.saturating_sub(block << BLOCK_SHIFT))
                    .unwrap_or(0)
                    .min(av::BLOCK_SIZE);
                (&c.values[..], len)
            }
            None => {
                let (bytes, len) = adaptive_block(data, entry, adaptive, block)?;
                let rest = out.len().saturating_sub(done);
                if within == 0 && rest >= len {
                    // The whole block lands in `out`: decode it in place.
                    let end = done.saturating_add(len);
                    av::decode_block(bytes, &mut out[done..end])?;
                    done = end;
                    continue;
                }
                let c = cache.get_or_insert_with(|| {
                    Box::new(AdaptiveBlock {
                        index: -1,
                        last: -1,
                        values: [0; av::BLOCK_SIZE],
                    })
                });
                c.index = -1;
                av::decode_block(bytes, &mut c.values[..len])?;
                c.index = block;
                c.last = block;
                (&c.values[..], len)
            }
        };
        if within >= len {
            return Err(Error::DocOutOfRange(
                i32::try_from(ordinal).unwrap_or(i32::MAX),
                entry.num_values,
            ));
        }
        let n = (len.saturating_sub(within)).min(out.len().saturating_sub(done));
        let end = done.saturating_add(n);
        out[done..end].copy_from_slice(&src[within..within.saturating_add(n)]);
        done = end;
    }
    Ok(())
}

/// Decodes the raw bit-packed value at `ordinal` (doc id for dense fields,
/// rank-among-present-docs for sparse ones) and applies the table or
/// GCD-delta transform to get the final value.
fn decode_value(data: &[u8], entry: &NumericEntry, ordinal: i64) -> Result<i64> {
    if let Some(adaptive) = &entry.adaptive {
        // One value, no reader to keep a block for: decode just it.
        use crate::adaptive_values::{self as av, BLOCK_SHIFT};
        if ordinal < 0 || ordinal >= entry.num_values {
            return Err(Error::DocOutOfRange(
                i32::try_from(ordinal).unwrap_or(i32::MAX),
                entry.num_values,
            ));
        }
        let (bytes, _) = adaptive_block(data, entry, adaptive, ordinal >> BLOCK_SHIFT)?;
        let within = (ordinal & low_bits_mask(BLOCK_SHIFT)) as usize;
        return Ok(av::get_one(bytes, within)?);
    }
    if let Some(shift) = entry.block_shift {
        return decode_value_varying_bpv(data, entry, shift, ordinal);
    }
    if entry.bits_per_value == 0 {
        return Ok(entry.min_value);
    }
    let values = region(data, entry.values_offset, entry.values_length)?;
    let raw = direct_reader::get(values, entry.bits_per_value, ordinal)?;

    if let Some(table) = &entry.table {
        let idx = usize::try_from(raw).map_err(|_| lucene_store::Error::Eof { offset: 0 })?;
        table
            .get(idx)
            .copied()
            .ok_or(lucene_store::Error::Eof { offset: 0 }.into())
    } else {
        Ok(entry.gcd.wrapping_mul(raw).wrapping_add(entry.min_value))
    }
}

/// Decodes `ordinal`'s value out of a varying-bits-per-value field
/// (`entry.block_shift.is_some()`), mirroring
/// `Lucene90DocValuesProducer.VaryingBPVReader.getLongValue`.
///
/// Each block's own start offset (absolute in the whole `.dvd` file) is
/// looked up directly in the per-field jump table at
/// `entry.value_jump_table_offset + block * 8` -- no need to walk prior
/// blocks first, unlike a sequential scan. A block's own header is then:
/// one byte bits-per-value, an `i64` delta (the block's own min value, or
/// its single constant value when bits-per-value is 0), and -- only when
/// bits-per-value is non-zero -- an `i32` byte length followed by that many
/// bytes of bit-packed `(value - delta) / gcd` values (`entry.gcd` is the
/// whole field's GCD, reused by every block).
fn decode_value_varying_bpv(
    data: &[u8],
    entry: &NumericEntry,
    shift: u32,
    ordinal: i64,
) -> Result<i64> {
    let block = read_varying_block(data, entry, ordinal >> shift)?;
    if block.bits_per_value == 0 {
        return Ok(block.delta);
    }
    let values = data
        .get(block.values_start..block.values_end)
        .ok_or(lucene_store::Error::Eof { offset: 0 })?;
    let mask = low_bits_mask(shift);
    let raw = direct_reader::get(values, block.bits_per_value, ordinal & mask)?;
    Ok(entry.gcd.wrapping_mul(raw).wrapping_add(block.delta))
}

/// Reads the binary doc-values value for `doc`, handling all three
/// docs-with-a-value shapes (empty/dense/sparse) and both length shapes
/// (fixed/variable). `data` is the whole `.dvd` file's bytes. `Ok(None)`
/// means `doc` legitimately has no value.
pub fn binary_value<'d>(data: &'d [u8], entry: &BinaryEntry, doc: i32) -> Result<Option<&'d [u8]>> {
    if entry.mask.is_some() {
        return Err(crate::value_mask::masked_error("binary_value").into());
    }
    if entry.substring.is_some() {
        return Err(Error::CompressedBinary(entry.field_number));
    }
    if doc < 0 {
        return Err(Error::DocOutOfRange(doc, entry.num_docs_with_field as i64));
    }
    if entry.is_empty_field() {
        return Ok(None);
    }

    let ordinal = if entry.is_dense() {
        if doc >= entry.num_docs_with_field {
            return Err(Error::DocOutOfRange(doc, entry.num_docs_with_field as i64));
        }
        doc as i64
    } else {
        let region = entry.docs_region(data)?;
        // See `numeric_value` for why this is a cursor and not a full decode.
        match indexed_disi::DisiCursor::new(
            region,
            entry.dense_rank_power,
            entry.jump_table_entry_count,
        )
        .advance_exact(doc)?
        {
            Some(ordinal) => ordinal as i64,
            None => return Ok(None),
        }
    };

    binary_value_at_ordinal(data, entry, ordinal).map(Some)
}

/// The value-bytes half of [`binary_value`], split out so [`BinaryReader`] can
/// share it verbatim: given an already-resolved ordinal, slice the value out of
/// the data region (fixed-length arithmetic, or the
/// [`direct_monotonic`] end-offset array for a variable-length field).
fn binary_value_at_ordinal<'d>(
    data: &'d [u8],
    entry: &BinaryEntry,
    ordinal: i64,
) -> Result<&'d [u8]> {
    let bytes_region = region(data, entry.data_offset, entry.data_length)?;

    // The two branches produce a `[start, end)` pair rather than an offset
    // and a length: subtracting two `i64`s out of the address array and adding
    // the difference back would overflow twice over on a non-monotonic pair.
    let (start, end) = if let Some(addrs) = &entry.addresses {
        let addr_data = region(data, addrs.offset, addrs.length)?;
        let start = direct_monotonic::get(addr_data, &addrs.meta, ordinal)?;
        // ARITH: `ordinal` is either `doc as i64` for a dense field (bounded
        // by `num_docs_with_field`, an `i32`) or an `IndexedDISI` ordinal,
        // which `DisiCursor` returns as an `i32`. Either way it is far below
        // `i64::MAX`.
        #[allow(clippy::arithmetic_side_effects)]
        let end = direct_monotonic::get(addr_data, &addrs.meta, ordinal + 1)?;
        (start, end)
    } else {
        // ARITH: `ordinal` is bounded by `i32::MAX` (see above) and
        // `max_length` is an `i32`, so the product is at most 2^62 and the
        // `+ length` that follows cannot reach `i64::MAX`.
        #[allow(clippy::arithmetic_side_effects)]
        let range = {
            let length = entry.max_length as i64;
            (ordinal * length, ordinal * length + length)
        };
        range
    };

    slice_at(bytes_region, start, end)
}

/// [`NumericReader`] for BINARY doc values: a forward-only cursor that walks a
/// sparse field's [`indexed_disi`] region **once** across a scan, instead of
/// re-deriving it per lookup the way the free [`binary_value`] must.
///
/// The same rule applies as to [`NumericReader`]: anything reading more than
/// one document's value from a field -- a merge, a whole-column rewrite for a
/// doc-values update generation, a facet count -- should hold one of these.
/// `binary_value` in a `for doc in 0..max_doc` loop is `O(maxDoc x blocks)`
/// where Lucene's own iterator is `O(maxDoc + cardinality)`.
pub struct BinaryReader<'a> {
    data: &'a [u8],
    entry: &'a BinaryEntry,
    /// Forward-only cursor over the docs-with-field region of a sparse field;
    /// `None` for a dense or empty one, and `None` when the region's byte range
    /// is not inside `data` (a lookup then falls through to [`binary_value`],
    /// which raises the error).
    docs: Option<indexed_disi::DisiCursor<'a>>,
    /// The values' bytes and a variable-width field's addresses, resolved
    /// once as `getBinary` slices them once (`bytesSlice`, `addresses`);
    /// `None` when a region is not inside `data`, and each lookup then takes
    /// [`binary_value_at_ordinal`], which reports it.
    resolved: Option<ResolvedBinary<'a>>,
    /// A `Substring90` entry's index and its last decompressed block.
    substring: Option<
        Box<(
            crate::substring::SubstringIndex<'a>,
            crate::substring::Block,
        )>,
    >,
    /// The masked form of the value [`Self::value_ref`] last answered.
    shown: Vec<u8>,
}

/// [`BinaryReader`]'s per-iterator state.
struct ResolvedBinary<'a> {
    bytes: &'a [u8],
    /// `None` for a fixed-width field.
    addresses: Option<direct_monotonic::Reader<'a>>,
}

impl<'a> ResolvedBinary<'a> {
    fn new(data: &'a [u8], entry: &'a BinaryEntry) -> Option<Self> {
        let bytes = region(data, entry.data_offset, entry.data_length).ok()?;
        let addresses = match &entry.addresses {
            Some(addrs) => Some(direct_monotonic::Reader::new(
                region(data, addrs.offset, addrs.length).ok()?,
                &addrs.meta,
            )),
            None => None,
        };
        Some(ResolvedBinary { bytes, addresses })
    }

    /// [`binary_value_at_ordinal`] over the resolved regions.
    #[inline]
    fn value_at_ordinal(&self, entry: &BinaryEntry, ordinal: i64) -> Result<&'a [u8]> {
        let (start, end) = match &self.addresses {
            Some(addrs) => {
                let start = addrs.get(ordinal)?;
                // ARITH: as in `binary_value_at_ordinal`: `ordinal` is
                // `i32`-bounded.
                #[allow(clippy::arithmetic_side_effects)]
                let end = addrs.get(ordinal + 1)?;
                (start, end)
            }
            // ARITH: as in `binary_value_at_ordinal`.
            #[allow(clippy::arithmetic_side_effects)]
            None => {
                let length = entry.max_length as i64;
                (ordinal * length, ordinal * length + length)
            }
        };
        slice_at(self.bytes, start, end)
    }
}

impl<'a> BinaryReader<'a> {
    /// Opens a cursor over `entry`'s values in `data` (the whole `.dvd` file).
    /// Allocates nothing and reads nothing up front.
    pub fn new(data: &'a [u8], entry: &'a BinaryEntry) -> Self {
        let docs = if entry.is_empty_field() || entry.is_dense() {
            None
        } else {
            entry.docs_region(data).ok().map(|region| {
                indexed_disi::DisiCursor::new(
                    region,
                    entry.dense_rank_power,
                    entry.jump_table_entry_count,
                )
            })
        };
        let substring = entry.substring.as_deref().and_then(|e| {
            crate::substring::SubstringIndex::new(data, e)
                .ok()
                .map(|index| Box::new((index, crate::substring::Block::new())))
        });
        let resolved = if entry.is_empty_field() || entry.substring.is_some() {
            None
        } else {
            ResolvedBinary::new(data, entry)
        };
        Self {
            data,
            entry,
            docs,
            resolved,
            substring,
            shown: Vec::new(),
        }
    }

    /// This document's value, or `None` when it has none: [`Self::value`],
    /// and for a `Substring90` entry the value decompressed into this
    /// reader (so it lives until the next call).
    pub fn value_ref(&mut self, doc: i32) -> Result<Option<&[u8]>> {
        let Some(mask) = self.entry.mask.clone() else {
            return self.raw_value_ref(doc);
        };
        // `binaryValue()` of the masking wrapper: the stored bytes through the mask.
        match self.raw_value_ref(doc)? {
            None => Ok(None),
            Some(raw) => {
                let shown = mask.apply(raw)?;
                self.shown = shown;
                Ok(Some(&self.shown))
            }
        }
    }

    /// [`Self::value_ref`] without the mask: the stored bytes. Private: only the masking path
    /// above reads them.
    fn raw_value_ref(&mut self, doc: i32) -> Result<Option<&[u8]>> {
        if self.entry.substring.is_none() {
            return self.raw_value(doc);
        }
        match self.substring.as_deref_mut() {
            Some((index, block)) => index.value(doc, block),
            // The index did not open: its error.
            None => match self.entry.substring.as_deref() {
                Some(e) => crate::substring::SubstringIndex::new(self.data, e).map(|_| None),
                None => Ok(None),
            },
        }
    }

    /// This document's value, or `None` when it legitimately has none.
    /// Identical to `binary_value(data, entry, doc)`, without the per-call
    /// rederivation. Ascending `doc` walks the region once; going backwards
    /// rewinds the cursor, which costs a re-walk and still allocates nothing.
    pub fn value(&mut self, doc: i32) -> Result<Option<&'a [u8]>> {
        if self.entry.mask.is_some() {
            // The value borrows the file's bytes, which are not what the field shows.
            return Err(crate::value_mask::masked_error("BinaryReader::value").into());
        }
        self.raw_value(doc)
    }

    /// [`Self::value`] without the mask check: the stored bytes.
    fn raw_value(&mut self, doc: i32) -> Result<Option<&'a [u8]>> {
        if self.entry.substring.is_some() {
            return Err(Error::CompressedBinary(self.entry.field_number));
        }
        if doc < 0 {
            return Err(Error::DocOutOfRange(
                doc,
                self.entry.num_docs_with_field as i64,
            ));
        }
        if self.entry.is_empty_field() {
            return Ok(None);
        }
        let ordinal = if self.entry.is_dense() {
            if doc >= self.entry.num_docs_with_field {
                return Err(Error::DocOutOfRange(
                    doc,
                    self.entry.num_docs_with_field as i64,
                ));
            }
            doc as i64
        } else {
            let Some(cursor) = self.docs.as_mut() else {
                // The region is not inside `data`; let the free function raise
                // the same error it would have.
                return binary_value(self.data, self.entry, doc);
            };
            if doc < cursor.doc_id() {
                cursor.reset();
            }
            match cursor.advance_exact(doc)? {
                Some(ordinal) => ordinal as i64,
                None => return Ok(None),
            }
        };
        match &self.resolved {
            Some(r) => r.value_at_ordinal(self.entry, ordinal).map(Some),
            None => binary_value_at_ordinal(self.data, self.entry, ordinal).map(Some),
        }
    }
}

/// Reads a SORTED field's ordinal for `doc` -- exactly [`numeric_value`]
/// applied to the entry's `ords`, since Java's `readSorted` reuses the
/// NUMERIC entry layout verbatim for ordinals.
pub fn sorted_ord(data: &[u8], entry: &SortedEntry, doc: i32) -> Result<Option<i64>> {
    numeric_value(data, &entry.ords, doc)
}

/// Reads a SORTED_NUMERIC field's values for `doc` (zero or more numbers,
/// in the order Lucene wrote them), or a SORTED_SET field's ordinals when
/// applied to its `ords` entry -- both share this exact decode. Unlike
/// [`numeric_value`], a doc's *ordinal into the flat values array* isn't
/// simply its doc id/rank: that only holds when every doc-with-a-value has
/// exactly one value (`entry.addresses.is_none()`); otherwise each doc's
/// rank indexes an address range `[start, end)` covering its values.
pub fn sorted_numeric_values(
    data: &[u8],
    entry: &SortedNumericEntry,
    doc: i32,
) -> Result<Vec<i64>> {
    if doc < 0 {
        return Err(Error::DocOutOfRange(doc, entry.numeric.num_values));
    }
    if entry.numeric.is_empty_field() {
        return Ok(Vec::new());
    }

    let rank: i64 = if entry.numeric.is_dense() {
        // A dense entry means `numDocsWithField == maxDoc`
        // (`Lucene90DocValuesConsumer.writeValues` writes the `-1` marker on
        // exactly that condition), so `num_docs_with_field` is this field's
        // `maxDoc` and doubles as the bound Java's iterator gets from
        // `SegmentReadState`. Without this check a constant-encoded field
        // (`bitsPerValue == 0`) happily answers for any doc id at all, since
        // `decode_value` never indexes anything.
        if doc >= entry.num_docs_with_field {
            return Err(Error::DocOutOfRange(doc, entry.numeric.num_values));
        }
        doc as i64
    } else {
        let region = entry.numeric.docs_region(data)?;
        // See `numeric_value` for why this is a cursor and not a full decode.
        match indexed_disi::DisiCursor::new(
            region,
            entry.numeric.dense_rank_power,
            entry.numeric.jump_table_entry_count,
        )
        .advance_exact(doc)?
        {
            Some(r) => r as i64,
            None => return Ok(Vec::new()),
        }
    };

    match &entry.addresses {
        None => Ok(vec![decode_value(data, &entry.numeric, rank)?]),
        Some(addrs) => {
            let addr_region = region(data, addrs.offset, addrs.length)?;
            let start = direct_monotonic::get(addr_region, &addrs.meta, rank)?;
            // ARITH: `rank` is either `doc as i64` or an `IndexedDISI`
            // ordinal, both `i32`-bounded, so `rank + 1` cannot overflow.
            #[allow(clippy::arithmetic_side_effects)]
            let end = direct_monotonic::get(addr_region, &addrs.meta, rank + 1)?;
            // The address array indexes the field's flat value array, whose
            // length is `numValues`. Without that bound `start..end` is a
            // range two `.dvd` `i64`s chose: for a constant-encoded field
            // (`bitsPerValue == 0`) `decode_value` never indexes anything and
            // never fails, so the `collect` below would build a `Vec` of up
            // to 2^63 `i64`s -- an allocation **abort**, which `catch_unwind`
            // at the FFI boundary cannot intercept.
            if start < 0 || end < start || end > entry.numeric.num_values {
                return Err(Error::CorruptAddressRange {
                    field_number: entry.field_number,
                    start,
                    end,
                    num_values: entry.numeric.num_values,
                });
            }
            (start..end)
                .map(|i| decode_value(data, &entry.numeric, i))
                .collect()
        }
    }
}

/// A forward-only cursor over a SORTED_NUMERIC column --
/// `Lucene90DocValuesProducer.getSortedNumeric`'s iterator: one
/// `IndexedDISI` walk over the docs-with-field region for the whole scan,
/// and each document's values decoded straight into a caller-owned buffer.
/// [`sorted_numeric_values`] answers the same question for one document but
/// re-walks the region from its start and allocates each time; a merge,
/// which visits every document of a source in order, holds one of these.
#[derive(Debug, Clone)]
pub struct SortedNumericReader<'a> {
    data: &'a [u8],
    entry: &'a SortedNumericEntry,
    /// `None` for a dense or empty field, and when the region is not inside
    /// `data` (a lookup then falls through to [`sorted_numeric_values`],
    /// which raises the error).
    docs: Option<indexed_disi::DisiCursor<'a>>,
    /// The values array read by ordinal with one load each, when its shape
    /// allows ([`FastDense::values`]); `None` decodes each value in full.
    fast: Option<FastDense<'a>>,
    /// The addresses, resolved once as `getSortedNumeric` builds its
    /// `DirectMonotonicReader` once; `None` for a singleton field and when
    /// the region is not inside `data` (each lookup then resolves it, and
    /// reports that).
    addresses: Option<direct_monotonic::Reader<'a>>,
    /// The address at a rank already read (the last document's end), so a
    /// forward scan reads one address per document.
    next_address: Option<(i64, i64)>,
    /// The last varying-bits-per-value block header read, as
    /// [`NumericReader`] keeps it: a document's values, and the next
    /// document's, sit in one block.
    block: Option<VaryingBlock>,
    /// The last decoded adaptive delta block, if any.
    adaptive_block: Option<Box<AdaptiveBlock>>,
}

impl<'a> SortedNumericReader<'a> {
    /// Opens a cursor over `entry`'s values in `data` (the whole `.dvd`).
    /// Allocates nothing and reads nothing up front.
    pub fn new(data: &'a [u8], entry: &'a SortedNumericEntry) -> Self {
        let numeric = &entry.numeric;
        let docs = if numeric.is_empty_field() || numeric.is_dense() {
            None
        } else {
            numeric.docs_region(data).ok().map(|region| {
                indexed_disi::DisiCursor::new(
                    region,
                    numeric.dense_rank_power,
                    numeric.jump_table_entry_count,
                )
            })
        };
        let fast = FastDense::values(data, numeric);
        let addresses = entry.addresses.as_ref().and_then(|addrs| {
            region(data, addrs.offset, addrs.length)
                .ok()
                .map(|r| direct_monotonic::Reader::new(r, &addrs.meta))
        });
        Self {
            data,
            entry,
            docs,
            fast,
            addresses,
            next_address: None,
            block: None,
            adaptive_block: None,
        }
    }

    /// The field's entry: `numDocsWithField` is the iterator's `cost()`.
    pub fn entry(&self) -> &'a SortedNumericEntry {
        self.entry
    }

    /// Replaces `out` with document `doc`'s values, in stored (ascending)
    /// order; leaves it empty when the document has none. Ascending `doc`
    /// walks the region once; going backwards rewinds the cursor (a re-walk),
    /// as [`NumericReader::value`] does -- a merge that reorders documents
    /// visits a source out of order.
    pub fn values(&mut self, doc: i32, out: &mut Vec<i64>) -> Result<()> {
        out.clear();
        let numeric = &self.entry.numeric;
        if numeric.is_empty_field() {
            return Ok(());
        }
        let rank: i64 = if numeric.is_dense() {
            if doc < 0 || doc >= self.entry.num_docs_with_field {
                return Err(Error::DocOutOfRange(doc, numeric.num_values));
            }
            i64::from(doc)
        } else {
            let Some(cursor) = self.docs.as_mut().filter(|_| doc >= 0) else {
                *out = sorted_numeric_values(self.data, self.entry, doc)?;
                return Ok(());
            };
            if doc < cursor.doc_id() {
                cursor.reset();
            }
            match cursor.advance_exact(doc)? {
                Some(r) => r as i64,
                None => return Ok(()),
            }
        };
        self.push_values_at(rank, out)
    }

    /// Replaces `out` with the values of the `rank`-th document that has
    /// any (its position among them, which a caller holding the field's
    /// documents already knows; a dense field's rank is the document):
    /// [`Self::values`] without a walk of the docs-with-field region.
    ///
    /// # Errors
    /// [`Error::DocOutOfRange`] for a rank outside `0..numDocsWithField`;
    /// what [`Self::values`] reports for the values themselves.
    #[inline]
    pub fn values_at_rank(&mut self, rank: i64, out: &mut Vec<i64>) -> Result<()> {
        out.clear();
        if rank < 0 || rank >= i64::from(self.entry.num_docs_with_field) {
            return Err(Error::DocOutOfRange(
                i32::try_from(rank).unwrap_or(i32::MAX),
                self.entry.numeric.num_values,
            ));
        }
        self.push_values_at(rank, out)
    }

    /// Appends the values of the document of rank `rank`, which the caller
    /// has checked is one of the field's.
    #[inline(always)]
    fn push_values_at(&mut self, rank: i64, out: &mut Vec<i64>) -> Result<()> {
        let numeric = &self.entry.numeric;
        match &self.entry.addresses {
            // A singleton: the value of rank `rank`, through the resolved
            // values array when its shape allows (one load), as below.
            None => {
                let fast = i32::try_from(rank)
                    .ok()
                    .and_then(|i| self.fast.as_ref()?.get(i));
                out.push(match fast {
                    Some(v) => v,
                    None => decode_value_cached(
                        self.data,
                        numeric,
                        rank,
                        &mut self.block,
                        &mut self.adaptive_block,
                    )?,
                });
            }
            Some(addrs) => {
                let (start, end) = match &self.addresses {
                    // ARITH: `rank` is a doc id or an `IndexedDISI` ordinal,
                    // both `i32`-bounded, so `rank + 1` cannot overflow.
                    #[allow(clippy::arithmetic_side_effects)]
                    Some(reader) => {
                        // The previous document's end is this one's start
                        // when the ranks run on (a forward scan).
                        let start = match self.next_address {
                            Some((r, address)) if r == rank => address,
                            _ => reader.get(rank)?,
                        };
                        let end = reader.get(rank + 1)?;
                        self.next_address = Some((rank + 1, end));
                        (start, end)
                    }
                    None => {
                        let addr_region = region(self.data, addrs.offset, addrs.length)?;
                        let start = direct_monotonic::get(addr_region, &addrs.meta, rank)?;
                        // ARITH: as above.
                        #[allow(clippy::arithmetic_side_effects)]
                        let end = direct_monotonic::get(addr_region, &addrs.meta, rank + 1)?;
                        (start, end)
                    }
                };
                // Same bound as `sorted_numeric_values`: the range must lie in
                // the flat value array, or a constant-encoded field would
                // "decode" an address-chosen number of values.
                if start < 0 || end < start || end > numeric.num_values {
                    return Err(Error::CorruptAddressRange {
                        field_number: self.entry.field_number,
                        start,
                        end,
                        num_values: numeric.num_values,
                    });
                }
                // `end <= num_values`, so every index is an `i32` whenever
                // `end` is; each value through the resolved values array
                // when its shape allows, decoded in full otherwise.
                match (&self.fast, i32::try_from(end)) {
                    (Some(fast), Ok(_)) => {
                        for i in start..end {
                            out.push(match fast.get(i as i32) {
                                Some(v) => v,
                                None => decode_value_cached(
                                    self.data,
                                    numeric,
                                    i,
                                    &mut self.block,
                                    &mut self.adaptive_block,
                                )?,
                            });
                        }
                    }
                    _ => {
                        for i in start..end {
                            out.push(decode_value_cached(
                                self.data,
                                numeric,
                                i,
                                &mut self.block,
                                &mut self.adaptive_block,
                            )?);
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// The values of the documents `start..start + len` in one forward pass:
    /// document `start + i`'s values are `values[offsets[i]..offsets[i + 1]]`
    /// (`offsets` gets `len + 1` entries), each what [`Self::values`]
    /// answers, which the tests check document for document.
    ///
    /// For a window-at-a-time consumer. Where the values are a single-width
    /// array: a dense field's ranks are its documents, a sparse one's come
    /// off its `IndexedDISI` a block window at a time
    /// (`DisiCursor::window_bits`); either way they run consecutively, so one
    /// address per document is read (the previous one's end is the next one's
    /// start) and the values are decoded a chunk at a time. Any other shape
    /// (a SPARSE block, a window across blocks, values not in one width) reads
    /// nothing and answers `false`: the caller asks [`Self::values`] for the
    /// documents it wants, rather than every document of the window.
    ///
    /// # Errors
    /// What [`Self::values`] reports, including an address range outside the
    /// values array.
    pub fn fill_window(
        &mut self,
        start: i32,
        len: usize,
        offsets: &mut Vec<u32>,
        values: &mut Vec<i64>,
    ) -> Result<bool> {
        offsets.clear();
        values.clear();
        offsets.push(0);
        let fits = i32::try_from(len).ok().and_then(|l| start.checked_add(l));
        if start < 0 || fits.is_none() {
            return Err(Error::DocOutOfRange(start, len as i64));
        }
        // A window behind the cursor rewinds it, whichever path reads it.
        if let Some(cursor) = self.docs.as_mut() {
            if start < cursor.doc_id() {
                cursor.reset();
            }
        }
        let numeric = &self.entry.numeric;
        // The documents with values and the rank of the first, when they
        // come in one piece.
        let mut present = vec![0u64; len.div_ceil(64)];
        let first_rank: Option<i64> = match (&self.entry.addresses, self.fast) {
            (Some(_), Some(_)) if numeric.is_empty_field() => None,
            (Some(_), Some(_)) if numeric.is_dense() => {
                let have = usize::try_from(self.entry.num_docs_with_field.saturating_sub(start))
                    .unwrap_or(0)
                    .min(len);
                // ARITH: `i < have <= len`, and `i >> 6 < present.len()`.
                #[allow(clippy::arithmetic_side_effects)]
                for i in 0..have {
                    present[i >> 6] |= 1 << (i & 63);
                }
                Some(i64::from(start))
            }
            (Some(_), Some(_)) if len.is_multiple_of(64) => match self.docs.as_mut() {
                Some(cursor) => cursor.window_bits(start, &mut present)?,
                None => None,
            },
            _ => None,
        };
        let (Some(rank0), Some(addrs), Some(fast)) = (first_rank, &self.entry.addresses, self.fast)
        else {
            // An empty field has nothing in any window.
            if numeric.is_empty_field() {
                // ARITH: `len` fits an `i32` (checked above).
                #[allow(clippy::arithmetic_side_effects)]
                offsets.resize(len + 1, 0);
                return Ok(true);
            }
            return Ok(false);
        };
        let addr_region = region(self.data, addrs.offset, addrs.length)?;
        let count: usize = present.iter().map(|w| w.count_ones() as usize).sum();
        // The addresses of ranks `rank0..=rank0 + count`, in one pass: each
        // present document's values run from its address to the next.
        // ARITH: `count <= len`, which fits an `i32`.
        #[allow(clippy::arithmetic_side_effects)]
        let mut bounds = vec![0i64; if count == 0 { 0 } else { count + 1 }];
        direct_monotonic::fill(addr_region, &addrs.meta, rank0, &mut bounds)?;
        let first = bounds.first().copied().unwrap_or(0);
        let mut lo = first;
        let mut k = 0usize;
        // ARITH: `i < len`, `k < count` for every present document, and every
        // address is checked against `num_values` before it sizes anything.
        #[allow(clippy::arithmetic_side_effects)]
        {
            if lo < 0 || lo > numeric.num_values {
                return Err(Error::CorruptAddressRange {
                    field_number: self.entry.field_number,
                    start: lo,
                    end: lo,
                    num_values: numeric.num_values,
                });
            }
            for i in 0..len {
                if present[i >> 6] >> (i & 63) & 1 == 1 {
                    k += 1;
                    let hi = bounds[k];
                    if hi < lo || hi > numeric.num_values {
                        return Err(Error::CorruptAddressRange {
                            field_number: self.entry.field_number,
                            start: lo,
                            end: hi,
                            num_values: numeric.num_values,
                        });
                    }
                    lo = hi;
                }
                // Checked just above: `first <= lo <= num_values`, and the
                // window's values fit the `u32` offsets a window's length
                // bounds in practice (saturating otherwise, and caught below).
                offsets.push(u32::try_from(lo - first).unwrap_or(u32::MAX));
            }
            // The window's values, ranks `first..lo`, a chunk at a time.
            let n = usize::try_from(lo - first).unwrap_or(0);
            values.resize(n, 0);
            let mut done = 0;
            while done < n {
                let at = first + done as i64;
                let got = i32::try_from(at).map_or(0, |a| fast.fill(a, &mut values[done..]));
                if got == 0 {
                    values[done] = decode_value_cached(
                        self.data,
                        numeric,
                        at,
                        &mut self.block,
                        &mut self.adaptive_block,
                    )?;
                    done += 1;
                } else {
                    done += got;
                }
            }
            if offsets.last().is_some_and(|&o| o as usize != n) {
                return Err(Error::CorruptAddressRange {
                    field_number: self.entry.field_number,
                    start: first,
                    end: lo,
                    num_values: numeric.num_values,
                });
            }
        }
        Ok(true)
    }

    /// Calls `f(doc, values)` for every document in `start..end` that has
    /// values, ascending, each document's values ascending -- [`Self::values`]
    /// over a range, the way an aggregation over a match-all reads a column.
    ///
    /// A field whose values [`FastDense`] can read is streamed: the documents
    /// with values come from the field's `IndexedDISI` as a bit set (a sparse
    /// field, read from document 0) or are every document (a dense one), so
    /// their value ranks run consecutively; one address per document (the
    /// previous document's end is this one's start), and the values decoded a
    /// chunk at a time. Every other shape goes through [`Self::values`]
    /// document by document.
    ///
    /// # Errors
    /// What [`Self::values`] reports, including an address range outside the
    /// values array.
    pub fn for_each_doc(&mut self, start: i32, end: i32, f: impl FnMut(i32, &[i64])) -> Result<()> {
        self.for_each_accepted(start, end, |_| true, f)
    }

    /// [`Self::for_each_doc`] for the documents `accept` takes only: a
    /// rejected document's values are neither decoded nor handed to `f`, and
    /// on the streamed path its addresses are not read either -- what an
    /// aggregation over a query's matches, a fraction of the column, costs.
    ///
    /// # Errors
    /// What [`Self::for_each_doc`] reports, for the accepted documents.
    pub fn for_each_accepted(
        &mut self,
        start: i32,
        end: i32,
        accept: impl FnMut(i32) -> bool,
        f: impl FnMut(i32, &[i64]),
    ) -> Result<()> {
        let start = start.max(0);
        if start >= end {
            return Ok(());
        }
        let numeric = &self.entry.numeric;
        let streamable = self.entry.addresses.is_some()
            && FastDense::values(self.data, numeric).is_some()
            && (numeric.is_dense() || start == 0);
        if !streamable {
            return self.for_each_doc_slow(start, end, accept, f);
        }
        if numeric.is_dense() {
            let end = end.min(self.entry.num_docs_with_field);
            return self.stream(i64::from(start), start..end, accept, f);
        }
        // Sparse: the documents with values, as bits.
        let Some(words_len) = usize::try_from(end).ok().map(|e| e.div_ceil(64)) else {
            return Ok(());
        };
        let mut words = vec![0u64; words_len];
        let filled = numeric.docs_region(self.data).ok().and_then(|r| {
            indexed_disi::or_into_words(r, numeric.dense_rank_power, &mut words).ok()
        });
        if filled.is_none() {
            return self.for_each_doc_slow(start, end, accept, f);
        }
        // ARITH: `w < words_len`, itself `end / 64` rounded up, and `b < 64`,
        // so `w * 64 + b` is below `words_len * 64 <= end + 63`.
        #[allow(clippy::arithmetic_side_effects)]
        let docs = words.iter().enumerate().flat_map(|(w, &bits)| {
            let base = w * 64;
            BitIter(bits).map(move |b| base + b)
        });
        // ARITH: a set bit is below `words_len * 64`, derived from an `i32`.
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        let docs = docs.map(|d| d as i32).take_while(|&d| d < end);
        self.stream(0, docs, accept, f)
    }

    /// [`Self::for_each_doc`] one [`Self::values`] call per document.
    fn for_each_doc_slow(
        &mut self,
        start: i32,
        end: i32,
        mut accept: impl FnMut(i32) -> bool,
        mut f: impl FnMut(i32, &[i64]),
    ) -> Result<()> {
        let mut vals = Vec::new();
        for doc in start..end {
            if !accept(doc) {
                continue;
            }
            self.values(doc, &mut vals)?;
            if !vals.is_empty() {
                f(doc, &vals);
            }
        }
        Ok(())
    }

    /// The streaming half of [`Self::for_each_doc`]: `docs` are documents with
    /// values whose ranks run consecutively from `first_rank`.
    fn stream(
        &self,
        first_rank: i64,
        docs: impl Iterator<Item = i32>,
        mut accept: impl FnMut(i32) -> bool,
        mut f: impl FnMut(i32, &[i64]),
    ) -> Result<()> {
        const CHUNK: usize = 256;
        let numeric = &self.entry.numeric;
        let (Some(addrs), Some(fast)) =
            (&self.entry.addresses, FastDense::values(self.data, numeric))
        else {
            return Ok(());
        };
        let addr_region = region(self.data, addrs.offset, addrs.length)?;
        let mut rank = first_rank;
        // The start address of the document at `rank`, when the previous
        // document's end already gave it; read afresh after a rejected one.
        let mut next_lo: Option<i64> = None;
        let mut buf = [0i64; CHUNK];
        // `buf[..buf_n]` holds the values at ordinals `buf_lo..buf_lo + buf_n`.
        let (mut buf_lo, mut buf_n) = (0i64, 0usize);
        let mut long = Vec::new();
        let mut block = None;
        for doc in docs {
            let at = rank;
            // ARITH: a rank is below the field's document count, an `i32`.
            #[allow(clippy::arithmetic_side_effects)]
            {
                rank += 1;
            }
            if !accept(doc) {
                next_lo = None;
                continue;
            }
            let lo = match next_lo {
                Some(lo) => lo,
                None => direct_monotonic::get(addr_region, &addrs.meta, at)?,
            };
            let hi = direct_monotonic::get(addr_region, &addrs.meta, rank)?;
            if lo < 0 || hi < lo || hi > numeric.num_values {
                return Err(Error::CorruptAddressRange {
                    field_number: self.entry.field_number,
                    start: lo,
                    end: hi,
                    num_values: numeric.num_values,
                });
            }
            // ARITH: `0 <= lo <= hi <= num_values`, and `buf_lo + buf_n` is a
            // value ordinal the buffer was filled from, so every difference
            // below is non-negative and in range.
            #[allow(clippy::arithmetic_side_effects)]
            {
                if hi > buf_lo + buf_n as i64 {
                    buf_n = match i32::try_from(lo) {
                        Ok(at) => fast.fill(at, &mut buf),
                        Err(_) => 0,
                    };
                    buf_lo = lo;
                }
                if hi <= buf_lo + buf_n as i64 {
                    f(doc, &buf[(lo - buf_lo) as usize..(hi - buf_lo) as usize]);
                } else {
                    // More values than a chunk, or ones the fast reader
                    // cannot answer: decode them one by one.
                    long.clear();
                    for i in lo..hi {
                        long.push(decode_value_cached(
                            self.data, numeric, i, &mut block, &mut None,
                        )?);
                    }
                    f(doc, &long);
                }
            }
            next_lo = Some(hi);
        }
        Ok(())
    }
}

/// The set bits of a word, lowest first.
struct BitIter(u64);

impl Iterator for BitIter {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<usize> {
        if self.0 == 0 {
            return None;
        }
        let b = self.0.trailing_zeros() as usize;
        // ARITH: the word is non-zero here, so `- 1` does not wrap.
        #[allow(clippy::arithmetic_side_effects)]
        {
            self.0 &= self.0 - 1;
        }
        Some(b)
    }
}

/// A SORTED column whose dictionary is already built: `dict` sorted and
/// unique, `ords[k]` the ordinal of document `docs[k]`, `docs` strictly
/// ascending. What `DocValuesConsumer.mergeSortedField` hands
/// `addSortedField` -- an `OrdinalMap` over the sources' dictionaries and the
/// documents' remapped ordinals -- so a merge never rebuilds a dictionary
/// from per-document bytes.
#[derive(Debug, Clone, Default)]
pub struct SortedOrdsColumn {
    pub dict: Vec<Vec<u8>>,
    pub docs: Vec<i32>,
    pub ords: Vec<i64>,
}

/// The SORTED_SET counterpart of [`SortedOrdsColumn`]: document `docs[k]`
/// has `counts[k]` ordinals (at least one), stored consecutively in `ords`,
/// ascending and unique within the document.
#[derive(Debug, Clone, Default)]
pub struct SortedSetOrdsColumn {
    pub dict: Vec<Vec<u8>>,
    pub docs: Vec<i32>,
    pub counts: Vec<u32>,
    pub ords: Vec<i64>,
}

/// A SORTED_NUMERIC column streamed in doc order: document `docs[k]` has
/// `counts[k]` values (at least one), ascending, stored consecutively in
/// `values`. The flat shape `Lucene90DocValuesConsumer.addSortedNumericField`
/// consumes from a merge's iterator, without a vector per document.
#[derive(Debug, Clone, Default)]
pub struct SortedNumericColumn {
    pub docs: Vec<i32>,
    pub counts: Vec<u32>,
    pub values: Vec<i64>,
}

/// A BINARY column streamed in doc order: document `docs[k]`'s value is the
/// next `lengths[k]` bytes of `bytes`.
#[derive(Debug, Clone, Default)]
pub struct BinaryColumn {
    pub docs: Vec<i32>,
    pub lengths: Vec<u32>,
    pub bytes: Vec<u8>,
}

/// Write-side error, kept separate from the read-side [`Error`] since none
/// of its variants are decode failures.
#[derive(Debug, thiserror::Error)]
pub enum WriteError {
    /// `FieldInfo`'s rule: a doc-values skip index is for NUMERIC,
    /// SORTED_NUMERIC, SORTED and SORTED_SET fields only.
    #[error("field {0}: a doc-values skip index cannot be written for a BINARY field")]
    SkipIndexOnBinary(i32),
    #[error("this format writes a .dvp file: finish it with finish_files")]
    DisiFileNotReturned,
    #[error("dense doc-values write requires values.len() == max_doc (every doc must have a value); got {values} values for max_doc={max_doc}")]
    NotDense { values: usize, max_doc: i32 },
    #[error("dense sorted-numeric write requires every doc to have at least one value; doc {0} has none")]
    EmptyMultiValuedDoc(i32),
    #[error("sparse doc-values write requires strictly ascending doc ids; doc {0} is out of order or duplicated")]
    DocIdsNotAscending(i32),
    #[error("sparse doc-values write requires doc {0} < max_doc={1}")]
    DocIdOutOfRange(i32, i32),
    #[error("write_dense_fields requires at least one field")]
    EmptyFieldList,
    /// `Lucene90DocValuesFormat(int)`'s argument check.
    #[error("skipIndexIntervalSize must be > 1, got [{0}]")]
    InvalidSkipIndexIntervalSize(i32),
    #[error(
        "write_dense_fields requires distinct field numbers; field {0} appears more than once"
    )]
    DuplicateFieldNumber(i32),
}

pub type WriteResult<T> = std::result::Result<T, WriteError>;

/// One field's dense (every doc `0..max_doc` has a value) doc-values input,
/// as accepted by [`write_dense_fields`] -- one variant per
/// [`crate::field_infos::DocValuesType`]. Each variant's `values` slice must
/// have exactly `max_doc` entries (the same `max_doc` [`write_dense_fields`]
/// is called with, shared across every field in the call since they all
/// describe the same segment), mirroring each single-field function's own
/// `values.len() == max_doc` requirement ([`WriteError::NotDense`]).
///
/// `SortedNumeric`'s per-doc entry must be non-empty (a doc with zero values
/// is a sparse concept, out of scope for this dense-only entry point --
/// [`WriteError::EmptyMultiValuedDoc`]), same restriction
/// [`write_single_dense_sorted_numeric_field`] already enforces.
/// `SortedSet`'s per-doc entry has the same restriction.
#[derive(Debug, Clone, Copy)]
pub enum DenseField<'a> {
    Numeric(i32, &'a [i64]),
    /// The one **sparse** shape this multi-field writer accepts:
    /// `(doc_id, value)` pairs for the docs that have a value, exactly the
    /// input [`write_single_sparse_numeric_field`] takes, written through the
    /// same `IndexedDISI` + values body. It is here because an *index sort*
    /// over more than one tier needs every tier's column in the same
    /// `.dvm`/`.dvd`, and a sort field whose documents may lack a value is
    /// the normal case (Java's `SortField.setMissingValue`), not an exotic
    /// one -- so a dense-only multi-field writer cannot express the sorts
    /// Lucene routinely writes.
    SparseNumeric(i32, &'a [(i32, i64)]),
    Binary(i32, &'a [Vec<u8>]),
    Sorted(i32, &'a [Vec<u8>]),
    SortedNumeric(i32, &'a [Vec<i64>]),
    SortedSet(i32, &'a [Vec<Vec<u8>>]),
    /// The BINARY/SORTED/SORTED_NUMERIC/SORTED_SET counterparts of
    /// [`DenseField::SparseNumeric`], each taking `(doc_id, value)` pairs for
    /// the documents that have a value.
    ///
    /// They exist because a **merge** needs them, not for symmetry:
    /// `DocValuesConsumer.getMerged*DocValues` drops any source reader that
    /// has no column for the field -- those documents are simply missing the
    /// value -- so merging an ordinary index that started writing a
    /// doc-values field partway through its life produces a sparse column for
    /// every one of the five types. Without these variants
    /// `lucene-index`'s `execute_merge` had to *fail* such a merge for the
    /// four non-NUMERIC types, an error on a perfectly valid index.
    /// `SparseNumeric` serves the same purpose on the sort path.
    SparseBinary(i32, &'a [(i32, Vec<u8>)]),
    SparseSorted(i32, &'a [(i32, Vec<u8>)]),
    SparseSortedNumeric(i32, &'a [(i32, Vec<i64>)]),
    SparseSortedSet(i32, &'a [(i32, Vec<Vec<u8>>)]),
    /// A SORTED column with its dictionary already built, dense or sparse
    /// (see [`SortedOrdsColumn`]).
    SortedOrds(i32, &'a SortedOrdsColumn),
    /// A SORTED_SET column with its dictionary already built, dense or
    /// sparse (see [`SortedSetOrdsColumn`]).
    SortedSetOrds(i32, &'a SortedSetOrdsColumn),
    /// A SORTED_NUMERIC column in flat form, dense or sparse (see
    /// [`SortedNumericColumn`]).
    SortedNumericColumn(i32, &'a SortedNumericColumn),
    /// A BINARY column in flat form, dense or sparse (see [`BinaryColumn`]).
    BinaryColumn(i32, &'a BinaryColumn),
}

impl DenseField<'_> {
    /// What `writeSkipIndex` iterates: every document with a value, in doc
    /// order, with its values ascending -- the numbers themselves for
    /// NUMERIC and SORTED_NUMERIC, the ordinals for SORTED and SORTED_SET
    /// (`doAddSortedField`'s and `addSortedSetField`'s views) -- flat, as
    /// [`SortedNumericColumn`] lays a column out.
    // ARITH: indices into in-memory slices.
    #[allow(clippy::arithmetic_side_effects)]
    fn skip_index_stream(&self, max_doc: i32) -> WriteResult<SortedNumericColumn> {
        let mut out = SortedNumericColumn::default();
        let push =
            |out: &mut SortedNumericColumn, doc: i32, values: &mut dyn Iterator<Item = i64>| {
                let start = out.values.len();
                out.values.extend(values);
                out.values[start..].sort_unstable();
                out.docs.push(doc);
                out.counts.push((out.values.len() - start) as u32);
            };
        fn set_ords(per_doc: &[&Vec<Vec<u8>>]) -> Vec<Vec<i64>> {
            let mut dict: Vec<&Vec<u8>> = per_doc.iter().flat_map(|v| v.iter()).collect();
            dict.sort_unstable();
            dict.dedup();
            per_doc
                .iter()
                .map(|values| {
                    let mut ords: Vec<i64> = values
                        .iter()
                        .map(|v| dict.binary_search(&v).unwrap_or(0) as i64)
                        .collect();
                    ords.sort_unstable();
                    ords.dedup();
                    ords
                })
                .collect()
        }
        match self {
            DenseField::Numeric(_, values) => {
                for (doc, &v) in values.iter().enumerate() {
                    push(&mut out, doc as i32, &mut std::iter::once(v));
                }
            }
            DenseField::SparseNumeric(_, pairs) => {
                for (doc, v) in sort_sparse_pairs(pairs, max_doc)? {
                    push(&mut out, doc, &mut std::iter::once(v));
                }
            }
            DenseField::Sorted(_, values) => {
                let ords = build_sorted_dict_and_ords(values).1;
                for (doc, o) in ords.into_iter().enumerate() {
                    push(&mut out, doc as i32, &mut std::iter::once(o));
                }
            }
            DenseField::SparseSorted(_, pairs) => {
                let sorted = sort_sparse_pairs(pairs, max_doc)?;
                let values: Vec<Vec<u8>> = sorted.iter().map(|(_, v)| v.clone()).collect();
                let ords = build_sorted_dict_and_ords(&values).1;
                for ((doc, _), o) in sorted.iter().zip(ords) {
                    push(&mut out, *doc, &mut std::iter::once(o));
                }
            }
            DenseField::SortedOrds(_, c) => {
                for (&doc, &o) in c.docs.iter().zip(&c.ords) {
                    push(&mut out, doc, &mut std::iter::once(o));
                }
            }
            DenseField::SortedNumeric(_, values) => {
                for (doc, v) in values.iter().enumerate() {
                    push(&mut out, doc as i32, &mut v.iter().copied());
                }
            }
            DenseField::SparseSortedNumeric(_, pairs) => {
                for (doc, v) in sort_sparse_pairs(pairs, max_doc)? {
                    push(&mut out, doc, &mut v.into_iter());
                }
            }
            DenseField::SortedNumericColumn(_, c) => {
                let mut at = 0usize;
                for (&doc, &n) in c.docs.iter().zip(&c.counts) {
                    let n = n as usize;
                    push(&mut out, doc, &mut c.values[at..at + n].iter().copied());
                    at += n;
                }
            }
            DenseField::SortedSet(_, values) => {
                let per_doc: Vec<&Vec<Vec<u8>>> = values.iter().collect();
                for (doc, ords) in set_ords(&per_doc).into_iter().enumerate() {
                    push(&mut out, doc as i32, &mut ords.into_iter());
                }
            }
            DenseField::SparseSortedSet(_, pairs) => {
                let sorted = sort_sparse_pairs(pairs, max_doc)?;
                let per_doc: Vec<&Vec<Vec<u8>>> = sorted.iter().map(|(_, v)| v).collect();
                for ((doc, _), ords) in sorted.iter().zip(set_ords(&per_doc)) {
                    push(&mut out, *doc, &mut ords.into_iter());
                }
            }
            DenseField::SortedSetOrds(_, c) => {
                let mut at = 0usize;
                for (&doc, &n) in c.docs.iter().zip(&c.counts) {
                    let n = n as usize;
                    push(&mut out, doc, &mut c.ords[at..at + n].iter().copied());
                    at += n;
                }
            }
            DenseField::Binary(n, _)
            | DenseField::SparseBinary(n, _)
            | DenseField::BinaryColumn(n, _) => {
                return Err(WriteError::SkipIndexOnBinary(*n));
            }
        }
        Ok(out)
    }

    fn field_number(&self) -> i32 {
        match self {
            DenseField::Numeric(n, _)
            | DenseField::SparseNumeric(n, _)
            | DenseField::Binary(n, _)
            | DenseField::Sorted(n, _)
            | DenseField::SortedNumeric(n, _)
            | DenseField::SortedSet(n, _)
            | DenseField::SparseBinary(n, _)
            | DenseField::SparseSorted(n, _)
            | DenseField::SparseSortedNumeric(n, _)
            | DenseField::SparseSortedSet(n, _)
            | DenseField::SortedOrds(n, _)
            | DenseField::SortedSetOrds(n, _)
            | DenseField::SortedNumericColumn(n, _)
            | DenseField::BinaryColumn(n, _) => *n,
        }
    }

    fn len(&self) -> usize {
        match self {
            DenseField::Numeric(_, v) => v.len(),
            // Not a per-doc count, so `write_dense_fields`' density check
            // skips this variant entirely (see there).
            DenseField::SparseNumeric(_, v) => v.len(),
            DenseField::Binary(_, v) => v.len(),
            DenseField::Sorted(_, v) => v.len(),
            DenseField::SortedNumeric(_, v) => v.len(),
            DenseField::SortedSet(_, v) => v.len(),
            // As for `SparseNumeric`: a count of docs *with* a value, not a
            // per-doc count, so the density check skips these too.
            DenseField::SparseBinary(_, v) => v.len(),
            DenseField::SparseSorted(_, v) => v.len(),
            DenseField::SparseSortedNumeric(_, v) => v.len(),
            DenseField::SparseSortedSet(_, v) => v.len(),
            DenseField::SortedOrds(_, c) => c.docs.len(),
            DenseField::SortedSetOrds(_, c) => c.docs.len(),
            DenseField::SortedNumericColumn(_, c) => c.docs.len(),
            DenseField::BinaryColumn(_, c) => c.docs.len(),
        }
    }

    fn is_sparse(&self) -> bool {
        matches!(
            self,
            DenseField::SparseNumeric(..)
                | DenseField::SparseBinary(..)
                | DenseField::SparseSorted(..)
                | DenseField::SparseSortedNumeric(..)
                | DenseField::SparseSortedSet(..)
                | DenseField::SortedOrds(..)
                | DenseField::SortedSetOrds(..)
                | DenseField::SortedNumericColumn(..)
                | DenseField::BinaryColumn(..)
        )
    }
}

/// Writes **one or more** distinct dense doc-values fields into a single
/// `.dvm`/`.dvd`/`.dvs` triple -- the multi-field analogue of
/// [`write_single_dense_numeric_field`]/[`write_single_dense_binary_field`]/
/// [`write_single_dense_sorted_field`]/
/// [`write_single_dense_sorted_numeric_field`]/
/// [`write_single_dense_sorted_set_field`], all five of which are now thin
/// one-element-slice wrappers over this function (kept so existing callers
/// are unaffected, same precedent as
/// [`crate::postings_writer::write_single_field`] over
/// [`crate::postings_writer::write_fields`]). `numFields` worth of per-field
/// meta entries are interleaved into the *same* physical meta/data buffers,
/// exactly like a real multi-field segment's `.dvm`/`.dvd`.
///
/// Every field in `fields` must be dense over the same `max_doc` (`values.len()
/// == max_doc` for every field -- [`WriteError::NotDense`]) and have a
/// distinct field number ([`WriteError::DuplicateFieldNumber`]); `fields` must
/// be non-empty ([`WriteError::EmptyFieldList`]).
// ARITH: writer-side arithmetic over data already in memory. Every length is
// a `Vec`/slice length (so at most `isize::MAX`), every running offset is a
// prefix sum of such lengths into a buffer that has already been built, and
// every `data.len() as i64 - X` subtracts an offset recorded from an earlier
// `data.len()` of the same, only-ever-appended buffer. Nothing here comes off
// disk -- this is the encode half.
#[allow(clippy::arithmetic_side_effects)]
pub fn write_dense_fields(
    fields: &[DenseField<'_>],
    max_doc: i32,
    segment_id: &[u8; ID_LENGTH],
    segment_suffix: &str,
) -> WriteResult<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    write_fields_with_skip_indexes(fields, &[], max_doc, segment_id, segment_suffix)
}

/// [`write_dense_fields`], plus a doc-values skip index
/// (`DocValuesSkipIndexType.RANGE`) for every field whose number is in
/// `skip_index_fields` -- `Lucene90DocValuesConsumer.writeSkipIndex`, written
/// into `.dvs` with its summary in the field's `.dvm` entry right after the
/// type byte, as Java does. A skip index is refused on a BINARY field, as
/// `FieldInfo` refuses it.
// ARITH: as `write_dense_fields`.
#[allow(clippy::arithmetic_side_effects)]
pub fn write_fields_with_skip_indexes(
    fields: &[DenseField<'_>],
    skip_index_fields: &[i32],
    max_doc: i32,
    segment_id: &[u8; ID_LENGTH],
    segment_suffix: &str,
) -> WriteResult<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    write_fields_with_format(
        fields,
        skip_index_fields,
        max_doc,
        segment_id,
        segment_suffix,
        Lucene90DocValuesFormat::default(),
    )
}

/// The `.dvd` buffer being written, and the `.dvp` one beside it when the
/// format keeps sparse fields' `IndexedDISI` there ([`VERSION_DISI_SEPARATE_FILE`]).
/// It derefs to the `.dvd` bytes, so a writer that never writes a DISI treats
/// it as the plain buffer.
#[derive(Debug, Default)]
struct DvData {
    buf: Vec<u8>,
    disi: Option<Vec<u8>>,
}

impl DvData {
    fn new(buf: Vec<u8>, disi: Option<Vec<u8>>) -> Self {
        Self { buf, disi }
    }

    /// Appends one field's `IndexedDISI` and returns its `(offset, length)`
    /// in the file it lands in (`.dvp`, or `.dvd` before format 3).
    fn write_disi(&mut self, bytes: &[u8]) -> (i64, i64) {
        let target = self.disi.as_mut().unwrap_or(&mut self.buf);
        let offset = target.len() as i64;
        target.extend_from_slice(bytes);
        (offset, bytes.len() as i64)
    }
}

impl std::ops::Deref for DvData {
    type Target = Vec<u8>;
    fn deref(&self) -> &Vec<u8> {
        &self.buf
    }
}

impl std::ops::DerefMut for DvData {
    fn deref_mut(&mut self) -> &mut Vec<u8> {
        &mut self.buf
    }
}

/// `Lucene90DocValuesFormat(skipIndexIntervalSize)`: the doc-values format a
/// field is routed to (`PerFieldDocValuesFormat`). Its one setting is the
/// number of documents a skip-index interval holds before it may close;
/// [`Default`] is `Lucene90DocValuesFormat()`,
/// [`Self::DEFAULT_SKIP_INDEX_INTERVAL_SIZE`].
///
/// Java groups fields by format *instance*; this port compares formats by
/// value, so two equal instances share one set of files unless
/// [`Self::as_instance`] tells them apart.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Lucene90DocValuesFormat {
    skip_index_interval_size: i32,
    /// Distinguishes equal formats Java holds as separate instances.
    instance: u32,
    /// `getName()`: [`Self::NAME`], or the name of a format that wraps this
    /// one without changing its files (`AssertingDocValuesFormat`'s
    /// `Asserting`).
    name: &'static str,
    /// Whether NUMERIC and SORTED_NUMERIC values and the ordinals of SORTED
    /// and SORTED_SET fields may be written as [`crate::adaptive_values`]
    /// blocks: the `Adaptive90` format ([`Self::rust_adaptive`]), which Lucene
    /// cannot read.
    adaptive: bool,
    /// The block shift BINARY values are written in as
    /// [`crate::substring`] blocks: the `Substring90` format
    /// ([`Self::rust_substring`]), which Lucene cannot read.
    substring: Option<u8>,
    /// Writes format version 3 (Lucene 10.6): sparse fields' `IndexedDISI` in
    /// a `.dvp` file. Off writes version 2, which Lucene 10.5 reads as well.
    disi_file: bool,
}

/// The files of one written doc-values format instance.
#[derive(Debug)]
pub struct DocValuesFiles {
    pub dvm: Vec<u8>,
    pub dvd: Vec<u8>,
    pub dvs: Vec<u8>,
    /// `Some` for format version 3 ([`Lucene90DocValuesFormat::with_disi_file`]).
    pub dvp: Option<Vec<u8>>,
}

impl Lucene90DocValuesFormat {
    /// This format writing Lucene 10.6's version 3 (sparse fields'
    /// `IndexedDISI` in a `.dvp` file) when `on`, version 2 otherwise.
    pub fn with_disi_file(self, on: bool) -> Self {
        Lucene90DocValuesFormat {
            disi_file: on,
            ..self
        }
    }

    /// Whether this format writes a `.dvp` file.
    pub fn has_disi_file(&self) -> bool {
        self.disi_file
    }

    fn write_version(&self) -> i32 {
        if self.disi_file {
            VERSION_DISI_SEPARATE_FILE
        } else {
            VERSION_WRITE_DEFAULT
        }
    }

    /// `Lucene90DocValuesFormat.DEFAULT_SKIP_INDEX_INTERVAL_SIZE`.
    pub const DEFAULT_SKIP_INDEX_INTERVAL_SIZE: i32 = 4096;
    /// `Lucene90DocValuesFormat.getName()`.
    pub const NAME: &'static str = "Lucene90";

    /// `new Lucene90DocValuesFormat(skipIndexIntervalSize)`; below 2 is
    /// Java's `IllegalArgumentException`.
    pub fn new(skip_index_interval_size: i32) -> WriteResult<Self> {
        if skip_index_interval_size < 2 {
            return Err(WriteError::InvalidSkipIndexIntervalSize(
                skip_index_interval_size,
            ));
        }
        Ok(Lucene90DocValuesFormat {
            skip_index_interval_size,
            instance: 0,
            name: Self::NAME,
            adaptive: false,
            substring: None,
            disi_file: false,
        })
    }

    /// `getName()` of [`Self::rust_adaptive`].
    pub const RUST_ADAPTIVE_NAME: &'static str = "Adaptive90";

    /// The `Adaptive90` format: Lucene90's files, with each NUMERIC or
    /// SORTED_NUMERIC value array written as [`crate::adaptive_values`]
    /// blocks whenever that is smaller. Stock Lucene has no format by this
    /// name, so it refuses such a field rather than misreading it; the
    /// plugin registers a reader for it on the JVM side.
    pub fn rust_adaptive() -> Self {
        Lucene90DocValuesFormat {
            name: Self::RUST_ADAPTIVE_NAME,
            adaptive: true,
            ..Self::default()
        }
    }

    /// Whether this is the `Adaptive90` format.
    pub fn is_adaptive(&self) -> bool {
        self.adaptive
    }

    /// `getName()` of [`Self::rust_substring`].
    pub const RUST_SUBSTRING_NAME: &'static str = "Substring90";

    /// The `Substring90` format: Lucene90's files, with each BINARY field
    /// written as [`crate::substring`] blocks of `2^block_shift` documents
    /// (capped at [`crate::substring::MAX_BLOCK_SHIFT`]) with a trigram
    /// index, searchable by substring, wildcard and regexp. Stock Lucene has
    /// no format by this name, so it refuses such a field rather than
    /// misreading it.
    pub fn rust_substring(block_shift: u8) -> Self {
        Lucene90DocValuesFormat {
            name: Self::RUST_SUBSTRING_NAME,
            substring: Some(block_shift.min(crate::substring::MAX_BLOCK_SHIFT)),
            ..Self::default()
        }
    }

    /// The block shift of the `Substring90` format, `None` for any other.
    pub fn substring_block_shift(&self) -> Option<u8> {
        self.substring
    }

    /// The interval size this format writes skip indexes with.
    pub fn skip_index_interval_size(&self) -> i32 {
        self.skip_index_interval_size
    }

    /// `getName()`: what `PerFieldDocValuesFormat.format` records and the
    /// stem of the instance's codec suffix.
    pub fn name(&self) -> &'static str {
        self.name
    }

    /// This format's files under another format's name: a wrapper such as
    /// `AssertingDocValuesFormat` whose files are its delegate's.
    pub fn named(self, name: &'static str) -> Self {
        Lucene90DocValuesFormat { name, ..self }
    }

    /// This format as Java's `instance`th separate object with equal
    /// settings (`RandomCodec`'s two default candidates): it groups its
    /// fields apart from the others, under the next suffix of its name.
    pub fn as_instance(self, instance: u32) -> Self {
        Lucene90DocValuesFormat { instance, ..self }
    }
}

impl Default for Lucene90DocValuesFormat {
    fn default() -> Self {
        Lucene90DocValuesFormat {
            skip_index_interval_size: Self::DEFAULT_SKIP_INDEX_INTERVAL_SIZE,
            instance: 0,
            name: Self::NAME,
            adaptive: false,
            substring: None,
            disi_file: false,
        }
    }
}

/// Java: `Lucene90DocValuesConsumer.close` without any `add*Field` calls.
/// Returns `.dvm`, `.dvd` and `.dvs` buffers containing their index headers,
/// the metadata's `-1` field-list terminator and the three checksummed footers.
/// Use this explicit no-field path when closing an unused consumer.
pub fn write_no_fields(
    segment_id: &[u8; ID_LENGTH],
    segment_suffix: &str,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut meta = new_meta_output(segment_id, segment_suffix);
    let mut data = DvData::new(new_data_output(segment_id, segment_suffix), None);
    let skip_index =
        finish_field_list_and_footers(&mut meta, &mut data, segment_id, segment_suffix);
    (meta, data.buf, skip_index)
}

/// [`write_no_fields`] as `format` writes it: the files of a consumer closed
/// without a field, `.dvp` included for a format that writes one.
pub fn write_no_fields_files(
    segment_id: &[u8; ID_LENGTH],
    segment_suffix: &str,
    format: Lucene90DocValuesFormat,
) -> DocValuesFiles {
    FieldsWriter::new(&[], 0, segment_id, segment_suffix, format).close_files()
}

/// [`write_fields_with_skip_indexes`] as `format` writes it -- its skip
/// indexes close an interval after `format`'s `skipIndexIntervalSize`
/// documents.
// ARITH: as `write_dense_fields`.
#[allow(clippy::arithmetic_side_effects)]
pub fn write_fields_with_format(
    fields: &[DenseField<'_>],
    skip_index_fields: &[i32],
    max_doc: i32,
    segment_id: &[u8; ID_LENGTH],
    segment_suffix: &str,
    format: Lucene90DocValuesFormat,
) -> WriteResult<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    if format.disi_file {
        return Err(WriteError::DisiFileNotReturned);
    }
    let files = write_fields_with_format_files(
        fields,
        skip_index_fields,
        max_doc,
        segment_id,
        segment_suffix,
        format,
    )?;
    Ok((files.dvm, files.dvd, files.dvs))
}

/// [`write_fields_with_format`] returning every file, `.dvp` included.
// ARITH: as `write_dense_fields`.
#[allow(clippy::arithmetic_side_effects)]
pub fn write_fields_with_format_files(
    fields: &[DenseField<'_>],
    skip_index_fields: &[i32],
    max_doc: i32,
    segment_id: &[u8; ID_LENGTH],
    segment_suffix: &str,
    format: Lucene90DocValuesFormat,
) -> WriteResult<DocValuesFiles> {
    if fields.is_empty() {
        return Err(WriteError::EmptyFieldList);
    }
    for field in fields {
        if field.is_sparse() {
            continue;
        }
        if field.len() != max_doc as usize {
            return Err(WriteError::NotDense {
                values: field.len(),
                max_doc,
            });
        }
    }
    for i in 0..fields.len() {
        for j in (i + 1)..fields.len() {
            if fields[i].field_number() == fields[j].field_number() {
                return Err(WriteError::DuplicateFieldNumber(fields[i].field_number()));
            }
        }
    }
    // Validate multi-valued emptiness up front, before any bytes are written,
    // same "fail before touching the buffers" order the single-field
    // functions use.
    for field in fields {
        if let DenseField::SortedNumeric(_, values) = field {
            for (doc, per_doc) in values.iter().enumerate() {
                if per_doc.is_empty() {
                    return Err(WriteError::EmptyMultiValuedDoc(doc as i32));
                }
            }
        }
        if let DenseField::SortedSet(_, values) = field {
            for (doc, per_doc) in values.iter().enumerate() {
                if per_doc.is_empty() {
                    return Err(WriteError::EmptyMultiValuedDoc(doc as i32));
                }
            }
        }
    }

    let mut writer = FieldsWriter::new(
        skip_index_fields,
        max_doc,
        segment_id,
        segment_suffix,
        format,
    );
    for field in fields {
        writer.write_field(field)?;
    }
    writer.finish_files()
}

/// [`write_fields_with_format`] one field at a time, so a merge can build each
/// field's column, write it and drop it before building the next instead of
/// holding every column of the segment at once. Fields go to the `.dvm`,
/// `.dvd` and `.dvs` in the order they are written; the caller writes them in
/// ascending field number, as the slice form is given them.
pub struct FieldsWriter<'f> {
    meta: Vec<u8>,
    data: DvData,
    skip_index: Vec<u8>,
    skip_index_fields: &'f [i32],
    max_doc: i32,
    format: Lucene90DocValuesFormat,
    written: Vec<i32>,
}

impl<'f> FieldsWriter<'f> {
    pub fn new(
        skip_index_fields: &'f [i32],
        max_doc: i32,
        segment_id: &[u8; ID_LENGTH],
        segment_suffix: &str,
        format: Lucene90DocValuesFormat,
    ) -> Self {
        let version = format.write_version();
        let disi = format.disi_file.then(|| {
            let mut disi: Vec<u8> = Vec::new();
            codec_util::write_index_header(
                &mut disi,
                DISI_CODEC,
                version,
                segment_id,
                segment_suffix,
            );
            disi
        });
        Self {
            meta: new_meta_output_v(segment_id, segment_suffix, version),
            data: DvData::new(new_data_output_v(segment_id, segment_suffix, version), disi),
            skip_index: new_skip_index_output(segment_id, segment_suffix, version),
            skip_index_fields,
            max_doc,
            format,
            written: Vec::new(),
        }
    }

    /// Validates `field` as the slice writer does, then appends it.
    // ARITH: as `write_dense_fields`.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn write_field(&mut self, field: &DenseField<'_>) -> WriteResult<()> {
        let max_doc = self.max_doc;
        let field_number = field.field_number();
        if self.written.contains(&field_number) {
            return Err(WriteError::DuplicateFieldNumber(field_number));
        }
        if !field.is_sparse() && field.len() != max_doc as usize {
            return Err(WriteError::NotDense {
                values: field.len(),
                max_doc,
            });
        }
        match field {
            DenseField::SortedNumeric(_, values) => {
                if let Some(doc) = values.iter().position(|v| v.is_empty()) {
                    return Err(WriteError::EmptyMultiValuedDoc(doc as i32));
                }
            }
            DenseField::SortedSet(_, values) => {
                if let Some(doc) = values.iter().position(|v| v.is_empty()) {
                    return Err(WriteError::EmptyMultiValuedDoc(doc as i32));
                }
            }
            _ => {}
        }
        self.written.push(field_number);
        self.meta.write_i32(field_number);
        if self.skip_index_fields.contains(&field_number) {
            // The type byte, then the skip index, then the rest of the
            // entry: `addNumericField`/`doAddSortedField`/
            // `doAddSortedNumericField`'s order. The entry is written first
            // (its type byte leads it) and spliced around the skip index.
            let stream = field.skip_index_stream(max_doc)?;
            let mut entry = Vec::new();
            write_field_entry(&mut entry, &mut self.data, field, max_doc, &self.format)?;
            self.meta.push(entry[0]);
            write_skip_index(
                &mut self.meta,
                &mut self.skip_index,
                &stream,
                self.format.skip_index_interval_size,
            );
            self.meta.extend_from_slice(&entry[1..]);
        } else {
            write_field_entry(&mut self.meta, &mut self.data, field, max_doc, &self.format)?;
        }
        Ok(())
    }

    /// The finished `(dvm, dvd, dvs)`; `EmptyFieldList` when no field was written.
    ///
    /// For a format with a `.dvp` ([`Lucene90DocValuesFormat::with_disi_file`])
    /// use [`Self::finish_files`]: this refuses rather than drop the file.
    pub fn finish(self) -> WriteResult<(Vec<u8>, Vec<u8>, Vec<u8>)> {
        if self.format.disi_file {
            return Err(WriteError::DisiFileNotReturned);
        }
        let files = self.finish_files()?;
        Ok((files.dvm, files.dvd, files.dvs))
    }

    /// The finished files, `.dvp` included when the format writes one.
    pub fn finish_files(self) -> WriteResult<DocValuesFiles> {
        if self.written.is_empty() {
            return Err(WriteError::EmptyFieldList);
        }
        Ok(self.close_files())
    }

    fn close_files(mut self) -> DocValuesFiles {
        self.meta.write_i32(-1); // field list terminator
        codec_util::write_footer(&mut self.meta);
        codec_util::write_footer(&mut self.data.buf);
        codec_util::write_footer(&mut self.skip_index);
        let dvp = self.data.disi.take().map(|mut dvp| {
            codec_util::write_footer(&mut dvp);
            dvp
        });
        DocValuesFiles {
            dvm: self.meta,
            dvd: self.data.buf,
            dvs: self.skip_index,
            dvp,
        }
    }
}

fn new_skip_index_output(
    segment_id: &[u8; ID_LENGTH],
    segment_suffix: &str,
    version: i32,
) -> Vec<u8> {
    let mut skip_index: Vec<u8> = Vec::new();
    codec_util::write_index_header(
        &mut skip_index,
        SKIP_INDEX_META_CODEC,
        version,
        segment_id,
        segment_suffix,
    );
    skip_index
}

/// One field's `.dvm` entry after its field number -- the type byte and the
/// body -- and its `.dvd` data.
// ARITH: as `write_dense_fields`.
#[allow(clippy::arithmetic_side_effects)]
fn write_field_entry(
    meta: &mut Vec<u8>,
    data: &mut DvData,
    field: &DenseField<'_>,
    max_doc: i32,
    format: &Lucene90DocValuesFormat,
) -> WriteResult<()> {
    let adaptive = format.adaptive;
    if let Some(shift) = format.substring {
        match field {
            DenseField::Binary(_, values) => {
                let docs: Vec<i32> = (0..values.len() as i32).collect();
                meta.push(DOC_VALUES_TYPE_BINARY);
                write_substring_entry_body(meta, data, &docs, |i| &values[i], max_doc, shift);
                return Ok(());
            }
            DenseField::SparseBinary(_, pairs) => {
                let sorted = sort_sparse_pairs(pairs, max_doc)?;
                let docs: Vec<i32> = sorted.iter().map(|(doc, _)| *doc).collect();
                meta.push(DOC_VALUES_TYPE_BINARY);
                write_substring_entry_body(meta, data, &docs, |i| &sorted[i].1, max_doc, shift);
                return Ok(());
            }
            DenseField::BinaryColumn(_, column) => {
                check_ascending_docs(&column.docs, max_doc)?;
                let mut starts = Vec::with_capacity(column.lengths.len() + 1);
                let mut at = 0usize;
                for &l in &column.lengths {
                    starts.push(at);
                    at = at.saturating_add(l as usize);
                }
                starts.push(at);
                meta.push(DOC_VALUES_TYPE_BINARY);
                write_substring_entry_body(
                    meta,
                    data,
                    &column.docs,
                    |i| &column.bytes[starts[i]..starts[i + 1]],
                    max_doc,
                    shift,
                );
                return Ok(());
            }
            _ => {}
        }
    }
    {
        match field {
            DenseField::Numeric(_, values) => {
                meta.push(DOC_VALUES_TYPE_NUMERIC);
                write_dense_numeric_entry_body_with(meta, data, values, adaptive, true, false);
            }
            DenseField::SparseNumeric(_, pairs) => {
                let mut sorted: Vec<(i32, i64)> = pairs.to_vec();
                sorted.sort_unstable_by_key(|&(doc, _)| doc);
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
                let doc_ids: Vec<i32> = sorted.iter().map(|&(doc, _)| doc).collect();
                let values: Vec<i64> = sorted.iter().map(|&(_, v)| v).collect();
                meta.push(DOC_VALUES_TYPE_NUMERIC);
                if doc_ids.is_empty() {
                    write_empty_docs_with_field(meta);
                    write_numeric_values_body(meta, data, &values);
                } else if adaptive {
                    write_numeric_entry_body_with(meta, data, &doc_ids, &values, max_doc, true);
                } else if doc_ids.len() == max_doc as usize {
                    // `Lucene90DocValuesConsumer.writeValues` writes the
                    // `[-1, 0]` dense marker when `docsWithField.cardinality()
                    // == maxDoc`, with no `IndexedDISI` region at all. A
                    // caller handing every document a value through this
                    // variant would otherwise get bytes Lucene never writes --
                    // readable, but a different encoding of the same column,
                    // which is exactly the kind of divergence a round-trip
                    // through this port's own reader cannot see.
                    write_dense_numeric_entry_body(meta, data, &values);
                } else {
                    write_sparse_numeric_entry_body(meta, data, &doc_ids, &values);
                }
            }
            DenseField::Binary(_, values) => {
                meta.push(DOC_VALUES_TYPE_BINARY);
                write_dense_binary_entry_body(meta, data, values, max_doc);
            }
            DenseField::Sorted(_, values) => {
                meta.push(DOC_VALUES_TYPE_SORTED);
                let (dict, ords) = build_sorted_dict_and_ords(values);
                write_dense_numeric_entry_body_with(meta, data, &ords, adaptive, true, true);
                write_terms_dict(meta, data, &dict);
            }
            DenseField::SortedNumeric(_, values) => {
                meta.push(DOC_VALUES_TYPE_SORTED_NUMERIC);
                write_dense_sorted_numeric_entry_body(meta, data, values, adaptive);
            }
            DenseField::SortedSet(_, values) => {
                meta.push(DOC_VALUES_TYPE_SORTED_SET);
                write_dense_sorted_set_entry_body(meta, data, values, adaptive);
            }
            DenseField::SparseBinary(_, pairs) => {
                let sorted = sort_sparse_pairs(pairs, max_doc)?;
                let doc_ids: Vec<i32> = sorted.iter().map(|(doc, _)| *doc).collect();
                let values: Vec<Vec<u8>> = sorted.into_iter().map(|(_, v)| v).collect();
                meta.push(DOC_VALUES_TYPE_BINARY);
                write_binary_entry_body(meta, data, &doc_ids, &values, max_doc);
            }
            DenseField::SparseSorted(_, pairs) => {
                let sorted = sort_sparse_pairs(pairs, max_doc)?;
                let doc_ids: Vec<i32> = sorted.iter().map(|(doc, _)| *doc).collect();
                let values: Vec<Vec<u8>> = sorted.into_iter().map(|(_, v)| v).collect();
                let (dict, ords) = build_sorted_dict_and_ords(&values);
                meta.push(DOC_VALUES_TYPE_SORTED);
                write_numeric_entry_body_as(meta, data, &doc_ids, &ords, max_doc, adaptive, true);
                write_terms_dict(meta, data, &dict);
            }
            DenseField::SparseSortedNumeric(_, pairs) => {
                let sorted = sort_sparse_pairs(pairs, max_doc)?;
                for (doc, per_doc) in &sorted {
                    if per_doc.is_empty() {
                        return Err(WriteError::EmptyMultiValuedDoc(*doc));
                    }
                }
                let doc_ids: Vec<i32> = sorted.iter().map(|(doc, _)| *doc).collect();
                let per_doc: Vec<Vec<i64>> = sorted.into_iter().map(|(_, v)| v).collect();
                meta.push(DOC_VALUES_TYPE_SORTED_NUMERIC);
                write_sorted_numeric_entry_body(meta, data, &doc_ids, &per_doc, max_doc, adaptive);
            }
            DenseField::SparseSortedSet(_, pairs) => {
                let sorted = sort_sparse_pairs(pairs, max_doc)?;
                for (doc, per_doc) in &sorted {
                    if per_doc.is_empty() {
                        return Err(WriteError::EmptyMultiValuedDoc(*doc));
                    }
                }
                let doc_ids: Vec<i32> = sorted.iter().map(|(doc, _)| *doc).collect();
                let per_doc: Vec<Vec<Vec<u8>>> = sorted.into_iter().map(|(_, v)| v).collect();
                meta.push(DOC_VALUES_TYPE_SORTED_SET);
                write_sorted_set_entry_body(meta, data, &doc_ids, &per_doc, max_doc, adaptive);
            }
            DenseField::SortedOrds(_, column) => {
                check_ascending_docs(&column.docs, max_doc)?;
                meta.push(DOC_VALUES_TYPE_SORTED);
                write_numeric_entry_body_as(
                    meta,
                    data,
                    &column.docs,
                    &column.ords,
                    max_doc,
                    adaptive,
                    true,
                );
                write_terms_dict(meta, data, &column.dict);
            }
            DenseField::SortedSetOrds(_, column) => {
                check_ascending_docs(&column.docs, max_doc)?;
                if let Some(k) = column.counts.iter().position(|&c| c == 0) {
                    return Err(WriteError::EmptyMultiValuedDoc(column.docs[k]));
                }
                meta.push(DOC_VALUES_TYPE_SORTED_SET);
                write_sorted_set_ords_entry_body(meta, data, column, max_doc, adaptive);
            }
            DenseField::SortedNumericColumn(_, column) => {
                check_ascending_docs(&column.docs, max_doc)?;
                if let Some(k) = column.counts.iter().position(|&c| c == 0) {
                    return Err(WriteError::EmptyMultiValuedDoc(column.docs[k]));
                }
                meta.push(DOC_VALUES_TYPE_SORTED_NUMERIC);
                write_numeric_entry_body_with(
                    meta,
                    data,
                    &column.docs,
                    &column.values,
                    max_doc,
                    adaptive,
                );
                meta.write_i32(column.docs.len() as i32); // numDocsWithField
                if column.docs.len() != column.values.len() {
                    write_address_array(
                        meta,
                        data,
                        &end_offsets(column.counts.iter().map(|&c| c as usize)),
                    );
                }
            }
            DenseField::BinaryColumn(_, column) => {
                check_ascending_docs(&column.docs, max_doc)?;
                meta.push(DOC_VALUES_TYPE_BINARY);
                write_binary_column_body(meta, data, column, max_doc);
            }
        }
    }
    Ok(())
}

/// `Lucene90DocValuesConsumer.SkipAccumulator`.
#[derive(Clone, Copy)]
struct SkipAccumulator {
    min_doc_id: i32,
    max_doc_id: i32,
    doc_count: i32,
    min_value: i64,
    max_value: i64,
}

// ARITH: counters over in-memory documents, each below `max_doc`.
#[allow(clippy::arithmetic_side_effects)]
impl SkipAccumulator {
    fn new(doc: i32) -> Self {
        Self {
            min_doc_id: doc,
            max_doc_id: doc,
            doc_count: 0,
            min_value: i64::MAX,
            max_value: i64::MIN,
        }
    }

    /// `isDone`: an interval closes once it holds `skipIndexIntervalSize`
    /// documents, unless the next document keeps it a dense run of one
    /// single value.
    fn is_done(&self, interval: i32, value_count: usize, next_value: i64, next_doc: i32) -> bool {
        if self.doc_count < interval {
            return false;
        }
        value_count > 1
            || self.min_value != self.max_value
            || self.min_value != next_value
            || self.doc_count != next_doc - self.min_doc_id
    }

    fn accumulate(&mut self, value: i64) {
        self.min_value = self.min_value.min(value);
        self.max_value = self.max_value.max(value);
    }

    fn next_doc(&mut self, doc: i32) {
        self.max_doc_id = doc;
        self.doc_count += 1;
    }

    /// `SkipAccumulator.merge(list, index, length)`.
    fn merge(list: &[SkipAccumulator]) -> Self {
        let mut acc = Self::new(list[0].min_doc_id);
        for other in list {
            acc.max_doc_id = other.max_doc_id;
            acc.min_value = acc.min_value.min(other.min_value);
            acc.max_value = acc.max_value.max(other.max_value);
            acc.doc_count += other.doc_count;
        }
        acc
    }
}

/// Port of `Lucene90DocValuesConsumer.writeSkipIndex`: intervals of about
/// `interval` (`skipIndexIntervalSize`) documents with their doc-id and value ranges,
/// grouped into up to `SKIP_INDEX_MAX_LEVEL` levels of `2^LEVEL_SHIFT`, into
/// `skip_index`; the field's summary into `meta`.
// ARITH: counters and offsets over in-memory data; `skip_index` only grows.
#[allow(clippy::arithmetic_side_effects)]
fn write_skip_index(
    meta: &mut Vec<u8>,
    skip_index: &mut Vec<u8>,
    stream: &SortedNumericColumn,
    interval: i32,
) {
    let start = skip_index.len() as i64;
    let mut global_max_value = i64::MIN;
    let mut global_min_value = i64::MAX;
    let mut global_doc_count = 0i32;
    let mut global_max_value_count = 0i32;
    let mut max_doc_id = -1i32;
    let max_accumulators =
        1usize << (SKIP_INDEX_LEVEL_SHIFT * (u32::from(SKIP_INDEX_MAX_LEVEL) - 1));
    let mut accumulators: Vec<SkipAccumulator> = Vec::new();
    let mut current: Option<SkipAccumulator> = None;
    let mut at = 0usize;
    for (&doc, &count) in stream.docs.iter().zip(&stream.counts) {
        let value_count = count as usize;
        let values = &stream.values[at..at + value_count];
        at += value_count;
        let Some(&first_value) = values.first() else {
            continue;
        };
        global_max_value_count = global_max_value_count.max(value_count as i32);
        if let Some(acc) = current {
            if acc.is_done(interval, value_count, first_value, doc) {
                global_max_value = global_max_value.max(acc.max_value);
                global_min_value = global_min_value.min(acc.min_value);
                global_doc_count += acc.doc_count;
                max_doc_id = acc.max_doc_id;
                accumulators.push(acc);
                current = None;
                if accumulators.len() == max_accumulators {
                    write_skip_levels(skip_index, &accumulators);
                    accumulators.clear();
                }
            }
        }
        let acc = current.get_or_insert_with(|| SkipAccumulator::new(doc));
        acc.next_doc(doc);
        for &v in values {
            acc.accumulate(v);
        }
    }
    if let Some(acc) = current {
        global_max_value = global_max_value.max(acc.max_value);
        global_min_value = global_min_value.min(acc.min_value);
        global_doc_count += acc.doc_count;
        max_doc_id = acc.max_doc_id;
        accumulators.push(acc);
    }
    if !accumulators.is_empty() {
        write_skip_levels(skip_index, &accumulators);
    }
    meta.write_i64(start);
    meta.write_i64(skip_index.len() as i64 - start);
    meta.write_i64(global_max_value);
    meta.write_i64(global_min_value);
    meta.write_i32(global_doc_count);
    meta.write_i32(max_doc_id);
    meta.write_i32(global_max_value_count);
}

/// `writeLevels`: each interval preceded by its level count, then its
/// intervals from the highest level down.
// ARITH: indices into in-memory lists.
#[allow(clippy::arithmetic_side_effects)]
fn write_skip_levels(skip_index: &mut Vec<u8>, accumulators: &[SkipAccumulator]) {
    let level_size = 1usize << SKIP_INDEX_LEVEL_SHIFT;
    let mut levels: Vec<Vec<SkipAccumulator>> = vec![accumulators.to_vec()];
    for i in 0..usize::from(SKIP_INDEX_MAX_LEVEL - 1) {
        let below = &levels[i];
        let built: Vec<SkipAccumulator> = below
            .chunks_exact(level_size)
            .map(SkipAccumulator::merge)
            .collect();
        levels.push(built);
    }
    let total = accumulators.len();
    for index in 0..total {
        let count = skip_levels_for(index, total);
        skip_index.push(count as u8);
        for level in (0..count).rev() {
            let acc = levels[level][index >> (SKIP_INDEX_LEVEL_SHIFT as usize * level)];
            skip_index.write_i32(acc.max_doc_id);
            skip_index.write_i32(acc.min_doc_id);
            skip_index.write_i64(acc.max_value);
            skip_index.write_i64(acc.min_value);
            skip_index.write_i32(acc.doc_count);
        }
    }
}

/// `getLevels`.
// ARITH: `index < size`, both list lengths.
#[allow(clippy::arithmetic_side_effects)]
fn skip_levels_for(index: usize, size: usize) -> usize {
    let shift = SKIP_INDEX_LEVEL_SHIFT as usize;
    if index == 0 || index.trailing_zeros() as usize >= shift {
        let left = size - index;
        for level in (1..usize::from(SKIP_INDEX_MAX_LEVEL)).rev() {
            let intervals = 1usize << (shift * level);
            if left >= intervals && index.is_multiple_of(intervals) {
                return level + 1;
            }
        }
    }
    1
}

/// Sorts a sparse `(doc_id, value)` list by doc id and rejects a duplicate or
/// out-of-range doc, the validation every `Sparse*` variant shares.
// ARITH: the loop starts at 1, so `i - 1` never underflows.
#[allow(clippy::arithmetic_side_effects)]
fn sort_sparse_pairs<V: Clone>(pairs: &[(i32, V)], max_doc: i32) -> WriteResult<Vec<(i32, V)>> {
    let mut sorted: Vec<(i32, V)> = pairs.to_vec();
    sorted.sort_unstable_by_key(|(doc, _)| *doc);
    for i in 1..sorted.len() {
        if sorted[i - 1].0 == sorted[i].0 {
            return Err(WriteError::DocIdsNotAscending(sorted[i].0));
        }
    }
    for (doc, _) in &sorted {
        if *doc < 0 || *doc >= max_doc {
            return Err(WriteError::DocIdOutOfRange(*doc, max_doc));
        }
    }
    Ok(sorted)
}

/// The docs-with-field header in whichever of `Lucene90DocValuesConsumer`'s
/// three shapes applies: `-2` when no document has a value, `-1` when every
/// one does, an [`indexed_disi`] region otherwise.
///
/// The `-1` collapse is not cosmetic. Java writes it on
/// `numDocsWithField == maxDoc` (`addBinaryField`, and `writeValues` for the
/// other four types), so a sparse writer that emitted a full `IndexedDISI`
/// region for a column that happens to be complete would produce bytes real
/// Lucene never writes -- readable, but a different encoding of the same
/// column, which a round-trip through this port's own reader cannot see.
/// `DenseField::SparseNumeric` already did this; the other four now do too.
// ARITH: writer-side arithmetic over data already in memory: `data.len() as
// i64 - offset` subtracts an offset recorded from an earlier `data.len()` of
// the same, only-ever-appended buffer.
#[allow(clippy::arithmetic_side_effects)]
fn write_docs_with_field(meta: &mut Vec<u8>, data: &mut DvData, doc_ids: &[i32], max_doc: i32) {
    if doc_ids.is_empty() {
        write_empty_docs_with_field(meta);
    } else if doc_ids.len() == max_doc as usize {
        meta.write_i64(DOCS_WITH_FIELD_DENSE);
        meta.write_i64(0); // docsWithFieldLength
        meta.write_i16(-1); // jumpTableEntryCount
        meta.push(0xFF); // denseRankPower (-1 as u8)
    } else {
        let (disi_bytes, jump_table_entry_count) = indexed_disi::write_with_dense_rank_power(
            doc_ids,
            indexed_disi::DEFAULT_DENSE_RANK_POWER,
        );
        // `docsWithFieldLength` spans the jump table too -- it is what the
        // reader subtracts the table's bytes from (`createBlockSlice`).
        let (docs_with_field_offset, docs_with_field_length) = data.write_disi(&disi_bytes);
        meta.write_i64(docs_with_field_offset);
        meta.write_i64(docs_with_field_length);
        meta.write_i16(jump_table_entry_count);
        meta.push(indexed_disi::DEFAULT_DENSE_RANK_POWER);
    }
}

/// The `DirectMonotonicWriter` address array trailer shared by BINARY's
/// variable-length values and both multi-valued types' per-doc ranges.
// ARITH: as `write_docs_with_field`.
#[allow(clippy::arithmetic_side_effects)]
fn write_address_array(meta: &mut Vec<u8>, data: &mut Vec<u8>, ends: &[i64]) {
    let block_shift = DIRECT_MONOTONIC_BLOCK_SHIFT;
    let (addr_meta, addr_data) = direct_monotonic::write(ends, block_shift);
    let addresses_offset = data.len() as i64;
    data.extend_from_slice(&addr_data);
    let addresses_length = data.len() as i64 - addresses_offset;
    meta.write_i64(addresses_offset);
    meta.write_vint(block_shift as i32);
    meta.extend_from_slice(&addr_meta);
    meta.write_i64(addresses_length);
}

/// Exclusive prefix sums of a per-element length, `[0, l0, l0+l1, ...]` --
/// the `ends` array `write_address_array` encodes.
// ARITH: a running sum of in-memory slice lengths, bounded by `isize::MAX`.
#[allow(clippy::arithmetic_side_effects)]
fn end_offsets(lengths: impl ExactSizeIterator<Item = usize>) -> Vec<i64> {
    let mut ends: Vec<i64> = Vec::with_capacity(lengths.len().saturating_add(1));
    let mut end = 0i64;
    ends.push(0);
    for l in lengths {
        end += l as i64;
        ends.push(end);
    }
    ends
}

/// A NUMERIC entry body for a column that may be sparse, its values as adaptive blocks when
/// `adaptive` and that is smaller, with [`crate::adaptive_zones`] statistics
/// when every document has exactly one value.
fn write_numeric_entry_body_with(
    meta: &mut Vec<u8>,
    data: &mut DvData,
    doc_ids: &[i32],
    values: &[i64],
    max_doc: i32,
    adaptive: bool,
) {
    write_numeric_entry_body_as(meta, data, doc_ids, values, max_doc, adaptive, false);
}

/// [`write_numeric_entry_body_with`] for the ordinals of a SORTED or
/// SORTED_SET field when `ords`: see [`ORDS_SLACK_PERCENT`].
fn write_numeric_entry_body_as(
    meta: &mut Vec<u8>,
    data: &mut DvData,
    doc_ids: &[i32],
    values: &[i64],
    max_doc: i32,
    adaptive: bool,
    ords: bool,
) {
    write_docs_with_field(meta, data, doc_ids, max_doc);
    // Dense with one value per document: value `i` is document `i`'s.
    let per_doc = doc_ids.len() == max_doc as usize && values.len() == doc_ids.len();
    write_numeric_values_body_with(meta, data, values, adaptive, per_doc, ords);
}

/// [`write_dense_numeric_entry_body`], its values as adaptive blocks when
/// `adaptive` and that is smaller, with [`crate::adaptive_zones`] statistics
/// when `per_doc` (value `i` is document `i`'s only value).
fn write_dense_numeric_entry_body_with(
    meta: &mut Vec<u8>,
    data: &mut Vec<u8>,
    values: &[i64],
    adaptive: bool,
    per_doc: bool,
    ords: bool,
) {
    if adaptive {
        meta.write_i64(DOCS_WITH_FIELD_DENSE);
        meta.write_i64(0);
        meta.write_i16(-1); // jumpTableEntryCount
        meta.push(0xFF); // denseRankPower (-1 as u8)
        write_numeric_values_body_with(meta, data, values, true, per_doc, ords);
    } else {
        write_dense_numeric_entry_body(meta, data, values);
    }
}

/// [`write_numeric_values_body`], or -- when `adaptive` and smaller -- the
/// same values as [`crate::adaptive_values`] blocks behind an
/// [`ADAPTIVE_TABLE_SIZE`] entry. Both are written and the larger dropped:
/// Lucene90's table and constant encodings still win on columns with few
/// distinct values. With `zones` (the values are the documents in order),
/// the adaptive entry also carries [`crate::adaptive_zones`] statistics, as
/// an [`ADAPTIVE_ZONES_TABLE_SIZE`] entry.
// ARITH: lengths and offsets of in-memory, append-only buffers.
#[allow(clippy::arithmetic_side_effects)]
fn write_numeric_values_body_with(
    meta: &mut Vec<u8>,
    data: &mut Vec<u8>,
    values: &[i64],
    adaptive: bool,
    zones: bool,
    ords: bool,
) {
    let (meta_start, data_start) = (meta.len(), data.len());
    write_numeric_values_body(meta, data, values);
    if !adaptive || values.len() < crate::adaptive_values::BLOCK_SIZE {
        return;
    }
    let lucene90 = (meta.len() - meta_start) + (data.len() - data_start);
    let (blocks, addresses) = crate::adaptive_values::encode(values);
    let (addr_meta, addr_data) = direct_monotonic::write(&addresses, ADAPTIVE_ADDRESS_BLOCK_SHIFT);
    // numValues, tableSize, bpv, min, gcd, offsets and lengths, block shift.
    let fixed = 8 + 4 + 1 + 8 * 5 + 5 + 16;
    let adaptive_len = fixed + addr_meta.len() + blocks.len() + addr_data.len();
    // Ordinals may cost a little more than Lucene90's packing: the blocks
    // and statistics are what whole-window folds read.
    let budget = if ords {
        lucene90 + lucene90 * ORDS_SLACK_PERCENT / 100
    } else {
        lucene90.saturating_sub(1)
    };
    if adaptive_len > budget {
        return;
    }
    // The statistics only where adaptive blocks still fit with them.
    let zone_bytes = if zones {
        crate::adaptive_zones::encode_with(values, !ords)
            .filter(|z| adaptive_len + z.len() <= budget)
    } else {
        None
    };
    if ords && zone_bytes.is_none() && adaptive_len >= lucene90 {
        // Without statistics an ordinal array must be strictly smaller.
        return;
    }
    meta.truncate(meta_start);
    data.truncate(data_start);
    meta.write_i64(values.len() as i64);
    meta.write_i32(if zone_bytes.is_some() {
        ADAPTIVE_ZONES_TABLE_SIZE
    } else {
        ADAPTIVE_TABLE_SIZE
    });
    meta.push(0xFF); // bitsPerValue: per block
    meta.write_i64(values.iter().copied().min().unwrap_or(0));
    meta.write_i64(1); // gcd: per block
    let values_offset = data.len() as i64;
    data.extend_from_slice(&blocks);
    meta.write_i64(values_offset);
    meta.write_i64(data.len() as i64 - values_offset);
    let addresses_offset = data.len() as i64;
    data.extend_from_slice(&addr_data);
    let addresses_length = data.len() as i64 - addresses_offset;
    // The block-address descriptor (see `AdaptiveEntry`), where Lucene90
    // keeps a varying-width field's jump table.
    meta.write_i64(data.len() as i64);
    data.write_vint(ADAPTIVE_ADDRESS_BLOCK_SHIFT as i32);
    data.extend_from_slice(&addr_meta);
    data.write_i64(addresses_offset);
    data.write_i64(addresses_length);
    if let Some(z) = zone_bytes {
        data.extend_from_slice(&z);
    }
}

/// A BINARY entry body in the `Substring90` layout: Lucene90's prefix
/// (data region, docs with a value, value count) with
/// [`crate::substring::MIN_LENGTH_MARKER`] as `minLength`, then the
/// substring metadata.
// ARITH: lengths and offsets of in-memory, append-only buffers.
#[allow(clippy::arithmetic_side_effects)]
fn write_substring_entry_body<'v>(
    meta: &mut Vec<u8>,
    data: &mut DvData,
    doc_ids: &[i32],
    value: impl Fn(usize) -> &'v [u8],
    max_doc: i32,
    block_shift: u8,
) {
    let max_length = (0..doc_ids.len())
        .map(|i| value(i).len())
        .max()
        .unwrap_or(0) as i32;
    let data_offset = data.len() as i64;
    let substring_meta = crate::substring::write_data(data, doc_ids, value, max_doc, block_shift);
    meta.write_i64(data_offset);
    meta.write_i64(data.len() as i64 - data_offset);
    write_docs_with_field(meta, data, doc_ids, max_doc);
    meta.write_i32(doc_ids.len() as i32);
    meta.write_i32(crate::substring::MIN_LENGTH_MARKER);
    meta.write_i32(max_length);
    meta.extend_from_slice(&substring_meta);
}

/// A BINARY entry body for a column that may be sparse -- the sparse
/// counterpart of [`write_dense_binary_entry_body`], sharing every decision
/// it makes except the docs-with-field header.
// ARITH: as `write_docs_with_field`.
#[allow(clippy::arithmetic_side_effects)]
fn write_binary_entry_body(
    meta: &mut Vec<u8>,
    data: &mut DvData,
    doc_ids: &[i32],
    values: &[Vec<u8>],
    max_doc: i32,
) {
    let data_offset = data.len() as i64;
    for v in values {
        data.extend_from_slice(v);
    }
    let data_length = data.len() as i64 - data_offset;
    meta.write_i64(data_offset);
    meta.write_i64(data_length);

    write_docs_with_field(meta, data, doc_ids, max_doc);

    meta.write_i32(values.len() as i32); // numDocsWithField
                                         // Java's `minLength`/`maxLength` trackers start at
                                         // `(Integer.MAX_VALUE, 0)` and stay there when no document has
                                         // a value, which is what `addBinaryField` then writes. Nothing
                                         // reads them back (the `-2` marker means no doc ever resolves
                                         // to an ordinal), but matching keeps the bytes identical to
                                         // real Lucene's -- and, because `maxLength > minLength` is
                                         // then false, no address array is written.
    let min_length = values
        .iter()
        .map(|v| v.len() as i32)
        .min()
        .unwrap_or(i32::MAX);
    let max_length = values.iter().map(|v| v.len() as i32).max().unwrap_or(0);
    meta.write_i32(min_length);
    meta.write_i32(max_length);

    if min_length < max_length {
        write_address_array(meta, data, &end_offsets(values.iter().map(|v| v.len())));
    }
}

/// [`write_binary_entry_body`] for a [`BinaryColumn`]: the same bytes, from
/// one flat buffer instead of a vector per document.
// ARITH: writer-side offsets into buffers already in memory.
#[allow(clippy::arithmetic_side_effects)]
fn write_binary_column_body(
    meta: &mut Vec<u8>,
    data: &mut DvData,
    column: &BinaryColumn,
    max_doc: i32,
) {
    let data_offset = data.len() as i64;
    data.extend_from_slice(&column.bytes);
    meta.write_i64(data_offset);
    meta.write_i64(column.bytes.len() as i64);
    write_docs_with_field(meta, data, &column.docs, max_doc);
    meta.write_i32(column.docs.len() as i32); // numDocsWithField
                                              // Java's trackers start at `(Integer.MAX_VALUE, 0)`; see
                                              // `write_binary_entry_body`.
    let min_length = column
        .lengths
        .iter()
        .map(|&l| l as i32)
        .min()
        .unwrap_or(i32::MAX);
    let max_length = column.lengths.iter().map(|&l| l as i32).max().unwrap_or(0);
    meta.write_i32(min_length);
    meta.write_i32(max_length);
    if min_length < max_length {
        write_address_array(
            meta,
            data,
            &end_offsets(column.lengths.iter().map(|&l| l as usize)),
        );
    }
}

/// A SORTED_NUMERIC entry body for a column that may be sparse.
fn write_sorted_numeric_entry_body(
    meta: &mut Vec<u8>,
    data: &mut DvData,
    doc_ids: &[i32],
    per_doc: &[Vec<i64>],
    max_doc: i32,
    adaptive: bool,
) {
    let flat: Vec<i64> = per_doc.iter().flatten().copied().collect();
    write_numeric_entry_body_with(meta, data, doc_ids, &flat, max_doc, adaptive);

    let num_docs_with_field = per_doc.len() as i32;
    meta.write_i32(num_docs_with_field);
    if num_docs_with_field as i64 != flat.len() as i64 {
        write_address_array(meta, data, &end_offsets(per_doc.iter().map(|v| v.len())));
    }
}

/// `docs` strictly ascending and every one below `max_doc` -- what the
/// `Sparse*` variants establish by sorting a copy, checked in place for the
/// columns that arrive already in doc order.
fn check_ascending_docs(docs: &[i32], max_doc: i32) -> WriteResult<()> {
    for w in docs.windows(2) {
        if w[0] >= w[1] {
            return Err(WriteError::DocIdsNotAscending(w[1]));
        }
    }
    if let Some(&doc) = docs.iter().find(|&&d| d < 0 || d >= max_doc) {
        return Err(WriteError::DocIdOutOfRange(doc, max_doc));
    }
    Ok(())
}

/// `Lucene90DocValuesConsumer.addSortedSetField` for a column whose
/// dictionary is already built ([`SortedSetOrdsColumn`]). As in Java, a
/// column in which every document has exactly one ordinal is written in the
/// SORTED shape (`multiValued = 0`, `isSingleValued`), dense or sparse;
/// otherwise the flattened ordinals, the docs-with-field count and, when some
/// document has more than one, the address array.
// ARITH: writer-side sums of per-document counts of an in-memory column.
#[allow(clippy::arithmetic_side_effects)]
fn write_sorted_set_ords_entry_body(
    meta: &mut Vec<u8>,
    data: &mut DvData,
    column: &SortedSetOrdsColumn,
    max_doc: i32,
    adaptive: bool,
) {
    let ords = |meta: &mut Vec<u8>, data: &mut DvData| {
        write_numeric_entry_body_as(
            meta,
            data,
            &column.docs,
            &column.ords,
            max_doc,
            adaptive,
            true,
        );
    };
    if column.counts.iter().all(|&c| c == 1) {
        meta.push(0); // multiValued = false: the plain SORTED shape.
        ords(meta, data);
    } else {
        meta.push(1); // multiValued = true.
        ords(meta, data);
        let num_docs_with_field = column.docs.len() as i32;
        meta.write_i32(num_docs_with_field);
        if column.docs.len() != column.ords.len() {
            write_address_array(
                meta,
                data,
                &end_offsets(column.counts.iter().map(|&c| c as usize)),
            );
        }
    }
    write_terms_dict(meta, data, &column.dict);
}

/// A SORTED_SET entry body for a column that may be sparse.
///
/// As in Java (`addSortedSetField`'s `isSingleValued`), a column in which
/// every present document has exactly one ordinal takes the plain SORTED
/// shape (`multiValued = 0`). That is not only a size optimization: Java's
/// reader never expects the multi-valued shape without an address array --
/// which is what "every document has one ordinal" writes -- and dereferences
/// the missing addresses (`Lucene90DocValuesProducer.getSortedSet`'s
/// single-block specialization). The docs-with-field collapse is applied
/// too -- see [`write_docs_with_field`].
fn write_sorted_set_entry_body(
    meta: &mut Vec<u8>,
    data: &mut DvData,
    doc_ids: &[i32],
    per_doc: &[Vec<Vec<u8>>],
    max_doc: i32,
    adaptive: bool,
) {
    let mut dict: Vec<Vec<u8>> = per_doc.iter().flatten().cloned().collect();
    dict.sort_unstable();
    dict.dedup();

    let per_doc_ords: Vec<Vec<i64>> = per_doc
        .iter()
        .map(|values| {
            let mut ords: Vec<i64> = values
                .iter()
                .map(|v| dict.binary_search(v).unwrap() as i64)
                .collect();
            ords.sort_unstable();
            ords.dedup();
            ords
        })
        .collect();

    let flat: Vec<i64> = per_doc_ords.iter().flatten().copied().collect();
    if per_doc_ords.iter().all(|ords| ords.len() == 1) {
        meta.push(0); // multiValued = false: the plain SORTED shape.
        write_numeric_entry_body_as(meta, data, doc_ids, &flat, max_doc, adaptive, true);
        write_terms_dict(meta, data, &dict);
        return;
    }
    meta.push(1); // multiValued = true.
    write_numeric_entry_body_as(meta, data, doc_ids, &flat, max_doc, adaptive, true);

    let num_docs_with_field = per_doc_ords.len() as i32;
    meta.write_i32(num_docs_with_field);
    if num_docs_with_field as i64 != flat.len() as i64 {
        write_address_array(
            meta,
            data,
            &end_offsets(per_doc_ords.iter().map(|v| v.len())),
        );
    }

    write_terms_dict(meta, data, &dict);
}

/// The BINARY entry body [`write_single_dense_binary_field`] writes after its
/// leading field-number/type byte -- extracted so [`write_dense_fields`] can
/// share it verbatim.
// ARITH: writer-side arithmetic over data already in memory. Every length is
// a `Vec`/slice length (so at most `isize::MAX`), every running offset is a
// prefix sum of such lengths into a buffer that has already been built, and
// every `data.len() as i64 - X` subtracts an offset recorded from an earlier
// `data.len()` of the same, only-ever-appended buffer. Nothing here comes off
// disk -- this is the encode half.
#[allow(clippy::arithmetic_side_effects)]
fn write_dense_binary_entry_body(
    meta: &mut Vec<u8>,
    data: &mut Vec<u8>,
    values: &[Vec<u8>],
    max_doc: i32,
) {
    let data_offset = data.len() as i64;
    for v in values {
        data.extend_from_slice(v);
    }
    let data_length = data.len() as i64 - data_offset;
    meta.write_i64(data_offset);
    meta.write_i64(data_length);

    // numDocsWithValue == maxDoc: [-1, 0], no IndexedDISI structure.
    meta.write_i64(DOCS_WITH_FIELD_DENSE);
    meta.write_i64(0);
    meta.write_i16(-1); // jumpTableEntryCount
    meta.push(0xFF); // denseRankPower (-1 as u8)

    meta.write_i32(max_doc);
    let min_length = values.iter().map(|v| v.len() as i32).min().unwrap_or(0);
    let max_length = values.iter().map(|v| v.len() as i32).max().unwrap_or(0);
    meta.write_i32(min_length);
    meta.write_i32(max_length);

    if min_length < max_length {
        let mut end = 0i64;
        let mut ends: Vec<i64> = Vec::with_capacity(values.len() + 1);
        ends.push(0);
        for v in values {
            end += v.len() as i64;
            ends.push(end);
        }
        let block_shift = DIRECT_MONOTONIC_BLOCK_SHIFT;
        let (addr_meta, addr_data) = direct_monotonic::write(&ends, block_shift);
        let addresses_offset = data.len() as i64;
        data.extend_from_slice(&addr_data);
        let addresses_length = data.len() as i64 - addresses_offset;

        meta.write_i64(addresses_offset);
        meta.write_vint(block_shift as i32);
        meta.extend_from_slice(&addr_meta);
        meta.write_i64(addresses_length);
    }
}

/// The SORTED_NUMERIC entry body [`write_single_dense_sorted_numeric_field`]
/// writes after its leading field-number/type byte -- extracted so
/// [`write_dense_fields`] can share it verbatim. Caller must already have
/// checked every `values[doc]` is non-empty.
// ARITH: writer-side arithmetic over data already in memory. Every length is
// a `Vec`/slice length (so at most `isize::MAX`), every running offset is a
// prefix sum of such lengths into a buffer that has already been built, and
// every `data.len() as i64 - X` subtracts an offset recorded from an earlier
// `data.len()` of the same, only-ever-appended buffer. Nothing here comes off
// disk -- this is the encode half.
#[allow(clippy::arithmetic_side_effects)]
fn write_dense_sorted_numeric_entry_body(
    meta: &mut Vec<u8>,
    data: &mut Vec<u8>,
    values: &[Vec<i64>],
    adaptive: bool,
) {
    let num_docs_with_field = values.len() as i32;
    let flat: Vec<i64> = values.iter().flatten().copied().collect();

    let per_doc = flat.len() == values.len();
    write_dense_numeric_entry_body_with(meta, data, &flat, adaptive, per_doc, false);

    meta.write_i32(num_docs_with_field);
    if num_docs_with_field as i64 != flat.len() as i64 {
        let mut end = 0i64;
        let mut ends: Vec<i64> = Vec::with_capacity(values.len() + 1);
        ends.push(0);
        for per_doc in values {
            end += per_doc.len() as i64;
            ends.push(end);
        }
        let block_shift = DIRECT_MONOTONIC_BLOCK_SHIFT;
        let (addr_meta, addr_data) = direct_monotonic::write(&ends, block_shift);
        let addresses_offset = data.len() as i64;
        data.extend_from_slice(&addr_data);
        let addresses_length = data.len() as i64 - addresses_offset;

        meta.write_i64(addresses_offset);
        meta.write_vint(block_shift as i32);
        meta.extend_from_slice(&addr_meta);
        meta.write_i64(addresses_length);
    }
}

/// The SORTED_SET entry body [`write_single_dense_sorted_set_field`] writes
/// after its leading field-number/type byte -- extracted so
/// [`write_dense_fields`] can share it verbatim. Caller must already have
/// checked every `values[doc]` is non-empty.
// ARITH: writer-side arithmetic over data already in memory. Every length is
// a `Vec`/slice length (so at most `isize::MAX`), every running offset is a
// prefix sum of such lengths into a buffer that has already been built, and
// every `data.len() as i64 - X` subtracts an offset recorded from an earlier
// `data.len()` of the same, only-ever-appended buffer. Nothing here comes off
// disk -- this is the encode half.
#[allow(clippy::arithmetic_side_effects)]
fn write_dense_sorted_set_entry_body(
    meta: &mut Vec<u8>,
    data: &mut DvData,
    values: &[Vec<Vec<u8>>],
    adaptive: bool,
) {
    let all_values: Vec<&[u8]> = values.iter().flatten().map(Vec::as_slice).collect();
    let (dict, value_ords) = sorted_dict_and_ords(&all_values);

    // Each document's ordinals, ascending and distinct, one run after
    // another; `ends[d + 1]` is where document `d`'s run ends.
    let mut flat: Vec<i64> = Vec::with_capacity(value_ords.len());
    let mut ends: Vec<i64> = Vec::with_capacity(values.len().saturating_add(1));
    ends.push(0);
    let mut all_single = true;
    let mut doc_ords: Vec<i64> = Vec::new();
    let mut rest = value_ords.as_slice();
    for per_doc in values {
        let (this_doc, tail) = rest.split_at(per_doc.len().min(rest.len()));
        rest = tail;
        doc_ords.clear();
        doc_ords.extend_from_slice(this_doc);
        doc_ords.sort_unstable();
        doc_ords.dedup();
        all_single &= doc_ords.len() == 1;
        flat.extend_from_slice(&doc_ords);
        ends.push(flat.len() as i64);
    }

    let num_docs_with_field = values.len() as i32;

    if all_single {
        meta.push(0); // multiValued = false: plain SORTED shape.
        write_dense_numeric_entry_body_with(meta, data, &flat, adaptive, true, true);
    } else {
        meta.push(1); // multiValued = true.
        write_dense_numeric_entry_body_with(meta, data, &flat, adaptive, false, true);

        meta.write_i32(num_docs_with_field);
        if num_docs_with_field as i64 != flat.len() as i64 {
            let block_shift = DIRECT_MONOTONIC_BLOCK_SHIFT;
            let (addr_meta, addr_data) = direct_monotonic::write(&ends, block_shift);
            let addresses_offset = data.len() as i64;
            data.extend_from_slice(&addr_data);
            let addresses_length = data.len() as i64 - addresses_offset;

            meta.write_i64(addresses_offset);
            meta.write_vint(block_shift as i32);
            meta.extend_from_slice(&addr_meta);
            meta.write_i64(addresses_length);
        }
    }

    write_terms_dict(meta, data, &dict);
}

/// Writes just the NUMERIC entry body (`addNumericField` -> `writeValues`,
/// the `numDocsWithValue == maxDoc` / no-blocks branches, feeding
/// `writeValuesSingleBlock`) into an already-open meta/data pair -- shared by
/// [`write_single_dense_numeric_field`] (a standalone NUMERIC field) and
/// [`write_single_dense_sorted_numeric_field`] (whose per-doc value counts
/// collapse to this exact same flat layout). Does **not** write the leading
/// `field_number`/type byte -- callers that need those (a bare NUMERIC
/// field) write them first, exactly as [`read_numeric_entry`] expects them
/// already consumed by its caller.
///
/// Always dense (`docsWithFieldOffset = -1`, i.e. `values[i]` is doc `i`'s
/// value / doc `i`'s rank into a shared value array); the per-value encoding
/// itself (plain delta, GCD-delta, or table compression) is chosen by
/// [`write_numeric_values_body`] exactly like real Lucene -- see that
/// function's doc comment. Varying-bits-per-value blocks are still deferred
/// -- see [`write_single_dense_numeric_field`]'s doc comment for the full
/// scope statement, which applies here identically.
fn write_dense_numeric_entry_body(meta: &mut Vec<u8>, data: &mut Vec<u8>, values: &[i64]) {
    // numDocsWithValue == maxDoc: meta[-1, 0], no IndexedDISI structure.
    meta.write_i64(DOCS_WITH_FIELD_DENSE);
    meta.write_i64(0);
    meta.write_i16(-1); // jumpTableEntryCount
    meta.push(0xFF); // denseRankPower (-1 as u8)

    write_numeric_values_body(meta, data, values);
}

/// Writes a **sparse** NUMERIC entry body: an [`indexed_disi`]-backed
/// docs-with-field structure (only `doc_ids` have a value, out of `max_doc`
/// total docs) followed by the same per-rank value encoding
/// [`write_dense_numeric_entry_body`] uses for per-doc values -- a sparse
/// field's value array is indexed by *rank among present docs*, not by doc
/// id, exactly what [`numeric_value`]'s sparse branch already expects
/// (`indexed_disi::rank_of` then [`decode_value`] at that rank).
///
/// `doc_ids` must be strictly ascending and every id `< max_doc` -- the
/// caller's job, same contract [`indexed_disi::write`] documents. For a
/// single-valued caller (plain NUMERIC), `values.len() == doc_ids.len()`
/// (each `values[i]` is `doc_ids[i]`'s value). Multi-valued callers (SORTED_
/// NUMERIC) instead pass every doc's values flattened in doc-id order, so
/// `values.len() >= doc_ids.len()` -- `values` only feeds
/// [`write_numeric_values_body`]'s own record count below, which never
/// assumes it lines up 1:1 with `doc_ids`.
///
/// Writes the DENSE rank table at [`indexed_disi::DEFAULT_DENSE_RANK_POWER`],
/// which is what real Lucene's `Lucene90DocValuesConsumer` does
/// (`IndexedDISI.writeBitSet(it, data)` defaults to power 9). It costs 256
/// bytes per DENSE block and turns a cold random lookup inside one from ~844
/// word reads into one rank read plus at most 8 -- measured at ~420 ns vs
/// ~16 ns.
/// `Lucene90DocValuesConsumer`'s `numDocsWithValue == 0` branch, shared by
/// `writeValues` and `addBinaryField`: `meta[-2, 0]`, "no documents with
/// values". Written **instead of** an [`indexed_disi`] structure -- a
/// zero-document DISI has no blocks at all, and a reader that took the sparse
/// path over it would rank-index into an empty value array rather than
/// short-circuiting on [`NumericEntry::is_empty_field`]/[`BinaryEntry::is_empty_field`].
///
/// `denseRankPower` is Java's `(byte) -1` here, not
/// [`indexed_disi::DEFAULT_DENSE_RANK_POWER`]: there is no DISI to describe.
fn write_empty_docs_with_field(meta: &mut Vec<u8>) {
    meta.write_i64(DOCS_WITH_FIELD_EMPTY);
    meta.write_i64(0); // docsWithFieldLength
    meta.write_i16(-1); // jumpTableEntryCount
    meta.push(0xFF); // denseRankPower (-1 as u8)
}

// ARITH: writer-side arithmetic over data already in memory. Every length is
// a `Vec`/slice length (so at most `isize::MAX`), every running offset is a
// prefix sum of such lengths into a buffer that has already been built, and
// every `data.len() as i64 - X` subtracts an offset recorded from an earlier
// `data.len()` of the same, only-ever-appended buffer. Nothing here comes off
// disk -- this is the encode half.
#[allow(clippy::arithmetic_side_effects)]
fn write_sparse_numeric_entry_body(
    meta: &mut Vec<u8>,
    data: &mut DvData,
    doc_ids: &[i32],
    values: &[i64],
) {
    let (disi_bytes, jump_table_entry_count) =
        indexed_disi::write_with_dense_rank_power(doc_ids, indexed_disi::DEFAULT_DENSE_RANK_POWER);
    // `docsWithFieldLength` spans the jump table too -- it is what the reader
    // subtracts the table's bytes from (`createBlockSlice`).
    let (docs_with_field_offset, docs_with_field_length) = data.write_disi(&disi_bytes);

    meta.write_i64(docs_with_field_offset);
    meta.write_i64(docs_with_field_length);
    meta.write_i16(jump_table_entry_count);
    meta.push(indexed_disi::DEFAULT_DENSE_RANK_POWER); // denseRankPower

    write_numeric_values_body(meta, data, values);
}

/// Euclidean GCD of two `i64`s (matches `MathUtil.gcd`'s contract: always
/// non-negative, `gcd(0, x) == |x|`, and `i64::MIN` read as `2^63`, which the
/// `as i64` turns back into `i64::MIN` exactly as Java's does). In `u64`:
/// `unsigned_abs` of every `i64` fits, and a 64-bit remainder is several
/// times cheaper than the `u128` one this once used. Euclid rather than
/// `MathUtil.gcd`'s binary loop: once the running GCD is small, as in any
/// field worth GCD-encoding, each value costs one remainder.
// ARITH: `b != 0` is the loop condition, so the remainder never divides by
// zero; both operands are unsigned.
#[allow(clippy::arithmetic_side_effects)]
fn gcd_i64(a: i64, b: i64) -> i64 {
    let (mut a, mut b) = (a.unsigned_abs(), b.unsigned_abs());
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a as i64
}

/// `Lucene90DocValuesConsumer.writeValues`'s GCD accumulation: the greatest
/// common divisor of every value's offset from the first one, or `0` when
/// every value is identical (including an empty array).
///
/// Real Lucene's overflow guard only checks the *current* value against
/// `[MIN/2, MAX/2]` -- but Java's `long` subtraction wraps silently on
/// overflow, while Rust's default arithmetic panics in debug builds. If the
/// first value itself falls outside that safe range (e.g. `i64::MIN`) while a
/// later value is in-range, `v - first_value` can still overflow even though
/// the per-`v` guard passed. `wrapping_sub` reproduces Java's exact
/// (silently-wrapping, "gcd may come out numerically odd but never crashes")
/// behaviour instead of panicking -- a correctness-irrelevant edge case Java
/// itself doesn't handle "correctly" either, just without a hard crash.
fn compute_gcd(values: &[i64]) -> i64 {
    let Some(&first_value) = values.first() else {
        return 0;
    };
    let mut gcd: i64 = 0;
    for &v in values {
        if gcd == 1 {
            break;
        }
        if !(i64::MIN / 2..=i64::MAX / 2).contains(&v) {
            gcd = 1;
        } else {
            gcd = gcd_i64(gcd, v.wrapping_sub(first_value));
        }
    }
    gcd
}

/// `Lucene90DocValuesFormat.NUMERIC_BLOCK_SHIFT`.
const NUMERIC_BLOCK_SHIFT: u32 = 14;
/// `Lucene90DocValuesFormat.NUMERIC_BLOCK_SIZE`.
const NUMERIC_BLOCK_SIZE: usize = 1 << NUMERIC_BLOCK_SHIFT;

/// `MinMaxTracker.finish`'s `spaceInBits` for one run of values: the bits
/// one uniform width over `[min, max]` needs, 0 for a constant run.
// ARITH: `max - min` wraps exactly as Java's `long` subtraction does for a
// pathological range; a width of at most 64 times a slice length cannot
// overflow an `i64`.
#[allow(clippy::arithmetic_side_effects)]
fn space_in_bits(values: &[i64]) -> i64 {
    let (Some(&min), Some(&max)) = (values.iter().min(), values.iter().max()) else {
        return 0;
    };
    if max > min {
        i64::from(direct_reader::unsigned_bits_required(max.wrapping_sub(min)))
            * values.len() as i64
    } else {
        0
    }
}

/// `Lucene90DocValuesConsumer.writeValuesMultipleBlocks`: every
/// `NUMERIC_BLOCK_SIZE` values as one `writeBlock`, then the jump table (each
/// block's absolute `.dvd` offset, then the table's own offset). Returns the
/// jump table's offset.
fn write_values_multiple_blocks(data: &mut Vec<u8>, values: &[i64], gcd: i64) -> i64 {
    let mut offsets = Vec::with_capacity(values.len().div_ceil(NUMERIC_BLOCK_SIZE));
    for block in values.chunks(NUMERIC_BLOCK_SIZE) {
        offsets.push(data.len() as i64);
        write_numeric_block(data, block, gcd);
    }
    let origo = data.len() as i64;
    for offset in offsets {
        data.write_i64(offset);
    }
    data.write_i64(origo);
    origo
}

/// `Lucene90DocValuesConsumer.writeBlock`: a constant block as `0u8` and its
/// value; otherwise its width, its own min, the packed length and the
/// `DirectWriter`-packed `(v - min) / gcd`.
// ARITH: `gcd` divides every `v - min` of the column (it is the column's
// GCD, never 0 or -1 on this path: the column is not constant); the
// subtraction wraps as Java's does; the packed length is an in-memory size.
#[allow(clippy::arithmetic_side_effects)]
fn write_numeric_block(data: &mut Vec<u8>, block: &[i64], gcd: i64) {
    let min = block.iter().copied().min().unwrap_or(0);
    let max = block.iter().copied().max().unwrap_or(0);
    if min == max {
        data.push(0);
        data.write_i64(min);
        return;
    }
    let bits_per_value = direct_reader::unsigned_bits_required(max.wrapping_sub(min) / gcd);
    let raw: Vec<i64> = block.iter().map(|&v| v.wrapping_sub(min) / gcd).collect();
    let mut packed = direct_reader::encode(&raw, bits_per_value);
    packed.extend(std::iter::repeat_n(
        0u8,
        direct_reader::padding_bytes_needed(bits_per_value),
    ));
    data.push(bits_per_value);
    data.write_i64(min);
    data.write_i32(packed.len() as i32);
    data.extend_from_slice(&packed);
}

fn new_meta_output(segment_id: &[u8; ID_LENGTH], segment_suffix: &str) -> Vec<u8> {
    new_meta_output_v(segment_id, segment_suffix, VERSION_WRITE_DEFAULT)
}

fn new_meta_output_v(segment_id: &[u8; ID_LENGTH], segment_suffix: &str, version: i32) -> Vec<u8> {
    let mut meta: Vec<u8> = Vec::new();
    codec_util::write_index_header(&mut meta, META_CODEC, version, segment_id, segment_suffix);
    meta
}

fn new_data_output(segment_id: &[u8; ID_LENGTH], segment_suffix: &str) -> Vec<u8> {
    new_data_output_v(segment_id, segment_suffix, VERSION_WRITE_DEFAULT)
}

fn new_data_output_v(segment_id: &[u8; ID_LENGTH], segment_suffix: &str, version: i32) -> Vec<u8> {
    let mut data: Vec<u8> = Vec::new();
    codec_util::write_index_header(
        &mut data,
        DATA_META_CODEC,
        version,
        segment_id,
        segment_suffix,
    );
    data
}

fn finish_field_list_and_footers(
    meta: &mut Vec<u8>,
    data: &mut Vec<u8>,
    segment_id: &[u8; ID_LENGTH],
    segment_suffix: &str,
) -> Vec<u8> {
    meta.write_i32(-1); // field list terminator
    codec_util::write_footer(meta);
    codec_util::write_footer(data);

    let mut skip_index: Vec<u8> = Vec::new();
    codec_util::write_index_header(
        &mut skip_index,
        SKIP_INDEX_META_CODEC,
        VERSION_WRITE_DEFAULT,
        segment_id,
        segment_suffix,
    );
    codec_util::write_footer(&mut skip_index);
    skip_index
}

/// Port of `Lucene90DocValuesConsumer`, scoped to exactly one shape: **a
/// single NUMERIC field, DENSE** (every doc from `0` to `max_doc - 1` has a
/// value) -- the `numDocsWithValue == maxDoc` branch of `writeValues`.
/// The per-value encoding itself (plain delta, GCD-delta, table
/// compression, or varying-bits-per-value blocks) is chosen by
/// [`write_numeric_values_body`] exactly like real Lucene's own
/// `uniqueValues`/`gcd` logic -- see that function's doc comment.
///
/// Sparse NUMERIC fields are handled by the sibling
/// [`write_single_sparse_numeric_field`], not here.
///
/// Skip indexes and multiple fields in one `.dvm`/`.dvd`/`.dvs` triple are
/// [`write_fields_with_skip_indexes`]'s. BINARY
/// ([`write_single_dense_binary_field`]), SORTED_NUMERIC
/// ([`write_single_dense_sorted_numeric_field`]), SORTED
/// ([`write_single_dense_sorted_field`]), and SORTED_SET
/// ([`write_single_dense_sorted_set_field`]) write sides are siblings of this
/// function, all built on this same dense scope. Sparse BINARY
/// ([`write_single_sparse_binary_field`]), sparse SORTED
/// ([`write_single_sparse_sorted_field`]), sparse SORTED_NUMERIC
/// ([`write_single_sparse_sorted_numeric_field`]), and sparse SORTED_SET
/// ([`write_single_sparse_sorted_set_field`]) writers cover the other types.
///
/// Returns `(meta_bytes, data_bytes, skip_index_bytes)` -- three separate
/// buffers matching the real writer's three `IndexOutput`s (`.dvm`, `.dvd`,
/// `.dvs`); `.dvs` is always just a header+footer here since no field
/// written here ever has a skip index (`Lucene90DocValuesProducer`
/// unconditionally opens `.dvs` once `VERSION_CURRENT >=
/// VERSION_SKIPPER_SEPARATE_FILE`, which this port's `VERSION_CURRENT`
/// always is, so it must exist and pass header/footer checks even when
/// empty).
pub fn write_single_dense_numeric_field(
    field_number: i32,
    values: &[i64],
    max_doc: i32,
    segment_id: &[u8; ID_LENGTH],
    segment_suffix: &str,
) -> WriteResult<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    write_dense_fields(
        &[DenseField::Numeric(field_number, values)],
        max_doc,
        segment_id,
        segment_suffix,
    )
}

/// Port of `Lucene90DocValuesConsumer.addNumericField`'s **sparse** branch
/// (`numDocsWithValue != maxDoc`, feeding an [`indexed_disi`]-backed
/// docs-with-field structure instead of the `-1`/DENSE marker
/// [`write_single_dense_numeric_field`] always writes); see that function's
/// doc comment for the rest of this module's scope. The sparse siblings are
/// [`write_single_sparse_binary_field`], [`write_single_sparse_sorted_field`],
/// [`write_single_sparse_sorted_numeric_field`] and
/// [`write_single_sparse_sorted_set_field`]. [`indexed_disi::write`] makes
/// Lucene's SPARSE-vs-DENSE choice per 65536-doc block, so a field spanning
/// several blocks at varying densities exercises all three on-disk block
/// shapes; only the jump table and DENSE rank table are never written, both
/// pure random-access speedups.
///
/// `doc_values` need not be sorted by the caller; this function sorts a
/// clone by doc id itself. Each doc id must be unique and `< max_doc`.
// ARITH: writer-side arithmetic over data already in memory. Every length is
// a `Vec`/slice length (so at most `isize::MAX`), every running offset is a
// prefix sum of such lengths into a buffer that has already been built, and
// every `data.len() as i64 - X` subtracts an offset recorded from an earlier
// `data.len()` of the same, only-ever-appended buffer. Nothing here comes off
// disk -- this is the encode half.
#[allow(clippy::arithmetic_side_effects)]
pub fn write_single_sparse_numeric_field(
    field_number: i32,
    doc_values: &[(i32, i64)],
    max_doc: i32,
    segment_id: &[u8; ID_LENGTH],
    segment_suffix: &str,
) -> WriteResult<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    let mut sorted: Vec<(i32, i64)> = doc_values.to_vec();
    sorted.sort_unstable_by_key(|&(doc, _)| doc);
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

    let doc_ids: Vec<i32> = sorted.iter().map(|&(doc, _)| doc).collect();
    let values: Vec<i64> = sorted.iter().map(|&(_, v)| v).collect();

    let mut meta = new_meta_output(segment_id, segment_suffix);
    let mut data = DvData::new(new_data_output(segment_id, segment_suffix), None);

    meta.write_i32(field_number);
    meta.push(DOC_VALUES_TYPE_NUMERIC);
    if doc_ids.is_empty() {
        write_empty_docs_with_field(&mut meta);
        write_numeric_values_body(&mut meta, &mut data, &values);
    } else {
        write_sparse_numeric_entry_body(&mut meta, &mut data, &doc_ids, &values);
    }

    let skip_index =
        finish_field_list_and_footers(&mut meta, &mut data, segment_id, segment_suffix);
    Ok((meta, data.buf, skip_index))
}

/// Port of `Lucene90DocValuesConsumer.addBinaryField`, scoped to exactly one
/// shape: **a single BINARY field, DENSE** (every doc from `0` to `max_doc -
/// 1` has a value, one `Vec<u8>` per doc in `values`, empty slices allowed --
/// an empty value is still a present value, distinct from "no value").
/// Handles both length shapes real Lucene distinguishes: **fixed-length**
/// (every value the same length -- no address array, `ordinal * length`
/// indexing, matching [`BinaryEntry::is_fixed_length`]) and
/// **variable-length** (a [`crate::direct_monotonic`] end-offset array,
/// [`write`](direct_monotonic::write) with
/// [`DIRECT_MONOTONIC_BLOCK_SHIFT`], the same 16 real Lucene uses).
///
/// Deliberately not attempted here: per-field doc-values skip indexes, and
/// multiple fields in one `.dvm`/`.dvd`/`.dvs` triple. Sparse BINARY fields
/// (`IndexedDISI`) are no longer out of scope -- see
/// [`write_single_sparse_binary_field`] below.
///
/// Returns `(meta_bytes, data_bytes, skip_index_bytes)`, same shape as
/// [`write_single_dense_numeric_field`].
pub fn write_single_dense_binary_field(
    field_number: i32,
    values: &[Vec<u8>],
    max_doc: i32,
    segment_id: &[u8; ID_LENGTH],
    segment_suffix: &str,
) -> WriteResult<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    write_dense_fields(
        &[DenseField::Binary(field_number, values)],
        max_doc,
        segment_id,
        segment_suffix,
    )
}

/// Port of `Lucene90DocValuesConsumer.addBinaryField`'s **sparse** branch
/// (`numDocsWithValue != maxDoc`), the BINARY analogue of
/// [`write_single_sparse_numeric_field`]: an [`indexed_disi`]-backed
/// docs-with-field structure instead of the `-1`/DENSE marker
/// [`write_single_dense_binary_field`] always writes, followed by the value
/// bytes for *only* the docs that have one -- no placeholder bytes for
/// missing docs, matching real Lucene exactly (a sparse field's value array
/// is indexed by rank among present docs, not by doc id, same as sparse
/// NUMERIC's value array).
///
/// `doc_values` need not be sorted by the caller; this function sorts a
/// clone by doc id itself, same contract as
/// [`write_single_sparse_numeric_field`]. Each doc id must be unique and
/// `< max_doc`. Empty `Vec<u8>` values are allowed (an empty value is still
/// present, distinct from absent).
///
/// Handles both length shapes real Lucene distinguishes, same as
/// [`write_single_dense_binary_field`]: fixed-length (no address array,
/// `rank * length` indexing) and variable-length ([`direct_monotonic`]
/// end-offset array over the present docs' values, in rank order).
// ARITH: writer-side arithmetic over data already in memory. Every length is
// a `Vec`/slice length (so at most `isize::MAX`), every running offset is a
// prefix sum of such lengths into a buffer that has already been built, and
// every `data.len() as i64 - X` subtracts an offset recorded from an earlier
// `data.len()` of the same, only-ever-appended buffer. Nothing here comes off
// disk -- this is the encode half.
#[allow(clippy::arithmetic_side_effects)]
pub fn write_single_sparse_binary_field(
    field_number: i32,
    doc_values: &[(i32, Vec<u8>)],
    max_doc: i32,
    segment_id: &[u8; ID_LENGTH],
    segment_suffix: &str,
) -> WriteResult<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    write_dense_fields(
        &[DenseField::SparseBinary(field_number, doc_values)],
        max_doc,
        segment_id,
        segment_suffix,
    )
}

/// Port of `Lucene90DocValuesConsumer.addSortedNumericField`, scoped to
/// exactly one shape: **a single SORTED_NUMERIC field, DENSE** (every doc
/// from `0` to `max_doc - 1` has **at least one** value -- `values[doc]`
/// must be non-empty, [`WriteError::EmptyMultiValuedDoc`] otherwise; zero
/// values for a doc would need `IndexedDISI`, out of scope here same as
/// every other sparse case in this module). Flattens `values` into one flat
/// array (Java's shared value array across all docs) and writes the address
/// range per doc via [`direct_monotonic::write`] -- **except** when every doc
/// has exactly one value, in which case real Lucene's `readSortedNumeric`
/// collapses the address array away entirely (a doc's rank already *is* its
/// value's index), so this function detects that case and omits the address
/// array too, to stay byte-compatible with what the read side actually
/// expects (`num_docs_with_field == numeric.num_values` is not a stored
/// flag -- the read side infers "no addresses" from that equality, so
/// writing one when it holds would desync the two sides).
///
/// Deliberately not attempted here, same as [`write_single_dense_numeric_field`]:
/// sparse per-doc value *presence* (every doc must have >= 1 value),
/// per-field doc-values skip indexes, and multiple fields in one
/// `.dvm`/`.dvd`/`.dvs` triple.
///
/// Returns `(meta_bytes, data_bytes, skip_index_bytes)`, same shape as
/// [`write_single_dense_numeric_field`].
pub fn write_single_dense_sorted_numeric_field(
    field_number: i32,
    values: &[Vec<i64>],
    segment_id: &[u8; ID_LENGTH],
    segment_suffix: &str,
) -> WriteResult<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    let max_doc = values.len() as i32;
    write_dense_fields(
        &[DenseField::SortedNumeric(field_number, values)],
        max_doc,
        segment_id,
        segment_suffix,
    )
}

/// Port of `Lucene90DocValuesConsumer.addSortedNumericField`'s **sparse**
/// branch (`numDocsWithField != maxDoc`), the SORTED_NUMERIC analogue of
/// [`write_single_sparse_numeric_field`]. Real Lucene's sparse test here is
/// specifically "does this doc have *zero* values", not "does it have
/// exactly one" -- a doc with 3 values is just as absent from the
/// docs-with-field structure's complement as a doc with 1, and a doc with 0
/// values gets no entry anywhere (no per-doc value-count slot, no address
/// range) and reads back as an empty `Vec` via the same
/// [`indexed_disi::rank_of`]-returns-`None` path
/// [`write_single_sparse_numeric_field`]'s dense-vs-sparse NUMERIC entry
/// already handles.
///
/// This reuses [`write_sparse_numeric_entry_body`] directly rather than
/// reinventing an IndexedDISI dance: that function's `doc_ids`/`values`
/// contract ("`docs_with_field`'s presence bitset over `doc_ids`, then
/// whatever flat value array `values` is") doesn't actually require
/// `values.len() == doc_ids.len()` -- that equality only happens to hold for
/// a bare sparse NUMERIC field (one value per present doc). Here `doc_ids` is
/// every doc with **at least one** value, and `values` is those docs' values
/// flattened in doc order, so `values.len() >= doc_ids.len()`, exactly
/// mirroring how [`write_single_dense_sorted_numeric_field`] feeds its own
/// (dense) `numeric` sub-entry. The `num_docs_with_field`/address-array tail
/// that follows is the same collapse rule that function uses: an address
/// array (via [`direct_monotonic`]) unless every present doc has exactly one
/// value, in which case the read side infers "no addresses" from
/// `num_docs_with_field == numeric.num_values` and this function omits the
/// array to match.
///
/// `doc_values` need not be sorted by the caller; this function sorts a
/// clone by doc id itself, same contract as
/// [`write_single_sparse_numeric_field`]. Each doc id must be unique and
/// `< max_doc`. Every doc in `doc_values` must have a non-empty value list
/// ([`WriteError::EmptyMultiValuedDoc`] otherwise) -- a doc with zero values
/// is represented by *omitting* it from `doc_values` entirely, not by
/// passing an empty `Vec` for it, since "present with zero values" isn't a
/// distinct on-disk state from "absent" in this format.
// ARITH: writer-side arithmetic over data already in memory. Every length is
// a `Vec`/slice length (so at most `isize::MAX`), every running offset is a
// prefix sum of such lengths into a buffer that has already been built, and
// every `data.len() as i64 - X` subtracts an offset recorded from an earlier
// `data.len()` of the same, only-ever-appended buffer. Nothing here comes off
// disk -- this is the encode half.
#[allow(clippy::arithmetic_side_effects)]
pub fn write_single_sparse_sorted_numeric_field(
    field_number: i32,
    doc_values: &[(i32, Vec<i64>)],
    max_doc: i32,
    segment_id: &[u8; ID_LENGTH],
    segment_suffix: &str,
) -> WriteResult<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    write_dense_fields(
        &[DenseField::SparseSortedNumeric(field_number, doc_values)],
        max_doc,
        segment_id,
        segment_suffix,
    )
}

/// Longest common byte prefix of `a` and `b` (`StringHelper.bytesDifference`,
/// scoped to the "always in order, never equal" case terms-dict callers
/// guarantee -- real Lucene's version also doubles as an out-of-order
/// assertion, not needed here since [`write_terms_dict`]'s caller already
/// sorts+dedups).
fn common_prefix_len(a: &[u8], b: &[u8]) -> usize {
    a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
}

/// Port of `Lucene90DocValuesConsumer.addSortedField`, scoped to exactly one
/// shape: **a single SORTED field, DENSE** (every doc from `0` to `max_doc -
/// 1` has a value -- `values[doc]` is that doc's raw term bytes; an empty
/// slice is a legitimate present value, same convention
/// [`write_single_dense_binary_field`] uses). Builds the sorted, deduplicated
/// distinct-value dictionary (`BTreeSet` over `values`), maps each doc to its
/// term's ordinal into that dictionary (written as a plain dense NUMERIC
/// entry via [`write_dense_numeric_entry_body`] -- exactly how real Lucene's
/// `doAddSortedField` always wraps a `SortedDocValues` as a singleton
/// `SortedNumericDocValues`, which collapses back to a bare per-doc ordinal
/// with no address array since it's single-valued), then the dictionary
/// itself via [`write_terms_dict`] (shared, byte-for-byte, with what a future
/// SORTED_SET writer would call for the same values).
///
/// Sparse SORTED fields (`IndexedDISI`, i.e. some docs with no value at all)
/// are handled by the sibling [`write_single_sparse_sorted_field`], not here.
///
/// Deliberately not attempted here, same as
/// [`write_single_dense_numeric_field`]: per-field doc-values skip indexes,
/// and multiple fields in one `.dvm`/`.dvd`/`.dvs` triple.
///
/// Returns `(meta_bytes, data_bytes, skip_index_bytes)`, same shape as
/// [`write_single_dense_numeric_field`].
pub fn write_single_dense_sorted_field(
    field_number: i32,
    values: &[Vec<u8>],
    max_doc: i32,
    segment_id: &[u8; ID_LENGTH],
    segment_suffix: &str,
) -> WriteResult<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    write_dense_fields(
        &[DenseField::Sorted(field_number, values)],
        max_doc,
        segment_id,
        segment_suffix,
    )
}

/// Builds the sorted, deduplicated distinct-value dictionary for a SORTED
/// field's raw per-doc term bytes (a `BTreeSet`-equivalent `sort_unstable` +
/// `dedup` over `values`) and maps each entry of `values` to its ordinal into
/// that dictionary. Shared by [`write_single_dense_sorted_field`] and
/// [`write_single_sparse_sorted_field`] -- in the sparse case, `values` holds
/// only the present docs' terms, in doc-id order, so the returned `ords`
/// align with those docs' *ranks*, exactly what
/// [`write_sparse_numeric_entry_body`] expects.
fn build_sorted_dict_and_ords(values: &[Vec<u8>]) -> (Vec<&[u8]>, Vec<i64>) {
    let values: Vec<&[u8]> = values.iter().map(Vec::as_slice).collect();
    sorted_dict_and_ords(&values)
}

/// [`build_sorted_dict_and_ords`] over borrowed values, as
/// `SortedDocValuesWriter` builds it: the distinct values hashed in first
/// (its `BytesRefHash`), only those sorted, every value then numbered by its
/// distinct value's place -- no value copied, none searched for.
fn sorted_dict_and_ords<'v>(values: &[&'v [u8]]) -> (Vec<&'v [u8]>, Vec<i64>) {
    let mut ids: HashMap<&[u8], u32, BuildHasherDefault<SetHasher>> = HashMap::default();
    let mut distinct: Vec<&[u8]> = Vec::new();
    let value_ids: Vec<u32> = values
        .iter()
        .map(|&v| {
            *ids.entry(v).or_insert_with(|| {
                distinct.push(v);
                u32::try_from(distinct.len().saturating_sub(1)).unwrap_or(u32::MAX)
            })
        })
        .collect();
    let mut order: Vec<usize> = (0..distinct.len()).collect();
    order.sort_unstable_by(|&a, &b| distinct[a].cmp(distinct[b]));
    let mut ord_of = vec![0i64; distinct.len()];
    for (ord, &id) in order.iter().enumerate() {
        ord_of[id] = ord as i64;
    }
    let dict = order.iter().map(|&id| distinct[id]).collect();
    let ords = value_ids.iter().map(|&id| ord_of[id as usize]).collect();
    (dict, ords)
}

/// Port of `Lucene90DocValuesConsumer.addSortedField`'s **sparse** branch
/// (`numDocsWithValue != maxDoc`), the SORTED analogue of
/// [`write_single_sparse_numeric_field`]/[`write_single_sparse_binary_field`]:
/// an [`indexed_disi`]-backed docs-with-field structure instead of the
/// `-1`/DENSE marker [`write_single_dense_sorted_field`] always writes,
/// followed by per-doc ordinals for *only* the docs that have a value --
/// missing docs get no ordinal at all, not even a placeholder, matching
/// [`write_sparse_numeric_entry_body`]'s rank-indexed (not doc-id-indexed)
/// value array. The terms dictionary itself
/// ([`build_sorted_dict_and_ords`]/[`write_terms_dict`]) is built only from
/// the present docs' values, same as real Lucene's `addSortedField`, which
/// only ever sees values for docs `DocIdSetIterator` actually advances to.
///
/// `doc_values` need not be sorted by the caller; this function sorts a
/// clone by doc id itself, same contract as
/// [`write_single_sparse_numeric_field`]. Each doc id must be unique and
/// `< max_doc`. Empty `Vec<u8>` values are allowed (an empty value is still
/// present, distinct from absent).
// ARITH: writer-side arithmetic over data already in memory. Every length is
// a `Vec`/slice length (so at most `isize::MAX`), every running offset is a
// prefix sum of such lengths into a buffer that has already been built, and
// every `data.len() as i64 - X` subtracts an offset recorded from an earlier
// `data.len()` of the same, only-ever-appended buffer. Nothing here comes off
// disk -- this is the encode half.
#[allow(clippy::arithmetic_side_effects)]
pub fn write_single_sparse_sorted_field(
    field_number: i32,
    doc_values: &[(i32, Vec<u8>)],
    max_doc: i32,
    segment_id: &[u8; ID_LENGTH],
    segment_suffix: &str,
) -> WriteResult<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    write_dense_fields(
        &[DenseField::SparseSorted(field_number, doc_values)],
        max_doc,
        segment_id,
        segment_suffix,
    )
}

/// Port of `Lucene90DocValuesConsumer.addSortedSetField`, scoped to exactly
/// one shape: **a single SORTED_SET field, DENSE** (every doc from `0` to
/// `max_doc - 1` has **at least one** value -- `values[doc]` must be
/// non-empty, [`WriteError::EmptyMultiValuedDoc`] otherwise, same
/// "every doc must have a value" contract as
/// [`write_single_dense_sorted_numeric_field`]). Builds the sorted,
/// deduplicated distinct-value dictionary across *every* value of *every*
/// doc (not one value per doc -- a doc's own value set can itself repeat a
/// value, which is deduped away per-doc too, since a sorted set never
/// stores the same ordinal twice for one doc), then writes the per-doc
/// ordinals following the exact same collapse rule
/// [`write_single_dense_sorted_numeric_field`] uses: when every doc has
/// exactly one distinct value, this is written as a plain [`SortedEntry`]
/// (`multiValued = 0`, per-doc ordinal via [`write_dense_numeric_entry_body`],
/// no address array) exactly like [`write_single_dense_sorted_field`] would;
/// otherwise as a true multi-valued form (`multiValued = 1`, flattened
/// ordinals via [`write_dense_numeric_entry_body`] plus a
/// [`direct_monotonic`] address-range array). This matches
/// [`read_sorted_set_entry`]'s own inference exactly -- it decides
/// single-vs-multi purely from the stored `multiValued` flag byte (unlike
/// [`write_single_dense_sorted_numeric_field`]'s address array, whose
/// presence the read side infers from a count equality rather than a flag),
/// and its `Multi` branch in turn infers the address array's own presence
/// from `num_docs_with_field == numeric.num_values` -- which always holds
/// exactly when every doc has one value, i.e. never in the `multiValued = 1`
/// branch this function takes (docs must have >= 1 value each, so any doc
/// with more than one forces the total above `num_docs_with_field`).
/// The dictionary itself is written via [`write_terms_dict`] (shared,
/// byte-for-byte, with [`write_single_dense_sorted_field`]).
///
/// Sparse SORTED_SET fields (some docs with zero values at all) are handled
/// by the sibling [`write_single_sparse_sorted_set_field`], not here.
///
/// Deliberately not attempted here, same as [`write_single_dense_numeric_field`]:
/// per-field doc-values skip indexes, and multiple fields in one
/// `.dvm`/`.dvd`/`.dvs` triple.
///
/// Returns `(meta_bytes, data_bytes, skip_index_bytes)`, same shape as
/// [`write_single_dense_numeric_field`].
pub fn write_single_dense_sorted_set_field(
    field_number: i32,
    values: &[Vec<Vec<u8>>],
    max_doc: i32,
    segment_id: &[u8; ID_LENGTH],
    segment_suffix: &str,
) -> WriteResult<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    write_dense_fields(
        &[DenseField::SortedSet(field_number, values)],
        max_doc,
        segment_id,
        segment_suffix,
    )
}

/// Port of `Lucene90DocValuesConsumer.addSortedSetField`'s **sparse** branch
/// (some docs have zero values at all), the SORTED_SET analogue of
/// [`write_single_sparse_sorted_numeric_field`]: an [`indexed_disi`]-backed
/// docs-with->=1-ordinal structure instead of the "every doc has a value"
/// contract [`write_single_dense_sorted_set_field`] requires, always taking
/// the `multiValued = 1` shape (this function never collapses to the
/// single-valued plain-`SortedEntry` shape
/// [`write_single_dense_sorted_set_field`] takes when every doc has exactly
/// one distinct value -- [`SortedSetKind::Multi`] decodes correctly
/// regardless of whether every present doc happens to have exactly one
/// ordinal, so that extra collapse is a size optimization real Lucene applies
/// but this port's read side doesn't require).
///
/// A doc's zero-or-more raw values are deduplicated *and* sorted into
/// ordinals, same as [`write_single_dense_sorted_set_field`] (a sorted set
/// never stores the same ordinal twice for one doc, and iterates its ordinals
/// in order) -- unlike SORTED_NUMERIC, whose values keep whatever
/// order/duplicates the caller passed. The terms dictionary is built only
/// from the present docs' values, same rationale as
/// [`write_single_sparse_sorted_field`]: real Lucene's `addSortedSetField`
/// only ever sees values for docs its iterator actually advances to. Per-doc
/// ordinal counts/addresses are written only for present docs, in rank order,
/// same [`write_sparse_numeric_entry_body`]-then-address-array structure
/// [`write_single_sparse_sorted_numeric_field`] uses for its own flattened
/// values.
///
/// `doc_values` need not be sorted by the caller; this function sorts a
/// clone by doc id itself, same contract as
/// [`write_single_sparse_sorted_numeric_field`]. Each doc id must be unique
/// and `< max_doc`. Every doc in `doc_values` must have a non-empty value set
/// ([`WriteError::EmptyMultiValuedDoc`] otherwise) -- a doc with zero values
/// is represented by *omitting* it from `doc_values` entirely, not by passing
/// an empty `Vec` for it, same "absent, not present-with-zero" contract as
/// [`write_single_sparse_sorted_numeric_field`].
// ARITH: writer-side arithmetic over data already in memory. Every length is
// a `Vec`/slice length (so at most `isize::MAX`), every running offset is a
// prefix sum of such lengths into a buffer that has already been built, and
// every `data.len() as i64 - X` subtracts an offset recorded from an earlier
// `data.len()` of the same, only-ever-appended buffer. Nothing here comes off
// disk -- this is the encode half.
#[allow(clippy::arithmetic_side_effects)]
pub fn write_single_sparse_sorted_set_field(
    field_number: i32,
    doc_values: &[(i32, Vec<Vec<u8>>)],
    max_doc: i32,
    segment_id: &[u8; ID_LENGTH],
    segment_suffix: &str,
) -> WriteResult<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    write_dense_fields(
        &[DenseField::SparseSortedSet(field_number, doc_values)],
        max_doc,
        segment_id,
        segment_suffix,
    )
}

/// Sets bits `0..n` of `bits` (which the caller cleared), a word at a time.
fn set_first_bits(bits: &mut [u64], n: usize) {
    let (whole, rest) = (n / 64, n % 64);
    for w in bits.iter_mut().take(whole) {
        *w = u64::MAX;
    }
    if rest != 0 {
        if let Some(w) = bits.get_mut(whole) {
            *w |= !(u64::MAX << rest);
        }
    }
}

#[cfg(test)]
mod tests;
