//! Port of `org.apache.lucene.codecs.lucene99.Lucene99SegmentInfoFormat` (`.si` files).
//!
//! Both directions ported: [`parse`] (read path, PLAN.md Phase 2) and
//! [`write`] (write path, PLAN.md Phase 5) are exact byte-level inverses of
//! each other.
//!
//! Wire format (all ints little-endian; header/footer per `codec_util`):
//! ```text
//! IndexHeader(codec="Lucene90SegmentInfo", version=0, id, suffix="")
//! SegVersion    --> i32 major, i32 minor, i32 bugfix
//! HasMinVersion --> u8 (0 or 1)
//! SegMinVersion --> [i32 major, i32 minor, i32 bugfix] iff HasMinVersion == 1
//! SegSize       --> i32 (maxDoc)
//! IsCompoundFile--> u8 (`SegmentInfo.YES` == 1, `SegmentInfo.NO` == -1/0xFF;
//!                    every reader tests `== YES`, so any other byte is NO)
//! HasBlocks     --> u8 (same YES/NO encoding)
//! Diagnostics   --> MapOfStrings
//! Files         --> SetOfStrings
//! Attributes    --> MapOfStrings
//! NumSortFields --> vint (0, or N for an N-field index `Sort`, priority-
//!                    ordered: SortField[0] is the primary key, SortField[1]
//!                    breaks ties in SortField[0], etc)
//! SortField     --> repeated NumSortFields times:
//!                    ProviderName (string), then that provider's own
//!                    bytestream (see below)
//! Footer
//! ```
//!
//! # Index-sort encoding (`SortFieldProvider`)
//!
//! Real Lucene writes each sort field as `writeString(provider name)` followed
//! by the bytes that provider's `writeSortField` emits
//! (`Lucene99SegmentInfoFormat.writeSegmentInfo` ->
//! `SortFieldProvider.write`). Four providers are registered in
//! `lucene-core`'s `META-INF/services/org.apache.lucene.index.SortFieldProvider`
//! and this module decodes all four:
//!
//! ```text
//! "SortField"               --> String field, String type (a `SortField.Type`
//!                               enum *name*, e.g. "LONG"), i32 reverse (1 ==
//!                               descending), i32 hasMissing; if hasMissing == 1:
//!                                 STRING -> i32 (1 == STRING_FIRST, else STRING_LAST)
//!                                 INT    -> i32   LONG   -> i64
//!                                 FLOAT  -> i32 (NumericUtils.floatToSortableInt)
//!                                 DOUBLE -> i64 (NumericUtils.doubleToSortableLong)
//! "SortedNumericSortField"  --> String field, String type, i32 reverse,
//!                               i32 selector (SortedNumericSelector.Type
//!                               ordinal: 0 == MIN, 1 == MAX), i32 hasMissing,
//!                               then the same numeric missing value as above
//! "SortedSetSortField"      --> String field, i32 reverse, i32 selector
//!                               (SortedSetSelector.Type ordinal: 0 == MIN,
//!                               1 == MAX, 2 == MIDDLE_MIN, 3 == MIDDLE_MAX),
//!                               i32 missing (0 == none, 1 == STRING_FIRST,
//!                               2 == STRING_LAST)
//! "BinarySortField"         --> String field, i32 reverse, i32 missing
//!                               (same 0/1/2 encoding as SortedSetSortField)
//! ```
//!
//! All of those ints are `DataInput.readInt` (little-endian since Lucene 9),
//! not vints.
//!
//! **What this port models.** [`IndexSortField`] is Java's `SortField`
//! as far as `SortFieldProvider` can round-trip it: all four providers, every
//! `SortField.Type` that can legally be an index sort, both selector enums,
//! `reverse`, and every form the missing value takes -- an arbitrary numeric
//! sentinel, `STRING_FIRST`/`STRING_LAST`, or **no missing value at all**
//! (which Java treats as `0` for a numeric sort and as the smallest ordinal
//! for a string one). [`write`] is the exact byte-level inverse of [`parse`]
//! for every one of them.
//!
//! Until c35 it carried only `(field, reverse, missing-first-or-last)` -- the
//! shape this port's own sort-on-flush writer produces -- and [`parse`]
//! *rejected* everything else. That was honest but it meant an index a real
//! `IndexWriter` wrote with an ordinary sort (a numeric sort with no missing
//! value, an arbitrary sentinel, a `MAX` selector, a string sort) could not
//! be **opened by this port at all**.
//!
//! Two things are still refused, and both are refused by Java too:
//!
//! - A `SortField.Type` that cannot be an index sort (`SCORE`, `DOC`,
//!   `CUSTOM`, `STRING_VAL`, `REWRITEABLE`). `SortField.serialize` throws on
//!   a missing value for them, and `IndexWriterConfig.setIndexSort` rejects
//!   any `SortField` whose `getIndexSorter()` is `null` -- which is all five
//!   -- so no `.si` real Lucene wrote can contain one.
//! - A `SortedNumericSortField` whose type is `STRING`, which is an
//!   `AssertionError` inside Java's own `serialize`.
//!
//! What this port cannot yet *act on* is narrower than what it can read, and
//! is stated per consumer rather than by refusing the file:
//! [`IndexSortField::key_comparison`] gives the comparator for every kind
//! whose per-document key is a single `i64` (the four numeric types, both
//! numeric selectors, and term ordinals for `STRING`/`SortedSetSortField`);
//! a `BinarySortField` sorts on raw bytes and has none. See `docs/parity.md`
//! for which consumer honours which.

use lucene_store::codec_util::{self, ID_LENGTH};
use lucene_store::data_input::{DataInput, SliceInput};
use lucene_store::data_output::DataOutput;

const CODEC_NAME: &str = "Lucene90SegmentInfo";
const VERSION_START: i32 = 0;
const VERSION_CURRENT: i32 = 0;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Store(#[from] lucene_store::Error),
    #[error("invalid docCount: {0}")]
    InvalidDocCount(i32),
    #[error("illegal boolean value for hasMinVersion: {0}")]
    IllegalHasMinVersion(u8),
    #[error("invalid index sort field count: {0}")]
    InvalidSortFieldCount(i32),
    #[error("unknown SortFieldProvider name: {0:?}")]
    UnknownSortFieldProvider(String),
    #[error("can't deserialize SortField - unknown type {0:?}")]
    UnknownSortFieldType(String),
    #[error("can't deserialize {provider} - unknown selector type {selector}")]
    UnknownSortSelector {
        provider: &'static str,
        selector: i32,
    },
    #[error("index sort field {field:?}: {reason}")]
    UnsupportedSortField { field: String, reason: String },
    #[error("illegal {which} version: {value}")]
    IllegalVersion { which: &'static str, value: i32 },
}

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LuceneVersion {
    pub major: i32,
    pub minor: i32,
    pub bugfix: i32,
}

/// A `SortField.Type` that can be an index sort, paired with the missing
/// value that goes with it -- Java's `SortField(field, type, reverse,
/// missingValue)` for the four numeric types, where `None` means
/// `setMissingValue` was never called.
///
/// Type and missing value are **one** enum rather than two fields because
/// Java's pairing of them is total: a `Type.INT` sort's missing value is an
/// `Integer` that `serialize` writes with `writeInt`, a `Type.DOUBLE`'s is a
/// `Double` written as `NumericUtils.doubleToSortableLong`. Two independent
/// fields would make `(INT, Some(3.5))` representable and nothing on disk
/// can hold it.
///
/// `None` is not a synonym for any sentinel: `IndexSorter.LongSorter`
/// pre-fills its comparison array with `missingValue` only when it is
/// non-null, so a document with no value compares as **`0`**, which is
/// neither first nor last. That case is exactly what this port used to
/// refuse to open.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NumericSortKey {
    /// `SortField.Type.INT`; `IndexSorter.IntSorter`, `Integer.compare`.
    Int(Option<i32>),
    /// `SortField.Type.LONG`; `IndexSorter.LongSorter`, `Long.compare`.
    Long(Option<i64>),
    /// `SortField.Type.FLOAT`; `IndexSorter.FloatSorter`, `Float.compare`.
    Float(Option<f32>),
    /// `SortField.Type.DOUBLE`; `IndexSorter.DoubleSorter`, `Double.compare`.
    Double(Option<f64>),
}

impl NumericSortKey {
    /// The `SortField.Type` enum *name* this key serializes as -- what
    /// `SortField.serialize` writes with `out.writeString(type.toString())`.
    fn type_name(&self) -> &'static str {
        match self {
            NumericSortKey::Int(_) => TYPE_INT,
            NumericSortKey::Long(_) => TYPE_LONG,
            NumericSortKey::Float(_) => TYPE_FLOAT,
            NumericSortKey::Double(_) => TYPE_DOUBLE,
        }
    }

    /// How this key's per-document value is compared, and the value a
    /// document with no value takes -- `IndexSorter.{Int,Long,Float,Double}
    /// Sorter.getDocComparator`, whose array is pre-filled with
    /// `missingValue` when it is non-null and with the JVM's zero default
    /// otherwise.
    ///
    /// The sentinel is in the same encoding the doc-values column holds:
    /// raw `Float.floatToRawIntBits`/`Double.doubleToRawLongBits`, matching
    /// `FloatDocValuesField`/`DoubleDocValuesField` and the
    /// `Float.intBitsToFloat((int) dvs.longValue())` those sorters apply.
    fn key_comparison(&self) -> (SortKeyKind, i64) {
        match *self {
            NumericSortKey::Int(m) => (SortKeyKind::Int, m.unwrap_or(0) as i64),
            NumericSortKey::Long(m) => (SortKeyKind::Long, m.unwrap_or(0)),
            NumericSortKey::Float(m) => {
                (SortKeyKind::Float, m.unwrap_or(0.0).to_bits() as i32 as i64)
            }
            NumericSortKey::Double(m) => (SortKeyKind::Double, m.unwrap_or(0.0).to_bits() as i64),
        }
    }
}

/// The missing value of a sort whose keys are *terms*: `SortField.STRING_FIRST`,
/// `SortField.STRING_LAST`, or none at all.
///
/// `IndexSorter.StringSorter` reads it as `missingValue == STRING_LAST ?
/// Integer.MAX_VALUE : Integer.MIN_VALUE`, so **`None` behaves like `First`**
/// -- the two are distinguishable on disk (`hasMissing == 0` versus an
/// explicit marker) but not in the comparator. Both are kept so [`write`]
/// reproduces the bytes it read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StringMissingValue {
    /// `setMissingValue` was never called.
    None,
    /// `SortField.STRING_FIRST`.
    First,
    /// `SortField.STRING_LAST`.
    Last,
}

impl StringMissingValue {
    /// `IndexSorter.StringSorter`'s `missingOrd`.
    fn missing_ord(self) -> i64 {
        match self {
            StringMissingValue::Last => i32::MAX as i64,
            StringMissingValue::None | StringMissingValue::First => i32::MIN as i64,
        }
    }
}

/// `SortedNumericSelector.Type` -- which of a multi-valued field's values a
/// document sorts by.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SortedNumericSelector {
    /// Ordinal 0.
    Min,
    /// Ordinal 1.
    Max,
}

/// `SortedSetSelector.Type`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SortedSetSelector {
    /// Ordinal 0.
    Min,
    /// Ordinal 1.
    Max,
    /// Ordinal 2.
    MiddleMin,
    /// Ordinal 3.
    MiddleMax,
}

/// Which `SortFieldProvider` wrote a sort field, and everything that provider
/// carries beyond the field name and `reverse` -- the two every provider has.
#[derive(Debug, Clone, PartialEq)]
pub enum IndexSortKind {
    /// `SortField` provider, one of the four numeric `SortField.Type`s, over
    /// a single-valued NUMERIC doc-values column.
    Numeric(NumericSortKey),
    /// `SortField` provider with `SortField.Type.STRING`, over a SORTED
    /// doc-values column, compared by term ordinal.
    String(StringMissingValue),
    /// `SortedNumericSortField` provider: a SORTED_NUMERIC column reduced to
    /// one value per document by `selector`, then compared as `key` says.
    SortedNumeric {
        key: NumericSortKey,
        selector: SortedNumericSelector,
    },
    /// `SortedSetSortField` provider: a SORTED_SET column reduced to one
    /// ordinal per document by `selector`.
    SortedSet {
        selector: SortedSetSelector,
        missing: StringMissingValue,
    },
    /// `BinarySortField` provider: a BINARY column compared as raw unsigned
    /// bytes. The one kind whose per-document key is not a single `i64`, so
    /// [`IndexSortField::key_comparison`] has none for it.
    Binary(StringMissingValue),
}

/// How a per-document sort key that has been read out of doc values as one
/// `i64` is compared. Each variant names the Java `compare` the corresponding
/// `IndexSorter` applies, and they differ: `Long.compare` and
/// `Float.compare` disagree on the same 64 bits, and `Integer.compare` over
/// the low 32 disagrees with both.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SortKeyKind {
    /// `Integer.compare` over `(int) value`, as `IntSorter` does after its
    /// `values[docID] = (int) dvs.longValue()`.
    Int,
    /// `Long.compare`.
    Long,
    /// `Float.compare(Float.intBitsToFloat((int) value), ...)` over a value
    /// holding `Float.floatToRawIntBits`, which is what a **NUMERIC** column
    /// holds (`FloatDocValuesField`).
    Float,
    /// `Double.compare(Double.longBitsToDouble(value), ...)`, the NUMERIC
    /// twin of [`SortKeyKind::Float`].
    Double,
    /// [`SortKeyKind::Float`] over a value holding
    /// `NumericUtils.floatToSortableInt` instead, which is what a
    /// **SORTED_NUMERIC** column holds (`FloatField` writes
    /// `SortedNumericDocValuesField(name, floatToSortableInt(value))`).
    /// `SortedNumericSelector.wrap` undoes it with
    /// `NumericUtils.sortableFloatBits` before `FloatSorter` ever sees a
    /// value; this variant is that `FilterNumericDocValues`.
    SortableFloat,
    /// The 64-bit twin of [`SortKeyKind::SortableFloat`]
    /// (`NumericUtils.sortableDoubleBits`).
    SortableDouble,
    /// `Integer.compare` over term ordinals (`StringSorter`).
    Ordinal,
}

/// One field of an index sort descriptor. Real Lucene's
/// `SegmentInfo.indexSort` is a `Sort` of one or more `SortField`s; this
/// port's [`SegmentInfo::index_sort`] is a priority-ordered, non-empty
/// `Vec<IndexSortField>` and this is one element of it.
///
/// `PartialEq` but not `Eq`: a `FLOAT`/`DOUBLE` missing value is a float, and
/// `NaN != NaN`. Nothing keys a map on a sort field, and the one comparison
/// that matters -- `IndexWriter::set_index_sort`'s congruence check against
/// an existing segment's sort -- wants exactly `PartialEq`'s answer.
#[derive(Debug, Clone, PartialEq)]
pub struct IndexSortField {
    pub field: String,
    /// `false` == ascending, `true` == descending (real Lucene's
    /// `SortField.reverse`).
    pub reverse: bool,
    pub kind: IndexSortKind,
}

impl IndexSortField {
    /// The `SortField(field, Type.LONG, reverse)` with an explicit
    /// `missingValue` that this port's own sort-on-flush and sort-preserving
    /// merge writers produce -- the whole of what [`IndexSortField`] could
    /// represent before c35, and still the common case.
    pub fn long(field: impl Into<String>, reverse: bool, missing: Option<i64>) -> Self {
        Self {
            field: field.into(),
            reverse,
            kind: IndexSortKind::Numeric(NumericSortKey::Long(missing)),
        }
    }

    /// How this sort's per-document key is compared and what a document with
    /// no value compares as, or `None` when the key is not a single `i64` --
    /// which is exactly `BinarySortField`, whose `IndexSorter.BinarySorter`
    /// compares raw `BytesRef`s.
    ///
    /// This is the whole comparator contract: a consumer reads each
    /// document's key out of the right doc-values column (applying the
    /// selector for the multi-valued kinds), substitutes the sentinel for a
    /// document that has none, compares as [`SortKeyKind`] says, and then
    /// applies `reverse` -- **including to the sentinel**, which is an
    /// ordinary value inside `reverseMul * X.compare(a, b)`.
    pub fn key_comparison(&self) -> Option<(SortKeyKind, i64)> {
        match &self.kind {
            IndexSortKind::Numeric(key) => Some(key.key_comparison()),
            // A SORTED_NUMERIC FLOAT/DOUBLE column holds
            // `NumericUtils.floatToSortableInt`/`doubleToSortableLong`, not
            // the raw bits a NUMERIC one holds, and
            // `SortedNumericSelector.wrap` undoes that before the sorter sees
            // a value. Comparing the stored form as raw bits instead would
            // reverse the whole negative half of the ordering.
            IndexSortKind::SortedNumeric { key, .. } => Some(match *key {
                NumericSortKey::Float(m) => (
                    SortKeyKind::SortableFloat,
                    float_to_sortable_int(m.unwrap_or(0.0)) as i64,
                ),
                NumericSortKey::Double(m) => (
                    SortKeyKind::SortableDouble,
                    double_to_sortable_long(m.unwrap_or(0.0)),
                ),
                key => key.key_comparison(),
            }),
            IndexSortKind::String(missing) => Some((SortKeyKind::Ordinal, missing.missing_ord())),
            IndexSortKind::SortedSet { missing, .. } => {
                Some((SortKeyKind::Ordinal, missing.missing_ord()))
            }
            IndexSortKind::Binary(_) => None,
        }
    }
}

/// One sort field's comparator, resolved once out of an [`IndexSortField`].
///
/// Java rebuilds this per segment in `IndexSorter.*Sorter.getDocComparator`,
/// which closes over `reverseMul`, the pre-filled sentinel and the type's own
/// `compare`; resolving it once here is the same thing, and it makes the
/// unsupportable case ([`IndexSortKind::Binary`], whose keys are raw bytes)
/// *unconstructible* rather than a branch inside every comparison.
///
/// # The sentinel is an ordinary value
///
/// Lucene's comparator for a numeric sort is
///
/// ```text
/// long[] values = new long[maxDoc];
/// Arrays.fill(values, missingValue);          // missing docs get the sentinel
/// ... values[docID] = dvs.longValue(); ...
/// return (d1, d2) -> reverseMul * Long.compare(values[d1], values[d2]);
/// ```
///
/// so **`reverse` applies to the sentinel too**: under a descending sort a
/// document whose missing value is `Long.MAX_VALUE` sorts *first*. A missing
/// value names which sentinel is substituted, not which end of the finished
/// order the document lands at; the two coincide only for an ascending sort.
/// `CheckIndex.testSort` -- Lucene's and this port's -- is what rejects a
/// segment whose physical order disagrees.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SortKeyComparator {
    kind: SortKeyKind,
    sentinel: i64,
    reverse: bool,
}

impl SortKeyComparator {
    /// `None` for a sort whose per-document key is not a single `i64` --
    /// see [`IndexSortField::key_comparison`].
    pub fn new(sort: &IndexSortField) -> Option<Self> {
        let (kind, sentinel) = sort.key_comparison()?;
        Some(Self {
            kind,
            sentinel,
            reverse: sort.reverse,
        })
    }

    /// Compares two documents' keys, `None` meaning "this document has no
    /// value for the sort field".
    ///
    /// The comparison is done on `Ordering` rather than by negating a
    /// difference: negating `i64::MIN` overflows, and `Long.compare`
    /// returning `-1/0/1` is what Java's `reverseMul` multiplies.
    pub fn compare(&self, a: Option<i64>, b: Option<i64>) -> std::cmp::Ordering {
        let (a, b) = (a.unwrap_or(self.sentinel), b.unwrap_or(self.sentinel));
        let ord = match self.kind {
            SortKeyKind::Int | SortKeyKind::Ordinal => (a as i32).cmp(&(b as i32)),
            SortKeyKind::Long => a.cmp(&b),
            SortKeyKind::Float => java_float_compare(
                f32::from_bits(a as i32 as u32),
                f32::from_bits(b as i32 as u32),
            ),
            SortKeyKind::Double => {
                java_double_compare(f64::from_bits(a as u64), f64::from_bits(b as u64))
            }
            SortKeyKind::SortableFloat => java_float_compare(
                sortable_int_to_float(a as i32),
                sortable_int_to_float(b as i32),
            ),
            SortKeyKind::SortableDouble => {
                java_double_compare(sortable_long_to_double(a), sortable_long_to_double(b))
            }
        };
        if self.reverse {
            ord.reverse()
        } else {
            ord
        }
    }
}

/// `java.lang.Float.compare`: `-0.0f < 0.0f`, every NaN is equal to every
/// other NaN and greater than `+Infinity`.
///
/// Not `f32::total_cmp`, which orders a *negative* NaN below `-Infinity` --
/// a real difference, because a FLOAT doc-values column holds whatever bits
/// `Float.floatToRawIntBits` produced and Java compares them through
/// `floatToIntBits`, which canonicalizes.
fn java_float_compare(a: f32, b: f32) -> std::cmp::Ordering {
    if a < b {
        std::cmp::Ordering::Less
    } else if a > b {
        std::cmp::Ordering::Greater
    } else {
        canonical_float_bits(a).cmp(&canonical_float_bits(b))
    }
}

/// `Float.floatToIntBits`.
fn canonical_float_bits(v: f32) -> i32 {
    if v.is_nan() {
        0x7fc0_0000u32 as i32
    } else {
        v.to_bits() as i32
    }
}

/// `java.lang.Double.compare`, the 64-bit twin of [`java_float_compare`].
fn java_double_compare(a: f64, b: f64) -> std::cmp::Ordering {
    if a < b {
        std::cmp::Ordering::Less
    } else if a > b {
        std::cmp::Ordering::Greater
    } else {
        canonical_double_bits(a).cmp(&canonical_double_bits(b))
    }
}

/// `Double.doubleToLongBits`.
fn canonical_double_bits(v: f64) -> i64 {
    if v.is_nan() {
        0x7ff8_0000_0000_0000u64 as i64
    } else {
        v.to_bits() as i64
    }
}

#[derive(Debug, Clone)]
pub struct SegmentInfo {
    pub id: [u8; ID_LENGTH],
    pub version: LuceneVersion,
    pub min_version: Option<LuceneVersion>,
    pub doc_count: i32,
    pub is_compound_file: bool,
    pub has_blocks: bool,
    pub diagnostics: Vec<(String, String)>,
    pub files: Vec<String>,
    pub attributes: Vec<(String, String)>,
    /// `None` == unsorted segment (`numSortFields == 0`). `Some(fields)` is
    /// always non-empty and priority-ordered: `fields[0]` is the primary sort
    /// key, `fields[1]` breaks ties in `fields[0]`, and so on -- mirroring
    /// real Lucene's `Sort` being an array of `SortField`s applied in order.
    /// See [`IndexSortField`] for the per-field scope.
    pub index_sort: Option<Vec<IndexSortField>>,
}

/// `SortFieldProvider` SPI names, exactly as Java's
/// `META-INF/services/org.apache.lucene.index.SortFieldProvider` registers
/// them (`SortField.Provider.NAME` and friends).
const PROVIDER_SORT_FIELD: &str = "SortField";
const PROVIDER_SORTED_NUMERIC: &str = "SortedNumericSortField";
const PROVIDER_SORTED_SET: &str = "SortedSetSortField";
const PROVIDER_BINARY: &str = "BinarySortField";

/// Parses a whole `.si` file already read into memory, verifying header, footer,
/// and checksum. `segment_id` is the id Lucene stores alongside the segment in
/// `segments_N` and must match the id embedded in the `.si` file's index header.
pub fn parse(buf: &[u8], segment_id: &[u8; ID_LENGTH]) -> Result<SegmentInfo> {
    let mut input = SliceInput::new(buf);

    codec_util::check_index_header(
        &mut input,
        CODEC_NAME,
        VERSION_START,
        VERSION_CURRENT,
        segment_id,
        "",
    )?;

    let version = read_version(&mut input)?;

    let has_min_version = input.read_byte()?;
    let min_version = match has_min_version {
        0 => None,
        1 => Some(read_version(&mut input)?),
        other => return Err(Error::IllegalHasMinVersion(other)),
    };

    let doc_count = input.read_i32()?;
    if doc_count < 0 {
        return Err(Error::InvalidDocCount(doc_count));
    }

    let is_compound_file = input.read_byte()? == 1;
    let has_blocks = input.read_byte()? == 1;

    let diagnostics = input.read_map_of_strings()?;
    let files = input.read_set_of_strings()?;
    let attributes = input.read_map_of_strings()?;

    let num_sort_fields = input.read_vint()?;
    if num_sort_fields < 0 {
        return Err(Error::InvalidSortFieldCount(num_sort_fields));
    }
    // A sort field is at minimum a provider-name string plus a field name, a
    // direction byte, a missing-value marker and a type byte -- several bytes
    // each. Bounding the count by the bytes left in the file before reserving
    // is what keeps a corrupt `.si` from *aborting* the process:
    // `IndexSortField` is a multi-`String` struct, so an unbounded vint count
    // reserves hundreds of gigabytes, and an allocation failure is an abort
    // that `catch_unwind` cannot keep out of the JVM. Java sizes a
    // `SortField[]` from the same unbounded value and gets a catchable
    // `OutOfMemoryError` instead.
    if num_sort_fields as usize > input.remaining() {
        return Err(Error::InvalidSortFieldCount(num_sort_fields));
    }
    let index_sort = if num_sort_fields == 0 {
        None
    } else {
        let mut fields = Vec::with_capacity(num_sort_fields as usize);
        for _ in 0..num_sort_fields {
            fields.push(read_sort_field(&mut input)?);
        }
        Some(fields)
    };

    let payload_end = input.position();
    codec_util::check_footer(&mut input, buf.len())?;
    // ARITH: `check_footer` has already returned `Ok`, which it only does for a
    // `buf` that is at least `FOOTER_LENGTH` long (it reads the footer out of
    // the tail), so the subtraction cannot underflow.
    #[allow(clippy::arithmetic_side_effects)]
    {
        debug_assert!(payload_end <= buf.len() - codec_util::FOOTER_LENGTH);
    }

    Ok(SegmentInfo {
        id: *segment_id,
        version,
        min_version,
        doc_count,
        is_compound_file,
        has_blocks,
        diagnostics,
        files,
        attributes,
        index_sort,
    })
}

/// `NumericUtils.floatToSortableInt` -- Java writes a FLOAT missing value in
/// this sortable form, so a round trip has to go through the same space.
///
/// `Float.floatToIntBits` (which is what `NumericUtils` uses, not
/// `floatToRawIntBits`) collapses every NaN to the canonical `0x7fc00000`,
/// so this does too: without it a signalling NaN would write bytes Java
/// never writes, and `write` would not be `parse`'s inverse for the value
/// `parse` produced.
fn float_to_sortable_int(v: f32) -> i32 {
    let bits = if v.is_nan() {
        0x7fc0_0000u32 as i32
    } else {
        v.to_bits() as i32
    };
    sortable_float_bits(bits)
}

/// `NumericUtils.sortableFloatBits` -- its own inverse.
fn sortable_float_bits(bits: i32) -> i32 {
    bits ^ ((bits >> 31) & 0x7fff_ffff)
}

/// `NumericUtils.sortableIntToFloat`.
fn sortable_int_to_float(encoded: i32) -> f32 {
    f32::from_bits(sortable_float_bits(encoded) as u32)
}

/// `NumericUtils.doubleToSortableLong`, with `Double.doubleToLongBits`'
/// NaN canonicalization for the same reason as [`float_to_sortable_int`].
fn double_to_sortable_long(v: f64) -> i64 {
    let bits = if v.is_nan() {
        0x7ff8_0000_0000_0000u64 as i64
    } else {
        v.to_bits() as i64
    };
    sortable_double_bits(bits)
}

/// `NumericUtils.sortableDoubleBits` -- its own inverse.
fn sortable_double_bits(bits: i64) -> i64 {
    bits ^ ((bits >> 63) & 0x7fff_ffff_ffff_ffff)
}

/// `NumericUtils.sortableLongToDouble`.
fn sortable_long_to_double(encoded: i64) -> f64 {
    f64::from_bits(sortable_double_bits(encoded) as u64)
}

fn unsupported<T>(field: &str, reason: impl Into<String>) -> Result<T> {
    Err(Error::UnsupportedSortField {
        field: field.to_string(),
        reason: reason.into(),
    })
}

/// Reads Java's `writeInt(reverse ? 1 : 0)`. Java's readers all test
/// `readInt() == 1`, so any other value means "ascending" -- mirrored exactly
/// rather than rejected, so we never refuse a file real Lucene accepts.
fn read_reverse(input: &mut SliceInput) -> Result<bool> {
    Ok(input.read_i32()? == 1)
}

const TYPE_INT: &str = "INT";
const TYPE_LONG: &str = "LONG";
const TYPE_FLOAT: &str = "FLOAT";
const TYPE_DOUBLE: &str = "DOUBLE";
const TYPE_STRING: &str = "STRING";
/// Every remaining `SortField.Type` constant. They are legal enum names (so
/// `readType` accepts them), Java's own `SortField.Provider` refuses to
/// deserialize a *missing value* for them, and
/// `IndexWriterConfig.setIndexSort` refuses the `SortField` outright because
/// `getIndexSorter()` is `null` -- so none can be an index sort.
const TYPES_NOT_SORTABLE_ON_DISK: [&str; 5] =
    ["SCORE", "DOC", "CUSTOM", "STRING_VAL", "REWRITEABLE"];

/// `SortField.readType`: the type is written as the enum *name*, not an
/// ordinal, so an unknown name is a hard error (Java throws
/// `IllegalArgumentException`).
fn read_type(input: &mut SliceInput) -> Result<String> {
    let name = input.read_string()?;
    if name == TYPE_INT
        || name == TYPE_LONG
        || name == TYPE_FLOAT
        || name == TYPE_DOUBLE
        || name == TYPE_STRING
        || TYPES_NOT_SORTABLE_ON_DISK.contains(&name.as_str())
    {
        Ok(name)
    } else {
        Err(Error::UnknownSortFieldType(name))
    }
}

/// The `hasMissing` int plus, when it is 1, that type's own missing-value
/// encoding -- `SortField.Provider.readSortField`'s numeric cases and
/// `SortedNumericSortField.Provider.readSortField`'s, which are the same
/// four lines.
fn read_numeric_key(
    input: &mut SliceInput,
    field: &str,
    type_name: &str,
) -> Result<NumericSortKey> {
    let has_missing = input.read_i32()? == 1;
    Ok(match type_name {
        TYPE_INT => NumericSortKey::Int(if has_missing {
            Some(input.read_i32()?)
        } else {
            None
        }),
        TYPE_LONG => NumericSortKey::Long(if has_missing {
            Some(input.read_i64()?)
        } else {
            None
        }),
        TYPE_FLOAT => NumericSortKey::Float(if has_missing {
            Some(sortable_int_to_float(input.read_i32()?))
        } else {
            None
        }),
        TYPE_DOUBLE => NumericSortKey::Double(if has_missing {
            Some(sortable_long_to_double(input.read_i64()?))
        } else {
            None
        }),
        other => {
            return unsupported(
                field,
                format!(
                    "sort type {other} cannot be an index sort: \
                     IndexWriterConfig.setIndexSort refuses a SortField whose getIndexSorter() \
                     is null, which is all of SCORE/DOC/CUSTOM/STRING_VAL/REWRITEABLE, and \
                     SortedNumericSortField.getIndexSorter() likewise asserts on STRING \
                     (SortedNumericSelector.wrap: \"numericType must be a numeric type\"). \
                     Java can deserialize such a SortField when it carries no missing value, \
                     and throws on one that does; either way no .si real Lucene wrote holds it \
                     as an index sort"
                ),
            )
        }
    })
}

/// `SortField.Provider.readSortField`.
fn read_plain_sort_field(input: &mut SliceInput) -> Result<IndexSortField> {
    let field = input.read_string()?;
    let type_name = read_type(input)?;
    let reverse = read_reverse(input)?;
    let kind = if type_name == TYPE_STRING {
        // Java: `hasMissing == 0` -> null; else `missingString == 1` ->
        // STRING_FIRST, anything else -> STRING_LAST.
        IndexSortKind::String(if input.read_i32()? == 1 {
            if input.read_i32()? == 1 {
                StringMissingValue::First
            } else {
                StringMissingValue::Last
            }
        } else {
            StringMissingValue::None
        })
    } else {
        IndexSortKind::Numeric(read_numeric_key(input, &field, &type_name)?)
    };
    Ok(IndexSortField {
        field,
        reverse,
        kind,
    })
}

/// `SortedNumericSortField.readSelectorType`: an out-of-range ordinal is a
/// hard error (Java throws, or indexes past the end of `values()`).
fn read_sorted_numeric_selector(input: &mut SliceInput) -> Result<SortedNumericSelector> {
    match input.read_i32()? {
        0 => Ok(SortedNumericSelector::Min),
        1 => Ok(SortedNumericSelector::Max),
        selector => Err(Error::UnknownSortSelector {
            provider: PROVIDER_SORTED_NUMERIC,
            selector,
        }),
    }
}

/// `SortedSetSortField.readSelectorType`.
fn read_sorted_set_selector(input: &mut SliceInput) -> Result<SortedSetSelector> {
    match input.read_i32()? {
        0 => Ok(SortedSetSelector::Min),
        1 => Ok(SortedSetSelector::Max),
        2 => Ok(SortedSetSelector::MiddleMin),
        3 => Ok(SortedSetSelector::MiddleMax),
        selector => Err(Error::UnknownSortSelector {
            provider: PROVIDER_SORTED_SET,
            selector,
        }),
    }
}

/// `SortedNumericSortField.Provider.readSortField`. Java's `default ->
/// throw new AssertionError()` covers `STRING` and the five unsortable
/// types; [`read_numeric_key`] is that `default`.
fn read_sorted_numeric_sort_field(input: &mut SliceInput) -> Result<IndexSortField> {
    let field = input.read_string()?;
    let type_name = read_type(input)?;
    let reverse = read_reverse(input)?;
    let selector = read_sorted_numeric_selector(input)?;
    let key = read_numeric_key(input, &field, &type_name)?;
    Ok(IndexSortField {
        field,
        reverse,
        kind: IndexSortKind::SortedNumeric { key, selector },
    })
}

/// The shared tail of `SortedSetSortField.serialize` and
/// `BinarySortField.serialize`: `1 == STRING_FIRST`, `2 == STRING_LAST`,
/// **anything else == no missing value**. Java's two readers both fall
/// through to the `null` constructor rather than validating, so this does
/// too -- being stricter than the format would refuse a file Lucene reads.
fn read_string_missing_marker(input: &mut SliceInput) -> Result<StringMissingValue> {
    Ok(match input.read_i32()? {
        1 => StringMissingValue::First,
        2 => StringMissingValue::Last,
        _ => StringMissingValue::None,
    })
}

/// `SortedSetSortField.Provider.readSortField`.
fn read_sorted_set_sort_field(input: &mut SliceInput) -> Result<IndexSortField> {
    let field = input.read_string()?;
    let reverse = read_reverse(input)?;
    let selector = read_sorted_set_selector(input)?;
    let missing = read_string_missing_marker(input)?;
    Ok(IndexSortField {
        field,
        reverse,
        kind: IndexSortKind::SortedSet { selector, missing },
    })
}

/// `BinarySortField.Provider.readSortField`.
fn read_binary_sort_field(input: &mut SliceInput) -> Result<IndexSortField> {
    let field = input.read_string()?;
    let reverse = read_reverse(input)?;
    let missing = read_string_missing_marker(input)?;
    Ok(IndexSortField {
        field,
        reverse,
        kind: IndexSortKind::Binary(missing),
    })
}

/// `Lucene99SegmentInfoFormat.parseSegmentInfo`'s per-sort-field body:
/// `SortFieldProvider.forName(input.readString()).readSortField(input)`.
fn read_sort_field(input: &mut SliceInput) -> Result<IndexSortField> {
    let provider = input.read_string()?;
    match provider.as_str() {
        PROVIDER_SORT_FIELD => read_plain_sort_field(input),
        PROVIDER_SORTED_NUMERIC => read_sorted_numeric_sort_field(input),
        PROVIDER_SORTED_SET => read_sorted_set_sort_field(input),
        PROVIDER_BINARY => read_binary_sort_field(input),
        _ => Err(Error::UnknownSortFieldProvider(provider)),
    }
}

/// `SortField.serialize`'s missing-value tail for the four numeric types --
/// `writeInt(0)` when there is none, otherwise `writeInt(1)` followed by that
/// type's own encoding. Shared by the `SortField` and
/// `SortedNumericSortField` providers, whose `serialize`s are the same four
/// cases.
fn write_numeric_missing(out: &mut Vec<u8>, key: &NumericSortKey) {
    match *key {
        NumericSortKey::Int(Some(v)) => {
            out.write_i32(1);
            out.write_i32(v);
        }
        NumericSortKey::Long(Some(v)) => {
            out.write_i32(1);
            out.write_i64(v);
        }
        NumericSortKey::Float(Some(v)) => {
            out.write_i32(1);
            out.write_i32(float_to_sortable_int(v));
        }
        NumericSortKey::Double(Some(v)) => {
            out.write_i32(1);
            out.write_i64(double_to_sortable_long(v));
        }
        NumericSortKey::Int(None)
        | NumericSortKey::Long(None)
        | NumericSortKey::Float(None)
        | NumericSortKey::Double(None) => out.write_i32(0),
    }
}

/// `SortedSetSortField.serialize`/`BinarySortField.serialize`'s trailing
/// marker.
fn write_string_missing_marker(out: &mut Vec<u8>, missing: StringMissingValue) {
    out.write_i32(match missing {
        StringMissingValue::First => 1,
        StringMissingValue::Last => 2,
        StringMissingValue::None => 0,
    });
}

/// Writes one sort field exactly as `SortFieldProvider.write` does: the
/// provider's SPI name, then that provider's own `serialize` bytes. The
/// byte-level inverse of [`read_sort_field`] for every [`IndexSortKind`].
fn write_sort_field(out: &mut Vec<u8>, sf: &IndexSortField) {
    let reverse = if sf.reverse { 1 } else { 0 };
    match &sf.kind {
        IndexSortKind::Numeric(key) => {
            out.write_string(PROVIDER_SORT_FIELD);
            out.write_string(&sf.field);
            out.write_string(key.type_name());
            out.write_i32(reverse);
            write_numeric_missing(out, key);
        }
        IndexSortKind::String(missing) => {
            out.write_string(PROVIDER_SORT_FIELD);
            out.write_string(&sf.field);
            out.write_string(TYPE_STRING);
            out.write_i32(reverse);
            match missing {
                // `serialize`'s STRING case: STRING_LAST is `writeInt(0)`
                // and STRING_FIRST is `writeInt(1)` -- the *opposite* way
                // round from `SortedSetSortField`'s marker, which is why
                // these two are written separately rather than shared.
                StringMissingValue::None => out.write_i32(0),
                StringMissingValue::First => {
                    out.write_i32(1);
                    out.write_i32(1);
                }
                StringMissingValue::Last => {
                    out.write_i32(1);
                    out.write_i32(0);
                }
            }
        }
        IndexSortKind::SortedNumeric { key, selector } => {
            out.write_string(PROVIDER_SORTED_NUMERIC);
            out.write_string(&sf.field);
            out.write_string(key.type_name());
            out.write_i32(reverse);
            out.write_i32(match selector {
                SortedNumericSelector::Min => 0,
                SortedNumericSelector::Max => 1,
            });
            write_numeric_missing(out, key);
        }
        IndexSortKind::SortedSet { selector, missing } => {
            out.write_string(PROVIDER_SORTED_SET);
            out.write_string(&sf.field);
            out.write_i32(reverse);
            out.write_i32(match selector {
                SortedSetSelector::Min => 0,
                SortedSetSelector::Max => 1,
                SortedSetSelector::MiddleMin => 2,
                SortedSetSelector::MiddleMax => 3,
            });
            write_string_missing_marker(out, *missing);
        }
        IndexSortKind::Binary(missing) => {
            out.write_string(PROVIDER_BINARY);
            out.write_string(&sf.field);
            out.write_i32(reverse);
            write_string_missing_marker(out, *missing);
        }
    }
}

/// A `Sort` rendered exactly as Java's `Sort.toString()` renders it --
/// `SortField.toString`/`SortedNumericSortField.toString`/
/// `SortedSetSortField.toString` joined by commas, `<none>` for no sort.
///
/// It is what the `cannot change previous indexSort=` error quotes, and it is
/// public because the fixture manifests carry Lucene's own `Sort.toString()`
/// output: `tests/index_sort_wide_fixtures.rs` compares this against it, which
/// is the only way the rendering is pinned to Java's rather than to a reading
/// of it.
pub fn describe_index_sort(sort: Option<&[IndexSortField]>) -> String {
    /// `SortField.getMissingValue()`'s `toString()`, which is what
    /// `SortField.toString` appends after ` missingValue=`.
    fn numeric_missing(key: &NumericSortKey) -> Option<String> {
        match *key {
            NumericSortKey::Int(m) => m.map(|v| v.to_string()),
            NumericSortKey::Long(m) => m.map(|v| v.to_string()),
            // `Float.toString`/`Double.toString` always print a decimal
            // point; Rust's `Display` prints `1` where Java prints `1.0`.
            NumericSortKey::Float(m) => m.map(java_float_string),
            NumericSortKey::Double(m) => m.map(java_double_string),
        }
    }

    fn string_missing(missing: StringMissingValue) -> Option<&'static str> {
        match missing {
            StringMissingValue::None => None,
            StringMissingValue::First => Some("SortField.STRING_FIRST"),
            StringMissingValue::Last => Some("SortField.STRING_LAST"),
        }
    }

    fn type_name(key: &NumericSortKey) -> &'static str {
        match key {
            NumericSortKey::Int(_) => "INT",
            NumericSortKey::Long(_) => "LONG",
            NumericSortKey::Float(_) => "FLOAT",
            NumericSortKey::Double(_) => "DOUBLE",
        }
    }

    match sort {
        None => "<none>".to_string(),
        Some(fields) => fields
            .iter()
            .map(|sf| {
                // `SortField.toString`/`SortedNumericSortField.toString`/
                // `SortedSetSortField.toString`, reproduced exactly: the
                // fixture manifests carry Lucene's own `Sort.toString()` and
                // `tests/index_sort_fixtures.rs` compares against it.
                let (head, tail) = match &sf.kind {
                    IndexSortKind::Numeric(key) => (
                        format!(
                            "<{}: \"{}\">",
                            type_name(key).to_ascii_lowercase(),
                            sf.field
                        ),
                        String::new(),
                    ),
                    IndexSortKind::String(_) => {
                        (format!("<string: \"{}\">", sf.field), String::new())
                    }
                    IndexSortKind::SortedNumeric { key, selector } => (
                        format!("<sortednumeric: \"{}\">", sf.field),
                        format!(
                            " selector={} type={}",
                            match selector {
                                SortedNumericSelector::Min => "MIN",
                                SortedNumericSelector::Max => "MAX",
                            },
                            type_name(key)
                        ),
                    ),
                    IndexSortKind::SortedSet { selector, .. } => (
                        format!("<sortedset: \"{}\">", sf.field),
                        format!(
                            " selector={}",
                            match selector {
                                SortedSetSelector::Min => "MIN",
                                SortedSetSelector::Max => "MAX",
                                SortedSetSelector::MiddleMin => "MIDDLE_MIN",
                                SortedSetSelector::MiddleMax => "MIDDLE_MAX",
                            }
                        ),
                    ),
                    // `BinarySortField.toString` -- its own, not the
                    // `Type.CUSTOM` branch it would otherwise inherit from
                    // `SortField`.
                    IndexSortKind::Binary(_) => {
                        (format!("<binary: \"{}\">", sf.field), String::new())
                    }
                };
                let missing = match &sf.kind {
                    IndexSortKind::Numeric(key) | IndexSortKind::SortedNumeric { key, .. } => {
                        numeric_missing(key)
                    }
                    IndexSortKind::String(m)
                    | IndexSortKind::SortedSet { missing: m, .. }
                    | IndexSortKind::Binary(m) => string_missing(*m).map(str::to_string),
                };
                format!(
                    "{head}{}{}{tail}",
                    if sf.reverse { "!" } else { "" },
                    missing
                        .map(|m| format!(" missingValue={m}"))
                        .unwrap_or_default(),
                )
            })
            .collect::<Vec<_>>()
            .join(","),
    }
}

/// `Float.toString` for the values a `SortField` missing value can hold.
///
/// Three differences from Rust's `Display`, all of them real: Java always
/// prints a decimal point (`1.0` where Rust prints `1`), spells the
/// infinities `Infinity`/`-Infinity` where Rust prints `inf`/`-inf`, and
/// agrees on `NaN`. Java also switches to scientific notation outside
/// `1e-3 ..= 1e7` where Rust does not; that is *not* reproduced, and the
/// only consumer of the difference would be an error message quoting a
/// missing value of that magnitude.
fn java_float_string(v: f32) -> String {
    if v.is_infinite() {
        return if v > 0.0 { "Infinity" } else { "-Infinity" }.to_string();
    }
    with_decimal_point(format!("{v}"))
}

/// `Double.toString`, the 64-bit twin of [`java_float_string`].
fn java_double_string(v: f64) -> String {
    if v.is_infinite() {
        return if v > 0.0 { "Infinity" } else { "-Infinity" }.to_string();
    }
    with_decimal_point(format!("{v}"))
}

/// Java's "a float always prints a decimal point" rule, applied to a finite
/// non-NaN rendering.
fn with_decimal_point(s: String) -> String {
    if s.contains(['.', 'e', 'E', 'N']) {
        s
    } else {
        format!("{s}.0")
    }
}

/// Port of `Lucene99SegmentInfoFormat.write`: the exact byte-level inverse of
/// [`parse`]. Emits `numSortFields = 0` unless `si.index_sort` is `Some`, in
/// which case it emits one `SortField` per entry, in order (see this
/// module's doc comment for the byte-format disclaimer). Callers are
/// responsible for `files` containing
/// only names prefixed by the segment's own name (the real writer enforces
/// this via `IndexFileNames.parseSegmentName`; this function does not
/// re-validate it, matching the parser's stance that a hand-built writer is
/// trusted).
pub fn write(si: &SegmentInfo, segment_suffix: &str) -> Vec<u8> {
    let mut out: Vec<u8> = Vec::new();

    codec_util::write_index_header(
        &mut out,
        CODEC_NAME,
        VERSION_CURRENT,
        &si.id,
        segment_suffix,
    );

    write_version(&mut out, si.version);

    match si.min_version {
        Some(mv) => {
            out.write_byte(1);
            write_version(&mut out, mv);
        }
        None => out.write_byte(0),
    }

    out.write_i32(si.doc_count);
    out.write_byte(yes_no(si.is_compound_file));
    out.write_byte(yes_no(si.has_blocks));
    out.write_map_of_strings(&si.diagnostics);
    out.write_set_of_strings(&si.files);
    out.write_map_of_strings(&si.attributes);
    match &si.index_sort {
        Some(fields) => {
            out.write_vint(fields.len() as i32);
            for sf in fields {
                write_sort_field(&mut out, sf);
            }
        }
        None => out.write_vint(0),
    }

    codec_util::write_footer(&mut out);
    out
}

/// `SegmentInfo.YES` (1) / `SegmentInfo.NO` (**-1**, not 0). Java writes
/// `(byte) (flag ? SegmentInfo.YES : SegmentInfo.NO)`, so a false flag is
/// `0xFF` on disk. Both readers only test `== YES`, so writing `0` would
/// still *read* correctly -- but it would not be the bytes Lucene writes,
/// and this port's `.si` output is byte-compared against real Lucene's in
/// `tests/segment_info_fixtures.rs`.
fn yes_no(flag: bool) -> u8 {
    if flag {
        1
    } else {
        0xff
    }
}

fn write_version(out: &mut Vec<u8>, v: LuceneVersion) {
    out.write_i32(v.major);
    out.write_i32(v.minor);
    out.write_i32(v.bugfix);
}

/// `Version.fromBits`: each component is encoded into one byte, so anything
/// outside `0..=255` is an `IllegalArgumentException` in Java. Without this a
/// corrupt `.si` silently yields a nonsense version other checks then compare
/// against.
pub(crate) fn check_version_component(which: &'static str, value: i32) -> Result<i32> {
    if !(0..=255).contains(&value) {
        return Err(Error::IllegalVersion { which, value });
    }
    Ok(value)
}

fn read_version(input: &mut SliceInput) -> Result<LuceneVersion> {
    let major = check_version_component("major", input.read_i32()?)?;
    let minor = check_version_component("minor", input.read_i32()?)?;
    let bugfix = check_version_component("bugfix", input.read_i32()?)?;
    Ok(LuceneVersion {
        major,
        minor,
        bugfix,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test-only `.si` byte builder: independent of the Java fixtures under
    /// `tests/segment_info_fixtures.rs` (which exercise real Lucene-written
    /// bytes) — this covers the parser's own corruption/error handling, which
    /// needs deliberately-invalid inputs a real Lucene codec would never write.
    struct SiBuilder {
        id: [u8; ID_LENGTH],
        has_min_version: u8,
        doc_count: i32,
        is_compound_file: u8,
        has_blocks: u8,
        num_sort_fields: i32,
        /// Raw bytes for the single sort field's payload, written verbatim
        /// after `num_sort_fields` when it's `1` -- lets tests hand-build
        /// both well-formed and deliberately-corrupt sort-field encodings.
        sort_field_bytes: Vec<u8>,
    }

    impl SiBuilder {
        fn valid() -> Self {
            Self {
                id: [1u8; ID_LENGTH],
                has_min_version: 0,
                doc_count: 5,
                is_compound_file: 1,
                has_blocks: 0,
                num_sort_fields: 0,
                sort_field_bytes: Vec::new(),
            }
        }

        /// The bytes real Lucene's `SortField.Provider` writes for a
        /// single-valued sort: provider name, field, `SortField.Type` enum
        /// name, reverse flag, missing-present flag, and the missing value in
        /// that type's own encoding.
        fn plain_sort_field_bytes(
            field: &str,
            type_name: &str,
            reverse: i32,
            missing: Option<&[u8]>,
        ) -> Vec<u8> {
            let mut b = Vec::new();
            write_string(&mut b, PROVIDER_SORT_FIELD);
            write_string(&mut b, field);
            write_string(&mut b, type_name);
            b.extend_from_slice(&reverse.to_le_bytes());
            match missing {
                Some(bytes) => {
                    b.extend_from_slice(&1i32.to_le_bytes());
                    b.extend_from_slice(bytes);
                }
                None => b.extend_from_slice(&0i32.to_le_bytes()),
            }
            b
        }

        /// A LONG sort field with the sort-first/sort-last sentinel this
        /// port's own writer emits.
        fn long_sort_field_bytes(field: &str, reverse: bool, missing_last: bool) -> Vec<u8> {
            let sentinel = if missing_last { i64::MAX } else { i64::MIN };
            Self::plain_sort_field_bytes(
                field,
                TYPE_LONG,
                if reverse { 1 } else { 0 },
                Some(&sentinel.to_le_bytes()),
            )
        }

        fn build(&self) -> Vec<u8> {
            let mut out = Vec::new();
            out.extend_from_slice(&codec_util::CODEC_MAGIC.to_be_bytes());
            write_string(&mut out, CODEC_NAME);
            out.extend_from_slice(&(VERSION_CURRENT as u32).to_be_bytes());
            out.extend_from_slice(&self.id);
            out.push(0); // empty suffix

            out.extend_from_slice(&10i32.to_le_bytes()); // version major
            out.extend_from_slice(&0i32.to_le_bytes()); // minor
            out.extend_from_slice(&0i32.to_le_bytes()); // bugfix
            out.push(self.has_min_version);
            if self.has_min_version == 1 {
                out.extend_from_slice(&9i32.to_le_bytes());
                out.extend_from_slice(&0i32.to_le_bytes());
                out.extend_from_slice(&0i32.to_le_bytes());
            }
            out.extend_from_slice(&self.doc_count.to_le_bytes());
            out.push(self.is_compound_file);
            out.push(self.has_blocks);
            write_vint(&mut out, 0); // diagnostics: empty map
            write_vint(&mut out, 0); // files: empty set
            write_vint(&mut out, 0); // attributes: empty map
            write_vint(&mut out, self.num_sort_fields);
            out.extend_from_slice(&self.sort_field_bytes);

            out.extend_from_slice(&codec_util::FOOTER_MAGIC.to_be_bytes());
            out.extend_from_slice(&0u32.to_be_bytes());
            let checksum = crc32fast::hash(&out) as u64;
            out.extend_from_slice(&checksum.to_be_bytes());
            out
        }
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

    #[test]
    fn valid_segment_info_parses() {
        let b = SiBuilder::valid();
        let si = parse(&b.build(), &b.id).unwrap();
        assert_eq!(si.doc_count, 5);
        assert!(si.is_compound_file);
        assert!(!si.has_blocks);
        assert!(si.min_version.is_none());
    }

    #[test]
    fn min_version_present_is_parsed() {
        let mut b = SiBuilder::valid();
        b.has_min_version = 1;
        let si = parse(&b.build(), &b.id).unwrap();
        let mv = si.min_version.unwrap();
        assert_eq!((mv.major, mv.minor, mv.bugfix), (9, 0, 0));
    }

    #[test]
    fn illegal_has_min_version_byte_rejected() {
        let b = SiBuilder::valid();
        let mut bytes = b.build();
        // has_min_version byte sits right after the 3 SegVersion i32s (12 bytes)
        // in the payload, following the fixed-size index header.
        let header_len =
            codec_util::CODEC_MAGIC.to_be_bytes().len() + 1 + CODEC_NAME.len() + 4 + ID_LENGTH + 1;
        let has_min_version_offset = header_len + 12;
        bytes[has_min_version_offset] = 7; // neither 0 nor 1
        assert!(matches!(
            parse(&bytes, &b.id),
            Err(Error::IllegalHasMinVersion(7))
        ));
    }

    #[test]
    fn negative_doc_count_rejected() {
        let mut b = SiBuilder::valid();
        b.doc_count = -1;
        assert!(matches!(
            parse(&b.build(), &b.id),
            Err(Error::InvalidDocCount(-1))
        ));
    }

    /// A `SortField(field, LONG, reverse)` with the `Long.MIN_VALUE`/
    /// `Long.MAX_VALUE` sentinel this port's own sorted writers emit.
    fn long_sort(field: &str, reverse: bool, missing_last: bool) -> IndexSortField {
        IndexSortField::long(
            field,
            reverse,
            Some(if missing_last { i64::MAX } else { i64::MIN }),
        )
    }

    #[test]
    fn two_field_sort_count_parses_both_fields() {
        let mut b = SiBuilder::valid();
        b.num_sort_fields = 2;
        let mut bytes = SiBuilder::long_sort_field_bytes("price", false, true);
        bytes.extend(SiBuilder::long_sort_field_bytes("timestamp", true, false));
        b.sort_field_bytes = bytes;
        let si = parse(&b.build(), &b.id).unwrap();
        let fields = si.index_sort.unwrap();
        assert_eq!(
            fields,
            vec![
                long_sort("price", false, true),
                long_sort("timestamp", true, false)
            ]
        );
    }

    #[test]
    fn three_field_sort_round_trips_with_mixed_policies() {
        let mut si = sample_si();
        si.index_sort = Some(vec![
            long_sort("a", false, false),
            long_sort("b", true, true),
            long_sort("c", false, true),
        ]);
        let bytes = write(&si, "");
        let parsed = parse(&bytes, &si.id).unwrap();
        let fields = parsed.index_sort.unwrap();
        assert_eq!(fields, si.index_sort.unwrap());
    }

    #[test]
    fn negative_sort_field_count_rejected() {
        let mut b = SiBuilder::valid();
        b.num_sort_fields = -1;
        assert!(matches!(
            parse(&b.build(), &b.id),
            Err(Error::InvalidSortFieldCount(-1))
        ));
    }

    /// `numSortFields` is a vint that sized `Vec<IndexSortField>` — a
    /// multi-`String` struct — with no bound at all. `i32::MAX` of them is a
    /// hundreds-of-gigabytes reservation, and an allocation failure is an
    /// **abort**: `catch_unwind` cannot intercept it, so it takes the JVM down
    /// through the FFI rather than surfacing as a corrupt-index error. Java
    /// allocates the same unbounded `SortField[]` but gets a catchable
    /// `OutOfMemoryError`.
    #[test]
    fn absurd_sort_field_count_errors_instead_of_reserving_for_it() {
        let mut b = SiBuilder::valid();
        b.num_sort_fields = i32::MAX;
        assert!(matches!(
            parse(&b.build(), &b.id),
            Err(Error::InvalidSortFieldCount(i32::MAX))
        ));
    }

    #[test]
    fn single_numeric_sort_field_parses() {
        let mut b = SiBuilder::valid();
        b.num_sort_fields = 1;
        b.sort_field_bytes = SiBuilder::long_sort_field_bytes("price", false, true);
        let si = parse(&b.build(), &b.id).unwrap();
        assert_eq!(
            si.index_sort.unwrap(),
            vec![long_sort("price", false, true)]
        );
    }

    #[test]
    fn unknown_sort_field_type_name_rejected() {
        let mut b = SiBuilder::valid();
        b.num_sort_fields = 1;
        b.sort_field_bytes =
            SiBuilder::plain_sort_field_bytes("price", "NOT_A_TYPE", 0, Some(&0i64.to_le_bytes()));
        assert!(matches!(
            parse(&b.build(), &b.id),
            Err(Error::UnknownSortFieldType(t)) if t == "NOT_A_TYPE"
        ));
    }

    #[test]
    fn unknown_sort_field_provider_rejected() {
        let mut b = SiBuilder::valid();
        b.num_sort_fields = 1;
        let mut bytes = Vec::new();
        write_string(&mut bytes, "MyCustomSortField");
        b.sort_field_bytes = bytes;
        assert!(matches!(
            parse(&b.build(), &b.id),
            Err(Error::UnknownSortFieldProvider(p)) if p == "MyCustomSortField"
        ));
    }

    /// A `reverse` int other than 1 means "ascending" in every one of Java's
    /// own `readSortField` implementations (`in.readInt() == 1`) -- we must
    /// not be stricter than the format and reject a file Java reads.
    #[test]
    fn non_one_reverse_int_reads_as_ascending() {
        let mut b = SiBuilder::valid();
        b.num_sort_fields = 1;
        b.sort_field_bytes =
            SiBuilder::plain_sort_field_bytes("price", TYPE_LONG, 7, Some(&i64::MIN.to_le_bytes()));
        let si = parse(&b.build(), &b.id).unwrap();
        assert!(!si.index_sort.unwrap()[0].reverse);
    }

    /// **The c35 fix, read direction.** A numeric sort with *no* missing
    /// value is what `new SortField("f", Type.LONG)` produces when nobody
    /// calls `setMissingValue` -- the most ordinary sort there is -- and
    /// until c35 this port refused to open the index at all rather than
    /// model it. Java sorts such a document as if it held `0`.
    #[test]
    fn a_numeric_sort_with_no_missing_value_parses_and_compares_as_zero() {
        let mut b = SiBuilder::valid();
        b.num_sort_fields = 1;
        b.sort_field_bytes = SiBuilder::plain_sort_field_bytes("price", TYPE_LONG, 0, None);
        let sf = &parse(&b.build(), &b.id).unwrap().index_sort.unwrap()[0];
        assert_eq!(sf.kind, IndexSortKind::Numeric(NumericSortKey::Long(None)));
        let cmp = SortKeyComparator::new(sf).unwrap();
        // A missing document compares as 0: above -1, below 1.
        assert_eq!(cmp.compare(None, Some(-1)), std::cmp::Ordering::Greater);
        assert_eq!(cmp.compare(None, Some(1)), std::cmp::Ordering::Less);
        assert_eq!(cmp.compare(None, Some(0)), std::cmp::Ordering::Equal);
    }

    /// An arbitrary numeric missing value (Java allows any long) is carried
    /// through, sentinel and all -- it used to be rejected because
    /// `(first|last)` could not hold it.
    #[test]
    fn an_arbitrary_numeric_missing_value_round_trips() {
        let mut si = sample_si();
        si.index_sort = Some(vec![IndexSortField::long("price", true, Some(42))]);
        let bytes = write(&si, "");
        let parsed = parse(&bytes, &si.id).unwrap();
        assert_eq!(parsed.index_sort, si.index_sort);
        let sf = &parsed.index_sort.unwrap()[0];
        let cmp = SortKeyComparator::new(sf).unwrap();
        // Descending, so a missing (42) document sorts *before* 100 and
        // *after* 7 -- the sentinel is compared like any other value.
        assert_eq!(cmp.compare(None, Some(100)), std::cmp::Ordering::Greater);
        assert_eq!(cmp.compare(None, Some(7)), std::cmp::Ordering::Less);
    }

    /// Every `SortField.Type`/missing-value combination the plain provider
    /// can hold, round-tripped through `write` -> `parse` including the
    /// no-missing-value form.
    #[test]
    fn every_numeric_type_and_missing_form_round_trips() {
        let kinds = [
            NumericSortKey::Int(None),
            NumericSortKey::Int(Some(i32::MIN)),
            NumericSortKey::Int(Some(-7)),
            NumericSortKey::Int(Some(i32::MAX)),
            NumericSortKey::Long(None),
            NumericSortKey::Long(Some(i64::MIN)),
            NumericSortKey::Long(Some(i64::MAX)),
            NumericSortKey::Float(None),
            NumericSortKey::Float(Some(f32::NEG_INFINITY)),
            NumericSortKey::Float(Some(-1.5)),
            NumericSortKey::Float(Some(f32::INFINITY)),
            NumericSortKey::Double(None),
            NumericSortKey::Double(Some(f64::NEG_INFINITY)),
            NumericSortKey::Double(Some(2.25)),
            NumericSortKey::Double(Some(f64::INFINITY)),
        ];
        for kind in kinds {
            for reverse in [false, true] {
                let sf = IndexSortField {
                    field: "f".to_string(),
                    reverse,
                    kind: IndexSortKind::Numeric(kind),
                };
                let mut si = sample_si();
                si.index_sort = Some(vec![sf.clone()]);
                let parsed = parse(&write(&si, ""), &si.id).unwrap();
                assert_eq!(parsed.index_sort, Some(vec![sf]), "{kind:?}");
            }
        }
    }

    /// The STRING form of the plain provider, whose missing-value encoding
    /// is `writeInt(1)` for `STRING_FIRST` and `writeInt(0)` for
    /// `STRING_LAST` -- the *opposite* way round from the sorted-set marker,
    /// which is exactly the kind of detail a shared helper would have got
    /// wrong.
    #[test]
    fn the_string_sort_field_round_trips_all_three_missing_forms() {
        for missing in [
            StringMissingValue::None,
            StringMissingValue::First,
            StringMissingValue::Last,
        ] {
            let sf = IndexSortField {
                field: "name".to_string(),
                reverse: true,
                kind: IndexSortKind::String(missing),
            };
            let mut si = sample_si();
            si.index_sort = Some(vec![sf.clone()]);
            let parsed = parse(&write(&si, ""), &si.id).unwrap();
            assert_eq!(parsed.index_sort, Some(vec![sf]), "{missing:?}");
        }
    }

    /// Both selector enums, every ordinal, both providers -- and the
    /// `BinarySortField` provider, which has no selector at all.
    #[test]
    fn every_selector_and_provider_round_trips() {
        let mut fields = Vec::new();
        for selector in [SortedNumericSelector::Min, SortedNumericSelector::Max] {
            fields.push(IndexSortField {
                field: format!("sn{selector:?}"),
                reverse: true,
                kind: IndexSortKind::SortedNumeric {
                    key: NumericSortKey::Int(Some(3)),
                    selector,
                },
            });
        }
        for selector in [
            SortedSetSelector::Min,
            SortedSetSelector::Max,
            SortedSetSelector::MiddleMin,
            SortedSetSelector::MiddleMax,
        ] {
            fields.push(IndexSortField {
                field: format!("ss{selector:?}"),
                reverse: false,
                kind: IndexSortKind::SortedSet {
                    selector,
                    missing: StringMissingValue::Last,
                },
            });
        }
        for missing in [
            StringMissingValue::None,
            StringMissingValue::First,
            StringMissingValue::Last,
        ] {
            fields.push(IndexSortField {
                field: format!("bin{missing:?}"),
                reverse: true,
                kind: IndexSortKind::Binary(missing),
            });
        }
        let mut si = sample_si();
        si.index_sort = Some(fields.clone());
        let parsed = parse(&write(&si, ""), &si.id).unwrap();
        assert_eq!(parsed.index_sort, Some(fields));
    }

    /// The exact provider byte streams Java writes, decoded -- hand-built
    /// rather than produced by [`write`], so the two sides cannot agree on a
    /// misreading of the format.
    #[test]
    fn sorted_numeric_sorted_set_and_binary_providers_all_decode() {
        // SortedNumericSortField: field, type, reverse, selector, hasMissing, value
        let mut sn = Vec::new();
        write_string(&mut sn, PROVIDER_SORTED_NUMERIC);
        write_string(&mut sn, "sn");
        write_string(&mut sn, TYPE_LONG);
        sn.extend_from_slice(&1i32.to_le_bytes()); // reverse
        sn.extend_from_slice(&1i32.to_le_bytes()); // selector MAX
        sn.extend_from_slice(&1i32.to_le_bytes()); // hasMissing
        sn.extend_from_slice(&i64::MAX.to_le_bytes());

        // SortedSetSortField: field, reverse, selector, missing marker
        let mut ss = Vec::new();
        write_string(&mut ss, PROVIDER_SORTED_SET);
        write_string(&mut ss, "ss");
        ss.extend_from_slice(&0i32.to_le_bytes());
        ss.extend_from_slice(&2i32.to_le_bytes()); // selector MIDDLE_MIN
        ss.extend_from_slice(&1i32.to_le_bytes()); // STRING_FIRST

        // BinarySortField: field, reverse, missing marker
        let mut bin = Vec::new();
        write_string(&mut bin, PROVIDER_BINARY);
        write_string(&mut bin, "bin");
        bin.extend_from_slice(&0i32.to_le_bytes());
        bin.extend_from_slice(&2i32.to_le_bytes()); // STRING_LAST

        let mut b = SiBuilder::valid();
        b.num_sort_fields = 3;
        let mut bytes = sn;
        bytes.extend(ss);
        bytes.extend(bin);
        b.sort_field_bytes = bytes;

        let fields = parse(&b.build(), &b.id).unwrap().index_sort.unwrap();
        assert_eq!(
            fields,
            vec![
                IndexSortField {
                    field: "sn".to_string(),
                    reverse: true,
                    kind: IndexSortKind::SortedNumeric {
                        key: NumericSortKey::Long(Some(i64::MAX)),
                        selector: SortedNumericSelector::Max,
                    },
                },
                IndexSortField {
                    field: "ss".to_string(),
                    reverse: false,
                    kind: IndexSortKind::SortedSet {
                        selector: SortedSetSelector::MiddleMin,
                        missing: StringMissingValue::First,
                    },
                },
                IndexSortField {
                    field: "bin".to_string(),
                    reverse: false,
                    kind: IndexSortKind::Binary(StringMissingValue::Last),
                },
            ]
        );
    }

    #[test]
    fn out_of_range_selector_ordinal_rejected() {
        for (provider, extra_int) in [
            (PROVIDER_SORTED_SET, true),
            (PROVIDER_SORTED_NUMERIC, false),
        ] {
            let mut bytes = Vec::new();
            write_string(&mut bytes, provider);
            write_string(&mut bytes, "f");
            if provider == PROVIDER_SORTED_NUMERIC {
                write_string(&mut bytes, TYPE_LONG);
            }
            bytes.extend_from_slice(&0i32.to_le_bytes()); // reverse
            bytes.extend_from_slice(&9i32.to_le_bytes()); // no such selector
            if extra_int {
                bytes.extend_from_slice(&1i32.to_le_bytes());
            }
            let mut b = SiBuilder::valid();
            b.num_sort_fields = 1;
            b.sort_field_bytes = bytes;
            assert!(
                matches!(
                    parse(&b.build(), &b.id),
                    Err(Error::UnknownSortSelector { selector: 9, .. })
                ),
                "{provider}"
            );
        }
    }

    /// A `SortField.Type` that cannot be an index sort at all -- Java's own
    /// `IndexWriterConfig.setIndexSort` refuses it because
    /// `getIndexSorter()` is null, so no real `.si` holds one, and
    /// `SortField.serialize` throws on a missing value for it.
    #[test]
    fn a_type_that_cannot_be_an_index_sort_is_rejected() {
        for type_name in TYPES_NOT_SORTABLE_ON_DISK {
            let mut b = SiBuilder::valid();
            b.num_sort_fields = 1;
            b.sort_field_bytes = SiBuilder::plain_sort_field_bytes(
                "f",
                type_name,
                0,
                Some(&0i64.to_le_bytes()[..4]),
            );
            assert!(
                matches!(
                    parse(&b.build(), &b.id),
                    Err(Error::UnsupportedSortField { .. })
                ),
                "{type_name}"
            );
        }
    }

    /// A `SortedNumericSortField` whose type is `STRING` is an
    /// `AssertionError` inside Java's own `serialize`, so it is refused here
    /// too rather than lowered onto something.
    #[test]
    fn a_sorted_numeric_sort_field_with_a_string_type_is_rejected() {
        let mut bytes = Vec::new();
        write_string(&mut bytes, PROVIDER_SORTED_NUMERIC);
        write_string(&mut bytes, "f");
        write_string(&mut bytes, TYPE_STRING);
        bytes.extend_from_slice(&0i32.to_le_bytes());
        bytes.extend_from_slice(&0i32.to_le_bytes());
        bytes.extend_from_slice(&0i32.to_le_bytes());
        let mut b = SiBuilder::valid();
        b.num_sort_fields = 1;
        b.sort_field_bytes = bytes;
        assert!(matches!(
            parse(&b.build(), &b.id),
            Err(Error::UnsupportedSortField { .. })
        ));
    }

    /// Java's two marker readers fall through to "no missing value" for any
    /// int that is neither 1 nor 2, so being stricter would refuse a file
    /// Lucene reads.
    #[test]
    fn an_unknown_string_missing_marker_reads_as_no_missing_value() {
        let mut bytes = Vec::new();
        write_string(&mut bytes, PROVIDER_BINARY);
        write_string(&mut bytes, "bin");
        bytes.extend_from_slice(&0i32.to_le_bytes());
        bytes.extend_from_slice(&77i32.to_le_bytes());
        let mut b = SiBuilder::valid();
        b.num_sort_fields = 1;
        b.sort_field_bytes = bytes;
        let fields = parse(&b.build(), &b.id).unwrap().index_sort.unwrap();
        assert_eq!(
            fields[0].kind,
            IndexSortKind::Binary(StringMissingValue::None)
        );
    }

    /// `key_comparison` is the whole comparator contract, so every kind's
    /// answer is pinned: which `compare` and which sentinel.
    #[test]
    fn key_comparison_names_the_right_compare_and_sentinel_per_kind() {
        use std::cmp::Ordering;
        let sf = |kind| IndexSortField {
            field: "f".to_string(),
            reverse: false,
            kind,
        };
        assert_eq!(
            sf(IndexSortKind::Numeric(NumericSortKey::Int(Some(5)))).key_comparison(),
            Some((SortKeyKind::Int, 5))
        );
        assert_eq!(
            sf(IndexSortKind::Numeric(NumericSortKey::Long(None))).key_comparison(),
            Some((SortKeyKind::Long, 0))
        );
        assert_eq!(
            sf(IndexSortKind::Numeric(NumericSortKey::Float(Some(1.0)))).key_comparison(),
            Some((SortKeyKind::Float, 1.0f32.to_bits() as i32 as i64))
        );
        assert_eq!(
            sf(IndexSortKind::Numeric(NumericSortKey::Double(Some(-1.0)))).key_comparison(),
            Some((SortKeyKind::Double, (-1.0f64).to_bits() as i64))
        );
        // `None` behaves like `First` in the comparator even though the two
        // are distinguishable on disk.
        for missing in [StringMissingValue::None, StringMissingValue::First] {
            assert_eq!(
                sf(IndexSortKind::String(missing)).key_comparison(),
                Some((SortKeyKind::Ordinal, i32::MIN as i64))
            );
        }
        assert_eq!(
            sf(IndexSortKind::SortedSet {
                selector: SortedSetSelector::Max,
                missing: StringMissingValue::Last,
            })
            .key_comparison(),
            Some((SortKeyKind::Ordinal, i32::MAX as i64))
        );
        // The one kind with no single-`i64` key.
        assert_eq!(
            sf(IndexSortKind::Binary(StringMissingValue::Last)).key_comparison(),
            None
        );
        assert!(
            SortKeyComparator::new(&sf(IndexSortKind::Binary(StringMissingValue::Last))).is_none()
        );

        // INT compares the low 32 bits, so a value whose 64-bit form is
        // larger can still be the smaller INT.
        let int_cmp =
            SortKeyComparator::new(&sf(IndexSortKind::Numeric(NumericSortKey::Int(None)))).unwrap();
        assert_eq!(
            int_cmp.compare(Some(0xFFFF_FFFF), Some(1)),
            Ordering::Less,
            "0xFFFFFFFF is -1 as an int"
        );
    }

    /// `Float.compare`/`Double.compare`, not `f32::total_cmp`: every NaN is
    /// equal and greater than `+Infinity`, and `-0.0 < 0.0`.
    #[test]
    fn float_and_double_sorts_use_javas_compare() {
        use std::cmp::Ordering;
        let f = IndexSortField {
            field: "f".to_string(),
            reverse: false,
            kind: IndexSortKind::Numeric(NumericSortKey::Float(None)),
        };
        let cmp = SortKeyComparator::new(&f).unwrap();
        let bits = |v: f32| Some(v.to_bits() as i32 as i64);
        // The ordinary cases first: `Float.compare`'s `<` and `>` arms, which
        // are the ones that make a FLOAT sort a *float* sort -- comparing the
        // same 64 bits as longs would order -1.5 above 1.5.
        assert_eq!(cmp.compare(bits(-1.5), bits(1.5)), Ordering::Less);
        assert_eq!(cmp.compare(bits(1.5), bits(-1.5)), Ordering::Greater);
        assert_eq!(cmp.compare(bits(1.5), bits(1.5)), Ordering::Equal);
        assert_eq!(cmp.compare(bits(-0.0), bits(0.0)), Ordering::Less);
        assert_eq!(
            cmp.compare(bits(f32::NAN), bits(f32::INFINITY)),
            Ordering::Greater
        );
        assert_eq!(
            cmp.compare(bits(f32::NAN), bits(-f32::NAN)),
            Ordering::Equal,
            "Java canonicalizes every NaN"
        );

        let d = IndexSortField {
            field: "d".to_string(),
            reverse: true,
            kind: IndexSortKind::Numeric(NumericSortKey::Double(None)),
        };
        let cmp = SortKeyComparator::new(&d).unwrap();
        let dbits = |v: f64| Some(v.to_bits() as i64);
        // Reversed, so `<` becomes `Greater`.
        assert_eq!(cmp.compare(dbits(-2.5), dbits(2.5)), Ordering::Greater);
        assert_eq!(cmp.compare(dbits(2.5), dbits(-2.5)), Ordering::Less);
        assert_eq!(cmp.compare(dbits(-0.0), dbits(0.0)), Ordering::Greater);
        assert_eq!(
            cmp.compare(dbits(f64::NAN), dbits(-f64::NAN)),
            Ordering::Equal
        );
        // The sentinel for a no-missing-value DOUBLE sort is `0.0`, and
        // `reverse` applies to it.
        assert_eq!(cmp.compare(None, dbits(1.0)), Ordering::Greater);
    }

    /// `describe_index_sort` is Java's `Sort.toString()`, and the fixture
    /// manifests compare it against Lucene's own output
    /// (`tests/index_sort_wide_fixtures.rs`,
    /// `examples/write_segment_info_fixture.rs` -> `VerifySegmentInfo`). This
    /// pins every branch of it -- including the ones no fixture happens to
    /// use -- so a kind added later cannot render as something Java never
    /// prints.
    #[test]
    fn describe_index_sort_renders_every_kind_the_way_java_does() {
        let sf = |field: &str, reverse: bool, kind| IndexSortField {
            field: field.to_string(),
            reverse,
            kind,
        };
        assert_eq!(describe_index_sort(None), "<none>");
        let all = vec![
            sf(
                "i",
                true,
                IndexSortKind::Numeric(NumericSortKey::Int(Some(-7))),
            ),
            sf(
                "l",
                false,
                IndexSortKind::Numeric(NumericSortKey::Long(None)),
            ),
            sf(
                "f",
                false,
                IndexSortKind::Numeric(NumericSortKey::Float(Some(-1.5))),
            ),
            // An *integral* float, where Java prints `1.0` and Rust's
            // `Display` prints `1`.
            sf(
                "f2",
                false,
                IndexSortKind::Numeric(NumericSortKey::Float(Some(1.0))),
            ),
            sf(
                "d",
                true,
                IndexSortKind::Numeric(NumericSortKey::Double(Some(2.25))),
            ),
            sf(
                "d2",
                false,
                IndexSortKind::Numeric(NumericSortKey::Double(Some(-3.0))),
            ),
            sf("s", true, IndexSortKind::String(StringMissingValue::Last)),
            sf("s2", false, IndexSortKind::String(StringMissingValue::None)),
            sf(
                "sn",
                false,
                IndexSortKind::SortedNumeric {
                    key: NumericSortKey::Int(Some(9)),
                    selector: SortedNumericSelector::Max,
                },
            ),
            sf(
                "sn2",
                false,
                IndexSortKind::SortedNumeric {
                    key: NumericSortKey::Double(None),
                    selector: SortedNumericSelector::Min,
                },
            ),
            sf(
                "ss",
                true,
                IndexSortKind::SortedSet {
                    selector: SortedSetSelector::MiddleMin,
                    missing: StringMissingValue::First,
                },
            ),
            sf(
                "ss2",
                false,
                IndexSortKind::SortedSet {
                    selector: SortedSetSelector::MiddleMax,
                    missing: StringMissingValue::None,
                },
            ),
            sf(
                "ss3",
                false,
                IndexSortKind::SortedSet {
                    selector: SortedSetSelector::Max,
                    missing: StringMissingValue::Last,
                },
            ),
            sf("b", false, IndexSortKind::Binary(StringMissingValue::None)),
            sf("b2", true, IndexSortKind::Binary(StringMissingValue::First)),
        ];
        assert_eq!(
            describe_index_sort(Some(&all)),
            "<int: \"i\">! missingValue=-7,\
             <long: \"l\">,\
             <float: \"f\"> missingValue=-1.5,\
             <float: \"f2\"> missingValue=1.0,\
             <double: \"d\">! missingValue=2.25,\
             <double: \"d2\"> missingValue=-3.0,\
             <string: \"s\">! missingValue=SortField.STRING_LAST,\
             <string: \"s2\">,\
             <sortednumeric: \"sn\"> missingValue=9 selector=MAX type=INT,\
             <sortednumeric: \"sn2\"> selector=MIN type=DOUBLE,\
             <sortedset: \"ss\">! missingValue=SortField.STRING_FIRST selector=MIDDLE_MIN,\
             <sortedset: \"ss2\"> selector=MIDDLE_MAX,\
             <sortedset: \"ss3\"> missingValue=SortField.STRING_LAST selector=MAX,\
             <binary: \"b\">,\
             <binary: \"b2\">! missingValue=SortField.STRING_FIRST"
        );
        // `Float.toString`/`Double.toString` on the shapes Rust renders
        // differently: infinities and NaN keep Java's spelling.
        assert_eq!(java_float_string(f32::INFINITY), "Infinity");
        assert_eq!(java_float_string(f32::NEG_INFINITY), "-Infinity");
        assert_eq!(java_double_string(f64::NEG_INFINITY), "-Infinity");
        assert_eq!(java_double_string(f64::INFINITY), "Infinity");
        assert_eq!(java_double_string(f64::NAN), "NaN");
    }

    /// A **SORTED_NUMERIC** FLOAT/DOUBLE column holds
    /// `NumericUtils.floatToSortableInt`, not the raw bits a NUMERIC one
    /// holds, and `SortedNumericSelector.wrap` undoes that with
    /// `sortableFloatBits` before `FloatSorter` ever sees a value.
    ///
    /// Comparing the stored form as raw bits instead reverses the whole
    /// negative half of the ordering -- `-1.0f` is stored as `0xC07FFFFF`,
    /// which reads back as `-3.99999` -- so this pins the two kinds against
    /// each other on the same values.
    #[test]
    fn a_sorted_numeric_float_sort_undoes_the_sortable_encoding() {
        use std::cmp::Ordering;
        let sorted_numeric = |key| IndexSortField {
            field: "f".to_string(),
            reverse: false,
            kind: IndexSortKind::SortedNumeric {
                key,
                selector: SortedNumericSelector::Min,
            },
        };

        let cmp = SortKeyComparator::new(&sorted_numeric(NumericSortKey::Float(None))).unwrap();
        // What `FloatField` stores: `floatToSortableInt`.
        let stored = |v: f32| Some(float_to_sortable_int(v) as i64);
        for (a, b) in [(-2.0f32, -1.0), (-1.0, 0.0), (0.0, 1.0), (1.0, 2.0)] {
            assert_eq!(
                cmp.compare(stored(a), stored(b)),
                Ordering::Less,
                "{a} < {b}"
            );
        }
        // The sentinel is in the same space, and a no-missing-value sort
        // compares a document with no value as `0.0`.
        assert_eq!(cmp.compare(None, stored(-1.0)), Ordering::Greater);
        assert_eq!(cmp.compare(None, stored(1.0)), Ordering::Less);
        // With an explicit sentinel it lands where that sentinel does.
        let cmp =
            SortKeyComparator::new(&sorted_numeric(NumericSortKey::Float(Some(-5.0)))).unwrap();
        assert_eq!(cmp.compare(None, stored(-1.0)), Ordering::Less);

        let cmp =
            SortKeyComparator::new(&sorted_numeric(NumericSortKey::Double(Some(-5.0)))).unwrap();
        let stored = |v: f64| Some(double_to_sortable_long(v));
        assert_eq!(cmp.compare(stored(-2.0), stored(-1.0)), Ordering::Less);
        assert_eq!(cmp.compare(None, stored(-1.0)), Ordering::Less);

        // And the NUMERIC kind over the *same* field is the other encoding:
        // raw bits, so the sortable form would be read as a different value
        // entirely. This is the assertion that fails if the two kinds are
        // conflated.
        let raw = SortKeyComparator::new(&IndexSortField {
            field: "f".to_string(),
            reverse: false,
            kind: IndexSortKind::Numeric(NumericSortKey::Float(None)),
        })
        .unwrap();
        assert_eq!(
            raw.compare(
                Some(float_to_sortable_int(-2.0) as i64),
                Some(float_to_sortable_int(-1.0) as i64)
            ),
            Ordering::Greater,
            "read as raw bits the sortable encoding of -2.0 is larger, which is the bug"
        );
    }

    /// A FLOAT/DOUBLE missing value survives the `sortableInt`/`sortableLong`
    /// encoding Java writes it in -- including a NaN, which Java
    /// canonicalizes on the way out.
    #[test]
    fn float_missing_values_round_trip_through_the_sortable_encoding() {
        for v in [0.0f32, -0.0, 1.5, -1.5, f32::MIN, f32::MAX, f32::EPSILON] {
            assert_eq!(
                sortable_int_to_float(float_to_sortable_int(v)).to_bits(),
                v.to_bits(),
                "{v}"
            );
        }
        assert!(sortable_int_to_float(float_to_sortable_int(f32::NAN)).is_nan());
        for v in [0.0f64, -0.0, 1.5, -1.5, f64::MIN, f64::MAX] {
            assert_eq!(
                sortable_long_to_double(double_to_sortable_long(v)).to_bits(),
                v.to_bits(),
                "{v}"
            );
        }
        assert!(sortable_long_to_double(double_to_sortable_long(f64::NAN)).is_nan());
    }

    /// Illegal `Version.fromBits` components (each is encoded into one byte)
    /// must be rejected, not silently carried into later comparisons.
    #[test]
    fn out_of_range_version_component_rejected() {
        let b = SiBuilder::valid();
        let mut bytes = b.build();
        let header_len =
            codec_util::CODEC_MAGIC.to_be_bytes().len() + 1 + CODEC_NAME.len() + 4 + ID_LENGTH + 1;
        // minor version sits at header_len + 4
        bytes[header_len + 4..header_len + 8].copy_from_slice(&999i32.to_le_bytes());
        assert!(matches!(
            parse(&bytes, &b.id),
            Err(Error::IllegalVersion {
                which: "minor",
                value: 999
            })
        ));
    }

    #[test]
    fn wrong_id_rejected_with_store_error() {
        let b = SiBuilder::valid();
        let wrong_id = [9u8; ID_LENGTH];
        assert!(matches!(parse(&b.build(), &wrong_id), Err(Error::Store(_))));
    }

    // --- write() round-trips through parse() ---

    fn sample_si() -> SegmentInfo {
        SegmentInfo {
            id: [3u8; ID_LENGTH],
            version: LuceneVersion {
                major: 10,
                minor: 0,
                bugfix: 0,
            },
            min_version: None,
            doc_count: 42,
            is_compound_file: false,
            has_blocks: false,
            diagnostics: vec![],
            files: vec![],
            attributes: vec![],
            index_sort: None,
        }
    }

    #[test]
    fn write_minimal_round_trips() {
        let si = sample_si();
        let bytes = write(&si, "");
        let parsed = parse(&bytes, &si.id).unwrap();
        assert_eq!(parsed.doc_count, 42);
        assert!(!parsed.is_compound_file);
        assert!(!parsed.has_blocks);
        assert!(parsed.min_version.is_none());
        assert_eq!(parsed.version, si.version);
        assert!(parsed.diagnostics.is_empty());
        assert!(parsed.files.is_empty());
        assert!(parsed.attributes.is_empty());
    }

    #[test]
    fn write_compound_file_round_trips() {
        let mut si = sample_si();
        si.is_compound_file = true;
        si.has_blocks = true;
        let bytes = write(&si, "");
        let parsed = parse(&bytes, &si.id).unwrap();
        assert!(parsed.is_compound_file);
        assert!(parsed.has_blocks);
    }

    #[test]
    fn write_min_version_round_trips() {
        let mut si = sample_si();
        si.min_version = Some(LuceneVersion {
            major: 9,
            minor: 12,
            bugfix: 0,
        });
        let bytes = write(&si, "");
        let parsed = parse(&bytes, &si.id).unwrap();
        let mv = parsed.min_version.unwrap();
        assert_eq!((mv.major, mv.minor, mv.bugfix), (9, 12, 0));
    }

    #[test]
    fn write_diagnostics_files_attributes_round_trip() {
        let mut si = sample_si();
        si.diagnostics = vec![
            ("source".to_string(), "flush".to_string()),
            ("os".to_string(), "Linux".to_string()),
        ];
        si.files = vec!["_0.fdt".to_string(), "_0.fdx".to_string()];
        si.attributes = vec![(
            "Lucene90StoredFieldsFormat.mode".to_string(),
            "BEST_SPEED".to_string(),
        )];
        let bytes = write(&si, "");
        let parsed = parse(&bytes, &si.id).unwrap();
        assert_eq!(parsed.diagnostics, si.diagnostics);
        assert_eq!(parsed.files, si.files);
        assert_eq!(parsed.attributes, si.attributes);
    }

    #[test]
    fn write_with_segment_suffix_round_trips() {
        let si = sample_si();
        let bytes = write(&si, "suffix1");
        let parsed = parse(&bytes, &si.id);
        // parse() with the wrong (empty) suffix must fail; the exact suffix must match.
        assert!(parsed.is_err());
        let parsed_ok = {
            let mut input = SliceInput::new(&bytes);
            codec_util::check_index_header(
                &mut input,
                CODEC_NAME,
                VERSION_START,
                VERSION_CURRENT,
                &si.id,
                "suffix1",
            )
        };
        assert!(parsed_ok.is_ok());
    }

    #[test]
    fn write_zero_doc_count_round_trips() {
        let mut si = sample_si();
        si.doc_count = 0;
        let bytes = write(&si, "");
        let parsed = parse(&bytes, &si.id).unwrap();
        assert_eq!(parsed.doc_count, 0);
    }

    #[test]
    fn write_no_index_sort_round_trips_as_none() {
        let si = sample_si();
        assert!(si.index_sort.is_none());
        let bytes = write(&si, "");
        let parsed = parse(&bytes, &si.id).unwrap();
        assert!(parsed.index_sort.is_none());
    }

    #[test]
    fn write_ascending_index_sort_round_trips() {
        let mut si = sample_si();
        si.index_sort = Some(vec![long_sort("timestamp", false, false)]);
        let parsed = parse(&write(&si, ""), &si.id).unwrap();
        assert_eq!(parsed.index_sort, si.index_sort);
    }

    #[test]
    fn write_descending_index_sort_with_missing_last_round_trips() {
        let mut si = sample_si();
        si.index_sort = Some(vec![long_sort("score", true, true)]);
        let parsed = parse(&write(&si, ""), &si.id).unwrap();
        assert_eq!(parsed.index_sort, si.index_sort);
    }
}
