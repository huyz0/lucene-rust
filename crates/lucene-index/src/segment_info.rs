//! Port of `org.apache.lucene.codecs.lucene99.Lucene99SegmentInfoFormat` (`.si` files).
//!
//! Both directions ported: [`parse`] (read path, PLAN.md Phase 2) and
//! [`write()`] (write path, PLAN.md Phase 5) are exact byte-level inverses of
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
//! for a string one). [`write()`] is the exact byte-level inverse of [`parse`]
//! for every one of them.
//!
//! So an index a real `IndexWriter` wrote with an ordinary sort (a numeric
//! sort with no missing value, an arbitrary sentinel, a `MAX` selector, a
//! string sort) opens here.
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
//! [`IndexSortField::key_comparison`] gives the comparator for every kind:
//! a single `i64` per document for the four numeric types and both numeric
//! selectors, and an ordinal rank for `STRING`, `SortedSetSortField` and
//! `BinarySortField` (computed by [`crate::index_sorter`]). The writer, the
//! merge and `CheckIndex` all act on every kind.
//!
//! [`parse_simpletext`] and [`write_simpletext`] use the same sort-provider
//! implementations for `SimpleTextSegmentInfoFormat`. Its text framing lives
//! in `lucene-codecs`; the typed sort belongs here with the binary `.si` model.
//! Differs: the text parser checks the checksum before these adapters decode
//! the provider bytes, whereas Java decodes them before checking the footer.

pub use lucene_codecs::codecs::SegmentInfoFormat;
use lucene_store::codec_util::{self, ID_LENGTH};
use lucene_store::data_input::{DataInput, SliceInput};
use lucene_store::data_output::DataOutput;
use lucene_util::numeric_utils::{
    double_to_sortable_long, float_to_sortable_int, sortable_int_to_float, sortable_long_to_double,
};

const CODEC_NAME: &str = "Lucene90SegmentInfo";
const VERSION_START: i32 = 0;
const VERSION_CURRENT: i32 = 0;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Store(#[from] lucene_store::Error),
    #[error(transparent)]
    SimpleText(#[from] lucene_codecs::simpletext::Error),
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
    #[error("unknown codec: {0:?}")]
    UnknownCodec(String),
}

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LuceneVersion {
    pub major: i32,
    pub minor: i32,
    pub bugfix: i32,
}

/// `IndexWriter.SOURCE_FLUSH`.
pub const SOURCE_FLUSH: &str = "flush";
/// `IndexWriter.SOURCE_MERGE`.
pub const SOURCE_MERGE: &str = "merge";
/// `IndexWriter.SOURCE_ADDINDEXES_READERS`.
pub const SOURCE_ADDINDEXES_READERS: &str = "addIndexes(CodecReader...)";

/// `IndexWriter.setDiagnostics(info, source, details)`: the diagnostics a
/// flushed, merged or added segment's `.si` records -- Java's keys (`source`,
/// `lucene.version`, `os`, `os.arch`, `os.version`, `java.runtime.version`,
/// `java.vendor`, `timestamp`), then `details` (a merge's `mergeFactor` and
/// `mergeMaxNumSegments`).
///
/// The platform values are the JVM's spelling of this process's platform:
/// `os` is `os.name`'s (`Linux`, `Mac OS X`, `Windows`), `os.arch` is
/// `os.arch`'s (`amd64` for x86-64 outside macOS), `os.version` the Linux
/// kernel release (`/proc/sys/kernel/osrelease`) and `unknown` elsewhere.
/// There is no JVM, so `java.vendor` is `lucene-rust` and
/// `java.runtime.version` this crate's version. `timestamp` is the wall
/// clock in epoch milliseconds, as `Instant.now().toEpochMilli()`.
pub fn writer_diagnostics(
    source: &str,
    version: LuceneVersion,
    details: &[(&str, String)],
) -> Vec<(String, String)> {
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_millis());
    let mut out: Vec<(String, String)> = [
        ("source", source.to_string()),
        (
            "lucene.version",
            format!("{}.{}.{}", version.major, version.minor, version.bugfix),
        ),
        ("os", java_os_name(std::env::consts::OS).to_string()),
        (
            "os.arch",
            java_os_arch(std::env::consts::OS, std::env::consts::ARCH).to_string(),
        ),
        ("os.version", os_version().to_string()),
        (
            "java.runtime.version",
            env!("CARGO_PKG_VERSION").to_string(),
        ),
        ("java.vendor", "lucene-rust".to_string()),
        ("timestamp", timestamp.to_string()),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v))
    .collect();
    out.extend(details.iter().map(|(k, v)| ((*k).to_string(), v.clone())));
    out
}

/// The JVM's `os.name` for a Rust `std::env::consts::OS`.
fn java_os_name(os: &str) -> &str {
    match os {
        "linux" => "Linux",
        "macos" => "Mac OS X",
        "windows" => "Windows",
        "freebsd" => "FreeBSD",
        other => other,
    }
}

/// The JVM's `os.arch` for a Rust `std::env::consts::ARCH`: HotSpot says
/// `amd64` for x86-64 everywhere but macOS, and `x86` for 32-bit x86.
fn java_os_arch<'a>(os: &str, arch: &'a str) -> &'a str {
    match (os, arch) {
        ("macos", "x86_64") => "x86_64",
        (_, "x86_64") => "amd64",
        (_, "x86") => "x86",
        (_, other) => other,
    }
}

/// `os.version`: the kernel release on Linux, read once.
fn os_version() -> &'static str {
    static VERSION: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    VERSION.get_or_init(|| {
        std::fs::read_to_string("/proc/sys/kernel/osrelease")
            .ok()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "unknown".to_string())
    })
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
/// neither first nor last.
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
/// explicit marker) but not in the comparator. Both are kept so [`write()`]
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
    /// bytes -- by rank among the values compared, see
    /// [`crate::index_sorter`].
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
    /// merge writers produce -- the common case.
    pub fn long(field: impl Into<String>, reverse: bool, missing: Option<i64>) -> Self {
        Self {
            field: field.into(),
            reverse,
            kind: IndexSortKind::Numeric(NumericSortKey::Long(missing)),
        }
    }

    /// How this sort's per-document key is compared and what a document with
    /// no value compares as.
    ///
    /// This is the whole comparator contract: a consumer reads each
    /// document's key out of the right doc-values column (applying the
    /// selector for the multi-valued kinds), substitutes the sentinel for a
    /// document that has none, compares as [`SortKeyKind`] says, and then
    /// applies `reverse` -- **including to the sentinel**, which is an
    /// ordinary value inside `reverseMul * X.compare(a, b)`.
    ///
    /// The three byte-keyed kinds (`STRING`, `SortedSetSortField`,
    /// `BinarySortField`) all compare [`SortKeyKind::Ordinal`]: the key is
    /// the value's rank among the distinct values being compared, which is
    /// the term ordinal for a SORTED/SORTED_SET column and orders a BINARY
    /// column exactly as `BytesRef.compareTo` does -- see
    /// [`crate::index_sorter`], which computes it for every consumer.
    pub fn key_comparison(&self) -> (SortKeyKind, i64) {
        match &self.kind {
            IndexSortKind::Numeric(key) => key.key_comparison(),
            // A SORTED_NUMERIC FLOAT/DOUBLE column holds
            // `NumericUtils.floatToSortableInt`/`doubleToSortableLong`, not
            // the raw bits a NUMERIC one holds, and
            // `SortedNumericSelector.wrap` undoes that before the sorter sees
            // a value. Comparing the stored form as raw bits instead would
            // reverse the whole negative half of the ordering.
            IndexSortKind::SortedNumeric { key, .. } => match *key {
                NumericSortKey::Float(m) => (
                    SortKeyKind::SortableFloat,
                    float_to_sortable_int(m.unwrap_or(0.0)) as i64,
                ),
                NumericSortKey::Double(m) => (
                    SortKeyKind::SortableDouble,
                    double_to_sortable_long(m.unwrap_or(0.0)),
                ),
                key => key.key_comparison(),
            },
            IndexSortKind::String(missing)
            | IndexSortKind::SortedSet { missing, .. }
            | IndexSortKind::Binary(missing) => (SortKeyKind::Ordinal, missing.missing_ord()),
        }
    }
}

/// One sort field's comparator, resolved once out of an [`IndexSortField`].
///
/// Java rebuilds this per segment in `IndexSorter.*Sorter.getDocComparator`,
/// which closes over `reverseMul`, the pre-filled sentinel and the type's own
/// `compare`; resolving it once here is the same thing.
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
    /// The comparator for `sort` -- see [`IndexSortField::key_comparison`].
    pub fn new(sort: &IndexSortField) -> Self {
        let (kind, sentinel) = sort.key_comparison();
        Self {
            kind,
            sentinel,
            reverse: sort.reverse,
        }
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

/// SimpleText's metadata with typed index-sort fields. Unlike the binary
/// `.si` descriptor, its `Version` retains the prerelease component and the
/// segment name is part of the model passed to the writer.
pub type SimpleTextSegmentInfo =
    lucene_codecs::simpletext::segment_info::SimpleTextSegmentInfo<IndexSortField>;

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
    parse_with_format(buf, segment_id, SegmentInfoFormat::Lucene99)
}

/// `SimpleTextSegmentInfoFormat.read`, including typed `SortFieldProvider`
/// decoding. The `type` description line is informational, as in Java; the
/// named provider and its serialized bytes determine the sort.
///
/// # Errors
/// Invalid SimpleText framing, checksum, ID or sort-provider bytes. Trailing
/// provider bytes are rejected (Java checks this with `assert bytes.eof()`).
pub fn parse_simpletext(
    buf: &[u8],
    segment_name: &str,
    segment_id: &[u8; ID_LENGTH],
) -> Result<SimpleTextSegmentInfo> {
    let raw = lucene_codecs::simpletext::segment_info::read(buf, segment_name, segment_id)?;
    let mut fields = Vec::new();
    for entry in raw.index_sort {
        let mut input = SliceInput::new(&entry.bytes);
        fields.push(read_sort_field_provider(&mut input, &entry.provider)?);
        if input.remaining() != 0 {
            return Err(lucene_store::Error::Corrupted(format!(
                "trailing bytes in {} sort field: {}",
                entry.provider,
                input.remaining()
            ))
            .into());
        }
    }
    Ok(SimpleTextSegmentInfo {
        name: raw.name,
        id: raw.id,
        version: raw.version,
        min_version: raw.min_version,
        doc_count: raw.doc_count,
        is_compound_file: raw.is_compound_file,
        has_blocks: raw.has_blocks,
        diagnostics: raw.diagnostics,
        files: raw.files,
        attributes: raw.attributes,
        index_sort: fields,
    })
}

/// `SimpleTextSegmentInfoFormat.write`: encode the typed index sort using
/// each provider's own bytes. Adds the `.si` file to `si.files`, as Java does.
///
/// # Errors
/// An invalid codec file name.
pub fn write_simpletext(si: &mut SimpleTextSegmentInfo) -> Result<Vec<u8>> {
    use lucene_codecs::simpletext::segment_info::{self as simpletext, SortFieldEntry};
    let mut index_sort = Vec::new();
    for field in &si.index_sort {
        let mut bytes = Vec::new();
        write_sort_field_value(&mut bytes, field);
        index_sort.push(SortFieldEntry {
            provider: sort_field_provider(field).to_string(),
            description: describe_index_sort(Some(std::slice::from_ref(field))),
            bytes,
        });
    }
    let mut raw = simpletext::SimpleTextSegmentInfo {
        name: si.name.clone(),
        version: si.version,
        min_version: si.min_version,
        doc_count: si.doc_count,
        is_compound_file: si.is_compound_file,
        has_blocks: si.has_blocks,
        diagnostics: si.diagnostics.clone(),
        attributes: si.attributes.clone(),
        files: si.files.clone(),
        id: si.id,
        index_sort,
    };
    let bytes = simpletext::write(&mut raw)?;
    si.files = raw.files;
    Ok(bytes)
}

/// `SegmentInfo.namedForThisSegment` applied to every listed file: Lucene's
/// `SegmentInfo.setFiles` -- which every `.si` read ends in -- renames each
/// file to its own segment's name. `IndexWriter.addIndexes(Directory...)`
/// copies a segment's `.si` byte for byte under a new segment name, so the
/// list on disk still names the segment it came from.
pub fn named_for_segment(segment_name: &str, files: &[String]) -> Vec<String> {
    files
        .iter()
        .map(|f| {
            format!(
                "{segment_name}{}",
                lucene_codecs::simpletext::compound::strip_segment_name(f)
            )
        })
        .collect()
}

/// [`parse_for_segment`] for a committed segment.
pub fn parse_for_commit(
    buf: &[u8],
    sci: &crate::segment_infos::SegmentCommitInfo,
) -> Result<SegmentInfo> {
    parse_for_segment(buf, &sci.segment_name, &sci.segment_id, &sci.codec_name)
}

/// [`parse_for_codec`] for the segment `segment_name`: its file list is named
/// for that segment (see [`named_for_segment`]), whatever segment name the
/// `.si` was written under.
pub fn parse_for_segment(
    buf: &[u8],
    segment_name: &str,
    segment_id: &[u8; ID_LENGTH],
    codec_name: &str,
) -> Result<SegmentInfo> {
    let mut si = parse_for_codec(buf, segment_id, codec_name)?;
    si.files = named_for_segment(segment_name, &si.files);
    Ok(si)
}

/// [`parse`] for a segment written by codec `codec_name` (the name its
/// `segments_N` entry records): `Lucene90`..`Lucene95` read the `.si` with
/// `Lucene90SegmentInfoFormat`, which has no `HasBlocks` byte, and everything
/// from `Lucene99` on with `Lucene99SegmentInfoFormat` -- under the same
/// codec name and version, so only the codec can tell them apart (see
/// [`lucene_codecs::codecs`]). An unknown codec is
/// [`Error::UnknownCodec`], as `Codec.forName` throws.
pub fn parse_for_codec(
    buf: &[u8],
    segment_id: &[u8; ID_LENGTH],
    codec_name: &str,
) -> Result<SegmentInfo> {
    let codec = lucene_codecs::codecs::for_name(codec_name)
        .ok_or_else(|| Error::UnknownCodec(codec_name.to_string()))?;
    parse_with_format(buf, segment_id, codec.segment_info)
}

/// [`parse`] with an explicit `.si` format generation: the only difference
/// is whether `HasBlocks` is on the wire (`Lucene90SegmentInfoFormat` reads
/// every segment as `hasBlocks == false`).
pub fn parse_with_format(
    buf: &[u8],
    segment_id: &[u8; ID_LENGTH],
    format: SegmentInfoFormat,
) -> Result<SegmentInfo> {
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
    let has_blocks = match format {
        SegmentInfoFormat::Lucene90 => false,
        SegmentInfoFormat::Lucene99 => input.read_byte()? == 1,
    };

    let diagnostics = input.read_map_of_strings()?;
    let files = input.read_set_of_strings()?;
    let attributes = input.read_map_of_strings()?;

    let index_sort = read_index_sort(&mut input)?;

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

/// An index sort encoded as a `.si` encodes it (`Lucene99SegmentInfoFormat`):
/// a vint count, then each field's `SortFieldProvider` name and body.
/// `None` for a count of 0.
///
/// # Errors
/// A truncated or malformed encoding, or a sort field this port cannot read.
pub fn parse_index_sort(buf: &[u8]) -> Result<Option<Vec<IndexSortField>>> {
    read_index_sort(&mut SliceInput::new(buf))
}

/// The `.si`'s sort block: a vint count, then each field's provider name and
/// body. `None` for a count of 0.
fn read_index_sort(input: &mut SliceInput) -> Result<Option<Vec<IndexSortField>>> {
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
    if num_sort_fields == 0 {
        return Ok(None);
    }
    let mut fields = Vec::with_capacity(num_sort_fields as usize);
    for _ in 0..num_sort_fields {
        fields.push(read_sort_field(input)?);
    }
    Ok(Some(fields))
}

/// `Lucene99SegmentInfoFormat.parseSegmentInfo`'s per-sort-field body:
/// `SortFieldProvider.forName(input.readString()).readSortField(input)`.
fn read_sort_field(input: &mut SliceInput) -> Result<IndexSortField> {
    let provider = input.read_string()?;
    read_sort_field_provider(input, &provider)
}

/// `SortFieldProvider.forName(provider).readSortField(input)`, shared by
/// binary segment infos and SimpleText's separately named byte arrays.
fn read_sort_field_provider(input: &mut SliceInput, provider: &str) -> Result<IndexSortField> {
    match provider {
        PROVIDER_SORT_FIELD => read_plain_sort_field(input),
        PROVIDER_SORTED_NUMERIC => read_sorted_numeric_sort_field(input),
        PROVIDER_SORTED_SET => read_sorted_set_sort_field(input),
        PROVIDER_BINARY => read_binary_sort_field(input),
        _ => Err(Error::UnknownSortFieldProvider(provider.to_string())),
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

/// Writes the binary `.si` provider's SPI name, then its `serialize` bytes. The
/// byte-level inverse of [`read_sort_field`] for every [`IndexSortKind`].
fn write_sort_field(out: &mut Vec<u8>, sf: &IndexSortField) {
    out.write_string(sort_field_provider(sf));
    write_sort_field_value(out, sf);
}

fn sort_field_provider(sf: &IndexSortField) -> &'static str {
    match sf.kind {
        IndexSortKind::Numeric(_) | IndexSortKind::String(_) => PROVIDER_SORT_FIELD,
        IndexSortKind::SortedNumeric { .. } => PROVIDER_SORTED_NUMERIC,
        IndexSortKind::SortedSet { .. } => PROVIDER_SORTED_SET,
        IndexSortKind::Binary(_) => PROVIDER_BINARY,
    }
}

/// `SortFieldProvider.write` writes only the provider's `serialize` bytes;
/// SimpleText stores its SPI name on a separate text line.
fn write_sort_field_value(out: &mut Vec<u8>, sf: &IndexSortField) {
    let reverse = if sf.reverse { 1 } else { 0 };
    match &sf.kind {
        IndexSortKind::Numeric(key) => {
            out.write_string(&sf.field);
            out.write_string(key.type_name());
            out.write_i32(reverse);
            write_numeric_missing(out, key);
        }
        IndexSortKind::String(missing) => {
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

/// `Float.toString`, including scientific notation and Java's tie rules.
fn java_float_string(v: f32) -> String {
    lucene_util::geo::java_float_string(v)
}

/// `Double.toString`, the 64-bit twin of [`java_float_string`].
fn java_double_string(v: f64) -> String {
    lucene_util::geo::java_double_string(v)
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
mod tests;
