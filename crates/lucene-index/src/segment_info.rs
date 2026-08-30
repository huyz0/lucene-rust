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
//! **What this port models.** [`IndexSortField`] carries only
//! `(field, reverse, missing-first-or-last)` — the shape this port's
//! sort-on-flush writer (`segment_writer::flush_sorted_stored_only_segment`)
//! and merge-with-sort path actually produce, which is a single-valued
//! `LONG` sort whose missing docs are pinned to `Long.MIN_VALUE`
//! (missing-first) or `Long.MAX_VALUE` (missing-last). [`write`] therefore
//! emits exactly that: provider `"SortField"`, type `"LONG"`, an explicit
//! missing value of `i64::MIN`/`i64::MAX` — real, byte-compatible Lucene
//! bytes a Java `SortFieldProvider.forName("SortField").readSortField` reads
//! back as the identical `SortField`.
//!
//! [`parse`] is deliberately more permissive than [`write`]: it decodes every
//! one of the four providers above and every `SortField.Type` Java can
//! round-trip, and only then *lowers* the result onto [`IndexSortField`].
//! An encoding that is valid Lucene but that `(field, reverse, first/last)`
//! cannot represent faithfully — a numeric sort with no missing value (Java
//! treats missing as `0`, which is neither first nor last), a numeric sort
//! with an arbitrary missing sentinel, a non-`MIN` multi-value selector, or a
//! `SCORE`/`DOC`/`CUSTOM`/`REWRITEABLE`/`STRING_VAL` type — is rejected as
//! [`Error::UnsupportedSortField`] naming exactly what it was, rather than
//! silently lowered onto a sort order this port would then get wrong. See
//! `docs/parity.md` for the tracked gap.

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

/// Which **sentinel** real Lucene substitutes for a document that has no
/// value for the index-sort field (`SortField.setMissingValue`), which
/// [`write_sort_field`] emits verbatim into the `.si`.
///
/// It names the sentinel, **not** the end of the finished order the document
/// lands at. Lucene's comparator for a `LONG` sort is
/// `reverseMul * Long.compare(values[d1], values[d2])` over an array
/// pre-filled with the sentinel (`IndexSorter.LongSorter.getDocComparator`),
/// so the sentinel is compared like any other value and `reverse` applies to
/// it too. The two readings coincide only for an ascending sort:
///
/// | variant | sentinel | ascending | descending |
/// |---|---|---|---|
/// | `First` | `Long.MIN_VALUE` | missing docs first | missing docs **last** |
/// | `Last` | `Long.MAX_VALUE` | missing docs last | missing docs **first** |
///
/// `crate::segment_writer::sort_key_rank` is the one place this is
/// implemented, and `CheckIndex.testSort` -- Lucene's and this port's -- is
/// what rejects a segment whose physical order disagrees with it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SortMissingValue {
    /// `SortField.setMissingValue(Long.MIN_VALUE)`: a doc with no value
    /// compares as the smallest possible value, so it sorts **first** under
    /// an ascending sort and **last** under a reversed one.
    First,
    /// `SortField.setMissingValue(Long.MAX_VALUE)`: a doc with no value
    /// compares as the largest possible value, so it sorts **last** under an
    /// ascending sort and **first** under a reversed one.
    Last,
}

/// One field of an index sort descriptor (real Lucene's
/// `SegmentInfo.indexSort` is a `Sort` of one or more `SortField`s -- this
/// port's [`SegmentInfo::index_sort`] is a priority-ordered, non-empty
/// `Vec<IndexSortField>`). This carries only the `(field, reverse,
/// missing-first-or-last)` triple this port's writers produce and its
/// sorters act on -- see this module's "Index-sort encoding" doc section for
/// exactly which real-Lucene encodings [`parse`] lowers onto it and which it
/// rejects.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexSortField {
    pub field: String,
    /// `false` == ascending, `true` == descending (real Lucene's
    /// `SortField.reverse`).
    pub reverse: bool,
    pub missing: SortMissingValue,
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

/// `SortedNumericSelector.Type.MIN` / `SortedSetSelector.Type.MIN` ordinal --
/// the only multi-value selector [`IndexSortField`] can represent (it has no
/// selector of its own, so anything else would silently change which value a
/// doc sorts by).
const SELECTOR_MIN: i32 = 0;
/// `SortedNumericSelector.Type.values().length` (MIN, MAX).
const SORTED_NUMERIC_SELECTOR_COUNT: i32 = 2;
/// `SortedSetSelector.Type.values().length` (MIN, MAX, MIDDLE_MIN, MIDDLE_MAX).
const SORTED_SET_SELECTOR_COUNT: i32 = 4;

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
/// this sortable form, so the "is it +/-infinity" test has to be done in the
/// same space.
fn float_to_sortable_int(v: f32) -> i32 {
    let bits = v.to_bits() as i32;
    bits ^ ((bits >> 31) & 0x7fff_ffff)
}

/// `NumericUtils.doubleToSortableLong`.
fn double_to_sortable_long(v: f64) -> i64 {
    let bits = v.to_bits() as i64;
    bits ^ ((bits >> 63) & 0x7fff_ffff_ffff_ffff)
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

/// Lowers a decoded numeric missing value onto [`SortMissingValue`]. Only the
/// two sentinels this port's own sorters mean by "first"/"last" are
/// representable; anything else (including Java's default *absent* missing
/// value, which it treats as `0`) is rejected rather than silently rounded to
/// one of them.
fn numeric_missing(field: &str, type_name: &str, raw: i64) -> Result<SortMissingValue> {
    let (first, last) = match type_name {
        TYPE_INT => (i32::MIN as i64, i32::MAX as i64),
        TYPE_LONG => (i64::MIN, i64::MAX),
        TYPE_FLOAT => (
            float_to_sortable_int(f32::NEG_INFINITY) as i64,
            float_to_sortable_int(f32::INFINITY) as i64,
        ),
        TYPE_DOUBLE => (
            double_to_sortable_long(f64::NEG_INFINITY),
            double_to_sortable_long(f64::INFINITY),
        ),
        _ => unreachable!("numeric_missing called with non-numeric type {type_name}"),
    };
    if raw == first {
        Ok(SortMissingValue::First)
    } else if raw == last {
        Ok(SortMissingValue::Last)
    } else {
        unsupported(
            field,
            format!(
                "{type_name} missing value {raw} is neither the sort-first nor the sort-last \
                 sentinel; this port models only first/last placement"
            ),
        )
    }
}

const TYPE_INT: &str = "INT";
const TYPE_LONG: &str = "LONG";
const TYPE_FLOAT: &str = "FLOAT";
const TYPE_DOUBLE: &str = "DOUBLE";
const TYPE_STRING: &str = "STRING";
/// Every remaining `SortField.Type` constant. They are legal enum names (so
/// `readType` accepts them), but Java's own `SortField.Provider` refuses to
/// deserialize a missing value for them and none can be an index sort.
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

/// Reads the numeric missing value for `type_name` in Java's encoding
/// (`SortField.serialize`'s `case INT/LONG/FLOAT/DOUBLE`).
fn read_numeric_missing(
    input: &mut SliceInput,
    field: &str,
    type_name: &str,
) -> Result<SortMissingValue> {
    let raw = match type_name {
        TYPE_INT | TYPE_FLOAT => input.read_i32()? as i64,
        TYPE_LONG | TYPE_DOUBLE => input.read_i64()?,
        other => {
            return unsupported(
                field,
                format!("cannot deserialize a missing value for sort type {other}"),
            )
        }
    };
    numeric_missing(field, type_name, raw)
}

/// `SortField.Provider.readSortField`.
fn read_plain_sort_field(input: &mut SliceInput) -> Result<IndexSortField> {
    let field = input.read_string()?;
    let type_name = read_type(input)?;
    let reverse = read_reverse(input)?;
    let has_missing = input.read_i32()? == 1;
    if !has_missing {
        return unsupported(
            &field,
            format!(
                "{type_name} sort has no missing value; Java sorts such docs as if they held \
                 0/the empty ord, which is neither first nor last"
            ),
        );
    }
    let missing = if type_name == TYPE_STRING {
        // Java: `missingString == 1` -> STRING_FIRST, else STRING_LAST.
        if input.read_i32()? == 1 {
            SortMissingValue::First
        } else {
            SortMissingValue::Last
        }
    } else {
        read_numeric_missing(input, &field, &type_name)?
    };
    Ok(IndexSortField {
        field,
        reverse,
        missing,
    })
}

/// `SortedNumericSortField.readSelectorType` / `SortedSetSortField.readSelectorType`:
/// an out-of-range ordinal is a hard error, a valid non-`MIN` one is a
/// representable-scope rejection.
fn read_selector(
    input: &mut SliceInput,
    provider: &'static str,
    count: i32,
    field: &str,
) -> Result<()> {
    let selector = input.read_i32()?;
    if selector >= count || selector < 0 {
        return Err(Error::UnknownSortSelector { provider, selector });
    }
    if selector != SELECTOR_MIN {
        return unsupported(
            field,
            format!("{provider} selector ordinal {selector} is not MIN; this port has no selector"),
        );
    }
    Ok(())
}

/// `SortedNumericSortField.Provider.readSortField`.
fn read_sorted_numeric_sort_field(input: &mut SliceInput) -> Result<IndexSortField> {
    let field = input.read_string()?;
    let type_name = read_type(input)?;
    let reverse = read_reverse(input)?;
    read_selector(
        input,
        PROVIDER_SORTED_NUMERIC,
        SORTED_NUMERIC_SELECTOR_COUNT,
        &field,
    )?;
    let has_missing = input.read_i32()? == 1;
    if !has_missing {
        return unsupported(
            &field,
            format!("{type_name} sorted-numeric sort has no missing value"),
        );
    }
    let missing = read_numeric_missing(input, &field, &type_name)?;
    Ok(IndexSortField {
        field,
        reverse,
        missing,
    })
}

/// The shared tail of `SortedSetSortField.serialize` and
/// `BinarySortField.serialize`: `0 == no missing value`, `1 == STRING_FIRST`,
/// `2 == STRING_LAST`.
fn read_string_missing_marker(input: &mut SliceInput, field: &str) -> Result<SortMissingValue> {
    match input.read_i32()? {
        1 => Ok(SortMissingValue::First),
        2 => Ok(SortMissingValue::Last),
        0 => unsupported(field, "sort has no missing value"),
        other => unsupported(field, format!("unknown missing-value marker {other}")),
    }
}

/// `SortedSetSortField.Provider.readSortField`.
fn read_sorted_set_sort_field(input: &mut SliceInput) -> Result<IndexSortField> {
    let field = input.read_string()?;
    let reverse = read_reverse(input)?;
    read_selector(
        input,
        PROVIDER_SORTED_SET,
        SORTED_SET_SELECTOR_COUNT,
        &field,
    )?;
    let missing = read_string_missing_marker(input, &field)?;
    Ok(IndexSortField {
        field,
        reverse,
        missing,
    })
}

/// `BinarySortField.Provider.readSortField`.
fn read_binary_sort_field(input: &mut SliceInput) -> Result<IndexSortField> {
    let field = input.read_string()?;
    let reverse = read_reverse(input)?;
    let missing = read_string_missing_marker(input, &field)?;
    Ok(IndexSortField {
        field,
        reverse,
        missing,
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

/// Writes one sort field as `SortField.Provider` + `SortField.serialize`
/// would: a single-valued `LONG` sort whose missing docs are pinned to
/// `Long.MIN_VALUE` (first) or `Long.MAX_VALUE` (last), which is exactly what
/// this port's sort-on-flush/sort-on-merge writers produce (see
/// `segment_writer::sort_key_rank`).
fn write_sort_field(out: &mut Vec<u8>, sf: &IndexSortField) {
    out.write_string(PROVIDER_SORT_FIELD);
    out.write_string(&sf.field);
    out.write_string(TYPE_LONG);
    out.write_i32(if sf.reverse { 1 } else { 0 });
    out.write_i32(1); // missing value present
    out.write_i64(match sf.missing {
        SortMissingValue::First => i64::MIN,
        SortMissingValue::Last => i64::MAX,
    });
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

    #[test]
    fn two_field_sort_count_parses_both_fields() {
        let mut b = SiBuilder::valid();
        b.num_sort_fields = 2;
        let mut bytes = SiBuilder::long_sort_field_bytes("price", false, true);
        bytes.extend(SiBuilder::long_sort_field_bytes("timestamp", true, false));
        b.sort_field_bytes = bytes;
        let si = parse(&b.build(), &b.id).unwrap();
        let fields = si.index_sort.unwrap();
        assert_eq!(fields.len(), 2);
        assert_eq!(fields[0].field, "price");
        assert!(!fields[0].reverse);
        assert_eq!(fields[0].missing, SortMissingValue::Last);
        assert_eq!(fields[1].field, "timestamp");
        assert!(fields[1].reverse);
        assert_eq!(fields[1].missing, SortMissingValue::First);
    }

    #[test]
    fn three_field_sort_round_trips_with_mixed_policies() {
        let mut si = sample_si();
        si.index_sort = Some(vec![
            IndexSortField {
                field: "a".to_string(),
                reverse: false,
                missing: SortMissingValue::First,
            },
            IndexSortField {
                field: "b".to_string(),
                reverse: true,
                missing: SortMissingValue::Last,
            },
            IndexSortField {
                field: "c".to_string(),
                reverse: false,
                missing: SortMissingValue::Last,
            },
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
        let fields = si.index_sort.unwrap();
        assert_eq!(fields.len(), 1);
        assert_eq!(fields[0].field, "price");
        assert!(!fields[0].reverse);
        assert_eq!(fields[0].missing, SortMissingValue::Last);
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

    #[test]
    fn missing_less_numeric_sort_is_rejected_not_guessed() {
        let mut b = SiBuilder::valid();
        b.num_sort_fields = 1;
        b.sort_field_bytes = SiBuilder::plain_sort_field_bytes("price", TYPE_LONG, 0, None);
        let err = parse(&b.build(), &b.id).unwrap_err();
        assert!(
            matches!(&err, Error::UnsupportedSortField { field, .. } if field == "price"),
            "unexpected error: {err}"
        );
    }

    /// An arbitrary numeric missing value (Java allows any long) is neither
    /// "first" nor "last" -- rejecting is the only honest lowering.
    #[test]
    fn arbitrary_numeric_missing_value_is_rejected() {
        let mut b = SiBuilder::valid();
        b.num_sort_fields = 1;
        b.sort_field_bytes =
            SiBuilder::plain_sort_field_bytes("price", TYPE_LONG, 0, Some(&42i64.to_le_bytes()));
        assert!(matches!(
            parse(&b.build(), &b.id),
            Err(Error::UnsupportedSortField { .. })
        ));
    }

    #[test]
    fn int_float_double_and_string_sentinels_all_lower() {
        let cases: Vec<(&str, Vec<u8>, SortMissingValue)> = vec![
            (
                TYPE_INT,
                i32::MIN.to_le_bytes().to_vec(),
                SortMissingValue::First,
            ),
            (
                TYPE_INT,
                i32::MAX.to_le_bytes().to_vec(),
                SortMissingValue::Last,
            ),
            (
                TYPE_FLOAT,
                float_to_sortable_int(f32::NEG_INFINITY)
                    .to_le_bytes()
                    .to_vec(),
                SortMissingValue::First,
            ),
            (
                TYPE_FLOAT,
                float_to_sortable_int(f32::INFINITY).to_le_bytes().to_vec(),
                SortMissingValue::Last,
            ),
            (
                TYPE_DOUBLE,
                double_to_sortable_long(f64::NEG_INFINITY)
                    .to_le_bytes()
                    .to_vec(),
                SortMissingValue::First,
            ),
            (
                TYPE_DOUBLE,
                double_to_sortable_long(f64::INFINITY)
                    .to_le_bytes()
                    .to_vec(),
                SortMissingValue::Last,
            ),
            // STRING: 1 == STRING_FIRST, anything else == STRING_LAST.
            (
                TYPE_STRING,
                1i32.to_le_bytes().to_vec(),
                SortMissingValue::First,
            ),
            (
                TYPE_STRING,
                0i32.to_le_bytes().to_vec(),
                SortMissingValue::Last,
            ),
        ];
        for (type_name, missing, expected) in cases {
            let mut b = SiBuilder::valid();
            b.num_sort_fields = 1;
            b.sort_field_bytes =
                SiBuilder::plain_sort_field_bytes("f", type_name, 0, Some(&missing));
            let si = parse(&b.build(), &b.id)
                .unwrap_or_else(|e| panic!("type {type_name} should parse: {e}"));
            assert_eq!(si.index_sort.unwrap()[0].missing, expected, "{type_name}");
        }
    }

    #[test]
    fn sorted_numeric_sorted_set_and_binary_providers_all_decode() {
        // SortedNumericSortField: field, type, reverse, selector, hasMissing, value
        let mut sn = Vec::new();
        write_string(&mut sn, PROVIDER_SORTED_NUMERIC);
        write_string(&mut sn, "sn");
        write_string(&mut sn, TYPE_LONG);
        sn.extend_from_slice(&1i32.to_le_bytes()); // reverse
        sn.extend_from_slice(&0i32.to_le_bytes()); // selector MIN
        sn.extend_from_slice(&1i32.to_le_bytes()); // hasMissing
        sn.extend_from_slice(&i64::MAX.to_le_bytes());

        // SortedSetSortField: field, reverse, selector, missing marker
        let mut ss = Vec::new();
        write_string(&mut ss, PROVIDER_SORTED_SET);
        write_string(&mut ss, "ss");
        ss.extend_from_slice(&0i32.to_le_bytes());
        ss.extend_from_slice(&0i32.to_le_bytes()); // selector MIN
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
        assert_eq!(fields.len(), 3);
        assert_eq!(fields[0].field, "sn");
        assert!(fields[0].reverse);
        assert_eq!(fields[0].missing, SortMissingValue::Last);
        assert_eq!(fields[1].field, "ss");
        assert_eq!(fields[1].missing, SortMissingValue::First);
        assert_eq!(fields[2].field, "bin");
        assert_eq!(fields[2].missing, SortMissingValue::Last);
    }

    #[test]
    fn out_of_range_selector_ordinal_rejected() {
        let mut ss = Vec::new();
        write_string(&mut ss, PROVIDER_SORTED_SET);
        write_string(&mut ss, "ss");
        ss.extend_from_slice(&0i32.to_le_bytes());
        ss.extend_from_slice(&9i32.to_le_bytes()); // no such SortedSetSelector.Type
        ss.extend_from_slice(&1i32.to_le_bytes());
        let mut b = SiBuilder::valid();
        b.num_sort_fields = 1;
        b.sort_field_bytes = ss;
        assert!(matches!(
            parse(&b.build(), &b.id),
            Err(Error::UnknownSortSelector { selector: 9, .. })
        ));
    }

    /// A valid-but-unrepresentable selector (MAX) must be refused, not
    /// silently treated as MIN -- it changes which value a doc sorts by.
    #[test]
    fn non_min_selector_rejected_rather_than_silently_downgraded() {
        let mut ss = Vec::new();
        write_string(&mut ss, PROVIDER_SORTED_SET);
        write_string(&mut ss, "ss");
        ss.extend_from_slice(&0i32.to_le_bytes());
        ss.extend_from_slice(&1i32.to_le_bytes()); // MAX
        ss.extend_from_slice(&1i32.to_le_bytes());
        let mut b = SiBuilder::valid();
        b.num_sort_fields = 1;
        b.sort_field_bytes = ss;
        assert!(matches!(
            parse(&b.build(), &b.id),
            Err(Error::UnsupportedSortField { .. })
        ));
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
        si.index_sort = Some(vec![IndexSortField {
            field: "timestamp".to_string(),
            reverse: false,
            missing: SortMissingValue::First,
        }]);
        let bytes = write(&si, "");
        let parsed = parse(&bytes, &si.id).unwrap();
        let fields = parsed.index_sort.unwrap();
        let sf = &fields[0];
        assert_eq!(sf.field, "timestamp");
        assert!(!sf.reverse);
        assert_eq!(sf.missing, SortMissingValue::First);
    }

    #[test]
    fn write_descending_index_sort_with_missing_last_round_trips() {
        let mut si = sample_si();
        si.index_sort = Some(vec![IndexSortField {
            field: "score".to_string(),
            reverse: true,
            missing: SortMissingValue::Last,
        }]);
        let bytes = write(&si, "");
        let parsed = parse(&bytes, &si.id).unwrap();
        let fields = parsed.index_sort.unwrap();
        let sf = &fields[0];
        assert_eq!(sf.field, "score");
        assert!(sf.reverse);
        assert_eq!(sf.missing, SortMissingValue::Last);
    }
}
