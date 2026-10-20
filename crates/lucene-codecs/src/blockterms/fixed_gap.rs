//! Port of `FixedGapTermsIndexWriter` and `FixedGapTermsIndexReader`: every
//! Nth term is an index term, its shortest prefix that sorts after the term
//! before it held in one byte array, with two monotonic packed arrays per
//! field -- each index term's block offset in the `.tib` and its prefix's
//! end in the bytes. Supports ords: index term `i` is term `i * interval`.

use std::collections::BTreeMap;

use lucene_store::codec_util::{self, FOOTER_LENGTH};
use lucene_store::data_input::{DataInput, SliceInput};
use lucene_store::data_output::DataOutput;

use super::{corrupt, seek_dir, Error, Result, TermStats, TermsIndexWriterBase};
use crate::field_infos::FieldInfo;
use crate::monotonic_block_packed::{MonotonicBlockPackedReader, MonotonicBlockPackedWriter};
use crate::segment_state::{SegmentReadState, SegmentWriteState};

/// `FixedGapTermsIndexWriter.TERMS_INDEX_EXTENSION`.
pub const TERMS_INDEX_EXTENSION: &str = "tii";
/// `FixedGapTermsIndexWriter.CODEC_NAME`.
pub const CODEC_NAME: &str = "FixedGapTermsIndex";
/// `FixedGapTermsIndexWriter.VERSION_CURRENT` (and `VERSION_START`).
pub const VERSION_CURRENT: i32 = 4;
/// `FixedGapTermsIndexWriter.BLOCKSIZE`: the packed arrays' block size.
pub const BLOCKSIZE: i32 = 4096;
/// `FixedGapTermsIndexWriter.DEFAULT_TERM_INDEX_INTERVAL`.
pub const DEFAULT_TERM_INDEX_INTERVAL: i32 = 32;
/// `PackedInts.VERSION_CURRENT`, which the `.tii` records.
const PACKED_INTS_VERSION_CURRENT: i32 = 2;

/// `FixedGapTermsIndexWriter.SimpleFieldWriter`.
#[derive(Debug)]
struct SimpleFieldWriter {
    number: i32,
    num_index_terms: i32,
    index_start: i64,
    terms_start: i64,
    packed_index_start: i64,
    packed_offsets_start: i64,
    num_terms: i64,
    /// `termOffsets`: the values the packed writer takes, `0` first.
    term_offsets: Vec<i64>,
    current_offset: i64,
    /// `termAddresses`.
    term_addresses: Vec<i64>,
    last_term: Vec<u8>,
}

/// `FixedGapTermsIndexWriter`: the `.tii` of one group.
#[derive(Debug)]
pub struct FixedGapTermsIndexWriter {
    /// `out`.
    out: Vec<u8>,
    term_index_interval: i64,
    /// The finished fields, in the order they were added.
    fields: Vec<SimpleFieldWriter>,
    /// The field being written.
    current: Option<SimpleFieldWriter>,
}

impl FixedGapTermsIndexWriter {
    /// `new FixedGapTermsIndexWriter(state, termIndexInterval)`: the header
    /// and the interval, packed-ints version and block size.
    ///
    /// # Errors
    /// [`Error::IllegalArgument`] for an interval `<= 0`.
    pub fn new(state: &SegmentWriteState<'_>, term_index_interval: i32) -> Result<Self> {
        if term_index_interval <= 0 {
            return Err(Error::IllegalArgument(format!(
                "invalid termIndexInterval: {term_index_interval}"
            )));
        }
        let mut out = Vec::new();
        codec_util::write_index_header(
            &mut out,
            CODEC_NAME,
            VERSION_CURRENT,
            state.segment_id,
            state.segment_suffix,
        );
        out.write_vint(term_index_interval);
        out.write_vint(PACKED_INTS_VERSION_CURRENT);
        out.write_vint(BLOCKSIZE);
        Ok(FixedGapTermsIndexWriter {
            out,
            term_index_interval: i64::from(term_index_interval),
            fields: Vec::new(),
            current: None,
        })
    }
}

/// `indexedTermPrefixLength(priorTerm, indexedTerm)`:
/// `StringHelper.sortKeyLength`.
fn indexed_term_prefix_length(prior_term: &[u8], indexed_term: &[u8]) -> Result<usize> {
    lucene_util::string_helper::sort_key_length(prior_term, indexed_term)
        .filter(|&n| n <= indexed_term.len())
        .ok_or_else(|| Error::IllegalArgument("terms out of order".into()))
}

/// The packed values of `values`, through a `MonotonicBlockPackedWriter`.
fn write_monotonic(out: &mut Vec<u8>, values: &[i64]) {
    let mut writer = MonotonicBlockPackedWriter::new(out, BLOCKSIZE as usize);
    for &v in values {
        writer.add(v);
    }
    writer.finish();
}

impl TermsIndexWriterBase for FixedGapTermsIndexWriter {
    /// The `.tii`.
    type Files = Vec<u8>;

    /// `addField(field, termsFilePointer)`: a `SimpleFieldWriter`, its
    /// offsets starting with `0`.
    fn add_field(&mut self, field: &FieldInfo, terms_file_pointer: i64) -> Result<()> {
        self.current = Some(SimpleFieldWriter {
            number: field.number,
            num_index_terms: 0,
            index_start: self.out.len() as i64,
            terms_start: terms_file_pointer,
            packed_index_start: 0,
            packed_offsets_start: 0,
            num_terms: 0,
            term_offsets: vec![0],
            current_offset: 0,
            term_addresses: Vec::new(),
            last_term: Vec::new(),
        });
        Ok(())
    }

    /// `SimpleFieldWriter.checkIndexTerm`: the first term and every
    /// `interval`th after it; the term before an index term is kept.
    // ARITH: `num_terms` counts a field's terms; the interval is positive.
    #[allow(clippy::arithmetic_side_effects)]
    fn check_index_term(&mut self, text: &[u8], _stats: TermStats) -> bool {
        let interval = self.term_index_interval;
        let Some(f) = self.current.as_mut() else {
            return false;
        };
        let n = f.num_terms;
        f.num_terms += 1;
        // First term is first indexed term:
        if n % interval == 0 {
            true
        } else {
            if f.num_terms % interval == 0 {
                // save last term just before next index term so we can
                // compute wasted suffix
                f.last_term.clear();
                f.last_term.extend_from_slice(text);
            }
            false
        }
    }

    /// `SimpleFieldWriter.add`: the index term's distinguishing prefix, its
    /// block's offset and the prefix's end.
    // ARITH: the offset sums prefix lengths of the field's index terms; a
    // block's file pointer is past the field's start.
    #[allow(clippy::arithmetic_side_effects)]
    fn add(&mut self, text: &[u8], _stats: TermStats, terms_file_pointer: i64) -> Result<()> {
        let f = self
            .current
            .as_mut()
            .ok_or(Error::IllegalState("no field was added"))?;
        let indexed_term_length = if f.num_index_terms == 0 {
            // no previous term: no bytes to write
            0
        } else {
            indexed_term_prefix_length(&f.last_term, text)?
        };
        // write only the min prefix that shows the diff against prior term
        self.out.write_bytes(&text[..indexed_term_length]);
        // save delta terms pointer
        f.term_addresses.push(terms_file_pointer - f.terms_start);
        // save term length (in bytes)
        f.current_offset += indexed_term_length as i64;
        f.term_offsets.push(f.current_offset);
        f.last_term.clear();
        f.last_term.extend_from_slice(text);
        f.num_index_terms += 1;
        Ok(())
    }

    /// `SimpleFieldWriter.finish`: the two packed arrays.
    fn finish(&mut self, _terms_file_pointer: i64) -> Result<()> {
        let mut f = self
            .current
            .take()
            .ok_or(Error::IllegalState("no field was added"))?;
        // write primary terms dict offsets
        f.packed_index_start = self.out.len() as i64;
        write_monotonic(&mut self.out, &f.term_addresses);
        f.packed_offsets_start = self.out.len() as i64;
        // write offsets into the byte[] terms
        write_monotonic(&mut self.out, &f.term_offsets);
        f.term_addresses = Vec::new();
        f.term_offsets = Vec::new();
        self.fields.push(f);
        Ok(())
    }

    /// `close()`: the directory of the fields with index terms, its offset
    /// and the footer.
    fn close(mut self) -> Result<Vec<u8>> {
        let dir_start = self.out.len() as i64;
        let non_null: Vec<&SimpleFieldWriter> = self
            .fields
            .iter()
            .filter(|f| f.num_index_terms > 0)
            .collect();
        let count = i32::try_from(non_null.len())
            .map_err(|_| Error::IllegalArgument("too many fields".into()))?;
        let mut dir = Vec::new();
        dir.write_vint(count);
        for f in non_null {
            dir.write_vint(f.number);
            dir.write_vint(f.num_index_terms);
            dir.write_vlong(f.terms_start);
            dir.write_vlong(f.index_start);
            dir.write_vlong(f.packed_index_start);
            dir.write_vlong(f.packed_offsets_start);
        }
        self.out.write_bytes(&dir);
        // `writeTrailer(dirStart)`.
        self.out.write_i64(dir_start);
        codec_util::write_footer(&mut self.out);
        Ok(self.out)
    }
}

/// `FixedGapTermsIndexReader.FieldIndexData`.
#[derive(Debug)]
struct FieldIndexData {
    /// `termOffsets`, decoded: each index term's prefix as a range of the
    /// reader's bytes.
    index_terms: Vec<(usize, usize)>,
    /// `termsStart + termsDictOffsets`, decoded: each index term's block.
    block_pointers: Vec<i64>,
    num_index_terms: i64,
}

/// `FixedGapTermsIndexReader`: every field's index terms, loaded at open.
#[derive(Debug)]
pub struct FixedGapTermsIndexReader {
    index_interval: i64,
    /// `termBytesReader`: all fields' prefixes (Java's `PagedBytes`).
    term_bytes: Vec<u8>,
    fields: BTreeMap<String, FieldIndexData>,
}

impl FixedGapTermsIndexReader {
    /// `new FixedGapTermsIndexReader(state)` over the `.tii`: its header,
    /// whole-file checksum, settings and directory, and each field's prefix
    /// bytes and packed arrays.
    ///
    /// # Errors
    /// `Corrupted` for a bad header or checksum, an interval below 1, a
    /// directory entry Java refuses (or one with no index terms, which Java
    /// would fail on at the first seek), a field number `state.field_infos`
    /// lacks, a duplicate field, or packed arrays that do not decode.
    pub fn open(tii: &[u8], state: &SegmentReadState<'_>) -> Result<Self> {
        let mut input = SliceInput::new(tii);
        codec_util::check_index_header(
            &mut input,
            CODEC_NAME,
            VERSION_CURRENT,
            VERSION_CURRENT,
            state.segment_id,
            state.segment_suffix,
        )?;
        // `CodecUtil.checksumEntireFile(in)`.
        let end = tii
            .len()
            .checked_sub(FOOTER_LENGTH)
            .ok_or_else(|| corrupt("file too small to contain a codec footer"))?;
        codec_util::check_whole_file_footer(tii, end)?;
        let index_interval = i64::from(input.read_vint()?);
        if index_interval < 1 {
            return Err(corrupt(format!("invalid indexInterval: {index_interval}")));
        }
        let packed_ints_version = input.read_vint()?;
        let blocksize = input.read_vint()?;
        let blocksize = usize::try_from(blocksize)
            .map_err(|_| corrupt(format!("invalid blocksize: {blocksize}")))?;
        seek_dir(&mut input, tii.len())?;

        let num_fields = input.read_vint()?;
        if num_fields < 0 {
            return Err(corrupt(format!("invalid numFields: {num_fields}")));
        }
        let mut term_bytes = Vec::new();
        let mut fields = BTreeMap::new();
        for _ in 0..num_fields {
            let field = input.read_vint()?;
            let num_index_terms = i64::from(input.read_vint()?);
            if num_index_terms <= 0 {
                return Err(corrupt(format!("invalid numIndexTerms: {num_index_terms}")));
            }
            let terms_start = input.read_vlong()?;
            let index_start = input.read_vlong()?;
            let packed_index_start = input.read_vlong()?;
            let _packed_offsets_start = input.read_vlong()?;
            if packed_index_start < index_start {
                return Err(corrupt(format!(
                    "invalid packedIndexStart: {packed_index_start} indexStart: {index_start} \
                     numIndexTerms: {num_index_terms}"
                )));
            }
            let info = state
                .field_infos
                .field_by_number(field)
                .ok_or_else(|| corrupt(format!("invalid field number: {field}")))?;
            // `new FieldIndexData(in, termBytes, indexStart, ...)`: slurp in
            // the images from disk.
            let start = super::file_pointer(index_start, end, "indexStart")?;
            let packed = super::file_pointer(packed_index_start, end, "packedIndexStart")?;
            let mut clone = SliceInput::new(&tii[..end]);
            clone.seek(start)?;
            let term_bytes_start = term_bytes.len();
            term_bytes.extend_from_slice(&tii[start..packed]);
            clone.seek(packed)?;
            // Every index term but the first has a non-empty prefix
            // (`indexedTermPrefixLength`), so a count the prefix bytes cannot
            // hold is refused before the arrays below are sized from it.
            // ARITH: both are non-negative `usize`s/`i64`s, `num_index_terms`
            // an `int`, `start <= packed`.
            #[allow(clippy::arithmetic_side_effects)]
            let too_many = num_index_terms - 1 > (packed - start) as i64;
            if too_many {
                return Err(corrupt(format!(
                    "invalid numIndexTerms: {num_index_terms} for {} prefix bytes",
                    packed.saturating_sub(start)
                )));
            }
            // `num_index_terms` is a positive `int`.
            let count = num_index_terms as u64;
            let terms_dict_offsets =
                MonotonicBlockPackedReader::of(&mut clone, packed_ints_version, blocksize, count)?;
            let term_offsets = MonotonicBlockPackedReader::of(
                &mut clone,
                packed_ints_version,
                blocksize,
                count.saturating_add(1),
            )?;
            // Not in Java (stage-3 optimisation, `cx_bterms_seek`): both
            // packed arrays are decoded here, each prefix's range checked once,
            // so a seek's binary search reads no packed bits.
            let index_terms = (0..count)
                .map(|i| {
                    // ARITH: `i < count`, and `term_offsets` holds `count + 1`.
                    #[allow(clippy::arithmetic_side_effects)]
                    let (offset, end) = (term_offsets.get(i), term_offsets.get(i + 1));
                    usize::try_from(offset)
                        .ok()
                        .zip(usize::try_from(end).ok())
                        .and_then(|(s, e)| {
                            let s = term_bytes_start.checked_add(s)?;
                            let e = term_bytes_start.checked_add(e)?;
                            (s <= e && e <= term_bytes.len()).then_some((s, e))
                        })
                        .ok_or_else(|| {
                            corrupt(format!("invalid index term offsets {offset}..{end}"))
                        })
                })
                .collect::<Result<Vec<_>>>()?;
            let block_pointers = (0..count)
                .map(|i| {
                    terms_start
                        .checked_add(terms_dict_offsets.get(i))
                        .ok_or_else(|| corrupt("invalid terms dict offset"))
                })
                .collect::<Result<Vec<_>>>()?;
            let data = FieldIndexData {
                index_terms,
                block_pointers,
                num_index_terms,
            };
            if fields.insert(info.name.clone(), data).is_some() {
                return Err(corrupt(format!("duplicate field: {}", info.name)));
            }
        }
        Ok(FixedGapTermsIndexReader {
            index_interval,
            term_bytes,
            fields,
        })
    }

    /// `supportsOrd()`.
    pub fn supports_ord(&self) -> bool {
        true
    }

    /// `indexInterval`.
    pub fn index_interval(&self) -> i64 {
        self.index_interval
    }

    /// `getFieldEnum(fieldInfo)`: `None` when the field has no index terms.
    pub fn field_enum(&self, field: &str) -> Option<FixedGapIndexEnum<'_>> {
        self.fields.get(field).map(|f| FixedGapIndexEnum {
            reader: self,
            field: f,
            term: (0, 0),
            ord: 0,
        })
    }

    /// The fields with index terms, in name order.
    pub fn field_names(&self) -> impl Iterator<Item = &str> {
        self.fields.keys().map(String::as_str)
    }
}

/// `FixedGapTermsIndexReader.IndexEnum`.
#[derive(Debug)]
pub struct FixedGapIndexEnum<'a> {
    reader: &'a FixedGapTermsIndexReader,
    field: &'a FieldIndexData,
    /// `term`, as a range of the reader's bytes.
    term: (usize, usize),
    ord: i64,
}

impl<'a> FixedGapIndexEnum<'a> {
    /// `term()`: the index term last seeked to (its distinguishing prefix).
    pub fn term(&self) -> &'a [u8] {
        let reader: &'a FixedGapTermsIndexReader = self.reader;
        &reader.term_bytes[self.term.0..self.term.1]
    }

    /// `ord()`: the ord of the index term last seeked to.
    pub fn ord(&self) -> i64 {
        self.ord
    }

    /// `termBytesReader.fillSlice(term, termBytesStart + offset, length)`
    /// for index term `idx` (`< numIndexTerms`).
    fn fill(&mut self, idx: i64) -> Result<()> {
        self.term = *usize::try_from(idx)
            .ok()
            .and_then(|i| self.field.index_terms.get(i))
            .ok_or_else(|| corrupt(format!("invalid index term: {idx}")))?;
        Ok(())
    }

    /// `termsStart + termsDictOffsets.get(idx)`.
    fn block_pointer(&self, idx: i64) -> Result<i64> {
        usize::try_from(idx)
            .ok()
            .and_then(|i| self.field.block_pointers.get(i).copied())
            .ok_or_else(|| corrupt(format!("invalid index term: {idx}")))
    }

    /// `seek(target)`: the last index term `<= target` (the first when every
    /// one is greater), and its block's file pointer.
    ///
    /// # Errors
    /// `Corrupted` for offsets outside the index's bytes.
    // ARITH: `lo`, `hi` and `mid` stay in `-1..numIndexTerms` (an `int`), so
    // `lo + hi` and `hi * indexInterval` fit an `i64`.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn seek(&mut self, target: &[u8]) -> Result<i64> {
        let interval = self.reader.index_interval;
        let mut lo: i64 = 0; // binary search
        let mut hi: i64 = self.field.num_index_terms - 1;
        while hi >= lo {
            let mid = ((lo + hi) as u64 >> 1) as i64;
            self.fill(mid)?;
            match target.cmp(self.term()) {
                std::cmp::Ordering::Less => hi = mid - 1,
                std::cmp::Ordering::Greater => lo = mid + 1,
                std::cmp::Ordering::Equal => {
                    self.ord = mid * interval;
                    return self.block_pointer(mid);
                }
            }
        }
        if hi < 0 {
            hi = 0;
        }
        self.fill(hi)?;
        self.ord = hi * interval;
        self.block_pointer(hi)
    }

    /// `next()`: the next index term's block, or `None` past the last
    /// (Java: `-1`).
    ///
    /// # Errors
    /// `Corrupted` for offsets outside the index's bytes.
    // A cursor that also seeks, as Java's `FieldIndexEnum`, not an `Iterator`.
    #[allow(clippy::should_implement_trait)]
    // ARITH: `ord` is an index term's ord, `idx * interval` with `idx` an
    // `int`.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn next(&mut self) -> Result<Option<i64>> {
        let interval = self.reader.index_interval;
        let idx = 1 + self.ord / interval;
        if idx >= self.field.num_index_terms {
            return Ok(None);
        }
        self.ord += interval;
        self.fill(idx)?;
        self.block_pointer(idx).map(Some)
    }

    /// `seek(ord)`: the index term at or before `ord`, and its block.
    ///
    /// # Errors
    /// [`Error::IllegalArgument`] for an ord past the index (Java: an
    /// `assert`); `Corrupted` for offsets outside the index's bytes.
    // ARITH: `ord >= 0`, so `idx` is in `0..numIndexTerms` once checked.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn seek_ord(&mut self, ord: i64) -> Result<i64> {
        let interval = self.reader.index_interval;
        if ord < 0 || ord / interval >= self.field.num_index_terms {
            return Err(Error::IllegalArgument(format!("ord out of bounds: {ord}")));
        }
        let idx = ord / interval;
        self.fill(idx)?;
        self.ord = idx * interval;
        self.block_pointer(idx)
    }
}
