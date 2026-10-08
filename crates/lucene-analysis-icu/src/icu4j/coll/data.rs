//! `com.ibm.icu.impl.coll.CollationData` and `CollationDataReader`: the
//! mappings of one collation (root `coll/ucadata.icu`, or a tailoring's
//! `%%CollationBin` over it), and the reader of their binary form (data
//! format `UCol`, format version 5).
//!
//! Layout after the ICU header: an `int32` index count, the indexes (the
//! options word, the Jamo CE32 start, then the byte offsets of each part),
//! and the parts in order -- reorder codes (and ranges), reorder table, the
//! CE32 trie (`Trie2_32`), CEs (`int64`), CE32s, root elements, contexts
//! (prefix and contraction tries, UTF-16), the unsafe-backward set, the
//! fast-Latin table, script data, compressible lead bytes. Read and kept
//! as Java keeps them, but for the parts only comparison uses: the
//! unsafe-backward set (backward iteration), the fast-Latin table (`compare`
//! on Latin text) and the root elements (the rule builder) are checked and
//! skipped, since sort keys -- what Lucene asks for -- never read them.

use std::sync::Arc;

use crate::icu4j::binary::{read_header, ByteReader};
use crate::icu4j::coll::collation::{self as c};
use crate::icu4j::coll::settings::CollationSettings;
use crate::icu4j::normalizer2_impl::Normalizer2Impl;
use crate::icu4j::trie2::Trie2_32;
use crate::{IcuError, IcuErrorKind};

/// `Collator.ReorderCodes.FIRST` (`SPACE`).
pub const REORDER_CODE_FIRST: i32 = 0x1000;
/// `Collator.ReorderCodes.DEFAULT`.
pub const REORDER_CODE_DEFAULT: i32 = -1;
/// `Collator.ReorderCodes.NONE` (`UScript.UNKNOWN`).
pub const REORDER_CODE_NONE: i32 = 103;
/// `Collator.ReorderCodes.CURRENCY`.
pub const REORDER_CODE_CURRENCY: i32 = 0x1003;
/// `Collator.ReorderCodes.LIMIT`.
pub const REORDER_CODE_LIMIT: i32 = 0x1005;
const SCRIPT_LATIN: i32 = 25;
const SCRIPT_UNKNOWN: i32 = 103;
const REORDER_RESERVED_BEFORE_LATIN: i32 = REORDER_CODE_FIRST + 14;
const REORDER_RESERVED_AFTER_LATIN: i32 = REORDER_CODE_FIRST + 15;
const MAX_NUM_SPECIAL_REORDER_CODES: i32 = 8;
/// `JAMO_CE32S_LENGTH`.
pub const JAMO_CE32S_LENGTH: usize = 19 + 21 + 27;

const IX_OPTIONS: usize = 1;
const IX_JAMO_CE32S_START: usize = 4;
const IX_REORDER_CODES_OFFSET: usize = 5;
const IX_REORDER_TABLE_OFFSET: usize = 6;
const IX_TRIE_OFFSET: usize = 7;
const IX_RESERVED8_OFFSET: usize = 8;
const IX_CES_OFFSET: usize = 9;
const IX_RESERVED10_OFFSET: usize = 10;
const IX_CE32S_OFFSET: usize = 11;
const IX_ROOT_ELEMENTS_OFFSET: usize = 12;
const IX_CONTEXTS_OFFSET: usize = 13;
const IX_UNSAFE_BWD_OFFSET: usize = 14;
const IX_FAST_LATIN_TABLE_OFFSET: usize = 15;
const IX_SCRIPTS_OFFSET: usize = 16;
const IX_COMPRESSIBLE_BYTES_OFFSET: usize = 17;
const IX_RESERVED18_OFFSET: usize = 18;
const IX_TOTAL_SIZE: usize = 19;
/// `CollationRootElements.IX_COMMON_SEC_AND_TER_CE` / `IX_SEC_TER_BOUNDARIES`.
const ROOT_IX_COMMON_SEC_AND_TER_CE: usize = 3;
const ROOT_IX_SEC_TER_BOUNDARIES: usize = 4;
/// `CollationKeys.SEC_COMMON_HIGH`.
const SEC_COMMON_HIGH: i64 = (c::COMMON_BYTE as i64) + 0x40;
const DATA_FORMAT: u32 = 0x5543_6f6c; // "UCol"

fn icu_exception(msg: &str) -> IcuError {
    IcuError::new(msg)
}

fn illegal(msg: impl Into<String>) -> IcuError {
    IcuError::with_kind(IcuErrorKind::IllegalArgument, msg)
}

/// `CollationData`.
#[derive(Debug)]
pub struct CollationData {
    pub trie: Trie2_32,
    pub ce32s: Vec<i32>,
    pub ces: Vec<i64>,
    pub contexts: Vec<u16>,
    pub base: Option<Arc<CollationData>>,
    pub jamo_ce32s: Vec<i32>,
    pub nfc_impl: Arc<Normalizer2Impl>,
    pub numeric_primary: i64,
    pub compressible_bytes: Vec<bool>,
    pub num_scripts: i32,
    pub scripts_index: Vec<u16>,
    pub script_starts: Vec<u16>,
}

impl CollationData {
    /// `getCE32(c)`.
    #[inline]
    pub fn get_ce32(&self, c: i32) -> i32 {
        self.trie.get(c)
    }

    /// `ce32s[i]` (unassigned past the end: corrupt data only).
    #[inline]
    pub fn ce32_at(&self, i: usize) -> i32 {
        self.ce32s.get(i).copied().unwrap_or(c::UNASSIGNED_CE32)
    }

    /// `ces[i]` (`NO_CE` past the end: corrupt data only).
    #[inline]
    pub fn ce_at(&self, i: usize) -> i64 {
        self.ces.get(i).copied().unwrap_or(c::NO_CE)
    }

    /// `jamoCE32s[i]`.
    #[inline]
    pub fn jamo_ce32(&self, i: usize) -> i32 {
        self.jamo_ce32s
            .get(i)
            .copied()
            .unwrap_or(c::UNASSIGNED_CE32)
    }

    /// The data a `FALLBACK_CE32` defers to: the base, or (root data,
    /// which has none) itself.
    #[inline]
    pub fn base_or_self(&self) -> &CollationData {
        self.base.as_deref().unwrap_or(self)
    }

    /// `isCompressibleLeadByte(b)`.
    #[inline]
    pub fn is_compressible_lead_byte(&self, b: usize) -> bool {
        self.compressible_bytes.get(b).copied().unwrap_or(false)
    }

    /// `getCE32FromContexts(index)`.
    #[inline]
    pub fn get_ce32_from_contexts(&self, index: usize) -> i32 {
        let hi = self.contexts.get(index).copied().unwrap_or(0);
        let lo = self
            .contexts
            .get(index.saturating_add(1))
            .copied()
            .unwrap_or(0);
        (i32::from(hi) << 16) | i32::from(lo)
    }

    /// `getCEFromOffsetCE32(c, ce32)`.
    pub fn get_ce_from_offset_ce32(&self, cp: i32, ce32: i32) -> i64 {
        let data_ce = self.ce_at(c::index_from_ce32(ce32));
        c::make_ce(c::get_three_byte_primary_for_offset_data(cp, data_ce))
    }

    /// `getFCD16(c)`.
    #[inline]
    pub fn get_fcd16(&self, cp: i32) -> i32 {
        self.nfc_impl.get_fcd16(cp)
    }

    fn script_start(&self, i: usize) -> i32 {
        self.script_starts.get(i).map_or(0, |&s| i32::from(s))
    }

    fn script_index_at(&self, i: i32) -> usize {
        usize::try_from(i)
            .ok()
            .and_then(|i| self.scripts_index.get(i))
            .map_or(0, |&s| usize::from(s))
    }

    /// `getScriptIndex(script)`.
    fn get_script_index(&self, script: i32) -> usize {
        if script < 0 {
            0
        } else if script < self.num_scripts {
            self.script_index_at(script)
        } else if script < REORDER_CODE_FIRST {
            0
        } else {
            let s = script.saturating_sub(REORDER_CODE_FIRST);
            if s < MAX_NUM_SPECIAL_REORDER_CODES {
                self.script_index_at(self.num_scripts.saturating_add(s))
            } else {
                0
            }
        }
    }

    /// `getLastPrimaryForGroup(script)`.
    pub fn get_last_primary_for_group(&self, script: i32) -> i64 {
        let index = self.get_script_index(script);
        if index == 0 {
            return 0;
        }
        let limit = i64::from(self.script_start(index.saturating_add(1)));
        (limit << 16).saturating_sub(1)
    }

    /// `makeReorderRanges(reorder, ranges)`.
    pub fn make_reorder_ranges(&self, reorder: &[i32]) -> Result<Vec<i32>, IcuError> {
        self.make_reorder_ranges_impl(reorder, false)
    }

    // ARITH: script starts are u16 weights (lead byte << 8 | second byte),
    // shifted and offset within i32; Java's int arithmetic, unchanged.
    #[allow(clippy::arithmetic_side_effects)]
    fn make_reorder_ranges_impl(
        &self,
        reorder: &[i32],
        latin_must_move: bool,
    ) -> Result<Vec<i32>, IcuError> {
        let mut ranges = Vec::new();
        let mut length = reorder.len();
        if length == 0 || (length == 1 && reorder[0] == SCRIPT_UNKNOWN) {
            return Ok(ranges);
        }
        let n_starts = self.script_starts.len();
        if n_starts < 2 {
            return Ok(ranges);
        }
        let mut table = vec![0i32; n_starts - 1];
        for code in [REORDER_RESERVED_BEFORE_LATIN, REORDER_RESERVED_AFTER_LATIN] {
            let index = self.script_index_at(self.num_scripts + code - REORDER_CODE_FIRST);
            if index != 0 {
                if let Some(t) = table.get_mut(index) {
                    *t = 0xff;
                }
            }
        }
        let mut low_start = self.script_start(1);
        let mut high_limit = self.script_start(n_starts - 1);
        let mut specials = 0i32;
        for &code in reorder {
            let rc = code.wrapping_sub(REORDER_CODE_FIRST);
            if (0..MAX_NUM_SPECIAL_REORDER_CODES).contains(&rc) {
                specials |= 1 << rc;
            }
        }
        for i in 0..MAX_NUM_SPECIAL_REORDER_CODES {
            let index = self.script_index_at(self.num_scripts + i);
            if index != 0 && specials & (1 << i) == 0 {
                low_start = self.add_low_script_range(&mut table, index, low_start);
            }
        }
        let mut skipped_reserved = 0;
        if specials == 0 && reorder[0] == SCRIPT_LATIN && !latin_must_move {
            let index = self.script_index_at(SCRIPT_LATIN);
            let start = self.script_start(index);
            skipped_reserved = start - low_start;
            low_start = start;
        }
        let mut has_reorder_to_end = false;
        let mut i = 0usize;
        while i < length {
            let script = reorder[i];
            i += 1;
            if script == SCRIPT_UNKNOWN {
                has_reorder_to_end = true;
                while i < length {
                    length -= 1;
                    let script = reorder[length];
                    if script == SCRIPT_UNKNOWN {
                        return Err(illegal("setReorderCodes(): duplicate UScript.UNKNOWN"));
                    }
                    if script == REORDER_CODE_DEFAULT {
                        return Err(illegal(
                            "setReorderCodes(): UScript.DEFAULT together with other scripts",
                        ));
                    }
                    let index = self.get_script_index(script);
                    if index == 0 {
                        continue;
                    }
                    if table.get(index).copied().unwrap_or(0) != 0 {
                        return Err(illegal(format!(
                            "setReorderCodes(): duplicate or equivalent script {}",
                            script_code_string(script)
                        )));
                    }
                    high_limit = self.add_high_script_range(&mut table, index, high_limit);
                }
                break;
            }
            if script == REORDER_CODE_DEFAULT {
                return Err(illegal(
                    "setReorderCodes(): UScript.DEFAULT together with other scripts",
                ));
            }
            let index = self.get_script_index(script);
            if index == 0 {
                continue;
            }
            if table.get(index).copied().unwrap_or(0) != 0 {
                return Err(illegal(format!(
                    "setReorderCodes(): duplicate or equivalent script {}",
                    script_code_string(script)
                )));
            }
            low_start = self.add_low_script_range(&mut table, index, low_start);
        }
        for i in 1..n_starts - 1 {
            if table[i] != 0 {
                continue;
            }
            let start = self.script_start(i);
            if !has_reorder_to_end && start > low_start {
                low_start = start;
            }
            low_start = self.add_low_script_range(&mut table, i, low_start);
        }
        if low_start > high_limit {
            if low_start - (skipped_reserved & 0xff00) <= high_limit {
                return self.make_reorder_ranges_impl(reorder, true);
            }
            return Err(icu_exception(
                "setReorderCodes(): reordering too many partial-primary-lead-byte scripts",
            ));
        }
        let mut offset = 0i32;
        let mut i = 1usize;
        loop {
            let mut next_offset = offset;
            while i < n_starts - 1 {
                let new_lead_byte = table[i];
                if new_lead_byte != 0xff {
                    next_offset = new_lead_byte - (self.script_start(i) >> 8);
                    if next_offset != offset {
                        break;
                    }
                }
                i += 1;
            }
            if offset != 0 || i < n_starts - 1 {
                ranges.push((self.script_start(i) << 16) | (offset & 0xffff));
            }
            if i == n_starts - 1 {
                break;
            }
            offset = next_offset;
            i += 1;
        }
        Ok(ranges)
    }

    // ARITH: as for make_reorder_ranges_impl.
    #[allow(clippy::arithmetic_side_effects)]
    fn add_low_script_range(&self, table: &mut [i32], index: usize, low_start: i32) -> i32 {
        let start = self.script_start(index);
        let mut low_start = low_start;
        if (start & 0xff) < (low_start & 0xff) {
            low_start += 0x100;
        }
        if let Some(t) = table.get_mut(index) {
            // Java: (short)(lowStart >> 8).
            *t = i32::from((low_start >> 8) as i16);
        }
        let limit = self.script_start(index + 1);
        ((low_start & 0xff00) + ((limit & 0xff00) - (start & 0xff00))) | (limit & 0xff)
    }

    // ARITH: as for make_reorder_ranges_impl.
    #[allow(clippy::arithmetic_side_effects)]
    fn add_high_script_range(&self, table: &mut [i32], index: usize, high_limit: i32) -> i32 {
        let limit = self.script_start(index + 1);
        let mut high_limit = high_limit;
        if (limit & 0xff) > (high_limit & 0xff) {
            high_limit -= 0x100;
        }
        let start = self.script_start(index);
        let high_limit =
            ((high_limit & 0xff00) - ((limit & 0xff00) - (start & 0xff00))) | (start & 0xff);
        if let Some(t) = table.get_mut(index) {
            *t = i32::from((high_limit >> 8) as i16);
        }
        high_limit
    }
}

/// `scriptCodeString(script)`.
fn script_code_string(script: i32) -> String {
    if script < REORDER_CODE_FIRST {
        script.to_string()
    } else {
        format!("0x{script:x}")
    }
}

/// `CollationTailoring`: data, settings, and where they came from.
#[derive(Debug, Clone)]
pub struct CollationTailoring {
    pub data: Arc<CollationData>,
    pub settings: CollationSettings,
    /// `actualLocale` (`getName()` form; empty for root).
    pub actual_locale: String,
    /// `version` (the data version from the header).
    pub version: i32,
}

impl CollationTailoring {
    /// `getUCAVersion()`.
    fn uca_version(version: i32) -> i32 {
        ((version >> 12) & 0xff0) | ((version >> 14) & 3)
    }
}

/// `CollationDataReader.read(base, inBytes, tailoring)`: reads the root
/// (`base == None`) or a tailoring over `base`.
// ARITH: offsets and lengths come from the data's index words and are
// checked (negative lengths fail the skip); Java's int arithmetic.
#[allow(clippy::arithmetic_side_effects)]
pub fn read(
    base: Option<&CollationTailoring>,
    bytes: &[u8],
) -> Result<CollationTailoring, IcuError> {
    let (mut r, _) = read_header(bytes, DATA_FORMAT, |v| v[0] == 5)?;
    let version = bytes
        .get(20..24)
        .map_or(0, |v| i32::from_be_bytes([v[0], v[1], v[2], v[3]]));
    if let Some(b) = base {
        if CollationTailoring::uca_version(b.version) != CollationTailoring::uca_version(version) {
            return Err(icu_exception(
                "Tailoring UCA version differs from base data UCA version",
            ));
        }
    }
    let in_length = r.remaining();
    if in_length < 8 {
        return Err(icu_exception("not enough bytes"));
    }
    let indexes_length = r.i32()?;
    if indexes_length < 2 || (in_length as i64) < i64::from(indexes_length) * 4 {
        return Err(icu_exception("not enough indexes"));
    }
    let mut ix = [-1i32; IX_TOTAL_SIZE + 1];
    ix[0] = indexes_length;
    let n = usize::try_from(indexes_length).unwrap_or(0);
    for slot in ix.iter_mut().take(n).skip(1) {
        *slot = r.i32()?;
    }
    if n > ix.len() {
        r.skip((n - ix.len()) * 4)?;
    }
    let length = if n > IX_TOTAL_SIZE {
        ix[IX_TOTAL_SIZE]
    } else if n > IX_REORDER_CODES_OFFSET {
        ix[n - 1]
    } else {
        0
    };
    if (in_length as i64) < i64::from(length) {
        return Err(icu_exception("not enough bytes"));
    }
    let part = |i: usize| -> i32 { ix[i + 1].wrapping_sub(ix[i]) };
    // ICUBinary.skipBytes: a negative length (an index past the count,
    // read as -1) skips nothing.
    let skip = |r: &mut ByteReader<'_>, len: i32| -> Result<(), IcuError> {
        match usize::try_from(len) {
            Ok(n) => r.skip(n),
            Err(_) => Ok(()),
        }
    };
    let base_data = base.map(|b| b.data.clone());

    // Reorder codes, then ranges with bits set in their upper half.
    let length = part(IX_REORDER_CODES_OFFSET);
    let mut reorder_codes: Vec<i32> = Vec::new();
    let mut reorder_codes_length = 0usize;
    if length >= 4 {
        if base_data.is_none() {
            return Err(icu_exception(
                "Collation base data must not reorder scripts",
            ));
        }
        let count = (length / 4) as usize;
        reorder_codes = r.i32s(count)?;
        skip(&mut r, length & 3)?;
        let mut ranges = 0usize;
        while ranges < count && reorder_codes[count - ranges - 1] as u32 & 0xffff_0000 != 0 {
            ranges += 1;
        }
        reorder_codes_length = count - ranges;
    } else {
        skip(&mut r, length)?;
    }

    let mut length = part(IX_REORDER_TABLE_OFFSET);
    let mut reorder_table: Option<[u8; 256]> = None;
    if length >= 256 {
        if reorder_codes_length == 0 {
            return Err(icu_exception("Reordering table without reordering codes"));
        }
        let mut t = [0u8; 256];
        t.copy_from_slice(r.take(256)?);
        reorder_table = Some(t);
        length -= 256;
    }
    skip(&mut r, length)?;

    let options = ix[IX_OPTIONS];
    if let Some(bd) = &base_data {
        if bd.numeric_primary != i64::from(options) & 0xff00_0000 {
            return Err(icu_exception(
                "Tailoring numeric primary weight differs from base data",
            ));
        }
    }

    // The mappings.
    let mut length = part(IX_TRIE_OFFSET);
    let mut trie: Option<Trie2_32> = None;
    if length >= 8 {
        let t = Trie2_32::from_serialized(&mut r)?;
        let trie_length = t.serialized_length();
        if trie_length > usize::try_from(length).unwrap_or(0) {
            return Err(icu_exception("Not enough bytes for the mappings trie"));
        }
        length -= trie_length as i32;
        trie = Some(t);
    } else if base_data.is_none() {
        return Err(icu_exception("Missing collation data mappings"));
    }
    skip(&mut r, length)?;
    skip(&mut r, part(IX_RESERVED8_OFFSET))?;

    let length = part(IX_CES_OFFSET);
    let mut ces = Vec::new();
    if length >= 8 {
        if trie.is_none() {
            return Err(icu_exception("Tailored ces without tailored trie"));
        }
        ces = r.i64s((length / 8) as usize)?;
        skip(&mut r, length & 7)?;
    } else {
        skip(&mut r, length)?;
    }
    skip(&mut r, part(IX_RESERVED10_OFFSET))?;

    let length = part(IX_CE32S_OFFSET);
    let mut ce32s: Option<Vec<i32>> = None;
    if length >= 4 {
        if trie.is_none() {
            return Err(icu_exception("Tailored ce32s without tailored trie"));
        }
        ce32s = Some(r.i32s((length / 4) as usize)?);
        skip(&mut r, length & 3)?;
    } else {
        skip(&mut r, length)?;
    }

    let jamo_start = ix[IX_JAMO_CE32S_START];
    let mut jamo_ce32s: Option<Vec<i32>> = None;
    if jamo_start >= 0 {
        let Some(c32) = &ce32s else {
            return Err(icu_exception(
                "JamoCE32sStart index into non-existent ce32s[]",
            ));
        };
        let start = jamo_start as usize;
        let slice = c32
            .get(start..start.saturating_add(JAMO_CE32S_LENGTH))
            .ok_or_else(|| icu_exception("JamoCE32sStart index into non-existent ce32s[]"))?;
        jamo_ce32s = Some(slice.to_vec());
    } else if trie.is_some() && base_data.is_none() {
        return Err(icu_exception("Missing Jamo CE32s for Hangul processing"));
    }

    let mut length = part(IX_ROOT_ELEMENTS_OFFSET);
    if length >= 4 {
        let count = (length / 4) as usize;
        if trie.is_none() {
            return Err(icu_exception("Root elements but no mappings"));
        }
        if count <= ROOT_IX_SEC_TER_BOUNDARIES {
            return Err(icu_exception("Root elements array too short"));
        }
        let elements = r.i32s(count)?;
        let common_sec_ter = i64::from(elements[ROOT_IX_COMMON_SEC_AND_TER_CE]) & 0xffff_ffff;
        if common_sec_ter != c::COMMON_SEC_AND_TER_CE {
            return Err(icu_exception(
                "Common sec/ter weights in base data differ from the hardcoded value",
            ));
        }
        let boundaries = i64::from(elements[ROOT_IX_SEC_TER_BOUNDARIES]) & 0xffff_ffff;
        if (boundaries >> 24) < SEC_COMMON_HIGH {
            return Err(icu_exception(
                "[fixed last secondary common byte] is too low",
            ));
        }
        length &= 3;
    }
    skip(&mut r, length)?;

    let length = part(IX_CONTEXTS_OFFSET);
    let mut contexts: Option<Vec<u16>> = None;
    if length >= 2 {
        if trie.is_none() {
            return Err(icu_exception("Tailored contexts without tailored trie"));
        }
        contexts = Some(r.u16s((length / 2) as usize)?);
        skip(&mut r, length & 1)?;
    } else {
        skip(&mut r, length)?;
    }

    // The unsafe-backward set: only backward iteration reads it.
    let length = part(IX_UNSAFE_BWD_OFFSET);
    if length >= 2 && trie.is_none() {
        return Err(icu_exception("Unsafe-backward-set but no mappings"));
    }
    if length < 2 && trie.is_some() && base_data.is_none() {
        return Err(icu_exception("Missing unsafe-backward-set"));
    }
    skip(&mut r, length)?;

    // The fast-Latin table: only `compare` reads it.
    skip(&mut r, part(IX_FAST_LATIN_TABLE_OFFSET))?;

    let length = part(IX_SCRIPTS_OFFSET);
    let mut scripts: Option<(i32, Vec<u16>, Vec<u16>)> = None;
    if length >= 2 {
        if trie.is_none() {
            return Err(icu_exception("Script order data but no mappings"));
        }
        let scripts_length = (length / 2) as usize;
        let units = r.u16s(scripts_length)?;
        skip(&mut r, length & 1)?;
        let num_scripts = usize::from(units[0]);
        let starts_length = scripts_length as i64 - (1 + num_scripts as i64 + 16);
        if starts_length <= 2 {
            return Err(icu_exception("Script order data too short"));
        }
        let index = units[1..1 + num_scripts + 16].to_vec();
        let starts = units[1 + num_scripts + 16..].to_vec();
        let last = starts.len() - 1;
        if !(starts[0] == 0
            && i32::from(starts[1]) == (c::MERGE_SEPARATOR_BYTE + 1) << 8
            && i32::from(starts[last]) == c::TRAIL_WEIGHT_BYTE << 8)
        {
            return Err(icu_exception("Script order data not valid"));
        }
        scripts = Some((num_scripts as i32, index, starts));
    } else {
        skip(&mut r, length)?;
    }

    let mut length = part(IX_COMPRESSIBLE_BYTES_OFFSET);
    let mut compressible: Option<Vec<bool>> = None;
    if length >= 256 {
        if trie.is_none() {
            return Err(icu_exception(
                "Data for compressible primary lead bytes but no mappings",
            ));
        }
        compressible = Some(r.take(256)?.iter().map(|&b| b != 0).collect());
        length -= 256;
    } else if trie.is_some() && base_data.is_none() {
        return Err(icu_exception(
            "Missing data for compressible primary lead bytes",
        ));
    }
    skip(&mut r, length)?;
    skip(&mut r, part(IX_RESERVED18_OFFSET))?;

    // Assemble: a tailoring without mappings uses its base's data.
    let data = match trie {
        Some(trie) => {
            let bd = base_data.clone();
            let (num_scripts, scripts_index, script_starts) = match scripts {
                Some(s) => s,
                None => match &bd {
                    Some(b) => (
                        b.num_scripts,
                        b.scripts_index.clone(),
                        b.script_starts.clone(),
                    ),
                    None => (0, Vec::new(), Vec::new()),
                },
            };
            Arc::new(CollationData {
                trie,
                ce32s: ce32s.unwrap_or_default(),
                ces,
                contexts: contexts.unwrap_or_default(),
                jamo_ce32s: match jamo_ce32s {
                    Some(j) => j,
                    None => bd
                        .as_ref()
                        .map(|b| b.jamo_ce32s.clone())
                        .unwrap_or_default(),
                },
                compressible_bytes: match compressible {
                    Some(cb) => cb,
                    None => bd
                        .as_ref()
                        .map(|b| b.compressible_bytes.clone())
                        .unwrap_or_default(),
                },
                nfc_impl: crate::icu4j::normalizer2::nfc_impl(),
                numeric_primary: i64::from(options) & 0xff00_0000,
                num_scripts,
                scripts_index,
                script_starts,
                base: bd,
            })
        }
        None => base_data
            .clone()
            .ok_or_else(|| icu_exception("Missing collation data mappings"))?,
    };

    let mut settings = base.map(|b| b.settings.clone()).unwrap_or_default();
    settings.options = options & 0xffff;
    settings.variable_top =
        data.get_last_primary_for_group(REORDER_CODE_FIRST.saturating_add(settings.max_variable()));
    if settings.variable_top == 0 {
        return Err(icu_exception(
            "The maxVariable could not be mapped to a variableTop",
        ));
    }
    if reorder_codes_length != 0 {
        let bd = base_data
            .as_deref()
            .ok_or_else(|| icu_exception("Collation base data must not reorder scripts"))?;
        settings.alias_reordering(bd, &reorder_codes, reorder_codes_length, reorder_table)?;
    }
    Ok(CollationTailoring {
        data,
        settings,
        actual_locale: String::new(),
        version,
    })
}
