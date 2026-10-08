//! `com.ibm.icu.impl.Normalizer2Impl`: Unicode normalization over ICU's
//! `.nrm` data (format version 5, data format `Nrm2`), with its
//! `ReorderingBuffer`, `Hangul` and `UTF16Plus` helpers.
//!
//! Data layout (`Normalizer2Impl.load`): after the ICU header, `int32
//! indexes[]` (`indexes[IX_NORM_TRIE_OFFSET]/4` of them, at least 19): byte
//! offsets of the normalization trie, the extra data and the small-FCD
//! bitmap, then the code point and `norm16` thresholds that partition the
//! 16-bit trie values (`minYesNo`, `minNoNo`, `limitNoNo`, `minMaybeNo`,
//! ...). Then a `Fast16` [`CodePointTrie`] of `norm16` values; the extra
//! data, UTF-16 units of decomposition mappings and composition lists
//! (`norm16 >> OFFSET_SHIFT` indexes them); and 256 bytes with one bit per
//! 32 BMP code points that may have a non-zero FCD value.
//!
//! The algorithms are ICU4J 77.1's, method for method: `decompose`,
//! `compose`, `composeQuickCheck`, `makeFCD`, the `*AndAppend` variants for
//! `normalizeSecondAndAppend`, `recompose`, `combine`, and the boundary
//! tests. Strings are UTF-16 code unit slices, as Java's `CharSequence`s
//! are. Rust-forced changes: the destination is always a `Vec<u16>` (Java's
//! `ReorderingBuffer` also accepts a non-`StringBuilder` `Appendable`, which
//! no caller here uses); every read of the extra data is bounds-checked, so
//! a corrupt custom `.nrm` file produces wrong text rather than a panic, and
//! a mapped value that is not a code point is appended as U+FFFD where Java
//! throws `IllegalArgumentException`. The canonical-iterator data
//! (`ensureCanonIterData`, `getCanonStartSet`) and `addPropertyStarts` are
//! not ported: nothing in Lucene's module reaches them.

use crate::icu4j::binary::read_header;
use crate::icu4j::code_point_trie::{CodePointTrie, TrieType, ValueWidth};
use crate::icu4j::utf16;
use crate::IcuError;

/// `Normalizer2Impl.Hangul`.
pub mod hangul {
    pub const JAMO_L_BASE: i32 = 0x1100;
    pub const JAMO_V_BASE: i32 = 0x1161;
    pub const JAMO_T_BASE: i32 = 0x11a7;
    pub const HANGUL_BASE: i32 = 0xac00;
    pub const JAMO_L_COUNT: i32 = 19;
    pub const JAMO_V_COUNT: i32 = 21;
    pub const JAMO_T_COUNT: i32 = 28;
    pub const HANGUL_COUNT: i32 = JAMO_L_COUNT * JAMO_V_COUNT * JAMO_T_COUNT;
    pub const HANGUL_LIMIT: i32 = HANGUL_BASE + HANGUL_COUNT;

    /// `isHangulLV(c)`.
    pub fn is_hangul_lv(c: i32) -> bool {
        let c = c.wrapping_sub(HANGUL_BASE);
        (0..HANGUL_COUNT).contains(&c) && c % JAMO_T_COUNT == 0
    }

    /// `decompose(c, buffer)` for a Hangul syllable: its two or three jamo.
    // ARITH: c is a Hangul syllable, so every intermediate is small.
    #[allow(clippy::arithmetic_side_effects)]
    pub(crate) fn decompose(c: i32) -> ([u16; 3], usize) {
        let c = c - HANGUL_BASE;
        let c2 = c % JAMO_T_COUNT;
        let c = c / JAMO_T_COUNT;
        let l = (JAMO_L_BASE + c / JAMO_V_COUNT) as u16;
        let v = (JAMO_V_BASE + c % JAMO_V_COUNT) as u16;
        if c2 == 0 {
            ([l, v, 0], 2)
        } else {
            ([l, v, (JAMO_T_BASE + c2) as u16], 3)
        }
    }
}

pub const MIN_YES_YES_WITH_CC: i32 = 0xfe02;
pub const JAMO_VT: i32 = 0xfe00;
pub const MIN_NORMAL_MAYBE_YES: i32 = 0xfc00;
pub const JAMO_L: i32 = 2;
pub const INERT: i32 = 1;
pub const HAS_COMP_BOUNDARY_AFTER: i32 = 1;
pub const OFFSET_SHIFT: i32 = 1;
pub const DELTA_TCCC_1: i32 = 2;
pub const DELTA_TCCC_MASK: i32 = 6;
pub const DELTA_SHIFT: i32 = 3;
pub const MAX_DELTA: i32 = 0x40;

const IX_NORM_TRIE_OFFSET: usize = 0;
const IX_EXTRA_DATA_OFFSET: usize = 1;
const IX_SMALL_FCD_OFFSET: usize = 2;
const IX_MIN_DECOMP_NO_CP: usize = 8;
const IX_MIN_COMP_NO_MAYBE_CP: usize = 9;
const IX_MIN_YES_NO: usize = 10;
const IX_MIN_NO_NO: usize = 11;
const IX_LIMIT_NO_NO: usize = 12;
const IX_MIN_MAYBE_YES: usize = 13;
const IX_MIN_YES_NO_MAPPINGS_ONLY: usize = 14;
const IX_MIN_NO_NO_COMP_BOUNDARY_BEFORE: usize = 15;
const IX_MIN_NO_NO_COMP_NO_MAYBE_CC: usize = 16;
const IX_MIN_NO_NO_EMPTY: usize = 17;
const IX_MIN_LCCC_CP: usize = 18;
const IX_MIN_MAYBE_NO: usize = 20;
const IX_MIN_MAYBE_NO_COMBINES_FWD: usize = 21;

pub const MAPPING_HAS_CCC_LCCC_WORD: i32 = 0x80;
pub const MAPPING_HAS_RAW_MAPPING: i32 = 0x40;
pub const MAPPING_LENGTH_MASK: i32 = 0x1f;
pub const COMP_1_LAST_TUPLE: i32 = 0x8000;
pub const COMP_1_TRIPLE: i32 = 1;
pub const COMP_1_TRAIL_LIMIT: i32 = 0x3400;
pub const COMP_1_TRAIL_MASK: i32 = 0x7ffe;
pub const COMP_1_TRAIL_SHIFT: i32 = 9;
pub const COMP_2_TRAIL_SHIFT: i32 = 6;
pub const COMP_2_TRAIL_MASK: i32 = 0xffc0;

const DATA_FORMAT: u32 = 0x4e72_6d32; // "Nrm2"

/// The normalization data of one `.nrm` file.
#[derive(Debug, Clone)]
pub struct Normalizer2Impl {
    min_decomp_no_cp: i32,
    min_comp_no_maybe_cp: i32,
    min_lccc_cp: i32,
    min_yes_no: i32,
    min_yes_no_mappings_only: i32,
    min_no_no: i32,
    min_no_no_comp_boundary_before: i32,
    min_no_no_comp_no_maybe_cc: i32,
    min_no_no_empty: i32,
    limit_no_no: i32,
    center_no_no_delta: i32,
    min_maybe_no: i32,
    min_maybe_no_combines_fwd: i32,
    min_maybe_yes: i32,
    norm_trie: CodePointTrie,
    extra_data: Vec<u16>,
    small_fcd: [u8; 0x100],
    /// Per `Mode` (in declaration order), the ASCII table
    /// `Normalizer2::ascii_map` derives from this data, computed once.
    pub(crate) ascii_maps: [std::sync::OnceLock<Option<std::sync::Arc<[u8; 128]>>>; 4],
}

/// `Normalizer2Impl.load(ByteBuffer)`.
impl Normalizer2Impl {
    /// Reads a `.nrm` file (Java: `load(ByteBuffer)`).
    pub fn load(bytes: &[u8]) -> Result<Normalizer2Impl, IcuError> {
        let (mut r, _) = read_header(bytes, DATA_FORMAT, |v| v[0] == 5)?;
        let first = r.i32()?;
        let indexes_length = usize::try_from(first / 4).unwrap_or(0);
        if indexes_length <= IX_MIN_LCCC_CP {
            return Err(IcuError::new("Normalizer2 data: not enough indexes"));
        }
        let mut ix = vec![0i32; indexes_length.max(IX_MIN_MAYBE_NO_COMBINES_FWD + 1)];
        ix[0] = first;
        for slot in ix.iter_mut().take(indexes_length).skip(1) {
            *slot = r.i32()?;
        }
        let min_maybe_no = ix[IX_MIN_MAYBE_NO];
        let offset = ix[IX_NORM_TRIE_OFFSET];
        let next_offset = ix[IX_EXTRA_DATA_OFFSET];
        let trie_position = r.position();
        let norm_trie =
            CodePointTrie::from_binary(Some(TrieType::Fast), Some(ValueWidth::Bits16), &mut r)?;
        let trie_length = r.position().saturating_sub(trie_position);
        let room = usize::try_from(next_offset.saturating_sub(offset))
            .map_err(|_| IcuError::new("Normalizer2 data: not enough bytes for normTrie"))?;
        if trie_length > room {
            return Err(IcuError::new(
                "Normalizer2 data: not enough bytes for normTrie",
            ));
        }
        r.skip(room.saturating_sub(trie_length))?;
        let offset = next_offset;
        let next_offset = ix[IX_SMALL_FCD_OFFSET];
        let num_chars = usize::try_from(next_offset.saturating_sub(offset) / 2)
            .map_err(|_| IcuError::new("Normalizer2 data: bad extra data length"))?;
        let extra_data = r.u16s(num_chars)?;
        let mut small_fcd = [0u8; 0x100];
        small_fcd.copy_from_slice(r.take(0x100)?);
        Ok(Normalizer2Impl {
            min_decomp_no_cp: ix[IX_MIN_DECOMP_NO_CP],
            min_comp_no_maybe_cp: ix[IX_MIN_COMP_NO_MAYBE_CP],
            min_lccc_cp: ix[IX_MIN_LCCC_CP],
            min_yes_no: ix[IX_MIN_YES_NO],
            min_yes_no_mappings_only: ix[IX_MIN_YES_NO_MAPPINGS_ONLY],
            min_no_no: ix[IX_MIN_NO_NO],
            min_no_no_comp_boundary_before: ix[IX_MIN_NO_NO_COMP_BOUNDARY_BEFORE],
            min_no_no_comp_no_maybe_cc: ix[IX_MIN_NO_NO_COMP_NO_MAYBE_CC],
            min_no_no_empty: ix[IX_MIN_NO_NO_EMPTY],
            limit_no_no: ix[IX_LIMIT_NO_NO],
            center_no_no_delta: (min_maybe_no >> DELTA_SHIFT)
                .wrapping_sub(MAX_DELTA)
                .wrapping_sub(1),
            min_maybe_no,
            min_maybe_no_combines_fwd: ix[IX_MIN_MAYBE_NO_COMBINES_FWD],
            min_maybe_yes: ix[IX_MIN_MAYBE_YES],
            norm_trie,
            extra_data,
            small_fcd,
            ascii_maps: Default::default(),
        })
    }
}

/// A position in a UTF-16 slice and the arithmetic on it: every position
/// indexes a slice, so it is below `isize::MAX` and `+ 2` cannot overflow.
#[inline]
fn char_count(c: i32) -> usize {
    if c > 0xffff {
        2
    } else {
        1
    }
}

impl Normalizer2Impl {
    /// One unit of the extra data; 0xffff past its end (corrupt data).
    #[inline]
    fn ed(&self, i: i32) -> i32 {
        usize::try_from(i)
            .ok()
            .and_then(|i| self.extra_data.get(i))
            .map_or(0xffff, |&u| i32::from(u))
    }

    /// The extra data in `[start, start + len)`, clamped to its end.
    fn ed_slice(&self, start: i32, len: i32) -> &[u16] {
        let s = usize::try_from(start).unwrap_or(usize::MAX);
        let n = usize::try_from(len).unwrap_or(0);
        let s = s.min(self.extra_data.len());
        let e = s.saturating_add(n).min(self.extra_data.len());
        &self.extra_data[s..e]
    }

    /// `getNorm16(c)`.
    #[inline]
    pub fn get_norm16(&self, c: i32) -> i32 {
        if (c & !0x3ff) == 0xd800 {
            INERT
        } else {
            self.norm_trie.get(c)
        }
    }

    /// `getRawNorm16(c)`.
    #[inline]
    pub fn get_raw_norm16(&self, c: i32) -> i32 {
        self.norm_trie.get(c)
    }

    /// `getCompQuickCheck(norm16)`: 1 yes, 2 maybe, 0 no.
    pub fn get_comp_quick_check(&self, norm16: i32) -> i32 {
        if norm16 < self.min_no_no || MIN_YES_YES_WITH_CC <= norm16 {
            1
        } else if self.min_maybe_no <= norm16 {
            2
        } else {
            0
        }
    }

    pub fn is_algorithmic_no_no(&self, norm16: i32) -> bool {
        self.limit_no_no <= norm16 && norm16 < self.min_maybe_no
    }

    pub fn is_decomp_yes(&self, norm16: i32) -> bool {
        norm16 < self.min_yes_no || self.min_maybe_yes <= norm16
    }

    /// `getCC(norm16)`.
    pub fn get_cc(&self, norm16: i32) -> i32 {
        if norm16 >= MIN_NORMAL_MAYBE_YES {
            return get_cc_from_normal_yes_or_maybe(norm16);
        }
        if norm16 < self.min_no_no || self.limit_no_no <= norm16 {
            return 0;
        }
        self.get_cc_from_no_no(norm16)
    }

    /// `getCCFromYesOrMaybeYesCP(c)`.
    pub fn get_cc_from_yes_or_maybe_yes_cp(&self, c: i32) -> i32 {
        if c < self.min_comp_no_maybe_cp {
            return 0;
        }
        get_cc_from_yes_or_maybe_yes(self.get_norm16(c))
    }

    /// `getFCD16(c)`.
    pub fn get_fcd16(&self, c: i32) -> i32 {
        if c < self.min_decomp_no_cp
            || (c <= 0xffff && !self.single_lead_might_have_non_zero_fcd16(c))
        {
            return 0;
        }
        self.get_fcd16_from_norm_data(c)
    }

    /// `singleLeadMightHaveNonZeroFCD16(lead)`.
    #[inline]
    pub fn single_lead_might_have_non_zero_fcd16(&self, lead: i32) -> bool {
        let bits = self
            .small_fcd
            .get(usize::try_from(lead >> 8).unwrap_or(0x100))
            .copied()
            .unwrap_or(0);
        if bits == 0 {
            return false;
        }
        ((bits >> ((lead >> 5) & 7)) & 1) != 0
    }

    /// `getFCD16FromNormData(c)`.
    pub fn get_fcd16_from_norm_data(&self, mut c: i32) -> i32 {
        let mut norm16 = self.get_norm16(c);
        if norm16 >= self.limit_no_no {
            if norm16 >= MIN_NORMAL_MAYBE_YES {
                norm16 = get_cc_from_normal_yes_or_maybe(norm16);
                return norm16 | (norm16 << 8);
            } else if norm16 >= self.min_maybe_yes {
                return 0;
            } else if norm16 < self.min_maybe_no {
                let delta_trail_cc = norm16 & DELTA_TCCC_MASK;
                if delta_trail_cc <= DELTA_TCCC_1 {
                    return delta_trail_cc >> OFFSET_SHIFT;
                }
                c = self.map_algorithmic(c, norm16);
                norm16 = self.get_raw_norm16(c);
            }
        }
        if norm16 <= self.min_yes_no || self.is_hangul_lvt(norm16) {
            return 0;
        }
        let mapping = self.get_data(norm16);
        let first_unit = self.ed(mapping);
        let mut fcd16 = first_unit >> 8;
        if (first_unit & MAPPING_HAS_CCC_LCCC_WORD) != 0 {
            fcd16 |= self.ed(mapping.wrapping_sub(1)) & 0xff00;
        }
        fcd16
    }

    /// `getFCD16FromMaybeOrNonZeroCC(norm16)`.
    fn get_fcd16_from_maybe_or_non_zero_cc(&self, mut norm16: i32) -> i32 {
        if norm16 >= MIN_NORMAL_MAYBE_YES {
            norm16 = get_cc_from_normal_yes_or_maybe(norm16);
            return norm16 | (norm16 << 8);
        } else if norm16 >= self.min_maybe_yes {
            return 0;
        }
        let mapping = self.get_data_for_maybe(norm16);
        self.ed(mapping) >> 8
    }

    /// `getDecomposition(c)`.
    pub fn get_decomposition(&self, mut c: i32) -> Option<Vec<u16>> {
        if c < self.min_decomp_no_cp {
            return None;
        }
        let mut norm16 = self.get_norm16(c);
        if self.is_maybe_yes_or_non_zero_cc(norm16) {
            return None;
        }
        let mut decomp = -1;
        if self.is_decomp_no_algorithmic(norm16) {
            c = self.map_algorithmic(c, norm16);
            decomp = c;
            norm16 = self.get_raw_norm16(c);
        }
        if norm16 < self.min_yes_no {
            if decomp < 0 {
                return None;
            }
            let mut v = Vec::new();
            utf16::push_code_point(&mut v, decomp);
            return Some(v);
        } else if self.is_hangul_lv(norm16) || self.is_hangul_lvt(norm16) {
            let (j, n) = hangul::decompose(c);
            return Some(j[..n].to_vec());
        }
        let mapping = self.get_data(norm16);
        let length = self.ed(mapping) & MAPPING_LENGTH_MASK;
        Some(self.ed_slice(mapping.wrapping_add(1), length).to_vec())
    }

    /// `getRawDecomposition(c)`.
    // ARITH: mapping offsets are u16 values shifted right, mLength is
    // masked to 5 bits.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn get_raw_decomposition(&self, c: i32) -> Option<Vec<u16>> {
        if c < self.min_decomp_no_cp {
            return None;
        }
        let norm16 = self.get_norm16(c);
        if self.is_decomp_yes(norm16) {
            return None;
        } else if self.is_hangul_lv(norm16) || self.is_hangul_lvt(norm16) {
            let c2 = c.wrapping_sub(hangul::HANGUL_BASE);
            let t = c2 % hangul::JAMO_T_COUNT;
            if t == 0 {
                let (j, _) = hangul::decompose(c);
                return Some(j[..2].to_vec());
            }
            return Some(vec![(c - t) as u16, (hangul::JAMO_T_BASE + t) as u16]);
        } else if self.is_decomp_no_algorithmic(norm16) {
            let mut v = Vec::new();
            utf16::push_code_point(&mut v, self.map_algorithmic(c, norm16));
            return Some(v);
        }
        let mapping = self.get_data(norm16);
        let first_unit = self.ed(mapping);
        let m_length = first_unit & MAPPING_LENGTH_MASK;
        if (first_unit & MAPPING_HAS_RAW_MAPPING) != 0 {
            let raw_mapping = mapping - ((first_unit >> 7) & 1) - 1;
            let rm0 = self.ed(raw_mapping);
            if rm0 <= MAPPING_LENGTH_MASK {
                Some(self.ed_slice(raw_mapping - rm0, rm0).to_vec())
            } else {
                let mut v = vec![rm0 as u16];
                v.extend_from_slice(self.ed_slice(mapping + 3, m_length - 2));
                Some(v)
            }
        } else {
            Some(self.ed_slice(mapping + 1, m_length).to_vec())
        }
    }

    /// Whether every code point of UTF-8 `s` passes the first loop of
    /// `composeQuickCheck` (`decompose(..., null)` with `decompose`), so
    /// that the UTF-16 quick check of `s` would answer "yes" outright -- the
    /// same per-unit tests, a supplementary character judged by its lead
    /// surrogate's value first, as the UTF-16 loop judges it. `false` means
    /// "not known": the caller runs the UTF-16 check.
    pub fn quick_yes_utf8(&self, s: &str, decompose: bool) -> bool {
        let (min, yes): (i32, fn(&Self, i32) -> bool) = if decompose {
            (self.min_decomp_no_cp, Self::is_most_decomp_yes_and_zero_cc)
        } else {
            (self.min_comp_no_maybe_cp, Self::is_comp_yes_and_zero_cc)
        };
        for ch in s.chars() {
            let c = ch as i32;
            if c < min {
                continue;
            }
            if c <= 0xffff {
                if !yes(self, self.norm_trie.bmp_get(c as u32)) {
                    return false;
                }
                continue;
            }
            let (lead, trail) = utf16::surrogates(c);
            if yes(self, self.norm_trie.bmp_get(u32::from(lead))) {
                // The lead unit passes alone; the trail unit is then judged
                // as a BMP unit of its own.
                if i32::from(trail) >= min && !yes(self, self.norm_trie.bmp_get(u32::from(trail))) {
                    return false;
                }
            } else if !yes(self, self.norm_trie.supp_get(c as u32)) {
                return false;
            }
        }
        true
    }

    // --- decomposition ---------------------------------------------------

    /// `decompose(s, src, limit, buffer)`; with `buffer == None` it is the
    /// quick check and returns the end of the normalized prefix.
    // ARITH: src < limit <= s.len(); a code point's char count is 1 or 2.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn decompose(
        &self,
        s: &[u16],
        mut src: usize,
        limit: usize,
        mut buffer: Option<&mut ReorderingBuffer<'_>>,
    ) -> usize {
        let min_no_cp = self.min_decomp_no_cp;
        let mut c = 0;
        let mut norm16 = 0;
        let mut prev_boundary = src;
        let mut prev_cc = 0;
        loop {
            let prev_src = src;
            while src != limit {
                c = i32::from(s[src]);
                if c < min_no_cp || {
                    norm16 = self.norm_trie.bmp_get(c as u32);
                    self.is_most_decomp_yes_and_zero_cc(norm16)
                } {
                    src += 1;
                } else if !utf16::is_lead(c) {
                    break;
                } else if src + 1 != limit && utf16::is_trail(i32::from(s[src + 1])) {
                    c = utf16::to_code_point(c, i32::from(s[src + 1]));
                    norm16 = self.norm_trie.supp_get(c as u32);
                    if self.is_most_decomp_yes_and_zero_cc(norm16) {
                        src += 2;
                    } else {
                        break;
                    }
                } else {
                    src += 1; // unpaired lead surrogate: inert
                }
            }
            if src != prev_src {
                if let Some(b) = buffer.as_deref_mut() {
                    b.flush_and_append_zero_cc(s, prev_src, src);
                } else {
                    prev_cc = 0;
                    prev_boundary = src;
                }
            }
            if src == limit {
                break;
            }
            src += char_count(c);
            if let Some(b) = buffer.as_deref_mut() {
                self.decompose_cp(c, norm16, b);
            } else {
                if self.is_decomp_yes(norm16) {
                    let cc = get_cc_from_yes_or_maybe_yes(norm16);
                    if prev_cc <= cc || cc == 0 {
                        prev_cc = cc;
                        if cc <= 1 {
                            prev_boundary = src;
                        }
                        continue;
                    }
                }
                return prev_boundary;
            }
        }
        src
    }

    /// `decomposeAndAppend(s, doDecompose, buffer)`.
    // ARITH: src < limit <= s.len().
    #[allow(clippy::arithmetic_side_effects)]
    pub fn decompose_and_append(
        &self,
        s: &[u16],
        do_decompose: bool,
        buffer: &mut ReorderingBuffer<'_>,
    ) {
        let limit = s.len();
        if limit == 0 {
            return;
        }
        if do_decompose {
            self.decompose(s, 0, limit, Some(buffer));
            return;
        }
        let mut c = utf16::code_point_at(s, 0);
        let mut src = 0;
        let mut cc = self.get_cc(self.get_norm16(c));
        let first_cc = cc;
        let mut prev_cc = cc;
        while cc != 0 {
            prev_cc = cc;
            src += char_count(c);
            if src >= limit {
                break;
            }
            c = utf16::code_point_at(s, src);
            cc = self.get_cc(self.get_norm16(c));
        }
        buffer.append_range(s, 0, src, false, first_cc, prev_cc);
        buffer.append_zero_cc_range(s, src, limit);
    }

    // --- composition -----------------------------------------------------

    /// `compose(s, src, limit, onlyContiguous, doCompose, buffer)`.
    // ARITH: positions stay within s (each step checks src != limit before
    // advancing); Hangul arithmetic is on values already range-checked
    // against the jamo counts.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn compose(
        &self,
        s: &[u16],
        mut src: usize,
        limit: usize,
        only_contiguous: bool,
        do_compose: bool,
        buffer: &mut ReorderingBuffer<'_>,
    ) -> bool {
        let mut prev_boundary = src;
        let min_no_maybe_cp = self.min_comp_no_maybe_cp;
        loop {
            let mut prev_src;
            let mut c;
            let mut norm16;
            loop {
                if src == limit {
                    if prev_boundary != limit && do_compose {
                        buffer.append_zero_cc_range(s, prev_boundary, limit);
                    }
                    return true;
                }
                c = i32::from(s[src]);
                if c < min_no_maybe_cp {
                    src += 1;
                    continue;
                }
                norm16 = self.norm_trie.bmp_get(c as u32);
                if self.is_comp_yes_and_zero_cc(norm16) {
                    src += 1;
                } else {
                    prev_src = src;
                    src += 1;
                    if !utf16::is_lead(c) {
                        break;
                    } else if src != limit && utf16::is_trail(i32::from(s[src])) {
                        c = utf16::to_code_point(c, i32::from(s[src]));
                        src += 1;
                        norm16 = self.norm_trie.supp_get(c as u32);
                        if !self.is_comp_yes_and_zero_cc(norm16) {
                            break;
                        }
                    }
                }
            }
            // c is a "noNo" or "maybe" character (or a non-zero-cc one).
            if norm16 < self.min_maybe_no {
                if !do_compose {
                    return false;
                }
                if self.is_decomp_no_algorithmic(norm16) {
                    if self.norm16_has_comp_boundary_after(norm16, only_contiguous)
                        || self.has_comp_boundary_before_at(s, src, limit)
                    {
                        if prev_boundary != prev_src {
                            buffer.append_zero_cc_range(s, prev_boundary, prev_src);
                        }
                        buffer.append_cp(self.map_algorithmic(c, norm16), 0);
                        prev_boundary = src;
                        continue;
                    }
                } else if norm16 < self.min_no_no_comp_boundary_before {
                    if self.norm16_has_comp_boundary_after(norm16, only_contiguous)
                        || self.has_comp_boundary_before_at(s, src, limit)
                    {
                        if prev_boundary != prev_src {
                            buffer.append_zero_cc_range(s, prev_boundary, prev_src);
                        }
                        let mapping = self.get_data_for_yes_or_no(norm16);
                        let length = self.ed(mapping) & MAPPING_LENGTH_MASK;
                        let m = self.ed_slice(mapping.wrapping_add(1), length);
                        buffer.append_zero_cc_range(m, 0, m.len());
                        prev_boundary = src;
                        continue;
                    }
                } else if norm16 >= self.min_no_no_empty
                    && (self.has_comp_boundary_before_at(s, src, limit)
                        || self.has_comp_boundary_after_at(
                            s,
                            prev_boundary,
                            prev_src,
                            only_contiguous,
                        ))
                {
                    if prev_boundary != prev_src {
                        buffer.append_zero_cc_range(s, prev_boundary, prev_src);
                    }
                    prev_boundary = src;
                    continue;
                }
            } else if is_jamo_vt(norm16) && prev_boundary != prev_src {
                let prev = i32::from(s[prev_src - 1]);
                if c < hangul::JAMO_T_BASE {
                    let l = prev.wrapping_sub(hangul::JAMO_L_BASE) & 0xffff;
                    if l < hangul::JAMO_L_COUNT {
                        if !do_compose {
                            return false;
                        }
                        let t;
                        if src != limit && {
                            let tt = i32::from(s[src]) - hangul::JAMO_T_BASE;
                            0 < tt && tt < hangul::JAMO_T_COUNT
                        } {
                            t = i32::from(s[src]) - hangul::JAMO_T_BASE;
                            src += 1;
                        } else if self.has_comp_boundary_before_at(s, src, limit) {
                            t = 0;
                        } else {
                            t = -1;
                        }
                        if t >= 0 {
                            let syllable = hangul::HANGUL_BASE
                                + (l * hangul::JAMO_V_COUNT + (c - hangul::JAMO_V_BASE))
                                    * hangul::JAMO_T_COUNT
                                + t;
                            prev_src -= 1; // Replace the Jamo L as well.
                            if prev_boundary != prev_src {
                                buffer.append_zero_cc_range(s, prev_boundary, prev_src);
                            }
                            buffer.append_char(syllable as u16);
                            prev_boundary = src;
                            continue;
                        }
                    }
                } else if hangul::is_hangul_lv(prev) {
                    if !do_compose {
                        return false;
                    }
                    let syllable = prev + c - hangul::JAMO_T_BASE;
                    prev_src -= 1; // Replace the Hangul LV as well.
                    if prev_boundary != prev_src {
                        buffer.append_zero_cc_range(s, prev_boundary, prev_src);
                    }
                    buffer.append_char(syllable as u16);
                    prev_boundary = src;
                    continue;
                }
            } else if norm16 > JAMO_VT {
                // norm16 >= MIN_YES_YES_WITH_CC
                let mut cc = get_cc_from_normal_yes_or_maybe(norm16);
                if only_contiguous && self.get_previous_trail_cc(s, prev_boundary, prev_src) > cc {
                    if !do_compose {
                        return false;
                    }
                } else {
                    let mut n16;
                    loop {
                        if src == limit {
                            if do_compose {
                                buffer.append_zero_cc_range(s, prev_boundary, limit);
                            }
                            return true;
                        }
                        let prev_cc = cc;
                        c = utf16::code_point_at(s, src);
                        n16 = self.norm_trie.get(c);
                        if n16 >= MIN_YES_YES_WITH_CC {
                            cc = get_cc_from_normal_yes_or_maybe(n16);
                            if prev_cc > cc {
                                if !do_compose {
                                    return false;
                                }
                                break;
                            }
                        } else {
                            break;
                        }
                        src += char_count(c);
                    }
                    if self.norm16_has_comp_boundary_before(n16) {
                        if self.is_comp_yes_and_zero_cc(n16) {
                            src += char_count(c);
                        }
                        continue;
                    }
                }
            }
            // Slow path: decompose from the last boundary and recompose.
            if prev_boundary != prev_src && !self.norm16_has_comp_boundary_before(norm16) {
                let c2 = utf16::code_point_before(s, prev_src);
                let n2 = self.norm_trie.get(c2);
                if !self.norm16_has_comp_boundary_after(n2, only_contiguous) {
                    prev_src -= char_count(c2);
                }
            }
            if do_compose && prev_boundary != prev_src {
                buffer.append_zero_cc_range(s, prev_boundary, prev_src);
            }
            let recompose_start_index = buffer.len();
            self.decompose_short(s, prev_src, src, false, only_contiguous, buffer);
            src = self.decompose_short(s, src, limit, true, only_contiguous, buffer);
            self.recompose(buffer, recompose_start_index, only_contiguous);
            if !do_compose {
                if !buffer.equals(s, prev_src, src) {
                    return false;
                }
                buffer.remove();
            }
            prev_boundary = src;
        }
    }

    /// `composeQuickCheck(s, src, limit, onlyContiguous, doSpan)`: the end
    /// of the "yes" prefix shifted left by one, ORed with 1 for "maybe".
    // ARITH: positions stay within s.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn compose_quick_check(
        &self,
        s: &[u16],
        mut src: usize,
        limit: usize,
        only_contiguous: bool,
        do_span: bool,
    ) -> usize {
        let mut qc_result = 0usize;
        let mut prev_boundary = src;
        let min_no_maybe_cp = self.min_comp_no_maybe_cp;
        loop {
            let prev_src;
            let mut c;
            let mut norm16;
            loop {
                if src == limit {
                    return (src << 1) | qc_result;
                }
                c = i32::from(s[src]);
                if c < min_no_maybe_cp {
                    src += 1;
                    continue;
                }
                norm16 = self.norm_trie.bmp_get(c as u32);
                if self.is_comp_yes_and_zero_cc(norm16) {
                    src += 1;
                } else {
                    let p = src;
                    src += 1;
                    if !utf16::is_lead(c) {
                        prev_src = p;
                        break;
                    } else if src != limit && utf16::is_trail(i32::from(s[src])) {
                        c = utf16::to_code_point(c, i32::from(s[src]));
                        src += 1;
                        norm16 = self.norm_trie.supp_get(c as u32);
                        if !self.is_comp_yes_and_zero_cc(norm16) {
                            prev_src = p;
                            break;
                        }
                    }
                }
            }
            let mut prev_norm16 = INERT;
            if prev_boundary != prev_src {
                prev_boundary = prev_src;
                if !self.norm16_has_comp_boundary_before(norm16) {
                    let c2 = utf16::code_point_before(s, prev_src);
                    let n16 = self.get_norm16(c2);
                    if !self.norm16_has_comp_boundary_after(n16, only_contiguous) {
                        prev_boundary -= char_count(c2);
                        prev_norm16 = n16;
                    }
                }
            }
            if norm16 >= self.min_maybe_no {
                let mut fcd16 = self.get_fcd16_from_maybe_or_non_zero_cc(norm16);
                let mut cc = (fcd16 >> 8) & 0xff;
                if !(only_contiguous
                    && cc != 0
                    && self.get_trail_cc_from_comp_yes_and_zero_cc(prev_norm16) > cc)
                {
                    loop {
                        if norm16 < MIN_YES_YES_WITH_CC {
                            if !do_span {
                                qc_result = 1;
                            } else {
                                return prev_boundary << 1;
                            }
                        }
                        if src == limit {
                            return (src << 1) | qc_result;
                        }
                        let prev_cc = fcd16 & 0xff;
                        c = utf16::code_point_at(s, src);
                        norm16 = self.get_norm16(c);
                        if norm16 >= self.min_maybe_no {
                            fcd16 = self.get_fcd16_from_maybe_or_non_zero_cc(norm16);
                            cc = (fcd16 >> 8) & 0xff;
                            if !(prev_cc <= cc || cc == 0) {
                                break;
                            }
                        } else {
                            break;
                        }
                        src += char_count(c);
                    }
                    if self.is_comp_yes_and_zero_cc(norm16) {
                        prev_boundary = src;
                        src += char_count(c);
                        continue;
                    }
                }
            }
            return prev_boundary << 1; // "no"
        }
    }

    /// `composeAndAppend(s, doCompose, onlyContiguous, buffer)`.
    // ARITH: lengths of slices.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn compose_and_append(
        &self,
        s: &[u16],
        do_compose: bool,
        only_contiguous: bool,
        buffer: &mut ReorderingBuffer<'_>,
    ) {
        let mut src = 0;
        let limit = s.len();
        if !buffer.is_empty() {
            let first_starter_in_src = self.find_next_comp_boundary(s, 0, limit, only_contiguous);
            if first_starter_in_src != 0 {
                let last_starter_in_dest =
                    self.find_previous_comp_boundary(buffer.str(), buffer.len(), only_contiguous);
                let mut middle: Vec<u16> = buffer.str()[last_starter_in_dest..].to_vec();
                buffer.remove_suffix(buffer.len() - last_starter_in_dest);
                middle.extend_from_slice(&s[..first_starter_in_src]);
                self.compose(&middle, 0, middle.len(), only_contiguous, true, buffer);
                src = first_starter_in_src;
            }
        }
        if do_compose {
            self.compose(s, src, limit, only_contiguous, true, buffer);
        } else {
            buffer.append_zero_cc_range(s, src, limit);
        }
    }

    // --- FCD -------------------------------------------------------------

    /// `makeFCD(s, src, limit, buffer)`; with `buffer == None` the quick
    /// check.
    // ARITH: positions stay within s.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn make_fcd(
        &self,
        s: &[u16],
        mut src: usize,
        limit: usize,
        mut buffer: Option<&mut ReorderingBuffer<'_>>,
    ) -> usize {
        let mut prev_boundary = src;
        let mut c = 0;
        let mut prev_fcd16 = 0;
        let mut fcd16 = 0;
        loop {
            let mut prev_src = src;
            while src != limit {
                c = i32::from(s[src]);
                if c < self.min_lccc_cp {
                    prev_fcd16 = !c;
                    src += 1;
                } else if !self.single_lead_might_have_non_zero_fcd16(c) {
                    prev_fcd16 = 0;
                    src += 1;
                } else {
                    if utf16::is_lead(c)
                        && src + 1 != limit
                        && utf16::is_trail(i32::from(s[src + 1]))
                    {
                        c = utf16::to_code_point(c, i32::from(s[src + 1]));
                    }
                    fcd16 = self.get_fcd16_from_norm_data(c);
                    if fcd16 <= 0xff {
                        prev_fcd16 = fcd16;
                        src += char_count(c);
                    } else {
                        break;
                    }
                }
            }
            if src != prev_src {
                if src == limit {
                    if let Some(b) = buffer.as_deref_mut() {
                        b.flush_and_append_zero_cc(s, prev_src, src);
                    }
                    break;
                }
                prev_boundary = src;
                if prev_fcd16 < 0 {
                    let prev = !prev_fcd16;
                    if prev < self.min_decomp_no_cp {
                        prev_fcd16 = 0;
                    } else {
                        prev_fcd16 = self.get_fcd16_from_norm_data(prev);
                        if prev_fcd16 > 1 {
                            prev_boundary -= 1;
                        }
                    }
                } else {
                    let mut p = src - 1;
                    if utf16::is_trail(i32::from(s[p]))
                        && prev_src < p
                        && utf16::is_lead(i32::from(s[p - 1]))
                    {
                        p -= 1;
                        prev_fcd16 = self.get_fcd16_from_norm_data(utf16::to_code_point(
                            i32::from(s[p]),
                            i32::from(s[p + 1]),
                        ));
                    }
                    if prev_fcd16 > 1 {
                        prev_boundary = p;
                    }
                }
                if let Some(b) = buffer.as_deref_mut() {
                    b.flush_and_append_zero_cc(s, prev_src, prev_boundary);
                    b.append_zero_cc_range(s, prev_boundary, src);
                }
                prev_src = src;
            } else if src == limit {
                break;
            }
            src += char_count(c);
            if (prev_fcd16 & 0xff) <= (fcd16 >> 8) {
                if (fcd16 & 0xff) <= 1 {
                    prev_boundary = src;
                }
                if let Some(b) = buffer.as_deref_mut() {
                    b.append_zero_cc(c);
                }
                prev_fcd16 = fcd16;
                continue;
            } else if let Some(b) = buffer.as_deref_mut() {
                b.remove_suffix(prev_src - prev_boundary);
                src = self.find_next_fcd_boundary(s, src, limit);
                self.decompose_short(s, prev_boundary, src, false, false, b);
                prev_boundary = src;
                prev_fcd16 = 0;
            } else {
                return prev_boundary;
            }
        }
        src
    }

    /// `makeFCDAndAppend(s, doMakeFCD, buffer)`.
    // ARITH: lengths of slices.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn make_fcd_and_append(
        &self,
        s: &[u16],
        do_make_fcd: bool,
        buffer: &mut ReorderingBuffer<'_>,
    ) {
        let mut src = 0;
        let limit = s.len();
        if !buffer.is_empty() {
            let first_boundary_in_src = self.find_next_fcd_boundary(s, 0, limit);
            if first_boundary_in_src != 0 {
                let last_boundary_in_dest =
                    self.find_previous_fcd_boundary(buffer.str(), buffer.len());
                let mut middle: Vec<u16> = buffer.str()[last_boundary_in_dest..].to_vec();
                buffer.remove_suffix(buffer.len() - last_boundary_in_dest);
                middle.extend_from_slice(&s[..first_boundary_in_src]);
                self.make_fcd(&middle, 0, middle.len(), Some(buffer));
                src = first_boundary_in_src;
            }
        }
        if do_make_fcd {
            self.make_fcd(s, src, limit, Some(buffer));
        } else {
            buffer.append_zero_cc_range(s, src, limit);
        }
    }

    // --- boundaries ------------------------------------------------------

    pub fn has_decomp_boundary_before(&self, c: i32) -> bool {
        c < self.min_lccc_cp
            || (c <= 0xffff && !self.single_lead_might_have_non_zero_fcd16(c))
            || self.norm16_has_decomp_boundary_before(self.get_norm16(c))
    }

    pub fn norm16_has_decomp_boundary_before(&self, norm16: i32) -> bool {
        if norm16 < self.min_no_no_comp_no_maybe_cc {
            return true;
        }
        if norm16 >= self.limit_no_no {
            return norm16 <= MIN_NORMAL_MAYBE_YES || norm16 == JAMO_VT;
        }
        let mapping = self.get_data_for_yes_or_no(norm16);
        let first_unit = self.ed(mapping);
        (first_unit & MAPPING_HAS_CCC_LCCC_WORD) == 0
            || (self.ed(mapping.wrapping_sub(1)) & 0xff00) == 0
    }

    pub fn has_decomp_boundary_after(&self, c: i32) -> bool {
        if c < self.min_decomp_no_cp {
            return true;
        }
        if c <= 0xffff && !self.single_lead_might_have_non_zero_fcd16(c) {
            return true;
        }
        self.norm16_has_decomp_boundary_after(self.get_norm16(c))
    }

    pub fn norm16_has_decomp_boundary_after(&self, norm16: i32) -> bool {
        if norm16 <= self.min_yes_no || self.is_hangul_lvt(norm16) {
            return true;
        }
        if norm16 >= self.limit_no_no {
            if self.is_maybe_yes_or_non_zero_cc(norm16) {
                return norm16 <= MIN_NORMAL_MAYBE_YES || norm16 == JAMO_VT;
            } else if norm16 < self.min_maybe_no {
                return (norm16 & DELTA_TCCC_MASK) <= DELTA_TCCC_1;
            }
        }
        let mapping = self.get_data(norm16);
        let first_unit = self.ed(mapping);
        if first_unit > 0x1ff {
            return false;
        }
        if first_unit <= 0xff {
            return true;
        }
        (first_unit & MAPPING_HAS_CCC_LCCC_WORD) == 0
            || (self.ed(mapping.wrapping_sub(1)) & 0xff00) == 0
    }

    pub fn is_decomp_inert(&self, c: i32) -> bool {
        self.is_decomp_yes_and_zero_cc(self.get_norm16(c))
    }

    pub fn has_comp_boundary_before(&self, c: i32) -> bool {
        c < self.min_comp_no_maybe_cp || self.norm16_has_comp_boundary_before(self.get_norm16(c))
    }

    pub fn has_comp_boundary_after(&self, c: i32, only_contiguous: bool) -> bool {
        self.norm16_has_comp_boundary_after(self.get_norm16(c), only_contiguous)
    }

    pub fn is_comp_inert(&self, c: i32, only_contiguous: bool) -> bool {
        let norm16 = self.get_norm16(c);
        self.is_comp_yes_and_zero_cc(norm16)
            && (norm16 & HAS_COMP_BOUNDARY_AFTER) != 0
            && (!only_contiguous
                || norm16 == INERT
                || self.ed(self.get_data_for_yes_or_no(norm16)) <= 0x1ff)
    }

    pub fn is_fcd_inert(&self, c: i32) -> bool {
        self.get_fcd16(c) <= 1
    }

    fn is_maybe(&self, norm16: i32) -> bool {
        self.min_maybe_no <= norm16 && norm16 <= JAMO_VT
    }

    fn is_maybe_yes_or_non_zero_cc(&self, norm16: i32) -> bool {
        norm16 >= self.min_maybe_yes
    }

    fn hangul_lvt(&self) -> i32 {
        self.min_yes_no_mappings_only | HAS_COMP_BOUNDARY_AFTER
    }

    fn is_hangul_lv(&self, norm16: i32) -> bool {
        norm16 == self.min_yes_no
    }

    fn is_hangul_lvt(&self, norm16: i32) -> bool {
        norm16 == self.hangul_lvt()
    }

    #[inline]
    fn is_comp_yes_and_zero_cc(&self, norm16: i32) -> bool {
        norm16 < self.min_no_no
    }

    fn is_decomp_yes_and_zero_cc(&self, norm16: i32) -> bool {
        norm16 < self.min_yes_no
            || norm16 == JAMO_VT
            || (self.min_maybe_yes <= norm16 && norm16 <= MIN_NORMAL_MAYBE_YES)
    }

    #[inline]
    fn is_most_decomp_yes_and_zero_cc(&self, norm16: i32) -> bool {
        norm16 < self.min_yes_no || norm16 == MIN_NORMAL_MAYBE_YES || norm16 == JAMO_VT
    }

    fn is_decomp_no_algorithmic(&self, norm16: i32) -> bool {
        self.limit_no_no <= norm16 && norm16 < self.min_maybe_no
    }

    fn get_cc_from_no_no(&self, norm16: i32) -> i32 {
        let mapping = self.get_data_for_yes_or_no(norm16);
        if (self.ed(mapping) & MAPPING_HAS_CCC_LCCC_WORD) != 0 {
            self.ed(mapping.wrapping_sub(1)) & 0xff
        } else {
            0
        }
    }

    fn get_trail_cc_from_comp_yes_and_zero_cc(&self, norm16: i32) -> i32 {
        if norm16 <= self.min_yes_no {
            0
        } else {
            self.ed(self.get_data_for_yes_or_no(norm16)) >> 8
        }
    }

    fn map_algorithmic(&self, c: i32, norm16: i32) -> i32 {
        c.wrapping_add(norm16 >> DELTA_SHIFT)
            .wrapping_sub(self.center_no_no_delta)
    }

    fn get_data_for_yes_or_no(&self, norm16: i32) -> i32 {
        norm16 >> OFFSET_SHIFT
    }

    fn get_data_for_maybe(&self, norm16: i32) -> i32 {
        norm16
            .wrapping_sub(self.min_maybe_no)
            .wrapping_add(self.limit_no_no)
            >> OFFSET_SHIFT
    }

    fn get_data(&self, mut norm16: i32) -> i32 {
        if norm16 >= self.min_maybe_no {
            norm16 = norm16
                .wrapping_sub(self.min_maybe_no)
                .wrapping_add(self.limit_no_no);
        }
        norm16 >> OFFSET_SHIFT
    }

    // SENTINEL: `-1` = no compositions list (Java's own result).
    fn get_compositions_list_for_decomp_yes(&self, norm16: i32) -> i32 {
        if !(JAMO_L..MIN_NORMAL_MAYBE_YES).contains(&norm16) {
            -1
        } else {
            self.get_data(norm16)
        }
    }

    fn get_compositions_list_for_composite(&self, norm16: i32) -> i32 {
        let list = self.get_data(norm16);
        let first_unit = self.ed(list);
        list.wrapping_add(1)
            .wrapping_add(first_unit & MAPPING_LENGTH_MASK)
    }

    /// `decomposeShort(s, src, limit, stopAtCompBoundary, onlyContiguous,
    /// buffer)`.
    // ARITH: src < limit <= s.len().
    #[allow(clippy::arithmetic_side_effects)]
    fn decompose_short(
        &self,
        s: &[u16],
        mut src: usize,
        limit: usize,
        stop_at_comp_boundary: bool,
        only_contiguous: bool,
        buffer: &mut ReorderingBuffer<'_>,
    ) -> usize {
        while src < limit {
            let c = utf16::code_point_at(s, src);
            if stop_at_comp_boundary && c < self.min_comp_no_maybe_cp {
                return src;
            }
            let norm16 = self.get_norm16(c);
            if stop_at_comp_boundary && self.norm16_has_comp_boundary_before(norm16) {
                return src;
            }
            src += char_count(c);
            self.decompose_cp(c, norm16, buffer);
            if stop_at_comp_boundary && self.norm16_has_comp_boundary_after(norm16, only_contiguous)
            {
                return src;
            }
        }
        src
    }

    /// `decompose(c, norm16, buffer)`.
    fn decompose_cp(&self, mut c: i32, mut norm16: i32, buffer: &mut ReorderingBuffer<'_>) {
        if norm16 >= self.limit_no_no {
            if self.is_maybe_yes_or_non_zero_cc(norm16) {
                buffer.append_cp(c, get_cc_from_yes_or_maybe_yes(norm16));
                return;
            } else if norm16 < self.min_maybe_no {
                c = self.map_algorithmic(c, norm16);
                norm16 = self.get_raw_norm16(c);
            }
        }
        if norm16 < self.min_yes_no {
            buffer.append_cp(c, 0);
        } else if self.is_hangul_lv(norm16) || self.is_hangul_lvt(norm16) {
            let (j, n) = hangul::decompose(c);
            for &u in &j[..n] {
                buffer.append_char(u);
            }
        } else {
            let mapping = self.get_data(norm16);
            let first_unit = self.ed(mapping);
            let length = first_unit & MAPPING_LENGTH_MASK;
            let trail_cc = first_unit >> 8;
            let lead_cc = if (first_unit & MAPPING_HAS_CCC_LCCC_WORD) != 0 {
                self.ed(mapping.wrapping_sub(1)) >> 8
            } else {
                0
            };
            let m = self.ed_slice(mapping.wrapping_add(1), length);
            buffer.append_range(m, 0, m.len(), true, lead_cc, trail_cc);
        }
    }

    /// `combine(list, trail)` (`combine_list` here): the composite (shifted left by one, ORed
    /// with its "combines forward" bit), or -1.
    // ARITH: list walks the extra data, which is at most 0xffff units
    // long, so `list + 3` is far from i32::MAX before ed() reads 0xffff
    // past the end and stops the walk; trail is a code point.
    // SENTINEL: `-1` = the pair does not combine (Java's own result).
    #[allow(clippy::arithmetic_side_effects)]
    fn combine_list(&self, mut list: i32, trail: i32) -> i32 {
        let mut first_unit;
        if trail < COMP_1_TRAIL_LIMIT {
            let key1 = trail << 1;
            loop {
                first_unit = self.ed(list);
                if key1 > first_unit {
                    list += 2 + (first_unit & COMP_1_TRIPLE);
                    if list > 0x1_0000 {
                        return -1;
                    }
                } else {
                    break;
                }
            }
            if key1 == (first_unit & COMP_1_TRAIL_MASK) {
                if (first_unit & COMP_1_TRIPLE) != 0 {
                    return (self.ed(list + 1) << 16) | self.ed(list + 2);
                } else {
                    return self.ed(list + 1);
                }
            }
        } else {
            let key1 = COMP_1_TRAIL_LIMIT + ((trail >> COMP_1_TRAIL_SHIFT) & !COMP_1_TRIPLE);
            let key2 = (trail << COMP_2_TRAIL_SHIFT) & 0xffff;
            loop {
                if list > 0x1_0000 {
                    break;
                }
                first_unit = self.ed(list);
                if key1 > first_unit {
                    list += 2 + (first_unit & COMP_1_TRIPLE);
                } else if key1 == (first_unit & COMP_1_TRAIL_MASK) {
                    let second_unit = self.ed(list + 1);
                    if key2 > second_unit {
                        if (first_unit & COMP_1_LAST_TUPLE) != 0 {
                            break;
                        } else {
                            list += 3;
                        }
                    } else if key2 == (second_unit & COMP_2_TRAIL_MASK) {
                        return ((second_unit & !COMP_2_TRAIL_MASK) << 16) | self.ed(list + 2);
                    } else {
                        break;
                    }
                } else {
                    break;
                }
            }
        }
        -1
    }

    /// `recompose(buffer, recomposeStartIndex, onlyContiguous)`.
    // ARITH: p, starter and pRemove index the buffer's string; Hangul
    // arithmetic is on range-checked jamo.
    #[allow(clippy::arithmetic_side_effects)]
    fn recompose(
        &self,
        buffer: &mut ReorderingBuffer<'_>,
        recompose_start_index: usize,
        only_contiguous: bool,
    ) {
        let mut p = recompose_start_index;
        if p == buffer.len() {
            return;
        }
        let mut starter = 0usize;
        let mut compositions_list = -1;
        let mut starter_is_supplementary = false;
        let mut prev_cc = 0;
        let sb = &mut *buffer.str;
        loop {
            let c = utf16::code_point_at(sb, p);
            p += char_count(c);
            let norm16 = self.get_norm16(c);
            let cc = get_cc_from_yes_or_maybe_yes(norm16);
            if self.is_maybe(norm16) && compositions_list >= 0 && (prev_cc < cc || prev_cc == 0) {
                if is_jamo_vt(norm16) {
                    if c < hangul::JAMO_T_BASE {
                        let prev = (i32::from(sb[starter]) - hangul::JAMO_L_BASE) & 0xffff;
                        if prev < hangul::JAMO_L_COUNT {
                            let p_remove = p - 1;
                            let mut syllable = hangul::HANGUL_BASE
                                + (prev * hangul::JAMO_V_COUNT + (c - hangul::JAMO_V_BASE))
                                    * hangul::JAMO_T_COUNT;
                            if p != sb.len() {
                                let t = (i32::from(sb[p]) - hangul::JAMO_T_BASE) & 0xffff;
                                if t < hangul::JAMO_T_COUNT {
                                    p += 1;
                                    syllable += t;
                                }
                            }
                            sb[starter] = syllable as u16;
                            sb.drain(p_remove..p);
                            p = p_remove;
                        }
                    }
                    if p == sb.len() {
                        break;
                    }
                    compositions_list = -1;
                    continue;
                } else {
                    let composite_and_fwd = self.combine_list(compositions_list, c);
                    if composite_and_fwd >= 0 {
                        let composite = composite_and_fwd >> 1;
                        let p_remove = p - char_count(c);
                        sb.drain(p_remove..p);
                        p = p_remove;
                        if starter_is_supplementary {
                            if composite > 0xffff {
                                let (lead, trail) = utf16::surrogates(composite);
                                sb[starter] = lead;
                                sb[starter + 1] = trail;
                            } else {
                                sb[starter] = c as u16;
                                sb.remove(starter + 1);
                                starter_is_supplementary = false;
                                p -= 1;
                            }
                        } else if composite > 0xffff {
                            starter_is_supplementary = true;
                            let (lead, trail) = utf16::surrogates(composite);
                            sb[starter] = lead;
                            sb.insert(starter + 1, trail);
                            p += 1;
                        } else {
                            sb[starter] = composite as u16;
                        }
                        if p == sb.len() {
                            break;
                        }
                        if (composite_and_fwd & 1) != 0 {
                            compositions_list = self.get_compositions_list_for_composite(
                                self.get_raw_norm16(composite),
                            );
                        } else {
                            compositions_list = -1;
                        }
                        continue;
                    }
                }
            }
            prev_cc = cc;
            if p == sb.len() {
                break;
            }
            if cc == 0 {
                compositions_list = self.get_compositions_list_for_decomp_yes(norm16);
                if compositions_list >= 0 {
                    if c <= 0xffff {
                        starter_is_supplementary = false;
                        starter = p - 1;
                    } else {
                        starter_is_supplementary = true;
                        starter = p - 2;
                    }
                }
            } else if only_contiguous {
                compositions_list = -1;
            }
        }
        buffer.flush();
    }

    /// `composePair(a, b)`.
    // SENTINEL: `-1` = no composite (Java's own result).
    // ARITH: Hangul arithmetic on range-checked values; list offsets are
    // u16-derived.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn compose_pair(&self, a: i32, mut b: i32) -> i32 {
        let norm16 = self.get_norm16(a);
        let mut list;
        if norm16 == INERT {
            return -1;
        } else if norm16 < self.min_yes_no_mappings_only {
            if norm16 == JAMO_L {
                b = b.wrapping_sub(hangul::JAMO_V_BASE);
                if (0..hangul::JAMO_V_COUNT).contains(&b) {
                    return hangul::HANGUL_BASE
                        + ((a - hangul::JAMO_L_BASE) * hangul::JAMO_V_COUNT + b)
                            * hangul::JAMO_T_COUNT;
                } else {
                    return -1;
                }
            } else if self.is_hangul_lv(norm16) {
                b = b.wrapping_sub(hangul::JAMO_T_BASE);
                if 0 < b && b < hangul::JAMO_T_COUNT {
                    return a + b;
                } else {
                    return -1;
                }
            } else {
                list = self.get_data_for_yes_or_no(norm16);
                if norm16 > self.min_yes_no {
                    list += 1 + (self.ed(list) & MAPPING_LENGTH_MASK);
                }
            }
        } else if norm16 < self.min_maybe_no_combines_fwd || MIN_NORMAL_MAYBE_YES <= norm16 {
            return -1;
        } else {
            list = self.get_data_for_maybe(norm16);
            if norm16 < self.min_maybe_yes {
                list += 1 + (self.ed(list) & MAPPING_LENGTH_MASK);
            }
        }
        if !(0..=0x10ffff).contains(&b) {
            return -1;
        }
        self.combine_list(list, b) >> 1
    }

    fn has_comp_boundary_before_cp(&self, c: i32, norm16: i32) -> bool {
        c < self.min_comp_no_maybe_cp || self.norm16_has_comp_boundary_before(norm16)
    }

    #[inline]
    fn norm16_has_comp_boundary_before(&self, norm16: i32) -> bool {
        norm16 < self.min_no_no_comp_no_maybe_cc || self.is_algorithmic_no_no(norm16)
    }

    fn has_comp_boundary_before_at(&self, s: &[u16], src: usize, limit: usize) -> bool {
        src == limit || self.has_comp_boundary_before(utf16::code_point_at(s, src))
    }

    #[inline]
    fn norm16_has_comp_boundary_after(&self, norm16: i32, only_contiguous: bool) -> bool {
        (norm16 & HAS_COMP_BOUNDARY_AFTER) != 0
            && (!only_contiguous || self.is_trail_cc01_for_comp_boundary_after(norm16))
    }

    fn has_comp_boundary_after_at(
        &self,
        s: &[u16],
        start: usize,
        p: usize,
        only_contiguous: bool,
    ) -> bool {
        start == p || self.has_comp_boundary_after(utf16::code_point_before(s, p), only_contiguous)
    }

    fn is_trail_cc01_for_comp_boundary_after(&self, norm16: i32) -> bool {
        norm16 == INERT
            || if self.is_decomp_no_algorithmic(norm16) {
                (norm16 & DELTA_TCCC_MASK) <= DELTA_TCCC_1
            } else {
                self.ed(self.get_data_for_yes_or_no(norm16)) <= 0x1ff
            }
    }

    // ARITH: p > 0 and a code point's char count is at most p.
    #[allow(clippy::arithmetic_side_effects)]
    fn find_previous_comp_boundary(&self, s: &[u16], mut p: usize, only_contiguous: bool) -> usize {
        while p > 0 {
            let c = utf16::code_point_before(s, p);
            let norm16 = self.get_norm16(c);
            if self.norm16_has_comp_boundary_after(norm16, only_contiguous) {
                break;
            }
            p -= char_count(c);
            if self.has_comp_boundary_before_cp(c, norm16) {
                break;
            }
        }
        p
    }

    // ARITH: p < limit <= s.len().
    #[allow(clippy::arithmetic_side_effects)]
    fn find_next_comp_boundary(
        &self,
        s: &[u16],
        mut p: usize,
        limit: usize,
        only_contiguous: bool,
    ) -> usize {
        while p < limit {
            let c = utf16::code_point_at(s, p);
            let norm16 = self.norm_trie.get(c);
            if self.has_comp_boundary_before_cp(c, norm16) {
                break;
            }
            p += char_count(c);
            if self.norm16_has_comp_boundary_after(norm16, only_contiguous) {
                break;
            }
        }
        p
    }

    // ARITH: p > 0 and a code point's char count is at most p.
    #[allow(clippy::arithmetic_side_effects)]
    fn find_previous_fcd_boundary(&self, s: &[u16], mut p: usize) -> usize {
        while p > 0 {
            let c = utf16::code_point_before(s, p);
            if c < self.min_decomp_no_cp {
                break;
            }
            let norm16 = self.get_norm16(c);
            if self.norm16_has_decomp_boundary_after(norm16) {
                break;
            }
            p -= char_count(c);
            if self.norm16_has_decomp_boundary_before(norm16) {
                break;
            }
        }
        p
    }

    // ARITH: p < limit <= s.len().
    #[allow(clippy::arithmetic_side_effects)]
    fn find_next_fcd_boundary(&self, s: &[u16], mut p: usize, limit: usize) -> usize {
        while p < limit {
            let c = utf16::code_point_at(s, p);
            if c < self.min_lccc_cp {
                break;
            }
            let norm16 = self.get_norm16(c);
            if self.norm16_has_decomp_boundary_before(norm16) {
                break;
            }
            p += char_count(c);
            if self.norm16_has_decomp_boundary_after(norm16) {
                break;
            }
        }
        p
    }

    fn get_previous_trail_cc(&self, s: &[u16], start: usize, p: usize) -> i32 {
        if start == p {
            return 0;
        }
        self.get_fcd16(utf16::code_point_before(s, p))
    }
}

#[inline]
fn is_jamo_vt(norm16: i32) -> bool {
    norm16 == JAMO_VT
}

/// `getCCFromNormalYesOrMaybe(norm16)`.
#[inline]
pub fn get_cc_from_normal_yes_or_maybe(norm16: i32) -> i32 {
    (norm16 >> OFFSET_SHIFT) & 0xff
}

/// `getCCFromYesOrMaybeYes(norm16)`.
#[inline]
pub fn get_cc_from_yes_or_maybe_yes(norm16: i32) -> i32 {
    if norm16 >= MIN_NORMAL_MAYBE_YES {
        get_cc_from_normal_yes_or_maybe(norm16)
    } else {
        0
    }
}

/// `Normalizer2Impl.ReorderingBuffer` over a `StringBuilder` destination.
pub struct ReorderingBuffer<'a> {
    imp: &'a Normalizer2Impl,
    str: &'a mut Vec<u16>,
    reorder_start: usize,
    last_cc: i32,
    code_point_start: usize,
    code_point_limit: usize,
}

impl<'a> ReorderingBuffer<'a> {
    /// `ReorderingBuffer(ni, dest, destCapacity)`.
    pub fn new(imp: &'a Normalizer2Impl, dest: &'a mut Vec<u16>, capacity: usize) -> Self {
        dest.reserve(capacity);
        let mut b = ReorderingBuffer {
            imp,
            str: dest,
            reorder_start: 0,
            last_cc: 0,
            code_point_start: 0,
            code_point_limit: 0,
        };
        if !b.str.is_empty() {
            b.set_iterator();
            b.last_cc = b.previous_cc();
            if b.last_cc > 1 {
                while b.previous_cc() > 1 {}
            }
            b.reorder_start = b.code_point_limit;
        }
        b
    }

    pub fn is_empty(&self) -> bool {
        self.str.is_empty()
    }

    pub fn len(&self) -> usize {
        self.str.len()
    }

    pub fn str(&self) -> &[u16] {
        self.str
    }

    /// `equals(s, start, limit)`.
    pub fn equals(&self, s: &[u16], start: usize, limit: usize) -> bool {
        s.get(start..limit) == Some(&self.str[..])
    }

    /// `append(c, cc)`.
    pub fn append_cp(&mut self, c: i32, cc: i32) {
        if self.last_cc <= cc || cc == 0 {
            utf16::push_code_point(self.str, c);
            self.last_cc = cc;
            if cc <= 1 {
                self.reorder_start = self.str.len();
            }
        } else {
            self.insert(c, cc);
        }
    }

    /// `append(s, start, limit, isNFD, leadCC, trailCC)`.
    // ARITH: start < limit <= s.len(); lengths of vectors.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn append_range(
        &mut self,
        s: &[u16],
        mut start: usize,
        limit: usize,
        is_nfd: bool,
        mut lead_cc: i32,
        trail_cc: i32,
    ) {
        if start == limit {
            return;
        }
        if self.last_cc <= lead_cc || lead_cc == 0 {
            if trail_cc <= 1 {
                self.reorder_start = self.str.len() + (limit - start);
            } else if lead_cc <= 1 {
                self.reorder_start = self.str.len() + 1;
            }
            self.str.extend_from_slice(&s[start..limit]);
            self.last_cc = trail_cc;
        } else {
            let mut c = utf16::code_point_at(s, start);
            start += char_count(c);
            self.insert(c, lead_cc);
            while start < limit {
                c = utf16::code_point_at(s, start);
                start += char_count(c);
                if start < limit {
                    lead_cc = if is_nfd {
                        get_cc_from_yes_or_maybe_yes(self.imp.get_norm16(c))
                    } else {
                        self.imp.get_cc(self.imp.get_norm16(c))
                    };
                } else {
                    lead_cc = trail_cc;
                }
                self.append_cp(c, lead_cc);
            }
        }
    }

    /// `append(char c)`.
    pub fn append_char(&mut self, c: u16) {
        self.str.push(c);
        self.last_cc = 0;
        self.reorder_start = self.str.len();
    }

    /// `appendZeroCC(c)`.
    pub fn append_zero_cc(&mut self, c: i32) {
        utf16::push_code_point(self.str, c);
        self.last_cc = 0;
        self.reorder_start = self.str.len();
    }

    /// `append(CharSequence s, int start, int limit)`.
    pub fn append_zero_cc_range(&mut self, s: &[u16], start: usize, limit: usize) {
        if start != limit {
            self.str.extend_from_slice(&s[start..limit]);
            self.last_cc = 0;
            self.reorder_start = self.str.len();
        }
    }

    /// `flush()`.
    pub fn flush(&mut self) {
        self.reorder_start = self.str.len();
        self.last_cc = 0;
    }

    /// `flushAndAppendZeroCC(s, start, limit)`.
    pub fn flush_and_append_zero_cc(&mut self, s: &[u16], start: usize, limit: usize) {
        self.str.extend_from_slice(&s[start..limit]);
        self.reorder_start = self.str.len();
        self.last_cc = 0;
    }

    /// `remove()`.
    pub fn remove(&mut self) {
        self.str.clear();
        self.last_cc = 0;
        self.reorder_start = 0;
    }

    /// `removeSuffix(suffixLength)`.
    pub fn remove_suffix(&mut self, suffix_length: usize) {
        let new_len = self.str.len().saturating_sub(suffix_length);
        self.str.truncate(new_len);
        self.last_cc = 0;
        self.reorder_start = self.str.len();
    }

    /// `insert(c, cc)`: requires `0 < cc < lastCC`.
    // ARITH: code_point_limit indexes the string.
    #[allow(clippy::arithmetic_side_effects)]
    fn insert(&mut self, c: i32, cc: i32) {
        self.set_iterator();
        self.skip_previous();
        while self.previous_cc() > cc {}
        let at = self.code_point_limit;
        if c <= 0xffff {
            self.str.insert(at, c as u16);
            if cc <= 1 {
                self.reorder_start = at + 1;
            }
        } else {
            let (lead, trail) = utf16::surrogates(c);
            self.str.insert(at, trail);
            self.str.insert(at, lead);
            if cc <= 1 {
                self.reorder_start = at + 2;
            }
        }
    }

    fn set_iterator(&mut self) {
        self.code_point_start = self.str.len();
    }

    /// `skipPrevious()`: requires `0 < codePointStart`.
    // ARITH: code_point_start > 0.
    #[allow(clippy::arithmetic_side_effects)]
    fn skip_previous(&mut self) {
        self.code_point_limit = self.code_point_start;
        let c = utf16::code_point_before(self.str, self.code_point_start);
        self.code_point_start -= char_count(c);
    }

    /// `previousCC()`.
    // ARITH: reorder_start < code_point_start, so code_point_start > 0.
    #[allow(clippy::arithmetic_side_effects)]
    fn previous_cc(&mut self) -> i32 {
        self.code_point_limit = self.code_point_start;
        if self.reorder_start >= self.code_point_start {
            return 0;
        }
        let c = utf16::code_point_before(self.str, self.code_point_start);
        self.code_point_start -= char_count(c);
        self.imp.get_cc_from_yes_or_maybe_yes_cp(c)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hangul_helpers() {
        assert!(hangul::is_hangul_lv(0xac00));
        assert!(!hangul::is_hangul_lv(0xac01));
        assert!(!hangul::is_hangul_lv(0xab00));
        assert_eq!(hangul::decompose(0xac00), ([0x1100, 0x1161, 0], 2));
        assert_eq!(hangul::decompose(0xac01), ([0x1100, 0x1161, 0x11a8], 3));
        assert_eq!(get_cc_from_yes_or_maybe_yes(5), 0);
    }

    /// Every per-code-point query of every vendored normalizer over the
    /// ranges where the data is dense, and two-code-point strings through
    /// the quick checks and appends: the paths the fixtures reach only for
    /// some forms.
    #[test]
    fn per_code_point_queries() {
        use crate::icu4j::normalizer2::{Mode, Normalizer2};
        let forms = ["nfc", "nfkc", "nfkc_cf", "uts46"];
        let ranges = [
            0..0x700,
            0x900..0xb00,
            0xf00..0xf90,
            0x1100..0x1200,
            0x1e00..0x2200,
            0x3000..0x3400,
            0xac00..0xac40,
            0xf900..0xfb50,
            0xff00..0xfff0,
            0x1d150..0x1d1c0,
            0x2f800..0x2f810,
        ];
        let mut seen = 0u64;
        for form in forms {
            let imp = Normalizer2::get_instance(form, Mode::Compose).unwrap();
            for mode in [
                Mode::Compose,
                Mode::Decompose,
                Mode::Fcd,
                Mode::ComposeContiguous,
            ] {
                let n = Normalizer2::get_instance(form, mode).unwrap();
                for r in ranges.clone() {
                    for c in r {
                        seen += u64::from(n.has_boundary_before(c))
                            + u64::from(n.has_boundary_after(c))
                            + u64::from(n.is_inert(c));
                        if mode == Mode::Compose {
                            seen += n.get_decomposition(c).map_or(0, |d| d.len() as u64);
                            seen += n.get_raw_decomposition(c).map_or(0, |d| d.len() as u64);
                            seen += n.get_combining_class(c) as u64;
                            seen += u64::from(imp.compose_pair(c, 0x301) > 0);
                            seen += u64::from(imp.compose_pair(0x41, c) > 0);
                        }
                        let mut s = Vec::new();
                        utf16::push_code_point(&mut s, c);
                        utf16::push_code_point(&mut s, 0x301);
                        s.insert(0, 0x41);
                        seen += n.span_quick_check_yes(&s) as u64;
                        seen += u64::from(n.is_normalized(&s));
                        let mut first = vec![0x41, 0x316];
                        n.append(&mut first, &s[1..]);
                        let mut second = utf16::units("\u{301}\u{316}");
                        n.normalize_second_and_append(&mut second, &s);
                        seen += (first.len() + second.len()) as u64;
                    }
                }
            }
        }
        assert!(seen > 1_000_000, "{seen}");
    }

    #[test]
    fn rejects_short_index_table() {
        let mut b = crate::icu4j::normalizer2::NFC_DATA.to_vec();
        // indexes[0] (the trie offset) / 4 = number of indexes: make it 4.
        let h = usize::from(u16::from_be_bytes([b[0], b[1]]));
        b[h..h + 4].copy_from_slice(&16i32.to_be_bytes());
        assert!(Normalizer2Impl::load(&b)
            .unwrap_err()
            .message()
            .contains("not enough indexes"));
        let mut b = crate::icu4j::normalizer2::NFC_DATA.to_vec();
        // the extra-data offset before the trie's end
        b[h + 4..h + 8].copy_from_slice(&0x60i32.to_be_bytes());
        assert!(Normalizer2Impl::load(&b).is_err());
        let mut b = crate::icu4j::normalizer2::NFC_DATA.to_vec();
        b[h + 4..h + 8].copy_from_slice(&(-8i32).to_be_bytes());
        assert!(Normalizer2Impl::load(&b).is_err());
    }
}
