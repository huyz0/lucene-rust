//! `com.ibm.icu.impl.coll.CollationIterator` with its two UTF-16 text
//! sources, `UTF16CollationIterator` (normalization off) and
//! `FCDUTF16CollationIterator` (on: segments that are not FCD are
//! decomposed to NFD on the fly), forward only -- the direction a sort key
//! is written in.
//!
//! `nextCE` turns code points into CEs through the data's CE32s: simple and
//! long-primary/secondary CE32s directly; expansions; prefix (look-behind)
//! and contraction (look-ahead, including discontiguous contractions over
//! skipped combining marks) tries; digit runs as numeric primaries; Hangul
//! syllables through their Jamo; offset and implicit primaries for
//! ranges; a tailoring's `FALLBACK_CE32` deferring to its base. Rust-forced
//! changes: one type with a source enum where Java subclasses; every data
//! index is bounds-checked (corrupt data yields `NO_CE`/unassigned, where
//! Java throws). Backward iteration (`previousCE`) is not ported: only
//! `compare` uses it.

use crate::icu4j::coll::collation::{self as c};
use crate::icu4j::coll::data::CollationData;
use crate::icu4j::coll::fcd;
use crate::icu4j::normalizer2_impl::ReorderingBuffer;
use crate::icu4j::tries::{CharsTrie, CharsTrieState, TrieResult};
use crate::icu4j::utf16;

/// `NO_CP_AND_CE32`: no code point, `FALLBACK_CE32`.
const NO_CP: i32 = -1;

/// `SkippedState`: the code points a discontiguous contraction skipped
/// over, to be returned to the input in their place.
#[derive(Debug, Default)]
struct SkippedState {
    old_buffer: Vec<u16>,
    new_buffer: Vec<u16>,
    pos: usize,
    skip_length_at_match: usize,
    state: CharsTrieState,
}

// ARITH: (the whole impl) positions within the skipped buffer, n at most
// the code points skipped (pos > length is checked before pos - length).
#[allow(clippy::arithmetic_side_effects)]
impl SkippedState {
    fn clear(&mut self) {
        self.old_buffer.clear();
        self.pos = 0;
    }

    fn is_empty(&self) -> bool {
        self.old_buffer.is_empty()
    }

    fn has_next(&self) -> bool {
        self.pos < self.old_buffer.len()
    }

    fn next(&mut self) -> i32 {
        let cp = utf16::code_point_at(&self.old_buffer, self.pos);
        self.pos = self.pos.saturating_add(char_count(cp));
        cp
    }

    fn inc_beyond(&mut self) {
        self.pos = self.pos.saturating_add(1);
    }

    /// `backwardNumCodePoints(n)`: how many of the `n` were beyond the
    /// buffer (the caller backs the input up by those).
    fn backward_num_code_points(&mut self, n: usize) -> usize {
        let length = self.old_buffer.len();
        let beyond = self.pos.saturating_sub(length);
        if self.pos > length {
            if beyond >= n {
                self.pos -= n;
                n
            } else {
                self.pos = offset_by_code_points_back(&self.old_buffer, length, n - beyond);
                beyond
            }
        } else {
            self.pos = offset_by_code_points_back(&self.old_buffer, self.pos, n);
            0
        }
    }

    fn set_first_skipped(&mut self, cp: i32) {
        self.skip_length_at_match = 0;
        self.new_buffer.clear();
        utf16::push_code_point(&mut self.new_buffer, cp);
    }

    fn skip(&mut self, cp: i32) {
        utf16::push_code_point(&mut self.new_buffer, cp);
    }

    fn record_match(&mut self) {
        self.skip_length_at_match = self.new_buffer.len();
    }

    fn replace_match(&mut self) {
        let old_length = self.old_buffer.len();
        if self.pos > old_length {
            self.pos = old_length;
        }
        let keep = self
            .new_buffer
            .get(..self.skip_length_at_match)
            .unwrap_or(&[]);
        self.old_buffer.splice(..self.pos, keep.iter().copied());
        self.pos = 0;
    }
}

/// `Character.charCount(cp)`.
#[inline]
fn char_count(cp: i32) -> usize {
    if cp >= 0x10000 {
        2
    } else {
        1
    }
}

/// `offsetByCodePoints(index, -n)`: `n` code points back from `index`.
fn offset_by_code_points_back(s: &[u16], index: usize, n: usize) -> usize {
    let mut i = index;
    for _ in 0..n {
        if i == 0 {
            break;
        }
        let cp = utf16::code_point_before(s, i);
        i = i.saturating_sub(char_count(cp));
    }
    i
}

/// The text source: Java's two subclasses.
#[derive(Debug)]
enum Source {
    /// `UTF16CollationIterator`.
    Utf16,
    /// `FCDUTF16CollationIterator`'s state beyond its superclass's.
    Fcd {
        /// Whether `seq` is the raw text (`seq == rawSeq`), not `normalized`.
        raw: bool,
        segment_start: usize,
        segment_limit: usize,
        raw_limit: usize,
        normalized: Vec<u16>,
        /// `checkDir`: 1 forward checking, 0 inside a checked segment, -1
        /// backward checking.
        check_dir: i32,
    },
}

/// `CollationIterator` over UTF-16 text.
#[derive(Debug)]
pub struct CollationIterator<'a> {
    data: &'a CollationData,
    text: &'a [u16],
    source: Source,
    start: usize,
    pos: usize,
    limit: usize,
    ce_buffer: Vec<i64>,
    ces_index: usize,
    skipped: Option<SkippedState>,
    num_cp_fwd: i32,
    is_numeric: bool,
}

// ARITH: (the whole impl) positions stay within [start, limit] of the
// current sequence (each step checks the bound first), the CE index within
// the CE buffer, and the code point counters within the text's length --
// Java's int arithmetic on the same values.
#[allow(clippy::arithmetic_side_effects)]
impl<'a> CollationIterator<'a> {
    /// `new UTF16CollationIterator(d, numeric, s, 0)` (`fcd == false`) or
    /// `new FCDUTF16CollationIterator(d, numeric, s, 0)`.
    pub fn new(data: &'a CollationData, numeric: bool, text: &'a [u16], fcd: bool) -> Self {
        let source = if fcd {
            Source::Fcd {
                raw: true,
                segment_start: 0,
                segment_limit: 0,
                raw_limit: text.len(),
                normalized: Vec::new(),
                check_dir: 1,
            }
        } else {
            Source::Utf16
        };
        CollationIterator {
            data,
            text,
            source,
            start: 0,
            pos: 0,
            limit: text.len(),
            ce_buffer: Vec::with_capacity(40),
            ces_index: 0,
            skipped: None,
            num_cp_fwd: -1,
            is_numeric: numeric,
        }
    }

    /// The unit at `i` of the current sequence (`seq.charAt(i)`).
    #[inline]
    fn seq_at(&self, i: usize) -> u16 {
        let seq: &[u16] = match &self.source {
            Source::Fcd {
                raw: false,
                normalized,
                ..
            } => normalized,
            _ => self.text,
        };
        seq.get(i).copied().unwrap_or(0)
    }

    /// `clearCEsIfNoneRemaining()`.
    pub fn clear_ces_if_none_remaining(&mut self) {
        if self.ces_index == self.ce_buffer.len() {
            self.ces_index = 0;
            self.ce_buffer.clear();
        }
    }

    /// `nextCE()`.
    pub fn next_ce(&mut self) -> i64 {
        if self.ces_index < self.ce_buffer.len() {
            let ce = self.ce_buffer[self.ces_index];
            self.ces_index += 1;
            return ce;
        }
        let (cp, mut ce32) = self.handle_next_ce32();
        let mut t = ce32 & 0xff;
        if t < c::SPECIAL_CE32_LOW_BYTE {
            return self.push_returned(c::ce_from_simple_ce32(ce32));
        }
        let mut d = self.data;
        if t == c::SPECIAL_CE32_LOW_BYTE {
            if cp < 0 {
                return self.push_returned(c::NO_CE);
            }
            d = self.data.base_or_self();
            ce32 = d.get_ce32(cp);
            t = ce32 & 0xff;
            if t < c::SPECIAL_CE32_LOW_BYTE {
                return self.push_returned(c::ce_from_simple_ce32(ce32));
            }
        }
        if t == c::LONG_PRIMARY_CE32_LOW_BYTE {
            return self
                .push_returned((i64::from(ce32.wrapping_sub(t)) << 32) | c::COMMON_SEC_AND_TER_CE);
        }
        // nextCEFromCE32
        self.append_ces_from_ce32(d, cp, ce32, true);
        let ce = self
            .ce_buffer
            .get(self.ces_index)
            .copied()
            .unwrap_or(c::NO_CE);
        self.ces_index += 1;
        ce
    }

    #[inline]
    fn push_returned(&mut self, ce: i64) -> i64 {
        self.ce_buffer.push(ce);
        self.ces_index += 1;
        ce
    }

    // ---------------------------------------------------------- the source

    /// `nextCodePoint()`.
    fn next_code_point(&mut self) -> i32 {
        let c = match self.source {
            Source::Utf16 => {
                if self.pos == self.limit {
                    return c::SENTINEL_CP;
                }
                let c = self.seq_at(self.pos);
                self.pos += 1;
                c
            }
            Source::Fcd { .. } => match self.fcd_next_unit() {
                Some(c) => c,
                None => return c::SENTINEL_CP,
            },
        };
        let c = i32::from(c);
        if utf16::is_lead(c) && self.pos != self.limit {
            let trail = i32::from(self.seq_at(self.pos));
            if utf16::is_trail(trail) {
                self.pos += 1;
                return utf16::to_code_point(c, trail);
            }
        }
        c
    }

    /// `previousCodePoint()`.
    fn previous_code_point(&mut self) -> i32 {
        let c = match self.source {
            Source::Utf16 => {
                if self.pos == self.start {
                    return c::SENTINEL_CP;
                }
                self.pos -= 1;
                self.seq_at(self.pos)
            }
            Source::Fcd { .. } => match self.fcd_previous_unit() {
                Some(c) => c,
                None => return c::SENTINEL_CP,
            },
        };
        let c = i32::from(c);
        if utf16::is_trail(c) && self.pos != self.start {
            let lead = i32::from(self.seq_at(self.pos - 1));
            if utf16::is_lead(lead) {
                self.pos -= 1;
                return utf16::to_code_point(lead, c);
            }
        }
        c
    }

    /// `handleNextCE32()`: the next code unit and its CE32 (a lead
    /// surrogate's own value: the `LEAD_SURROGATE_TAG` case reads the
    /// trail).
    fn handle_next_ce32(&mut self) -> (i32, i32) {
        let c = match self.source {
            Source::Utf16 => {
                if self.pos == self.limit {
                    return (NO_CP, c::FALLBACK_CE32);
                }
                let c = self.seq_at(self.pos);
                self.pos += 1;
                c
            }
            Source::Fcd { .. } => match self.fcd_next_unit() {
                Some(c) => c,
                None => return (NO_CP, c::FALLBACK_CE32),
            },
        };
        (i32::from(c), self.data.trie.get_from_u16_single_lead(c))
    }

    /// `handleGetTrailSurrogate()`.
    fn handle_get_trail_surrogate(&mut self) -> i32 {
        if self.pos == self.limit {
            return 0;
        }
        let trail = i32::from(self.seq_at(self.pos));
        if utf16::is_trail(trail) {
            self.pos += 1;
        }
        trail
    }

    /// `forwardNumCodePoints(num)`.
    fn forward_num_code_points(&mut self, mut num: i32) {
        match self.source {
            Source::Utf16 => {
                while num > 0 && self.pos != self.limit {
                    let c = i32::from(self.seq_at(self.pos));
                    self.pos += 1;
                    num -= 1;
                    if utf16::is_lead(c)
                        && self.pos != self.limit
                        && utf16::is_trail(i32::from(self.seq_at(self.pos)))
                    {
                        self.pos += 1;
                    }
                }
            }
            Source::Fcd { .. } => {
                while num > 0 && self.next_code_point() >= 0 {
                    num -= 1;
                }
            }
        }
    }

    /// `backwardNumCodePoints(num)`.
    fn backward_num_code_points(&mut self, mut num: i32) {
        match self.source {
            Source::Utf16 => {
                while num > 0 && self.pos != self.start {
                    self.pos -= 1;
                    let c = i32::from(self.seq_at(self.pos));
                    num -= 1;
                    if utf16::is_trail(c)
                        && self.pos != self.start
                        && utf16::is_lead(i32::from(self.seq_at(self.pos - 1)))
                    {
                        self.pos -= 1;
                    }
                }
            }
            Source::Fcd { .. } => {
                while num > 0 && self.previous_code_point() >= 0 {
                    num -= 1;
                }
            }
        }
    }

    // ------------------------------------------- FCDUTF16CollationIterator

    fn check_dir(&self) -> i32 {
        match self.source {
            Source::Fcd { check_dir, .. } => check_dir,
            Source::Utf16 => 0,
        }
    }

    /// The code-unit loop shared by `nextCodePoint` and `handleNextCE32`.
    fn fcd_next_unit(&mut self) -> Option<u16> {
        loop {
            let dir = self.check_dir();
            if dir > 0 {
                if self.pos == self.limit {
                    return None;
                }
                let mut c = self.seq_at(self.pos);
                self.pos += 1;
                if fcd::has_tccc(i32::from(c))
                    && (fcd::maybe_tibetan_composite_vowel(i32::from(c))
                        || (self.pos != self.limit
                            && fcd::has_lccc(i32::from(self.seq_at(self.pos)))))
                {
                    self.pos -= 1;
                    self.next_segment();
                    c = self.seq_at(self.pos);
                    self.pos += 1;
                }
                return Some(c);
            } else if dir == 0 && self.pos != self.limit {
                let c = self.seq_at(self.pos);
                self.pos += 1;
                return Some(c);
            } else {
                self.switch_to_forward();
            }
        }
    }

    /// The code-unit loop of `previousCodePoint`.
    fn fcd_previous_unit(&mut self) -> Option<u16> {
        loop {
            let dir = self.check_dir();
            if dir < 0 {
                if self.pos == self.start {
                    return None;
                }
                self.pos -= 1;
                let mut c = self.seq_at(self.pos);
                if fcd::has_lccc(i32::from(c))
                    && (fcd::maybe_tibetan_composite_vowel(i32::from(c))
                        || (self.pos != self.start
                            && fcd::has_tccc(i32::from(self.seq_at(self.pos - 1)))))
                {
                    self.pos += 1;
                    self.previous_segment();
                    self.pos -= 1;
                    c = self.seq_at(self.pos);
                }
                return Some(c);
            } else if dir == 0 && self.pos != self.start {
                self.pos -= 1;
                return Some(self.seq_at(self.pos));
            } else {
                self.switch_to_backward();
            }
        }
    }

    /// `switchToForward()`.
    fn switch_to_forward(&mut self) {
        let pos = self.pos;
        let Source::Fcd {
            raw,
            segment_start,
            segment_limit,
            raw_limit,
            check_dir,
            ..
        } = &mut self.source
        else {
            return;
        };
        if *check_dir < 0 {
            self.start = pos;
            *segment_start = pos;
            if pos == *segment_limit {
                self.limit = *raw_limit;
                *check_dir = 1;
            } else {
                *check_dir = 0;
            }
        } else {
            if !*raw {
                *raw = true;
                self.pos = *segment_limit;
                self.start = *segment_limit;
                *segment_start = *segment_limit;
            }
            self.limit = *raw_limit;
            *check_dir = 1;
        }
    }

    /// `switchToBackward()`.
    fn switch_to_backward(&mut self) {
        let pos = self.pos;
        let Source::Fcd {
            raw,
            segment_start,
            segment_limit,
            check_dir,
            ..
        } = &mut self.source
        else {
            return;
        };
        if *check_dir > 0 {
            self.limit = pos;
            *segment_limit = pos;
            if pos == *segment_start {
                self.start = 0;
                *check_dir = -1;
            } else {
                *check_dir = 0;
            }
        } else {
            if !*raw {
                *raw = true;
                self.pos = *segment_start;
                self.limit = *segment_start;
                *segment_limit = *segment_start;
            }
            self.start = 0;
            *check_dir = -1;
        }
    }

    /// `nextSegment()`.
    fn next_segment(&mut self) {
        let data = self.data;
        let nfc = &*data.nfc_impl;
        let raw_limit = match self.source {
            Source::Fcd { raw_limit, .. } => raw_limit,
            Source::Utf16 => return,
        };
        let text = self.text;
        let mut p = self.pos;
        let mut prev_cc = 0;
        loop {
            let mut q = p;
            let cp = utf16::code_point_at(text, p);
            p += char_count(cp);
            let fcd16 = nfc.get_fcd16(cp);
            let lead_cc = fcd16 >> 8;
            if lead_cc == 0 && q != self.pos {
                self.limit = q;
                self.set_segment_limit(q);
                break;
            }
            if lead_cc != 0
                && (prev_cc > lead_cc || fcd::is_fcd16_of_tibetan_composite_vowel(fcd16))
            {
                loop {
                    q = p;
                    if p == raw_limit {
                        break;
                    }
                    let cp = utf16::code_point_at(text, p);
                    p += char_count(cp);
                    if nfc.get_fcd16(cp) <= 0xff {
                        break;
                    }
                }
                self.normalize(self.pos, q);
                self.pos = self.start;
                break;
            }
            prev_cc = fcd16 & 0xff;
            if p == raw_limit || prev_cc == 0 {
                self.limit = p;
                self.set_segment_limit(p);
                break;
            }
        }
        if let Source::Fcd { check_dir, .. } = &mut self.source {
            *check_dir = 0;
        }
    }

    /// `previousSegment()`.
    fn previous_segment(&mut self) {
        let data = self.data;
        let nfc = &*data.nfc_impl;
        let text = self.text;
        let mut p = self.pos;
        let mut next_cc = 0;
        loop {
            let mut q = p;
            let cp = utf16::code_point_before(text, p);
            p -= char_count(cp);
            let mut fcd16 = nfc.get_fcd16(cp);
            let trail_cc = fcd16 & 0xff;
            if trail_cc == 0 && q != self.pos {
                self.start = q;
                self.set_segment_start(q);
                break;
            }
            if trail_cc != 0
                && ((next_cc != 0 && trail_cc > next_cc)
                    || fcd::is_fcd16_of_tibetan_composite_vowel(fcd16))
            {
                loop {
                    q = p;
                    if fcd16 <= 0xff || p == 0 {
                        break;
                    }
                    let cp = utf16::code_point_before(text, p);
                    p -= char_count(cp);
                    fcd16 = nfc.get_fcd16(cp);
                    if fcd16 == 0 {
                        break;
                    }
                }
                self.normalize(q, self.pos);
                self.pos = self.limit;
                break;
            }
            next_cc = fcd16 >> 8;
            if p == 0 || next_cc == 0 {
                self.start = p;
                self.set_segment_start(p);
                break;
            }
        }
        if let Source::Fcd { check_dir, .. } = &mut self.source {
            *check_dir = 0;
        }
    }

    fn set_segment_limit(&mut self, v: usize) {
        if let Source::Fcd { segment_limit, .. } = &mut self.source {
            *segment_limit = v;
        }
    }

    fn set_segment_start(&mut self, v: usize) {
        if let Source::Fcd { segment_start, .. } = &mut self.source {
            *segment_start = v;
        }
    }

    /// `normalize(from, to)`: the raw text's `[from, to)` in NFD becomes
    /// the current sequence.
    fn normalize(&mut self, from: usize, to: usize) {
        let data = self.data;
        let nfc = &*data.nfc_impl;
        let text = self.text;
        let Source::Fcd {
            raw,
            segment_start,
            segment_limit,
            normalized,
            ..
        } = &mut self.source
        else {
            return;
        };
        normalized.clear();
        {
            let mut buffer = ReorderingBuffer::new(nfc, normalized, to.saturating_sub(from));
            nfc.decompose(text, from, to, Some(&mut buffer));
        }
        *segment_start = from;
        *segment_limit = to;
        *raw = false;
        self.start = 0;
        self.limit = normalized.len();
    }

    // ------------------------------------------------- CollationIterator

    /// `appendCEsFromCE32(d, c, ce32, forward)` with `forward == true`.
    // ARITH: Hangul syllable decomposition (c in AC00..D7A3) and indexes
    // into the data, bounds-checked by the accessors.
    #[allow(clippy::arithmetic_side_effects)]
    fn append_ces_from_ce32(
        &mut self,
        mut d: &'a CollationData,
        mut cp: i32,
        mut ce32: i32,
        forward: bool,
    ) {
        while c::is_special_ce32(ce32) {
            match c::tag_from_ce32(ce32) {
                c::FALLBACK_TAG | c::RESERVED_TAG_3 | c::BUILDER_DATA_TAG => {
                    // Java: ICUException ("should be unreachable"); runtime
                    // data never holds these.
                    self.ce_buffer.push(c::NO_CE);
                    return;
                }
                c::LONG_PRIMARY_TAG => {
                    self.ce_buffer.push(c::ce_from_long_primary_ce32(ce32));
                    return;
                }
                c::LONG_SECONDARY_TAG => {
                    self.ce_buffer.push(c::ce_from_long_secondary_ce32(ce32));
                    return;
                }
                c::LATIN_EXPANSION_TAG => {
                    self.ce_buffer.push(c::latin_ce0_from_ce32(ce32));
                    self.ce_buffer.push(c::latin_ce1_from_ce32(ce32));
                    return;
                }
                c::EXPANSION32_TAG => {
                    let index = c::index_from_ce32(ce32);
                    for i in 0..c::length_from_ce32(ce32).max(1) {
                        self.ce_buffer.push(c::ce_from_ce32(d.ce32_at(index + i)));
                    }
                    return;
                }
                c::EXPANSION_TAG => {
                    let index = c::index_from_ce32(ce32);
                    for i in 0..c::length_from_ce32(ce32).max(1) {
                        self.ce_buffer.push(d.ce_at(index + i));
                    }
                    return;
                }
                c::PREFIX_TAG => {
                    if forward {
                        self.backward_num_code_points(1);
                    }
                    ce32 = self.get_ce32_from_prefix(d, ce32);
                    if forward {
                        self.forward_num_code_points(1);
                    }
                }
                c::CONTRACTION_TAG => {
                    let index = c::index_from_ce32(ce32);
                    let default_ce32 = d.get_ce32_from_contexts(index);
                    if !forward {
                        ce32 = default_ce32;
                        continue;
                    }
                    let next_cp;
                    if self.skipped.is_none() && self.num_cp_fwd < 0 {
                        next_cp = self.next_code_point();
                        if next_cp < 0 {
                            ce32 = default_ce32;
                            continue;
                        } else if ce32 & c::CONTRACT_NEXT_CCC != 0 && !fcd::may_have_lccc(next_cp) {
                            self.backward_num_code_points(1);
                            ce32 = default_ce32;
                            continue;
                        }
                    } else {
                        next_cp = self.next_skipped_code_point();
                        if next_cp < 0 {
                            ce32 = default_ce32;
                            continue;
                        } else if ce32 & c::CONTRACT_NEXT_CCC != 0 && !fcd::may_have_lccc(next_cp) {
                            self.backward_num_skipped(1);
                            ce32 = default_ce32;
                            continue;
                        }
                    }
                    ce32 =
                        self.next_ce32_from_contraction(d, ce32, index + 2, default_ce32, next_cp);
                    if ce32 == c::NO_CE32 {
                        return;
                    }
                }
                c::DIGIT_TAG => {
                    if self.is_numeric {
                        self.append_numeric_ces(ce32);
                        return;
                    }
                    ce32 = d.ce32_at(c::index_from_ce32(ce32));
                }
                c::U0000_TAG => {
                    ce32 = d.ce32_at(0);
                }
                c::HANGUL_TAG => {
                    let mut s = cp - 0xac00;
                    let t = s % 28;
                    s /= 28;
                    let v = s % 21;
                    s /= 21;
                    let (l, v, t) = (s as usize, v as usize, t as usize);
                    if ce32 & c::HANGUL_NO_SPECIAL_JAMO != 0 {
                        self.ce_buffer.push(c::ce_from_ce32(d.jamo_ce32(l)));
                        self.ce_buffer.push(c::ce_from_ce32(d.jamo_ce32(19 + v)));
                        if t != 0 {
                            self.ce_buffer.push(c::ce_from_ce32(d.jamo_ce32(39 + t)));
                        }
                        return;
                    }
                    self.append_ces_from_ce32(d, c::SENTINEL_CP, d.jamo_ce32(l), forward);
                    self.append_ces_from_ce32(d, c::SENTINEL_CP, d.jamo_ce32(19 + v), forward);
                    if t == 0 {
                        return;
                    }
                    ce32 = d.jamo_ce32(39 + t);
                    cp = c::SENTINEL_CP;
                }
                c::LEAD_SURROGATE_TAG => {
                    let trail = self.handle_get_trail_surrogate();
                    if utf16::is_trail(trail) {
                        cp = utf16::to_code_point(cp, trail);
                        ce32 &= c::LEAD_TYPE_MASK;
                        if ce32 == c::LEAD_ALL_UNASSIGNED {
                            ce32 = c::UNASSIGNED_CE32;
                        } else if ce32 == c::LEAD_ALL_FALLBACK || {
                            ce32 = d.get_ce32(cp);
                            ce32 == c::FALLBACK_CE32
                        } {
                            d = d.base_or_self();
                            ce32 = d.get_ce32(cp);
                        }
                    } else {
                        ce32 = c::UNASSIGNED_CE32;
                    }
                }
                c::OFFSET_TAG => {
                    self.ce_buffer.push(d.get_ce_from_offset_ce32(cp, ce32));
                    return;
                }
                _ => {
                    // IMPLICIT_TAG (UTF-16 iterators allow surrogate code
                    // points: forbidSurrogateCodePoints() is false).
                    self.ce_buffer.push(c::unassigned_ce_from_code_point(cp));
                    return;
                }
            }
        }
        self.ce_buffer.push(c::ce_from_simple_ce32(ce32));
    }

    /// `getCE32FromPrefix(d, ce32)`.
    fn get_ce32_from_prefix(&mut self, d: &'a CollationData, ce32: i32) -> i32 {
        let index = c::index_from_ce32(ce32);
        let mut ce32 = d.get_ce32_from_contexts(index);
        let mut prefixes = CharsTrie::new(&d.contexts, (index + 2).try_into().unwrap_or(i32::MAX));
        let mut look_behind = 0;
        loop {
            let cp = self.previous_code_point();
            if cp < 0 {
                break;
            }
            look_behind += 1;
            let m = prefixes.next_for_code_point(cp);
            if m.has_value() {
                ce32 = prefixes.get_value();
            }
            if !m.has_next() {
                break;
            }
        }
        self.forward_num_code_points(look_behind);
        ce32
    }

    /// `nextSkippedCodePoint()`.
    fn next_skipped_code_point(&mut self) -> i32 {
        if let Some(s) = &mut self.skipped {
            if s.has_next() {
                return s.next();
            }
        }
        if self.num_cp_fwd == 0 {
            return c::SENTINEL_CP;
        }
        let cp = self.next_code_point();
        if let Some(s) = &mut self.skipped {
            if !s.is_empty() && cp >= 0 {
                s.inc_beyond();
            }
        }
        if self.num_cp_fwd > 0 && cp >= 0 {
            self.num_cp_fwd -= 1;
        }
        cp
    }

    /// `backwardNumSkipped(n)`.
    fn backward_num_skipped(&mut self, n: i32) {
        let mut n = n;
        if let Some(s) = &mut self.skipped {
            if !s.is_empty() {
                n = s.backward_num_code_points(usize::try_from(n).unwrap_or(0)) as i32;
            }
        }
        self.backward_num_code_points(n);
        if self.num_cp_fwd >= 0 {
            self.num_cp_fwd += n;
        }
    }

    fn skipped_nonempty(&self) -> bool {
        self.skipped.as_ref().is_some_and(|s| !s.is_empty())
    }

    /// `nextCE32FromContraction(d, contractionCE32, trieChars, trieOffset,
    /// ce32, c)`.
    // ARITH: look-ahead counters bounded by the text length.
    #[allow(clippy::arithmetic_side_effects)]
    fn next_ce32_from_contraction(
        &mut self,
        d: &'a CollationData,
        contraction_ce32: i32,
        trie_offset: usize,
        mut ce32: i32,
        mut cp: i32,
    ) -> i32 {
        let mut look_ahead = 1;
        let mut since_match = 1;
        let mut suffixes = CharsTrie::new(&d.contexts, trie_offset.try_into().unwrap_or(i32::MAX));
        if self.skipped_nonempty() {
            if let Some(s) = &mut self.skipped {
                s.state = suffixes.save_state();
            }
        }
        let mut m = suffixes.first_for_code_point(cp);
        loop {
            if m.has_value() {
                ce32 = suffixes.get_value();
                if !m.has_next() {
                    return ce32;
                }
                cp = self.next_skipped_code_point();
                if cp < 0 {
                    return ce32;
                }
                if self.skipped_nonempty() {
                    if let Some(s) = &mut self.skipped {
                        s.state = suffixes.save_state();
                    }
                }
                since_match = 1;
            } else if m == TrieResult::NoMatch || {
                let next = self.next_skipped_code_point();
                if next >= 0 {
                    cp = next;
                }
                next < 0
            } {
                if contraction_ce32 & c::CONTRACT_TRAILING_CCC != 0
                    && (contraction_ce32 & c::CONTRACT_SINGLE_CP_NO_MATCH == 0
                        || since_match < look_ahead)
                {
                    if since_match > 1 {
                        self.backward_num_skipped(since_match);
                        cp = self.next_skipped_code_point();
                        look_ahead -= since_match - 1;
                        since_match = 1;
                    }
                    if d.get_fcd16(cp) > 0xff {
                        return self.next_ce32_from_discontiguous_contraction(
                            d, suffixes, ce32, look_ahead, cp,
                        );
                    }
                }
                break;
            } else {
                since_match += 1;
            }
            look_ahead += 1;
            m = suffixes.next_for_code_point(cp);
        }
        self.backward_num_skipped(since_match);
        ce32
    }

    /// `nextCE32FromDiscontiguousContraction(d, suffixes, ce32, lookAhead, c)`.
    // ARITH: as for next_ce32_from_contraction.
    #[allow(clippy::arithmetic_side_effects)]
    fn next_ce32_from_discontiguous_contraction(
        &mut self,
        d: &'a CollationData,
        mut suffixes: CharsTrie<'a>,
        mut ce32: i32,
        mut look_ahead: i32,
        mut cp: i32,
    ) -> i32 {
        let mut fcd16 = d.get_fcd16(cp);
        let next_cp = self.next_skipped_code_point();
        if next_cp < 0 {
            self.backward_num_skipped(1);
            return ce32;
        }
        look_ahead += 1;
        let mut prev_cc = fcd16 & 0xff;
        fcd16 = d.get_fcd16(next_cp);
        if fcd16 <= 0xff {
            self.backward_num_skipped(2);
            return ce32;
        }
        if !self.skipped_nonempty() {
            if self.skipped.is_none() {
                self.skipped = Some(SkippedState::default());
            }
            suffixes.reset();
            if look_ahead > 2 {
                self.backward_num_code_points(look_ahead);
                let first = self.next_code_point();
                suffixes.first_for_code_point(first);
                for _ in 3..look_ahead {
                    let n = self.next_code_point();
                    suffixes.next_for_code_point(n);
                }
                self.forward_num_code_points(2);
            }
            if let Some(s) = &mut self.skipped {
                s.state = suffixes.save_state();
            }
        } else if let Some(s) = &self.skipped {
            suffixes.reset_to_state(s.state);
        }
        if let Some(s) = &mut self.skipped {
            s.set_first_skipped(cp);
        }
        let mut since_match = 2;
        cp = next_cp;
        loop {
            let mut matched = false;
            if prev_cc < (fcd16 >> 8) {
                let m = suffixes.next_for_code_point(cp);
                if m.has_value() {
                    matched = true;
                    ce32 = suffixes.get_value();
                    since_match = 0;
                    if let Some(s) = &mut self.skipped {
                        s.record_match();
                    }
                    if !m.has_next() {
                        break;
                    }
                    if let Some(s) = &mut self.skipped {
                        s.state = suffixes.save_state();
                    }
                }
            }
            if !matched {
                if let Some(s) = &mut self.skipped {
                    s.skip(cp);
                    suffixes.reset_to_state(s.state);
                }
                prev_cc = fcd16 & 0xff;
            }
            cp = self.next_skipped_code_point();
            if cp < 0 {
                break;
            }
            since_match += 1;
            fcd16 = d.get_fcd16(cp);
            if fcd16 <= 0xff {
                break;
            }
        }
        self.backward_num_skipped(since_match);
        let is_top_discontiguous = !self.skipped_nonempty();
        if let Some(s) = &mut self.skipped {
            s.replace_match();
        }
        if is_top_discontiguous && self.skipped_nonempty() {
            let mut d = d;
            cp = c::SENTINEL_CP;
            loop {
                self.append_ces_from_ce32(d, cp, ce32, true);
                let Some(s) = &mut self.skipped else { break };
                if !s.has_next() {
                    break;
                }
                cp = s.next();
                ce32 = self.data.get_ce32(cp);
                if ce32 == c::FALLBACK_CE32 {
                    d = self.data.base_or_self();
                    ce32 = d.get_ce32(cp);
                } else {
                    d = self.data;
                }
            }
            if let Some(s) = &mut self.skipped {
                s.clear();
            }
            ce32 = c::NO_CE32;
        }
        ce32
    }

    /// `appendNumericCEs(ce32, forward)` with `forward == true`.
    fn append_numeric_ces(&mut self, mut ce32: i32) {
        let mut digits: Vec<u8> = Vec::new();
        loop {
            digits.push(c::digit_from_ce32(ce32));
            if self.num_cp_fwd == 0 {
                break;
            }
            let cp = self.next_code_point();
            if cp < 0 {
                break;
            }
            ce32 = self.data.get_ce32(cp);
            if ce32 == c::FALLBACK_CE32 {
                ce32 = self.data.base_or_self().get_ce32(cp);
            }
            if !c::has_ce32_tag(ce32, c::DIGIT_TAG) {
                self.backward_num_code_points(1);
                break;
            }
            if self.num_cp_fwd > 0 {
                self.num_cp_fwd = self.num_cp_fwd.saturating_sub(1);
            }
        }
        let mut pos = 0usize;
        loop {
            while pos < digits.len().saturating_sub(1) && digits[pos] == 0 {
                pos += 1;
            }
            let segment_length = (digits.len() - pos).min(254);
            self.append_numeric_segment_ces(&digits[pos..pos + segment_length]);
            pos += segment_length;
            if pos >= digits.len() {
                break;
            }
        }
    }

    /// `appendNumericSegmentCEs(digits)`.
    // ARITH: at most 254 digits, values below 10^7 in the short forms.
    #[allow(clippy::arithmetic_side_effects)]
    fn append_numeric_segment_ces(&mut self, digits: &[u8]) {
        let mut length = digits.len();
        let numeric_primary = self.data.numeric_primary;
        if length <= 7 {
            let mut value = i64::from(digits[0]);
            for &d in &digits[1..length] {
                value = value * 10 + i64::from(d);
            }
            let mut first_byte = 2i64;
            let mut num_bytes = 74i64;
            if value < num_bytes {
                let primary = numeric_primary | ((first_byte + value) << 16);
                self.ce_buffer.push(c::make_ce(primary));
                return;
            }
            value -= num_bytes;
            first_byte += num_bytes;
            num_bytes = 40;
            if value < num_bytes * 254 {
                let primary =
                    numeric_primary | ((first_byte + value / 254) << 16) | ((2 + value % 254) << 8);
                self.ce_buffer.push(c::make_ce(primary));
                return;
            }
            value -= num_bytes * 254;
            first_byte += num_bytes;
            num_bytes = 16;
            if value < num_bytes * 254 * 254 {
                let mut primary = numeric_primary | (2 + value % 254);
                value /= 254;
                primary |= (2 + value % 254) << 8;
                value /= 254;
                primary |= (first_byte + value % 254) << 16;
                self.ce_buffer.push(c::make_ce(primary));
                return;
            }
        }
        let num_pairs = length.div_ceil(2) as i64;
        let mut primary = numeric_primary | ((132 - 4 + num_pairs) << 16);
        while length >= 2 && digits[length - 1] == 0 && digits[length - 2] == 0 {
            length -= 2;
        }
        let (mut pair, mut pos) = if length & 1 != 0 {
            (i64::from(digits[0]), 1)
        } else {
            (i64::from(digits[0]) * 10 + i64::from(digits[1]), 2)
        };
        pair = 11 + 2 * pair;
        let mut shift = 8;
        while pos < length {
            if shift == 0 {
                primary |= pair;
                self.ce_buffer.push(c::make_ce(primary));
                primary = numeric_primary;
                shift = 16;
            } else {
                primary |= pair << shift;
                shift -= 8;
            }
            pair = 11 + 2 * (i64::from(digits[pos]) * 10 + i64::from(digits[pos + 1]));
            pos += 2;
        }
        primary |= (pair - 1) << shift;
        self.ce_buffer.push(c::make_ce(primary));
    }
}
