//! `com.ibm.icu.impl.coll.CollationKeys.writeSortKeyUpToQuaternary`,
//! `BOCSU.writeIdenticalLevelRun` and `RuleBasedCollator.writeSortKey`:
//! a string's sort key, byte for byte.
//!
//! The primary level is written straight into the key (compressible lead
//! bytes written once per run, with the compression terminators); the
//! secondary, case, tertiary and quaternary levels are collected apart, runs
//! of common weights compressed into single bytes counting up from below
//! or down from above the common weight, and appended after level
//! separators. The identical level is the NFD text in BOCSU, a
//! difference encoding of code points. The key ends with a `0` terminator
//! (which Lucene's `ICUCollatedTermAttributeImpl` keeps: `key.size` counts
//! it).

use crate::icu4j::coll::collation::{self as c};
use crate::icu4j::coll::iter::CollationIterator;
use crate::icu4j::coll::settings::{self as s, CollationSettings};
use crate::icu4j::normalizer2_impl::{Normalizer2Impl, ReorderingBuffer};
use crate::icu4j::utf16;

const SEC_COMMON_LOW: i32 = c::COMMON_BYTE;
const SEC_COMMON_MIDDLE: i32 = SEC_COMMON_LOW + 0x20;
const SEC_COMMON_HIGH: i32 = SEC_COMMON_LOW + 0x40;
const SEC_COMMON_MAX_COUNT: i32 = 0x21;
const CASE_LOWER_FIRST_COMMON_LOW: i32 = 1;
const CASE_LOWER_FIRST_COMMON_MIDDLE: i32 = 7;
const CASE_LOWER_FIRST_COMMON_HIGH: i32 = 13;
const CASE_LOWER_FIRST_COMMON_MAX_COUNT: i32 = 7;
const CASE_UPPER_FIRST_COMMON_LOW: i32 = 3;
const CASE_UPPER_FIRST_COMMON_MAX_COUNT: i32 = 13;
const TER_ONLY_COMMON_LOW: i32 = c::COMMON_BYTE;
const TER_ONLY_COMMON_MIDDLE: i32 = TER_ONLY_COMMON_LOW + 0x60;
const TER_ONLY_COMMON_HIGH: i32 = TER_ONLY_COMMON_LOW + 0xc0;
const TER_ONLY_COMMON_MAX_COUNT: i32 = 0x61;
const TER_LOWER_FIRST_COMMON_LOW: i32 = c::COMMON_BYTE;
const TER_LOWER_FIRST_COMMON_MIDDLE: i32 = TER_LOWER_FIRST_COMMON_LOW + 0x20;
const TER_LOWER_FIRST_COMMON_HIGH: i32 = TER_LOWER_FIRST_COMMON_LOW + 0x40;
const TER_LOWER_FIRST_COMMON_MAX_COUNT: i32 = 0x21;
const TER_UPPER_FIRST_COMMON_LOW: i32 = c::COMMON_BYTE + 0x80;
const TER_UPPER_FIRST_COMMON_MIDDLE: i32 = TER_UPPER_FIRST_COMMON_LOW + 0x20;
const TER_UPPER_FIRST_COMMON_HIGH: i32 = TER_UPPER_FIRST_COMMON_LOW + 0x40;
const TER_UPPER_FIRST_COMMON_MAX_COUNT: i32 = 0x21;
const QUAT_COMMON_LOW: i32 = 0x1c;
const QUAT_COMMON_MIDDLE: i32 = QUAT_COMMON_LOW + 0x70;
const QUAT_COMMON_HIGH: i32 = QUAT_COMMON_LOW + 0xe0;
const QUAT_COMMON_MAX_COUNT: i32 = 0x71;
const QUAT_SHIFTED_LIMIT_BYTE: i32 = QUAT_COMMON_LOW - 1;

/// `levelMasks[strength]`.
fn level_mask(strength: i32) -> i32 {
    match strength {
        s::PRIMARY => 2,
        s::SECONDARY => 6,
        s::TERTIARY => 0x16,
        s::QUATERNARY | s::IDENTICAL => 0x36,
        _ => 0,
    }
}

/// `SortKeyLevel`: one level's bytes. Java's `(byte)` casts keep the low
/// byte of an `int`.
#[derive(Debug, Default)]
struct Level(Vec<u8>);

impl Level {
    #[inline]
    fn append_byte(&mut self, b: i32) {
        self.0.push(b as u8);
    }

    fn append_weight16(&mut self, w: i32) {
        let b0 = (w >> 8) as u8;
        let b1 = w as u8;
        self.0.push(b0);
        if b1 != 0 {
            self.0.push(b1);
        }
    }

    fn append_weight32(&mut self, w: i64) {
        let bytes = [(w >> 24) as u8, (w >> 16) as u8, (w >> 8) as u8, w as u8];
        self.0.push(bytes[0]);
        if bytes[1] != 0 {
            self.0.push(bytes[1]);
            if bytes[2] != 0 {
                self.0.push(bytes[2]);
                if bytes[3] != 0 {
                    self.0.push(bytes[3]);
                }
            }
        }
    }

    fn append_reverse_weight16(&mut self, w: i32) {
        let b0 = (w >> 8) as u8;
        let b1 = w as u8;
        if b1 == 0 {
            self.0.push(b0);
        } else {
            self.0.push(b1);
            self.0.push(b0);
        }
    }

    /// `appendTo(sink)`: all but the trailing level separator.
    fn append_to(&self, sink: &mut Vec<u8>) {
        let n = self.0.len().saturating_sub(1);
        sink.extend_from_slice(&self.0[..n]);
    }
}

/// `writeSortKeyUpToQuaternary(iter, compressibleBytes, settings, sink,
/// PRIMARY_LEVEL, SIMPLE_LEVEL_FALLBACK, true)`.
// ARITH: run counters bounded by the text's CE count; weights are bytes
// and 16-bit values of CEs, combined as Java's int arithmetic does.
#[allow(clippy::arithmetic_side_effects)]
pub fn write_sort_key_up_to_quaternary(
    iter: &mut CollationIterator<'_>,
    data_compressible: &dyn Fn(usize) -> bool,
    settings: &CollationSettings,
    sink: &mut Vec<u8>,
) {
    let options = settings.options;
    let mut levels = level_mask(CollationSettings::strength_of(options));
    if options & s::CASE_LEVEL != 0 {
        levels |= c::CASE_LEVEL_FLAG;
    }
    // minLevel == PRIMARY_LEVEL: levels &= ~((1 << 1) - 1).
    levels &= !1;
    if levels == 0 {
        return;
    }
    let variable_top = if options & s::ALTERNATE_MASK == 0 {
        0
    } else {
        settings.variable_top + 1
    };
    let tertiary_mask = CollationSettings::tertiary_mask(options);
    let mut cases = Level::default();
    let mut secondaries = Level::default();
    let mut tertiaries = Level::default();
    let mut quaternaries = Level::default();
    let mut prev_reordered_primary: i64 = 0;
    let mut common_cases = 0;
    let mut common_secondaries = 0;
    let mut common_tertiaries = 0;
    let mut common_quaternaries = 0;
    let mut prev_secondary = 0;
    let mut sec_segment_start = 0usize;
    loop {
        iter.clear_ces_if_none_remaining();
        let mut ce = iter.next_ce();
        let mut p = ((ce as u64) >> 32) as i64;
        if p < variable_top && p > c::MERGE_SEPARATOR_PRIMARY {
            if common_quaternaries != 0 {
                common_quaternaries -= 1;
                while common_quaternaries >= QUAT_COMMON_MAX_COUNT {
                    quaternaries.append_byte(QUAT_COMMON_MIDDLE);
                    common_quaternaries -= QUAT_COMMON_MAX_COUNT;
                }
                quaternaries.append_byte(QUAT_COMMON_LOW + common_quaternaries);
                common_quaternaries = 0;
            }
            loop {
                if levels & c::QUATERNARY_LEVEL_FLAG != 0 {
                    if settings.has_reordering() {
                        p = settings.reorder(p);
                    }
                    if ((p as u32) >> 24) as i32 >= QUAT_SHIFTED_LIMIT_BYTE {
                        quaternaries.append_byte(QUAT_SHIFTED_LIMIT_BYTE);
                    }
                    quaternaries.append_weight32(p);
                }
                loop {
                    ce = iter.next_ce();
                    p = ((ce as u64) >> 32) as i64;
                    if p != 0 {
                        break;
                    }
                }
                if !(p < variable_top && p > c::MERGE_SEPARATOR_PRIMARY) {
                    break;
                }
            }
        }
        if p > c::NO_CE_PRIMARY && levels & c::PRIMARY_LEVEL_FLAG != 0 {
            let is_compressible = data_compressible(((p as u32) >> 24) as usize);
            if settings.has_reordering() {
                p = settings.reorder(p);
            }
            let p1 = ((p as u32) >> 24) as i32;
            if !is_compressible || p1 != ((prev_reordered_primary as u32) >> 24) as i32 {
                if prev_reordered_primary != 0 {
                    if p < prev_reordered_primary {
                        if p1 > c::MERGE_SEPARATOR_BYTE {
                            sink.push(c::PRIMARY_COMPRESSION_LOW_BYTE as u8);
                        }
                    } else {
                        sink.push(c::PRIMARY_COMPRESSION_HIGH_BYTE as u8);
                    }
                }
                sink.push(p1 as u8);
                prev_reordered_primary = if is_compressible { p } else { 0 };
            }
            let p2 = (p >> 16) as u8;
            if p2 != 0 {
                let p3 = (p >> 8) as u8;
                let p4 = p as u8;
                sink.push(p2);
                if p3 != 0 {
                    sink.push(p3);
                    if p4 != 0 {
                        sink.push(p4);
                    }
                }
            }
        }
        let lower32 = ce as i32;
        if lower32 == 0 {
            continue;
        }
        if levels & c::SECONDARY_LEVEL_FLAG != 0 {
            let s = ((lower32 as u32) >> 16) as i32;
            if s == 0 {
                // secondary ignorable
            } else if s == c::COMMON_WEIGHT16
                && (options & s::BACKWARD_SECONDARY == 0 || p != c::MERGE_SEPARATOR_PRIMARY)
            {
                common_secondaries += 1;
            } else if options & s::BACKWARD_SECONDARY == 0 {
                if common_secondaries != 0 {
                    common_secondaries -= 1;
                    while common_secondaries >= SEC_COMMON_MAX_COUNT {
                        secondaries.append_byte(SEC_COMMON_MIDDLE);
                        common_secondaries -= SEC_COMMON_MAX_COUNT;
                    }
                    let b = if s < c::COMMON_WEIGHT16 {
                        SEC_COMMON_LOW + common_secondaries
                    } else {
                        SEC_COMMON_HIGH - common_secondaries
                    };
                    secondaries.append_byte(b);
                    common_secondaries = 0;
                }
                secondaries.append_weight16(s);
            } else {
                if common_secondaries != 0 {
                    common_secondaries -= 1;
                    let remainder = common_secondaries % SEC_COMMON_MAX_COUNT;
                    let b = if prev_secondary < c::COMMON_WEIGHT16 {
                        SEC_COMMON_LOW + remainder
                    } else {
                        SEC_COMMON_HIGH - remainder
                    };
                    secondaries.append_byte(b);
                    common_secondaries -= remainder;
                    while common_secondaries > 0 {
                        secondaries.append_byte(SEC_COMMON_MIDDLE);
                        common_secondaries -= SEC_COMMON_MAX_COUNT;
                    }
                }
                if 0 < p && p <= c::MERGE_SEPARATOR_PRIMARY {
                    // Reverse the segment since the last separator.
                    let secs = &mut secondaries.0;
                    if !secs.is_empty() {
                        let mut last = secs.len() - 1;
                        while sec_segment_start < last {
                            secs.swap(sec_segment_start, last);
                            sec_segment_start += 1;
                            last -= 1;
                        }
                    }
                    secondaries.append_byte(if p == c::NO_CE_PRIMARY {
                        c::LEVEL_SEPARATOR_BYTE
                    } else {
                        c::MERGE_SEPARATOR_BYTE
                    });
                    prev_secondary = 0;
                    sec_segment_start = secondaries.0.len();
                } else {
                    secondaries.append_reverse_weight16(s);
                    prev_secondary = s;
                }
            }
        }
        if levels & c::CASE_LEVEL_FLAG != 0 {
            let ignorable = if CollationSettings::strength_of(options) == s::PRIMARY {
                p == 0
            } else {
                ((lower32 as u32) >> 16) == 0
            };
            if !ignorable {
                let mut cb = ((lower32 as u32) >> 8) as i32 & 0xff;
                if cb & 0xc0 == 0 && cb > c::LEVEL_SEPARATOR_BYTE {
                    common_cases += 1;
                } else {
                    if options & s::UPPER_FIRST == 0 {
                        if common_cases != 0
                            && (cb > c::LEVEL_SEPARATOR_BYTE || !cases.0.is_empty())
                        {
                            common_cases -= 1;
                            while common_cases >= CASE_LOWER_FIRST_COMMON_MAX_COUNT {
                                cases.append_byte(CASE_LOWER_FIRST_COMMON_MIDDLE << 4);
                                common_cases -= CASE_LOWER_FIRST_COMMON_MAX_COUNT;
                            }
                            let b = if cb <= c::LEVEL_SEPARATOR_BYTE {
                                CASE_LOWER_FIRST_COMMON_LOW + common_cases
                            } else {
                                CASE_LOWER_FIRST_COMMON_HIGH - common_cases
                            };
                            cases.append_byte(b << 4);
                            common_cases = 0;
                        }
                        if cb > c::LEVEL_SEPARATOR_BYTE {
                            cb = (CASE_LOWER_FIRST_COMMON_HIGH + (cb >> 6)) << 4;
                        }
                    } else {
                        if common_cases != 0 {
                            common_cases -= 1;
                            while common_cases >= CASE_UPPER_FIRST_COMMON_MAX_COUNT {
                                cases.append_byte(CASE_UPPER_FIRST_COMMON_LOW << 4);
                                common_cases -= CASE_UPPER_FIRST_COMMON_MAX_COUNT;
                            }
                            cases.append_byte((CASE_UPPER_FIRST_COMMON_LOW + common_cases) << 4);
                            common_cases = 0;
                        }
                        if cb > c::LEVEL_SEPARATOR_BYTE {
                            cb = (CASE_UPPER_FIRST_COMMON_LOW - (cb >> 6)) << 4;
                        }
                    }
                    cases.append_byte(cb);
                }
            }
        }
        if levels & c::TERTIARY_LEVEL_FLAG != 0 {
            let mut t = lower32 & tertiary_mask;
            if t == c::COMMON_WEIGHT16 {
                common_tertiaries += 1;
            } else if tertiary_mask & 0x8000 == 0 {
                if common_tertiaries != 0 {
                    common_tertiaries -= 1;
                    while common_tertiaries >= TER_ONLY_COMMON_MAX_COUNT {
                        tertiaries.append_byte(TER_ONLY_COMMON_MIDDLE);
                        common_tertiaries -= TER_ONLY_COMMON_MAX_COUNT;
                    }
                    let b = if t < c::COMMON_WEIGHT16 {
                        TER_ONLY_COMMON_LOW + common_tertiaries
                    } else {
                        TER_ONLY_COMMON_HIGH - common_tertiaries
                    };
                    tertiaries.append_byte(b);
                    common_tertiaries = 0;
                }
                if t > c::COMMON_WEIGHT16 {
                    t += 0xc000;
                }
                tertiaries.append_weight16(t);
            } else if options & s::UPPER_FIRST == 0 {
                if common_tertiaries != 0 {
                    common_tertiaries -= 1;
                    while common_tertiaries >= TER_LOWER_FIRST_COMMON_MAX_COUNT {
                        tertiaries.append_byte(TER_LOWER_FIRST_COMMON_MIDDLE);
                        common_tertiaries -= TER_LOWER_FIRST_COMMON_MAX_COUNT;
                    }
                    let b = if t < c::COMMON_WEIGHT16 {
                        TER_LOWER_FIRST_COMMON_LOW + common_tertiaries
                    } else {
                        TER_LOWER_FIRST_COMMON_HIGH - common_tertiaries
                    };
                    tertiaries.append_byte(b);
                    common_tertiaries = 0;
                }
                if t > c::COMMON_WEIGHT16 {
                    t += 0x4000;
                }
                tertiaries.append_weight16(t);
            } else {
                if t <= c::NO_CE_WEIGHT16 {
                    // Keep separators unchanged.
                } else if ((lower32 as u32) >> 16) != 0 {
                    t ^= 0xc000;
                    if t < (TER_UPPER_FIRST_COMMON_HIGH << 8) {
                        t -= 0x4000;
                    }
                } else {
                    t += 0x4000;
                }
                if common_tertiaries != 0 {
                    common_tertiaries -= 1;
                    while common_tertiaries >= TER_UPPER_FIRST_COMMON_MAX_COUNT {
                        tertiaries.append_byte(TER_UPPER_FIRST_COMMON_MIDDLE);
                        common_tertiaries -= TER_UPPER_FIRST_COMMON_MAX_COUNT;
                    }
                    let b = if t < (TER_UPPER_FIRST_COMMON_LOW << 8) {
                        TER_UPPER_FIRST_COMMON_LOW + common_tertiaries
                    } else {
                        TER_UPPER_FIRST_COMMON_HIGH - common_tertiaries
                    };
                    tertiaries.append_byte(b);
                    common_tertiaries = 0;
                }
                tertiaries.append_weight16(t);
            }
        }
        if levels & c::QUATERNARY_LEVEL_FLAG != 0 {
            let mut q = lower32 & 0xffff;
            if q & 0xc0 == 0 && q > c::NO_CE_WEIGHT16 {
                common_quaternaries += 1;
            } else if q == c::NO_CE_WEIGHT16
                && options & s::ALTERNATE_MASK == 0
                && quaternaries.0.is_empty()
            {
                quaternaries.append_byte(c::LEVEL_SEPARATOR_BYTE);
            } else {
                if q == c::NO_CE_WEIGHT16 {
                    q = c::LEVEL_SEPARATOR_BYTE;
                } else {
                    q = 0xfc + ((q >> 6) & 3);
                }
                if common_quaternaries != 0 {
                    common_quaternaries -= 1;
                    while common_quaternaries >= QUAT_COMMON_MAX_COUNT {
                        quaternaries.append_byte(QUAT_COMMON_MIDDLE);
                        common_quaternaries -= QUAT_COMMON_MAX_COUNT;
                    }
                    let b = if q < QUAT_COMMON_LOW {
                        QUAT_COMMON_LOW + common_quaternaries
                    } else {
                        QUAT_COMMON_HIGH - common_quaternaries
                    };
                    quaternaries.append_byte(b);
                    common_quaternaries = 0;
                }
                quaternaries.append_byte(q);
            }
        }
        if ((lower32 as u32) >> 24) as i32 == c::LEVEL_SEPARATOR_BYTE {
            break; // ce == NO_CE
        }
    }
    let sep = c::LEVEL_SEPARATOR_BYTE as u8;
    if levels & c::SECONDARY_LEVEL_FLAG != 0 {
        sink.push(sep);
        secondaries.append_to(sink);
    }
    if levels & c::CASE_LEVEL_FLAG != 0 {
        sink.push(sep);
        let length = cases.0.len().saturating_sub(1);
        let mut b: u8 = 0;
        for &cb in &cases.0[..length] {
            if b == 0 {
                b = cb;
            } else {
                sink.push(b | ((cb >> 4) & 0xf));
                b = 0;
            }
        }
        if b != 0 {
            sink.push(b);
        }
    }
    if levels & c::TERTIARY_LEVEL_FLAG != 0 {
        sink.push(sep);
        tertiaries.append_to(sink);
    }
    if levels & c::QUATERNARY_LEVEL_FLAG != 0 {
        sink.push(sep);
        quaternaries.append_to(sink);
    }
}

// --------------------------------------------------------------- BOCSU

const SLOPE_MIN: i32 = 3;
const SLOPE_MAX: i32 = 0xff;
const SLOPE_MIDDLE: i32 = 0x81;
const SLOPE_TAIL_COUNT: i32 = SLOPE_MAX - SLOPE_MIN + 1;
const SLOPE_SINGLE: i32 = 80;
const SLOPE_LEAD_2: i32 = 42;
const SLOPE_LEAD_3: i32 = 3;
const SLOPE_REACH_POS_1: i32 = SLOPE_SINGLE;
const SLOPE_REACH_NEG_1: i32 = -SLOPE_SINGLE;
const SLOPE_REACH_POS_2: i32 = SLOPE_LEAD_2 * SLOPE_TAIL_COUNT + SLOPE_LEAD_2 - 1;
const SLOPE_REACH_NEG_2: i32 = -SLOPE_REACH_POS_2 - 1;
const SLOPE_REACH_POS_3: i32 = SLOPE_LEAD_3 * SLOPE_TAIL_COUNT * SLOPE_TAIL_COUNT
    + (SLOPE_LEAD_3 - 1) * SLOPE_TAIL_COUNT
    + (SLOPE_TAIL_COUNT - 1);
const SLOPE_REACH_NEG_3: i32 = -SLOPE_REACH_POS_3 - 1;
const SLOPE_START_POS_2: i32 = SLOPE_MIDDLE + SLOPE_SINGLE + 1;
const SLOPE_START_POS_3: i32 = SLOPE_START_POS_2 + SLOPE_LEAD_2;
const SLOPE_START_NEG_2: i32 = SLOPE_MIDDLE + SLOPE_REACH_NEG_1;
const SLOPE_START_NEG_3: i32 = SLOPE_START_NEG_2 - SLOPE_LEAD_2;

/// `getNegDivMod(number, factor)`: floor division and its non-negative
/// remainder.
// ARITH: factor is SLOPE_TAIL_COUNT (253).
#[allow(clippy::arithmetic_side_effects)]
fn neg_div_mod(number: i32, factor: i32) -> (i32, i32) {
    let mut modulo = number % factor;
    let mut result = number / factor;
    if modulo < 0 {
        result -= 1;
        modulo += factor;
    }
    (result, modulo)
}

/// `writeDiff(diff, buffer, offset)`.
// ARITH: diff is the difference of two code points (|diff| < 0x220000).
#[allow(clippy::arithmetic_side_effects)]
fn write_diff(mut diff: i32, out: &mut Vec<u8>) {
    if diff >= SLOPE_REACH_NEG_1 {
        if diff <= SLOPE_REACH_POS_1 {
            out.push((SLOPE_MIDDLE + diff) as u8);
        } else if diff <= SLOPE_REACH_POS_2 {
            out.push((SLOPE_START_POS_2 + diff / SLOPE_TAIL_COUNT) as u8);
            out.push((SLOPE_MIN + diff % SLOPE_TAIL_COUNT) as u8);
        } else if diff <= SLOPE_REACH_POS_3 {
            let b2 = (SLOPE_MIN + diff % SLOPE_TAIL_COUNT) as u8;
            diff /= SLOPE_TAIL_COUNT;
            let b1 = (SLOPE_MIN + diff % SLOPE_TAIL_COUNT) as u8;
            let b0 = (SLOPE_START_POS_3 + diff / SLOPE_TAIL_COUNT) as u8;
            out.extend_from_slice(&[b0, b1, b2]);
        } else {
            let b3 = (SLOPE_MIN + diff % SLOPE_TAIL_COUNT) as u8;
            diff /= SLOPE_TAIL_COUNT;
            let b2 = (SLOPE_MIN + diff % SLOPE_TAIL_COUNT) as u8;
            diff /= SLOPE_TAIL_COUNT;
            let b1 = (SLOPE_MIN + diff % SLOPE_TAIL_COUNT) as u8;
            out.extend_from_slice(&[SLOPE_MAX as u8, b1, b2, b3]);
        }
    } else {
        let (d, modulo) = neg_div_mod(diff, SLOPE_TAIL_COUNT);
        if diff >= SLOPE_REACH_NEG_2 {
            out.push((SLOPE_START_NEG_2 + d) as u8);
            out.push((SLOPE_MIN + modulo) as u8);
        } else if diff >= SLOPE_REACH_NEG_3 {
            let b2 = (SLOPE_MIN + modulo) as u8;
            let (d, modulo) = neg_div_mod(d, SLOPE_TAIL_COUNT);
            let b1 = (SLOPE_MIN + modulo) as u8;
            let b0 = (SLOPE_START_NEG_3 + d) as u8;
            out.extend_from_slice(&[b0, b1, b2]);
        } else {
            let b3 = (SLOPE_MIN + modulo) as u8;
            let (d, modulo) = neg_div_mod(d, SLOPE_TAIL_COUNT);
            let b2 = (SLOPE_MIN + modulo) as u8;
            let (_, modulo) = neg_div_mod(d, SLOPE_TAIL_COUNT);
            let b1 = (SLOPE_MIN + modulo) as u8;
            out.extend_from_slice(&[SLOPE_MIN as u8, b1, b2, b3]);
        }
    }
}

/// `BOCSU.writeIdenticalLevelRun(prev, s, 0, length, sink)`.
// ARITH: prev and c are code points.
#[allow(clippy::arithmetic_side_effects)]
fn write_identical_level_run(mut prev: i32, s: &[u16], out: &mut Vec<u8>) -> i32 {
    let mut i = 0;
    while i < s.len() {
        if !(0x4e00..0xa000).contains(&prev) {
            prev = (prev & !0x7f) - SLOPE_REACH_NEG_1;
        } else {
            prev = 0x9fff - SLOPE_REACH_POS_2;
        }
        let cp = utf16::code_point_at(s, i);
        i += if cp >= 0x10000 { 2 } else { 1 };
        if cp == 0xfffe {
            out.push(2);
            prev = 0;
        } else {
            write_diff(cp - prev, out);
            prev = cp;
        }
    }
    prev
}

/// `RuleBasedCollator.writeIdenticalLevel(s, sink)`: the level separator
/// and the NFD of `s` in BOCSU (the NFD quick-check-yes prefix as is).
pub fn write_identical_level(nfc: &Normalizer2Impl, s: &[u16], out: &mut Vec<u8>) {
    let nfd_qc_yes_limit = nfc.decompose(s, 0, s.len(), None);
    out.push(c::LEVEL_SEPARATOR_BYTE as u8);
    let mut prev = 0;
    if nfd_qc_yes_limit != 0 {
        prev = write_identical_level_run(prev, &s[..nfd_qc_yes_limit], out);
    }
    if nfd_qc_yes_limit < s.len() {
        let mut nfd = Vec::new();
        {
            let mut buffer =
                ReorderingBuffer::new(nfc, &mut nfd, s.len().saturating_sub(nfd_qc_yes_limit));
            nfc.decompose(s, nfd_qc_yes_limit, s.len(), Some(&mut buffer));
        }
        write_identical_level_run(prev, &nfd, out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bocsu_diffs() {
        let enc = |d: i32| {
            let mut v = Vec::new();
            write_diff(d, &mut v);
            v
        };
        assert_eq!(enc(0), vec![0x81]);
        assert_eq!(enc(80), vec![0xd1]);
        assert_eq!(enc(-80), vec![0x31]);
        assert_eq!(enc(81).len(), 2);
        assert_eq!(enc(-81).len(), 2);
        assert_eq!(enc(SLOPE_REACH_POS_2 + 1).len(), 3);
        assert_eq!(enc(SLOPE_REACH_NEG_2 - 1).len(), 3);
        assert_eq!(enc(SLOPE_REACH_POS_3 + 1).len(), 4);
        assert_eq!(enc(SLOPE_REACH_NEG_3 - 1).len(), 4);
        assert_eq!(neg_div_mod(-1, 253), (-1, 252));
        let mut v = Vec::new();
        assert_eq!(
            write_identical_level_run(0, &[0x4e00, 0xfffe, 0x41], &mut v),
            0x41
        );
        assert!(v.contains(&2));
    }

    #[test]
    fn levels() {
        let mut l = Level::default();
        l.append_weight16(0x0500);
        l.append_weight16(0x1234);
        l.append_reverse_weight16(0x0600);
        l.append_reverse_weight16(0x1234);
        l.append_weight32(0x1200_0000);
        l.append_weight32(0x1234_0000);
        l.append_weight32(0x1234_5600);
        l.append_weight32(0x1234_5678);
        l.append_byte(1);
        assert_eq!(
            l.0,
            vec![
                5, 0x12, 0x34, 6, 0x34, 0x12, 0x12, 0x12, 0x34, 0x12, 0x34, 0x56, 0x12, 0x34, 0x56,
                0x78, 1
            ]
        );
        let mut out = Vec::new();
        l.append_to(&mut out);
        assert_eq!(out.len(), 16);
        assert_eq!(level_mask(7), 0);
    }
}
