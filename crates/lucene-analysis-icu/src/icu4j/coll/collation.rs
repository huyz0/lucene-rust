//! `com.ibm.icu.impl.coll.Collation`: the collation element (CE) and CE32
//! encodings and their constants.
//!
//! A CE is a 64-bit `pppppppp ssss tttt` (primary, secondary, tertiary with
//! the case bits on top and the quaternary bits in the tertiary's low
//! byte); a CE32 is the 32-bit form stored in the data, either a simple
//! `pppp ss tt` or, when its low byte is `>= 0xc0`, a special one whose low
//! nibble is a tag and whose upper bits index the data. Java's `long` CE and
//! `int` CE32 are `i64` and `i32`; the shifts and masks are Java's, so a
//! CE32 with its top bit set stays negative where Java's does.

pub const SENTINEL_CP: i32 = -1;
pub const TERMINATOR_BYTE: i32 = 0;
pub const LEVEL_SEPARATOR_BYTE: i32 = 1;
pub const MERGE_SEPARATOR_BYTE: i32 = 2;
pub const MERGE_SEPARATOR_PRIMARY: i64 = 0x0200_0000;
pub const PRIMARY_COMPRESSION_LOW_BYTE: i32 = 3;
pub const PRIMARY_COMPRESSION_HIGH_BYTE: i32 = 0xff;
pub const COMMON_BYTE: i32 = 5;
pub const COMMON_WEIGHT16: i32 = 0x0500;
pub const COMMON_SECONDARY_CE: i64 = 0x0500_0000;
pub const COMMON_TERTIARY_CE: i64 = 0x0500;
pub const COMMON_SEC_AND_TER_CE: i64 = 0x0500_0500;
pub const CASE_MASK: i32 = 0xc000;
pub const ONLY_TERTIARY_MASK: i32 = 0x3f3f;
pub const CASE_AND_TERTIARY_MASK: i32 = CASE_MASK | ONLY_TERTIARY_MASK;
pub const UNASSIGNED_IMPLICIT_BYTE: i64 = 0xfe;
pub const TRAIL_WEIGHT_BYTE: i32 = 0xff;
pub const MAX_PRIMARY: i64 = 0xffff_0000;
pub const MAX_REGULAR_CE32: i32 = 0xffff_0505_u32 as i32;
pub const FFFD_CE32: i32 = MAX_REGULAR_CE32.wrapping_sub(0x20000);
pub const SPECIAL_CE32_LOW_BYTE: i32 = 0xc0;
pub const FALLBACK_CE32: i32 = SPECIAL_CE32_LOW_BYTE;
pub const LONG_PRIMARY_CE32_LOW_BYTE: i32 = 0xc1;
pub const UNASSIGNED_CE32: i32 = -1;
pub const NO_CE32: i32 = 1;
pub const NO_CE_PRIMARY: i64 = 1;
pub const NO_CE_WEIGHT16: i32 = 0x0100;
pub const NO_CE: i64 = 0x1_0100_0100;

pub const PRIMARY_LEVEL: i32 = 1;
pub const SECONDARY_LEVEL: i32 = 2;
pub const CASE_LEVEL: i32 = 3;
pub const TERTIARY_LEVEL: i32 = 4;
pub const QUATERNARY_LEVEL: i32 = 5;

pub const PRIMARY_LEVEL_FLAG: i32 = 2;
pub const SECONDARY_LEVEL_FLAG: i32 = 4;
pub const CASE_LEVEL_FLAG: i32 = 8;
pub const TERTIARY_LEVEL_FLAG: i32 = 0x10;
pub const QUATERNARY_LEVEL_FLAG: i32 = 0x20;

pub const FALLBACK_TAG: i32 = 0;
pub const LONG_PRIMARY_TAG: i32 = 1;
pub const LONG_SECONDARY_TAG: i32 = 2;
pub const RESERVED_TAG_3: i32 = 3;
pub const LATIN_EXPANSION_TAG: i32 = 4;
pub const EXPANSION32_TAG: i32 = 5;
pub const EXPANSION_TAG: i32 = 6;
pub const BUILDER_DATA_TAG: i32 = 7;
pub const PREFIX_TAG: i32 = 8;
pub const CONTRACTION_TAG: i32 = 9;
pub const DIGIT_TAG: i32 = 10;
pub const U0000_TAG: i32 = 11;
pub const HANGUL_TAG: i32 = 12;
pub const LEAD_SURROGATE_TAG: i32 = 13;
pub const OFFSET_TAG: i32 = 14;
pub const IMPLICIT_TAG: i32 = 15;

pub const CONTRACT_SINGLE_CP_NO_MATCH: i32 = 0x100;
pub const CONTRACT_NEXT_CCC: i32 = 0x200;
pub const CONTRACT_TRAILING_CCC: i32 = 0x400;
pub const HANGUL_NO_SPECIAL_JAMO: i32 = 0x100;
pub const LEAD_ALL_UNASSIGNED: i32 = 0;
pub const LEAD_ALL_FALLBACK: i32 = 0x100;
pub const LEAD_TYPE_MASK: i32 = 0x300;

/// `isSpecialCE32`.
#[inline]
pub fn is_special_ce32(ce32: i32) -> bool {
    (ce32 & 0xff) >= SPECIAL_CE32_LOW_BYTE
}

/// `tagFromCE32`.
#[inline]
pub fn tag_from_ce32(ce32: i32) -> i32 {
    ce32 & 0xf
}

/// `hasCE32Tag`.
#[inline]
pub fn has_ce32_tag(ce32: i32, tag: i32) -> bool {
    is_special_ce32(ce32) && tag_from_ce32(ce32) == tag
}

/// `isSimpleOrLongCE32`.
pub fn is_simple_or_long_ce32(ce32: i32) -> bool {
    !is_special_ce32(ce32)
        || tag_from_ce32(ce32) == LONG_PRIMARY_TAG
        || tag_from_ce32(ce32) == LONG_SECONDARY_TAG
}

/// `ceFromLongPrimaryCE32`.
#[inline]
pub fn ce_from_long_primary_ce32(ce32: i32) -> i64 {
    (i64::from(ce32 & !0xff) << 32) | COMMON_SEC_AND_TER_CE
}

/// `ceFromLongSecondaryCE32`.
#[inline]
pub fn ce_from_long_secondary_ce32(ce32: i32) -> i64 {
    i64::from(ce32) & 0xffff_ff00
}

/// `latinCE0FromCE32`.
#[inline]
pub fn latin_ce0_from_ce32(ce32: i32) -> i64 {
    (i64::from(ce32 & (0xff00_0000_u32 as i32)) << 32)
        | COMMON_SECONDARY_CE
        | i64::from((ce32 & 0xff_0000) >> 8)
}

/// `latinCE1FromCE32`.
#[inline]
pub fn latin_ce1_from_ce32(ce32: i32) -> i64 {
    ((i64::from(ce32) & 0xff00) << 16) | COMMON_TERTIARY_CE
}

/// `indexFromCE32`: the unsigned top 19 bits.
#[inline]
pub fn index_from_ce32(ce32: i32) -> usize {
    // u32 -> usize never truncates on the targets this crate builds for.
    ((ce32 as u32) >> 13) as usize
}

/// `lengthFromCE32`.
#[inline]
pub fn length_from_ce32(ce32: i32) -> usize {
    ((ce32 >> 8) & 31) as usize
}

/// `digitFromCE32`.
#[inline]
pub fn digit_from_ce32(ce32: i32) -> u8 {
    ((ce32 >> 8) & 0xf) as u8
}

/// `ceFromSimpleCE32`.
#[inline]
pub fn ce_from_simple_ce32(ce32: i32) -> i64 {
    (i64::from(ce32 & (0xffff_0000_u32 as i32)) << 32)
        | (i64::from(ce32 & 0xff00) << 16)
        | i64::from((ce32 & 0xff) << 8)
}

/// `ceFromCE32`.
pub fn ce_from_ce32(ce32: i32) -> i64 {
    let tertiary = ce32 & 0xff;
    if tertiary < SPECIAL_CE32_LOW_BYTE {
        ce_from_simple_ce32(ce32)
    } else {
        let ce32 = ce32.wrapping_sub(tertiary);
        if tertiary & 0xf == LONG_PRIMARY_TAG {
            (i64::from(ce32) << 32) | COMMON_SEC_AND_TER_CE
        } else {
            // LONG_SECONDARY_TAG (any other tag is not a CE here)
            i64::from(ce32) & 0xffff_ffff
        }
    }
}

/// `makeCE(p)`.
#[inline]
pub fn make_ce(p: i64) -> i64 {
    (p << 32) | COMMON_SEC_AND_TER_CE
}

/// `incThreeBytePrimaryByOffset`.
// ARITH: Java `int` arithmetic on a three-byte primary and a code point
// offset times a step below 0x80: no overflow for any code point.
#[allow(clippy::arithmetic_side_effects)]
pub fn inc_three_byte_primary_by_offset(
    base_primary: i64,
    is_compressible: bool,
    offset: i32,
) -> i64 {
    let mut offset = offset + ((base_primary >> 8) as i32 & 0xff) - 2;
    let mut primary = i64::from(((offset % 254) + 2) << 8);
    offset /= 254;
    if is_compressible {
        offset += ((base_primary >> 16) as i32 & 0xff) - 4;
        primary |= i64::from(((offset % 251) + 4) << 16);
        offset /= 251;
    } else {
        offset += ((base_primary >> 16) as i32 & 0xff) - 2;
        primary |= i64::from(((offset % 254) + 2) << 16);
        offset /= 254;
    }
    primary | ((base_primary & 0xff00_0000) + (i64::from(offset) << 24))
}

/// `getThreeBytePrimaryForOffsetData`.
// ARITH: c and the base code point are code points, the step below 0x80.
#[allow(clippy::arithmetic_side_effects)]
pub fn get_three_byte_primary_for_offset_data(c: i32, data_ce: i64) -> i64 {
    let p = ((data_ce as u64) >> 32) as i64;
    let lower32 = data_ce as i32;
    let offset = c.wrapping_sub(lower32 >> 8).wrapping_mul(lower32 & 0x7f);
    let is_compressible = lower32 & 0x80 != 0;
    inc_three_byte_primary_by_offset(p, is_compressible, offset)
}

/// `unassignedPrimaryFromCodePoint`.
// ARITH: c is a code point (or -1 + 1 = 0 for the sentinel), small.
#[allow(clippy::arithmetic_side_effects)]
pub fn unassigned_primary_from_code_point(c: i32) -> i64 {
    let mut c = i64::from(c) + 1;
    let mut primary = 2 + (c % 18) * 14;
    c /= 18;
    primary |= (2 + (c % 254)) << 8;
    c /= 254;
    primary |= (4 + (c % 251)) << 16;
    primary | (UNASSIGNED_IMPLICIT_BYTE << 24)
}

/// `unassignedCEFromCodePoint`.
pub fn unassigned_ce_from_code_point(c: i32) -> i64 {
    make_ce(unassigned_primary_from_code_point(c))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ce_shapes() {
        assert_eq!(ce_from_simple_ce32(0x1234_0567), 0x1234_0000_0500_6700);
        assert_eq!(ce_from_ce32(0x1234_56c1), 0x1234_5600_0500_0500);
        assert_eq!(ce_from_ce32(0x1234_56c2), 0x1234_5600);
        assert_eq!(
            ce_from_long_primary_ce32(0x1234_56c1),
            0x1234_5600_0500_0500
        );
        assert_eq!(ce_from_long_secondary_ce32(0x1234_56c2), 0x1234_5600);
        assert!(is_simple_or_long_ce32(0x0505));
        assert!(is_simple_or_long_ce32(0x1234_56c2));
        assert!(!is_simple_or_long_ce32(0x1234_56c9));
        assert_eq!(index_from_ce32(-1), 0x7ffff);
        assert_eq!(length_from_ce32(0x1f00), 31);
        assert_eq!(digit_from_ce32(0x7ca), 7);
        assert_eq!(latin_ce0_from_ce32(0x1234_5678), 0x1200_0000_0500_3400);
        assert_eq!(latin_ce1_from_ce32(0x1234_5678), 0x5600_0500);
        assert_eq!((unassigned_ce_from_code_point(-1) as u64) >> 56, 0xfe);
        assert_eq!(make_ce(0x02), 0x2_0500_0500);
    }
}
