//! `com.ibm.icu.impl.coll.CollationSettings`: the options word (strength,
//! alternate handling, max variable, case first/level, backward
//! secondaries, numeric, FCD check), the variable top, and the script
//! reordering (a lead-byte permutation table plus ranges for split lead
//! bytes).

use std::sync::Arc;

use crate::icu4j::coll::collation;
use crate::icu4j::coll::data::{CollationData, REORDER_CODE_FIRST, REORDER_CODE_NONE};
use crate::IcuError;

pub const CHECK_FCD: i32 = 1;
pub const NUMERIC: i32 = 2;
pub const SHIFTED: i32 = 4;
pub const ALTERNATE_MASK: i32 = 0xc;
pub const MAX_VARIABLE_SHIFT: i32 = 4;
pub const MAX_VARIABLE_MASK: i32 = 0x70;
pub const UPPER_FIRST: i32 = 0x100;
pub const CASE_FIRST: i32 = 0x200;
pub const CASE_FIRST_AND_UPPER_MASK: i32 = CASE_FIRST | UPPER_FIRST;
pub const CASE_LEVEL: i32 = 0x400;
pub const BACKWARD_SECONDARY: i32 = 0x800;
pub const STRENGTH_SHIFT: i32 = 12;
pub const STRENGTH_MASK: i32 = 0xf000;

/// `Collator.PRIMARY` .. `IDENTICAL`.
pub const PRIMARY: i32 = 0;
pub const SECONDARY: i32 = 1;
pub const TERTIARY: i32 = 2;
pub const QUATERNARY: i32 = 3;
pub const IDENTICAL: i32 = 15;

/// `CollationSettings`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CollationSettings {
    pub options: i32,
    pub variable_top: i64,
    /// `reorderTable` (Java's `byte[]`, read unsigned); `None` without
    /// reordering.
    pub reorder_table: Option<Arc<[u8; 256]>>,
    pub min_high_no_reorder: i64,
    pub reorder_ranges: Vec<i64>,
    pub reorder_codes: Vec<i32>,
}

impl Default for CollationSettings {
    fn default() -> Self {
        CollationSettings {
            options: (TERTIARY << STRENGTH_SHIFT) | (1 << MAX_VARIABLE_SHIFT),
            variable_top: 0,
            reorder_table: None,
            min_high_no_reorder: 0,
            reorder_ranges: Vec::new(),
            reorder_codes: Vec::new(),
        }
    }
}

fn illegal(msg: impl Into<String>) -> IcuError {
    IcuError::with_kind(crate::IcuErrorKind::IllegalArgument, msg)
}

impl CollationSettings {
    /// `resetReordering()`.
    pub fn reset_reordering(&mut self) {
        self.reorder_table = None;
        self.min_high_no_reorder = 0;
        self.reorder_ranges.clear();
        self.reorder_codes.clear();
    }

    /// `aliasReordering(data, codesAndRanges, codesLength, table)`: the
    /// reordering a tailoring's data carries.
    pub fn alias_reordering(
        &mut self,
        data: &CollationData,
        codes_and_ranges: &[i32],
        codes_length: usize,
        table: Option<[u8; 256]>,
    ) -> Result<(), IcuError> {
        let codes = codes_and_ranges
            .get(..codes_length)
            .unwrap_or(codes_and_ranges)
            .to_vec();
        let ranges = codes_and_ranges.get(codes_length..).unwrap_or(&[]);
        if let Some(table) = table {
            let usable = if ranges.is_empty() {
                !reorder_table_has_split_bytes(&table)
            } else {
                ranges.len() >= 2
                    && ranges[0] & 0xffff == 0
                    && ranges[ranges.len().saturating_sub(1)] & 0xffff != 0
            };
            if usable {
                self.reorder_table = Some(Arc::new(table));
                self.reorder_codes = codes;
                let first_split = ranges.iter().position(|&r| r & 0xff_0000 != 0);
                match first_split {
                    None => {
                        self.min_high_no_reorder = 0;
                        self.reorder_ranges.clear();
                    }
                    Some(i) => {
                        let last = ranges[ranges.len().saturating_sub(1)];
                        self.min_high_no_reorder = i64::from(last) & 0xffff_0000;
                        self.set_reorder_ranges(&ranges[i..]);
                    }
                }
                return Ok(());
            }
        }
        self.set_reordering(data, &codes)
    }

    /// `setReordering(data, codes)`.
    pub fn set_reordering(&mut self, data: &CollationData, codes: &[i32]) -> Result<(), IcuError> {
        if codes.is_empty() || (codes.len() == 1 && codes[0] == REORDER_CODE_NONE) {
            self.reset_reordering();
            return Ok(());
        }
        let ranges = data.make_reorder_ranges(codes)?;
        if ranges.is_empty() {
            self.reset_reordering();
            return Ok(());
        }
        self.min_high_no_reorder = i64::from(ranges[ranges.len().saturating_sub(1)]) & 0xffff_0000;
        let mut table = [0u8; 256];
        let mut b = 0usize;
        let mut first_split: Option<usize> = None;
        for (i, &pair) in ranges.iter().enumerate() {
            let limit1 = ((pair as u32) >> 24) as usize;
            while b < limit1 {
                // Java: (byte)(b + pair): the low byte of the sum.
                table[b] = (b as i32).wrapping_add(pair) as u8;
                b = b.saturating_add(1);
            }
            if pair & 0xff_0000 != 0 {
                if let Some(slot) = table.get_mut(limit1) {
                    *slot = 0;
                }
                b = limit1.saturating_add(1);
                first_split.get_or_insert(i);
            }
        }
        while b <= 0xff {
            table[b] = b as u8;
            b = b.saturating_add(1);
        }
        self.reorder_table = Some(Arc::new(table));
        self.reorder_codes = codes.to_vec();
        match first_split {
            None => self.reorder_ranges.clear(),
            Some(i) => self.set_reorder_ranges(&ranges[i..]),
        }
        Ok(())
    }

    fn set_reorder_ranges(&mut self, ranges: &[i32]) {
        self.reorder_ranges = ranges.iter().map(|&r| i64::from(r) & 0xffff_ffff).collect();
    }

    /// `copyReorderingFrom(other)`.
    pub fn copy_reordering_from(&mut self, other: &CollationSettings) {
        if !other.has_reordering() {
            self.reset_reordering();
            return;
        }
        self.min_high_no_reorder = other.min_high_no_reorder;
        self.reorder_table.clone_from(&other.reorder_table);
        self.reorder_ranges.clone_from(&other.reorder_ranges);
        self.reorder_codes.clone_from(&other.reorder_codes);
    }

    /// `hasReordering()`.
    #[inline]
    pub fn has_reordering(&self) -> bool {
        self.reorder_table.is_some()
    }

    /// `reorder(p)`: a primary moved by the reordering.
    #[inline]
    pub fn reorder(&self, p: i64) -> i64 {
        let Some(table) = &self.reorder_table else {
            return p;
        };
        let b = table[usize::from((p as u32 >> 24) as u8)];
        if b != 0 || p <= collation::NO_CE_PRIMARY {
            (i64::from(b) << 24) | (p & 0xff_ffff)
        } else {
            self.reorder_ex(p)
        }
    }

    fn reorder_ex(&self, p: i64) -> i64 {
        if p >= self.min_high_no_reorder {
            return p;
        }
        let q = p | 0xffff;
        let r = self
            .reorder_ranges
            .iter()
            .copied()
            .find(|&r| q < r)
            .unwrap_or(0);
        // Java: p + ((long)(short)r << 24).
        p.wrapping_add(i64::from(r as i16) << 24)
    }

    /// `setStrength(value)`.
    pub fn set_strength(&mut self, value: i32) -> Result<(), IcuError> {
        let no_strength = self.options & !STRENGTH_MASK;
        match value {
            PRIMARY | SECONDARY | TERTIARY | QUATERNARY | IDENTICAL => {
                self.options = no_strength | (value << STRENGTH_SHIFT);
                Ok(())
            }
            _ => Err(illegal(format!("illegal strength value {value}"))),
        }
    }

    /// `getStrength(options)`.
    #[inline]
    pub fn strength_of(options: i32) -> i32 {
        options >> STRENGTH_SHIFT
    }

    /// `getStrength()`.
    pub fn strength(&self) -> i32 {
        Self::strength_of(self.options)
    }

    /// `setFlag(bit, value)`.
    pub fn set_flag(&mut self, bit: i32, value: bool) {
        if value {
            self.options |= bit;
        } else {
            self.options &= !bit;
        }
    }

    /// `getFlag(bit)`.
    pub fn flag(&self, bit: i32) -> bool {
        self.options & bit != 0
    }

    /// `setCaseFirst(value)`: 0, `CASE_FIRST` or `CASE_FIRST_AND_UPPER_MASK`.
    pub fn set_case_first(&mut self, value: i32) {
        self.options = (self.options & !CASE_FIRST_AND_UPPER_MASK) | value;
    }

    /// `getCaseFirst()`.
    pub fn case_first(&self) -> i32 {
        self.options & CASE_FIRST_AND_UPPER_MASK
    }

    /// `setAlternateHandlingShifted(value)`.
    pub fn set_alternate_handling_shifted(&mut self, value: bool) {
        let no_alternate = self.options & !ALTERNATE_MASK;
        self.options = if value {
            no_alternate | SHIFTED
        } else {
            no_alternate
        };
    }

    /// `getAlternateHandling()`.
    pub fn alternate_handling(&self) -> bool {
        self.options & ALTERNATE_MASK != 0
    }

    /// `setMaxVariable(value, defaultOptions)`.
    pub fn set_max_variable(&mut self, value: i32, default_options: i32) -> Result<(), IcuError> {
        let no_max = self.options & !MAX_VARIABLE_MASK;
        match value {
            0..=3 => self.options = no_max | (value << MAX_VARIABLE_SHIFT),
            -1 => self.options = no_max | (default_options & MAX_VARIABLE_MASK),
            _ => return Err(illegal(format!("illegal maxVariable value {value}"))),
        }
        Ok(())
    }

    /// `getMaxVariable()`.
    pub fn max_variable(&self) -> i32 {
        (self.options & MAX_VARIABLE_MASK) >> MAX_VARIABLE_SHIFT
    }

    /// `getTertiaryMask(options)`.
    pub fn tertiary_mask(options: i32) -> i32 {
        if options & (CASE_LEVEL | CASE_FIRST) == CASE_FIRST {
            collation::CASE_AND_TERTIARY_MASK
        } else {
            collation::ONLY_TERTIARY_MASK
        }
    }

    /// `dontCheckFCD()`.
    pub fn dont_check_fcd(&self) -> bool {
        self.options & CHECK_FCD == 0
    }

    /// `isNumeric()`.
    pub fn is_numeric(&self) -> bool {
        self.options & NUMERIC != 0
    }
}

/// `reorderTableHasSplitBytes(table)`.
fn reorder_table_has_split_bytes(table: &[u8; 256]) -> bool {
    table[1..].contains(&0)
}

/// `Collator.ReorderCodes.FIRST` + `getMaxVariable()`: the group code.
pub fn max_variable_group(max_variable: i32) -> i32 {
    REORDER_CODE_FIRST.saturating_add(max_variable)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn options() {
        let mut s = CollationSettings::default();
        assert_eq!(s.strength(), TERTIARY);
        assert_eq!(s.max_variable(), 1);
        s.set_strength(IDENTICAL).unwrap();
        assert_eq!(s.strength(), IDENTICAL);
        assert!(s.set_strength(7).is_err());
        s.set_flag(NUMERIC, true);
        assert!(s.is_numeric() && s.flag(NUMERIC));
        s.set_flag(NUMERIC, false);
        assert!(!s.is_numeric());
        s.set_case_first(CASE_FIRST_AND_UPPER_MASK);
        assert_eq!(s.case_first(), CASE_FIRST_AND_UPPER_MASK);
        assert_eq!(
            CollationSettings::tertiary_mask(s.options),
            collation::CASE_AND_TERTIARY_MASK
        );
        s.set_flag(CASE_LEVEL, true);
        assert_eq!(
            CollationSettings::tertiary_mask(s.options),
            collation::ONLY_TERTIARY_MASK
        );
        s.set_alternate_handling_shifted(true);
        assert!(s.alternate_handling());
        s.set_alternate_handling_shifted(false);
        assert!(!s.alternate_handling());
        s.set_max_variable(3, 0).unwrap();
        assert_eq!(s.max_variable(), 3);
        s.set_max_variable(-1, 0x10).unwrap();
        assert_eq!(s.max_variable(), 1);
        assert!(s.set_max_variable(4, 0).is_err());
        assert!(s.dont_check_fcd());
        assert_eq!(max_variable_group(2), REORDER_CODE_FIRST + 2);
    }

    #[test]
    fn reorder_table_and_ranges() {
        let mut s = CollationSettings::default();
        assert_eq!(s.reorder(0x1234_5678), 0x1234_5678);
        let mut table = [0u8; 256];
        for (i, t) in table.iter_mut().enumerate() {
            *t = i as u8;
        }
        table[0x20] = 0x30;
        table[0x40] = 0; // split lead byte
        s.reorder_table = Some(Arc::new(table));
        s.min_high_no_reorder = 0x5000_0000;
        s.reorder_ranges = vec![0x4080_0000 | 2, 0x5000_0000];
        assert_eq!(s.reorder(0x2012_3456), 0x3012_3456);
        assert_eq!(s.reorder(1), 1);
        assert_eq!(s.reorder(0x4010_0000), 0x4210_0000);
        assert_eq!(s.reorder(0x4090_0000), 0x4090_0000);
        assert!(reorder_table_has_split_bytes(&table));
        let mut t = CollationSettings::default();
        t.copy_reordering_from(&s);
        assert!(t.has_reordering());
        t.copy_reordering_from(&CollationSettings::default());
        assert!(!t.has_reordering());
    }
}
