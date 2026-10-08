//! `com.ibm.icu.impl.RBBIDataWrapper`: compiled break rules (`.brk`, data
//! format `Brk `, format version 6).
//!
//! Layout after the ICU header: a 20-`int32` header -- magic `0xb1a0`,
//! format version, total length, character category count, then offset and
//! length pairs of the forward table, reverse table, category trie, rule
//! source (UTF-8) and rule status table, then six reserved words. A state
//! table is five `int32`s (state count, row length in bytes, first
//! dictionary category, look-ahead result count, flags: hard break, BOF
//! required, 8-bit rows) and its rows, each `accepting, lookahead, tags
//! index` and one next state per category, in 8- or 16-bit units. The
//! category trie is a `FAST` [`CodePointTrie`].

use crate::icu4j::binary::{read_header, ByteReader};
use crate::icu4j::code_point_trie::{CodePointTrie, TrieType};
use crate::IcuError;

/// `RBBIDataWrapper.ACCEPTING` etc.: the fixed columns of a row.
pub const ACCEPTING: usize = 0;
pub const LOOKAHEAD: usize = 1;
pub const TAGSIDX: usize = 2;
pub const NEXTSTATES: usize = 3;
/// `ACCEPTING_UNCONDITIONAL`.
pub const ACCEPTING_UNCONDITIONAL: u16 = 1;
/// `RBBI_BOF_REQUIRED`.
pub const RBBI_BOF_REQUIRED: i32 = 2;
/// `RBBI_8BITS_ROWS`.
pub const RBBI_8BITS_ROWS: i32 = 4;

const DATA_FORMAT: u32 = 0x4272_6b20; // "Brk "
const FORMAT_VERSION: [u8; 4] = [6, 0, 0, 0];
const DH_SIZE: i32 = 20;

/// `RBBIDataWrapper.RBBIStateTable`.
#[derive(Debug, Clone)]
pub struct StateTable {
    pub num_states: i32,
    pub row_len: i32,
    pub dict_categories_start: i32,
    pub look_ahead_results_size: i32,
    pub flags: i32,
    pub table: Vec<u16>,
}

impl StateTable {
    /// `RBBIStateTable.get(bytes, length)`.
    fn get(r: &mut ByteReader<'_>, length: i32) -> Result<Option<StateTable>, IcuError> {
        if length == 0 {
            return Ok(None);
        }
        if length < 20 {
            return Err(IcuError::new("Invalid RBBI state table length."));
        }
        let num_states = r.i32()?;
        let row_len = r.i32()?;
        let dict_categories_start = r.i32()?;
        let look_ahead_results_size = r.i32()?;
        let flags = r.i32()?;
        let len = usize::try_from(length.saturating_sub(20))
            .map_err(|_| IcuError::new("Invalid RBBI state table length."))?;
        let table = if flags & RBBI_8BITS_ROWS != 0 {
            let t = r.take(len)?.iter().map(|&b| u16::from(b)).collect();
            r.skip(len & 1)?;
            t
        } else {
            let t = r.u16s(len / 2)?;
            r.skip(len & 1)?;
            t
        };
        Ok(Some(StateTable {
            num_states,
            row_len,
            dict_categories_start,
            look_ahead_results_size,
            flags,
            table,
        }))
    }

    /// A table cell (0 past the end: a corrupt table stops the machine).
    #[inline]
    pub fn at(&self, i: usize) -> u16 {
        self.table.get(i).copied().unwrap_or(0)
    }
}

/// `RBBIDataWrapper`.
#[derive(Debug, Clone)]
pub struct RbbiData {
    /// `fHeader.fCatCount`.
    pub cat_count: i32,
    /// `fFTable`.
    pub ftable: StateTable,
    /// `fRTable`.
    pub rtable: Option<StateTable>,
    /// `fTrie`.
    pub trie: CodePointTrie,
    /// `fRuleSource`.
    pub rule_source: String,
    /// `fStatusTable`.
    pub status_table: Vec<i32>,
}

fn corrupt() -> IcuError {
    IcuError::new("Break iterator Rule data corrupt")
}

impl RbbiData {
    /// `RBBIDataWrapper.get(bytes)`.
    pub fn get(bytes: &[u8]) -> Result<RbbiData, IcuError> {
        let (r0, _) = read_header(bytes, DATA_FORMAT, |v| *v == FORMAT_VERSION)?;
        let base = r0.position();
        let mut r = r0;
        let magic = r.i32()?;
        let format_version = r.take(4)?;
        let ok = magic == 0xb1a0 && format_version == FORMAT_VERSION;
        let length = r.i32()?;
        let cat_count = r.i32()?;
        let f_table = r.i32()?;
        let f_table_len = r.i32()?;
        let r_table = r.i32()?;
        let r_table_len = r.i32()?;
        let trie_off = r.i32()?;
        let _trie_len = r.i32()?;
        let rule_source = r.i32()?;
        let rule_source_len = r.i32()?;
        let status_table = r.i32()?;
        let status_table_len = r.i32()?;
        r.skip(6 * 4)?;
        if !ok {
            return Err(IcuError::new(
                "Break Iterator Rule Data Magic Number Incorrect, or unsupported data version.",
            ));
        }
        let to = |off: i32| -> Result<usize, IcuError> {
            usize::try_from(off)
                .ok()
                .and_then(|o| base.checked_add(o))
                .ok_or_else(corrupt)
        };
        let pos = DH_SIZE * 4;
        if f_table < pos || f_table > length {
            return Err(corrupt());
        }
        r.seek(to(f_table)?)?;
        let ftable = StateTable::get(&mut r, f_table_len)?.ok_or_else(corrupt)?;
        r.seek(to(r_table)?)?;
        let rtable = StateTable::get(&mut r, r_table_len)?;
        r.seek(to(trie_off)?)?;
        let trie = CodePointTrie::from_binary(Some(TrieType::Fast), None, &mut r)?;
        if trie_off > status_table {
            return Err(corrupt());
        }
        r.seek(to(status_table)?)?;
        let n = usize::try_from(status_table_len / 4).map_err(|_| corrupt())?;
        let status = r.i32s(n)?;
        if status_table.saturating_add(status_table_len) > rule_source {
            return Err(corrupt());
        }
        r.seek(to(rule_source)?)?;
        let src_len = usize::try_from(rule_source_len).map_err(|_| corrupt())?;
        let src = String::from_utf8_lossy(r.take(src_len)?).into_owned();
        Ok(RbbiData {
            cat_count,
            ftable,
            rtable,
            trie,
            rule_source: src,
            status_table: status,
        })
    }

    /// `getRowIndex(state)`.
    #[inline]
    pub fn row_index(&self, state: u16) -> usize {
        usize::from(state).saturating_mul(
            usize::try_from(self.cat_count)
                .unwrap_or(0)
                .saturating_add(NEXTSTATES),
        )
    }

    /// A status table entry (0 past the end).
    #[inline]
    pub fn status(&self, i: i32) -> i32 {
        usize::try_from(i)
            .ok()
            .and_then(|i| self.status_table.get(i))
            .copied()
            .unwrap_or(0)
    }
}
