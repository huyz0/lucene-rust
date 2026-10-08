//! `com.ibm.icu.util.BytesTrie` and `com.ibm.icu.util.CharsTrie`: the
//! read-only string tries of ICU's word-break dictionaries (`.dict`).
//!
//! Only matching is ported -- `first`/`next` per byte or code unit (and
//! per code point for `CharsTrie`), `current`, `getValue` -- the operations
//! `BytesDictionaryMatcher`/`CharsDictionaryMatcher` use; iteration,
//! `getUniqueValue` and the builders are not. Node layouts are ICU's:
//! branch nodes (a split-branch binary search down to linear lists of at
//! most five units), linear-match nodes, and value nodes, with values and
//! jump deltas in 1-5 byte (or 1-3 unit) encodings. Rust-forced change:
//! every read past the trie's end reads 0 and a corrupt trie stops matching
//! instead of throwing `ArrayIndexOutOfBoundsException` (the data is ICU's
//! own, vendored).

/// `BytesTrie.Result` / `CharsTrie.Result` (`BytesTrie.Result` is shared in
/// ICU4J).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrieResult {
    /// `NO_MATCH`.
    NoMatch,
    /// `NO_VALUE`.
    NoValue,
    /// `FINAL_VALUE`.
    FinalValue,
    /// `INTERMEDIATE_VALUE`.
    IntermediateValue,
}

impl TrieResult {
    /// `hasValue()`.
    pub fn has_value(self) -> bool {
        matches!(self, TrieResult::FinalValue | TrieResult::IntermediateValue)
    }

    /// `hasNext()`.
    pub fn has_next(self) -> bool {
        matches!(self, TrieResult::NoValue | TrieResult::IntermediateValue)
    }
}

mod b {
    pub const MAX_BRANCH_LINEAR_SUB_NODE_LENGTH: i32 = 5;
    pub const MIN_LINEAR_MATCH: i32 = 0x10;
    pub const MIN_VALUE_LEAD: i32 = 0x20;
    pub const VALUE_IS_FINAL: i32 = 1;
    pub const MIN_ONE_BYTE_VALUE_LEAD: i32 = 0x10;
    pub const MIN_TWO_BYTE_VALUE_LEAD: i32 = 0x51;
    pub const MIN_THREE_BYTE_VALUE_LEAD: i32 = 0x6c;
    pub const FOUR_BYTE_VALUE_LEAD: i32 = 0x7e;
    pub const MIN_TWO_BYTE_DELTA_LEAD: i32 = 0xc0;
    pub const MIN_THREE_BYTE_DELTA_LEAD: i32 = 0xf0;
    pub const FOUR_BYTE_DELTA_LEAD: i32 = 0xfe;
}

/// A `BytesTrie` over a dictionary's bytes.
#[derive(Debug, Clone)]
pub struct BytesTrie<'a> {
    bytes: &'a [u8],
    root: i32,
    pos: i32,
    remaining_match_length: i32,
}

// ARITH: (the whole impl) positions are i32 offsets into a slice of at
// most a few megabytes; a position is only advanced by a few units or a
// decoded delta, and every read goes through `at`, which reads 0 outside
// the slice. Deltas are read as Java reads them (an `int`), so a corrupt
// one can only move the position somewhere that reads as 0s or stops.
#[allow(clippy::arithmetic_side_effects)]
impl<'a> BytesTrie<'a> {
    /// `new BytesTrie(trieBytes, offset)`.
    pub fn new(bytes: &'a [u8], offset: i32) -> Self {
        BytesTrie {
            bytes,
            root: offset,
            pos: offset,
            remaining_match_length: -1,
        }
    }

    #[inline]
    fn at(&self, pos: i32) -> i32 {
        usize::try_from(pos)
            .ok()
            .and_then(|p| self.bytes.get(p))
            .map_or(0, |&b| i32::from(b))
    }

    /// `at` as a signed byte (Java's `bytes[pos]`).
    #[inline]
    fn at_signed(&self, pos: i32) -> i32 {
        i32::from(self.at(pos) as u8 as i8)
    }

    fn value_result(node: i32) -> TrieResult {
        if node & b::VALUE_IS_FINAL != 0 {
            TrieResult::FinalValue
        } else {
            TrieResult::IntermediateValue
        }
    }

    /// `current()`.
    pub fn current(&self) -> TrieResult {
        if self.pos < 0 {
            return TrieResult::NoMatch;
        }
        let node = self.at(self.pos);
        if self.remaining_match_length < 0 && node >= b::MIN_VALUE_LEAD {
            Self::value_result(node)
        } else {
            TrieResult::NoValue
        }
    }

    /// `first(inByte)`.
    pub fn first(&mut self, mut in_byte: i32) -> TrieResult {
        self.remaining_match_length = -1;
        if in_byte < 0 {
            in_byte += 0x100;
        }
        self.next_impl(self.root, in_byte)
    }

    /// `next(inByte)`.
    pub fn next(&mut self, mut in_byte: i32) -> TrieResult {
        let mut pos = self.pos;
        if pos < 0 {
            return TrieResult::NoMatch;
        }
        if in_byte < 0 {
            in_byte += 0x100;
        }
        let mut length = self.remaining_match_length;
        if length >= 0 {
            let b = self.at(pos);
            pos += 1;
            if in_byte == b {
                length -= 1;
                self.remaining_match_length = length;
                self.pos = pos;
                let node = self.at(pos);
                return if length < 0 && node >= b::MIN_VALUE_LEAD {
                    Self::value_result(node)
                } else {
                    TrieResult::NoValue
                };
            }
            self.stop();
            return TrieResult::NoMatch;
        }
        self.next_impl(pos, in_byte)
    }

    /// `getValue()`.
    pub fn get_value(&self) -> i32 {
        let pos = self.pos;
        let lead = self.at(pos);
        self.read_value(pos + 1, lead >> 1)
    }

    fn stop(&mut self) {
        self.pos = -1;
    }

    fn read_value(&self, pos: i32, lead: i32) -> i32 {
        if lead < b::MIN_TWO_BYTE_VALUE_LEAD {
            lead - b::MIN_ONE_BYTE_VALUE_LEAD
        } else if lead < b::MIN_THREE_BYTE_VALUE_LEAD {
            ((lead - b::MIN_TWO_BYTE_VALUE_LEAD) << 8) | self.at(pos)
        } else if lead < b::FOUR_BYTE_VALUE_LEAD {
            ((lead - b::MIN_THREE_BYTE_VALUE_LEAD) << 16) | (self.at(pos) << 8) | self.at(pos + 1)
        } else if lead == b::FOUR_BYTE_VALUE_LEAD {
            (self.at(pos) << 16) | (self.at(pos + 1) << 8) | self.at(pos + 2)
        } else {
            (self.at_signed(pos) << 24)
                | (self.at(pos + 1) << 16)
                | (self.at(pos + 2) << 8)
                | self.at(pos + 3)
        }
    }

    fn skip_value_lead(pos: i32, lead: i32) -> i32 {
        if lead >= (b::MIN_TWO_BYTE_VALUE_LEAD << 1) {
            if lead < (b::MIN_THREE_BYTE_VALUE_LEAD << 1) {
                return pos + 1;
            } else if lead < (b::FOUR_BYTE_VALUE_LEAD << 1) {
                return pos + 2;
            } else {
                return pos + 3 + ((lead >> 1) & 1);
            }
        }
        pos
    }

    fn skip_value(&self, pos: i32) -> i32 {
        let lead = self.at(pos);
        Self::skip_value_lead(pos + 1, lead)
    }

    fn jump_by_delta(&self, mut pos: i32) -> i32 {
        let mut delta = self.at(pos);
        pos += 1;
        if delta < b::MIN_TWO_BYTE_DELTA_LEAD {
        } else if delta < b::MIN_THREE_BYTE_DELTA_LEAD {
            delta = ((delta - b::MIN_TWO_BYTE_DELTA_LEAD) << 8) | self.at(pos);
            pos += 1;
        } else if delta < b::FOUR_BYTE_DELTA_LEAD {
            delta = ((delta - b::MIN_THREE_BYTE_DELTA_LEAD) << 16)
                | (self.at(pos) << 8)
                | self.at(pos + 1);
            pos += 2;
        } else if delta == b::FOUR_BYTE_DELTA_LEAD {
            delta = (self.at(pos) << 16) | (self.at(pos + 1) << 8) | self.at(pos + 2);
            pos += 3;
        } else {
            delta = (self.at_signed(pos) << 24)
                | (self.at(pos + 1) << 16)
                | (self.at(pos + 2) << 8)
                | self.at(pos + 3);
            pos += 4;
        }
        pos.wrapping_add(delta)
    }

    fn skip_delta(&self, mut pos: i32) -> i32 {
        let delta = self.at(pos);
        pos += 1;
        if delta >= b::MIN_TWO_BYTE_DELTA_LEAD {
            if delta < b::MIN_THREE_BYTE_DELTA_LEAD {
                pos += 1;
            } else if delta < b::FOUR_BYTE_DELTA_LEAD {
                pos += 2;
            } else {
                pos += 3 + (delta & 1);
            }
        }
        pos
    }

    /// `branchNext(pos, length, inByte)`.
    fn branch_next(&mut self, mut pos: i32, mut length: i32, in_byte: i32) -> TrieResult {
        if length == 0 {
            length = self.at(pos);
            pos += 1;
        }
        length += 1;
        while length > b::MAX_BRANCH_LINEAR_SUB_NODE_LENGTH {
            let unit = self.at(pos);
            pos += 1;
            if in_byte < unit {
                length >>= 1;
                pos = self.jump_by_delta(pos);
            } else {
                length -= length >> 1;
                pos = self.skip_delta(pos);
            }
            if pos < 0 || pos as usize >= self.bytes.len() {
                self.stop();
                return TrieResult::NoMatch;
            }
        }
        loop {
            let unit = self.at(pos);
            pos += 1;
            if in_byte == unit {
                let mut node = self.at(pos);
                let result;
                if node & b::VALUE_IS_FINAL != 0 {
                    result = TrieResult::FinalValue;
                } else {
                    pos += 1;
                    node >>= 1;
                    let delta;
                    if node < b::MIN_TWO_BYTE_VALUE_LEAD {
                        delta = node - b::MIN_ONE_BYTE_VALUE_LEAD;
                    } else if node < b::MIN_THREE_BYTE_VALUE_LEAD {
                        delta = ((node - b::MIN_TWO_BYTE_VALUE_LEAD) << 8) | self.at(pos);
                        pos += 1;
                    } else if node < b::FOUR_BYTE_VALUE_LEAD {
                        delta = ((node - b::MIN_THREE_BYTE_VALUE_LEAD) << 16)
                            | (self.at(pos) << 8)
                            | self.at(pos + 1);
                        pos += 2;
                    } else if node == b::FOUR_BYTE_VALUE_LEAD {
                        delta = (self.at(pos) << 16) | (self.at(pos + 1) << 8) | self.at(pos + 2);
                        pos += 3;
                    } else {
                        delta = (self.at_signed(pos) << 24)
                            | (self.at(pos + 1) << 16)
                            | (self.at(pos + 2) << 8)
                            | self.at(pos + 3);
                        pos += 4;
                    }
                    pos = pos.wrapping_add(delta);
                    node = self.at(pos);
                    result = if node >= b::MIN_VALUE_LEAD {
                        Self::value_result(node)
                    } else {
                        TrieResult::NoValue
                    };
                }
                self.pos = pos;
                return result;
            }
            length -= 1;
            pos = self.skip_value(pos);
            if length <= 1 {
                break;
            }
        }
        let unit = self.at(pos);
        pos += 1;
        if in_byte == unit {
            self.pos = pos;
            let node = self.at(pos);
            if node >= b::MIN_VALUE_LEAD {
                Self::value_result(node)
            } else {
                TrieResult::NoValue
            }
        } else {
            self.stop();
            TrieResult::NoMatch
        }
    }

    /// `nextImpl(pos, inByte)`.
    fn next_impl(&mut self, mut pos: i32, in_byte: i32) -> TrieResult {
        // A value node is followed by a match node, so a well-formed trie
        // takes at most two steps; the bound only stops corrupt data.
        for _ in 0..64 {
            let node = self.at(pos);
            pos += 1;
            if node < b::MIN_LINEAR_MATCH {
                return self.branch_next(pos, node, in_byte);
            } else if node < b::MIN_VALUE_LEAD {
                let mut length = node - b::MIN_LINEAR_MATCH;
                let unit = self.at(pos);
                pos += 1;
                if in_byte == unit {
                    length -= 1;
                    self.remaining_match_length = length;
                    self.pos = pos;
                    let node = self.at(pos);
                    return if length < 0 && node >= b::MIN_VALUE_LEAD {
                        Self::value_result(node)
                    } else {
                        TrieResult::NoValue
                    };
                }
                break;
            } else if node & b::VALUE_IS_FINAL != 0 {
                break;
            } else {
                pos = Self::skip_value_lead(pos, node);
            }
        }
        self.stop();
        TrieResult::NoMatch
    }
}

mod c {
    pub const MAX_BRANCH_LINEAR_SUB_NODE_LENGTH: i32 = 5;
    pub const MIN_LINEAR_MATCH: i32 = 0x30;
    pub const MIN_VALUE_LEAD: i32 = 0x40;
    pub const NODE_TYPE_MASK: i32 = 0x3f;
    pub const VALUE_IS_FINAL: i32 = 0x8000;
    pub const MIN_TWO_UNIT_VALUE_LEAD: i32 = 0x4000;
    pub const THREE_UNIT_VALUE_LEAD: i32 = 0x7fff;
    pub const MIN_TWO_UNIT_NODE_VALUE_LEAD: i32 = 0x4040;
    pub const THREE_UNIT_NODE_VALUE_LEAD: i32 = 0x7fc0;
    pub const MIN_TWO_UNIT_DELTA_LEAD: i32 = 0xfc00;
    pub const THREE_UNIT_DELTA_LEAD: i32 = 0xffff;
}

/// `CharsTrie.State`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CharsTrieState {
    pos: i32,
    remaining_match_length: i32,
}

/// A `CharsTrie` over a dictionary's UTF-16 units.
#[derive(Debug, Clone)]
pub struct CharsTrie<'a> {
    chars: &'a [u16],
    root: i32,
    pos: i32,
    remaining_match_length: i32,
}

// ARITH: (the whole impl) as for `BytesTrie`.
#[allow(clippy::arithmetic_side_effects)]
impl<'a> CharsTrie<'a> {
    /// `new CharsTrie(trieChars, offset)`.
    pub fn new(chars: &'a [u16], offset: i32) -> Self {
        CharsTrie {
            chars,
            root: offset,
            pos: offset,
            remaining_match_length: -1,
        }
    }

    #[inline]
    fn at(&self, pos: i32) -> i32 {
        usize::try_from(pos)
            .ok()
            .and_then(|p| self.chars.get(p))
            .map_or(0, |&u| i32::from(u))
    }

    fn value_result(node: i32) -> TrieResult {
        if node >> 15 != 0 {
            TrieResult::FinalValue
        } else {
            TrieResult::IntermediateValue
        }
    }

    /// `current()`.
    pub fn current(&self) -> TrieResult {
        if self.pos < 0 {
            return TrieResult::NoMatch;
        }
        let node = self.at(self.pos);
        if self.remaining_match_length < 0 && node >= c::MIN_VALUE_LEAD {
            Self::value_result(node)
        } else {
            TrieResult::NoValue
        }
    }

    /// `first(inUnit)`.
    pub fn first(&mut self, in_unit: i32) -> TrieResult {
        self.remaining_match_length = -1;
        self.next_impl(self.root, in_unit)
    }

    /// `reset()`.
    pub fn reset(&mut self) {
        self.pos = self.root;
        self.remaining_match_length = -1;
    }

    /// `saveState(state)`: the position within this trie.
    pub fn save_state(&self) -> CharsTrieState {
        CharsTrieState {
            pos: self.pos,
            remaining_match_length: self.remaining_match_length,
        }
    }

    /// `resetToState(state)` (a state saved from this trie).
    pub fn reset_to_state(&mut self, state: CharsTrieState) {
        self.pos = state.pos;
        self.remaining_match_length = state.remaining_match_length;
    }

    /// `firstForCodePoint(cp)`.
    pub fn first_for_code_point(&mut self, cp: i32) -> TrieResult {
        if cp <= 0xffff {
            self.first(cp)
        } else {
            let (lead, trail) = crate::icu4j::utf16::surrogates(cp);
            if self.first(i32::from(lead)).has_next() {
                self.next(i32::from(trail))
            } else {
                TrieResult::NoMatch
            }
        }
    }

    /// `next(inUnit)`.
    pub fn next(&mut self, in_unit: i32) -> TrieResult {
        let mut pos = self.pos;
        if pos < 0 {
            return TrieResult::NoMatch;
        }
        let mut length = self.remaining_match_length;
        if length >= 0 {
            let u = self.at(pos);
            pos += 1;
            if in_unit == u {
                length -= 1;
                self.remaining_match_length = length;
                self.pos = pos;
                let node = self.at(pos);
                return if length < 0 && node >= c::MIN_VALUE_LEAD {
                    Self::value_result(node)
                } else {
                    TrieResult::NoValue
                };
            }
            self.stop();
            return TrieResult::NoMatch;
        }
        self.next_impl(pos, in_unit)
    }

    /// `nextForCodePoint(cp)`.
    pub fn next_for_code_point(&mut self, cp: i32) -> TrieResult {
        if cp <= 0xffff {
            self.next(cp)
        } else {
            let (lead, trail) = crate::icu4j::utf16::surrogates(cp);
            if self.next(i32::from(lead)).has_next() {
                self.next(i32::from(trail))
            } else {
                TrieResult::NoMatch
            }
        }
    }

    /// `getValue()`.
    pub fn get_value(&self) -> i32 {
        let pos = self.pos;
        let lead = self.at(pos);
        if lead & c::VALUE_IS_FINAL != 0 {
            self.read_value(pos + 1, lead & 0x7fff)
        } else {
            self.read_node_value(pos + 1, lead)
        }
    }

    fn stop(&mut self) {
        self.pos = -1;
    }

    fn read_value(&self, pos: i32, lead: i32) -> i32 {
        if lead < c::MIN_TWO_UNIT_VALUE_LEAD {
            lead
        } else if lead < c::THREE_UNIT_VALUE_LEAD {
            ((lead - c::MIN_TWO_UNIT_VALUE_LEAD) << 16) | self.at(pos)
        } else {
            (self.at(pos) << 16) | self.at(pos + 1)
        }
    }

    fn skip_value_lead(pos: i32, lead: i32) -> i32 {
        if lead >= c::MIN_TWO_UNIT_VALUE_LEAD {
            if lead < c::THREE_UNIT_VALUE_LEAD {
                return pos + 1;
            }
            return pos + 2;
        }
        pos
    }

    fn skip_value(&self, pos: i32) -> i32 {
        let lead = self.at(pos);
        Self::skip_value_lead(pos + 1, lead & 0x7fff)
    }

    fn read_node_value(&self, pos: i32, lead: i32) -> i32 {
        if lead < c::MIN_TWO_UNIT_NODE_VALUE_LEAD {
            (lead >> 6) - 1
        } else if lead < c::THREE_UNIT_NODE_VALUE_LEAD {
            (((lead & 0x7fc0) - c::MIN_TWO_UNIT_NODE_VALUE_LEAD) << 10) | self.at(pos)
        } else {
            (self.at(pos) << 16) | self.at(pos + 1)
        }
    }

    fn skip_node_value(pos: i32, lead: i32) -> i32 {
        if lead >= c::MIN_TWO_UNIT_NODE_VALUE_LEAD {
            if lead < c::THREE_UNIT_NODE_VALUE_LEAD {
                return pos + 1;
            }
            return pos + 2;
        }
        pos
    }

    fn jump_by_delta(&self, mut pos: i32) -> i32 {
        let mut delta = self.at(pos);
        pos += 1;
        if delta >= c::MIN_TWO_UNIT_DELTA_LEAD {
            if delta == c::THREE_UNIT_DELTA_LEAD {
                delta = (self.at(pos) << 16) | self.at(pos + 1);
                pos += 2;
            } else {
                delta = ((delta - c::MIN_TWO_UNIT_DELTA_LEAD) << 16) | self.at(pos);
                pos += 1;
            }
        }
        pos.wrapping_add(delta)
    }

    fn skip_delta(&self, mut pos: i32) -> i32 {
        let delta = self.at(pos);
        pos += 1;
        if delta >= c::MIN_TWO_UNIT_DELTA_LEAD {
            if delta == c::THREE_UNIT_DELTA_LEAD {
                pos += 2;
            } else {
                pos += 1;
            }
        }
        pos
    }

    /// `branchNext(pos, length, inUnit)`.
    fn branch_next(&mut self, mut pos: i32, mut length: i32, in_unit: i32) -> TrieResult {
        if length == 0 {
            length = self.at(pos);
            pos += 1;
        }
        length += 1;
        while length > c::MAX_BRANCH_LINEAR_SUB_NODE_LENGTH {
            let unit = self.at(pos);
            pos += 1;
            if in_unit < unit {
                length >>= 1;
                pos = self.jump_by_delta(pos);
            } else {
                length -= length >> 1;
                pos = self.skip_delta(pos);
            }
            if pos < 0 || pos as usize >= self.chars.len() {
                self.stop();
                return TrieResult::NoMatch;
            }
        }
        loop {
            let unit = self.at(pos);
            pos += 1;
            if in_unit == unit {
                let mut node = self.at(pos);
                let result;
                if node & c::VALUE_IS_FINAL != 0 {
                    result = TrieResult::FinalValue;
                } else {
                    pos += 1;
                    let delta;
                    if node < c::MIN_TWO_UNIT_VALUE_LEAD {
                        delta = node;
                    } else if node < c::THREE_UNIT_VALUE_LEAD {
                        delta = ((node - c::MIN_TWO_UNIT_VALUE_LEAD) << 16) | self.at(pos);
                        pos += 1;
                    } else {
                        delta = (self.at(pos) << 16) | self.at(pos + 1);
                        pos += 2;
                    }
                    pos = pos.wrapping_add(delta);
                    node = self.at(pos);
                    result = if node >= c::MIN_VALUE_LEAD {
                        Self::value_result(node)
                    } else {
                        TrieResult::NoValue
                    };
                }
                self.pos = pos;
                return result;
            }
            length -= 1;
            pos = self.skip_value(pos);
            if length <= 1 {
                break;
            }
        }
        let unit = self.at(pos);
        pos += 1;
        if in_unit == unit {
            self.pos = pos;
            let node = self.at(pos);
            if node >= c::MIN_VALUE_LEAD {
                Self::value_result(node)
            } else {
                TrieResult::NoValue
            }
        } else {
            self.stop();
            TrieResult::NoMatch
        }
    }

    /// `nextImpl(pos, inUnit)`.
    fn next_impl(&mut self, mut pos: i32, in_unit: i32) -> TrieResult {
        let mut node = self.at(pos);
        pos += 1;
        // As in `BytesTrie::next_impl`, a bound only corrupt data reaches.
        for _ in 0..64 {
            if node < c::MIN_LINEAR_MATCH {
                return self.branch_next(pos, node, in_unit);
            } else if node < c::MIN_VALUE_LEAD {
                let mut length = node - c::MIN_LINEAR_MATCH;
                let unit = self.at(pos);
                pos += 1;
                if in_unit == unit {
                    length -= 1;
                    self.remaining_match_length = length;
                    self.pos = pos;
                    let n = self.at(pos);
                    return if length < 0 && n >= c::MIN_VALUE_LEAD {
                        Self::value_result(n)
                    } else {
                        TrieResult::NoValue
                    };
                }
                break;
            } else if node & c::VALUE_IS_FINAL != 0 {
                break;
            } else {
                pos = Self::skip_node_value(pos, node);
                node &= c::NODE_TYPE_MASK;
            }
        }
        self.stop();
        TrieResult::NoMatch
    }
}
