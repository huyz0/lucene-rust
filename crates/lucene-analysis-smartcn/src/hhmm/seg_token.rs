//! `org.apache.lucene.analysis.cn.smart.hhmm.SegToken`: a segment of a
//! sentence -- its text (or, for a run of letters or digits, the
//! dictionary's placeholder word), offsets, type and frequency.

/// `SegToken`. Java's `equals`/`hashCode` (all fields) are the derived
/// ones.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SegToken {
    /// `charArray`.
    pub char_array: Vec<u16>,
    /// `startOffset`: -1 for the sentence-begin token.
    pub start_offset: i32,
    /// `endOffset`.
    pub end_offset: i32,
    /// `wordType` ([`crate::utility::word_type`]).
    pub word_type: i32,
    /// `weight`: the word's frequency.
    pub weight: i32,
    /// `index`: its number in [`super::SegGraph::make_index`]'s order.
    pub index: i32,
}

impl SegToken {
    /// `new SegToken(idArray, start, end, wordType, weight)`.
    pub fn new(char_array: &[u16], start: i32, end: i32, word_type: i32, weight: i32) -> Self {
        SegToken {
            char_array: char_array.to_vec(),
            start_offset: start,
            end_offset: end,
            word_type,
            weight,
            index: 0,
        }
    }
}
