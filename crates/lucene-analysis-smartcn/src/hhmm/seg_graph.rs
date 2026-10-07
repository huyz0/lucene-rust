//! `org.apache.lucene.analysis.cn.smart.hhmm.SegGraph`: every candidate
//! segment of a sentence, grouped by start offset.
//!
//! Java keys the groups by start offset in an `IntObjectHashMap` and walks
//! the keys upward from -1; the keys are exactly the offsets -1..=length, so
//! here they index a vector (slot `start + 1`), which walks the same way.

use super::seg_token::SegToken;

/// `SegGraph`.
#[derive(Debug, Default)]
pub struct SegGraph {
    /// `tokenListTable`, by `start + 1`.
    token_list_table: Vec<Vec<SegToken>>,
    /// `maxStart`.
    max_start: i32,
}

/// The slot of a start offset (`start >= -1`).
fn slot(start: i32) -> Option<usize> {
    usize::try_from(start.checked_add(1)?).ok()
}

impl SegGraph {
    pub fn new() -> Self {
        SegGraph {
            token_list_table: Vec::new(),
            max_start: -1,
        }
    }

    /// `isStartExist(s)`.
    pub fn is_start_exist(&self, s: i32) -> bool {
        self.get_start_list(s).is_some()
    }

    /// `getStartList(s)`: the tokens starting at `s`.
    pub fn get_start_list(&self, s: i32) -> Option<&[SegToken]> {
        let list = self.token_list_table.get(slot(s)?)?;
        (!list.is_empty()).then_some(list.as_slice())
    }

    /// `getMaxStart()`.
    pub fn get_max_start(&self) -> i32 {
        self.max_start
    }

    /// `makeIndex()`: numbers every token by start offset, then insertion
    /// order ([`Self::into_token_list`] returns them in that order).
    pub fn make_index(&mut self) {
        let mut index = 0;
        for list in &mut self.token_list_table {
            for st in list.iter_mut() {
                st.index = index;
                // ARITH: at most one token per (start, end) pair of a
                // sentence of at most 1,024 units.
                #[allow(clippy::arithmetic_side_effects)]
                {
                    index += 1;
                }
            }
        }
    }

    /// The tokens in [`Self::make_index`]'s order, taken.
    pub fn into_token_list(self) -> Vec<SegToken> {
        self.token_list_table.into_iter().flatten().collect()
    }

    /// `addToken(token)` (`token.start_offset >= -1`).
    pub fn add_token(&mut self, token: SegToken) {
        let s = token.start_offset;
        let Some(at) = slot(s) else {
            return;
        };
        if self.token_list_table.len() <= at {
            self.token_list_table
                .resize_with(at.saturating_add(1), Vec::new);
        }
        self.token_list_table[at].push(token);
        if s > self.max_start {
            self.max_start = s;
        }
    }

    /// `toTokenList()`: every token, by start offset.
    pub fn to_token_list(&self) -> Vec<&SegToken> {
        self.token_list_table.iter().flatten().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn groups_by_start_and_numbers_in_order() {
        let mut g = SegGraph::new();
        assert_eq!(g.get_max_start(), -1);
        assert!(!g.is_start_exist(0));
        g.add_token(SegToken::new(&[1], 2, 3, 0, 0));
        g.add_token(SegToken::new(&[2], -1, 0, 0, 0));
        g.add_token(SegToken::new(&[3], 2, 4, 0, 0));
        g.add_token(SegToken::new(&[4], -7, 4, 0, 0)); // ignored: no such start
        assert_eq!(g.get_max_start(), 2);
        assert!(g.is_start_exist(-1));
        assert!(!g.is_start_exist(0));
        assert!(!g.is_start_exist(-2));
        assert_eq!(g.get_start_list(2).unwrap().len(), 2);
        g.make_index();
        assert_eq!(g.get_start_list(2).unwrap()[1].index, 2);
        assert_eq!(g.to_token_list().len(), 3);
        let idx = g.into_token_list();
        let order: Vec<(u16, i32)> = idx.iter().map(|t| (t.char_array[0], t.index)).collect();
        assert_eq!(order, [(2, 0), (1, 1), (3, 2)]);
    }
}
