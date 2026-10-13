//! `com.ibm.icu.text.RBBISetBuilder`: the sets of the rules, split into
//! non-overlapping ranges of code points, ranges that belong to exactly the
//! same sets grouped into one character category, and each category given a
//! number the state table is indexed by.
//!
//! Categories 0, 1 and 2 are reserved (0 is never used; 1 is the end of the
//! text, 2 its start); the others start at 3. Categories of the characters of
//! a set variable named `$dictionary` come last, from `dictCategoriesStart`.
//! Not ported: `mergeCategories`, which ICU4J's `optimizeTables` uses to merge
//! categories whose columns turn out equal -- it changes the table's size,
//! not what it matches.

use super::node::{LEAF_CHAR, OP_OR, VAR_REF};
use super::RuleBuilder;
use crate::icu4j::rbbi_data::CategoryMap;
use crate::icu4j::utf16;
use crate::IcuError;

/// `RBBISetBuilder.RangeDescriptor`.
#[derive(Debug, Clone)]
pub(super) struct Range {
    pub start: u32,
    pub end: u32,
    /// `fNum`: the category.
    pub num: i32,
    pub includes_dict: bool,
    pub first_in_group: bool,
    /// `fIncludesSets`: the `uset` nodes holding the range.
    pub sets: Vec<usize>,
}

/// `RBBISetBuilder`'s fields.
#[derive(Debug, Default)]
pub(super) struct SetBuilderState {
    pub ranges: Vec<Range>,
    pub group_count: i32,
    pub dict_categories_start: i32,
    pub saw_bof: bool,
}

// ARITH: (the whole impl) code points are at most 0x10ffff, category numbers
// are bounded by the number of ranges (at most twice the number of ranges of
// the rules' sets), and indexes walk the range list.
#[allow(clippy::arithmetic_side_effects)]
impl RuleBuilder {
    /// `RangeDescriptor.split(where)`.
    fn split_range(&mut self, at: usize, where_: u32) {
        let mut tail = self.sb.ranges[at].clone();
        tail.start = where_;
        self.sb.ranges[at].end = where_ - 1;
        self.sb.ranges.insert(at + 1, tail);
    }

    /// `RangeDescriptor.isDictionaryRange()`: whether a set of the range is
    /// the value of a variable named `dictionary`.
    fn is_dictionary_range(&self, range: usize) -> bool {
        self.sb.ranges[range].sets.iter().any(|&u| {
            let nodes = &self.arena.nodes;
            nodes[u]
                .parent
                .and_then(|set_ref| nodes[set_ref].parent)
                .is_some_and(|var| nodes[var].kind == VAR_REF && nodes[var].text == "dictionary")
        })
    }

    /// `addValToSet(usetNode, val)`: category `val` joins the characters of
    /// the set's node (an or of leaf nodes when it has several).
    fn add_val_to_set(&mut self, uset: usize, val: i32) {
        let leaf = self.arena.new_node(LEAF_CHAR);
        self.arena.nodes[leaf].val = val;
        match self.arena.nodes[uset].left {
            None => {
                self.arena.nodes[uset].left = Some(leaf);
                self.arena.nodes[leaf].parent = Some(uset);
            }
            Some(existing) => {
                let or = self.arena.new_node(OP_OR);
                self.arena.nodes[or].left = Some(existing);
                self.arena.nodes[or].right = Some(leaf);
                self.arena.nodes[existing].parent = Some(or);
                self.arena.nodes[leaf].parent = Some(or);
                self.arena.nodes[uset].left = Some(or);
                self.arena.nodes[or].parent = Some(uset);
            }
        }
    }

    /// `addValToSets(sets, val)`.
    fn add_val_to_sets(&mut self, sets: &[usize], val: i32) {
        for &u in sets {
            self.add_val_to_set(u, val);
        }
    }

    /// `buildRanges()`.
    pub(super) fn build_ranges(&mut self) -> Result<(), IcuError> {
        self.sb.ranges = vec![Range {
            start: 0,
            end: 0x10ffff,
            num: 0,
            includes_dict: false,
            first_in_group: false,
            sets: Vec::new(),
        }];
        // split the code space at every boundary of every set
        let usets = self.usets.clone();
        for &u in &usets {
            let set_ranges: Vec<(u32, u32)> = self.arena.nodes[u]
                .input_set
                .as_ref()
                .map(|s| s.ranges().to_vec())
                .unwrap_or_default();
            let mut i = 0;
            let mut rl = 0;
            while i < set_ranges.len() {
                let (begin, end) = set_ranges[i];
                while rl < self.sb.ranges.len() && self.sb.ranges[rl].end < begin {
                    rl += 1;
                }
                if rl >= self.sb.ranges.len() {
                    break;
                }
                if self.sb.ranges[rl].start < begin {
                    self.split_range(rl, begin);
                    continue;
                }
                if self.sb.ranges[rl].end > end {
                    self.split_range(rl, end + 1);
                }
                if !self.sb.ranges[rl].sets.contains(&u) {
                    self.sb.ranges[rl].sets.push(u);
                }
                if end == self.sb.ranges[rl].end {
                    i += 1;
                }
                rl += 1;
            }
        }
        // ranges in the same sets share a category
        let mut dict_group_count = 0;
        for ri in 0..self.sb.ranges.len() {
            let same = (0..ri).find(|&rj| self.sb.ranges[ri].sets == self.sb.ranges[rj].sets);
            if let Some(rj) = same {
                self.sb.ranges[ri].num = self.sb.ranges[rj].num;
                self.sb.ranges[ri].includes_dict = self.sb.ranges[rj].includes_dict;
            }
            if self.sb.ranges[ri].num == 0 {
                self.sb.ranges[ri].first_in_group = true;
                if self.is_dictionary_range(ri) {
                    dict_group_count += 1;
                    self.sb.ranges[ri].num = dict_group_count;
                    self.sb.ranges[ri].includes_dict = true;
                } else {
                    self.sb.group_count += 1;
                    let num = self.sb.group_count + 2;
                    self.sb.ranges[ri].num = num;
                    let sets = self.sb.ranges[ri].sets.clone();
                    self.add_val_to_sets(&sets, num);
                }
            }
        }
        // the dictionary categories follow the others
        self.sb.dict_categories_start = self.sb.group_count + 3;
        for ri in 0..self.sb.ranges.len() {
            if self.sb.ranges[ri].includes_dict {
                self.sb.ranges[ri].num += self.sb.dict_categories_start - 1;
                if self.sb.ranges[ri].first_in_group {
                    let (sets, num) = (self.sb.ranges[ri].sets.clone(), self.sb.ranges[ri].num);
                    self.add_val_to_sets(&sets, num);
                }
            }
        }
        self.sb.group_count += dict_group_count;
        // the strings "eof" and "bof" in a set are the end and start of text
        let (eof, bof) = (utf16::units("eof"), utf16::units("bof"));
        for &u in &usets {
            let (has_eof, has_bof) =
                self.arena.nodes[u]
                    .input_set
                    .as_ref()
                    .map_or((false, false), |s| {
                        (
                            s.strings().any(|t| t == eof.as_slice()),
                            s.strings().any(|t| t == bof.as_slice()),
                        )
                    });
            if has_eof {
                self.add_val_to_set(u, 1);
            }
            if has_bof {
                self.add_val_to_set(u, 2);
                self.sb.saw_bof = true;
            }
        }
        Ok(())
    }

    /// `getNumCharCategories()`.
    pub(super) fn num_char_categories(&self) -> i32 {
        self.sb.group_count + 3
    }

    /// `buildTrie()`: every range's category, as the lookup the break
    /// iterator reads.
    pub(super) fn category_map(&self) -> CategoryMap {
        CategoryMap::from_ranges(
            self.sb
                .ranges
                .iter()
                .map(|r| (r.start, r.end, r.num))
                .collect(),
        )
    }
}
