//! `com.ibm.icu.text.RBBITableBuilder`, the forward table: the rules' syntax
//! tree made into a deterministic state machine -- nullable, first, last and
//! follow positions of every node, the subset construction over the
//! character categories, and the accepting, look-ahead and rule-status
//! information of each state -- exported as the table `RbbiData` reads.
//!
//! Not ported: the safe reverse table, which only backward iteration reads,
//! and `optimizeTables` (merging equal columns and states), which changes the
//! table's size and not what it matches.

use std::collections::{BTreeSet, HashMap};

use super::node::{
    END_MARK, LEAF_CHAR, LOOK_AHEAD, OP_CAT, OP_OR, OP_PLUS, OP_QUESTION, OP_STAR,
    RECURSIVE_DEPTH_LIMIT, SET_REF, TAG,
};
use super::RuleBuilder;
use crate::icu4j::rbbi_data::{
    StateTable, ACCEPTING_UNCONDITIONAL, NEXTSTATES, RBBI_8BITS_ROWS, RBBI_BOF_REQUIRED,
};
use crate::IcuError;

/// `RBBI_LOOKAHEAD_HARD_BREAK`.
const RBBI_LOOKAHEAD_HARD_BREAK: i32 = 1;

/// `RBBITableBuilder.RBBIStateDescriptor`.
#[derive(Debug)]
pub(super) struct DState {
    marked: bool,
    accepting: i32,
    look_ahead: i32,
    tag_vals: BTreeSet<i32>,
    tags_idx: i32,
    positions: BTreeSet<usize>,
    /// `fDtran`: the next state on each character category.
    dtran: Vec<i32>,
}

impl DState {
    fn new(last_input_symbol: usize) -> DState {
        DState {
            marked: false,
            accepting: 0,
            look_ahead: 0,
            tag_vals: BTreeSet::new(),
            tags_idx: 0,
            positions: BTreeSet::new(),
            dtran: vec![0; last_input_symbol.saturating_add(1)],
        }
    }
}

/// The table builder's state.
#[derive(Debug, Default)]
pub(super) struct TableBuilderState {
    pub dstates: Vec<DState>,
    pub look_ahead_rule_map: Vec<i32>,
    pub la_slots_in_use: i32,
    /// `RBBIRuleBuilder.fStatusSets`: a state's set of rule-status values to
    /// its index in the status table.
    pub status_sets: HashMap<BTreeSet<i32>, i32>,
}

fn too_long() -> IcuError {
    IcuError::illegal_argument("The input is too long")
}

fn internal() -> IcuError {
    IcuError::illegal_argument("Error 66049 in the break rule table builder")
}

// ARITH: (the whole impl) state, category and rule numbers are bounded by the
// rules' size, state numbers are bounded to 16 bits when exported, and the
// recursions carry a depth limit.
#[allow(clippy::arithmetic_side_effects)]
impl RuleBuilder {
    /// `buildForwardTable()`, then the exported table.
    pub(super) fn build_forward_table(&mut self) -> Result<StateTable, IcuError> {
        let Some(root0) = self.tree_roots[0] else {
            return Err(internal());
        };
        let mut root = self.arena.flatten_variables(root0, 0)?;
        if self.sb.saw_bof {
            // the start of the text is a character that precedes the rules
            let cat = self.arena.new_node(OP_CAT);
            let bof = self.arena.new_node(LEAF_CHAR);
            self.arena.nodes[cat].left = Some(bof);
            self.arena.nodes[cat].right = Some(root);
            self.arena.nodes[bof].parent = Some(cat);
            self.arena.nodes[bof].val = 2;
            root = cat;
        }
        // one end marker follows every rule
        let cat = self.arena.new_node(OP_CAT);
        self.arena.nodes[cat].left = Some(root);
        self.arena.nodes[root].parent = Some(cat);
        let end_mark = self.arena.new_node(END_MARK);
        self.arena.nodes[cat].right = Some(end_mark);
        self.arena.nodes[end_mark].parent = Some(cat);
        root = cat;
        self.tree_roots[0] = Some(root);
        self.arena.flatten_sets(root)?;

        self.calc_nullable(Some(root), 0)?;
        self.calc_first_pos(Some(root), 0)?;
        self.calc_last_pos(Some(root), 0)?;
        self.calc_follow_pos(Some(root), 0)?;
        if self.chain_rules {
            self.calc_chained_follow_pos(root, end_mark)?;
        }
        if self.sb.saw_bof {
            self.bof_fixup(root)?;
        }
        self.build_state_table(root)?;
        self.map_look_ahead_rules()?;
        self.flag_accepting_states(root)?;
        self.flag_look_ahead_states(root)?;
        self.flag_tagged_states(root)?;
        self.merge_rule_status_vals();
        Ok(self.export_table())
    }

    fn calc_nullable(&mut self, n: Option<usize>, depth: usize) -> Result<(), IcuError> {
        let Some(n) = n else { return Ok(()) };
        if depth > RECURSIVE_DEPTH_LIMIT {
            return Err(too_long());
        }
        let kind = self.arena.nodes[n].kind;
        if kind == SET_REF || kind == END_MARK {
            self.arena.nodes[n].nullable = false;
            return Ok(());
        }
        if kind == LOOK_AHEAD || kind == TAG {
            self.arena.nodes[n].nullable = true;
            return Ok(());
        }
        let (l, r) = (self.arena.nodes[n].left, self.arena.nodes[n].right);
        self.calc_nullable(l, depth + 1)?;
        self.calc_nullable(r, depth + 1)?;
        let nullable_of = |a: &Self, c: Option<usize>| c.is_some_and(|c| a.arena.nodes[c].nullable);
        self.arena.nodes[n].nullable = match kind {
            OP_OR => nullable_of(self, l) || nullable_of(self, r),
            OP_CAT => nullable_of(self, l) && nullable_of(self, r),
            OP_STAR | OP_QUESTION => true,
            _ => false,
        };
        Ok(())
    }

    fn calc_first_pos(&mut self, n: Option<usize>, depth: usize) -> Result<(), IcuError> {
        let Some(n) = n else { return Ok(()) };
        if depth > RECURSIVE_DEPTH_LIMIT {
            return Err(too_long());
        }
        let kind = self.arena.nodes[n].kind;
        if matches!(kind, LEAF_CHAR | END_MARK | LOOK_AHEAD | TAG) {
            self.arena.nodes[n].first_pos_set.insert(n);
            return Ok(());
        }
        let (l, r) = (self.arena.nodes[n].left, self.arena.nodes[n].right);
        self.calc_first_pos(l, depth + 1)?;
        self.calc_first_pos(r, depth + 1)?;
        let set_of = |a: &Self, c: Option<usize>| {
            c.map(|c| a.arena.nodes[c].first_pos_set.clone())
                .unwrap_or_default()
        };
        match kind {
            OP_OR => {
                let mut s = set_of(self, l);
                s.extend(set_of(self, r));
                self.arena.nodes[n].first_pos_set.extend(s);
            }
            OP_CAT => {
                let mut s = set_of(self, l);
                if l.is_some_and(|l| self.arena.nodes[l].nullable) {
                    s.extend(set_of(self, r));
                }
                self.arena.nodes[n].first_pos_set.extend(s);
            }
            OP_STAR | OP_QUESTION | OP_PLUS => {
                let s = set_of(self, l);
                self.arena.nodes[n].first_pos_set.extend(s);
            }
            _ => {}
        }
        Ok(())
    }

    fn calc_last_pos(&mut self, n: Option<usize>, depth: usize) -> Result<(), IcuError> {
        let Some(n) = n else { return Ok(()) };
        if depth > RECURSIVE_DEPTH_LIMIT {
            return Err(too_long());
        }
        let kind = self.arena.nodes[n].kind;
        if matches!(kind, LEAF_CHAR | END_MARK | LOOK_AHEAD | TAG) {
            self.arena.nodes[n].last_pos_set.insert(n);
            return Ok(());
        }
        let (l, r) = (self.arena.nodes[n].left, self.arena.nodes[n].right);
        self.calc_last_pos(l, depth + 1)?;
        self.calc_last_pos(r, depth + 1)?;
        let set_of = |a: &Self, c: Option<usize>| {
            c.map(|c| a.arena.nodes[c].last_pos_set.clone())
                .unwrap_or_default()
        };
        match kind {
            OP_OR => {
                let mut s = set_of(self, l);
                s.extend(set_of(self, r));
                self.arena.nodes[n].last_pos_set.extend(s);
            }
            OP_CAT => {
                let mut s = set_of(self, r);
                if r.is_some_and(|r| self.arena.nodes[r].nullable) {
                    s.extend(set_of(self, l));
                }
                self.arena.nodes[n].last_pos_set.extend(s);
            }
            OP_STAR | OP_QUESTION | OP_PLUS => {
                let s = set_of(self, l);
                self.arena.nodes[n].last_pos_set.extend(s);
            }
            _ => {}
        }
        Ok(())
    }

    fn calc_follow_pos(&mut self, n: Option<usize>, depth: usize) -> Result<(), IcuError> {
        let Some(n) = n else { return Ok(()) };
        if depth > RECURSIVE_DEPTH_LIMIT {
            return Err(too_long());
        }
        let kind = self.arena.nodes[n].kind;
        if kind == LEAF_CHAR || kind == END_MARK {
            return Ok(());
        }
        let (l, r) = (self.arena.nodes[n].left, self.arena.nodes[n].right);
        self.calc_follow_pos(l, depth + 1)?;
        self.calc_follow_pos(r, depth + 1)?;
        if kind == OP_CAT {
            if let (Some(l), Some(r)) = (l, r) {
                let last = self.arena.nodes[l].last_pos_set.clone();
                let first = self.arena.nodes[r].first_pos_set.clone();
                for i in last {
                    self.arena.nodes[i].follow_pos.extend(first.iter().copied());
                }
            }
        }
        if kind == OP_STAR || kind == OP_PLUS {
            let last = self.arena.nodes[n].last_pos_set.clone();
            let first = self.arena.nodes[n].first_pos_set.clone();
            for i in last {
                self.arena.nodes[i].follow_pos.extend(first.iter().copied());
            }
        }
        Ok(())
    }

    /// `addRuleRootNodes`: the topmost nodes marked as the root of a rule.
    fn add_rule_root_nodes(
        &self,
        list: &mut Vec<usize>,
        n: Option<usize>,
        depth: usize,
    ) -> Result<(), IcuError> {
        let Some(n) = n else { return Ok(()) };
        if depth > RECURSIVE_DEPTH_LIMIT {
            return Err(too_long());
        }
        if self.arena.nodes[n].rule_root {
            list.push(n);
            return Ok(());
        }
        self.add_rule_root_nodes(list, self.arena.nodes[n].left, depth + 1)?;
        self.add_rule_root_nodes(list, self.arena.nodes[n].right, depth + 1)
    }

    /// `calcChainedFollowPos(tree, endMarkNode)`: a character that can end a
    /// match and is of the same category as one that can start a chainable
    /// rule is followed by what follows that start.
    fn calc_chained_follow_pos(&mut self, tree: usize, end_mark: usize) -> Result<(), IcuError> {
        let mut leaf_nodes = Vec::new();
        self.arena.find_nodes(tree, LEAF_CHAR, &mut leaf_nodes, 0)?;
        let mut rule_roots = Vec::new();
        self.add_rule_root_nodes(&mut rule_roots, Some(tree), 0)?;
        let mut match_start: BTreeSet<usize> = BTreeSet::new();
        for &r in &rule_roots {
            if self.arena.nodes[r].chain_in {
                match_start.extend(self.arena.nodes[r].first_pos_set.iter().copied());
            }
        }
        for &node in &leaf_nodes {
            if !self.arena.nodes[node].follow_pos.contains(&end_mark) {
                continue;
            }
            for &start in &match_start {
                if self.arena.nodes[start].kind != LEAF_CHAR {
                    continue;
                }
                if self.arena.nodes[node].val == self.arena.nodes[start].val {
                    let follow = self.arena.nodes[start].follow_pos.clone();
                    self.arena.nodes[node].follow_pos.extend(follow);
                }
            }
        }
        Ok(())
    }

    /// `bofFixup()`: the start-of-text character is followed by what follows
    /// any other character of its category at the start of a rule.
    fn bof_fixup(&mut self, root: usize) -> Result<(), IcuError> {
        let rules_cat = self.arena.nodes[root].left.ok_or_else(internal)?;
        let bof = self.arena.nodes[rules_cat].left.ok_or_else(internal)?;
        let rules = self.arena.nodes[rules_cat].right.ok_or_else(internal)?;
        let starts = self.arena.nodes[rules].first_pos_set.clone();
        let bof_val = self.arena.nodes[bof].val;
        for n in starts {
            if self.arena.nodes[n].kind != LEAF_CHAR {
                continue;
            }
            if self.arena.nodes[n].val == bof_val {
                let follow = self.arena.nodes[n].follow_pos.clone();
                self.arena.nodes[bof].follow_pos.extend(follow);
            }
        }
        Ok(())
    }

    /// `buildStateTable()`: the subset construction. State 0 is the failure
    /// state, state 1 the start.
    fn build_state_table(&mut self, root: usize) -> Result<(), IcuError> {
        let last_input_symbol = usize::try_from(self.num_char_categories() - 1).unwrap_or(0);
        let mut states = vec![DState::new(last_input_symbol)];
        let mut start = DState::new(last_input_symbol);
        start.positions = self.arena.nodes[root].first_pos_set.clone();
        states.push(start);
        while let Some(t) = (1..states.len()).find(|&i| !states[i].marked) {
            states[t].marked = true;
            for a in 1..=last_input_symbol {
                let mut u: BTreeSet<usize> = BTreeSet::new();
                for &p in &states[t].positions {
                    let node = &self.arena.nodes[p];
                    if node.kind == LEAF_CHAR && usize::try_from(node.val).ok() == Some(a) {
                        u.extend(node.follow_pos.iter().copied());
                    }
                }
                let idx = match states.iter().position(|s| s.positions == u) {
                    Some(i) => i,
                    None => {
                        let mut s = DState::new(last_input_symbol);
                        s.positions = u;
                        states.push(s);
                        states.len() - 1
                    }
                };
                states[t].dtran[a] = i32::try_from(idx).unwrap_or(i32::MAX);
            }
        }
        self.tb.dstates = states;
        Ok(())
    }

    /// `mapLookAheadRules()`: each look-ahead rule gets a slot, shared by the
    /// rules that can be pending in the same state, in which the iterator
    /// remembers where the look-ahead matched.
    fn map_look_ahead_rules(&mut self) -> Result<(), IcuError> {
        let rules = usize::try_from(self.sc.rule_num).unwrap_or(0);
        let mut map = vec![0i32; rules + 1];
        self.tb.la_slots_in_use = 1;
        for si in 0..self.tb.dstates.len() {
            let look_aheads: Vec<usize> = self.tb.dstates[si]
                .positions
                .iter()
                .copied()
                .filter(|&p| self.arena.nodes[p].kind == LOOK_AHEAD)
                .collect();
            if look_aheads.is_empty() {
                continue;
            }
            let mut slot = 0;
            for &p in &look_aheads {
                let rule = usize::try_from(self.arena.nodes[p].val).map_err(|_| internal())?;
                let existing = *map.get(rule).ok_or_else(internal)?;
                if existing != 0 && slot == 0 {
                    slot = existing;
                }
            }
            if slot == 0 {
                self.tb.la_slots_in_use += 1;
                slot = self.tb.la_slots_in_use;
            }
            for &p in &look_aheads {
                let rule = usize::try_from(self.arena.nodes[p].val).map_err(|_| internal())?;
                *map.get_mut(rule).ok_or_else(internal)? = slot;
            }
        }
        self.tb.look_ahead_rule_map = map;
        Ok(())
    }

    /// `ruleMap[rule]` for a node's rule number.
    fn look_ahead_slot(&self, node: usize) -> Result<i32, IcuError> {
        let rule = usize::try_from(self.arena.nodes[node].val).map_err(|_| internal())?;
        self.tb
            .look_ahead_rule_map
            .get(rule)
            .copied()
            .ok_or_else(internal)
    }

    /// `flagAcceptingStates()`: a state holding an end marker accepts --
    /// unconditionally (1), or, for a look-ahead rule's marker, once its
    /// slot's look-ahead matched.
    fn flag_accepting_states(&mut self, root: usize) -> Result<(), IcuError> {
        let mut end_marks = Vec::new();
        self.arena.find_nodes(root, END_MARK, &mut end_marks, 0)?;
        for em in end_marks {
            let slot = self.look_ahead_slot(em)?;
            let val = self.arena.nodes[em].val;
            for st in &mut self.tb.dstates {
                if !st.positions.contains(&em) {
                    continue;
                }
                if st.accepting == 0 {
                    st.accepting = slot;
                    if st.accepting == 0 {
                        st.accepting = i32::from(ACCEPTING_UNCONDITIONAL);
                    }
                }
                if st.accepting == i32::from(ACCEPTING_UNCONDITIONAL) && val != 0 {
                    st.accepting = slot;
                }
            }
        }
        Ok(())
    }

    /// `flagLookAheadStates()`.
    fn flag_look_ahead_states(&mut self, root: usize) -> Result<(), IcuError> {
        let mut nodes = Vec::new();
        self.arena.find_nodes(root, LOOK_AHEAD, &mut nodes, 0)?;
        for n in nodes {
            let slot = self.look_ahead_slot(n)?;
            for st in &mut self.tb.dstates {
                if st.positions.contains(&n) {
                    st.look_ahead = slot;
                }
            }
        }
        Ok(())
    }

    /// `flagTaggedStates()`.
    fn flag_tagged_states(&mut self, root: usize) -> Result<(), IcuError> {
        let mut nodes = Vec::new();
        self.arena.find_nodes(root, TAG, &mut nodes, 0)?;
        for n in nodes {
            let val = self.arena.nodes[n].val;
            for st in &mut self.tb.dstates {
                if st.positions.contains(&n) {
                    st.tag_vals.insert(val);
                }
            }
        }
        Ok(())
    }

    /// `mergeRuleStatusVals()`: the status table lists each distinct set of
    /// tag values once, as its count then the values; a state keeps the
    /// index of its set.
    fn merge_rule_status_vals(&mut self) {
        if self.rule_status_vals.is_empty() {
            self.rule_status_vals.push(1);
            self.rule_status_vals.push(0);
            self.tb.status_sets.insert(BTreeSet::new(), 0);
            self.tb.status_sets.insert(BTreeSet::from([0]), 0);
        }
        for si in 0..self.tb.dstates.len() {
            let tags = self.tb.dstates[si].tag_vals.clone();
            let idx = match self.tb.status_sets.get(&tags) {
                Some(&i) => i,
                None => {
                    let i = i32::try_from(self.rule_status_vals.len()).unwrap_or(i32::MAX);
                    self.rule_status_vals
                        .push(i32::try_from(tags.len()).unwrap_or(i32::MAX));
                    self.rule_status_vals.extend(tags.iter().copied());
                    self.tb.status_sets.insert(tags, i);
                    i
                }
            };
            self.tb.dstates[si].tags_idx = idx;
        }
    }

    /// `exportTable()`.
    fn export_table(&self) -> StateTable {
        let num_states = self.tb.dstates.len();
        let cats = usize::try_from(self.num_char_categories()).unwrap_or(0);
        let eight_bit = num_states <= 255;
        let row_units = NEXTSTATES + cats;
        let mut flags = 0;
        if self.look_ahead_hard_break {
            flags |= RBBI_LOOKAHEAD_HARD_BREAK;
        }
        if self.sb.saw_bof {
            flags |= RBBI_BOF_REQUIRED;
        }
        if eight_bit {
            flags |= RBBI_8BITS_ROWS;
        }
        let mut table = vec![0u16; num_states * row_units];
        for (i, st) in self.tb.dstates.iter().enumerate() {
            let row = i * row_units;
            table[row] = st.accepting as u16;
            table[row + 1] = st.look_ahead as u16;
            table[row + 2] = st.tags_idx as u16;
            for (c, &next) in st.dtran.iter().enumerate().take(cats) {
                table[row + NEXTSTATES + c] = next as u16;
            }
        }
        StateTable {
            num_states: i32::try_from(num_states).unwrap_or(i32::MAX),
            row_len: i32::try_from(if eight_bit { row_units } else { row_units * 2 })
                .unwrap_or(i32::MAX),
            dict_categories_start: self.sb.dict_categories_start,
            look_ahead_results_size: if self.tb.la_slots_in_use == 1 {
                0
            } else {
                self.tb.la_slots_in_use + 1
            },
            flags,
            table,
        }
    }
}
