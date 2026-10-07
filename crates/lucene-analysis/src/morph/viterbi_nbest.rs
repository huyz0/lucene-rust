//! `org.apache.lucene.analysis.morph.ViterbiNBest`: the n-best paths
//! through a backtraced fragment's lattice (`Lattice`), and the pending
//! list fix-up that merges them with the best path.
//!
//! As in [`super::viterbi`], Java's abstract class is split into state
//! ([`NBestState`], [`Lattice`]) and the language hook
//! ([`NBestLang::register_node`]); [`backtrace_nbest`] and
//! [`fixup_pending_list`] are Java's final methods. The lattice's fourteen
//! parallel arrays are one vector of node records. Costs wrap as Java
//! `int`s do.

use std::sync::Arc;

use super::connection_costs::ConnectionCosts;
use super::resource::io_error;
use super::token::{MorphToken, TokenType};
use super::viterbi::{Viterbi, ViterbiLang, WrappedPositionArray};
use crate::AnalysisError;

/// `ViterbiNBest`'s own fields: `nBestCost` and the reused `lattice`.
#[derive(Debug, Clone, Default)]
pub struct NBestState {
    n_best_cost: i32,
    lattice: Option<Lattice>,
}

impl NBestState {
    /// `setNBestCost(value)`: n-best output is on when the cost is
    /// positive.
    pub fn set_n_best_cost<T>(&mut self, v: &mut Viterbi<T>, value: i32) {
        self.n_best_cost = value;
        v.output_nbest = 0 < value;
    }

    /// `getNBestCost()`.
    pub fn n_best_cost(&self) -> i32 {
        self.n_best_cost
    }

    /// `getLatticeRootBase()` (`None` before the first n-best backtrace,
    /// where Java throws `NullPointerException`).
    pub fn lattice_root_base(&self) -> Option<i32> {
        self.lattice.as_ref().map(|l| l.root_base)
    }

    /// `probeDelta(start, end)` (`None` as for
    /// [`Self::lattice_root_base`]).
    pub fn probe_delta(&self, start: i32, end: i32) -> Option<i32> {
        self.lattice.as_ref().map(|l| l.probe_delta(start, end))
    }
}

/// A language with n-best output (Kuromoji's `ViterbiNBest`).
pub trait NBestLang<T>: ViterbiLang<T> {
    /// The n-best state.
    fn nbest_state(&mut self) -> &mut NBestState;

    /// `registerNode(node, fragment)`: adds the tokens of an n-best node to
    /// the pending list.
    fn register_node(
        &self,
        v: &mut Viterbi<T>,
        lattice: &Lattice,
        node: usize,
        fragment: &Arc<[u16]>,
    ) -> Result<(), AnalysisError>;
}

/// `ViterbiNBest.backtraceNBest(endPosData, useEOS)`.
pub fn backtrace_nbest<T, L: NBestLang<T>>(
    v: &mut Viterbi<T>,
    lang: &mut L,
    end_pos: i32,
    use_eos: bool,
) -> Result<(), AnalysisError> {
    let mut lattice = lang.nbest_state().lattice.take().unwrap_or_default();
    let n_best_cost = lang.nbest_state().n_best_cost;
    let last = v.last_back_trace_pos;
    let fragment = v.fragment(last, end_pos.wrapping_sub(last));
    lattice.setup(&*lang, &mut v.positions, last, end_pos, use_eos)?;
    lattice.mark_unreachable();
    lattice.calc_left_cost(&v.costs);
    lattice.calc_right_cost(&v.costs);

    let best_cost = lattice.best_cost();
    let result = (|| {
        for node in lattice.best_path_node_list()? {
            lang.register_node(v, &lattice, node, &fragment)?;
        }
        let mut n = 2i32;
        loop {
            let nbest = lattice.n_best_node_list(n);
            let Some(&first) = nbest.first() else {
                break;
            };
            let cost = lattice.cost(first);
            if best_cost.wrapping_add(n_best_cost) < cost {
                break;
            }
            for node in nbest {
                lang.register_node(v, &lattice, node, &fragment)?;
            }
            n = n.wrapping_add(1);
        }
        Ok(())
    })();
    lang.nbest_state().lattice = Some(lattice);
    result
}

/// `ViterbiNBest.fixupPendingList()`: sorts the pending tokens, drops the
/// repeats of one span, sets each position length from the token edges,
/// and reverses the list (it is served from the end).
pub fn fixup_pending_list<T: MorphToken>(pending: &mut Vec<T>) {
    // Sort for removing same tokens. USER token should be ahead from
    // normal one: order of Type is KNOWN, UNKNOWN, USER, so reversed.
    pending.sort_by(|a, b| {
        let (a, b) = (a.base(), b.base());
        a.offset
            .cmp(&b.offset)
            .then(a.length.cmp(&b.length))
            .then(b.token_type.cmp(&a.token_type))
    });
    // Remove same token.
    pending
        .dedup_by(|b, a| a.base().offset == b.base().offset && a.base().length == b.base().length);

    // Get unique and sorted list of all edge positions of tokens.
    let mut offsets: Vec<i32> = Vec::with_capacity(pending.len().saturating_mul(2));
    for t in pending.iter() {
        offsets.push(t.base().offset);
        offsets.push(t.base().offset.wrapping_add(t.base().length));
    }
    offsets.sort_unstable();
    offsets.dedup();
    let rank = |o: i32| offsets.binary_search(&o).map_or(0, |r| r as i32);
    for t in pending.iter_mut() {
        let start = rank(t.base().offset);
        let end = rank(t.base().offset.wrapping_add(t.base().length));
        t.base_mut().pos_len = end.wrapping_sub(start);
    }
    // Make PENDING to be reversed order to fit its usage.
    pending.reverse();
}

/// One lattice node.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct Node {
    dic_type: TokenType,
    word_id: i32,
    /// -1: excluded, 0: unused, 1: best path, N: N-best path.
    mark: i32,
    left_id: i32,
    right_id: i32,
    word_cost: i32,
    left_cost: i32,
    right_cost: i32,
    /// The left/right node of the least-cost path through this one.
    left_node: i32,
    right_node: i32,
    /// Start/end offset in the fragment (-1 for BOS's left, EOS's right).
    left: i32,
    right: i32,
    left_chain: i32,
    right_chain: i32,
}

/// `ViterbiNBest.Lattice`: the fragment's nodes, chained by start offset
/// (`lRoot`/`leftChain`) and by end offset (`rRoot`/`rightChain`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Lattice {
    use_eos: bool,
    root_size: i32,
    root_base: i32,
    l_root: Vec<i32>,
    r_root: Vec<i32>,
    nodes: Vec<Node>,
}

fn slot(i: i32) -> usize {
    usize::try_from(i).unwrap_or(usize::MAX)
}

impl Lattice {
    fn node(&self, node: i32) -> Option<&Node> {
        self.nodes.get(slot(node))
    }

    /// `getNodeLeft(node)`.
    pub fn node_left(&self, node: usize) -> i32 {
        self.nodes.get(node).map_or(0, |n| n.left)
    }
    /// `getNodeRight(node)`.
    pub fn node_right(&self, node: usize) -> i32 {
        self.nodes.get(node).map_or(0, |n| n.right)
    }
    /// `getNodeDicType(node)`.
    pub fn node_dic_type(&self, node: usize) -> TokenType {
        self.nodes
            .get(node)
            .map_or(TokenType::Known, |n| n.dic_type)
    }
    /// `getNodeWordID(node)`.
    pub fn node_word_id(&self, node: usize) -> i32 {
        self.nodes.get(node).map_or(-1, |n| n.word_id)
    }
    /// `getRootBase()`.
    pub fn root_base(&self) -> i32 {
        self.root_base
    }

    /// `setupRoot(baseOffset, lastOffset)`.
    fn setup_root(&mut self, base_offset: i32, last_offset: i32) {
        let size = last_offset.wrapping_sub(base_offset).wrapping_add(1).max(0);
        // ALLOC: one root per position from the last backtrace to
        // `last_offset`, positions the search has already read from the
        // input (at most MAX_BACKTRACE_GAP-ish past a frontier, never more
        // than the input's length).
        self.l_root.clear();
        self.l_root.resize(slot(size), -1);
        // ALLOC: as `l_root`.
        self.r_root.clear();
        self.r_root.resize(slot(size), -1);
        self.root_size = size;
        self.root_base = base_offset;
    }

    /// `addNode(dicType, wordID, left, right)`.
    fn add_node<T, L: ViterbiLang<T> + ?Sized>(
        &mut self,
        lang: &L,
        dic_type: TokenType,
        word_id: i32,
        left: i32,
        right: i32,
    ) -> usize {
        let node = self.nodes.len();
        let node_id = i32::try_from(node).unwrap_or(i32::MAX);
        let mut n = Node {
            dic_type,
            word_id,
            left,
            right,
            ..Node::default()
        };
        if word_id >= 0 {
            (n.left_id, n.right_id, n.word_cost) = lang.connection(dic_type, word_id);
        }
        n.left_chain = -1;
        if let Some(root) = self.l_root.get_mut(slot(left)) {
            n.left_chain = *root;
            *root = node_id;
        }
        n.right_chain = -1;
        if let Some(root) = self.r_root.get_mut(slot(right)) {
            n.right_chain = *root;
            *root = node_id;
        }
        self.nodes.push(n);
        node
    }

    /// `setup(fragment, dictionaryMap, positions, prevOffset, endOffset,
    /// useEOS)`.
    fn setup<T, L: ViterbiLang<T> + ?Sized>(
        &mut self,
        lang: &L,
        positions: &mut WrappedPositionArray,
        prev_offset: i32,
        end_offset: i32,
        use_eos: bool,
    ) -> Result<(), AnalysisError> {
        self.use_eos = use_eos;
        // Initialize lRoot and rRoot.
        self.setup_root(prev_offset, end_offset);
        // setupNodePool
        self.nodes.clear();

        // substitute for BOS = 0
        let first = positions.get(prev_offset);
        let (bos_type, bos_id) = (first.back_type(0)?, first.back_id(0)?);
        self.add_node(lang, bos_type, bos_id, -1, 0);
        // EOS = 1
        self.add_node(
            lang,
            TokenType::Known,
            -1,
            end_offset.wrapping_sub(self.root_base),
            -1,
        );

        let mut offset = end_offset;
        while prev_offset < offset {
            let right = offset.wrapping_sub(self.root_base);
            // optimize: exclude disconnected nodes.
            if self.l_root.get(slot(right)).is_some_and(|&r| 0 <= r) {
                let pos = positions.get(offset);
                for i in 0..pos.count() {
                    let (t, id, back_pos) = (pos.back_type(i)?, pos.back_id(i)?, pos.back_pos(i)?);
                    self.add_node(lang, t, id, back_pos.wrapping_sub(self.root_base), right);
                }
            }
            offset = offset.wrapping_sub(1);
        }
        Ok(())
    }

    /// Node `node` of a chain being walked (its slot and a copy), or `None`
    /// at the chain's end -- or once `*steps` passes the node count, which
    /// only a cycle could (no lattice this code builds has one).
    #[inline]
    fn step(&self, node: i32, steps: &mut usize) -> Option<(usize, Node)> {
        let n = *self.node(node)?;
        *steps = steps.saturating_add(1);
        if *steps > self.nodes.len().saturating_add(1) {
            return None;
        }
        Some((slot(node), n))
    }

    /// `markUnreachable()`: nodes starting where no node ends are excluded.
    fn mark_unreachable(&mut self) {
        for index in 1..self.root_size.wrapping_sub(1).max(1) {
            if self.r_root.get(slot(index)).is_some_and(|&r| r < 0) {
                let mut node = self.l_root.get(slot(index)).copied().unwrap_or(-1);
                let mut steps = 0;
                while let Some((i, n)) = self.step(node, &mut steps) {
                    self.nodes[i].mark = -1;
                    node = n.left_chain;
                }
            }
        }
    }

    /// `connectionCost(costs, left, right)`.
    fn connection_cost(&self, costs: &ConnectionCosts, left: &Node, right: &Node) -> i32 {
        if right.left_id == 0 && !self.use_eos {
            0
        } else {
            costs.get(left.right_id, right.left_id)
        }
    }

    /// `calcLeftCost(costs)`.
    fn calc_left_cost(&mut self, costs: &ConnectionCosts) {
        for index in 0..self.root_size {
            let (l_head, r_head) = (
                self.l_root.get(slot(index)).copied().unwrap_or(-1),
                self.r_root.get(slot(index)).copied().unwrap_or(-1),
            );
            let mut node = l_head;
            let mut steps = 0;
            while let Some((i, n)) = self.step(node, &mut steps) {
                node = n.left_chain;
                if n.mark < 0 {
                    continue;
                }
                let mut least_node = -1i32;
                let mut least_cost = i32::MAX;
                let mut left_node = r_head;
                let mut left_steps = 0;
                while let Some((j, l)) = self.step(left_node, &mut left_steps) {
                    left_node = l.right_chain;
                    if 0 <= l.mark {
                        let cost = l
                            .left_cost
                            .wrapping_add(l.word_cost)
                            .wrapping_add(self.connection_cost(costs, &l, &self.nodes[i]));
                        if cost < least_cost {
                            least_cost = cost;
                            least_node = j as i32;
                        }
                    }
                }
                self.nodes[i].left_node = least_node;
                self.nodes[i].left_cost = least_cost;
            }
        }
    }

    /// `calcRightCost(costs)`.
    fn calc_right_cost(&mut self, costs: &ConnectionCosts) {
        let mut index = self.root_size.wrapping_sub(1);
        while 0 <= index {
            let (l_head, r_head) = (
                self.l_root.get(slot(index)).copied().unwrap_or(-1),
                self.r_root.get(slot(index)).copied().unwrap_or(-1),
            );
            let mut node = r_head;
            let mut steps = 0;
            while let Some((i, n)) = self.step(node, &mut steps) {
                node = n.right_chain;
                if n.mark < 0 {
                    continue;
                }
                let mut least_node = -1i32;
                let mut least_cost = i32::MAX;
                let mut right_node = l_head;
                let mut right_steps = 0;
                while let Some((j, r)) = self.step(right_node, &mut right_steps) {
                    right_node = r.left_chain;
                    if 0 <= r.mark {
                        let cost = r
                            .right_cost
                            .wrapping_add(r.word_cost)
                            .wrapping_add(self.connection_cost(costs, &self.nodes[i], &r));
                        if cost < least_cost {
                            least_cost = cost;
                            least_node = j as i32;
                        }
                    }
                }
                self.nodes[i].right_node = least_node;
                self.nodes[i].right_cost = least_cost;
            }
            index = index.wrapping_sub(1);
        }
    }

    /// `markSameSpanNode(refNode, value)`: every node of the same span.
    fn mark_same_span_node(&mut self, ref_node: usize, value: i32) {
        let Some(&Node { left, right, .. }) = self.nodes.get(ref_node) else {
            return;
        };
        let mut node = self.l_root.get(slot(left)).copied().unwrap_or(-1);
        let mut steps = 0;
        while let Some((i, n)) = self.step(node, &mut steps) {
            node = n.left_chain;
            if n.right == right {
                self.nodes[i].mark = value;
            }
        }
    }

    /// `bestPathNodeList()`.
    fn best_path_node_list(&mut self) -> Result<Vec<usize>, AnalysisError> {
        let mut list = Vec::new();
        let mut node = self.node(0).map_or(-1, |n| n.right_node);
        while node != 1 {
            let next = self
                .node(node)
                .map(|n| n.right_node)
                .filter(|_| list.len() <= self.nodes.len())
                .ok_or_else(|| io_error("ArrayIndexOutOfBoundsException", node))?;
            list.push(slot(node));
            self.mark_same_span_node(slot(node), 1);
            node = next;
        }
        Ok(list)
    }

    /// `cost(node)`.
    fn cost(&self, node: usize) -> i32 {
        self.nodes.get(node).map_or(i32::MAX, |n| {
            n.left_cost
                .wrapping_add(n.word_cost)
                .wrapping_add(n.right_cost)
        })
    }

    /// `nBestNodeList(N)`: the unused nodes of the least cost, one per
    /// span.
    fn n_best_node_list(&mut self, n: i32) -> Vec<usize> {
        let mut list = Vec::new();
        let mut least_cost = i32::MAX;
        let mut least_left = -1;
        let mut least_right = -1;
        for node in 2..self.nodes.len() {
            if self.nodes[node].mark == 0 {
                let cost = self.cost(node);
                let (left, right) = (self.nodes[node].left, self.nodes[node].right);
                if cost < least_cost {
                    least_cost = cost;
                    least_left = left;
                    least_right = right;
                    list.clear();
                    list.push(node);
                } else if cost == least_cost && (left != least_left || right != least_right) {
                    list.push(node);
                }
            }
        }
        for &node in &list {
            self.mark_same_span_node(node, n);
        }
        list
    }

    /// `bestCost()`.
    fn best_cost(&self) -> i32 {
        self.node(1).map_or(i32::MAX, |n| n.left_cost)
    }

    /// `probeDelta(start, end)`.
    fn probe_delta(&self, start: i32, end: i32) -> i32 {
        let left = start.wrapping_sub(self.root_base);
        let right = end.wrapping_sub(self.root_base);
        if left < 0 || self.root_size < right {
            return i32::MAX;
        }
        let mut probed = i32::MAX;
        let mut node = self.l_root.get(slot(left)).copied().unwrap_or(-1);
        let mut steps = 0;
        while let Some((i, n)) = self.step(node, &mut steps) {
            node = n.left_chain;
            if n.right == right {
                probed = probed.min(self.cost(i));
            }
        }
        probed.wrapping_sub(self.best_cost())
    }
}
