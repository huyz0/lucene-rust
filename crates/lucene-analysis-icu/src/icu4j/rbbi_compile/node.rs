//! `com.ibm.icu.text.RBBINode`: the syntax tree of the break rules, as an
//! arena (a node is an index), since parents, children and the position sets
//! of the table builder all point at each other.

use std::collections::BTreeSet;

use crate::icu4j::unicode_set::UnicodeSet;
use crate::IcuError;

/// `RBBINode.setRef`.
pub(super) const SET_REF: u8 = 0;
/// `RBBINode.uset`.
pub(super) const USET: u8 = 1;
/// `RBBINode.varRef`.
pub(super) const VAR_REF: u8 = 2;
/// `RBBINode.leafChar`.
pub(super) const LEAF_CHAR: u8 = 3;
/// `RBBINode.lookAhead`.
pub(super) const LOOK_AHEAD: u8 = 4;
/// `RBBINode.tag`.
pub(super) const TAG: u8 = 5;
/// `RBBINode.endMark`.
pub(super) const END_MARK: u8 = 6;
/// `RBBINode.opStart`.
pub(super) const OP_START: u8 = 7;
/// `RBBINode.opCat`.
pub(super) const OP_CAT: u8 = 8;
/// `RBBINode.opOr`.
pub(super) const OP_OR: u8 = 9;
/// `RBBINode.opStar`.
pub(super) const OP_STAR: u8 = 10;
/// `RBBINode.opPlus`.
pub(super) const OP_PLUS: u8 = 11;
/// `RBBINode.opQuestion`.
pub(super) const OP_QUESTION: u8 = 12;
/// `RBBINode.opLParen`.
pub(super) const OP_LPAREN: u8 = 15;

/// `RBBINode.precStart`.
pub(super) const PREC_START: u8 = 1;
/// `RBBINode.precLParen`.
pub(super) const PREC_LPAREN: u8 = 2;
/// `RBBINode.precOpOr`.
pub(super) const PREC_OP_OR: u8 = 3;
/// `RBBINode.precOpCat`.
pub(super) const PREC_OP_CAT: u8 = 4;

/// `RBBINode.kRecursiveDepthLimit`: how deep a tree this port walks. Java
/// bounds only `flattenVariables`; the other walks would overflow its stack
/// on a tree this deep, so they refuse it the same way.
pub(super) const RECURSIVE_DEPTH_LIMIT: usize = 3500;

/// One node of the rule tree.
#[derive(Debug, Clone)]
pub(super) struct Node {
    pub kind: u8,
    pub parent: Option<usize>,
    pub left: Option<usize>,
    pub right: Option<usize>,
    /// `fInputSet`, a `uset` node's characters.
    pub input_set: Option<UnicodeSet>,
    pub precedence: u8,
    /// `fText`, the rule source the node was scanned from.
    pub text: String,
    pub first_pos: usize,
    pub last_pos: usize,
    pub nullable: bool,
    /// `fVal`: a rule number, a tag value or a character category.
    pub val: i32,
    pub rule_root: bool,
    pub chain_in: bool,
    pub first_pos_set: BTreeSet<usize>,
    pub last_pos_set: BTreeSet<usize>,
    pub follow_pos: BTreeSet<usize>,
}

impl Node {
    /// `new RBBINode(type)`.
    fn new(kind: u8) -> Node {
        let precedence = match kind {
            OP_CAT => PREC_OP_CAT,
            OP_OR => PREC_OP_OR,
            OP_START => PREC_START,
            OP_LPAREN => PREC_LPAREN,
            _ => 0,
        };
        Node {
            kind,
            parent: None,
            left: None,
            right: None,
            input_set: None,
            precedence,
            text: String::new(),
            first_pos: 0,
            last_pos: 0,
            nullable: false,
            val: 0,
            rule_root: false,
            chain_in: false,
            first_pos_set: BTreeSet::new(),
            last_pos_set: BTreeSet::new(),
            follow_pos: BTreeSet::new(),
        }
    }
}

fn too_long() -> IcuError {
    IcuError::illegal_argument("The input is too long")
}

/// The nodes of one rule compilation.
#[derive(Debug, Default)]
pub(super) struct Arena {
    pub nodes: Vec<Node>,
}

impl Arena {
    /// `new RBBINode(type)`, added to the arena.
    pub fn new_node(&mut self, kind: u8) -> usize {
        self.nodes.push(Node::new(kind));
        self.nodes.len().saturating_sub(1)
    }

    /// `RBBINode(RBBINode other)`: a copy of the node alone (no children, not
    /// a rule root).
    fn copy_node(&mut self, other: usize) -> usize {
        let mut n = self.nodes[other].clone();
        n.parent = None;
        n.left = None;
        n.right = None;
        n.rule_root = false;
        self.nodes.push(n);
        self.nodes.len().saturating_sub(1)
    }

    /// `cloneTree()`: a variable reference is its definition, cloned; a
    /// `uset` is shared; any other node is copied with its children.
    pub fn clone_tree(&mut self, id: usize) -> Result<usize, IcuError> {
        self.clone_tree_at(id, 0)
    }

    fn clone_tree_at(&mut self, id: usize, depth: usize) -> Result<usize, IcuError> {
        if depth > RECURSIVE_DEPTH_LIMIT {
            return Err(too_long());
        }
        let kind = self.nodes[id].kind;
        if kind == VAR_REF {
            return match self.nodes[id].left {
                Some(l) => self.clone_tree_at(l, depth.saturating_add(1)),
                None => Err(IcuError::illegal_argument("undefined variable")),
            };
        }
        if kind == USET {
            return Ok(id);
        }
        let n = self.copy_node(id);
        if let Some(l) = self.nodes[id].left {
            let c = self.clone_tree_at(l, depth.saturating_add(1))?;
            self.nodes[n].left = Some(c);
            self.nodes[c].parent = Some(n);
        }
        if let Some(r) = self.nodes[id].right {
            let c = self.clone_tree_at(r, depth.saturating_add(1))?;
            self.nodes[n].right = Some(c);
            self.nodes[c].parent = Some(n);
        }
        Ok(n)
    }

    /// `flattenVariables(depth)`: every variable reference below `id` is its
    /// definition's tree; returns the tree that replaces `id`.
    pub fn flatten_variables(&mut self, id: usize, depth: usize) -> Result<usize, IcuError> {
        if depth > RECURSIVE_DEPTH_LIMIT {
            return Err(too_long());
        }
        if self.nodes[id].kind == VAR_REF {
            let l = self.nodes[id]
                .left
                .ok_or_else(|| IcuError::illegal_argument("undefined variable"))?;
            let ret = self.clone_tree(l)?;
            self.nodes[ret].rule_root = self.nodes[id].rule_root;
            self.nodes[ret].chain_in = self.nodes[id].chain_in;
            return Ok(ret);
        }
        if let Some(l) = self.nodes[id].left {
            let c = self.flatten_variables(l, depth.saturating_add(1))?;
            self.nodes[id].left = Some(c);
            self.nodes[c].parent = Some(id);
        }
        if let Some(r) = self.nodes[id].right {
            let c = self.flatten_variables(r, depth.saturating_add(1))?;
            self.nodes[id].right = Some(c);
            self.nodes[c].parent = Some(id);
        }
        Ok(id)
    }

    /// `flattenSets()`: a set reference below `id` becomes the tree of
    /// character categories its `uset` was given.
    pub fn flatten_sets(&mut self, id: usize) -> Result<(), IcuError> {
        self.flatten_sets_at(id, 0)
    }

    fn flatten_sets_at(&mut self, id: usize, depth: usize) -> Result<(), IcuError> {
        if depth > RECURSIVE_DEPTH_LIMIT {
            return Err(too_long());
        }
        for right in [false, true] {
            let child = if right {
                self.nodes[id].right
            } else {
                self.nodes[id].left
            };
            let Some(child) = child else { continue };
            if self.nodes[child].kind == SET_REF {
                let repl = self.set_replacement(child)?;
                let c = self.clone_tree(repl)?;
                self.nodes[c].parent = Some(id);
                if right {
                    self.nodes[id].right = Some(c);
                } else {
                    self.nodes[id].left = Some(c);
                }
            } else {
                self.flatten_sets_at(child, depth.saturating_add(1))?;
            }
        }
        Ok(())
    }

    /// The category tree of a set reference: `setRef.left.left`.
    fn set_replacement(&self, set_ref: usize) -> Result<usize, IcuError> {
        self.nodes[set_ref]
            .left
            .and_then(|u| self.nodes[u].left)
            .ok_or_else(|| IcuError::illegal_argument("a set without character categories"))
    }

    /// `findNodes(dest, kind)`: the nodes of `kind` at or below `id`, in
    /// pre-order.
    pub fn find_nodes(
        &self,
        id: usize,
        kind: u8,
        dest: &mut Vec<usize>,
        depth: usize,
    ) -> Result<(), IcuError> {
        if depth > RECURSIVE_DEPTH_LIMIT {
            return Err(too_long());
        }
        if self.nodes[id].kind == kind {
            dest.push(id);
        }
        if let Some(l) = self.nodes[id].left {
            self.find_nodes(l, kind, dest, depth.saturating_add(1))?;
        }
        if let Some(r) = self.nodes[id].right {
            self.find_nodes(r, kind, dest, depth.saturating_add(1))?;
        }
        Ok(())
    }
}
