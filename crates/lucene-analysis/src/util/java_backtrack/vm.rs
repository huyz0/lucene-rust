//! The backtracking matcher's machine: the pattern tree ([`Node`]) compiled
//! to a flat program and run with explicit stacks on the heap, making the
//! choices [`super`]'s module docs list for each construct.
//!
//! The tree used to be matched by continuation passing, each node calling
//! the rest of the match as a closure, so a match's backtracking depth was
//! native stack. Here the same choices are data: where that matcher called
//! its continuation and went on to an alternative when it failed, the
//! machine pushes an [`Entry`] -- the alternative, a capture to restore, a
//! repetition's back-off position, a loop's failed-position memo to record
//! -- and goes on; a failure pops entries until one resumes. A match's
//! depth costs heap (at most [`MAX_ENTRIES`] entries, Java's
//! `StackOverflowError` past that), never stack.
//!
//! Only a construct that needs the *first* match of a sub-pattern -- a
//! lookaround, an atomic group, a quantifier over anything but one-way
//! nodes -- runs it as a nested [`Code::run`], whose entries it drops (a
//! success) or has popped (a failure) before it goes on. That recursion is
//! as deep as the pattern nests such constructs ([`Code::depth`]), which
//! [`super::Program`] bounds before it matches on the caller's stack.
//!
//! Runs of nodes with one way to match and no side effects (characters,
//! literals, backreferences, assertions, fixed repetitions of them) are one
//! instruction ([`SOp`]s), walked without an entry each.

use std::collections::HashSet;

use super::{
    count_chars, has_base, is_word, line_terminator, overflow, CharSet, FastHash, Greed, Node,
    Program, RepMode, State,
};
use crate::java_character::{self as jc, char_count, get_type, to_lower_case, to_upper_case};
use crate::AnalysisError;

/// The most entries (and back-off positions) one match may hold before it
/// fails as Java's `StackOverflowError` (some 24 bytes each). Java's own
/// limit, on a 1 MiB thread stack, comes after some 1,500 iterations of
/// `(a|b)*`; this allows several hundred thousand.
pub(crate) const MAX_ENTRIES: usize = 1 << 20;

/// A zero-width assertion without side effects.
#[derive(Debug, Clone, Copy)]
enum Assert {
    /// `\A`, and `^` without `MULTILINE`.
    Begin,
    /// `\z`.
    End,
    /// `^` with `MULTILINE`.
    Caret { unix: bool },
    /// `$` and `\Z`.
    Dollar { multiline: bool, unix: bool },
    /// `\b` (`not`: `\B`).
    Bound { not: bool, unicode: bool },
    /// `\G`.
    LastMatch,
}

/// One step of a run of one-way nodes.
#[derive(Debug, Clone)]
enum SOp {
    /// One character of a set: one unit when the set is BMP-only.
    Char(CharSet),
    /// A run of literals, one code point each.
    Slice(Box<[CharSet]>),
    BackRef {
        group: usize,
        ci: Option<bool>,
    },
    Assert(Assert),
    /// A fixed repetition of a character, literals or a backreference.
    Times(u32, Box<SOp>),
}

/// What a quantifier, a lookaround or an atomic group applies to: a run of
/// one-way nodes, or a sub-program (ending in [`Inst::End`]) run nested.
#[derive(Debug)]
enum Atom {
    Simple(Box<[SOp]>),
    Code(usize),
}

/// One instruction; unless it names another, the next one follows it.
#[derive(Debug)]
enum Inst {
    /// A run of one-way nodes.
    Simple(Box<[SOp]>),
    /// A capturing group around a run of one-way nodes, and the run of
    /// one-way steps after it.
    CaptureSimple {
        g: usize,
        ops: Box<[SOp]>,
        then: Box<[SOp]>,
    },
    /// A capturing group's start, into register `reg`.
    GroupOpen {
        reg: usize,
    },
    /// A capturing group's end: its span from register `reg`.
    GroupClose {
        g: usize,
        reg: usize,
    },
    Jmp(usize),
    /// Alternatives, each ending in a jump past the last.
    Alt(Branches),
    Look {
        negate: bool,
        body: Atom,
        next: usize,
    },
    Behind {
        negate: bool,
        body: Atom,
        min: i32,
        max: i32,
        by_code_point: bool,
        next: usize,
    },
    Atomic {
        body: Atom,
        next: usize,
    },
    /// `Curly` over the atom's first match; `unit`: the atom is one BMP
    /// set, one unit per repetition.
    Curly {
        atom: Atom,
        min: u32,
        max: u32,
        greed: Greed,
        unit: bool,
        next: usize,
    },
    Ques {
        atom: Atom,
        greed: Greed,
        next: usize,
    },
    GroupCurly {
        body: Atom,
        min: u32,
        max: u32,
        greed: Greed,
        capture: Option<usize>,
        next: usize,
    },
    /// `Loop`: the body follows, ending in [`Inst::LoopTail`]; registers
    /// `reg` (the iteration under way) and `reg + 1` (where it began).
    LoopInit {
        min: u32,
        max: u32,
        lazy: bool,
        memo: Option<usize>,
        reg: usize,
        next: usize,
    },
    /// The end of a `Loop`'s body (its [`Inst::LoopInit`] at `init`).
    LoopTail {
        init: usize,
    },
    /// `\R`.
    LineEnding,
    /// The end of the pattern or of a sub-program.
    End,
}

/// An alternation's alternatives: where each starts, and the characters it
/// must start with when that is known.
#[derive(Debug)]
struct Branches {
    pcs: Box<[usize]>,
    firsts: Box<[Option<CharSet>]>,
    /// Per ASCII unit, the alternatives it does not rule out (bit `k` for
    /// the `k`th), when there are at most 64.
    ascii: Option<Box<[u64; 128]>>,
}

impl Branches {
    fn new(pcs: Vec<usize>, firsts: Vec<Option<CharSet>>) -> Self {
        let ascii = (pcs.len() <= 64).then(|| {
            let mut t = Box::new([0u64; 128]);
            for (u, slot) in t.iter_mut().enumerate() {
                for (k, f) in firsts.iter().enumerate() {
                    if f.as_ref().is_none_or(|set| set.contains(u as u32)) {
                        *slot |= 1 << k;
                    }
                }
            }
            t
        });
        Branches {
            pcs: pcs.into_boxed_slice(),
            firsts: firsts.into_boxed_slice(),
            ascii,
        }
    }
}

/// A choice to come back to, or state to restore, when what follows fails.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Entry {
    /// Go on at `pc` from `i`.
    Resume { pc: u32, i: usize },
    /// The alternatives of the `Alt` at `pc` from the `b`th, at `i`.
    AltNext { pc: u32, b: u32, i: usize },
    /// A capture's previous span.
    Group { g: u32, s: i32, e: i32 },
    /// A register's previous value.
    Reg { r: u32, v: usize },
    /// A span a `GroupCurly` records again once the (sub-)match succeeds.
    OnSuccess { g: u32, s: i32, e: i32 },
    /// A greedy unit `Curly`'s back-off: what follows from `cur - 1` down
    /// to `lo`.
    CurlyUnit { next: u32, lo: usize, cur: usize },
    /// A greedy `Curly`'s back-off through `positions[base..top]`.
    CurlyBack { next: u32, base: u32, top: u32 },
    /// A lazy `Curly` at `pos` after `count` repetitions.
    CurlyLazy { pc: u32, pos: usize, count: u32 },
    /// A lazy `Ques`'s atom at `i`.
    QuesLazy { pc: u32, i: usize },
    /// A greedy `GroupCurly`'s back-off: `positions[base]` its start,
    /// `[base + 1]`, `[base + 2]` its capture there, then the spans.
    GroupBack { pc: u32, base: u32, top: u32 },
    /// A lazy `GroupCurly` at `pos` after `count` iterations.
    GroupLazy { pc: u32, pos: usize, count: u32 },
    /// A lazy `Loop`'s first iteration, at `i`.
    LoopLazyFirst { pc: u32, i: usize },
    /// A lazy `Loop`'s next iteration, at `i` after the `count`th.
    LoopLazyNext { pc: u32, i: usize, count: u32 },
    /// A greedy `Loop`'s exit at `i`, after another iteration failed.
    LoopExit { pc: u32, i: usize },
}

/// A match's stacks, kept between matches.
#[derive(Debug, Default, Clone)]
pub(crate) struct Scratch {
    stack: Vec<Entry>,
    regs: Vec<usize>,
    positions: Vec<usize>,
    /// `localsPos`: per memoised `Loop`, the positions where another
    /// iteration failed (for one `find`).
    pub(crate) failed: HashSet<(usize, usize), FastHash>,
}

impl Scratch {
    pub(crate) fn clear(&mut self) {
        self.stack.clear();
        self.positions.clear();
        if !self.failed.is_empty() {
            self.failed.clear();
        }
    }
}

/// A compiled pattern.
#[derive(Debug)]
pub(crate) struct Code {
    insts: Vec<Inst>,
    regs: usize,
    /// How deeply sub-programs nest (the native recursion of a match).
    pub(crate) depth: usize,
    /// A capturing `GroupCurly` exists ([`Entry::OnSuccess`]).
    on_success: bool,
}

// ---------------------------------------------------------------- compile

/// The one-way steps of `n`, when it is a run of one-way nodes.
fn simple_ops(n: &Node, out: &mut Vec<SOp>) -> bool {
    match n {
        Node::Empty => true,
        Node::Char(cs) => {
            out.push(SOp::Char(cs.clone()));
            true
        }
        Node::Slice(sets) => {
            out.push(SOp::Slice(sets.clone().into_boxed_slice()));
            true
        }
        Node::BackRef { group, ci } => {
            out.push(SOp::BackRef {
                group: *group,
                ci: *ci,
            });
            true
        }
        Node::Group(None, body) => simple_ops(body, out),
        Node::Seq(items) => items.iter().all(|n| simple_ops(n, out)),
        Node::Repeat {
            atom,
            min,
            max,
            mode: RepMode::First,
            ..
        } if min == max
            && matches!(
                **atom,
                Node::Char(_) | Node::Slice(_) | Node::BackRef { .. }
            ) =>
        {
            let mut one = Vec::new();
            simple_ops(atom, &mut one);
            match one.pop() {
                Some(op) => {
                    out.push(SOp::Times(*min, Box::new(op)));
                    true
                }
                None => false,
            }
        }
        Node::Begin => assert_op(out, Assert::Begin),
        Node::End => assert_op(out, Assert::End),
        Node::Caret { unix } => assert_op(out, Assert::Caret { unix: *unix }),
        Node::Dollar { multiline, unix } => assert_op(
            out,
            Assert::Dollar {
                multiline: *multiline,
                unix: *unix,
            },
        ),
        Node::Bound { not, unicode } => assert_op(
            out,
            Assert::Bound {
                not: *not,
                unicode: *unicode,
            },
        ),
        Node::LastMatch => assert_op(out, Assert::LastMatch),
        _ => false,
    }
}

fn assert_op(out: &mut Vec<SOp>, a: Assert) -> bool {
    out.push(SOp::Assert(a));
    true
}

/// A `Curly` of one one-way step with a minimum (`X{n,m}`, `n > 0`, `m >
/// n`): the step, the minimum, and the repetition of the rest (its `next`
/// to be filled in), so the minimum can join the run of steps before it.
fn split_min(n: &Node) -> Option<(SOp, u32, Inst)> {
    let Node::Repeat {
        atom,
        min,
        max,
        greed,
        mode: RepMode::First,
        ..
    } = n
    else {
        return None;
    };
    let ops = simple(atom)?;
    let [op] = &ops[..] else {
        return None;
    };
    if *min == 0 || max <= min || matches!(op, SOp::Times(..)) {
        return None;
    }
    let unit = matches!(op, SOp::Char(cs) if cs.bmp);
    Some((
        op.clone(),
        *min,
        Inst::Curly {
            atom: Atom::Simple(Box::new([op.clone()])),
            min: 0,
            max: max - min,
            greed: *greed,
            unit,
            next: usize::MAX,
        },
    ))
}

/// `n`'s one-way steps, or `None`.
fn simple(n: &Node) -> Option<Vec<SOp>> {
    let mut ops = Vec::new();
    simple_ops(n, &mut ops).then_some(ops)
}

#[derive(Default)]
struct Compiler {
    insts: Vec<Inst>,
    regs: usize,
    depth: usize,
    nesting: usize,
    on_success: bool,
}

impl Code {
    /// The program of a parsed (and annotated) tree.
    pub(crate) fn compile(root: &Node) -> Code {
        let mut c = Compiler::default();
        c.node(root);
        c.insts.push(Inst::End);
        Code {
            insts: c.insts,
            regs: c.regs,
            depth: c.depth,
            on_success: c.on_success,
        }
    }
}

impl Compiler {
    fn reg(&mut self, n: usize) -> usize {
        let r = self.regs;
        self.regs += n;
        r
    }

    /// An instruction to fill in once what follows it is laid out.
    fn placeholder(&mut self) -> usize {
        self.insts.push(Inst::End);
        self.insts.len() - 1
    }

    /// `inst`, going on to the instruction after it.
    fn push_next(&mut self, mut inst: Inst) {
        let after = self.insts.len() + 1;
        if let Inst::Curly { next, .. } = &mut inst {
            *next = after;
        }
        self.insts.push(inst);
    }

    fn simple(&mut self, ops: Vec<SOp>) {
        if !ops.is_empty() {
            self.insts.push(Inst::Simple(ops.into_boxed_slice()));
        }
    }

    /// `n` as an atom: its steps, or a sub-program laid out here.
    fn atom(&mut self, n: &Node) -> Atom {
        if let Some(ops) = simple(n) {
            return Atom::Simple(ops.into_boxed_slice());
        }
        let pc = self.insts.len();
        self.nesting += 1;
        self.depth = self.depth.max(self.nesting);
        self.node(n);
        self.nesting -= 1;
        self.insts.push(Inst::End);
        Atom::Code(pc)
    }

    /// A sequence: runs of one-way nodes as one instruction each, a
    /// one-step repetition's minimum in the run before it, and the one-way
    /// steps after a one-way capture checked before its span is pushed.
    fn seq(&mut self, items: &[Node]) {
        let mut run = Vec::new();
        let mut k = 0;
        while k < items.len() {
            let item = &items[k];
            k += 1;
            if let Some(ops) = simple(item) {
                run.extend(ops);
                continue;
            }
            if let Some((op, min, rest)) = split_min(item) {
                run.push(SOp::Times(min, Box::new(op)));
                self.simple(std::mem::take(&mut run));
                self.push_next(rest);
                continue;
            }
            self.simple(std::mem::take(&mut run));
            let Node::Group(Some(g), body) = item else {
                self.node(item);
                continue;
            };
            let Some(ops) = simple(body) else {
                self.node(item);
                continue;
            };
            let mut then = Vec::new();
            let mut rest = None;
            while let Some(next) = items.get(k) {
                if let Some(ops) = simple(next) {
                    then.extend(ops);
                } else if let Some((op, min, r)) = split_min(next) {
                    then.push(SOp::Times(min, Box::new(op)));
                    rest = Some(r);
                    k += 1;
                    break;
                } else {
                    break;
                }
                k += 1;
            }
            self.insts.push(Inst::CaptureSimple {
                g: *g,
                ops: ops.into_boxed_slice(),
                then: then.into_boxed_slice(),
            });
            if let Some(r) = rest {
                self.push_next(r);
            }
        }
        self.simple(run);
    }

    fn node(&mut self, n: &Node) {
        if let Some(ops) = simple(n) {
            self.simple(ops);
            return;
        }
        match n {
            Node::Seq(items) => self.seq(items),
            Node::Group(None, body) => self.node(body),
            Node::Group(Some(g), body) => match simple(body) {
                Some(ops) => self.insts.push(Inst::CaptureSimple {
                    g: *g,
                    ops: ops.into_boxed_slice(),
                    then: Box::new([]),
                }),
                None => {
                    let reg = self.reg(1);
                    self.insts.push(Inst::GroupOpen { reg });
                    self.node(body);
                    self.insts.push(Inst::GroupClose { g: *g, reg });
                }
            },
            Node::Alt(alts, firsts) => {
                let at = self.placeholder();
                let mut pcs = Vec::with_capacity(alts.len());
                let mut jumps = Vec::with_capacity(alts.len());
                for a in alts {
                    pcs.push(self.insts.len());
                    self.node(a);
                    jumps.push(self.placeholder());
                }
                let end = self.insts.len();
                for j in jumps {
                    self.insts[j] = Inst::Jmp(end);
                }
                let firsts = (0..alts.len())
                    .map(|k| firsts.get(k).cloned().flatten())
                    .collect();
                self.insts[at] = Inst::Alt(Branches::new(pcs, firsts));
            }
            Node::LookAhead { negate, body } => {
                let at = self.placeholder();
                let body = self.atom(body);
                self.insts[at] = Inst::Look {
                    negate: *negate,
                    body,
                    next: self.insts.len(),
                };
            }
            Node::LookBehind {
                negate,
                body,
                min,
                max,
                by_code_point,
            } => {
                let at = self.placeholder();
                let body = self.atom(body);
                self.insts[at] = Inst::Behind {
                    negate: *negate,
                    body,
                    min: *min,
                    max: *max,
                    by_code_point: *by_code_point,
                    next: self.insts.len(),
                };
            }
            Node::Atomic(body) => {
                let at = self.placeholder();
                let body = self.atom(body);
                self.insts[at] = Inst::Atomic {
                    body,
                    next: self.insts.len(),
                };
            }
            Node::Repeat {
                atom,
                min,
                max,
                greed,
                mode,
                capture,
            } => {
                let at = self.placeholder();
                let (min, max, greed) = (*min, *max, *greed);
                self.insts[at] = match mode {
                    RepMode::First => {
                        let atom = self.atom(atom);
                        let unit = matches!(&atom, Atom::Simple(ops) if matches!(&ops[..], [SOp::Char(cs)] if cs.bmp));
                        match &atom {
                            // A one-step atom's minimum is a run of steps
                            // before the repetition (which then fails
                            // before it pushes anything).
                            Atom::Simple(ops)
                                if min > 0
                                    && ops.len() == 1
                                    && !matches!(ops[0], SOp::Times(..)) =>
                            {
                                let once = ops[0].clone();
                                self.insts.push(Inst::Curly {
                                    atom,
                                    min: 0,
                                    max: max - min,
                                    greed,
                                    unit,
                                    next: at + 2,
                                });
                                Inst::Simple(Box::new([SOp::Times(min, Box::new(once))]))
                            }
                            _ => Inst::Curly {
                                atom,
                                min,
                                max,
                                greed,
                                unit,
                                next: self.insts.len(),
                            },
                        }
                    }
                    RepMode::Ques => Inst::Ques {
                        atom: self.atom(atom),
                        greed,
                        next: self.insts.len(),
                    },
                    RepMode::GroupCurly => {
                        self.on_success |= capture.is_some();
                        Inst::GroupCurly {
                            body: self.atom(atom),
                            min,
                            max,
                            greed,
                            capture: *capture,
                            next: self.insts.len(),
                        }
                    }
                    RepMode::Loop => {
                        let reg = self.reg(2);
                        self.node(atom);
                        self.insts.push(Inst::LoopTail { init: at });
                        Inst::LoopInit {
                            min,
                            max,
                            lazy: greed == Greed::Lazy,
                            memo: *capture,
                            reg,
                            next: self.insts.len(),
                        }
                    }
                };
            }
            Node::LineEnding => self.insts.push(Inst::LineEnding),
            // One-way nodes are `simple` above.
            _ => {}
        }
    }
}

// ------------------------------------------------------------ one-way steps

/// Where the run of one-way steps `ops` at `i` ends.
#[inline(always)]
fn run_simple(ops: &[SOp], i: usize, st: &State<'_>) -> Option<usize> {
    let mut j = i;
    for op in ops {
        j = match op {
            SOp::Times(n, op) => {
                for _ in 0..*n {
                    j = step(op, j, st)?;
                }
                j
            }
            _ => step(op, j, st)?,
        };
    }
    Some(j)
}

/// One step other than a repetition (which holds no repetition).
#[inline(always)]
fn step(op: &SOp, i: usize, st: &State<'_>) -> Option<usize> {
    match op {
        SOp::Char(cs) => char_step(cs, i, st),
        SOp::Slice(sets) => {
            let mut j = i;
            for set in sets.iter() {
                j = char_step_cp(set, j, st)?;
            }
            Some(j)
        }
        SOp::BackRef { group, ci } => backref_step(*group, *ci, i, st),
        SOp::Assert(a) => assertion(*a, i, st).then_some(i),
        SOp::Times(..) => None,
    }
}

/// Where one character of `cs` at `i` ends.
#[inline]
fn char_step(cs: &CharSet, i: usize, st: &State<'_>) -> Option<usize> {
    let u = *st.text.get(i)?;
    if cs.bmp || !(0xD800..=0xDBFF).contains(&u) {
        return cs.contains(u32::from(u)).then_some(i + 1);
    }
    let c = st.cp(i);
    cs.contains(c).then(|| i + char_count(c))
}

/// One code point of `cs` at `i` (a literal of a run reads code points).
#[inline]
fn char_step_cp(cs: &CharSet, i: usize, st: &State<'_>) -> Option<usize> {
    let u = *st.text.get(i)?;
    if !(0xD800..=0xDBFF).contains(&u) {
        return cs.contains(u32::from(u)).then_some(i + 1);
    }
    let c = st.cp(i);
    cs.contains(c).then(|| i + char_count(c))
}

/// Where a backreference to `group` at `i` ends.
fn backref_step(group: usize, ci: Option<bool>, i: usize, st: &State<'_>) -> Option<usize> {
    let (s, e) = st.group(group);
    if s < 0 {
        return None;
    }
    let (s, size) = (s as usize, (e - s) as usize);
    let Some(ci) = ci else {
        // Element by element: these are short, and `memcmp` costs a call.
        let (Some(here), Some(there)) = (st.text.get(i..i + size), st.text.get(s..s + size)) else {
            return None;
        };
        return here
            .iter()
            .zip(there)
            .all(|(a, b)| a == b)
            .then_some(i + size);
    };
    backref_ci(s, size, ci, i, st)
}

/// [`backref_step`] without regard to case.
fn backref_ci(s: usize, size: usize, unicode: bool, i: usize, st: &State<'_>) -> Option<usize> {
    let len = st.len();
    if i + size > len {
        return None;
    }
    let (mut x, mut j) = (i, s);
    for _ in 0..size {
        if x >= len || j >= len {
            return None;
        }
        let (c1, c2) = (st.cp(x), st.cp(j));
        if c1 != c2 {
            if unicode {
                let (u1, u2) = (to_upper_case(c1), to_upper_case(c2));
                if u1 != u2 && to_lower_case(u1) != to_lower_case(u2) {
                    return None;
                }
            } else if ascii_lower(c1) != ascii_lower(c2) {
                return None;
            }
        }
        x += char_count(c1);
        j += char_count(c2);
    }
    Some(i + size)
}

fn ascii_lower(c: u32) -> u32 {
    if (0x41..=0x5A).contains(&c) {
        c + 0x20
    } else {
        c
    }
}

/// A zero-width assertion at `i`.
fn assertion(a: Assert, i: usize, st: &State<'_>) -> bool {
    let len = st.len();
    match a {
        Assert::Begin => i == 0,
        Assert::End => i == len,
        Assert::Caret { unix } => {
            if i == len {
                false
            } else if i == 0 {
                true
            } else {
                let c = st.text[i - 1];
                if unix {
                    c == 0x0A
                } else {
                    line_terminator(c) && !(c == 0x0D && st.text[i] == 0x0A)
                }
            }
        }
        Assert::Dollar { multiline, unix } => {
            if unix {
                i >= len || (st.text[i] == 0x0A && (multiline || i == len - 1))
            } else if !multiline
                && (i + 2 < len || (i + 2 == len && (st.text[i] != 0x0D || st.text[i + 1] != 0x0A)))
            {
                false
            } else if i < len {
                let c = st.text[i];
                if c == 0x0A {
                    !(i > 0 && st.text[i - 1] == 0x0D)
                } else {
                    line_terminator(c)
                }
            } else {
                true
            }
        }
        Assert::Bound { not, unicode } => {
            let word = |c: u32, at: usize| {
                is_word(c, unicode)
                    || (c >= 0x300 && get_type(c) == jc::NON_SPACING_MARK && has_base(st, at))
            };
            // An ASCII unit is a word character by `\w`'s ASCII rule, which
            // `(?U)`'s agrees with there.
            let left = i > 0
                && match st.text[i - 1] {
                    u @ 0..=0x7F => is_word(u32::from(u), false),
                    _ => word(st.cp_before(i), i - 1),
                };
            let right = i < len
                && match st.text[i] {
                    u @ 0..=0x7F => is_word(u32::from(u), false),
                    _ => word(st.cp(i), i),
                };
            (left != right) != not
        }
        Assert::LastMatch => i == st.old_last,
    }
}

// ---------------------------------------------------------------- machine

fn pc32(pc: usize) -> u32 {
    u32::try_from(pc).unwrap_or(u32::MAX)
}

/// Where to go on: an instruction and a position, or `None` to backtrack.
type Go = Option<(usize, usize)>;

impl Code {
    fn push(st: &mut State<'_>, e: Entry) {
        st.scratch.stack.push(e);
    }

    fn set_reg(st: &mut State<'_>, r: usize, v: usize) {
        let old = st.scratch.regs[r];
        Self::push(st, Entry::Reg { r: pc32(r), v: old });
        st.scratch.regs[r] = v;
    }

    fn save_group(st: &mut State<'_>, g: usize) {
        let (s, e) = st.group(g);
        Self::push(st, Entry::Group { g: pc32(g), s, e });
    }

    /// The first match of `atom` at `i`: where it ends.
    #[inline]
    fn first(
        &self,
        st: &mut State<'_>,
        atom: &Atom,
        i: usize,
    ) -> Result<Option<usize>, AnalysisError> {
        match atom {
            Atom::Simple(ops) => Ok(run_simple(ops, i, st)),
            Atom::Code(pc) => self.run(st, *pc, i, None),
        }
    }

    /// Runs from `pc` at `i` to the first [`Inst::End`] reached (at
    /// `target`, when given): where it is, or `None`. A success keeps the
    /// captures it set (its entries are dropped, unpopped); a failure has
    /// restored them.
    pub(crate) fn run(
        &self,
        st: &mut State<'_>,
        pc: usize,
        i: usize,
        target: Option<usize>,
    ) -> Result<Option<usize>, AnalysisError> {
        Ok(self.exec(st, pc, i, target, None)?.map(|(_, e)| e))
    }

    /// [`Self::run`]; with `starts`, from every start position the search
    /// tries (the program from its first instruction at each) until one
    /// matches: where it started and ended.
    fn exec(
        &self,
        st: &mut State<'_>,
        mut pc: usize,
        mut i: usize,
        target: Option<usize>,
        starts: Option<&Starts<'_>>,
    ) -> Result<Option<(usize, usize)>, AnalysisError> {
        let mut start = i;
        let base = st.scratch.stack.len();
        let pbase = st.scratch.positions.len();
        loop {
            let go: Go = match &self.insts[pc] {
                Inst::Simple(ops) => run_simple(ops, i, st).map(|j| (pc + 1, j)),
                Inst::CaptureSimple { g, ops, then } => match run_simple(ops, i, st) {
                    Some(j) => {
                        let (s, e) = st.group(*g);
                        st.set_group(*g, i as i32, j as i32);
                        match run_simple(then, j, st) {
                            Some(k) => {
                                Self::push(st, Entry::Group { g: pc32(*g), s, e });
                                Some((pc + 1, k))
                            }
                            None => {
                                // What failed was one-way: no entry needed.
                                st.set_group(*g, s, e);
                                None
                            }
                        }
                    }
                    None => None,
                },
                Inst::GroupOpen { reg } => {
                    Self::set_reg(st, *reg, i);
                    Some((pc + 1, i))
                }
                Inst::GroupClose { g, reg } => {
                    let s = st.scratch.regs[*reg];
                    Self::save_group(st, *g);
                    st.set_group(*g, s as i32, i as i32);
                    Some((pc + 1, i))
                }
                Inst::Jmp(t) => Some((*t, i)),
                Inst::Alt(_) => self.alt(st, pc, 0, i),
                Inst::Look { negate, body, next } => {
                    let matched = self.first(st, body, i)?.is_some();
                    (matched != *negate).then_some((*next, i))
                }
                Inst::Behind {
                    negate,
                    body,
                    min,
                    max,
                    by_code_point,
                    next,
                } => {
                    let matched = self.behind(st, body, i, *min, *max, *by_code_point)?;
                    (matched != *negate).then_some((*next, i))
                }
                Inst::Atomic { body, next } => self.first(st, body, i)?.map(|e| (*next, e)),
                Inst::Curly { .. } => self.curly(st, pc, i)?,
                Inst::Ques { atom, greed, next } => match greed {
                    Greed::Greedy => match self.first(st, atom, i)? {
                        Some(e) => {
                            Self::push(st, Entry::Resume { pc: pc32(*next), i });
                            Some((*next, e))
                        }
                        None => Some((*next, i)),
                    },
                    Greed::Lazy => {
                        Self::push(st, Entry::QuesLazy { pc: pc32(pc), i });
                        Some((*next, i))
                    }
                    Greed::Possessive => Some((*next, self.first(st, atom, i)?.unwrap_or(i))),
                },
                Inst::GroupCurly { .. } => self.group_curly(st, pc, i)?,
                Inst::LoopInit {
                    min,
                    max,
                    lazy,
                    reg,
                    next,
                    ..
                } => {
                    if 0 < *min {
                        Some(self.loop_enter(st, pc, *reg, 1, i))
                    } else if *lazy {
                        Self::push(st, Entry::LoopLazyFirst { pc: pc32(pc), i });
                        Some((*next, i))
                    } else if 0 < *max {
                        Self::push(st, Entry::Resume { pc: pc32(*next), i });
                        Some(self.loop_enter(st, pc, *reg, 1, i))
                    } else {
                        Some((*next, i))
                    }
                }
                Inst::LoopTail { init } => {
                    // Only a loop's iterations grow the stack without bound
                    // (each instruction pushes a few entries at most).
                    if st.scratch.stack.len() > MAX_ENTRIES {
                        return Err(overflow());
                    }
                    Some(self.loop_tail(st, *init, i))
                }
                Inst::LineEnding => {
                    let len = st.len();
                    match st.text.get(i).copied() {
                        Some(0x0A | 0x0B | 0x0C | 0x85 | 0x2028 | 0x2029) => Some((pc + 1, i + 1)),
                        Some(0x0D) => {
                            if i + 1 < len && st.text[i + 1] == 0x0A {
                                Self::push(
                                    st,
                                    Entry::Resume {
                                        pc: pc32(pc + 1),
                                        i: i + 1,
                                    },
                                );
                                Some((pc + 1, i + 2))
                            } else {
                                Some((pc + 1, i + 1))
                            }
                        }
                        _ => None,
                    }
                }
                Inst::End => {
                    if target.is_none_or(|t| t == i) {
                        self.succeed(st, base, pbase);
                        return Ok(Some((start, i)));
                    }
                    None
                }
            };
            match go {
                Some((p, j)) => (pc, i) = (p, j),
                None => match self.backtrack(st, base)? {
                    Some((p, j)) => (pc, i) = (p, j),
                    None => {
                        st.scratch.positions.truncate(pbase);
                        match starts.and_then(|sx| sx.after(st, start)) {
                            Some(next) => (start, pc, i) = (next, 0, next),
                            None => return Ok(None),
                        }
                    }
                },
            }
        }
    }

    /// A (sub-)match succeeded: the spans its `GroupCurly`s record again,
    /// innermost first, then its entries dropped.
    fn succeed(&self, st: &mut State<'_>, base: usize, pbase: usize) {
        if self.on_success {
            let mut k = st.scratch.stack.len();
            while k > base {
                k -= 1;
                if let Entry::OnSuccess { g, s, e } = st.scratch.stack[k] {
                    st.set_group(g as usize, s, e);
                }
            }
        }
        st.scratch.stack.truncate(base);
        st.scratch.positions.truncate(pbase);
    }

    /// Pops entries above `base` until one resumes: where.
    fn backtrack(&self, st: &mut State<'_>, base: usize) -> Result<Go, AnalysisError> {
        while st.scratch.stack.len() > base {
            let Some(entry) = st.scratch.stack.pop() else {
                break;
            };
            let go = match entry {
                Entry::Group { g, s, e } => {
                    st.set_group(g as usize, s, e);
                    None
                }
                Entry::Reg { r, v } => {
                    st.scratch.regs[r as usize] = v;
                    None
                }
                Entry::OnSuccess { .. } => None,
                Entry::Resume { pc, i } => Some((pc as usize, i)),
                Entry::AltNext { pc, b, i } => self.alt(st, pc as usize, b as usize, i),
                Entry::CurlyUnit { next, lo, cur } => {
                    let c = cur - 1;
                    if c > lo {
                        Self::push(st, Entry::CurlyUnit { next, lo, cur: c });
                    }
                    Some((next as usize, c))
                }
                Entry::CurlyBack { next, base, top } => {
                    let p = &mut st.scratch.positions;
                    p.truncate(top as usize);
                    let pos = p.pop().unwrap_or(0);
                    if top - 1 > base {
                        Self::push(
                            st,
                            Entry::CurlyBack {
                                next,
                                base,
                                top: top - 1,
                            },
                        );
                    }
                    Some((next as usize, pos))
                }
                Entry::CurlyLazy { pc, pos, count } => {
                    let Inst::Curly {
                        atom, max, next, ..
                    } = &self.insts[pc as usize]
                    else {
                        continue;
                    };
                    if count >= *max {
                        None
                    } else {
                        match self.first(st, atom, pos)? {
                            Some(e) if e != pos => {
                                Self::push(
                                    st,
                                    Entry::CurlyLazy {
                                        pc,
                                        pos: e,
                                        count: count + 1,
                                    },
                                );
                                Some((*next, e))
                            }
                            _ => None,
                        }
                    }
                }
                Entry::QuesLazy { pc, i } => {
                    let Inst::Ques { atom, next, .. } = &self.insts[pc as usize] else {
                        continue;
                    };
                    self.first(st, atom, i)?.map(|e| (*next, e))
                }
                Entry::GroupBack { pc, base, top } => {
                    Some(self.group_back(st, pc as usize, base as usize, top as usize))
                }
                Entry::GroupLazy { pc, pos, count } => {
                    let Inst::GroupCurly {
                        body,
                        max,
                        capture,
                        next,
                        ..
                    } = &self.insts[pc as usize]
                    else {
                        continue;
                    };
                    if count >= *max {
                        None
                    } else {
                        match self.first(st, body, pos)? {
                            Some(e) if e != pos => {
                                if let Some(g) = capture {
                                    st.set_group(*g, pos as i32, e as i32);
                                }
                                Self::push(
                                    st,
                                    Entry::GroupLazy {
                                        pc,
                                        pos: e,
                                        count: count + 1,
                                    },
                                );
                                Some((*next, e))
                            }
                            _ => None,
                        }
                    }
                }
                Entry::LoopLazyFirst { pc, i } => {
                    let Inst::LoopInit { max, reg, .. } = &self.insts[pc as usize] else {
                        continue;
                    };
                    (0 < *max).then(|| self.loop_enter(st, pc as usize, *reg, 1, i))
                }
                Entry::LoopLazyNext { pc, i, count } => {
                    let Inst::LoopInit { max, reg, .. } = &self.insts[pc as usize] else {
                        continue;
                    };
                    let count = count as usize;
                    (count < *max as usize)
                        .then(|| self.loop_enter(st, pc as usize, *reg, count + 1, i))
                }
                Entry::LoopExit { pc, i } => {
                    let Inst::LoopInit { memo, next, .. } = &self.insts[pc as usize] else {
                        continue;
                    };
                    if let Some(id) = memo {
                        st.scratch.failed.insert((*id, i));
                    }
                    Some((*next, i))
                }
            };
            if go.is_some() {
                return Ok(go);
            }
        }
        Ok(None)
    }

    /// The `Alt` at `pc` from its `b`th alternative at `i`: the first one
    /// the text's next `char` does not rule out, with the next such one
    /// pushed.
    #[inline(always)]
    fn alt(&self, st: &mut State<'_>, pc: usize, b: usize, i: usize) -> Go {
        let Inst::Alt(br) = &self.insts[pc] else {
            return None;
        };
        let unit = st.text.get(i).copied();
        if let (Some(table), Some(u @ 0..=127)) = (&br.ascii, unit) {
            let mask = table[usize::from(u)] & u64::MAX.checked_shl(b as u32).unwrap_or(0);
            if mask == 0 {
                return None;
            }
            let rest = mask & (mask - 1);
            if rest != 0 {
                Self::push(
                    st,
                    Entry::AltNext {
                        pc: pc32(pc),
                        b: rest.trailing_zeros(),
                        i,
                    },
                );
            }
            return Some((br.pcs[mask.trailing_zeros() as usize], i));
        }
        self.alt_slow(st, pc, br, b, i)
    }

    /// [`Self::alt`] past ASCII (or with more than 64 alternatives).
    fn alt_slow(&self, st: &mut State<'_>, pc: usize, br: &Branches, b: usize, i: usize) -> Go {
        let unit = st.text.get(i).copied();
        // A non-surrogate `char` an alternative cannot start with.
        let unit = unit.filter(|u| !(0xD800..=0xDFFF).contains(u));
        let at_end = i >= st.len();
        let viable = |k: usize| match (&br.firsts[k], unit) {
            (Some(set), Some(u)) => set.contains(u32::from(u)),
            (Some(_), None) => !at_end,
            (None, _) => true,
        };
        let n = br.pcs.len();
        let first = (b..n).find(|&k| viable(k))?;
        if let Some(later) = (first + 1..n).find(|&k| viable(k)) {
            Self::push(
                st,
                Entry::AltNext {
                    pc: pc32(pc),
                    b: pc32(later),
                    i,
                },
            );
        }
        Some((br.pcs[first], i))
    }

    /// Lookbehind: a match of `body` from `i - min` back to `i - max` that
    /// ends at `i` (Java's `int` arithmetic: `i - rmax` may wrap).
    fn behind(
        &self,
        st: &mut State<'_>,
        body: &Atom,
        i: usize,
        min: i32,
        max: i32,
        by_code_point: bool,
    ) -> Result<bool, AnalysisError> {
        let ii = i as i32;
        let (from, start) = if by_code_point {
            let rmax_chars = count_chars(st, ii, max.wrapping_neg());
            let rmin_chars = count_chars(st, ii, min.wrapping_neg());
            (
                ii.wrapping_sub(rmax_chars).max(0),
                ii.wrapping_sub(rmin_chars),
            )
        } else {
            (ii.wrapping_sub(max).max(0), ii.wrapping_sub(min))
        };
        let mut j = start;
        while j >= from && j <= ii {
            let matched = match body {
                Atom::Simple(ops) => run_simple(ops, j as usize, st) == Some(i),
                Atom::Code(pc) => self.run(st, *pc, j as usize, Some(i))?.is_some(),
            };
            if matched {
                return Ok(true);
            }
            j -= if by_code_point && j > from {
                count_chars(st, j, -1)
            } else {
                1
            };
        }
        Ok(false)
    }

    /// `Curly`/`Ques` over the atom's first match.
    fn curly(&self, st: &mut State<'_>, pc: usize, i: usize) -> Result<Go, AnalysisError> {
        let Inst::Curly {
            atom,
            min,
            max,
            greed,
            unit,
            next,
        } = &self.insts[pc]
        else {
            return Ok(None);
        };
        let (min, max, next) = (*min, *max, *next);
        let mut pos = i;
        for _ in 0..min {
            match self.first(st, atom, pos)? {
                Some(e) => pos = e,
                None => return Ok(None),
            }
        }
        let mut count = min;
        Ok(Some(match greed {
            Greed::Greedy if *unit => {
                // One BMP set: one unit per repetition, the back-off by
                // arithmetic.
                if let Atom::Simple(ops) = atom {
                    if let [SOp::Char(cs)] = &ops[..] {
                        let lo = pos;
                        let room = (max - count) as usize;
                        let stop = st.len().min(pos.saturating_add(room));
                        while pos < stop && cs.contains(u32::from(st.text[pos])) {
                            pos += 1;
                        }
                        if pos > lo {
                            Self::push(
                                st,
                                Entry::CurlyUnit {
                                    next: pc32(next),
                                    lo,
                                    cur: pos,
                                },
                            );
                        }
                    }
                }
                (next, pos)
            }
            Greed::Greedy => {
                // The positions to back off through.
                let base = st.scratch.positions.len();
                st.scratch.positions.push(pos);
                while count < max {
                    match self.first(st, atom, pos)? {
                        Some(e) if e != pos => {
                            pos = e;
                            st.scratch.positions.push(e);
                            count += 1;
                            if st.scratch.positions.len() > MAX_ENTRIES {
                                return Err(overflow());
                            }
                        }
                        _ => break,
                    }
                }
                // The last is tried now, the others on backtracking.
                let top = st.scratch.positions.len() - 1;
                st.scratch.positions.truncate(top);
                if top > base {
                    Self::push(
                        st,
                        Entry::CurlyBack {
                            next: pc32(next),
                            base: pc32(base),
                            top: pc32(top),
                        },
                    );
                }
                (next, pos)
            }
            Greed::Lazy => {
                Self::push(
                    st,
                    Entry::CurlyLazy {
                        pc: pc32(pc),
                        pos,
                        count,
                    },
                );
                (next, pos)
            }
            Greed::Possessive => {
                while count < max {
                    match self.first(st, atom, pos)? {
                        Some(e) if e != pos => {
                            pos = e;
                            count += 1;
                        }
                        _ => break,
                    }
                }
                (next, pos)
            }
        }))
    }

    /// `GroupCurly`: a deterministic group's body repeated by its first
    /// match, each iteration's span recorded in its capture.
    fn group_curly(&self, st: &mut State<'_>, pc: usize, i: usize) -> Result<Go, AnalysisError> {
        let Inst::GroupCurly {
            body,
            min,
            max,
            greed,
            capture,
            next,
        } = &self.insts[pc]
        else {
            return Ok(None);
        };
        let (min, max, next, capture) = (*min, *max, *next, *capture);
        if let Some(g) = capture {
            // Restored when everything after fails.
            Self::save_group(st, g);
        }
        let mut pos = i;
        let mut count = 0;
        while count < min {
            match self.first(st, body, pos)? {
                Some(e) => {
                    if let Some(g) = capture {
                        st.set_group(g, pos as i32, e as i32);
                    }
                    pos = e;
                    count += 1;
                }
                None => return Ok(None),
            }
        }
        if *greed == Greed::Lazy {
            Self::push(
                st,
                Entry::GroupLazy {
                    pc: pc32(pc),
                    pos,
                    count,
                },
            );
            return Ok(Some((next, pos)));
        }
        let at_min = capture.map_or((-1, -1), |g| st.group(g));
        let base = st.scratch.positions.len();
        st.scratch
            .positions
            .extend([pos, at_min.0 as u32 as usize, at_min.1 as u32 as usize]);
        while count < max {
            match self.first(st, body, pos)? {
                Some(e) if e > pos => {
                    st.scratch.positions.extend([pos, e]);
                    pos = e;
                    count += 1;
                    if st.scratch.positions.len() > MAX_ENTRIES {
                        return Err(overflow());
                    }
                }
                _ => break,
            }
        }
        let top = st.scratch.positions.len();
        Ok(Some(self.group_back(st, pc, base, top)))
    }

    /// A greedy `GroupCurly`'s next back-off: the last span left, recorded
    /// (and again on success); past the first, its start with the capture
    /// it had there.
    fn group_back(&self, st: &mut State<'_>, pc: usize, base: usize, top: usize) -> (usize, usize) {
        let (capture, next) = match &self.insts[pc] {
            Inst::GroupCurly { capture, next, .. } => (*capture, *next),
            _ => (None, pc + 1),
        };
        let p = &mut st.scratch.positions;
        p.truncate(top);
        if top >= base + 5 {
            let (s, e) = (p[top - 2], p[top - 1]);
            p.truncate(top - 2);
            Self::push(
                st,
                Entry::GroupBack {
                    pc: pc32(pc),
                    base: pc32(base),
                    top: pc32(top - 2),
                },
            );
            if let Some(g) = capture {
                st.set_group(g, s as i32, e as i32);
                Self::push(
                    st,
                    Entry::OnSuccess {
                        g: pc32(g),
                        s: s as i32,
                        e: e as i32,
                    },
                );
            }
            return (next, e);
        }
        let (start, s, e) = (
            p[base],
            p[base + 1] as u32 as i32,
            p[base + 2] as u32 as i32,
        );
        p.truncate(base);
        if let Some(g) = capture {
            st.set_group(g, s, e);
        }
        (next, start)
    }

    /// Starts iteration `count` of the `Loop` at `pc` from `i`.
    fn loop_enter(
        &self,
        st: &mut State<'_>,
        pc: usize,
        reg: usize,
        count: usize,
        i: usize,
    ) -> (usize, usize) {
        Self::set_reg(st, reg, count);
        Self::set_reg(st, reg + 1, i);
        (pc + 1, i)
    }

    /// `Loop.match`: the iteration under way ended at `i`.
    fn loop_tail(&self, st: &mut State<'_>, init: usize, i: usize) -> (usize, usize) {
        let Inst::LoopInit {
            min,
            max,
            lazy,
            memo,
            reg,
            next,
        } = &self.insts[init]
        else {
            return (init + 1, i);
        };
        let (count, begin) = (st.scratch.regs[*reg], st.scratch.regs[*reg + 1]);
        if i > begin {
            if count < *min as usize {
                return self.loop_enter(st, init, *reg, count + 1, i);
            }
            if *lazy {
                Self::push(
                    st,
                    Entry::LoopLazyNext {
                        pc: pc32(init),
                        i,
                        count: pc32(count),
                    },
                );
                return (*next, i);
            }
            if count < *max as usize {
                // `posIndex`: another iteration from here already failed.
                if memo.is_some_and(|id| st.scratch.failed.contains(&(id, i))) {
                    return (*next, i);
                }
                Self::push(st, Entry::LoopExit { pc: pc32(init), i });
                return self.loop_enter(st, init, *reg, count + 1, i);
            }
        }
        (*next, i)
    }

    /// Empty stacks and memo, and registers, for a match (a failed one may
    /// have left entries behind).
    fn prepare(&self, st: &mut State<'_>) {
        st.scratch.clear();
        if st.scratch.regs.len() != self.regs {
            st.scratch.regs.resize(self.regs, 0);
        }
    }
}

// ----------------------------------------------------------- the searches

impl Program {
    /// `Start`/`StartS`: the first match at or after `from`; its groups in
    /// `st`.
    pub(crate) fn search(&self, from: usize, st: &mut State<'_>) -> Result<bool, AnalysisError> {
        self.on_stack(st, |p, st| p.search_here(from, st))
    }

    /// `matches()`: the whole text.
    pub(crate) fn matches_all(&self, st: &mut State<'_>) -> Result<bool, AnalysisError> {
        self.on_stack(st, |p, st| {
            let len = st.len();
            p.code.prepare(st);
            if p.code.run(st, 0, 0, Some(len))?.is_some() {
                st.set_group(0, 0, len as i32);
                return Ok(true);
            }
            Ok(false)
        })
    }

    /// Runs `f` here, or -- for a pattern whose sub-programs nest deeper
    /// than the caller's stack is trusted with -- on a deep stack.
    fn on_stack<'t>(
        &self,
        st: &mut State<'t>,
        f: impl Fn(&Program, &mut State<'t>) -> Result<bool, AnalysisError> + Sync,
    ) -> Result<bool, AnalysisError> {
        if self.code.depth <= super::SHALLOW_NESTING {
            return f(self, st);
        }
        super::on_deep_stack(|| f(self, st)).unwrap_or_else(|| Err(overflow()))
    }

    fn search_here(&self, from: usize, st: &mut State<'_>) -> Result<bool, AnalysisError> {
        let Some(guard) = st.len().checked_sub(self.min_len) else {
            return Ok(false);
        };
        self.code.prepare(st);
        let starts = Starts {
            first: self.root_first.as_ref(),
            supplementary: self.has_supplementary,
            guard,
        };
        let Some(i) = starts.from(st, from) else {
            return Ok(false);
        };
        match self.code.exec(st, 0, i, None, Some(&starts))? {
            Some((s, e)) => {
                st.set_group(0, s as i32, e as i32);
                Ok(true)
            }
            None => Ok(false),
        }
    }
}

/// The positions a search starts from.
struct Starts<'p> {
    /// The characters every match starts with, when known.
    first: Option<&'p CharSet>,
    /// `StartS`: the low half of a pair is not a start.
    supplementary: bool,
    /// The last start a match fits after (`len - minLength`).
    guard: usize,
}

impl Starts<'_> {
    /// The first start at or after `i`: not one whose `char` no match
    /// starts with (the attempt would fail at once).
    #[inline]
    fn from(&self, st: &State<'_>, mut i: usize) -> Option<usize> {
        while i <= self.guard {
            let Some(first) = self.first else {
                return Some(i);
            };
            let u = st.text.get(i).copied().unwrap_or(0xD800);
            if (0xD800..=0xDFFF).contains(&u) || first.contains(u32::from(u)) {
                return Some(i);
            }
            i += 1;
        }
        None
    }

    /// The start after a failed one at `i`.
    #[inline]
    fn after(&self, st: &State<'_>, i: usize) -> Option<usize> {
        let len = st.len();
        let pair = self.supplementary
            && i + 1 < len
            && (0xD800..=0xDBFF).contains(&st.text[i])
            && (0xDC00..=0xDFFF).contains(&st.text[i + 1]);
        self.from(st, if pair { i + 2 } else { i + 1 })
    }
}
