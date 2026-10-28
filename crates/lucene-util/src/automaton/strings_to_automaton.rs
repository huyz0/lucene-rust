//! `StringsToAutomaton`: the minimal deterministic automaton for a sorted
//! set of strings, built incrementally (Daciuk, Mihov, Watson & Watson,
//! "Incremental Construction of Minimal Acyclic Finite-State Automata").
//!
//! Java's graph of `State` objects becomes an arena of nodes; the register of
//! frozen states is keyed by what Java's `equals` compares (finality, labels
//! and child *identity*), and a registered node is never modified again.

use std::hash::{Hash, Hasher};

use super::automata::MAX_STRING_UNION_TERM_LENGTH;
use super::automaton::{Automaton, Builder};
use super::error::AutomatonError;
use crate::fx_hash::{FxHashMap, FxHasher};

/// A node: its outgoing `(label, child)` edges in label order (one `Vec`,
/// so one allocation per node with children) and its finality.
#[derive(Default)]
struct Node {
    children: Vec<(i32, usize)>,
    is_final: bool,
}

impl Node {
    /// The register key Java's `State.equals`/`hashCode` use: finality,
    /// labels and child identity.
    fn key_hash(&self) -> u64 {
        let mut h = FxHasher::default();
        self.is_final.hash(&mut h);
        self.children.hash(&mut h);
        h.finish()
    }

    fn same_key(&self, other: &Node) -> bool {
        self.is_final == other.is_final && self.children == other.children
    }
}

/// No next node in a register chain.
const NONE: usize = usize::MAX;

struct StringsBuilder {
    nodes: Vec<Node>,
    /// The register: key hash -> first registered node with that hash; the
    /// rest of the chain runs through `next_registered`. Nodes are compared
    /// in place, so a probe clones nothing.
    registry: FxHashMap<u64, usize>,
    /// Per node: the next registered node with the same key hash.
    next_registered: Vec<usize>,
    /// The previous term (for the order check), reused across terms.
    previous: Vec<u8>,
    has_previous: bool,
    /// Scratch for the current term's labels.
    labels: Vec<i32>,
}

const ROOT: usize = 0;

impl StringsBuilder {
    fn new() -> Self {
        StringsBuilder {
            nodes: vec![Node::default()],
            registry: FxHashMap::default(),
            next_registered: Vec::new(),
            previous: Vec::new(),
            has_previous: false,
            labels: Vec::new(),
        }
    }

    fn last_child_with(&self, s: usize, label: i32) -> Option<usize> {
        let n = &self.nodes[s];
        match n.children.last() {
            Some(&(l, child)) if l == label => Some(child),
            _ => None,
        }
    }

    fn new_state(&mut self, s: usize, label: i32) -> usize {
        let id = self.nodes.len();
        self.nodes.push(Node::default());
        let n = &mut self.nodes[s];
        n.children.push((label, id));
        id
    }

    fn replace_or_register(&mut self, s: usize) {
        let child = self.nodes[s].children.last().expect("has children").1;
        if !self.nodes[child].children.is_empty() {
            self.replace_or_register(child);
        }
        let hash = self.nodes[child].key_hash();
        let head = self.registry.get(&hash).copied().unwrap_or(NONE);
        let mut r = head;
        while r != NONE {
            if self.nodes[r].same_key(&self.nodes[child]) {
                self.nodes[s].children.last_mut().expect("has children").1 = r;
                return;
            }
            r = self.next_registered[r];
        }
        if self.next_registered.len() < self.nodes.len() {
            self.next_registered.resize(self.nodes.len(), NONE);
        }
        self.next_registered[child] = head;
        self.registry.insert(hash, child);
    }

    fn add(&mut self, current: &[u8], as_binary: bool) -> Result<(), AutomatonError> {
        if current.len() > MAX_STRING_UNION_TERM_LENGTH {
            return Err(AutomatonError::IllegalArgument(format!(
                "This builder doesn't allow terms that are larger than {MAX_STRING_UNION_TERM_LENGTH} UTF-8 bytes, got {}",
                bytes_ref_string(current)
            )));
        }
        if self.has_previous && self.previous.as_slice() > current {
            return Err(AutomatonError::IllegalArgument(format!(
                "Input must be in sorted UTF-8 order: {} >= {}",
                bytes_ref_string(&self.previous),
                bytes_ref_string(current)
            )));
        }
        self.previous.clear();
        self.previous.extend_from_slice(current);
        self.has_previous = true;
        let mut labels = std::mem::take(&mut self.labels);
        labels.clear();
        if as_binary {
            labels.extend(current.iter().map(|&b| i32::from(b)));
        } else if let Err(e) = decode_code_points(current, &mut labels) {
            self.labels = labels;
            return Err(e);
        }
        let mut pos = 0;
        let mut state = ROOT;
        while pos < labels.len() {
            match self.last_child_with(state, labels[pos]) {
                Some(next) => {
                    state = next;
                    pos += 1;
                }
                None => break,
            }
        }
        if !self.nodes[state].children.is_empty() {
            self.replace_or_register(state);
        }
        while pos < labels.len() {
            state = self.new_state(state, labels[pos]);
            pos += 1;
        }
        self.nodes[state].is_final = true;
        self.labels = labels;
        Ok(())
    }

    /// `visited[node]` is its converted state, or -1.
    fn convert(&self, b: &mut Builder, s: usize, visited: &mut [i32]) -> i32 {
        if visited[s] != -1 {
            return visited[s];
        }
        let converted = b.create_state();
        b.set_accept(converted, self.nodes[s].is_final);
        visited[s] = converted;
        for &(label, target) in &self.nodes[s].children {
            let dest = self.convert(b, target, visited);
            b.add_transition_label(converted, dest, label);
        }
        converted
    }

    fn complete_and_convert(mut self) -> Automaton {
        if !self.nodes[ROOT].children.is_empty() {
            self.replace_or_register(ROOT);
        }
        let mut b = Builder::new();
        let mut visited = vec![-1i32; self.nodes.len()];
        self.convert(&mut b, ROOT, &mut visited);
        b.finish()
    }
}

/// `BytesRef.toString()`: `[61 62 63]`, lowercase hex bytes.
fn bytes_ref_string(b: &[u8]) -> String {
    let inner: Vec<String> = b.iter().map(|x| format!("{x:x}")).collect();
    format!("[{}]", inner.join(" "))
}

/// `UnicodeUtil.codePointAt` over a whole UTF-8 string: lenient decoding of
/// each lead byte's length, as Java does (no continuation-byte checks).
fn decode_code_points(bytes: &[u8], out: &mut Vec<i32>) -> Result<(), AutomatonError> {
    let mut pos = 0;
    while pos < bytes.len() {
        let lead = i32::from(bytes[pos]);
        let (num, mut v) = match lead {
            0x00..=0x7F => (1, lead),
            0xC0..=0xDF => (2, lead & 31),
            0xE0..=0xEF => (3, lead & 15),
            0xF0..=0xF7 => (4, lead & 7),
            _ => {
                return Err(AutomatonError::IllegalArgument(format!(
                    "Invalid UTF8 header byte: 0x{lead:x}"
                )));
            }
        };
        if pos + num > bytes.len() {
            return Err(AutomatonError::IllegalArgument(
                "truncated UTF-8 sequence".into(),
            ));
        }
        for &b in &bytes[pos + 1..pos + num] {
            v = (v << 6) | (i32::from(b) & 63);
        }
        out.push(v);
        pos += num;
    }
    Ok(())
}

/// `StringsToAutomaton.build(input, asBinary)`: the minimal automaton
/// accepting exactly the given strings, which must arrive in byte order.
/// With `as_binary` the labels are bytes, otherwise the UTF-8 is decoded to
/// code points.
///
/// # Errors
/// `IllegalArgument` for an unsorted input, a string over
/// [`MAX_STRING_UNION_TERM_LENGTH`] bytes, or (code points) an invalid UTF-8
/// lead byte.
pub fn build<'a, I>(input: I, as_binary: bool) -> Result<Automaton, AutomatonError>
where
    I: IntoIterator<Item = &'a [u8]>,
{
    let mut b = StringsBuilder::new();
    for s in input {
        b.add(s, as_binary)?;
    }
    Ok(b.complete_and_convert())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::automaton::operations::run;

    #[test]
    fn builds_minimal_union() {
        let words: Vec<&[u8]> = vec![b"cat", b"cats", b"dog", b"dogs", "\u{e9}t\u{e9}".as_bytes()];
        let a = build(words.iter().copied(), false).unwrap();
        assert!(a.is_deterministic());
        for w in ["cat", "cats", "dog", "dogs", "\u{e9}t\u{e9}"] {
            assert!(run(&a, w), "{w}");
        }
        assert!(!run(&a, "ca") && !run(&a, "catss"));
        // root; "c"/"a" and "d"/"o"; one final state shared by "cat"/"dog"
        // with its 's' child, which is also where "\u{e9}t\u{e9}" ends
        // (final, no children); "\u{e9}" and "\u{e9}t".
        assert_eq!(a.get_num_states(), 9);
        let dup = build([b"a".as_slice(), b"a"], true).unwrap();
        assert_eq!(dup.get_num_states(), 2);
        assert!(build([b"b".as_slice(), b"a"], true).is_err());
        let long = vec![b'x'; 1001];
        assert!(build([long.as_slice()], true).is_err());
        assert!(build([[0xFFu8].as_slice()], false).is_err());
        assert!(build([[0xE2u8].as_slice()], false).is_err());
        assert_eq!(bytes_ref_string(b"ab"), "[61 62]");
    }
}
