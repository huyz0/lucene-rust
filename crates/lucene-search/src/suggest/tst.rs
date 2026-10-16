//! `org.apache.lucene.search.suggest.tst` (M13 T13.3b): `TSTLookup`, a suggester
//! over a ternary search tree, with `TSTAutocomplete`'s insert and prefix
//! completion and `TernaryTreeNode`.
//!
//! The tree splits on UTF-16 code units, as Java's `char` does, so a
//! supplementary character is two nodes. Lookup results come out in
//! `prefixCompletion`'s depth-first order (the high child first), which is
//! not sorted; `onlyMorePopular` ranks them by weight instead.

use std::cmp::Ordering;
use std::collections::BTreeSet;

use lucene_store::{DataInput, DataOutput, Error, Result};

use super::input::{InputIterator, SortedInputIterator};
use super::lookup::{Lookup, LookupPriorityQueue, LookupResult};

const LO_KID: u8 = 0x01;
const EQ_KID: u8 = 0x02;
const HI_KID: u8 = 0x04;
const HAS_TOKEN: u8 = 0x08;
const HAS_VALUE: u8 = 0x10;

/// `TernaryTreeNode`: one UTF-16 unit, three children, and (where a key ends
/// here) the key and its value.
#[derive(Debug, Default, Clone)]
struct Node {
    split: u16,
    lo: Option<Box<Node>>,
    eq: Option<Box<Node>>,
    hi: Option<Box<Node>>,
    token: Option<String>,
    val: Option<i64>,
}

impl Node {
    fn new(split: u16) -> Self {
        Self {
            split,
            ..Self::default()
        }
    }
}

/// `TSTAutocomplete.insert`, iterative so a long key cannot overflow the
/// stack. A new node splits on the unit being placed, as Java's does; an empty
/// key changes nothing.
fn insert(root: &mut Node, s: &[u16], token: &str, val: i64) {
    if s.is_empty() {
        return;
    }
    let mut x = 0;
    let mut cur = root;
    loop {
        let c = s[x];
        match cur.split.cmp(&c) {
            Ordering::Greater => {
                cur = cur.lo.get_or_insert_with(|| Box::new(Node::new(c)));
            }
            Ordering::Equal => {
                if x + 1 < s.len() {
                    x += 1;
                    let next = s[x];
                    cur = cur.eq.get_or_insert_with(|| Box::new(Node::new(next)));
                } else {
                    cur.token = Some(token.to_string());
                    cur.val = Some(val);
                    return;
                }
            }
            Ordering::Less => {
                cur = cur.hi.get_or_insert_with(|| Box::new(Node::new(c)));
            }
        }
    }
}

/// `TSTAutocomplete.balancedTree`: inserts the middle of `lo..=hi` first, then
/// each half, so sorted input builds a balanced tree.
fn balanced_tree(tokens: &[String], vals: &[i64], lo: isize, hi: isize, root: &mut Node) {
    if lo > hi {
        return;
    }
    let mid = (lo + hi) / 2;
    let i = mid as usize;
    let units: Vec<u16> = tokens[i].encode_utf16().collect();
    insert(root, &units, &tokens[i], vals[i]);
    balanced_tree(tokens, vals, lo, mid - 1, root);
    balanced_tree(tokens, vals, mid + 1, hi, root);
}

/// `TSTAutocomplete.prefixCompletion` for a non-empty `s`: the nodes whose
/// token starts with `s`, the node for `s` itself first, then its subtree in
/// the order a stack that pushes `eq`, `lo`, `hi` pops them.
fn prefix_completion<'a>(root: &'a Node, s: &[u16]) -> Vec<&'a Node> {
    let mut suggest = Vec::new();
    let mut p = Some(root);
    let mut x = 0;
    while let Some(n) = p {
        match s[x].cmp(&n.split) {
            Ordering::Less => p = n.lo.as_deref(),
            Ordering::Equal => {
                if x == s.len() - 1 {
                    break;
                }
                x += 1;
                p = n.eq.as_deref();
            }
            Ordering::Greater => p = n.hi.as_deref(),
        }
    }
    let Some(p) = p else {
        return suggest;
    };
    let Some(eq) = p.eq.as_deref() else {
        if p.token.is_some() {
            suggest.push(p);
        }
        return suggest;
    };
    if p.token.is_some() {
        suggest.push(p);
    }
    let mut stack = vec![eq];
    while let Some(top) = stack.pop() {
        if top.token.is_some() {
            suggest.push(top);
        }
        if let Some(k) = top.eq.as_deref() {
            stack.push(k);
        }
        if let Some(k) = top.lo.as_deref() {
            stack.push(k);
        }
        if let Some(k) = top.hi.as_deref() {
            stack.push(k);
        }
    }
    suggest
}

/// `TSTLookup.utf8SortedAsUTF16SortOrder`: bytes compare unsigned, except that
/// the lead bytes `EE` and `EF` (U+E000..U+FFFF) sort after the surrogate-pair
/// lead bytes `F0..F4`, which is where UTF-16 puts them.
fn utf8_sorted_as_utf16_sort_order(a: &[u8], b: &[u8]) -> Ordering {
    for (&x, &y) in a.iter().zip(b) {
        if x != y {
            let (mut p, mut q) = (i32::from(x), i32::from(y));
            if p >= 0xee && q >= 0xee {
                if p & 0xfe == 0xee {
                    p += 0xe;
                }
                if q & 0xfe == 0xee {
                    q += 0xe;
                }
            }
            return p.cmp(&q);
        }
    }
    a.len().cmp(&b.len())
}

/// A key as UTF-16 units; Java throws on an empty one (`charAt(0)`).
fn key_units(key: &str) -> Result<Vec<u16>> {
    let units: Vec<u16> = key.encode_utf16().collect();
    if units.is_empty() {
        return Err(Error::IllegalArgument(
            "index 0 out of bounds for length 0".into(),
        ));
    }
    Ok(units)
}

/// `TSTLookup`.
#[derive(Debug)]
pub struct TstLookup {
    root: Node,
    count: u64,
}

impl Default for TstLookup {
    fn default() -> Self {
        Self::new()
    }
}

impl TstLookup {
    /// `new TSTLookup()`. The root is an empty node splitting on `U+0000`, as
    /// in Java, so every key but one starting with NUL hangs off its `hi` child.
    pub fn new() -> Self {
        Self {
            root: Node::new(0),
            count: 0,
        }
    }

    /// `add(key, value)`.
    pub fn add(&mut self, key: &str, value: i64) -> bool {
        let units: Vec<u16> = key.encode_utf16().collect();
        insert(&mut self.root, &units, key, value);
        true
    }

    /// `get(key)`: the value stored for exactly `key`.
    pub fn get(&self, key: &str) -> Result<Option<i64>> {
        let units = key_units(key)?;
        Ok(prefix_completion(&self.root, &units)
            .into_iter()
            .find(|n| n.token.as_deref() == Some(key))
            .and_then(|n| n.val))
    }
}

fn write_recursively(out: &mut dyn DataOutput, node: &Node) {
    // A lone surrogate is written as U+FFFD, as Java's `writeString` does,
    // so a supplementary character does not survive store/load.
    let ch = char::from_u32(u32::from(node.split)).unwrap_or('\u{FFFD}');
    out.write_string(&ch.to_string());
    let mut mask = 0u8;
    if node.eq.is_some() {
        mask |= EQ_KID;
    }
    if node.lo.is_some() {
        mask |= LO_KID;
    }
    if node.hi.is_some() {
        mask |= HI_KID;
    }
    if node.token.is_some() {
        mask |= HAS_TOKEN;
    }
    if node.val.is_some() {
        mask |= HAS_VALUE;
    }
    out.write_byte(mask);
    if let Some(t) = &node.token {
        out.write_string(t);
    }
    if let Some(v) = node.val {
        out.write_i64(v);
    }
    for kid in [&node.lo, &node.eq, &node.hi].into_iter().flatten() {
        write_recursively(out, kid);
    }
}

fn read_recursively(input: &mut dyn DataInput, node: &mut Node) -> Result<()> {
    let s = input.read_string()?;
    node.split = s
        .encode_utf16()
        .next()
        .ok_or_else(|| Error::Corrupted("empty split character".into()))?;
    let mask = input.read_byte()?;
    if mask & HAS_TOKEN != 0 {
        node.token = Some(input.read_string()?);
    }
    if mask & HAS_VALUE != 0 {
        node.val = Some(input.read_i64()?);
    }
    for (bit, kid) in [
        (LO_KID, &mut node.lo),
        (EQ_KID, &mut node.eq),
        (HI_KID, &mut node.hi),
    ] {
        if mask & bit != 0 {
            let mut child = Box::new(Node::default());
            read_recursively(input, &mut child)?;
            *kid = Some(child);
        }
    }
    Ok(())
}

impl Lookup for TstLookup {
    fn get_count(&self) -> Result<u64> {
        Ok(self.count)
    }

    fn build(&mut self, input: &mut dyn InputIterator) -> Result<()> {
        if input.has_payloads() {
            return Err(Error::IllegalArgument(
                "this suggester doesn't support payloads".into(),
            ));
        }
        if input.has_contexts() {
            return Err(Error::IllegalArgument(
                "this suggester doesn't support contexts".into(),
            ));
        }
        self.root = Node::new(0);
        let mut sorted =
            SortedInputIterator::with_comparator(input, utf8_sorted_as_utf16_sort_order);
        self.count = 0;
        let (mut tokens, mut vals) = (Vec::new(), Vec::new());
        while let Some(term) = sorted.next() {
            tokens.push(String::from_utf8_lossy(&term).into_owned());
            vals.push(sorted.weight());
            self.count += 1;
        }
        balanced_tree(&tokens, &vals, 0, tokens.len() as isize - 1, &mut self.root);
        Ok(())
    }

    fn lookup(
        &self,
        key: &str,
        contexts: Option<&BTreeSet<Vec<u8>>>,
        only_more_popular: bool,
        num: usize,
    ) -> Result<Vec<LookupResult>> {
        if contexts.is_some() {
            return Err(Error::IllegalArgument(
                "this suggester doesn't support contexts".into(),
            ));
        }
        let units = key_units(key)?;
        let list = prefix_completion(&self.root, &units);
        let result_of = |n: &Node| {
            LookupResult::new(
                n.token.clone().unwrap_or_default(),
                n.val.unwrap_or_default(),
            )
        };
        if only_more_popular {
            let mut queue = LookupPriorityQueue::new(num);
            for n in &list {
                queue.insert_with_overflow(result_of(n));
            }
            Ok(queue.get_results())
        } else {
            Ok(list.iter().take(num).map(|n| result_of(n)).collect())
        }
    }

    fn store(&self, output: &mut dyn DataOutput) -> Result<bool> {
        output.write_vlong(i64::try_from(self.count).unwrap_or(i64::MAX));
        write_recursively(output, &self.root);
        Ok(true)
    }

    fn load(&mut self, input: &mut dyn DataInput) -> Result<bool> {
        self.count = u64::try_from(input.read_vlong()?)
            .map_err(|_| Error::Corrupted("negative suggestion count".into()))?;
        let mut root = Node::default();
        read_recursively(input, &mut root)?;
        self.root = root;
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::suggest::input::{input_empty, InputIteratorWrapper};
    use lucene_store::{SliceInput, VecDataOutput};

    #[test]
    fn add_then_get_and_prefix_lookup() {
        let mut t = TstLookup::default();
        assert!(t.add("car", 3));
        assert!(t.add("cart", 9));
        t.add("", 1); // an empty key changes nothing
        assert_eq!(t.get("car").unwrap(), Some(3));
        assert_eq!(t.get("ca").unwrap(), None);
        assert_eq!(t.get("zzz").unwrap(), None);
        let got = t.lookup_no_contexts("car", true, 5).unwrap();
        let keys: Vec<(&str, i64)> = got.iter().map(|r| (r.key.as_str(), r.value)).collect();
        assert_eq!(keys, vec![("cart", 9), ("car", 3)]);
    }

    #[test]
    fn an_empty_key_is_refused_as_javas_charat_is() {
        let t = TstLookup::new();
        assert!(t.get("").is_err());
        assert!(t.lookup_no_contexts("", false, 3).is_err());
    }

    #[test]
    fn contexts_and_payloads_are_refused() {
        let t = TstLookup::new();
        let ctx: BTreeSet<Vec<u8>> = [b"c".to_vec()].into();
        assert!(t.lookup("a", Some(&ctx), false, 1).is_err());

        struct Flags(bool, bool);
        impl InputIterator for Flags {
            fn next(&mut self) -> Option<Vec<u8>> {
                None
            }
            fn weight(&self) -> i64 {
                0
            }
            fn payload(&self) -> Option<&[u8]> {
                None
            }
            fn has_payloads(&self) -> bool {
                self.0
            }
            fn contexts(&self) -> Option<&BTreeSet<Vec<u8>>> {
                None
            }
            fn has_contexts(&self) -> bool {
                self.1
            }
        }
        let mut t = TstLookup::new();
        assert!(t.build(&mut Flags(true, false)).is_err());
        assert!(t.build(&mut Flags(false, true)).is_err());
    }

    #[test]
    fn build_of_nothing_is_empty_and_rebuild_replaces() {
        let mut t = TstLookup::new();
        t.build(&mut input_empty()).unwrap();
        assert_eq!(t.get_count().unwrap(), 0);
        let words = vec![b"ab".to_vec(), b"abc".to_vec()];
        t.build(&mut InputIteratorWrapper::new(words.into_iter()))
            .unwrap();
        assert_eq!(t.get_count().unwrap(), 2);
        assert_eq!(t.get("ab").unwrap(), Some(1));
        t.build(&mut input_empty()).unwrap();
        assert_eq!(t.get("ab").unwrap(), None);
    }

    #[test]
    fn utf8_order_places_u_e000_to_u_ffff_after_supplementary() {
        // U+FFFD is EF BF BD, U+1F600 is F0 9F 98 80: UTF-16 puts the
        // surrogate pair first, plain byte order the other way.
        let bmp = "\u{FFFD}".as_bytes();
        let supp = "\u{1F600}".as_bytes();
        assert_eq!(utf8_sorted_as_utf16_sort_order(supp, bmp), Ordering::Less);
        assert_eq!(
            utf8_sorted_as_utf16_sort_order(bmp, supp),
            Ordering::Greater
        );
        // both in the EE/EF range: ordinary comparison
        assert_eq!(
            utf8_sorted_as_utf16_sort_order("\u{E000}".as_bytes(), bmp),
            Ordering::Less
        );
        // a plain difference, then a proper prefix
        assert_eq!(utf8_sorted_as_utf16_sort_order(b"a", b"b"), Ordering::Less);
        assert_eq!(
            utf8_sorted_as_utf16_sort_order(b"ab", b"a"),
            Ordering::Greater
        );
        assert_eq!(utf8_sorted_as_utf16_sort_order(b"a", b"a"), Ordering::Equal);
    }

    #[test]
    fn a_supplementary_character_does_not_survive_store_and_load() {
        // Java stores each half of the surrogate pair as U+FFFD.
        let mut t = TstLookup::new();
        t.add("x\u{1F600}", 4);
        assert_eq!(t.get("x\u{1F600}").unwrap(), Some(4));
        let mut out = VecDataOutput::new();
        assert!(t.store(&mut out).unwrap());
        let bytes = out.into_inner();
        let mut loaded = TstLookup::new();
        assert!(loaded.load(&mut SliceInput::new(&bytes)).unwrap());
        assert_eq!(loaded.get("x\u{1F600}").unwrap(), None);
    }

    #[test]
    fn load_refuses_corrupt_input() {
        let mut t = TstLookup::new();
        // an empty split string
        let mut out = VecDataOutput::new();
        out.write_vlong(1);
        out.write_string("");
        assert!(t.load(&mut SliceInput::new(&out.into_inner())).is_err());
        // truncated after the count
        let mut out = VecDataOutput::new();
        out.write_vlong(1);
        assert!(t.load(&mut SliceInput::new(&out.into_inner())).is_err());
    }
}
