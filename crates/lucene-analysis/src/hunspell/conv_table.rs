//! `ConvTable`: an `ICONV`/`OCONV` replacement table, applied by longest
//! match from left to right.
//!
//! Differs: Lucene keeps the mappings in an `FST<CharsRef>` over UTF-16
//! units; this is a trie over the same units, walked the same way (the
//! longest key matching at each position wins), so the replacements are
//! identical.

use std::collections::{BTreeMap, HashMap};

/// One trie node: children by UTF-16 unit, and the replacement if a key
/// ends here.
#[derive(Debug, Default)]
struct Node {
    children: HashMap<u16, usize>,
    output: Option<Vec<u16>>,
}

/// `ConvTable`.
#[derive(Debug)]
pub(crate) struct ConvTable {
    nodes: Vec<Node>,
    first_char_hashes: Vec<bool>,
    modulus: usize,
}

impl ConvTable {
    /// `new ConvTable(TreeMap<String, String>)` (keys non-empty).
    pub(crate) fn new(mappings: &BTreeMap<Vec<u16>, Vec<u16>>) -> Self {
        // `Math.max(256, Integer.highestOneBit(size) << 1)`.
        let highest = if mappings.is_empty() {
            0
        } else {
            1usize << (usize::BITS - 1 - mappings.len().leading_zeros())
        };
        let modulus = 256.max(highest << 1);
        let mut table = ConvTable {
            nodes: vec![Node::default()],
            first_char_hashes: vec![false; modulus],
            modulus,
        };
        for (key, value) in mappings {
            table.first_char_hashes[usize::from(key[0]) % modulus] = true;
            let mut node = 0;
            for &u in key {
                node = match table.nodes[node].children.get(&u) {
                    Some(&n) => n,
                    None => {
                        table.nodes.push(Node::default());
                        let n = table.nodes.len() - 1;
                        table.nodes[node].children.insert(u, n);
                        n
                    }
                };
            }
            table.nodes[node].output = Some(value.clone());
        }
        table
    }

    /// `ConvTable.applyMappings`.
    pub(crate) fn apply_mappings(&self, sb: &mut Vec<u16>) {
        let mut i = 0;
        while i < sb.len() {
            if !self.might_replace_char(sb[i]) {
                i += 1;
                continue;
            }
            let mut node = 0;
            let mut longest: Option<(usize, &Vec<u16>)> = None;
            for (j, &ch) in sb.iter().enumerate().skip(i) {
                match self.nodes[node].children.get(&ch) {
                    Some(&n) => node = n,
                    None => break,
                }
                if let Some(out) = &self.nodes[node].output {
                    longest = Some((j, out));
                }
            }
            if let Some((end, out)) = longest {
                let out = out.clone();
                let n = out.len();
                sb.splice(i..=end, out);
                // Java: `i += longestOutput.length - 1`, then the loop's `i++`.
                // (An empty replacement re-examines position `i`.)
                i += n;
                continue;
            }
            i += 1;
        }
    }

    /// `ConvTable.mightReplaceChar`.
    pub(crate) fn might_replace_char(&self, c: u16) -> bool {
        self.first_char_hashes[usize::from(c) % self.modulus]
    }
}
