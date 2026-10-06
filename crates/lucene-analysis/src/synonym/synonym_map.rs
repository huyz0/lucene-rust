//! `org.apache.lucene.analysis.synonym.SynonymMap`, its `Builder` and the
//! `Parser` base the rule-file parsers share.
//!
//! A map from an input phrase to the output phrases that replace it. Words of
//! a phrase are joined by [`WORD_SEPARATOR`] (U+0000), as in Java.
//!
//! Differs: Java compiles the entries into an `FST<BytesRef>` over the input's
//! code points (`INPUT_TYPE.BYTE4`), each output a byte string of
//! `vint(count << 1 | !includeOrig)` then `count` vint word ords. The FST is
//! never serialised -- the filters only walk it code point by code point,
//! test `isFinal`, and decode the output at a final arc -- so the port holds
//! the same keys in a code-point trie ([`SynonymMap::step`],
//! [`SynonymMap::entry`]) and the decoded output as a [`SynonymEntry`]. Every
//! lookup visits the same keys and yields the same ords in the same order.
//! `BytesRefHash words` is a `Vec<String>` indexed by ord, ords assigned in
//! insertion order as `BytesRefHash.add` assigns them.

use std::collections::HashMap;

use crate::analyzer::Analyzer;
use crate::token_stream::TokenStream;
use crate::AnalysisError;

/// `SynonymMap.WORD_SEPARATOR`.
pub const WORD_SEPARATOR: char = '\0';

/// One trie node: children sorted by code point, and the entry of the key
/// ending here (`Arc.isFinal()`).
#[derive(Debug, Default, Clone)]
struct Node {
    children: Vec<(u32, u32)>,
    entry: Option<u32>,
}

/// A key's decoded FST output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SynonymEntry {
    /// `(code & 1) == 0`: the matched input is kept alongside the outputs.
    pub keep_orig: bool,
    /// The output phrases' word ords, in the order added (deduplicated when
    /// the builder dedups).
    pub ords: Vec<u32>,
}

/// `SynonymMap`.
#[derive(Debug, Clone)]
pub struct SynonymMap {
    nodes: Vec<Node>,
    entries: Vec<SynonymEntry>,
    /// `words`: map<ord, output phrase>.
    words: Vec<String>,
    /// `maxHorizontalContext`: the most words on either side of a rule.
    pub max_horizontal_context: usize,
}

/// A trie node id; [`SynonymMap::root`] is `fst.getFirstArc`.
pub type NodeId = u32;

impl SynonymMap {
    /// The node before any input (`fst.getFirstArc`).
    pub fn root(&self) -> NodeId {
        0
    }

    /// `fst.findTargetArc(label, ...)`: the node after `label`, if any key
    /// continues with it.
    #[inline]
    pub fn step(&self, node: NodeId, label: u32) -> Option<NodeId> {
        let children = &self.nodes[node as usize].children;
        if children.len() <= 8 {
            return children.iter().find(|c| c.0 == label).map(|c| c.1);
        }
        children
            .binary_search_by_key(&label, |c| c.0)
            .ok()
            .map(|i| children[i].1)
    }

    /// The entry of the key ending at `node` (`isFinal()` and its output).
    #[inline]
    pub fn entry(&self, node: NodeId) -> Option<&SynonymEntry> {
        self.nodes[node as usize]
            .entry
            .map(|e| &self.entries[e as usize])
    }

    /// `words.get(ord)`: an output phrase, words joined by [`WORD_SEPARATOR`].
    pub fn word(&self, ord: u32) -> &str {
        &self.words[ord as usize]
    }

    /// The number of distinct output phrases (`words.size()`).
    pub fn word_count(&self) -> usize {
        self.words.len()
    }

    /// Every key and its entry in code-point order (`IntsRefFSTEnum`'s
    /// order), keys' words joined by [`WORD_SEPARATOR`].
    pub fn entries(&self) -> Vec<(String, &SynonymEntry)> {
        let mut out = Vec::new();
        let mut key = String::new();
        self.collect(0, &mut key, &mut out);
        out
    }

    fn collect<'a>(
        &'a self,
        node: u32,
        key: &mut String,
        out: &mut Vec<(String, &'a SynonymEntry)>,
    ) {
        let n = &self.nodes[node as usize];
        if let Some(e) = n.entry {
            out.push((key.clone(), &self.entries[e as usize]));
        }
        for &(label, child) in &n.children {
            let len = key.len();
            key.push(char::from_u32(label).unwrap_or(char::REPLACEMENT_CHARACTER));
            self.collect(child, key, out);
            key.truncate(len);
        }
    }
}

/// `SynonymMap.Builder.MapEntry`.
#[derive(Debug, Default)]
struct MapEntry {
    include_orig: bool,
    ords: Vec<u32>,
}

/// `SynonymMap.Builder`.
#[derive(Debug)]
pub struct SynonymMapBuilder {
    working_set: HashMap<String, MapEntry>,
    words: Vec<String>,
    word_ords: HashMap<String, u32>,
    max_horizontal_context: usize,
    dedup: bool,
}

impl Default for SynonymMapBuilder {
    /// `new Builder()`: `dedup = true`.
    fn default() -> Self {
        Self::new(true)
    }
}

impl SynonymMapBuilder {
    /// `new Builder(dedup)`: when `dedup`, a rule added twice (same input,
    /// same output) is kept once.
    pub fn new(dedup: bool) -> Self {
        SynonymMapBuilder {
            working_set: HashMap::new(),
            words: Vec::new(),
            word_ords: HashMap::new(),
            max_horizontal_context: 0,
            dedup,
        }
    }

    /// `Builder.join(String[], CharsRefBuilder)`: the words joined by
    /// [`WORD_SEPARATOR`].
    pub fn join(words: &[&str]) -> String {
        words.join("\0")
    }

    fn count_words(chars: &str) -> usize {
        1 + chars.chars().filter(|&c| c == WORD_SEPARATOR).count()
    }

    /// `add(CharsRef input, CharsRef output, boolean includeOrig)`: a
    /// phrase -> phrase rule.
    pub fn add(
        &mut self,
        input: &str,
        output: &str,
        include_orig: bool,
    ) -> Result<(), AnalysisError> {
        let (ni, no) = (Self::count_words(input), Self::count_words(output));
        self.add_counted(input, ni, output, no, include_orig)
    }

    // Java: Builder.add(CharsRef, int, CharsRef, int, boolean)
    fn add_counted(
        &mut self,
        input: &str,
        num_input_words: usize,
        output: &str,
        num_output_words: usize,
        include_orig: bool,
    ) -> Result<(), AnalysisError> {
        if input.is_empty() {
            return Err(AnalysisError::IllegalArgument(
                "input.length must be > 0 (got 0)".to_string(),
            ));
        }
        if output.is_empty() {
            return Err(AnalysisError::IllegalArgument(
                "output.length must be > 0 (got 0)".to_string(),
            ));
        }
        let ord = match self.word_ords.get(output) {
            Some(&ord) => ord,
            None => {
                let ord = self.words.len() as u32;
                self.words.push(output.to_string());
                self.word_ords.insert(output.to_string(), ord);
                ord
            }
        };
        let e = self.working_set.entry(input.to_string()).or_default();
        e.ords.push(ord);
        e.include_orig |= include_orig;
        self.max_horizontal_context = self
            .max_horizontal_context
            .max(num_input_words)
            .max(num_output_words);
        Ok(())
    }

    /// `build()`. An empty map is an `IllegalState` error: Lucene 10.5.0's
    /// `build()` throws a `NullPointerException` there (`FST.fromFSTReader`
    /// of the `null` an empty `FSTCompiler` compiles to).
    pub fn build(self) -> Result<SynonymMap, AnalysisError> {
        if self.working_set.is_empty() {
            return Err(AnalysisError::IllegalState(
                "NullPointerException: an empty SynonymMap has no FST".to_string(),
            ));
        }
        let mut nodes = vec![Node::default()];
        let mut entries = Vec::with_capacity(self.working_set.len());
        for (input, e) in self.working_set {
            let mut ords = Vec::with_capacity(e.ords.len());
            for ord in e.ords {
                // The dedup set is per entry, as Java clears it per key.
                if self.dedup && ords.contains(&ord) {
                    continue;
                }
                ords.push(ord);
            }
            let mut node = 0usize;
            for c in input.chars() {
                let label = c as u32;
                let next = match nodes[node].children.binary_search_by_key(&label, |c| c.0) {
                    Ok(i) => nodes[node].children[i].1 as usize,
                    Err(i) => {
                        let id = nodes.len();
                        nodes.push(Node::default());
                        nodes[node].children.insert(i, (label, id as u32));
                        id
                    }
                };
                node = next;
            }
            nodes[node].entry = Some(entries.len() as u32);
            entries.push(SynonymEntry {
                keep_orig: e.include_orig,
                ords,
            });
        }
        Ok(SynonymMap {
            nodes,
            entries,
            words: self.words,
            max_horizontal_context: self.max_horizontal_context,
        })
    }
}

/// `SynonymMap.Parser`'s shared half: the builder and the analyzer rule
/// phrases go through.
pub struct SynonymParserBase<'a> {
    /// The rules parsed so far (`Parser extends Builder`).
    pub builder: SynonymMapBuilder,
    analyzer: &'a Analyzer,
}

impl<'a> SynonymParserBase<'a> {
    /// `new Parser(dedup, analyzer)`.
    pub fn new(dedup: bool, analyzer: &'a Analyzer) -> Self {
        SynonymParserBase {
            builder: SynonymMapBuilder::new(dedup),
            analyzer,
        }
    }

    /// `Parser.analyze(String, CharsRefBuilder)`: the analyzer's terms for
    /// `text`, joined by [`WORD_SEPARATOR`]. A zero-length term, a position
    /// increment other than 1 or no term at all is an `IllegalArgument`.
    pub fn analyze(&self, text: &str) -> Result<String, AnalysisError> {
        let mut reuse = String::new();
        let mut ts = self.analyzer.token_stream("", text)?;
        let result = (|| {
            ts.reset()?;
            while ts.increment_token()? {
                let a = ts.attributes();
                if a.term().is_empty() {
                    return Err(AnalysisError::IllegalArgument(format!(
                        "term: {text} analyzed to a zero-length token"
                    )));
                }
                if a.position_increment() != 1 {
                    return Err(AnalysisError::IllegalArgument(format!(
                        "term: {text} analyzed to a token ({}) with position increment != 1 (got: {})",
                        a.term(),
                        a.position_increment()
                    )));
                }
                if !reuse.is_empty() {
                    reuse.push(WORD_SEPARATOR);
                }
                reuse.push_str(a.term());
            }
            ts.end()
        })();
        let closed = ts.close();
        result?;
        closed?;
        if reuse.is_empty() {
            return Err(AnalysisError::IllegalArgument(format!(
                "term: {text} was completely eliminated by analyzer"
            )));
        }
        Ok(reuse)
    }
}

/// `java.text.ParseException("Invalid synonym rule at line N")`, its cause
/// the `IllegalArgumentException` the rule raised.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum SynonymParseError {
    /// The `ParseException` both parsers throw; `line` is
    /// `LineNumberReader.getLineNumber()` when the rule failed.
    #[error("Invalid synonym rule at line {line}: {cause}")]
    InvalidRule { line: usize, cause: AnalysisError },
    /// A WordNet line too short for its synset id or without its quoted
    /// word, where Java's `substring` throws `StringIndexOutOfBoundsException`
    /// (not wrapped in a `ParseException`).
    #[error("malformed WordNet line {line}")]
    Malformed { line: usize },
}

/// `LineNumberReader.readLine()`'s lines: split at `\n`, `\r` or `\r\n`.
pub(crate) fn java_lines(text: &str) -> Vec<&str> {
    let mut lines = Vec::new();
    let b = text.as_bytes();
    let (mut start, mut i) = (0, 0);
    while i < b.len() {
        match b[i] {
            b'\n' => {
                lines.push(&text[start..i]);
                i += 1;
                start = i;
            }
            b'\r' => {
                lines.push(&text[start..i]);
                i += 1;
                if i < b.len() && b[i] == b'\n' {
                    i += 1;
                }
                start = i;
            }
            _ => i += 1,
        }
    }
    if start < b.len() {
        lines.push(&text[start..]);
    }
    lines
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_assigns_ords_in_insertion_order_and_dedups_per_entry() {
        let mut b = SynonymMapBuilder::new(true);
        b.add("a", "x", false).unwrap();
        b.add("a", "y\0z", true).unwrap();
        b.add("a", "x", false).unwrap();
        b.add("b\0c", "x", false).unwrap();
        let m = b.build().unwrap();
        assert_eq!(m.max_horizontal_context, 2);
        assert_eq!(m.word_count(), 2);
        assert_eq!(m.word(1), "y\0z");
        let e = m.entries();
        assert_eq!(e.len(), 2);
        assert_eq!(e[0].0, "a");
        assert_eq!(e[0].1.ords, vec![0, 1]);
        assert!(e[0].1.keep_orig);
        assert_eq!(e[1].0, "b\0c");
        assert!(!e[1].1.keep_orig);

        let mut b = SynonymMapBuilder::new(false);
        b.add("a", "x", false).unwrap();
        b.add("a", "x", false).unwrap();
        assert_eq!(b.build().unwrap().entries()[0].1.ords, vec![0, 0]);
        assert!(SynonymMapBuilder::default().build().is_err());
    }

    #[test]
    fn walking_the_trie_finds_keys_and_prefixes() {
        let mut b = SynonymMapBuilder::default();
        for (i, k) in ["abc", "ab", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
            .iter()
            .enumerate()
        {
            b.add(k, &format!("w{i}"), false).unwrap();
        }
        let m = b.build().unwrap();
        let root = m.root();
        let a = m.step(root, 'a' as u32).unwrap();
        assert!(m.entry(a).is_none());
        let ab = m.step(a, 'b' as u32).unwrap();
        assert_eq!(m.entry(ab).unwrap().ords, vec![1]);
        // A wide node is binary-searched.
        assert!(m.step(root, 'j' as u32).is_some());
        assert!(m.step(root, 'z' as u32).is_none());
        assert!(m.step(ab, 'z' as u32).is_none());
    }

    #[test]
    fn empty_phrases_are_rejected_and_join_uses_the_separator() {
        let mut b = SynonymMapBuilder::default();
        assert!(matches!(
            b.add("", "x", false),
            Err(AnalysisError::IllegalArgument(_))
        ));
        assert!(matches!(
            b.add("x", "", false),
            Err(AnalysisError::IllegalArgument(_))
        ));
        assert_eq!(SynonymMapBuilder::join(&["a", "b"]), "a\0b");
    }

    #[test]
    fn analyze_joins_terms_and_rejects_what_java_rejects() {
        let ws = Analyzer::new(crate::core_analysis::WhitespaceAnalyzer::default());
        let p = SynonymParserBase::new(true, &ws);
        assert_eq!(p.analyze("a b").unwrap(), "a\0b");
        assert!(p
            .analyze("  ")
            .unwrap_err()
            .to_string()
            .contains("eliminated"));
        let stop = Analyzer::new(crate::core_analysis::StopAnalyzer::new(
            std::sync::Arc::new(crate::CharArraySet::from_words(["the"], false)),
        ));
        let p = SynonymParserBase::new(true, &stop);
        assert!(p
            .analyze("a the b")
            .unwrap_err()
            .to_string()
            .contains("increment"));
        let kw = Analyzer::keyword();
        let p = SynonymParserBase::new(true, &kw);
        assert!(p
            .analyze("")
            .unwrap_err()
            .to_string()
            .contains("zero-length"));
    }

    #[test]
    fn java_lines_splits_like_line_number_reader() {
        assert_eq!(java_lines("a\nb\r\nc\rd"), vec!["a", "b", "c", "d"]);
        assert_eq!(java_lines("a\n\n"), vec!["a", ""]);
        assert_eq!(java_lines(""), Vec::<&str>::new());
        assert_eq!(java_lines("x\r"), vec!["x"]);
    }
}
