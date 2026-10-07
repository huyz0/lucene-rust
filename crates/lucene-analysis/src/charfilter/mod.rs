//! `org.apache.lucene.analysis.charfilter`: `BaseCharFilter`'s offset
//! corrections, `NormalizeCharMap`, `MappingCharFilter`, `HTMLStripCharFilter` (and
//! `analysis/util/RollingCharBuffer`, which it reads through).

use std::collections::BTreeMap;

mod html_strip;

pub use html_strip::HTMLStripCharFilter;

use crate::reader::{CharFilter, CharReader};
use crate::AnalysisError;

/// `BaseCharFilter`'s `offsets`/`diffs` map and `correct`.
#[derive(Debug, Clone, Default)]
pub struct OffsetCorrections {
    offsets: Vec<i32>,
    diffs: Vec<i32>,
}

impl OffsetCorrections {
    /// `BaseCharFilter.correct(int)`.
    pub fn correct(&self, current_off: i32) -> i32 {
        if self.offsets.is_empty() {
            return current_off;
        }
        // Arrays.binarySearch: the last recorded offset <= current_off.
        let index = match self.offsets.binary_search(&current_off) {
            Ok(i) => i as isize,
            Err(ins) => ins as isize - 1,
        };
        let diff = if index < 0 {
            0
        } else {
            self.diffs[index as usize]
        };
        current_off + diff
    }

    /// `getLastCumulativeDiff()`.
    pub fn last_cumulative_diff(&self) -> i32 {
        self.diffs.last().copied().unwrap_or(0)
    }

    /// `addOffCorrectMap(int off, int cumulativeDiff)`.
    pub fn add(&mut self, off: i32, cumulative_diff: i32) {
        debug_assert!(self.offsets.last().is_none_or(|&l| off >= l));
        if self.offsets.last() == Some(&off) {
            *self.diffs.last_mut().expect("parallel arrays") = cumulative_diff;
        } else {
            self.offsets.push(off);
            self.diffs.push(cumulative_diff);
        }
    }

    /// Forget every correction (a reused filter).
    pub fn clear(&mut self) {
        self.offsets.clear();
        self.diffs.clear();
    }
}

/// `org.apache.lucene.analysis.util.RollingCharBuffer`: the window of input
/// units from the first one not yet freed.
///
/// Differs: one contiguous vector whose freed front is dropped in bulk
/// (Java's ring of `char`s wraps), so a buffered unit is one index and a
/// span of them one slice.
#[derive(Debug, Default)]
pub struct RollingCharBuffer {
    /// `window[head..]` holds positions `first_pos..`.
    window: Vec<u16>,
    head: usize,
    /// Position of `window[head]`.
    first_pos: i32,
    end: bool,
    chunk: Vec<u16>,
}

impl RollingCharBuffer {
    /// `reset(Reader)`.
    pub fn reset(&mut self) {
        self.window.clear();
        self.head = 0;
        self.first_pos = 0;
        self.end = false;
    }

    /// `get(int pos)`: the unit at `pos`, reading on demand; `None` at the
    /// end of the input.
    #[inline]
    pub fn get(
        &mut self,
        reader: &mut dyn CharReader,
        pos: i32,
    ) -> Result<Option<u16>, AnalysisError> {
        match self.peek(pos) {
            Some(u) => Ok(Some(u)),
            None => self.read_to(reader, pos),
        }
    }

    /// [`Self::get`] of a unit not buffered yet.
    fn read_to(
        &mut self,
        reader: &mut dyn CharReader,
        pos: i32,
    ) -> Result<Option<u16>, AnalysisError> {
        let buffered = |b: &Self| b.first_pos + (b.window.len() - b.head) as i32;
        while pos >= buffered(self) {
            if self.end {
                return Ok(None);
            }
            if self.chunk.is_empty() {
                self.chunk = vec![0; 512];
            }
            let n = reader.read(&mut self.chunk)?;
            if n == 0 {
                self.end = true;
                return Ok(None);
            }
            self.window.extend_from_slice(&self.chunk[..n]);
        }
        debug_assert!(pos >= self.first_pos, "pos {pos} was freed");
        Ok(self.peek(pos))
    }

    /// The unit at `pos` if it is already buffered (read and not freed),
    /// without reading.
    #[inline]
    pub fn peek(&self, pos: i32) -> Option<u16> {
        let idx = usize::try_from(pos.checked_sub(self.first_pos)?).ok()?;
        self.window.get(self.head.checked_add(idx)?).copied()
    }

    /// `get(int posStart, int length)`: `length` buffered units from
    /// `pos_start` (units not buffered are left out).
    pub fn slice(&self, pos_start: i32, length: i32) -> &[u16] {
        let Some(start) = pos_start
            .checked_sub(self.first_pos)
            .and_then(|s| usize::try_from(s).ok())
        else {
            return &[];
        };
        let live = &self.window[self.head..];
        let start = start.min(live.len());
        let len = usize::try_from(length).unwrap_or(0).min(live.len() - start);
        &live[start..start + len]
    }

    /// `freeBefore(int pos)`.
    pub fn free_before(&mut self, pos: i32) {
        let live = self.window.len() - self.head;
        let n = usize::try_from(pos.saturating_sub(self.first_pos))
            .unwrap_or(0)
            .min(live);
        self.head += n;
        self.first_pos += n as i32;
        // Drop the freed front once it is most of the vector.
        if self.head >= 1024 && self.head * 2 >= self.window.len() {
            self.window.drain(..self.head);
            self.head = 0;
        }
    }
}

/// One node of [`NormalizeCharMap`]'s trie.
#[derive(Debug, Clone, Default)]
struct Node {
    children: BTreeMap<u16, usize>,
    output: Option<Vec<u16>>,
}

/// `org.apache.lucene.analysis.charfilter.NormalizeCharMap`.
///
/// Differs: Java compiles the pairs into an `FST<CharsRef>` over UTF-16
/// units; this is a trie over the same units, walked the same way
/// (longest match, stopping where the FST has no further arcs).
#[derive(Debug, Clone, Default)]
pub struct NormalizeCharMap {
    nodes: Vec<Node>,
}

/// `NormalizeCharMap.Builder`.
#[derive(Debug, Clone, Default)]
pub struct NormalizeCharMapBuilder {
    pending: BTreeMap<String, String>,
}

impl NormalizeCharMapBuilder {
    /// `new Builder()`.
    pub fn new() -> Self {
        Self::default()
    }

    /// `add(String match, String replacement)`.
    pub fn add(&mut self, m: &str, replacement: &str) -> Result<(), AnalysisError> {
        if m.is_empty() {
            return Err(AnalysisError::IllegalArgument(
                "cannot match the empty string".into(),
            ));
        }
        if self.pending.contains_key(m) {
            return Err(AnalysisError::IllegalArgument(format!(
                "match \"{m}\" was already added"
            )));
        }
        self.pending.insert(m.to_string(), replacement.to_string());
        Ok(())
    }

    /// `build()`.
    pub fn build(self) -> NormalizeCharMap {
        let mut map = NormalizeCharMap {
            nodes: vec![Node::default()],
        };
        for (k, v) in self.pending {
            let mut n = 0;
            for u in k.encode_utf16() {
                let next = map.nodes.len();
                n = *map.nodes[n].children.entry(u).or_insert(next);
                if n == next {
                    map.nodes.push(Node::default());
                }
            }
            map.nodes[n].output = Some(v.encode_utf16().collect());
        }
        map
    }
}

/// `org.apache.lucene.analysis.charfilter.MappingCharFilter`: replaces the
/// longest matching key at each position with its value, recording offset
/// corrections.
pub struct MappingCharFilter<R> {
    input: R,
    map: std::sync::Arc<NormalizeCharMap>,
    buffer: RollingCharBuffer,
    replacement: Vec<u16>,
    replacement_pointer: usize,
    input_off: i32,
    corrections: OffsetCorrections,
}

impl<R: CharReader> MappingCharFilter<R> {
    /// `new MappingCharFilter(NormalizeCharMap, Reader)`.
    pub fn new(map: std::sync::Arc<NormalizeCharMap>, input: R) -> Self {
        MappingCharFilter {
            input,
            map,
            buffer: RollingCharBuffer::default(),
            replacement: Vec::new(),
            replacement_pointer: 0,
            input_off: 0,
            corrections: OffsetCorrections::default(),
        }
    }

    // Java: MappingCharFilter.read()
    fn read_one(&mut self) -> Result<Option<u16>, AnalysisError> {
        loop {
            if self.replacement_pointer < self.replacement.len() {
                let c = self.replacement[self.replacement_pointer];
                self.replacement_pointer += 1;
                return Ok(Some(c));
            }
            let mut last_match: Option<(i32, usize)> = None; // (len, node)
            if let Some(first) = self.buffer.get(&mut self.input, self.input_off)? {
                if let Some(&root_child) =
                    self.map.nodes.first().and_then(|r| r.children.get(&first))
                {
                    let mut node = root_child;
                    let mut lookahead = 0;
                    loop {
                        lookahead += 1;
                        if self.map.nodes[node].output.is_some() {
                            last_match = Some((lookahead, node));
                        }
                        if self.map.nodes[node].children.is_empty() {
                            break;
                        }
                        let Some(ch) = self
                            .buffer
                            .get(&mut self.input, self.input_off + lookahead)?
                        else {
                            break;
                        };
                        match self.map.nodes[node].children.get(&ch) {
                            Some(&next) => node = next,
                            None => break,
                        }
                    }
                }
            }
            if let Some((len, node)) = last_match {
                self.input_off += len;
                let out = self.map.nodes[node].output.clone().expect("a final node");
                let diff = len - out.len() as i32;
                if diff != 0 {
                    let prev = self.corrections.last_cumulative_diff();
                    if diff > 0 {
                        self.corrections
                            .add(self.input_off - diff - prev, prev + diff);
                    } else {
                        let output_start = self.input_off - prev;
                        for extra in 0..-diff {
                            self.corrections.add(output_start + extra, prev - extra - 1);
                        }
                    }
                }
                self.replacement = out;
                self.replacement_pointer = 0;
            } else {
                let ret = self.buffer.get(&mut self.input, self.input_off)?;
                if ret.is_some() {
                    self.input_off += 1;
                    self.buffer.free_before(self.input_off);
                }
                return Ok(ret);
            }
        }
    }
}

impl<R: CharReader> CharFilter for MappingCharFilter<R> {
    fn input(&self) -> &dyn CharReader {
        &self.input
    }

    fn input_mut(&mut self) -> &mut dyn CharReader {
        &mut self.input
    }

    // Java: MappingCharFilter.read(char[], int, int)
    fn read_filtered(&mut self, buf: &mut [u16]) -> Result<usize, AnalysisError> {
        let mut n = 0;
        while n < buf.len() {
            match self.read_one()? {
                Some(c) => {
                    buf[n] = c;
                    n += 1;
                }
                None => break,
            }
        }
        Ok(n)
    }

    fn correct(&self, current_off: i32) -> i32 {
        self.corrections.correct(current_off)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reader::StrReader;

    fn run(map: &std::sync::Arc<NormalizeCharMap>, text: &str) -> (String, Vec<i32>) {
        let mut f = MappingCharFilter::new(map.clone(), StrReader::new(text));
        let mut out = Vec::new();
        let mut buf = [0u16; 3];
        loop {
            let n = f.read(&mut buf).unwrap();
            if n == 0 {
                break;
            }
            out.extend_from_slice(&buf[..n]);
        }
        let s = String::from_utf16(&out).unwrap();
        let corr = (0..=out.len() as i32)
            .map(|o| f.correct_offset(o))
            .collect();
        (s, corr)
    }

    #[test]
    fn maps_longest_match_and_corrects_offsets() {
        let mut b = NormalizeCharMapBuilder::new();
        b.add("a", "x").unwrap();
        b.add("ab", "yy").unwrap();
        b.add("abc", "").unwrap();
        b.add("ß", "ss").unwrap();
        assert!(b.add("a", "z").is_err());
        assert!(b.add("", "z").is_err());
        let map = std::sync::Arc::new(b.build());
        assert_eq!(run(&map, "abcab").0, "yy");
        assert_eq!(run(&map, "abd").0, "yyd");
        let (s, corr) = run(&map, "aßq");
        assert_eq!(s, "xssq");
        assert_eq!(corr, vec![0, 1, 1, 2, 3]);
        let (s, corr) = run(&map, "abcq");
        assert_eq!(s, "q");
        assert_eq!(corr, vec![3, 4]);
        let empty = std::sync::Arc::new(NormalizeCharMapBuilder::new().build());
        assert_eq!(run(&empty, "x").0, "x");
    }

    #[test]
    fn corrections_and_rolling_buffer() {
        let mut c = OffsetCorrections::default();
        assert_eq!((c.correct(5), c.last_cumulative_diff()), (5, 0));
        c.add(2, 1);
        c.add(2, 3);
        c.add(4, 5);
        assert_eq!(
            (c.correct(1), c.correct(2), c.correct(3), c.correct(9)),
            (1, 5, 6, 14)
        );
        c.clear();
        assert_eq!(c.correct(3), 3);
        let mut r = RollingCharBuffer::default();
        let mut input = StrReader::new("x".repeat(600));
        assert_eq!(r.get(&mut input, 550).unwrap(), Some(u16::from(b'x')));
        r.free_before(500);
        assert_eq!(r.get(&mut input, 599).unwrap(), Some(u16::from(b'x')));
        assert_eq!(r.get(&mut input, 600).unwrap(), None);
        assert_eq!(r.get(&mut input, 601).unwrap(), None);
        assert_eq!(r.peek(500), Some(u16::from(b'x')));
        assert_eq!((r.peek(499), r.peek(600)), (None, None));
        assert_eq!(r.slice(598, 5), [u16::from(b'x'); 2]);
        assert!(r.slice(10, 2).is_empty());
        assert!(r.slice(650, 2).is_empty());
        r.reset();
        // A freed front past 1,024 units is dropped; positions stay put.
        let text: Vec<u16> = (0..3000u16).map(|i| 0x3040 + i % 80).collect();
        let mut input = StrReader::new(String::from_utf16(&text).unwrap());
        assert_eq!(r.get(&mut input, 1500).unwrap(), Some(text[1500]));
        r.free_before(1400);
        assert_eq!((r.peek(1399), r.peek(1400)), (None, Some(text[1400])));
        assert_eq!(r.get(&mut input, 2999).unwrap(), Some(text[2999]));
        assert_eq!(r.slice(1400, 3), &text[1400..1403]);
        r.free_before(5000);
        assert_eq!(r.peek(2999), None);
        assert_eq!(r.get(&mut input, 3000).unwrap(), None);
    }
}
