//! Morfologik's finite state automata (`morfologik.fsa`, 2.1.9): the
//! `FSA5` and `CFSA2` formats a dictionary is stored in, `FSATraversal`'s
//! `match` and `ByteSequenceIterator`.
//!
//! # Format
//!
//! A file starts `\fsa` and a version byte: `5` (FSA5) or `0xC6` (CFSA2);
//! the older `CFSA` (`0xC5`) is refused with a typed error (Morfologik
//! 2.1.9 still reads it; no dictionary Lucene loads uses it).
//!
//! **FSA5**: `filler`, `annotation`, then a byte whose low nibble is the
//! width of an arc's address (`gtl`) and high nibble the bytes of per-node
//! data, then the arcs. An arc is its label byte and `gtl` bytes,
//! little-endian, whose low three bits are flags (final, last arc of the
//! node, target is the next arc) and the rest the target's offset; a
//! "next" arc is two bytes. The root is the target of the first arc of the
//! node after node 0's first (epsilon) arc.
//!
//! **CFSA2**: a big-endian `short` of flags (`FSAFlags`: only the known
//! bits), a label-mapping table (a length byte and that many labels), then
//! the arcs: a flag byte (target-next `0x80`, last `0x40`, final `0x20`, and
//! a 5-bit index into the label table, `0` meaning an explicit label byte
//! follows), then, unless target-next, the target as a v-int. With the
//! `NUMBERS` flag every node starts with a v-int count.
//!
//! Every offset read from the file is checked: a corrupt automaton is a
//! [`MorfologikError`], never a panic (Java throws
//! `ArrayIndexOutOfBoundsException` from the lookup that reaches it).

use crate::MorfologikError;

/// `FSAFlags.NUMBERS`.
const NUMBERS: i16 = 1 << 8;
/// Every bit `FSAFlags` defines.
const KNOWN_FLAGS: i16 = 1 | 2 | 4 | 8 | NUMBERS | (1 << 9);

fn corrupt() -> MorfologikError {
    MorfologikError::new("ArrayIndexOutOfBoundsException: the automaton reads past its arcs")
}

/// A byte of `arcs`, or the error a corrupt offset is.
fn byte(arcs: &[u8], i: usize) -> Result<u8, MorfologikError> {
    arcs.get(i).copied().ok_or_else(corrupt)
}

fn add(a: usize, b: usize) -> Result<usize, MorfologikError> {
    a.checked_add(b).ok_or_else(corrupt)
}

/// `morfologik.fsa.FSA5`.
#[derive(Debug, Clone)]
pub struct Fsa5 {
    arcs: Vec<u8>,
    node_data_length: usize,
    gtl: usize,
}

impl Fsa5 {
    // Java: FSA5(InputStream)
    fn read(body: &[u8]) -> Result<Fsa5, MorfologikError> {
        let [_filler, _annotation, hgtl, ref arcs @ ..] = *body else {
            return Err(MorfologikError::new("EOFException: truncated FSA5 header"));
        };
        Ok(Fsa5 {
            arcs: arcs.to_vec(),
            node_data_length: usize::from((hgtl >> 4) & 0x0F),
            gtl: usize::from(hgtl & 0x0F),
        })
    }

    fn flags(&self, arc: usize) -> Result<u8, MorfologikError> {
        byte(&self.arcs, add(arc, 1)?)
    }

    // Java: FSA5.skipArc
    fn skip_arc(&self, offset: usize) -> Result<usize, MorfologikError> {
        let next = self.flags(offset)? & 4 != 0;
        add(offset, if next { 2 } else { add(1, self.gtl)? })
    }

    // Java: FSA5.getDestinationNodeOffset
    fn destination(&self, arc: usize) -> Result<usize, MorfologikError> {
        if self.flags(arc)? & 4 != 0 {
            return self.skip_arc(arc);
        }
        // decodeFromBytes(arcs, arc + 1, gtl) >>> 3: a Java int, so a
        // width over four bytes keeps only the last four shifted in.
        let start = add(arc, 1)?;
        let mut r: i32 = 0;
        for i in (0..self.gtl).rev() {
            r = r.wrapping_shl(8) | i32::from(byte(&self.arcs, add(start, i)?)?);
        }
        Ok(((r as u32) >> 3) as usize)
    }

    // Java: FSA5.getFirstArc
    fn first_arc(&self, node: usize) -> Result<usize, MorfologikError> {
        add(self.node_data_length, node)
    }

    // Java: FSA5.isArcLast
    fn is_last(&self, arc: usize) -> Result<bool, MorfologikError> {
        Ok(self.flags(arc)? & 2 != 0)
    }
}

/// `morfologik.fsa.CFSA2`.
#[derive(Debug, Clone)]
pub struct Cfsa2 {
    arcs: Vec<u8>,
    label_mapping: Vec<u8>,
    has_numbers: bool,
}

impl Cfsa2 {
    // Java: CFSA2(InputStream)
    fn read(body: &[u8]) -> Result<Cfsa2, MorfologikError> {
        let [f0, f1, size, ref rest @ ..] = *body else {
            return Err(MorfologikError::new("EOFException: truncated CFSA2 header"));
        };
        let flags = i16::from_be_bytes([f0, f1]);
        if flags & !KNOWN_FLAGS != 0 {
            return Err(MorfologikError::new(format!(
                "IOException: Unrecognized flags: 0x{:x}",
                i32::from(flags)
            )));
        }
        let size = usize::from(size);
        let label_mapping = rest
            .get(..size)
            .ok_or_else(|| MorfologikError::new("EOFException: truncated CFSA2 label mapping"))?
            .to_vec();
        Ok(Cfsa2 {
            arcs: rest[size..].to_vec(),
            label_mapping,
            has_numbers: flags & NUMBERS != 0,
        })
    }

    // Java: CFSA2.readVInt -- a Java int: shifts past 31 wrap, as `<<` does.
    fn read_vint(&self, mut offset: usize) -> Result<usize, MorfologikError> {
        let mut b = byte(&self.arcs, offset)?;
        let mut value: i32 = i32::from(b & 0x7F);
        let mut shift: u32 = 7;
        while b & 0x80 != 0 {
            offset = add(offset, 1)?;
            b = byte(&self.arcs, offset)?;
            value |= i32::from(b & 0x7F).wrapping_shl(shift);
            shift = shift.wrapping_add(7);
        }
        usize::try_from(value).map_err(|_| corrupt())
    }

    // Java: CFSA2.skipVInt
    fn skip_vint(&self, mut offset: usize) -> Result<usize, MorfologikError> {
        loop {
            let b = byte(&self.arcs, offset)?;
            offset = add(offset, 1)?;
            if b & 0x80 == 0 {
                return Ok(offset);
            }
        }
    }

    // Java: CFSA2.skipArc
    fn skip_arc(&self, offset: usize) -> Result<usize, MorfologikError> {
        let flag = byte(&self.arcs, offset)?;
        let mut offset = add(offset, 1)?;
        if flag & 0x1F == 0 {
            offset = add(offset, 1)?;
        }
        if flag & 0x80 == 0 {
            offset = self.skip_vint(offset)?;
        }
        Ok(offset)
    }

    // Java: CFSA2.getDestinationNodeOffset
    fn destination(&self, mut arc: usize) -> Result<usize, MorfologikError> {
        let flag = byte(&self.arcs, arc)?;
        if flag & 0x80 != 0 {
            while byte(&self.arcs, arc)? & 0x40 == 0 {
                arc = self.skip_arc(arc)?;
            }
            return self.skip_arc(arc);
        }
        self.read_vint(add(arc, if flag & 0x1F == 0 { 2 } else { 1 })?)
    }

    // Java: CFSA2.getFirstArc
    fn first_arc(&self, node: usize) -> Result<usize, MorfologikError> {
        if self.has_numbers {
            self.skip_vint(node)
        } else {
            Ok(node)
        }
    }
}

/// `morfologik.fsa.FSA`: one of the two formats read.
#[derive(Debug, Clone)]
pub enum Fsa {
    /// FSA5.
    Fsa5(Fsa5),
    /// CFSA2.
    Cfsa2(Cfsa2),
}

/// `MatchResult.kind`s `FSATraversal.match` answers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Match {
    /// `EXACT_MATCH`.
    Exact,
    /// `NO_MATCH`.
    No,
    /// `AUTOMATON_HAS_PREFIX`.
    AutomatonHasPrefix,
    /// `SEQUENCE_IS_A_PREFIX`, at this node.
    SequenceIsAPrefix(usize),
}

/// The longest sequence [`Fsa::sequences`] follows: a cycle in a corrupt
/// automaton would otherwise never end.
const MAX_DEPTH: usize = 1 << 16;

impl Fsa {
    /// `FSA.read(InputStream)`.
    pub fn read(bytes: &[u8]) -> Result<Fsa, MorfologikError> {
        let Some(rest) = bytes.strip_prefix(b"\\fsa") else {
            return Err(MorfologikError::new(
                "IOException: Invalid file header, probably not an FSA.",
            ));
        };
        let Some((&version, body)) = rest.split_first() else {
            return Err(MorfologikError::new(
                "IOException: Truncated file, no version number.",
            ));
        };
        match version {
            5 => Ok(Fsa::Fsa5(Fsa5::read(body)?)),
            0xC6 => Ok(Fsa::Cfsa2(Cfsa2::read(body)?)),
            v => Err(MorfologikError::new(format!(
                "IOException: Unsupported automaton version: 0x{v:02x}"
            ))),
        }
    }

    /// `getRootNode()`.
    pub fn root(&self) -> Result<usize, MorfologikError> {
        match self {
            Fsa::Fsa5(f) => {
                let epsilon = f.skip_arc(f.first_arc(0)?)?;
                f.destination(f.first_arc(epsilon)?)
            }
            Fsa::Cfsa2(c) => c.destination(c.first_arc(0)?),
        }
    }

    /// `getFirstArc(node)`.
    pub fn first_arc(&self, node: usize) -> Result<usize, MorfologikError> {
        match self {
            Fsa::Fsa5(f) => f.first_arc(node),
            Fsa::Cfsa2(c) => c.first_arc(node),
        }
    }

    /// `getNextArc(arc)`: `0` after the node's last arc.
    pub fn next_arc(&self, arc: usize) -> Result<usize, MorfologikError> {
        match self {
            Fsa::Fsa5(f) => {
                if f.is_last(arc)? {
                    Ok(0)
                } else {
                    f.skip_arc(arc)
                }
            }
            Fsa::Cfsa2(c) => {
                if byte(&c.arcs, arc)? & 0x40 != 0 {
                    Ok(0)
                } else {
                    c.skip_arc(arc)
                }
            }
        }
    }

    /// `getArcLabel(arc)`.
    pub fn label(&self, arc: usize) -> Result<u8, MorfologikError> {
        match self {
            Fsa::Fsa5(f) => byte(&f.arcs, arc),
            Fsa::Cfsa2(c) => {
                let index = usize::from(byte(&c.arcs, arc)? & 0x1F);
                if index > 0 {
                    c.label_mapping.get(index).copied().ok_or_else(corrupt)
                } else {
                    byte(&c.arcs, add(arc, 1)?)
                }
            }
        }
    }

    /// `isArcFinal(arc)`.
    pub fn is_final(&self, arc: usize) -> Result<bool, MorfologikError> {
        Ok(match self {
            Fsa::Fsa5(f) => f.flags(arc)? & 1 != 0,
            Fsa::Cfsa2(c) => byte(&c.arcs, arc)? & 0x20 != 0,
        })
    }

    /// `getEndNode(arc)`: the target (`0` for a terminal arc,
    /// `isArcTerminal`).
    pub fn end_node(&self, arc: usize) -> Result<usize, MorfologikError> {
        match self {
            Fsa::Fsa5(f) => f.destination(arc),
            Fsa::Cfsa2(c) => c.destination(arc),
        }
    }

    /// `getArc(node, label)`: `0` when the node has no such arc.
    pub fn arc(&self, node: usize, label: u8) -> Result<usize, MorfologikError> {
        let mut arc = self.first_arc(node)?;
        while arc != 0 {
            if self.label(arc)? == label {
                return Ok(arc);
            }
            arc = self.next_arc(arc)?;
        }
        Ok(0)
    }

    /// `FSATraversal.match(sequence, root)`.
    pub fn match_sequence(&self, sequence: &[u8], node: usize) -> Result<Match, MorfologikError> {
        if node == 0 {
            return Ok(Match::No);
        }
        let mut node = node;
        for (i, &b) in sequence.iter().enumerate() {
            let arc = self.arc(node, b)?;
            if arc == 0 {
                return Ok(if i > 0 {
                    Match::AutomatonHasPrefix
                } else {
                    Match::No
                });
            }
            let last = i.checked_add(1) == Some(sequence.len());
            if last && self.is_final(arc)? {
                return Ok(Match::Exact);
            }
            let end = self.end_node(arc)?;
            if end == 0 {
                return Ok(Match::AutomatonHasPrefix);
            }
            node = end;
        }
        Ok(Match::SequenceIsAPrefix(node))
    }

    /// `ByteSequenceIterator` from `node`: every sequence its final arcs
    /// end, depth first in arc order.
    pub fn sequences(&self, node: usize) -> Result<Vec<Vec<u8>>, MorfologikError> {
        let mut out = Vec::new();
        let first = self.first_arc(node)?;
        if first == 0 {
            return Ok(out);
        }
        let mut arcs: Vec<usize> = vec![first];
        let mut buffer: Vec<u8> = Vec::new();
        while let Some(&arc) = arcs.last() {
            let last_index = arcs.len().checked_sub(1).ok_or_else(corrupt)?;
            if arc == 0 {
                arcs.pop();
                continue;
            }
            arcs[last_index] = self.next_arc(arc)?;
            buffer.truncate(last_index);
            buffer.push(self.label(arc)?);
            let end = self.end_node(arc)?;
            if end != 0 {
                if arcs.len() >= MAX_DEPTH {
                    return Err(MorfologikError::new(
                        "IOException: the automaton has a cycle",
                    ));
                }
                arcs.push(self.first_arc(end)?);
            }
            if self.is_final(arc)? {
                out.push(buffer.clone());
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::arithmetic_side_effects)] // test code: no value read off disk
    use super::*;

    /// A CFSA2 holding "ab" and "ac" (explicit labels, no mapping).
    fn cfsa2() -> Vec<u8> {
        // Node 0 (epsilon): one arc, label 0, last, target = 3 (v-int).
        // Node 3 (root): 'a' last, target 7.
        // Node 7: 'b' final, terminal (target 0); 'c' final last, terminal.
        let mut f = b"\\fsa\xC6".to_vec();
        f.extend_from_slice(&[0, 0, 0]); // flags, empty label mapping
        let arcs = [
            0x40, 0x00, 0x03, // 0: epsilon arc -> 3
            0x40, b'a', 0x07, // 3: 'a' -> 7
            0x00, // pad (6)
            0x20, b'b', 0x00, // 7: 'b' final, terminal
            0x60, b'c', 0x00, // 10: 'c' final last, terminal
        ];
        f.extend_from_slice(&arcs);
        f
    }

    #[test]
    fn reads_and_walks_a_cfsa2() {
        let fsa = Fsa::read(&cfsa2()).unwrap();
        let root = fsa.root().unwrap();
        assert_eq!(root, 3);
        assert_eq!(fsa.match_sequence(b"ab", root).unwrap(), Match::Exact);
        assert_eq!(
            fsa.match_sequence(b"a", root).unwrap(),
            Match::SequenceIsAPrefix(7)
        );
        assert_eq!(
            fsa.match_sequence(b"abc", root).unwrap(),
            Match::AutomatonHasPrefix
        );
        assert_eq!(
            fsa.match_sequence(b"ax", root).unwrap(),
            Match::AutomatonHasPrefix
        );
        assert_eq!(fsa.match_sequence(b"x", root).unwrap(), Match::No);
        assert_eq!(fsa.match_sequence(b"x", 0).unwrap(), Match::No);
        assert_eq!(
            fsa.sequences(root).unwrap(),
            vec![b"ab".to_vec(), b"ac".to_vec()]
        );
        assert_eq!(
            fsa.sequences(7).unwrap(),
            vec![b"b".to_vec(), b"c".to_vec()]
        );
    }

    #[test]
    fn refuses_bad_headers() {
        for bad in [
            &b"\\fs"[..],
            b"xfsa\x05",
            b"\\fsa",
            b"\\fsa\xC5",
            b"\\fsa\x05\x00",
            b"\\fsa\xC6\x00",
            b"\\fsa\xC6\x00\x00\x05ab",
            b"\\fsa\xC6\x40\x00\x00",
        ] {
            assert!(Fsa::read(bad).is_err(), "{bad:?}");
        }
        assert!(Fsa::read(b"\\fsa\xC6\x01\x00\x00").is_ok());
    }

    #[test]
    fn corrupt_offsets_are_errors() {
        let mut f = cfsa2();
        let len = f.len();
        f[len - 4] = 0x50; // 'b' targets a node past the arcs
        let fsa = Fsa::read(&f).unwrap();
        assert!(fsa.sequences(7).is_err());
        let mut f = cfsa2();
        f[len - 6] = 0x21; // 'b': a label index past the (empty) mapping
        let fsa = Fsa::read(&f).unwrap();
        assert!(fsa.sequences(7).is_err());
        // A truncated automaton.
        let fsa = Fsa::read(&cfsa2()[..12]).unwrap();
        assert!(fsa.root().is_err() || fsa.sequences(3).is_err());
        // A cycle: the 'a' arc points back at its own node.
        let mut c = cfsa2();
        c[8 + 5] = 0x03;
        let fsa = Fsa::read(&c).unwrap();
        assert!(fsa.sequences(3).is_err());
    }

    #[test]
    fn vints_wrap_as_java_ints() {
        let c = Cfsa2 {
            arcs: vec![0xFF, 0xFF, 0xFF, 0xFF, 0x7F],
            label_mapping: vec![],
            has_numbers: true,
        };
        assert!(c.read_vint(0).is_err()); // negative as an int
        let c = Cfsa2 {
            arcs: vec![0x81, 0x01, 0x02],
            label_mapping: vec![],
            has_numbers: true,
        };
        assert_eq!(c.read_vint(0).unwrap(), 129);
        assert_eq!(c.first_arc(0).unwrap(), 2);
    }

    /// An FSA5 with `gtl` 2 holding "ab": a dummy arc at 0, the epsilon
    /// node at 3 pointing at the root, 6.
    #[test]
    fn reads_and_walks_an_fsa5() {
        let arc = |label: u8, target: u16, flags: u16| {
            let v = (target << 3) | flags;
            [label, (v & 0xFF) as u8, (v >> 8) as u8]
        };
        let mut f = b"\\fsa\x05_+\x02".to_vec();
        f.extend_from_slice(&arc(0, 0, 2));
        f.extend_from_slice(&arc(0, 6, 2));
        f.extend_from_slice(&arc(b'a', 9, 2));
        f.extend_from_slice(&arc(b'b', 0, 3));
        let fsa = Fsa::read(&f).unwrap();
        let root = fsa.root().unwrap();
        assert_eq!(root, 6);
        assert_eq!(fsa.match_sequence(b"ab", root).unwrap(), Match::Exact);
        assert_eq!(fsa.sequences(root).unwrap(), vec![b"ab".to_vec()]);
        // A "next" arc: the target follows it.
        let mut n = b"\\fsa\x05_+\x02".to_vec();
        n.extend_from_slice(&arc(0, 0, 2));
        n.extend_from_slice(&arc(0, 6, 2));
        n.extend_from_slice(&[b'a', 0x06]); // 6: 'a', last + next -> 8
        n.extend_from_slice(&arc(b'b', 0, 3)); // 8
        let fsa = Fsa::read(&n).unwrap();
        let root = fsa.root().unwrap();
        assert_eq!(fsa.sequences(root).unwrap(), vec![b"ab".to_vec()]);
        assert!(Fsa::read(b"\\fsa\x05_").is_err());
    }
}
