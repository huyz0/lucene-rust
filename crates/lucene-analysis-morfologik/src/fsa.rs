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

/// One arc, read once: what `getArcLabel`, `isArcFinal`, `getNextArc` and
/// `getEndNode` (`0`: terminal) answer for it.
#[derive(Debug, Clone, Copy)]
struct ArcData {
    label: u8,
    is_final: bool,
    next: usize,
    target: usize,
}

/// `morfologik.fsa.FSA5`.
#[derive(Debug, Clone)]
pub struct Fsa5 {
    arcs: Vec<u8>,
    node_data_length: usize,
    gtl: usize,
}

impl Fsa5 {
    // Java: FSA5(InputStream) -- `bytes` is the whole file, its first
    // `start` bytes (magic and version) already read. The arcs keep the
    // file's allocation: the header is drained in place.
    fn read(mut bytes: Vec<u8>, start: usize) -> Result<Fsa5, MorfologikError> {
        let Some(&[_filler, _annotation, hgtl]) = bytes.get(start..).and_then(|b| b.get(..3))
        else {
            return Err(MorfologikError::new("EOFException: truncated FSA5 header"));
        };
        // ARITH: `start + 3` bytes were just read, so the sum is in range.
        #[allow(clippy::arithmetic_side_effects)]
        bytes.drain(..start + 3);
        Ok(Fsa5 {
            arcs: bytes,
            node_data_length: usize::from((hgtl >> 4) & 0x0F),
            gtl: usize::from(hgtl & 0x0F),
        })
    }

    // Java: FSA5.getArc
    fn find_arc(&self, node: usize, label: u8) -> Result<usize, MorfologikError> {
        let mut arc = self.first_arc(node)?;
        while arc != 0 {
            if byte(&self.arcs, arc)? == label {
                return Ok(arc);
            }
            arc = if self.is_last(arc)? {
                0
            } else {
                self.skip_arc(arc)?
            };
        }
        Ok(0)
    }

    /// The arc at `arc`, read once.
    fn decode(&self, arc: usize) -> Result<ArcData, MorfologikError> {
        let label = byte(&self.arcs, arc)?;
        let flags = self.flags(arc)?;
        Ok(ArcData {
            label,
            is_final: flags & 1 != 0,
            next: if flags & 2 != 0 {
                0
            } else {
                self.skip_arc(arc)?
            },
            target: self.destination(arc)?,
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
    // Java: CFSA2(InputStream) -- as `Fsa5::read`.
    fn read(mut bytes: Vec<u8>, start: usize) -> Result<Cfsa2, MorfologikError> {
        let body = bytes.get(start..).unwrap_or_default();
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
        let has_numbers = flags & NUMBERS != 0;
        // ARITH: the header (`start + 3` bytes) and the `size` mapping bytes
        // were just read, so the sum is at most `bytes.len()`.
        #[allow(clippy::arithmetic_side_effects)]
        bytes.drain(..start + 3 + size);
        Ok(Cfsa2 {
            arcs: bytes,
            label_mapping,
            has_numbers,
        })
    }

    /// `getArcLabel` of the arc at `arc`, whose flag byte is `flag`, and
    /// the offset past the label.
    #[inline]
    fn label_at(&self, arc: usize, flag: u8) -> Result<(u8, usize), MorfologikError> {
        let index = usize::from(flag & 0x1F);
        // ARITH: `arc` indexes `arcs` (its flag was read), so `arc + 1` is at
        // most `arcs.len()`; `arc + 2` follows a successful read of `arc + 1`.
        #[allow(clippy::arithmetic_side_effects)]
        if index > 0 {
            let label = self.label_mapping.get(index).copied().ok_or_else(corrupt)?;
            Ok((label, arc + 1))
        } else {
            Ok((byte(&self.arcs, arc + 1)?, arc + 2))
        }
    }

    // Java: CFSA2.getArc -- `getArcLabel` and `getNextArc` with the flag
    // byte read once per arc.
    fn find_arc(&self, node: usize, label: u8) -> Result<usize, MorfologikError> {
        let mut arc = self.first_arc(node)?;
        if arc == 0 {
            return Ok(0);
        }
        loop {
            let flag = byte(&self.arcs, arc)?;
            let (l, after) = self.label_at(arc, flag)?;
            if l == label {
                return Ok(arc);
            }
            if flag & 0x40 != 0 {
                return Ok(0);
            }
            arc = if flag & 0x80 != 0 {
                after
            } else {
                self.skip_vint(after)?
            };
        }
    }

    /// The arc at `arc`, read once.
    fn decode(&self, arc: usize) -> Result<ArcData, MorfologikError> {
        let flag = byte(&self.arcs, arc)?;
        let (label, after) = self.label_at(arc, flag)?;
        let (target, after) = if flag & 0x80 == 0 {
            self.read_vint_at(after)?
        } else if flag & 0x40 != 0 {
            // Target-next on the node's last arc: the node right after it.
            (after, after)
        } else {
            (self.destination(arc)?, after)
        };
        Ok(ArcData {
            label,
            is_final: flag & 0x20 != 0,
            next: if flag & 0x40 != 0 { 0 } else { after },
            target,
        })
    }

    // Java: CFSA2.readVInt -- a Java int: shifts past 31 wrap, as `<<` does.
    fn read_vint(&self, offset: usize) -> Result<usize, MorfologikError> {
        Ok(self.read_vint_at(offset)?.0)
    }

    /// `readVInt` at `offset` and the offset past the v-int.
    // ARITH: `offset` is only advanced past a byte just read from `arcs`, so
    // it stays at most `arcs.len()`.
    #[allow(clippy::arithmetic_side_effects)]
    #[inline]
    fn read_vint_at(&self, mut offset: usize) -> Result<(usize, usize), MorfologikError> {
        let mut b = byte(&self.arcs, offset)?;
        offset += 1;
        let mut value: i32 = i32::from(b & 0x7F);
        let mut shift: u32 = 7;
        while b & 0x80 != 0 {
            b = byte(&self.arcs, offset)?;
            offset += 1;
            value |= i32::from(b & 0x7F).wrapping_shl(shift);
            shift = shift.wrapping_add(7);
        }
        Ok((usize::try_from(value).map_err(|_| corrupt())?, offset))
    }

    // Java: CFSA2.skipVInt
    // ARITH: as `read_vint_at`.
    #[allow(clippy::arithmetic_side_effects)]
    #[inline]
    fn skip_vint(&self, mut offset: usize) -> Result<usize, MorfologikError> {
        loop {
            let b = byte(&self.arcs, offset)?;
            offset += 1;
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

/// The longest sequence [`Fsa::visit_sequences`] follows (and no longer
/// than the automaton has arc bytes, which no acyclic path can exceed): a
/// cycle in a corrupt automaton would otherwise never end.
const MAX_DEPTH: usize = 1 << 16;

/// The most sequences one [`Fsa::visit_sequences`] reports. Morfologik has
/// no bound; a 102-byte automaton of two-way branches holds 2^18.
pub const MAX_SEQUENCES: usize = 1 << 16;

/// The most bytes (summed over the sequences) one
/// [`Fsa::visit_sequences`] reports: a chain of `n` final arcs holds
/// `n(n+1)/2`.
pub const MAX_SEQUENCE_BYTES: usize = 1 << 24;

/// The stack and buffer [`Fsa::visit_sequences`] reuses
/// (`ByteSequenceIterator`'s `arcs` and `buffer`).
#[derive(Debug, Clone, Default)]
pub struct SequenceScratch {
    arcs: Vec<usize>,
    buffer: Vec<u8>,
}

impl Fsa {
    /// `FSA.read(InputStream)`.
    pub fn read(bytes: &[u8]) -> Result<Fsa, MorfologikError> {
        Fsa::from_vec(bytes.to_vec())
    }

    /// `FSA.read(InputStream)` over an owned file: the automaton keeps the
    /// file's allocation.
    pub fn from_vec(bytes: Vec<u8>) -> Result<Fsa, MorfologikError> {
        if !bytes.starts_with(b"\\fsa") {
            return Err(MorfologikError::new(
                "IOException: Invalid file header, probably not an FSA.",
            ));
        }
        let Some(&version) = bytes.get(4) else {
            return Err(MorfologikError::new(
                "IOException: Truncated file, no version number.",
            ));
        };
        match version {
            5 => Ok(Fsa::Fsa5(Fsa5::read(bytes, 5)?)),
            0xC6 => Ok(Fsa::Cfsa2(Cfsa2::read(bytes, 5)?)),
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
        match self {
            Fsa::Fsa5(f) => f.find_arc(node, label),
            Fsa::Cfsa2(c) => c.find_arc(node, label),
        }
    }

    fn decode(&self, arc: usize) -> Result<ArcData, MorfologikError> {
        match self {
            Fsa::Fsa5(f) => f.decode(arc),
            Fsa::Cfsa2(c) => c.decode(arc),
        }
    }

    fn arcs_len(&self) -> usize {
        match self {
            Fsa::Fsa5(f) => f.arcs.len(),
            Fsa::Cfsa2(c) => c.arcs.len(),
        }
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

    /// `ByteSequenceIterator` from `node`, collected: every sequence its
    /// final arcs end, depth first in arc order.
    pub fn sequences(&self, node: usize) -> Result<Vec<Vec<u8>>, MorfologikError> {
        let mut out = Vec::new();
        self.visit_sequences(node, &mut SequenceScratch::default(), |s| {
            out.push(s.to_vec());
            Ok(())
        })?;
        Ok(out)
    }

    /// `ByteSequenceIterator` from `node`: `visit` sees every sequence its
    /// final arcs end, depth first in arc order, in `scratch`'s buffer (as
    /// Java's iterator hands out one reused `ByteBuffer`). A path deeper
    /// than the automaton has arc bytes (a cycle), more than
    /// [`MAX_SEQUENCES`] sequences or more than [`MAX_SEQUENCE_BYTES`] of
    /// them are errors; Morfologik loops or runs out of memory there.
    pub fn visit_sequences<F>(
        &self,
        node: usize,
        scratch: &mut SequenceScratch,
        mut visit: F,
    ) -> Result<(), MorfologikError>
    where
        F: FnMut(&[u8]) -> Result<(), MorfologikError>,
    {
        let SequenceScratch { arcs, buffer } = scratch;
        arcs.clear();
        buffer.clear();
        let first = self.first_arc(node)?;
        if first == 0 {
            return Ok(());
        }
        let max_depth = MAX_DEPTH.min(self.arcs_len());
        arcs.push(first);
        let (mut count, mut bytes) = (0usize, 0usize);
        while let Some(&arc) = arcs.last() {
            // ARITH: `arcs` is not empty.
            #[allow(clippy::arithmetic_side_effects)]
            let last_index = arcs.len() - 1;
            if arc == 0 {
                arcs.pop();
                continue;
            }
            let a = self.decode(arc)?;
            arcs[last_index] = a.next;
            buffer.truncate(last_index);
            buffer.push(a.label);
            if a.target != 0 {
                if arcs.len() > max_depth {
                    return Err(MorfologikError::new(
                        "IOException: the automaton has a cycle",
                    ));
                }
                arcs.push(self.first_arc(a.target)?);
            }
            if a.is_final {
                count = count.saturating_add(1);
                bytes = bytes.saturating_add(buffer.len());
                if count > MAX_SEQUENCES {
                    return Err(MorfologikError::new(format!(
                        "IOException: a lookup reaches more than {MAX_SEQUENCES} sequences"
                    )));
                }
                if bytes > MAX_SEQUENCE_BYTES {
                    return Err(MorfologikError::new(format!(
                        "IOException: a lookup reaches more than {MAX_SEQUENCE_BYTES} bytes of sequences"
                    )));
                }
                visit(buffer)?;
            }
        }
        Ok(())
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

    /// A CFSA2 whose root reads `a`, then `+`, then the node at 9: `tail`.
    fn after_separator(tail: &[u8]) -> Vec<u8> {
        let mut f = b"\\fsa\xC6\x00\x00\x00".to_vec();
        f.extend_from_slice(&[0x40, b'x', 3, 0x40, b'a', 6, 0x40, b'+', 9]);
        f.extend_from_slice(tail);
        f
    }

    #[test]
    fn a_final_cycle_stops_at_the_automatons_size() {
        // 9: 'b', final and last, back to 9 -- every "b...b" is a sequence.
        // Bounded only by the 65,536-level cap, the walk held 2 GB of them
        // before it stopped.
        let fsa = Fsa::read(&after_separator(&[0x60, b'b', 9])).unwrap();
        let e = fsa.sequences(9).unwrap_err();
        assert!(e.message().contains("cycle"), "{}", e.message());
    }

    #[test]
    fn a_lookups_sequences_are_capped() {
        // 17 levels of two final arcs each: 2^18 - 2 sequences from 102
        // bytes of arcs.
        let mut dag = Vec::new();
        for k in 0..17u8 {
            let next = if k == 16 { 0 } else { 9 + 6 * (k + 1) };
            dag.extend_from_slice(&[0x20, b'b', next, 0x60, b'c', next]);
        }
        let fsa = Fsa::read(&after_separator(&dag)).unwrap();
        let e = fsa.sequences(9).unwrap_err();
        assert!(e.message().contains("65536 sequences"), "{}", e.message());
        // A chain of n final arcs: n sequences, n(n+1)/2 bytes.
        let chain = |n: usize| {
            let mut c = Vec::new();
            for _ in 1..n {
                c.extend_from_slice(&[0xE0, b'b']); // final, last, target next
            }
            c.extend_from_slice(&[0x60, b'b', 0]);
            Fsa::read(&after_separator(&c)).unwrap()
        };
        let e = chain(6000).sequences(9).unwrap_err();
        assert!(e.message().contains("bytes"), "{}", e.message());
        let ok = chain(100).sequences(9).unwrap();
        assert_eq!(ok.len(), 100);
        assert_eq!(ok[99], vec![b'b'; 100]);
    }

    #[test]
    fn target_next_and_mapped_labels() {
        // Mapping [_, 'q']. Node 3: 'b' (final, target next, not last),
        // 'c' (final, last, terminal); node 8: mapped 'q' (final, last,
        // terminal).
        let mut f = b"\\fsa\xC6\x00\x00\x02\x00q".to_vec();
        f.extend_from_slice(&[0x40, 0x00, 0x03, 0xA0, b'b', 0x60, b'c', 0x00, 0x61, 0x00]);
        let fsa = Fsa::read(&f).unwrap();
        assert_eq!(fsa.root().unwrap(), 3);
        assert_eq!(
            fsa.sequences(3).unwrap(),
            vec![b"b".to_vec(), b"bq".to_vec(), b"c".to_vec()]
        );
        assert_eq!(fsa.arc(3, b'c').unwrap(), 5);
        assert_eq!(fsa.arc(3, b'x').unwrap(), 0);
        assert_eq!(fsa.arc(8, b'q').unwrap(), 8);
        assert_eq!(fsa.label(8).unwrap(), b'q');
        assert_eq!(fsa.next_arc(3).unwrap(), 5);
        assert_eq!(fsa.next_arc(5).unwrap(), 0);
        assert!(fsa.is_final(3).unwrap());
        assert_eq!(fsa.end_node(3).unwrap(), 8);
        assert_eq!(fsa.end_node(5).unwrap(), 0);
        assert_eq!(fsa.match_sequence(b"bq", 3).unwrap(), Match::Exact);
        // The same scans over a truncated copy fail cleanly.
        let fsa = Fsa::read(&f[..f.len() - 3]).unwrap();
        assert!(fsa.sequences(3).is_err());
        assert!(fsa.arc(8, b'x').is_err());
    }

    #[test]
    fn fsa5_arcs_one_by_one() {
        let arc = |label: u8, target: u16, flags: u16| {
            let v = (target << 3) | flags;
            [label, (v & 0xFF) as u8, (v >> 8) as u8]
        };
        // Root 6: 'a' (not last) -> 12, 'b' (final, last, terminal);
        // 12: 'c' (final, last, terminal).
        let mut f = b"\\fsa\x05_+\x02".to_vec();
        f.extend_from_slice(&arc(0, 0, 2));
        f.extend_from_slice(&arc(0, 6, 2));
        f.extend_from_slice(&arc(b'a', 12, 0));
        f.extend_from_slice(&arc(b'b', 0, 3));
        f.extend_from_slice(&arc(b'c', 0, 3));
        let fsa = Fsa::from_vec(f.clone()).unwrap();
        assert_eq!(fsa.arc(6, b'b').unwrap(), 9);
        assert_eq!(fsa.arc(6, b'x').unwrap(), 0);
        assert_eq!(fsa.label(9).unwrap(), b'b');
        assert_eq!(fsa.next_arc(6).unwrap(), 9);
        assert_eq!(fsa.next_arc(9).unwrap(), 0);
        assert!(!fsa.is_final(6).unwrap());
        assert_eq!(fsa.end_node(6).unwrap(), 12);
        assert_eq!(
            fsa.sequences(6).unwrap(),
            vec![b"ac".to_vec(), b"b".to_vec()]
        );
        let fsa = Fsa::read(&f[..f.len() - 2]).unwrap();
        assert!(fsa.sequences(6).is_err());
        assert!(fsa.arc(12, b'x').is_err());
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
