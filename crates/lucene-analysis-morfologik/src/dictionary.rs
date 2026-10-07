//! Morfologik's `Dictionary` and `DictionaryLookup` (`morfologik.stemming`,
//! 2.1.9): an automaton of `surface SEP encoded-base SEP tag` byte
//! sequences, and the lookup of a word's `(base, tag)` pairs, with the four
//! `ISequenceEncoder`s' `decode`.
//!
//! Morfologik reuses its `WordData` objects across lookups, and a form
//! whose entry has no tag keeps the previous form's tag bytes at that
//! position (`WordData.update` only `clear()`s the buffer, so its limit is
//! the old capacity): [`DictionaryLookup`] keeps the same buffers per
//! position, so the port answers what Java answers there too.

use std::sync::Arc;

use crate::fsa::{Fsa, Match};
use crate::metadata::{DictionaryMetadata, EncoderType};
use crate::MorfologikError;

/// `morfologik.stemming.Dictionary`: an automaton and its metadata.
#[derive(Debug, Clone)]
pub struct Dictionary {
    fsa: Fsa,
    metadata: DictionaryMetadata,
    root: usize,
}

impl Dictionary {
    /// `Dictionary.read(InputStream fsa, InputStream metadata)`.
    pub fn read(fsa: &[u8], metadata: &str) -> Result<Dictionary, MorfologikError> {
        let fsa = Fsa::read(fsa)?;
        let metadata = DictionaryMetadata::read(metadata)?;
        let root = fsa.root()?;
        Ok(Dictionary {
            fsa,
            metadata,
            root,
        })
    }

    /// The metadata.
    pub fn metadata(&self) -> &DictionaryMetadata {
        &self.metadata
    }

    /// The automaton.
    pub fn fsa(&self) -> &Fsa {
        &self.fsa
    }
}

/// `ISequenceEncoder.decode(reuse, source, encoded)`: the base form's bytes
/// from the surface form's and the entry's encoded part.
pub fn decode(
    encoder: EncoderType,
    source: &[u8],
    encoded: &[u8],
) -> Result<Vec<u8>, MorfologikError> {
    const REMOVE_EVERYTHING: usize = 255;
    let bad = || {
        MorfologikError::new("IndexOutOfBoundsException: a dictionary entry decodes past its word")
    };
    let code = |i: usize| -> Result<usize, MorfologikError> {
        Ok(usize::from(
            encoded.get(i).ok_or_else(bad)?.wrapping_sub(b'A'),
        ))
    };
    let tail = |from: usize| encoded.get(from..).ok_or_else(bad);
    let mut out = Vec::new();
    match encoder {
        EncoderType::None => out.extend_from_slice(encoded),
        EncoderType::Suffix => {
            let mut truncate = code(0)?;
            if truncate == REMOVE_EVERYTHING {
                truncate = source.len();
            }
            let keep = source.len().checked_sub(truncate).ok_or_else(bad)?;
            out.extend_from_slice(&source[..keep]);
            out.extend_from_slice(tail(1)?);
        }
        EncoderType::Prefix => {
            let (mut prefix, mut suffix) = (code(0)?, code(1)?);
            if prefix == REMOVE_EVERYTHING || suffix == REMOVE_EVERYTHING {
                prefix = source.len();
                suffix = 0;
            }
            let end = source.len().checked_sub(suffix).ok_or_else(bad)?;
            out.extend_from_slice(source.get(prefix..end).ok_or_else(bad)?);
            out.extend_from_slice(tail(2)?);
        }
        EncoderType::Infix => {
            let (mut index, mut length, mut suffix) = (code(0)?, code(1)?, code(2)?);
            if length == REMOVE_EVERYTHING || suffix == REMOVE_EVERYTHING {
                index = 0;
                length = source.len();
                suffix = 0;
            }
            let after = index.checked_add(length).ok_or_else(bad)?;
            let end = source.len().checked_sub(suffix).ok_or_else(bad)?;
            out.extend_from_slice(source.get(..index).ok_or_else(bad)?);
            out.extend_from_slice(source.get(after..end).ok_or_else(bad)?);
            out.extend_from_slice(tail(3)?);
        }
    }
    Ok(out)
}

/// `DictionaryLookup.applyReplacements`: each pair in order, every
/// occurrence left to right.
pub fn apply_replacements(word: &[u16], pairs: &[(Vec<u16>, Vec<u16>)]) -> Vec<u16> {
    let mut sb = word.to_vec();
    for (key, value) in pairs {
        if key.is_empty() {
            continue;
        }
        let mut from = 0;
        while let Some(i) = sb
            .get(from..)
            .and_then(|s| s.windows(key.len()).position(|w| w == key.as_slice()))
        {
            // ARITH: from + i is an index into sb.
            #[allow(clippy::arithmetic_side_effects)]
            let at = from + i;
            sb.splice(at..at.saturating_add(key.len()), value.iter().copied());
            // Java resumes at index + key.length() in the edited text.
            from = at.saturating_add(key.len());
        }
    }
    sb
}

/// One reused `WordData`: the stem and the tag's backing buffer, whose
/// whole capacity a tagless entry reads.
#[derive(Debug, Clone, Default)]
struct Slot {
    stem: Vec<u8>,
    tag_buf: Vec<u8>,
    tag_len: usize,
}

/// A form a lookup found: `WordData.getStem()` and `getTag()`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WordData {
    /// The base form; `None` for an empty one (Java's `null`).
    pub stem: Option<Vec<u16>>,
    /// The tag; `None` for none.
    pub tag: Option<Vec<u16>>,
}

/// `morfologik.stemming.DictionaryLookup`, with its reused `WordData`s.
#[derive(Debug)]
pub struct DictionaryLookup {
    dictionary: Arc<Dictionary>,
    slots: Vec<Slot>,
}

impl DictionaryLookup {
    /// `new DictionaryLookup(dictionary)`.
    pub fn new(dictionary: Arc<Dictionary>) -> Self {
        DictionaryLookup {
            dictionary,
            slots: Vec::new(),
        }
    }

    /// `DictionaryLookup.lookup(word)`: every `(stem, tag)` of the word's
    /// entries, in the automaton's order; empty for a word holding the
    /// separator or that the charset cannot encode.
    pub fn lookup(&mut self, word: &[u16]) -> Result<Vec<WordData>, MorfologikError> {
        let d = Arc::clone(&self.dictionary);
        let m = &d.metadata;
        let converted;
        let word = if m.input_conversion.is_empty() {
            word
        } else {
            converted = apply_replacements(word, &m.input_conversion);
            &converted
        };
        if word.contains(&m.separator_char) {
            return Ok(Vec::new());
        }
        let Some(bytes) = m.charset.encode(word) else {
            return Ok(Vec::new());
        };
        let Match::SequenceIsAPrefix(node) = d.fsa.match_sequence(&bytes, d.root)? else {
            return Ok(Vec::new());
        };
        let arc = d.fsa.arc(node, m.separator)?;
        if arc == 0 || d.fsa.is_final(arc)? {
            return Ok(Vec::new());
        }
        let end = d.fsa.end_node(arc)?;
        let sequences = if end == 0 {
            Vec::new()
        } else {
            d.fsa.sequences(end)?
        };
        let prefix_bytes = m.encoder.prefix_bytes();
        let mut out = Vec::with_capacity(sequences.len());
        for (k, ba) in sequences.iter().enumerate() {
            if self.slots.len() <= k {
                self.slots.resize_with(k.saturating_add(10), Slot::default);
            }
            let slot = &mut self.slots[k];
            // WordData.update: the tag buffer cleared to its capacity.
            slot.tag_len = slot.tag_buf.len();
            let sep_pos = ba
                .iter()
                .enumerate()
                .skip(prefix_bytes)
                .find(|&(_, &b)| b == m.separator)
                .map_or(ba.len(), |(i, _)| i);
            if ba.len() < prefix_bytes {
                // Java asserts this (and, without -ea, reads stale bytes).
                return Err(MorfologikError::new(
                    "AssertionError: an entry shorter than its encoder's prefix",
                ));
            }
            slot.stem = decode(m.encoder, &bytes, &ba[..sep_pos])?;
            let tag_start = sep_pos.saturating_add(1);
            if let Some(tag) = ba.get(tag_start..).filter(|t| !t.is_empty()) {
                if slot.tag_buf.len() < tag.len() {
                    slot.tag_buf = vec![0; tag.len()];
                }
                slot.tag_buf[..tag.len()].copy_from_slice(tag);
                slot.tag_len = tag.len();
            }
            let decode_chars = |b: &[u8]| -> Result<Option<Vec<u16>>, MorfologikError> {
                let chars = m.charset.decode(b).ok_or_else(|| {
                    MorfologikError::new("RuntimeException: Input cannot be mapped to bytes using the dictionary's encoding")
                })?;
                Ok((!chars.is_empty()).then_some(chars))
            };
            out.push(WordData {
                stem: decode_chars(&slot.stem)?,
                tag: decode_chars(&slot.tag_buf[..slot.tag_len])?,
            });
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decoders() {
        assert_eq!(
            decode(EncoderType::Suffix, b"kotami", b"Dy").unwrap(),
            b"koty"
        );
        assert_eq!(
            decode(EncoderType::Suffix, b"ab", &[b'A'.wrapping_add(255), b'x']).unwrap(),
            b"x"
        );
        assert!(decode(EncoderType::Suffix, b"ab", b"E").is_err());
        assert!(decode(EncoderType::Suffix, b"ab", b"").is_err());
        assert_eq!(
            decode(EncoderType::Prefix, b"najlepszy", b"DAdobry").unwrap(),
            b"lepszydobry"
        );
        assert_eq!(
            decode(
                EncoderType::Prefix,
                b"ab",
                &[b'A'.wrapping_add(255), b'A', b'z']
            )
            .unwrap(),
            b"z"
        );
        assert_eq!(decode(EncoderType::Prefix, b"ab", b"CAz").unwrap(), b"z");
        assert!(decode(EncoderType::Prefix, b"ab", b"DAz").is_err());
        assert_eq!(
            decode(EncoderType::Infix, b"abcde", b"BBAz").unwrap(),
            b"acdez"
        );
        assert_eq!(
            decode(
                EncoderType::Infix,
                b"abcde",
                &[b'A', b'A'.wrapping_add(255), b'A', b'q']
            )
            .unwrap(),
            b"q"
        );
        assert!(decode(EncoderType::Infix, b"ab", b"DAAz").is_err());
        assert!(decode(EncoderType::Infix, b"ab", b"AA").is_err());
        assert_eq!(decode(EncoderType::None, b"ab", b"xy").unwrap(), b"xy");
    }

    /// A CFSA2 holding `a` `+` then `tail` (or a terminal `+` arc).
    fn dict(tail: Option<u8>, encoder: &str) -> Dictionary {
        let mut f = b"\\fsa\xC6\x00\x00\x00".to_vec();
        f.extend_from_slice(&[0x40, 0x00, 0x03, 0x40, b'a', 0x07, 0x00]);
        match tail {
            Some(t) => f.extend_from_slice(&[0x40, b'+', 0x0B, 0x00, 0x60, t, 0x00]),
            None => f.extend_from_slice(&[0x40, b'+', 0x00]),
        }
        let info =
            format!("fsa.dict.separator=+\nfsa.dict.encoding=UTF-8\nfsa.dict.encoder={encoder}");
        Dictionary::read(&f, &info).unwrap()
    }

    #[test]
    fn short_and_empty_entries() {
        let d = dict(Some(b'X'), "PREFIX");
        assert_eq!(d.metadata().encoder, EncoderType::Prefix);
        assert!(d.fsa().root().is_ok());
        let mut l = DictionaryLookup::new(Arc::new(d));
        assert!(l.lookup(&[u16::from(b'a')]).is_err());
        let mut l = DictionaryLookup::new(Arc::new(dict(None, "SUFFIX")));
        assert!(l.lookup(&[u16::from(b'a')]).unwrap().is_empty());
        // A stem that is not UTF-8.
        let mut l = DictionaryLookup::new(Arc::new(dict(Some(0xFF), "NONE")));
        assert!(l.lookup(&[u16::from(b'a')]).is_err());
    }

    #[test]
    fn replacements() {
        let u = |s: &str| s.encode_utf16().collect::<Vec<u16>>();
        let pairs = vec![(u("ab"), u("x")), (u("x"), u("yy")), (u(""), u("q"))];
        // Java resumes `key.length()` past the replacement's start, so the
        // second "ab" (now at 1) is skipped.
        assert_eq!(apply_replacements(&u("abab-x"), &pairs), u("yyab-yy"));
        assert_eq!(apply_replacements(&u("aaa"), &[(u("a"), u("b"))]), u("bbb"));
    }
}
