//! Hunspell flags: `Dictionary.FlagParsingStrategy` (the four `FLAG`
//! encodings) and `FlagEnumerator` (deduplicated, sorted flag sets).
//!
//! A flag is a Java `char` (`u16`), as in Lucene: `FLAG long` packs two
//! ASCII characters into one, `FLAG num` stores the number.

use std::collections::HashMap;

use super::{HunspellError, FLAG_UNSET};

/// `Dictionary.DEFAULT_FLAGS`: flags at or above this are internal (the
/// hidden flag) and never printed.
pub(crate) const DEFAULT_FLAGS: u16 = 65510;

/// `Dictionary.FlagParsingStrategy` and its four subclasses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FlagParsing {
    /// `SimpleFlagParsingStrategy`: each character is a flag.
    Simple,
    /// `DefaultAsUtf8FlagParsingStrategy`: `FLAG UTF-8` in an ISO-8859-1
    /// file -- the raw flag bytes are re-decoded as UTF-8.
    DefaultAsUtf8,
    /// `NumFlagParsingStrategy`: comma-separated decimal numbers.
    Num,
    /// `DoubleASCIIFlagParsingStrategy`: two ASCII characters per flag.
    DoubleAscii,
}

impl FlagParsing {
    /// `FlagParsingStrategy.parseFlags`.
    pub(crate) fn parse_flags(self, raw: &[u16]) -> Result<Vec<u16>, HunspellError> {
        match self {
            FlagParsing::Simple => Ok(raw.to_vec()),
            FlagParsing::DefaultAsUtf8 => {
                // `new String(raw.getBytes(ISO_8859_1), UTF_8)`: unmappable
                // units become `?`, malformed UTF-8 becomes U+FFFD.
                let bytes: Vec<u8> = raw
                    .iter()
                    .map(|&u| if u < 256 { u as u8 } else { b'?' })
                    .collect();
                Ok(String::from_utf8_lossy(&bytes).encode_utf16().collect())
            }
            FlagParsing::Num => {
                let mut result = Vec::new();
                let mut group = String::new();
                for i in 0..=raw.len() {
                    if i == raw.len() || raw[i] == u16::from(b',') {
                        if !group.is_empty() {
                            // `Integer.parseInt`: past `i32::MAX` is a
                            // `NumberFormatException`.
                            let flag: i32 = group.parse().map_err(|_| {
                                HunspellError::NumberFormat(format!(
                                    "For input string: \"{group}\""
                                ))
                            })?;
                            if flag >= i32::from(DEFAULT_FLAGS) {
                                return Err(HunspellError::IllegalArgument(format!(
                                    "Num flags should be between 0 and {DEFAULT_FLAGS}, found {flag}"
                                )));
                            }
                            result.push(flag as u16);
                            group.clear();
                        }
                    } else if (u16::from(b'0')..=u16::from(b'9')).contains(&raw[i]) {
                        group.push(char::from(raw[i] as u8));
                    }
                }
                Ok(result)
            }
            FlagParsing::DoubleAscii => {
                let mut flags = Vec::with_capacity(raw.len() / 2);
                for pair in raw.chunks_exact(2) {
                    let (f1, f2) = (pair[0], pair[1]);
                    if f1 >= 256 || f2 >= 256 {
                        return Err(HunspellError::IllegalArgument(format!(
                            "Invalid flags (LONG flags must be double ASCII): {}",
                            String::from_utf16_lossy(raw)
                        )));
                    }
                    flags.push(f1 << 8 | f2);
                }
                Ok(flags)
            }
        }
    }

    /// `FlagParsingStrategy.parseFlag`: the first flag of `raw`
    /// (`checkFlags` is off, so longer sequences are accepted).
    pub(crate) fn parse_flag(self, raw: &[u16]) -> Result<u16, HunspellError> {
        self.parse_flags(raw)?.first().copied().ok_or_else(|| {
            HunspellError::IndexOutOfBounds(format!(
                "Index 0 out of bounds for length 0: {}",
                String::from_utf16_lossy(raw)
            ))
        })
    }

    /// `FlagParsingStrategy.parseUtfFlags`: flags written in the dictionary's
    /// own text (a compound rule), already decoded.
    pub(crate) fn parse_utf_flags(self, raw: &[u16]) -> Result<Vec<u16>, HunspellError> {
        match self {
            FlagParsing::DefaultAsUtf8 => Ok(raw.to_vec()),
            other => other.parse_flags(raw),
        }
    }

    /// `FlagParsingStrategy.printFlag`.
    pub(crate) fn print_flag(self, flag: u16) -> String {
        match self {
            FlagParsing::Num => flag.to_string(),
            FlagParsing::DoubleAscii => String::from_utf16_lossy(&[flag >> 8, flag & 0xff]),
            FlagParsing::Simple | FlagParsing::DefaultAsUtf8 => String::from_utf16_lossy(&[flag]),
        }
    }

    /// `FlagParsingStrategy.printFlags`: internal flags dropped, the rest
    /// printed and sorted (`,`-joined for `num`).
    pub(crate) fn print_flags(self, flags: &[u16]) -> String {
        let mut printed: Vec<String> = flags
            .iter()
            .filter(|&&f| f < DEFAULT_FLAGS)
            .map(|&f| self.print_flag(f))
            .collect();
        printed.sort_by(|a, b| a.encode_utf16().cmp(b.encode_utf16()));
        printed.join(if self == FlagParsing::Num { "," } else { "" })
    }
}

/// `FlagEnumerator`: deduplicates sorted flag sets, giving each the offset
/// of its `[length, flags...]` record in one `u16` array.
#[derive(Debug)]
pub(crate) struct FlagEnumerator {
    data: Vec<u16>,
    indices: HashMap<Vec<u16>, i32>,
}

impl FlagEnumerator {
    /// `new FlagEnumerator()`: the empty set is id 0.
    pub(crate) fn new() -> Self {
        let mut e = FlagEnumerator {
            data: Vec::new(),
            indices: HashMap::new(),
        };
        e.add(&mut Vec::new())
            .expect("the empty flag set always fits");
        e
    }

    /// `FlagEnumerator.add`: sorts `flags` in place and returns its id.
    pub(crate) fn add(&mut self, flags: &mut Vec<u16>) -> Result<i32, HunspellError> {
        flags.sort_unstable();
        if flags.len() > usize::from(u16::MAX) {
            return Err(HunspellError::IllegalArgument(format!(
                "Too many flags: {}",
                String::from_utf16_lossy(flags)
            )));
        }
        if let Some(&id) = self.indices.get(flags.as_slice()) {
            return Ok(id);
        }
        let id = self.data.len() as i32;
        self.indices.insert(flags.clone(), id);
        self.data.push(flags.len() as u16);
        self.data.extend_from_slice(flags);
        Ok(id)
    }

    /// `FlagEnumerator.finish`.
    pub(crate) fn finish(self) -> FlagLookup {
        FlagLookup { data: self.data }
    }
}

/// `FlagEnumerator.hasFlagInSortedArray`.
fn has_flag_in_sorted(flag: u16, flags: &[u16]) -> bool {
    if flag == FLAG_UNSET {
        return false;
    }
    for &c in flags {
        if c == flag {
            return true;
        }
        if c > flag {
            return false;
        }
    }
    false
}

/// `FlagEnumerator.Lookup`.
#[derive(Debug, Default)]
pub(crate) struct FlagLookup {
    data: Vec<u16>,
}

impl FlagLookup {
    fn flags_of(&self, entry_id: i32) -> &[u16] {
        let start = entry_id as usize;
        let len = usize::from(self.data[start]);
        &self.data[start + 1..start + 1 + len]
    }

    /// `Lookup.hasFlag` (a negative id has no flags).
    pub(crate) fn has_flag(&self, entry_id: i32, flag: u16) -> bool {
        entry_id >= 0 && has_flag_in_sorted(flag, self.flags_of(entry_id))
    }

    /// `Lookup.getFlags`.
    pub(crate) fn get_flags(&self, entry_id: i32) -> Vec<u16> {
        self.flags_of(entry_id).to_vec()
    }
}
