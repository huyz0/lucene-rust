//! `CharBlockArray`: an append-only UTF-16 character sequence stored in
//! fixed-size blocks, so growing it never copies what is already there.
//! (Package-private in Java, and used by nothing else in Lucene 10.5.0; kept
//! for the package's completeness.)

use crate::{Error, Result};

/// `DefaultBlockSize`: 32K characters.
const DEFAULT_BLOCK_SIZE: usize = 32 * 1024;

#[derive(Debug, Clone)]
struct Block {
    chars: Vec<u16>,
}

/// `CharBlockArray`.
#[derive(Debug, Clone)]
pub struct CharBlockArray {
    blocks: Vec<Block>,
    block_size: usize,
    length: usize,
}

impl Default for CharBlockArray {
    fn default() -> Self {
        Self::new()
    }
}

impl CharBlockArray {
    /// `new CharBlockArray()`.
    pub fn new() -> Self {
        Self::with_block_size(DEFAULT_BLOCK_SIZE)
    }

    /// `new CharBlockArray(blockSize)`; a zero block size is taken as one.
    pub fn with_block_size(block_size: usize) -> Self {
        let block_size = block_size.max(1);
        CharBlockArray {
            blocks: vec![Block {
                chars: Vec::with_capacity(block_size.min(DEFAULT_BLOCK_SIZE)),
            }],
            block_size,
            length: 0,
        }
    }

    /// `addBlock()`: refused past 2 GB of characters, as in Java.
    fn add_block(&mut self) -> Result<()> {
        let total = (self.blocks.len() as u64)
            .saturating_add(1)
            .saturating_mul(self.block_size as u64);
        if total > i32::MAX as u64 {
            return Err(Error::IllegalState(
                "cannot store more than 2 GB in CharBlockArray".to_string(),
            ));
        }
        self.blocks.push(Block {
            chars: Vec::with_capacity(self.block_size.min(DEFAULT_BLOCK_SIZE)),
        });
        Ok(())
    }

    fn current_full(&self) -> bool {
        self.blocks
            .last()
            .is_none_or(|b| b.chars.len() == self.block_size)
    }

    /// `blockIndex(index)`.
    pub fn block_index(&self, index: usize) -> usize {
        index / self.block_size
    }

    /// `indexInBlock(index)`.
    pub fn index_in_block(&self, index: usize) -> usize {
        index % self.block_size
    }

    /// `append(char)`.
    pub fn append_char(&mut self, c: u16) -> Result<&mut Self> {
        self.append_units(&[c])
    }

    /// `append(char[], start, length)`.
    pub fn append_units(&mut self, chars: &[u16]) -> Result<&mut Self> {
        let mut rest = chars;
        while !rest.is_empty() {
            if self.current_full() {
                self.add_block()?;
            }
            let current = self.blocks.last_mut().expect("at least one block");
            let room = self.block_size.saturating_sub(current.chars.len());
            let take = room.min(rest.len());
            current.chars.extend_from_slice(&rest[..take]);
            rest = &rest[take..];
            self.length = self.length.saturating_add(take);
        }
        Ok(self)
    }

    /// `append(String)` / `append(CharSequence)`: the string's UTF-16 units.
    pub fn append_str(&mut self, s: &str) -> Result<&mut Self> {
        let units: Vec<u16> = s.encode_utf16().collect();
        self.append_units(&units)
    }

    /// `charAt(index)`: `None` past the end (Java throws).
    pub fn char_at(&self, index: usize) -> Option<u16> {
        if index >= self.length {
            return None;
        }
        self.blocks
            .get(self.block_index(index))?
            .chars
            .get(self.index_in_block(index))
            .copied()
    }

    /// `length()`.
    pub fn len(&self) -> usize {
        self.length
    }

    /// `length() == 0`.
    pub fn is_empty(&self) -> bool {
        self.length == 0
    }

    /// `subSequence(start, end)` as UTF-16 units; `None` for a range outside
    /// the array (Java throws).
    pub fn sub_sequence_units(&self, start: usize, end: usize) -> Option<Vec<u16>> {
        if start > end || end > self.length {
            return None;
        }
        let mut out = Vec::with_capacity(end.saturating_sub(start));
        let mut block = self.block_index(start);
        let mut in_block = self.index_in_block(start);
        while out.len() < end.saturating_sub(start) {
            let b = &self.blocks[block];
            let take = end
                .saturating_sub(start)
                .saturating_sub(out.len())
                .min(b.chars.len().saturating_sub(in_block));
            out.extend_from_slice(&b.chars[in_block..in_block.saturating_add(take)]);
            block = block.saturating_add(1);
            in_block = 0;
        }
        Some(out)
    }

    /// `subSequence(start, end).toString()`.
    pub fn sub_sequence(&self, start: usize, end: usize) -> Option<String> {
        self.sub_sequence_units(start, end)
            .map(|u| String::from_utf16_lossy(&u))
    }
}

impl std::fmt::Display for CharBlockArray {
    /// `toString()`.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let units: Vec<u16> = self
            .blocks
            .iter()
            .flat_map(|b| b.chars.iter().copied())
            .collect();
        f.write_str(&String::from_utf16_lossy(&units))
    }
}

#[cfg(test)]
mod tests;
