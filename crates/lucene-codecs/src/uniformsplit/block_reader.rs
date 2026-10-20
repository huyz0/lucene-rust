//! Port of `BlockReader`: the field's terms enum, seeking a block through
//! the dictionary and then the term in the block, from its middle line or
//! its first.

use std::cmp::Ordering;

use super::dictionary::Browser;
use super::{
    corrupt, read_vint, BlockHeader, BlockLineReader, DeltaBaseTermStateSerializer, Error, Result,
    SeekStatus, TermBytes, UniformSplitTermState, UniformSplitTerms,
};
use crate::postings::{DocInput, Postings, PostingsFlags};

/// `BlockReader`: a `TermsEnum` over one field.
pub struct BlockReader<'a> {
    pub(super) terms: &'a UniformSplitTerms,
    /// `dictionaryBrowser`, created on the first seek.
    browser: Option<Browser<'a>>,
    /// `blockInput.getFilePointer()`: just past the current block.
    pub(super) block_input_fp: usize,
    /// `blockStartFP`: `-1` before the first block is read.
    pub(super) block_start_fp: i64,
    /// `blockHeader`.
    pub(super) block_header: Option<BlockHeader>,
    /// The decoded block, when the field has a decoder (`scratchBlockBytes`
    /// otherwise borrows the `.ustb`: [`Self::block_range`]).
    block_decoded: Option<Vec<u8>>,
    /// Where the undecoded block lies in the `.ustb`.
    block_range: (usize, usize),
    /// `blockReadBuffer.getPosition()`, in the block.
    block_pos: usize,
    /// `blockFirstLineStart`.
    block_first_line_start: usize,
    /// `lineIndexInBlock`.
    pub(super) line_index_in_block: i32,
    /// `blockLine`: its term state offset and `mdpLength`; its term is
    /// [`Self::line_reader`]'s.
    pub(super) line: Option<(i32, usize)>,
    /// `blockLineReader`.
    pub(super) line_reader: BlockLineReader,
    /// `termState`.
    pub(super) term_state: Option<UniformSplitTermState>,
    /// `termStateForced`.
    term_state_forced: bool,
    /// `forcedTerm`.
    forced_term: Vec<u8>,
}

impl std::fmt::Debug for BlockReader<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BlockReader")
            .field("field", &self.terms.metadata.field_info.name)
            .field("block_start_fp", &self.block_start_fp)
            .field("line_index_in_block", &self.line_index_in_block)
            .finish()
    }
}

/// The current block's bytes: borrowed from the `.ustb`, or decoded.
fn block_bytes<'x>(
    ustb: &'x [u8],
    decoded: &'x Option<Vec<u8>>,
    range: (usize, usize),
) -> &'x [u8] {
    match decoded {
        Some(bytes) => bytes,
        None => &ustb[range.0..range.1],
    }
}

impl<'a> BlockReader<'a> {
    /// `new BlockReader(dictionaryBrowserSupplier, blockInput,
    /// postingsReader, fieldMetadata, blockDecoder)`.
    pub(crate) fn new(terms: &'a UniformSplitTerms) -> Self {
        BlockReader {
            terms,
            browser: None,
            block_input_fp: 0,
            block_start_fp: -1,
            block_header: None,
            block_decoded: None,
            block_range: (0, 0),
            block_pos: 0,
            block_first_line_start: 0,
            line_index_in_block: 0,
            line: None,
            line_reader: BlockLineReader::default(),
            term_state: None,
            term_state_forced: false,
            forced_term: Vec::new(),
        }
    }

    /// `seekCeil(searchedTerm)`.
    ///
    /// # Errors
    /// `Corrupted` for a malformed block or dictionary.
    pub fn seek_ceil(&mut self, searched_term: &[u8]) -> Result<SeekStatus> {
        if self.is_current_term(searched_term) {
            return Ok(SeekStatus::Found);
        }
        self.clear_term_state();
        let block_start_fp = self
            .get_or_create_dictionary_browser()
            .seek_block(searched_term)?
            .max(self.terms.metadata.first_block_start_fp);
        if self.is_beyond_last_term(searched_term, block_start_fp) {
            return Ok(SeekStatus::End);
        }
        let seek_status = self.seek_in_block_at(searched_term, block_start_fp)?;
        if seek_status != SeekStatus::End {
            return Ok(seek_status);
        }
        Ok(if self.next_term()?.is_none() {
            SeekStatus::End
        } else {
            SeekStatus::NotFound
        })
    }

    /// `seekExact(searchedTerm)`.
    ///
    /// # Errors
    /// `Corrupted` for a malformed block or dictionary.
    pub fn seek_exact(&mut self, searched_term: &[u8]) -> Result<bool> {
        if self.is_current_term(searched_term) {
            return Ok(true);
        }
        self.clear_term_state();
        let block_start_fp = self
            .get_or_create_dictionary_browser()
            .seek_block(searched_term)?;
        if block_start_fp < self.terms.metadata.first_block_start_fp
            || self.is_beyond_last_term(searched_term, block_start_fp)
        {
            return Ok(false);
        }
        Ok(self.seek_in_block_at(searched_term, block_start_fp)? == SeekStatus::Found)
    }

    /// `isCurrentTerm(searchedTerm)`.
    fn is_current_term(&self, searched_term: &[u8]) -> bool {
        self.term() == Some(searched_term)
    }

    /// `isBeyondLastTerm(searchedTerm, blockStartFP)`.
    pub(super) fn is_beyond_last_term(&self, searched_term: &[u8], block_start_fp: i64) -> bool {
        block_start_fp == self.terms.metadata.last_block_start_fp
            && searched_term > self.terms.metadata.last_term.as_slice()
    }

    /// `seekInBlock(searchedTerm, blockStartFP)`.
    pub(super) fn seek_in_block_at(
        &mut self,
        searched_term: &[u8],
        block_start_fp: i64,
    ) -> Result<SeekStatus> {
        self.initialize_header(Some(searched_term), block_start_fp)?;
        if self.block_header.is_none() {
            return Err(self.corrupt_at("Illegal absence of block", block_start_fp));
        }
        self.seek_in_block(searched_term)
    }

    /// `seekInBlock(searchedTerm)`: the scan compares from the first byte
    /// the line before did not already match.
    pub(super) fn seek_in_block(&mut self, searched_term: &[u8]) -> Result<SeekStatus> {
        if self.compare_to_middle_and_jump(searched_term)? == Ordering::Equal {
            return Ok(SeekStatus::Found);
        }
        let mut comparison_offset = 0;
        loop {
            let Some((_, mdp_length)) = self.read_line_in_block()? else {
                return Ok(SeekStatus::End);
            };
            let line_term = &self.line_reader.current_term;
            let suffix_offset = TermBytes::suffix_offset(mdp_length);
            let start = comparison_offset.min(suffix_offset);
            let end = searched_term.len().min(line_term.len());
            let mut comparison = searched_term.len().cmp(&line_term.len());
            for i in start..end {
                let byte_diff = searched_term[i].cmp(&line_term[i]);
                if byte_diff != Ordering::Equal {
                    comparison = byte_diff;
                    break;
                }
                // ARITH: `i < end`, a slice length.
                #[allow(clippy::arithmetic_side_effects)]
                {
                    comparison_offset = i + 1;
                }
            }
            match comparison {
                Ordering::Equal => return Ok(SeekStatus::Found),
                Ordering::Less => return Ok(SeekStatus::NotFound),
                Ordering::Greater => {}
            }
        }
    }

    /// `compareToMiddleAndJump(searchedTerm)`: from the first line, reads
    /// the middle one and goes back to the first unless `searched_term` is
    /// at least the middle line's term.
    fn compare_to_middle_and_jump(&mut self, searched_term: &[u8]) -> Result<Ordering> {
        if self.line_index_in_block != 0 {
            // Java's `-1`: not compared.
            return Ok(Ordering::Less);
        }
        let header = self.header()?;
        let (middle_line_offset, middle_line_index) =
            (header.middle_line_offset, header.middle_line_index);
        // `blockReadBuffer.skipBytes(middleLineOffset)`; checked when the
        // line is read.
        self.block_pos = self
            .block_pos
            .saturating_add(usize::try_from(middle_line_offset).unwrap_or(usize::MAX));
        self.line_index_in_block = middle_line_index;
        if self.read_line_in_block()?.is_none() {
            return Err(self.corrupt_at_none("Illegal absence of line at the middle of the block"));
        }
        let compare = searched_term.cmp(self.line_reader.current_term.as_slice());
        if compare == Ordering::Less {
            self.block_pos = self.block_first_line_start;
            self.line_index_in_block = 0;
        }
        Ok(compare)
    }

    /// `readLineInBlock()`.
    pub(super) fn read_line_in_block(&mut self) -> Result<Option<(i32, usize)>> {
        let header = self.header()?;
        if self.line_index_in_block >= header.lines_count {
            self.line = None;
            return Ok(None);
        }
        let is_incremental_encoding_seed =
            self.line_index_in_block == 0 || self.line_index_in_block == header.middle_line_index;
        // ARITH: below `lines_count`, an `i32`.
        #[allow(clippy::arithmetic_side_effects)]
        {
            self.line_index_in_block += 1;
        }
        let bytes = block_bytes(&self.terms.ustb, &self.block_decoded, self.block_range);
        let line =
            self.line_reader
                .read_line(bytes, &mut self.block_pos, is_incremental_encoding_seed)?;
        self.line = Some(line);
        Ok(self.line)
    }

    /// `seekExact(term, state)`: positions on a term from its state without
    /// reading a block.
    pub fn seek_exact_state(&mut self, term: &[u8], state: &UniformSplitTermState) {
        self.term_state_forced = true;
        self.term_state = Some(*state);
        self.forced_term.clear();
        self.forced_term.extend_from_slice(term);
    }

    /// `seekExact(ord)`.
    ///
    /// # Errors
    /// Always [`Error::Unsupported`].
    pub fn seek_exact_ord(&mut self, _ord: i64) -> Result<()> {
        Err(Error::Unsupported("UniformSplit terms have no ords"))
    }

    /// `ord()`.
    ///
    /// # Errors
    /// Always [`Error::Unsupported`].
    pub fn ord(&self) -> Result<i64> {
        Err(Error::Unsupported("UniformSplit terms have no ords"))
    }

    /// `next()`: after a forced state, from its line.
    ///
    /// # Errors
    /// `Corrupted` for a malformed block.
    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> Result<Option<&[u8]>> {
        if self.term_state_forced {
            let (block_file_pointer, term_block_ord) = self
                .term_state
                .map_or((-1, 0), |s| (s.block_file_pointer, s.term_block_ord));
            let forced_term = std::mem::take(&mut self.forced_term);
            let initialized = self.initialize_header(Some(&forced_term), block_file_pointer);
            self.forced_term = forced_term;
            initialized?;
            if self.block_header.is_none() {
                return Err(
                    self.corrupt_at("Illegal absence of block for TermState", block_file_pointer)
                );
            }
            // At most `lines_count` lines: `read_line_in_block` stops there.
            for _ in self.line_index_in_block..term_block_ord {
                if self.read_line_in_block()?.is_none() {
                    break;
                }
            }
        }
        self.clear_term_state();
        self.next_term()
    }

    /// `nextTerm()`.
    pub(super) fn next_term(&mut self) -> Result<Option<&[u8]>> {
        if self.block_header.is_none() {
            let first = self.terms.metadata.first_block_start_fp;
            self.initialize_header(None, first)?;
            if self.block_header.is_none() {
                return Err(self.corrupt_at("Illegal absence of first block", first));
            }
        }
        if self.read_line_in_block()?.is_none() {
            self.initialize_header(None, self.block_input_fp as i64)?;
            if self.block_header.is_none() {
                return Ok(None);
            }
            self.read_line_in_block()?;
        }
        Ok(self.term())
    }

    /// `initializeHeader(searchedTerm, targetBlockStartFP)`: the block at
    /// `target_block_start_fp`, from its first line unless `searched_term`
    /// is after the current line of that same block.
    pub(super) fn initialize_header(
        &mut self,
        searched_term: Option<&[u8]>,
        target_block_start_fp: i64,
    ) -> Result<()> {
        if self.block_start_fp == target_block_start_fp {
            if self.block_header.is_none() {
                return Err(self.corrupt_at("Illegal absence of block", self.block_start_fp));
            }
            let after_line = match (searched_term, self.line) {
                (Some(term), Some(_)) => term > self.line_reader.current_term.as_slice(),
                _ => false,
            };
            if !after_line {
                self.block_pos = self.block_first_line_start;
                self.line_index_in_block = 0;
            }
        } else {
            // `blockInput.seek(targetBlockStartFP)`.
            self.block_input_fp = usize::try_from(target_block_start_fp)
                .ok()
                .filter(|&fp| fp <= self.terms.ustb.len())
                .ok_or_else(|| self.corrupt_at("Illegal block start", target_block_start_fp))?;
            self.block_start_fp = target_block_start_fp;
            self.read_header()?;
            self.block_first_line_start = self.block_pos;
            self.line_index_in_block = 0;
        }
        Ok(())
    }

    /// `readHeader()`: `None` past the field's last block.
    fn read_header(&mut self) -> Result<()> {
        if self.block_input_fp as i64 > self.terms.metadata.last_block_start_fp {
            self.block_header = None;
            return Ok(());
        }
        // Cleared before reading: a block that fails to read leaves no
        // header or line behind. (Past the last block, above, the line is
        // kept, as Java keeps `blockLine`.)
        self.block_header = None;
        self.line = None;
        let ustb: &[u8] = &self.terms.ustb;
        let mut pos = self.block_input_fp;
        let num_block_bytes = read_vint(ustb, &mut pos)?;
        let num_block_bytes = usize::try_from(num_block_bytes)
            .map_err(|_| corrupt(format!("negative block length {num_block_bytes}")))?;
        let start = pos;
        // `decodeBlockBytesIfNeeded(numBlockBytes)`.
        let block = super::read_slice(ustb, &mut pos, num_block_bytes)?;
        self.block_input_fp = pos;
        self.block_decoded = match self.terms.decoder.as_deref() {
            None => None,
            Some(decoder) => Some(decoder.decode(block)?),
        };
        self.block_range = (start, pos);
        self.block_pos = 0;
        let bytes = block_bytes(&self.terms.ustb, &self.block_decoded, self.block_range);
        self.block_header = Some(BlockHeader::read(bytes, &mut self.block_pos)?);
        Ok(())
    }

    /// `readTermStateIfNotRead()`.
    ///
    /// # Errors
    /// [`Error::Unsupported`] when not positioned on a term (Java's
    /// `NullPointerException`); `Corrupted` for a malformed term state.
    pub(super) fn read_term_state_if_not_read(&mut self) -> Result<UniformSplitTermState> {
        if let Some(state) = self.term_state {
            return Ok(state);
        }
        let mut state = self.read_term_state()?;
        state.term_block_ord = self.line_index_in_block;
        state.block_file_pointer = self.block_start_fp;
        self.term_state = Some(state);
        Ok(state)
    }

    /// `readTermState()`.
    fn read_term_state(&self) -> Result<UniformSplitTermState> {
        let (Some(header), Some((rel_offset, _))) = (self.block_header.as_ref(), self.line) else {
            return Err(Error::Unsupported(
                "the terms enum is not positioned on a term",
            ));
        };
        let mut pos = usize::try_from(header.term_states_base_offset)
            .ok()
            .zip(usize::try_from(rel_offset).ok())
            .and_then(|(base, rel)| {
                self.block_first_line_start
                    .checked_add(base)?
                    .checked_add(rel)
            })
            .ok_or_else(|| corrupt("term state offset overflows"))?;
        let bytes = block_bytes(&self.terms.ustb, &self.block_decoded, self.block_range);
        let info = &self.terms.metadata.field_info;
        DeltaBaseTermStateSerializer::read_term_state(
            header.base_docs_fp,
            header.base_positions_fp,
            header.base_payloads_fp,
            bytes,
            &mut pos,
            info.index_options,
            info.store_payloads,
        )
    }

    /// `term()`.
    pub fn term(&self) -> Option<&[u8]> {
        if self.term_state_forced {
            return Some(&self.forced_term);
        }
        self.line.map(|_| self.line_reader.current_term.as_slice())
    }

    /// `docFreq()`.
    ///
    /// # Errors
    /// See [`Self::term_state`].
    pub fn doc_freq(&mut self) -> Result<i32> {
        Ok(self.read_term_state_if_not_read()?.doc_freq)
    }

    /// `totalTermFreq()`.
    ///
    /// # Errors
    /// See [`Self::term_state`].
    pub fn total_term_freq(&mut self) -> Result<i64> {
        Ok(self.read_term_state_if_not_read()?.total_term_freq)
    }

    /// `termState()`.
    ///
    /// # Errors
    /// [`Error::Unsupported`] when not positioned on a term; `Corrupted` for
    /// a malformed term state.
    pub fn term_state(&mut self) -> Result<UniformSplitTermState> {
        self.read_term_state_if_not_read()
    }

    /// `postings(reuse, flags)`.
    ///
    /// # Errors
    /// See [`Self::term_state`] and [`UniformSplitTerms::postings`].
    pub fn postings(&mut self, doc_in: &DocInput<'_>, flags: PostingsFlags) -> Result<Postings> {
        let state = self.read_term_state_if_not_read()?;
        self.terms.postings(&state, doc_in, flags)
    }

    /// `getOrCreateDictionaryBrowser()`.
    pub(super) fn get_or_create_dictionary_browser(&mut self) -> &mut Browser<'a> {
        let terms = self.terms;
        self.browser
            .get_or_insert_with(|| terms.dictionary.browser())
    }

    /// `clearTermState()`.
    pub(super) fn clear_term_state(&mut self) {
        self.term_state = None;
        self.term_state_forced = false;
    }

    /// The current block's header, which the callers have read.
    fn header(&self) -> Result<BlockHeader> {
        self.block_header
            .ok_or_else(|| self.corrupt_at_none("Illegal absence of block"))
    }

    /// `newCorruptIndexException(msg, fp)`.
    pub(super) fn corrupt_at(&self, msg: &str, fp: i64) -> Error {
        corrupt(format!(
            "{msg} at FP {fp} for field \"{}\"",
            self.terms.metadata.field_info.name
        ))
    }

    /// `newCorruptIndexException(msg, null)`.
    fn corrupt_at_none(&self, msg: &str) -> Error {
        corrupt(format!(
            "{msg} for field \"{}\"",
            self.terms.metadata.field_info.name
        ))
    }
}
