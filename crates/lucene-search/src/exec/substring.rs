//! [`SubstringQuery`] per segment: the candidate blocks of the field's
//! trigram index, each decompressed once and its values tested
//! ([`lucene_codecs::substring`]); without that index, every value of the
//! field's BINARY doc values.

use std::sync::Arc;

use lucene_codecs::substring::{Block, Candidates, SubstringIndex, SubstringMatcher};

use super::build::LeafContext;
use super::leaf::{ConstantScorer, DocList};
use super::{BoxScorer, Mode, Scorer, NO_MORE_DOCS};
use crate::extended_query::SubstringQuery;
use crate::reader::LeafReader;
use crate::Result;

/// `Weight.scorer` for a [`SubstringQuery`].
pub(crate) fn substring<'a>(
    ctx: &LeafContext<'a>,
    q: &SubstringQuery,
    boost: f32,
    mode: Mode,
) -> Result<Option<BoxScorer<'a>>> {
    let reader = ctx
        .reader
        .ok_or_else(|| crate::Error::MissingSegmentReader(q.field.clone()))?;
    let inner: BoxScorer<'a> = match reader.substring_index(&q.field)? {
        Some(index) => {
            let blocks = match index.candidate_blocks(q.matcher().grams())? {
                Candidates::All => None,
                Candidates::Blocks(b) if b.is_empty() => return Ok(None),
                Candidates::Blocks(b) => Some(b),
            };
            Box::new(BlockScorer::new(index, Arc::clone(q.matcher()), blocks))
        }
        None => {
            let Some(mut values) = reader.binary_doc_values(&q.field)? else {
                return Ok(None);
            };
            let mut docs = Vec::new();
            let mut scratch = Vec::new();
            while values.next_doc()? != NO_MORE_DOCS {
                if q.matcher().matches(values.binary_value(), &mut scratch) {
                    docs.push(values.doc_id());
                }
            }
            if docs.is_empty() {
                return Ok(None);
            }
            Box::new(DocList::new(docs, Vec::new()))
        }
    };
    Ok(Some(Box::new(ConstantScorer::new(
        inner,
        boost,
        mode == Mode::TopScores,
    ))))
}

/// The matching documents of the candidate blocks, a block at a time.
struct BlockScorer<'a> {
    index: SubstringIndex<'a>,
    matcher: Arc<SubstringMatcher>,
    /// `None`: every block.
    blocks: Option<Vec<u32>>,
    /// The next candidate to read.
    next: usize,
    block: Block,
    hits: Vec<i32>,
    hit: usize,
    scratch: Vec<u8>,
    doc: i32,
}

impl<'a> BlockScorer<'a> {
    fn new(
        index: SubstringIndex<'a>,
        matcher: Arc<SubstringMatcher>,
        blocks: Option<Vec<u32>>,
    ) -> Self {
        BlockScorer {
            index,
            matcher,
            blocks,
            next: 0,
            block: Block::new(),
            hits: Vec::new(),
            hit: 0,
            scratch: Vec::new(),
            doc: -1,
        }
    }

    fn candidates(&self) -> usize {
        match &self.blocks {
            Some(b) => b.len(),
            None => usize::try_from(self.index.num_blocks()).unwrap_or(0),
        }
    }

    fn candidate(&self, i: usize) -> i64 {
        match &self.blocks {
            Some(b) => i64::from(b[i]),
            None => i as i64,
        }
    }

    /// Moves to the first candidate block that may hold `target` or later.
    fn skip_to(&mut self, target: i32) {
        let block = i64::from(target) >> self.index.block_shift();
        let n = self.candidates();
        match &self.blocks {
            Some(b) => {
                let from = self.next.min(b.len());
                self.next =
                    from.saturating_add(b[from..].partition_point(|&x| i64::from(x) < block));
            }
            None => self.next = self.next.max(usize::try_from(block).unwrap_or(n)).min(n),
        }
    }

    /// The next matching document at or after `target`.
    fn next_at_or_after(&mut self, target: i32) -> Result<i32> {
        loop {
            while self.hit < self.hits.len() {
                let d = self.hits[self.hit];
                self.hit = self.hit.saturating_add(1);
                if d >= target {
                    self.doc = d;
                    return Ok(d);
                }
            }
            if self.next >= self.candidates() {
                self.doc = NO_MORE_DOCS;
                return Ok(NO_MORE_DOCS);
            }
            let b = self.candidate(self.next);
            self.next = self.next.saturating_add(1);
            self.index.read_block(b, &mut self.block)?;
            self.hits.clear();
            self.hit = 0;
            self.matcher
                .collect(&self.block, &mut self.hits, &mut self.scratch);
        }
    }
}

impl Scorer for BlockScorer<'_> {
    fn doc_id(&self) -> i32 {
        self.doc
    }

    fn next_doc(&mut self) -> Result<i32> {
        let target = self.doc.saturating_add(1);
        self.next_at_or_after(target)
    }

    fn advance(&mut self, target: i32) -> Result<i32> {
        if target == NO_MORE_DOCS {
            self.doc = NO_MORE_DOCS;
            return Ok(NO_MORE_DOCS);
        }
        // Past the hits of the current block: jump over the blocks between.
        if self.hits.last().is_none_or(|&last| last < target) {
            self.hits.clear();
            self.hit = 0;
            self.skip_to(target);
        }
        self.next_at_or_after(target)
    }

    fn cost(&self) -> i64 {
        (self.candidates() as i64).saturating_mul(1i64 << self.index.block_shift())
    }

    fn score(&mut self) -> Result<f32> {
        Ok(0.0)
    }

    fn max_score(&mut self, _up_to: i32) -> Result<f32> {
        Ok(0.0)
    }
}
