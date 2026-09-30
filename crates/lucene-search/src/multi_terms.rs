//! Port of the composite term views over a multi-leaf reader:
//! `FieldInfos.getIndexedFields` (the indexed field names), `MultiTerms`
//! (one field's terms across leaves, with summed statistics),
//! `MultiTermsEnum` (their union, in term order) and `MultiPostingsEnum` (a
//! term's postings across leaves, in top-level doc ids, with positions,
//! offsets and payloads when asked for).
//!
//! Built over the [`crate::reader`] layer, so the leaves may be segments or
//! any view over them.
//!
//! # What differs from Java
//!
//! - `MultiTermsEnum` keeps its subs in a priority queue; here the smallest
//!   current term is found by a scan of the subs -- the same answers, and a
//!   reader has few leaves.
//! - [`MultiPostingsEnum`] is filled from each leaf's postings when it is
//!   created rather than stepping through them lazily; it answers the same.
//! - Like Java's, the postings and statistics include deleted documents:
//!   `PostingsEnum` never filters live docs (see [`crate::multi_bits`]).
//! - `ord()`/`seekExact(long)` are unsupported, and `impacts()` is a
//!   [`SlowImpactsEnum`], both as in Java.

use lucene_codecs::automaton::ByteDfa;
use lucene_codecs::blocktree::SeekStatus;

use crate::reader::{
    DocIdSetIterator, ImpactsEnum, IndexReader, MaterializedPostings, Position, PostingsEnum,
    PostingsFlags, SlowImpactsEnum, Terms, TermsEnum,
};
use crate::Result;

pub use crate::reader::NO_MORE_DOCS;

/// `FieldInfos.getIndexedFields(reader)`: the names of the fields any leaf
/// indexes, sorted, each once.
pub fn indexed_fields<R: IndexReader + ?Sized>(reader: &R) -> Vec<String> {
    let mut names: Vec<String> = reader
        .leaves()
        .iter()
        .flat_map(|l| {
            l.reader
                .field_infos()
                .fields
                .iter()
                .filter(|f| f.index_options != lucene_codecs::field_infos::IndexOptions::None)
                .map(|f| f.name.clone())
                .collect::<Vec<_>>()
        })
        .collect();
    names.sort();
    names.dedup();
    names
}

/// `MultiTerms`: one field's terms across the leaves that have it.
pub struct MultiTerms<'a> {
    /// Each leaf's terms, with that leaf's doc base (`subSlices`).
    subs: Vec<(Box<dyn Terms + 'a>, i32)>,
    has_freqs: bool,
    has_offsets: bool,
    has_positions: bool,
    has_payloads: bool,
}

impl<'a> MultiTerms<'a> {
    /// `new MultiTerms(subs, subSlices)`.
    pub fn new(subs: Vec<(Box<dyn Terms + 'a>, i32)>) -> Self {
        let has_freqs = subs.iter().all(|(t, _)| t.has_freqs());
        let has_offsets = subs.iter().all(|(t, _)| t.has_offsets());
        let has_positions = subs.iter().all(|(t, _)| t.has_positions());
        // "if all subs have pos, and at least one has payloads".
        let has_payloads = has_positions && subs.iter().any(|(t, _)| t.has_payloads());
        Self {
            subs,
            has_freqs,
            has_offsets,
            has_positions,
            has_payloads,
        }
    }

    /// `MultiTerms.getTerms(reader, field)`: `None` when no leaf has terms
    /// for `field`; a single leaf's own terms when the reader has one leaf.
    ///
    /// # Errors
    /// A leaf's terms fail to open.
    pub fn get_terms<R: IndexReader + ?Sized>(
        reader: &'a R,
        field: &str,
    ) -> Result<Option<Box<dyn Terms + 'a>>> {
        let leaves = reader.leaves();
        if leaves.len() == 1 {
            return leaves[0].reader.terms(field);
        }
        let mut subs = Vec::new();
        for leaf in leaves {
            if let Some(t) = leaf.reader.terms(field)? {
                subs.push((t, leaf.doc_base));
            }
        }
        Ok(if subs.is_empty() {
            None
        } else {
            Some(Box::new(MultiTerms::new(subs)))
        })
    }

    fn enum_over(subs: Vec<(Box<dyn TermsEnum + '_>, i32)>) -> MultiTermsEnum<'_> {
        MultiTermsEnum {
            subs: subs
                .into_iter()
                .map(|(te, doc_base)| Sub {
                    te,
                    term: None,
                    doc_base,
                })
                .collect(),
            current: None,
            top: Vec::new(),
            started: false,
        }
    }
}

impl Terms for MultiTerms<'_> {
    fn iterator(&self) -> Result<Box<dyn TermsEnum + '_>> {
        let subs = self
            .subs
            .iter()
            .map(|(t, base)| Ok((t.iterator()?, *base)))
            .collect::<Result<Vec<_>>>()?;
        Ok(Box::new(Self::enum_over(subs)))
    }

    /// `intersect(compiled, startTerm)`: each leaf intersects its own terms,
    /// and the results are merged as `iterator()`'s are.
    fn intersect<'s>(
        &'s self,
        dfa: &'s ByteDfa,
        start_term: Option<&[u8]>,
    ) -> Result<Box<dyn TermsEnum + 's>> {
        let subs = self
            .subs
            .iter()
            .map(|(t, base)| Ok((t.intersect(dfa, start_term)?, *base)))
            .collect::<Result<Vec<_>>>()?;
        Ok(Box::new(Self::enum_over(subs)))
    }

    /// `size()`: unknown (`-1`), as in Java -- terms shared by leaves are
    /// counted once, which only a full enumeration can tell.
    // SENTINEL: `-1` = unknown, Java's `Terms.size()` contract; no caller
    // indexes with it.
    fn size(&self) -> i64 {
        -1
    }

    fn sum_total_term_freq(&self) -> i64 {
        self.subs.iter().fold(0i64, |acc, (t, _)| {
            acc.saturating_add(t.sum_total_term_freq())
        })
    }

    fn sum_doc_freq(&self) -> i64 {
        self.subs
            .iter()
            .fold(0i64, |acc, (t, _)| acc.saturating_add(t.sum_doc_freq()))
    }

    fn doc_count(&self) -> i32 {
        self.subs
            .iter()
            .fold(0i32, |acc, (t, _)| acc.saturating_add(t.doc_count()))
    }

    fn has_freqs(&self) -> bool {
        self.has_freqs
    }
    fn has_offsets(&self) -> bool {
        self.has_offsets
    }
    fn has_positions(&self) -> bool {
        self.has_positions
    }
    fn has_payloads(&self) -> bool {
        self.has_payloads
    }

    /// `getMin()`: the smallest term of any leaf.
    fn min(&self) -> Result<Option<Vec<u8>>> {
        let mut out: Option<Vec<u8>> = None;
        for (t, _) in &self.subs {
            if let Some(m) = t.min()? {
                if out.as_ref().is_none_or(|o| m < *o) {
                    out = Some(m);
                }
            }
        }
        Ok(out)
    }

    /// `getMax()`: the largest term of any leaf.
    fn max(&self) -> Result<Option<Vec<u8>>> {
        let mut out: Option<Vec<u8>> = None;
        for (t, _) in &self.subs {
            if let Some(m) = t.max()? {
                if out.as_ref().is_none_or(|o| m > *o) {
                    out = Some(m);
                }
            }
        }
        Ok(out)
    }
}

struct Sub<'a> {
    te: Box<dyn TermsEnum + 'a>,
    /// The sub's current term; `None` before it starts and once exhausted.
    term: Option<Vec<u8>>,
    doc_base: i32,
}

impl Sub<'_> {
    fn advance(&mut self) -> Result<()> {
        self.term = self.te.next()?.map(<[u8]>::to_vec);
        Ok(())
    }

    fn try_seek_ceil(&mut self, target: &[u8]) -> Result<()> {
        self.term = match self.te.try_seek_ceil(target)? {
            SeekStatus::End => None,
            SeekStatus::Found | SeekStatus::NotFound => self.te.term().map(<[u8]>::to_vec),
        };
        Ok(())
    }
}

/// `MultiTermsEnum`.
pub struct MultiTermsEnum<'a> {
    subs: Vec<Sub<'a>>,
    /// `current`: the term the enum stands on.
    current: Option<Vec<u8>>,
    /// `top`: the subs standing on `current`, ascending (leaf order).
    top: Vec<usize>,
    started: bool,
}

impl MultiTermsEnum<'_> {
    /// `pullTop()`: the smallest term among the subs, and every sub on it.
    fn pull_top(&mut self) {
        let min = self
            .subs
            .iter()
            .filter_map(|s| s.term.as_deref())
            .min()
            .map(<[u8]>::to_vec);
        self.top = match &min {
            None => Vec::new(),
            Some(m) => (0..self.subs.len())
                .filter(|&i| self.subs[i].term.as_deref() == Some(m.as_slice()))
                .collect(),
        };
        self.current = min;
    }
}

impl TermsEnum for MultiTermsEnum<'_> {
    fn next(&mut self) -> Result<Option<&[u8]>> {
        if !self.started {
            self.started = true;
            for sub in &mut self.subs {
                sub.advance()?;
            }
        } else {
            // `pushTop()`: the subs on the current term move on.
            for &i in &self.top {
                self.subs[i].advance()?;
            }
        }
        self.pull_top();
        Ok(self.current.as_deref())
    }

    /// `seekCeil(target)`: positions every sub at its smallest term `>=
    /// target`, and the enum on the smallest of those.
    fn try_seek_ceil(&mut self, target: &[u8]) -> Result<SeekStatus> {
        self.started = true;
        for sub in &mut self.subs {
            sub.try_seek_ceil(target)?;
        }
        self.pull_top();
        Ok(match &self.current {
            None => SeekStatus::End,
            Some(t) if t.as_slice() == target => SeekStatus::Found,
            Some(_) => SeekStatus::NotFound,
        })
    }

    fn term(&self) -> Option<&[u8]> {
        self.current.as_deref()
    }

    /// `docFreq()`: summed over the leaves holding the current term.
    fn doc_freq(&mut self) -> Result<i32> {
        let mut sum = 0i32;
        for &i in &self.top {
            sum = sum.saturating_add(self.subs[i].te.doc_freq()?);
        }
        Ok(sum)
    }

    /// `totalTermFreq()`: summed over the leaves holding the current term.
    fn total_term_freq(&mut self) -> Result<i64> {
        let mut sum = 0i64;
        for &i in &self.top {
            sum = sum.saturating_add(self.subs[i].te.total_term_freq()?);
        }
        Ok(sum)
    }

    /// `postings(null, flags)`: the current term's documents across the
    /// leaves holding it, in top-level doc ids.
    fn postings(&mut self, flags: PostingsFlags) -> Result<Box<dyn PostingsEnum>> {
        let mut docs = Vec::new();
        let mut freqs = Vec::new();
        let mut positions: Option<Vec<Vec<Position>>> = flags.wants_positions().then(Vec::new);
        for &i in &self.top {
            let sub = &mut self.subs[i];
            let pe = sub.te.postings(flags)?;
            drain_into(pe, sub.doc_base, &mut docs, &mut freqs, positions.as_mut())?;
        }
        Ok(Box::new(MultiPostingsEnum(MaterializedPostings::new(
            docs, freqs, positions,
        )?)))
    }

    /// `impacts(flags)`: a `SlowImpactsEnum`, "implemented to not fail
    /// CheckIndex, but you shouldn't be using impacts on a slow reader".
    fn impacts(&mut self, flags: PostingsFlags) -> Result<Box<dyn ImpactsEnum>> {
        Ok(Box::new(SlowImpactsEnum::new(self.postings(flags)?)))
    }
}

/// Reads every document (and, when `positions` is given, every position with
/// its offsets and payload) out of `pe`, shifting doc ids by `doc_base`.
pub(crate) fn drain_into(
    mut pe: Box<dyn PostingsEnum>,
    doc_base: i32,
    docs: &mut Vec<i32>,
    freqs: &mut Vec<i32>,
    mut positions: Option<&mut Vec<Vec<Position>>>,
) -> Result<()> {
    loop {
        let doc = pe.next_doc()?;
        if doc == NO_MORE_DOCS {
            return Ok(());
        }
        let freq = pe.freq();
        docs.push(doc.saturating_add(doc_base));
        freqs.push(freq);
        if let Some(all) = positions.as_deref_mut() {
            let mut list = Vec::with_capacity(usize::try_from(freq).unwrap_or(0));
            for _ in 0..freq {
                let position = pe.next_position()?;
                list.push(Position {
                    position,
                    start_offset: pe.start_offset(),
                    end_offset: pe.end_offset(),
                    payload: pe.payload().map(<[u8]>::to_vec).unwrap_or_default(),
                });
            }
            all.push(list);
        }
    }
}

/// `MultiPostingsEnum`: a term's postings across leaves, in top-level doc
/// ids.
#[derive(Debug, Clone)]
pub struct MultiPostingsEnum(MaterializedPostings);

impl DocIdSetIterator for MultiPostingsEnum {
    fn doc_id(&self) -> i32 {
        self.0.doc_id()
    }
    fn next_doc(&mut self) -> Result<i32> {
        self.0.next_doc()
    }
    fn advance(&mut self, target: i32) -> Result<i32> {
        self.0.advance(target)
    }
    fn cost(&self) -> i64 {
        self.0.cost()
    }
}

impl PostingsEnum for MultiPostingsEnum {
    fn freq(&self) -> i32 {
        self.0.freq()
    }
    fn next_position(&mut self) -> Result<i32> {
        self.0.next_position()
    }
    fn start_offset(&self) -> i32 {
        self.0.start_offset()
    }
    fn end_offset(&self) -> i32 {
        self.0.end_offset()
    }
    fn payload(&self) -> Option<&[u8]> {
        self.0.payload()
    }
}
