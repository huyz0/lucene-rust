//! `PerThreadPKLookup`: finds a live document by its primary key, looking
//! in the largest segments first.
//!
//! Differs: Java keeps one `TermsEnum` and `PostingsEnum` per segment and
//! hands them on through `reopen` for the segments the new reader shares.
//! An enum here borrows the `Terms` it came from, which no struct can hold
//! beside it, so `lookup` opens each segment's terms afresh; the
//! segments, their order and the answers are Java's.

use lucene_search::reader::{IndexReader, LeafReaderContext, PostingsFlags, NO_MORE_DOCS};
use lucene_search::Result;

/// `PerThreadPKLookup` over a reader's leaves.
pub struct PerThreadPkLookup<'r> {
    id_field: String,
    /// The leaves with the id field, largest (`numDocs`) first.
    leaves: Vec<LeafReaderContext<'r>>,
    has_deletions: bool,
}

impl<'r> PerThreadPkLookup<'r> {
    /// `new PerThreadPKLookup(reader, idFieldName)`.
    ///
    /// # Errors
    /// What reading a leaf's terms fails with.
    pub fn new(reader: &'r dyn IndexReader, id_field: &str) -> Result<Self> {
        let mut leaves = reader.leaves();
        // Larger segments are more likely to have the id (a stable sort, as
        // Java's `List.sort`).
        leaves.sort_by_key(|c| std::cmp::Reverse(c.reader.num_docs()));
        let mut kept = Vec::with_capacity(leaves.len());
        let mut has_deletions = false;
        for c in leaves {
            if c.reader.terms(id_field)?.is_some() {
                has_deletions |= c.reader.live_docs().is_some();
                kept.push(c);
            }
        }
        Ok(Self {
            id_field: id_field.to_string(),
            leaves: kept,
            has_deletions,
        })
    }

    /// `lookup(id)`: the live document holding `id`, `None` when there is
    /// none.
    ///
    /// # Errors
    /// What reading the terms or postings fails with.
    pub fn lookup(&self, id: &[u8]) -> Result<Option<i32>> {
        for c in &self.leaves {
            let Some(terms) = c.reader.terms(&self.id_field)? else {
                continue;
            };
            let mut te = terms.iterator()?;
            if !te.try_seek_exact(id)? {
                continue;
            }
            let mut postings = te.postings(PostingsFlags::None)?;
            let live = c.reader.live_docs();
            loop {
                let doc = postings.next_doc()?;
                if doc == NO_MORE_DOCS {
                    break;
                }
                if live.is_none_or(|bits| bits.get(doc as usize)) {
                    return Ok(Some(c.doc_base + doc));
                }
            }
            assert!(
                self.has_deletions,
                "every document holding the id is deleted in a reader without deletions"
            );
        }
        Ok(None)
    }

    /// `reopen(reader)`: a lookup over `reader`, `None` for no reader.
    ///
    /// # Errors
    /// What reading a leaf's terms fails with.
    pub fn reopen<'n>(
        &self,
        reader: Option<&'n dyn IndexReader>,
    ) -> Result<Option<PerThreadPkLookup<'n>>> {
        reader
            .map(|r| PerThreadPkLookup::new(r, &self.id_field))
            .transpose()
    }

    /// The leaves searched, in order: their doc bases.
    pub fn doc_bases(&self) -> Vec<i32> {
        self.leaves.iter().map(|c| c.doc_base).collect()
    }
}
