//! `PKIndexSplitter`: splits an index into two by a query, usually a range
//! over a primary-key term -- the documents the query matches go to the first
//! output, the rest to the second. Both outputs are written with
//! `addIndexes(CodecReader...)` over the input's segments, a reader per
//! segment hiding the other half's documents, so each output is one merged
//! segment.

use lucene_store::directory::Directory;

use super::{index_error, live_flags, match_flags, open_writer};
use crate::directory_reader::DirectoryReader;
use crate::extended_query::{
    ExtendedQuery, MultiTermQuery, MultiTermSource, RewriteMethod, TermRangeQuery,
};
use crate::query::{BooleanQuery, Clause};
use crate::Result;

/// `PKIndexSplitter`.
pub struct PkIndexSplitter<'d> {
    input: &'d dyn Directory,
    dir1: &'d dyn Directory,
    dir2: &'d dyn Directory,
    docs_in_first_index: BooleanQuery,
}

impl<'d> PkIndexSplitter<'d> {
    /// `PKIndexSplitter(input, dir1, dir2, docsInFirstIndex)`.
    pub fn new(
        input: &'d dyn Directory,
        dir1: &'d dyn Directory,
        dir2: &'d dyn Directory,
        docs_in_first_index: BooleanQuery,
    ) -> Self {
        Self {
            input,
            dir1,
            dir2,
            docs_in_first_index,
        }
    }

    /// `PKIndexSplitter(input, dir1, dir2, midTerm)`: the first output gets
    /// the documents whose `field` term is below `mid` (`new
    /// TermRangeQuery(field, null, mid, true, false)`), the second the rest.
    pub fn with_mid_term(
        input: &'d dyn Directory,
        dir1: &'d dyn Directory,
        dir2: &'d dyn Directory,
        field: &str,
        mid: &[u8],
    ) -> Self {
        let range = TermRangeQuery::new(field, None, Some(mid.to_vec()), true, false);
        let multi = MultiTermQuery::new(
            MultiTermSource::TermRange(range),
            RewriteMethod::ConstantScoreBlended,
        );
        let query = BooleanQuery {
            must: vec![Clause::Extended(Box::new(ExtendedQuery::MultiTerm(multi)))],
            ..Default::default()
        };
        Self::new(input, dir1, dir2, query)
    }

    /// `split()`: writes the first output (the matches), then the second (the
    /// documents that are not).
    pub fn split(&self) -> Result<()> {
        let reader = DirectoryReader::open(self.input)?;
        let matches = match_flags(&reader, &self.docs_in_first_index)?;
        let live = live_flags(&reader);
        self.create_index(self.dir1, &live, &matches, false)?;
        self.create_index(self.dir2, &live, &matches, true)
    }

    /// `createIndex`: the kept documents are the live ones the query matches
    /// (or, `negate`, does not).
    fn create_index(
        &self,
        target: &dyn Directory,
        live: &[Vec<bool>],
        matches: &[Vec<bool>],
        negate: bool,
    ) -> Result<()> {
        let mut w = open_writer(target)?;
        w.add_indexes_filtered(&[self.input], &|_, segment, doc| {
            let d = usize::try_from(doc).unwrap_or(0);
            live[segment][d] && (matches[segment][d] != negate)
        })
        .map_err(index_error)?;
        w.close().map_err(index_error)
    }
}
