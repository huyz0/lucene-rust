//! The score of chosen documents under a query, without visiting the rest.

use crate::index_searcher::IndexSearcher;
use crate::{BooleanQuery, Error, Result};

impl IndexSearcher<'_, '_> {
    /// The score `query` gives each of `docs` (global ids, ascending), and
    /// `None` for one it does not match: one scorer a segment, advanced
    /// through that segment's documents in order as `QueryRescorer`'s second
    /// pass does, so the work follows `docs` and not the query's matches. A
    /// deleted document scores as any other, a `Scorer` never reading live
    /// docs.
    ///
    /// # Errors
    /// [`Error::IllegalArgument`] for a document outside every segment or
    /// ids that are not ascending, and whatever scoring reports.
    pub fn scores_at(&self, query: &BooleanQuery, docs: &[i32]) -> Result<Vec<Option<f32>>> {
        let mut out = Vec::with_capacity(docs.len());
        if docs.is_empty() {
            return Ok(out);
        }
        let (rewritten, global) = self.prepare(query)?;
        let query = rewritten.as_ref().unwrap_or(query);
        let mut start = 0;
        let mut previous = i32::MIN;
        while start < docs.len() {
            let leaf = self.segment_of(docs[start]).ok_or_else(|| {
                Error::IllegalArgument(format!("doc {} is in no segment", docs[start]))
            })?;
            let base = self.segments()[leaf].doc_base;
            let mut locals = Vec::new();
            let mut end = start;
            while end < docs.len() && self.segment_of(docs[end]) == Some(leaf) {
                if docs[end] < previous {
                    return Err(Error::IllegalArgument("docs are not ascending".to_owned()));
                }
                previous = docs[end];
                locals.push(docs[end] - base);
                end += 1;
            }
            if end == start {
                return Err(Error::IllegalArgument(format!(
                    "doc {} is in no segment",
                    docs[start]
                )));
            }
            out.extend(self.leaf_scores_at(query, &global, leaf, &locals)?);
            start = end;
        }
        Ok(out)
    }
}
