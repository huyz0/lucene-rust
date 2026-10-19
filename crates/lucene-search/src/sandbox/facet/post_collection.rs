//! `org.apache.lucene.sandbox.facet.utils.PostCollectionFaceting`: runs
//! collector managers over documents a [`FacetsCollector`] (and per dim, a
//! drill-sideways one) already collected, rather than over a query.
//!
//! The collectors move to a segment through the searcher that collected the
//! documents ([`IndexSearcher::segments`], by the matching documents' leaf
//! ordinal) -- the Rust [`SegmentCollector`] is handed an [`OpenSegment`](crate::multi_segment::OpenSegment),
//! not a `LeafReaderContext`. The slices run on the calling thread (Java's
//! `executor == null`).

use std::collections::HashMap;

use crate::facets::{FacetsCollector, MatchingDocs};
use crate::grouping::GroupingCollectorManager;
use crate::index_searcher::IndexSearcher;
use crate::leaf_collector::SegmentCollector;
use crate::reader::DocIdSetIterator;
use crate::{Error, Result};

use lucene_codecs::postings::NO_MORE_DOCS;

/// `PostCollectionFaceting.MIN_DOCS_PER_SLICE`.
pub const MIN_DOCS_PER_SLICE: i32 = 100;

/// `PostCollectionFaceting.Result`.
#[derive(Debug)]
pub struct PostCollectionResult<T, R> {
    /// `drillDownResult`: `None` without a drill-down manager.
    pub drill_down_result: Option<T>,
    /// `drillSidewaysResults`: per dim with both a manager and documents.
    pub drill_sideways_results: HashMap<String, R>,
}

/// `PostCollectionFaceting`.
pub struct PostCollectionFaceting<'m, 'c, 's, 'a, M, S> {
    searcher: &'s IndexSearcher<'s, 'a>,
    drill_down: Option<&'m M>,
    drill_down_docs: &'c FacetsCollector<'a>,
    /// The dims with both a manager and collected documents, in the
    /// managers' order.
    sideways: Vec<(String, &'m S, &'c FacetsCollector<'a>)>,
}

impl<'m, 'c, 's, 'a, M, S> PostCollectionFaceting<'m, 'c, 's, 'a, M, S>
where
    M: GroupingCollectorManager<'a>,
    S: GroupingCollectorManager<'a>,
{
    /// `new PostCollectionFaceting(drillDownCollectorManager,
    /// drillSidewaysCollectorManagers, drillDownFacetsCollector,
    /// drillSidewaysFacetsCollectors, null)`: a dim's manager is kept only
    /// when the dim has collected documents.
    pub fn new(
        searcher: &'s IndexSearcher<'s, 'a>,
        drill_down: Option<&'m M>,
        sideways_managers: &[(&str, &'m S)],
        drill_down_docs: &'c FacetsCollector<'a>,
        sideways_docs: &[(&str, &'c FacetsCollector<'a>)],
    ) -> Self {
        let sideways = sideways_managers
            .iter()
            .filter_map(|&(dim, manager)| {
                sideways_docs
                    .iter()
                    .find(|(d, _)| *d == dim)
                    .map(|&(_, docs)| (dim.to_string(), manager, docs))
            })
            .collect();
        PostCollectionFaceting {
            searcher,
            drill_down,
            drill_down_docs,
            sideways,
        }
    }

    /// `getPerLeafMatchingDocs()`: per leaf ordinal, the drill-down
    /// documents then each dim's.
    fn per_leaf(&self) -> Vec<Vec<Option<&'c MatchingDocs<'a>>>> {
        let all = std::iter::once(self.drill_down_docs)
            .chain(self.sideways.iter().map(|(_, _, docs)| *docs));
        let leaves = all
            .clone()
            .flat_map(|c| c.matching_docs().iter().map(|m| m.context.ord + 1))
            .max()
            .unwrap_or(0);
        let mut out = vec![vec![None; self.sideways.len() + 1]; leaves];
        for (i, c) in all.enumerate() {
            for m in c.matching_docs() {
                out[m.context.ord][i] = Some(m);
            }
        }
        out
    }

    /// `collect()`: one collector per slice and manager, every slice's
    /// leaves merge-collected, then each manager's `reduce`.
    ///
    /// # Errors
    /// [`Error::IllegalState`] when a collector needs scores the drill-down
    /// collector did not keep; whatever a collector or a reduction reports.
    pub fn collect(&self) -> Result<PostCollectionResult<M::Output, S::Output>> {
        let per_leaf = self.per_leaf();
        let slices = slices(MIN_DOCS_PER_SLICE, &per_leaf);
        let mut drill_down = Vec::with_capacity(slices.len());
        let mut sideways: Vec<Vec<S::Collector>> =
            self.sideways.iter().map(|_| Vec::new()).collect();
        for slice in slices {
            let mut main = self.drill_down.map(|m| m.new_collector()).transpose()?;
            let mut side = self
                .sideways
                .iter()
                .map(|(_, m, _)| m.new_collector())
                .collect::<Result<Vec<_>>>()?;
            for leaf in slice {
                let mut collectors: Vec<Option<&mut dyn SegmentCollector<'a>>> =
                    Vec::with_capacity(leaf.len());
                collectors.push(main.as_mut().map(|c| c as &mut dyn SegmentCollector<'a>));
                collectors.extend(
                    side.iter_mut()
                        .map(|c| Some(c as &mut dyn SegmentCollector<'a>)),
                );
                self.collect_leaf(leaf, collectors)?;
            }
            drill_down.extend(main);
            for (slot, c) in sideways.iter_mut().zip(side) {
                slot.push(c);
            }
        }
        let mut drill_sideways_results = HashMap::with_capacity(self.sideways.len());
        for ((dim, manager, _), collectors) in self.sideways.iter().zip(sideways) {
            drill_sideways_results.insert(dim.clone(), manager.reduce(collectors)?);
        }
        let drill_down_result = match self.drill_down {
            Some(m) => Some(m.reduce(drill_down)?),
            None => None,
        };
        Ok(PostCollectionResult {
            drill_down_result,
            drill_sideways_results,
        })
    }

    /// `collectLeaf`: each document of the leaf, in order, to every
    /// collector whose documents hold it; then every leaf collector's
    /// `finish`.
    fn collect_leaf(
        &self,
        docs: &[Option<&MatchingDocs<'a>>],
        mut collectors: Vec<Option<&mut dyn SegmentCollector<'a>>>,
    ) -> Result<()> {
        let mut iterators = Vec::with_capacity(docs.len());
        let mut needs_scores = Vec::with_capacity(docs.len());
        let mut current = NO_MORE_DOCS;
        for (m, c) in docs.iter().zip(collectors.iter_mut()) {
            let (Some(m), Some(c)) = (m, c) else {
                iterators.push(None);
                needs_scores.push(false);
                continue;
            };
            // `getLeafCollector(context)`.
            let segment = self.searcher.segments().get(m.context.ord).ok_or_else(|| {
                Error::IllegalArgument(format!(
                    "matching documents of leaf {} are not of this searcher",
                    m.context.ord
                ))
            })?;
            c.set_next_reader(m.context.ord, segment)?;
            // `createScorer`.
            let needs = c.score_mode().needs_scores();
            if needs && m.scores.is_none() {
                return Err(Error::IllegalState(
                    "Collector requires scores, but FacetCollector doesn't have them.".into(),
                ));
            }
            needs_scores.push(needs);
            let mut it = m.bits.iterator();
            let first = it.next_doc()?;
            if first != NO_MORE_DOCS && (current == NO_MORE_DOCS || current > first) {
                current = first;
            }
            iterators.push(Some(it));
        }
        while current < NO_MORE_DOCS {
            let mut next = i32::MAX;
            for (i, slot) in iterators.iter_mut().enumerate() {
                let Some(it) = slot else {
                    continue;
                };
                if it.doc_id() == current {
                    let score = match docs[i].and_then(|m| m.scores.as_ref()) {
                        Some(scores) if needs_scores[i] => scores
                            .get(usize::try_from(current).unwrap_or(usize::MAX))
                            .copied()
                            .unwrap_or(0.0),
                        _ => 0.0,
                    };
                    if let Some(c) = collectors[i].as_mut() {
                        c.collect(current, score)?;
                    }
                    let n = it.next_doc()?;
                    if n == NO_MORE_DOCS {
                        *slot = None;
                    } else if next > n {
                        next = n;
                    }
                } else if next > it.doc_id() {
                    next = it.doc_id();
                }
            }
            current = next;
        }
        for (m, c) in docs.iter().zip(collectors.iter_mut()) {
            if let (Some(_), Some(c)) = (m, c) {
                c.finish()?;
            }
        }
        Ok(())
    }
}

/// `getSlices(minDocsPerSlice, perLeafMatchingDocs)`: consecutive leaves
/// until their hits reach `min_docs`, the rest (if any hit) last.
fn slices<'p, 'd, 'a>(
    min_docs: i32,
    per_leaf: &'p [Vec<Option<&'d MatchingDocs<'a>>>],
) -> Vec<&'p [Vec<Option<&'d MatchingDocs<'a>>>]> {
    let mut out = Vec::new();
    let mut size = 0i32;
    let mut start = 0usize;
    for (leaf, docs) in per_leaf.iter().enumerate() {
        for m in docs.iter().flatten() {
            size = size.wrapping_add(m.total_hits);
        }
        if size >= min_docs {
            out.push(&per_leaf[start..=leaf]);
            size = 0;
            start = leaf + 1;
        }
    }
    if size > 0 {
        out.push(&per_leaf[start..]);
    }
    out
}
