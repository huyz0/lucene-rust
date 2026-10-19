//! `RandomSamplingFacetsCollector`: a [`FacetsCollector`] whose matching
//! documents are, once there are more than the sample size of them, a
//! random sample -- one document per bin of `1 / samplingRate` hits, drawn
//! by the same XOR-shift generator as Java's, so a seed samples the same
//! documents -- plus [`RandomSamplingFacetsCollector::amortize_facet_counts`],
//! which scales counts over the sample back up.
//!
//! Differs: Java's `getMatchingDocs()` override is
//! [`RandomSamplingFacetsCollector::matching_docs`], which hands back the
//! collector to count over (the original one while the hits fit the sample,
//! else one holding the sample) rather than its list; the collector is
//! built by [`RandomSamplingFacetsCollectorManager`] (`createManager`).

use lucene_util::fixed_bit_set::FixedBitSet;

use super::facets_collector::{DocIdSet, FacetsCollector, FacetsCollectorManager, MatchingDocs};
use super::{path_to_string, FacetResult, FacetValue, FacetsConfig, LabelAndValue};
use crate::collector::ScoreMode;
use crate::index_searcher::IndexSearcher;
use crate::leaf_collector::{search_slices, SegmentCollector};
use crate::multi_segment::OpenSegment;
use crate::query::BooleanQuery;
use crate::reader::{DocIdSetIterator, IndexReader, NO_MORE_DOCS};
use crate::{Error, Result};

/// `XORShift64Random`.
#[derive(Debug, Clone)]
struct XorShift64 {
    x: i64,
}

impl XorShift64 {
    /// Seed `0` stands for Java's `0xdeadbeef`: an `int` literal, so
    /// negative, sign-extended to `long`.
    fn new(seed: i64) -> Self {
        XorShift64 {
            x: if seed == 0 {
                i64::from(0xdead_beef_u32 as i32)
            } else {
                seed
            },
        }
    }

    /// `randomLong()`.
    fn random_long(&mut self) -> i64 {
        self.x ^= self.x << 21;
        self.x ^= ((self.x as u64) >> 35) as i64;
        self.x ^= self.x << 4;
        self.x
    }

    /// `nextInt(n)`: `|(int) (randomLong() % n)|`, a negative `n` too (a
    /// negative sample size makes one). `n == 0`, where Java divides by
    /// zero, is refused by `create_sample` before it gets here; it gives `0`
    /// rather than panicking.
    fn next_int(&mut self, n: i32) -> i32 {
        let x = self.random_long();
        if n == 0 {
            return 0;
        }
        (x.wrapping_rem(i64::from(n)) as i32).wrapping_abs()
    }
}

/// `RandomSamplingFacetsCollector.NOT_CALCULATED`.
// SENTINEL: -1 = not calculated, Java's own marker for the leftover bin.
const NOT_CALCULATED: i32 = -1;

/// `RandomSamplingFacetsCollector`.
pub struct RandomSamplingFacetsCollector<'a> {
    inner: FacetsCollector<'a>,
    sample_size: i32,
    random: XorShift64,
    sampling_rate: f64,
    /// The sample, once drawn.
    sampled: Option<FacetsCollector<'a>>,
    /// `totalHits`: `None` until [`Self::matching_docs`] summed it.
    total_hits: Option<i32>,
    leftover_bin: i32,
    leftover_index: i32,
}

impl std::fmt::Debug for RandomSamplingFacetsCollector<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RandomSamplingFacetsCollector")
            .field("sample_size", &self.sample_size)
            .field("sampling_rate", &self.sampling_rate)
            .field("total_hits", &self.total_hits)
            .finish()
    }
}

impl<'a> RandomSamplingFacetsCollector<'a> {
    /// `new RandomSamplingFacetsCollector(sampleSize)`: seed `0`.
    pub fn new(sample_size: i32) -> Self {
        Self::with_seed(sample_size, 0)
    }

    /// `new RandomSamplingFacetsCollector(sampleSize, seed)`; seed `0`
    /// stands for Java's `0xdeadbeef` (see `XorShift64::new`).
    pub fn with_seed(sample_size: i32, seed: i64) -> Self {
        Self::over(sample_size, seed, FacetsCollector::new(false))
    }

    fn over(sample_size: i32, seed: i64, inner: FacetsCollector<'a>) -> Self {
        RandomSamplingFacetsCollector {
            inner,
            sample_size,
            random: XorShift64::new(seed),
            sampling_rate: 0.0,
            sampled: None,
            total_hits: None,
            leftover_bin: NOT_CALCULATED,
            leftover_index: NOT_CALCULATED,
        }
    }

    /// `getMatchingDocs()`: the collector to count over -- the original
    /// hits while they number no more than the sample size, else the sample
    /// (drawn on the first call).
    ///
    /// # Errors
    /// Whatever iterating the hits reports.
    pub fn matching_docs(&mut self) -> Result<&FacetsCollector<'a>> {
        let total = *self.total_hits.get_or_insert_with(|| {
            self.inner
                .matching_docs()
                .iter()
                .fold(0i32, |t, m| t.wrapping_add(m.total_hits))
        });
        if total <= self.sample_size {
            return Ok(&self.inner);
        }
        if self.sampled.is_none() {
            self.sampling_rate = f64::from(self.sample_size) / f64::from(total);
            let mut sampled = FacetsCollector::new(false);
            // Not a `map`: each sample advances the generator and leftovers.
            for i in 0..self.inner.matching_docs().len() {
                let docs = &self.inner.matching_docs()[i];
                let (context, total_hits) = (docs.context, docs.total_hits);
                let bits = self.create_sample(i)?;
                sampled.push_matching_docs(MatchingDocs {
                    context,
                    bits,
                    total_hits,
                    scores: None,
                });
            }
            self.sampled = Some(sampled);
        }
        Ok(self.sampled.as_ref().unwrap_or(&self.inner))
    }

    /// `getOriginalMatchingDocs()`.
    pub fn original_matching_docs(&self) -> &FacetsCollector<'a> {
        &self.inner
    }

    /// `getSamplingRate()`: `0` until a sample was drawn.
    pub fn sampling_rate(&self) -> f64 {
        self.sampling_rate
    }

    /// `createSample(docs)`: one document per bin of the segment's hits, a
    /// bin begun in the previous segment finishing in this one.
    fn create_sample(&mut self, i: usize) -> Result<DocIdSet> {
        let docs = &self.inner.matching_docs()[i];
        let max_doc = usize::try_from(docs.context.reader.max_doc()).unwrap_or(0);
        let mut sample = FixedBitSet::new(max_doc);
        // `(int) (1.0 / samplingRate)`: `as` saturates as Java's cast does.
        let bin_size = (1.0 / self.sampling_rate) as i32;
        if bin_size == 0 {
            // A sample size below `-totalHits`: Java's `nextInt(0)` throws
            // `ArithmeticException` here rather than sampling anything.
            return Err(Error::IllegalArgument(format!(
                "/ by zero: sample size {} over {} hits leaves bins of 0 documents",
                self.sample_size,
                self.total_hits.unwrap_or(0)
            )));
        }
        let mut counter = 0i32;
        let (mut limit, mut random_index) = if self.leftover_bin != NOT_CALCULATED {
            (self.leftover_bin, self.leftover_index)
        } else {
            (bin_size, self.random.next_int(bin_size))
        };
        let mut it = docs.bits.iterator();
        let mut doc = it.next_doc()?;
        while doc != NO_MORE_DOCS {
            if counter == random_index {
                if let Ok(d) = usize::try_from(doc) {
                    if d < max_doc {
                        // FBS: d < max_doc, the set's length.
                        sample.set(d);
                    }
                }
            }
            counter = counter.wrapping_add(1);
            if counter >= limit {
                counter = 0;
                limit = bin_size;
                random_index = self.random.next_int(bin_size);
            }
            doc = it.next_doc()?;
        }
        if counter == 0 {
            self.leftover_bin = NOT_CALCULATED;
            self.leftover_index = NOT_CALCULATED;
        } else {
            self.leftover_bin = limit.wrapping_sub(counter);
            if random_index > counter {
                self.leftover_index = random_index.wrapping_sub(counter);
            } else if random_index < counter {
                self.leftover_index = NOT_CALCULATED;
            }
            // `randomIndex == counter` leaves the previous leftover index,
            // as Java's does.
        }
        let cost = i64::try_from(sample.cardinality()).unwrap_or(i64::MAX);
        Ok(DocIdSet::Bits { bits: sample, cost })
    }

    /// `amortizeFacetCounts(res, config, searcher)`: counts over the sample
    /// divided by the sampling rate, each capped by its label's document
    /// frequency (and the total by the reader's live documents). `res` comes
    /// back unchanged when no sample was drawn.
    ///
    /// # Errors
    /// Whatever reading the document frequencies reports;
    /// [`Error::IllegalArgument`] for an empty path component.
    pub fn amortize_facet_counts(
        &self,
        res: Option<FacetResult>,
        config: &FacetsConfig,
        reader: &dyn IndexReader,
    ) -> Result<Option<FacetResult>> {
        let Some(res) = res else {
            return Ok(None);
        };
        if self.total_hits.is_none_or(|t| t <= self.sample_size) {
            return Ok(Some(res));
        }
        let index_field = &config.dim_config(&res.dim).index_field_name;
        let mut child_path: Vec<&str> = res.path.iter().map(String::as_str).collect();
        let mut fixed = Vec::with_capacity(res.label_values.len());
        for lv in &res.label_values {
            child_path.push(&lv.label);
            let full = path_to_string(&res.dim, &child_path)
                .map_err(|e| Error::IllegalArgument(e.to_string()))?;
            child_path.pop();
            let max = reader.doc_freq(index_field, full.as_bytes())?;
            let max = i32::try_from(max).unwrap_or(i32::MAX);
            let corrected = (lv.value.double_value() / self.sampling_rate) as i32;
            fixed.push(LabelAndValue::new(
                lv.label.clone(),
                FacetValue::Int(max.min(corrected)),
            ));
        }
        let mut total = res.value.int_value();
        if total > 0 {
            total = reader
                .num_docs()
                .min((res.value.double_value() / self.sampling_rate) as i32);
        }
        Ok(Some(FacetResult::new(
            res.dim,
            res.path,
            FacetValue::Int(total),
            fixed,
            res.child_count,
        )))
    }
}

impl<'a> SegmentCollector<'a> for RandomSamplingFacetsCollector<'a> {
    fn score_mode(&self) -> ScoreMode {
        self.inner.score_mode()
    }

    fn set_next_reader(&mut self, ord: usize, leaf: &OpenSegment<'a>) -> Result<()> {
        self.inner.set_next_reader(ord, leaf)
    }

    #[inline]
    fn collect(&mut self, doc: i32, score: f32) -> Result<()> {
        self.inner.collect(doc, score)
    }

    fn collect_range(&mut self, min: i32, max: i32, score: f32) -> Result<()> {
        self.inner.collect_range(min, max, score)
    }

    fn finish(&mut self) -> Result<()> {
        self.inner.finish()
    }
}

/// `RandomSamplingFacetsCollector.createManager(sampleSize, seed)`.
#[derive(Debug, Clone, Copy)]
pub struct RandomSamplingFacetsCollectorManager {
    sample_size: i32,
    seed: i64,
}

impl RandomSamplingFacetsCollectorManager {
    /// `createManager(sampleSize, seed)`.
    pub fn new(sample_size: i32, seed: i64) -> Self {
        RandomSamplingFacetsCollectorManager { sample_size, seed }
    }

    /// `newCollector()`.
    pub fn new_collector<'a>(&self) -> RandomSamplingFacetsCollector<'a> {
        RandomSamplingFacetsCollector::with_seed(self.sample_size, self.seed)
    }

    /// `reduce(collectors)`: none is a new collector, one is itself, several
    /// are a new collector over their merged hits
    /// (`ReducedRandomSamplingFacetsCollector`).
    ///
    /// # Errors
    /// Whatever merging the hits reports.
    pub fn reduce<'a>(
        &self,
        mut collectors: Vec<RandomSamplingFacetsCollector<'a>>,
    ) -> Result<RandomSamplingFacetsCollector<'a>> {
        if collectors.len() <= 1 {
            return Ok(collectors.pop().unwrap_or_else(|| self.new_collector()));
        }
        let inners = collectors.into_iter().map(|c| c.inner).collect();
        let merged = FacetsCollectorManager::new(false).reduce(inners)?;
        Ok(RandomSamplingFacetsCollector::over(
            self.sample_size,
            self.seed,
            merged,
        ))
    }

    /// `searcher.search(query, manager)`.
    ///
    /// # Errors
    /// Whatever the search reports.
    pub fn collect<'a>(
        &self,
        searcher: &IndexSearcher<'_, 'a>,
        query: &BooleanQuery,
    ) -> Result<RandomSamplingFacetsCollector<'a>> {
        let collectors = search_slices(searcher, query, || Ok(self.new_collector()))?;
        self.reduce(collectors)
    }
}

#[cfg(test)]
mod tests;
