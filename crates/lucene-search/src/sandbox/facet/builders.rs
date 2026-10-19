//! `org.apache.lucene.sandbox.facet.utils`' builders and orchestrators: a
//! [`FacetBuilder`] per requested facet, collected together by a
//! [`FacetOrchestrator`] (or a [`DrillSidewaysFacetOrchestrator`]) and read
//! back as a [`FacetResult`].
//!
//! Java's `FacetBuilder` hierarchy (`BaseFacetBuilder`,
//! `CommonFacetBuilder`, `TaxonomyFacetBuilder`, `LongValueFacetBuilder`)
//! is one [`FacetBuilder`] with a kind: its abstract methods are
//! package-private, so no caller can add a kind. A builder's collection
//! state lives behind a `RefCell`, so an orchestrator borrows the builders
//! its caller keeps and reads the results from.

use std::cell::{Cell, RefCell};
use std::sync::Arc;

use crate::collector::ScoreMode;
use crate::facets::multi_double_values_source::{self, MultiDoubleValuesSource};
use crate::facets::multi_long_values_source::{self, MultiLongValuesSource};
use crate::facets::range::DoubleRange;
use crate::facets::range::LongRange;
use crate::facets::taxonomy::facet_label::FacetLabel;
use crate::facets::taxonomy::taxonomy_reader::TaxonomyReader;
use crate::facets::{
    DrillDownQuery, DrillSideways, FacetResult, FacetValue, FacetsConfig, LabelAndValue,
};
use crate::grouping::{search_manager, GroupingCollectorManager};
use crate::index_searcher::IndexSearcher;
use crate::leaf_collector::SegmentCollector;
use crate::multi_segment::OpenSegment;
use crate::query::BooleanQuery;
use crate::{Error, Result};

use super::comparable::{self, ByCount, ByLongValue, ByOrdinal, ComparableSupplier};
use super::cutters::{LongValueFacetCutter, TaxonomyFacetsCutter};
use super::iterators::{
    LengthOrdinalIterator, TaxonomyChildrenOrdinalIterator, TopnOrdinalIterator,
};
use super::labels::{LabelToOrd, OrdToLabel, RangeOrdToLabel, TaxonomyOrdLabelBiMap};
use super::ranges::{DoubleRangeFacetCutter, LongRangeFacetCutter};
use super::recorders::CountFacetRecorder;
use super::{
    BoxOrdinalIterator, FacetCutter, FacetFieldCollector, FacetFieldCollectorManager,
    FacetRecorder, OrdinalIterator,
};

/// `BaseFacetBuilder.sortOrderSupplier`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SortOrder {
    Count,
    Ordinal,
    LongValue,
}

/// What kind of builder this is (Java's subclasses).
enum Kind<'t> {
    /// `CommonFacetBuilder`.
    Common {
        cutter: Arc<dyn FacetCutter + 't>,
        labels: Arc<dyn OrdToLabel + 't>,
    },
    /// `TaxonomyFacetBuilder`.
    Taxonomy {
        config: &'t FacetsConfig,
        taxo: &'t dyn TaxonomyReader,
        multi_valued: bool,
        require_dim_count: bool,
        index_field_name: String,
        labels: TaxonomyOrdLabelBiMap<'t>,
        /// `parentOrd`: looked up at the first need, again while negative.
        parent_ord: Cell<i32>,
    },
    /// `LongValueFacetBuilder`: its cutter, set (or shared) at collection.
    LongValue {
        cutter: RefCell<Option<Arc<LongValueFacetCutter>>>,
    },
}

/// `FacetBuilder.collectionKey()`: builders with equal keys share one
/// collection.
#[derive(Debug, Clone, PartialEq, Eq)]
enum CollectionKey {
    Taxonomy(String),
    LongValue(String),
    /// A builder that shares only with itself (Java's `collectionKey()`
    /// returning `this`): its address, so a builder listed twice is
    /// collected once.
    Identity(usize),
}

/// `FacetBuilder` and its subclasses: one facet request -- what to cut
/// documents into, how to sort and cut the result -- collected by an
/// orchestrator, then read with [`get_result`](Self::get_result).
pub struct FacetBuilder<'t> {
    dimension: String,
    path: Vec<String>,
    top_n: i32,
    sort: SortOrder,
    kind: Kind<'t>,
    /// `countRecorder`: set (or shared with a similar builder) at
    /// collection.
    count_recorder: RefCell<Option<Arc<CountFacetRecorder>>>,
}

impl<'t> FacetBuilder<'t> {
    fn with_kind(dimension: String, path: Vec<String>, kind: Kind<'t>) -> Self {
        FacetBuilder {
            dimension,
            path,
            top_n: -1,
            sort: SortOrder::Count,
            kind,
            count_recorder: RefCell::new(None),
        }
    }

    /// `new CommonFacetBuilder(dimension, cutter, ordToLabel)`: counts, sorted
    /// by count; the overall value is `-1`.
    pub fn common(
        dimension: impl Into<String>,
        cutter: Arc<dyn FacetCutter + 't>,
        labels: Arc<dyn OrdToLabel + 't>,
    ) -> Self {
        Self::with_kind(
            dimension.into(),
            Vec::new(),
            Kind::Common { cutter, labels },
        )
    }

    /// `new TaxonomyFacetBuilder(facetsConfig, taxonomyReader, dimension,
    /// path...)`: the counts of `dimension/path`'s children, sorted by count.
    ///
    /// # Errors
    /// [`Error::IllegalArgument`] when `dimension` has no explicit config.
    pub fn taxonomy<S: AsRef<str>>(
        config: &'t FacetsConfig,
        taxo: &'t dyn TaxonomyReader,
        dimension: &str,
        path: &[S],
    ) -> Result<Self> {
        if !config.is_dim_configured(dimension) {
            return Err(Error::IllegalArgument(format!(
                "Dimension config for {dimension} is required. Call one of the FacetsConfig's \
                 setter with a default value."
            )));
        }
        let dim_config = config.dim_config(dimension);
        Ok(Self::with_kind(
            dimension.to_string(),
            path.iter().map(|p| p.as_ref().to_string()).collect(),
            Kind::Taxonomy {
                config,
                taxo,
                multi_valued: dim_config.multi_valued,
                require_dim_count: dim_config.require_dim_count,
                index_field_name: dim_config.index_field_name.clone(),
                labels: TaxonomyOrdLabelBiMap::new(taxo),
                parent_ord: Cell::new(-1),
            },
        ))
    }

    /// `new LongValueFacetBuilder(field)`: the counts of the field's
    /// distinct values, sorted by value.
    pub fn long_value(field: impl Into<String>) -> Self {
        let mut b = Self::with_kind(
            field.into(),
            Vec::new(),
            Kind::LongValue {
                cutter: RefCell::new(None),
            },
        );
        b.sort = SortOrder::LongValue;
        b
    }

    /// `withTopN(n)`: only the top `n` (`-1`, the default: all of them).
    #[must_use]
    pub fn with_top_n(mut self, n: i32) -> Self {
        self.top_n = n;
        self
    }

    /// `withSortByCount()`.
    #[must_use]
    pub fn with_sort_by_count(mut self) -> Self {
        self.sort = SortOrder::Count;
        self
    }

    /// `withSortByOrdinal()` (package-private in Java; the range factories'
    /// order).
    #[must_use]
    fn with_sort_by_ordinal(mut self) -> Self {
        self.sort = SortOrder::Ordinal;
        self
    }

    fn collection_key(&self) -> CollectionKey {
        match &self.kind {
            Kind::Taxonomy {
                index_field_name, ..
            } => CollectionKey::Taxonomy(index_field_name.clone()),
            Kind::LongValue { .. } => CollectionKey::LongValue(self.dimension.clone()),
            Kind::Common { .. } => CollectionKey::Identity(std::ptr::from_ref(self).addr()),
        }
    }

    /// `getFacetCutter()`.
    // The cutters are shared within one search, never across threads: no
    // `Send`/`Sync` bound (see the module docs).
    #[allow(clippy::arc_with_non_send_sync)]
    fn facet_cutter(&self) -> Result<Arc<dyn FacetCutter + 't>> {
        Ok(match &self.kind {
            Kind::Common { cutter, .. } => cutter.clone(),
            Kind::Taxonomy {
                config,
                taxo,
                index_field_name,
                ..
            } => Arc::new(TaxonomyFacetsCutter::new(
                index_field_name.clone(),
                config,
                *taxo,
            )),
            Kind::LongValue { cutter } => cutter.borrow().clone().ok_or_else(not_collected)?,
        })
    }

    /// `initOrReuseCollector(similar)`: shares a similar builder's recorder
    /// (and long-value cutter), or starts a collection of its own and
    /// returns its manager.
    fn init_or_reuse_collector(
        &self,
        similar: Option<&FacetBuilder<'t>>,
    ) -> Result<Option<FacetFieldCollectorManager<'t, CountFacetRecorder>>> {
        if let Kind::LongValue { cutter } = &self.kind {
            let shared = match similar.map(|s| &s.kind) {
                Some(Kind::LongValue { cutter: theirs }) => theirs.borrow().clone(),
                _ => Some(Arc::new(LongValueFacetCutter::new(self.dimension.clone()))),
            };
            *cutter.borrow_mut() = shared;
        }
        if let Some(similar) = similar {
            *self.count_recorder.borrow_mut() = similar.count_recorder.borrow().clone();
            return Ok(None);
        }
        let recorder = Arc::new(CountFacetRecorder::new());
        *self.count_recorder.borrow_mut() = Some(recorder.clone());
        Ok(Some(FacetFieldCollectorManager::new(
            self.facet_cutter()?,
            recorder,
        )))
    }

    fn recorder(&self) -> Result<Arc<CountFacetRecorder>> {
        self.count_recorder
            .borrow()
            .clone()
            .ok_or_else(not_collected)
    }

    fn long_value_cutter(&self) -> Result<Option<Arc<LongValueFacetCutter>>> {
        match &self.kind {
            Kind::LongValue { cutter } => {
                cutter.borrow().clone().map(Some).ok_or_else(not_collected)
            }
            _ => Ok(None),
        }
    }

    /// `TaxonomyFacetBuilder.getParentOrd()`, over a taxonomy builder's
    /// labels and cached ordinal.
    fn parent_ord(
        &self,
        labels: &TaxonomyOrdLabelBiMap<'t>,
        parent_ord: &Cell<i32>,
    ) -> Result<i32> {
        if parent_ord.get() < 0 {
            let label = FacetLabel::from_dim_path(&self.dimension, &self.path)?;
            parent_ord.set(labels.get_ord(&label)?);
        }
        Ok(parent_ord.get())
    }

    /// `getOverallValue()`: the parent's count for a taxonomy dim that is
    /// not multi-valued or requires a dim count, else `-1`.
    // SENTINEL: -1 is Java's "no overall value", reported as the result's
    // value exactly as `getResult` reports it, never used as an ordinal.
    fn overall_value(&self, recorder: &CountFacetRecorder) -> Result<i32> {
        match &self.kind {
            Kind::Taxonomy {
                multi_valued,
                require_dim_count,
                labels,
                parent_ord,
                ..
            } if !*multi_valued || *require_dim_count => {
                recorder.get_count(self.parent_ord(labels, parent_ord)?)
            }
            _ => Ok(-1),
        }
    }

    /// `getResult()`: the matching ordinals, sorted and cut to the top `n`,
    /// labelled with their last path component and their count.
    ///
    /// # Errors
    /// [`Error::IllegalState`] before the builder was collected (Java's
    /// assertion); whatever reading the counts or labels reports.
    pub fn get_result(&self) -> Result<FacetResult> {
        let recorder = self.recorder()?;
        let long_cutter = self.long_value_cutter()?;
        let recorded: BoxOrdinalIterator<'static> = recorder.recorded_ords()?;
        let matching: BoxOrdinalIterator<'_> = match &self.kind {
            Kind::Taxonomy {
                taxo,
                labels,
                parent_ord,
                ..
            } => Box::new(TaxonomyChildrenOrdinalIterator::new(
                recorded,
                taxo.parallel_taxonomy_arrays()?,
                self.parent_ord(labels, parent_ord)?,
            )),
            _ => recorded,
        };
        let mut length = LengthOrdinalIterator::new(matching);
        let ords = match (self.sort, long_cutter.as_deref()) {
            (SortOrder::LongValue, Some(c)) => self.ranked(&mut length, ByLongValue::new(c))?,
            (SortOrder::Ordinal, _) => self.ranked(&mut length, ByOrdinal)?,
            _ => self.ranked(&mut length, ByCount::new(&recorder))?,
        };
        let labels = match &self.kind {
            Kind::Common { labels, .. } => labels.get_labels(&ords)?,
            Kind::Taxonomy { labels, .. } => labels.get_labels(&ords)?,
            Kind::LongValue { .. } => long_cutter
                .as_deref()
                .ok_or_else(not_collected)?
                .get_labels(&ords)?,
        };
        let mut values = Vec::with_capacity(ords.len());
        for (&ord, label) in ords.iter().zip(labels) {
            // Java dereferences a `null` label: an NPE.
            let label = label
                .ok_or_else(|| Error::IllegalState(format!("facet ordinal {ord} has no label")))?;
            let count = recorder.get_count(ord)?;
            values.push(LabelAndValue::with_count(
                label.last_component()?,
                FacetValue::Int(count),
                count,
            ));
        }
        Ok(FacetResult::new(
            self.dimension.clone(),
            self.path.clone(),
            // SENTINEL-OK: -1 is the result's value as Java's `getResult` reports it.
            FacetValue::Int(self.overall_value(&recorder)?),
            values,
            length.length()?,
        ))
    }

    /// The top `n` through `TopnOrdinalIterator`, or every ordinal sorted.
    fn ranked<I: OrdinalIterator, S: ComparableSupplier>(
        &self,
        source: &mut I,
        supplier: S,
    ) -> Result<Vec<i32>> {
        if self.top_n != -1 {
            return TopnOrdinalIterator::new(source, supplier, self.top_n)?.to_array();
        }
        let mut ords = source.to_array()?;
        comparable::sort(&mut ords, &supplier)?;
        Ok(ords)
    }
}

impl<I: OrdinalIterator + ?Sized> OrdinalIterator for &mut I {
    fn next_ord(&mut self) -> Result<i32> {
        (**self).next_ord()
    }
}

fn not_collected() -> Error {
    Error::IllegalState("must not be called before collect".into())
}

/// `RangeFacetBuilderFactory`: [`FacetBuilder::common`] builders over
/// ranges, sorted by range order.
pub struct RangeFacetBuilderFactory;

impl RangeFacetBuilderFactory {
    /// `forLongRanges(field, ranges...)`: the field's `SORTED_NUMERIC` (or
    /// `NUMERIC`) values.
    pub fn for_long_ranges<'t>(field: &str, ranges: &[LongRange]) -> FacetBuilder<'t> {
        Self::for_long_ranges_source(
            field,
            multi_long_values_source::from_long_field(field),
            ranges,
        )
    }

    /// `forLongRanges(dimension, valuesSource, ranges...)`.
    pub fn for_long_ranges_source<'t>(
        dimension: &str,
        source: Arc<dyn MultiLongValuesSource>,
        ranges: &[LongRange],
    ) -> FacetBuilder<'t> {
        FacetBuilder::common(
            dimension,
            Arc::new(LongRangeFacetCutter::create(source, ranges)),
            Arc::new(RangeOrdToLabel::for_long_ranges(ranges)),
        )
        .with_sort_by_ordinal()
    }

    /// `forDoubleRanges(field, ranges...)`.
    pub fn for_double_ranges<'t>(field: &str, ranges: &[DoubleRange]) -> FacetBuilder<'t> {
        Self::for_double_ranges_source(
            field,
            multi_double_values_source::from_double_field(field),
            ranges,
        )
    }

    /// `forDoubleRanges(dimension, valuesSource, ranges...)`.
    pub fn for_double_ranges_source<'t>(
        dimension: &str,
        source: Arc<dyn MultiDoubleValuesSource>,
        ranges: &[DoubleRange],
    ) -> FacetBuilder<'t> {
        FacetBuilder::common(
            dimension,
            Arc::new(DoubleRangeFacetCutter::new(source, ranges)),
            Arc::new(RangeOrdToLabel::for_double_ranges(ranges)),
        )
        .with_sort_by_ordinal()
    }
}

/// `FacetOrchestrator.collectorManagerForBuilders`: one manager per
/// collection key, the other builders with that key sharing its recorder
/// (in first-seen order; Java's `HashMap` order only orders the
/// collectors).
fn collector_managers<'t>(
    builders: &[&FacetBuilder<'t>],
) -> Result<Vec<FacetFieldCollectorManager<'t, CountFacetRecorder>>> {
    let mut seen: Vec<(CollectionKey, usize)> = Vec::new();
    let mut managers = Vec::new();
    for (i, b) in builders.iter().enumerate() {
        let key = b.collection_key();
        let similar = seen
            .iter()
            .find(|(k, _)| *k == key)
            .map(|&(_, j)| builders[j]);
        if let Some(m) = b.init_or_reuse_collector(similar)? {
            seen.push((key, i));
            managers.push(m);
        }
    }
    Ok(managers)
}

/// `FacetOrchestrator.createMainCollector`'s `MultiCollectorManager`: the
/// facet managers beside an optional main manager.
struct MainCollectorManager<'m, 't, M> {
    main: Option<&'m M>,
    facets: Vec<FacetFieldCollectorManager<'t, CountFacetRecorder>>,
}

impl<'m, 't, M> MainCollectorManager<'m, 't, M> {
    /// `new MultiCollectorManager(managers)`.
    ///
    /// # Errors
    /// [`Error::IllegalArgument`] for no manager at all, as Java's.
    fn new(
        main: Option<&'m M>,
        facets: Vec<FacetFieldCollectorManager<'t, CountFacetRecorder>>,
    ) -> Result<Self> {
        if main.is_none() && facets.is_empty() {
            return Err(Error::IllegalArgument(
                "There must be at least one collector manager".into(),
            ));
        }
        Ok(MainCollectorManager { main, facets })
    }
}

/// Its collector: `MultiCollector`.
pub struct MainCollector<'t, 'a, C> {
    main: Option<C>,
    facets: Vec<FacetFieldCollector<'t, 'a>>,
}

impl<'a, C: SegmentCollector<'a>> SegmentCollector<'a> for MainCollector<'_, 'a, C> {
    /// `MultiCollector.scoreMode()`: the collectors' common mode, else
    /// `COMPLETE` when one needs scores, else `COMPLETE_NO_SCORES`.
    fn score_mode(&self) -> ScoreMode {
        match &self.main {
            Some(m) if self.facets.is_empty() => m.score_mode(),
            Some(m) if m.score_mode() == ScoreMode::CompleteNoScores => ScoreMode::CompleteNoScores,
            Some(m) if m.score_mode().needs_scores() => ScoreMode::Complete,
            _ => ScoreMode::CompleteNoScores,
        }
    }

    fn set_next_reader(&mut self, ord: usize, leaf: &OpenSegment<'a>) -> Result<()> {
        if let Some(m) = &mut self.main {
            m.set_next_reader(ord, leaf)?;
        }
        for f in &mut self.facets {
            f.set_next_reader(ord, leaf)?;
        }
        Ok(())
    }

    fn collect(&mut self, doc: i32, score: f32) -> Result<()> {
        if let Some(m) = &mut self.main {
            m.collect(doc, score)?;
        }
        for f in &mut self.facets {
            f.collect_doc(doc)?;
        }
        Ok(())
    }

    fn finish(&mut self) -> Result<()> {
        if let Some(m) = &mut self.main {
            m.finish()?;
        }
        for f in &mut self.facets {
            f.leave()?;
        }
        Ok(())
    }
}

impl<'t, 'a, M: GroupingCollectorManager<'a>> GroupingCollectorManager<'a>
    for MainCollectorManager<'_, 't, M>
{
    type Collector = MainCollector<'t, 'a, M::Collector>;
    type Output = Option<M::Output>;

    fn new_collector(&self) -> Result<Self::Collector> {
        Ok(MainCollector {
            main: self.main.map(|m| m.new_collector()).transpose()?,
            facets: self
                .facets
                .iter()
                .map(|f| f.new_facet_collector())
                .collect(),
        })
    }

    fn reduce(&self, collectors: Vec<Self::Collector>) -> Result<Option<M::Output>> {
        let mut mains = Vec::with_capacity(collectors.len());
        let mut facets: Vec<Vec<FacetFieldCollector<'t, 'a>>> =
            self.facets.iter().map(|_| Vec::new()).collect();
        for c in collectors {
            mains.extend(c.main);
            for (slot, f) in facets.iter_mut().zip(c.facets) {
                slot.push(f);
            }
        }
        let main = match self.main {
            Some(m) => Some(m.reduce(mains)?),
            None => None,
        };
        for (manager, collectors) in self.facets.iter().zip(facets) {
            manager.reduce_collectors(collectors)?;
        }
        Ok(main)
    }
}

/// A collector manager that is never used: the type of
/// [`FacetOrchestrator::collect`]'s missing main manager.
pub struct NoMainCollector;

impl<'a> GroupingCollectorManager<'a> for NoMainCollector {
    type Collector = NoCollector;
    type Output = ();

    fn new_collector(&self) -> Result<NoCollector> {
        Ok(NoCollector)
    }

    fn reduce(&self, _collectors: Vec<NoCollector>) -> Result<()> {
        Ok(())
    }
}

/// [`NoMainCollector`]'s collector.
pub struct NoCollector;

impl<'a> SegmentCollector<'a> for NoCollector {
    fn score_mode(&self) -> ScoreMode {
        ScoreMode::CompleteNoScores
    }

    fn set_next_reader(&mut self, _ord: usize, _leaf: &OpenSegment<'a>) -> Result<()> {
        Ok(())
    }

    fn collect(&mut self, _doc: i32, _score: f32) -> Result<()> {
        Ok(())
    }
}

/// `FacetOrchestrator`: collects several [`FacetBuilder`]s (and an optional
/// main collector) in one search, builders over the same index field or
/// long field sharing one collection.
#[derive(Default)]
pub struct FacetOrchestrator<'b, 't> {
    builders: Vec<&'b FacetBuilder<'t>>,
}

impl<'b, 't> FacetOrchestrator<'b, 't> {
    /// `new FacetOrchestrator()`.
    pub fn new() -> Self {
        FacetOrchestrator {
            builders: Vec::new(),
        }
    }

    /// `addBuilder(request)`.
    #[must_use]
    pub fn add_builder(mut self, builder: &'b FacetBuilder<'t>) -> Self {
        self.builders.push(builder);
        self
    }

    /// `collect(query, searcher)`.
    ///
    /// # Errors
    /// What the search, a cutter or a recorder reports.
    pub fn collect(&self, query: &BooleanQuery, searcher: &IndexSearcher<'_, '_>) -> Result<()> {
        self.collect_with(query, searcher, None::<&NoMainCollector>)
            .map(|_| ())
    }

    /// `collect(query, searcher, mainCollector)`: the main manager's result
    /// (`None` without one).
    ///
    /// # Errors
    /// What the search, a collector, a cutter or a recorder reports.
    pub fn collect_with<'a, M: GroupingCollectorManager<'a>>(
        &self,
        query: &BooleanQuery,
        searcher: &IndexSearcher<'_, 'a>,
        main: Option<&M>,
    ) -> Result<Option<M::Output>> {
        let manager = MainCollectorManager::new(main, collector_managers(&self.builders)?)?;
        search_manager(searcher, query, &manager)
    }
}

/// `DrillSidewaysFacetOrchestrator`: drill-down builders over the
/// documents matching the whole query, and per drilled dim, builders over
/// its sideways documents.
#[derive(Default)]
pub struct DrillSidewaysFacetOrchestrator<'b, 't> {
    drill_down: Vec<&'b FacetBuilder<'t>>,
    /// Per dim (in first-added order), its builders.
    sideways: Vec<(String, Vec<&'b FacetBuilder<'t>>)>,
}

impl<'b, 't> DrillSidewaysFacetOrchestrator<'b, 't> {
    /// `new DrillSidewaysFacetOrchestrator()`.
    pub fn new() -> Self {
        DrillSidewaysFacetOrchestrator {
            drill_down: Vec::new(),
            sideways: Vec::new(),
        }
    }

    /// `addDrillSidewaysBuilder(dim, facetBuilder)`: dims are numbered in
    /// the order first added, which must be the query's dim order.
    #[must_use]
    pub fn add_drill_sideways_builder(mut self, dim: &str, builder: &'b FacetBuilder<'t>) -> Self {
        match self.sideways.iter_mut().find(|(d, _)| d == dim) {
            Some((_, list)) => list.push(builder),
            None => self.sideways.push((dim.to_string(), vec![builder])),
        }
        self
    }

    /// `addDrillDownBuilder(facetBuilder)`.
    #[must_use]
    pub fn add_drill_down_builder(mut self, builder: &'b FacetBuilder<'t>) -> Self {
        self.drill_down.push(builder);
        self
    }

    /// `collect(query, drillSideways)`.
    ///
    /// # Errors
    /// As [`collect_with`](Self::collect_with).
    pub fn collect<'a>(
        &self,
        query: &DrillDownQuery<'_>,
        drill_sideways: &DrillSideways<'_, '_, 'a, '_>,
    ) -> Result<()> {
        self.collect_with(query, drill_sideways, None::<&NoMainCollector>)
            .map(|_| ())
    }

    /// `collect(query, drillSideways, mainCollector)`: the main manager's
    /// result over the drill-down documents.
    ///
    /// # Errors
    /// [`Error::IllegalArgument`] when a dim's builders need more than one
    /// collection, or the dims do not match the query's; what the search
    /// reports.
    pub fn collect_with<'a, M: GroupingCollectorManager<'a>>(
        &self,
        query: &DrillDownQuery<'_>,
        drill_sideways: &DrillSideways<'_, '_, 'a, '_>,
        main: Option<&M>,
    ) -> Result<Option<M::Output>> {
        let drill_down = MainCollectorManager::new(main, collector_managers(&self.drill_down)?)?;
        let mut sideways = Vec::with_capacity(self.sideways.len());
        for (_, builders) in &self.sideways {
            let mut managers = collector_managers(builders)?;
            if managers.len() != 1 {
                return Err(Error::IllegalArgument(format!(
                    "Expected exactly one collector manager per dimension but got {}",
                    managers.len()
                )));
            }
            sideways.extend(managers.pop());
        }
        let result = drill_sideways.search_managers(query, &drill_down, Some(&sideways))?;
        Ok(result.drill_down_result)
    }
}
