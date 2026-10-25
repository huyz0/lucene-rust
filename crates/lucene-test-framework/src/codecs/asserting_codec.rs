//! Bounded Java `AssertingCodec` composition, with eight eager asserting formats,
//! three cached per-field selector routers and six suppressible accessors.
//!
//! The explicit provider supplies each default independently of the retained
//! delegate. Its factory calls follow Java's initializer order; suppression is
//! resolved only afterwards. `Rc` preserves reference identity and `OnceCell`
//! binds escaped routers to the same selector hooks without self references.
//! Stage 3 stores each router as a typed handle over that shared state instead
//! of a second Rc allocation. Router clones preserve logical instance identity
//! (router type plus shared state); selected format Rc identities stay exact.
//! `Result` preserves factory/display errors. Resolved suppression replaces
//! RandomizedContext annotation lookup; thread failures panic and render Rust
//! thread IDs instead of Java Thread descriptions. No native default graph,
//! per-field format I/O, SPI or IndexWriter integration is supplied. The eight
//! wrapper values are the existing adapters; their operation trait bounds apply
//! when callers use their I/O methods, not when this composition stores them.
//! Selector hooks return asserting wrapper handles only. Arbitrary Java subclass
//! returns, including RandomCodec's non-asserting FST/Direct formats, require a
//! broader selection type and are not supported by this bounded seam.

use super::{
    asserting_doc_values::AssertingDocValuesFormat,
    asserting_knn_vectors::AssertingKnnVectorsFormat, asserting_live_docs::AssertingLiveDocsFormat,
    asserting_norms::AssertingNormsFormat, asserting_points::AssertingPointsFormat,
    asserting_postings::AssertingPostingsFormat,
    asserting_stored_fields::AssertingStoredFieldsFormat,
    asserting_term_vectors::AssertingTermVectorsFormat, cranky_live_docs::LiveDocsFormat,
};
use std::{cell::OnceCell, fmt, rc::Rc, thread::ThreadId};

/// Java: `AssertingCodec.Format`, in EnumSet iteration order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    StoredFields,
    TermVectors,
    Norms,
    LiveDocs,
    Points,
    KnnVectors,
}
impl Format {
    pub const ALL: [Self; 6] = [
        Self::StoredFields,
        Self::TermVectors,
        Self::Norms,
        Self::LiveDocs,
        Self::Points,
        Self::KnnVectors,
    ];
    pub fn name(self) -> &'static str {
        match self {
            Self::StoredFields => "STORED_FIELDS",
            Self::TermVectors => "TERM_VECTORS",
            Self::Norms => "NORMS",
            Self::LiveDocs => "LIVE_DOCS",
            Self::Points => "POINTS",
            Self::KnnVectors => "KNN_VECTORS",
        }
    }
}
/// Resolved inherited annotation values; duplicates and input order are immaterial.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Suppression(u8);
impl Suppression {
    pub fn new(values: impl IntoIterator<Item = Format>) -> Self {
        let mut mask = 0;
        for value in values {
            mask |= 1 << value as u8;
        }
        Self(mask)
    }
    pub fn contains(self, format: Format) -> bool {
        self.0 & (1 << format as u8) != 0
    }
}

/// Retained delegate factories. Handles remain opaque here; each factory is
/// called once per forwarded accessor, and its exact returned handle is kept.
pub trait Delegate {
    type Error;
    type DocValues;
    type FieldInfos;
    type LiveDocs;
    type Norms;
    type Postings;
    type SegmentInfo;
    type StoredFields;
    type TermVectors;
    type Compound;
    type Points;
    type KnnVectors;
    type SelectedKnnVectors;
    fn name(&self) -> &str;
    fn display(&self) -> Result<String, Self::Error>;
    fn doc_values_format(&self) -> Result<Self::DocValues, Self::Error>;
    fn field_infos_format(&self) -> Result<Self::FieldInfos, Self::Error>;
    fn live_docs_format(&self) -> Result<Self::LiveDocs, Self::Error>;
    fn norms_format(&self) -> Result<Self::Norms, Self::Error>;
    fn postings_format(&self) -> Result<Self::Postings, Self::Error>;
    fn segment_info_format(&self) -> Result<Self::SegmentInfo, Self::Error>;
    fn stored_fields_format(&self) -> Result<Self::StoredFields, Self::Error>;
    fn term_vectors_format(&self) -> Result<Self::TermVectors, Self::Error>;
    fn compound_format(&self) -> Result<Self::Compound, Self::Error>;
    fn points_format(&self) -> Result<Self::Points, Self::Error>;
    fn knn_vectors_format(&self) -> Result<Self::KnnVectors, Self::Error>;
    /// Explicit selector seam for the delegate's per-field KNN handle.
    fn select_knn_vectors(
        &self,
        router: &Self::KnnVectors,
        field: Option<&str>,
    ) -> Result<Self::SelectedKnnVectors, Self::Error>;
}

/// Java's hardcoded TestUtil defaults, provided explicitly. The eight returned
/// values are inner formats, not already asserting wrappers. In particular the
/// default KNN is TestUtil's threshold-zero format, distinct from the retained
/// Lucene104 delegate's threshold-100 format. This trait supplies no native graph.
pub trait DefaultProvider {
    type Error;
    type Delegate: Delegate<Error = Self::Error>;
    type TermVectors;
    type StoredFields;
    type Norms;
    type LiveDocs: LiveDocsFormat;
    type Postings;
    type DocValues;
    type Points;
    type KnnVectors;
    fn default_codec(&mut self) -> Result<Self::Delegate, Self::Error>;
    fn term_vectors(&mut self) -> Result<Self::TermVectors, Self::Error>;
    fn stored_fields(&mut self) -> Result<Self::StoredFields, Self::Error>;
    fn norms(&mut self) -> Result<Self::Norms, Self::Error>;
    fn live_docs(&mut self) -> Result<Self::LiveDocs, Self::Error>;
    fn postings(&mut self) -> Result<Self::Postings, Self::Error>;
    fn doc_values(&mut self) -> Result<Self::DocValues, Self::Error>;
    fn points(&mut self) -> Result<Self::Points, Self::Error>;
    fn knn_vectors(&mut self) -> Result<Self::KnnVectors, Self::Error>;
}
/// The eight actual existing asserting adapters; all are constructed even when
/// their public accessor is suppressed. Handles are stable for the codec's life.
pub struct Defaults<P: DefaultProvider> {
    pub term_vectors: Rc<AssertingTermVectorsFormat<P::TermVectors>>,
    pub stored_fields: Rc<AssertingStoredFieldsFormat<P::StoredFields>>,
    pub norms: Rc<AssertingNormsFormat<P::Norms>>,
    pub live_docs: Rc<AssertingLiveDocsFormat<P::LiveDocs>>,
    pub postings: Rc<AssertingPostingsFormat<P::Postings>>,
    pub doc_values: Rc<AssertingDocValuesFormat<P::DocValues>>,
    pub points: Rc<AssertingPointsFormat<P::Points>>,
    pub knn_vectors: Rc<AssertingKnnVectorsFormat<P::KnnVectors>>,
}
/// Stable selected asserting handles used by both direct and routed selectors.
pub type PostingsSelection<P> = Rc<AssertingPostingsFormat<<P as DefaultProvider>::Postings>>;
pub type DocValuesSelection<P> = Rc<AssertingDocValuesFormat<<P as DefaultProvider>::DocValues>>;
pub type KnnSelection<P> = Rc<AssertingKnnVectorsFormat<<P as DefaultProvider>::KnnVectors>>;
/// Suppressible typed accessor results; operation traits remain on each branch.
pub type TermVectorsChoice<P> = FormatChoice<
    AssertingTermVectorsFormat<<P as DefaultProvider>::TermVectors>,
    <<P as DefaultProvider>::Delegate as Delegate>::TermVectors,
>;
pub type StoredFieldsChoice<P> = FormatChoice<
    AssertingStoredFieldsFormat<<P as DefaultProvider>::StoredFields>,
    <<P as DefaultProvider>::Delegate as Delegate>::StoredFields,
>;
pub type NormsChoice<P> = FormatChoice<
    AssertingNormsFormat<<P as DefaultProvider>::Norms>,
    <<P as DefaultProvider>::Delegate as Delegate>::Norms,
>;
pub type LiveDocsChoice<P> = FormatChoice<
    AssertingLiveDocsFormat<<P as DefaultProvider>::LiveDocs>,
    <<P as DefaultProvider>::Delegate as Delegate>::LiveDocs,
>;
pub type PointsChoice<P> = FormatChoice<
    AssertingPointsFormat<<P as DefaultProvider>::Points>,
    <<P as DefaultProvider>::Delegate as Delegate>::Points,
>;
/// A cached typed KNN router handle or the delegate's exact native router.
pub enum KnnRouterChoice<P: DefaultProvider, H> {
    Cached(PerFieldKnnVectors<P, H>),
    Delegate(<P::Delegate as Delegate>::KnnVectors),
}
pub type PublicKnnSelection<P> = FormatChoice<
    AssertingKnnVectorsFormat<<P as DefaultProvider>::KnnVectors>,
    <<P as DefaultProvider>::Delegate as Delegate>::SelectedKnnVectors,
>;
/// Subclass selector overrides. A field may be empty or null, as in Java.
pub trait Selectors<P: DefaultProvider> {
    fn postings(
        &self,
        defaults: &Defaults<P>,
        _field: Option<&str>,
    ) -> Result<PostingsSelection<P>, P::Error> {
        Ok(defaults.postings.clone())
    }
    fn doc_values(
        &self,
        defaults: &Defaults<P>,
        _field: Option<&str>,
    ) -> Result<DocValuesSelection<P>, P::Error> {
        Ok(defaults.doc_values.clone())
    }
    fn knn_vectors(
        &self,
        defaults: &Defaults<P>,
        _field: Option<&str>,
    ) -> Result<KnnSelection<P>, P::Error> {
        Ok(defaults.knn_vectors.clone())
    }
}
#[derive(Default)]
pub struct DefaultSelectors;
impl<P: DefaultProvider> Selectors<P> for DefaultSelectors {}
struct SelectorState<P: DefaultProvider, H> {
    defaults: OnceCell<Defaults<P>>,
    hooks: H,
}
impl<P: DefaultProvider, H> SelectorState<P, H> {
    #[inline]
    fn defaults(&self) -> &Defaults<P> {
        self.defaults
            .get()
            .expect("eager defaults initialized before publication")
    }
}
/// Java's anonymous PerFieldPostingsFormat, bound to the codec's overrides.
pub struct PerFieldPostings<P: DefaultProvider, H> {
    state: Rc<SelectorState<P, H>>,
}
impl<P: DefaultProvider, H> Clone for PerFieldPostings<P, H> {
    #[inline]
    fn clone(&self) -> Self {
        Self {
            state: self.state.clone(),
        }
    }
}
impl<P: DefaultProvider, H: Selectors<P>> PerFieldPostings<P, H> {
    /// Logical identity of a cached router, retained across clones and codec drop.
    /// Different codecs have different instances; the router type distinguishes
    /// the three routers sharing one selector allocation.
    #[inline]
    pub fn same_instance(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.state, &other.state)
    }
    pub const NAME: &str = "PerField40";
    #[inline]
    pub fn select(&self, field: Option<&str>) -> Result<PostingsSelection<P>, P::Error> {
        self.state.hooks.postings(self.state.defaults(), field)
    }
}
/// Java's anonymous PerFieldDocValuesFormat; selector calls perform no I/O.
pub struct PerFieldDocValues<P: DefaultProvider, H> {
    state: Rc<SelectorState<P, H>>,
}
impl<P: DefaultProvider, H> Clone for PerFieldDocValues<P, H> {
    #[inline]
    fn clone(&self) -> Self {
        Self {
            state: self.state.clone(),
        }
    }
}
impl<P: DefaultProvider, H: Selectors<P>> PerFieldDocValues<P, H> {
    /// Logical identity of a cached router, retained across clones and codec drop.
    /// Different codecs have different instances; the router type distinguishes
    /// the three routers sharing one selector allocation.
    #[inline]
    pub fn same_instance(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.state, &other.state)
    }
    pub const NAME: &str = "PerFieldDV40";
    #[inline]
    pub fn select(&self, field: Option<&str>) -> Result<DocValuesSelection<P>, P::Error> {
        self.state.hooks.doc_values(self.state.defaults(), field)
    }
}
/// Java's cached anonymous PerFieldKnnVectorsFormat, including when suppressed.
pub struct PerFieldKnnVectors<P: DefaultProvider, H> {
    state: Rc<SelectorState<P, H>>,
}
impl<P: DefaultProvider, H> Clone for PerFieldKnnVectors<P, H> {
    #[inline]
    fn clone(&self) -> Self {
        Self {
            state: self.state.clone(),
        }
    }
}
impl<P: DefaultProvider, H: Selectors<P>> PerFieldKnnVectors<P, H> {
    /// Logical identity of a cached router, retained across clones and codec drop.
    /// Different codecs have different instances; the router type distinguishes
    /// the three routers sharing one selector allocation.
    #[inline]
    pub fn same_instance(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.state, &other.state)
    }
    pub const NAME: &str = "PerFieldVectors90";
    #[inline]
    pub fn select(&self, field: Option<&str>) -> Result<KnnSelection<P>, P::Error> {
        self.state.hooks.knn_vectors(self.state.defaults(), field)
    }
}
/// An accessor's exact cached asserting value or exact forwarded delegate value.
pub enum FormatChoice<A, D> {
    Cached(Rc<A>),
    Delegate(D),
}

/// Java: AssertingCodec. No factories or selectors run during access to cached
/// branches. Suppressed branches call the retained delegate on every invocation.
pub struct AssertingCodec<P: DefaultProvider, H = DefaultSelectors> {
    delegate: P::Delegate,
    postings: PerFieldPostings<P, H>,
    doc_values: PerFieldDocValues<P, H>,
    knn_vectors: PerFieldKnnVectors<P, H>,
    state: Rc<SelectorState<P, H>>,
    suppressed: Suppression,
}
impl<P: DefaultProvider, H: Selectors<P>> AssertingCodec<P, H> {
    /// Java constructor ordering. The context resolver runs after every eager
    /// initializer, and its errors propagate unchanged (no broad exception catch).
    pub fn new(
        provider: &mut P,
        resolve: impl FnOnce() -> Result<Suppression, P::Error>,
        hooks: H,
    ) -> Result<Self, P::Error> {
        let delegate = provider.default_codec()?;
        let state = Rc::new(SelectorState {
            defaults: OnceCell::new(),
            hooks,
        });
        let postings = PerFieldPostings {
            state: state.clone(),
        };
        let doc_values = PerFieldDocValues {
            state: state.clone(),
        };
        let knn_vectors = PerFieldKnnVectors {
            state: state.clone(),
        };
        let term_vectors = Rc::new(AssertingTermVectorsFormat::new(provider.term_vectors()?));
        let stored_fields = Rc::new(AssertingStoredFieldsFormat::new(provider.stored_fields()?));
        let norms = Rc::new(AssertingNormsFormat::new(provider.norms()?));
        let live_docs = Rc::new(AssertingLiveDocsFormat::new(provider.live_docs()?));
        let default_postings = Rc::new(AssertingPostingsFormat::new(provider.postings()?));
        let default_doc_values = Rc::new(AssertingDocValuesFormat::new(provider.doc_values()?));
        let points = Rc::new(AssertingPointsFormat::new(provider.points()?));
        let default_knn = Rc::new(AssertingKnnVectorsFormat::new(provider.knn_vectors()?));
        let _ = state.defaults.set(Defaults {
            term_vectors,
            stored_fields,
            norms,
            live_docs,
            postings: default_postings,
            doc_values: default_doc_values,
            points,
            knn_vectors: default_knn,
        });
        let suppressed = resolve()?;
        Ok(Self {
            delegate,
            postings,
            doc_values,
            knn_vectors,
            state,
            suppressed,
        })
    }
    pub const NAME: &str = "Asserting";
    pub fn name(&self) -> &'static str {
        Self::NAME
    }
    pub fn delegate(&self) -> &P::Delegate {
        &self.delegate
    }
    #[inline]
    pub fn defaults(&self) -> &Defaults<P> {
        self.state.defaults()
    }
    /// Bytes in the retained shared selector allocation, including two Rc
    /// counters, value alignment and trailing padding. This uses the pinned
    /// standard library's repr(C) Rc allocation layout. It excludes allocator
    /// metadata, cached wrappers/routers and the delegate or its inner graph.
    pub fn selector_state_bytes(&self) -> usize {
        rc_allocation_bytes::<SelectorState<P, H>>()
    }
    pub fn suppression(&self) -> Suppression {
        self.suppressed
    }
    pub fn hooks(&self) -> &H {
        &self.state.hooks
    }
    /// Java: toString; primary fallible API retains the delegate's exact error.
    pub fn display(&self) -> Result<String, P::Error> {
        let mut result = format!("Asserting({}", self.delegate.display()?);
        if self.suppressed != Suppression::default() {
            result.push_str(", suppressed: ");
            let mut first = true;
            for format in Format::ALL {
                if self.suppressed.contains(format) {
                    if !first {
                        result.push_str(", ");
                    }
                    first = false;
                    result.push_str(format.name());
                }
            }
        }
        result.push(')');
        Ok(result)
    }
    #[inline]
    pub fn postings_format(&self) -> PerFieldPostings<P, H> {
        self.postings.clone()
    }
    #[inline]
    pub fn doc_values_format(&self) -> PerFieldDocValues<P, H> {
        self.doc_values.clone()
    }
    /// The cached asserting router remains bound to the outer selector under suppression.
    #[inline]
    pub fn cached_knn_vectors_format(&self) -> PerFieldKnnVectors<P, H> {
        self.knn_vectors.clone()
    }
    pub fn get_postings_format_for_field(
        &self,
        field: Option<&str>,
    ) -> Result<PostingsSelection<P>, P::Error> {
        self.postings.select(field)
    }
    pub fn get_doc_values_format_for_field(
        &self,
        field: Option<&str>,
    ) -> Result<DocValuesSelection<P>, P::Error> {
        self.doc_values.select(field)
    }
    pub fn get_knn_vectors_format_for_field(
        &self,
        field: Option<&str>,
    ) -> Result<KnnSelection<P>, P::Error> {
        self.knn_vectors.select(field)
    }
    pub fn field_infos_format(&self) -> Result<<P::Delegate as Delegate>::FieldInfos, P::Error> {
        self.delegate.field_infos_format()
    }
    pub fn segment_info_format(&self) -> Result<<P::Delegate as Delegate>::SegmentInfo, P::Error> {
        self.delegate.segment_info_format()
    }
    pub fn compound_format(&self) -> Result<<P::Delegate as Delegate>::Compound, P::Error> {
        self.delegate.compound_format()
    }
    pub fn term_vectors_format(&self) -> Result<TermVectorsChoice<P>, P::Error> {
        if self.suppressed.contains(Format::TermVectors) {
            Ok(FormatChoice::Delegate(self.delegate.term_vectors_format()?))
        } else {
            Ok(FormatChoice::Cached(self.defaults().term_vectors.clone()))
        }
    }
    pub fn stored_fields_format(&self) -> Result<StoredFieldsChoice<P>, P::Error> {
        if self.suppressed.contains(Format::StoredFields) {
            Ok(FormatChoice::Delegate(
                self.delegate.stored_fields_format()?,
            ))
        } else {
            Ok(FormatChoice::Cached(self.defaults().stored_fields.clone()))
        }
    }
    pub fn norms_format(&self) -> Result<NormsChoice<P>, P::Error> {
        if self.suppressed.contains(Format::Norms) {
            Ok(FormatChoice::Delegate(self.delegate.norms_format()?))
        } else {
            Ok(FormatChoice::Cached(self.defaults().norms.clone()))
        }
    }
    pub fn live_docs_format(&self) -> Result<LiveDocsChoice<P>, P::Error> {
        if self.suppressed.contains(Format::LiveDocs) {
            Ok(FormatChoice::Delegate(self.delegate.live_docs_format()?))
        } else {
            Ok(FormatChoice::Cached(self.defaults().live_docs.clone()))
        }
    }
    pub fn points_format(&self) -> Result<PointsChoice<P>, P::Error> {
        if self.suppressed.contains(Format::Points) {
            Ok(FormatChoice::Delegate(self.delegate.points_format()?))
        } else {
            Ok(FormatChoice::Cached(self.defaults().points.clone()))
        }
    }
    pub fn knn_vectors_format(&self) -> Result<KnnRouterChoice<P, H>, P::Error> {
        if self.suppressed.contains(Format::KnnVectors) {
            Ok(KnnRouterChoice::Delegate(
                self.delegate.knn_vectors_format()?,
            ))
        } else {
            Ok(KnnRouterChoice::Cached(self.knn_vectors.clone()))
        }
    }
    pub fn select_public_knn_vectors(
        &self,
        field: Option<&str>,
    ) -> Result<PublicKnnSelection<P>, P::Error> {
        match self.knn_vectors_format()? {
            KnnRouterChoice::Cached(router) => Ok(FormatChoice::Cached(router.select(field)?)),
            KnnRouterChoice::Delegate(router) => Ok(FormatChoice::Delegate(
                self.delegate.select_knn_vectors(&router, field)?,
            )),
        }
    }
}
impl<P: DefaultProvider, H: Selectors<P>> fmt::Display for AssertingCodec<P, H> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.display().map_err(|_| fmt::Error)?)
    }
}
fn rc_allocation_bytes<T>() -> usize {
    let header = std::alloc::Layout::new::<[usize; 2]>();
    let (allocation, _) = header
        .extend(std::alloc::Layout::new::<T>())
        .expect("Rc value layout");
    allocation.pad_to_align().size()
}
/// Java: AssertingCodec.assertThread, unconditional even in release builds.
/// Rust ThreadId replaces Java's thread identity; null creation threads fail.
pub fn assert_thread(object: Option<&str>, creation_thread: Option<ThreadId>) {
    let current = std::thread::current().id();
    if creation_thread != Some(current) {
        let owner = creation_thread.map_or_else(|| "null".into(), |thread| format!("{thread:?}"));
        panic!("{} are only supposed to be consumed in the thread in which they have been acquired. But was acquired in {} and consumed in {:?}.", object.unwrap_or("null"), owner, current);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn allocation_accounting_retains_payload_counters_and_alignment() {
        // Eight cached default handles and a concrete hook payload are retained
        // by the shared state. The OnceCell tag and all alignment remain counted.
        type State = OnceCell<([Rc<()>; 8], [u8; 23])>;
        let bytes = rc_allocation_bytes::<State>();
        assert!(bytes >= 2 * std::mem::size_of::<usize>() + std::mem::size_of::<State>());
        #[repr(align(64))]
        struct Aligned([u8; 3]);
        let payload = Aligned([1, 2, 3]);
        assert_eq!(payload.0, [1, 2, 3]);
        assert_eq!(rc_allocation_bytes::<Aligned>(), 128);
    }
    #[test]
    fn suppression_is_an_enum_set_in_declaration_order() {
        let set = Suppression::new([
            Format::KnnVectors,
            Format::Points,
            Format::StoredFields,
            Format::Points,
        ]);
        assert_eq!(
            set,
            Suppression::new([Format::StoredFields, Format::Points, Format::KnnVectors])
        );
        let names: Vec<_> = Format::ALL
            .into_iter()
            .filter(|&format| set.contains(format))
            .map(Format::name)
            .collect();
        assert_eq!(names, ["STORED_FIELDS", "POINTS", "KNN_VECTORS"]);
        assert!(!Suppression::default().contains(Format::TermVectors));
    }
    #[test]
    fn thread_ownership_is_unconditional_and_null_owner_fails() {
        let owner = std::thread::current().id();
        assert_thread(None, Some(owner));
        let error = std::panic::catch_unwind(|| assert_thread(None, None)).unwrap_err();
        let actual = error.downcast_ref::<String>().unwrap();
        assert_eq!(actual, &format!("null are only supposed to be consumed in the thread in which they have been acquired. But was acquired in null and consumed in {owner:?}."));
        assert!(
            std::thread::spawn(move || std::panic::catch_unwind(|| assert_thread(
                Some(""),
                Some(owner)
            ))
            .is_err())
            .join()
            .unwrap()
        );
    }
}
