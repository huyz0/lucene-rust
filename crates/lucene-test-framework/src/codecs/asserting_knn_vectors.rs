//! Java's `AssertingKnnVectorsFormat`, including its merge-reader state machine.
//! Reader assertions precede delegation and inspect a fresh iterator without
//! wrapping the returned vector values. Merge completion marks the handle before
//! delegation, but increments the parent's completion count only after success.
//! Base reader close delegates twice, then checks the Java completion predicate.
//! Writer callbacks, field handles and optional deferred tasks pass through intact.
//!
//! Local vector/AcceptDocs iterator seams preserve Java's callback order without
//! changing production reader traits. Values and delegates may be absent at the
//! seam so assertion failures can be tested; signed dimensions are represented by
//! usize, making negative dimensions unrepresentable. All assertions stay enabled.
//! AssertingAcceptDocs checks the creation thread and a fresh iterator's docID;
//! already asserting inputs are retained. No wrapper callback consumes randomness.
//!
//! The explicit unquantized Lucene99 reader opens `.vec/.vemf/.vex/.vem`, checks
//! FieldInfos, and uses native graph searches with document-space filters and
//! ordinal translation. Native production readers check all four CRCs on open;
//! Java defers data-file CRC computation to checkIntegrity. Immutable bytes/parsed metadata are borrowed; native
//! handles close independently, while borrowed values/graphs remain usable.
//! This models ByteBuffers inputs, not a backend that invalidates borrowed inputs.
//! IOContext, Codec/IndexWriter routing and native writer/merge adapters are absent.
//! Generic writer forwarding and its inherited two-phase merge are represented.

use std::collections::BTreeMap;
use std::io;
use std::sync::{
    atomic::{AtomicI32, Ordering},
    Arc,
};
use std::thread::{self, ThreadId};

use lucene_codecs::field_infos::{FieldInfo, FieldInfos, VectorEncoding};
use lucene_codecs::hnsw::{HnswGraphView, KnnCollect};
use lucene_codecs::hnsw_vectors::{self, HnswVectorsReader, OffHeapHnswGraph, SearchOptions};
use lucene_codecs::segment_state::SegmentReadState;
use lucene_codecs::vectors::{self, FlatVectorsReader};
use lucene_search::reader::{DocIdSetIterator, NO_MORE_DOCS};
use lucene_store::codec_util;
use lucene_util::fixed_bit_set::FixedBitSet;

pub type Result<T> = io::Result<T>;
fn failure(error: impl std::fmt::Display) -> io::Error {
    io::Error::other(error.to_string())
}

/// Java AcceptDocs callbacks; a missing bit slice accepts every document.
pub trait AcceptDocs {
    fn bits(&self) -> Result<Option<&[bool]>>;
    fn cost(&self) -> Result<i32>;
    fn iterator(&self) -> Result<Box<dyn DocIdSetIterator + '_>>;
    fn is_asserting(&self) -> bool {
        false
    }
}
pub struct AssertingAcceptDocs<'a, D: AcceptDocs + ?Sized = dyn AcceptDocs + 'a> {
    delegate: &'a D,
    creation_thread: ThreadId,
}
pub enum AcceptDocsGuard<'a> {
    Existing(&'a dyn AcceptDocs),
    New(AssertingAcceptDocs<'a>),
}
impl<'a> AssertingAcceptDocs<'a> {
    /// Java: AssertingAcceptDocs.wrap, including the existing-wrapper identity.
    pub fn wrap(delegate: &'a dyn AcceptDocs) -> AcceptDocsGuard<'a> {
        if delegate.is_asserting() {
            AcceptDocsGuard::Existing(delegate)
        } else {
            AcceptDocsGuard::New(Self {
                delegate,
                creation_thread: thread::current().id(),
            })
        }
    }
}
impl<'a, D: AcceptDocs + ?Sized> AssertingAcceptDocs<'a, D> {
    pub fn new(delegate: &'a D) -> Self {
        Self {
            delegate,
            creation_thread: thread::current().id(),
        }
    }
    fn check_thread(&self) {
        assert_eq!(
            self.creation_thread,
            thread::current().id(),
            "Usage of AcceptDocs should be confined to a single thread"
        );
    }
}
impl<D: AcceptDocs + ?Sized> AcceptDocs for AssertingAcceptDocs<'_, D> {
    fn bits(&self) -> Result<Option<&[bool]>> {
        self.check_thread();
        self.delegate.bits()
    }
    fn cost(&self) -> Result<i32> {
        self.check_thread();
        self.delegate.cost()
    }
    fn iterator(&self) -> Result<Box<dyn DocIdSetIterator + '_>> {
        self.check_thread();
        let iterator = self.delegate.iterator()?;
        assert_eq!(iterator.doc_id(), -1, "Iterator must be unpositioned");
        Ok(iterator)
    }
    fn is_asserting(&self) -> bool {
        true
    }
}
impl AcceptDocs for AcceptDocsGuard<'_> {
    fn bits(&self) -> Result<Option<&[bool]>> {
        match self {
            Self::Existing(v) => v.bits(),
            Self::New(v) => v.bits(),
        }
    }
    fn cost(&self) -> Result<i32> {
        match self {
            Self::Existing(v) => v.cost(),
            Self::New(v) => v.cost(),
        }
    }
    fn iterator(&self) -> Result<Box<dyn DocIdSetIterator + '_>> {
        match self {
            Self::Existing(v) => v.iterator(),
            Self::New(v) => v.iterator(),
        }
    }
    fn is_asserting(&self) -> bool {
        true
    }
}
pub struct BitsAcceptDocs<'a> {
    bits: Option<&'a [bool]>,
    max_doc: i32,
}
impl<'a> BitsAcceptDocs<'a> {
    pub fn new(bits: Option<&'a [bool]>, max_doc: i32) -> Self {
        assert!(max_doc >= 0);
        if let Some(bits) = bits {
            assert_eq!(bits.len(), max_doc as usize);
        }
        Self { bits, max_doc }
    }
}
struct Docs<'a> {
    bits: Option<&'a [bool]>,
    max_doc: i32,
    doc: i32,
}
impl DocIdSetIterator for Docs<'_> {
    fn doc_id(&self) -> i32 {
        self.doc
    }
    fn next_doc(&mut self) -> lucene_search::Result<i32> {
        if self.doc == NO_MORE_DOCS {
            return Ok(NO_MORE_DOCS);
        }
        let mut next = self.doc + 1;
        while next < self.max_doc && self.bits.is_some_and(|bits| !bits[next as usize]) {
            next += 1;
        }
        self.doc = if next < self.max_doc {
            next
        } else {
            NO_MORE_DOCS
        };
        Ok(self.doc)
    }
    fn advance(&mut self, target: i32) -> lucene_search::Result<i32> {
        self.doc = target.saturating_sub(1);
        self.next_doc()
    }
    fn cost(&self) -> i64 {
        i64::from(self.max_doc)
    }
}
impl AcceptDocs for BitsAcceptDocs<'_> {
    fn bits(&self) -> Result<Option<&[bool]>> {
        Ok(self.bits)
    }
    fn cost(&self) -> Result<i32> {
        Ok(self.max_doc)
    }
    fn iterator(&self) -> Result<Box<dyn DocIdSetIterator + '_>> {
        Ok(Box::new(Docs {
            bits: self.bits,
            max_doc: self.max_doc,
            doc: -1,
        }))
    }
}
/// The wrapper only inspects size, dimension and a fresh iterator; values pass through.
pub trait VectorValues<T> {
    fn size(&self) -> i32;
    fn dimension(&self) -> usize;
    fn iterator(&self) -> Result<Box<dyn DocIdSetIterator + '_>>;
    fn ord_to_doc(&self, ord: i32) -> Result<i32>;
    fn vector(&self, ord: i32) -> Result<Vec<T>>;
}
pub trait KnnVectorsReader {
    type Floats<'a>: VectorValues<f32>
    where
        Self: 'a;
    type Bytes<'a>: VectorValues<u8>
    where
        Self: 'a;
    type Graph<'a>: HnswGraphView
    where
        Self: 'a;
    fn float_values(&self, field: &str) -> Result<Option<Self::Floats<'_>>>;
    fn byte_values(&self, field: &str) -> Result<Option<Self::Bytes<'_>>>;
    fn search_float(
        &self,
        field: &str,
        target: &[f32],
        collector: &mut dyn KnnCollect,
        accept: &dyn AcceptDocs,
    ) -> Result<()>;
    fn search_byte(
        &self,
        field: &str,
        target: &[u8],
        collector: &mut dyn KnnCollect,
        accept: &dyn AcceptDocs,
    ) -> Result<()>;
    fn check_integrity(&self) -> Result<()>;
    fn get_merge_instance(&self) -> Result<Option<Self>>
    where
        Self: Sized;
    fn finish_merge(&mut self) -> Result<()>;
    fn off_heap_byte_size(&self, field: &FieldInfo) -> BTreeMap<String, i64>;
    fn graph(&self, field: &str) -> Result<Option<Self::Graph<'_>>>;
    fn close(&mut self) -> Result<()>;
}
#[derive(Default)]
struct Counts {
    acquired: AtomicI32,
    finished: AtomicI32,
}
pub struct AssertingKnnVectorsReader<'a, R> {
    delegate: R,
    fields: &'a FieldInfos,
    counts: Arc<Counts>,
    parent: Option<Arc<Counts>>,
    finished: bool,
}
impl<'a, R: KnnVectorsReader> AssertingKnnVectorsReader<'a, R> {
    pub fn new(delegate: R, fields: &'a FieldInfos) -> Self {
        Self {
            delegate,
            fields,
            counts: Arc::default(),
            parent: None,
            finished: false,
        }
    }
    pub fn delegate(&self) -> &R {
        &self.delegate
    }
    pub fn delegate_mut(&mut self) -> &mut R {
        &mut self.delegate
    }
    fn field(&self, name: &str, encoding: VectorEncoding) {
        let info = self
            .fields
            .field_by_name(name)
            .expect("vector field must exist");
        assert!(info.vector_dimension > 0 && info.vector_encoding == encoding);
    }
    fn check_values<T>(values: &impl VectorValues<T>) -> Result<()> {
        assert_eq!(values.iterator()?.doc_id(), -1);
        assert!(values.size() >= 0);
        assert!(values.dimension() > 0);
        Ok(())
    }
    fn assert_ordinary(&self) {
        assert!(
            self.parent.is_none(),
            "This instance should only be used for merging"
        );
    }
}
impl<R: KnnVectorsReader> KnnVectorsReader for AssertingKnnVectorsReader<'_, R> {
    type Floats<'a>
        = R::Floats<'a>
    where
        Self: 'a;
    type Bytes<'a>
        = R::Bytes<'a>
    where
        Self: 'a;
    type Graph<'a>
        = R::Graph<'a>
    where
        Self: 'a;
    fn float_values(&self, field: &str) -> Result<Option<Self::Floats<'_>>> {
        self.field(field, VectorEncoding::Float32);
        let values = self
            .delegate
            .float_values(field)?
            .expect("float vector values must not be null");
        Self::check_values(&values)?;
        Ok(Some(values))
    }
    fn byte_values(&self, field: &str) -> Result<Option<Self::Bytes<'_>>> {
        self.field(field, VectorEncoding::Byte);
        let values = self
            .delegate
            .byte_values(field)?
            .expect("byte vector values must not be null");
        Self::check_values(&values)?;
        Ok(Some(values))
    }
    fn search_float(
        &self,
        field: &str,
        target: &[f32],
        collector: &mut dyn KnnCollect,
        accept: &dyn AcceptDocs,
    ) -> Result<()> {
        self.assert_ordinary();
        self.field(field, VectorEncoding::Float32);
        match AssertingAcceptDocs::wrap(accept) {
            AcceptDocsGuard::Existing(original) => self
                .delegate
                .search_float(field, target, collector, original),
            AcceptDocsGuard::New(wrapper) => self
                .delegate
                .search_float(field, target, collector, &wrapper),
        }
    }
    fn search_byte(
        &self,
        field: &str,
        target: &[u8],
        collector: &mut dyn KnnCollect,
        accept: &dyn AcceptDocs,
    ) -> Result<()> {
        self.assert_ordinary();
        self.field(field, VectorEncoding::Byte);
        match AssertingAcceptDocs::wrap(accept) {
            AcceptDocsGuard::Existing(original) => self
                .delegate
                .search_byte(field, target, collector, original),
            AcceptDocsGuard::New(wrapper) => self
                .delegate
                .search_byte(field, target, collector, &wrapper),
        }
    }
    fn check_integrity(&self) -> Result<()> {
        self.delegate.check_integrity()
    }
    fn get_merge_instance(&self) -> Result<Option<Self>> {
        self.assert_ordinary();
        let delegate = self
            .delegate
            .get_merge_instance()?
            .expect("merge reader must not be null");
        self.counts.acquired.fetch_add(1, Ordering::SeqCst);
        let mut merged = Self::new(delegate, self.fields);
        merged.parent = Some(self.counts.clone());
        Ok(Some(merged))
    }
    fn finish_merge(&mut self) -> Result<()> {
        let parent = self
            .parent
            .as_ref()
            .expect("only a merge instance can finish merge");
        assert!(!self.finished, "Merging already finished");
        self.finished = true;
        self.delegate.finish_merge()?;
        parent.finished.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
    fn off_heap_byte_size(&self, field: &FieldInfo) -> BTreeMap<String, i64> {
        self.delegate.off_heap_byte_size(field)
    }
    fn graph(&self, field: &str) -> Result<Option<Self::Graph<'_>>> {
        self.delegate.graph(field)
    }
    fn close(&mut self) -> Result<()> {
        self.assert_ordinary();
        self.delegate.close()?;
        self.delegate.close()?;
        let finished = self.counts.finished.load(Ordering::SeqCst);
        assert!(finished <= 0 || self.counts.acquired.load(Ordering::SeqCst) == finished);
        Ok(())
    }
}
pub trait DeferredWork {
    fn run(&mut self) -> Result<()>;
}
/// Minimal inherited merge state: abort/integrity first, all fields in phase one,
/// then deferred tasks in order and finish. Mapping belongs to the delegate.
pub struct MergeState<'a> {
    pub fields: &'a FieldInfos,
    pub integrity: &'a [&'a dyn Fn() -> Result<()>],
    pub check_aborted: &'a dyn Fn() -> Result<()>,
}
pub trait KnnVectorsWriter {
    type FieldWriter;
    type Deferred: DeferredWork;
    fn add_field(&mut self, field: &FieldInfo) -> Result<Self::FieldWriter>;
    fn flush(&mut self, max_doc: i32, sort_map: Option<&[i32]>) -> Result<()>;
    fn merge_one_field(
        &mut self,
        field: Option<&FieldInfo>,
        state: Option<&MergeState<'_>>,
    ) -> Result<Option<Self::Deferred>>;
    fn finish(&mut self) -> Result<()>;
    fn close(&mut self) -> Result<()>;
    fn ram_bytes_used(&self) -> usize;
    fn merge(&mut self, state: &MergeState<'_>) -> Result<()> {
        for reader in state.integrity {
            (state.check_aborted)()?;
            reader()?;
        }
        // Java FieldInfos iterates by field number even when its constructor
        // receives the source metadata in a different order.
        let mut fields: Vec<_> = state.fields.fields.iter().collect();
        fields.sort_unstable_by_key(|field| field.number);
        let mut work = Vec::new();
        for field in fields {
            if field.vector_dimension > 0 {
                if let Some(task) = self.merge_one_field(Some(field), Some(state))? {
                    work.push(task);
                }
            }
        }
        for mut task in work {
            task.run()?;
        }
        self.finish()
    }
}
pub struct AssertingKnnVectorsWriter<W> {
    delegate: W,
}
impl<W> AssertingKnnVectorsWriter<W> {
    pub fn new(delegate: W) -> Self {
        Self { delegate }
    }
    pub fn delegate(&self) -> &W {
        &self.delegate
    }
    pub fn delegate_mut(&mut self) -> &mut W {
        &mut self.delegate
    }
}
impl<W: KnnVectorsWriter> KnnVectorsWriter for AssertingKnnVectorsWriter<W> {
    type FieldWriter = W::FieldWriter;
    type Deferred = W::Deferred;
    fn add_field(&mut self, field: &FieldInfo) -> Result<Self::FieldWriter> {
        self.delegate.add_field(field)
    }
    fn flush(&mut self, max_doc: i32, sort_map: Option<&[i32]>) -> Result<()> {
        self.delegate.flush(max_doc, sort_map)
    }
    fn merge_one_field(
        &mut self,
        field: Option<&FieldInfo>,
        state: Option<&MergeState<'_>>,
    ) -> Result<Option<Self::Deferred>> {
        assert!(field.is_some());
        assert!(state.is_some());
        self.delegate.merge_one_field(field, state)
    }
    fn finish(&mut self) -> Result<()> {
        self.delegate.finish()
    }
    fn close(&mut self) -> Result<()> {
        self.delegate.close()
    }
    fn ram_bytes_used(&self) -> usize {
        self.delegate.ram_bytes_used()
    }
}
/// Explicit delegate selection replaces TestUtil's global default codec lookup.
pub trait KnnVectorsFormat {
    type Reader<'a>: KnnVectorsReader
    where
        Self: 'a;
    type Writer: KnnVectorsWriter;
    fn fields_reader<'a>(&'a self, state: ReadState<'a>) -> Result<Self::Reader<'a>>;
    fn fields_writer(&mut self) -> Result<Self::Writer>;
}
pub struct AssertingKnnVectorsFormat<F> {
    delegate: F,
}
impl<F> AssertingKnnVectorsFormat<F> {
    pub const NAME: &str = "Asserting";
    pub const DEFAULT_MAX_DIMENSIONS: usize = 1024;
    pub fn new(delegate: F) -> Self {
        Self { delegate }
    }
    pub fn max_dimensions(&self, _field: &str) -> usize {
        Self::DEFAULT_MAX_DIMENSIONS
    }
    pub fn delegate(&self) -> &F {
        &self.delegate
    }
    pub fn delegate_mut(&mut self) -> &mut F {
        &mut self.delegate
    }
}
impl<F: KnnVectorsFormat> KnnVectorsFormat for AssertingKnnVectorsFormat<F> {
    type Reader<'a>
        = AssertingKnnVectorsReader<'a, F::Reader<'a>>
    where
        Self: 'a;
    type Writer = AssertingKnnVectorsWriter<F::Writer>;
    fn fields_reader<'a>(&'a self, state: ReadState<'a>) -> Result<Self::Reader<'a>> {
        Ok(AssertingKnnVectorsReader::new(
            self.delegate.fields_reader(state)?,
            state.segment.field_infos,
        ))
    }
    fn fields_writer(&mut self) -> Result<Self::Writer> {
        Ok(AssertingKnnVectorsWriter::new(
            self.delegate.fields_writer()?,
        ))
    }
}
#[derive(Clone, Copy)]
pub struct Files<'a> {
    pub vec: &'a [u8],
    pub vemf: &'a [u8],
    pub vex: &'a [u8],
    pub vem: &'a [u8],
}
#[derive(Clone, Copy)]
pub struct ReadState<'a> {
    pub files: Files<'a>,
    pub segment: SegmentReadState<'a>,
}
struct NativeParsed<'a> {
    flat: FlatVectorsReader<'a>,
    graph: HnswVectorsReader<'a>,
}
pub struct Lucene99KnnVectorsReader<'a> {
    state: ReadState<'a>,
    parsed: Arc<NativeParsed<'a>>,
    closed: bool,
}
impl<'a> Lucene99KnnVectorsReader<'a> {
    pub fn open(state: ReadState<'a>) -> Result<Self> {
        let flat = FlatVectorsReader::open(
            state.files.vemf,
            state.files.vec,
            state.segment.segment_id,
            state.segment.segment_suffix,
        )
        .map_err(failure)?;
        let graph = HnswVectorsReader::open(
            state.files.vem,
            state.files.vex,
            state.segment.segment_id,
            state.segment.segment_suffix,
        )
        .map_err(failure)?;
        for field in flat.fields() {
            let info = state
                .segment
                .field_infos
                .field_by_number(field.field_number)
                .ok_or_else(|| failure("unknown vector field"))?;
            if info.vector_dimension != field.dimension
                || info.vector_encoding != field.encoding
                || info.vector_similarity_function != field.similarity
            {
                return Err(failure("inconsistent vector field metadata"));
            }
            let entry = graph
                .field(field.field_number)
                .ok_or_else(|| failure("missing graph field"))?;
            if entry.dimension != field.dimension
                || entry.size != field.size
                || entry.encoding != field.encoding
                || entry.similarity != field.similarity
            {
                return Err(failure("inconsistent graph field metadata"));
            }
        }
        for entry in graph.fields() {
            let info = state
                .segment
                .field_infos
                .field_by_number(entry.field_number)
                .ok_or_else(|| failure("unknown graph vector field"))?;
            if entry.dimension != info.vector_dimension
                || entry.similarity != info.vector_similarity_function
            {
                return Err(failure("inconsistent graph field metadata"));
            }
        }
        Ok(Self {
            state,
            parsed: Arc::new(NativeParsed { flat, graph }),
            closed: false,
        })
    }
    fn open_check(&self) -> Result<()> {
        if self.closed {
            Err(failure("vector reader is closed"))
        } else {
            Ok(())
        }
    }
    fn field_number(&self, name: &str) -> Result<i32> {
        self.state
            .segment
            .field_infos
            .field_by_name(name)
            .filter(|f| f.vector_dimension > 0)
            .map(|f| f.number)
            .ok_or_else(|| failure("unknown vector field"))
    }
    fn filter(&self, accept: &dyn AcceptDocs, docs: &[i32]) -> Result<(Option<FixedBitSet>, i32)> {
        let cost = accept.cost()?;
        let bits = accept.bits()?;
        let ords = if let Some(bits) = bits {
            let mut ords = FixedBitSet::new(docs.len());
            for (ord, &doc) in (0..ords.len()).zip(docs) {
                let doc = usize::try_from(doc).map_err(failure)?;
                if *bits
                    .get(doc)
                    .ok_or_else(|| failure("accept bits too short"))?
                {
                    ords.set(ord);
                }
            }
            Some(ords)
        } else {
            None
        };
        Ok((ords, cost))
    }
}
pub struct NativeFloatValues<'a>(vectors::FloatVectorValues<'a>);
pub struct NativeByteValues<'a>(vectors::ByteVectorValues<'a>);
trait NativeOrdinals {
    fn count(&self) -> i32;
    fn doc(&self, ord: i32) -> std::result::Result<i32, vectors::Error>;
}
impl NativeOrdinals for vectors::FloatVectorValues<'_> {
    fn count(&self) -> i32 {
        self.size()
    }
    fn doc(&self, ord: i32) -> std::result::Result<i32, vectors::Error> {
        self.ord_to_doc(ord)
    }
}
impl NativeOrdinals for vectors::ByteVectorValues<'_> {
    fn count(&self) -> i32 {
        self.size()
    }
    fn doc(&self, ord: i32) -> std::result::Result<i32, vectors::Error> {
        self.ord_to_doc(ord)
    }
}
struct VectorDocs<'a, V> {
    values: &'a V,
    ord: i32,
    doc: i32,
}
impl<V: NativeOrdinals> DocIdSetIterator for VectorDocs<'_, V> {
    fn doc_id(&self) -> i32 {
        self.doc
    }
    fn next_doc(&mut self) -> lucene_search::Result<i32> {
        if self.doc == NO_MORE_DOCS {
            return Ok(self.doc);
        }
        self.ord += 1;
        self.doc = if self.ord < self.values.count() {
            self.values.doc(self.ord)?
        } else {
            NO_MORE_DOCS
        };
        Ok(self.doc)
    }
    fn advance(&mut self, target: i32) -> lucene_search::Result<i32> {
        while self.next_doc()? < target {}
        Ok(self.doc)
    }
    fn cost(&self) -> i64 {
        i64::from(self.values.count())
    }
}
macro_rules! native_values {
    ($name:ident,$component:ty,$value:expr) => {
        impl VectorValues<$component> for $name<'_> {
            fn size(&self) -> i32 {
                self.0.size()
            }
            fn dimension(&self) -> usize {
                self.0.dimension()
            }
            fn iterator(&self) -> Result<Box<dyn DocIdSetIterator + '_>> {
                Ok(Box::new(VectorDocs {
                    values: &self.0,
                    ord: -1,
                    doc: -1,
                }))
            }
            fn ord_to_doc(&self, ord: i32) -> Result<i32> {
                self.0.ord_to_doc(ord).map_err(failure)
            }
            fn vector(&self, ord: i32) -> Result<Vec<$component>> {
                ($value)(self, ord)
            }
        }
    };
}
native_values!(NativeFloatValues, f32, |v: &NativeFloatValues<'_>, ord| v
    .0
    .vector(ord)
    .map_err(failure));
native_values!(NativeByteValues, u8, |v: &NativeByteValues<'_>, ord| v
    .0
    .vector(ord)
    .map(|value| value.to_vec())
    .map_err(failure));
struct Translated<'a> {
    delegate: &'a mut dyn KnnCollect,
    docs: Vec<i32>,
}
impl KnnCollect for Translated<'_> {
    fn k(&self) -> usize {
        self.delegate.k()
    }
    fn early_terminated(&self) -> bool {
        self.delegate.early_terminated()
    }
    fn inc_visited_count(&mut self, count: usize) {
        self.delegate.inc_visited_count(count)
    }
    fn visited_count(&self) -> u64 {
        self.delegate.visited_count()
    }
    fn visit_limit(&self) -> u64 {
        self.delegate.visit_limit()
    }
    fn collect(&mut self, ord: i32, score: f32) -> bool {
        self.delegate.collect(self.docs[ord as usize], score)
    }
    fn min_competitive_similarity(&self) -> f32 {
        self.delegate.min_competitive_similarity()
    }
    fn next_vectors_block(&mut self) {
        self.delegate.next_vectors_block()
    }
}
impl KnnVectorsReader for Lucene99KnnVectorsReader<'_> {
    type Floats<'a>
        = NativeFloatValues<'a>
    where
        Self: 'a;
    type Bytes<'a>
        = NativeByteValues<'a>
    where
        Self: 'a;
    type Graph<'a>
        = OffHeapHnswGraph<'a>
    where
        Self: 'a;
    fn float_values(&self, field: &str) -> Result<Option<Self::Floats<'_>>> {
        self.open_check()?;
        self.parsed
            .flat
            .float_vector_values(self.field_number(field)?)
            .map(|v| Some(NativeFloatValues(v)))
            .map_err(failure)
    }
    fn byte_values(&self, field: &str) -> Result<Option<Self::Bytes<'_>>> {
        self.open_check()?;
        self.parsed
            .flat
            .byte_vector_values(self.field_number(field)?)
            .map(|v| Some(NativeByteValues(v)))
            .map_err(failure)
    }
    fn search_float(
        &self,
        field: &str,
        target: &[f32],
        collector: &mut dyn KnnCollect,
        accept: &dyn AcceptDocs,
    ) -> Result<()> {
        self.open_check()?;
        let field = self.field_number(field)?;
        let values = self
            .parsed
            .flat
            .float_vector_values(field)
            .map_err(failure)?;
        if values.size() == 0 || collector.k() == 0 {
            return Ok(());
        }
        let mut scorer = values.scorer(target).map_err(failure)?;
        let mut docs = Vec::new();
        values.ord_to_docs(&mut docs).map_err(failure)?;
        let (accept_ords, cost) = self.filter(accept, &docs)?;
        let mut collector = Translated {
            delegate: collector,
            docs,
        };
        let graph = self.parsed.graph.graph(field).map_err(failure)?;
        hnsw_vectors::search_with(
            &mut scorer,
            graph.as_ref(),
            &mut collector,
            SearchOptions {
                accept_ords: accept_ords.as_ref(),
                filtered_doc_count: Some(cost),
                ..SearchOptions::default()
            },
        )
        .map_err(failure)
    }
    fn search_byte(
        &self,
        field: &str,
        target: &[u8],
        collector: &mut dyn KnnCollect,
        accept: &dyn AcceptDocs,
    ) -> Result<()> {
        self.open_check()?;
        let field = self.field_number(field)?;
        let values = self
            .parsed
            .flat
            .byte_vector_values(field)
            .map_err(failure)?;
        if values.size() == 0 || collector.k() == 0 {
            return Ok(());
        }
        let mut scorer = values.scorer(target).map_err(failure)?;
        let mut docs = Vec::new();
        values.ord_to_docs(&mut docs).map_err(failure)?;
        let (accept_ords, cost) = self.filter(accept, &docs)?;
        let mut collector = Translated {
            delegate: collector,
            docs,
        };
        let graph = self.parsed.graph.graph(field).map_err(failure)?;
        hnsw_vectors::search_with(
            &mut scorer,
            graph.as_ref(),
            &mut collector,
            SearchOptions {
                accept_ords: accept_ords.as_ref(),
                filtered_doc_count: Some(cost),
                ..SearchOptions::default()
            },
        )
        .map_err(failure)
    }
    fn check_integrity(&self) -> Result<()> {
        self.open_check()?;
        for file in [
            self.state.files.vec,
            self.state.files.vemf,
            self.state.files.vex,
            self.state.files.vem,
        ] {
            codec_util::check_whole_file_footer(file, file.len().saturating_sub(16))
                .map_err(failure)?;
        }
        Ok(())
    }
    fn get_merge_instance(&self) -> Result<Option<Self>> {
        self.open_check()?;
        Ok(Some(Self {
            state: self.state,
            parsed: self.parsed.clone(),
            closed: false,
        }))
    }
    fn finish_merge(&mut self) -> Result<()> {
        self.open_check()
    }
    fn off_heap_byte_size(&self, field: &FieldInfo) -> BTreeMap<String, i64> {
        let mut sizes = BTreeMap::new();
        if let Some(entry) = self.parsed.flat.field(field.number) {
            sizes.insert("vec".into(), entry.vector_data_length);
        }
        if let Some(entry) = self.parsed.graph.field(field.number) {
            sizes.insert("vex".into(), entry.vector_index_length);
        }
        sizes
    }
    fn graph(&self, field: &str) -> Result<Option<Self::Graph<'_>>> {
        self.open_check()?;
        self.parsed
            .graph
            .graph(self.field_number(field)?)
            .map_err(failure)
    }
    fn close(&mut self) -> Result<()> {
        self.closed = true;
        Ok(())
    }
}

#[cfg(test)]
mod tests;
