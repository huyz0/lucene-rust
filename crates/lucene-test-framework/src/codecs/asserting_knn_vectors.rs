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
mod tests {
    use super::*;
    use lucene_codecs::field_infos::VectorSimilarityFunction;
    use lucene_codecs::hnsw::KnnCollector;
    use std::cell::RefCell;
    use std::panic::{catch_unwind, AssertUnwindSafe};
    use std::rc::Rc;

    #[derive(Default)]
    struct State {
        events: Vec<String>,
        fail: String,
        size: i32,
        dimension: usize,
        initial: i32,
        closes: i32,
        missing: bool,
        accept_identity: Option<usize>,
    }
    type Shared = Rc<RefCell<State>>;
    fn shared() -> Shared {
        Rc::new(RefCell::new(State {
            size: 3,
            dimension: 2,
            initial: -1,
            ..State::default()
        }))
    }
    fn event(s: &Shared, name: impl Into<String>) -> Result<()> {
        let name = name.into();
        let mut s = s.borrow_mut();
        s.events.push(name.clone());
        if name == "close" {
            s.closes += 1;
        }
        if s.fail == name || (s.fail == "close2" && name == "close" && s.closes == 2) {
            Err(failure("delegate failure"))
        } else {
            Ok(())
        }
    }
    fn infos() -> FieldInfos {
        FieldInfos::new(vec![
            FieldInfo::new("float", 0).with_vectors(
                2,
                VectorEncoding::Float32,
                VectorSimilarityFunction::Euclidean,
            ),
            FieldInfo::new("byte", 1).with_vectors(
                2,
                VectorEncoding::Byte,
                VectorSimilarityFunction::Euclidean,
            ),
            FieldInfo::new("plain", 2),
        ])
        .unwrap()
    }
    fn outcome(action: impl FnOnce() -> Result<()>) -> String {
        match catch_unwind(AssertUnwindSafe(action)) {
            Ok(Ok(())) => "ok".into(),
            Ok(Err(e)) => format!("io:{e}"),
            Err(_) => "assert".into(),
        }
    }
    struct ProbeIterator {
        state: Option<Shared>,
        doc: i32,
    }
    impl DocIdSetIterator for ProbeIterator {
        fn doc_id(&self) -> i32 {
            if let Some(s) = &self.state {
                s.borrow_mut().events.push("docID".into());
            }
            self.doc
        }
        fn next_doc(&mut self) -> lucene_search::Result<i32> {
            self.doc = if self.doc < 2 {
                self.doc + 1
            } else {
                NO_MORE_DOCS
            };
            Ok(self.doc)
        }
        fn advance(&mut self, target: i32) -> lucene_search::Result<i32> {
            self.doc = target;
            Ok(target)
        }
        fn cost(&self) -> i64 {
            3
        }
    }
    struct ProbeValues<T> {
        state: Shared,
        identity: Rc<()>,
        marker: std::marker::PhantomData<T>,
    }
    impl<T> Clone for ProbeValues<T> {
        fn clone(&self) -> Self {
            Self {
                state: self.state.clone(),
                identity: self.identity.clone(),
                marker: std::marker::PhantomData,
            }
        }
    }
    impl<T> VectorValues<T> for ProbeValues<T> {
        fn size(&self) -> i32 {
            let mut s = self.state.borrow_mut();
            s.events.push("size".into());
            s.size
        }
        fn dimension(&self) -> usize {
            let mut s = self.state.borrow_mut();
            s.events.push("dimension".into());
            s.dimension
        }
        fn iterator(&self) -> Result<Box<dyn DocIdSetIterator + '_>> {
            let mut s = self.state.borrow_mut();
            s.events.push("iterator".into());
            Ok(Box::new(ProbeIterator {
                state: Some(self.state.clone()),
                doc: s.initial,
            }))
        }
        fn ord_to_doc(&self, ord: i32) -> Result<i32> {
            Ok(ord)
        }
        fn vector(&self, _ord: i32) -> Result<Vec<T>> {
            Ok(Vec::new())
        }
    }
    #[derive(Clone, Copy)]
    struct EmptyGraph;
    impl HnswGraphView for EmptyGraph {
        fn size(&self) -> i32 {
            0
        }
        fn num_levels(&self) -> i32 {
            0
        }
        fn entry_node(&self) -> i32 {
            -1
        }
        fn max_conn(&self) -> i32 {
            0
        }
        fn neighbors_into(
            &self,
            _level: i32,
            _node: i32,
            out: &mut Vec<i32>,
        ) -> std::result::Result<(), lucene_codecs::vectors::Error> {
            out.clear();
            Ok(())
        }
        fn sorted_nodes_on_level(
            &self,
            _level: i32,
        ) -> std::result::Result<Vec<i32>, lucene_codecs::vectors::Error> {
            Ok(Vec::new())
        }
    }
    struct ProbeReader {
        state: Shared,
        float: ProbeValues<f32>,
        byte: ProbeValues<u8>,
    }
    impl ProbeReader {
        fn new(state: Shared) -> Self {
            Self {
                float: ProbeValues {
                    state: state.clone(),
                    identity: Rc::new(()),
                    marker: std::marker::PhantomData,
                },
                byte: ProbeValues {
                    state: state.clone(),
                    identity: Rc::new(()),
                    marker: std::marker::PhantomData,
                },
                state,
            }
        }
    }
    impl KnnVectorsReader for ProbeReader {
        type Floats<'a> = ProbeValues<f32>;
        type Bytes<'a> = ProbeValues<u8>;
        type Graph<'a> = EmptyGraph;
        fn float_values(&self, field: &str) -> Result<Option<Self::Floats<'_>>> {
            event(&self.state, format!("float:{field}"))?;
            Ok((!self.state.borrow().missing).then(|| self.float.clone()))
        }
        fn byte_values(&self, field: &str) -> Result<Option<Self::Bytes<'_>>> {
            event(&self.state, format!("byte:{field}"))?;
            Ok((!self.state.borrow().missing).then(|| self.byte.clone()))
        }
        fn search_float(
            &self,
            field: &str,
            _target: &[f32],
            _collector: &mut dyn KnnCollect,
            accept: &dyn AcceptDocs,
        ) -> Result<()> {
            self.search(field, accept)
        }
        fn search_byte(
            &self,
            field: &str,
            _target: &[u8],
            _collector: &mut dyn KnnCollect,
            accept: &dyn AcceptDocs,
        ) -> Result<()> {
            self.search(field, accept)
        }
        fn check_integrity(&self) -> Result<()> {
            event(&self.state, "integrity")
        }
        fn get_merge_instance(&self) -> Result<Option<Self>> {
            event(&self.state, "merge")?;
            Ok((!self.state.borrow().missing).then(|| Self::new(self.state.clone())))
        }
        fn finish_merge(&mut self) -> Result<()> {
            event(&self.state, "finishMerge")
        }
        fn close(&mut self) -> Result<()> {
            event(&self.state, "close")
        }
        fn off_heap_byte_size(&self, field: &FieldInfo) -> BTreeMap<String, i64> {
            self.state
                .borrow_mut()
                .events
                .push(format!("memory:{}", field.name));
            BTreeMap::from([("vec".into(), 17)])
        }
        fn graph(&self, field: &str) -> Result<Option<Self::Graph<'_>>> {
            event(&self.state, format!("graph:{field}"))?;
            Ok(Some(EmptyGraph))
        }
    }
    impl ProbeReader {
        fn search(&self, field: &str, accept: &dyn AcceptDocs) -> Result<()> {
            if let Some(expected) = self.state.borrow().accept_identity {
                assert_eq!(
                    accept as *const dyn AcceptDocs as *const () as usize,
                    expected
                );
            }
            event(&self.state, format!("search:{field}"))?;
            self.state.borrow_mut().events.push(format!(
                "acceptWrapped:{}",
                if accept.is_asserting() {
                    "AssertingAcceptDocs"
                } else {
                    "ProbeAccept"
                }
            ));
            accept.bits()?;
            accept.cost()?;
            accept.iterator()?;
            Ok(())
        }
    }
    struct ProbeAccept(Shared);
    impl AcceptDocs for ProbeAccept {
        fn bits(&self) -> Result<Option<&[bool]>> {
            event(&self.0, "bits")?;
            Ok(None)
        }
        fn cost(&self) -> Result<i32> {
            event(&self.0, "cost")?;
            Ok(3)
        }
        fn iterator(&self) -> Result<Box<dyn DocIdSetIterator + '_>> {
            event(&self.0, "acceptIterator")?;
            Ok(Box::new(ProbeIterator {
                state: None,
                doc: self.0.borrow().initial,
            }))
        }
    }
    fn reader_trace(program: &str) -> (String, String) {
        let fields = infos();
        let state = shared();
        let raw = ProbeReader::new(state.clone());
        let float_identity = raw.float.identity.clone();
        let byte_identity = raw.byte.identity.clone();
        let mut base = AssertingKnnVectorsReader::new(raw, &fields);
        let mut merged = [None, None];
        let accept = ProbeAccept(state.clone());
        let mut results = Vec::new();
        for token in program.split(',') {
            results.push(outcome(|| {
                let p: Vec<_> = token.split(':').collect();
                let field = p.get(1).copied();
                let mut collector = KnnCollector::new(1, 100);
                match p[0] {
                    "f" => assert!(Rc::ptr_eq(
                        &base
                            .float_values(field.unwrap_or("float"))?
                            .unwrap()
                            .identity,
                        &float_identity
                    )),
                    "b" => assert!(Rc::ptr_eq(
                        &base.byte_values(field.unwrap_or("byte"))?.unwrap().identity,
                        &byte_identity
                    )),
                    "sf" => base.search_float(
                        field.unwrap_or("float"),
                        &[0., 1.],
                        &mut collector,
                        &accept,
                    )?,
                    "sb" => {
                        base.search_byte(field.unwrap_or("byte"), &[0, 1], &mut collector, &accept)?
                    }
                    "merge" => merged[0] = base.get_merge_instance()?,
                    "merge2" => merged[1] = base.get_merge_instance()?,
                    "mf" => {
                        merged[0].as_ref().unwrap().float_values("float")?;
                    }
                    "mb" => {
                        merged[0].as_ref().unwrap().byte_values("byte")?;
                    }
                    "msf" => merged[0].as_ref().unwrap().search_float(
                        "float",
                        &[0., 1.],
                        &mut collector,
                        &accept,
                    )?,
                    "msb" => merged[0].as_ref().unwrap().search_byte(
                        "byte",
                        &[0, 1],
                        &mut collector,
                        &accept,
                    )?,
                    "mm" => {
                        merged[0].as_ref().unwrap().get_merge_instance()?;
                    }
                    "mc" => merged[0].as_mut().unwrap().close()?,
                    "finish" => base.finish_merge()?,
                    "mfinish" => merged[0].as_mut().unwrap().finish_merge()?,
                    "finish2" => merged[1].as_mut().unwrap().finish_merge()?,
                    "close" => base.close()?,
                    "integrity" => base.check_integrity()?,
                    "memory" => assert_eq!(
                        base.off_heap_byte_size(&fields.fields[0]),
                        BTreeMap::from([("vec".into(), 17)])
                    ),
                    "graph" => assert!(base.graph("float")?.is_some()),
                    "fail" => state.borrow_mut().fail = token[5..].into(),
                    "ok" => state.borrow_mut().fail.clear(),
                    "size" => state.borrow_mut().size = p[1].parse().unwrap(),
                    "dim" => state.borrow_mut().dimension = p[1].parse().unwrap(),
                    "initial" => state.borrow_mut().initial = p[1].parse().unwrap(),
                    "missing" => state.borrow_mut().missing = true,
                    _ => panic!("unknown command {token}"),
                }
                Ok(())
            }));
        }
        let events = state.borrow().events.join(",");
        (results.join(";"), events)
    }
    #[derive(Clone)]
    struct Task(Shared);
    impl DeferredWork for Task {
        fn run(&mut self) -> Result<()> {
            event(&self.0, "deferred")
        }
    }
    struct ProbeWriter {
        state: Shared,
        handle: Rc<()>,
        task: Task,
    }
    impl KnnVectorsWriter for ProbeWriter {
        type FieldWriter = Rc<()>;
        type Deferred = Task;
        fn add_field(&mut self, field: &FieldInfo) -> Result<Self::FieldWriter> {
            event(&self.state, format!("add:{}", field.name))?;
            Ok(self.handle.clone())
        }
        fn flush(&mut self, max_doc: i32, map: Option<&[i32]>) -> Result<()> {
            event(&self.state, format!("flush:{max_doc}:{}", map.is_none()))
        }
        fn merge_one_field(
            &mut self,
            field: Option<&FieldInfo>,
            _state: Option<&MergeState<'_>>,
        ) -> Result<Option<Self::Deferred>> {
            event(&self.state, format!("mergeOne:{}", field.unwrap().name))?;
            Ok((!self.state.borrow().missing).then(|| self.task.clone()))
        }
        fn finish(&mut self) -> Result<()> {
            event(&self.state, "finish")
        }
        fn close(&mut self) -> Result<()> {
            event(&self.state, "close")
        }
        fn ram_bytes_used(&self) -> usize {
            self.state.borrow_mut().events.push("ram".into());
            17
        }
    }
    fn writer_trace(name: &str) -> (String, String) {
        if name.starts_with("merge-reversed") {
            return reversed_merge_trace(name);
        }
        let state = shared();
        let fields = infos();
        let handle = Rc::new(());
        let task = Task(state.clone());
        let mut writer = AssertingKnnVectorsWriter::new(ProbeWriter {
            state: state.clone(),
            handle: handle.clone(),
            task: task.clone(),
        });
        let merge = MergeState {
            fields: &fields,
            integrity: &[],
            check_aborted: &|| Ok(()),
        };
        let status = outcome(|| {
            match name {
                "all" => {
                    assert!(Rc::ptr_eq(&writer.add_field(&fields.fields[0])?, &handle));
                    writer.flush(7, None)?;
                    let mut deferred = writer
                        .merge_one_field(Some(&fields.fields[0]), Some(&merge))?
                        .unwrap();
                    assert!(Rc::ptr_eq(&deferred.0, &task.0));
                    deferred.run()?;
                    writer.finish()?;
                    assert_eq!(writer.ram_bytes_used(), 17);
                    writer.close()?;
                }
                "null-field" => {
                    writer.merge_one_field(None, Some(&merge))?;
                }
                "null-state" => {
                    writer.merge_one_field(Some(&fields.fields[0]), None)?;
                }
                "no-task" => {
                    state.borrow_mut().missing = true;
                    assert!(writer
                        .merge_one_field(Some(&fields.fields[0]), Some(&merge))?
                        .is_none());
                }
                "deferred-error" => {
                    state.borrow_mut().fail = "deferred".into();
                    writer
                        .merge_one_field(Some(&fields.fields[0]), Some(&merge))?
                        .unwrap()
                        .run()?;
                }
                "merge-order" => writer.merge(&merge)?,
                _ => {
                    state.borrow_mut().fail = name.into();
                    match name {
                        "add:float" => {
                            writer.add_field(&fields.fields[0])?;
                        }
                        "flush:7:true" => writer.flush(7, None)?,
                        "mergeOne:float" => {
                            writer.merge_one_field(Some(&fields.fields[0]), Some(&merge))?;
                        }
                        "finish" => writer.finish()?,
                        "close" => writer.close()?,
                        _ => panic!("unknown case"),
                    }
                }
            }
            Ok(())
        });
        let events = state.borrow().events.join(",");
        (status, events)
    }
    fn accept_trace(name: &str) -> (String, String) {
        let state = shared();
        let raw = ProbeAccept(state.clone());
        let wrapped = AssertingAcceptDocs::wrap(&raw);
        let status = outcome(|| {
            match name {
                "all" => {
                    let twice = AssertingAcceptDocs::wrap(&wrapped);
                    assert!(
                        matches!(twice,AcceptDocsGuard::Existing(v) if std::ptr::eq(v,&wrapped as &dyn AcceptDocs))
                    );
                    wrapped.bits()?;
                    wrapped.cost()?;
                    wrapped.iterator()?;
                }
                "positioned" => {
                    state.borrow_mut().initial = 0;
                    wrapped.iterator()?;
                }
                _ => {
                    state.borrow_mut().fail = name.into();
                    match name {
                        "bits" => {
                            wrapped.bits()?;
                        }
                        "cost" => {
                            wrapped.cost()?;
                        }
                        "acceptIterator" => {
                            wrapped.iterator()?;
                        }
                        _ => panic!("unknown case"),
                    }
                }
            }
            Ok(())
        });
        let events = state.borrow().events.join(",");
        (status, events)
    }
    #[test]
    fn actual_java_assertions_identity_callback_order_and_lifecycle() {
        let fixture =
            include_str!("../../../../fixtures/testframework/data/asserting_knn_vectors.tsv");
        let mut checked = 0;
        for row in fixture.lines() {
            let p: Vec<_> = row.split('\t').collect();
            if p[0] == "accept" && p[1] == "thread" {
                continue;
            }
            let actual = match p[0] {
                "reader" => reader_trace(p[1]),
                "writer" => writer_trace(p[1]),
                "accept" => accept_trace(p[1]),
                _ => panic!("unknown fixture"),
            };
            assert_eq!(
                actual,
                (p[2].into(), p.get(3).unwrap_or(&"").to_string()),
                "{row}"
            );
            checked += 1;
        }
        assert_eq!(checked, 42);
    }
    #[test]
    fn already_asserting_accept_docs_reach_both_delegate_searches_by_identity() {
        let state = shared();
        let fields = infos();
        let raw = ProbeAccept(state.clone());
        let accept = AssertingAcceptDocs::new(&raw);
        state.borrow_mut().accept_identity = Some(&accept as *const _ as usize);
        let reader = AssertingKnnVectorsReader::new(ProbeReader::new(state), &fields);
        let mut collector = KnnCollector::new(1, 100);
        reader
            .search_float("float", &[0., 1.], &mut collector, &accept)
            .unwrap();
        reader
            .search_byte("byte", &[0, 1], &mut collector, &accept)
            .unwrap();
    }
    struct OrderedTask(Shared, String);
    impl DeferredWork for OrderedTask {
        fn run(&mut self) -> Result<()> {
            event(&self.0, format!("deferred:{}", self.1))
        }
    }
    struct OrderedWriter(Shared);
    impl KnnVectorsWriter for OrderedWriter {
        type FieldWriter = ();
        type Deferred = OrderedTask;
        fn add_field(&mut self, _: &FieldInfo) -> Result<()> {
            unreachable!()
        }
        fn flush(&mut self, _: i32, _: Option<&[i32]>) -> Result<()> {
            unreachable!()
        }
        fn merge_one_field(
            &mut self,
            field: Option<&FieldInfo>,
            _: Option<&MergeState<'_>>,
        ) -> Result<Option<OrderedTask>> {
            let field = field.unwrap();
            event(&self.0, format!("mergeOne:{}", field.name))?;
            Ok(Some(OrderedTask(self.0.clone(), field.name.clone())))
        }
        fn finish(&mut self) -> Result<()> {
            event(&self.0, "finish")
        }
        fn close(&mut self) -> Result<()> {
            unreachable!()
        }
        fn ram_bytes_used(&self) -> usize {
            unreachable!()
        }
    }
    fn reversed_merge_trace(name: &str) -> (String, String) {
        let state = shared();
        state.borrow_mut().fail = match name {
            "merge-reversed-field-error" => "mergeOne:byte",
            "merge-reversed-deferred-error" => "deferred:float",
            _ => "",
        }
        .into();
        let mut fields = infos().fields;
        fields.reverse();
        let fields = FieldInfos::new(fields).unwrap();
        assert_eq!(fields.fields[0].name, "plain");
        let merge = MergeState {
            fields: &fields,
            integrity: &[],
            check_aborted: &|| Ok(()),
        };
        let mut writer = AssertingKnnVectorsWriter::new(OrderedWriter(state.clone()));
        let status = outcome(|| writer.merge(&merge));
        let events = state.borrow().events.join(",");
        (status, events)
    }
    #[test]
    fn accept_docs_cross_thread_checks_precede_callbacks() {
        let raw = BitsAcceptDocs::new(None, 3);
        let wrapped = AssertingAcceptDocs::new(&raw);
        for op in 0..3 {
            thread::scope(|scope| {
                let wrapped = &wrapped;
                assert!(scope
                    .spawn(move || catch_unwind(AssertUnwindSafe(|| match op {
                        0 => {
                            wrapped.bits().unwrap();
                        }
                        1 => {
                            wrapped.cost().unwrap();
                        }
                        _ => {
                            wrapped.iterator().unwrap();
                        }
                    }))
                    .is_err())
                    .join()
                    .unwrap());
            });
        }
    }
    #[test]
    fn bits_iterators_filter_advance_and_exhaustion() {
        let bits = [false, true, false, true];
        let accept = BitsAcceptDocs::new(Some(&bits), 4);
        assert_eq!(accept.bits().unwrap(), Some(bits.as_slice()));
        assert_eq!(accept.cost().unwrap(), 4);
        let mut iterator = accept.iterator().unwrap();
        assert_eq!(iterator.doc_id(), -1);
        assert_eq!(iterator.cost(), 4);
        assert_eq!(iterator.next_doc().unwrap(), 1);
        assert_eq!(iterator.advance(3).unwrap(), 3);
        assert_eq!(iterator.next_doc().unwrap(), NO_MORE_DOCS);
        assert_eq!(iterator.next_doc().unwrap(), NO_MORE_DOCS);
        assert!(catch_unwind(|| BitsAcceptDocs::new(None, -1)).is_err());
        assert!(catch_unwind(|| BitsAcceptDocs::new(Some(&bits), 3)).is_err());
        let all = BitsAcceptDocs::new(None, 2);
        assert!(all.bits().unwrap().is_none());
        let mut it = all.iterator().unwrap();
        assert_eq!(it.advance(1).unwrap(), 1);
        assert_eq!(it.next_doc().unwrap(), NO_MORE_DOCS);
    }
    #[test]
    fn inherited_merge_aborts_and_integrity_errors_before_any_field_callback() {
        let fields = infos();
        let state = shared();
        let mut w = AssertingKnnVectorsWriter::new(ProbeWriter {
            state: state.clone(),
            handle: Rc::new(()),
            task: Task(state.clone()),
        });
        let integrity = || event(&state, "integrity");
        let abort = || event(&state, "abort");
        let sources: [&dyn Fn() -> Result<()>; 1] = [&integrity];
        let merge = MergeState {
            fields: &fields,
            integrity: &sources,
            check_aborted: &abort,
        };
        state.borrow_mut().fail = "abort".into();
        assert!(w.merge(&merge).is_err());
        assert_eq!(state.borrow().events, ["abort"]);
        state.borrow_mut().events.clear();
        state.borrow_mut().fail = "integrity".into();
        assert!(w.merge(&merge).is_err());
        assert_eq!(state.borrow().events, ["abort", "integrity"]);
        state.borrow_mut().events.clear();
        state.borrow_mut().fail = "mergeOne:float".into();
        assert!(w.merge(&merge).is_err());
        assert_eq!(
            state.borrow().events,
            ["abort", "integrity", "mergeOne:float"]
        );
        assert!(Rc::ptr_eq(&w.delegate().state, &state));
        assert!(Rc::ptr_eq(&w.delegate_mut().state, &state));
    }
    struct ProbeFormat {
        state: Shared,
    }
    impl KnnVectorsFormat for ProbeFormat {
        type Reader<'a> = ProbeReader;
        type Writer = ProbeWriter;
        fn fields_reader<'a>(&'a self, _state: ReadState<'a>) -> Result<Self::Reader<'a>> {
            event(&self.state, "reader-open")?;
            Ok(ProbeReader::new(self.state.clone()))
        }
        fn fields_writer(&mut self) -> Result<Self::Writer> {
            event(&self.state, "writer-open")?;
            Ok(ProbeWriter {
                state: self.state.clone(),
                handle: Rc::new(()),
                task: Task(self.state.clone()),
            })
        }
    }
    #[test]
    fn format_factory_forwarding_constants_and_open_failures() {
        let state = shared();
        let mut format = AssertingKnnVectorsFormat::new(ProbeFormat {
            state: state.clone(),
        });
        let fields = infos();
        let id = [0; 16];
        let read = ReadState {
            files: Files {
                vec: &[],
                vemf: &[],
                vex: &[],
                vem: &[],
            },
            segment: SegmentReadState {
                max_doc: 3,
                segment_id: &id,
                segment_suffix: "",
                field_infos: &fields,
            },
        };
        assert_eq!(AssertingKnnVectorsFormat::<ProbeFormat>::NAME, "Asserting");
        assert_eq!(format.max_dimensions("float"), 1024);
        assert!(Rc::ptr_eq(&format.delegate().state, &state));
        assert!(Rc::ptr_eq(&format.delegate_mut().state, &state));
        let reader = format.fields_reader(read).unwrap();
        assert!(reader.float_values("float").unwrap().is_some());
        assert!(Rc::ptr_eq(&reader.delegate().state, &state));
        drop(reader);
        let mut writer = format.fields_writer().unwrap();
        writer.flush(3, Some(&[2, 1, 0])).unwrap();
        assert!(Rc::ptr_eq(&writer.delegate_mut().state, &state));
        state.borrow_mut().fail = "reader-open".into();
        assert!(format.fields_reader(read).is_err());
        state.borrow_mut().fail = "writer-open".into();
        assert!(format.fields_writer().is_err());
        let float = ProbeValues::<f32> {
            state: state.clone(),
            identity: Rc::new(()),
            marker: std::marker::PhantomData,
        };
        assert_eq!(float.ord_to_doc(2).unwrap(), 2);
        assert!(float.vector(0).unwrap().is_empty());
        let mut it = float.iterator().unwrap();
        assert_eq!(it.next_doc().unwrap(), 0);
        assert_eq!(it.advance(2).unwrap(), 2);
        assert_eq!(it.cost(), 3);
        assert_eq!(EmptyGraph.size(), 0);
        assert_eq!(EmptyGraph.num_levels(), 0);
        assert_eq!(EmptyGraph.entry_node(), -1);
        assert_eq!(EmptyGraph.max_conn(), 0);
        let mut neighbors = vec![1];
        EmptyGraph.neighbors_into(0, 0, &mut neighbors).unwrap();
        assert!(neighbors.is_empty());
        assert!(EmptyGraph.sorted_nodes_on_level(0).unwrap().is_empty());
    }
}
