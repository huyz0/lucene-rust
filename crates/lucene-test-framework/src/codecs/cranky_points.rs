//! Java's cranky points format: one shared RNG wraps the writer, reader,
//! values and original tree. Opens, values acquisition, tree acquisition and
//! navigation do not draw. Metadata getters draw before delegation; integrity,
//! finish, merge and visits draw before and after successful delegation; close
//! draws afterwards. A tree clone is deliberately the unwrapped delegate clone.
//!
//! The native Lucene90 adapter uses explicit state and borrowed file bytes in
//! place of Codec routing, IOContext and reader-owned IndexInputs. Its tree
//! cursor keeps native value nodes and ancestor frames rather than Java's
//! mutable per-level arrays. Native visitors use a local fallible callback
//! seam; the production codec's callbacks are infallible. Immutable borrows
//! prevent closing a native reader while its values or trees remain borrowed.
//! `Rc<RefCell<JavaRandom>>` is single-threaded, unlike Java Random.
//! The writer retains owned metadata/index/data buffers and publishes their
//! deltas after each field and finish; Java streams directly into its outputs.
//! Native output write failures follow `FsIndexOutput`'s sticky-error model:
//! the first failure is retained and returned by close.
//! Native one-dimensional merges retain one leaf's doc/value buffers per
//! source. Their infallible iterator interface carries failures in a shared
//! first-error slot, checked before the field's output can be published.
//! Java's merge configuration assertions are returned as errors here, so
//! mismatched native dimensions or widths cannot silently truncate values
//! when assertions are disabled.

use std::cell::RefCell;
use std::rc::Rc;

use lucene_codecs::field_infos::{FieldInfo, FieldInfos};
use lucene_codecs::points;
use lucene_util::java_random::JavaRandom;

/// The exact message used by every Java injection site.
pub const FAKE_ERROR: &str = "Fake IOException";
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Store(#[from] lucene_store::Error),
    #[error(transparent)]
    Points(#[from] points::Error),
    #[error(transparent)]
    Bkd(#[from] lucene_codecs::bkd_writer::BkdError),
    #[error("{0}")]
    Message(String),
}
pub type Result<T> = std::result::Result<T, Error>;
fn failure(message: impl Into<String>) -> Error {
    Error::Message(message.into())
}
fn guard(random: &RefCell<JavaRandom>) -> Result<()> {
    if random.borrow_mut().next_int_bounded(100) == 0 {
        Err(failure(FAKE_ERROR))
    } else {
        Ok(())
    }
}

/// Fallible point callbacks, matching Java's checked visitor exceptions.
pub trait IntersectVisitor {
    fn visit(&mut self, doc: i32) -> Result<()>;
    fn visit_value(&mut self, doc: i32, value: &[u8]) -> Result<()>;
    fn grow(&mut self, _count: usize) {}
    fn compare(&mut self, min: &[u8], max: &[u8]) -> points::Relation;
}
pub trait PointTree {
    type CloneTree: PointTree;
    /// Java: `PointTree.clone`; the current node becomes the clone's root.
    fn clone_tree(&self) -> Self::CloneTree;
    fn move_to_child(&mut self) -> Result<bool>;
    fn move_to_sibling(&mut self) -> Result<bool>;
    fn move_to_parent(&mut self) -> Result<bool>;
    fn min_packed_value(&self) -> &[u8];
    fn max_packed_value(&self) -> &[u8];
    fn size(&self) -> i64;
    fn visit_doc_ids<V: IntersectVisitor>(&mut self, visitor: &mut V) -> Result<()>;
    fn visit_doc_values<V: IntersectVisitor>(&mut self, visitor: &mut V) -> Result<()>;
}
pub trait PointValues {
    type Tree<'a>: PointTree
    where
        Self: 'a;
    fn point_tree(&self) -> Result<Self::Tree<'_>>;
    fn min_packed_value(&self) -> Result<&[u8]>;
    fn max_packed_value(&self) -> Result<&[u8]>;
    fn num_dimensions(&self) -> Result<i32>;
    fn num_index_dimensions(&self) -> Result<i32>;
    fn bytes_per_dimension(&self) -> Result<i32>;
    fn size(&self) -> i64;
    fn doc_count(&self) -> i32;
}
pub trait PointsReader {
    type Values<'a>: PointValues
    where
        Self: 'a;
    fn values(&self, field: &str) -> Result<Option<Self::Values<'_>>>;
    fn check_integrity(&self) -> Result<()>;
    fn close(&mut self) -> Result<()>;
    /// Only the concrete native reader supplies this, like Java's instanceof
    /// test before Lucene90's optimized merge; wrappers retain the default.
    fn native(&self) -> Option<&points::PointsReader<'_>> {
        None
    }
}
pub struct MergeSource<'a, R> {
    pub reader: Option<&'a R>,
    pub fields: &'a FieldInfos,
    /// Java DocMap: -1 means deleted; other entries are destination IDs.
    pub doc_map: &'a [i32],
}
pub struct MergeState<'a, R> {
    pub sources: &'a [MergeSource<'a, R>],
    pub fields: &'a FieldInfos,
    pub check_aborted: &'a dyn Fn() -> Result<()>,
}
pub trait PointsWriter {
    fn write_field<R: PointsReader>(&mut self, field: &FieldInfo, reader: &R) -> Result<()>;
    fn finish(&mut self) -> Result<()>;
    fn merge<R: PointsReader>(&mut self, state: &MergeState<'_, R>) -> Result<()>;
    fn close(&mut self) -> Result<()>;
}
pub trait PointsFormat {
    type Writer<'a>: PointsWriter;
    type Reader<'a>: PointsReader;
    fn fields_writer<'a>(&mut self, state: WriteState<'a>) -> Result<Self::Writer<'a>>;
    fn fields_reader<'a>(&mut self, state: ReadState<'a>) -> Result<Self::Reader<'a>>;
}
#[derive(Clone, Copy)]
pub struct WriteState<'a> {
    pub directory: &'a dyn lucene_store::Directory,
    pub segment: &'a str,
    pub suffix: &'a str,
    pub id: &'a [u8; 16],
    pub max_doc: i32,
    pub max_points_in_leaf_node: usize,
}
#[derive(Clone, Copy)]
pub struct ReadState<'a> {
    pub meta: &'a [u8],
    pub index: &'a [u8],
    pub data: &'a [u8],
    pub fields: &'a FieldInfos,
    pub suffix: &'a str,
    pub id: &'a [u8; 16],
}

pub struct CrankyPointsFormat<F> {
    delegate: F,
    random: Rc<RefCell<JavaRandom>>,
}
impl<F> CrankyPointsFormat<F> {
    pub fn new(delegate: F, random: Rc<RefCell<JavaRandom>>) -> Self {
        Self { delegate, random }
    }
    pub fn delegate(&self) -> &F {
        &self.delegate
    }
    pub fn delegate_mut(&mut self) -> &mut F {
        &mut self.delegate
    }
}
impl<F: PointsFormat> PointsFormat for CrankyPointsFormat<F> {
    type Writer<'a> = CrankyPointsWriter<F::Writer<'a>>;
    type Reader<'a> = CrankyPointsReader<F::Reader<'a>>;
    fn fields_writer<'a>(&mut self, state: WriteState<'a>) -> Result<Self::Writer<'a>> {
        Ok(CrankyPointsWriter::new(
            self.delegate.fields_writer(state)?,
            self.random.clone(),
        ))
    }
    fn fields_reader<'a>(&mut self, state: ReadState<'a>) -> Result<Self::Reader<'a>> {
        Ok(CrankyPointsReader::new(
            self.delegate.fields_reader(state)?,
            self.random.clone(),
        ))
    }
}
pub struct CrankyPointsWriter<W> {
    delegate: W,
    random: Rc<RefCell<JavaRandom>>,
}
impl<W> CrankyPointsWriter<W> {
    pub fn new(delegate: W, random: Rc<RefCell<JavaRandom>>) -> Self {
        Self { delegate, random }
    }
    pub fn delegate(&self) -> &W {
        &self.delegate
    }
    pub fn delegate_mut(&mut self) -> &mut W {
        &mut self.delegate
    }
}
impl<W: PointsWriter> PointsWriter for CrankyPointsWriter<W> {
    fn write_field<R: PointsReader>(&mut self, field: &FieldInfo, reader: &R) -> Result<()> {
        guard(&self.random)?;
        self.delegate.write_field(field, reader)
    }
    fn finish(&mut self) -> Result<()> {
        guard(&self.random)?;
        self.delegate.finish()?;
        guard(&self.random)
    }
    fn merge<R: PointsReader>(&mut self, state: &MergeState<'_, R>) -> Result<()> {
        guard(&self.random)?;
        self.delegate.merge(state)?;
        guard(&self.random)
    }
    fn close(&mut self) -> Result<()> {
        self.delegate.close()?;
        guard(&self.random)
    }
}
pub struct CrankyPointsReader<R> {
    delegate: R,
    random: Rc<RefCell<JavaRandom>>,
}
impl<R> CrankyPointsReader<R> {
    pub fn new(delegate: R, random: Rc<RefCell<JavaRandom>>) -> Self {
        Self { delegate, random }
    }
    pub fn delegate(&self) -> &R {
        &self.delegate
    }
    pub fn delegate_mut(&mut self) -> &mut R {
        &mut self.delegate
    }
}
impl<R: PointsReader> PointsReader for CrankyPointsReader<R> {
    type Values<'a>
        = CrankyPointValues<R::Values<'a>>
    where
        Self: 'a;
    fn values(&self, field: &str) -> Result<Option<Self::Values<'_>>> {
        Ok(self
            .delegate
            .values(field)?
            .map(|delegate| CrankyPointValues {
                delegate,
                random: self.random.clone(),
            }))
    }
    fn check_integrity(&self) -> Result<()> {
        guard(&self.random)?;
        self.delegate.check_integrity()?;
        guard(&self.random)
    }
    fn close(&mut self) -> Result<()> {
        self.delegate.close()?;
        guard(&self.random)
    }
}
pub struct CrankyPointValues<V> {
    delegate: V,
    random: Rc<RefCell<JavaRandom>>,
}
impl<V: PointValues> PointValues for CrankyPointValues<V> {
    type Tree<'a>
        = CrankyPointTree<V::Tree<'a>>
    where
        Self: 'a;
    fn point_tree(&self) -> Result<Self::Tree<'_>> {
        Ok(CrankyPointTree {
            delegate: self.delegate.point_tree()?,
            random: self.random.clone(),
        })
    }
    fn min_packed_value(&self) -> Result<&[u8]> {
        guard(&self.random)?;
        self.delegate.min_packed_value()
    }
    fn max_packed_value(&self) -> Result<&[u8]> {
        guard(&self.random)?;
        self.delegate.max_packed_value()
    }
    fn num_dimensions(&self) -> Result<i32> {
        guard(&self.random)?;
        self.delegate.num_dimensions()
    }
    fn num_index_dimensions(&self) -> Result<i32> {
        guard(&self.random)?;
        self.delegate.num_index_dimensions()
    }
    fn bytes_per_dimension(&self) -> Result<i32> {
        guard(&self.random)?;
        self.delegate.bytes_per_dimension()
    }
    fn size(&self) -> i64 {
        self.delegate.size()
    }
    fn doc_count(&self) -> i32 {
        self.delegate.doc_count()
    }
}
pub struct CrankyPointTree<T> {
    delegate: T,
    random: Rc<RefCell<JavaRandom>>,
}
impl<T: PointTree> PointTree for CrankyPointTree<T> {
    type CloneTree = T::CloneTree;
    fn clone_tree(&self) -> Self::CloneTree {
        self.delegate.clone_tree()
    }
    fn move_to_child(&mut self) -> Result<bool> {
        self.delegate.move_to_child()
    }
    fn move_to_sibling(&mut self) -> Result<bool> {
        self.delegate.move_to_sibling()
    }
    fn move_to_parent(&mut self) -> Result<bool> {
        self.delegate.move_to_parent()
    }
    fn min_packed_value(&self) -> &[u8] {
        self.delegate.min_packed_value()
    }
    fn max_packed_value(&self) -> &[u8] {
        self.delegate.max_packed_value()
    }
    fn size(&self) -> i64 {
        self.delegate.size()
    }
    fn visit_doc_ids<V: IntersectVisitor>(&mut self, visitor: &mut V) -> Result<()> {
        guard(&self.random)?;
        self.delegate.visit_doc_ids(visitor)?;
        guard(&self.random)
    }
    fn visit_doc_values<V: IntersectVisitor>(&mut self, visitor: &mut V) -> Result<()> {
        guard(&self.random)?;
        self.delegate.visit_doc_values(visitor)?;
        guard(&self.random)
    }
}

#[derive(Clone, Copy, Default)]
pub struct Lucene90PointsFormat;
pub struct Lucene90PointsReader<'a> {
    reader: Option<points::PointsReader<'a>>,
    index: &'a [u8],
    data: &'a [u8],
    fields: &'a FieldInfos,
}
pub struct NativeValues<'r, 'd> {
    reader: &'r points::PointsReader<'d>,
    number: i32,
}
pub struct NativeTree<'r, 'd> {
    reader: &'r points::PointsReader<'d>,
    node: points::PointTreeNode,
    size: i64,
    parents: Vec<(points::PointTreeNode, Option<points::PointTreeNode>)>,
}
impl PointsReader for Lucene90PointsReader<'_> {
    type Values<'a>
        = NativeValues<'a, 'a>
    where
        Self: 'a;
    fn values(&self, name: &str) -> Result<Option<Self::Values<'_>>> {
        let reader = self
            .reader
            .as_ref()
            .ok_or_else(|| failure("closed points reader"))?;
        let field = self
            .fields
            .field_by_name(name)
            .ok_or_else(|| failure(format!("field=\"{name}\" is unrecognized")))?;
        if field.point_dimension_count == 0 {
            return Err(failure(format!(
                "field=\"{name}\" did not index point values"
            )));
        }
        Ok(reader.field(field.number).map(|_| NativeValues {
            reader,
            number: field.number,
        }))
    }
    fn check_integrity(&self) -> Result<()> {
        if self.reader.is_none() {
            return Err(failure("closed points reader"));
        }
        for bytes in [self.index, self.data] {
            let end = bytes
                .len()
                .checked_sub(lucene_store::codec_util::FOOTER_LENGTH)
                .ok_or_else(|| failure("missing points footer"))?;
            lucene_store::codec_util::check_whole_file_footer(bytes, end)?;
        }
        Ok(())
    }
    fn close(&mut self) -> Result<()> {
        self.reader = None;
        Ok(())
    }
    fn native(&self) -> Option<&points::PointsReader<'_>> {
        self.reader.as_ref()
    }
}
impl NativeValues<'_, '_> {
    fn field(&self) -> &points::PointsField {
        self.reader
            .field(self.number)
            .expect("values acquired for existing field")
    }
}
impl PointValues for NativeValues<'_, '_> {
    type Tree<'a>
        = NativeTree<'a, 'a>
    where
        Self: 'a;
    fn point_tree(&self) -> Result<Self::Tree<'_>> {
        let node = self.reader.point_tree(self.number)?;
        let size = self.reader.point_tree_size(&node)?;
        Ok(NativeTree {
            reader: self.reader,
            node,
            size,
            parents: Vec::new(),
        })
    }
    fn min_packed_value(&self) -> Result<&[u8]> {
        Ok(&self.field().min_packed_value)
    }
    fn max_packed_value(&self) -> Result<&[u8]> {
        Ok(&self.field().max_packed_value)
    }
    fn num_dimensions(&self) -> Result<i32> {
        Ok(self.field().num_dims)
    }
    fn num_index_dimensions(&self) -> Result<i32> {
        Ok(self.field().num_index_dims)
    }
    fn bytes_per_dimension(&self) -> Result<i32> {
        Ok(self.field().bytes_per_dim)
    }
    fn size(&self) -> i64 {
        self.field().point_count
    }
    fn doc_count(&self) -> i32 {
        self.field().doc_count
    }
}
struct VisitorBridge<'a, V> {
    visitor: &'a mut V,
    error: Option<Error>,
}
impl<V: IntersectVisitor> points::IntersectVisitor for VisitorBridge<'_, V> {
    fn compare(&mut self, min: &[u8], max: &[u8]) -> points::Relation {
        if self.error.is_some() {
            points::Relation::CellOutsideQuery
        } else {
            self.visitor.compare(min, max)
        }
    }
    fn visit(&mut self, doc: i32) {
        if self.error.is_none() {
            self.error = self.visitor.visit(doc).err();
        }
    }
    fn visit_with_value(&mut self, doc: i32, value: &[u8]) {
        if self.error.is_none() {
            self.error = self.visitor.visit_value(doc, value).err();
        }
    }
    fn grow(&mut self, count: usize) {
        if self.error.is_none() {
            self.visitor.grow(count);
        }
    }
}
impl<'r, 'd> PointTree for NativeTree<'r, 'd> {
    type CloneTree = Self;
    fn clone_tree(&self) -> Self {
        Self {
            reader: self.reader,
            node: self.node.clone(),
            size: self.size,
            parents: Vec::new(),
        }
    }
    fn move_to_child(&mut self) -> Result<bool> {
        let Some((left, right)) = self.reader.point_tree_children(&self.node)? else {
            return Ok(false);
        };
        let size = self.reader.point_tree_size(&left)?;
        let parent = std::mem::replace(&mut self.node, left);
        self.parents.push((parent, Some(right)));
        self.size = size;
        Ok(true)
    }
    fn move_to_sibling(&mut self) -> Result<bool> {
        let Some((_, right)) = self.parents.last_mut() else {
            return Ok(false);
        };
        let Some(node) = right.as_ref() else {
            return Ok(false);
        };
        let size = self.reader.point_tree_size(node)?;
        self.node = right.take().expect("sibling present");
        self.size = size;
        Ok(true)
    }
    fn move_to_parent(&mut self) -> Result<bool> {
        let Some((node, _)) = self.parents.last() else {
            return Ok(false);
        };
        let size = self.reader.point_tree_size(node)?;
        self.node = self.parents.pop().expect("parent present").0;
        self.size = size;
        Ok(true)
    }
    fn min_packed_value(&self) -> &[u8] {
        self.node.min_packed()
    }
    fn max_packed_value(&self) -> &[u8] {
        self.node.max_packed()
    }
    fn size(&self) -> i64 {
        self.size
    }
    fn visit_doc_ids<V: IntersectVisitor>(&mut self, visitor: &mut V) -> Result<()> {
        let mut bridge = VisitorBridge {
            visitor,
            error: None,
        };
        let result = self.reader.visit_doc_ids(&self.node, &mut bridge);
        if let Some(error) = bridge.error {
            return Err(error);
        };
        Ok(result?)
    }
    fn visit_doc_values<V: IntersectVisitor>(&mut self, visitor: &mut V) -> Result<()> {
        let mut pending = vec![self.node.clone()];
        while let Some(node) = pending.pop() {
            if let Some((left, right)) = self.reader.point_tree_children(&node)? {
                pending.push(right);
                pending.push(left);
            } else {
                let mut bridge = VisitorBridge {
                    visitor,
                    error: None,
                };
                let result = self.reader.visit_leaf(&node, &mut bridge);
                // Production callbacks cannot abort decoding. Suppress every
                // later callback in this leaf and preserve the first visitor
                // failure over any later decode error; no later leaf is read.
                if let Some(error) = bridge.error {
                    return Err(error);
                };
                result?;
            }
        }
        Ok(())
    }
}

pub struct Lucene90PointsWriter<'a> {
    state: WriteState<'a>,
    meta: Vec<u8>,
    index: Vec<u8>,
    data: Vec<u8>,
    outputs: Option<[lucene_store::FsIndexOutput; 3]>,
    published: [usize; 3],
    finished: bool,
}
fn filename(state: WriteState<'_>, extension: &str) -> String {
    if state.suffix.is_empty() {
        format!("{}.{extension}", state.segment)
    } else {
        format!("{}_{}.{extension}", state.segment, state.suffix)
    }
}
impl PointsFormat for Lucene90PointsFormat {
    type Writer<'a> = Lucene90PointsWriter<'a>;
    type Reader<'a> = Lucene90PointsReader<'a>;
    fn fields_writer<'a>(&mut self, state: WriteState<'a>) -> Result<Self::Writer<'a>> {
        use lucene_store::DataOutput;
        let mut opened: Vec<(lucene_store::FsIndexOutput, Vec<u8>)> = Vec::new();
        for (extension, codec) in [
            ("kdd", "Lucene90PointsFormatData"),
            ("kdm", "Lucene90PointsFormatMeta"),
            ("kdi", "Lucene90PointsFormatIndex"),
        ] {
            let mut output = match state.directory.create_output(&filename(state, extension)) {
                Ok(output) => output,
                Err(error) => {
                    for (output, _) in opened {
                        let _ = output.close();
                    }
                    return Err(error.into());
                }
            };
            let mut bytes = Vec::new();
            lucene_store::codec_util::write_index_header(
                &mut bytes,
                codec,
                1,
                state.id,
                state.suffix,
            );
            output.write_bytes(&bytes);
            opened.push((output, bytes));
        }
        let mut opened = opened.into_iter();
        let (data_out, data) = opened.next().expect("data output");
        let (meta_out, meta) = opened.next().expect("metadata output");
        let (index_out, index) = opened.next().expect("index output");
        let published = [meta.len(), index.len(), data.len()];
        Ok(Lucene90PointsWriter {
            state,
            meta,
            index,
            data,
            outputs: Some([meta_out, index_out, data_out]),
            published,
            finished: false,
        })
    }
    fn fields_reader<'a>(&mut self, state: ReadState<'a>) -> Result<Self::Reader<'a>> {
        let reader = points::open(state.meta, state.index, state.data, state.id, state.suffix)?;
        Ok(Lucene90PointsReader {
            reader: Some(reader),
            index: state.index,
            data: state.data,
            fields: state.fields,
        })
    }
}
impl<'a> Lucene90PointsWriter<'a> {
    fn ensure_open(&self) -> Result<()> {
        if self.outputs.is_none() {
            Err(failure("closed points writer"))
        } else {
            Ok(())
        }
    }
    fn publish(&mut self) -> Result<()> {
        use lucene_store::DataOutput;
        let outputs = self
            .outputs
            .as_mut()
            .ok_or_else(|| failure("closed points writer"))?;
        for ((output, offset), bytes) in
            outputs
                .iter_mut()
                .zip(&mut self.published)
                .zip([&self.meta, &self.index, &self.data])
        {
            output.write_bytes(&bytes[*offset..]);
            *offset = bytes.len();
        }
        Ok(())
    }
    fn bkd(
        &self,
        field: &FieldInfo,
        count: i64,
    ) -> Result<lucene_codecs::bkd_writer::BkdWriter<'a>> {
        use lucene_codecs::bkd_writer::{
            BkdConfig, BkdWriter, DEFAULT_MAX_MB_SORT_IN_HEAP, VERSION_CURRENT,
        };
        let as_usize = |value: i32| {
            usize::try_from(value).map_err(|_| failure("negative point configuration"))
        };
        let config = BkdConfig::new(
            as_usize(field.point_dimension_count)?,
            as_usize(field.point_index_dimension_count)?,
            as_usize(field.point_num_bytes)?,
            self.state.max_points_in_leaf_node,
        )?;
        Ok(BkdWriter::new(
            as_usize(self.state.max_doc)?,
            Some(self.state.directory),
            self.state.segment,
            config,
            DEFAULT_MAX_MB_SORT_IN_HEAP,
            u64::try_from(count).map_err(|_| failure("negative point count"))?,
            VERSION_CURRENT,
        )?)
    }
}
struct AddVisitor<'b, 'd> {
    writer: &'b mut lucene_codecs::bkd_writer::BkdWriter<'d>,
    map: Option<&'b [i32]>,
}
impl IntersectVisitor for AddVisitor<'_, '_> {
    fn visit(&mut self, _: i32) -> Result<()> {
        Err(failure("missing packed value"))
    }
    fn visit_value(&mut self, doc: i32, value: &[u8]) -> Result<()> {
        let doc = if let Some(map) = self.map {
            let index = usize::try_from(doc).map_err(|_| failure("negative source document"))?;
            *map.get(index)
                .ok_or_else(|| failure("document map out of bounds"))?
        } else {
            doc
        };
        // SENTINEL: MergeState.DocMap returns -1 for a deleted source document.
        if doc != -1 {
            self.writer.add(value, doc)?;
        }
        Ok(())
    }
    fn compare(&mut self, _: &[u8], _: &[u8]) -> points::Relation {
        points::Relation::CellCrossesQuery
    }
}
/// Java: `BKDWriter.MergeIntersectsVisitor`, one reusable leaf's doc/value buffers.
struct MergeLeaf {
    docs: Vec<i32>,
    values: Vec<u8>,
    stride: usize,
}
impl IntersectVisitor for MergeLeaf {
    fn visit(&mut self, _: i32) -> Result<()> {
        Err(failure("missing packed value"))
    }
    fn visit_value(&mut self, doc: i32, value: &[u8]) -> Result<()> {
        self.docs.push(doc);
        self.values.extend_from_slice(value);
        Ok(())
    }
    fn grow(&mut self, count: usize) {
        self.docs.reserve(count);
        self.values.reserve(count.saturating_mul(self.stride));
    }
    fn compare(&mut self, _: &[u8], _: &[u8]) -> points::Relation {
        points::Relation::CellCrossesQuery
    }
}
/// Java: `BKDWriter.MergeReader`. Pending right nodes are the native cursor
/// representation; only the current leaf's points are retained. Iterator's
/// infallible signature requires an explicit shared first-error slot, checked
/// by the writer before any partial merge output is published.
struct NativeMergeSource<'r, 'd, 'm> {
    reader: &'r points::PointsReader<'d>,
    pending: Vec<points::PointTreeNode>,
    leaf: MergeLeaf,
    position: usize,
    map: &'m [i32],
    error: Rc<RefCell<Option<Error>>>,
}
impl<'r, 'd, 'm> NativeMergeSource<'r, 'd, 'm> {
    fn new(
        reader: &'r points::PointsReader<'d>,
        number: i32,
        target: &FieldInfo,
        map: &'m [i32],
        error: Rc<RefCell<Option<Error>>>,
    ) -> Result<Self> {
        let field = reader
            .field(number)
            .ok_or_else(|| failure("missing native point field"))?;
        if field.num_dims != 1
            || field.num_dims != target.point_dimension_count
            || field.num_index_dims != target.point_index_dimension_count
            || field.bytes_per_dim != target.point_num_bytes
        {
            return Err(failure("native merge point configuration mismatch"));
        }
        let stride = usize::try_from(field.bytes_per_dim)
            .map_err(|_| failure("negative point configuration"))?;
        Ok(Self {
            reader,
            pending: vec![reader.point_tree(number)?],
            leaf: MergeLeaf {
                docs: Vec::new(),
                values: Vec::new(),
                stride,
            },
            position: 0,
            map,
            error,
        })
    }
    fn advance(&mut self) -> Result<Option<(Vec<u8>, i32)>> {
        loop {
            if let Some(&old) = self.leaf.docs.get(self.position) {
                let index =
                    usize::try_from(old).map_err(|_| failure("negative source document"))?;
                let doc = *self
                    .map
                    .get(index)
                    .ok_or_else(|| failure("document map out of bounds"))?;
                let start = self
                    .position
                    .checked_mul(self.leaf.stride)
                    .ok_or_else(|| failure("point value offset overflow"))?;
                let end = start
                    .checked_add(self.leaf.stride)
                    .ok_or_else(|| failure("point value offset overflow"))?;
                self.position += 1;
                // SENTINEL: MergeState.DocMap returns -1 for a deleted source document.
                if doc != -1 {
                    return Ok(Some((self.leaf.values[start..end].to_vec(), doc)));
                }
                continue;
            }
            let Some(node) = self.pending.pop() else {
                return Ok(None);
            };
            if let Some((left, right)) = self.reader.point_tree_children(&node)? {
                self.pending.push(right);
                self.pending.push(left);
                continue;
            }
            self.leaf.docs.clear();
            self.leaf.values.clear();
            self.position = 0;
            let mut bridge = VisitorBridge {
                visitor: &mut self.leaf,
                error: None,
            };
            let result = self.reader.visit_leaf(&node, &mut bridge);
            if let Some(error) = bridge.error {
                return Err(error);
            }
            result?;
        }
    }
}
impl Iterator for NativeMergeSource<'_, '_, '_> {
    type Item = (Vec<u8>, i32);
    fn next(&mut self) -> Option<Self::Item> {
        if self.error.borrow().is_some() {
            return None;
        }
        match self.advance() {
            Ok(point) => point,
            Err(error) => {
                let mut first = self.error.borrow_mut();
                if first.is_none() {
                    *first = Some(error);
                }
                None
            }
        }
    }
}
impl PointsWriter for Lucene90PointsWriter<'_> {
    fn write_field<R: PointsReader>(&mut self, field: &FieldInfo, reader: &R) -> Result<()> {
        use lucene_store::DataOutput;
        self.ensure_open()?;
        let values = reader
            .values(&field.name)?
            .ok_or_else(|| failure("missing point values"))?;
        let mut tree = values.point_tree()?;
        let mut writer = self.bkd(field, tree.size())?;
        tree.visit_doc_values(&mut AddVisitor {
            writer: &mut writer,
            map: None,
        })?;
        // BKDWriter borrows only the directory, not the adapter buffers.
        let mut data = std::mem::take(&mut self.data);
        let plan = writer.finish(&mut data);
        self.data = data;
        let plan = plan?;
        if let Some(plan) = plan {
            self.meta.write_i32(field.number);
            writer.write_index(&mut self.meta, &mut self.index, &plan);
        }
        self.publish()
    }
    fn finish(&mut self) -> Result<()> {
        use lucene_store::DataOutput;
        self.ensure_open()?;
        if self.finished {
            return Err(failure("already finished"));
        }
        self.finished = true;
        self.meta.write_i32(-1);
        lucene_store::codec_util::write_footer(&mut self.index);
        lucene_store::codec_util::write_footer(&mut self.data);
        self.meta.write_i64(self.index.len() as i64);
        self.meta.write_i64(self.data.len() as i64);
        lucene_store::codec_util::write_footer(&mut self.meta);
        self.publish()
    }
    fn merge<R: PointsReader>(&mut self, state: &MergeState<'_, R>) -> Result<()> {
        use lucene_store::DataOutput;
        self.ensure_open()?;
        let optimized = state.sources.iter().all(|source| {
            source
                .reader
                .is_some_and(|reader| reader.native().is_some())
        });
        for source in state.sources {
            if let Some(reader) = source.reader {
                (state.check_aborted)()?;
                reader.check_integrity()?;
            }
        }
        for field in &state.fields.fields {
            if field.point_dimension_count == 0 {
                continue;
            }
            let mut count = 0i64;
            for source in state.sources {
                if !source
                    .fields
                    .field_by_name(&field.name)
                    .is_some_and(|field| field.point_dimension_count > 0)
                {
                    continue;
                }
                if let Some(reader) = source.reader {
                    if let Some(values) = reader.values(&field.name)? {
                        count = count
                            .checked_add(values.size())
                            .ok_or_else(|| failure("point count overflow"))?;
                    }
                }
            }
            let mut writer = self.bkd(field, count)?;
            let mut merged: Vec<Box<dyn Iterator<Item = (Vec<u8>, i32)> + '_>> = Vec::new();
            let merge_error = Rc::new(RefCell::new(None));
            for source in state.sources {
                if !source
                    .fields
                    .field_by_name(&field.name)
                    .is_some_and(|field| field.point_dimension_count > 0)
                {
                    continue;
                }
                let Some(reader) = source.reader else {
                    continue;
                };
                let Some(values) = reader.values(&field.name)? else {
                    continue;
                };
                if optimized && field.point_dimension_count == 1 {
                    let native = reader
                        .native()
                        .expect("optimized merge requires native readers");
                    let number = source
                        .fields
                        .field_by_name(&field.name)
                        .expect("source field checked above")
                        .number;
                    merged.push(Box::new(NativeMergeSource::new(
                        native,
                        number,
                        field,
                        source.doc_map,
                        merge_error.clone(),
                    )?));
                } else {
                    let mut tree = values.point_tree()?;
                    tree.visit_doc_values(&mut AddVisitor {
                        writer: &mut writer,
                        map: Some(source.doc_map),
                    })?;
                }
            }
            let mut data = std::mem::take(&mut self.data);
            let plan = if optimized && field.point_dimension_count == 1 {
                writer.merge(&mut data, merged)
            } else {
                writer.finish(&mut data)
            };
            self.data = data;
            if let Some(error) = merge_error.borrow_mut().take() {
                return Err(error);
            }
            let plan = plan?;
            if let Some(plan) = plan {
                self.meta.write_i32(field.number);
                writer.write_index(&mut self.meta, &mut self.index, &plan);
            }
            self.publish()?;
        }
        self.finish()
    }
    fn close(&mut self) -> Result<()> {
        let Some(outputs) = self.outputs.take() else {
            return Ok(());
        };
        let mut first = None;
        for output in outputs {
            if let Err(error) = output.close() {
                if first.is_none() {
                    first = Some(error);
                }
            }
        }
        if let Some(error) = first {
            return Err(error.into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
