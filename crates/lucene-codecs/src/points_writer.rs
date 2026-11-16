//! Optional same-wire points writers for native flush and merge operations.
//!
//! The operation owns its writer; the index layer owns output tracking, cleanup,
//! synchronization and publication. The native sink reserves the three outputs
//! in Java constructor order, buffers headers and flush fields until `finish`,
//! and encodes a merge in `merge` (which includes finish). Java streams bytes earlier;
//! this bounded sink reports encoder failures at finish and sticky output errors
//! at close. Native readers and BKD merge dispatch remain unchanged.

use std::{fmt, io, sync::Arc};

use lucene_store::{DataOutput, Directory, FsIndexOutput};

use crate::field_infos::FieldInfo;
use crate::points::{self, MergePointsField, PackedPointsField};

/// Native identity and document bound for one points output triple.
#[derive(Clone, Copy)]
pub struct PointsWriteState<'a> {
    pub directory: &'a dyn Directory,
    pub segment: &'a str,
    pub id: &'a [u8; 16],
    pub suffix: &'a str,
    pub max_doc: i32,
}

/// Java's `PointsWriter` callbacks over owned native point buffers.
///
/// `merge` includes finish and retains source order, mapped document IDs and
/// the deleted-inclusive source budgets. `native_readers` records Java's
/// `instanceof Lucene90PointsReader` check, including point-free sources.
pub trait PointsWriteSink {
    fn write_field(&mut self, field: &FieldInfo, values: PackedPointsField) -> points::Result<()>;
    fn merge(&mut self, fields: Vec<MergePointsField>, native_readers: bool) -> points::Result<()>;
    fn finish(&mut self) -> points::Result<()>;
    fn close(&mut self) -> points::Result<()>;
}

/// Shared codec configuration opening an operation-local points writer.
pub trait PointsWriterFactory: Send + Sync + fmt::Debug {
    fn fields_writer<'a>(
        &self,
        state: PointsWriteState<'a>,
    ) -> points::Result<Box<dyn PointsWriteSink + 'a>>;
}

/// Closes after a returned success or error, preserving the operation's error.
/// A close error fails an otherwise successful operation. Panics unwind without
/// an automatic close or cleanup guarantee, as for the other writer seams.
pub fn with_writer<T, E>(
    factory: &dyn PointsWriterFactory,
    state: PointsWriteState<'_>,
    operation: impl FnOnce(&mut dyn PointsWriteSink) -> Result<T, E>,
) -> Result<T, E>
where
    E: From<points::Error>,
{
    let mut writer = factory.fields_writer(state).map_err(E::from)?;
    let result = operation(writer.as_mut());
    let closed = writer.close().map_err(E::from);
    match result {
        Ok(value) => closed.map(|()| value),
        Err(error) => Err(error),
    }
}

fn lifecycle_error(message: &str) -> points::Error {
    lucene_store::Error::Io(io::Error::new(io::ErrorKind::InvalidInput, message)).into()
}

/// A `Lucene90PointsWriter` subclass overriding `writeField` to add each
/// point to a fresh `BKDWriter` and finish it (`RandomCodec`): its leaf
/// size, heap budget and `split` override. Its flush and merge both go
/// through [`points::write_added`], as Java's inherited `merge` reaches the
/// override through `mergeOneField` whenever a source is not a plain
/// `Lucene90PointsReader`.
pub trait AddFinishPolicy: Send + Sync + fmt::Debug {
    fn max_points_in_leaf_node(&self) -> i32;
    fn max_mb_sort_in_heap(&self) -> f64;
    /// The `BKDWriter.split` override for the field named `field_name`.
    fn split(&self, field_name: &str) -> Option<crate::bkd_writer::SplitOverride<'static>>;
}

/// Buffered production encoder retaining native field order and BKD settings.
pub struct NativePointsWriter<'a> {
    state: PointsWriteState<'a>,
    fields: Vec<PackedPointsField>,
    /// Each buffered field's name, for [`AddFinishPolicy::split`].
    names: Vec<String>,
    add_finish: Option<Arc<dyn AddFinishPolicy>>,
    meta: Option<FsIndexOutput>,
    index: Option<FsIndexOutput>,
    data: Option<FsIndexOutput>,
    finished: bool,
    closed: bool,
}

impl<'a> NativePointsWriter<'a> {
    /// Java: `Lucene90PointsWriter` constructor, data then meta then index.
    /// A failed open closes every successfully created output, preserving the
    /// open error. The caller's tracked directory retains cleanup ownership.
    pub fn new(state: PointsWriteState<'a>) -> points::Result<Self> {
        let mut writer = Self {
            state,
            fields: Vec::new(),
            names: Vec::new(),
            add_finish: None,
            meta: None,
            index: None,
            data: None,
            finished: false,
            closed: false,
        };
        let stem = if state.suffix.is_empty() {
            state.segment.to_owned()
        } else {
            format!("{}_{}", state.segment, state.suffix)
        };
        let result = (|| -> lucene_store::Result<()> {
            writer.data = Some(state.directory.create_output(&format!("{stem}.kdd"))?);
            writer.meta = Some(state.directory.create_output(&format!("{stem}.kdm"))?);
            writer.index = Some(state.directory.create_output(&format!("{stem}.kdi"))?);
            Ok(())
        })();
        if let Err(error) = result {
            let _ = writer.close();
            return Err(error.into());
        }
        Ok(writer)
    }

    /// [`Self::new`] for an [`AddFinishPolicy`] subclass.
    pub fn with_add_finish(
        state: PointsWriteState<'a>,
        policy: Arc<dyn AddFinishPolicy>,
    ) -> points::Result<Self> {
        let mut writer = Self::new(state)?;
        writer.add_finish = Some(policy);
        Ok(writer)
    }

    /// The policy's `.kdm`/`.kdi`/`.kdd` for `fields`.
    fn write_added(
        &self,
        policy: &dyn AddFinishPolicy,
        fields: Vec<MergePointsField>,
    ) -> points::Result<(Vec<u8>, Vec<u8>, Vec<u8>)> {
        let split = |name: &str| policy.split(name);
        points::write_added(
            fields,
            &points::AddFinishPoints {
                max_points_in_leaf_node: policy.max_points_in_leaf_node(),
                max_mb_sort_in_heap: policy.max_mb_sort_in_heap(),
                split: &split,
            },
            self.state.max_doc,
            self.state.directory,
            self.state.segment,
            self.state.id,
            self.state.suffix,
        )
    }

    fn writable(&self) -> points::Result<()> {
        if self.closed {
            Err(lifecycle_error("closed points writer"))
        } else if self.finished {
            Err(lifecycle_error("already finished"))
        } else {
            Ok(())
        }
    }

    fn publish(&mut self, bytes: (Vec<u8>, Vec<u8>, Vec<u8>)) {
        // Outputs remain owned until close; publication here follows the
        // buffered encoders' tuple order, not the constructor's open order.
        self.meta
            .as_mut()
            .expect("open metadata")
            .write_bytes(&bytes.0);
        self.index
            .as_mut()
            .expect("open index")
            .write_bytes(&bytes.1);
        self.data.as_mut().expect("open data").write_bytes(&bytes.2);
    }
}

impl PointsWriteSink for NativePointsWriter<'_> {
    fn write_field(&mut self, field: &FieldInfo, values: PackedPointsField) -> points::Result<()> {
        self.writable()?;
        if field.number != values.field_number
            || field.point_dimension_count != values.num_dims
            || field.point_index_dimension_count != values.num_index_dims
            || field.point_num_bytes != values.bytes_per_dim
        {
            return Err(lifecycle_error(
                "point field identity/configuration mismatch",
            ));
        }
        self.fields.push(values);
        self.names.push(field.name.clone());
        Ok(())
    }

    fn merge(&mut self, fields: Vec<MergePointsField>, native_readers: bool) -> points::Result<()> {
        self.writable()?;
        if !self.fields.is_empty() {
            return Err(lifecycle_error("cannot merge after writing flush fields"));
        }
        self.finished = true;
        if let Some(policy) = self.add_finish.clone() {
            let bytes = self.write_added(policy.as_ref(), fields)?;
            self.publish(bytes);
            return Ok(());
        }
        let bytes = points::write_merged_with_mode(
            fields,
            points::DEFAULT_MAX_POINTS_IN_LEAF_NODE,
            self.state.max_doc,
            self.state.directory,
            self.state.segment,
            self.state.id,
            self.state.suffix,
            native_readers,
        )?;
        self.publish(bytes);
        Ok(())
    }

    fn finish(&mut self) -> points::Result<()> {
        self.writable()?;
        self.finished = true;
        if let Some(policy) = self.add_finish.clone() {
            let fields = std::mem::take(&mut self.fields)
                .into_iter()
                .zip(std::mem::take(&mut self.names))
                .map(|(field, field_name)| MergePointsField {
                    field_number: field.field_number,
                    field_name,
                    num_dims: field.num_dims,
                    num_index_dims: field.num_index_dims,
                    bytes_per_dim: field.bytes_per_dim,
                    max_point_count: field.docs.len() as u64,
                    sources: vec![points::PointsMergeSource {
                        docs: field.docs,
                        values: field.values,
                    }],
                })
                .collect();
            let bytes = self.write_added(policy.as_ref(), fields)?;
            self.publish(bytes);
            return Ok(());
        }
        let bytes = points::write_packed(
            std::mem::take(&mut self.fields),
            points::DEFAULT_MAX_POINTS_IN_LEAF_NODE,
            self.state.max_doc,
            self.state.id,
            self.state.suffix,
        )?;
        self.publish(bytes);
        Ok(())
    }

    /// Java: `IOUtils.close(metaOut, indexOut, dataOut)`, including all closes
    /// after a failure. Taking each output makes repeated close harmless.
    fn close(&mut self) -> points::Result<()> {
        self.closed = true;
        self.fields.clear();
        self.names.clear();
        let mut first_error = None;
        for output in [&mut self.meta, &mut self.index, &mut self.data] {
            if let Some(output) = output.take() {
                if let Err(error) = output.close() {
                    first_error.get_or_insert(error);
                }
            }
        }
        first_error.map_or(Ok(()), |error| Err(error.into()))
    }
}

#[cfg(test)]
mod tests;
