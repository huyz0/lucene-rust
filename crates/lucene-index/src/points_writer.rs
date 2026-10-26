//! Installed points callbacks over the unchanged native packed/merge inputs.

use lucene_codecs::field_infos::FieldInfo;
use lucene_codecs::points::{self, MergePointsField, PackedPointsField};
use lucene_codecs::points_writer::{self, PointsWriteState, PointsWriterFactory};

use crate::created_outputs::CreatedOutputs;

/// Java: `IndexingChain.writePoints` and `SegmentMerger.mergePoints`.
/// The caller owns publication; returned errors remove only operation-created files.
pub(crate) fn write_installed(
    factory: &dyn PointsWriterFactory,
    state: PointsWriteState<'_>,
    operation: impl FnOnce(&mut dyn points_writer::PointsWriteSink) -> points::Result<()>,
) -> points::Result<Vec<String>> {
    let outputs = CreatedOutputs::new(state.directory);
    let result = points_writer::with_writer(
        factory,
        PointsWriteState {
            directory: &outputs,
            ..state
        },
        operation,
    );
    if let Err(error) = result {
        for name in outputs.files().iter() {
            let _ = state.directory.delete_file(name);
        }
        return Err(error);
    }
    let files = outputs.files().iter().cloned().collect();
    Ok(files)
}

pub(crate) fn flush(
    factory: &dyn PointsWriterFactory,
    state: PointsWriteState<'_>,
    field_infos: &[FieldInfo],
    fields: Vec<PackedPointsField>,
) -> points::Result<Vec<String>> {
    write_installed(factory, state, |writer| {
        for values in fields {
            let field = field_infos
                .iter()
                .find(|field| field.number == values.field_number)
                .ok_or(points::Error::IllegalFieldNumber(values.field_number))?;
            writer.write_field(field, values)?;
        }
        writer.finish()
    })
}

pub(crate) fn merge(
    factory: &dyn PointsWriterFactory,
    state: PointsWriteState<'_>,
    fields: Vec<MergePointsField>,
    native_readers: bool,
) -> points::Result<Vec<String>> {
    write_installed(factory, state, |writer| {
        writer.merge(fields, native_readers)
    })
}
