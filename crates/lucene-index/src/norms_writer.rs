//! Operation-local installed norms output and returned-error cleanup.

use crate::created_outputs::CreatedOutputs;
use lucene_codecs::field_infos::FieldInfo;
use lucene_codecs::norms::{self, NormsField};
use lucene_codecs::norms_writer::{self, NormsWriteState, NormsWriterFactory};
#[cfg(test)]
use lucene_store::Directory;

/// Writes borrowed columns through the installed consumer before publication.
/// Only files successfully created by this operation are removed on error.
pub(crate) fn write_installed(
    factory: &dyn NormsWriterFactory,
    state: NormsWriteState<'_>,
    field_infos: &[FieldInfo],
    fields: &[NormsField<'_>],
) -> norms::Result<Vec<String>> {
    let outputs = CreatedOutputs::new(state.directory);
    let result = norms_writer::with_writer(
        factory,
        NormsWriteState {
            directory: &outputs,
            ..state
        },
        |writer| {
            for values in fields {
                let number = match values {
                    NormsField::Dense(number, _) | NormsField::Sparse(number, _) => *number,
                };
                let field = field_infos
                    .iter()
                    .find(|field| field.number == number)
                    .ok_or(norms::Error::UnknownFieldNumber(number))?;
                let values = match values {
                    NormsField::Dense(number, values) => NormsField::Dense(*number, values),
                    NormsField::Sparse(number, values) => NormsField::Sparse(*number, values),
                };
                writer.add_norms_field(field, values)?;
            }
            Ok::<_, norms::Error>(())
        },
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

#[cfg(test)]
mod tests;
