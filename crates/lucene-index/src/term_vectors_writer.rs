//! Installed native term-vector callbacks and operation-owned error cleanup.
//!
//! The existing owned document model resolves field metadata before replay
//! and validates occurrence shapes before term callbacks. Unlike Java's lazy
//! Fields/Terms/Postings graph, malformed owned input can therefore fail before
//! an earlier delegate callback. Valid inputs use Java's callback sequence.

use lucene_codecs::field_infos::FieldInfo;
use lucene_codecs::term_vectors::{self, TermVectorsDocument};
use lucene_codecs::term_vectors_writer::{
    self, TermVectorsWriteSink, TermVectorsWriteState, TermVectorsWriterFactory,
};

use crate::created_outputs::CreatedOutputs;

/// Java: `TermVectorsConsumer.flush` and `SegmentMerger.mergeTermVectors`.
/// Publication belongs to the caller. Only successful creations are removed
/// on returned errors; a failed create never owns a preexisting collision.
pub(crate) fn write_installed<E>(
    factory: &dyn TermVectorsWriterFactory,
    state: TermVectorsWriteState<'_>,
    operation: impl FnOnce(&mut dyn TermVectorsWriteSink) -> Result<(), E>,
) -> Result<Vec<String>, E>
where
    E: From<term_vectors::Error>,
{
    let outputs = CreatedOutputs::new(state.directory);
    let result = term_vectors_writer::with_writer(
        factory,
        TermVectorsWriteState {
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

fn invalid(message: impl Into<String>) -> term_vectors::Error {
    lucene_store::Error::Corrupted(message.into()).into()
}

/// Java: `TermVectorsWriter.addAllDocVectors`. The owned document model
/// replaces Java's lazy Fields/Terms/Postings graph; shape errors remain
/// returned errors before indexing occurrence arrays.
pub(crate) fn write_document(
    writer: &mut dyn TermVectorsWriteSink,
    document: &TermVectorsDocument,
    infos: &[FieldInfo],
) -> term_vectors::Result<()> {
    let mut fields = document
        .fields
        .iter()
        .map(|field| {
            let info = infos
                .iter()
                .find(|info| info.number == field.field_number)
                .ok_or_else(|| {
                    invalid(format!("unknown term-vector field {}", field.field_number))
                })?;
            Ok((info, field))
        })
        .collect::<term_vectors::Result<Vec<_>>>()?;
    // Java String.compareTo compares UTF-16 code units, including astral names.
    fields.sort_by(|(a, _), (b, _)| a.name.encode_utf16().cmp(b.name.encode_utf16()));
    let count = i32::try_from(fields.len()).map_err(|_| invalid("too many vector fields"))?;
    writer.start_document(count)?;
    for (info, field) in fields {
        let count =
            i32::try_from(field.terms.len()).map_err(|_| invalid("too many vector terms"))?;
        writer.start_field(
            info,
            count,
            field.has_positions,
            field.has_offsets,
            field.has_payloads,
        )?;
        for term in &field.terms {
            let frequency = usize::try_from(term.freq)
                .ok()
                .filter(|frequency| *frequency > 0)
                .ok_or_else(|| invalid("nonpositive term-vector frequency"))?;
            let positions = term.positions.as_deref();
            let starts = term.start_offsets.as_deref();
            let ends = term.end_offsets.as_deref();
            let payloads = term.payloads.as_deref();
            let matching = |enabled: bool, length: Option<usize>| {
                if enabled {
                    length == Some(frequency)
                } else {
                    length.is_none()
                }
            };
            if !matching(field.has_positions, positions.map(<[i32]>::len))
                || !matching(field.has_offsets, starts.map(<[i32]>::len))
                || !matching(field.has_offsets, ends.map(<[i32]>::len))
                || !matching(field.has_payloads, payloads.map(<[Vec<u8>]>::len))
                || field.has_payloads && !field.has_positions
            {
                return Err(invalid(
                    "term-vector occurrence arrays disagree with flags or frequency",
                ));
            }
            writer.start_term(&term.term, term.freq)?;
            if field.has_positions || field.has_offsets {
                for occurrence in 0..frequency {
                    // SENTINEL: Java passes -1 for each axis absent from this field.
                    // Enabled arrays have exactly frequency entries, checked above.
                    let position = positions.map_or(-1, |values| values[occurrence]);
                    let start = starts.map_or(-1, |values| values[occurrence]);
                    let end = ends.map_or(-1, |values| values[occurrence]);
                    let payload = payloads
                        .and_then(|values| values.get(occurrence))
                        .filter(|value| !value.is_empty())
                        .map(Vec::as_slice);
                    writer.add_position(position, start, end, payload)?;
                }
            }
            writer.finish_term()?;
        }
        writer.finish_field()?;
    }
    writer.finish_document()
}

pub(crate) fn flush(
    factory: &dyn TermVectorsWriterFactory,
    state: TermVectorsWriteState<'_>,
    infos: &[FieldInfo],
    documents: &[TermVectorsDocument],
) -> term_vectors::Result<Vec<String>> {
    write_installed(factory, state, |writer| {
        for document in documents {
            write_document(writer, document, infos)?;
        }
        writer.finish(state.max_doc)
    })
}

#[cfg(test)]
mod tests;
