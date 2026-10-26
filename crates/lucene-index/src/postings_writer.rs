//! Installed native postings groups with operation-owned output cleanup.
//!
//! Java: `FreqProxTermsWriter.flush`, `SegmentMerger.mergeTerms` and
//! `PerFieldPostingsFormat.FieldsWriter`. Readers and mapped merge data remain
//! native; the lower layer owns group consumers until their outer close.

use lucene_codecs::per_field_postings::{self, GroupOutput, Lucene104PostingsFormat};
use lucene_codecs::postings_consumer::{PostingsWriteState, PostingsWriterFactory};
use lucene_codecs::postings_writer::{self, FieldNorms, FieldPostingsInput, TermSource};

use crate::created_outputs::CreatedOutputs;

pub(crate) fn write_installed<T: TermSource>(
    factory: &dyn PostingsWriterFactory,
    state: PostingsWriteState<'_>,
    inputs: &[FieldPostingsInput<'_, T>],
    norms: &[FieldNorms<'_>],
    name_of: &dyn Fn(i32) -> Option<String>,
    format_for: &dyn Fn(&str) -> Lucene104PostingsFormat,
    merge: bool,
) -> postings_writer::Result<Vec<GroupOutput>> {
    let outputs = CreatedOutputs::new(state.directory);
    let result = per_field_postings::write_installed(
        inputs,
        norms,
        name_of,
        format_for,
        factory,
        PostingsWriteState {
            directory: &outputs,
            ..state
        },
        merge,
    );
    if result.is_err() {
        for name in outputs.files().iter() {
            let _ = state.directory.delete_file(name);
        }
    }
    result
}
