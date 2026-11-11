//! The Lucene fields one pattern-text value is indexed as.

use lucene_index::document::{
    Document, Field, FieldType, IndexOptions, SortedDocValuesField, StoredValue,
};

use crate::format::{encode_args, split};

/// The suffix of the `SORTED` doc-values field that holds the template.
pub const TEMPLATE_SUFFIX: &str = ".template";
/// The suffix of the stored field that holds the variables.
pub const ARGS_SUFFIX: &str = ".args";

/// The indexed message's type: tokenized, docs only, no norms.
fn message_type() -> FieldType {
    let mut t = FieldType::new();
    // Each setter fails only on a frozen type; this one is fresh.
    let _ = t.set_tokenized(true);
    let _ = t.set_index_options(IndexOptions::Docs);
    let _ = t.set_omit_norms(true);
    t.freeze();
    t
}

/// Adds `message` to `doc` as field `name`: the message indexed docs-only,
/// `name.template` as a `SORTED` doc value, and `name.args` stored when the
/// message has variables.
///
/// # Errors
/// What [`Field::from_string`] refuses (never, for this type).
pub fn fields(doc: &mut Document, name: &str, message: &str) -> lucene_index::document::Result<()> {
    doc.add(Field::from_string(name, message, message_type())?);
    let s = split(message);
    doc.add(SortedDocValuesField::new(
        format!("{name}{TEMPLATE_SUFFIX}"),
        s.template.into_bytes(),
    ));
    if !s.args.is_empty() {
        doc.add(Field::stored(
            format!("{name}{ARGS_SUFFIX}"),
            StoredValue::Binary(encode_args(&s.args)),
        ));
    }
    Ok(())
}
