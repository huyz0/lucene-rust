//! The Lucene fields one pattern-text value is indexed as.

use lucene_index::document::{
    Document, Error, Field, FieldType, IndexOptions, SortedDocValuesField, StoredValue,
};

use crate::format::{encode_args, split};

/// The suffix of the `SORTED` doc-values field that holds the template.
pub const TEMPLATE_SUFFIX: &str = ".template";
/// The suffix of the stored field that holds the variables.
pub const ARGS_SUFFIX: &str = ".args";

/// `IndexWriter.MAX_TERM_LENGTH`: the longest `SORTED` value Lucene writes.
pub const MAX_TEMPLATE_BYTES: usize = 32766;

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
/// message has variables. The field is single-valued: call this once per
/// document and name.
///
/// # Errors
/// [`Error::IllegalArgument`] for a template over [`MAX_TEMPLATE_BYTES`],
/// which Lucene's `SortedDocValuesWriter` refuses too.
pub fn fields(doc: &mut Document, name: &str, message: &str) -> lucene_index::document::Result<()> {
    let s = split(message);
    if s.template.len() > MAX_TEMPLATE_BYTES {
        return Err(Error::IllegalArgument(format!(
            "pattern_text field [{name}]: the message's template is {} bytes, over the {MAX_TEMPLATE_BYTES}-byte doc-values limit",
            s.template.len()
        )));
    }
    doc.add(Field::from_string(name, message, message_type())?);
    let args = (!s.args.is_empty()).then(|| encode_args(&s.args));
    doc.add(SortedDocValuesField::new(
        format!("{name}{TEMPLATE_SUFFIX}"),
        s.template.into_bytes(),
    ));
    if let Some(args) = args {
        doc.add(Field::stored(
            format!("{name}{ARGS_SUFFIX}"),
            StoredValue::Binary(args),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_template_over_the_doc_values_limit_is_refused() {
        let mut doc = Document::new();
        let long = "x".repeat(MAX_TEMPLATE_BYTES + 1);
        assert!(fields(&mut doc, "m", &long).is_err());
        let ok = format!("{} 1", "y".repeat(MAX_TEMPLATE_BYTES - 2));
        assert!(fields(&mut doc, "m", &ok).is_ok());
    }
}
