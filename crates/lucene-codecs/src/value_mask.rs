//! A mask over doc values: what a reader wrapper that rewrites values (the security plugin's
//! field masking) shows in place of a keyword or binary field's stored bytes.
//!
//! Lucene's wrapper keeps the ordinals and the sort order of the underlying values and rewrites
//! only what it hands out: `lookupOrd`, `TermsEnum.term()` and `binaryValue()` show the masked
//! bytes, while `TermsEnum.next()`, the seeks and every comparison keep the raw ones. This port
//! puts the mask on the entry that describes the values ([`crate::terms_dict::TermsDictEntry`],
//! [`crate::doc_values::BinaryEntry`]), so each reader that opens a masked entry is the same
//! choke point:
//!
//! * [`crate::terms_dict::TermsDict::lookup_ord`] -- the ordinal-to-bytes conversion -- answers
//!   the masked bytes; [`crate::terms_dict::TermsDict::raw_term_at`], the seeks and the
//!   sequential [`crate::terms_dict::TermsCursor`] answer the raw ones, and the name says so;
//! * [`crate::doc_values::BinaryReader::value_ref`] answers the masked bytes, and every other
//!   path to a masked binary value ([`crate::doc_values::BinaryReader::value`],
//!   [`crate::doc_values::binary_value`]) refuses, so a caller that has not been taught about
//!   masks cannot read around it.

use std::fmt::Debug;
use std::sync::Arc;

use lucene_store::Result;

/// The function a masked field's values go through.
pub trait ValueMask: Send + Sync + Debug {
    /// The bytes shown in place of `raw`.
    ///
    /// # Errors
    /// The mask could not be applied (a callback into the host failed): the read fails rather
    /// than showing the raw value.
    fn apply(&self, raw: &[u8]) -> Result<Vec<u8>>;
}

/// A mask shared by the entries of one masked field.
pub type SharedMask = Arc<dyn ValueMask>;

/// The error a path that would show raw bytes of a masked field answers.
pub fn masked_error(what: &str) -> lucene_store::Error {
    lucene_store::Error::IllegalArgument(format!(
        "{what} of a masked field would show its raw values"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug)]
    struct Upper;

    impl ValueMask for Upper {
        fn apply(&self, raw: &[u8]) -> Result<Vec<u8>> {
            Ok(raw.to_ascii_uppercase())
        }
    }

    #[test]
    fn a_shared_mask_applies() {
        let mask: SharedMask = Arc::new(Upper);
        assert_eq!(mask.apply(b"ab").unwrap(), b"AB");
    }

    #[test]
    fn the_refusal_names_what_it_refuses() {
        assert!(masked_error("binary_value")
            .to_string()
            .contains("binary_value of a masked field"));
    }
}
