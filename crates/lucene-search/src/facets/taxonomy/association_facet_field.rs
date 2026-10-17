//! `AssociationFacetField`, `IntAssociationFacetField` and
//! `FloatAssociationFacetField`: a flat taxonomy facet label with a value
//! associated to it in the document, which
//! [`FacetsConfig::build`](crate::facets::FacetsConfig::build) indexes as
//! `(ordinal, value)` pairs in a binary doc-values field.

use crate::facets::facet_field::verify_label;
use crate::facets::taxonomy::facet_label::java_array_to_string;
use crate::{Error, Result};
use lucene_util::geo::java_float_string;

/// Which Java class an [`AssociationFacetField`] is -- what
/// `FacetsConfig.build` checks a field's associations against.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AssociationKind {
    /// `AssociationFacetField` itself: opaque bytes.
    Bytes,
    /// `IntAssociationFacetField`.
    Int,
    /// `FloatAssociationFacetField`.
    Float,
}

impl AssociationKind {
    /// The type name `FacetsConfig.build` records per index field.
    pub fn type_name(self) -> &'static str {
        match self {
            AssociationKind::Bytes => "bytes",
            AssociationKind::Int => "int",
            AssociationKind::Float => "float",
        }
    }
}

/// `AssociationFacetField`: `dim`, a non-empty `path`, and the associated
/// bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AssociationFacetField {
    pub dim: String,
    pub path: Vec<String>,
    /// `assoc`.
    pub assoc: Vec<u8>,
    kind: AssociationKind,
}

impl AssociationFacetField {
    /// `new AssociationFacetField(assoc, dim, path...)`.
    pub fn new<S: AsRef<str>>(assoc: Vec<u8>, dim: &str, path: &[S]) -> Result<Self> {
        Self::of_kind(AssociationKind::Bytes, assoc, dim, path)
    }

    /// `new IntAssociationFacetField(assoc, dim, path...)`.
    pub fn int<S: AsRef<str>>(assoc: i32, dim: &str, path: &[S]) -> Result<Self> {
        Self::of_kind(
            AssociationKind::Int,
            IntAssociationFacetField::int_to_bytes_ref(assoc),
            dim,
            path,
        )
    }

    /// `new FloatAssociationFacetField(assoc, dim, path...)`.
    pub fn float<S: AsRef<str>>(assoc: f32, dim: &str, path: &[S]) -> Result<Self> {
        Self::of_kind(
            AssociationKind::Float,
            FloatAssociationFacetField::float_to_bytes_ref(assoc),
            dim,
            path,
        )
    }

    fn of_kind<S: AsRef<str>>(
        kind: AssociationKind,
        assoc: Vec<u8>,
        dim: &str,
        path: &[S],
    ) -> Result<Self> {
        verify_label(dim)?;
        for label in path {
            verify_label(label.as_ref())?;
        }
        if path.is_empty() {
            return Err(Error::IllegalArgument(
                "path must have at least one element".to_string(),
            ));
        }
        Ok(AssociationFacetField {
            dim: dim.to_string(),
            path: path.iter().map(|s| s.as_ref().to_string()).collect(),
            assoc,
            kind,
        })
    }

    /// The Java class this field is.
    pub fn kind(&self) -> AssociationKind {
        self.kind
    }
}

impl std::fmt::Display for AssociationFacetField {
    /// `toString()` of the field's Java class.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let path = java_array_to_string(&self.path);
        match self.kind {
            AssociationKind::Int => write!(
                f,
                "IntAssociationFacetField(dim={} path={path} value={})",
                self.dim,
                IntAssociationFacetField::bytes_ref_to_int(&self.assoc).unwrap_or_default()
            ),
            AssociationKind::Float => write!(
                f,
                "FloatAssociationFacetField(dim={} path={path} value={})",
                self.dim,
                java_float_string(
                    FloatAssociationFacetField::bytes_ref_to_float(&self.assoc).unwrap_or_default()
                )
            ),
            AssociationKind::Bytes => {
                // `BytesRef.toString()`: `[` hex bytes `]`.
                let bytes: Vec<String> = self.assoc.iter().map(|b| format!("{b:x}")).collect();
                write!(
                    f,
                    "AssociationFacetField(dim={} path={path} bytes=[{}])",
                    self.dim,
                    bytes.join(" ")
                )
            }
        }
    }
}

/// `IntAssociationFacetField`'s static helpers; the field itself is an
/// [`AssociationFacetField::int`].
#[derive(Debug, Clone, Copy)]
pub struct IntAssociationFacetField;

impl IntAssociationFacetField {
    /// `intToBytesRef(v)`: big-endian.
    pub fn int_to_bytes_ref(v: i32) -> Vec<u8> {
        v.to_be_bytes().to_vec()
    }

    /// `bytesRefToInt(b)`: `None` for fewer than four bytes (Java's
    /// `ArrayIndexOutOfBoundsException`).
    pub fn bytes_ref_to_int(b: &[u8]) -> Option<i32> {
        let head: [u8; 4] = b.get(..4)?.try_into().ok()?;
        Some(i32::from_be_bytes(head))
    }
}

/// `FloatAssociationFacetField`'s static helpers; the field itself is an
/// [`AssociationFacetField::float`].
#[derive(Debug, Clone, Copy)]
pub struct FloatAssociationFacetField;

impl FloatAssociationFacetField {
    /// `floatToBytesRef(v)`: `floatToIntBits`, big-endian (every `NaN` the
    /// canonical one).
    pub fn float_to_bytes_ref(v: f32) -> Vec<u8> {
        IntAssociationFacetField::int_to_bytes_ref(crate::facets::facet_result::float_to_int_bits(
            v,
        ))
    }

    /// `bytesRefToFloat(b)`.
    pub fn bytes_ref_to_float(b: &[u8]) -> Option<f32> {
        IntAssociationFacetField::bytes_ref_to_int(b).map(|bits| f32::from_bits(bits as u32))
    }
}

#[cfg(test)]
mod tests;
