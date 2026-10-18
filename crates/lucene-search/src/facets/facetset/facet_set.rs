//! `FacetSet` and its `long`/`int`/`float`/`double` forms,
//! `FacetSetsField` and `FacetSetDecoder`.

use lucene_index::document::BinaryDocValuesField;
use lucene_util::numeric_utils::{
    double_to_sortable_long, float_to_sortable_int, int_to_sortable_bytes, long_to_sortable_bytes,
    sortable_bytes_to_int, sortable_bytes_to_long,
};

use crate::{Error, Result};

/// `FacetSet`: one tuple of `dims` values.
pub trait FacetSet {
    /// `dims`.
    fn dims(&self) -> usize;

    /// `getComparableValues()`: the values as comparable `long`s.
    fn get_comparable_values(&self) -> Vec<i64>;

    /// `packValues(buf, start)`: the values packed into `buf` from `start`;
    /// returns the bytes written. The default packs each comparable value
    /// as `LongPoint.encodeDimension`.
    ///
    /// # Panics
    /// When `buf` is too short (Java: an index out of bounds).
    fn pack_values(&self, buf: &mut [u8], start: usize) -> usize {
        let values = self.get_comparable_values();
        for (i, &v) in values.iter().enumerate() {
            long_to_sortable_bytes(v, buf, start + i * 8);
        }
        values.len() * 8
    }

    /// `sizePackedBytes()`.
    fn size_packed_bytes(&self) -> usize {
        self.dims() * 8
    }

    /// The Java class (`getClass()`), as `FacetSetsField.create`'s
    /// same-type check and message print it.
    fn class_name(&self) -> &str;
}

/// The constructors' `validateValuesAndGetNumDims`.
fn validate<T>(values: &[T]) -> Result<()> {
    if values.is_empty() {
        return Err(Error::IllegalArgument(
            "values cannot be null or empty".into(),
        ));
    }
    Ok(())
}

macro_rules! facet_set {
    ($(#[$doc:meta])* $name:ident, $t:ty, $class:literal) => {
        $(#[$doc])*
        #[derive(Debug, Clone, PartialEq)]
        pub struct $name {
            /// `values`.
            pub values: Vec<$t>,
        }

        impl $name {
            #[doc = concat!("`new ", stringify!($name), "(values...)`.")]
            ///
            /// # Errors
            /// [`Error::IllegalArgument`] for no values.
            pub fn new(values: Vec<$t>) -> Result<Self> {
                validate(&values)?;
                Ok($name { values })
            }

            const CLASS: &'static str = $class;
        }
    };
}

facet_set!(
    /// `LongFacetSet`.
    LongFacetSet,
    i64,
    "org.apache.lucene.facet.facetset.LongFacetSet"
);
facet_set!(
    /// `IntFacetSet`: packed as `IntPoint`s.
    IntFacetSet,
    i32,
    "org.apache.lucene.facet.facetset.IntFacetSet"
);
facet_set!(
    /// `FloatFacetSet`: packed as `FloatPoint`s (sortable `int`s).
    FloatFacetSet,
    f32,
    "org.apache.lucene.facet.facetset.FloatFacetSet"
);
facet_set!(
    /// `DoubleFacetSet`: packed as sortable `long`s.
    DoubleFacetSet,
    f64,
    "org.apache.lucene.facet.facetset.DoubleFacetSet"
);

impl FacetSet for LongFacetSet {
    fn dims(&self) -> usize {
        self.values.len()
    }
    fn get_comparable_values(&self) -> Vec<i64> {
        self.values.clone()
    }
    fn class_name(&self) -> &str {
        Self::CLASS
    }
}

impl FacetSet for IntFacetSet {
    fn dims(&self) -> usize {
        self.values.len()
    }
    fn get_comparable_values(&self) -> Vec<i64> {
        self.values.iter().map(|&v| i64::from(v)).collect()
    }
    fn pack_values(&self, buf: &mut [u8], start: usize) -> usize {
        for (i, &v) in self.values.iter().enumerate() {
            int_to_sortable_bytes(v, buf, start + i * 4);
        }
        self.values.len() * 4
    }
    fn size_packed_bytes(&self) -> usize {
        self.values.len() * 4
    }
    fn class_name(&self) -> &str {
        Self::CLASS
    }
}

impl FacetSet for FloatFacetSet {
    fn dims(&self) -> usize {
        self.values.len()
    }
    fn get_comparable_values(&self) -> Vec<i64> {
        self.values
            .iter()
            .map(|&v| i64::from(float_to_sortable_int(v)))
            .collect()
    }
    fn pack_values(&self, buf: &mut [u8], start: usize) -> usize {
        for (i, &v) in self.values.iter().enumerate() {
            int_to_sortable_bytes(float_to_sortable_int(v), buf, start + i * 4);
        }
        self.values.len() * 4
    }
    fn size_packed_bytes(&self) -> usize {
        self.values.len() * 4
    }
    fn class_name(&self) -> &str {
        Self::CLASS
    }
}

impl FacetSet for DoubleFacetSet {
    fn dims(&self) -> usize {
        self.values.len()
    }
    fn get_comparable_values(&self) -> Vec<i64> {
        self.values
            .iter()
            .map(|&v| double_to_sortable_long(v))
            .collect()
    }
    fn class_name(&self) -> &str {
        Self::CLASS
    }
}

/// `FacetSetsField`: a document's facet sets as one `BINARY` doc value --
/// the dimension count (`IntPoint`-encoded), then each set packed.
#[derive(Debug)]
pub struct FacetSetsField;

impl FacetSetsField {
    /// `FacetSetsField.create(name, facetSets...)`.
    ///
    /// # Errors
    /// [`Error::IllegalArgument`] for no sets, and for sets of different
    /// dimensions or types.
    pub fn create(name: &str, facet_sets: &[&dyn FacetSet]) -> Result<BinaryDocValuesField> {
        let Some(first) = facet_sets.first() else {
            return Err(Error::IllegalArgument(
                "FacetSets cannot be null or empty!".into(),
            ));
        };
        let num_dims = first.dims();
        let expected_class = first.class_name();
        let mut buf = vec![0u8; 4 + first.size_packed_bytes() * facet_sets.len()];
        int_to_sortable_bytes(i32::try_from(num_dims).unwrap_or(i32::MAX), &mut buf, 0);
        let mut offset = 4usize;
        for fs in facet_sets {
            if fs.dims() != num_dims {
                return Err(Error::IllegalArgument(format!(
                    "All FacetSets must have the same number of dimensions. Expected {num_dims} \
                     found {}",
                    fs.dims()
                )));
            }
            if fs.class_name() != expected_class {
                return Err(Error::IllegalArgument(format!(
                    "All FacetSets must be the same type. Expected class {expected_class} found \
                     class {}",
                    fs.class_name()
                )));
            }
            offset += fs.pack_values(&mut buf, offset);
        }
        buf.truncate(offset);
        Ok(BinaryDocValuesField::new(name, buf))
    }
}

/// `FacetSetDecoder`: decodes the set at `start` of `bytes` into `dest`
/// (one value per dimension), returning the bytes read.
pub type FacetSetDecoder = dyn Fn(&[u8], usize, &mut [i64]) -> Result<usize> + Send + Sync;

/// A packed value too short for the decoder.
fn truncated(len: usize, start: usize, need: usize) -> Error {
    Error::IllegalState(format!(
        "facet set of {need} bytes at {start} runs past the {len}-byte value"
    ))
}

/// `FacetSetDecoder.decodeLongs`: `LongPoint.unpack`.
///
/// # Errors
/// [`Error::IllegalState`] when `bytes` ends before the set does.
pub fn decode_longs(bytes: &[u8], start: usize, dest: &mut [i64]) -> Result<usize> {
    let need = dest.len() * 8;
    if start.checked_add(need).is_none_or(|end| end > bytes.len()) {
        return Err(truncated(bytes.len(), start, need));
    }
    for (i, d) in dest.iter_mut().enumerate() {
        *d = sortable_bytes_to_long(bytes, start + i * 8);
    }
    Ok(need)
}

/// `FacetSetDecoder.decodeInts`: each dimension as `IntPoint.decodeDimension`.
///
/// # Errors
/// [`Error::IllegalState`] when `bytes` ends before the set does.
pub fn decode_ints(bytes: &[u8], start: usize, dest: &mut [i64]) -> Result<usize> {
    let need = dest.len() * 4;
    if start.checked_add(need).is_none_or(|end| end > bytes.len()) {
        return Err(truncated(bytes.len(), start, need));
    }
    for (i, d) in dest.iter_mut().enumerate() {
        *d = i64::from(sortable_bytes_to_int(bytes, start + i * 4));
    }
    Ok(need)
}

#[cfg(test)]
mod tests;
