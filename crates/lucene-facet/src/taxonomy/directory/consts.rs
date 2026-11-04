//! `Consts`: the taxonomy index's field names.

/// `Consts.FULL`: the category path, as both an indexed term and a binary
/// doc value.
pub const FULL: &str = "$full_path$";

/// `Consts.FIELD_PARENT_ORDINAL_NDV`: the parent ordinal, a numeric doc
/// value.
pub const FIELD_PARENT_ORDINAL_NDV: &str = "$parent_ndv$";
