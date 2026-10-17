//! `ParallelTaxonomyArrays`: three parallel arrays over the ordinals --
//! `parents[i]`, `children[i]` (the youngest child, the one with the largest
//! ordinal) and `siblings[i]` (the next older sibling) -- with
//! [`super::INVALID_ORDINAL`] for "none".

/// `ParallelTaxonomyArrays.IntArray`.
pub trait IntArray: std::fmt::Debug + Send + Sync {
    /// `get(i)`. Panics for `i` outside `0..length()`, Java's
    /// `ArrayIndexOutOfBoundsException`.
    fn get(&self, i: i32) -> i32;
    /// `length()`.
    fn length(&self) -> i32;
}

/// `ParallelTaxonomyArrays`.
pub trait ParallelTaxonomyArrays: std::fmt::Debug + Send + Sync {
    /// `parents()`.
    fn parents(&self) -> &dyn IntArray;
    /// `children()`.
    fn children(&self) -> &dyn IntArray;
    /// `siblings()`.
    fn siblings(&self) -> &dyn IntArray;
}
