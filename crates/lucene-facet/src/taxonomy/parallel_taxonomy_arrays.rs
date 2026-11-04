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
    /// The values as [`IntChunks`], when the array keeps them that way: a
    /// children/siblings walk then reads them without a virtual call per
    /// step, as the JIT inlines Java's monomorphic `get`. `None` (the
    /// default) walks through [`IntArray::get`].
    fn chunks(&self) -> Option<IntChunks<'_>> {
        None
    }
}

/// An [`IntArray`]'s values as chunks of `1 << bits` ints, every chunk but
/// the last full. Only this crate's arrays build one (its constructor is
/// crate-private), so the shape is the taxonomy arrays' own, never an
/// outside implementation's.
#[derive(Debug, Clone, Copy)]
pub struct IntChunks<'a> {
    bits: u32,
    chunks: &'a [std::sync::Arc<[i32]>],
}

impl<'a> IntChunks<'a> {
    /// Chunks of `1 << bits` ints; `bits` below `usize::BITS`.
    pub(crate) fn new(bits: u32, chunks: &'a [std::sync::Arc<[i32]>]) -> Self {
        IntChunks { bits, chunks }
    }

    /// The one chunk of an array no longer than a chunk.
    fn single(&self) -> Option<&'a [i32]> {
        match self.chunks {
            [only] => Some(only),
            _ => None,
        }
    }

    /// `get(i)`, panicking as [`IntArray::get`] does.
    #[inline]
    pub fn get(&self, i: i32) -> i32 {
        let i = usize::try_from(i).expect("negative taxonomy array index");
        // ARITH: a shift of a usize by fewer bits than it has, and its mask.
        #[allow(clippy::arithmetic_side_effects)]
        let (chunk, at) = (i >> self.bits, i & ((1usize << self.bits) - 1));
        self.chunks[chunk][at]
    }
}

/// An [`IntArray`] read through its [`IntChunks`] when it has them -- and
/// as one slice when they are one chunk (a taxonomy under 8192 ordinals),
/// one dependent load per walk step instead of two.
#[derive(Clone, Copy)]
pub(crate) enum IntArrayReader<'a> {
    Single(&'a [i32]),
    Chunks(IntChunks<'a>),
    Dyn(&'a dyn IntArray),
}

impl<'a> IntArrayReader<'a> {
    pub(crate) fn new(array: &'a dyn IntArray) -> Self {
        match array.chunks() {
            Some(chunks) => match chunks.single() {
                Some(single) => IntArrayReader::Single(single),
                None => IntArrayReader::Chunks(chunks),
            },
            None => IntArrayReader::Dyn(array),
        }
    }

    #[inline]
    pub(crate) fn get(&self, i: i32) -> i32 {
        match self {
            IntArrayReader::Single(values) => {
                values[usize::try_from(i).expect("negative taxonomy array index")]
            }
            IntArrayReader::Chunks(chunks) => chunks.get(i),
            IntArrayReader::Dyn(array) => array.get(i),
        }
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug)]
    struct Plain(Vec<i32>);

    impl IntArray for Plain {
        fn get(&self, i: i32) -> i32 {
            self.0[usize::try_from(i).unwrap()]
        }
        fn length(&self) -> i32 {
            self.0.len() as i32
        }
    }

    /// An array without chunks is read through `get`; chunks of two ints
    /// split an index into chunk and offset.
    #[test]
    fn readers_answer_get() {
        let plain = Plain(vec![4, 5, 6]);
        assert!(plain.chunks().is_none());
        let reader = IntArrayReader::new(&plain);
        assert!(matches!(reader, IntArrayReader::Dyn(_)));
        assert_eq!((reader.get(0), reader.get(2)), (4, 6));
        let chunks: Vec<std::sync::Arc<[i32]>> = vec![vec![1, 2].into(), vec![3].into()];
        let c = IntChunks::new(1, &chunks);
        assert_eq!((c.get(0), c.get(1), c.get(2)), (1, 2, 3));
        assert!(c.single().is_none());
        assert_eq!(IntChunks::new(1, &chunks[..1]).single(), Some(&[1, 2][..]));
    }
}
