//! Port of `org.apache.lucene.sandbox.codecs.quantization.SampleReader`: a
//! view over a subset of another vector source's ordinals, chosen by
//! reservoir sampling.

use lucene_store::error::Error;
use lucene_util::java_random::JavaRandom;

use super::FloatVectorSource;

/// `SampleReader`: ordinal `i` reads the origin's ordinal `samples[i]`.
///
/// Java's `copy()`, `ordToDoc`, `getAcceptOrds` throw "Not supported" and
/// `getSlice()` hands out the origin's `IndexInput`; this port's
/// [`FloatVectorSource`] has none of the four, so they are absent rather
/// than present-and-throwing.
#[derive(Debug)]
pub struct SampleReader<'a, V: ?Sized> {
    origin: &'a V,
    samples: Vec<usize>,
}

impl<'a, V: FloatVectorSource + ?Sized> SampleReader<'a, V> {
    /// `SampleReader.createSampleReader(origin, k, seed)`.
    ///
    /// # Errors
    ///
    /// `k` or the origin's size beyond `i32::MAX`, which Java's `int`
    /// arguments cannot express.
    pub fn create(origin: &'a V, k: usize, seed: i64) -> Result<Self, Error> {
        let samples = reservoir_sample(origin.size(), k, seed)?;
        Ok(SampleReader { origin, samples })
    }

    /// The origin ordinal sample `ord` reads.
    pub fn origin_ord(&self, ord: usize) -> usize {
        self.samples[ord]
    }
}

impl<V: FloatVectorSource + ?Sized> FloatVectorSource for SampleReader<'_, V> {
    fn size(&self) -> usize {
        self.samples.len()
    }

    fn dimension(&self) -> usize {
        self.origin.dimension()
    }

    fn vector_value<'s>(&'s self, ord: usize, scratch: &'s mut [f32]) -> Result<&'s [f32], Error> {
        let target = *self.samples.get(ord).ok_or_else(|| {
            Error::IllegalArgument(format!("ordinal {ord} out of bounds for {}", self.size()))
        })?;
        self.origin.vector_value(target, scratch)
    }
}

fn java_int(v: usize, what: &str) -> Result<i32, Error> {
    i32::try_from(v).map_err(|_| Error::IllegalArgument(format!("{what} {v} exceeds an int")))
}

/// `SampleReader.reservoirSample(n, k, seed)`: `k` of `0..n`, Algorithm R
/// over `java.util.Random(seed)`.
///
/// Java allocates `int[k]` and fills it with `0..k` whatever `n` is, so
/// `k > n` returns ordinals past `n`; kept (`KMeans` never asks for that).
///
/// # Errors
///
/// `n` or `k` beyond `i32::MAX`.
pub fn reservoir_sample(n: usize, k: usize, seed: i64) -> Result<Vec<usize>, Error> {
    java_int(n, "n")?;
    java_int(k, "k")?;
    let mut rnd = JavaRandom::new(seed);
    let mut reservoir: Vec<usize> = (0..k).collect();
    for i in k..n {
        // `i + 1 <= n <= i32::MAX`, checked above.
        let bound = java_int(i, "n")?.saturating_add(1);
        let j = rnd.next_int_bounded(bound) as usize;
        if j < k {
            reservoir[j] = i;
        }
    }
    Ok(reservoir)
}

/// `SampleReader.reservoirSampleFromArray(origin, k, seed)`: `origin`
/// itself when `k >= origin.len()`.
///
/// # Errors
///
/// `origin.len()` beyond `i32::MAX`.
pub fn reservoir_sample_from_array(origin: &[i32], k: usize, seed: i64) -> Result<Vec<i32>, Error> {
    java_int(origin.len(), "origin length")?;
    let mut rnd = JavaRandom::new(seed);
    if k >= origin.len() {
        return Ok(origin.to_vec());
    }
    let mut reservoir = origin[..k].to_vec();
    for (i, &v) in origin.iter().enumerate().skip(k) {
        let bound = java_int(i, "origin length")?.saturating_add(1);
        let j = rnd.next_int_bounded(bound) as usize;
        if j < k {
            reservoir[j] = v;
        }
    }
    Ok(reservoir)
}
