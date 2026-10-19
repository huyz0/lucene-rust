//! Port of lucene-sandbox's `org.apache.lucene.sandbox.codecs.quantization`:
//! `KMeans` clustering of float vectors and the reservoir-sampling
//! `SampleReader` it clusters a large set through.
//!
//! The arithmetic is Java's, in Java's order and widths -- `float` distances
//! summed into a `double`, the Kahan-compensated `float` centroid sums of the
//! final assignment pass, `float / int` means -- and the random draws are
//! `java.util.Random`'s, so a clustering reproduces Lucene's bit for bit
//! wherever the distance kernel does (see [`crate::vectors::square_distance`]:
//! every dimension up to 16, beyond that within the last ulp, as Lucene's own
//! scalar and Panama kernels differ).
//!
//! FORGY initialization takes its centroids in the order an `IntHashSet`
//! iterates the chosen ordinals, which in Java follows a process-wide seed
//! (`HashContainers.ITERATION_SEED`) and so differs between runs; this port
//! takes them in the order they were drawn. The *set* of initial centroids,
//! and so the clustering up to a renumbering, is the same.

mod sample_reader;

#[cfg(test)]
mod tests;

pub use sample_reader::{reservoir_sample, reservoir_sample_from_array, SampleReader};

use lucene_store::error::Error;
use lucene_util::java_random::JavaRandom;

use crate::hnsw::NeighborQueue;
use crate::vectors::{square_distance, FloatVectorValues};

/// A source of float vectors by ordinal: the part of Lucene's
/// `FloatVectorValues` that `KMeans` and `SampleReader` use.
pub trait FloatVectorSource {
    /// `size()`: the number of vectors.
    fn size(&self) -> usize;
    /// `dimension()`.
    fn dimension(&self) -> usize;
    /// `vectorValue(ord)`: borrowed from the source when it holds the vector
    /// decoded, otherwise decoded into `scratch` (`dimension()` long).
    ///
    /// # Errors
    ///
    /// An ordinal out of range, or the source's own read error.
    fn vector_value<'s>(&'s self, ord: usize, scratch: &'s mut [f32]) -> Result<&'s [f32], Error>;
}

/// `FloatVectorValues.fromFloats(vectors, dimension)`.
#[derive(Debug, Clone, Copy)]
pub struct InMemoryFloatVectors<'a> {
    vectors: &'a [Vec<f32>],
    dimension: usize,
}

impl<'a> InMemoryFloatVectors<'a> {
    /// `FloatVectorValues.fromFloats(vectors, dimension)`.
    pub fn new(vectors: &'a [Vec<f32>], dimension: usize) -> Self {
        InMemoryFloatVectors { vectors, dimension }
    }
}

impl FloatVectorSource for InMemoryFloatVectors<'_> {
    fn size(&self) -> usize {
        self.vectors.len()
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn vector_value<'s>(&'s self, ord: usize, _: &'s mut [f32]) -> Result<&'s [f32], Error> {
        self.vectors.get(ord).map(Vec::as_slice).ok_or_else(|| {
            Error::IllegalArgument(format!(
                "ordinal {ord} out of bounds for {}",
                self.vectors.len()
            ))
        })
    }
}

impl FloatVectorSource for FloatVectorValues<'_> {
    fn size(&self) -> usize {
        usize::try_from(FloatVectorValues::size(self)).unwrap_or(0)
    }

    fn dimension(&self) -> usize {
        FloatVectorValues::dimension(self)
    }

    fn vector_value<'s>(&'s self, ord: usize, scratch: &'s mut [f32]) -> Result<&'s [f32], Error> {
        let ord = i32::try_from(ord)
            .map_err(|_| Error::IllegalArgument(format!("ordinal {ord} out of bounds")))?;
        self.vector_into(ord, scratch).map_err(|e| match e {
            crate::vectors::Error::Store(e) => e,
            other => Error::IllegalArgument(other.to_string()),
        })?;
        Ok(scratch)
    }
}

/// `KMeans.MAX_NUM_CENTROIDS`: `Short.MAX_VALUE`.
pub const MAX_NUM_CENTROIDS: usize = 32_767;
/// `KMeans.DEFAULT_RESTARTS`.
pub const DEFAULT_RESTARTS: usize = 5;
/// `KMeans.DEFAULT_ITRS`.
pub const DEFAULT_ITRS: usize = 10;
/// `KMeans.DEFAULT_SAMPLE_SIZE`.
pub const DEFAULT_SAMPLE_SIZE: usize = 100_000;
/// `KMeans.cluster`'s default seed.
pub const DEFAULT_SEED: i64 = 42;

/// `KMeans.KmeansInitializationMethod`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KmeansInitializationMethod {
    /// `FORGY`: distinct random vectors.
    Forgy,
    /// `RESERVOIR_SAMPLING`: a reservoir over all vectors.
    ReservoirSampling,
    /// `PLUS_PLUS`: k-means++.
    PlusPlus,
}

/// `KMeans.Results`.
#[derive(Debug, Clone, PartialEq)]
pub struct Results {
    /// `centroids()`.
    pub centroids: Vec<Vec<f32>>,
    /// `vectorCentroids()`: each vector's centroid, `None` when not asked
    /// for.
    pub vector_centroids: Option<Vec<i16>>,
}

/// The expert arguments of `KMeans.cluster`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ClusterOptions {
    /// `assignCentroidsToVectors`.
    pub assign_centroids_to_vectors: bool,
    /// `seed`.
    pub seed: i64,
    /// `initializationMethod`.
    pub initialization_method: KmeansInitializationMethod,
    /// `normalizeCenters`: spherical k-means, for COSINE.
    pub normalize_centers: bool,
    /// `restarts`.
    pub restarts: usize,
    /// `iters`.
    pub iters: usize,
    /// `sampleSize`.
    pub sample_size: usize,
}

impl ClusterOptions {
    /// The options `KMeans.cluster(vectors, similarity, numClusters)` uses.
    pub fn defaults(cosine: bool) -> Self {
        ClusterOptions {
            assign_centroids_to_vectors: true,
            seed: DEFAULT_SEED,
            initialization_method: KmeansInitializationMethod::PlusPlus,
            normalize_centers: cosine,
            restarts: DEFAULT_RESTARTS,
            iters: DEFAULT_ITRS,
            sample_size: DEFAULT_SAMPLE_SIZE,
        }
    }
}

/// `KMeans.cluster(vectors, similarityFunction, numClusters)`: `cosine` is
/// `similarityFunction == COSINE` (whose vectors must be normalized).
///
/// # Errors
///
/// As [`cluster_with`].
pub fn cluster<V: FloatVectorSource + ?Sized>(
    vectors: &V,
    cosine: bool,
    num_clusters: usize,
) -> Result<Option<Results>, Error> {
    cluster_with(vectors, num_clusters, &ClusterOptions::defaults(cosine))
}

/// `KMeans.cluster(vectors, numClusters, assignCentroidsToVectors, seed,
/// initializationMethod, normalizeCenters, restarts, iters, sampleSize)`.
/// `None` for an empty vector set, as Java's `null`.
///
/// # Errors
///
/// Java's "\[numClusters\] must be between \[1\] and \[32767\]"; `restarts`
/// of zero where Java would compute centroids (Java: `null` centroids, an
/// NPE when assigning); a
/// k-means++ draw over vectors that all coincide with the centroids so far
/// (Java reads ordinal -1); and the source's read errors.
pub fn cluster_with<V: FloatVectorSource + ?Sized>(
    vectors: &V,
    num_clusters: usize,
    options: &ClusterOptions,
) -> Result<Option<Results>, Error> {
    let size = vectors.size();
    if size == 0 {
        return Ok(None);
    }
    if !(1..=MAX_NUM_CENTROIDS).contains(&num_clusters) {
        return Err(Error::IllegalArgument(format!(
            "[numClusters] must be between [1] and [{MAX_NUM_CENTROIDS}]"
        )));
    }
    let mut num_clusters = num_clusters;
    // adjust sampleSize and numClusters (`100 * numClusters` fits: <= 3_276_700)
    let mut sample_size = options.sample_size.max(num_clusters.saturating_mul(100));
    if sample_size > size {
        sample_size = size;
        let max_num_clusters = (sample_size / 100).max(1);
        num_clusters = num_clusters.min(max_num_clusters);
    }
    let random = JavaRandom::new(options.seed);
    let dimension = vectors.dimension();
    let mut centroids = if num_clusters == 1 {
        vec![vec![0.0f32; dimension]]
    } else if size <= sample_size {
        KMeans::new(vectors, num_clusters, random, options)
            .compute_centroids(options.normalize_centers)?
    } else {
        let sample = SampleReader::create(vectors, sample_size, options.seed)?;
        KMeans::new(&sample, num_clusters, random, options)
            .compute_centroids(options.normalize_centers)?
    };
    let mut vector_centroids = None;
    if options.assign_centroids_to_vectors {
        let mut assigned = vec![0i16; size];
        // Kahan summation for the final, whole-set pass.
        run_kmeans_step(
            vectors,
            &mut centroids,
            &mut assigned,
            true,
            options.normalize_centers,
        )?;
        vector_centroids = Some(assigned);
    }
    Ok(Some(Results {
        centroids,
        vector_centroids,
    }))
}

/// The `KMeans` instance: a (possibly sampled) vector set, the cluster
/// count and the shared `Random`.
struct KMeans<'a, V: ?Sized> {
    vectors: &'a V,
    num_vectors: usize,
    num_centroids: usize,
    random: JavaRandom,
    initialization_method: KmeansInitializationMethod,
    restarts: usize,
    iters: usize,
}

impl<'a, V: FloatVectorSource + ?Sized> KMeans<'a, V> {
    fn new(
        vectors: &'a V,
        num_centroids: usize,
        random: JavaRandom,
        options: &ClusterOptions,
    ) -> Self {
        KMeans {
            vectors,
            num_vectors: vectors.size(),
            num_centroids,
            random,
            initialization_method: options.initialization_method,
            restarts: options.restarts,
            iters: options.iters,
        }
    }

    /// `random.nextInt(numVectors)`.
    fn next_ord(&mut self, bound: usize) -> usize {
        // `bound` is a vector count or cluster count, both checked to fit an
        // `int` by the sampling/argument checks; never 0 here.
        self.random
            .next_int_bounded(i32::try_from(bound).unwrap_or(i32::MAX)) as usize
    }

    fn copy_of(&self, ord: usize, scratch: &mut [f32]) -> Result<Vec<f32>, Error> {
        Ok(self.vectors.vector_value(ord, scratch)?.to_vec())
    }

    fn compute_centroids(&mut self, normalize_centers: bool) -> Result<Vec<Vec<f32>>, Error> {
        // Java runs no restart and returns `null` centroids (an NPE once
        // they are assigned); one centroid never gets here, as in Java.
        if self.restarts == 0 {
            return Err(Error::IllegalArgument(
                "[restarts] must be at least [1]".to_string(),
            ));
        }
        let mut vector_centroids = vec![0i16; self.num_vectors];
        let mut min_squared_dist = f64::MAX;
        let mut squared_dist = 0.0f64;
        let mut best_centroids = None;
        for _ in 0..self.restarts {
            let mut centroids = match self.initialization_method {
                KmeansInitializationMethod::Forgy => self.initialize_forgy()?,
                KmeansInitializationMethod::ReservoirSampling => {
                    self.initialize_reservoir_sampling()?
                }
                KmeansInitializationMethod::PlusPlus => self.initialize_plus_plus()?,
            };
            let mut prev_squared_dist = f64::MAX;
            for _ in 0..self.iters {
                squared_dist = run_kmeans_step(
                    self.vectors,
                    &mut centroids,
                    &mut vector_centroids,
                    false,
                    normalize_centers,
                )?;
                // Check for convergence
                if prev_squared_dist <= squared_dist + 1e-6 {
                    break;
                }
                prev_squared_dist = squared_dist;
            }
            if squared_dist < min_squared_dist {
                min_squared_dist = squared_dist;
                best_centroids = Some(centroids);
            }
        }
        // Unreachable with `restarts >= 1` (checked): the first restart's
        // distance is a finite sum (never NaN -- a NaN distance never wins
        // the `<`), so it beats `Double.MAX_VALUE`. Java would return `null`.
        best_centroids.ok_or_else(|| {
            Error::IllegalArgument("k-means produced no finite clustering".to_string())
        })
    }

    /// FORGY: `numCentroids` distinct random vectors, in the order drawn
    /// (Java: the `IntHashSet`'s run-dependent iteration order).
    fn initialize_forgy(&mut self) -> Result<Vec<Vec<f32>>, Error> {
        let mut seen = vec![false; self.num_vectors];
        let mut selection = Vec::with_capacity(self.num_centroids);
        while selection.len() < self.num_centroids {
            let ord = self.next_ord(self.num_vectors);
            if !seen[ord] {
                seen[ord] = true;
                selection.push(ord);
            }
        }
        let mut scratch = vec![0.0f32; self.vectors.dimension()];
        selection
            .into_iter()
            .map(|ord| self.copy_of(ord, &mut scratch))
            .collect()
    }

    /// RESERVOIR_SAMPLING.
    fn initialize_reservoir_sampling(&mut self) -> Result<Vec<Vec<f32>>, Error> {
        let mut initial: Vec<Vec<f32>> = vec![Vec::new(); self.num_centroids];
        let mut scratch = vec![0.0f32; self.vectors.dimension()];
        for index in 0..self.num_vectors {
            if index < self.num_centroids {
                initial[index] = self.copy_of(index, &mut scratch)?;
            } else {
                if self.random.next_double() < self.num_centroids as f64 * (1.0 / index as f64) {
                    let c = self.next_ord(self.num_centroids);
                    initial[c] = self.copy_of(index, &mut scratch)?;
                }
            }
        }
        Ok(initial)
    }

    /// PLUS_PLUS: k-means++.
    fn initialize_plus_plus(&mut self) -> Result<Vec<Vec<f32>>, Error> {
        let mut initial: Vec<Vec<f32>> = Vec::with_capacity(self.num_centroids);
        let mut scratch = vec![0.0f32; self.vectors.dimension()];
        // Choose the first centroid uniformly at random
        let first = self.next_ord(self.num_vectors);
        initial.push(self.copy_of(first, &mut scratch)?);
        // Distances of each point to the nearest centroid
        let mut min_distances = vec![f32::MAX; self.num_vectors];
        for _ in 1..self.num_centroids {
            let last = initial.last().expect("the first centroid is pushed above");
            // Update distances with the new centroid
            let mut total_sum = 0.0f64;
            for (j, min) in min_distances.iter_mut().enumerate() {
                let dist = square_distance(self.vectors.vector_value(j, &mut scratch)?, last);
                if dist < *min {
                    *min = dist;
                }
                total_sum += f64::from(*min);
            }
            // Randomly select next centroid
            let r = total_sum * self.random.next_double();
            let mut cumulative_sum = 0.0f64;
            let mut next = None;
            for (j, &min) in min_distances.iter().enumerate() {
                cumulative_sum += f64::from(min);
                if cumulative_sum >= r && min > 0.0 {
                    next = Some(j);
                    break;
                }
            }
            let next = next.ok_or_else(|| {
                Error::IllegalArgument(
                    "k-means++ found no vector away from the centroids chosen so far (Java reads ordinal -1)"
                        .to_string(),
                )
            })?;
            let value = self.copy_of(next, &mut scratch)?;
            initial.push(value);
        }
        Ok(initial)
    }
}

/// `KMeans.runKMeansStep`: assign every vector to its nearest centroid
/// (the first of equals), move each centroid to its vectors' mean (Kahan
/// sums when `use_kahan_summation`), refill empty ones from outliers, and
/// normalize if asked. Returns the summed squared distances (0 for one
/// centroid, whose distances Java never computes).
// ARITH: float/double accumulation (no overflow trap) and per-centroid
// counts bounded by the vector count.
#[allow(clippy::arithmetic_side_effects)]
fn run_kmeans_step<V: FloatVectorSource + ?Sized>(
    vectors: &V,
    centroids: &mut [Vec<f32>],
    doc_centroids: &mut [i16],
    use_kahan_summation: bool,
    normalize_centroids: bool,
) -> Result<f64, Error> {
    let num_centroids = centroids.len();
    let dim = centroids[0].len();
    let mut new_centroids = vec![vec![0.0f32; dim]; num_centroids];
    let mut new_centroid_size = vec![0i32; num_centroids];
    let mut compensations = if use_kahan_summation {
        vec![vec![0.0f32; dim]; num_centroids]
    } else {
        Vec::new()
    };
    let mut scratch = vec![0.0f32; vectors.dimension()];
    let mut sum_squared_dist = 0.0f64;
    for (doc, slot) in doc_centroids.iter_mut().enumerate().take(vectors.size()) {
        let vector = vectors.vector_value(doc, &mut scratch)?;
        let mut best = 0usize;
        if num_centroids > 1 {
            let mut min_squared_dist = f32::MAX;
            for (c, centroid) in centroids.iter().enumerate() {
                let d = square_distance(centroid, vector);
                if d < min_squared_dist {
                    best = c;
                    min_squared_dist = d;
                }
            }
            sum_squared_dist += f64::from(min_squared_dist);
        }
        new_centroid_size[best] += 1;
        let sums = &mut new_centroids[best];
        if use_kahan_summation {
            let comp = &mut compensations[best];
            for ((s, c), &x) in sums.iter_mut().zip(comp.iter_mut()).zip(vector) {
                let y = x - *c;
                let t = *s + y;
                *c = (t - *s) - y;
                *s = t;
            }
        } else {
            for (s, &x) in sums.iter_mut().zip(vector) {
                *s += x;
            }
        }
        // `best < num_centroids <= MAX_NUM_CENTROIDS`, a `short`.
        *slot = best as i16;
    }
    let mut unassigned = Vec::new();
    for (c, (centroid, sums)) in centroids.iter_mut().zip(&new_centroids).enumerate() {
        let count = new_centroid_size[c];
        if count > 0 {
            // Java's `float / int`: the count widened to `float`.
            let count = count as f32;
            for (dst, &s) in centroid.iter_mut().zip(sums) {
                *dst = s / count;
            }
        } else {
            unassigned.push(c);
        }
    }
    if !unassigned.is_empty() {
        assign_centroids(vectors, centroids, &unassigned)?;
    }
    if normalize_centroids {
        for centroid in centroids.iter_mut() {
            // A zero centroid is left alone (`throwOnZero` false).
            let _ = lucene_util::vector_util::l2normalize(centroid, false);
        }
    }
    Ok(sum_squared_dist)
}

/// `KMeans.assignCentroids`: give each empty centroid one of the vectors
/// farthest from the centroids that did get vectors, farthest first being
/// the *last* popped -- a min-heap of the `unassigned.len()` largest
/// (vector, distance-to-one-assigned-centroid) pairs, popped smallest first.
/// A vector far from several centroids holds several slots, as in Java.
fn assign_centroids<V: FloatVectorSource + ?Sized>(
    vectors: &V,
    centroids: &mut [Vec<f32>],
    unassigned: &[usize],
) -> Result<(), Error> {
    let assigned: Vec<usize> = (0..centroids.len())
        .filter(|i| !unassigned.contains(i))
        .collect();
    let mut queue = NeighborQueue::new(unassigned.len(), false);
    let mut scratch = vec![0.0f32; vectors.dimension()];
    for i in 0..vectors.size() {
        let vector = vectors.vector_value(i, &mut scratch)?;
        // `i < size`, which the caller's `i16` assignment array bounds
        // far below `i32::MAX`... but the full set may not be; checked.
        let node = i32::try_from(i)
            .map_err(|_| Error::IllegalArgument(format!("ordinal {i} exceeds an int")))?;
        for &c in &assigned {
            queue.insert_with_overflow(node, square_distance(&centroids[c], vector));
        }
    }
    for &c in unassigned {
        let node = queue.top_node() as usize;
        centroids[c] = vectors.vector_value(node, &mut scratch)?.to_vec();
        queue.pop();
    }
    Ok(())
}
