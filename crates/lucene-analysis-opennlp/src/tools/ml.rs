//! `opennlp.tools.ml`: the event models a model package holds and the beam
//! search the sequence taggers run over them.
//!
//! * [`MaxentModel`]: `GenericModelReader` over the `.model` artifact --
//!   `DataInputStream` big-endian ints and doubles and modified-UTF-8
//!   strings (`ModelParameterChunker`'s `CHUNKED-MODEL-PARAMS:` strings
//!   joined) -- for the two model types OpenNLP's trainers write by default,
//!   `GIS` (maxent: `GISModel.eval`, a uniform log prior, `StrictMath.exp`,
//!   normalised) and `Perceptron` (`PerceptronModel.eval`: scores scaled by
//!   the largest magnitude, then the same). `QN` and `NaiveBayes` models are
//!   refused (`UnsupportedOperationException`): their trainers are opt-in
//!   and the port does not carry their evaluation.
//! * [`BeamSearch`] and its `Sequence`s, over a port of
//!   `java.util.PriorityQueue`'s binary heap, so that equal scores are
//!   ranked in Java's order.
//!
//! Differs: an outcome id past the model's outcomes in a parameter pattern
//! that some predicate uses is refused when the model loads, whether or not
//! that predicate is ever seen; Java loads it and throws an
//! `ArrayIndexOutOfBoundsException` only when such a context is evaluated.
//! A pattern no predicate uses is not checked, as in Java.

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

use lucene_analysis::factory::{FactoryError, JavaException};
use lucene_analysis::AnalysisError;
use lucene_util::strict_math;

use super::text::{self, JString};

/// A multiply-rotate hasher (FxHash's) for the feature maps: the keys are
/// short strings, looked up once per feature of every evaluation, where
/// Java's `String.hashCode` is cached on the string. It is unseeded, so a
/// model built to collide its predicates loads and evaluates slowly (no
/// more: lookups stay correct) -- the model files are the user's own
/// choice, trusted as OpenNLP trusts them.
#[derive(Default, Clone, Copy)]
pub struct FxHasher {
    hash: u64,
}

const FX_SEED: u64 = 0x51_7c_c1_b7_27_22_0a_95;

impl FxHasher {
    #[inline]
    fn add(&mut self, word: u64) {
        self.hash = (self.hash.rotate_left(5) ^ word).wrapping_mul(FX_SEED);
    }
}

impl Hasher for FxHasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        let mut chunks = bytes.chunks_exact(8);
        for c in &mut chunks {
            self.add(u64::from_le_bytes([
                c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7],
            ]));
        }
        for &b in chunks.remainder() {
            self.add(u64::from(b));
        }
    }

    #[inline]
    fn write_u16(&mut self, i: u16) {
        self.add(u64::from(i));
    }

    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.add(i as u64);
    }

    #[inline]
    fn finish(&self) -> u64 {
        self.hash
    }
}

/// A `HashMap` over [`FxHasher`].
pub type FxHashMap<K, V> = HashMap<K, V, BuildHasherDefault<FxHasher>>;

fn io(message: impl Into<String>) -> FactoryError {
    FactoryError::new(JavaException::Io, message)
}

/// `DataInputStream` over a byte array, as `BinaryFileDataReader` reads.
pub struct DataReader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> DataReader<'a> {
    /// A reader at the start of `bytes`.
    pub fn new(bytes: &'a [u8]) -> Self {
        DataReader { bytes, pos: 0 }
    }

    /// The bytes not yet read.
    // ARITH: `pos <= bytes.len()` (only `take` advances it, after checking).
    #[allow(clippy::arithmetic_side_effects)]
    pub fn remaining(&self) -> usize {
        self.bytes.len() - self.pos
    }

    // ARITH: `n <= remaining()` is checked first.
    #[allow(clippy::arithmetic_side_effects)]
    fn take(&mut self, n: usize) -> Result<&'a [u8], FactoryError> {
        if n > self.remaining() {
            return Err(io("EOFException"));
        }
        let s = &self.bytes[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }

    /// `readInt()`.
    pub fn read_int(&mut self) -> Result<i32, FactoryError> {
        let b = self.take(4)?;
        Ok(i32::from_be_bytes([b[0], b[1], b[2], b[3]]))
    }

    /// `readDouble()`.
    pub fn read_double(&mut self) -> Result<f64, FactoryError> {
        let b = self.take(8)?;
        Ok(f64::from_be_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    /// `DataInputStream.readUTF()`: a 16-bit length, then that many bytes of
    /// modified UTF-8.
    // ARITH: offsets within the string's `len` bytes (each read checked).
    #[allow(clippy::arithmetic_side_effects)]
    fn read_raw_utf(&mut self) -> Result<JString, FactoryError> {
        let b = self.take(2)?;
        let len = usize::from(u16::from_be_bytes([b[0], b[1]]));
        let bytes = self.take(len)?;
        let malformed = |at: usize| {
            io(format!(
                "UTFDataFormatException: malformed input around byte {at}"
            ))
        };
        let partial = || io("UTFDataFormatException: malformed input: partial character at end");
        let mut out = Vec::with_capacity(len);
        let mut i = 0;
        while i < len {
            let c = bytes[i];
            match c >> 4 {
                0..=7 => {
                    out.push(u16::from(c));
                    i += 1;
                }
                12 | 13 => {
                    let c2 = *bytes.get(i + 1).ok_or_else(partial)?;
                    if c2 & 0xC0 != 0x80 {
                        return Err(malformed(i + 1));
                    }
                    out.push((u16::from(c & 0x1F) << 6) | u16::from(c2 & 0x3F));
                    i += 2;
                }
                14 => {
                    let (c2, c3) = match (bytes.get(i + 1), bytes.get(i + 2)) {
                        (Some(&c2), Some(&c3)) => (c2, c3),
                        _ => return Err(partial()),
                    };
                    if c2 & 0xC0 != 0x80 || c3 & 0xC0 != 0x80 {
                        return Err(malformed(i + 2));
                    }
                    out.push(
                        (u16::from(c & 0x0F) << 12)
                            | (u16::from(c2 & 0x3F) << 6)
                            | u16::from(c3 & 0x3F),
                    );
                    i += 3;
                }
                _ => return Err(malformed(i)),
            }
        }
        Ok(out)
    }

    /// `ModelParameterChunker.readUTF`: a `CHUNKED-MODEL-PARAMS:<n>` string
    /// is followed by `n` strings, joined.
    // ARITH: offsets within the in-memory string.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn read_utf(&mut self) -> Result<JString, FactoryError> {
        const SIGNATURE: &str = "CHUNKED-MODEL-PARAMS:";
        let data = self.read_raw_utf()?;
        let sig = text::units(SIGNATURE);
        if !data.starts_with(&sig) {
            return Ok(data);
        }
        // `data.replace(SIGNATURE, "")`, then `Integer.parseInt`.
        let mut count_text = Vec::new();
        let mut i = 0;
        while i < data.len() {
            if data[i..].starts_with(&sig) {
                i += sig.len();
            } else {
                count_text.push(data[i]);
                i += 1;
            }
        }
        let chunks = text::parse_int(&count_text).ok_or_else(|| {
            FactoryError::new(
                JavaException::NumberFormat,
                format!("For input string: \"{}\"", text::to_string(&count_text)),
            )
        })?;
        let mut out = Vec::new();
        for _ in 0..chunks {
            out.extend(self.read_raw_utf()?);
        }
        Ok(out)
    }

    /// A count read with `readInt`, for an array of elements at least
    /// `min_bytes` long each: Java's `NegativeArraySizeException` for a
    /// negative one, and an end of input (where Java would first try the
    /// allocation) for one the remaining bytes cannot hold.
    fn read_count(&mut self, min_bytes: usize) -> Result<usize, FactoryError> {
        let n = self.read_int()?;
        let n = usize::try_from(n).map_err(|_| {
            FactoryError::new(
                JavaException::Runtime,
                format!("NegativeArraySizeException: {n}"),
            )
        })?;
        if n.saturating_mul(min_bytes) > self.remaining() {
            return Err(io("EOFException"));
        }
        Ok(n)
    }
}

/// The two event model types the port evaluates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelType {
    /// `GISModel` (maxent).
    Gis,
    /// `PerceptronModel`.
    Perceptron,
}

/// `Context`: the outcomes a predicate has parameters for, and those
/// parameters.
#[derive(Debug, Clone)]
struct Context {
    outcomes: Vec<u32>,
    params: Vec<f64>,
}

/// `MaxentModel` as `GenericModelReader` builds it.
#[derive(Debug, Clone)]
pub struct MaxentModel {
    model_type: ModelType,
    outcomes: Vec<JString>,
    outcome_names: Vec<String>,
    /// Each predicate's `Context`, as the range of its parameters in
    /// `params` and of their outcomes in `context_outcomes` (the contexts
    /// flattened: two allocations for the model, where a `Context` per
    /// predicate was two each).
    pmap: FxHashMap<JString, (u32, u32)>,
    context_outcomes: Vec<u32>,
    params: Vec<f64>,
    /// `UniformPrior.r`: `StrictMath.log(1.0 / numOutcomes)`.
    log_prior: f64,
}

impl MaxentModel {
    /// `GenericModelReader.getModel()` over the artifact's bytes.
    // ARITH: `pid` is checked against the predicate count before it is used.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn read(bytes: &[u8]) -> Result<MaxentModel, FactoryError> {
        let mut r = DataReader::new(bytes);
        let model_type = text::to_string(&r.read_utf()?);
        let model_type = match model_type.as_str() {
            "GIS" => {
                // the correction constant and parameter, unused
                r.read_int()?;
                r.read_double()?;
                ModelType::Gis
            }
            "Perceptron" => ModelType::Perceptron,
            "QN" | "NaiveBayes" => {
                return Err(FactoryError::new(
                    JavaException::UnsupportedOperation,
                    format!("OpenNLP {model_type} models are not supported by this port (GIS and Perceptron are)"),
                ))
            }
            other => return Err(io(format!("Unknown model format: {other}"))),
        };
        // getOutcomes
        let n = r.read_count(2)?;
        let mut outcomes = Vec::with_capacity(n);
        for _ in 0..n {
            outcomes.push(r.read_utf()?);
        }
        // getOutcomePatterns
        let n = r.read_count(2)?;
        let mut patterns: Vec<Vec<i32>> = Vec::with_capacity(n);
        for _ in 0..n {
            let s = text::to_string(&r.read_utf()?);
            let mut ints = Vec::new();
            for tok in s.split(' ').filter(|t| !t.is_empty()) {
                let v = text::parse_int(&text::units(tok)).ok_or_else(|| {
                    FactoryError::new(
                        JavaException::NumberFormat,
                        format!("For input string: \"{tok}\""),
                    )
                })?;
                ints.push(v);
            }
            if ints.is_empty() {
                return Err(FactoryError::new(
                    JavaException::ArrayIndexOutOfBounds,
                    "Index 0 out of bounds for length 0",
                ));
            }
            patterns.push(ints);
        }
        // getPredicates
        let n_preds = r.read_count(2)?;
        let mut preds = Vec::with_capacity(n_preds);
        for _ in 0..n_preds {
            preds.push(r.read_utf()?);
        }
        // getParameters
        let mut contexts: Vec<Option<Context>> = vec![None; n_preds];
        let mut pid = 0usize;
        for pattern in &patterns {
            // a pattern no predicate uses is never read again (Java does not
            // look at its outcome ids either)
            if pattern[0] <= 0 {
                continue;
            }
            let pattern_outcomes = pattern[1..]
                .iter()
                .map(|&o| match u32::try_from(o) {
                    Ok(o) if (o as usize) < outcomes.len() => Ok(o),
                    _ => Err(FactoryError::new(
                        JavaException::ArrayIndexOutOfBounds,
                        format!(
                            "outcome {o} of a parameter pattern is not one of the model's {} outcomes",
                            outcomes.len()
                        ),
                    )),
                })
                .collect::<Result<Vec<u32>, _>>()?;
            for _ in 0..pattern[0] {
                if pid >= n_preds {
                    return Err(FactoryError::new(
                        JavaException::ArrayIndexOutOfBounds,
                        format!("Index {pid} out of bounds for length {n_preds}"),
                    ));
                }
                if pattern_outcomes.len().saturating_mul(8) > r.remaining() {
                    return Err(io("EOFException"));
                }
                let mut params = Vec::with_capacity(pattern_outcomes.len());
                for _ in 0..pattern_outcomes.len() {
                    params.push(r.read_double()?);
                }
                contexts[pid] = Some(Context {
                    outcomes: pattern_outcomes.clone(),
                    params,
                });
                pid += 1;
            }
        }
        let mut pmap = FxHashMap::default();
        let mut context_outcomes = Vec::new();
        let mut params = Vec::new();
        for (label, context) in preds.into_iter().zip(contexts) {
            match context {
                Some(c) => {
                    let start = params.len();
                    context_outcomes.extend_from_slice(&c.outcomes);
                    params.extend_from_slice(&c.params);
                    let range = match (u32::try_from(start), u32::try_from(params.len())) {
                        (Ok(s), Ok(e)) => (s, e),
                        _ => {
                            return Err(FactoryError::new(
                                JavaException::Runtime,
                                "too many model parameters",
                            ))
                        }
                    };
                    pmap.insert(label, range);
                }
                // `pmap.put(label, null)`: the label evaluates as unknown
                None => {
                    pmap.remove(&label);
                }
            }
        }
        let outcome_names = outcomes.iter().map(|o| text::to_string(o)).collect();
        let log_prior = strict_math::log(1.0 / outcomes.len() as f64);
        Ok(MaxentModel {
            model_type,
            outcomes,
            outcome_names,
            pmap,
            context_outcomes,
            params,
            log_prior,
        })
    }

    /// The model type.
    pub fn model_type(&self) -> ModelType {
        self.model_type
    }

    /// `getNumOutcomes()`.
    pub fn num_outcomes(&self) -> usize {
        self.outcomes.len()
    }

    /// `getOutcome(i)`.
    pub fn outcome(&self, i: usize) -> &[u16] {
        &self.outcomes[i]
    }

    /// `getOutcome(i)` as a Rust string.
    pub fn outcome_name(&self, i: usize) -> &str {
        &self.outcome_names[i]
    }

    /// Every outcome.
    pub fn outcomes(&self) -> &[JString] {
        &self.outcomes
    }

    /// `getIndex(outcome)`: the first equal outcome.
    pub fn index(&self, outcome: &str) -> Option<usize> {
        self.outcome_names.iter().position(|o| o == outcome)
    }

    /// `eval(context, probs)`: the outcome distribution of the predicates
    /// `context` into `probs` (one slot per outcome).
    pub fn eval<F: AsRef<[u16]>>(&self, context: &[F], probs: &mut [f64]) {
        match self.model_type {
            // UniformPrior.logPrior
            ModelType::Gis => probs.fill(self.log_prior),
            ModelType::Perceptron => probs.fill(0.0),
        }
        // ArrayMath.sumFeatures (values == null: each weight times 1)
        for feature in context {
            if let Some(&(start, end)) = self.pmap.get(feature.as_ref()) {
                let range = start as usize..end as usize;
                for (&oid, &p) in self.context_outcomes[range.clone()]
                    .iter()
                    .zip(&self.params[range])
                {
                    probs[oid as usize] += p;
                }
            }
        }
        match self.model_type {
            ModelType::Gis => {
                let mut normal = 0.0;
                for p in probs.iter_mut() {
                    *p = strict_math::exp(*p);
                    normal += *p;
                }
                for p in probs.iter_mut() {
                    *p /= normal;
                }
            }
            ModelType::Perceptron => {
                let mut max_prior = 1.0;
                for p in probs.iter() {
                    if max_prior < p.abs() {
                        max_prior = p.abs();
                    }
                }
                let mut normal = 0.0;
                for p in probs.iter_mut() {
                    *p = strict_math::exp(*p / max_prior);
                    normal += *p;
                }
                for p in probs.iter_mut() {
                    *p /= normal;
                }
            }
        }
    }

    /// `getBestOutcome(probs)`'s index: `ArrayMath.argmax`, the first
    /// maximum (`NaN` never replaces).
    pub fn best_outcome(probs: &[f64]) -> usize {
        let mut max = 0;
        for i in 1..probs.len() {
            if probs[max] < probs[i] {
                max = i;
            }
        }
        max
    }
}

/// `Double.compare(a, b)`: `-0.0` below `0.0`, `NaN` above everything and
/// equal to itself.
pub fn java_double_compare(a: f64, b: f64) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    if a < b {
        return Ordering::Less;
    }
    if a > b {
        return Ordering::Greater;
    }
    let bits = |x: f64| {
        // `doubleToLongBits`: every NaN is the canonical one
        if x.is_nan() {
            0x7ff8_0000_0000_0000i64
        } else {
            x.to_bits() as i64
        }
    };
    bits(a).cmp(&bits(b))
}

/// `Sequence`: the outcomes so far, the sum of their log probabilities
/// and (when a caller asks for them) the probabilities themselves.
#[derive(Debug, Clone)]
struct Sequence {
    outcomes: Vec<u32>,
    probs: Vec<f64>,
    score: f64,
}

impl Sequence {
    /// `Sequence.compareTo`: the higher score first.
    fn cmp(&self, other: &Sequence) -> std::cmp::Ordering {
        java_double_compare(other.score, self.score)
    }
}

/// `java.util.PriorityQueue<Sequence>`: Java's sift-up and sift-down, so
/// sequences of equal score leave in Java's order.
#[derive(Default)]
struct JavaPriorityQueue {
    heap: Vec<Sequence>,
}

impl JavaPriorityQueue {
    fn len(&self) -> usize {
        self.heap.len()
    }

    fn clear(&mut self) {
        self.heap.clear();
    }

    /// `offer`: `siftUpComparable`.
    // ARITH: heap indices below the heap's length (`k > 0` before `k - 1`).
    #[allow(clippy::arithmetic_side_effects)]
    fn add(&mut self, key: Sequence) {
        let mut k = self.heap.len();
        self.heap.push(key);
        while k > 0 {
            let parent = (k - 1) >> 1;
            if self.heap[k].cmp(&self.heap[parent]).is_ge() {
                break;
            }
            self.heap.swap(k, parent);
            k = parent;
        }
    }

    /// `poll`: the head, the last element sifted down from the root.
    // ARITH: heap indices below the heap's length (`child < n` before use).
    #[allow(clippy::arithmetic_side_effects)]
    fn poll(&mut self) -> Option<Sequence> {
        let n = self.heap.len().checked_sub(1)?;
        let last = self.heap.pop()?;
        if n == 0 {
            return Some(last);
        }
        let result = std::mem::replace(&mut self.heap[0], last);
        // siftDownComparable(0, x)
        let mut k = 0;
        let half = n >> 1;
        while k < half {
            let mut child = 2 * k + 1;
            let right = child + 1;
            if right < n && self.heap[child].cmp(&self.heap[right]).is_gt() {
                child = right;
            }
            if self.heap[k].cmp(&self.heap[child]).is_le() {
                break;
            }
            self.heap.swap(k, child);
            k = child;
        }
        Some(result)
    }
}

/// What a beam search tags: `BeamSearchContextGenerator` and
/// `SequenceValidator` over one sequence, the outcomes named by index into
/// the model's outcomes.
pub trait SequenceContext {
    /// `getContext(i, sequence, priorDecisions, additionalContext)`: the
    /// features of position `i` after the outcomes `prior`, into `features`
    /// (empty on entry).
    fn context(
        &mut self,
        i: usize,
        prior: &[u32],
        outcomes: &[JString],
        features: &mut Vec<JString>,
    );

    /// `validSequence(i, sequence, priorDecisions, outcome)`.
    fn valid(&self, i: usize, prior: &[u32], outcome: u32, outcomes: &[JString]) -> bool;
}

/// `BeamSearch` (without a contexts cache: every sequence model
/// `getPosSequenceModel` & co. build has none).
#[derive(Debug, Clone)]
pub struct BeamSearch {
    size: i32,
}

/// `BeamSearch.zeroLog`: the default `minSequenceScore`.
const ZERO_LOG: f64 = -100_000.0;

impl BeamSearch {
    /// `new BeamSearch(size, model)`.
    pub fn new(size: i32) -> Self {
        BeamSearch { size }
    }

    /// `bestSequence(sequence, additionalContext, cg, validator)`'s
    /// outcomes over `len` positions. Java returns `null` when every
    /// sequence was pruned, which its callers dereference: here a
    /// `NullPointerException`. A size below 1 is the
    /// `IllegalArgumentException` of `new PriorityQueue<>(size)`, the
    /// search's first statement, whatever the sentence.
    pub fn best_sequence(
        &self,
        model: &MaxentModel,
        len: usize,
        cg: &mut dyn SequenceContext,
    ) -> Result<Vec<u32>, AnalysisError> {
        self.search(model, len, cg, false).map(|(o, _)| o)
    }

    /// [`best_sequence`](Self::best_sequence) with each outcome's
    /// probability (`Sequence.getProbs()`).
    pub fn best_sequence_with_probs(
        &self,
        model: &MaxentModel,
        len: usize,
        cg: &mut dyn SequenceContext,
    ) -> Result<(Vec<u32>, Vec<f64>), AnalysisError> {
        self.search(model, len, cg, true)
    }

    // ARITH: counts of the beam (at most its size) and a sequence one longer than its prefix.
    #[allow(clippy::arithmetic_side_effects)]
    fn search(
        &self,
        model: &MaxentModel,
        len: usize,
        cg: &mut dyn SequenceContext,
        with_probs: bool,
    ) -> Result<(Vec<u32>, Vec<f64>), AnalysisError> {
        let size = match usize::try_from(self.size) {
            Ok(n) if n >= 1 => n,
            _ => return Err(AnalysisError::IllegalArgument(String::new())),
        };
        let outcomes = model.outcomes();
        let mut prev = JavaPriorityQueue::default();
        let mut next = JavaPriorityQueue::default();
        prev.add(Sequence {
            outcomes: Vec::new(),
            probs: Vec::new(),
            score: 0.0,
        });
        let mut probs = vec![0.0; model.num_outcomes()];
        let mut sorted = vec![0.0; model.num_outcomes()];
        let mut features = Vec::new();
        for i in 0..len {
            let sz = size.min(prev.len());
            let mut sc = 0;
            while prev.len() > 0 && sc < sz {
                sc += 1;
                let Some(top) = prev.poll() else { break };
                features.clear();
                cg.context(i, &top.outcomes, outcomes, &mut features);
                model.eval(&features, &mut probs);
                sorted.copy_from_slice(&probs);
                // The `size`-th largest probability: a selection, not a
                // full sort (`java_double_compare` is a total order, so the
                // value is the sorted array's at that index).
                let k = sorted.len().saturating_sub(size);
                let min = *sorted
                    .select_nth_unstable_by(k, |a, b| java_double_compare(*a, *b))
                    .1;
                let extend = |p: usize, next: &mut JavaPriorityQueue| {
                    if cg.valid(i, &top.outcomes, p as u32, outcomes) {
                        let score = top.score + strict_math::log(probs[p]);
                        if score > ZERO_LOG {
                            let mut o = Vec::with_capacity(top.outcomes.len() + 1);
                            o.extend_from_slice(&top.outcomes);
                            o.push(p as u32);
                            let mut probs_so_far = Vec::new();
                            if with_probs {
                                probs_so_far.reserve(top.probs.len() + 1);
                                probs_so_far.extend_from_slice(&top.probs);
                                probs_so_far.push(probs[p]);
                            }
                            next.add(Sequence {
                                outcomes: o,
                                probs: probs_so_far,
                                score,
                            });
                        }
                    }
                };
                for (p, &prob) in probs.iter().enumerate() {
                    if prob >= min {
                        extend(p, &mut next);
                    }
                }
                if next.len() == 0 {
                    // if no advanced sequences, advance all valid
                    for p in 0..probs.len() {
                        extend(p, &mut next);
                    }
                }
            }
            prev.clear();
            std::mem::swap(&mut prev, &mut next);
        }
        prev.poll().map(|s| (s.outcomes, s.probs)).ok_or_else(|| {
            AnalysisError::IllegalState(
                "NullPointerException: the beam search pruned every sequence".to_string(),
            )
        })
    }
}
