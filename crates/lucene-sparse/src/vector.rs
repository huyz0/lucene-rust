//! Quantized sparse vectors: neural-search's `SparseVector` and `ByteQuantizer`.
//!
//! A vector holds tokens folded into `0..32768` (`token % 32768`), sorted, with
//! one unsigned-byte weight each; tokens that fold together keep the larger
//! weight. Scores are integer dot products of the weights.

/// Tokens are folded modulo this before they are stored (`MODULUS_FOR_SHORT`).
pub const TOKEN_MODULUS: u32 = 32768;

/// `ByteQuantizer.quantize`: `round(clamp(value, 0, ceiling) * 255 / ceiling)`,
/// computed in `f32` as Java does, rounding half up.
pub fn quantize(value: f32, ceiling: f32) -> u8 {
    // `Math.max(0f, Math.min(ceil, v))` keeps a NaN, and `Math.round(NaN)` is 0;
    // Rust's `min`/`max` would drop it instead.
    if value.is_nan() {
        return 0;
    }
    let clamped = 0.0f32.max(ceiling.min(value));
    let scaled = clamped * 255.0 / ceiling;
    // `Math.round(float)` is floor(x + 0.5) without intermediate rounding; f64
    // holds every f32 exactly and the sum of an f32 and 0.5 exactly too.
    let rounded = (f64::from(scaled) + 0.5).floor();
    // In range by construction: 0 <= scaled <= 255.
    rounded.clamp(0.0, 255.0) as u8
}

/// `SparseVector`: folded tokens in ascending order with their weights.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SparseVector {
    tokens: Vec<u16>,
    weights: Vec<u8>,
}

impl SparseVector {
    /// Builds a vector from raw `(token, weight)` items: folds, sorts, and
    /// keeps the larger weight of tokens that fold together.
    pub fn from_items(mut items: Vec<(u32, u8)>) -> Self {
        for item in &mut items {
            item.0 %= TOKEN_MODULUS;
        }
        items.sort_by_key(|&(t, _)| t);
        let mut tokens: Vec<u16> = Vec::with_capacity(items.len());
        let mut weights: Vec<u8> = Vec::with_capacity(items.len());
        for (token, weight) in items {
            // Folded tokens are below 32768, so they fit.
            let token = token as u16;
            match (tokens.last(), weights.last_mut()) {
                (Some(&last), Some(w)) if last == token => *w = (*w).max(weight),
                _ => {
                    tokens.push(token);
                    weights.push(weight);
                }
            }
        }
        SparseVector { tokens, weights }
    }

    /// `new SparseVector(map, new ByteQuantizer(ceiling))` over raw float
    /// weights. A token listed twice keeps its last value, as the `HashMap`
    /// it is read into in Java.
    pub fn quantized(pairs: &[(u32, f32)], ceiling: f32) -> Self {
        let mut last: Vec<(u32, f32)> = Vec::with_capacity(pairs.len());
        for &(t, v) in pairs {
            match last.iter_mut().find(|(lt, _)| *lt == t) {
                Some(slot) => slot.1 = v,
                None => last.push((t, v)),
            }
        }
        Self::from_items(
            last.into_iter()
                .map(|(t, v)| (t, quantize(v, ceiling)))
                .collect(),
        )
    }

    /// A vector whose tokens are already folded and sorted (a cluster summary
    /// as stored on disk).
    pub(crate) fn from_sorted(tokens: Vec<u16>, weights: Vec<u8>) -> Self {
        SparseVector { tokens, weights }
    }

    /// Folded tokens, ascending.
    pub fn tokens(&self) -> &[u16] {
        &self.tokens
    }

    /// Weights, aligned with [`Self::tokens`].
    pub fn weights(&self) -> &[u8] {
        &self.weights
    }

    /// `toDenseVector`: a weight per token from 0 to the largest token.
    pub fn to_dense(&self) -> Vec<u8> {
        let Some(&max) = self.tokens.last() else {
            return Vec::new();
        };
        let mut dense = vec![0u8; usize::from(max) + 1];
        for (&t, &w) in self.tokens.iter().zip(&self.weights) {
            dense[usize::from(t)] = w;
        }
        dense
    }

    /// `dotProduct(dense)`: the integer dot product against a dense query.
    pub fn dot(&self, dense: &[u8]) -> i32 {
        let mut score = 0i32;
        for (&t, &w) in self.tokens.iter().zip(&self.weights) {
            // Tokens ascend, so the first one past the query ends the scan.
            let Some(&q) = dense.get(usize::from(t)) else {
                break;
            };
            // At most 255 * 255, so the product cannot overflow; the sum wraps as Java's int does.
            score = score.wrapping_add(i32::from(w).wrapping_mul(i32::from(q)));
        }
        score
    }
}

/// Reads a forward-index value: big-endian `(int32 token, float32 weight)`
/// pairs, as neural-search's mapper writes them. Trailing bytes short of a
/// pair are ignored, as `DataInputStream.available()` loops do.
pub fn decode_forward(bytes: &[u8]) -> Vec<(u32, f32)> {
    bytes
        .chunks_exact(8)
        .map(|c| {
            let token = i32::from_be_bytes([c[0], c[1], c[2], c[3]]);
            let value = f32::from_be_bytes([c[4], c[5], c[6], c[7]]);
            // Negative tokens are rejected at index time; reinterpret as Java's
            // `%` would leave them negative, which no query token can match.
            (token as u32, value)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantize_rounds_half_up_and_clamps() {
        assert_eq!(quantize(0.0, 3.0), 0);
        assert_eq!(quantize(3.0, 3.0), 255);
        assert_eq!(quantize(10.0, 3.0), 255);
        assert_eq!(quantize(-1.0, 3.0), 0);
        assert_eq!(quantize(f32::NAN, 3.0), 0);
        // 0.5 * 255 / 1 = 127.5 rounds up.
        assert_eq!(quantize(0.5, 1.0), 128);
        assert_eq!(quantize(1.0, 16.0), 16);
    }

    #[test]
    fn folding_keeps_the_larger_weight_and_sorts() {
        let v = SparseVector::from_items(vec![(32769, 9), (1, 4), (5, 7), (32773, 2)]);
        assert_eq!(v.tokens(), &[1, 5]);
        assert_eq!(v.weights(), &[9, 7]);
        let empty = SparseVector::from_items(Vec::new());
        assert!(empty.to_dense().is_empty());
        assert_eq!(empty.dot(&[1, 2]), 0);
    }

    #[test]
    fn duplicate_raw_tokens_keep_the_last_value() {
        let v = SparseVector::quantized(&[(3, 3.0), (3, 0.0)], 3.0);
        assert_eq!(v.weights(), &[0]);
    }

    #[test]
    fn dot_stops_at_the_end_of_the_query() {
        let doc = SparseVector::from_items(vec![(0, 2), (2, 3), (9, 100)]);
        let query = SparseVector::from_items(vec![(0, 5), (2, 7)]).to_dense();
        assert_eq!(query, vec![5, 0, 7]);
        assert_eq!(doc.dot(&query), 2 * 5 + 3 * 7);
    }

    #[test]
    fn forward_values_are_big_endian_pairs() {
        let mut b = Vec::new();
        b.extend_from_slice(&7i32.to_be_bytes());
        b.extend_from_slice(&1.5f32.to_be_bytes());
        b.push(0);
        assert_eq!(decode_forward(&b), vec![(7, 1.5)]);
    }
}
