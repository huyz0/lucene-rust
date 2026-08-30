//! The `org.apache.lucene.util.NumericUtils` float helpers: the
//! order-preserving `f32` <-> `i32` mapping.
//!
//! Only the float half lives here so far; the `long`/`double` and
//! `*ToSortableBytes` halves are still duplicated in
//! `lucene_index::segment_info` and `lucene_search::facets` (see
//! `docs/sweep/m2/LEDGER.md`).

/// `NumericUtils.sortableFloatBits`.
pub fn sortable_float_bits(bits: i32) -> i32 {
    bits ^ ((bits >> 31) & 0x7fff_ffff)
}

/// `NumericUtils.floatToSortableInt`.
pub fn float_to_sortable_int(value: f32) -> i32 {
    sortable_float_bits(value.to_bits() as i32)
}

/// `NumericUtils.sortableIntToFloat`.
pub fn sortable_int_to_float(encoded: i32) -> f32 {
    f32::from_bits(sortable_float_bits(encoded) as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The point of the mapping: the `i32` order is the `f32` order, across
    /// zero and across the sign boundary, and the transform is its own
    /// inverse. `NeighborQueue` packs a score through this so a plain integer
    /// heap orders by score.
    #[test]
    fn the_encoding_round_trips_and_preserves_order() {
        for v in [-1e30f32, -1.0, -0.0, 0.0, 1.0, 1e30] {
            assert_eq!(sortable_int_to_float(float_to_sortable_int(v)), v);
        }
        let ordered = [
            f32::NEG_INFINITY,
            -1e30,
            -1.0,
            -f32::MIN_POSITIVE,
            -0.0,
            0.0,
            f32::MIN_POSITIVE,
            1.0,
            1e30,
            f32::INFINITY,
        ];
        let encoded: Vec<i32> = ordered.iter().copied().map(float_to_sortable_int).collect();
        let mut sorted = encoded.clone();
        sorted.sort_unstable();
        assert_eq!(encoded, sorted, "the i32 order must be the f32 order");
        assert!(float_to_sortable_int(-1.0) < float_to_sortable_int(0.0));
        assert!(float_to_sortable_int(0.0) < float_to_sortable_int(1.0));
    }

    /// `sortableFloatBits` is an involution on the bit pattern.
    #[test]
    fn sortable_float_bits_is_its_own_inverse() {
        for bits in [0i32, 1, -1, i32::MIN, i32::MAX, 0x4048_f5c3] {
            assert_eq!(sortable_float_bits(sortable_float_bits(bits)), bits);
        }
    }
}
