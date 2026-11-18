use super::*;

/// Shared tail of a NUMERIC entry body -- everything after the
/// docs-with-field header, common to both the dense and sparse shapes:
/// `numValues`, the constant/table/GCD-delta encoding choice, and the
/// bit-packed (or absent, for the constant case) value array itself.
///
/// Port of `Lucene90DocValuesConsumer.writeValues`'s encoding-choice logic,
/// including its `doBlocks` varying-bits-per-value split
/// ([`write_values_multiple_blocks`]): computes a running GCD of every value's difference
/// from the first value (mirrors `MathUtil.gcd` accumulation, including
/// Java's overflow guard: values outside `[i64::MIN/2, i64::MAX/2]` abandon
/// GCD tracking for the rest of the scan rather than risk `v - firstValue`
/// overflowing), and a `<= 256`-entry distinct-value set (abandoned, same as
/// Java's `uniqueValues`, the moment a 257th distinct value appears). Then
/// picks whichever of table-compression (`bitsPerValue =
/// unsignedBitsRequired(uniqueValues.len() - 1)`) or GCD/plain-delta
/// (`bitsPerValue = unsignedBitsRequired((max - min) / gcd)`) needs fewer
/// bits, strictly preferring delta on a tie (`<`, not `<=`, exactly like
/// Java) -- this also means a field whose values are already a dense
/// contiguous ordinal range (e.g. a SORTED field's per-doc dictionary
/// ordinals, which reuse this exact function) never accidentally picks table
/// compression: `uniqueValues.len() - 1 == max - min` there, so the `<`
/// comparison always favors delta, matching `Lucene90DocValuesConsumer`'s own
/// explicit `ords ? null : new LongHashSet()` (ordinals never build a
/// `uniqueValues` set at all; this port gets the same *outcome* without
/// needing that special case, since the two paths tie).
// ARITH: the two divisions are by `gcd`, which the `if gcd == 0 { gcd = 1 }`
// below forces non-zero, and `gcd` is `unsigned_abs`-derived so it is never
// `-1` (no `i64::MIN / -1`). The two `unique.len() as i64 - 1` uses are
// reached only past `min >= max`, which means at least two distinct values,
// so the set is non-empty. `data.len() as i64 - start_offset` subtracts an
// offset recorded from an earlier `data.len()` of the same append-only
// buffer. The two `wrapping_sub`s are deliberate and documented in place.
#[allow(clippy::arithmetic_side_effects)]
pub(super) fn write_numeric_values_body(meta: &mut Vec<u8>, data: &mut Vec<u8>, values: &[i64]) {
    let num_values = values.len() as i64;
    meta.write_i64(num_values);

    // `MinMaxTracker` starts at `(Long.MAX_VALUE, Long.MIN_VALUE)` and stays
    // there for an empty value array, which is what Java then writes as this
    // entry's `min`. Nothing reads it back (an empty field carries the `-2`
    // docs-with-field marker, so no doc ever resolves to an ordinal), but
    // matching it keeps the bytes identical to real Lucene's.
    let min = values.iter().copied().min().unwrap_or(i64::MAX);
    let max = values.iter().copied().max().unwrap_or(i64::MIN);

    // Java runs the GCD scan before the `min >= max` branch and writes
    // whatever it produced either way -- `0` when every value is identical
    // (every `v - firstValue` is 0), or `1` when a value outside
    // `[MIN/2, MAX/2]` tripped the overflow guard. `bitsPerValue == 0` means
    // it is never read back, but writing the same number keeps this writer
    // byte-identical to `Lucene90DocValuesConsumer` for the constant case.
    let mut gcd = compute_gcd(values);

    if min >= max {
        // All values equal (including the empty-values case, which can't
        // actually happen for a standalone NUMERIC field since
        // `values.len() == max_doc` and `max_doc` is a real field count, but
        // can for a SORTED_NUMERIC field's flat value array when every doc
        // has zero values -- Java's `min >= max` check covers both "all
        // equal" and "no values" the same way).
        meta.write_i32(-1); // tableSize
        meta.push(0); // bitsPerValue
        meta.write_i64(min);
        meta.write_i64(gcd);
        let start_offset = data.len() as i64;
        meta.write_i64(start_offset);
        meta.write_i64(0); // valuesLength
        meta.write_i64(-1); // valueJumpTableOffset
        return;
    }

    // A non-constant value array always yields a non-zero GCD (some value
    // differs from the first), so this only guards the theoretically
    // unreachable case; Java has no equivalent because `DirectWriter` would
    // divide by it either way.
    if gcd == 0 {
        gcd = 1;
    }

    // Distinct-value set, capped at 256 entries (table compression's max
    // table size), same as Java's `uniqueValues` -- a hash set, as Java's
    // `LongHashSet`; sorted only if it becomes the table.
    let mut unique: HashSet<i64, BuildHasherDefault<SetHasher>> =
        HashSet::with_capacity_and_hasher(512, BuildHasherDefault::default());
    let mut unique_tracked = true;
    for &v in values {
        if !unique_tracked {
            break;
        }
        unique.insert(v);
        if unique.len() > 256 {
            unique_tracked = false;
        }
    }

    // Same wraparound-not-panic reasoning as the GCD loop above: `max - min`
    // can overflow i64 for a pathological value set (e.g. min == i64::MIN),
    // just like Java's `long` subtraction would -- wrap instead of crash.
    let range_bits = direct_reader::unsigned_bits_required(max.wrapping_sub(min) / gcd);
    let use_table = unique_tracked
        && unique.len() > 1
        && direct_reader::unsigned_bits_required(unique.len() as i64 - 1) < range_bits;

    // `doBlocks`: per-16384-value blocks, each packed at its own width, when
    // that appears to save 10% or more over one width for the whole column.
    let do_blocks = !use_table && {
        let whole = space_in_bits(values);
        let blocks: i64 = values
            .chunks(NUMERIC_BLOCK_SIZE)
            .map(space_in_bits)
            .fold(0i64, i64::wrapping_add);
        whole > 0 && blocks as f64 / whole as f64 <= 0.9
    };

    let (bits_per_value, min, gcd, table): (u8, i64, i64, Option<Vec<i64>>) = if use_table {
        let mut sorted_unique: Vec<i64> = unique.into_iter().collect();
        sorted_unique.sort_unstable();
        let bpv = direct_reader::unsigned_bits_required(sorted_unique.len() as i64 - 1);
        (bpv, 0, 1, Some(sorted_unique))
    } else if do_blocks {
        // `numBitsPerValue = 0xFF`; `min` is the column's own, never zeroed.
        (0xFF, min, gcd, None)
    } else {
        let mut bpv = range_bits;
        let mut min = min;
        // Java: if gcd==1 && min>0 && bits(max) == bits(max-min), drop the
        // min-shift (store raw values instead) since it doesn't save space.
        if gcd == 1 && min > 0 && direct_reader::unsigned_bits_required(max) == bpv {
            min = 0;
            bpv = direct_reader::unsigned_bits_required(max);
        }
        (bpv, min, gcd, None)
    };

    match &table {
        Some(t) => {
            meta.write_i32(t.len() as i32);
            for &v in t {
                meta.write_i64(v);
            }
        }
        None if do_blocks => meta.write_i32(-2 - NUMERIC_BLOCK_SHIFT as i32),
        None => meta.write_i32(-1),
    }
    meta.push(bits_per_value);
    meta.write_i64(min);
    meta.write_i64(gcd);

    let start_offset = data.len() as i64;
    meta.write_i64(start_offset);

    if do_blocks {
        let jump_table_offset = write_values_multiple_blocks(data, values, gcd);
        meta.write_i64(data.len() as i64 - start_offset); // valuesLength
        meta.write_i64(jump_table_offset);
        return;
    }
    if bits_per_value != 0 {
        let raw: Vec<i64> = match &table {
            Some(t) => values
                .iter()
                .map(|v| t.binary_search(v).expect("value must be in its own table") as i64)
                .collect(),
            // `v - min` for `v` in `values` is normally in `[0, max - min]`
            // (min/max are this array's own extrema), but `wrapping_sub`
            // keeps the same non-panicking-on-pathological-input guarantee
            // as `range_bits`'s `max.wrapping_sub(min)` above for the same
            // reason -- both derive from the same possibly-huge min/max.
            None => values.iter().map(|&v| v.wrapping_sub(min) / gcd).collect(),
        };
        let mut packed = direct_reader::encode(&raw, bits_per_value);
        packed.extend(std::iter::repeat_n(
            0u8,
            direct_reader::padding_bytes_needed(bits_per_value),
        ));
        data.extend_from_slice(&packed);
    }

    meta.write_i64(data.len() as i64 - start_offset); // valuesLength
    meta.write_i64(-1); // valueJumpTableOffset: no varying-bpv blocks
}
