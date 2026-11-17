//! Dense `f32` matrix products for batch distance work (clustering, centroid
//! ranking): the tiled micro-kernels of the `matrixmultiply` crate behind a
//! safe, row-major interface.

/// `out[i * rows_b + j] = a_i . b_j` for every row `a_i` of `a` and `b_j` of
/// `b` (rows of `dim` floats): the product `A * B^T`, one inner product per
/// pair, computed with a blocked kernel that reads each operand tile once.
///
/// # Panics
/// When a slice length is not the row count times `dim`.
// ARITH: the three length checks below bound every product and offset.
#[allow(clippy::arithmetic_side_effects)]
pub fn dot_rows(a: &[f32], b: &[f32], dim: usize, out: &mut [f32]) {
    assert!(
        dim > 0 && a.len().is_multiple_of(dim) && b.len().is_multiple_of(dim),
        "ragged rows"
    );
    let (m, n) = (a.len() / dim, b.len() / dim);
    assert_eq!(out.len(), m * n, "output of {m} x {n}");
    if m == 0 || n == 0 {
        return;
    }
    // SAFETY: `a` is `m` rows of `dim` (row stride `dim`, column stride 1),
    // `b` read as `dim x n` with strides (1, `dim`) stays inside its `n * dim`
    // floats, and `out` is `m * n` with row stride `n`; the lengths were
    // asserted above, `a`/`b` are read-only and `out` does not alias them.
    unsafe {
        matrixmultiply::sgemm(
            m,
            dim,
            n,
            1.0,
            a.as_ptr(),
            dim as isize,
            1,
            b.as_ptr(),
            1,
            dim as isize,
            0.0,
            out.as_mut_ptr(),
            n as isize,
            1,
        );
    }
}

#[cfg(test)]
mod tests;
