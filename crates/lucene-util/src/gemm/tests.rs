use super::dot_rows;

fn naive(a: &[f32], b: &[f32], dim: usize) -> Vec<f32> {
    let mut out = Vec::new();
    for x in a.chunks_exact(dim) {
        for y in b.chunks_exact(dim) {
            out.push(x.iter().zip(y).map(|(p, q)| p * q).sum());
        }
    }
    out
}

#[test]
fn matches_naive_products() {
    for (m, n, dim) in [(1, 1, 1), (5, 7, 3), (33, 17, 100), (130, 9, 64)] {
        let a: Vec<f32> = (0..m * dim)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) / 17.0)
            .collect();
        let b: Vec<f32> = (0..n * dim)
            .map(|i| ((i * 53 % 97) as f32 - 48.0) / 13.0)
            .collect();
        let mut out = vec![0.0; m * n];
        dot_rows(&a, &b, dim, &mut out);
        for (got, want) in out.iter().zip(naive(&a, &b, dim)) {
            assert!(
                (got - want).abs() <= 1e-3 * want.abs().max(1.0),
                "{got} vs {want}"
            );
        }
    }
}

#[test]
fn empty_operands_write_nothing() {
    let mut out: Vec<f32> = Vec::new();
    dot_rows(&[], &[1.0, 2.0], 2, &mut out);
    dot_rows(&[1.0, 2.0], &[], 2, &mut out);
}

#[test]
#[should_panic(expected = "ragged")]
fn ragged_rows_panic() {
    dot_rows(&[1.0, 2.0, 3.0], &[1.0, 2.0], 2, &mut [0.0; 1]);
}

#[test]
#[should_panic(expected = "output")]
fn wrong_output_length_panics() {
    dot_rows(&[1.0, 2.0], &[1.0, 2.0], 2, &mut [0.0; 3]);
}
