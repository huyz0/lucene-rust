//! Seeded random rotations applied to base vectors and queries before
//! indexing: none, IsoQuant's 4-dimensional block rotations (full SO(4) from
//! a left and a right unit quaternion, or the fast left-only one), a
//! randomized Hadamard transform without padding (sign flips and
//! overlapping power-of-two FWHT windows, three rounds), and a dense random
//! orthogonal matrix (TurboQuant's rotation).

use rayon::prelude::*;

pub enum Rotation {
    None,
    /// One row-major 4x4 matrix per block of four dimensions.
    Blocks(Vec<[f32; 16]>),
    /// Per round: signs, then FWHT on the first and the last `p` coordinates.
    Hadamard {
        signs: Vec<Vec<f32>>,
        p: usize,
    },
    Dense(Vec<f32>),
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn uniform(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn gauss(&mut self) -> f64 {
        let (u, v) = (self.uniform().max(1e-300), self.uniform());
        (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
    }
    fn quaternion(&mut self) -> [f32; 4] {
        let g = [self.gauss(), self.gauss(), self.gauss(), self.gauss()];
        let n = g.iter().map(|x| x * x).sum::<f64>().sqrt();
        g.map(|x| (x / n) as f32)
    }
}

fn left([a, b, c, d]: [f32; 4]) -> [f32; 16] {
    [a, -b, -c, -d, b, a, -d, c, c, d, a, -b, d, -c, b, a]
}

fn right([a, b, c, d]: [f32; 4]) -> [f32; 16] {
    [a, -b, -c, -d, b, a, d, -c, c, -d, a, b, d, c, -b, a]
}

fn mul(x: &[f32; 16], y: &[f32; 16]) -> [f32; 16] {
    let mut o = [0f32; 16];
    for r in 0..4 {
        for c in 0..4 {
            o[r * 4 + c] = (0..4).map(|k| x[r * 4 + k] * y[k * 4 + c]).sum();
        }
    }
    o
}

fn fwht(x: &mut [f32]) {
    let n = x.len();
    let mut h = 1;
    while h < n {
        for i in (0..n).step_by(2 * h) {
            for j in i..i + h {
                let (a, b) = (x[j], x[j + h]);
                x[j] = a + b;
                x[j + h] = a - b;
            }
        }
        h *= 2;
    }
    let s = 1.0 / (n as f32).sqrt();
    x.iter_mut().for_each(|v| *v *= s);
}

impl Rotation {
    pub fn new(kind: &str, dim: usize, seed: u64) -> Self {
        let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
        match kind {
            "none" => Rotation::None,
            "iso" | "isofast" => {
                assert!(dim % 4 == 0, "iso needs a multiple of 4 dimensions");
                Rotation::Blocks(
                    (0..dim / 4)
                        .map(|_| {
                            let l = left(rng.quaternion());
                            if kind == "iso" {
                                mul(&l, &right(rng.quaternion()))
                            } else {
                                l
                            }
                        })
                        .collect(),
                )
            }
            "hadamard" => {
                let p = 1usize << (usize::BITS - 1 - dim.leading_zeros());
                let signs = (0..6)
                    .map(|_| {
                        (0..dim)
                            .map(|_| if rng.next() & 1 == 0 { 1.0 } else { -1.0 })
                            .collect()
                    })
                    .collect();
                Rotation::Hadamard { signs, p }
            }
            "dense" => {
                // Gaussian matrix, rows orthonormalized (modified Gram-Schmidt).
                let mut m: Vec<f64> = (0..dim * dim).map(|_| rng.gauss()).collect();
                for i in 0..dim {
                    for j in 0..i {
                        let d: f64 = (0..dim).map(|k| m[i * dim + k] * m[j * dim + k]).sum();
                        for k in 0..dim {
                            m[i * dim + k] -= d * m[j * dim + k];
                        }
                    }
                    let n = (0..dim).map(|k| m[i * dim + k].powi(2)).sum::<f64>().sqrt();
                    for k in 0..dim {
                        m[i * dim + k] /= n;
                    }
                }
                Rotation::Dense(m.into_iter().map(|x| x as f32).collect())
            }
            other => panic!("unknown rotation {other}"),
        }
    }

    pub fn apply(&self, v: &[f32]) -> Vec<f32> {
        match self {
            Rotation::None => v.to_vec(),
            Rotation::Blocks(ms) => {
                let mut out = vec![0f32; v.len()];
                for (b, m) in ms.iter().enumerate() {
                    let x = &v[4 * b..4 * b + 4];
                    for r in 0..4 {
                        out[4 * b + r] = m[r * 4] * x[0]
                            + m[r * 4 + 1] * x[1]
                            + m[r * 4 + 2] * x[2]
                            + m[r * 4 + 3] * x[3];
                    }
                }
                out
            }
            Rotation::Hadamard { signs, p } => {
                let mut x = v.to_vec();
                let n = x.len();
                for round in 0..3 {
                    x.iter_mut()
                        .zip(&signs[2 * round])
                        .for_each(|(a, s)| *a *= s);
                    fwht(&mut x[..*p]);
                    x.iter_mut()
                        .zip(&signs[2 * round + 1])
                        .for_each(|(a, s)| *a *= s);
                    fwht(&mut x[n - p..]);
                }
                x
            }
            Rotation::Dense(m) => {
                let d = v.len();
                (0..d)
                    .map(|i| lucene_util::simd::dot_f32(&m[i * d..(i + 1) * d], v))
                    .collect()
            }
        }
    }

    pub fn apply_all(&self, vs: &[f32]) -> Vec<f32> {
        let dim = match self {
            Rotation::None => return vs.to_vec(),
            Rotation::Blocks(ms) => ms.len() * 4,
            Rotation::Hadamard { signs, .. } => signs[0].len(),
            Rotation::Dense(m) => (m.len() as f64).sqrt() as usize,
        };
        vs.par_chunks_exact(dim)
            .flat_map_iter(|v| self.apply(v))
            .collect()
    }
}
