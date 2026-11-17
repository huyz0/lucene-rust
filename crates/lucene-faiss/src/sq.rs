//! `QT_8bit` scalar quantization (`SQ8`): training, encoding, and the
//! distances from a float query to a code, as Faiss computes them.

use crate::format::{Metric, Sq8};

/// `RS_minmax`, the range statistic the `SQ8` factory string trains with.
const RS_MINMAX: u32 = 0;

impl Sq8 {
    /// Trains on `rows` (`d` floats each): per-dimension minimum and range.
    pub(crate) fn train(d: usize, rows: &[f32]) -> Sq8 {
        let mut vmin = vec![f32::INFINITY; d];
        let mut vmax = vec![f32::NEG_INFINITY; d];
        for row in rows.chunks_exact(d.max(1)) {
            for ((lo, hi), &x) in vmin.iter_mut().zip(&mut vmax).zip(row) {
                *lo = lo.min(x);
                *hi = hi.max(x);
            }
        }
        // Nothing to learn from: a unit range around zero.
        for (lo, hi) in vmin.iter_mut().zip(&mut vmax) {
            if lo > hi {
                *lo = 0.0;
                *hi = 1.0;
            }
        }
        let vdiff = vmin.iter().zip(&vmax).map(|(lo, hi)| hi - lo).collect();
        Sq8 {
            range_stat: RS_MINMAX,
            range_stat_arg: 0.0,
            vmin,
            vdiff,
        }
    }

    /// Encodes `x` into `out` (`Codec8bit::encode_component`).
    pub(crate) fn encode(&self, x: &[f32], out: &mut [u8]) {
        for (((o, &v), &lo), &diff) in out.iter_mut().zip(x).zip(&self.vmin).zip(&self.vdiff) {
            let xi = if diff == 0.0 { 0.0 } else { (v - lo) / diff };
            *o = (255.0 * xi.clamp(0.0, 1.0)) as u8;
        }
    }

    /// The value `code` decodes to in a dimension with minimum `lo` and
    /// range `diff`: `lo + (code + 0.5) / 255 * diff`, as Faiss's vector
    /// kernels compute it (multiply by `1 / 255`, fused multiply-add).
    #[inline(always)]
    fn component(code: u8, lo: f32, diff: f32) -> f32 {
        ((f32::from(code) + 0.5) * (1.0 / 255.0)).mul_add(diff, lo)
    }

    /// Decodes `code` into `out`.
    pub(crate) fn decode(&self, code: &[u8], out: &mut [f32]) {
        for (((o, &c), &lo), &diff) in out.iter_mut().zip(code).zip(&self.vmin).zip(&self.vdiff) {
            *o = Self::component(c, lo, diff);
        }
    }

    /// The distance of `metric` from the float vector `q` to `code`: squared
    /// L2 or inner product, in eight lanes like the plain float kernels.
    pub(crate) fn distance(&self, metric: Metric, q: &[f32], code: &[u8]) -> f32 {
        let mut acc = [0f32; 8];
        let (cq, cc) = (q.chunks_exact(8), code.chunks_exact(8));
        let (cl, cd) = (self.vmin.chunks_exact(8), self.vdiff.chunks_exact(8));
        let (rq, rc) = (cq.remainder(), cc.remainder());
        let tail = q.len() - rq.len();
        for (((q8, c8), lo8), diff8) in cq.zip(cc).zip(cl).zip(cd) {
            for i in 0..8 {
                let x = Self::component(c8[i], lo8[i], diff8[i]);
                acc[i] = match metric {
                    Metric::L2 => {
                        let d = q8[i] - x;
                        d.mul_add(d, acc[i])
                    }
                    Metric::InnerProduct => q8[i].mul_add(x, acc[i]),
                };
            }
        }
        let mut sum = acc.iter().sum::<f32>();
        for (j, (&qj, &c)) in rq.iter().zip(rc).enumerate() {
            let x = Self::component(c, self.vmin[tail + j], self.vdiff[tail + j]);
            sum += match metric {
                Metric::L2 => (qj - x) * (qj - x),
                Metric::InnerProduct => qj * x,
            };
        }
        sum
    }
}
