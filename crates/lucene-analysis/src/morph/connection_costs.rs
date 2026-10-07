//! `org.apache.lucene.analysis.morph.ConnectionCosts`: the bigram cost of a
//! word whose right id is `forwardId` followed by one whose left id is
//! `backwardId`.
//!
//! File: `CodecUtil` header, `vint forwardSize`, `vint backwardSize`, then
//! `forwardSize * backwardSize` zints, row by row of `backwardId`, each the
//! delta from the previous cost (accumulated in Java `int`, stored as
//! `short`).
//!
//! Differs: Java allocates the matrix before reading it and fails on the
//! first missing value; this refuses a matrix with more cells than the file
//! has bytes left (each cell takes at least one) before allocating, with the
//! `EOFException` Java would reach. An id outside the matrix -- a dictionary
//! that does not belong to these costs -- reads cost 0 where Java throws
//! `IndexOutOfBoundsException`.

use super::resource::{io_error, ResourceInput};
use crate::AnalysisError;

/// `ConnectionCosts`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConnectionCosts {
    /// The matrix, `backwardId`-major.
    costs: Vec<i16>,
    forward_size: usize,
    backward_size: usize,
}

impl ConnectionCosts {
    /// `new ConnectionCosts(resource, codecHeader, version)`.
    pub fn read(bytes: &[u8], codec_header: &str, version: i32) -> Result<Self, AnalysisError> {
        let mut input = ResourceInput::new(bytes);
        input.check_header(codec_header, version, version)?;
        let forward = input.read_vint()?;
        let backward = input.read_vint()?;
        let (Ok(forward_size), Ok(backward_size)) =
            (usize::try_from(forward), usize::try_from(backward))
        else {
            return Err(io_error(
                "IllegalArgumentException",
                format!("capacity < 0: ({forward} * {backward} * 2 < 0)"),
            ));
        };
        let size = forward_size
            .checked_mul(backward_size)
            .filter(|&n| n <= input.remaining())
            .ok_or_else(|| io_error("EOFException", "read past EOF"))?;
        let mut costs = Vec::with_capacity(size);
        let mut accum: i32 = 0;
        for _ in 0..size {
            accum = accum.wrapping_add(input.read_zint()?);
            costs.push(accum as i16);
        }
        Ok(ConnectionCosts {
            costs,
            forward_size,
            backward_size,
        })
    }

    /// `get(forwardId, backwardId)`.
    #[inline]
    pub fn get(&self, forward_id: i32, backward_id: i32) -> i32 {
        let (Ok(f), Ok(b)) = (usize::try_from(forward_id), usize::try_from(backward_id)) else {
            return 0;
        };
        if f >= self.forward_size {
            return 0;
        }
        // `b < backward_size` is only checked by the lookup, after the
        // product: an overflow is an id outside the matrix.
        b.checked_mul(self.forward_size)
            .and_then(|row| row.checked_add(f))
            .and_then(|i| self.costs.get(i))
            .map_or(0, |&c| i32::from(c))
    }

    /// `forwardSize` and `backwardSize`.
    pub fn dimensions(&self) -> (usize, usize) {
        (self.forward_size, self.backward_size)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    #![allow(clippy::arithmetic_side_effects)]
    use super::*;
    use crate::morph::resource::test_util::{header, vlong};

    /// A costs file of `rows` (`backwardId`-major), as
    /// `ConnectionCostsWriter` writes it.
    pub(crate) fn costs_file(codec: &str, forward: usize, rows: &[i32]) -> Vec<u8> {
        let mut b = header(codec, 1);
        vlong(&mut b, forward as u64);
        vlong(&mut b, (rows.len() / forward.max(1)) as u64);
        let mut last = 0i32;
        for &c in rows {
            let d = c.wrapping_sub(last);
            last = c;
            vlong(&mut b, ((d << 1) ^ (d >> 31)) as u32 as u64);
        }
        b
    }

    #[test]
    fn reads_and_looks_up_like_java() {
        let f = costs_file("ko_cc", 3, &[1, -2, 3, 40000, 5, 6]);
        let c = ConnectionCosts::read(&f, "ko_cc", 1).unwrap();
        assert_eq!(c.dimensions(), (3, 2));
        assert_eq!(c.get(0, 0), 1);
        assert_eq!(c.get(1, 0), -2);
        // Stored as a Java short.
        assert_eq!(c.get(0, 1), 40000i32 as i16 as i32);
        assert_eq!(c.get(2, 1), 6);
        for (f, b) in [(3, 0), (0, 2), (-1, 0), (0, -1), (i32::MAX, i32::MAX)] {
            assert_eq!(c.get(f, b), 0);
        }
    }

    #[test]
    fn hostile_sizes_fail_without_allocating() {
        let mut b = header("ko_cc", 1);
        vlong(&mut b, 1 << 20);
        vlong(&mut b, 1 << 20);
        assert!(ConnectionCosts::read(&b, "ko_cc", 1).is_err());
        let mut neg = header("ko_cc", 1);
        neg.extend_from_slice(&[0xFF, 0xFF, 0xFF, 0xFF, 0x0F, 1]);
        let e = ConnectionCosts::read(&neg, "ko_cc", 1).unwrap_err();
        assert!(e.to_string().contains("IllegalArgumentException"), "{e}");
        let f = costs_file("ko_cc", 2, &[1, 2, 3, 4]);
        for cut in 0..f.len() {
            assert!(ConnectionCosts::read(&f[..cut], "ko_cc", 1).is_err());
        }
        assert!(ConnectionCosts::read(&f, "ja_cc", 1).is_err());
    }
}
