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
//! `EOFException` Java would reach. `get` computes Java's offset
//! `(backwardId * forwardSize + forwardId) * 2` in wrapping `int`
//! arithmetic, so an id outside its own dimension (`forwardId >=
//! forwardSize`, or a negative one) reads a neighbouring cell exactly as
//! Java's `ByteBuffer.getShort` does; only an offset outside the buffer --
//! negative, or past its end, where Java throws
//! `IndexOutOfBoundsException` -- reads cost 0 instead.

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

    /// The costs of every `forwardId` before `backward_id` (`get(f,
    /// backward_id)` is `row[f]` for each `f` in the row); empty for an id
    /// outside the matrix, whose cells [`Self::get`] reads.
    #[inline]
    pub fn row(&self, backward_id: i32) -> &[i16] {
        usize::try_from(backward_id)
            .ok()
            .filter(|&b| b < self.backward_size)
            .and_then(|b| {
                let start = b.checked_mul(self.forward_size)?;
                self.costs.get(start..start.checked_add(self.forward_size)?)
            })
            .unwrap_or(&[])
    }

    /// `get(forwardId, backwardId)`.
    #[inline]
    pub fn get(&self, forward_id: i32, backward_id: i32) -> i32 {
        // Java: `buffer.getShort((backwardId * forwardSize + forwardId) * 2)`,
        // `int` arithmetic. The forward size came from a vint, so it fits.
        let forward_size = self.forward_size as i32;
        let offset = backward_id
            .wrapping_mul(forward_size)
            .wrapping_add(forward_id)
            .wrapping_mul(2);
        // An even, non-negative offset is cell `offset / 2`; a negative one
        // is Java's `IndexOutOfBoundsException`, read as 0 here.
        usize::try_from(offset)
            .ok()
            .and_then(|o| self.costs.get(o >> 1))
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
        // Java reads a neighbouring row for an id outside its own dimension.
        assert_eq!(c.get(3, 0), 40000i32 as i16 as i32);
        assert_eq!(c.get(-1, 1), 3);
        assert_eq!(c.get(5, 0), 6);
        // Past the buffer, or a negative offset: Java throws, this reads 0.
        for (f, b) in [(6, 0), (0, 2), (-1, 0), (0, -1), (i32::MAX, i32::MAX)] {
            assert_eq!(c.get(f, b), 0);
        }
        // The offset wraps as Java's `int` does: (1 << 30) * 2 = i32::MIN.
        assert_eq!(c.get(1 << 30, 0), 0);
        assert_eq!(c.get(0, i32::MIN), 1);
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
