//! `MemoryIndex.Info.terms`, Java's `BytesRefHash`: each distinct term of a
//! field once, numbered in first-seen order. The terms live end to end in
//! one byte pool and the table is open addressing over their ords, so
//! looking a term up allocates nothing and adding one copies its bytes once
//! -- a field indexes every token of its value through here.

/// The terms of one field, by ord.
#[derive(Debug, Clone, Default)]
pub(super) struct TermHash {
    /// Every term's bytes, end to end.
    pool: Vec<u8>,
    /// `ends[ord]`: where term `ord`'s bytes end in `pool` (it starts where
    /// the previous one ends).
    ends: Vec<usize>,
    /// `ord + 1` per slot, `0` for an empty one; its length a power of two,
    /// at least twice the number of terms.
    table: Vec<u32>,
}

/// FxHash's multiply-rotate over 8-byte words, and the length.
fn hash(bytes: &[u8]) -> u64 {
    const K: u64 = 0x51_7c_c1_b7_27_22_0a_95;
    let mut h = bytes.len() as u64;
    let mut words = bytes.chunks_exact(8);
    for w in &mut words {
        let v = u64::from_le_bytes(w.try_into().expect("an 8-byte chunk"));
        h = (h.rotate_left(5) ^ v).wrapping_mul(K);
    }
    let mut tail = [0u8; 8];
    let rest = words.remainder();
    tail[..rest.len()].copy_from_slice(rest);
    (h.rotate_left(5) ^ u64::from_le_bytes(tail)).wrapping_mul(K)
}

impl TermHash {
    /// The number of terms.
    pub(super) fn len(&self) -> usize {
        self.ends.len()
    }

    /// Term `ord`'s bytes.
    pub(super) fn get(&self, ord: usize) -> &[u8] {
        let start = match ord {
            0 => 0,
            _ => self.ends[ord - 1],
        };
        &self.pool[start..self.ends[ord]]
    }

    /// `BytesRefHash.add`: `term`'s ord, added when new.
    pub(super) fn add(&mut self, term: &[u8]) -> usize {
        if self.table.len() < self.ends.len().saturating_add(1).saturating_mul(2) {
            self.grow();
        }
        let mask = self.table.len() - 1;
        let mut slot = hash(term) as usize & mask;
        loop {
            match self.table[slot] {
                0 => break,
                id => {
                    let ord = id as usize - 1;
                    if self.get(ord) == term {
                        return ord;
                    }
                }
            }
            slot = (slot + 1) & mask;
        }
        let ord = self.ends.len();
        self.pool.extend_from_slice(term);
        self.ends.push(self.pool.len());
        self.table[slot] = u32::try_from(ord + 1).expect("fewer than 2^32 terms");
        ord
    }

    /// Doubles the table (from 16 slots) and re-places every ord.
    fn grow(&mut self) {
        let size = (self.table.len() * 2).max(16);
        self.table = vec![0; size];
        let mask = size - 1;
        for ord in 0..self.ends.len() {
            let mut slot = hash(self.get(ord)) as usize & mask;
            while self.table[slot] != 0 {
                slot = (slot + 1) & mask;
            }
            self.table[slot] = u32::try_from(ord + 1).expect("fewer than 2^32 terms");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn terms_keep_their_first_seen_ords_through_growth() {
        let mut h = TermHash::default();
        let terms: Vec<Vec<u8>> = (0..1000u32)
            .map(|i| format!("term-{i}-{}", "x".repeat(i as usize % 13)).into_bytes())
            .collect();
        for (i, t) in terms.iter().enumerate() {
            assert_eq!(h.add(t), i);
        }
        for (i, t) in terms.iter().enumerate().rev() {
            assert_eq!(h.add(t), i);
            assert_eq!(h.get(i), &t[..]);
        }
        assert_eq!(h.len(), 1000);
        assert_eq!(h.add(b""), 1000);
        assert_eq!(h.get(1000), b"");
        assert_eq!(h.add(b""), 1000);
    }
}
