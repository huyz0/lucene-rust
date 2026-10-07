//! `org.apache.lucene.analysis.cn.smart.hhmm.BigramDictionary`: the word
//! bigram dictionary, `bigramdict.mem`.
//!
//! The file is Java's serialization of `bigramHashTable` (`long[402137]`)
//! and `frequencyTable` (`int[402137]`): a double-hashing table keyed by the
//! FNV hash ([`hash1`]) of `word1@word2`, probed with [`hash2`] as the step;
//! a slot holds the key's hash, not the key, so two pairs whose hashes agree
//! share a frequency, as in Java.
//!
//! Java computes the probe `(hash1 + i * hash2) % PRIME_BIGRAM_LENGTH` in
//! `int`, which overflows after about 5,340 probes and would index the table
//! with a negative number (an `ArrayIndexOutOfBoundsException`); a lookup
//! that gets there reports the pair absent. At the table's 82% load no
//! lookup does.

use std::sync::{Arc, LazyLock};

use super::abstract_dictionary::{hash1, hash2, probe};
use crate::serialized::{ArrayData, ObjectStream};
use crate::SmartcnError;

/// `PRIME_BIGRAM_LENGTH`: the table's size.
pub const PRIME_BIGRAM_LENGTH: usize = 402_137;
/// `WORD_SEGMENT_CHAR`: the separator of a pair's two words, `@`.
pub const WORD_SEGMENT_CHAR: u16 = 0x40;

const BIGRAM_DICT_Z: &[u8] = include_bytes!("../resources/bigramdict.mem.z");

/// `BigramDictionary`.
#[derive(Debug)]
pub struct BigramDictionary {
    bigram_hash_table: Vec<i64>,
    frequency_table: Vec<i32>,
}

static DEFAULT: LazyLock<Arc<BigramDictionary>> = LazyLock::new(|| {
    let bytes = miniz_oxide::inflate::decompress_to_vec_zlib(BIGRAM_DICT_Z)
        .expect("the vendored bigramdict.mem inflates");
    Arc::new(BigramDictionary::from_mem(&bytes).expect("the vendored bigramdict.mem loads"))
});

impl BigramDictionary {
    /// `BigramDictionary.getInstance()`: Lucene's `bigramdict.mem`, loaded
    /// once.
    pub fn get_instance() -> Arc<BigramDictionary> {
        Arc::clone(&DEFAULT)
    }

    /// `loadFromInputStream`: a dictionary from a `bigramdict.mem`'s bytes.
    pub fn from_mem(bytes: &[u8]) -> Result<BigramDictionary, SmartcnError> {
        let (stream, top) = ObjectStream::read(bytes)?;
        let get = |i: usize| {
            top.get(i)
                .copied()
                .flatten()
                .and_then(|h| stream.array_at(h))
        };
        let (Some((_, ArrayData::Long(hashes))), Some((_, ArrayData::Int(freqs)))) =
            (get(0), get(1))
        else {
            return Err(SmartcnError::new(
                "ClassCastException: bigramdict.mem is not a long[] and an int[]",
            ));
        };
        if hashes.len() != PRIME_BIGRAM_LENGTH || freqs.len() != PRIME_BIGRAM_LENGTH {
            return Err(SmartcnError::new(
                "corrupt bigramdict.mem: tables are not PRIME_BIGRAM_LENGTH long",
            ));
        }
        Ok(BigramDictionary {
            bigram_hash_table: hashes.clone(),
            frequency_table: freqs.clone(),
        })
    }

    /// `getBigramItemIndex(carray)`: the slot of the pair's hash, if present.
    fn get_bigram_item_index(&self, carray: &[u16]) -> Option<usize> {
        // PRIME_BIGRAM_LENGTH fits an i32.
        const PRIME: i32 = PRIME_BIGRAM_LENGTH as i32;
        let hash_id = hash1(carray);
        let (h1, h2) = probe(hash_id, hash2(carray), PRIME);
        let mut index = h1;
        let mut i = 1;
        let at = |index: i32| {
            usize::try_from(index)
                .ok()
                .map(|x| self.bigram_hash_table[x])
        };
        loop {
            let slot = at(index)?;
            if slot == 0 || slot == hash_id || i >= PRIME {
                break;
            }
            // Java's int arithmetic, wrapping (see the module docs).
            index = h1.wrapping_add(i.wrapping_mul(h2)).wrapping_rem(PRIME);
            // ARITH: i < PRIME_BIGRAM_LENGTH here.
            #[allow(clippy::arithmetic_side_effects)]
            {
                i += 1;
            }
        }
        (i < PRIME && at(index) == Some(hash_id)).then_some(index as usize)
    }

    /// `getFrequency(carray)`: the pair `word1@word2`'s frequency, 0 if
    /// absent.
    pub fn get_frequency(&self, carray: &[u16]) -> i32 {
        self.get_bigram_item_index(carray)
            .map_or(0, |i| self.frequency_table[i])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::serialized::test_util::{Value, Writer};

    fn u(s: &str) -> Vec<u16> {
        s.encode_utf16().collect()
    }

    #[test]
    fn the_vendored_dictionary() {
        let d = BigramDictionary::get_instance();
        assert!(d.get_frequency(&u("始##始@中国")) > 0);
        assert_eq!(d.get_frequency(&u("没有@这样的@三元组")), 0);
    }

    #[test]
    fn built_tables_and_errors() {
        let key = u("a@b");
        let (h1, _) = probe(hash1(&key), hash2(&key), PRIME_BIGRAM_LENGTH as i32);
        let mut hashes = vec![0i64; PRIME_BIGRAM_LENGTH];
        let mut freqs = vec![0i32; PRIME_BIGRAM_LENGTH];
        hashes[h1 as usize] = hash1(&key);
        freqs[h1 as usize] = 42;
        let mut w = Writer::new();
        w.write(&Value::Long(hashes.clone()));
        w.write(&Value::Int(freqs.clone()));
        let d = BigramDictionary::from_mem(&w.out).unwrap();
        assert_eq!(d.get_frequency(&key), 42);
        assert_eq!(d.get_frequency(&u("a@c")), 0);
        // A full table: absent keys probe to the end.
        let full = BigramDictionary {
            bigram_hash_table: vec![1; PRIME_BIGRAM_LENGTH],
            frequency_table: freqs,
        };
        assert_eq!(full.get_frequency(&u("x@y")), 0);
        let mut w = Writer::new();
        w.write(&Value::Long(vec![0; 3]));
        w.write(&Value::Int(vec![0; 3]));
        let e = BigramDictionary::from_mem(&w.out).err().unwrap();
        assert!(e.to_string().contains("PRIME_BIGRAM_LENGTH"));
        let mut w = Writer::new();
        w.write(&Value::Int(vec![0; 3]));
        let e = BigramDictionary::from_mem(&w.out).err().unwrap();
        assert!(e.to_string().contains("ClassCastException"));
        assert!(BigramDictionary::from_mem(&[1, 2]).is_err());
    }
}
