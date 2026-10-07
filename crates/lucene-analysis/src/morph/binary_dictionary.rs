//! `org.apache.lucene.analysis.morph.BinaryDictionary`: the target map
//! (FST output or character class -> the word ids it names) and the entry
//! buffer a language's `MorphData` decodes.
//!
//! `$targetMap.dat`: `CodecUtil` header, `vint targetMapLength`,
//! `vint offsetsLength`, then `targetMapLength` vints, each
//! `delta << 1 | startsNewSource` (word ids accumulate the deltas in Java
//! `int`). `$buffer.dat`: header, `vint size`, then `size` bytes.
//!
//! Differs: lengths are checked against the bytes left before anything is
//! allocated (a vint is at least one byte), and an offsets array longer than
//! the target map can fill is refused up front -- Java allocates first and
//! fails later (`ArrayIndexOutOfBoundsException` or its own "targetMap file
//! format broken" `IOException`, which is the error given here). A source id
//! outside the map (from a hostile FST) names no words, where Java throws
//! `ArrayIndexOutOfBoundsException`.

use super::resource::{io_error, ResourceInput};
use crate::AnalysisError;

/// `BinaryDictionary.DICT_FILENAME_SUFFIX`.
pub const DICT_FILENAME_SUFFIX: &str = "$buffer.dat";
/// `BinaryDictionary.TARGETMAP_FILENAME_SUFFIX`.
pub const TARGETMAP_FILENAME_SUFFIX: &str = "$targetMap.dat";
/// `BinaryDictionary.POSDICT_FILENAME_SUFFIX`.
pub const POSDICT_FILENAME_SUFFIX: &str = "$posDict.dat";

/// `BinaryDictionary`'s target map and buffer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BinaryDictionary {
    /// `u32` (Java's `int[]`): half the cache lines of `usize` on the
    /// per-match lookup.
    target_map_offsets: Vec<u32>,
    target_map: Vec<i32>,
    buffer: Vec<u8>,
}

fn length(input: &mut ResourceInput<'_>) -> Result<usize, AnalysisError> {
    let n = input.read_vint()?;
    usize::try_from(n).map_err(|_| io_error("NegativeArraySizeException", n))
}

impl BinaryDictionary {
    /// `new BinaryDictionary(targetMap, dict, targetMapHeader, dictHeader,
    /// version)`.
    pub fn read(
        target_map_bytes: &[u8],
        dict_bytes: &[u8],
        target_map_codec_header: &str,
        dict_codec_header: &str,
        version: i32,
    ) -> Result<Self, AnalysisError> {
        let mut input = ResourceInput::new(target_map_bytes);
        input.check_header(target_map_codec_header, version, version)?;
        let map_len = length(&mut input)?;
        let offsets_len = length(&mut input)?;
        if map_len > input.remaining() {
            return Err(io_error("EOFException", "read past EOF"));
        }
        if offsets_len == 0 || offsets_len > map_len.saturating_add(1) {
            return Err(Self::broken(map_len, offsets_len, 0));
        }
        let mut target_map = Vec::with_capacity(map_len);
        let mut target_map_offsets = vec![0u32; offsets_len];
        // `map_len` is a non-negative vint, so it fits.
        let map_len_u32 = map_len as u32;
        let (mut accum, mut source_id) = (0i32, 0usize);
        for ofs in 0..map_len_u32 {
            let val = input.read_vint()?;
            if val & 0x01 != 0 {
                let slot = target_map_offsets
                    .get_mut(source_id)
                    .ok_or_else(|| Self::broken(map_len, offsets_len, source_id))?;
                *slot = ofs;
                source_id = source_id.saturating_add(1);
            }
            accum = accum.wrapping_add(((val as u32) >> 1) as i32);
            target_map.push(accum);
        }
        if source_id.checked_add(1) != Some(offsets_len) {
            return Err(Self::broken(map_len, offsets_len, source_id));
        }
        if let Some(last) = target_map_offsets.get_mut(source_id) {
            *last = map_len_u32;
        }

        let mut input = ResourceInput::new(dict_bytes);
        input.check_header(dict_codec_header, version, version)?;
        let size = length(&mut input)?;
        let buffer = input
            .read_bytes(size)
            .map_err(|_| io_error("EOFException", "Cannot read whole dictionary"))?
            .to_vec();
        Ok(BinaryDictionary {
            target_map_offsets,
            target_map,
            buffer,
        })
    }

    fn broken(map_len: usize, offsets_len: usize, source_id: usize) -> AnalysisError {
        io_error(
            "IOException",
            format!(
                "targetMap file format broken; targetMap.length={map_len}, targetMapOffsets.length={offsets_len}, sourceId={source_id}"
            ),
        )
    }

    /// `lookupWordIds(sourceId, ref)`: the word ids of `source_id`.
    #[inline]
    pub fn lookup_word_ids(&self, source_id: i32) -> &[i32] {
        let Ok(sid) = usize::try_from(source_id) else {
            return &[];
        };
        let (Some(&start), Some(&end)) = (
            self.target_map_offsets.get(sid),
            sid.checked_add(1)
                .and_then(|n| self.target_map_offsets.get(n)),
        ) else {
            return &[];
        };
        self.target_map
            .get(start as usize..end as usize)
            .unwrap_or(&[])
    }

    /// Every word id of the map (to validate the entries they point at).
    pub fn word_ids(&self) -> &[i32] {
        &self.target_map
    }

    /// The entry buffer (`ByteBuffer buffer`).
    pub fn buffer(&self) -> &[u8] {
        &self.buffer
    }
}

#[cfg(test)]
pub(crate) mod tests {
    #![allow(clippy::arithmetic_side_effects)]
    use super::*;
    use crate::morph::resource::test_util::{header, vlong};

    /// A target map of `sources`, each a list of word ids, as
    /// `BinaryDictionaryWriter.writeTargetMap` writes it.
    pub(crate) fn target_map_file(codec: &str, sources: &[&[i32]]) -> Vec<u8> {
        let mut vals = Vec::new();
        let mut prev = 0i32;
        for ids in sources {
            for (i, &id) in ids.iter().enumerate() {
                let delta = id - prev;
                prev = id;
                vals.push(((delta as u32) << 1 | u32::from(i == 0)) as u64);
            }
        }
        let mut b = header(codec, 1);
        vlong(&mut b, vals.len() as u64);
        vlong(&mut b, sources.len() as u64 + 1);
        for v in vals {
            vlong(&mut b, v);
        }
        b
    }

    pub(crate) fn buffer_file(codec: &str, bytes: &[u8]) -> Vec<u8> {
        let mut b = header(codec, 1);
        vlong(&mut b, bytes.len() as u64);
        b.extend_from_slice(bytes);
        b
    }

    #[test]
    fn target_map_round_trip() {
        let map = target_map_file("m", &[&[0, 8], &[12], &[16, 20, 24]]);
        let buf = buffer_file("d", &[1, 2, 3]);
        let d = BinaryDictionary::read(&map, &buf, "m", "d", 1).unwrap();
        assert_eq!(d.lookup_word_ids(0), [0, 8]);
        assert_eq!(d.lookup_word_ids(1), [12]);
        assert_eq!(d.lookup_word_ids(2), [16, 20, 24]);
        assert!(d.lookup_word_ids(3).is_empty());
        assert!(d.lookup_word_ids(-1).is_empty());
        assert_eq!(d.word_ids().len(), 6);
        assert_eq!(d.buffer(), [1, 2, 3]);
    }

    #[test]
    fn broken_maps_fail() {
        let buf = buffer_file("d", &[1]);
        // More offsets than sources.
        let mut b = header("m", 1);
        for v in [2u64, 5, 1, 1] {
            vlong(&mut b, v);
        }
        let e = BinaryDictionary::read(&b, &buf, "m", "d", 1).unwrap_err();
        assert!(
            e.to_string().contains("targetMap file format broken"),
            "{e}"
        );
        // More sources than offsets.
        let mut b = header("m", 1);
        for v in [2u64, 2, 1, 1] {
            vlong(&mut b, v);
        }
        assert!(BinaryDictionary::read(&b, &buf, "m", "d", 1).is_err());
        // Zero offsets.
        let mut b = header("m", 1);
        for v in [0u64, 0] {
            vlong(&mut b, v);
        }
        assert!(BinaryDictionary::read(&b, &buf, "m", "d", 1).is_err());
        // A length past the end, a negative length.
        let mut b = header("m", 1);
        for v in [1000u64, 2] {
            vlong(&mut b, v);
        }
        assert!(BinaryDictionary::read(&b, &buf, "m", "d", 1).is_err());
        let mut b = header("m", 1);
        b.extend_from_slice(&[0xFF, 0xFF, 0xFF, 0xFF, 0x0F]);
        assert!(BinaryDictionary::read(&b, &buf, "m", "d", 1).is_err());
        let map = target_map_file("m", &[&[0]]);
        let e = BinaryDictionary::read(&map, &buf[..buf.len() - 1], "m", "d", 1).unwrap_err();
        assert!(
            e.to_string().contains("Cannot read whole dictionary"),
            "{e}"
        );
        let mut neg = header("d", 1);
        neg.extend_from_slice(&[0xFF, 0xFF, 0xFF, 0xFF, 0x0F]);
        assert!(BinaryDictionary::read(&map, &neg, "m", "d", 1).is_err());
        for cut in 0..map.len() {
            assert!(BinaryDictionary::read(&map[..cut], &buf, "m", "d", 1).is_err());
        }
    }
}
