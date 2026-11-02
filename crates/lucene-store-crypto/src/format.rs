//! The encrypted file format: a header, then the plaintext in fixed-size
//! chunks, each sealed with AES-256-GCM under the file's own data key.
//!
//! ```text
//! header  magic "LRSE" | version u8 = 1 | algorithm u8 = 1 (AES-256-GCM)
//!         chunk_size u32 BE | key_id_len u16 BE | key_id (UTF-8)
//!         wrapped_len u16 BE | wrapped data key
//! chunk i nonce [12] | ciphertext (chunk_size bytes, the last 0..=chunk_size) | tag [16]
//! ```
//!
//! - Every file has its own random 256-bit data key (DEK), stored only
//!   wrapped by the key provider's key-encryption key `key_id` (envelope
//!   encryption). Rotating the provider's current key changes what new files
//!   use; old files keep naming the key that wrapped them, and Lucene's merges
//!   rewrite them under the new one over time.
//! - Each chunk's 96-bit nonce is drawn by the crypto module's DRBG (the
//!   FIPS-approved way to use GCM) and stored before it. A data key seals at
//!   most 2^32 chunks, NIST's bound for random GCM nonces under one key.
//! - Each chunk's associated data is the whole header, the chunk's index
//!   (u32 BE) and a final-chunk flag, so a changed header, reordered or
//!   dropped chunks, and a truncated file all fail authentication. There is always a final chunk (an empty
//!   file has one, empty), and only the final chunk can be short.
//! - File names are deliberately *not* authenticated: Lucene renames
//!   `pending_segments_N` to `segments_N` to commit. Swapping two encrypted
//!   files under each other's names is caught by Lucene itself (each file's
//!   codec header carries its segment's random id).

use crate::backend::Aes256Gcm;
use lucene_store::{Error, Result};
use zeroize::Zeroizing;

/// `"LRSE"`: lucene-rust store encryption.
pub const MAGIC: [u8; 4] = *b"LRSE";
/// The format version this crate writes and reads.
pub const VERSION: u8 = 1;
/// AES-256-GCM, the one algorithm so far.
pub const ALG_AES_256_GCM: u8 = 1;
/// A data key's length: AES-256.
pub const DATA_KEY_LEN: usize = 32;
/// An AES-GCM tag.
pub const TAG_LEN: usize = crate::backend::TAG_LEN;
/// A chunk's nonce.
pub const NONCE_LEN: usize = crate::backend::NONCE_LEN;
/// The bytes each chunk adds: its nonce and tag.
pub const CHUNK_OVERHEAD: usize = NONCE_LEN + TAG_LEN;
/// The default chunk: 64 KiB of plaintext per nonce and tag (0.04% overhead).
pub const DEFAULT_CHUNK_SIZE: u32 = 64 * 1024;
/// The largest chunk accepted, so a corrupt header cannot ask for a huge
/// allocation.
pub const MAX_CHUNK_SIZE: u32 = 16 * 1024 * 1024;
const FIXED_HEADER_LEN: usize = 4 + 1 + 1 + 4 + 2 + 2;
/// Chunks per parallel decryption task: 1 MiB of plaintext at the default
/// chunk size. A file of at most this many chunks is decrypted on the
/// calling thread.
const CHUNKS_PER_TASK: usize = 16;

/// A 256-bit key that is wiped from memory when dropped.
pub type DataKey = Zeroizing<[u8; DATA_KEY_LEN]>;

/// A parsed header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Header {
    /// Plaintext bytes per chunk.
    pub chunk_size: u32,
    /// The key provider's id of the key that wrapped the data key.
    pub key_id: String,
    /// The data key, wrapped.
    pub wrapped_key: Vec<u8>,
}

impl Header {
    /// The header's bytes, as written at the start of the file (and used as
    /// every chunk's associated data).
    pub fn encode(&self) -> Result<Vec<u8>> {
        let key_id_len = u16::try_from(self.key_id.len())
            .map_err(|_| Error::IllegalArgument("key id longer than 65535 bytes".into()))?;
        let wrapped_len = u16::try_from(self.wrapped_key.len())
            .map_err(|_| Error::IllegalArgument("wrapped key longer than 65535 bytes".into()))?;
        let mut out = Vec::with_capacity(
            FIXED_HEADER_LEN
                .saturating_add(self.key_id.len())
                .saturating_add(self.wrapped_key.len()),
        );
        out.extend_from_slice(&MAGIC);
        out.push(VERSION);
        out.push(ALG_AES_256_GCM);
        out.extend_from_slice(&self.chunk_size.to_be_bytes());
        out.extend_from_slice(&key_id_len.to_be_bytes());
        out.extend_from_slice(self.key_id.as_bytes());
        out.extend_from_slice(&wrapped_len.to_be_bytes());
        out.extend_from_slice(&self.wrapped_key);
        Ok(out)
    }

    /// Parses the header at the start of `file`, returning it and its
    /// length in bytes.
    pub fn decode(file: &[u8], name: &str) -> Result<(Header, usize)> {
        let corrupt = |what: &str| Error::Corrupted(format!("encrypted file {name}: {what}"));
        let mut r = Reader {
            bytes: file,
            pos: 0,
        };
        if r.take(4).ok_or_else(|| corrupt("too short for a header"))? != MAGIC {
            return Err(corrupt(
                "not an encrypted file (no LRSE magic); is it a plaintext index?",
            ));
        }
        let version = r.u8().ok_or_else(|| corrupt("truncated header"))?;
        if version != VERSION {
            return Err(corrupt(&format!("unsupported format version {version}")));
        }
        let alg = r.u8().ok_or_else(|| corrupt("truncated header"))?;
        if alg != ALG_AES_256_GCM {
            return Err(corrupt(&format!("unsupported algorithm {alg}")));
        }
        let chunk_size = r.u32().ok_or_else(|| corrupt("truncated header"))?;
        if chunk_size == 0 || chunk_size > MAX_CHUNK_SIZE {
            return Err(corrupt(&format!("chunk size {chunk_size} out of range")));
        }
        let key_id_len = r.u16().ok_or_else(|| corrupt("truncated header"))?;
        let key_id = r
            .take(usize::from(key_id_len))
            .ok_or_else(|| corrupt("truncated key id"))?;
        let key_id =
            String::from_utf8(key_id.to_vec()).map_err(|_| corrupt("key id is not UTF-8"))?;
        let wrapped_len = r.u16().ok_or_else(|| corrupt("truncated header"))?;
        let wrapped_key = r
            .take(usize::from(wrapped_len))
            .ok_or_else(|| corrupt("truncated wrapped key"))?
            .to_vec();
        Ok((
            Header {
                chunk_size,
                key_id,
                wrapped_key,
            },
            r.pos,
        ))
    }
}

struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        let end = self.pos.checked_add(n)?;
        let out = self.bytes.get(self.pos..end)?;
        self.pos = end;
        Some(out)
    }
    fn u8(&mut self) -> Option<u8> {
        self.take(1).map(|b| b[0])
    }
    fn u16(&mut self) -> Option<u16> {
        self.take(2).map(|b| u16::from_be_bytes([b[0], b[1]]))
    }
    fn u32(&mut self) -> Option<u32> {
        self.take(4)
            .map(|b| u32::from_be_bytes([b[0], b[1], b[2], b[3]]))
    }
}

/// The plaintext length of an encrypted body (everything after the header)
/// of `body_len` bytes in chunks of `chunk_size`, or `None` when no
/// well-formed body has that length.
pub fn plaintext_len(body_len: u64, chunk_size: u32) -> Option<u64> {
    let per_chunk = u64::from(chunk_size).checked_add(CHUNK_OVERHEAD as u64)?;
    let chunks = body_len.div_ceil(per_chunk).max(1);
    let overhead = chunks.checked_mul(CHUNK_OVERHEAD as u64)?;
    body_len.checked_sub(overhead)
}

/// Seals and opens one file's chunks.
pub struct ChunkCipher {
    aead: Aes256Gcm,
    /// The header, then five bytes [`ChunkCipher::aad`] fills per chunk.
    aad: Vec<u8>,
}

impl ChunkCipher {
    /// A cipher for the file whose encoded header is `header_bytes`.
    pub fn new(data_key: &DataKey, header_bytes: Vec<u8>) -> Result<Self> {
        let aead = Aes256Gcm::new(data_key)?;
        let mut aad = header_bytes;
        aad.extend_from_slice(&[0; 5]);
        Ok(Self { aead, aad })
    }

    /// Chunk `index`'s associated data, written into `aad` (a copy of
    /// `self.aad`): the header, the index, the final flag.
    fn aad(aad: &mut [u8], index: u32, last: bool) -> &[u8] {
        let n = aad.len();
        let tail = &mut aad[n.saturating_sub(5)..];
        tail[..4].copy_from_slice(&index.to_be_bytes());
        tail[4] = u8::from(last);
        aad
    }

    /// Encrypts `chunk` in place as chunk `index`, returning its nonce and
    /// tag.
    pub fn seal(
        &mut self,
        index: u32,
        last: bool,
        chunk: &mut [u8],
    ) -> Result<([u8; NONCE_LEN], [u8; TAG_LEN])> {
        let aad = Self::aad(&mut self.aad, index, last);
        self.aead.seal(aad, chunk)
    }

    /// Decrypts a whole body (everything after the header): chunks in
    /// parallel on the rayon pool when there are more than one task's worth.
    pub fn open_body(&self, body: &[u8], chunk_size: u32, name: &str) -> Result<Vec<u8>> {
        let corrupt =
            || Error::Corrupted(format!("encrypted file {name}: body length {}", body.len()));
        let plain_len = plaintext_len(body.len() as u64, chunk_size).ok_or_else(corrupt)?;
        let plain_len = usize::try_from(plain_len).map_err(|_| corrupt())?;
        let chunk = chunk_size as usize;
        let per_chunk = chunk.saturating_add(CHUNK_OVERHEAD);
        let chunks = body.len().div_ceil(per_chunk).max(1);
        if u32::try_from(chunks).is_err() {
            return Err(corrupt());
        }
        // The last chunk carries at least its nonce and tag (a short
        // remainder would have made `plaintext_len` fail).
        let mut out = vec![0u8; plain_len];
        let task_plain = chunk.saturating_mul(CHUNKS_PER_TASK);
        let task_sealed = per_chunk.saturating_mul(CHUNKS_PER_TASK);
        if chunks <= CHUNKS_PER_TASK {
            self.open_run(body, 0, chunks, &mut out, chunk, name)?;
        } else {
            use rayon::prelude::*;
            out.par_chunks_mut(task_plain)
                .zip(body.par_chunks(task_sealed))
                .enumerate()
                .try_for_each(|(task, (plain, sealed))| {
                    let first = task.saturating_mul(CHUNKS_PER_TASK);
                    self.open_run(sealed, first, chunks, plain, chunk, name)
                })?;
        }
        Ok(out)
    }

    /// Opens the consecutive chunks in `sealed`, the first of them chunk
    /// `first` of `total`, into `plain`. Each chunk's ciphertext and tag are
    /// copied into its place in `plain` and opened there (the tag's 16 bytes
    /// land where the next chunk goes, before it is written); a chunk whose
    /// tag would run past `plain` goes through a scratch buffer instead.
    fn open_run(
        &self,
        sealed: &[u8],
        first: usize,
        total: usize,
        plain: &mut [u8],
        chunk: usize,
        name: &str,
    ) -> Result<()> {
        let mut aad = self.aad.clone();
        let mut scratch = Vec::new();
        let per_chunk = chunk.saturating_add(CHUNK_OVERHEAD);
        for (j, piece) in sealed.chunks(per_chunk).enumerate() {
            let index = first.saturating_add(j);
            let last = index.saturating_add(1) == total;
            let bad = || {
                Error::Corrupted(format!(
                    "encrypted file {name}: chunk {index} failed authentication \
                     (wrong key, tampered or truncated file)"
                ))
            };
            let ct_len = piece.len().checked_sub(CHUNK_OVERHEAD).ok_or_else(bad)?;
            if !last && ct_len != chunk {
                return Err(bad());
            }
            let (nonce, ct_and_tag) = piece.split_at(NONCE_LEN);
            let nonce: &[u8; NONCE_LEN] = nonce.try_into().map_err(|_| bad())?;
            let at = j.saturating_mul(chunk);
            let index32 = u32::try_from(index).map_err(|_| bad())?;
            let aad = Self::aad(&mut aad, index32, last);
            let end = at.saturating_add(ct_and_tag.len());
            if end <= plain.len() {
                let dst = &mut plain[at..end];
                dst.copy_from_slice(ct_and_tag);
                self.aead.open(nonce, aad, dst).map_err(|_| bad())?;
            } else {
                scratch.clear();
                scratch.extend_from_slice(ct_and_tag);
                self.aead
                    .open(nonce, aad, &mut scratch)
                    .map_err(|_| bad())?;
                let dst = plain
                    .get_mut(at..at.saturating_add(ct_len))
                    .ok_or_else(bad)?;
                dst.copy_from_slice(&scratch[..ct_len]);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    // Test fixtures' own arithmetic -- see `docs/arithmetic-gate.md`'s "Test code".
    #![allow(clippy::arithmetic_side_effects)]
    use super::*;

    fn header() -> Header {
        Header {
            chunk_size: 4,
            key_id: "k1".into(),
            wrapped_key: vec![9; 5],
        }
    }

    #[test]
    fn a_header_round_trips_and_every_truncation_is_reported() {
        let h = header();
        let bytes = h.encode().unwrap();
        let (back, len) = Header::decode(&bytes, "f").unwrap();
        assert_eq!((back, len), (h, bytes.len()));
        for cut in 0..bytes.len() {
            let err = Header::decode(&bytes[..cut], "f").unwrap_err().to_string();
            assert!(err.contains("encrypted file f"), "{cut}: {err}");
        }
    }

    #[test]
    fn a_header_rejects_bad_fields() {
        let mut h = header();
        let good = h.encode().unwrap();
        let mut bad = good.clone();
        bad[0] = b'X';
        assert!(Header::decode(&bad, "f")
            .unwrap_err()
            .to_string()
            .contains("plaintext"));
        let mut bad = good.clone();
        bad[4] = 2;
        assert!(Header::decode(&bad, "f")
            .unwrap_err()
            .to_string()
            .contains("version 2"));
        let mut bad = good.clone();
        bad[5] = 9;
        assert!(Header::decode(&bad, "f")
            .unwrap_err()
            .to_string()
            .contains("algorithm 9"));
        let mut bad = good.clone();
        bad[6..10].copy_from_slice(&0u32.to_be_bytes());
        assert!(Header::decode(&bad, "f")
            .unwrap_err()
            .to_string()
            .contains("chunk size 0"));
        let mut bad = good.clone();
        bad[12] = 0xff;
        assert!(Header::decode(&bad, "f")
            .unwrap_err()
            .to_string()
            .contains("UTF-8"));
        h.key_id = "x".repeat(70_000);
        assert!(h.encode().is_err());
        h.key_id = "k".into();
        h.wrapped_key = vec![0; 70_000];
        assert!(h.encode().is_err());
    }

    #[test]
    fn plaintext_len_inverts_the_chunking() {
        // chunk 4, overhead 28: 0 bytes -> one empty chunk of 28.
        assert_eq!(plaintext_len(28, 4), Some(0));
        assert_eq!(plaintext_len(32, 4), Some(4));
        assert_eq!(plaintext_len(32 + 29, 4), Some(5));
        assert_eq!(plaintext_len(27, 4), None);
        assert_eq!(plaintext_len(0, 4), None);
        assert_eq!(plaintext_len(32 + 20, 4), None);
    }

    /// Seals `plain` the way the writer does.
    fn seal_body(cipher: &mut ChunkCipher, plain: &[u8], chunk: usize) -> Vec<u8> {
        let pieces: Vec<&[u8]> = if plain.is_empty() {
            vec![&[]]
        } else {
            plain.chunks(chunk).collect()
        };
        let mut body = Vec::new();
        for (i, c) in pieces.iter().enumerate() {
            let mut c = c.to_vec();
            let (nonce, tag) = cipher
                .seal(i as u32, i + 1 == pieces.len(), &mut c)
                .unwrap();
            body.extend_from_slice(&nonce);
            body.extend_from_slice(&c);
            body.extend_from_slice(&tag);
        }
        body
    }

    #[test]
    fn bodies_round_trip_and_tampering_fails() {
        let key = DataKey::new([3; 32]);
        let h = header();
        let mut cipher = ChunkCipher::new(&key, h.encode().unwrap()).unwrap();
        for len in [0usize, 1, 4, 5, 8, 9, 13] {
            let plain: Vec<u8> = (0..len as u8).collect();
            let body = seal_body(&mut cipher, &plain, 4);
            assert_eq!(cipher.open_body(&body, 4, "f").unwrap(), plain, "{len}");
            assert_eq!(plaintext_len(body.len() as u64, 4), Some(len as u64));
            let mut flipped = body.clone();
            flipped[NONCE_LEN] ^= 1;
            assert!(cipher.open_body(&flipped, 4, "f").is_err());
            if len > 4 {
                // Drop the last chunk: the new last one was not sealed final.
                let n = len.div_ceil(4);
                let cut = (n - 1) * 32;
                assert!(cipher.open_body(&body[..cut], 4, "f").is_err(), "{len}");
                // Swap the first two chunks: the index is authenticated.
                if n > 2 {
                    let mut swapped = body[32..64].to_vec();
                    swapped.extend_from_slice(&body[..32]);
                    swapped.extend_from_slice(&body[64..]);
                    assert!(cipher.open_body(&swapped, 4, "f").is_err(), "{len}");
                }
            }
        }
        assert!(cipher.open_body(&[0; 3], 4, "f").is_err());
        // A short middle chunk is malformed even before authentication.
        assert!(cipher.open_body(&[0; 28 + 29], 4, "f").is_err());
    }

    #[test]
    fn large_bodies_decrypt_in_parallel_at_every_tail_length() {
        let key = DataKey::new([5; 32]);
        let mut cipher = ChunkCipher::new(&key, header().encode().unwrap()).unwrap();
        // 16-chunk tasks of 4-byte chunks; tails that do and do not leave
        // room for a spilled tag.
        for len in [64usize, 65, 67, 68, 79, 80, 129, 4 * 16 * 5 + 3] {
            let plain: Vec<u8> = (0..len).map(|i| (i * 7) as u8).collect();
            let body = seal_body(&mut cipher, &plain, 4);
            assert_eq!(cipher.open_body(&body, 4, "f").unwrap(), plain, "{len}");
            let mut bad = body.clone();
            let last = bad.len() - 1;
            bad[last] ^= 1;
            let err = cipher.open_body(&bad, 4, "f").unwrap_err().to_string();
            assert!(err.contains("failed authentication"), "{err}");
        }
    }
}
