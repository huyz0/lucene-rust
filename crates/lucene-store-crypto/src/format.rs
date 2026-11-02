//! The encrypted file format: a header, then the plaintext in fixed-size
//! chunks, each sealed with AES-256-GCM under the file's own data key.
//!
//! ```text
//! header  magic "LRSE" | version u8 = 1 | algorithm u8 = 1 (AES-256-GCM)
//!         chunk_size u32 BE | key_id_len u16 BE | key_id (UTF-8)
//!         wrapped_len u16 BE | wrapped data key | nonce_prefix [8]
//! chunk i ciphertext (chunk_size bytes, the last one 0..=chunk_size) | tag [16]
//! ```
//!
//! - Every file has its own random 256-bit data key (DEK), stored only
//!   wrapped by the key provider's key-encryption key `key_id` (envelope
//!   encryption). Rotating the provider's current key changes what new files
//!   use; old files keep naming the key that wrapped them, and Lucene's merges
//!   rewrite them under the new one over time.
//! - Chunk `i`'s nonce is `nonce_prefix || i` (u32 BE). A DEK is never
//!   reused across files, so nonces never repeat under one key.
//! - Each chunk's associated data is the whole header plus a final-chunk
//!   flag, so a changed header, reordered or dropped chunks, and a truncated
//!   file all fail authentication. There is always a final chunk (an empty
//!   file has one, empty), and only the final chunk can be short.
//! - File names are deliberately *not* authenticated: Lucene renames
//!   `pending_segments_N` to `segments_N` to commit. Swapping two encrypted
//!   files under each other's names is caught by Lucene itself (each file's
//!   codec header carries its segment's random id).

use aes_gcm::aead::AeadInPlace;
use aes_gcm::{Aes256Gcm, KeyInit, Nonce, Tag};
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
pub const TAG_LEN: usize = 16;
/// The default chunk: 64 KiB of plaintext per tag (0.02% overhead).
pub const DEFAULT_CHUNK_SIZE: u32 = 64 * 1024;
/// The largest chunk accepted, so a corrupt header cannot ask for a huge
/// allocation.
pub const MAX_CHUNK_SIZE: u32 = 16 * 1024 * 1024;
const NONCE_PREFIX_LEN: usize = 8;
const FIXED_HEADER_LEN: usize = 4 + 1 + 1 + 4 + 2 + 2 + NONCE_PREFIX_LEN;

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
    /// The first eight bytes of every chunk's nonce.
    pub nonce_prefix: [u8; NONCE_PREFIX_LEN],
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
        out.extend_from_slice(&self.nonce_prefix);
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
        let prefix = r
            .take(NONCE_PREFIX_LEN)
            .ok_or_else(|| corrupt("truncated header"))?;
        let mut nonce_prefix = [0u8; NONCE_PREFIX_LEN];
        nonce_prefix.copy_from_slice(prefix);
        Ok((
            Header {
                chunk_size,
                key_id,
                wrapped_key,
                nonce_prefix,
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
    let per_chunk = u64::from(chunk_size).checked_add(TAG_LEN as u64)?;
    let chunks = body_len.div_ceil(per_chunk).max(1);
    let tags = chunks.checked_mul(TAG_LEN as u64)?;
    body_len.checked_sub(tags)
}

/// Seals and opens one file's chunks.
pub struct ChunkCipher {
    aead: Aes256Gcm,
    header: Vec<u8>,
    nonce_prefix: [u8; NONCE_PREFIX_LEN],
}

impl ChunkCipher {
    /// A cipher for the file whose encoded header is `header_bytes`.
    pub fn new(data_key: &DataKey, header: &Header, header_bytes: Vec<u8>) -> Result<Self> {
        let aead = Aes256Gcm::new_from_slice(&data_key[..])
            .map_err(|_| Error::IllegalArgument("data key is not 256 bits".into()))?;
        Ok(Self {
            aead,
            header: header_bytes,
            nonce_prefix: header.nonce_prefix,
        })
    }

    fn nonce_and_aad(&self, index: u32, last: bool) -> ([u8; 12], Vec<u8>) {
        let mut nonce = [0u8; 12];
        nonce[..NONCE_PREFIX_LEN].copy_from_slice(&self.nonce_prefix);
        nonce[NONCE_PREFIX_LEN..].copy_from_slice(&index.to_be_bytes());
        let mut aad = Vec::with_capacity(self.header.len().saturating_add(1));
        aad.extend_from_slice(&self.header);
        aad.push(u8::from(last));
        (nonce, aad)
    }

    /// Encrypts `chunk` in place and returns its tag.
    pub fn seal(&self, index: u32, last: bool, chunk: &mut [u8]) -> Result<[u8; TAG_LEN]> {
        let (nonce, aad) = self.nonce_and_aad(index, last);
        let tag = self
            .aead
            .encrypt_in_place_detached(Nonce::from_slice(&nonce), &aad, chunk)
            .map_err(|_| Error::IllegalArgument("chunk too large to encrypt".into()))?;
        let mut out = [0u8; TAG_LEN];
        out.copy_from_slice(&tag);
        Ok(out)
    }

    /// Decrypts `chunk` in place, checking `tag`.
    pub fn open(
        &self,
        index: u32,
        last: bool,
        chunk: &mut [u8],
        tag: &[u8],
        name: &str,
    ) -> Result<()> {
        let (nonce, aad) = self.nonce_and_aad(index, last);
        self.aead
            .decrypt_in_place_detached(Nonce::from_slice(&nonce), &aad, chunk, Tag::from_slice(tag))
            .map_err(|_| {
                Error::Corrupted(format!(
                    "encrypted file {name}: chunk {index} failed authentication \
                     (wrong key, tampered or truncated file)"
                ))
            })
    }

    /// Decrypts a whole body (everything after the header).
    pub fn open_body(&self, body: &[u8], chunk_size: u32, name: &str) -> Result<Vec<u8>> {
        let corrupt =
            || Error::Corrupted(format!("encrypted file {name}: body length {}", body.len()));
        let plain_len = plaintext_len(body.len() as u64, chunk_size).ok_or_else(corrupt)?;
        let plain_len = usize::try_from(plain_len).map_err(|_| corrupt())?;
        let mut out = Vec::with_capacity(plain_len);
        let per_chunk = (chunk_size as usize).saturating_add(TAG_LEN);
        let mut index: u32 = 0;
        let mut rest = body;
        loop {
            let take = rest.len().min(per_chunk);
            let last = take == rest.len();
            let (sealed, tail) = rest.split_at(take);
            let split = sealed.len().checked_sub(TAG_LEN).ok_or_else(corrupt)?;
            let (ciphertext, tag) = sealed.split_at(split);
            if !last && ciphertext.len() != chunk_size as usize {
                return Err(corrupt());
            }
            let start = out.len();
            out.extend_from_slice(ciphertext);
            self.open(index, last, &mut out[start..], tag, name)?;
            if last {
                return Ok(out);
            }
            rest = tail;
            index = index.checked_add(1).ok_or_else(corrupt)?;
        }
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
            nonce_prefix: [7; 8],
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
        // chunk 4, tag 16: 0 bytes -> one empty chunk of 16.
        assert_eq!(plaintext_len(16, 4), Some(0));
        assert_eq!(plaintext_len(20, 4), Some(4));
        assert_eq!(plaintext_len(21 + 16, 4), Some(5));
        assert_eq!(plaintext_len(15, 4), None);
        assert_eq!(plaintext_len(0, 4), None);
        assert_eq!(plaintext_len(30, 4), None);
    }

    #[test]
    fn bodies_round_trip_and_tampering_fails() {
        let key = DataKey::new([3; 32]);
        let h = header();
        let cipher = ChunkCipher::new(&key, &h, h.encode().unwrap()).unwrap();
        for len in [0usize, 1, 4, 5, 8, 9, 13] {
            let plain: Vec<u8> = (0..len as u8).collect();
            let mut body = Vec::new();
            let chunks: Vec<&[u8]> = if plain.is_empty() {
                vec![&[]]
            } else {
                plain.chunks(4).collect()
            };
            for (i, c) in chunks.iter().enumerate() {
                let mut c = c.to_vec();
                let tag = cipher
                    .seal(i as u32, i + 1 == chunks.len(), &mut c)
                    .unwrap();
                body.extend_from_slice(&c);
                body.extend_from_slice(&tag);
            }
            assert_eq!(cipher.open_body(&body, 4, "f").unwrap(), plain, "{len}");
            assert_eq!(plaintext_len(body.len() as u64, 4), Some(len as u64));
            // Flip a bit, or drop the last chunk: authentication fails.
            let mut flipped = body.clone();
            flipped[0] ^= 1;
            assert!(cipher.open_body(&flipped, 4, "f").is_err());
            if chunks.len() > 1 {
                let cut = body.len() - (body.len() - 20 * (chunks.len() - 1));
                assert!(cipher.open_body(&body[..cut], 4, "f").is_err(), "{len}");
            }
        }
        assert!(cipher.open_body(&[0; 3], 4, "f").is_err());
        // A short middle chunk is malformed even before authentication.
        assert!(cipher.open_body(&[0; 16 + 17], 4, "f").is_err());
    }
}
