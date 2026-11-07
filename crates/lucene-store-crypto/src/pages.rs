//! Encrypted files read page by page through a directory that fetches and
//! caches pages itself ([`lucene_store::CachedBlobDirectory`]).
//!
//! The directory cuts an encrypted file into pages of whole chunks after
//! its header ([`lucene_store::UnitLayout`]): its cache holds ciphertext,
//! and each page is authenticated and decrypted on its own when it is
//! brought into memory. Only the header is read when the file opens.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use lucene_store::{Directory, Error, PageCodec, PageDecoder, Result, UnitLayout};

use crate::format::{plaintext_len, ChunkCipher, Header, CHUNK_OVERHEAD};
use crate::key_provider::KeyProvider;

/// Bytes read first for a header: enough for any key id and wrapped key a
/// real provider produces.
const HEADER_PROBE: usize = 4096;
/// The longest possible header: the fixed fields and two 16-bit-length
/// fields at their maximum.
const MAX_HEADER: usize = 12 + 2 * 65535;

/// `name`'s header and its encoded bytes, read by range.
pub(crate) fn read_header(dir: &dyn Directory, name: &str) -> Result<(Header, Vec<u8>)> {
    let probe = dir.read_at(name, 0, HEADER_PROBE)?;
    let bytes = match Header::decode(&probe, name) {
        Ok(_) => probe,
        Err(_) if probe.len() == HEADER_PROBE => dir.read_at(name, 0, MAX_HEADER)?,
        Err(e) => return Err(e),
    };
    let (header, len) = Header::decode(&bytes, name)?;
    let mut bytes = bytes;
    bytes.truncate(len);
    Ok((header, bytes))
}

/// The [`PageDecoder`] an [`crate::EncryptedDirectory`] hands its inner
/// directory: unwraps the file's data key and decrypts pages of chunks.
pub(crate) struct EncryptedDecoder<'a> {
    pub(crate) keys: &'a dyn KeyProvider,
    /// The directory's header lengths, filled as headers are read so its
    /// [`Directory::file_length`] does not read them again.
    pub(crate) header_lens: &'a Mutex<HashMap<String, (u64, u32)>>,
}

impl PageDecoder for EncryptedDecoder<'_> {
    fn codec(
        &self,
        name: &str,
        stored_len: u64,
        dir: &dyn Directory,
    ) -> Result<Arc<dyn PageCodec>> {
        let (header, header_bytes) = read_header(dir, name)?;
        let data_key = self.keys.unwrap_key(&header.key_id, &header.wrapped_key)?;
        let header_len = header_bytes.len() as u64;
        self.header_lens
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .insert(name.to_string(), (header_len, header.chunk_size));
        let cipher = ChunkCipher::new(&data_key, header_bytes)?;
        let corrupt = || Error::Corrupted(format!("encrypted file {name}: malformed length"));
        let body = stored_len.checked_sub(header_len).ok_or_else(corrupt)?;
        let plain_len = plaintext_len(body, header.chunk_size).ok_or_else(corrupt)?;
        let per_chunk = u64::from(header.chunk_size).saturating_add(CHUNK_OVERHEAD as u64);
        let total = usize::try_from(body.div_ceil(per_chunk).max(1)).map_err(|_| corrupt())?;
        Ok(Arc::new(EncryptedPages {
            cipher,
            chunk_size: header.chunk_size,
            header_len,
            plain_len,
            total,
            name: name.to_string(),
        }))
    }
}

/// One encrypted file's chunk layout and cipher.
struct EncryptedPages {
    cipher: ChunkCipher,
    chunk_size: u32,
    header_len: u64,
    plain_len: u64,
    total: usize,
    name: String,
}

impl PageCodec for EncryptedPages {
    fn layout(&self) -> UnitLayout {
        UnitLayout {
            origin: self.header_len,
            stored_unit: u64::from(self.chunk_size).saturating_add(CHUNK_OVERHEAD as u64),
            plain_unit: u64::from(self.chunk_size),
            plain_len: self.plain_len,
        }
    }

    fn decode(&self, first_unit: u64, stored: &[u8], out: &mut [u8]) -> Result<()> {
        let first = usize::try_from(first_unit).map_err(|_| {
            Error::Corrupted(format!("encrypted file {}: chunk out of range", self.name))
        })?;
        self.cipher
            .open_chunks(stored, first, self.total, out, self.chunk_size, &self.name)
    }
}
