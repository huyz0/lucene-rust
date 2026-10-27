//! `java.util.zip.ZipInputStream` as `TestUtil.unzip` reads a test zip: the
//! local file headers one after another (never the central directory), each
//! entry stored or deflated, checked against its CRC-32 and sizes.
//!
//! Errors are `std::io::Error`s carrying Java's messages: `InvalidData` for
//! a `ZipException`, `UnexpectedEof` for an `EOFException`.
//!
//! Differs: ZIP64 size fields are not read -- test zips are far below 4 GiB.

use std::io;

use miniz_oxide::inflate::stream::{inflate, InflateState};
use miniz_oxide::{DataFormat, MZFlush, MZStatus};

const LOCSIG: u32 = 0x0403_4b50;
const EXTSIG: u32 = 0x0807_4b50;
const STORED: u16 = 0;
const DEFLATED: u16 = 8;

fn zip_error(message: impl Into<String>) -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidData,
        format!("ZipException: {}", message.into()),
    )
}

fn eof(message: &str) -> io::Error {
    io::Error::new(
        io::ErrorKind::UnexpectedEof,
        format!("EOFException: {message}"),
    )
}

/// One entry: its name and its (inflated) bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ZipEntry {
    /// The entry name (UTF-8, as `ZipInputStream`'s default `ZipCoder`).
    pub name: String,
    /// The entry's bytes (empty for a directory).
    pub data: Vec<u8>,
}

impl ZipEntry {
    /// `ZipEntry.isDirectory()`: the name ends with `/`.
    pub fn is_directory(&self) -> bool {
        self.name.ends_with('/')
    }
}

/// A `ZipInputStream` over a byte array.
pub struct ZipReader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

fn u16_at(b: &[u8], at: usize) -> u16 {
    u16::from_le_bytes([b[at], b[at + 1]])
}

fn u32_at(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]])
}

impl<'a> ZipReader<'a> {
    /// A reader at the start of `bytes`.
    pub fn new(bytes: &'a [u8]) -> Self {
        ZipReader { bytes, pos: 0 }
    }

    fn take(&mut self, n: usize, what: &str) -> io::Result<&'a [u8]> {
        let end = self
            .pos
            .checked_add(n)
            .filter(|&e| e <= self.bytes.len())
            .ok_or_else(|| eof(&format!("Unexpected end of ZIP input stream ({what})")))?;
        let s = &self.bytes[self.pos..end];
        self.pos = end;
        Ok(s)
    }

    /// `getNextEntry()` and the entry's bytes; `None` past the last local
    /// header (a short read or another signature, as `readLOC`).
    ///
    /// # Errors
    /// A truncated or corrupt entry, as `ZipInputStream`'s exceptions.
    pub fn next_entry(&mut self) -> io::Result<Option<ZipEntry>> {
        let Ok(header) = self.take(30, "header") else {
            return Ok(None);
        };
        if u32_at(header, 0) != LOCSIG {
            return Ok(None);
        }
        let flag = u16_at(header, 6);
        let method = u16_at(header, 8);
        let crc = u32_at(header, 14);
        let csize = u32_at(header, 18);
        let size = u32_at(header, 22);
        let name_len = usize::from(u16_at(header, 26));
        let extra_len = usize::from(u16_at(header, 28));
        let name = String::from_utf8_lossy(self.take(name_len, "name")?).into_owned();
        if flag & 1 == 1 {
            return Err(zip_error("encrypted ZIP entry not supported"));
        }
        let descriptor = flag & 8 == 8;
        if descriptor && method != DEFLATED {
            return Err(zip_error("only DEFLATED entries can have EXT descriptor"));
        }
        self.take(extra_len, "extra")?;
        let data = match method {
            STORED => {
                let data = self.take(size as usize, "stored entry")?.to_vec();
                if size != csize {
                    return Err(size_mismatch("size", size, csize as usize));
                }
                check_crc(crc, &data)?;
                data
            }
            DEFLATED => {
                let (data, consumed) = inflate_raw(&self.bytes[self.pos..])?;
                self.pos += consumed;
                let (crc, csize, size) = if descriptor {
                    self.read_descriptor()?
                } else {
                    (crc, csize, size)
                };
                if size as usize != data.len() {
                    return Err(size_mismatch("size", size, data.len()));
                }
                if csize as usize != consumed {
                    return Err(size_mismatch("compressed size", csize, consumed));
                }
                check_crc(crc, &data)?;
                data
            }
            _ => return Err(zip_error("invalid compression method")),
        };
        Ok(Some(ZipEntry { name, data }))
    }

    /// The data descriptor after a deflated entry: an optional signature,
    /// then CRC, compressed and uncompressed size.
    fn read_descriptor(&mut self) -> io::Result<(u32, u32, u32)> {
        let first = u32_at(self.take(4, "descriptor")?, 0);
        let crc = if first == EXTSIG {
            u32_at(self.take(4, "descriptor")?, 0)
        } else {
            first
        };
        let sizes = self.take(8, "descriptor")?;
        Ok((crc, u32_at(sizes, 0), u32_at(sizes, 4)))
    }
}

/// `ZipInputStream`'s message for an entry whose `what` is not the header's.
fn size_mismatch(what: &str, expected: u32, got: usize) -> io::Error {
    zip_error(format!(
        "invalid entry {what} (expected {expected} but got {got} bytes)"
    ))
}

fn check_crc(expected: u32, data: &[u8]) -> io::Result<()> {
    let got = crc32fast::hash(data);
    if got != expected {
        return Err(zip_error(format!(
            "invalid entry CRC (expected 0x{expected:x} but got 0x{got:x})"
        )));
    }
    Ok(())
}

/// Inflates the raw deflate stream at the start of `input`: the bytes and
/// how much input the stream took.
fn inflate_raw(input: &[u8]) -> io::Result<(Vec<u8>, usize)> {
    let mut state = InflateState::new_boxed(DataFormat::Raw);
    let mut out = Vec::new();
    let mut chunk = vec![0u8; 64 * 1024];
    let mut consumed = 0;
    loop {
        let r = inflate(&mut state, &input[consumed..], &mut chunk, MZFlush::None);
        consumed += r.bytes_consumed;
        out.extend_from_slice(&chunk[..r.bytes_written]);
        match r.status {
            Ok(MZStatus::StreamEnd) => return Ok((out, consumed)),
            Ok(_) if r.bytes_consumed > 0 || r.bytes_written > 0 => {}
            Ok(_) | Err(miniz_oxide::MZError::Buf) => {
                return Err(eof("Unexpected end of ZLIB input stream"));
            }
            Err(_) => return Err(zip_error("invalid stored block lengths")),
        }
    }
}

/// Every entry of `zip`, directories included, in local-header order.
///
/// # Errors
/// A truncated or corrupt entry.
pub fn entries(zip: &[u8]) -> io::Result<Vec<ZipEntry>> {
    let mut r = ZipReader::new(zip);
    let mut out = Vec::new();
    while let Some(e) = r.next_entry()? {
        out.push(e);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn local(
        name: &str,
        flag: u16,
        method: u16,
        crc: u32,
        sizes: (u32, u32),
        data: &[u8],
    ) -> Vec<u8> {
        let mut v = Vec::new();
        v.extend_from_slice(&LOCSIG.to_le_bytes());
        v.extend_from_slice(&20u16.to_le_bytes());
        v.extend_from_slice(&flag.to_le_bytes());
        v.extend_from_slice(&method.to_le_bytes());
        v.extend_from_slice(&[0; 4]);
        v.extend_from_slice(&crc.to_le_bytes());
        v.extend_from_slice(&sizes.0.to_le_bytes());
        v.extend_from_slice(&sizes.1.to_le_bytes());
        v.extend_from_slice(&(name.len() as u16).to_le_bytes());
        v.extend_from_slice(&2u16.to_le_bytes());
        v.extend_from_slice(name.as_bytes());
        v.extend_from_slice(&[0xAA, 0xBB]);
        v.extend_from_slice(data);
        v
    }

    #[test]
    fn stored_and_deflated_entries() {
        let text = b"hello hello hello hello";
        let crc = crc32fast::hash(text);
        let deflated = miniz_oxide::deflate::compress_to_vec(text, 6);
        let mut zip = local("a.txt", 0, STORED, crc, (23, 23), text);
        zip.extend(local(
            "b.txt",
            0,
            DEFLATED,
            crc,
            (deflated.len() as u32, 23),
            &deflated,
        ));
        // a deflated entry with a data descriptor, with and without its signature
        zip.extend(local("c.bin", 8, DEFLATED, 0, (0, 0), &deflated));
        zip.extend_from_slice(&EXTSIG.to_le_bytes());
        zip.extend_from_slice(&crc.to_le_bytes());
        zip.extend_from_slice(&(deflated.len() as u32).to_le_bytes());
        zip.extend_from_slice(&23u32.to_le_bytes());
        zip.extend(local("d/e.bin", 8, DEFLATED, 0, (0, 0), &deflated));
        zip.extend_from_slice(&crc.to_le_bytes());
        zip.extend_from_slice(&(deflated.len() as u32).to_le_bytes());
        zip.extend_from_slice(&23u32.to_le_bytes());
        zip.extend(local("d/", 0, STORED, 0, (0, 0), &[]));
        // the central directory ends the local headers
        zip.extend_from_slice(&0x0201_4b50u32.to_le_bytes());
        let all = entries(&zip).unwrap();
        assert_eq!(all.len(), 5);
        assert!(all[..4].iter().all(|e| e.data == text && !e.is_directory()));
        assert!(all[4].is_directory());
        assert_eq!(all[1].name, "b.txt");
        assert!(entries(&[]).unwrap().is_empty());
    }

    #[test]
    fn corrupt_entries_are_zip_exceptions() {
        let text = b"abc";
        let crc = crc32fast::hash(text);
        let deflated = miniz_oxide::deflate::compress_to_vec(text, 6);
        let msg = |z: Vec<u8>| entries(&z).unwrap_err().to_string();
        assert!(msg(local("a", 1, STORED, crc, (3, 3), text)).contains("encrypted"));
        assert!(msg(local("a", 8, STORED, crc, (3, 3), text)).contains("EXT descriptor"));
        assert!(msg(local("a", 0, 9, crc, (3, 3), text)).contains("compression method"));
        assert!(msg(local("a", 0, STORED, crc ^ 1, (3, 3), text)).contains("invalid entry CRC"));
        assert!(msg(local("a", 0, STORED, crc, (2, 3), text)).contains("invalid entry size"));
        assert!(msg(local("a", 0, STORED, crc, (9, 9), text)).contains("EOFException"));
        let wrong_size = local("a", 0, DEFLATED, crc, (deflated.len() as u32, 4), &deflated);
        assert!(msg(wrong_size).contains("invalid entry size"));
        assert!(msg(local("a", 0, DEFLATED, crc, (99, 3), &deflated)).contains("compressed size"));
        assert!(msg(local("a", 0, DEFLATED, crc, (2, 3), &deflated[..2])).contains("EOFException"));
        assert!(
            msg(local("a", 0, DEFLATED, crc, (2, 3), &[0xFF, 0xFF, 0xFF])).contains("ZipException")
        );
        assert!(msg(local("a", 8, DEFLATED, 0, (0, 0), &deflated)).contains("EOFException"));
        let mut cut = local("abc", 0, STORED, crc, (3, 3), text);
        cut.truncate(32);
        assert_eq!(
            entries(&cut).unwrap_err().kind(),
            io::ErrorKind::UnexpectedEof
        );
    }
}
