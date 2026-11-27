//! A sidecar object holding the small files a reader reads whole at open,
//! so opening an index on an object store costs one request for them
//! instead of several per segment.
//!
//! The bundle is one extra object next to the index files. It changes no
//! Lucene file: `segments_N` and every segment file stay byte-identical,
//! and its name matches none of the patterns Lucene's file deleter owns
//! (`segments*`, `pending_*`, `_<segment>.<ext>`), so a stock reader or
//! writer ignores it. A reader that finds it missing, unreadable or out of
//! date reads each file by range as before: an entry is used only when the
//! store's listing has the same name with the same length, and files are
//! written once, so equal name and length mean equal bytes.
//!
//! Layout: magic, entry count, then per entry the name (u16 length +
//! UTF-8), the stored length (u64) and the bytes; a CRC-32 of everything
//! before it ends the object.

use crate::blob_directory::BlobStore;
use crate::error::Result;

/// The sidecar's object name.
pub const OPEN_BUNDLE_NAME: &str = "accel-open-bundle.bin";

const MAGIC: &[u8; 8] = b"ACCLOPN1";

/// What [`write_open_bundle`] stored.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BundleWritten {
    /// Files in the bundle.
    pub files: usize,
    /// The bundle's size.
    pub bytes: u64,
}

/// What priming a cache from a bundle did.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BundlePrimed {
    /// Files taken from the bundle into the cache.
    pub primed: usize,
    /// Bytes of those files.
    pub bytes: u64,
    /// Entries skipped because the listing had no file of that name and
    /// length.
    pub stale: usize,
    /// Whether a bundle was found and read.
    pub found: bool,
}

fn generation(name: &str) -> Option<u64> {
    u64::from_str_radix(name.strip_prefix("segments_")?, 36).ok()
}

/// Writes the bundle for the store's current newest commit: its
/// `segments_N` and every file `is_bundled` accepts (of at most `max_file`
/// bytes) that the listing shows.
pub fn write_open_bundle<S: BlobStore>(
    store: &S,
    is_bundled: &dyn Fn(&str) -> bool,
    max_file: u64,
) -> Result<BundleWritten> {
    let listing: Vec<(String, u64)> = match store.list_with_sizes()? {
        Some(l) => l,
        None => {
            let mut v = Vec::new();
            for n in store.list()? {
                let len = store.size(&n)?;
                v.push((n, len));
            }
            v
        }
    };
    let newest = listing
        .iter()
        .filter_map(|(n, _)| generation(n).map(|g| (g, n.clone())))
        .max()
        .map(|(_, n)| n);
    let mut out = Vec::new();
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&0u32.to_le_bytes());
    let mut files = 0usize;
    for (name, len) in &listing {
        if name == OPEN_BUNDLE_NAME || *len > max_file {
            continue;
        }
        let take = if name.starts_with("segments_") {
            Some(name) == newest.as_ref()
        } else {
            is_bundled(name)
        };
        if !take {
            continue;
        }
        let bytes = store.get(name)?;
        let name_bytes = name.as_bytes();
        out.extend_from_slice(&(name_bytes.len() as u16).to_le_bytes());
        out.extend_from_slice(name_bytes);
        out.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
        out.extend_from_slice(&bytes);
        files = files.saturating_add(1);
    }
    out[8..12].copy_from_slice(&(files as u32).to_le_bytes());
    let crc = crc32fast::hash(&out);
    out.extend_from_slice(&crc.to_le_bytes());
    let bytes = out.len() as u64;
    store.put(OPEN_BUNDLE_NAME, out)?;
    Ok(BundleWritten { files, bytes })
}

/// The entries of a bundle's bytes, or `None` when they are not a whole,
/// intact bundle.
// ARITH: `pos` only ever advances past a range `get` just returned, so it
// never exceeds `body.len()`; the one length read from the file, `len`, is
// added with `checked_add`. The other additions use small constants.
#[allow(clippy::arithmetic_side_effects)]
pub fn parse_open_bundle(buf: &[u8]) -> Option<Vec<(String, &[u8])>> {
    if buf.len() < 16 || &buf[..8] != MAGIC {
        return None;
    }
    let (body, tail) = buf.split_at(buf.len() - 4);
    if crc32fast::hash(body) != u32::from_le_bytes(tail.try_into().ok()?) {
        return None;
    }
    let count = u32::from_le_bytes(body[8..12].try_into().ok()?) as usize;
    let mut pos = 12usize;
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        let nl = u16::from_le_bytes(body.get(pos..pos + 2)?.try_into().ok()?) as usize;
        pos += 2;
        let name = std::str::from_utf8(body.get(pos..pos + nl)?)
            .ok()?
            .to_string();
        pos += nl;
        let len = u64::from_le_bytes(body.get(pos..pos + 8)?.try_into().ok()?) as usize;
        pos += 8;
        let end = pos.checked_add(len)?;
        let bytes = body.get(pos..end)?;
        pos = end;
        out.push((name, bytes));
    }
    (pos == body.len()).then_some(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blob_directory::InMemoryBlobStore;

    #[test]
    fn a_bundle_round_trips_and_ignores_the_bundle_itself_and_old_commits() {
        let store = InMemoryBlobStore::new();
        store.put("segments_1", vec![1; 10]).unwrap();
        store.put("segments_2", vec![2; 20]).unwrap();
        store.put("_0.si", vec![3; 30]).unwrap();
        store.put("_0.doc", vec![4; 40]).unwrap();
        let w = write_open_bundle(&store, &|n: &str| n.ends_with(".si"), 1 << 20).unwrap();
        assert_eq!(w.files, 2);
        // Writing again, with the bundle now in the listing, adds nothing.
        let again = write_open_bundle(&store, &|n: &str| n.ends_with(".si"), 1 << 20).unwrap();
        assert_eq!(again, w);
        let bytes = store.get(OPEN_BUNDLE_NAME).unwrap();
        let entries = parse_open_bundle(&bytes).unwrap();
        let names: Vec<&str> = entries.iter().map(|(n, _)| n.as_str()).collect();
        assert!(names.contains(&"segments_2") && names.contains(&"_0.si"));
        assert!(!names.contains(&"segments_1"));
    }

    #[test]
    fn damaged_bundles_are_refused() {
        let store = InMemoryBlobStore::new();
        store.put("segments_1", vec![1; 64]).unwrap();
        write_open_bundle(&store, &|_: &str| false, 1 << 20).unwrap();
        let good = store.get(OPEN_BUNDLE_NAME).unwrap()[..].to_vec();
        assert!(parse_open_bundle(&good).is_some());
        for cut in [0, 5, good.len() - 1] {
            assert!(parse_open_bundle(&good[..cut]).is_none());
        }
        let mut flipped = good.clone();
        flipped[20] ^= 1;
        assert!(parse_open_bundle(&flipped).is_none());
        let mut magic = good;
        magic[0] = b'X';
        assert!(parse_open_bundle(&magic).is_none());
    }

    #[test]
    fn files_over_the_cap_stay_out() {
        let store = InMemoryBlobStore::new();
        store.put("segments_1", vec![1; 10]).unwrap();
        store.put("_0.si", vec![3; 5000]).unwrap();
        let w = write_open_bundle(&store, &|_: &str| true, 1000).unwrap();
        assert_eq!(w.files, 1);
    }
}
