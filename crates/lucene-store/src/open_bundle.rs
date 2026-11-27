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

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use crate::blob_directory::BlobStore;
use crate::error::Result;
use crate::index_output::OutputSink;
use crate::lock::lock_ignoring_poison;

/// The sidecar's object name.
pub const OPEN_BUNDLE_NAME: &str = "accel-open-bundle.bin";

const MAGIC: &[u8; 8] = b"ACCLOPN1";

/// The largest file a bundle holds: bigger ones are read by range, as the
/// bundle's saving is in the many small files.
pub const MAX_BUNDLED_FILE: u64 = 1 << 20;

/// The most bytes a bundle holds in all.
pub const MAX_BUNDLE_BYTES: u64 = 8 << 20;

/// Whether `name` is a file a reader opens whole or nearly so: the commit
/// point and the segment files the page cache pins.
pub fn is_bundled_name(name: &str) -> bool {
    if name.starts_with("segments_") {
        return true;
    }
    let ext = name.rsplit_once('.').map_or("", |(_, e)| e);
    crate::page_cache::DEFAULT_PINNED_EXTENSIONS.contains(&ext)
}

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

fn encode(entries: &[(&str, &[u8])]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&(entries.len() as u32).to_le_bytes());
    for (name, bytes) in entries {
        let name_bytes = name.as_bytes();
        out.extend_from_slice(&(name_bytes.len() as u16).to_le_bytes());
        out.extend_from_slice(name_bytes);
        out.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
        out.extend_from_slice(bytes);
    }
    let crc = crc32fast::hash(&out);
    out.extend_from_slice(&crc.to_le_bytes());
    out
}

fn listing<S: BlobStore>(store: &S) -> Result<Vec<(String, u64)>> {
    match store.list_with_sizes()? {
        Some(l) => Ok(l),
        None => {
            let mut v = Vec::new();
            for n in store.list()? {
                let len = store.size(&n)?;
                v.push((n, len));
            }
            Ok(v)
        }
    }
}

/// The files a bundle for the store's current newest commit holds, read
/// from the store: its `segments_N` and every file `is_bundled` accepts (of
/// at most `max_file` bytes) that the listing shows.
fn collect<S: BlobStore>(
    store: &S,
    is_bundled: &dyn Fn(&str) -> bool,
    max_file: u64,
) -> Result<BTreeMap<String, Arc<[u8]>>> {
    let listing = listing(store)?;
    let newest = listing
        .iter()
        .filter_map(|(n, _)| generation(n).map(|g| (g, n.clone())))
        .max()
        .map(|(_, n)| n);
    let mut out = BTreeMap::new();
    for (name, len) in &listing {
        if name == OPEN_BUNDLE_NAME || *len > max_file {
            continue;
        }
        let take = if name.starts_with("segments_") {
            Some(name) == newest.as_ref()
        } else {
            is_bundled(name)
        };
        if take {
            out.insert(name.clone(), Arc::from(&store.get(name)?[..]));
        }
    }
    Ok(out)
}

/// Writes the bundle for the store's current newest commit: its
/// `segments_N` and every file `is_bundled` accepts (of at most `max_file`
/// bytes) that the listing shows.
pub fn write_open_bundle<S: BlobStore>(
    store: &S,
    is_bundled: &dyn Fn(&str) -> bool,
    max_file: u64,
) -> Result<BundleWritten> {
    let files = collect(store, is_bundled, max_file)?;
    let entries: Vec<(&str, &[u8])> = files.iter().map(|(n, b)| (n.as_str(), &b[..])).collect();
    let out = encode(&entries);
    let bytes = out.len() as u64;
    store.put(OPEN_BUNDLE_NAME, out)?;
    Ok(BundleWritten {
        files: entries.len(),
        bytes,
    })
}

/// What a directory that writes an index keeps to publish the bundle after
/// each commit without reading anything back: the bytes of the small files
/// it wrote, kept as they are written, and dropped as they are deleted.
pub(crate) struct BundleWriter {
    files: Mutex<BTreeMap<String, Arc<[u8]>>>,
    seeded: Mutex<bool>,
}

impl BundleWriter {
    pub(crate) fn new() -> Arc<Self> {
        Arc::new(Self {
            files: Mutex::new(BTreeMap::new()),
            seeded: Mutex::new(false),
        })
    }

    /// Whether the bytes of the output `name` are worth keeping: it is a
    /// bundled name, or the commit point while it is still pending.
    pub(crate) fn wants(name: &str) -> bool {
        is_bundled_name(name) || name.starts_with("pending_segments_")
    }

    /// Wraps an output so its bytes are kept (up to [`MAX_BUNDLED_FILE`]) once
    /// it closes.
    pub(crate) fn tee(
        self: &Arc<Self>,
        name: &str,
        inner: Box<dyn OutputSink>,
    ) -> Box<dyn OutputSink> {
        Box::new(Tee {
            inner,
            writer: Arc::clone(self),
            name: name.to_string(),
            buf: Some(Vec::new()),
        })
    }

    pub(crate) fn forget(&self, name: &str) {
        lock_ignoring_poison(&self.files).remove(name);
    }

    pub(crate) fn rename(&self, source: &str, dest: &str) {
        let mut files = lock_ignoring_poison(&self.files);
        if let Some(bytes) = files.remove(source) {
            files.insert(dest.to_string(), bytes);
        }
    }

    /// Writes the bundle for the commit just published as `segments_N`. The
    /// first time, the files the directory did not write itself (an index
    /// that existed before it) come from the bundle already in the store, or
    /// from the files themselves. It is best-effort: the sidecar is optional,
    /// so a failure never fails the commit.
    pub(crate) fn publish<S: BlobStore>(&self, store: &S, commit: &str) {
        let _ = self.try_publish(store, commit);
    }

    fn try_publish<S: BlobStore>(&self, store: &S, commit: &str) -> Result<()> {
        {
            let mut seeded = lock_ignoring_poison(&self.seeded);
            if !*seeded {
                let listed = listing(store)?;
                let live: std::collections::HashSet<&str> =
                    listed.iter().map(|(n, _)| n.as_str()).collect();
                let mut seed = match store.get(OPEN_BUNDLE_NAME) {
                    Ok(b) => parse_open_bundle(&b).map(|entries| {
                        entries
                            .into_iter()
                            .map(|(n, bytes)| (n, Arc::<[u8]>::from(bytes)))
                            .collect::<BTreeMap<_, _>>()
                    }),
                    Err(_) => None,
                };
                if seed.is_none() {
                    seed = Some(collect(store, &is_bundled_name, MAX_BUNDLED_FILE)?);
                }
                let mut files = lock_ignoring_poison(&self.files);
                for (name, bytes) in seed.unwrap_or_default() {
                    if live.contains(name.as_str()) {
                        files.entry(name).or_insert(bytes);
                    }
                }
                *seeded = true;
            }
        }
        let mut files = lock_ignoring_poison(&self.files);
        let newest = generation(commit);
        files.retain(|n, _| !n.starts_with("segments_") || generation(n) == newest);
        let mut entries: Vec<(&str, &[u8])> = Vec::new();
        let mut total = 0u64;
        for (name, bytes) in files.iter() {
            total = total.saturating_add(bytes.len() as u64);
            if total > MAX_BUNDLE_BYTES {
                break;
            }
            entries.push((name.as_str(), &bytes[..]));
        }
        store.put(OPEN_BUNDLE_NAME, encode(&entries))
    }
}

/// An output that keeps what it writes while it is small enough to bundle.
struct Tee {
    inner: Box<dyn OutputSink>,
    writer: Arc<BundleWriter>,
    name: String,
    buf: Option<Vec<u8>>,
}

impl OutputSink for Tee {
    fn write_all(&mut self, bytes: &[u8]) -> std::io::Result<()> {
        if let Some(buf) = self.buf.as_mut() {
            if (buf.len() as u64).saturating_add(bytes.len() as u64) > MAX_BUNDLED_FILE {
                self.buf = None;
            } else {
                buf.extend_from_slice(bytes);
            }
        }
        self.inner.write_all(bytes)
    }

    fn finish(self: Box<Self>) -> std::io::Result<()> {
        let Tee {
            inner,
            writer,
            name,
            buf,
        } = *self;
        inner.finish()?;
        if let Some(buf) = buf {
            lock_ignoring_poison(&writer.files).insert(name, Arc::from(buf));
        }
        Ok(())
    }
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
