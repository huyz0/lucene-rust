//! `IndexWriter::set_merge_mb_per_sec`: a merge's writes held to a rate
//! (`ConcurrentMergeScheduler`'s per-merge `MergeRateLimiter`).
//!
//! Rate limiting has no on-disk trace, so there is no Java fixture: what is
//! checked is that a throttled merge takes the time its rate implies, and
//! writes exactly the files an unthrottled one does.
#![allow(clippy::arithmetic_side_effects)]

use std::time::{Duration, Instant};

use lucene_codecs::field_infos::FieldInfo;
use lucene_codecs::stored_fields::{Document, FieldValue, StoredField};
use lucene_index::index_writer::{Error, IndexWriter};
use lucene_index::segment_info::LuceneVersion;
use lucene_store::directory::Directory;
use lucene_store::FsDirectory;
use lucene_util::test_support::TempDir;

/// ~1 KiB of hex from a fixed LCG: stored fields' LZ4 barely compresses it,
/// so the merged segment is about as large as the text.
fn doc(seed: &mut u64) -> Document {
    let mut s = String::with_capacity(1024);
    while s.len() < 1024 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        s.push_str(&format!("{:016x}", *seed));
    }
    Document {
        fields: vec![StoredField {
            field_number: 0,
            value: FieldValue::String(s),
        }],
    }
}

/// Two committed segments of 1500 documents each, then `forceMerge(1)` under
/// `rate`; returns how long the merge took, and every file of the result
/// but the `segments_N`/`.si`/`write.lock` with its length.
fn merge_under(rate: Option<f64>, name: &str) -> (Duration, Vec<(String, u64)>) {
    let tmp = TempDir::new(name);
    let dir = FsDirectory::open(tmp.path());
    let version = LuceneVersion {
        major: 10,
        minor: 5,
        bugfix: 0,
    };
    let mut w =
        IndexWriter::open(&dir, vec![FieldInfo::new("body", 0)], "Lucene104", version).unwrap();
    let mut seed = 42;
    for _ in 0..2 {
        for _ in 0..1500 {
            w.add_document(doc(&mut seed)).unwrap();
        }
        w.commit().unwrap();
    }
    w.set_merge_mb_per_sec(rate).unwrap();
    assert_eq!(w.merge_mb_per_sec(), rate);
    let start = Instant::now();
    w.force_merge(1).unwrap();
    let took = start.elapsed();
    drop(w);
    let files = dir
        .list_all()
        .unwrap()
        .into_iter()
        .filter(|f| !f.starts_with("segments") && !f.ends_with(".si") && f != "write.lock")
        .map(|f| {
            let len = dir.file_length(&f).unwrap();
            (f, len)
        })
        .collect();
    (took, files)
}

#[test]
fn a_throttled_merge_takes_its_rate_and_writes_the_same_files() {
    let (_, free) = merge_under(None, "merge-rate-free");
    let bytes: u64 = free.iter().map(|(_, len)| len).sum();
    assert!(
        bytes > 2_500_000,
        "{bytes} bytes: the merge must be big enough to time"
    );

    // 8 MB/s over ~3 MB: ~370 ms of pauses. Allow for the first write, which
    // never pauses, and the last, which pauses only when due.
    let rate = 8.0;
    let (took, throttled) = merge_under(Some(rate), "merge-rate-throttled");
    let expected = Duration::from_secs_f64(bytes as f64 / (rate * 1024.0 * 1024.0));
    assert!(
        took >= expected.mul_f64(0.6),
        "{took:?} for {bytes} bytes at {rate} MB/s; expected about {expected:?}"
    );
    assert_eq!(
        throttled, free,
        "throttling changes when bytes are written, not which"
    );
}

#[test]
fn a_negative_or_nan_merge_rate_is_refused() {
    let tmp = TempDir::new("merge-rate-bad");
    let dir = FsDirectory::open(tmp.path());
    let version = LuceneVersion {
        major: 10,
        minor: 5,
        bugfix: 0,
    };
    let mut w =
        IndexWriter::open(&dir, vec![FieldInfo::new("body", 0)], "Lucene104", version).unwrap();
    for bad in [-1.0, f64::NAN] {
        assert!(matches!(
            w.set_merge_mb_per_sec(Some(bad)),
            Err(Error::InvalidMergeRate(_))
        ));
    }
    assert_eq!(w.merge_mb_per_sec(), None);
    w.set_merge_mb_per_sec(Some(0.5)).unwrap();
    w.set_merge_mb_per_sec(None).unwrap();
    assert_eq!(w.merge_mb_per_sec(), None);
}
