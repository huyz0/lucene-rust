//! A Java-written index with a few bytes of one file flipped, set or cut off, run through the
//! `CheckIndex` equivalent: every decoder a search, a merge or a recovery reads (stored fields,
//! term vectors, doc values, norms, postings, points, vectors, HNSW graphs, compound files) sees
//! the damaged bytes. Corruption must come back as failed checks or an `Err`, never a panic, an
//! out-of-bounds read, a hang or a length-sized allocation. The input is one byte choosing the
//! index, then 6-byte edits: file, offset (u32), and an operation with its operand.
#![no_main]

use libfuzzer_sys::fuzz_target;
use lucene_index::check_index;
use lucene_store::FsDirectory;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

const FIXTURES: &[&str] = &[
    "multi_segment_scoring_index",
    "mixed_boolean_scoring_index",
    "stored_fields_index",
    "stored_fields_best_compression_index",
    "term_vectors_index",
    "doc_values_index",
    "points_index",
    "norms_index",
    "vectors_index",
    "vectors_multi_index",
    "compound_index",
    "blocktree_multilevel_index",
    "extended_knn_index",
];

const ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../../fixtures/data");

/// Every index file of a fixture, name and bytes, read once.
fn fixture(i: usize) -> &'static [(String, Vec<u8>)] {
    static ALL: OnceLock<Vec<Vec<(String, Vec<u8>)>>> = OnceLock::new();
    &ALL.get_or_init(|| {
        FIXTURES
            .iter()
            .map(|f| {
                let mut files: Vec<(String, Vec<u8>)> = std::fs::read_dir(Path::new(ROOT).join(f))
                    .unwrap_or_else(|e| panic!("fixture {f}: {e}"))
                    .filter_map(Result::ok)
                    .filter(|e| e.path().is_file())
                    .map(|e| (e.file_name().to_string_lossy().into_owned(), std::fs::read(e.path()).unwrap()))
                    .filter(|(n, _)| !n.ends_with(".raw") && n != "manifest.properties")
                    .collect();
                files.sort();
                files
            })
            .collect()
    })[i]
}

fn scratch() -> PathBuf {
    let dir = PathBuf::from("/dev/shm").join(format!("lucene-fuzz-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fuzz_target!(|data: &[u8]| {
    let Some((&which, edits)) = data.split_first() else { return };
    let mut files = fixture(usize::from(which) % FIXTURES.len()).to_vec();
    for e in edits.chunks_exact(6).take(8) {
        let (file, off) = (usize::from(e[0]) % files.len(), u32::from_le_bytes([e[1], e[2], e[3], e[4]]) as usize);
        let bytes = &mut files[file].1;
        if bytes.is_empty() {
            continue;
        }
        let off = off % bytes.len();
        match e[5] & 3 {
            0 => bytes[off] ^= e[5] | 1,
            1 => bytes[off] = e[5],
            2 => bytes.truncate(off),
            _ => bytes[off..].iter_mut().take(16).for_each(|b| *b = 0),
        }
    }
    let dir = scratch();
    for (name, bytes) in &files {
        std::fs::write(dir.join(name), bytes).unwrap();
    }
    let fs = FsDirectory::open(&dir);
    let _ = check_index::check_directory(&fs);
    let _ = std::fs::remove_dir_all(&dir);
});
