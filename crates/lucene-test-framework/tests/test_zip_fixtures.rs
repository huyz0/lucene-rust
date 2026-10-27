//! Differential test against `fixtures/src/testframework/GenTestFrameworkZip.java`:
//! [`unzip`] extracts the same files from a `ZipOutputStream`-written zip
//! (stored, deflated with data descriptors, a directory entry) as Lucene
//! 10.5.0's `TestUtil.unzip`, and `VocabularyAssert`'s zip overloads pass
//! over its vocabularies.
//!
//! Regenerate with `scripts/gen-test-framework-fixtures.sh --only TestFrameworkZip`.

use std::fs;
use std::path::{Path, PathBuf};

use lucene_test_framework::analysis::mock_analyzer::MockAnalyzer;
use lucene_test_framework::analysis::vocabulary_assert::{
    assert_vocabulary_tsv_zip, assert_vocabulary_zip,
};
use lucene_test_framework::util::test_util::unzip;
use lucene_util::java_random::JavaRandom;

fn data(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/testframework/data")
        .join(name)
}

/// `unzip.txt`'s listing of `root`: sorted, `dir/` or `path\tlen\tcrc`.
fn listing(root: &Path, dir: &Path, out: &mut Vec<String>) {
    for entry in fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        let rel = path
            .strip_prefix(root)
            .unwrap()
            .to_string_lossy()
            .replace('\\', "/");
        if path.is_dir() {
            out.push(format!("{rel}/"));
            listing(root, &path, out);
        } else {
            let bytes = fs::read(&path).unwrap();
            out.push(format!(
                "{rel}\t{}\t{:x}",
                bytes.len(),
                crc32fast::hash(&bytes)
            ));
        }
    }
}

#[test]
fn unzip_extracts_what_lucene_does() {
    let zip = fs::read(data("vocabulary.zip")).unwrap();
    let expected = fs::read_to_string(data("unzip.txt")).unwrap();
    let tmp = std::env::temp_dir().join(format!("lucene-unzip-{}", std::process::id()));
    let _ = fs::remove_dir_all(&tmp);
    fs::create_dir_all(&tmp).unwrap();
    unzip(&zip, &tmp).unwrap();
    let mut got = Vec::new();
    listing(&tmp, &tmp, &mut got);
    got.sort();
    fs::remove_dir_all(&tmp).unwrap();
    assert_eq!(got.join("\n") + "\n", expected);
}

#[test]
fn vocabulary_zip_overloads() {
    let zip = data("vocabulary.zip");
    let mut r = JavaRandom::new(7);
    let a = MockAnalyzer::new(&mut r).build();
    assert_vocabulary_zip(&mut r, &a, &zip, "voc.txt", "out.txt").unwrap();
    assert_vocabulary_tsv_zip(&mut r, &a, &zip, "dir/voc_out.txt").unwrap();

    let missing = assert_vocabulary_tsv_zip(&mut r, &a, &zip, "nope.txt").unwrap_err();
    assert!(
        missing.to_string().contains("NoSuchFileException"),
        "{missing}"
    );
    let dir = assert_vocabulary_tsv_zip(&mut r, &a, &zip, "dir/").unwrap_err();
    assert!(dir.to_string().contains("NoSuchFileException"), "{dir}");
    let no_zip = assert_vocabulary_zip(&mut r, &a, &data("no.zip"), "a", "b").unwrap_err();
    assert!(no_zip.to_string().contains("no.zip"), "{no_zip}");
    let wrong = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        assert_vocabulary_zip(&mut r, &a, &zip, "out.txt", "voc.txt")
    }));
    assert!(wrong.is_err(), "Dogs is not what dogs analyzes to");
}
