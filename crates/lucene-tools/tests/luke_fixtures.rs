//! M14.i, differentially against Lucene 10.5.0: `fixtures/src/GenToolsLuke.java`
//! ran each line of `tools/luke/cases.tsv` through Luke's own models
//! (`OverviewImpl`, `DocumentsImpl`, `CommitsImpl`, `SearchImpl`,
//! `AnalysisImpl`, `IndexToolsImpl`) and recorded what they returned. The
//! same lines through `lucene-tools Luke` must print the same bytes.
//!
//! The index (`index/`) is Java's: three commits kept by `NoDeletionPolicy`,
//! a deletion in the last, and a field of each kind Luke reports on. The
//! lines that change an index (`optimize`, `delete`, `add`, `create`) work
//! on copies in one scratch directory, in order, and the lines after them
//! read what the Rust writer wrote -- so those compare the Rust writer's
//! merges, deletes and commits with Java's, through what Luke reports.
//! All but one number: `SegmentInfos.version`, which Java's writer bumps on
//! each `SegmentInfos.changed()` -- calls the Rust writer does not make one
//! for one, so it counts fewer (15 against 17 after the force merge). It is
//! masked on the indexes the Rust writer wrote. And after `delete`, one
//! generation: the deletes leave enough of the index deleted for
//! `TieredMergePolicy` to merge, which Java's writer does after publishing
//! the deletes' commit and commits again on `close` (`segments_5`), while
//! the Rust writer's one commit already carries the merge (`segments_4`).
//! The segments, documents and user data are compared as they are.

use std::path::{Path, PathBuf};

use lucene_util::test_support::TempDir;

fn data() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/data/tools/luke")
}

fn read(path: &Path) -> String {
    std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// The cases reading an index the Rust writer changed.
const WRITTEN_BY_RUST: [&str; 5] = ["optimize-", "expunge-", "delete-", "add-", "create-"];

/// `version: N` as `version: {v}`, and for the `delete` cases the commit's
/// generation: `commit: segments_N (generation=N, segs=S)` as
/// `commit: {c} segs=S)`.
fn mask_version(id: &str, text: &str) -> String {
    text.lines()
        .map(|l| {
            if l.starts_with("version: ") {
                "version: {v}".to_string()
            } else if id.starts_with("delete-") && l.starts_with("commit: ") {
                match l.split_once(" segs=") {
                    Some((_, segs)) => format!("commit: {{c}} segs={segs}"),
                    None => l.to_string(),
                }
            } else {
                l.to_string()
            }
        })
        .map(|l| format!("{l}\n"))
        .collect()
}

#[test]
fn every_case_prints_what_lukes_models_returned() {
    let data = data();
    let tmp = TempDir::new("tools-luke");
    let fixture = data.to_string_lossy().into_owned();
    let scratch = tmp.path().to_string_lossy().into_owned();
    let cases = read(&data.join("cases.tsv"));
    let mut failures = Vec::new();
    let mut count = 0usize;
    for line in cases.lines().filter(|l| !l.is_empty()) {
        let mut cols = line.split('\t');
        let id = cols.next().unwrap();
        let mut args = vec!["Luke".to_string()];
        args.extend(cols.map(|c| c.replace("{fixture}", &fixture).replace("{tmp}", &scratch)));
        let (mut out, mut err) = (Vec::new(), Vec::new());
        let code = lucene_tools::run(&args, &mut out, &mut err);
        let out = String::from_utf8(out).unwrap();
        let err = String::from_utf8(err).unwrap();
        let mut printed = out
            .replace(&fixture, "{fixture}")
            .replace(&scratch, "{tmp}")
            .replace('\\', "/");
        let mut expected = read(&data.join("out").join(format!("{id}.out")));
        if WRITTEN_BY_RUST.iter().any(|p| id.starts_with(p)) {
            printed = mask_version(id, &printed);
            expected = mask_version(id, &expected);
        }
        let failed = expected.contains("\nerror: ") || expected.starts_with("error: ");
        if printed != expected || !err.is_empty() || (code != 0) != failed {
            failures.push(format!(
                "{id} (exit {code}):\n--- expected\n{expected}--- printed\n{printed}--- stderr\n{err}"
            ));
        }
        count += 1;
    }
    assert_eq!(count, 73, "every case ran");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
