//! M14.i, differentially against Lucene 10.5.0: `fixtures/src/GenToolsMisc.java`
//! ran each line of `tools/misc/cases.tsv` through the `misc` module's
//! command-line tools and recorded what they printed. The same lines run
//! through [`lucene_tools::run`] must print the same bytes to stdout and
//! stderr and end with the same exit code.
//!
//! The lines chain: a tool writes an index into the shared scratch directory
//! (`IndexSplitter`'s copy and removal, `IndexMergeTool`, `MultiPassIndexSplitter`)
//! and the next lines read it back with `HighFreqTerms`, `GetTermInfo` and
//! `IndexSplitter -l` -- so what the writing tools produced is checked by
//! what Java's own tools print over Java's own result.

use std::path::{Path, PathBuf};

use lucene_util::test_support::TempDir;

fn data() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/data/tools/misc")
}

fn read(path: &Path) -> String {
    std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

fn copy_tree(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).unwrap();
    for f in std::fs::read_dir(from).unwrap() {
        let f = f.unwrap();
        std::fs::copy(f.path(), to.join(f.file_name())).unwrap();
    }
}

/// `IndexSplitter -l` over a segment a force-merge wrote: the names and their
/// order compare, the sizes do not. Java's merged `.si` carries the JVM's
/// environment in its diagnostics (`os.version`, `java.runtime.version`, a
/// timestamp, ...), so its size is not even stable across machines, and the
/// Rust writer records only `source` and `lucene.version`.
const MERGED_LISTINGS: &[&str] = &["merge-again-list", "merge-logdoc-list"];

/// The first column of each `name size` line.
fn segment_names(listing: &str) -> String {
    listing
        .lines()
        .map(|l| l.split(' ').next().unwrap_or_default())
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn misc_tools_print_what_lucene_prints() {
    let data = data();
    let tmp = TempDir::new("tools-misc");
    // The tools read the fixture indexes from a copy: `IndexMergeTool` locks
    // its sources (Java's `addIndexes` does too), which leaves a `write.lock`.
    let copy = tmp.path().join("fixture");
    for index in ["index", "index2"] {
        copy_tree(&data.join(index), &copy.join(index));
    }
    let fixture = copy.to_str().unwrap().replace('\\', "/");
    let scratch = tmp.path().join("tmp");
    let scratch = scratch.to_str().unwrap().replace('\\', "/");
    let cases = read(&data.join("cases.tsv"));
    let mut ran = 0;
    for line in cases.lines() {
        let fields: Vec<&str> = line.split('\t').collect();
        let (id, args) = (fields[0], &fields[1..]);
        let args: Vec<String> = args
            .iter()
            .map(|a| a.replace("{fixture}", &fixture).replace("{tmp}", &scratch))
            .collect();
        let out_dir = data.join("out");
        let mut want_out = read(&out_dir.join(format!("{id}.out")));
        let mut want_err = read(&out_dir.join(format!("{id}.err")));
        let want_code = read(&out_dir.join(format!("{id}.code")));
        let want_code = want_code.trim_end();

        if args[0] == "copy" {
            copy_tree(Path::new(&args[1]), Path::new(&args[2]));
            continue;
        }
        let (mut out, mut err) = (Vec::new(), Vec::new());
        let code = lucene_tools::run(&args, &mut out, &mut err);
        let code_want: i32 = match want_code.parse() {
            Ok(c) => c,
            Err(_) => {
                // An exception out of `main`: the JVM's line for it, without
                // the stack trace.
                want_err.push_str(&format!("Exception in thread \"main\" {want_code}\n"));
                1
            }
        };
        let mut out = String::from_utf8(out).unwrap();
        let err = String::from_utf8(err).unwrap();
        if MERGED_LISTINGS.contains(&id) {
            out = segment_names(&out);
            want_out = segment_names(&want_out);
        }
        assert_eq!(out, want_out, "{id}: stdout");
        assert_eq!(err, want_err, "{id}: stderr");
        assert_eq!(code, code_want, "{id}: exit code");
        ran += 1;
    }
    assert!(ran >= 30, "only {ran} cases ran");
}
