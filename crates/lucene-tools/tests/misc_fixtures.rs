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

/// `IndexSplitter -l` over an index a force-merge wrote. A merged `.si`
/// records `IndexWriter.setDiagnostics`' map, six of whose values are the
/// writing process's environment: Java's are its JVM's and host's, the Rust
/// writer's its own (`segment_info::writer_diagnostics`). Every other byte of
/// the segment must agree, so each listed size is compared with Java's plus
/// the difference those six values make to the `.si`.
const MERGED_LISTINGS: &[&str] = &["merge-again-list", "merge-logdoc-list"];

/// The diagnostics a writer fills in from its environment.
const ENVIRONMENT_DIAGNOSTICS: [&str; 6] = [
    "os",
    "os.arch",
    "os.version",
    "java.runtime.version",
    "java.vendor",
    "timestamp",
];

/// The newest commit's diagnostics of segment `name` in the index at `path`.
fn diagnostics(path: &Path, name: &str) -> Vec<(String, String)> {
    let dir = lucene_store::FsDirectory::open(path);
    let commits = lucene_tools::luke::commits::Commits::new(&dir, path).unwrap();
    let generation = commits.list_commits()[0].generation;
    commits.segment_diagnostics(generation, name).unwrap()
}

/// `DataOutput.writeString`'s length: a vInt byte count, then the UTF-8.
fn written_len(s: &str) -> i64 {
    let n = s.len();
    let vint = (usize::BITS - (n | 1).leading_zeros()).div_ceil(7);
    i64::from(vint) + n as i64
}

/// What the environment values of `segment` add to its `.si` over those of
/// `java`, a segment Java's process wrote. Zero for a segment Java wrote.
fn environment_delta(segment: &[(String, String)], java: &[(String, String)]) -> i64 {
    let value = |d: &[(String, String)], k: &str| {
        d.iter()
            .find(|(key, _)| key == k)
            .map(|(_, v)| written_len(v))
            .unwrap_or_else(|| panic!("no {k} diagnostic in {d:?}"))
    };
    ENVIRONMENT_DIAGNOSTICS
        .iter()
        .map(|k| value(segment, k) - value(java, k))
        .sum()
}

/// Each `name size` line of an `IndexSplitter -l` listing, the size's
/// grouping commas dropped.
fn listed_sizes(listing: &str) -> Vec<(String, i64)> {
    listing
        .lines()
        .map(|l| {
            let (name, size) = l.split_once(' ').unwrap_or_else(|| panic!("{l:?}"));
            (name.to_string(), size.replace(',', "").parse().unwrap())
        })
        .collect()
}

#[test]
fn the_environment_delta_counts_vint_and_utf8_bytes() {
    assert_eq!(written_len(""), 1);
    assert_eq!(written_len("Linux"), 6);
    assert_eq!(written_len(&"x".repeat(127)), 128);
    assert_eq!(written_len(&"x".repeat(128)), 130);
    let d = |vendor: &str| {
        ENVIRONMENT_DIAGNOSTICS
            .iter()
            .map(|k| {
                let v = if *k == "java.vendor" { vendor } else { "v" };
                (k.to_string(), v.to_string())
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(
        environment_delta(&d("lucene-rust"), &d("Eclipse Adoptium")),
        -5
    );
    assert_eq!(environment_delta(&d("a"), &d("a")), 0);
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
    let mut merged_listings = 0;
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
            // Java's environment, from a segment the same process flushed.
            let java = diagnostics(&copy.join("index"), "_0");
            let listed = Path::new(&args[1]);
            let want: Vec<(String, i64)> = listed_sizes(&want_out)
                .into_iter()
                .map(|(name, size)| {
                    let delta = environment_delta(&diagnostics(listed, &name), &java);
                    (name, size + delta)
                })
                .collect();
            assert!(!want.is_empty(), "{id}: Java listed no segment");
            assert_eq!(listed_sizes(&out), want, "{id}: stdout");
            merged_listings += 1;
            out.clear();
            want_out.clear();
        }
        assert_eq!(out, want_out, "{id}: stdout");
        assert_eq!(err, want_err, "{id}: stderr");
        assert_eq!(code, code_want, "{id}: exit code");
        ran += 1;
    }
    assert!(ran >= 30, "only {ran} cases ran");
    assert_eq!(
        merged_listings,
        MERGED_LISTINGS.len(),
        "a merged listing is not in cases.tsv"
    );
}
