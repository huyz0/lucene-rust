//! M14.i, differentially against Lucene 10.5.0: `fixtures/src/GenToolsBenchmark.java`
//! ran each line of `tools/benchmark/cases.tsv` through the `benchmark`
//! module's `QueryDriver` and `QualityQueriesFinder` over `index/`, a
//! Java-written two-segment index, and recorded what they printed and the
//! TREC submission `QueryDriver` wrote. The same lines through
//! [`lucene_tools::run`] must print the same bytes, write the same
//! submission -- the same hits, in the same order, with the same scores to
//! four places -- and end with the same exit code.
//!
//! The times the tools measure (`Search Seconds`, `DocName Seconds`) are
//! written `{s}` on both sides.

use std::path::{Path, PathBuf};

use lucene_util::test_support::TempDir;

fn data() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/data/tools/benchmark")
}

fn read(path: &Path) -> String {
    std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// What a tool printed, with the directories written as names and each
/// measured time as `{s}`.
fn clean(s: &str, fixture: &str, tmp: &str) -> String {
    let s = s
        .replace('\\', "/")
        .replace(fixture, "{fixture}")
        .replace(tmp, "{tmp}");
    let mut out = String::with_capacity(s.len());
    for line in s.split_inclusive('\n') {
        let timed = ["Search Seconds: ", "DocName Seconds: "]
            .iter()
            .find_map(|label| line.find(label).map(|at| at + label.len()));
        match timed {
            Some(end) => {
                out.push_str(&line[..end]);
                out.push_str("{s}");
                if line.ends_with('\n') {
                    out.push('\n');
                }
            }
            None => out.push_str(line),
        }
    }
    out
}

#[test]
fn quality_tools_print_what_lucene_prints() {
    let data = data();
    let fixture = data.to_str().unwrap().replace('\\', "/");
    let tmp = TempDir::new("tools-benchmark");
    let scratch = tmp.path().to_str().unwrap().replace('\\', "/");
    let mut ran = 0;
    let mut failed = Vec::new();
    for line in read(&data.join("cases.tsv")).lines() {
        let fields: Vec<&str> = line.split('\t').collect();
        let (id, tool) = (fields[0], fields[1]);
        let args: Vec<String> = fields[2..]
            .iter()
            .map(|a| a.replace("{fixture}", &fixture).replace("{tmp}", &scratch))
            .collect();
        let out_dir = data.join("out");
        let want_out = read(&out_dir.join(format!("{id}.out")));
        let mut want_err = read(&out_dir.join(format!("{id}.err")));
        let want_code = read(&out_dir.join(format!("{id}.code")));
        let want_code = want_code.strip_suffix('\n').unwrap_or(&want_code);
        let code_want: i32 = match want_code.parse() {
            Ok(c) => c,
            Err(_) => {
                want_err.push_str(&format!("Exception in thread \"main\" {want_code}\n"));
                1
            }
        };

        let mut all = vec![tool.to_string()];
        all.extend_from_slice(&args);
        let (mut out, mut err) = (Vec::new(), Vec::new());
        let code = lucene_tools::run(&all, &mut out, &mut err);
        let out = clean(&String::from_utf8(out).unwrap(), &fixture, &scratch);
        let err = clean(&String::from_utf8(err).unwrap(), &fixture, &scratch);
        let mut problems = Vec::new();
        if out != want_out {
            problems.push(format!("out\n{out}\nwant\n{want_out}"));
        }
        if err != want_err || code != code_want {
            problems.push(format!(
                "err {err:?} code {code}\nwant {want_err:?} code {code_want}"
            ));
        }
        let want_sub = out_dir.join(format!("{id}.sub"));
        if want_sub.exists() {
            let got = std::fs::read_to_string(&args[2]).unwrap_or_default();
            let want = read(&want_sub);
            if got != want {
                problems.push(format!("submission\n{got}\nwant\n{want}"));
            }
        }
        if !problems.is_empty() {
            failed.push(format!("{id}: {}", problems.join("\n")));
        }
        ran += 1;
    }
    assert!(failed.is_empty(), "{}", failed.join("\n\n"));
    assert_eq!(ran, 11, "every case ran");
}
