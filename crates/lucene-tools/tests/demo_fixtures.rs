//! M14.i, differentially against Lucene 10.5.0: `fixtures/src/GenToolsDemo.java`
//! ran each line of `tools/demo/cases.tsv` through the `demo` module's
//! `IndexFiles` and `SearchFiles`, with the line's text on `System.in`, and
//! recorded what they printed. The same lines through
//! [`lucene_tools::run`] (and `SearchFiles` through
//! [`search_files`](lucene_tools::demo::search_files::search_files), which
//! takes its `System.in`) must print the same bytes and end with the same
//! exit code -- the same top hits, in the same order, with the same scores
//! where `-raw` prints them.
//!
//! The lines chain through one scratch directory: an index built a file at a
//! time (`-create`, then `-update`) is searched, and a directory indexed with
//! a word-vector dictionary is searched with `-knn_vector`. `SearchFiles` also
//! reads `index/`, the index Java's `IndexFiles.indexDoc` wrote.
//!
//! The dictionary itself is compared byte for byte (`dict/`), and the
//! embeddings computed over it bit for bit (`embeddings.tsv`).

use std::io::Cursor;
use std::path::{Path, PathBuf};

use lucene_store::directory::FsDirectory;
use lucene_tools::cli::Io;
use lucene_tools::demo::knn::{DemoEmbeddings, KnnVectorDict};
use lucene_tools::demo::search_files::search_files;
use lucene_util::test_support::TempDir;

fn data() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/data/tools/demo")
}

fn read(path: &Path) -> String {
    std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// The digits after each `prefix` (up to `suffix`) written `{ms}`, as the
/// generator writes the times the tools measure.
fn mask_times(s: &str, prefix: &str, suffix: &str) -> String {
    let mut out = String::new();
    let mut rest = s;
    while let Some(at) = rest.find(prefix) {
        let (head, tail) = rest.split_at(at + prefix.len());
        out.push_str(head);
        let digits = tail.bytes().take_while(u8::is_ascii_digit).count();
        if digits > 0 && tail[digits..].starts_with(suffix) {
            out.push_str("{ms}");
            rest = &tail[digits..];
        } else {
            rest = tail;
        }
    }
    out.push_str(rest);
    out
}

/// What a tool printed, with the directories written as names.
fn clean(s: &str, fixture: &str, tmp: &str) -> String {
    let s = s
        .replace('\\', "/")
        .replace(fixture, "{fixture}")
        .replace(tmp, "{tmp}");
    let s = mask_times(&s, "documents in ", " ms");
    mask_times(&s, "Time: ", "ms")
}

/// A walk lists the directory in the file system's order: its lines compare
/// as a multiset.
fn sorted_lines(s: &str) -> Vec<&str> {
    let mut lines: Vec<&str> = s.lines().collect();
    lines.sort_unstable();
    lines
}

fn run_case(tool: &str, args: &[String], stdin: &str) -> (i32, String, String) {
    let (mut out, mut err) = (Vec::new(), Vec::new());
    let code = if tool == "SearchFiles" {
        let mut io = Io {
            out: &mut out,
            err: &mut err,
        };
        match search_files(args, &mut io, &mut Cursor::new(stdin.as_bytes())) {
            Ok(code) => code,
            Err(e) => {
                writeln_err(&mut err, &e.to_string());
                1
            }
        }
    } else {
        let mut all = vec![tool.to_string()];
        all.extend_from_slice(args);
        lucene_tools::run(&all, &mut out, &mut err)
    };
    (
        code,
        String::from_utf8(out).unwrap(),
        String::from_utf8(err).unwrap(),
    )
}

/// What [`lucene_tools::run`] prints for an exception out of a tool.
fn writeln_err(err: &mut Vec<u8>, e: &str) {
    err.extend_from_slice(format!("Exception in thread \"main\" {e}\n").as_bytes());
}

#[test]
fn demo_tools_print_what_lucene_prints() {
    let data = data();
    let fixture = data.to_str().unwrap().replace('\\', "/");
    let tmp = TempDir::new("tools-demo");
    let scratch = tmp.path().to_str().unwrap().replace('\\', "/");
    let mut ran = 0;
    let mut failed = Vec::new();
    for line in read(&data.join("cases.tsv")).lines() {
        let fields: Vec<&str> = line.split('\t').collect();
        let (id, stdin, tool) = (fields[0], fields[1], fields[2]);
        let stdin = if stdin == "-" {
            String::new()
        } else {
            stdin.replace("\\n", "\n")
        };
        let args: Vec<String> = fields[3..]
            .iter()
            .map(|a| a.replace("{fixture}", &fixture).replace("{tmp}", &scratch))
            .collect();
        let out_dir = data.join("out");
        let want_out = read(&out_dir.join(format!("{id}.out")));
        let mut want_err = read(&out_dir.join(format!("{id}.err")));
        let want_code = read(&out_dir.join(format!("{id}.code")));
        // the generator ends the line; a message may itself end in spaces
        let want_code = want_code.strip_suffix('\n').unwrap_or(&want_code);
        let code_want: i32 = match want_code.parse() {
            Ok(c) => c,
            Err(_) => {
                want_err.push_str(&format!("Exception in thread \"main\" {want_code}\n"));
                1
            }
        };

        let (code, out, err) = run_case(tool, &args, &stdin);
        let out = clean(&out, &fixture, &scratch);
        let err = clean(&err, &fixture, &scratch);
        let same_out = if id.starts_with("walk-") && tool == "IndexFiles" {
            sorted_lines(&out) == sorted_lines(&want_out)
        } else {
            out == want_out
        };
        if !same_out || err != want_err || code != code_want {
            failed.push(format!(
                "{id}:
  out  {out:?}
  want {want_out:?}
  err  {err:?}
  want {want_err:?}
  code {code} want {code_want}"
            ));
        }
        ran += 1;
    }
    assert!(
        failed.is_empty(),
        "{}",
        failed.join(
            "
"
        )
    );
    assert!(ran >= 35, "only {ran} cases ran");
}

#[test]
fn the_dictionary_is_the_bytes_lucene_builds() {
    let data = data();
    let tmp = TempDir::new("tools-demo-dict");
    let dir = FsDirectory::open(tmp.path());
    KnnVectorDict::build(&data.join("vectors.txt"), &dir, "knn-dict").unwrap();
    for ext in ["fst", "bin"] {
        let name = format!("knn-dict.{ext}");
        let got = std::fs::read(tmp.path().join(&name)).unwrap();
        let want = std::fs::read(data.join("dict").join(&name)).unwrap();
        assert_eq!(got, want, "{name}");
    }
}

#[test]
fn the_embeddings_are_the_bits_lucene_computes() {
    let data = data();
    let dir = FsDirectory::open(data.join("dict"));
    let dict = KnnVectorDict::open(&dir, "knn-dict").unwrap();
    let embeddings = DemoEmbeddings::new(&dict);
    let mut checked = 0;
    for line in read(&data.join("embeddings.tsv")).lines() {
        let (text, bits) = line.split_once('\t').unwrap();
        let got: Vec<String> = embeddings
            .compute_embedding(text)
            .unwrap()
            .iter()
            .map(|f| format!("{:x}", f.to_bits()))
            .collect();
        assert_eq!(got.join(","), bits, "{text:?}");
        checked += 1;
    }
    assert!(checked >= 15, "only {checked} embeddings");
}

#[test]
fn the_time_masks_only_mask_a_time() {
    assert_eq!(
        mask_times(
            "documents in 12 ms, documents in x ms",
            "documents in ",
            " ms"
        ),
        "documents in {ms} ms, documents in x ms"
    );
    assert_eq!(mask_times("Time: 7s", "Time: ", "ms"), "Time: 7s");
}
