//! `MultiPassIndexSplitter`'s command line: split one or more indexes into
//! `numParts` new ones, `part-0` .. `part-<n-1>` under the output directory.
//!
//! ```text
//! lucene-tools MultiPassIndexSplitter -out <outputDir> -num <numParts> [-seq] <inputIndex1> [<inputIndex2> ...]
//! ```

use std::path::Path;

use lucene_search::misc::index::MultiPassIndexSplitter;
use lucene_store::directory::{Directory, FsDirectory};

use crate::cli::{arg, exception, parse_int, Io, ToolResult};

/// The usage, printed to `System.err` before `System.exit(-1)`.
pub const USAGE: &str = "Usage: MultiPassIndexSplitter -out <outputDir> -num <numParts> [-seq] <inputIndex1> [<inputIndex2 ...]\n\
\tinputIndex\tpath to input index, multiple values are ok\n\
\t-out ouputDir\tpath to output directory to contain partial indexes\n\
\t-num numParts\tnumber of parts to produce\n\
\t-seq\tsequential docid-range split (default is round-robin)\n";

/// `DirectoryReader.indexExists(dir)`: a `segments_N` file is there.
fn index_exists(dir: &dyn Directory) -> bool {
    dir.list_all()
        .is_ok_and(|files| files.iter().any(|f| f.starts_with("segments_")))
}

/// `MultiPassIndexSplitter.main(args)`. Fewer than five arguments print the
/// usage and exit -1. An input that is not a directory, or holds no index,
/// is reported to `System.err` and skipped. Progress goes to `System.err`:
/// `Writing part i ...` for each part, then `Done.`.
///
/// # Errors
/// Java's `Exception`s for a missing `-out`, a `-num` under 2 and no input
/// left, `NumberFormatException` for a `-num` that is not a number, and what
/// reading and writing the indexes report.
pub fn main(args: &[String], io: &mut Io<'_>) -> ToolResult {
    if args.len() < 5 {
        write!(io.err, "{USAGE}")?;
        return Ok(-1);
    }
    let mut inputs = Vec::new();
    let mut out_dir = None;
    let mut num_parts = -1;
    let mut seq = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-out" => {
                i = i.saturating_add(1);
                out_dir = Some(arg(args, i)?);
            }
            "-num" => {
                i = i.saturating_add(1);
                num_parts = parse_int(arg(args, i)?)
                    .map_err(|m| exception(format!("java.lang.NumberFormatException: {m}")))?;
            }
            "-seq" => seq = true,
            path => {
                if !Path::new(path).is_dir() {
                    writeln!(io.err, "Invalid input path - skipping: {path}")?;
                } else {
                    let dir = FsDirectory::open(path);
                    if index_exists(&dir) {
                        inputs.push(dir);
                    } else {
                        writeln!(io.err, "Invalid input index - skipping: {path}")?;
                    }
                }
            }
        }
        i = i.saturating_add(1);
    }
    let out_dir = out_dir.ok_or_else(|| {
        exception("java.lang.Exception: Required argument missing: -out outputDir")
    })?;
    if num_parts < 2 {
        return Err(exception(
            "java.lang.Exception: Invalid value of required argument: -num numParts",
        ));
    }
    if inputs.is_empty() {
        return Err(exception(
            "java.lang.Exception: No input indexes to process",
        ));
    }
    let out = Path::new(out_dir);
    let parts: Vec<FsDirectory> = (0..num_parts)
        .map(|i| {
            let part = out.join(format!("part-{i}"));
            std::fs::create_dir_all(&part).map(|()| FsDirectory::open(part))
        })
        .collect::<Result<_, _>>()?;
    let inputs: Vec<&dyn Directory> = inputs.iter().map(|d| d as &dyn Directory).collect();
    let parts: Vec<&dyn Directory> = parts.iter().map(|d| d as &dyn Directory).collect();
    MultiPassIndexSplitter.split(&inputs, &parts, seq)?;
    // Java prints each line as it starts the part; the library writes every
    // part in one call, so the lines follow it. Identical unless a part fails
    // to write.
    for i in 1..=num_parts {
        writeln!(io.err, "Writing part {i} ...")?;
    }
    writeln!(io.err, "Done.")?;
    Ok(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(args: &[&str]) -> (ToolResult, String, String) {
        let args: Vec<String> = args.iter().map(|s| s.to_string()).collect();
        let (mut out, mut err) = (Vec::new(), Vec::new());
        let r = main(
            &args,
            &mut Io {
                out: &mut out,
                err: &mut err,
            },
        );
        (
            r,
            String::from_utf8(out).unwrap(),
            String::from_utf8(err).unwrap(),
        )
    }

    #[test]
    fn too_few_arguments_print_the_usage() {
        let (r, out, err) = run(&["-out", "o", "-num", "2"]);
        assert_eq!(r.unwrap(), -1);
        assert!(out.is_empty());
        assert_eq!(err, USAGE);
    }

    #[test]
    fn the_argument_checks_are_javas() {
        let tmp = lucene_util::test_support::TempDir::new("tools-mpis");
        let empty = tmp.path().join("empty");
        std::fs::create_dir_all(&empty).unwrap();
        let empty = empty.to_str().unwrap();
        let nowhere = tmp.path().join("nowhere");
        let nowhere = nowhere.to_str().unwrap();

        let (r, _, err) = run(&["-num", "2", "-seq", empty, nowhere]);
        assert_eq!(
            r.unwrap_err().to_string(),
            "java.lang.Exception: Required argument missing: -out outputDir"
        );
        assert_eq!(
            err,
            format!(
                "Invalid input index - skipping: {empty}\nInvalid input path - skipping: {nowhere}\n"
            )
        );
        let (r, _, _) = run(&["-out", "o", "-num", "1", empty]);
        assert_eq!(
            r.unwrap_err().to_string(),
            "java.lang.Exception: Invalid value of required argument: -num numParts"
        );
        let (r, _, _) = run(&["-out", "o", "-num", "2", empty]);
        assert_eq!(
            r.unwrap_err().to_string(),
            "java.lang.Exception: No input indexes to process"
        );
        let (r, _, _) = run(&["-out", "o", empty, empty, "-num", "two"]);
        assert_eq!(
            r.unwrap_err().to_string(),
            "java.lang.NumberFormatException: For input string: \"two\""
        );
        let (r, _, _) = run(&[empty, empty, empty, empty, "-out"]);
        assert!(r
            .unwrap_err()
            .to_string()
            .starts_with("java.lang.ArrayIndexOutOfBoundsException"));
    }
}
