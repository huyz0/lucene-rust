#![forbid(unsafe_code)]
//! lucene-tools: the command-line tools of Lucene's `misc`, `demo`, `luke`
//! and `benchmark` modules, each under its Java class name, behind one
//! `lucene-tools` binary:
//!
//! ```text
//! lucene-tools GetTermInfo <index dir> field term
//! lucene-tools IndexMergeTool [OPTION...] <mergedIndex> <index1> <index2> ...
//! ```
//!
//! Every tool is a function of its arguments writing to an injected
//! `System.out`/`System.err` ([`cli::Io`]), so the binary is a thin shell over
//! [`run`] and the tools are tested -- against the Java tools' own output on
//! Java-written indexes -- without spawning a process.
//!
//! The top of the crate order: nothing depends on this crate.

use std::io::Write;

pub mod cli;
pub mod demo;
pub mod luke;
pub mod misc;

use cli::{Io, Tool};

/// Every tool, by the Java class name it is invoked as.
pub const TOOLS: &[(&str, Tool)] = &[
    ("GetTermInfo", misc::get_term_info::main),
    ("HighFreqTerms", misc::high_freq_terms::main),
    ("IndexFiles", demo::index_files::main),
    ("IndexMergeTool", misc::index_merge_tool::main),
    ("IndexSplitter", misc::index_splitter::main),
    ("Luke", luke::cli::main),
    (
        "MultiPassIndexSplitter",
        misc::multi_pass_index_splitter::main,
    ),
    ("SearchFiles", demo::search_files::main),
];

/// The tool called `name`.
pub fn tool(name: &str) -> Option<Tool> {
    TOOLS.iter().find(|(n, _)| *n == name).map(|(_, t)| *t)
}

/// `lucene-tools <Tool> [args...]`: runs the named tool and returns the
/// process exit code. An exception out of the tool is printed to `err` as the
/// JVM prints an uncaught one (`Exception in thread "main" ...`, without the
/// stack trace) and exits 1; an unknown or missing tool name prints the list
/// of tools and exits 2.
pub fn run(args: &[String], out: &mut dyn Write, err: &mut dyn Write) -> i32 {
    let Some(tool) = args.first().and_then(|name| tool(name)) else {
        let _ = writeln!(err, "usage: lucene-tools <Tool> [args...]\n\ntools:");
        for (name, _) in TOOLS {
            let _ = writeln!(err, "  {name}");
        }
        return 2;
    };
    let mut io = Io { out, err };
    let code = match tool(&args[1..], &mut io) {
        Ok(code) => code,
        Err(e) => {
            let _ = writeln!(io.err, "Exception in thread \"main\" {e}");
            1
        }
    };
    let _ = io.out.flush();
    code
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run_strs(args: &[&str]) -> (i32, String, String) {
        let args: Vec<String> = args.iter().map(|s| s.to_string()).collect();
        let (mut out, mut err) = (Vec::new(), Vec::new());
        let code = run(&args, &mut out, &mut err);
        (
            code,
            String::from_utf8(out).unwrap(),
            String::from_utf8(err).unwrap(),
        )
    }

    #[test]
    fn an_unknown_tool_lists_the_tools() {
        for args in [&[][..], &["NoSuchTool"][..]] {
            let (code, out, err) = run_strs(args);
            assert_eq!(code, 2);
            assert!(out.is_empty());
            assert!(err.starts_with("usage: lucene-tools <Tool>"));
            for (name, _) in TOOLS {
                assert!(err.contains(&format!("  {name}\n")), "{name}");
            }
        }
    }

    #[test]
    fn an_exception_is_printed_as_the_jvm_prints_it() {
        let dir = lucene_util::test_support::TempDir::new("tools-run");
        let missing = dir.path().join("none");
        let (code, out, err) = run_strs(&["GetTermInfo", missing.to_str().unwrap(), "f", "t"]);
        assert_eq!(code, 1);
        assert!(out.is_empty());
        assert!(err.starts_with("Exception in thread \"main\" "), "{err}");
    }
}
