//! `IndexMergeTool`: merges source indexes into a new one -- hard-linking
//! their files where it can -- and optionally force-merges the result.
//!
//! ```text
//! lucene-tools IndexMergeTool [OPTION...] <mergedIndex> <index1> <index2> [index3] ...
//! ```

use std::sync::Arc;

use lucene_index::index_writer::IndexWriter;
use lucene_index::merge_policy::api::{MergePolicy, TieredMergePolicy};
use lucene_index::merge_policy::log::{LogByteSizeMergePolicy, LogDocMergePolicy};
use lucene_index::merge_policy::MergePolicyConfig;
use lucene_index::segment_info::LuceneVersion;
use lucene_index::segment_infos;
use lucene_search::misc::store::HardlinkCopyDirectory;
use lucene_store::directory::{Directory, FsDirectory};

use crate::cli::{arg, exception, parse_int, Error, Io, ToolResult};

/// `IndexMergeTool.USAGE`.
pub const USAGE: &str = "Usage: IndexMergeTool [OPTION...] <mergedIndex> <index1> <index2> [index3] ...\n\
Merges source indexes 'index1' .. 'indexN' into 'mergedIndex'\n\
\n\
OPTIONS:\n \
-merge-policy ClassName  specifies MergePolicy class (must be in CLASSPATH).The default is\n                          \
'org.apache.lucene.index.TieredMergePolicy.TieredMergePolicy'\n \
-max-segments N          force-merge's the index to a maximum of N segments. Default is\n                          \
to execute only the merges according to the merge policy.\n \
-verbose                 print additional details.\n";

/// The codec and version every segment this tool writes records.
const CODEC: &str = "Lucene104";
const VERSION: LuceneVersion = LuceneVersion {
    major: 10,
    minor: 5,
    bugfix: 0,
};

/// `IndexMergeTool.Options`: what [`Options::parse`] read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Options {
    pub merged_index_path: String,
    pub index_paths: Vec<String>,
    /// `-merge-policy`'s class name; `None` is the writer's default, tiered.
    pub merge_policy: Option<String>,
    /// `-max-segments`; 0 (the default) does no force-merge.
    pub max_segments: i32,
    /// `-verbose`: Java sets the writer's `InfoStream` to `System.err`. This
    /// port's writer has no info stream, so it is accepted and prints nothing.
    pub verbose: bool,
}

/// What [`Options::parse`] refuses.
#[derive(Debug)]
pub enum ParseError {
    /// Java's `IllegalArgumentException` (and its `NumberFormatException`):
    /// `main` prints the message to `System.err` and exits 2.
    IllegalArgument(String),
    /// Anything else Java's parse throws, out of `main`: an option missing
    /// its value, a merge policy class that cannot be loaded.
    Exception(Error),
}

/// `Class.forName(name).asSubclass(MergePolicy.class).getConstructor()
/// .newInstance()` for the policies of `lucene-core` that have a public
/// no-argument constructor.
///
/// # Errors
/// Java's `ClassNotFoundException` for any other name, including classes
/// Java would load but not as a no-argument `MergePolicy`.
pub fn merge_policy(class_name: &str) -> Result<Arc<dyn MergePolicy>, Error> {
    Ok(match class_name {
        "org.apache.lucene.index.TieredMergePolicy" => {
            Arc::new(TieredMergePolicy::new(MergePolicyConfig::default()))
        }
        "org.apache.lucene.index.LogDocMergePolicy" => Arc::new(LogDocMergePolicy::new()),
        "org.apache.lucene.index.LogByteSizeMergePolicy" => Arc::new(LogByteSizeMergePolicy::new()),
        other => {
            return Err(exception(format!(
                "java.lang.ClassNotFoundException: {other}"
            )))
        }
    })
}

impl Options {
    /// `Options.parse(args)`: the options, up to the first argument not
    /// starting with `-`, then the merged index and at least two sources.
    /// Java's `"--"` check compares references and never matches an
    /// argument, so `--` is an unrecognized option here too.
    ///
    /// # Errors
    /// See [`ParseError`].
    pub fn parse(args: &[String]) -> Result<Options, ParseError> {
        let mut options = Options {
            merged_index_path: String::new(),
            index_paths: Vec::new(),
            merge_policy: None,
            max_segments: 0,
            verbose: false,
        };
        let mut index = 0usize;
        while index < args.len() {
            let a = args[index].as_str();
            if !a.starts_with('-') {
                break;
            }
            match a {
                "-merge-policy" => {
                    index = index.saturating_add(1);
                    let name = arg(args, index).map_err(ParseError::Exception)?;
                    merge_policy(name).map_err(ParseError::Exception)?;
                    options.merge_policy = Some(name.to_string());
                }
                "-max-segments" => {
                    index = index.saturating_add(1);
                    let n = arg(args, index).map_err(ParseError::Exception)?;
                    options.max_segments = parse_int(n).map_err(ParseError::IllegalArgument)?;
                }
                "-verbose" => options.verbose = true,
                _ => {
                    return Err(ParseError::IllegalArgument(format!(
                        "unrecognized option: '{a}'\n{USAGE}"
                    )))
                }
            }
            index = index.saturating_add(1);
        }
        let paths = &args[index..];
        if paths.len() < 3 {
            return Err(ParseError::IllegalArgument(format!(
                "not enough parameters.\n{USAGE}"
            )));
        }
        options.merged_index_path = paths[0].clone();
        options.index_paths = paths[1..].to_vec();
        Ok(options)
    }
}

/// `IndexMergeTool.main(args)`: prints `Merging...`, adds every source
/// index to a new index at the merged path (`OpenMode.CREATE`: an index
/// already there is replaced), prints `Force-merging to N...` and
/// force-merges when `-max-segments` is positive, closes the writer --
/// running the merges its policy asks for -- and prints `Done.`.
///
/// # Errors
/// What reading the sources and writing the merged index report.
pub fn main(args: &[String], io: &mut Io<'_>) -> ToolResult {
    let options = match Options::parse(args) {
        Ok(o) => o,
        Err(ParseError::IllegalArgument(m)) => {
            writeln!(io.err, "{m}")?;
            return Ok(2);
        }
        Err(ParseError::Exception(e)) => return Err(e),
    };
    std::fs::create_dir_all(&options.merged_index_path)?;
    // Try to use hardlinks to source segments, if possible.
    let merged = HardlinkCopyDirectory::new(FsDirectory::open(&options.merged_index_path));
    let sources: Vec<FsDirectory> = options.index_paths.iter().map(FsDirectory::open).collect();
    let sources: Vec<&dyn Directory> = sources.iter().map(|d| d as &dyn Directory).collect();

    let existing = segment_infos::read_latest(&merged).is_ok();
    let mut writer = IndexWriter::open(&merged, Vec::new(), CODEC, VERSION)?;
    if existing {
        // `OpenMode.CREATE` over an index: the new commit holds none of it.
        writer.delete_all()?;
    }
    if let Some(name) = &options.merge_policy {
        writer.set_pluggable_merge_policy(Some(merge_policy(name)?));
    }

    writeln!(io.out, "Merging...")?;
    writer.add_indexes(&sources)?;
    if options.max_segments > 0 {
        writeln!(io.out, "Force-merging to {}...", options.max_segments)?;
        writer.force_merge(usize::try_from(options.max_segments).unwrap_or(1))?;
    }
    writer.close()?;
    writeln!(io.out, "Done.")?;
    Ok(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strings(args: &[&str]) -> Vec<String> {
        args.iter().map(|s| s.to_string()).collect()
    }

    fn illegal(args: &[&str]) -> String {
        match Options::parse(&strings(args)) {
            Err(ParseError::IllegalArgument(m)) => m,
            other => panic!("{other:?}"),
        }
    }

    fn thrown(args: &[&str]) -> String {
        match Options::parse(&strings(args)) {
            Err(ParseError::Exception(e)) => e.to_string(),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn options_parse_as_javas() {
        let o = Options::parse(&strings(&[
            "-merge-policy",
            "org.apache.lucene.index.LogDocMergePolicy",
            "-max-segments",
            "1",
            "-verbose",
            "out",
            "a",
            "b",
            "-c",
        ]))
        .unwrap();
        assert_eq!(o.merged_index_path, "out");
        assert_eq!(o.index_paths, ["a", "b", "-c"]);
        assert_eq!(
            o.merge_policy.as_deref(),
            Some("org.apache.lucene.index.LogDocMergePolicy")
        );
        assert_eq!(o.max_segments, 1);
        assert!(o.verbose);

        assert_eq!(
            illegal(&["--", "out", "a", "b"]),
            format!("unrecognized option: '--'\n{USAGE}")
        );
        assert_eq!(
            illegal(&["out", "a"]),
            format!("not enough parameters.\n{USAGE}")
        );
        assert_eq!(
            illegal(&["-max-segments", "x", "out", "a", "b"]),
            "For input string: \"x\""
        );
        assert!(thrown(&["-max-segments"]).starts_with("java.lang.ArrayIndexOutOfBounds"));
        assert!(thrown(&["-merge-policy"]).starts_with("java.lang.ArrayIndexOutOfBounds"));
        assert_eq!(
            thrown(&[
                "-merge-policy",
                "org.apache.lucene.index.NoMergePolicy",
                "o",
                "a",
                "b"
            ]),
            "java.lang.ClassNotFoundException: org.apache.lucene.index.NoMergePolicy"
        );
        for name in [
            "org.apache.lucene.index.TieredMergePolicy",
            "org.apache.lucene.index.LogByteSizeMergePolicy",
        ] {
            assert!(merge_policy(name).is_ok());
        }
    }

    #[test]
    fn a_parse_error_prints_its_message_and_exits_2() {
        let (mut out, mut err) = (Vec::new(), Vec::new());
        let code = main(
            &strings(&["o"]),
            &mut Io {
                out: &mut out,
                err: &mut err,
            },
        )
        .unwrap();
        assert_eq!(code, 2);
        assert!(out.is_empty());
        assert_eq!(
            String::from_utf8(err).unwrap(),
            format!("not enough parameters.\n{USAGE}\n")
        );
        let r = main(
            &strings(&["-merge-policy"]),
            &mut Io {
                out: &mut out,
                err: &mut Vec::new(),
            },
        );
        assert!(r.is_err());
    }
}
