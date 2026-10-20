//! `IndexSplitter`'s command line: list an index's segments, remove some of
//! them from its latest commit, or copy some of them into a new index.
//!
//! ```text
//! lucene-tools IndexSplitter <srcDir> -l
//! lucene-tools IndexSplitter <srcDir> -d <segments>+
//! lucene-tools IndexSplitter <srcDir> <destDir> <segments>+
//! ```

use lucene_search::misc::index::IndexSplitter;
use lucene_store::directory::FsDirectory;

use crate::cli::{Io, ToolResult};

/// The usage, printed to `System.err` (and the exit code is 0).
pub const USAGE: &str = "Usage: IndexSplitter <srcDir> -l (list the segments and their sizes)\n\
IndexSplitter <srcDir> <destDir> <segments>+\n\
IndexSplitter <srcDir> -d (delete the following segments)\n";

/// `IndexSplitter.main(args)`. Java's own "srcdir doesn't exist" check comes
/// after reading the index, which already fails on a missing directory, so
/// it is never reached and is not reproduced.
///
/// # Errors
/// What reading, writing or copying the index reports, and a name in the
/// destination form that is not a segment.
pub fn main(args: &[String], io: &mut Io<'_>) -> ToolResult {
    let [src, command, segs @ ..] = args else {
        write!(io.err, "{USAGE}")?;
        return Ok(0);
    };
    let src_dir = FsDirectory::open(src);
    let mut splitter = IndexSplitter::new(&src_dir)?;
    let segs: Vec<&str> = segs.iter().map(String::as_str).collect();
    match command.as_str() {
        "-l" => {
            for line in splitter.list_segments()? {
                writeln!(io.out, "{line}")?;
            }
        }
        "-d" => splitter.remove(&segs)?,
        dest => {
            std::fs::create_dir_all(dest)?;
            splitter.split(&FsDirectory::open(dest), &segs)?;
        }
    }
    Ok(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn too_few_arguments_print_the_usage_to_stderr() {
        for n in [0, 1] {
            let args = vec!["x".to_string(); n];
            let (mut out, mut err) = (Vec::new(), Vec::new());
            let code = main(
                &args,
                &mut Io {
                    out: &mut out,
                    err: &mut err,
                },
            )
            .unwrap();
            assert_eq!(code, 0);
            assert!(out.is_empty());
            assert_eq!(String::from_utf8(err).unwrap(), USAGE);
        }
    }
}
