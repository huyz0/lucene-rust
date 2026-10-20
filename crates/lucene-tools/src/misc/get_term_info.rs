//! `GetTermInfo`: a term's total term frequency and document frequency.
//!
//! ```text
//! lucene-tools GetTermInfo <index dir> field term
//! ```

use std::io::Write;

use lucene_search::directory_reader::DirectoryReader;
use lucene_search::reader::IndexReader;
use lucene_store::directory::{Directory, FsDirectory};

use crate::cli::{group_digits, Error, Io, ToolResult};

/// `usage()`'s text, printed with `println` to `System.out`.
pub const USAGE: &str =
    "\n\nusage:\n\tjava org.apache.lucene.misc.GetTermInfo <index dir> field term \n\n";

/// `GetTermInfo.main(args)`: exactly three arguments, else the usage and
/// exit code 1.
pub fn main(args: &[String], io: &mut Io<'_>) -> ToolResult {
    let [dir, field, text] = args else {
        writeln!(io.out, "{USAGE}")?;
        return Ok(1);
    };
    get_term_info(&FsDirectory::open(dir), field, text, io.out)?;
    Ok(0)
}

/// `getTermInfo(dir, term)`: prints
/// `field:text \t totalTF = n \t doc freq = n \n`, the numbers grouped by
/// three (`%,d` under `Locale.ROOT`). A term the index does not hold prints
/// zeros.
///
/// # Errors
/// What opening and reading the index reports (no index at `dir` is Java's
/// `IndexNotFoundException`).
pub fn get_term_info(
    dir: &dyn Directory,
    field: &str,
    text: &str,
    out: &mut dyn Write,
) -> Result<(), Error> {
    let reader = DirectoryReader::open(dir)?;
    let total_tf = reader.total_term_freq(field, text.as_bytes())?;
    let doc_freq = reader.doc_freq(field, text.as_bytes())?;
    writeln!(
        out,
        "{field}:{text} \t totalTF = {} \t doc freq = {} ",
        group_digits(total_tf),
        group_digits(doc_freq)
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wrong_arity_prints_the_usage() {
        for n in [0, 2, 4] {
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
            assert_eq!(code, 1);
            assert_eq!(String::from_utf8(out).unwrap(), format!("{USAGE}\n"));
            assert!(err.is_empty());
        }
    }
}
