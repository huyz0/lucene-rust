//! `HighFreqTerms`: an index's most frequent terms, by document frequency or
//! (`-t`) by total term frequency.
//!
//! ```text
//! lucene-tools HighFreqTerms <index dir> [-t] [number_terms] [field]
//! ```

use lucene_search::directory_reader::DirectoryReader;
use lucene_search::misc::{get_high_freq_terms, TermOrder, DEFAULT_NUMTERMS};
use lucene_store::directory::FsDirectory;

use crate::cli::{exception, group_digits, parse_int, Io, ToolResult};

/// `ArrayUtil.MAX_ARRAY_LENGTH` on a 64-bit JVM with compressed oops: the
/// bound `PriorityQueue` puts on its size.
const MAX_ARRAY_LENGTH: i32 = i32::MAX - 16;

/// `usage()`'s text, printed with `println` to `System.out`.
pub const USAGE: &str = "\n\njava org.apache.lucene.misc.HighFreqTerms <index dir> [-t] [number_terms] [field]\n\t -t: order by totalTermFreq\n\n";

/// `HighFreqTerms.main(args)`. After the directory, `-t` orders by total
/// term frequency, a number is how many terms to print, and anything else is
/// the one field to read (every indexed field otherwise); a later argument of
/// the same kind wins. No arguments, or more than four, print the usage and
/// exit 1. A count of zero prints nothing.
///
/// # Errors
/// What opening and reading the index reports, and Java's
/// `RuntimeException` for a field with no terms or an index with no indexed
/// fields, then `PriorityQueue`'s `IllegalArgumentException` for a negative count.
pub fn main(args: &[String], io: &mut Io<'_>) -> ToolResult {
    if args.is_empty() || args.len() > 4 {
        writeln!(io.out, "{USAGE}")?;
        return Ok(1);
    }
    let dir = FsDirectory::open(&args[0]);
    let mut order = TermOrder::DocFreq;
    let mut num_terms = i32::try_from(DEFAULT_NUMTERMS).unwrap_or(i32::MAX);
    let mut field = None;
    for a in &args[1..] {
        if a == "-t" {
            order = TermOrder::TotalTermFreq;
        } else {
            match parse_int(a) {
                Ok(n) => num_terms = n,
                Err(_) => field = Some(a.as_str()),
            }
        }
    }
    let reader = DirectoryReader::open(&dir)?;
    // Java checks the field before it sizes the queue, which refuses a
    // negative size or one past `ArrayUtil.MAX_ARRAY_LENGTH`: a run at size
    // 0 reports the field's error first.
    let size = usize::try_from(num_terms)
        .ok()
        .filter(|_| num_terms < MAX_ARRAY_LENGTH);
    let terms =
        get_high_freq_terms(&reader, size.unwrap_or(0), field, order).map_err(|e| match e {
            lucene_search::Error::IllegalState(m) => {
                exception(format!("java.lang.RuntimeException: {m}"))
            }
            other => other.into(),
        })?;
    if size.is_none() {
        return Err(exception(format!(
            "java.lang.IllegalArgumentException: maxSize must be >= 0 and < {MAX_ARRAY_LENGTH}; got: {num_terms}"
        )));
    }
    for t in terms {
        writeln!(
            io.out,
            "{}:{} \t totalTF = {} \t docFreq = {} ",
            t.field,
            t.term_text(),
            group_digits(t.total_term_freq),
            group_digits(i64::from(t.doc_freq))
        )?;
    }
    Ok(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wrong_arity_prints_the_usage() {
        for n in [0, 5] {
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
        }
    }
}
