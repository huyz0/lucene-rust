//! `PrintTaxonomyStats`: a taxonomy's size, each dim's immediate and total
//! children, and optionally every category under each dim as an indented
//! tree.
//!
//! Differs: `main` is [`run`], which takes the arguments and a writer and
//! returns the exit status instead of printing to `System.out` and calling
//! `System.exit`; the taxonomy is opened from an [`FsDirectory`].

use std::io::Write;

use lucene_store::FsDirectory;

use super::{DirectoryTaxonomyReader, TaxonomyReader, INVALID_ORDINAL, ROOT_ORDINAL};
use crate::{Error, Result};

/// The usage line `main` prints for bad arguments.
pub const USAGE: &str =
    "\nUsage: java -classpath ... org.apache.lucene.facet.util.PrintTaxonomyStats \
                         [-printTree] /path/to/taxononmy/index\n";

/// `PrintTaxonomyStats.main(args)`: `[-printTree] /path/to/taxonomy`. Bad
/// arguments print [`USAGE`] and give `1`; otherwise the stats of the
/// taxonomy at the path, and `0`.
///
/// # Errors
/// Whatever opening or reading the taxonomy, or writing, reports.
pub fn run(args: &[&str], out: &mut dyn Write) -> Result<i32> {
    let mut print_tree = false;
    let mut path = None;
    for &a in args {
        if a == "-printTree" {
            print_tree = true;
        } else {
            path = Some(a);
        }
    }
    let wanted = if print_tree { 2 } else { 1 };
    let Some(path) = path.filter(|_| args.len() == wanted) else {
        writeln!(out, "{USAGE}").map_err(io)?;
        return Ok(1);
    };
    let dir = FsDirectory::open(path);
    let r = DirectoryTaxonomyReader::open(&dir)?;
    print_stats(&r, out, print_tree)?;
    r.close()?;
    Ok(0)
}

fn io(e: std::io::Error) -> Error {
    Error::Store(lucene_store::Error::Io(e))
}

/// `printStats(r, out, printTree)`.
///
/// # Errors
/// Whatever reading the taxonomy or writing reports.
pub fn print_stats(r: &dyn TaxonomyReader, out: &mut dyn Write, print_tree: bool) -> Result<()> {
    writeln!(out, "{} total categories.", r.get_size()?).map_err(io)?;
    let mut it = r.get_children(ROOT_ORDINAL)?;
    loop {
        let child = it.next_ordinal();
        if child == INVALID_ORDINAL {
            break;
        }
        let mut children = r.get_children(child)?;
        let mut immediate = 0i32;
        while children.next_ordinal() != INVALID_ORDINAL {
            immediate = immediate.saturating_add(1);
        }
        let cp = r.get_path(child)?;
        let dim = cp.path().first().map_or("", String::as_str);
        writeln!(
            out,
            "/{dim}: {immediate} immediate children; {} total categories",
            count_all_children(r, child)?.saturating_add(1)
        )
        .map_err(io)?;
        if print_tree {
            print_all_children(out, r, child, "  ", 1)?;
        }
    }
    Ok(())
}

/// `countAllChildren(r, ord)`.
fn count_all_children(r: &dyn TaxonomyReader, ord: i32) -> Result<i32> {
    let mut count = 0i32;
    let mut it = r.get_children(ord)?;
    loop {
        let child = it.next_ordinal();
        if child == INVALID_ORDINAL {
            return Ok(count);
        }
        count = count.saturating_add(count_all_children(r, child)?.saturating_add(1));
    }
}

/// `printAllChildren(out, r, ord, indent, depth)`.
fn print_all_children(
    out: &mut dyn Write,
    r: &dyn TaxonomyReader,
    ord: i32,
    indent: &str,
    depth: usize,
) -> Result<()> {
    let mut it = r.get_children(ord)?;
    loop {
        let child = it.next_ordinal();
        if child == INVALID_ORDINAL {
            return Ok(());
        }
        let cp = r.get_path(child)?;
        let label = cp.path().get(depth).map_or("", String::as_str);
        writeln!(out, "{indent}/{label}").map_err(io)?;
        print_all_children(
            out,
            r,
            child,
            &format!("{indent}  "),
            depth.saturating_add(1),
        )?;
    }
}

#[cfg(test)]
mod tests;
