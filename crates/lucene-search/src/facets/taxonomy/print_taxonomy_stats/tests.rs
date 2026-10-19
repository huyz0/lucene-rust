//! Unit tests.
#![allow(clippy::arithmetic_side_effects)]

use super::*;
use crate::facets::test_index::basic_index;

/// A writer that fails once `left` bytes were written.
struct FailAfter {
    left: usize,
}

impl Write for FailAfter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        if buf.len() > self.left {
            return Err(std::io::Error::other("full"));
        }
        self.left -= buf.len();
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[test]
fn a_failing_writer_fails_at_every_point() {
    let idx = basic_index();
    idx.with(|_, _, taxo| {
        let mut full = Vec::new();
        print_stats(taxo, &mut full, true).unwrap();
        // Every prefix of the output: the write that does not fit is the
        // error, wherever in the tree it falls.
        for left in 0..full.len() {
            let e = print_stats(taxo, &mut FailAfter { left }, true).unwrap_err();
            assert!(
                matches!(e, Error::Store(lucene_store::Error::Io(_))),
                "{left}"
            );
        }
        print_stats(taxo, &mut FailAfter { left: full.len() }, true).unwrap();
    });
    // The usage line too.
    let e = run(&[], &mut FailAfter { left: 0 }).unwrap_err();
    assert!(matches!(e, Error::Store(_)));
}
