//! `org.apache.lucene.search.spell` (`lucene-suggest`, M13 T13.3): the
//! spell-checking package. This file starts with the package's interface,
//! [`Dictionary`]; the suggest package's dictionaries implement it.

use crate::suggest::input::InputIterator;
use crate::Result;

/// `org.apache.lucene.search.spell.Dictionary`: a source of weighted terms
/// for a suggester to build from.
pub trait Dictionary {
    /// `getEntryIterator()`: the dictionary's entries. Java hands out a lazy
    /// iterator that can fail on any `next()`; the dictionaries here read
    /// their source when this is called, so a malformed entry is reported
    /// here instead.
    fn get_entry_iterator(&self) -> Result<Box<dyn InputIterator>>;
}
