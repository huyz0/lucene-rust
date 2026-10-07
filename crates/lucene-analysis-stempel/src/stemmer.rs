//! `org.apache.lucene.analysis.stempel.StempelStemmer`: a word's stem is the
//! command its table's `getLastOnPath` answers, applied by
//! [`crate::diff::apply`]; and Lucene's default Polish table
//! (`stemmer_20000.tbl`, vendored zlib-compressed from the 10.5.0 jar).

use std::sync::{Arc, LazyLock};

use crate::diff;
use crate::egothor::{OutOfBounds, Table};
use crate::StempelError;

/// `PolishAnalyzer.DEFAULT_STEMMER_FILE`, zlib-compressed.
const DEFAULT_TABLE_Z: &[u8] = include_bytes!("resources/stemmer_20000.tbl.z");

/// `PolishAnalyzer.getDefaultTable()`: the table, read once.
pub fn default_table() -> Arc<Table> {
    static TABLE: LazyLock<Arc<Table>> = LazyLock::new(|| {
        let bytes = miniz_oxide::inflate::decompress_to_vec_zlib(DEFAULT_TABLE_Z)
            .expect("the vendored stemmer table inflates");
        Arc::new(Table::load(&bytes).expect("the vendored stemmer table reads"))
    });
    Arc::clone(&TABLE)
}

/// `org.apache.lucene.analysis.stempel.StempelStemmer`.
#[derive(Debug, Clone)]
pub struct StempelStemmer {
    table: Arc<Table>,
}

impl StempelStemmer {
    /// `new StempelStemmer(Trie)`.
    pub fn new(table: Arc<Table>) -> Self {
        StempelStemmer { table }
    }

    /// `new StempelStemmer(InputStream)`: `StempelStemmer.load` of a
    /// table's bytes.
    pub fn load(bytes: &[u8]) -> Result<Self, StempelError> {
        Ok(Self::new(Arc::new(Table::load(bytes)?)))
    }

    /// The table.
    pub fn table(&self) -> &Arc<Table> {
        &self.table
    }

    /// `StempelStemmer.stem(CharSequence)`: the stem, or `None` (Java's
    /// `null`) when the table has no command or the stem is empty. `Err`
    /// for an empty word over a plain `Trie`, where Java throws
    /// `StringIndexOutOfBoundsException` (a `MultiTrie2` catches it).
    pub fn stem(&self, word: &[u16]) -> Result<Option<Vec<u16>>, OutOfBounds> {
        let Some(cmd) = self.table.get_last_on_path(word)? else {
            return Ok(None);
        };
        let mut buffer = word.to_vec();
        diff::apply(&mut buffer, &cmd);
        Ok((!buffer.is_empty()).then_some(buffer))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn u(s: &str) -> Vec<u16> {
        s.encode_utf16().collect()
    }

    #[test]
    fn default_table_stems() {
        let s = StempelStemmer::new(default_table());
        assert!(matches!(**s.table(), Table::Multi(_)));
        let stem = |w: &str| {
            s.stem(&u(w))
                .unwrap()
                .map(|v| String::from_utf16(&v).unwrap())
        };
        assert_eq!(stem("kotu").as_deref(), Some("kot"));
        assert_eq!(stem("przeciwdziałanie").as_deref(), Some("przeciwdziałanć"));
        assert_eq!(stem(""), None);
        assert!(StempelStemmer::load(&[0]).is_err());
    }

    #[test]
    fn empty_stems_are_none() {
        // A plain trie whose command deletes the whole word.
        let mut bytes = vec![0, 2, b'-', b'0'];
        bytes.push(1); // forward
        bytes.extend_from_slice(&0i32.to_be_bytes()); // root
        bytes.extend_from_slice(&1i32.to_be_bytes());
        bytes.extend_from_slice(&[0, 2, b'D', b'a']);
        bytes.extend_from_slice(&1i32.to_be_bytes()); // one row
        bytes.extend_from_slice(&1i32.to_be_bytes()); // one cell
        bytes.extend_from_slice(&u16::from(b'a').to_be_bytes());
        for v in [0i32, 1, -1, 0] {
            bytes.extend_from_slice(&v.to_be_bytes());
        }
        let s = StempelStemmer::load(&bytes).unwrap();
        assert_eq!(s.stem(&u("a")), Ok(None));
        assert_eq!(s.stem(&u("b")), Ok(None));
        assert_eq!(s.stem(&[]), Err(OutOfBounds));
    }
}
