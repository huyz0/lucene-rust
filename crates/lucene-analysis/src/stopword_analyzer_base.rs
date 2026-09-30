//! `org.apache.lucene.analysis.StopwordAnalyzerBase`.

use std::io::Read;
use std::path::Path;
use std::sync::Arc;

use crate::{wordlist_loader, AnalysisError, CharArraySet};

/// `StopwordAnalyzerBase`: the stopword set an analyzer carries, and the two
/// `loadStopwordSet` helpers.
///
/// Java's abstract base class becomes a value an analyzer embeds (see
/// [`crate::StandardAnalyzer`]); the set is held in an `Arc`, which is
/// Java's `CharArraySet.unmodifiableSet(CharArraySet.copy(stopwords))`: the
/// analyzer's set can be shared with every `StopFilter` it creates and can
/// no longer change.
#[derive(Debug, Clone)]
pub struct StopwordAnalyzerBase {
    stopwords: Arc<CharArraySet>,
}

impl Default for StopwordAnalyzerBase {
    fn default() -> Self {
        Self::new(None)
    }
}

impl StopwordAnalyzerBase {
    /// `StopwordAnalyzerBase(CharArraySet)`; `None` is `EMPTY_SET`.
    pub fn new(stopwords: Option<CharArraySet>) -> Self {
        StopwordAnalyzerBase {
            stopwords: Arc::new(stopwords.unwrap_or_else(CharArraySet::empty)),
        }
    }

    /// `getStopwordSet()`.
    pub fn stopword_set(&self) -> &Arc<CharArraySet> {
        &self.stopwords
    }

    /// `loadStopwordSet(Reader)`: [`wordlist_loader::get_word_set`].
    pub fn load_stopword_set(reader: impl Read) -> Result<CharArraySet, AnalysisError> {
        wordlist_loader::get_word_set(reader)
    }

    /// `loadStopwordSet(Path)`: the file read as UTF-8.
    pub fn load_stopword_set_from_path(path: &Path) -> Result<CharArraySet, AnalysisError> {
        let file = std::fs::File::open(path).map_err(|e| AnalysisError::Io(e.to_string()))?;
        wordlist_loader::get_word_set(std::io::BufReader::new(file))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_by_default_and_loads_from_reader_and_path() {
        assert!(StopwordAnalyzerBase::default().stopword_set().is_empty());
        let set = StopwordAnalyzerBase::load_stopword_set("the\na\n".as_bytes()).unwrap();
        let base = StopwordAnalyzerBase::new(Some(set));
        assert!(base.stopword_set().contains("the"));
        let dir = std::env::temp_dir().join(format!("stopwords-{}.txt", std::process::id()));
        std::fs::write(&dir, "x\ny\n").unwrap();
        let set = StopwordAnalyzerBase::load_stopword_set_from_path(&dir).unwrap();
        std::fs::remove_file(&dir).unwrap();
        assert_eq!(set.len(), 2);
        assert!(StopwordAnalyzerBase::load_stopword_set_from_path(&dir).is_err());
    }
}
