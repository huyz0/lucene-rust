//! `PlainTextDictionary` (M13 T13.3e): one word per line, each of weight 1.

use crate::spell::Dictionary;
use crate::suggest::dictionary::read_lines;
use crate::suggest::input::{InputIterator, InputIteratorWrapper};
use crate::{Error, Result};

/// `PlainTextDictionary`: every line of the text is an entry, in order,
/// through `InputIteratorWrapper` (weight 1, no payloads, no contexts).
///
/// Departure from Java: the text is held whole and read again from the start
/// by every [`get_entry_iterator`](Dictionary::get_entry_iterator) (Java's
/// `BufferedReader` is read once, then closed).
#[derive(Debug, Clone)]
pub struct PlainTextDictionary {
    text: String,
}

impl PlainTextDictionary {
    /// `new PlainTextDictionary(Reader)`: the text, already decoded.
    pub fn from_text(text: impl Into<String>) -> Self {
        Self { text: text.into() }
    }

    /// `new PlainTextDictionary(InputStream)` / `(Path)`: UTF-8, strictly
    /// decoded.
    ///
    /// # Errors
    /// Bytes that are not UTF-8.
    pub fn from_utf8(bytes: &[u8]) -> Result<Self> {
        let text = std::str::from_utf8(bytes)
            .map_err(|e| Error::IllegalArgument(format!("MalformedInputException: {e}")))?;
        Ok(Self::from_text(text))
    }
}

impl Dictionary for PlainTextDictionary {
    fn get_entry_iterator(&self) -> Result<Box<dyn InputIterator>> {
        let lines: Vec<Vec<u8>> = read_lines(&self.text)
            .into_iter()
            .map(|l| l.as_bytes().to_vec())
            .collect();
        Ok(Box::new(InputIteratorWrapper::new(lines.into_iter())))
    }
}
