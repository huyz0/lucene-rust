//! `org.apache.lucene.analysis.path`: `PathHierarchyTokenizer` and
//! `ReversePathHierarchyTokenizer`, reading UTF-16 units one at a time as
//! Java's `Reader.read()` does (through a small buffer).

use crate::attributes::AttributeSource;
use crate::reader::CharReader;
use crate::token_stream::{TokenStream, Tokenizer, TokenizerInput};
use crate::AnalysisError;

/// `PathHierarchyTokenizer.DEFAULT_DELIMITER`.
pub const DEFAULT_DELIMITER: char = '/';
/// `PathHierarchyTokenizer.DEFAULT_SKIP`.
pub const DEFAULT_SKIP: i32 = 0;

/// `Reader.read()`: one UTF-16 unit at a time, buffered.
#[derive(Default)]
struct UnitReader {
    buf: Vec<u16>,
    pos: usize,
    len: usize,
}

impl UnitReader {
    fn reset(&mut self) {
        self.pos = 0;
        self.len = 0;
    }

    fn read(&mut self, input: &mut TokenizerInput) -> Result<Option<u16>, AnalysisError> {
        if self.pos == self.len {
            if self.buf.is_empty() {
                self.buf = vec![0; 1024];
            }
            self.len = input.reader()?.read(&mut self.buf)?;
            self.pos = 0;
            if self.len == 0 {
                return Ok(None);
            }
        }
        let u = self.buf[self.pos];
        self.pos += 1;
        Ok(Some(u))
    }
}

fn check(skip: i32) -> Result<(), AnalysisError> {
    if skip < 0 {
        return Err(AnalysisError::IllegalArgument(
            "skip cannot be negative".into(),
        ));
    }
    Ok(())
}

fn unit(c: char) -> Result<u16, AnalysisError> {
    u16::try_from(u32::from(c))
        .map_err(|_| AnalysisError::IllegalArgument(format!("{c:?} is not a Java char")))
}

/// `org.apache.lucene.analysis.path.PathHierarchyTokenizer`: `/a/b/c` ->
/// `/a`, `/a/b`, `/a/b/c`.
pub struct PathHierarchyTokenizer {
    atts: AttributeSource,
    input: TokenizerInput,
    reader: UnitReader,
    delimiter: u16,
    replacement: u16,
    skip: i32,
    start_position: i32,
    skipped: i32,
    end_delimiter: bool,
    result_token: Vec<u16>,
    chars_read: i32,
    term: Vec<u16>,
}

impl Default for PathHierarchyTokenizer {
    fn default() -> Self {
        Self::new(DEFAULT_DELIMITER, DEFAULT_DELIMITER, DEFAULT_SKIP).expect("defaults are valid")
    }
}

impl PathHierarchyTokenizer {
    /// `new PathHierarchyTokenizer(char delimiter, char replacement, int skip)`.
    pub fn new(delimiter: char, replacement: char, skip: i32) -> Result<Self, AnalysisError> {
        check(skip)?;
        Ok(PathHierarchyTokenizer {
            atts: AttributeSource::new(),
            input: TokenizerInput::new(),
            reader: UnitReader::default(),
            delimiter: unit(delimiter)?,
            replacement: unit(replacement)?,
            skip,
            start_position: 0,
            skipped: 0,
            end_delimiter: false,
            result_token: Vec::new(),
            chars_read: 0,
            term: Vec::new(),
        })
    }

    fn set_token(&mut self, length: usize) -> Result<(), AnalysisError> {
        // `length` counts the units appended after `resultToken`.
        let total = self.term.len().min(self.result_token.len() + length);
        self.term.truncate(total);
        self.atts.set_term_utf16(&self.term);
        let start = self.input.correct_offset(self.start_position);
        let end = self
            .input
            .correct_offset(self.start_position + total as i32);
        self.atts.set_offset(start, end)
    }
}

impl TokenStream for PathHierarchyTokenizer {
    fn attributes(&self) -> &AttributeSource {
        &self.atts
    }

    fn attributes_mut(&mut self) -> &mut AttributeSource {
        &mut self.atts
    }

    // Java: PathHierarchyTokenizer.incrementToken
    fn increment_token(&mut self) -> Result<bool, AnalysisError> {
        self.atts.clear_attributes();
        self.term.clear();
        self.term.extend_from_slice(&self.result_token);
        self.atts.set_position_increment(1)?;
        let mut length = 0usize;
        let mut added = false;
        if self.end_delimiter {
            self.term.push(self.replacement);
            length += 1;
            self.end_delimiter = false;
            added = true;
        }
        loop {
            let Some(c) = self.reader.read(&mut self.input)? else {
                if self.skipped > self.skip {
                    self.set_token(length)?;
                    if added {
                        self.result_token.clone_from(&self.term);
                    }
                    return Ok(added);
                }
                return Ok(false);
            };
            self.chars_read += 1;
            if !added {
                added = true;
                self.skipped += 1;
                if self.skipped > self.skip {
                    self.term.push(if c == self.delimiter {
                        self.replacement
                    } else {
                        c
                    });
                    length += 1;
                } else {
                    self.start_position += 1;
                }
            } else if c == self.delimiter {
                if self.skipped > self.skip {
                    self.end_delimiter = true;
                    break;
                }
                self.skipped += 1;
                if self.skipped > self.skip {
                    self.term.push(self.replacement);
                    length += 1;
                } else {
                    self.start_position += 1;
                }
            } else if self.skipped > self.skip {
                self.term.push(c);
                length += 1;
            } else {
                self.start_position += 1;
            }
        }
        self.set_token(length)?;
        self.result_token.clone_from(&self.term);
        Ok(true)
    }

    // Java: PathHierarchyTokenizer.end
    fn end(&mut self) -> Result<(), AnalysisError> {
        self.atts.end_attributes();
        let f = self.input.correct_offset(self.chars_read);
        self.atts.set_offset(f, f)
    }

    // Java: PathHierarchyTokenizer.reset
    fn reset(&mut self) -> Result<(), AnalysisError> {
        self.input.reset();
        self.reader.reset();
        self.result_token.clear();
        self.chars_read = 0;
        self.end_delimiter = false;
        self.skipped = 0;
        self.start_position = 0;
        Ok(())
    }

    fn close(&mut self) -> Result<(), AnalysisError> {
        self.input.close()
    }

    fn as_tokenizer(&mut self) -> Option<&mut dyn Tokenizer> {
        Some(self)
    }
}

impl Tokenizer for PathHierarchyTokenizer {
    fn set_reader(&mut self, input: Box<dyn CharReader>) -> Result<(), AnalysisError> {
        self.input.set_reader(input)
    }
}

/// `org.apache.lucene.analysis.path.ReversePathHierarchyTokenizer`:
/// `www.site.co.uk` -> `www.site.co.uk`, `site.co.uk`, `co.uk`, `uk`.
pub struct ReversePathHierarchyTokenizer {
    atts: AttributeSource,
    input: TokenizerInput,
    reader: UnitReader,
    delimiter: u16,
    replacement: u16,
    skip: i32,
    end_position: i32,
    final_offset: i32,
    skipped: i32,
    delimiter_positions: Vec<i32>,
    delimiters_count: i32,
    result_token: Vec<u16>,
}

impl Default for ReversePathHierarchyTokenizer {
    fn default() -> Self {
        Self::new(DEFAULT_DELIMITER, DEFAULT_DELIMITER, DEFAULT_SKIP).expect("defaults are valid")
    }
}

impl ReversePathHierarchyTokenizer {
    /// `new ReversePathHierarchyTokenizer(char delimiter, char replacement, int skip)`.
    pub fn new(delimiter: char, replacement: char, skip: i32) -> Result<Self, AnalysisError> {
        check(skip)?;
        Ok(ReversePathHierarchyTokenizer {
            atts: AttributeSource::new(),
            input: TokenizerInput::new(),
            reader: UnitReader::default(),
            delimiter: unit(delimiter)?,
            replacement: unit(replacement)?,
            skip,
            end_position: 0,
            final_offset: 0,
            skipped: 0,
            delimiter_positions: Vec::new(),
            delimiters_count: -1,
            result_token: Vec::new(),
        })
    }
}

impl TokenStream for ReversePathHierarchyTokenizer {
    fn attributes(&self) -> &AttributeSource {
        &self.atts
    }

    fn attributes_mut(&mut self) -> &mut AttributeSource {
        &mut self.atts
    }

    // Java: ReversePathHierarchyTokenizer.incrementToken
    fn increment_token(&mut self) -> Result<bool, AnalysisError> {
        self.atts.clear_attributes();
        if self.delimiters_count == -1 {
            let mut length = 0i32;
            self.delimiter_positions.push(0);
            while let Some(c) = self.reader.read(&mut self.input)? {
                length += 1;
                if c == self.delimiter {
                    self.delimiter_positions.push(length);
                    self.result_token.push(self.replacement);
                } else {
                    self.result_token.push(c);
                }
            }
            self.delimiters_count = self.delimiter_positions.len() as i32;
            if *self.delimiter_positions.last().expect("0 pushed") < length {
                self.delimiter_positions.push(length);
                self.delimiters_count += 1;
            }
            let idx = self.delimiters_count - 1 - self.skip;
            if idx >= 0 {
                self.end_position = self.delimiter_positions[idx as usize];
            }
            self.final_offset = self.input.correct_offset(length);
        }
        self.atts.set_position_increment(1)?;
        if self.skipped < self.delimiters_count - self.skip - 1 {
            let start = self.delimiter_positions[self.skipped as usize];
            self.atts
                .set_term_utf16(&self.result_token[start as usize..self.end_position as usize]);
            let (s, e) = (
                self.input.correct_offset(start),
                self.input.correct_offset(self.end_position),
            );
            self.atts.set_offset(s, e)?;
            self.skipped += 1;
            return Ok(true);
        }
        Ok(false)
    }

    // Java: ReversePathHierarchyTokenizer.end
    fn end(&mut self) -> Result<(), AnalysisError> {
        self.atts.end_attributes();
        self.atts.set_offset(self.final_offset, self.final_offset)
    }

    // Java: ReversePathHierarchyTokenizer.reset
    fn reset(&mut self) -> Result<(), AnalysisError> {
        self.input.reset();
        self.reader.reset();
        self.result_token.clear();
        self.final_offset = 0;
        self.end_position = 0;
        self.skipped = 0;
        self.delimiters_count = -1;
        self.delimiter_positions.clear();
        Ok(())
    }

    fn close(&mut self) -> Result<(), AnalysisError> {
        self.input.close()
    }

    fn as_tokenizer(&mut self) -> Option<&mut dyn Tokenizer> {
        Some(self)
    }
}

impl Tokenizer for ReversePathHierarchyTokenizer {
    fn set_reader(&mut self, input: Box<dyn CharReader>) -> Result<(), AnalysisError> {
        self.input.set_reader(input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reader::StrReader;
    use crate::util::canned::render;

    fn run(t: &mut dyn Tokenizer, text: &str) -> String {
        t.set_reader(Box::new(StrReader::new(text))).unwrap();
        render(t)
    }

    #[test]
    fn forward() {
        let mut t = PathHierarchyTokenizer::default();
        assert_eq!(
            run(&mut t, "/a/b/c"),
            "/a:0:2:1:1 /a/b:0:4:1:1 /a/b/c:0:6:1:1|6|0"
        );
        let mut t = PathHierarchyTokenizer::new('\\', '/', 1).unwrap();
        assert_eq!(run(&mut t, "c:\\x\\y"), "/x:2:4:1:1 /x/y:2:6:1:1|6|0");
        assert_eq!(run(&mut t, "c:"), "|2|0");
        let mut t = PathHierarchyTokenizer::default();
        assert_eq!(run(&mut t, "a/"), "a:0:1:1:1 a/:0:2:1:1|2|0");
        assert_eq!(run(&mut t, ""), "|0|0");
        assert!(PathHierarchyTokenizer::new('/', '/', -1).is_err());
        assert!(PathHierarchyTokenizer::new('😀', '/', 0).is_err());
    }

    #[test]
    fn reverse() {
        let mut t = ReversePathHierarchyTokenizer::new('.', '.', 0).unwrap();
        assert_eq!(
            run(&mut t, "www.site.co"),
            "www.site.co:0:11:1:1 site.co:4:11:1:1 co:9:11:1:1|11|0"
        );
        let mut t = ReversePathHierarchyTokenizer::new('.', '.', 1).unwrap();
        assert_eq!(
            run(&mut t, "www.site.co"),
            "www.site.:0:9:1:1 site.:4:9:1:1|11|0"
        );
        let mut t = ReversePathHierarchyTokenizer::default();
        assert_eq!(run(&mut t, "/a/"), "/a/:0:3:1:1 a/:1:3:1:1|3|0");
        assert!(ReversePathHierarchyTokenizer::new('/', '/', -1).is_err());
    }
}
