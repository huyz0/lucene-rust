//! `org.apache.lucene.luke.models.util.twentynewsgroups`: the messages of a
//! 20 Newsgroups dump, and the document each one indexes as.
//!
//! Differs: Java visits a dump's files in the order the file system lists
//! them, which is not the same on two machines; here they are visited by
//! name, so the documents' order -- their ids -- is the same everywhere. A
//! file whose header never ends (no blank line) is Java's
//! `NullPointerException`, which `walkFileTree` lets through and so fails
//! the whole parse; here the header simply ends there and the message has
//! no body. A file that is not UTF-8 is Java's `MalformedInputException`,
//! which `visitFile` logs and skips; here it is skipped too.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use lucene_analysis::factory::spi;
use lucene_analysis::miscellaneous::PerFieldAnalyzerWrapper;
use lucene_analysis::Analyzer;
use lucene_index::document::{
    Document, Field, FieldType, IndexOptions, IntPoint, SortedNumericDocValuesField,
    SortedSetDocValuesField, Store, StoredValue, StringField, TextField,
};

use crate::cli::{java_lines, java_trim};

/// `Message`: one message's header fields and body.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Message {
    pub from: Option<String>,
    pub newsgroups: Option<Vec<String>>,
    pub subject: Option<String>,
    pub message_id: Option<String>,
    pub date: Option<String>,
    pub organization: Option<String>,
    /// Every body line, each followed by a space.
    pub body: Option<String>,
    pub lines: i32,
}

impl Message {
    /// `MessageFilesParser.parse(file)` of a file's text: header lines up to
    /// the first empty one (a line without `:` is skipped), then the body.
    pub fn parse(text: &str) -> Message {
        let mut message = Message::default();
        let mut lines = java_lines(text);
        let mut header_ended = false;
        for line in lines.by_ref() {
            if line.is_empty() {
                header_ended = true;
                break;
            }
            let Some((att, val)) = line.split_once(':') else {
                continue;
            };
            let val = java_trim(val);
            match java_trim(att) {
                "From" => message.from = Some(val.to_string()),
                "Newsgroups" => {
                    message.newsgroups = Some(java_split_comma(val));
                }
                "Subject" => message.subject = Some(val.to_string()),
                "Message-ID" => message.message_id = Some(val.to_string()),
                "Date" => message.date = Some(val.to_string()),
                "Organization" => message.organization = Some(val.to_string()),
                "Lines" => {
                    if let Ok(n) = crate::cli::parse_int(val) {
                        message.lines = n;
                    }
                }
                _ => {}
            }
        }
        let mut body = String::new();
        // The empty line that ended the header starts the body, as Java's
        // loop goes on from it.
        if header_ended {
            body.push(' ');
        }
        for line in lines {
            body.push_str(line);
            body.push(' ');
        }
        message.body = Some(body);
        message
    }

    /// `toLuceneDoc()`.
    pub fn to_lucene_doc(&self) -> Document {
        let mut doc = Document::new();
        if let Some(from) = &self.from {
            doc.add(TextField::new("from", from.clone(), Store::Yes));
        }
        if let Some(groups) = &self.newsgroups {
            for group in groups {
                doc.add(StringField::new("newsgroup", group.clone(), Store::Yes));
                doc.add(SortedSetDocValuesField::new(
                    "newsgroup_sort",
                    group.as_bytes().to_vec(),
                ));
            }
        }
        if let Some(subject) = &self.subject {
            doc.add(text_field("subject", subject, subject_field_type()));
        }
        if let Some(id) = &self.message_id {
            doc.add(StringField::new("messageId", id.clone(), Store::Yes));
        }
        if let Some(date) = &self.date {
            doc.add(Field::stored("date_raw", StoredValue::String(date.clone())));
        }
        if let Some(organization) = &self.organization {
            doc.add(TextField::new(
                "organization",
                organization.clone(),
                Store::Yes,
            ));
        }
        doc.add(IntPoint::new("lines_range", &[self.lines]).expect("one dimension"));
        doc.add(SortedNumericDocValuesField::new(
            "lines_sort",
            i64::from(self.lines),
        ));
        doc.add(Field::stored(
            "lines_raw",
            StoredValue::String(self.lines.to_string()),
        ));
        if let Some(body) = &self.body {
            doc.add(text_field("body", body, body_field_type()));
        }
        doc
    }

    /// `createLuceneAnalyzer()`: the standard analyzer, `from` through
    /// `UAX29URLEmailAnalyzer`.
    pub fn create_lucene_analyzer() -> Arc<Analyzer> {
        let email = spi::new_analyzer("org.apache.lucene.analysis.email.UAX29URLEmailAnalyzer")
            .expect("UAX29URLEmailAnalyzer is built in");
        let standard = spi::new_analyzer("org.apache.lucene.analysis.standard.StandardAnalyzer")
            .expect("StandardAnalyzer is built in");
        let mut map = HashMap::new();
        map.insert("from".to_string(), email);
        Arc::new(PerFieldAnalyzerWrapper::new(standard, map).into_analyzer())
    }
}

/// `String.split(",")`: trailing empty strings dropped, so `","` is `[]`;
/// `""` alone is `[""]` (no match: the input itself).
fn java_split_comma(s: &str) -> Vec<String> {
    if s.is_empty() {
        return vec![String::new()];
    }
    let mut parts: Vec<String> = s.split(',').map(str::to_string).collect();
    while parts.last().is_some_and(String::is_empty) {
        parts.pop();
    }
    parts
}

fn text_field(name: &str, value: &str, field_type: FieldType) -> Field {
    Field::from_string(name, value, field_type).expect("indexed and stored")
}

/// `SUBJECT_FIELD_TYPE`.
fn subject_field_type() -> FieldType {
    let mut t = FieldType::new();
    set(t.set_index_options(IndexOptions::DocsAndFreqsAndPositionsAndOffsets));
    set(t.set_tokenized(true));
    set(t.set_stored(true));
    t
}

/// `BODY_FIELD_TYPE`.
fn body_field_type() -> FieldType {
    let mut t = FieldType::new();
    set(t.set_index_options(IndexOptions::DocsAndFreqsAndPositions));
    set(t.set_tokenized(true));
    set(t.set_stored(true));
    set(t.set_store_term_vectors(true));
    set(t.set_store_term_vector_positions(true));
    set(t.set_store_term_vector_offsets(true));
    t
}

/// A setter of an unfrozen type cannot fail.
fn set(r: lucene_index::document::Result<()>) {
    r.expect("the type is not frozen");
}

/// `MessageFilesParser`: every regular file under a directory, as a message.
pub struct MessageFilesParser {
    root: PathBuf,
}

impl MessageFilesParser {
    pub fn new(root: &Path) -> Self {
        MessageFilesParser {
            root: root.to_path_buf(),
        }
    }

    /// `parseAll()`: the messages of every regular file under the root, by
    /// path.
    ///
    /// # Errors
    /// The root or a directory under it cannot be listed.
    pub fn parse_all(&self) -> std::io::Result<Vec<Message>> {
        let mut files = Vec::new();
        walk(&self.root, &mut files)?;
        files.sort();
        Ok(files
            .iter()
            // `visitFile` logs an unreadable file and goes on.
            .filter_map(|f| std::fs::read(f).ok())
            .filter_map(|bytes| String::from_utf8(bytes).ok())
            .map(|text| Message::parse(&text))
            .collect())
    }
}

fn walk(path: &Path, files: &mut Vec<PathBuf>) -> std::io::Result<()> {
    let meta = std::fs::metadata(path)?;
    if meta.is_file() {
        files.push(path.to_path_buf());
        return Ok(());
    }
    if meta.is_dir() {
        for entry in std::fs::read_dir(path)? {
            walk(&entry?.path(), files)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_message_is_its_header_and_body() {
        let m = Message::parse(
            "From: a@b.c (A)\r\nNewsgroups: x,y,\nno colon here\nLines: 7x\nSubject:  s \n\nbody one\rbody two\n",
        );
        assert_eq!(m.from.as_deref(), Some("a@b.c (A)"));
        assert_eq!(m.newsgroups, Some(vec!["x".to_string(), "y".to_string()]));
        assert_eq!(m.subject.as_deref(), Some("s"));
        assert_eq!(m.lines, 0, "not a number");
        assert_eq!(m.body.as_deref(), Some(" body one body two "));
        assert_eq!(Message::parse("Lines: -3\n\n").lines, -3);
        assert_eq!(Message::parse("Lines: 99999999999\n\n").lines, 0);
    }

    #[test]
    fn a_header_without_end_has_an_empty_body() {
        let m = Message::parse("Message-ID: <1>\nDate: d\nOrganization: o");
        assert_eq!(m.message_id.as_deref(), Some("<1>"));
        assert_eq!(m.date.as_deref(), Some("d"));
        assert_eq!(m.organization.as_deref(), Some("o"));
        assert_eq!(m.body.as_deref(), Some(""));
        assert_eq!(Message::parse("").body.as_deref(), Some(""));
    }

    #[test]
    fn splits_are_javas() {
        assert_eq!(java_split_comma(""), vec![String::new()]);
        assert_eq!(java_split_comma("a,,"), vec!["a".to_string()]);
        assert!(java_split_comma(",,").is_empty());
    }

    #[test]
    fn the_document_holds_every_present_field() {
        let m = Message::parse("From: f\nNewsgroups: g\n\nb\n");
        let doc = m.to_lucene_doc();
        assert_eq!(doc.get("from").as_deref(), Some("f"));
        assert_eq!(doc.get("newsgroup").as_deref(), Some("g"));
        assert_eq!(doc.get("lines_raw").as_deref(), Some("0"));
        assert_eq!(doc.get("body").as_deref(), Some(" b "));
        assert!(doc.get("subject").is_none());
        let full =
            Message::parse("From: f\nSubject: s\nMessage-ID: m\nDate: d\nOrganization: o\n\nb\n")
                .to_lucene_doc();
        for f in ["subject", "messageId", "date_raw", "organization"] {
            assert!(full.get(f).is_some(), "{f}");
        }
    }

    #[test]
    fn a_missing_root_is_an_error_and_files_are_read_by_path() {
        let dir = std::env::temp_dir().join(format!("luke-news-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        assert!(MessageFilesParser::new(&dir).parse_all().is_err());
        std::fs::create_dir_all(dir.join("g")).unwrap();
        std::fs::write(dir.join("g").join("2"), "Subject: two\n\nx\n").unwrap();
        std::fs::write(dir.join("1"), "Subject: one\n\nx\n").unwrap();
        std::fs::write(dir.join("bad"), [0xff, 0xfe]).unwrap();
        let messages = MessageFilesParser::new(&dir).parse_all().unwrap();
        let subjects: Vec<_> = messages
            .iter()
            .map(|m| m.subject.clone().unwrap())
            .collect();
        assert_eq!(subjects, ["one", "two"]);
        let single = MessageFilesParser::new(&dir.join("1")).parse_all().unwrap();
        assert_eq!(single.len(), 1);
        let _ = std::fs::remove_dir_all(&dir);
        let _ = Message::create_lucene_analyzer();
    }
}
