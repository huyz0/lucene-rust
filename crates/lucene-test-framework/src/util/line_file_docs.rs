//! `LineFileDocs`: documents read line by line from a Europarl or Wikipedia
//! line file (`title \t date \t body`), starting at a random position and
//! rewinding at the end.
//!
//! Lucene's default is `europarl.lines.txt.gz` from the test-framework jar
//! (9.7 MB). This crate commits a 200-line excerpt of it instead
//! (`data/europarl.lines.excerpt.txt.gz`: four gzip members of 50 lines and
//! the `.seek` file of their offsets, the layout
//! `dev-tools/scripts/create_line_file_docs.py` gives the full file), and
//! `TESTS_LINEDOCSFILE` points at any other line file -- the full Europarl
//! one included -- as `-Dtests.linedocsfile` does.
//!
//! Opening draws what Java draws from the same [`JavaRandom`]: one
//! `nextLong` to seed the instance's own stream, then either a skip point
//! from the `.seek` file (`.gz`) or a byte offset into the file (plain
//! text, after which the partial line is skipped). Each [`LineFileDocs::next_doc`]
//! draws the page-view count and the sparse-field dice Java rolls.

use std::io;
use std::path::{Path, PathBuf};

use lucene_util::java_random::JavaRandom;

/// One line's document: the fields Java's `DocState` fills.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LineDoc {
    /// `title` (also `titleTokenized`).
    pub title: String,
    /// `date`, as the line spells it.
    pub date: String,
    /// `body`.
    pub body: String,
    /// `docid` / `docid_int`: the running count since open or reset.
    pub id: i32,
    /// `page_views`: `random.nextInt(10_000)`.
    pub page_views: i64,
}

/// The committed excerpt of Lucene's `europarl.lines.txt.gz`.
pub fn default_excerpt() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("data/europarl.lines.excerpt.txt.gz")
}

/// `LuceneTestCase.TEST_LINE_DOCS_FILE`: `TESTS_LINEDOCSFILE`, else the
/// committed excerpt.
pub fn test_line_docs_file() -> PathBuf {
    std::env::var_os("TESTS_LINEDOCSFILE").map_or_else(default_excerpt, PathBuf::from)
}

/// `LineFileDocs` over one file.
#[derive(Debug)]
pub struct LineFileDocs {
    path: PathBuf,
    random: JavaRandom,
    lines: Vec<String>,
    pos: usize,
    id: i32,
}

impl LineFileDocs {
    /// `new LineFileDocs(random)`: over [`test_line_docs_file`].
    pub fn new(random: &mut JavaRandom) -> io::Result<Self> {
        Self::with_path(random, &test_line_docs_file())
    }

    /// `new LineFileDocs(random, path)`.
    pub fn with_path(random: &mut JavaRandom, path: &Path) -> io::Result<Self> {
        let mut docs = Self {
            path: path.to_path_buf(),
            random: JavaRandom::new(random.next_long()),
            lines: Vec::new(),
            pos: 0,
            id: 0,
        };
        docs.open()?;
        Ok(docs)
    }

    /// `randomSeekPos`: no draw at all for a file of three bytes or fewer.
    fn random_seek_pos(&mut self, size: u64) -> u64 {
        if size <= 3 {
            0
        } else {
            ((self.random.next_long() & i64::MAX) as u64) % (size / 3)
        }
    }

    fn open(&mut self) -> io::Result<()> {
        let bytes = std::fs::read(&self.path)?;
        let text = if self.path.extension().is_some_and(|e| e == "gz") {
            // "foo.txt.gz" -> "foo.txt.seek".
            let seek_path = self.path.with_extension("seek");
            let seek = std::fs::read_to_string(&seek_path)?;
            let mut skip_points = vec![0u64];
            for line in seek.lines() {
                skip_points.push(
                    line.trim()
                        .parse()
                        .map_err(|e| invalid(format!("{seek_path:?}: {e}")))?,
                );
            }
            let n = i32::try_from(skip_points.len()).map_err(|e| invalid(e.to_string()))?;
            let seek_to = skip_points[self.random.next_int_bounded(n) as usize];
            let from = usize::try_from(seek_to).ok().filter(|&s| s <= bytes.len());
            gunzip(
                &bytes[from
                    .ok_or_else(|| invalid(format!("skip point {seek_to} past the end")))?..],
            )?
        } else {
            let seek_to = self.random_seek_pos(bytes.len() as u64) as usize;
            let mut rest = &bytes[seek_to..];
            if seek_to > 0 {
                // Read up to and including the next CR or LF: the seek may
                // have landed inside a multi-byte character.
                let skip = rest
                    .iter()
                    .position(|&b| b == b'\r' || b == b'\n')
                    .map_or(rest.len(), |p| p + 1);
                rest = &rest[skip..];
            }
            rest.to_vec()
        };
        let text = String::from_utf8(text).map_err(|e| invalid(format!("{:?}: {e}", self.path)))?;
        self.lines = read_lines(&text);
        self.pos = 0;
        Ok(())
    }

    /// `reset()`: reopen (drawing a new start) and restart the ids.
    pub fn reset(&mut self) -> io::Result<()> {
        self.open()?;
        self.id = 0;
        Ok(())
    }

    /// `nextDoc()`: the next line's document, rewinding (and redrawing the
    /// start) at the end of the file.
    ///
    /// Java builds a second document with up to three sparse fields one time
    /// in five, then returns the first; the dice are rolled here too, so the
    /// stream stays in step, and nothing else is kept.
    pub fn next_doc(&mut self) -> io::Result<LineDoc> {
        if self.pos == self.lines.len() {
            self.open()?;
            if self.lines.is_empty() {
                return Err(invalid(format!("{:?} has no lines", self.path)));
            }
        }
        let line = &self.lines[self.pos];
        self.pos += 1;
        let bad = || invalid(format!("line: [{line}] is in an invalid format !"));
        let spot = line.find('\t').ok_or_else(bad)?;
        let spot2 = line[spot + 1..]
            .find('\t')
            .map(|p| p + spot + 1)
            .ok_or_else(bad)?;
        let doc = LineDoc {
            title: line[..spot].to_string(),
            date: line[spot + 1..spot2].to_string(),
            body: line[spot2 + 1..].to_string(),
            id: self.id,
            page_views: i64::from(self.random.next_int_bounded(10_000)),
        };
        self.id += 1;
        if self.random.next_int_bounded(5) == 4 {
            for _ in 0..3 {
                if self.random.next_int_bounded(3) == 1 {
                    self.random.next_int_bounded(4);
                }
            }
        }
        Ok(doc)
    }
}

fn invalid(msg: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg)
}

/// `BufferedReader.readLine` over the whole text: lines end at `\n`, `\r`
/// or `\r\n`, and a final line needs no terminator.
fn read_lines(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut rest = text;
    while !rest.is_empty() {
        match rest.find(['\r', '\n']) {
            Some(p) => {
                out.push(rest[..p].to_string());
                let skip = if rest[p..].starts_with("\r\n") { 2 } else { 1 };
                rest = &rest[p + skip..];
            }
            None => {
                out.push(rest.to_string());
                rest = "";
            }
        }
    }
    out
}

/// `GZIPInputStream` over concatenated members: each RFC 1952 header (with
/// its optional fields) skipped, the raw DEFLATE stream inflated, the 8-byte
/// trailer stepped over, until the input runs out.
pub fn gunzip(gz: &[u8]) -> io::Result<Vec<u8>> {
    use miniz_oxide::inflate::stream::{inflate, InflateState};
    use miniz_oxide::{DataFormat, MZFlush, MZStatus};
    let truncated = || invalid("truncated gzip member".to_string());
    let mut out = Vec::new();
    let mut member = 0;
    while member < gz.len() {
        let header = gz.get(member..member + 10).ok_or_else(truncated)?;
        if header[..3] != [0x1f, 0x8b, 8] {
            return Err(invalid("not in GZIP format".to_string()));
        }
        let flags = header[3];
        let mut at = member + 10;
        if flags & 4 != 0 {
            let xlen = gz.get(at..at + 2).ok_or_else(truncated)?;
            at += 2 + usize::from(u16::from_le_bytes([xlen[0], xlen[1]]));
        }
        for bit in [8u8, 16] {
            if flags & bit != 0 {
                let end = gz
                    .get(at..)
                    .and_then(|s| s.iter().position(|&b| b == 0))
                    .ok_or_else(truncated)?;
                at += end + 1;
            }
        }
        if flags & 2 != 0 {
            at += 2;
        }
        let mut input = gz.get(at..).ok_or_else(truncated)?;
        let mut state = InflateState::new_boxed(DataFormat::Raw);
        let mut buf = vec![0u8; 1 << 16];
        loop {
            let r = inflate(&mut state, input, &mut buf, MZFlush::None);
            input = &input[r.bytes_consumed..];
            out.extend_from_slice(&buf[..r.bytes_written]);
            match r.status {
                Ok(MZStatus::StreamEnd) => break,
                Ok(_) if r.bytes_consumed == 0 && r.bytes_written == 0 => return Err(truncated()),
                Ok(_) => {}
                Err(e) => return Err(invalid(format!("inflate: {e:?}"))),
            }
        }
        member = gz.len() - input.len() + 8;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use lucene_util::test_support::TempDir;

    fn gzip_member(data: &[u8], flags: u8) -> Vec<u8> {
        let mut out = vec![0x1f, 0x8b, 8, flags, 0, 0, 0, 0, 0, 0xff];
        if flags & 4 != 0 {
            out.extend_from_slice(&[2, 0, b'x', b'y']);
        }
        if flags & 8 != 0 {
            out.extend_from_slice(b"name\0");
        }
        if flags & 16 != 0 {
            out.extend_from_slice(b"comment\0");
        }
        if flags & 2 != 0 {
            out.extend_from_slice(&[0, 0]);
        }
        out.extend(miniz_oxide::deflate::compress_to_vec(data, 6));
        out.extend_from_slice(&[0; 8]); // CRC and size: not checked here
        out
    }

    #[test]
    fn gunzip_reads_every_member_and_header_field() {
        let mut gz = gzip_member(b"a\tb\tc\n", 0);
        gz.extend(gzip_member(b"d\te\tf\r\n", 4 | 8 | 16 | 2));
        assert_eq!(gunzip(&gz).unwrap(), b"a\tb\tc\nd\te\tf\r\n");
        assert!(gunzip(b"nope-nope-nope").is_err());
        assert!(gunzip(&gz[..5]).is_err());
        assert!(gunzip(&gz[..14]).is_err());
        assert!(gunzip(&[0x1f, 0x8b, 8, 4, 0, 0, 0, 0, 0, 0, 1]).is_err());
        assert!(gunzip(&[0x1f, 0x8b, 8, 8, 0, 0, 0, 0, 0, 0, 1]).is_err());
        let mut bad = gzip_member(b"x", 0);
        bad[10] = 0xff;
        assert!(gunzip(&bad).is_err());
    }

    #[test]
    fn lines_split_like_read_line() {
        assert_eq!(read_lines("a\r\nb\rc\nd"), ["a", "b", "c", "d"]);
        assert_eq!(read_lines("a\n\nb\n"), ["a", "", "b"]);
        assert!(read_lines("").is_empty());
    }

    #[test]
    fn excerpt_reads_and_rewinds() {
        let mut r = JavaRandom::new(1);
        let mut docs = LineFileDocs::with_path(&mut r, &default_excerpt()).unwrap();
        for i in 0..450 {
            let d = docs.next_doc().unwrap();
            assert!(!d.title.is_empty() && d.date.len() == 10, "{d:?}");
            assert!((0..10_000).contains(&d.page_views));
            assert_eq!(d.id, i);
        }
        docs.reset().unwrap();
        assert_eq!(docs.next_doc().unwrap().id, 0);
        assert!(test_line_docs_file().exists());
        let _ = LineFileDocs::new(&mut r).unwrap();
    }

    #[test]
    fn plain_files_seek_then_skip_the_partial_line() {
        let dir = TempDir::new("line-file-docs");
        let path = dir.join("docs.txt");
        let text: String = (0..50)
            .map(|i| format!("t{i}\t2024-01-01\tbody {i}\n"))
            .collect();
        std::fs::write(&path, &text).unwrap();
        for seed in 0..20 {
            let mut docs = LineFileDocs::with_path(&mut JavaRandom::new(seed), &path).unwrap();
            let d = docs.next_doc().unwrap();
            assert_eq!(d.body, format!("body {}", &d.title[1..]));
        }
        // Three bytes or fewer: no seek, no draw.
        let tiny = dir.join("tiny.txt");
        std::fs::write(&tiny, "a\tb").unwrap();
        let mut docs = LineFileDocs::with_path(&mut JavaRandom::new(1), &tiny).unwrap();
        assert!(docs.next_doc().is_err());
        let one = dir.join("one.txt");
        std::fs::write(&one, "a\tb\tc").unwrap();
        let mut docs = LineFileDocs::with_path(&mut JavaRandom::new(1), &one).unwrap();
        assert_eq!(docs.next_doc().unwrap().body, "c");
        assert_eq!(docs.next_doc().unwrap().id, 1);
    }

    #[test]
    fn bad_inputs_are_errors() {
        let dir = TempDir::new("line-file-docs-bad");
        let empty = dir.join("empty.txt");
        std::fs::write(&empty, "").unwrap();
        let mut docs = LineFileDocs::with_path(&mut JavaRandom::new(1), &empty).unwrap();
        assert!(docs.next_doc().is_err());
        let bad_utf8 = dir.join("bad.txt");
        std::fs::write(&bad_utf8, [0xff, 0xfe]).unwrap();
        assert!(LineFileDocs::with_path(&mut JavaRandom::new(1), &bad_utf8).is_err());
        let gz = dir.join("x.txt.gz");
        std::fs::write(&gz, gzip_member(b"a\tb\tc\n", 0)).unwrap();
        assert!(
            LineFileDocs::with_path(&mut JavaRandom::new(1), &gz).is_err(),
            "no seek file"
        );
        std::fs::write(dir.join("x.txt.seek"), "zz\n").unwrap();
        assert!(LineFileDocs::with_path(&mut JavaRandom::new(1), &gz).is_err());
        std::fs::write(dir.join("x.txt.seek"), "99999\n99999\n99999\n").unwrap();
        let far = (0..10).any(|s| LineFileDocs::with_path(&mut JavaRandom::new(s), &gz).is_err());
        assert!(far, "a skip point past the end");
        let missing = dir.join("missing.txt");
        assert!(LineFileDocs::with_path(&mut JavaRandom::new(1), &missing).is_err());
    }
}
