//! `org.apache.lucene.benchmark.quality.trec`: TREC topics, 1MQ topics and
//! relevance judgements (qrels), and `QueryDriver`, which runs topics
//! against an index and judges the results.
//!
//! ```text
//! lucene-tools QueryDriver <topicsFile> <qrelsFile> <submissionFile> <indexDir> [querySpec]
//! ```

use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;
use std::path::Path;

use lucene_search::directory_reader::DirectoryReader;
use lucene_search::field_norms::FieldNorms;
use lucene_search::index_searcher::IndexSearcher;
use lucene_store::directory::FsDirectory;

use super::quality::{
    sort_queries, Judge, QualityBenchmark, QualityQuery, QualitySearcher, QualityStats,
};
use super::quality_utils::{SimpleQQParser, SubmissionReport};
use crate::cli::{exception, java_lines, java_trim, Error, Io, ToolResult};

/// `TrecJudge`: the qrels' relevant documents of each query.
#[derive(Debug, Clone, Default)]
pub struct TrecJudge {
    /// Query ids in the order Java's `HashMap` first saw them.
    order: Vec<String>,
    judgements: HashMap<String, HashSet<String>>,
}

impl TrecJudge {
    /// `new TrecJudge(reader)`: lines of `qnum 0 doc-name is-relevant`,
    /// empty and `#` lines skipped, only the relevant (`is-relevant` other
    /// than `0`) kept.
    ///
    /// # Errors
    /// Java's `NoSuchElementException` for a line of fewer than four
    /// tokens.
    pub fn read(text: &str) -> Result<Self, Error> {
        let mut judge = TrecJudge::default();
        for line in java_lines(text) {
            let line = java_trim(line);
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            // `StringTokenizer`'s default delimiters
            let mut st = line
                .split([' ', '\t', '\n', '\r', '\u{c}'])
                .filter(|t| !t.is_empty());
            let mut next = || {
                st.next()
                    .ok_or_else(|| exception("java.util.NoSuchElementException"))
            };
            let query_id = next()?.to_string();
            next()?;
            let doc_name = next()?.to_string();
            let relevant = next()? != "0";
            if relevant {
                if !judge.judgements.contains_key(&query_id) {
                    judge.order.push(query_id.clone());
                }
                judge
                    .judgements
                    .entry(query_id)
                    .or_default()
                    .insert(doc_name);
            }
        }
        Ok(judge)
    }
}

impl Judge for TrecJudge {
    fn is_relevant(&self, doc_name: Option<&str>, query: &QualityQuery) -> bool {
        match (self.judgements.get(query.query_id()), doc_name) {
            (Some(docs), Some(name)) => docs.contains(name),
            // `containsKey(null)`: no document is named `null`
            _ => false,
        }
    }

    fn validate_data(&self, queries: &[QualityQuery], logger: Option<&mut String>) -> bool {
        let mut missing_queries: HashSet<&str> = self.order.iter().map(String::as_str).collect();
        let mut missing_judgements = Vec::new();
        for qq in queries {
            let id = qq.query_id();
            if !missing_queries.remove(id) {
                missing_judgements.push(id);
            }
        }
        let mut valid = true;
        let mut logger = logger;
        if !missing_judgements.is_empty() {
            valid = false;
            if let Some(log) = logger.as_deref_mut() {
                let _ = writeln!(
                    log,
                    "WARNING: {} queries have no judgments! - ",
                    missing_judgements.len()
                );
                for id in &missing_judgements {
                    let _ = writeln!(log, "   {id}");
                }
            }
        }
        if !missing_queries.is_empty() {
            valid = false;
            if let Some(log) = logger {
                let _ = writeln!(
                    log,
                    "WARNING: {} judgments match no query! - ",
                    missing_queries.len()
                );
                for id in copy_order(&self.order, &missing_queries) {
                    let _ = writeln!(log, "   {id}");
                }
            }
        }
        valid
    }

    fn max_recall(&self, query: &QualityQuery) -> i32 {
        self.judgements
            .get(query.query_id())
            .map_or(0, |docs| i32::try_from(docs.len()).unwrap_or(i32::MAX))
    }
}

/// `String.hashCode()`.
fn java_string_hash(s: &str) -> i32 {
    s.encode_utf16()
        .fold(0i32, |h, c| h.wrapping_mul(31).wrapping_add(i32::from(c)))
}

/// `HashMap`'s bucket of `key` in a table of `capacity`.
fn bucket(key: &str, capacity: u32) -> u32 {
    let h = java_string_hash(key) as u32;
    (h ^ (h >> 16)) & (capacity - 1)
}

/// The order `new HashMap<>(judgements)`, with every query found removed,
/// iterates the rest: the copy is sized for all of `judgements`
/// (`tableSizeFor(ceil(size / 0.75))`) and filled in the order the
/// original iterates (a table grown from 16), and removal keeps the order
/// of what is left.
///
/// Blind spot, as `lucene_search::mlt::java_hash_order`'s: a bucket of
/// nine or more keys (`treeifyBin`) is not modelled.
fn copy_order<'k>(inserted: &'k [String], left: &HashSet<&str>) -> Vec<&'k str> {
    let n = inserted.len();
    let mut source_cap: u32 = 16;
    while n > (source_cap / 4 * 3) as usize {
        source_cap = source_cap.saturating_mul(2);
    }
    let mut keys: Vec<&str> = inserted.iter().map(String::as_str).collect();
    // `sort_by_key` is stable: insertion order within a bucket
    keys.sort_by_key(|k| bucket(k, source_cap));
    // `tableSizeFor((int) Math.ceil(s / (double) 0.75f))`
    let copy_cap = ((n as f64 / 0.75).ceil() as u32).max(1).next_power_of_two();
    keys.sort_by_key(|k| bucket(k, copy_cap));
    keys.retain(|k| left.contains(k));
    keys
}

/// `TrecTopicsReader.readQueries`: `<top>` blocks of `<num>`, `<title>`,
/// `<desc>` and `<narr>`, sorted by id.
///
/// # Errors
/// Java's `NullPointerException` for a topic cut short before its
/// `<num>` or `<title>`.
pub fn read_trec_topics(text: &str) -> Result<Vec<QualityQuery>, Error> {
    let mut lines = java_lines(text).collect::<Vec<&str>>().into_iter();
    let mut res = Vec::new();
    // `read(reader, prefix, ...)`: up to a line starting with `prefix`
    let read = |lines: &mut std::vec::IntoIter<&str>, prefix: &str| -> Option<String> {
        lines.find(|l| l.starts_with(prefix)).map(|l| l.to_string())
    };
    let npe = || exception("java.lang.NullPointerException");
    while read(&mut lines, "<top>").is_some() {
        let num = read(&mut lines, "<num>").ok_or_else(npe)?;
        let id = java_trim(num.split_once(':').map_or(num.as_str(), |(_, r)| r)).to_string();
        let title = read(&mut lines, "<title>").ok_or_else(npe)?;
        let title = java_trim(title.split_once('>').map_or(title.as_str(), |(_, r)| r)).to_string();
        read(&mut lines, "<desc>");
        let mut description = String::new();
        for line in lines.by_ref() {
            if line.starts_with("<narr>") {
                break;
            }
            if !description.is_empty() {
                description.push(' ');
            }
            description.push_str(line);
        }
        let mut narrative = String::new();
        for line in lines.by_ref() {
            if line.starts_with("</top>") {
                break;
            }
            if !narrative.is_empty() {
                narrative.push(' ');
            }
            narrative.push_str(line);
        }
        res.push(QualityQuery::new(
            id,
            vec![
                ("title".to_string(), title),
                (
                    "description".to_string(),
                    java_trim(&description).to_string(),
                ),
                ("narrative".to_string(), java_trim(&narrative).to_string()),
            ],
        ));
    }
    sort_queries(&mut res);
    Ok(res)
}

/// `Trec1MQReader(name).readQueries`: `id:text` lines (`#` lines skipped),
/// the text under `name`, sorted by id.
///
/// # Errors
/// Java's `StringIndexOutOfBoundsException` for a line without `:`.
pub fn read_trec_1mq(text: &str, name: &str) -> Result<Vec<QualityQuery>, Error> {
    let mut res = Vec::new();
    for line in java_lines(text) {
        let line = java_trim(line);
        if line.starts_with('#') {
            continue;
        }
        let Some((id, qtext)) = line.split_once(':') else {
            return Err(exception(format!(
                "java.lang.StringIndexOutOfBoundsException: Range [0, -1) out of bounds for length {}",
                line.encode_utf16().count()
            )));
        };
        res.push(QualityQuery::new(
            java_trim(id),
            vec![(name.to_string(), java_trim(qtext).to_string())],
        ));
    }
    sort_queries(&mut res);
    Ok(res)
}

/// The usage `QueryDriver` prints to `System.err`.
pub const USAGE: &str = "Usage: QueryDriver <topicsFile> <qrelsFile> <submissionFile> <indexDir> [querySpec]
topicsFile: input file containing queries
qrelsFile: input file containing relevance judgements
submissionFile: output submission file for trec_eval
indexDir: index directory
querySpec: string composed of fields to use in query consisting of T=title,D=description,N=narrative:
\texample: TD (query on Title + Description). The default is T (title only)
";

/// `QueryDriver.main`: the topics searched over the index's `body`, each
/// judged (logged to standard output), their average, and a TREC
/// submission of every result.
///
/// # Errors
/// A file cannot be read or written, the index cannot be opened, or a
/// query cannot be parsed.
pub fn query_driver(args: &[String], io: &mut Io<'_>) -> ToolResult {
    if args.len() < 4 || args.len() > 5 {
        write!(io.err, "{USAGE}")?;
        return Ok(1);
    }
    let read = |p: &str| -> Result<String, Error> { Ok(std::fs::read_to_string(p)?) };
    let submission = std::io::BufWriter::new(std::fs::File::create(&args[2])?);
    let mut submit = SubmissionReport::new(Some(Box::new(submission)), "lucene");
    let field_spec = args.get(4).map_or("T", String::as_str);
    let dir = FsDirectory::open(Path::new(&args[3]));
    let reader = DirectoryReader::open(&dir)?;
    let opened = reader.open_segments()?;
    let segments = opened.as_open_segments();
    let mut fields: Vec<String> = reader
        .segment_readers()
        .iter()
        .flat_map(|s| s.field_infos().fields.iter().map(|f| f.name.clone()))
        .collect();
    fields.sort();
    fields.dedup();
    let owned: Vec<HashMap<String, FieldNorms<'_>>> = reader.field_norms_by_field(&fields)?;
    let norms: Vec<Option<&HashMap<String, FieldNorms<'_>>>> = owned.iter().map(Some).collect();
    let searcher = IndexSearcher::new(&segments, &norms)?;
    let searcher = QualitySearcher {
        reader: &reader,
        searcher: &searcher,
    };

    let queries = read_trec_topics(&read(&args[0])?)?;
    let judge = TrecJudge::read(&read(&args[1])?)?;
    // the logger prints, and flushes, each line as it comes
    let mut log = String::new();
    judge.validate_data(&queries, Some(&mut log));
    flush(&mut log, io)?;

    // `new HashSet<>()` of the parts asked for, in its iteration order
    let mut names = Vec::new();
    for (c, name) in [('T', "title"), ('D', "description"), ('N', "narrative")] {
        if field_spec.contains(c) {
            names.push(name);
        }
    }
    let names: Vec<&str> = lucene_search::mlt::java_hash_order(&names)
        .into_iter()
        .map(|i| names[i])
        .collect();
    let parser = SimpleQQParser::new(&names, "body");
    let mut run = QualityBenchmark::new(&queries, &parser, "docname");
    run.set_max_results(1000);
    let result = run.execute(&searcher, Some(&judge), Some(&mut submit), Some(&mut log));
    flush(&mut log, io)?;
    let stats: Vec<QualityStats> = result?.into_iter().flatten().collect();
    let avg = QualityStats::average(&stats)?;
    let result = avg.log(Some("SUMMARY"), 2, &mut log, "  ");
    flush(&mut log, io)?;
    result?;
    Ok(0)
}

/// What the autoflushing logger printed so far, to standard output.
fn flush(log: &mut String, io: &mut Io<'_>) -> Result<(), Error> {
    io.out.write_all(log.as_bytes())?;
    log.clear();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn q(id: &str) -> QualityQuery {
        QualityQuery::new(id, Vec::new())
    }

    #[test]
    fn qrels_keep_the_relevant_documents() {
        let judge = TrecJudge::read("# c\n\n1 0 a 1\n1 0 b 0\r\n 2\t0 c 2 \r3 0 d 0\n").unwrap();
        assert!(judge.is_relevant(Some("a"), &q("1")));
        assert!(!judge.is_relevant(Some("b"), &q("1")));
        assert!(!judge.is_relevant(None, &q("1")));
        assert!(!judge.is_relevant(Some("a"), &q("9")));
        assert_eq!(judge.max_recall(&q("2")), 1);
        assert_eq!(judge.max_recall(&q("3")), 0);
        let mut log = String::new();
        assert!(judge.validate_data(&[q("1"), q("2")], Some(&mut log)));
        assert!(log.is_empty());
        assert!(!judge.validate_data(&[q("1"), q("4")], Some(&mut log)));
        assert_eq!(
            log,
            "WARNING: 1 queries have no judgments! - \n   4\nWARNING: 1 judgments match no query! - \n   2\n"
        );
        assert!(!judge.validate_data(&[], None));
        let e = TrecJudge::read("1 0 a\n").unwrap_err().to_string();
        assert!(e.contains("java.util.NoSuchElementException"), "{e}");
    }

    #[test]
    fn a_copy_iterates_in_its_own_table() {
        // three keys: the copy's table has four buckets, the original 16
        let ids: Vec<String> = ["Aa", "17", "1"].iter().map(|s| s.to_string()).collect();
        let all: HashSet<&str> = ids.iter().map(String::as_str).collect();
        let order = copy_order(&ids, &all);
        let mut expected: Vec<&str> = ids.iter().map(String::as_str).collect();
        expected.sort_by_key(|k| bucket(k, 16));
        expected.sort_by_key(|k| bucket(k, 4));
        assert_eq!(order, expected);
        assert_eq!(copy_order(&[], &HashSet::new()), Vec::<&str>::new());
    }

    #[test]
    fn topics_read_as_trec_writes_them() {
        let text = "<top>\n<num> Number: 10\n<title> second\n\n<desc> Description:\nd1\nd2\n<narr> Narrative:\nn1\n</top>\n\
                    <top>\n<num> Number: 9\n<title> first\n<desc>\n<narr>\n</top>\n";
        let qs = read_trec_topics(text).unwrap();
        assert_eq!(qs.len(), 2);
        assert_eq!(qs[0].query_id(), "9");
        assert_eq!(qs[1].value("title"), Some("second"));
        assert_eq!(qs[1].value("description"), Some("d1 d2"));
        assert_eq!(qs[1].value("narrative"), Some("n1"));
        assert_eq!(qs[0].value("description"), Some(""));
        assert!(read_trec_topics("<top>\n").is_err());
        assert!(read_trec_topics("<top>\n<num> 1\n").is_err());
        assert!(read_trec_topics("").unwrap().is_empty());
    }

    #[test]
    fn one_million_query_lines() {
        let qs = read_trec_1mq("# c\n2: b c \n1:a\n", "title").unwrap();
        assert_eq!(qs[0].query_id(), "1");
        assert_eq!(qs[1].value("title"), Some("b c"));
        let e = read_trec_1mq("x\n", "title").unwrap_err().to_string();
        assert!(e.contains("length 1"), "{e}");
    }

    #[test]
    fn wrong_arity_prints_the_usage() {
        let (mut out, mut err) = (Vec::new(), Vec::new());
        let mut io = Io {
            out: &mut out,
            err: &mut err,
        };
        assert_eq!(query_driver(&["a".to_string()], &mut io).unwrap(), 1);
        assert_eq!(String::from_utf8(err).unwrap(), USAGE);
    }
}
