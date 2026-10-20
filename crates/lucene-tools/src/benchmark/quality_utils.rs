//! `org.apache.lucene.benchmark.quality.utils`: a document's name, the
//! simple quality-query parser, the TREC submission report, and
//! `QualityQueriesFinder`, which suggests queries from an index's terms.
//!
//! ```text
//! lucene-tools QualityQueriesFinder <index-dir>
//! ```

use std::io::Write;
use std::path::Path;
use std::sync::Arc;

use lucene_analysis::standard::StandardAnalyzer;
use lucene_analysis::Analyzer;
use lucene_codecs::stored_fields::FieldValue;
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::multi_terms::MultiTerms;
use lucene_search::query_visitor::Occur;
use lucene_search::queryparser::classic::QueryParser;
use lucene_search::queryparser::{ParsedBoolean, ParsedQuery};
use lucene_store::directory::FsDirectory;

use super::quality::{number_format, pad, QualityQuery, QualityQueryParser, QualitySearcher};
use crate::cli::{exception, Error, Io, ToolResult};

/// `DocNameExtractor`: a document's name, the first string value of a
/// stored field.
pub struct DocNameExtractor {
    doc_name_field: String,
}

impl DocNameExtractor {
    pub fn new(doc_name_field: &str) -> Self {
        DocNameExtractor {
            doc_name_field: doc_name_field.to_string(),
        }
    }

    /// `docName(storedFields, docid)`: `None` (Java's `null`) without one.
    ///
    /// # Errors
    /// The stored fields cannot be read.
    pub fn doc_name(&self, reader: &DirectoryReader, doc: i32) -> Result<Option<String>, Error> {
        let Some(seg) = reader
            .segment_readers()
            .iter()
            .find(|s| s.doc_base <= doc && doc - s.doc_base < s.max_doc)
        else {
            return Ok(None);
        };
        let Some(number) = seg
            .field_infos()
            .fields
            .iter()
            .find(|f| f.name == self.doc_name_field)
            .map(|f| f.number)
        else {
            return Ok(None);
        };
        let Some(document) = seg.stored_document(doc - seg.doc_base)? else {
            return Ok(None);
        };
        // `stringField` is the only value the visitor keeps
        Ok(document
            .fields
            .into_iter()
            .filter(|f| f.field_number == number)
            .find_map(|f| match f.value {
                FieldValue::String(s) => Some(s),
                _ => None,
            }))
    }
}

/// `SimpleQQParser`: each named part of a quality query, escaped and parsed
/// over one field with a `StandardAnalyzer`, as a `SHOULD` clause.
pub struct SimpleQQParser {
    qq_names: Vec<String>,
    parser: QueryParser,
}

impl SimpleQQParser {
    pub fn new(qq_names: &[&str], index_field: &str) -> Self {
        let analyzer: Arc<Analyzer> = Arc::new(Analyzer::new(StandardAnalyzer::new()));
        SimpleQQParser {
            qq_names: qq_names.iter().map(|s| s.to_string()).collect(),
            parser: QueryParser::new(index_field, analyzer),
        }
    }
}

impl QualityQueryParser for SimpleQQParser {
    fn parse(&self, query: &QualityQuery) -> Result<ParsedQuery, Error> {
        let mut clauses = Vec::with_capacity(self.qq_names.len());
        for name in &self.qq_names {
            // `QueryParserBase.escape(null)`
            let value = query
                .value(name)
                .ok_or_else(|| exception("java.lang.NullPointerException"))?;
            let parsed = self
                .parser
                .parse(&QueryParser::escape(value))
                .map_err(|e| {
                    let class = match e.class {
                        "ParseException" => "org.apache.lucene.queryparser.classic.ParseException",
                        "TooManyClauses" => "org.apache.lucene.search.IndexSearcher$TooManyClauses",
                        other => return exception(format!("java.lang.{other}: {}", e.message)),
                    };
                    exception(format!("{class}: {}", e.message))
                })?;
            clauses.push((Occur::Should, parsed));
        }
        Ok(ParsedQuery::Boolean(ParsedBoolean {
            clauses,
            minimum_should_match: 0,
        }))
    }
}

/// `SubmissionReport`: each result as a TREC submission line,
/// `qid Q0 docname rank score name`.
pub struct SubmissionReport {
    logger: Option<Box<dyn Write>>,
    name: String,
}

impl SubmissionReport {
    pub fn new(logger: Option<Box<dyn Write>>, name: &str) -> Self {
        SubmissionReport {
            logger,
            name: name.to_string(),
        }
    }

    /// `report(qq, td, docNameField, searcher)`.
    ///
    /// # Errors
    /// The stored fields cannot be read or the report written.
    pub fn report(
        &mut self,
        qq: &QualityQuery,
        hits: &[(i32, f32)],
        doc_name_field: &str,
        searcher: &QualitySearcher<'_, '_, '_>,
    ) -> Result<(), Error> {
        let Some(logger) = self.logger.as_mut() else {
            return Ok(());
        };
        let sep = " \t ";
        let xt = DocNameExtractor::new(doc_name_field);
        for (i, &(doc, score)) in hits.iter().enumerate() {
            let name = xt.doc_name(searcher.reader, doc)?;
            writeln!(
                logger,
                "{}{sep}Q0{sep}{}{sep}{}{sep}{}{sep}{}",
                qq.query_id(),
                pad(name.as_deref().unwrap_or(""), 20),
                pad(&i.to_string(), 7),
                // the `float` score widened to `double`
                number_format(f64::from(score), 4, true),
                self.name
            )?;
        }
        Ok(())
    }

    /// `flush()`.
    ///
    /// # Errors
    /// The report cannot be written.
    pub fn flush(&mut self) -> Result<(), Error> {
        if let Some(logger) = self.logger.as_mut() {
            logger.flush()?;
        }
        Ok(())
    }
}

/// `org.apache.lucene.util.PriorityQueue` of `TermDf`s, least `df` on top:
/// the same heap, so equal frequencies pop in Java's order.
struct TermsDfQueue {
    /// `heap[0]` unused, as Java's.
    heap: Vec<(String, i32)>,
    max_size: usize,
}

impl TermsDfQueue {
    fn new(max_size: usize) -> Self {
        TermsDfQueue {
            heap: vec![(String::new(), 0)],
            max_size,
        }
    }

    fn size(&self) -> usize {
        self.heap.len() - 1
    }

    fn less_than(a: &(String, i32), b: &(String, i32)) -> bool {
        a.1 < b.1
    }

    fn insert_with_overflow(&mut self, element: (String, i32)) {
        if self.size() < self.max_size {
            self.heap.push(element);
            self.up_heap(self.size());
        } else if self.size() > 0 && Self::less_than(&self.heap[1], &element) {
            self.heap[1] = element;
            self.down_heap(1);
        }
    }

    fn pop(&mut self) -> Option<(String, i32)> {
        if self.size() == 0 {
            return None;
        }
        let last = self.heap.pop()?;
        if self.size() == 0 {
            return Some(last);
        }
        let top = std::mem::replace(&mut self.heap[1], last);
        self.down_heap(1);
        Some(top)
    }

    fn up_heap(&mut self, mut i: usize) {
        let mut j = i >> 1;
        while j > 0 && Self::less_than(&self.heap[i], &self.heap[j]) {
            self.heap.swap(i, j);
            i = j;
            j >>= 1;
        }
    }

    fn down_heap(&mut self, mut i: usize) {
        let size = self.size();
        loop {
            let mut j = i << 1;
            let k = j + 1;
            if k <= size && Self::less_than(&self.heap[k], &self.heap[j]) {
                j = k;
            }
            if j <= size && Self::less_than(&self.heap[j], &self.heap[i]) {
                self.heap.swap(i, j);
                i = j;
            } else {
                break;
            }
        }
    }
}

/// `bestTerms(field, numTerms)`: the `num_terms` most frequent terms below
/// a tenth of the documents, least frequent first, each printed as it pops.
fn best_terms(
    reader: &DirectoryReader,
    field: &str,
    num_terms: usize,
    out: &mut dyn Write,
) -> Result<Vec<String>, Error> {
    let mut pq = TermsDfQueue::new(num_terms);
    let threshold = reader.max_doc() / 10;
    if let Some(terms) = MultiTerms::get_terms(reader, field)? {
        let mut te = terms.iterator()?;
        while let Some(term) = te.next()? {
            let term = term.to_vec();
            let df = te.doc_freq()?;
            if df < threshold {
                pq.insert_with_overflow((String::from_utf8_lossy(&term).into_owned(), df));
            }
        }
    }
    let mut res = Vec::with_capacity(pq.size());
    while let Some((word, df)) = pq.pop() {
        res.push(word);
        writeln!(out, "{}.   word:  {df}   {}", res.len(), res[res.len() - 1])?;
    }
    Ok(res)
}

/// `bestQueries(field, numQueries)`: four of the best terms per query, from
/// both ends of the quarters.
fn best_queries(
    reader: &DirectoryReader,
    num_queries: usize,
    out: &mut dyn Write,
) -> Result<Vec<String>, Error> {
    let words = best_terms(reader, "body", num_queries.saturating_mul(4), out)?;
    let n = words.len();
    let m = n / 4;
    Ok((0..m)
        .map(|i| {
            format!(
                "{} {}  {} {}",
                words[i],
                words[m + i],
                words[n - 1 - m - i],
                words[n - 1 - i]
            )
        })
        .collect())
}

/// `formatQueryAsTrecTopic(qnum, title, null, null)`.
fn format_query_as_trec_topic(qnum: usize, title: &str) -> String {
    format!(
        "<top>\n<num> Number: {qnum}\n\n<title> {title}\n\n<desc> Description:\n\n\n<narr> Narrative:\n\n\n</top>"
    )
}

/// `QualityQueriesFinder.main`: twenty TREC topics made of the index's
/// `body` terms.
///
/// # Errors
/// The index cannot be read.
pub fn quality_queries_finder(args: &[String], io: &mut Io<'_>) -> ToolResult {
    let Some(index) = args.first() else {
        writeln!(io.err, "Usage: java QualityQueriesFinder <index-dir>")?;
        return Ok(1);
    };
    let dir = FsDirectory::open(Path::new(index));
    let reader = DirectoryReader::open(&dir)?;
    let queries = best_queries(&reader, 20, io.out)?;
    for (i, q) in queries.iter().enumerate() {
        writeln!(io.out, "\n{}", format_query_as_trec_topic(i, q))?;
    }
    Ok(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_queue_pops_as_lucenes_does() {
        let mut pq = TermsDfQueue::new(3);
        for (w, df) in [("a", 5), ("b", 1), ("c", 3), ("d", 3), ("e", 0), ("f", 9)] {
            pq.insert_with_overflow((w.to_string(), df));
        }
        let popped: Vec<(String, i32)> = std::iter::from_fn(|| pq.pop()).collect();
        let words: Vec<&str> = popped.iter().map(|(w, _)| w.as_str()).collect();
        // `d` displaced `b`, `f` displaced `d`: `c`, tied with `d`, stayed
        assert_eq!(words, ["c", "a", "f"]);
        assert!(pq.pop().is_none());
        let mut empty = TermsDfQueue::new(0);
        empty.insert_with_overflow(("x".to_string(), 1));
        assert_eq!(empty.size(), 0);
    }

    #[test]
    fn a_topic_is_formatted_as_trec_reads_it() {
        let topic = format_query_as_trec_topic(3, "a b  c d");
        let qs = super::super::trec::read_trec_topics(&topic).unwrap();
        assert_eq!(qs[0].query_id(), "3");
        assert_eq!(qs[0].value("title"), Some("a b  c d"));
    }

    #[test]
    fn no_arguments_prints_the_usage() {
        let (mut out, mut err) = (Vec::new(), Vec::new());
        let mut io = Io {
            out: &mut out,
            err: &mut err,
        };
        assert_eq!(quality_queries_finder(&[], &mut io).unwrap(), 1);
        assert_eq!(
            String::from_utf8(err).unwrap(),
            "Usage: java QualityQueriesFinder <index-dir>\n"
        );
    }

    #[test]
    fn a_missing_part_is_javas_null() {
        let parser = SimpleQQParser::new(&["title"], "body");
        let q = QualityQuery::new("1", Vec::new());
        assert!(parser.parse(&q).is_err());
        let q = QualityQuery::new("1", vec![("title".to_string(), "a OR (b".to_string())]);
        // `escape` leaves the `OR` operator alone
        assert_eq!(parser.parse(&q).unwrap().to_string(""), "(body:a body:b)");
    }
}
