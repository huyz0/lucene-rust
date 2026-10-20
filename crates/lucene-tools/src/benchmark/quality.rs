//! `org.apache.lucene.benchmark.quality`: search quality -- precision,
//! recall, mean reciprocal rank -- of a set of queries against relevance
//! judgements, and a TREC submission of their results.
//!
//! Differs: Java's `PrintWriter` loggers are `&mut String`s the caller
//! prints; a search is the port's [`IndexSearcher`] over a
//! [`DirectoryReader`] ([`QualitySearcher`]) rather than any searcher.

use std::fmt::Write as _;
use std::time::Instant;

use lucene_search::directory_reader::DirectoryReader;
use lucene_search::index_searcher::IndexSearcher;
use lucene_search::query::{BooleanQuery, Clause};
use lucene_search::queryparser::ParsedQuery;

use crate::cli::{exception, java_trim, Error};

/// `QualityQuery`: a query's id and its named parts (`title`,
/// `description`, ...).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QualityQuery {
    query_id: String,
    name_val_pairs: Vec<(String, String)>,
}

impl QualityQuery {
    pub fn new(query_id: impl Into<String>, name_val_pairs: Vec<(String, String)>) -> Self {
        QualityQuery {
            query_id: query_id.into(),
            name_val_pairs,
        }
    }

    /// `getNames()`.
    pub fn names(&self) -> Vec<&str> {
        self.name_val_pairs
            .iter()
            .map(|(k, _)| k.as_str())
            .collect()
    }

    /// `getValue(name)`.
    pub fn value(&self, name: &str) -> Option<&str> {
        self.name_val_pairs
            .iter()
            .find(|(k, _)| k == name)
            .map(|(_, v)| v.as_str())
    }

    /// `getQueryID()`.
    pub fn query_id(&self) -> &str {
        &self.query_id
    }

    /// `compareTo`: as `int`s when both ids are, else as strings (UTF-16).
    pub fn compare(&self, other: &QualityQuery) -> std::cmp::Ordering {
        match (
            crate::cli::parse_int(&self.query_id),
            crate::cli::parse_int(&other.query_id),
        ) {
            // `n - nOther`, which overflows as Java's does
            (Ok(n), Ok(m)) => n.wrapping_sub(m).cmp(&0),
            _ => self
                .query_id
                .encode_utf16()
                .cmp(other.query_id.encode_utf16()),
        }
    }
}

/// `Arrays.sort(qq)`.
pub fn sort_queries(queries: &mut [QualityQuery]) {
    queries.sort_by(QualityQuery::compare);
}

/// `Judge`: whether a document is relevant to a query.
pub trait Judge {
    /// `isRelevant(docName, query)`.
    fn is_relevant(&self, doc_name: Option<&str>, query: &QualityQuery) -> bool;
    /// `validateData(qq, logger)`: whether the queries and the judgements
    /// match, the mismatches logged.
    fn validate_data(&self, queries: &[QualityQuery], logger: Option<&mut String>) -> bool;
    /// `maxRecall(query)`: how many documents are relevant to the query.
    fn max_recall(&self, query: &QualityQuery) -> i32;
}

/// `QualityQueryParser`: a quality query as a Lucene query.
pub trait QualityQueryParser {
    /// `parse(qq)`.
    ///
    /// # Errors
    /// The parser's `ParseException`.
    fn parse(&self, query: &QualityQuery) -> Result<ParsedQuery, Error>;
}

/// The index a quality run searches: `IndexSearcher.search(query, n)` and
/// the stored fields' `docName`.
pub struct QualitySearcher<'r, 's, 'a> {
    pub reader: &'r DirectoryReader,
    pub searcher: &'s IndexSearcher<'s, 'a>,
}

impl QualitySearcher<'_, '_, '_> {
    /// `searcher.search(q, n).scoreDocs`.
    ///
    /// # Errors
    /// The search fails.
    pub fn search(&self, query: &ParsedQuery, n: usize) -> Result<Vec<(i32, f32)>, Error> {
        let bq = match query.to_clause(self.reader.max_doc()).rewrite() {
            Clause::MatchNoDocs(_) => return Ok(Vec::new()),
            Clause::Boolean(b) => *b,
            other => {
                let mut b = BooleanQuery::new();
                b.must.push(other);
                b
            }
        };
        let top = self.searcher.search(&bq, n)?;
        Ok(top.score_docs.iter().map(|h| (h.doc, h.score)).collect())
    }

    /// `DocNameExtractor.docName(storedFields, docid)`: the first string
    /// value of the field, `None` without one.
    ///
    /// # Errors
    /// The stored fields cannot be read.
    pub fn doc_name(&self, doc: i32, field: &str) -> Result<Option<String>, Error> {
        super::quality_utils::DocNameExtractor::new(field).doc_name(self.reader, doc)
    }
}

/// `QualityStats.MAX_POINTS`.
pub const MAX_POINTS: usize = 20;

/// `QualityStats.RecallPoint`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RecallPoint {
    pub rank: i32,
    pub recall: f64,
}

/// `QualityStats`: the quality of one query's results, or their average.
#[derive(Debug, Clone, PartialEq)]
pub struct QualityStats {
    max_good_points: f64,
    recall: f64,
    /// `pAt[0]` unused.
    p_at: [f64; MAX_POINTS + 1],
    p_relevant_sum: f64,
    num_points: f64,
    num_good_points: f64,
    mrr: f64,
    search_time: i64,
    doc_names_extract_time: i64,
    recall_points: Vec<RecallPoint>,
}

impl QualityStats {
    pub fn new(max_good_points: f64, search_time: i64) -> Self {
        QualityStats {
            max_good_points,
            recall: 0.0,
            p_at: [0.0; MAX_POINTS + 1],
            p_relevant_sum: 0.0,
            num_points: 0.0,
            num_good_points: 0.0,
            mrr: 0.0,
            search_time,
            doc_names_extract_time: 0,
            recall_points: Vec::new(),
        }
    }

    /// `addResult(n, isRelevant, docNameExtractTime)`: the result at rank
    /// `n`, which must follow the last.
    ///
    /// # Errors
    /// Java's `IllegalArgumentException` for a rank out of turn.
    pub fn add_result(
        &mut self,
        n: i32,
        is_relevant: bool,
        extract_time: i64,
    ) -> Result<(), Error> {
        let nf = f64::from(n);
        if (self.num_points + 1.0 - nf).abs() > 1e-6 {
            return Err(exception(format!(
                "java.lang.IllegalArgumentException: point {n} illegal after {} points!",
                lucene_util::geo::java_double_string(self.num_points)
            )));
        }
        if is_relevant {
            self.num_good_points += 1.0;
            self.recall_points.push(RecallPoint {
                rank: n,
                recall: self.num_good_points,
            });
            if self.recall_points.len() == 1 && n <= 5 {
                self.mrr = 1.0 / nf;
            }
        }
        self.num_points = nf;
        let p = self.num_good_points / self.num_points;
        if is_relevant {
            self.p_relevant_sum += p;
        }
        if let Some(slot) = usize::try_from(n).ok().and_then(|i| self.p_at.get_mut(i)) {
            *slot = p;
        }
        self.recall = if self.max_good_points <= 0.0 {
            p
        } else {
            self.num_good_points / self.max_good_points
        };
        self.doc_names_extract_time = self.doc_names_extract_time.saturating_add(extract_time);
        Ok(())
    }

    /// `getPrecisionAt(n)`.
    ///
    /// # Errors
    /// Java's `IllegalArgumentException` for `n` outside `1..=20`.
    pub fn precision_at(&self, n: i32) -> Result<f64, Error> {
        let index = usize::try_from(n)
            .ok()
            .filter(|&i| (1..=MAX_POINTS).contains(&i));
        let Some(index) = index else {
            return Err(exception(format!(
                "java.lang.IllegalArgumentException: n={n} - but it must be in [1,{MAX_POINTS}] range!"
            )));
        };
        if f64::from(n) > self.num_points {
            // `(int) numPoints`, below `n` <= 20
            let at = self.num_points as usize;
            return Ok(
                (self.num_points * self.p_at.get(at).copied().unwrap_or(0.0)) / f64::from(n),
            );
        }
        Ok(self.p_at[index])
    }

    /// `getAvp()`.
    pub fn avp(&self) -> f64 {
        if self.max_good_points == 0.0 {
            0.0
        } else {
            self.p_relevant_sum / self.max_good_points
        }
    }

    pub fn recall(&self) -> f64 {
        self.recall
    }

    pub fn mrr(&self) -> f64 {
        self.mrr
    }

    pub fn num_points(&self) -> f64 {
        self.num_points
    }

    pub fn num_good_points(&self) -> f64 {
        self.num_good_points
    }

    pub fn max_good_points(&self) -> f64 {
        self.max_good_points
    }

    pub fn search_time(&self) -> i64 {
        self.search_time
    }

    pub fn doc_names_extract_time(&self) -> i64 {
        self.doc_names_extract_time
    }

    pub fn recall_points(&self) -> &[RecallPoint] {
        &self.recall_points
    }

    /// `log(title, paddLines, logger, prefix)`.
    ///
    /// # Errors
    /// Java's `StringIndexOutOfBoundsException` for a value without a
    /// fraction (`NaN`, as an average of no judged query is).
    pub fn log(
        &self,
        title: Option<&str>,
        padd_lines: usize,
        logger: &mut String,
        prefix: &str,
    ) -> Result<(), Error> {
        for _ in 0..padd_lines {
            logger.push('\n');
        }
        if let Some(title) = title.filter(|t| !java_trim(t).is_empty()) {
            logger.push_str(title);
            logger.push('\n');
        }
        let mut line = |label: &str, value: f64| -> Result<(), Error> {
            let value = frac_format(&number_format(value, 3, true))?;
            let _ = writeln!(logger, "{prefix}{}{value}", pad(label, 19));
            Ok(())
        };
        line("Search Seconds: ", self.search_time as f64 / 1000.0)?;
        line(
            "DocName Seconds: ",
            self.doc_names_extract_time as f64 / 1000.0,
        )?;
        line("Num Points: ", self.num_points)?;
        line("Num Good Points: ", self.num_good_points)?;
        line("Max Good Points: ", self.max_good_points)?;
        line("Average Precision: ", self.avp())?;
        line("MRR: ", self.mrr)?;
        line("Recall: ", self.recall)?;
        // `(int) numPoints`
        let points = self.num_points as i64;
        let mut i = 1i32;
        while i64::from(i) < points && (i as usize) <= MAX_POINTS {
            line(&format!("Precision At {i}: "), self.precision_at(i)?)?;
            i = i.saturating_add(1);
        }
        for _ in 0..padd_lines {
            logger.push('\n');
        }
        Ok(())
    }

    /// `average(stats)`: times by every query, the rest by the queries with
    /// a relevant document.
    pub fn average(stats: &[QualityStats]) -> Result<QualityStats, Error> {
        let mut avg = QualityStats::new(0.0, 0);
        if stats.is_empty() {
            return Ok(avg);
        }
        let mut m = 0i32;
        for stat in stats {
            avg.search_time = avg.search_time.wrapping_add(stat.search_time);
            avg.doc_names_extract_time = avg
                .doc_names_extract_time
                .wrapping_add(stat.doc_names_extract_time);
            if stat.max_good_points > 0.0 {
                m = m.saturating_add(1);
                avg.num_good_points += stat.num_good_points;
                avg.num_points += stat.num_points;
                avg.p_relevant_sum += stat.avp();
                avg.recall += stat.recall;
                avg.mrr += stat.mrr;
                avg.max_good_points += stat.max_good_points;
                for j in 1..=MAX_POINTS {
                    avg.p_at[j] += stat.precision_at(j as i32)?;
                }
            }
        }
        let count = i64::try_from(stats.len()).unwrap_or(i64::MAX);
        avg.search_time /= count;
        avg.doc_names_extract_time /= count;
        let m = f64::from(m);
        avg.num_good_points /= m;
        avg.num_points /= m;
        avg.recall /= m;
        avg.mrr /= m;
        avg.max_good_points /= m;
        for p in avg.p_at.iter_mut().skip(1) {
            *p /= m;
        }
        avg.p_relevant_sum /= m;
        avg.p_relevant_sum *= avg.max_good_points;
        Ok(avg)
    }
}

/// `format(s, minLen)`: `s` padded with spaces to `min_len`.
pub(crate) fn pad(s: &str, min_len: usize) -> String {
    let n = s.chars().count();
    let mut out = s.to_string();
    for _ in n..min_len {
        out.push(' ');
    }
    out
}

/// `fracFormat(frac)`: the integer part right-aligned in six columns.
fn frac_format(frac: &str) -> Result<String, Error> {
    let Some(k) = frac.find('.') else {
        return Err(exception(format!(
            "java.lang.StringIndexOutOfBoundsException: Range [0, -1) out of bounds for length {}",
            frac.encode_utf16().count()
        )));
    };
    let int_part = &frac[..k];
    let width = k.max(6);
    Ok(format!("{int_part:>width$}{}", &frac[k..]))
}

/// `NumberFormat.getInstance(Locale.ROOT)` with exactly `digits` fraction
/// digits (`HALF_EVEN`), its integer part grouped by three when
/// `grouping`.
pub(crate) fn number_format(value: f64, digits: usize, grouping: bool) -> String {
    if value.is_nan() {
        return "NaN".to_string();
    }
    if value.is_infinite() {
        return if value < 0.0 { "-\u{221e}" } else { "\u{221e}" }.to_string();
    }
    let s = format!("{:.*}", digits, value.abs());
    let (int_part, frac) = s.split_once('.').unwrap_or((s.as_str(), ""));
    let int_part = if grouping {
        group(int_part)
    } else {
        int_part.to_string()
    };
    let negative = value.is_sign_negative();
    let sign = if negative { "-" } else { "" };
    if frac.is_empty() {
        format!("{sign}{int_part}")
    } else {
        format!("{sign}{int_part}.{frac}")
    }
}

/// `int_part`'s digits grouped by three with `,`.
fn group(int_part: &str) -> String {
    let len = int_part.len();
    let mut out = String::with_capacity(len.saturating_mul(2));
    for (i, c) in int_part.chars().enumerate() {
        if i > 0 && len.saturating_sub(i) % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    out
}

/// `QualityBenchmark`: runs quality queries and judges, or reports, their
/// results.
pub struct QualityBenchmark<'q> {
    queries: &'q [QualityQuery],
    parser: &'q dyn QualityQueryParser,
    doc_name_field: String,
    max_queries: usize,
    max_results: usize,
}

impl<'q> QualityBenchmark<'q> {
    pub fn new(
        queries: &'q [QualityQuery],
        parser: &'q dyn QualityQueryParser,
        doc_name_field: &str,
    ) -> Self {
        QualityBenchmark {
            queries,
            parser,
            doc_name_field: doc_name_field.to_string(),
            max_queries: usize::MAX,
            max_results: 1000,
        }
    }

    pub fn max_queries(&self) -> usize {
        self.max_queries
    }

    pub fn set_max_queries(&mut self, max_queries: usize) {
        self.max_queries = max_queries;
    }

    pub fn max_results(&self) -> usize {
        self.max_results
    }

    pub fn set_max_results(&mut self, max_results: usize) {
        self.max_results = max_results;
    }

    /// `execute(judge, submitRep, qualityLog)`: each query's stats when
    /// judged (`None` for every query otherwise).
    ///
    /// # Errors
    /// A query cannot be parsed or searched.
    pub fn execute(
        &self,
        searcher: &QualitySearcher<'_, '_, '_>,
        judge: Option<&dyn Judge>,
        mut submit: Option<&mut super::quality_utils::SubmissionReport>,
        mut log: Option<&mut String>,
    ) -> Result<Vec<Option<QualityStats>>, Error> {
        let n = self.max_queries.min(self.queries.len());
        let mut stats = Vec::with_capacity(n);
        for qq in &self.queries[..n] {
            let q = self.parser.parse(qq)?;
            let start = Instant::now();
            let hits = searcher.search(&q, self.max_results)?;
            let search_time = millis(start);
            let mut stat = None;
            if let Some(judge) = judge {
                stat = Some(self.analyze_query_results(
                    searcher,
                    qq,
                    &q,
                    &hits,
                    judge,
                    log.as_deref_mut(),
                    search_time,
                )?);
            }
            if let Some(submit) = submit.as_deref_mut() {
                submit.report(qq, &hits, &self.doc_name_field, searcher)?;
            }
            stats.push(stat);
        }
        if let Some(submit) = submit {
            submit.flush()?;
        }
        Ok(stats)
    }

    #[allow(clippy::too_many_arguments)]
    fn analyze_query_results(
        &self,
        searcher: &QualitySearcher<'_, '_, '_>,
        qq: &QualityQuery,
        q: &ParsedQuery,
        hits: &[(i32, f32)],
        judge: &dyn Judge,
        log: Option<&mut String>,
        search_time: i64,
    ) -> Result<QualityStats, Error> {
        let mut stats = QualityStats::new(f64::from(judge.max_recall(qq)), search_time);
        let mut start = Instant::now();
        for (i, &(doc, _)) in hits.iter().enumerate() {
            let name = searcher.doc_name(doc, &self.doc_name_field)?;
            let extract_time = millis(start);
            start = Instant::now();
            let relevant = judge.is_relevant(name.as_deref(), qq);
            let rank = i32::try_from(i.saturating_add(1)).unwrap_or(i32::MAX);
            stats.add_result(rank, relevant, extract_time)?;
        }
        if let Some(log) = log {
            let _ = writeln!(log, "{}  -  {}", qq.query_id(), q.to_string(""));
            stats.log(Some(&format!("{} Stats:", qq.query_id())), 1, log, "  ")?;
        }
        Ok(stats)
    }
}

/// Milliseconds since `start`, as `System.currentTimeMillis()` differences.
fn millis(start: Instant) -> i64 {
    i64::try_from(start.elapsed().as_millis()).unwrap_or(i64::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numbers_format_as_number_format_does() {
        assert_eq!(number_format(0.0625, 3, true), "0.062");
        assert_eq!(number_format(0.1875, 3, true), "0.188");
        assert_eq!(number_format(1234.5, 3, true), "1,234.500");
        assert_eq!(number_format(-0.0, 3, true), "-0.000");
        assert_eq!(number_format(f64::NAN, 3, true), "NaN");
        assert_eq!(number_format(f64::INFINITY, 3, true), "\u{221e}");
        assert_eq!(number_format(f64::NEG_INFINITY, 4, false), "-\u{221e}");
        assert_eq!(number_format(12345.0, 0, false), "12345");
        assert_eq!(frac_format("1.500").unwrap(), "     1.500");
        assert_eq!(frac_format("1,234,567.5").unwrap(), "1,234,567.5");
        assert!(frac_format("NaN").is_err());
        assert_eq!(pad("ab", 4), "ab  ");
        assert_eq!(pad("abcde", 4), "abcde");
    }

    #[test]
    fn stats_follow_the_ranks() {
        let mut s = QualityStats::new(2.0, 7);
        s.add_result(1, false, 1).unwrap();
        s.add_result(2, true, 1).unwrap();
        assert!(s.add_result(4, true, 0).is_err());
        s.add_result(3, true, 0).unwrap();
        assert_eq!(s.mrr(), 0.5);
        assert_eq!(s.recall(), 1.0);
        assert_eq!(s.precision_at(2).unwrap(), 0.5);
        assert_eq!(s.precision_at(6).unwrap(), 3.0 * (2.0 / 3.0) / 6.0);
        assert!(s.precision_at(0).is_err());
        assert!(s.precision_at(21).is_err());
        assert_eq!(s.avp(), (0.5 + 2.0 / 3.0) / 2.0);
        assert_eq!(s.recall_points().len(), 2);
        assert_eq!(s.doc_names_extract_time(), 2);
        assert_eq!(
            (s.num_points(), s.num_good_points(), s.max_good_points()),
            (3.0, 2.0, 2.0)
        );
        assert_eq!(s.search_time(), 7);
        let mut log = String::new();
        s.log(Some("t"), 1, &mut log, "> ").unwrap();
        assert!(
            // SPACES-OK: Java's padded report column.
            log.starts_with("\nt\n> Search Seconds:         0.007\n"),
            "{log}"
        );
        // SPACES-OK: Java's padded report column.
        assert!(log.contains("> Precision At 2:         0.500\n"), "{log}");
        assert!(!log.contains("Precision At 3"), "{log}");

        let unjudged = QualityStats::new(0.0, 3);
        let avg = QualityStats::average(&[s.clone(), unjudged.clone()]).unwrap();
        assert_eq!(avg.search_time(), 5);
        assert_eq!(avg.avp(), s.avp());
        assert!(QualityStats::average(std::slice::from_ref(&unjudged))
            .unwrap()
            .log(None, 0, &mut String::new(), "")
            .is_err());
        assert_eq!(
            QualityStats::average(&[]).unwrap(),
            QualityStats::new(0.0, 0)
        );
        let mut no_judgement = QualityStats::new(0.0, 0);
        no_judgement.add_result(1, true, 0).unwrap();
        assert_eq!(no_judgement.recall(), 1.0);
        assert_eq!(no_judgement.avp(), 0.0);
    }

    #[test]
    fn queries_sort_by_number_then_text() {
        let q = |id: &str| QualityQuery::new(id, vec![("title".to_string(), id.to_string())]);
        let mut qs = vec![q("10"), q("9"), q("b"), q("a")];
        sort_queries(&mut qs);
        let ids: Vec<&str> = qs.iter().map(QualityQuery::query_id).collect();
        assert_eq!(ids, ["9", "10", "a", "b"]);
        assert_eq!(qs[0].names(), ["title"]);
        assert_eq!(qs[0].value("title"), Some("9"));
        assert_eq!(qs[0].value("x"), None);
    }
}
