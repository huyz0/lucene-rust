//! `SearchFiles`: the demo's searcher. Parses each query with the classic
//! query parser over the standard analyzer, prints the parsed query and the
//! matching documents' paths a page at a time -- and, with `-knn_vector`,
//! adds a KNN query on the query words' embedding to each.
//!
//! ```text
//! lucene-tools SearchFiles [-index dir] [-field f] [-repeat n] [-queries file] [-query string] [-raw] [-paging hitsPerPage] [-knn_vector knnHits]
//! ```

use std::collections::HashMap;
use std::io::BufRead;
use std::sync::Arc;
use std::time::Instant;

use lucene_analysis::standard::StandardAnalyzer;
use lucene_analysis::Analyzer;
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::extended_query::DocAndScoreQuery;
use lucene_search::field_norms::FieldNorms;
use lucene_search::index_searcher::IndexSearcher;
use lucene_search::query::{BooleanQuery, Clause, MatchNoDocsQuery};
use lucene_search::query_visitor::{self, QueryVisitor, Term};
use lucene_search::queryparser::classic::{QueryParser, QueryParserError};
use lucene_search::queryparser::ParsedQuery;
use lucene_search::vector_query::{
    search_knn_float_vector_query_multi_segment, KnnFloatVectorQuery, KnnSegment, VectorsInput,
};
use lucene_store::directory::FsDirectory;
use lucene_util::geo::java_float_string;

use super::index_files::KNN_DICT;
use super::knn::{DemoEmbeddings, KnnVectorDict};
use crate::cli::{arg, exception, java_trim, parse_int, Error, Io, ToolResult};

/// `main`'s usage text.
pub const USAGE: &str = "Usage:\tjava org.apache.lucene.demo.SearchFiles [-index dir] [-field f] [-repeat n] [-queries file] [-query string] [-raw] [-paging hitsPerPage] [-knn_vector knnHits]\n\nSee http://lucene.apache.org/core/9_0_0/demo/ for details.";

/// The vector field `IndexFiles` writes.
const VECTOR_FIELD: &str = "contents-vector";

/// `SearchFiles.main(args)`, reading interactive input from `System.in`.
///
/// # Errors
/// As [`search_files`].
pub fn main(args: &[String], io: &mut Io<'_>) -> ToolResult {
    search_files(args, io, &mut std::io::stdin().lock())
}

/// The options `main` reads.
struct Options {
    index: String,
    field: String,
    queries: Option<String>,
    query: Option<String>,
    repeat: i32,
    raw: bool,
    hits_per_page: i32,
    knn_vectors: i32,
}

/// `SearchFiles.main(args)` with `System.in` as `stdin`. `-h` or `-help`
/// first prints the usage and exits 0; an unknown argument or a page size
/// under 1 prints so and exits 1. Without `-query` or `-queries` it prompts
/// on `stdin` and pages interactively.
///
/// # Errors
/// What opening the index and the queries file report, a query that does
/// not parse (Java's `ParseException`), and a page number that is not one
/// (`NumberFormatException`).
pub fn search_files(args: &[String], io: &mut Io<'_>, stdin: &mut dyn BufRead) -> ToolResult {
    if args.first().is_some_and(|a| a == "-h" || a == "-help") {
        writeln!(io.out, "{USAGE}")?;
        return Ok(0);
    }
    let mut o = Options {
        index: "index".to_string(),
        field: "contents".to_string(),
        queries: None,
        query: None,
        repeat: 0,
        raw: false,
        hits_per_page: 10,
        knn_vectors: 0,
    };
    let int = |s: &str| {
        parse_int(s).map_err(|m| exception(format!("java.lang.NumberFormatException: {m}")))
    };
    let mut i = 0usize;
    while i < args.len() {
        let next = i.saturating_add(1);
        match args[i].as_str() {
            "-index" => o.index = arg(args, next)?.to_string(),
            "-field" => o.field = arg(args, next)?.to_string(),
            "-queries" => o.queries = Some(arg(args, next)?.to_string()),
            "-query" => o.query = Some(arg(args, next)?.to_string()),
            "-repeat" => o.repeat = int(arg(args, next)?)?,
            "-raw" => o.raw = true,
            "-paging" => {
                o.hits_per_page = int(arg(args, next)?)?;
                if o.hits_per_page <= 0 {
                    writeln!(io.err, "There must be at least 1 hit per page.")?;
                    return Ok(1);
                }
            }
            "-knn_vector" => o.knn_vectors = int(arg(args, next)?)?,
            other => {
                writeln!(io.err, "Unknown argument: {other}")?;
                return Ok(1);
            }
        }
        // the options with a value took it
        i = if matches!(args[i].as_str(), "-raw") {
            next
        } else {
            next.saturating_add(1)
        };
    }

    let dir = FsDirectory::open(&o.index);
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
    let search = Search {
        reader: &reader,
        searcher: &searcher,
    };
    let vector_dict = if o.knn_vectors > 0 {
        Some(KnnVectorDict::open(&dir, KNN_DICT)?)
    } else {
        None
    };
    let mut queries_file;
    let input: &mut dyn BufRead = match &o.queries {
        Some(path) => {
            queries_file = std::io::BufReader::new(std::fs::File::open(path)?);
            &mut queries_file
        }
        None => stdin,
    };
    let analyzer: Arc<Analyzer> = Arc::new(Analyzer::new(StandardAnalyzer::new()));
    let parser = QueryParser::new(o.field.clone(), analyzer);
    let interactive = o.queries.is_none() && o.query.is_none();
    loop {
        if interactive {
            writeln!(io.out, "Enter query: ")?;
        }
        let line = match &o.query {
            Some(q) => Some(q.clone()),
            None => read_line(input)?,
        };
        let Some(line) = line else { break };
        let line = java_trim(&line);
        if line.is_empty() {
            break;
        }
        let parsed = parser.parse(line).map_err(parse_error)?;
        let query = match &vector_dict {
            // `knnVectors > 0` opened the dictionary
            Some(dict) => {
                let k = usize::try_from(o.knn_vectors).unwrap_or(0);
                add_semantic_query(parsed, dict, k, &search)?
            }
            None => SearchQuery::Parsed(parsed),
        };
        writeln!(io.out, "Searching for: {}", query.to_string(&o.field))?;
        let clause = query.to_clause(&search)?;
        if o.repeat > 0 {
            let start = Instant::now();
            for _ in 0..o.repeat {
                search.top(&clause, 100)?;
            }
            writeln!(io.out, "Time: {}ms", start.elapsed().as_millis())?;
        }
        do_paging_search(
            input,
            &search,
            &clause,
            o.hits_per_page,
            o.raw,
            interactive,
            io,
        )?;
        if o.query.is_some() {
            break;
        }
    }
    Ok(0)
}

/// A `ParseException` (or another exception out of `parse`) as Java names it.
fn parse_error(e: QueryParserError) -> Error {
    let class = match e.class {
        "ParseException" => "org.apache.lucene.queryparser.classic.ParseException",
        "TooManyClauses" => "org.apache.lucene.search.IndexSearcher$TooManyClauses",
        other => return exception(format!("java.lang.{other}: {}", e.message)),
    };
    exception(format!("{class}: {}", e.message))
}

/// `BufferedReader.readLine()`: the next line without its terminator, or
/// `None` at the end.
fn read_line(input: &mut dyn BufRead) -> Result<Option<String>, Error> {
    let mut line = String::new();
    if input.read_line(&mut line)? == 0 {
        return Ok(None);
    }
    if line.ends_with('\n') {
        line.pop();
    }
    if line.ends_with('\r') {
        line.pop();
    }
    Ok(Some(line))
}

/// The index and its searcher.
struct Search<'r, 's, 'a> {
    reader: &'r DirectoryReader,
    searcher: &'s IndexSearcher<'s, 'a>,
}

impl Search<'_, '_, '_> {
    /// `searcher.search(query, n)`: the top `n` hits and the total.
    fn top(&self, clause: &Clause, n: usize) -> Result<(Vec<(i32, f32)>, u64), Error> {
        let bq = match clause.clone() {
            Clause::MatchNoDocs(_) => return Ok((Vec::new(), 0)),
            Clause::Boolean(b) => *b,
            other => {
                let mut b = BooleanQuery::new();
                b.must.push(other);
                b
            }
        };
        let top = self.searcher.search(&bq, n)?;
        Ok((
            top.score_docs.iter().map(|h| (h.doc, h.score)).collect(),
            top.total_hits.value,
        ))
    }

    /// `storedFields.document(doc).get(name)`.
    fn stored(&self, doc: i32, name: &str) -> Result<Option<String>, Error> {
        let Some(seg) = self
            .reader
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
            .find(|f| f.name == name)
            .map(|f| f.number)
        else {
            return Ok(None);
        };
        let Some(document) = seg.stored_document(doc - seg.doc_base)? else {
            return Ok(None);
        };
        use lucene_codecs::stored_fields::FieldValue;
        Ok(document
            .fields
            .into_iter()
            .filter(|f| f.field_number == number)
            .find_map(|f| match f.value {
                FieldValue::String(s) => Some(s),
                FieldValue::Int(v) => Some(v.to_string()),
                FieldValue::Long(v) => Some(v.to_string()),
                FieldValue::Float(v) => Some(java_float_string(v)),
                FieldValue::Double(v) => Some(lucene_util::geo::java_double_string(v)),
                FieldValue::Binary(_) => None,
            }))
    }
}

/// The query searched: the parsed one, or that one `SHOULD` with a KNN
/// query on its words' embedding.
enum SearchQuery {
    Parsed(ParsedQuery),
    WithKnn {
        parsed: ParsedQuery,
        knn: KnnFloatVectorQuery,
    },
}

impl SearchQuery {
    /// `query.toString(field)`.
    fn to_string(&self, field: &str) -> String {
        match self {
            SearchQuery::Parsed(q) => q.to_string(field),
            SearchQuery::WithKnn { parsed, knn } => {
                // `BooleanQuery.toString`: `SHOULD` adds no prefix, a nested
                // boolean is parenthesized.
                let first = match parsed {
                    ParsedQuery::Boolean(_) => format!("({})", parsed.to_string(field)),
                    _ => parsed.to_string(field),
                };
                let target0 = knn.target.first().copied().unwrap_or(0.0);
                format!(
                    "{first} KnnFloatVectorQuery:{}[{},...][{}]",
                    knn.field,
                    java_float_string(target0),
                    knn.k
                )
            }
        }
    }

    /// The query as a clause the searcher runs; the KNN query rewritten to
    /// its hits (`AbstractKnnVectorQuery.rewrite`).
    fn to_clause(&self, search: &Search<'_, '_, '_>) -> Result<Clause, Error> {
        let max_doc = search.reader.max_doc();
        match self {
            SearchQuery::Parsed(q) => Ok(q.to_clause(max_doc).rewrite()),
            SearchQuery::WithKnn { parsed, knn } => {
                let mut b = BooleanQuery::new();
                b.should.push(parsed.to_clause(max_doc).rewrite());
                b.should.push(knn_clause(search.reader, knn)?);
                Ok(Clause::Boolean(Box::new(b)))
            }
        }
    }
}

/// The KNN query's hits as a clause: an exhaustive scan of each segment's
/// flat vectors (no graph is opened), which on the demo's handful of
/// documents finds what Java's graph search finds.
fn knn_clause(reader: &DirectoryReader, knn: &KnnFloatVectorQuery) -> Result<Clause, Error> {
    let segs = reader.segment_readers();
    let flats = segs
        .iter()
        .map(|seg| {
            let number = seg
                .field_infos()
                .fields
                .iter()
                .find(|f| f.name == knn.field)
                .map(|f| f.number);
            match number {
                Some(n) => seg.flat_vectors_reader_for(n),
                None => Ok(None),
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    let knn_segments: Vec<KnnSegment<'_>> = segs
        .iter()
        .zip(flats)
        .filter_map(|(seg, flat)| {
            Some(KnnSegment {
                vectors: VectorsInput {
                    flat: flat?,
                    hnsw: None,
                    field_infos: seg.field_infos(),
                    live_docs: seg.live_docs(),
                    filter: None,
                    max_doc: seg.max_doc,
                },
                doc_base: seg.doc_base,
            })
        })
        .collect();
    let hits = search_knn_float_vector_query_multi_segment(&knn_segments, knn)?;
    if hits.is_empty() {
        return Ok(Clause::MatchNoDocs(MatchNoDocsQuery::new()));
    }
    let bases: Vec<i32> = segs.iter().map(|s| s.doc_base).collect();
    Ok(DocAndScoreQuery::new(hits.iter().map(|h| (h.doc_id, h.score)).collect(), &bases).into())
}

/// `QueryFieldTermExtractor`: the terms of one field, outside `MUST_NOT`
/// clauses.
struct FieldTermExtractor {
    field: String,
    terms: Vec<String>,
}

impl QueryVisitor for FieldTermExtractor {
    fn consume_terms(&mut self, _query: &Clause, terms: &[Term]) {
        for t in terms.iter().filter(|t| t.field == self.field) {
            self.terms
                .push(String::from_utf8_lossy(&t.bytes).into_owned());
        }
    }

    fn accept_field(&self, field: &str) -> bool {
        field == self.field
    }
}

/// `addSemanticQuery(query, vectorDict, k)`: the query, `SHOULD` with a KNN
/// query for the embedding of its `contents` terms -- or the query alone
/// when it has none.
fn add_semantic_query(
    parsed: ParsedQuery,
    dict: &KnnVectorDict,
    k: usize,
    search: &Search<'_, '_, '_>,
) -> Result<SearchQuery, Error> {
    let mut extractor = FieldTermExtractor {
        field: "contents".to_string(),
        terms: Vec::new(),
    };
    query_visitor::visit(&parsed.to_clause(search.reader.max_doc()), &mut extractor);
    let mut text = String::new();
    for t in &extractor.terms {
        text.push_str(t);
        text.push(' ');
    }
    if text.is_empty() {
        return Ok(SearchQuery::Parsed(parsed));
    }
    let target = DemoEmbeddings::new(dict).compute_embedding(&text)?;
    let knn = KnnFloatVectorQuery::new(VECTOR_FIELD, target, k)?;
    Ok(SearchQuery::WithKnn { parsed, knn })
}

/// `doPagingSearch(in, searcher, query, hitsPerPage, raw, interactive)`.
fn do_paging_search(
    input: &mut dyn BufRead,
    search: &Search<'_, '_, '_>,
    clause: &Clause,
    hits_per_page: i32,
    raw: bool,
    interactive: bool,
    io: &mut Io<'_>,
) -> Result<(), Error> {
    let page = usize::try_from(hits_per_page).unwrap_or(1);
    let (mut hits, total) = search.top(clause, page.saturating_mul(5))?;
    let num_total_hits = usize::try_from(total).unwrap_or(usize::MAX);
    writeln!(io.out, "{num_total_hits} total matching documents")?;
    let mut start = 0usize;
    let mut end = num_total_hits.min(page);
    loop {
        if end > hits.len() {
            writeln!(
                io.out,
                "Only results 1 - {} of {num_total_hits} total matching documents collected.",
                hits.len()
            )?;
            writeln!(io.out, "Collect more (y/n) ?")?;
            let line = read_line(input)?;
            if line
                .as_deref()
                .is_none_or(|l| l.is_empty() || l.starts_with('n'))
            {
                break;
            }
            hits = search.top(clause, num_total_hits)?.0;
        }
        end = hits.len().min(start.saturating_add(page));
        for (i, &(doc, score)) in hits.iter().enumerate().take(end).skip(start) {
            let rank = i.saturating_add(1);
            if raw {
                writeln!(io.out, "doc={doc} score={}", java_float_string(score))?;
                continue;
            }
            match search.stored(doc, "path")? {
                Some(path) => {
                    writeln!(io.out, "{rank}. {path}")?;
                    if let Some(title) = search.stored(doc, "title")? {
                        writeln!(io.out, "   Title: {title}")?;
                    }
                }
                None => writeln!(io.out, "{rank}. No path for this document")?,
            }
        }
        if !interactive || end == 0 {
            break;
        }
        if num_total_hits >= end {
            let mut quit = false;
            loop {
                write!(io.out, "Press ")?;
                if start >= page {
                    write!(io.out, "(p)revious page, ")?;
                }
                if start.saturating_add(page) < num_total_hits {
                    write!(io.out, "(n)ext page, ")?;
                }
                writeln!(io.out, "(q)uit or enter number to jump to a page.")?;
                let line = read_line(input)?;
                let Some(line) = line.filter(|l| !l.is_empty() && !l.starts_with('q')) else {
                    quit = true;
                    break;
                };
                if line.starts_with('p') {
                    start = start.saturating_sub(page);
                    break;
                } else if line.starts_with('n') {
                    if start.saturating_add(page) < num_total_hits {
                        start = start.saturating_add(page);
                    }
                    break;
                }
                let number = parse_int(&line)
                    .map_err(|m| exception(format!("java.lang.NumberFormatException: {m}")))?;
                // `(page - 1) * hitsPerPage`, in `int`
                let first = i64::from(number)
                    .saturating_sub(1)
                    .saturating_mul(i64::from(hits_per_page));
                if first < i64::try_from(num_total_hits).unwrap_or(i64::MAX) {
                    start = usize::try_from(first).unwrap_or(0);
                    break;
                }
                writeln!(io.out, "No such page")?;
            }
            if quit {
                break;
            }
            end = num_total_hits.min(start.saturating_add(page));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(args: &[&str], stdin: &str) -> (ToolResult, String, String) {
        let args: Vec<String> = args.iter().map(|s| s.to_string()).collect();
        let (mut out, mut err) = (Vec::new(), Vec::new());
        let r = search_files(
            &args,
            &mut Io {
                out: &mut out,
                err: &mut err,
            },
            &mut stdin.as_bytes(),
        );
        (
            r,
            String::from_utf8(out).unwrap(),
            String::from_utf8(err).unwrap(),
        )
    }

    #[test]
    fn the_argument_checks_are_javas() {
        let (r, out, _) = run(&["-help", "-x"], "");
        assert_eq!(r.unwrap(), 0);
        assert_eq!(out, format!("{USAGE}\n"));
        let (r, _, err) = run(&["-x"], "");
        assert_eq!(r.unwrap(), 1);
        assert_eq!(err, "Unknown argument: -x\n");
        let (r, _, err) = run(&["-paging", "0"], "");
        assert_eq!(r.unwrap(), 1);
        assert_eq!(err, "There must be at least 1 hit per page.\n");
        let (r, _, _) = run(&["-repeat", "x"], "");
        assert_eq!(
            r.unwrap_err().to_string(),
            "java.lang.NumberFormatException: For input string: \"x\""
        );
        let (r, _, _) = run(&["-query"], "");
        assert!(r
            .unwrap_err()
            .to_string()
            .starts_with("java.lang.ArrayIndexOutOfBoundsException"));
    }

    /// An index of three documents matching `alpha`: one with a path and a
    /// title, one with neither, one with every other stored value type.
    fn odd_index(dir: &std::path::Path) {
        use lucene_index::document::{Document, Field, Store, StoredValue, StringField, TextField};
        use lucene_index::index_writer::IndexWriter;
        let fs = FsDirectory::open(dir);
        std::fs::create_dir_all(dir).unwrap();
        let mut w = IndexWriter::open(
            &fs,
            Vec::new(),
            super::super::index_files::CODEC,
            super::super::index_files::VERSION,
        )
        .unwrap();
        let mut d = Document::new();
        d.add(TextField::new("contents", "alpha", Store::No));
        d.add(StringField::new("path", "p0", Store::Yes));
        d.add(StringField::new("title", "T0", Store::Yes));
        w.add_fields_document(&d).unwrap();
        let mut d = Document::new();
        d.add(TextField::new("contents", "alpha beta", Store::No));
        w.add_fields_document(&d).unwrap();
        let mut d = Document::new();
        d.add(TextField::new("contents", "alpha beta gamma", Store::No));
        d.add(Field::stored("i", StoredValue::Int(-3)));
        d.add(Field::stored("l", StoredValue::Long(4)));
        d.add(Field::stored("f", StoredValue::Float(0.5)));
        d.add(Field::stored("d", StoredValue::Double(1e-5)));
        d.add(Field::stored("b", StoredValue::Binary(vec![1])));
        w.add_fields_document(&d).unwrap();
        w.close().unwrap();
    }

    #[test]
    fn stored_values_titles_and_pathless_documents_print_as_javas() {
        let tmp = lucene_util::test_support::TempDir::new("tools-search-odd");
        let index = tmp.path().join("index");
        odd_index(&index);
        let (r, out, _) = run(&["-index", index.to_str().unwrap(), "-query", "alpha"], "");
        assert_eq!(r.unwrap(), 0);
        assert!(out.contains("   Title: T0\n"), "{out}");
        assert!(out.contains(". No path for this document\n"), "{out}");

        let fs = FsDirectory::open(&index);
        let reader = DirectoryReader::open(&fs).unwrap();
        let opened = reader.open_segments().unwrap();
        let segments = opened.as_open_segments();
        let owned = reader
            .field_norms_by_field(&["contents".to_string()])
            .unwrap();
        let norms: Vec<Option<&HashMap<String, FieldNorms<'_>>>> = owned.iter().map(Some).collect();
        let searcher = IndexSearcher::new(&segments, &norms).unwrap();
        let search = Search {
            reader: &reader,
            searcher: &searcher,
        };
        let stored = |doc, name| search.stored(doc, name).unwrap();
        assert_eq!(stored(2, "i").as_deref(), Some("-3"));
        assert_eq!(stored(2, "l").as_deref(), Some("4"));
        assert_eq!(stored(2, "f").as_deref(), Some("0.5"));
        assert_eq!(stored(2, "d").as_deref(), Some("1.0E-5"));
        assert_eq!(stored(2, "b"), None);
        assert_eq!(stored(2, "no-such-field"), None);
        assert_eq!(stored(99, "path"), None);
        let none = search
            .top(&Clause::MatchNoDocs(MatchNoDocsQuery::new()), 10)
            .unwrap();
        assert_eq!(none, (Vec::new(), 0));
    }

    #[test]
    fn lines_and_trimming_are_javas() {
        let mut input: &[u8] = b"a\r\nb\n\nc";
        let lines: Vec<Option<String>> = (0..5).map(|_| read_line(&mut input).unwrap()).collect();
        assert_eq!(
            lines,
            [
                Some("a".to_string()),
                Some("b".to_string()),
                Some(String::new()),
                Some("c".to_string()),
                None
            ]
        );
        assert_eq!(java_trim("\t x \u{1} "), "x");
    }

    #[test]
    fn parse_errors_are_named_as_java_names_them() {
        let e = |class: &'static str| {
            parse_error(QueryParserError {
                class,
                message: "m".to_string(),
            })
            .to_string()
        };
        assert_eq!(
            e("ParseException"),
            "org.apache.lucene.queryparser.classic.ParseException: m"
        );
        assert_eq!(
            e("TooManyClauses"),
            "org.apache.lucene.search.IndexSearcher$TooManyClauses: m"
        );
        assert_eq!(
            e("IllegalArgumentException"),
            "java.lang.IllegalArgumentException: m"
        );
    }
}
