//! `lucene-tools Luke`: Luke's models through a command line, printing what
//! each returns. Luke itself is a Swing application with no command line;
//! this one is the twin of the driver `fixtures/src/GenToolsLuke.java` runs
//! the Java models through, line for line, so the two print the same bytes.
//!
//! ```text
//! lucene-tools Luke overview <index> [field n]...
//! lucene-tools Luke documents <index>
//! lucene-tools Luke terms <index> <field> [seek <text>]...
//! lucene-tools Luke stored <index> <sort field>
//! lucene-tools Luke commits <index>
//! lucene-tools Luke commit <index> <generation>
//! lucene-tools Luke fields <index>
//! lucene-tools Luke search <index> [option]... <expression>
//! lucene-tools Luke mlt <index> <doc> <fields> [minDocFreq minTermFreq]
//! lucene-tools Luke components
//! lucene-tools Luke analyze [-step] [-analyzer CLASS | -tokenizer NAME [k=v]... [-cf NAME [k=v]...]... [-tf NAME [k=v]...]...] <text>
//! lucene-tools Luke export <index> <field> <delimiter> <dest dir>
//! lucene-tools Luke check <index>
//! lucene-tools Luke optimize <index> <maxSegments | expunge>
//! lucene-tools Luke delete <index> <field> <term>
//! lucene-tools Luke add <index> <field> <text>
//! lucene-tools Luke create <dir> [20 newsgroups dir]
//! lucene-tools Luke copy <from dir> <to dir>
//! ```
//!
//! A model's exception ends the command with the line `error: <simple class
//! name>: <message>` and exit code 1.

use std::path::Path;
use std::sync::Arc;

use lucene_analysis::factory::spi;
use lucene_analysis::Analyzer;
use lucene_index::buffered_updates::{DeleteQuery, Term};
use lucene_index::document::{Document, Store, TextField};
use lucene_search::directory_reader::DirectoryReader;
use lucene_store::directory::{Directory, FsDirectory};
use lucene_util::geo::java_float_string;

use super::analysis::{Analysis, ComponentConfig, CustomAnalyzerConfig, Token};
use super::commits::Commits;
use super::documents::{vector_similarity_name, Documents};
use super::overview::{Overview, TermCountsOrder};
use super::search::{
    LukeSortField, MltConfig, Operator, QueryParserConfig, Search, SearchResults, SimilarityConfig,
};
use super::tools::IndexTools;
use super::{bytes_ref_string, list_string, map_string, LukeError};
use crate::cli::{arg, exception, parse_int, Error, Io, ToolResult};

const STANDARD_ANALYZER: &str = "org.apache.lucene.analysis.standard.StandardAnalyzer";

/// `Luke <command> args...`.
///
/// # Errors
/// An unknown command, a missing argument, or an index that cannot be
/// opened; a model's own exception is printed, not thrown.
pub fn main(args: &[String], io: &mut Io<'_>) -> ToolResult {
    let command = arg(args, 0)?;
    let args = &args[1..];
    let mut out = String::new();
    let result = run(command, args, &mut out);
    io.out.write_all(out.as_bytes())?;
    match result {
        Ok(()) => Ok(0),
        Err(e) => match e.downcast_ref::<LukeError>() {
            Some(model) => {
                writeln!(io.out, "error: {model}")?;
                Ok(1)
            }
            None => Err(e),
        },
    }
}

fn run(command: &str, args: &[String], out: &mut String) -> Result<(), Error> {
    match command {
        "copy" => return copy_tree(Path::new(arg(args, 0)?), Path::new(arg(args, 1)?)),
        "components" => {
            components(out);
            return Ok(());
        }
        "analyze" => return analyze(args, out),
        "create" => return create(args, out),
        _ => {}
    }
    let path = arg(args, 0)?;
    let dir = FsDirectory::open(path);
    let reader = DirectoryReader::open(&dir as &dyn Directory)
        .map_err(|e| exception(format!("java.io.IOException: {e}")))?;
    match command {
        "overview" => overview(&reader, args, out),
        "documents" => documents(&reader, out),
        "terms" => terms(&reader, args, out),
        "stored" => stored(&reader, arg(args, 1)?, out),
        "commits" => commits(&dir, path, out),
        "commit" => commit(&dir, path, arg(args, 1)?, out),
        "fields" => fields(&reader, out),
        "search" => search(&reader, args, out),
        "mlt" => mlt(&reader, args, out),
        "export" => export(&reader, &dir, args, out),
        "check" => {
            let st = IndexTools::with_reader(&reader, &dir, false, false).check_index()?;
            line(out, format!("clean: {}", st.clean));
            line(out, format!("segments: {}", st.num_segments));
            line(out, format!("bad segments: {}", st.num_bad_segments));
            line(out, format!("lost documents: {}", st.tot_lose_doc_count));
            Ok(())
        }
        "optimize" => {
            let expunge = arg(args, 1)? == "expunge";
            let max = if expunge { 1 } else { int(arg(args, 1)?)? };
            Ok(IndexTools::with_reader(&reader, &dir, false, false).optimize(expunge, max)?)
        }
        "delete" => {
            let term = Term::new(arg(args, 1)?, arg(args, 2)?.as_bytes().to_vec());
            Ok(IndexTools::with_reader(&reader, &dir, false, false)
                .delete_documents(DeleteQuery::Term(term))?)
        }
        "add" => {
            let mut doc = Document::new();
            doc.add(TextField::new(arg(args, 1)?, arg(args, 2)?, Store::Yes));
            Ok(IndexTools::with_reader(&reader, &dir, false, false)
                .add_document(&doc, standard_analyzer())?)
        }
        _ => Err(LukeError::illegal_argument(format!("unknown command {command}")).into()),
    }
}

fn line(out: &mut String, s: impl AsRef<str>) {
    out.push_str(s.as_ref());
    out.push('\n');
}

/// `Optional.map(String::valueOf).orElse("--")`.
fn present<T: std::fmt::Display>(o: Option<T>) -> String {
    o.map_or_else(|| "--".to_string(), |v| v.to_string())
}

fn int(s: &str) -> Result<i32, Error> {
    parse_int(s).map_err(|m| exception(format!("java.lang.NumberFormatException: {m}")))
}

fn standard_analyzer() -> Arc<Analyzer> {
    Arc::new(spi::new_analyzer(STANDARD_ANALYZER).expect("StandardAnalyzer is built in"))
}

fn overview(reader: &DirectoryReader, args: &[String], out: &mut String) -> Result<(), Error> {
    let mut o = Overview::new(reader)?;
    line(out, format!("fields: {}", o.num_fields()));
    line(out, format!("documents: {}", o.num_documents()));
    line(out, format!("terms: {}", o.num_terms()));
    line(out, format!("deletions: {}", o.has_deletions()));
    line(out, format!("deleted documents: {}", o.num_deleted_docs()));
    line(out, format!("optimized: {}", present(o.is_optimized())));
    line(out, format!("version: {}", present(o.index_version())));
    line(out, format!("format: {}", present(o.index_format())));
    line(out, format!("commit: {}", present(o.commit_description())));
    line(out, format!("user data: {}", present(o.commit_user_data())));
    line(
        out,
        format!("field names: {}", list_string(o.field_names())),
    );
    for order in TermCountsOrder::ALL {
        let counts = o.sorted_term_counts(order);
        line(
            out,
            format!("term counts {}: {}", order.name(), map_string(counts)),
        );
    }
    let mut i = 1usize;
    while i.saturating_add(1) < args.len() {
        let field = &args[i];
        let n = int(&args[i.saturating_add(1)])?;
        line(out, format!("top {field} {n}:"));
        for t in o.top_terms(field, n)? {
            line(out, format!("  {t}"));
        }
        i = i.saturating_add(2);
    }
    Ok(())
}

fn documents(reader: &DirectoryReader, out: &mut String) -> Result<(), Error> {
    let d = Documents::new(reader);
    line(out, format!("max doc: {}", d.max_doc()));
    line(
        out,
        format!("field names: {}", list_string(d.field_names())),
    );
    for doc in 0..d.max_doc() {
        line(out, format!("doc {doc} live={}", d.is_live(doc)));
        for f in d.document_fields(doc)? {
            line(
                out,
                format!(
                    "  {f} hasPayloads={} hasNorms={} norm={} string={} binary={} numeric={} \
                     pointNumBytes={} vectorSimilarity={}",
                    f.has_payloads,
                    f.has_norms,
                    f.norm,
                    f.string_value().unwrap_or("null"),
                    f.binary_value()
                        .map_or_else(|| "null".to_string(), bytes_ref_string),
                    f.numeric_value().unwrap_or("null"),
                    f.point_num_bytes,
                    vector_similarity_name(f.vector_similarity)
                ),
            );
        }
        for field in d.field_names() {
            let tv = d.term_vectors(doc, &field)?;
            if !tv.is_empty() {
                line(out, format!("  term vector {field}:"));
                for e in tv {
                    line(out, format!("    {e}"));
                }
            }
            if let Some(dv) = d.doc_values(doc, &field)? {
                line(out, format!("  doc values {field}: {dv}"));
            }
        }
    }
    Ok(())
}

fn terms(reader: &DirectoryReader, args: &[String], out: &mut String) -> Result<(), Error> {
    let mut d = Documents::new(reader);
    let mut term = d.first_term(arg(args, 1)?)?;
    line(out, format!("first term: {}", present(term.as_ref())));
    line(
        out,
        format!("current field: {}", d.current_field().unwrap_or("null")),
    );
    while term.is_some() {
        postings(&mut d, out)?;
        term = d.next_term()?;
        line(out, format!("next term: {}", present(term.as_ref())));
    }
    line(out, format!("doc freq: {}", present(d.doc_freq()?)));
    line(out, format!("first doc: {}", present(d.first_term_doc()?)));
    line(
        out,
        format!("positions: {}", list_string(d.term_positions())),
    );
    let mut i = 2usize;
    while i.saturating_add(1) < args.len() {
        let text = &args[i.saturating_add(1)];
        let term = d.seek_term(text)?;
        line(out, format!("seek {text}: {}", present(term.as_ref())));
        if term.is_some() {
            postings(&mut d, out)?;
        }
        i = i.saturating_add(2);
    }
    Ok(())
}

fn postings(d: &mut Documents<'_>, out: &mut String) -> Result<(), Error> {
    line(out, format!("  doc freq: {}", present(d.doc_freq()?)));
    let mut doc = d.first_term_doc()?;
    while let Some(n) = doc {
        let positions: Vec<String> = d.term_positions().iter().map(ToString::to_string).collect();
        line(out, format!("  doc {n}: {}", positions.join(", ")));
        doc = d.next_term_doc();
    }
    line(out, format!("  next doc: {}", present(d.next_term_doc())));
    Ok(())
}

/// Every document's stored fields, the documents by the value of
/// `sort_field`.
fn stored(reader: &DirectoryReader, sort_field: &str, out: &mut String) -> Result<(), Error> {
    let d = Documents::new(reader);
    let mut docs: Vec<(String, Vec<String>)> = Vec::new();
    for doc in 0..d.max_doc() {
        let mut lines = Vec::new();
        let mut key = String::new();
        for f in d.document_fields(doc)? {
            if f.is_stored {
                let value = f.string_value().unwrap_or("null");
                lines.push(format!("{}={value}", f.name));
                if f.name == sort_field {
                    key = value.to_string();
                }
            }
        }
        docs.push((key, lines));
    }
    // `List.sort`: stable, by `String.compareTo`
    docs.sort_by(|a, b| java_compare(&a.0, &b.0));
    for (key, lines) in docs {
        line(out, format!("doc {key}"));
        for l in lines {
            line(out, format!("  {l}"));
        }
    }
    Ok(())
}

/// `String.compareTo`: by UTF-16 code unit.
fn java_compare(a: &str, b: &str) -> std::cmp::Ordering {
    a.encode_utf16().cmp(b.encode_utf16())
}

/// The diagnostics that differ between runs and machines, printed `*`.
const MASKED_DIAGNOSTICS: [&str; 6] = [
    "timestamp",
    "os",
    "os.arch",
    "os.version",
    "java.vendor",
    "java.runtime.version",
];

fn commits(dir: &FsDirectory, path: &str, out: &mut String) -> Result<(), Error> {
    let commits = Commits::new(dir, Path::new(path))?;
    for c in commits.list_commits() {
        let generation = c.generation;
        line(
            out,
            format!(
                "commit {generation} deleted={} segments={} user data={}",
                c.is_deleted, c.seg_count, c.user_data
            ),
        );
        line(
            out,
            format!("  same commit: {}", commits.commit(generation).is_some()),
        );
        for f in commits.files(generation) {
            line(out, format!("  file {} {}", f.file_name, f.display_size));
        }
        for s in commits.segments(generation)? {
            line(
                out,
                format!(
                    "  segment {} maxDoc={} delGen={} delCount={} version={} codec={} size={} \
                     compound={}",
                    s.name,
                    s.max_doc,
                    s.del_gen,
                    s.del_count,
                    s.lucene_ver,
                    s.codec_name,
                    s.display_size,
                    s.use_compound_file
                ),
            );
            let attributes = commits.segment_attributes(generation, &s.name)?;
            line(out, format!("    attributes: {}", map_string(attributes)));
            let diagnostics = commits
                .segment_diagnostics(generation, &s.name)?
                .into_iter()
                .map(|(k, v)| {
                    let v = if MASKED_DIAGNOSTICS.contains(&k.as_str()) {
                        "*".to_string()
                    } else {
                        v
                    };
                    (k, v)
                });
            line(out, format!("    diagnostics: {}", map_string(diagnostics)));
            let codec = commits.segment_codec(generation, &s.name)?;
            line(out, format!("    codec: {}", present(codec)));
        }
    }
    Ok(())
}

fn commit(dir: &FsDirectory, path: &str, generation: &str, out: &mut String) -> Result<(), Error> {
    let commits = Commits::new(dir, Path::new(path))?;
    let generation: i64 = generation.parse().map_err(|_| {
        exception(format!(
            "java.lang.NumberFormatException: For input string: \"{generation}\""
        ))
    })?;
    let c = commits.commit(generation).map(|c| {
        format!(
            "Commit{{generation={}, isDeleted={}, segCount={}, userData={}}}",
            c.generation, c.is_deleted, c.seg_count, c.user_data
        )
    });
    line(out, format!("commit {generation}: {}", present(c)));
    line(out, format!("files: {}", commits.files(generation).len()));
    line(
        out,
        format!("segments: {}", commits.segments(generation)?.len()),
    );
    let attributes = commits.segment_attributes(generation, "_0")?;
    line(out, format!("attributes: {}", map_string(attributes)));
    let diagnostics = commits.segment_diagnostics(generation, "_0")?;
    line(out, format!("diagnostics: {}", map_string(diagnostics)));
    let codec = commits.segment_codec(generation, "_0")?;
    line(out, format!("codec: {}", present(codec)));
    Ok(())
}

const SORT_TYPES: [&str; 7] = [
    "INT",
    "LONG",
    "FLOAT",
    "DOUBLE",
    "STRING",
    "STRING_VAL",
    "DOC",
];

fn fields(reader: &DirectoryReader, out: &mut String) -> Result<(), Error> {
    let s = Search::new(reader);
    line(
        out,
        format!("sortable: {}", list_string(s.sortable_field_names())),
    );
    line(
        out,
        format!("searchable: {}", list_string(s.searchable_field_names())),
    );
    line(
        out,
        format!(
            "range searchable: {}",
            list_string(s.range_searchable_field_names())
        ),
    );
    for field in s.field_names() {
        let types = s.guess_sort_types(&field)?;
        line(out, format!("sort types {field}: {}", list_string(types)));
        for ty in SORT_TYPES {
            for reverse in [false, true] {
                if let Some(sf) = s.sort_type(&field, ty, reverse)? {
                    let rev = if reverse { " reverse" } else { "" };
                    line(out, format!("  {ty}{rev}: {sf}"));
                }
            }
        }
    }
    if let Err(e) = s.guess_sort_types("nofield") {
        line(out, format!("nofield: {e}"));
    }
    Ok(())
}

/// What `search`'s options set.
struct SearchOptions {
    field: String,
    parser: QueryParserConfig,
    similarity: SimilarityConfig,
    page_size: i32,
    rewrite: bool,
    exact: bool,
    load: Option<Vec<String>>,
    sort: Option<(String, String)>,
    reverse: bool,
    nav: String,
    explain: i32,
}

fn float(s: &str) -> Result<f32, Error> {
    s.trim().parse().map_err(|_| {
        exception(format!(
            "java.lang.NumberFormatException: For input string: \"{s}\""
        ))
    })
}

/// The options before the expression (`args[0]` is the index).
fn search_options(args: &[String]) -> Result<(SearchOptions, usize), Error> {
    let mut o = SearchOptions {
        field: "text".to_string(),
        parser: QueryParserConfig::default(),
        similarity: SimilarityConfig::default(),
        page_size: 10,
        rewrite: false,
        exact: false,
        load: None,
        sort: None,
        reverse: false,
        nav: String::new(),
        explain: -1,
    };
    let mut i = 1usize;
    while i.saturating_add(1) < args.len() {
        let next = |i: &mut usize| -> Result<String, Error> {
            *i = i.saturating_add(1);
            Ok(arg(args, *i)?.to_string())
        };
        match args[i].as_str() {
            "-field" => o.field = next(&mut i)?,
            "-standard" => o.parser.use_classic_parser = false,
            "-and" => o.parser.default_operator = Operator::And,
            "-slop" => o.parser.phrase_slop = int(&next(&mut i)?)?,
            "-leading-wildcard" => o.parser.allow_leading_wildcard = true,
            "-rewrite" => o.rewrite = true,
            "-classic-sim" => o.similarity.use_classic_similarity = true,
            "-no-discount" => o.similarity.discount_overlaps = false,
            "-k1" => o.similarity.k1 = float(&next(&mut i)?)?,
            "-b" => o.similarity.b = float(&next(&mut i)?)?,
            "-page" => o.page_size = int(&next(&mut i)?)?,
            "-exact" => o.exact = true,
            "-load" => o.load = Some(next(&mut i)?.split(',').map(str::to_string).collect()),
            "-sort" => {
                let field = next(&mut i)?;
                o.sort = Some((field, next(&mut i)?));
            }
            "-reverse" => o.reverse = true,
            "-nav" => o.nav = next(&mut i)?,
            "-explain" => o.explain = int(&next(&mut i)?)?,
            other => {
                return Err(LukeError::illegal_argument(format!("unknown option {other}")).into())
            }
        }
        i = i.saturating_add(1);
    }
    Ok((o, i))
}

fn search(reader: &DirectoryReader, args: &[String], out: &mut String) -> Result<(), Error> {
    let mut s = Search::new(reader);
    let (o, i) = search_options(args)?;
    let expression = arg(args, i)?;
    let query = s.parse_query(
        expression,
        &o.field,
        standard_analyzer(),
        &o.parser,
        o.rewrite,
    )?;
    line(out, format!("query: {}", query.to_string("")));
    let mut sort: Option<LukeSortField> = None;
    if let Some((field, ty)) = &o.sort {
        sort = s.sort_type(field, ty, o.reverse)?;
        line(out, format!("sort: {}", present(sort.as_ref())));
    }
    let r = s.search(
        query.clone(),
        o.similarity,
        sort,
        o.load.clone(),
        o.page_size,
        o.exact,
    )?;
    results("page", &r, out);
    for step in o.nav.split(',').filter(|s| !s.is_empty()) {
        let (label, page) = if step == "n" {
            ("next", s.next_page()?)
        } else {
            ("previous", s.prev_page()?)
        };
        match page {
            Some(page) => results(label, &page, out),
            None => line(out, format!("{label}: --")),
        }
    }
    let current = s
        .current_query()
        .map_or_else(|| "null".to_string(), |q| q.to_string(""));
    line(out, format!("current query: {current}"));
    if o.explain >= 0 {
        line(out, format!("explain {}:", o.explain));
        out.push_str(&s.explain(&query, o.explain)?);
    }
    Ok(())
}

fn results(label: &str, r: &SearchResults, out: &mut String) {
    line(
        out,
        format!(
            "{label}: total={} offset={} size={}",
            r.total_hits,
            r.offset,
            r.hits.len()
        ),
    );
    for d in &r.hits {
        let mut l = format!("  doc={} score={}", d.doc_id, java_float_string(d.score));
        for (k, v) in &d.field_values {
            l.push_str(&format!(" {k}={}", list_string(v)));
        }
        line(out, l);
    }
}

fn mlt(reader: &DirectoryReader, args: &[String], out: &mut String) -> Result<(), Error> {
    let s = Search::new(reader);
    let mut config = MltConfig {
        fields: arg(args, 2)?.split(',').map(str::to_string).collect(),
        ..MltConfig::default()
    };
    if args.len() > 3 {
        config.min_doc_freq = int(arg(args, 3)?)?;
        config.min_term_freq = int(arg(args, 4)?)?;
    }
    let analyzer = standard_analyzer();
    let q = s.mlt_query(int(arg(args, 1)?)?, &config, &analyzer)?;
    line(out, format!("query: {q}"));
    Ok(())
}

fn components(out: &mut String) {
    line(
        out,
        format!(
            "char filters: {}",
            list_string(Analysis::available_char_filters())
        ),
    );
    line(
        out,
        format!(
            "tokenizers: {}",
            list_string(Analysis::available_tokenizers())
        ),
    );
    line(
        out,
        format!(
            "token filters: {}",
            list_string(Analysis::available_token_filters())
        ),
    );
}

/// The `k=v` parameters from `args[from]` on.
fn params(args: &[String], from: usize) -> Vec<(String, String)> {
    args.iter()
        .skip(from)
        .take_while(|a| a.contains('=') && !a.starts_with('-'))
        .map(|a| {
            let (k, v) = a.split_once('=').unwrap_or((a, ""));
            (k.to_string(), v.to_string())
        })
        .collect()
}

fn analyze(args: &[String], out: &mut String) -> Result<(), Error> {
    let mut a = Analysis::new();
    let text = args.last().ok_or_else(|| {
        exception("java.lang.ArrayIndexOutOfBoundsException: Index -1 out of bounds for length 0")
    })?;
    let mut step = false;
    let mut custom: Option<CustomAnalyzerConfig> = None;
    let last = args.len().saturating_sub(1);
    let mut i = 0usize;
    while i < last {
        match args[i].as_str() {
            "-step" => step = true,
            "-analyzer" => {
                i = i.saturating_add(1);
                a.create_analyzer_from_class_name(arg(args, i)?)?;
            }
            flag @ ("-tokenizer" | "-cf" | "-tf") => {
                let p = params(&args[..last], i.saturating_add(2));
                i = i.saturating_add(1);
                let component = ComponentConfig::new(arg(args, i)?, p.clone());
                i = i.saturating_add(p.len());
                match (flag, custom.as_mut()) {
                    ("-tokenizer", _) => custom = Some(CustomAnalyzerConfig::new(component)),
                    ("-cf", Some(c)) => c.char_filters.push(component),
                    (_, Some(c)) => c.token_filters.push(component),
                    (_, None) => {
                        return Err(exception("java.lang.NullPointerException"));
                    }
                }
            }
            other => {
                return Err(LukeError::illegal_argument(format!("unknown option {other}")).into())
            }
        }
        i = i.saturating_add(1);
    }
    if let Some(config) = &custom {
        a.build_custom_analyzer(config)?;
        step = true;
    }
    line(out, format!("analyzer: {}", a.current_analyzer_class()));
    tokens(&a.analyze(text)?, "", out);
    if step {
        let r = a.analyze_step_by_step(text)?;
        for t in &r.charfiltered_texts {
            line(out, format!("char filter {}: {}", t.name, t.text));
        }
        for t in &r.named_tokens {
            line(out, format!("stage {}:", t.name));
            tokens(&t.tokens, "  ", out);
        }
    }
    Ok(())
}

/// The attribute implementations the driver leaves out: which of them a
/// chain adds is not something the port, which holds all of them, knows.
const UNPRINTED_ATTRIBUTES: [&str; 5] = [
    "FlagsAttributeImpl",
    "KeywordAttributeImpl",
    "PayloadAttributeImpl",
    "SentenceAttributeImpl",
    "BoostAttributeImpl",
];

fn tokens(tokens: &[Token], indent: &str, out: &mut String) {
    for t in tokens {
        line(out, format!("{indent}token {}", t.term));
        for att in &t.attributes {
            if UNPRINTED_ATTRIBUTES.contains(&att.att_class.as_str()) {
                continue;
            }
            let mut values = att.att_values.clone();
            values.sort_by(|a, b| java_compare(&a.0, &b.0));
            line(
                out,
                format!("{indent}  {} {}", att.att_class, map_string(values)),
            );
        }
    }
}

fn export(
    reader: &DirectoryReader,
    dir: &FsDirectory,
    args: &[String],
    out: &mut String,
) -> Result<(), Error> {
    let delimiter = match arg(args, 2)? {
        "\\t" => "\t",
        d => d,
    };
    let file = IndexTools::with_reader(reader, dir, false, false).export_terms(
        arg(args, 3)?,
        arg(args, 1)?,
        delimiter,
    )?;
    let path = Path::new(&file);
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    line(out, format!("file: {}", mask_millis(&name)));
    out.push_str(&std::fs::read_to_string(path)?);
    std::fs::remove_file(path)?;
    Ok(())
}

/// `name.replaceAll("_[0-9]+\\.out$", "_{ms}.out")`.
fn mask_millis(name: &str) -> String {
    if let Some(stem) = name.strip_suffix(".out") {
        if let Some(at) = stem.rfind('_') {
            let digits = &stem[at.saturating_add(1)..];
            if !digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit()) {
                return format!("{}_{{ms}}.out", &stem[..at]);
            }
        }
    }
    name.to_string()
}

fn create(args: &[String], out: &mut String) -> Result<(), Error> {
    let path = arg(args, 0)?;
    std::fs::create_dir_all(path)?;
    let dir = FsDirectory::open(path);
    let tools = IndexTools::new(&dir, false, false);
    tools.create_new_index(args.get(1).map(Path::new))?;
    line(out, "created");
    Ok(())
}

/// Copies the files of `from` into `to`, which is created.
fn copy_tree(from: &Path, to: &Path) -> Result<(), Error> {
    std::fs::create_dir_all(to)?;
    for entry in std::fs::read_dir(from)? {
        let entry = entry?;
        std::fs::copy(entry.path(), to.join(entry.file_name()))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn export_file_names_mask_their_millis() {
        assert_eq!(mask_millis("terms_f_123.out"), "terms_f_{ms}.out");
        assert_eq!(mask_millis("terms_f_.out"), "terms_f_.out");
        assert_eq!(mask_millis("terms_f_1x.out"), "terms_f_1x.out");
        assert_eq!(mask_millis("terms.txt"), "terms.txt");
        assert_eq!(mask_millis("terms.out"), "terms.out");
    }

    #[test]
    fn params_stop_at_the_next_option() {
        let args: Vec<String> = ["-tokenizer", "x", "a=1", "b=c=d", "-tf", "y=2"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(
            params(&args, 2),
            vec![
                ("a".to_string(), "1".to_string()),
                ("b".to_string(), "c=d".to_string())
            ]
        );
        assert!(params(&args, 6).is_empty());
    }

    #[test]
    fn strings_compare_by_utf16() {
        use std::cmp::Ordering;
        assert_eq!(java_compare("\u{ff61}", "\u{10000}"), Ordering::Greater);
        assert_eq!(java_compare("a", "b"), Ordering::Less);
    }
}
