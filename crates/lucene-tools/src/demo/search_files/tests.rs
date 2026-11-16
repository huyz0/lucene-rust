#![cfg(test)]
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
