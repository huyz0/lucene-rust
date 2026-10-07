//! Differential test for `QueryBuilder` against Lucene 10.5.0:
//! `fixtures/src/GenQueryBuilder.java` records, for `StandardAnalyzer` text
//! and for canned token graphs (stacked synonyms, multi-token synonyms
//! spanning positions, holes), the query `QueryBuilder` builds -- its
//! `toString("body")` -- and its top 10 hits (score bits) and total over
//! one Java-written segment. This builds the same queries with
//! `lucene_search::query_builder::QueryBuilder`, compares the strings, and
//! runs them through `IndexSearcher`.

use std::collections::HashMap;

use lucene_analysis::{AnalysisError, Analyzer, AttributeSource, TokenStream};
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::field_norms::FieldNorms;
use lucene_search::index_searcher::IndexSearcher;
use lucene_search::query::{BooleanQuery, Clause};
use lucene_search::query_builder::{to_query_string, QueryBuilder};
use lucene_search::query_visitor::Occur;
use lucene_store::FsDirectory;

fn dir() -> String {
    concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/data/query_builder_index"
    )
    .to_string()
}

/// `GenQueryBuilder.Canned`: tokens `term:posInc:posLen`, offsets by index.
struct Canned {
    atts: AttributeSource,
    tokens: Vec<(String, i32, i32)>,
    upto: usize,
}

impl Canned {
    fn new(spec: &str) -> Self {
        let tokens = spec
            .split(',')
            .map(|t| {
                let p: Vec<&str> = t.split(':').collect();
                (
                    p[0].to_string(),
                    p[1].parse().unwrap(),
                    p[2].parse().unwrap(),
                )
            })
            .collect();
        Canned {
            atts: AttributeSource::new(),
            tokens,
            upto: 0,
        }
    }
}

impl TokenStream for Canned {
    /// A source, not a wrapper: no conditional wrapper below it.
    fn conditional_root(&mut self) -> Option<&mut dyn std::any::Any> {
        None
    }
    fn attributes(&self) -> &AttributeSource {
        &self.atts
    }
    fn attributes_mut(&mut self) -> &mut AttributeSource {
        &mut self.atts
    }
    fn increment_token(&mut self) -> Result<bool, AnalysisError> {
        let Some((t, inc, len)) = self.tokens.get(self.upto).cloned() else {
            return Ok(false);
        };
        self.atts.clear_attributes();
        self.atts.set_term(&t);
        self.atts.set_position_increment(inc)?;
        self.atts.set_position_length(len)?;
        self.atts
            .set_offset(self.upto as i32, self.upto as i32 + 1)?;
        self.upto += 1;
        Ok(true)
    }
    fn reset(&mut self) -> Result<(), AnalysisError> {
        self.upto = 0;
        Ok(())
    }
}

/// `GenQueryBuilder.build`.
fn build(analyzer: &Analyzer, method: &str, input: &str) -> Option<Clause> {
    let mut b = QueryBuilder::new(analyzer);
    let canned = method.starts_with('C');
    let mut m = if canned { &method[1..] } else { method };
    if let Some(rest) = m.strip_suffix('A') {
        b.set_auto_generate_multi_term_synonyms_phrase_query(true);
        m = rest;
    }
    if let Some(rest) = m.strip_suffix('G') {
        b.set_enable_graph_queries(false);
        m = rest;
    }
    if let Some(rest) = m.strip_suffix('I') {
        b.set_enable_position_increments(false);
        m = rest;
    }
    if m == "B" || m == "BM" {
        let op = if m == "B" { Occur::Should } else { Occur::Must };
        return if canned {
            b.create_field_query(&mut Canned::new(input), op, "body", false, 0)
        } else {
            b.create_boolean_query_with("body", input, op)
        }
        .unwrap();
    }
    if let Some(slop) = m.strip_prefix('P') {
        let slop: u32 = slop.parse().unwrap();
        return if canned {
            b.create_field_query(&mut Canned::new(input), Occur::Must, "body", true, slop)
        } else {
            b.create_phrase_query_with_slop("body", input, slop)
        }
        .unwrap();
    }
    let fraction: f32 = m.strip_prefix('M').expect("method").parse().unwrap();
    if !canned {
        return b
            .create_min_should_match_query("body", input, fraction)
            .unwrap();
    }
    let q = b
        .create_field_query(&mut Canned::new(input), Occur::Should, "body", false, 0)
        .unwrap();
    q.map(|q| match q {
        Clause::Boolean(mut bq) => {
            bq.minimum_should_match = (fraction * bq.should.len() as f32) as usize;
            Clause::Boolean(bq)
        }
        other => other,
    })
}

#[test]
fn query_builder_matches_lucene() {
    let dir_path = dir();
    let reader = DirectoryReader::open(&FsDirectory::open(format!("{dir_path}/index"))).unwrap();
    let opened = reader.open_segments().unwrap();
    let segments = opened.as_open_segments();
    let owned: Vec<HashMap<String, FieldNorms<'_>>> = reader
        .field_norms("body")
        .into_iter()
        .map(|n| n.into_iter().map(|n| ("body".to_string(), n)).collect())
        .collect();
    let norms: Vec<Option<&HashMap<String, FieldNorms<'_>>>> = owned.iter().map(Some).collect();
    let searcher = IndexSearcher::new(&segments, &norms).unwrap();
    let analyzer = Analyzer::standard(None);

    let cases = std::fs::read_to_string(format!("{dir_path}/cases.tsv")).unwrap();
    let mut failures = Vec::new();
    let mut checked = 0;
    for line in cases.lines() {
        let cols: Vec<&str> = line.split('\t').collect();
        let (method, input, want_string, want_hits, want_total) =
            (cols[0], cols[1], cols[2], cols[3], cols[4]);
        let q = build(&analyzer, method, input);
        let got_string = to_query_string(q.as_ref(), "body");
        if got_string != want_string {
            failures.push(format!(
                "{method} {input:?}: query {got_string}, Lucene {want_string}"
            ));
            continue;
        }
        checked += 1;
        let Some(q) = q else { continue };
        // `IndexSearcher.search` rewrites first (a graph query's nested
        // disjunction is flattened into its parent, which changes how its
        // scores are summed); this crate's searches take rewritten queries.
        let bq = match q.rewrite() {
            Clause::Boolean(b) => *b,
            other => {
                let mut b = BooleanQuery::new();
                b.must.push(other);
                b
            }
        };
        let top = searcher.search(&bq, 10).unwrap();
        let got_hits: Vec<String> = top
            .score_docs
            .iter()
            .map(|h| format!("{}:{:x}", h.doc, h.score.to_bits()))
            .collect();
        let got_hits = if got_hits.is_empty() {
            "-".to_string()
        } else {
            got_hits.join(",")
        };
        if got_hits != want_hits || top.total_hits.value.to_string() != want_total {
            failures.push(format!(
                "{method} {input:?} ({want_string}): hits {got_hits} total {}, Lucene {want_hits} total {want_total}",
                top.total_hits.value
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    assert!(checked >= 29, "cases checked: {checked}");
}
