//! Expression sources score the same through their batch path as through the
//! per-document calls: each query runs once with the expression as given and
//! once wrapped in [`PerDocument`], which hides `fill_batch`.

use std::sync::Arc;

use lucene_expressions::{compile, ExpressionValueSource, SimpleBindings};
use lucene_index::buffered_updates::Term;
use lucene_index::document::{self as d, Document, IndexableField, Store};
use lucene_index::index_writer::IndexWriter;
use lucene_index::segment_info::LuceneVersion;
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::explain::Explanation;
use lucene_search::function::{FunctionMatchQuery, FunctionScoreQuery, TopLevel, ValueSource};
use lucene_search::index_searcher::{IndexSearcher, SegmentNorms};
use lucene_search::query::{BooleanQuery, Clause, TermQuery};
use lucene_search::top_docs::TopDocs;
use lucene_search::values_source::{
    self as dvs, BoxDoubleValues, DoubleValues, DoubleValuesSource, ValuesContext,
};
use lucene_search::Result;
use lucene_store::FsDirectory;
use lucene_util::test_support::TempDir;

const VERSION: LuceneVersion = LuceneVersion {
    major: 10,
    minor: 5,
    bugfix: 0,
};

/// Two segments (5 000 and 300 documents), every eleventh document of the
/// first deleted. `i` is sparse (missing on every seventh), `n` and `f`
/// dense, `g` a sparse float column; `body` holds `a` everywhere, `b` on
/// every third (twice on every fifth), `c` on three in seven; `s` a sparse
/// `SORTED` column.
fn index(tmp: &TempDir) -> DirectoryReader {
    let dir = FsDirectory::open(tmp.path());
    let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", VERSION).unwrap();
    for (seg, count) in [(0, 5000i64), (1, 300)] {
        for i in 0..count {
            let mut doc = Document::new();
            let mut body = String::from("a");
            if i % 3 == 0 {
                body.push_str(" b");
                if i % 5 == 0 {
                    body.push_str(" b");
                }
            }
            if i % 7 < 3 {
                body.push_str(" c");
            }
            let fields: Vec<Box<dyn IndexableField>> = vec![
                Box::new(d::StringField::new("id", format!("{seg}-{i}"), Store::No)),
                Box::new(d::TextField::new("body", body, Store::No)),
                Box::new(d::NumericDocValuesField::new("n", i * 7919 % 100_000)),
                Box::new(d::NumericDocValuesField::new(
                    "f",
                    i64::from(((i * 13 % 997) as f32 / 10.0 - 5.0).to_bits()),
                )),
            ];
            for f in fields {
                doc.add_boxed(f);
            }
            if i % 7 != 0 {
                doc.add_boxed(Box::new(d::NumericDocValuesField::new("i", i * 37 % 1000)));
            }
            if i % 5 != 0 {
                doc.add_boxed(Box::new(d::SortedDocValuesField::new(
                    "s",
                    format!("w{}", i % 13).into_bytes(),
                )));
            }
            if i % 4 != 0 {
                doc.add_boxed(Box::new(d::NumericDocValuesField::new(
                    "g",
                    i64::from(((i % 50) as f32 * 1.5).to_bits()),
                )));
            }
            w.add_fields_document(&doc).unwrap();
        }
        w.commit().unwrap();
    }
    let deleted: Vec<Term> = (0..5000)
        .step_by(11)
        .map(|i| Term::new("id", format!("0-{i}")))
        .collect();
    w.delete_documents_by_term(&deleted).unwrap();
    w.commit().unwrap();
    drop(w);
    DirectoryReader::open(&dir).unwrap()
}

/// A source's values without the batch path: `batch_capable` is `false`, so
/// a search reads them one document at a time.
struct PerDocument(Arc<dyn DoubleValuesSource>);

struct PerDocumentValues<'c>(BoxDoubleValues<'c>);

impl DoubleValues for PerDocumentValues<'_> {
    fn advance_exact(&mut self, doc: i32) -> Result<bool> {
        self.0.advance_exact(doc)
    }
    fn double_value(&mut self) -> Result<f64> {
        self.0.double_value()
    }
}

impl DoubleValuesSource for PerDocument {
    fn get_values<'c>(
        &self,
        ctx: &ValuesContext<'c>,
        leaf: usize,
        scores: Option<BoxDoubleValues<'c>>,
    ) -> Result<BoxDoubleValues<'c>> {
        Ok(Box::new(PerDocumentValues(
            self.0.get_values(ctx, leaf, scores)?,
        )))
    }
    fn needs_scores(&self) -> bool {
        self.0.needs_scores()
    }
    fn needs_searcher(&self) -> bool {
        self.0.needs_searcher()
    }
    fn is_cacheable(&self, ctx: &ValuesContext<'_>, leaf: usize) -> bool {
        self.0.is_cacheable(ctx, leaf)
    }
    fn describe(&self) -> String {
        self.0.describe()
    }
    fn rewrite(&self, top: &TopLevel<'_>) -> Result<Option<Arc<dyn DoubleValuesSource>>> {
        Ok(self
            .0
            .rewrite(top)?
            .map(|r| Arc::new(PerDocument(r)) as Arc<dyn DoubleValuesSource>))
    }
    fn queries(&self) -> Vec<&Clause> {
        self.0.queries()
    }
    fn wrapped_value_source(&self) -> Option<Arc<dyn ValueSource>> {
        self.0.wrapped_value_source()
    }
    fn explain(
        &self,
        ctx: &ValuesContext<'_>,
        leaf: usize,
        doc: i32,
        score_explanation: &Explanation,
    ) -> Result<Explanation> {
        self.0.explain(ctx, leaf, doc, score_explanation)
    }
}

fn term(w: &str) -> Clause {
    Clause::Term(TermQuery::new("body", w.as_bytes().to_vec()))
}

/// The queries over `source(text, caching)`, for each term and caching mode.
fn queries(source: &dyn Fn(&str, bool) -> Arc<dyn DoubleValuesSource>) -> Vec<BooleanQuery> {
    let one = |c: Clause| BooleanQuery {
        must: vec![c],
        ..Default::default()
    };
    let mut out = Vec::new();
    for w in ["a", "b", "c"] {
        for caching in [false, true] {
            out.push(one(FunctionScoreQuery::new(
                term(w),
                source(
                    "sqrt(_score) + ln(1 + abs(i)) * g + (g > 5 ? nested : i)",
                    caching,
                ),
            )
            .into()));
            out.push(one(FunctionScoreQuery::boost_by_value(
                term(w),
                source("abs(i) * 0.5 + g", caching),
            )
            .into()));
            out.push(BooleanQuery {
                must: vec![term(w)],
                filter: vec![FunctionMatchQuery::new(
                    source("i + g", caching),
                    Arc::new(|v| v > 300.0),
                )
                .into()],
                ..Default::default()
            });
        }
    }
    out
}

fn run(searcher: &IndexSearcher<'_, '_>, q: &BooleanQuery, n: usize) -> String {
    match searcher.search(q, n) {
        Ok(TopDocs {
            total_hits,
            score_docs,
        }) => {
            let hits: Vec<(i32, u32)> = score_docs
                .iter()
                .map(|d| (d.doc, d.score.to_bits()))
                .collect();
            format!("{total_hits:?} {hits:?}")
        }
        Err(e) => format!("error {e}"),
    }
}

#[test]
fn expressions_score_the_same_with_and_without_batches() {
    let tmp = TempDir::new("expressions-batches");
    let reader = index(&tmp);
    let opened = reader.open_segments().unwrap();
    let mut segments = opened.as_open_segments();
    for seg in &mut segments {
        seg.cache = None;
    }
    let owned = reader.field_norms_by_field(&["body".to_string()]).unwrap();
    let norms: Vec<SegmentNorms<'_, '_>> = owned.iter().map(Some).collect();
    let searcher = IndexSearcher::new(&segments, &norms).unwrap();

    let mut bindings = SimpleBindings::new();
    bindings.add_source("i", dvs::from_int_field("i"));
    bindings.add_source("g", dvs::from_float_field("g"));
    bindings.add_source("_score", dvs::scores());
    bindings.add_expression("nested", Arc::new(compile("i * _score + g").unwrap()));
    let expr = |text: &str, caching: bool| -> Arc<dyn DoubleValuesSource> {
        let e = Arc::new(compile(text).unwrap());
        Arc::new(if caching {
            ExpressionValueSource::caching(&bindings, e).unwrap()
        } else {
            ExpressionValueSource::new(&bindings, e).unwrap()
        })
    };
    let per_doc = |text: &str, caching: bool| -> Arc<dyn DoubleValuesSource> {
        Arc::new(PerDocument(expr(text, caching)))
    };
    for (q, p) in queries(&expr).iter().zip(queries(&per_doc).iter()) {
        for n in [1, 10, 2000] {
            let batched = run(&searcher, q, n);
            let one_by_one = run(&searcher, p, n);
            assert_eq!(batched, one_by_one, "{q:?} top {n}");
            assert!(!batched.starts_with("error"), "{q:?}: {batched}");
        }
    }
}
