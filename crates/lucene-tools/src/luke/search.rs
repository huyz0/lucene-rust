//! `org.apache.lucene.luke.models.search`: parse a query, page through its
//! hits (by score or by a sort), explain a hit, and build a more-like-this
//! query.
//!
//! Differs: Java keeps one `IndexSearcher`; here each search builds one over
//! the reader's segments, with the similarity of the current search. The
//! query parsers run in the default locale and time zone of the port (Java's
//! `QueryParserConfig` takes the JVM's), and a rewrite (`Query.rewrite`) is the
//! port's `Clause::rewrite`: the boolean simplifications, while a multi-term
//! query stays as written, which is how it prints after Java's rewrite too.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use lucene_analysis::Analyzer;
use lucene_codecs::field_infos::{DocValuesType, FieldInfos, IndexOptions};
use lucene_codecs::stored_fields::{DocumentVisitor, FieldValue};
use lucene_search::collector::{ScoreDoc, TopDocsCollector, TotalHits, TotalHitsRelation};
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::field_norms::FieldNorms;
use lucene_search::index_searcher::IndexSearcher;
use lucene_search::mlt::MoreLikeThis;
use lucene_search::query::{BooleanQuery, Clause};
use lucene_search::query_builder::to_query_string;
use lucene_search::queryparser::{classic, flexible, ParsedQuery};
use lucene_search::reader::{leaf_for_doc, merged_field_infos, IndexReader};
use lucene_search::similarities::{Bm25Similarity, ClassicSimilarity, Similarity};
use lucene_search::top_field::{FieldDoc, SortField, SortType};
use lucene_util::geo::{java_double_string, java_float_string};

use super::{LukeError, Result};

/// `Search.DEFAULT_TOTAL_HITS_THRESHOLD`.
const DEFAULT_TOTAL_HITS_THRESHOLD: u64 = 1000;

/// `QueryParserConfig.Operator`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Operator {
    #[default]
    Or,
    And,
}

/// `QueryParserConfig`: how a query expression is parsed. The date
/// resolution, locale, time zone and points type map are left out: the
/// first is `MILLISECOND` (the parsers' own), the next two the port's
/// defaults, the map empty.
#[derive(Debug, Clone, PartialEq)]
pub struct QueryParserConfig {
    pub use_classic_parser: bool,
    pub enable_position_increments: bool,
    pub allow_leading_wildcard: bool,
    pub default_operator: Operator,
    pub fuzzy_min_sim: f32,
    pub fuzzy_prefix_length: i32,
    pub phrase_slop: i32,
    pub auto_generate_multi_term_synonyms_phrase_query: bool,
    pub auto_generate_phrase_queries: bool,
    pub split_on_whitespace: bool,
}

impl Default for QueryParserConfig {
    /// `new QueryParserConfig.Builder().build()`.
    fn default() -> Self {
        QueryParserConfig {
            use_classic_parser: true,
            enable_position_increments: true,
            allow_leading_wildcard: false,
            default_operator: Operator::Or,
            fuzzy_min_sim: 2.0,
            fuzzy_prefix_length: 0,
            phrase_slop: 0,
            auto_generate_multi_term_synonyms_phrase_query: false,
            auto_generate_phrase_queries: false,
            split_on_whitespace: false,
        }
    }
}

/// `SimilarityConfig`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SimilarityConfig {
    pub use_classic_similarity: bool,
    pub k1: f32,
    pub b: f32,
    pub discount_overlaps: bool,
}

impl Default for SimilarityConfig {
    fn default() -> Self {
        SimilarityConfig {
            use_classic_similarity: false,
            k1: 1.2,
            b: 0.75,
            discount_overlaps: true,
        }
    }
}

/// `MLTConfig`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MltConfig {
    pub fields: Vec<String>,
    pub max_doc_freq: i32,
    pub min_doc_freq: i32,
    pub min_term_freq: i32,
}

impl Default for MltConfig {
    /// `MoreLikeThis`' defaults.
    fn default() -> Self {
        MltConfig {
            fields: Vec::new(),
            max_doc_freq: i32::MAX,
            min_doc_freq: 5,
            min_term_freq: 2,
        }
    }
}

/// `SortField.Type`, as Luke names a sort type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SortTypeName {
    Int,
    Long,
    Float,
    Double,
    String,
    StringVal,
    Doc,
}

impl SortTypeName {
    /// `name()`.
    pub fn name(self) -> &'static str {
        match self {
            SortTypeName::Int => "INT",
            SortTypeName::Long => "LONG",
            SortTypeName::Float => "FLOAT",
            SortTypeName::Double => "DOUBLE",
            SortTypeName::String => "STRING",
            SortTypeName::StringVal => "STRING_VAL",
            SortTypeName::Doc => "DOC",
        }
    }

    fn numeric(self) -> Option<SortType> {
        match self {
            SortTypeName::Int => Some(SortType::Int),
            SortTypeName::Long => Some(SortType::Long),
            SortTypeName::Float => Some(SortType::Float),
            SortTypeName::Double => Some(SortType::Double),
            _ => None,
        }
    }
}

/// Which `SortField` class a sort is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SortKind {
    /// `SortField`.
    Plain,
    /// `SortedNumericSortField`, `MIN` selector.
    SortedNumeric,
    /// `SortedSetSortField`, `MIN` selector.
    SortedSet,
}

/// A sort Luke offers on a field.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LukeSortField {
    pub kind: SortKind,
    pub field: String,
    /// The type; a sorted-set sort has none (`STRING`).
    pub ty: SortTypeName,
    pub reverse: bool,
}

impl std::fmt::Display for LukeSortField {
    /// `SortField.toString()` of the class.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let rev = if self.reverse { "!" } else { "" };
        let field = &self.field;
        match self.kind {
            SortKind::SortedSet => write!(f, "<sortedset: \"{field}\">{rev} selector=MIN"),
            SortKind::SortedNumeric => write!(
                f,
                "<sortednumeric: \"{field}\">{rev} selector=MIN type={}",
                self.ty.name()
            ),
            SortKind::Plain => {
                let head = match self.ty {
                    SortTypeName::Doc => "<doc>".to_string(),
                    t => format!("<{}: \"{field}\">", t.name().to_lowercase()),
                };
                write!(f, "{head}{rev}")
            }
        }
    }
}

impl LukeSortField {
    fn new(kind: SortKind, field: &str, ty: SortTypeName, reverse: bool) -> Self {
        LukeSortField {
            kind,
            field: field.to_string(),
            ty,
            reverse,
        }
    }

    /// The key the searcher sorts by.
    fn sort_field(&self) -> SortField {
        let field = self.field.as_str();
        if self.kind == SortKind::SortedSet {
            return SortField::string(field, self.reverse);
        }
        match self.ty {
            SortTypeName::String => SortField::string(field, self.reverse),
            SortTypeName::StringVal => SortField::string_val(field, self.reverse),
            SortTypeName::Doc => {
                let mut f = SortField::doc();
                f.reverse = self.reverse;
                f
            }
            numeric => {
                let ty = numeric.numeric().unwrap_or(SortType::Long);
                SortField::numeric(field, ty, self.reverse)
            }
        }
    }
}

/// `SearchResults.Doc`: a hit and the values of its stored fields (by
/// field name; Java's is a `HashMap`).
#[derive(Debug, Clone, PartialEq)]
pub struct ResultDoc {
    pub doc_id: i32,
    pub score: f32,
    pub field_values: BTreeMap<String, Vec<String>>,
}

/// `SearchResults`: a page of hits.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchResults {
    pub total_hits: TotalHits,
    /// The page's first hit's rank; `0` for an empty page.
    pub offset: i32,
    pub hits: Vec<ResultDoc>,
}

/// A hit kept for paging.
#[derive(Debug, Clone)]
struct Hit {
    doc: i32,
    score: f32,
    /// The sort values of a sorted search's hit (a `FieldDoc`).
    field_doc: Option<FieldDoc>,
}

/// `SearchImpl`.
pub struct Search<'r> {
    reader: &'r DirectoryReader,
    field_infos: FieldInfos,
    page_size: i32,
    current_page: i32,
    total_hits: Option<TotalHits>,
    docs: Vec<Hit>,
    exact_hits_count: bool,
    query: Option<ParsedQuery>,
    sort: Option<LukeSortField>,
    fields_to_load: Option<Vec<String>>,
    similarity: SimilarityConfig,
}

impl<'r> Search<'r> {
    /// `new SearchImpl(reader)`.
    pub fn new(reader: &'r DirectoryReader) -> Self {
        Search {
            reader,
            field_infos: merged_field_infos(&reader.leaves()),
            page_size: 10,
            current_page: -1,
            total_hits: None,
            docs: Vec::new(),
            exact_hits_count: false,
            query: None,
            sort: None,
            fields_to_load: None,
            similarity: SimilarityConfig::default(),
        }
    }

    fn names_where(
        &self,
        keep: impl Fn(&lucene_codecs::field_infos::FieldInfo) -> bool,
    ) -> Vec<String> {
        self.field_infos
            .fields
            .iter()
            .filter(|f| keep(f))
            .map(|f| f.name.clone())
            .collect()
    }

    /// `getFieldNames()`.
    pub fn field_names(&self) -> Vec<String> {
        self.names_where(|_| true)
    }

    /// `getSortableFieldNames()`: the fields with doc values.
    pub fn sortable_field_names(&self) -> Vec<String> {
        self.names_where(|f| f.doc_values_type != DocValuesType::None)
    }

    /// `getSearchableFieldNames()`: the indexed fields.
    pub fn searchable_field_names(&self) -> Vec<String> {
        self.names_where(|f| f.index_options != IndexOptions::None)
    }

    /// `getRangeSearchableFieldNames()`: the fields with points, by name
    /// (Java's is a `HashSet`).
    pub fn range_searchable_field_names(&self) -> Vec<String> {
        let mut names = self.names_where(|f| f.point_dimension_count > 0);
        names.sort();
        names
    }

    /// `getCurrentQuery()`.
    pub fn current_query(&self) -> Option<&ParsedQuery> {
        self.query.as_ref()
    }

    /// `parseQuery(expression, defField, analyzer, config, rewrite)`.
    ///
    /// # Errors
    /// `LukeException` for an expression the parser refuses.
    pub fn parse_query(
        &self,
        expression: &str,
        def_field: &str,
        analyzer: Arc<Analyzer>,
        config: &QueryParserConfig,
        rewrite: bool,
    ) -> Result<ParsedQuery> {
        let parsed = if config.use_classic_parser {
            parse_classic(expression, def_field, analyzer, config)
        } else {
            parse_standard(expression, def_field, analyzer, config)
        }
        .map_err(|_| LukeError::luke(format!("Failed to parse query expression: {expression}")))?;
        Ok(if rewrite {
            ParsedQuery::from_clause(parsed.to_clause(self.reader.max_doc()).rewrite())
        } else {
            parsed
        })
    }

    /// `mltQuery(docid, mltConfig, analyzer)`: the query `toString("")`.
    ///
    /// # Errors
    /// `LukeException` when the document's terms cannot be read.
    pub fn mlt_query(&self, doc: i32, config: &MltConfig, analyzer: &Analyzer) -> Result<String> {
        let mut mlt = MoreLikeThis::new(self.reader);
        mlt.analyzer = Some(analyzer);
        mlt.field_names = Some(config.fields.clone());
        mlt.min_doc_freq = config.min_doc_freq;
        mlt.max_doc_freq = config.max_doc_freq;
        mlt.min_term_freq = config.min_term_freq;
        let q = mlt
            .like_doc(doc)
            .map_err(|_| LukeError::luke(format!("Failed to create MLT query for doc: {doc}")))?;
        Ok(to_query_string(Some(&Clause::Boolean(Box::new(q))), ""))
    }

    /// `search(query, simConfig, sort, fieldsToLoad, pageSize,
    /// exactHitsCount)`: the first page; `sort` `None` sorts by score.
    ///
    /// # Errors
    /// `LukeException` for a negative page size or a failed search.
    pub fn search(
        &mut self,
        query: ParsedQuery,
        similarity: SimilarityConfig,
        sort: Option<LukeSortField>,
        fields_to_load: Option<Vec<String>>,
        page_size: i32,
        exact_hits_count: bool,
    ) -> Result<SearchResults> {
        if page_size < 0 {
            return Err(LukeError::luke_caused_by(
                "java.lang.IllegalArgumentException",
                "Negative integer is not acceptable for page size.",
            ));
        }
        self.docs.clear();
        self.current_page = 0;
        self.page_size = page_size;
        self.exact_hits_count = exact_hits_count;
        self.query = Some(query);
        self.sort = sort;
        self.fields_to_load = fields_to_load;
        self.similarity = similarity;
        self.run_search()
    }

    /// The searcher over the reader, with the current similarity, handed
    /// to `f`.
    fn with_searcher<T>(
        &self,
        f: impl FnOnce(&IndexSearcher<'_, '_>) -> lucene_search::Result<T>,
    ) -> Result<T> {
        fn failed<E>(_: E) -> LukeError {
            LukeError::luke("Search Failed.")
        }
        let opened = self.reader.open_segments().map_err(failed)?;
        let segments = opened.as_open_segments();
        let fields = self.field_names();
        let owned: Vec<HashMap<String, FieldNorms<'_>>> = self.reader.field_norms_by_field(&fields);
        let norms: Vec<Option<&HashMap<String, FieldNorms<'_>>>> = owned.iter().map(Some).collect();
        let mut searcher = IndexSearcher::new(&segments, &norms).map_err(failed)?;
        let s = self.similarity;
        let similarity: Box<dyn Similarity> = if s.use_classic_similarity {
            Box::new(ClassicSimilarity::new(s.discount_overlaps))
        } else {
            Box::new(Bm25Similarity::new(s.k1, s.b, s.discount_overlaps).map_err(LukeError::luke)?)
        };
        searcher.set_similarity(similarity.as_ref());
        f(&searcher).map_err(failed)
    }

    /// The current query as the boolean query the searcher runs; `None`
    /// for one that matches nothing.
    fn boolean_query(&self) -> Option<BooleanQuery> {
        let query = self.query.as_ref()?;
        match query.to_clause(self.reader.max_doc()).rewrite() {
            Clause::MatchNoDocs(_) => None,
            Clause::Boolean(b) => Some(*b),
            other => {
                let mut b = BooleanQuery::new();
                b.must.push(other);
                Some(b)
            }
        }
    }

    /// `search()`: the next page's hits after the last one kept.
    fn run_search(&mut self) -> Result<SearchResults> {
        let n = usize::try_from(self.page_size).unwrap_or(0);
        let after = self.docs.last().cloned();
        let (total, hits) = match self.boolean_query() {
            None => (
                TotalHits {
                    value: 0,
                    relation: TotalHitsRelation::EqualTo,
                },
                Vec::new(),
            ),
            Some(bq) => match &self.sort {
                Some(sort) => {
                    let key = [sort.sort_field()];
                    let after = after.and_then(|a| a.field_doc);
                    let readers = self.reader.segment_readers();
                    let top = self.with_searcher(|s| {
                        s.search_sorted(readers, &bq, n, &key, after.as_ref())
                    })?;
                    let hits = top
                        .hits
                        .into_iter()
                        .map(|h| Hit {
                            doc: h.doc,
                            score: f32::NAN,
                            field_doc: Some(h),
                        })
                        .collect();
                    (top.total, hits)
                }
                None => {
                    let threshold = if self.exact_hits_count {
                        u64::MAX
                    } else {
                        DEFAULT_TOTAL_HITS_THRESHOLD
                    };
                    self.with_searcher(|s| {
                        let mut c = TopDocsCollector::with_total_hits_threshold(n, threshold);
                        if let Some(a) = &after {
                            c = c.with_after(ScoreDoc {
                                doc_id: a.doc,
                                score: a.score,
                            });
                        }
                        s.search_collector(&bq, &mut c)?;
                        let hits = c
                            .top_docs()
                            .iter()
                            .map(|h| Hit {
                                doc: h.doc_id,
                                score: h.score,
                                field_doc: None,
                            })
                            .collect();
                        Ok((c.total_hits(), hits))
                    })?
                }
            },
        };
        self.total_hits = Some(total);
        let offset = self.current_page.saturating_mul(self.page_size);
        let results = self.results(total, &hits, offset)?;
        self.docs.extend(hits);
        Ok(results)
    }

    /// `SearchResults.of`: the hits with their stored fields.
    fn results(&self, total_hits: TotalHits, hits: &[Hit], offset: i32) -> Result<SearchResults> {
        let mut docs = Vec::with_capacity(hits.len());
        for h in hits {
            docs.push(ResultDoc {
                doc_id: h.doc,
                score: h.score,
                field_values: self.field_values(h.doc)?,
            });
        }
        Ok(SearchResults {
            total_hits,
            offset: if docs.is_empty() { 0 } else { offset },
            hits: docs,
        })
    }

    /// `Document.getValues(name)` of each stored field the search loads: a
    /// field's string values, numbers printed; a binary value is none.
    fn field_values(&self, doc: i32) -> Result<BTreeMap<String, Vec<String>>> {
        fn failed<E>(_: E) -> LukeError {
            LukeError::luke("Search Failed.")
        }
        let (leaf, local) = leaf_for_doc(&self.reader.leaves(), doc).map_err(failed)?;
        let mut visitor = DocumentVisitor::all();
        leaf.reader.document(local, &mut visitor).map_err(failed)?;
        let infos = leaf.reader.field_infos();
        let mut values: BTreeMap<String, Vec<String>> = BTreeMap::new();
        for f in visitor.into_document().fields {
            let Some(name) = infos.field_by_number(f.field_number).map(|i| &i.name) else {
                continue;
            };
            if self
                .fields_to_load
                .as_ref()
                .is_some_and(|load| !load.contains(name))
            {
                continue;
            }
            let entry = values.entry(name.clone()).or_default();
            match f.value {
                FieldValue::String(s) => entry.push(s),
                FieldValue::Binary(_) => {}
                FieldValue::Int(i) => entry.push(i.to_string()),
                FieldValue::Long(l) => entry.push(l.to_string()),
                FieldValue::Float(v) => entry.push(java_float_string(v)),
                FieldValue::Double(v) => entry.push(java_double_string(v)),
            }
        }
        Ok(values)
    }

    fn session_started(&self) -> Result<TotalHits> {
        match self.total_hits {
            Some(t) if self.current_page >= 0 && self.query.is_some() => Ok(t),
            _ => Err(LukeError::luke_caused_by(
                "java.lang.IllegalStateException",
                "Search session not started.",
            )),
        }
    }

    /// The kept hits of the current page.
    fn page(&self, total: TotalHits) -> Result<SearchResults> {
        let from = self.current_page.saturating_mul(self.page_size);
        let start = usize::try_from(from).unwrap_or(0).min(self.docs.len());
        let end = start
            .saturating_add(usize::try_from(self.page_size).unwrap_or(0))
            .min(self.docs.len());
        self.results(total, &self.docs[start..end], from)
    }

    /// `nextPage()`: `None` past the last hit.
    ///
    /// # Errors
    /// `LukeException` before a search, or when the search fails.
    pub fn next_page(&mut self) -> Result<Option<SearchResults>> {
        let total = self.session_started()?;
        self.current_page = self.current_page.saturating_add(1);
        let from = i64::from(self.current_page).saturating_mul(i64::from(self.page_size));
        let past_end = total.relation == TotalHitsRelation::EqualTo
            && u64::try_from(from).is_ok_and(|f| f >= total.value);
        if total.value == 0 || past_end {
            return Ok(None);
        }
        if usize::try_from(from).is_ok_and(|f| f < self.docs.len()) {
            self.page(total).map(Some)
        } else {
            self.run_search().map(Some)
        }
    }

    /// `prevPage()`: `None` before the first page.
    ///
    /// # Errors
    /// `LukeException` before a search.
    pub fn prev_page(&mut self) -> Result<Option<SearchResults>> {
        let total = self.session_started()?;
        self.current_page = self.current_page.saturating_sub(1);
        if self.current_page < 0 {
            return Ok(None);
        }
        self.page(total).map(Some)
    }

    /// `guessSortTypes(name)`: the sorts the field's doc values allow.
    ///
    /// # Errors
    /// `LukeException` for a field the index does not have.
    pub fn guess_sort_types(&self, name: &str) -> Result<Vec<LukeSortField>> {
        let Some(fi) = self.field_infos.field_by_name(name) else {
            return Err(LukeError::luke(format!("No such field: {name}")));
        };
        let numeric = [
            SortTypeName::Int,
            SortTypeName::Long,
            SortTypeName::Float,
            SortTypeName::Double,
        ];
        let of = |kind, ty| LukeSortField::new(kind, name, ty, false);
        Ok(match fi.doc_values_type {
            DocValuesType::None => Vec::new(),
            DocValuesType::Numeric => numeric.iter().map(|t| of(SortKind::Plain, *t)).collect(),
            DocValuesType::SortedNumeric => numeric
                .iter()
                .map(|t| of(SortKind::SortedNumeric, *t))
                .collect(),
            DocValuesType::Sorted => vec![
                of(SortKind::Plain, SortTypeName::String),
                of(SortKind::Plain, SortTypeName::StringVal),
            ],
            DocValuesType::SortedSet => vec![of(SortKind::SortedSet, SortTypeName::String)],
            DocValuesType::Binary => vec![of(SortKind::Plain, SortTypeName::Doc)],
        })
    }

    /// `getSortType(name, type, reverse)`: the field's sort of the named
    /// type (any type, for a sorted-set field).
    ///
    /// # Errors
    /// `LukeException` for a field the index does not have.
    pub fn sort_type(&self, name: &str, ty: &str, reverse: bool) -> Result<Option<LukeSortField>> {
        for sf in self.guess_sort_types(name)? {
            if sf.kind == SortKind::SortedSet || sf.ty.name() == ty {
                return Ok(Some(LukeSortField { reverse, ..sf }));
            }
        }
        Ok(None)
    }

    /// `explain(query, docid)`: `Explanation.toString()`.
    ///
    /// # Errors
    /// `LukeException` when the explanation cannot be made.
    pub fn explain(&self, query: &ParsedQuery, doc: i32) -> Result<String> {
        let failed = || {
            LukeError::luke(format!(
                "Failed to create explanation for doc: {doc} for query: \"{}\"",
                query.to_string("")
            ))
        };
        let bq = match query.to_clause(self.reader.max_doc()).rewrite() {
            Clause::Boolean(b) => *b,
            other => {
                let mut b = BooleanQuery::new();
                b.must.push(other);
                b
            }
        };
        self.with_searcher(|s| s.explain(&bq, doc))
            .map(|e| e.to_string())
            .map_err(|_| failed())
    }
}

/// `parseByClassicParser`.
fn parse_classic(
    expression: &str,
    def_field: &str,
    analyzer: Arc<Analyzer>,
    config: &QueryParserConfig,
) -> std::result::Result<ParsedQuery, classic::QueryParserError> {
    let mut parser = classic::QueryParser::new(def_field, analyzer);
    parser.set_default_operator(match config.default_operator {
        Operator::Or => classic::Operator::Or,
        Operator::And => classic::Operator::And,
    });
    parser.set_split_on_whitespace(config.split_on_whitespace)?;
    parser.set_auto_generate_multi_term_synonyms_phrase_query(
        config.auto_generate_multi_term_synonyms_phrase_query,
    );
    parser.set_auto_generate_phrase_queries(config.auto_generate_phrase_queries)?;
    parser.set_enable_position_increments(config.enable_position_increments);
    parser.set_allow_leading_wildcard(config.allow_leading_wildcard);
    parser.set_fuzzy_min_sim(config.fuzzy_min_sim);
    parser.set_fuzzy_prefix_length(config.fuzzy_prefix_length);
    parser.set_phrase_slop(config.phrase_slop);
    parser.parse(expression)
}

/// `parseByStandardParser`.
fn parse_standard(
    expression: &str,
    def_field: &str,
    analyzer: Arc<Analyzer>,
    config: &QueryParserConfig,
) -> std::result::Result<ParsedQuery, classic::QueryParserError> {
    let mut parser = flexible::StandardQueryParser::with_analyzer(analyzer);
    parser.set_default_operator(match config.default_operator {
        Operator::Or => flexible::Operator::Or,
        Operator::And => flexible::Operator::And,
    });
    parser.set_enable_position_increments(config.enable_position_increments);
    parser.set_allow_leading_wildcard(config.allow_leading_wildcard);
    parser.set_fuzzy_min_sim(config.fuzzy_min_sim);
    parser.set_fuzzy_prefix_length(config.fuzzy_prefix_length);
    parser.set_phrase_slop(config.phrase_slop);
    parser.set_points_config_map(Some(HashMap::new()));
    parser.parse(expression, Some(def_field))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sorts_print_as_their_java_classes_do() {
        let s = LukeSortField::new(SortKind::Plain, "f", SortTypeName::StringVal, true);
        assert_eq!(s.to_string(), "<string_val: \"f\">!");
        let s = LukeSortField::new(SortKind::Plain, "f", SortTypeName::Doc, false);
        assert_eq!(s.to_string(), "<doc>");
        assert_eq!(s.sort_field().ty, SortType::Doc);
        let s = LukeSortField::new(SortKind::SortedNumeric, "f", SortTypeName::Float, true);
        assert_eq!(
            s.to_string(),
            "<sortednumeric: \"f\">! selector=MIN type=FLOAT"
        );
        assert_eq!(s.sort_field().ty, SortType::Float);
        let s = LukeSortField::new(SortKind::Plain, "f", SortTypeName::StringVal, false);
        assert_eq!(s.sort_field().ty, SortType::StringVal);
        let s = LukeSortField::new(SortKind::SortedSet, "f", SortTypeName::String, false);
        assert_eq!(s.sort_field().ty, SortType::String);
        assert_eq!(SimilarityConfig::default().k1, 1.2);
        assert_eq!(MltConfig::default().min_doc_freq, 5);
        assert!(QueryParserConfig::default().use_classic_parser);
        assert_eq!(SortTypeName::Double.numeric(), Some(SortType::Double));
        assert_eq!(SortTypeName::Doc.numeric(), None);
    }

    fn fixture() -> DirectoryReader {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../fixtures/data/tools/luke/index");
        let dir = lucene_store::directory::FsDirectory::open(path);
        DirectoryReader::open(&dir as &dyn lucene_store::directory::Directory).unwrap()
    }

    fn analyzer(class: &str) -> Arc<Analyzer> {
        Arc::new(lucene_analysis::factory::spi::new_analyzer(class).unwrap())
    }

    fn standard() -> Arc<Analyzer> {
        analyzer("org.apache.lucene.analysis.standard.StandardAnalyzer")
    }

    /// The session's edges: paging before a search, before the first page,
    /// a query of stop words only, exact counting, an `AND` default
    /// operator through the standard parser, and an explanation that fails.
    #[test]
    fn a_session_pages_and_fails_as_lukes_does() {
        let reader = fixture();
        let mut search = Search::new(&reader);
        let not_started =
            "LukeException: java.lang.IllegalStateException: Search session not started.";
        assert_eq!(search.next_page().unwrap_err().to_string(), not_started);
        assert_eq!(search.prev_page().unwrap_err().to_string(), not_started);

        let config = QueryParserConfig::default();
        let nothing = search
            .parse_query(
                "the",
                "text",
                analyzer("org.apache.lucene.analysis.en.EnglishAnalyzer"),
                &config,
                false,
            )
            .unwrap();
        let page = search
            .search(nothing, SimilarityConfig::default(), None, None, 10, true)
            .unwrap();
        assert_eq!(page.total_hits.value, 0);
        assert!(page.hits.is_empty());

        let config = QueryParserConfig {
            use_classic_parser: false,
            default_operator: Operator::And,
            ..QueryParserConfig::default()
        };
        let query = search
            .parse_query("faust", "title", standard(), &config, false)
            .unwrap();
        let page = search
            .search(
                query.clone(),
                SimilarityConfig::default(),
                None,
                None,
                1,
                true,
            )
            .unwrap();
        assert_eq!(page.total_hits.relation, TotalHitsRelation::EqualTo);
        assert_eq!(search.prev_page().unwrap(), None);

        let e = search.explain(&query, i32::MAX).unwrap_err();
        assert!(
            e.to_string()
                .starts_with("LukeException: Failed to create explanation for doc: "),
            "{e}"
        );
    }
}
