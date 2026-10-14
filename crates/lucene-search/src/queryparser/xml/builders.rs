//! `org.apache.lucene.queryparser.xml.builders`: one builder per query
//! element, as `CoreParser` registers them. A query builder returns a
//! query; a span builder returns a span query, and as a query it is wrapped
//! in a `BoostQuery` when its boost is not 1 (`SpanBuilderBase.getQuery`).
//!
//! Java's `DOMUtils.getAttribute(e, name, default)` reads `""` for an absent
//! attribute and falls back to the default for an empty one, so a number or
//! a boolean is parsed only when present and non-empty; a string attribute
//! read with `getAttribute(name)` is `""` when absent, not `null`.

use std::sync::Arc;

use lucene_analysis::attributes::java_float_to_string;
use lucene_analysis::{Analyzer, TokenStream};

use crate::extended_query::TermRangeQuery;
use crate::function::{parse_java_double, parse_java_float, parse_java_int, parse_java_long};
use crate::query_visitor::Occur;
use crate::queryparser::classic::{QueryParser, QueryParserError};
use crate::queryparser::xml::dom::Element;
use crate::queryparser::PointKind;
use crate::queryparser::{ParsedBoolean, ParsedQuery};
use crate::spans::payloads::{PayloadDecoder, PayloadFunction, PayloadScoreQuery};
use crate::spans::SpanNode;
use lucene_util::numeric_utils::{double_to_sortable_long, float_to_sortable_int};

use super::PARSER_EXCEPTION;

/// The elements from the document element down to a builder's element,
/// its parents last: what `getAttributeWithInheritance` climbs.
pub(crate) type Ancestors<'x> = Vec<&'x Element>;

/// `ParserException(message)`.
pub(crate) fn parser_error(message: impl Into<String>) -> QueryParserError {
    QueryParserError::runtime_class(PARSER_EXCEPTION, message)
}

/// `NumberFormatException`, from a `parseInt`/`parseFloat` refusal.
fn number_format(e: crate::Error) -> QueryParserError {
    QueryParserError::runtime_class("java.lang.NumberFormatException", error_text(e))
}

fn error_text(e: crate::Error) -> String {
    match e {
        crate::Error::IllegalArgument(m) => m,
        other => other.to_string(),
    }
}

/// `DOMUtils.getAttributeWithInheritance`: the first non-empty value of
/// `name` on the element or an ancestor.
fn inherited<'x>(e: &'x Element, anc: &[&'x Element], name: &str) -> Option<String> {
    let own = e.attribute(name);
    if !own.is_empty() {
        return Some(own.to_string());
    }
    anc.iter()
        .rev()
        .map(|p| p.attribute(name))
        .find(|v| !v.is_empty())
        .map(str::to_string)
}

/// `getAttributeWithInheritanceOrFail`.
fn inherited_or_fail<'x>(
    e: &'x Element,
    anc: &[&'x Element],
    name: &str,
) -> Result<String, QueryParserError> {
    inherited(e, anc, name)
        .ok_or_else(|| parser_error(format!("{} missing \"{name}\" attribute", e.name)))
}

/// `DOMUtils.getAttribute(e, name, float)`.
fn float_attr(e: &Element, name: &str, default: f32) -> Result<f32, QueryParserError> {
    match e.attribute(name) {
        "" => Ok(default),
        v => parse_java_float(v).map_err(number_format),
    }
}

/// `DOMUtils.getAttribute(e, name, int)`.
fn int_attr(e: &Element, name: &str, default: i32) -> Result<i32, QueryParserError> {
    match e.attribute(name) {
        "" => Ok(default),
        v => parse_java_int(v).map_err(number_format),
    }
}

/// `DOMUtils.getAttribute(e, name, boolean)`: `Boolean.valueOf`.
fn bool_attr(e: &Element, name: &str, default: bool) -> bool {
    match e.attribute(name) {
        "" => default,
        v => v.eq_ignore_ascii_case("true"),
    }
}

/// `DOMUtils.getNonBlankTextOrFail`: the text, trimmed of what Java's
/// `String.trim` strips (`<= ' '`), or the failure when nothing is left.
fn non_blank_text(e: &Element) -> Result<String, QueryParserError> {
    let v = e.text();
    let v = v.trim_matches(|c: char| c <= ' ');
    if v.is_empty() {
        return Err(parser_error(format!("{} has no text", e.name)));
    }
    Ok(v.to_string())
}

/// `getChildByTagOrFail`.
fn child_by_tag_or_fail<'a>(e: &'a Element, name: &str) -> Result<&'a Element, QueryParserError> {
    e.elements()
        .find(|c| c.name == name)
        .ok_or_else(|| parser_error(format!("{} missing \"{name}\" child element", e.name)))
}

/// `getFirstChildOrFail`.
fn first_child_or_fail(e: &Element) -> Result<&Element, QueryParserError> {
    e.elements()
        .next()
        .ok_or_else(|| parser_error(format!("{} does not contain a child element", e.name)))
}

/// `new BoostQuery(query, boost)`'s check: finite and not negative.
fn boosted(q: ParsedQuery, boost: f32) -> Result<ParsedQuery, QueryParserError> {
    if !boost.is_finite() || boost < 0.0 {
        return Err(QueryParserError::illegal_argument(format!(
            "boost must be a positive float, got {}",
            java_float_to_string(boost)
        )));
    }
    Ok(ParsedQuery::Boost(Box::new(q), boost))
}

/// The builder-level `if (boost != 1f) new BoostQuery(q, boost)`.
fn boosted_unless_one(q: ParsedQuery, boost: f32) -> Result<ParsedQuery, QueryParserError> {
    if boost != 1.0 {
        boosted(q, boost)
    } else {
        Ok(q)
    }
}

/// `analyzer.tokenStream(field, text)`'s terms, in order.
fn analyzed_terms(
    analyzer: &Analyzer,
    field: &str,
    text: &str,
) -> Result<Vec<Vec<u8>>, QueryParserError> {
    let mut ts = analyzer
        .token_stream(field, text)
        .map_err(QueryParserError::from_analysis)?;
    ts.reset().map_err(QueryParserError::from_analysis)?;
    let mut terms = Vec::new();
    while ts
        .increment_token()
        .map_err(QueryParserError::from_analysis)?
    {
        terms.push(ts.attributes().term_bytes().to_vec());
    }
    ts.end().map_err(QueryParserError::from_analysis)?;
    Ok(terms)
}

/// Every query and span builder of one `CoreParser`, with the analyzer and
/// the `UserQuery` parser it was given.
pub(crate) struct Builders {
    pub(crate) default_field: Option<String>,
    pub(crate) analyzer: Arc<Analyzer>,
    /// `new UserInputQueryBuilder(parser)`: the one parser every `UserQuery`
    /// uses. `None` makes a parser per query, on the query's field.
    pub(crate) parser: Option<QueryParser>,
}

impl Builders {
    /// `QueryBuilderFactory.getQuery`: the builder of `e`'s name, applied.
    pub(crate) fn query<'x>(
        &self,
        e: &'x Element,
        anc: &Ancestors<'x>,
    ) -> Result<ParsedQuery, QueryParserError> {
        match e.name.as_str() {
            "TermQuery" => self.term_query(e, anc),
            "TermsQuery" => self.terms_query(e, anc),
            "MatchAllDocsQuery" => Ok(ParsedQuery::MatchAll),
            "BooleanQuery" => self.boolean_query(e, anc),
            "RangeQuery" => self.range_query(e, anc),
            "PointRangeQuery" => self.point_range_query(e, anc),
            "DisjunctionMaxQuery" => self.disjunction_max_query(e, anc),
            "UserQuery" => self.user_query(e),
            "ConstantScoreQuery" => self.constant_score_query(e, anc),
            name if is_span(name) => {
                // `SpanBuilderBase.getQuery` reads the boost first
                let boost = float_attr(e, "boost", 1.0)?;
                boosted_unless_one(ParsedQuery::SpanNode(self.span(e, anc)?), boost)
            }
            name => Err(parser_error(format!(
                "No QueryObjectBuilder defined for node {name}"
            ))),
        }
    }

    /// `SpanQueryBuilderFactory.getSpanQuery`: the span builder of `e`'s name.
    pub(crate) fn span<'x>(
        &self,
        e: &'x Element,
        anc: &Ancestors<'x>,
    ) -> Result<SpanNode, QueryParserError> {
        match e.name.as_str() {
            "SpanNear" => self.span_near(e, anc),
            "BoostingTermQuery" => self.boosting_term(e, anc),
            "SpanTerm" => self.span_term(e, anc),
            "SpanOr" => self.span_or(e, anc),
            "SpanOrTerms" => self.span_or_terms(e, anc),
            "SpanFirst" => self.span_first(e, anc),
            "SpanPositionRange" => self.span_position_range(e, anc),
            "SpanNot" => self.span_not(e, anc),
            name => Err(parser_error(format!(
                "No SpanQueryObjectBuilder defined for node {name}"
            ))),
        }
    }

    fn child_anc<'x>(anc: &Ancestors<'x>, e: &'x Element) -> Ancestors<'x> {
        let mut next = anc.clone();
        next.push(e);
        next
    }

    fn term_query<'x>(
        &self,
        e: &'x Element,
        anc: &Ancestors<'x>,
    ) -> Result<ParsedQuery, QueryParserError> {
        let field = inherited_or_fail(e, anc, "fieldName")?;
        let value = non_blank_text(e)?;
        let tq = ParsedQuery::Term(crate::query::TermQuery::new(field, value.into_bytes()));
        boosted_unless_one(tq, float_attr(e, "boost", 1.0)?)
    }

    fn terms_query<'x>(
        &self,
        e: &'x Element,
        anc: &Ancestors<'x>,
    ) -> Result<ParsedQuery, QueryParserError> {
        let field = inherited_or_fail(e, anc, "fieldName")?;
        let text = non_blank_text(e)?;
        let msm = int_attr(e, "minimumNumberShouldMatch", 0)?;
        let clauses = analyzed_terms(&self.analyzer, &field, &text)?
            .into_iter()
            .map(|t| {
                (
                    Occur::Should,
                    ParsedQuery::Term(crate::query::TermQuery::new(field.clone(), t)),
                )
            })
            .collect();
        let bq = ParsedQuery::Boolean(ParsedBoolean {
            clauses,
            minimum_should_match: usize_msm(msm)?,
        });
        // `BoostQuery(bq.build(), boost)`, whatever the boost
        boosted(bq, float_attr(e, "boost", 1.0)?)
    }

    fn boolean_query<'x>(
        &self,
        e: &'x Element,
        anc: &Ancestors<'x>,
    ) -> Result<ParsedQuery, QueryParserError> {
        let msm = int_attr(e, "minimumNumberShouldMatch", 0)?;
        let next = Self::child_anc(anc, e);
        let mut clauses = Vec::new();
        for clause in e.elements().filter(|c| c.name == "Clause") {
            let occur = occurs(clause)?;
            let inner = first_child_or_fail(clause)?;
            clauses.push((occur, self.query(inner, &Self::child_anc(&next, clause))?));
        }
        let bq = ParsedQuery::Boolean(ParsedBoolean {
            clauses,
            minimum_should_match: usize_msm(msm)?,
        });
        boosted_unless_one(bq, float_attr(e, "boost", 1.0)?)
    }

    fn range_query<'x>(
        &self,
        e: &'x Element,
        anc: &Ancestors<'x>,
    ) -> Result<ParsedQuery, QueryParserError> {
        // a missing field is a null one: `MultiTermQuery` refuses it
        let field = inherited(e, anc, "fieldName").ok_or_else(|| {
            QueryParserError::runtime_class(
                "java.lang.NullPointerException",
                "field must not be null",
            )
        })?;
        let lower = e.attribute("lowerTerm").as_bytes().to_vec();
        let upper = e.attribute("upperTerm").as_bytes().to_vec();
        let include_lower = bool_attr(e, "includeLower", true);
        let include_upper = bool_attr(e, "includeUpper", true);
        Ok(ParsedQuery::TermRange(TermRangeQuery::new(
            field,
            Some(lower),
            Some(upper),
            include_lower,
            include_upper,
        )))
    }

    fn point_range_query<'x>(
        &self,
        e: &'x Element,
        anc: &Ancestors<'x>,
    ) -> Result<ParsedQuery, QueryParserError> {
        let field = inherited_or_fail(e, anc, "fieldName")?;
        // `getAttribute(e, name, null)`: absent or empty is no bound
        let bound = |name: &str| match e.attribute(name) {
            "" => None,
            v => Some(v.to_string()),
        };
        let (lower, upper) = (bound("lowerTerm"), bound("upperTerm"));
        let ty = match e.attribute("type") {
            "" => "int".to_string(),
            v => v.to_lowercase(),
        };
        // a bound that does not parse is a `ParserException`, not a
        // `NumberFormatException`
        fn bound_of<T>(
            v: &Option<String>,
            parse: fn(&str) -> crate::Result<T>,
            default: T,
        ) -> Result<T, QueryParserError> {
            match v {
                None => Ok(default),
                Some(s) => parse(s).map_err(|_| {
                    parser_error("Could not parse lowerTerm or upperTerm into a number")
                }),
            }
        }
        let (query, kind) = match ty.as_str() {
            "int" => {
                let lo = bound_of(&lower, parse_java_int, i32::MIN)?;
                let hi = bound_of(&upper, parse_java_int, i32::MAX)?;
                (
                    crate::extended_query::PointRangeQuery::int_range(field, &[lo], &[hi]),
                    PointKind::Int,
                )
            }
            "long" => {
                let lo = bound_of(&lower, parse_java_long, i64::MIN)?;
                let hi = bound_of(&upper, parse_java_long, i64::MAX)?;
                (
                    crate::extended_query::PointRangeQuery::long_range(field, &[lo], &[hi]),
                    PointKind::Long,
                )
            }
            "float" => {
                let lo = bound_of(&lower, parse_java_float, f32::NEG_INFINITY)?;
                let hi = bound_of(&upper, parse_java_float, f32::INFINITY)?;
                (
                    crate::extended_query::PointRangeQuery::int_range(
                        field,
                        &[float_to_sortable_int(lo)],
                        &[float_to_sortable_int(hi)],
                    ),
                    PointKind::Float,
                )
            }
            "double" => {
                let lo = bound_of(&lower, parse_java_double, f64::NEG_INFINITY)?;
                let hi = bound_of(&upper, parse_java_double, f64::INFINITY)?;
                (
                    crate::extended_query::PointRangeQuery::long_range(
                        field,
                        &[double_to_sortable_long(lo)],
                        &[double_to_sortable_long(hi)],
                    ),
                    PointKind::Double,
                )
            }
            _ => {
                return Err(parser_error(
                    "type attribute must be one of: [long, int, double, float]",
                ))
            }
        };
        Ok(ParsedQuery::PointRange {
            query: query.map_err(QueryParserError::from_search)?,
            kind,
        })
    }

    // `!(0.0..=1.0).contains(&t)` would refuse NaN, which Java's
    // `t < 0 || t > 1` accepts
    #[allow(clippy::manual_range_contains)]
    fn disjunction_max_query<'x>(
        &self,
        e: &'x Element,
        anc: &Ancestors<'x>,
    ) -> Result<ParsedQuery, QueryParserError> {
        let tie_breaker = float_attr(e, "tieBreaker", 0.0)?;
        let next = Self::child_anc(anc, e);
        let disjuncts = e
            .elements()
            .map(|d| self.query(d, &next))
            .collect::<Result<Vec<_>, _>>()?;
        // `tieBreakerMultiplier < 0 || > 1`: NaN passes, as in Java
        if tie_breaker < 0.0 || tie_breaker > 1.0 {
            return Err(QueryParserError::illegal_argument(
                "tieBreakerMultiplier must be in [0, 1]",
            ));
        }
        boosted_unless_one(
            ParsedQuery::DisjunctionMax {
                disjuncts,
                tie_breaker,
            },
            float_attr(e, "boost", 1.0)?,
        )
    }

    fn user_query(&self, e: &Element) -> Result<ParsedQuery, QueryParserError> {
        let text = e.text();
        let q = match &self.parser {
            Some(parser) => parser.parse(&text),
            None => {
                let field = match e.attribute("fieldName") {
                    "" => self.default_field.clone(),
                    f => Some(f.to_string()),
                }
                .unwrap_or_default();
                QueryParser::new(field, Arc::clone(&self.analyzer)).parse(&text)
            }
        }
        .map_err(|err| {
            // `catch (ParseException e1)`: any other exception goes through
            if err.class.ends_with("ParseException") {
                parser_error(err.message)
            } else {
                err
            }
        })?;
        // `new BoostQuery(q, boost)` even for a boost of 1
        boosted(q, float_attr(e, "boost", 1.0)?)
    }

    fn constant_score_query<'x>(
        &self,
        e: &'x Element,
        anc: &Ancestors<'x>,
    ) -> Result<ParsedQuery, QueryParserError> {
        let next = Self::child_anc(anc, e);
        let inner = self.query(first_child_or_fail(e)?, &next)?;
        boosted_unless_one(
            ParsedQuery::ConstantScore(Box::new(inner)),
            float_attr(e, "boost", 1.0)?,
        )
    }

    fn span_near<'x>(
        &self,
        e: &'x Element,
        anc: &Ancestors<'x>,
    ) -> Result<SpanNode, QueryParserError> {
        let slop = parse_int(e.attribute("slop"))?;
        let in_order = bool_attr(e, "inOrder", false);
        let next = Self::child_anc(anc, e);
        let clauses = e
            .elements()
            .map(|c| self.span(c, &next))
            .collect::<Result<Vec<_>, _>>()?;
        SpanNode::near(clauses, slop, in_order).map_err(QueryParserError::from_search)
    }

    fn boosting_term<'x>(
        &self,
        e: &'x Element,
        anc: &Ancestors<'x>,
    ) -> Result<SpanNode, QueryParserError> {
        let field = inherited_or_fail(e, anc, "fieldName")?;
        let value = non_blank_text(e)?;
        Ok(SpanNode::PayloadScore(Box::new(PayloadScoreQuery::new(
            SpanNode::term(field, value.into_bytes()),
            PayloadFunction::Average,
            PayloadDecoder::Float,
            true,
        ))))
    }

    fn span_term<'x>(
        &self,
        e: &'x Element,
        anc: &Ancestors<'x>,
    ) -> Result<SpanNode, QueryParserError> {
        let field = inherited_or_fail(e, anc, "fieldName")?;
        let value = non_blank_text(e)?;
        Ok(SpanNode::term(field, value.into_bytes()))
    }

    fn span_or<'x>(
        &self,
        e: &'x Element,
        anc: &Ancestors<'x>,
    ) -> Result<SpanNode, QueryParserError> {
        let next = Self::child_anc(anc, e);
        let clauses = e
            .elements()
            .map(|c| self.span(c, &next))
            .collect::<Result<Vec<_>, _>>()?;
        SpanNode::or(clauses).map_err(QueryParserError::from_search)
    }

    fn span_or_terms<'x>(
        &self,
        e: &'x Element,
        anc: &Ancestors<'x>,
    ) -> Result<SpanNode, QueryParserError> {
        let field = inherited_or_fail(e, anc, "fieldName")?;
        let text = non_blank_text(e)?;
        let clauses = analyzed_terms(&self.analyzer, &field, &text)?
            .into_iter()
            .map(|t| SpanNode::term(field.clone(), t))
            .collect();
        SpanNode::or(clauses).map_err(QueryParserError::from_search)
    }

    fn span_first<'x>(
        &self,
        e: &'x Element,
        anc: &Ancestors<'x>,
    ) -> Result<SpanNode, QueryParserError> {
        let end = int_attr(e, "end", 1)?;
        let next = Self::child_anc(anc, e);
        let child = e.elements().next().ok_or_else(|| {
            QueryParserError::runtime_class(
                "java.lang.NullPointerException",
                "Cannot invoke \"org.w3c.dom.Element.getNodeName()\" because \"e\" is null",
            )
        })?;
        Ok(SpanNode::first(self.span(child, &next)?, end))
    }

    fn span_position_range<'x>(
        &self,
        e: &'x Element,
        anc: &Ancestors<'x>,
    ) -> Result<SpanNode, QueryParserError> {
        let start = int_attr(e, "start", 1)?;
        let end = int_attr(e, "end", 1)?;
        let next = Self::child_anc(anc, e);
        let child = e.elements().next().ok_or_else(|| {
            QueryParserError::runtime_class(
                "java.lang.NullPointerException",
                "Cannot invoke \"org.w3c.dom.Element.getNodeName()\" because \"e\" is null",
            )
        })?;
        Ok(SpanNode::PositionRange {
            inner: Box::new(self.span(child, &next)?),
            start,
            end,
        })
    }

    fn span_not<'x>(
        &self,
        e: &'x Element,
        anc: &Ancestors<'x>,
    ) -> Result<SpanNode, QueryParserError> {
        let include_el = child_by_tag_or_fail(e, "Include")?;
        let include = first_child_or_fail(include_el)?;
        let exclude_el = child_by_tag_or_fail(e, "Exclude")?;
        let exclude = first_child_or_fail(exclude_el)?;
        let next = Self::child_anc(anc, e);
        // `Include`/`Exclude` are parents too: their attributes are inherited
        let include = self.span(include, &Self::child_anc(&next, include_el))?;
        let exclude = self.span(exclude, &Self::child_anc(&next, exclude_el))?;
        SpanNode::not(include, exclude, 0, 0).map_err(QueryParserError::from_search)
    }
}

/// `Integer.parseInt` of an attribute value.
fn parse_int(v: &str) -> Result<i32, QueryParserError> {
    parse_java_int(v).map_err(number_format)
}

/// `minimumNumberShouldMatch` as the boolean takes it: a count, not below 0.
fn usize_msm(msm: i32) -> Result<usize, QueryParserError> {
    usize::try_from(msm)
        .map_err(|_| QueryParserError::illegal_argument("minimumNumberShouldMatch must be >= 0"))
}

/// `BooleanQueryBuilder.getOccursValue`.
fn occurs(clause: &Element) -> Result<Occur, QueryParserError> {
    let occs = clause.attribute("occurs");
    if occs.eq_ignore_ascii_case("should") {
        Ok(Occur::Should)
    } else if occs.eq_ignore_ascii_case("must") {
        Ok(Occur::Must)
    } else if occs.eq_ignore_ascii_case("mustNot") {
        Ok(Occur::MustNot)
    } else if occs == "filter" {
        Ok(Occur::Filter)
    } else {
        Err(parser_error(format!(
            "Invalid value for \"occurs\" attribute of clause:{occs}"
        )))
    }
}

fn is_span(name: &str) -> bool {
    matches!(
        name,
        "SpanNear"
            | "BoostingTermQuery"
            | "SpanTerm"
            | "SpanOr"
            | "SpanOrTerms"
            | "SpanFirst"
            | "SpanPositionRange"
            | "SpanNot"
    )
}
