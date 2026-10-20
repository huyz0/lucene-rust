//! `StandardQueryNodeProcessorPipeline`: the processors that turn the
//! syntax tree into the tree the builders read, in Java's order. Each is a
//! `QueryNodeProcessorImpl` walk -- pre-process a node, process its
//! children (a child sees the node as its parent), let the processor order
//! the new children, post-process -- with the processor's own hooks.
//!
//! A field's points config (`points.rs`) and date resolution read its
//! terms and range bounds in Java's locale `NumberFormat` and `DateFormat`,
//! as far as `lucene_analysis::miscellaneous` reproduces them.

use super::nodes::{FieldData, Kind, Modifier, Node, Text};
use super::points::PointsConfig;
use super::{Operator, StandardQueryParser};
use crate::queryparser::classic::QueryParserError;
use lucene_analysis::TokenStream;

/// `org.apache.lucene.queryparser.flexible.core.QueryNodeException`.
pub(crate) const QUERY_NODE_EXCEPTION: &str =
    "org.apache.lucene.queryparser.flexible.core.QueryNodeException";

pub(crate) type Result<T> = std::result::Result<T, QueryParserError>;

fn illegal(message: &str) -> QueryParserError {
    QueryParserError::runtime_class("java.lang.IllegalArgumentException", message)
}

/// The parts of a processor that `QueryNodeProcessorImpl` calls.
trait Processor {
    fn pre(&mut self, node: Node, _parent: Option<&Kind>) -> Result<Node> {
        Ok(node)
    }
    fn post(&mut self, node: Node, _parent: Option<&Kind>) -> Result<Node> {
        Ok(node)
    }
    fn order(&mut self, children: Vec<Node>) -> Result<Vec<Node>> {
        Ok(children)
    }
    /// `processChildren` overridden to skip a node's children.
    fn skip_children(&mut self) -> bool {
        false
    }
}

/// `QueryNodeProcessorImpl.processIteration`.
fn walk(p: &mut dyn Processor, node: Node, parent: Option<&Kind>) -> Result<Node> {
    let mut node = p.pre(node, parent)?;
    if !p.skip_children() && !node.children.is_empty() {
        let children = std::mem::take(&mut node.children);
        let mut done = Vec::with_capacity(children.len());
        for child in children {
            let mut child = walk(p, child, Some(&node.kind))?;
            // `set(children)` makes the node every child's parent again
            child.stale_parent = None;
            done.push(child);
        }
        node.children = p.order(done)?;
    }
    p.post(node, parent)
}

/// `getFieldConfig(field)`: refuses no field.
fn check_field_config(field: Option<&str>) -> Result<()> {
    match field {
        Some(_) => Ok(()),
        None => Err(illegal("field name must not be null!")),
    }
}

/// `Analyzer.normalize(field, text).utf8ToString()`.
fn normalize(cfg: &StandardQueryParser, field: &str, text: &str) -> Result<String> {
    let analyzer = cfg.analyzer.as_ref().expect("an analyzer");
    let bytes = analyzer
        .normalize(field, text)
        .map_err(QueryParserError::from_analysis)?;
    Ok(String::from_utf8_lossy(&bytes).into_owned())
}

/// `toString()` of a node's field (`getFieldAsString()`): `null` reads as
/// `"null"` in Java's string concatenation; no processor here reaches it.
fn field_str(d: &FieldData) -> String {
    d.field.clone().unwrap_or_else(|| "null".to_string())
}

/// One processor over the whole tree: the root's parent is the one it
/// may still point at.
fn pass(p: &mut dyn Processor, tree: Node) -> Result<Node> {
    let parent = tree.stale_parent.as_deref().cloned();
    walk(p, tree, parent.as_ref())
}

pub(crate) fn run(cfg: &StandardQueryParser, tree: Node) -> Result<Node> {
    let mut tree = pass(&mut WildcardProc { cfg }, tree)?;
    tree = pass(&mut MultiFieldProc { cfg, skip: false }, tree)?;
    tree = pass(&mut FuzzyProc { cfg }, tree)?;
    tree = pass(&mut RegexpProc { cfg }, tree)?;
    tree = pass(&mut MatchAllProc, tree)?;
    tree = pass(&mut OpenRangeProc, tree)?;
    tree = pass(&mut PointProc { cfg }, tree)?;
    tree = pass(&mut PointRangeProc { cfg }, tree)?;
    tree = pass(&mut TermRangeProc { cfg }, tree)?;
    if !cfg.allow_leading_wildcard {
        tree = pass(&mut LeadingWildcardProc { cfg }, tree)?;
    }
    if cfg.analyzer.is_some() {
        tree = pass(&mut AnalyzerProc { cfg }, tree)?;
    }
    tree = pass(&mut PhraseSlopProc, tree)?;
    if !cfg.precedence {
        let root_parent = tree.stale_parent.as_deref().cloned();
        boolean_to_modifier(
            &mut tree,
            root_parent.as_ref(),
            cfg.default_operator == Operator::And,
        );
    }
    tree = pass(&mut NoChildOptimizationProc, tree)?;
    tree = pass(&mut RemoveDeletedProc, tree)?;
    if tree.is_deleted() && tree.kind != Kind::MatchNoDocs {
        tree = Node::leaf(Kind::MatchNoDocs);
    }
    tree = pass(&mut RemoveEmptyNonLeafProc, tree)?;
    if !tree.is_leaf() && tree.children.is_empty() {
        tree = Node::leaf(Kind::MatchNoDocs);
    }
    tree = pass(&mut BooleanSingleChildProc, tree)?;
    tree = pass(
        &mut DefaultPhraseSlopProc {
            slop: cfg.phrase_slop,
            skip: false,
        },
        tree,
    )?;
    tree = pass(&mut BoostProc { cfg }, tree)?;
    tree = pass(&mut IntervalProc { cfg }, tree)?;
    if cfg.precedence {
        tree = pass(
            &mut BooleanModifiersProc {
                using_and: cfg.default_operator == Operator::And,
            },
            tree,
        )?;
    }
    Ok(tree)
}

/// `BooleanModifiersQueryNodeProcessor` (`PrecedenceQueryParser`): every
/// child of an AND -- or of any boolean but an OR, under a default AND --
/// required.
struct BooleanModifiersProc {
    using_and: bool,
}

impl Processor for BooleanModifiersProc {
    fn post(&mut self, mut node: Node, _parent: Option<&Kind>) -> Result<Node> {
        let applies = node.kind == Kind::And
            || (self.using_and && node.is_boolean() && node.kind != Kind::Or);
        if applies {
            let children = std::mem::take(&mut node.children);
            node.children = children
                .into_iter()
                .map(|child| match child.modifier() {
                    None => Node::with(Kind::Modifier(Modifier::Req), vec![child]),
                    Some(Modifier::None) => {
                        let inner = child.unwrap_child(0);
                        Node::with(Kind::Modifier(Modifier::Req), vec![inner])
                    }
                    Some(_) => child,
                })
                .collect();
        }
        Ok(node)
    }
}

// ------------------------------------------------------------ wildcards

struct WildcardProc<'a> {
    cfg: &'a StandardQueryParser,
}

/// `isWildcard(text)`: an unescaped `*` or `?`.
fn is_wildcard(text: &Text) -> bool {
    (0..text.len()).rev().any(|i| {
        let c = text.char_at(i);
        (c == u16::from(b'*') || c == u16::from(b'?')) && !text.was_escaped(i)
    })
}

/// `isPrefixWildcard(text)`: one unescaped `*`, at the end, and no `?`.
fn is_prefix_wildcard(text: &Text) -> bool {
    if text.is_empty() || !is_wildcard(text) {
        return false;
    }
    let last = text.len() - 1;
    if text.char_at(last) != u16::from(b'*') || text.was_escaped(last) || text.len() == 1 {
        return false;
    }
    for i in 0..text.len() {
        let c = text.char_at(i);
        if c == u16::from(b'?') {
            return false;
        }
        if c == u16::from(b'*') && !text.was_escaped(i) {
            return i == last;
        }
    }
    false
}

/// `analyzeWildcard(analyzer, field, wildcard)`: the text between the
/// `[?*]+` runs normalised, the runs kept. Lucene's flexible
/// `WILDCARD_PATTERN` is `(\.)|([?*]+)`: its first group matches a literal
/// `.` and is skipped without moving `last`, so only the runs split the text
/// (unlike the classic parser's `(\\.)|([?*]+)`).
fn analyze_wildcard(cfg: &StandardQueryParser, field: &str, wildcard: &[u16]) -> Result<String> {
    let is_wild = |c: u16| c == u16::from(b'?') || c == u16::from(b'*');
    let mut out = String::new();
    let mut last = 0;
    let mut i = 0;
    while i < wildcard.len() {
        if !is_wild(wildcard[i]) {
            i += 1;
            continue;
        }
        let mut end = i + 1;
        while end < wildcard.len() && is_wild(wildcard[end]) {
            end += 1;
        }
        if i > 0 {
            out.push_str(&normalize(
                cfg,
                field,
                &String::from_utf16_lossy(&wildcard[last..i]),
            )?);
        }
        out.push_str(&String::from_utf16_lossy(&wildcard[i..end]));
        last = end;
        i = end;
    }
    if last < wildcard.len() {
        out.push_str(&normalize(
            cfg,
            field,
            &String::from_utf16_lossy(&wildcard[last..]),
        )?);
    }
    Ok(out)
}

impl Processor for WildcardProc<'_> {
    fn post(&mut self, node: Node, parent: Option<&Kind>) -> Result<Node> {
        let Some(d) = node.field_data() else {
            return Ok(node);
        };
        if matches!(parent, Some(Kind::TermRange { .. }))
            || matches!(node.kind, Kind::Quoted(_))
            || d.text.is_empty()
            || !is_wildcard(&d.text)
        {
            return Ok(node);
        }
        let mut text = d.text.clone();
        if self.cfg.analyzer.is_some() {
            text = Text::plain(&analyze_wildcard(self.cfg, &field_str(d), &text.units)?);
        }
        let data = FieldData::new(d.field.clone(), text);
        Ok(Node::leaf(if is_prefix_wildcard(&data.text) {
            Kind::PrefixWildcard(data)
        } else {
            Kind::Wildcard(data)
        }))
    }
}

// --------------------------------------------------------- multi-field

struct MultiFieldProc<'a> {
    cfg: &'a StandardQueryParser,
    skip: bool,
}

impl Processor for MultiFieldProc<'_> {
    fn pre(&mut self, mut node: Node, _parent: Option<&Kind>) -> Result<Node> {
        if !node.is_fieldable() {
            return Ok(node);
        }
        self.skip = true;
        if node.get_field().is_some() {
            return Ok(node);
        }
        let Some(fields) = &self.cfg.multi_fields else {
            return Err(illegal(
                "StandardQueryConfigHandler.ConfigurationKeys.MULTI_FIELDS should be set on the QueryConfigHandler",
            ));
        };
        if fields.is_empty() {
            return Ok(node);
        }
        node.set_field(&fields[0]);
        if fields.len() == 1 {
            return Ok(node);
        }
        let mut children = vec![node.clone()];
        for f in &fields[1..] {
            let mut copy = node.clone();
            copy.tags = Default::default();
            copy.set_field(f);
            children.push(copy);
        }
        Ok(Node::with(
            Kind::Group,
            vec![Node::with(Kind::Or, children)],
        ))
    }

    fn skip_children(&mut self) -> bool {
        std::mem::replace(&mut self.skip, false)
    }
}

// ----------------------------------------------------- fuzzy and regexp

struct FuzzyProc<'a> {
    cfg: &'a StandardQueryParser,
}

impl Processor for FuzzyProc<'_> {
    fn pre(&mut self, mut node: Node, _parent: Option<&Kind>) -> Result<Node> {
        if let Kind::Fuzzy {
            data,
            similarity,
            prefix_length,
        } = &mut node.kind
        {
            if self.cfg.analyzer.is_some() {
                let text = normalize(self.cfg, &field_str(data), &data.text.as_string())?;
                data.text = Text::plain(&text);
            }
            *prefix_length = self.cfg.fuzzy_prefix_length;
            if *similarity < 0.0 {
                *similarity = self.cfg.fuzzy_min_sim;
            }
        }
        Ok(node)
    }
}

struct RegexpProc<'a> {
    cfg: &'a StandardQueryParser,
}

impl Processor for RegexpProc<'_> {
    fn post(&mut self, mut node: Node, _parent: Option<&Kind>) -> Result<Node> {
        if let Kind::Regexp { field, text } = &mut node.kind {
            if self.cfg.analyzer.is_some() {
                let f = field.clone().unwrap_or_else(|| "null".to_string());
                *text = Text::plain(&normalize(self.cfg, &f, &text.as_string())?);
            }
        }
        Ok(node)
    }
}

struct MatchAllProc;

impl Processor for MatchAllProc {
    fn post(&mut self, node: Node, _parent: Option<&Kind>) -> Result<Node> {
        if let Some(d) = node.field_data() {
            if d.field.as_deref() == Some("*") && d.text.as_string() == "*" {
                return Ok(Node::leaf(Kind::MatchAll));
            }
        }
        Ok(node)
    }
}

// --------------------------------------------------------------- ranges

/// `OpenRangeQueryNodeProcessor`: an unescaped `*` bound is open (`""`).
struct OpenRangeProc;

impl Processor for OpenRangeProc {
    fn post(&mut self, mut node: Node, _parent: Option<&Kind>) -> Result<Node> {
        if matches!(node.kind, Kind::TermRange { .. }) {
            for bound in node.children.iter_mut() {
                if let Some(d) = bound.field_data_mut() {
                    if d.text.as_string() == "*" && !d.text.was_escaped(0) {
                        d.text = Text::plain("");
                    }
                }
            }
        }
        Ok(node)
    }
}

/// `QueryNodeParseException`, which a points term that does not parse
/// throws.
const QUERY_NODE_PARSE_EXCEPTION: &str =
    "org.apache.lucene.queryparser.flexible.core.QueryNodeParseException";

/// A point bound's text through its config: `None` for an empty text,
/// `COULD_NOT_PARSE_NUMBER` for one its format does not parse.
fn point_value(config: &PointsConfig, text: &str) -> Result<Option<super::points::PointValue>> {
    if text.is_empty() {
        return Ok(None);
    }
    let format = config.number_format();
    let number = format.parse(text).ok_or_else(|| {
        QueryParserError::runtime_class(
            QUERY_NODE_PARSE_EXCEPTION,
            format!(
                "Could not parse text \"{text}\" using {}",
                format.class_name()
            ),
        )
    })?;
    Ok(Some(config.point_type().convert(number)))
}

/// A `PointRangeQueryNode` over two `PointQueryNode`s of `field`.
fn point_range(
    field: Option<String>,
    config: &PointsConfig,
    values: [Option<super::points::PointValue>; 2],
    lower_inclusive: bool,
    upper_inclusive: bool,
) -> Node {
    let children = values
        .into_iter()
        .map(|value| {
            Node::leaf(Kind::Point {
                field: field.clone(),
                value,
                format: config.number_format().clone(),
            })
        })
        .collect();
    Node::with(
        Kind::PointRange {
            lower_inclusive,
            upper_inclusive,
            config: config.clone(),
        },
        children,
    )
}

/// `PointQueryNodeProcessor`: a field term (of any `FieldQueryNode` class)
/// outside a range, of a field with a points config, is the inclusive range
/// from its number to itself.
struct PointProc<'a> {
    cfg: &'a StandardQueryParser,
}

impl Processor for PointProc<'_> {
    fn post(&mut self, node: Node, parent: Option<&Kind>) -> Result<Node> {
        let Some(d) = node.field_data() else {
            return Ok(node);
        };
        if matches!(
            parent,
            Some(Kind::TermRange { .. } | Kind::PointRange { .. })
        ) {
            return Ok(node);
        }
        check_field_config(d.field.as_deref())?;
        let field = d.field.clone().unwrap_or_default();
        let Some(config) = self.cfg.points_config(&field) else {
            return Ok(node);
        };
        let Some(value) = point_value(config, &d.text.as_string())? else {
            return Err(QueryParserError::runtime_class(
                QUERY_NODE_PARSE_EXCEPTION,
                format!("Field \"{field}\" is numeric and cannot have an empty value."),
            ));
        };
        Ok(point_range(
            d.field.clone(),
            config,
            [Some(value), Some(value)],
            true,
            true,
        ))
    }
}

/// `PointRangeQueryNodeProcessor`: a term range of a field with a points
/// config is a point range, its bounds parsed (an empty one open), the
/// lower first.
struct PointRangeProc<'a> {
    cfg: &'a StandardQueryParser,
}

impl Processor for PointRangeProc<'_> {
    fn post(&mut self, node: Node, _parent: Option<&Kind>) -> Result<Node> {
        let Kind::TermRange {
            lower_inclusive,
            upper_inclusive,
        } = node.kind
        else {
            return Ok(node);
        };
        let field = node.get_field();
        check_field_config(field.as_deref())?;
        let Some(config) = self.cfg.points_config(field.as_deref().unwrap_or_default()) else {
            return Ok(node);
        };
        let text = |i: usize| {
            node.children
                .get(i)
                .and_then(Node::field_data)
                .map(|d| d.text.as_string())
                .unwrap_or_default()
        };
        let lower = point_value(config, &text(0))?;
        let upper = point_value(config, &text(1))?;
        Ok(point_range(
            field,
            config,
            [lower, upper],
            lower_inclusive,
            upper_inclusive,
        ))
    }
}

/// `TermRangeQueryNodeProcessor`: one `try` over both bounds, each
/// non-empty one replaced by its date at the field's resolution (the upper
/// one at the end of its day when inclusive); the first bound that is not a
/// date -- with no resolution, the first non-empty one -- ends it in the
/// `catch`, which normalises both bounds as they then are (a lower bound
/// already a date's string included).
struct TermRangeProc<'a> {
    cfg: &'a StandardQueryParser,
}

impl Processor for TermRangeProc<'_> {
    fn post(&mut self, mut node: Node, _parent: Option<&Kind>) -> Result<Node> {
        let Kind::TermRange {
            upper_inclusive, ..
        } = node.kind
        else {
            return Ok(node);
        };
        let field = node.get_field();
        check_field_config(field.as_deref())?;
        if node.children.len() != 2 {
            return Ok(node);
        }
        let resolution = self.cfg.date_resolution(field.as_deref().unwrap_or("null"));
        let mut caught = false;
        for (i, bound) in node.children.iter_mut().enumerate() {
            if let Some(d) = bound.field_data_mut() {
                let text = d.text.as_string();
                if text.is_empty() {
                    continue;
                }
                match self
                    .cfg
                    .dates
                    .bound(&text, resolution, i == 1 && upper_inclusive)
                    .map_err(|e| illegal(&e))?
                {
                    Some(date) => d.text = Text::plain(&date),
                    None => {
                        caught = true;
                        break;
                    }
                }
            }
        }
        if caught && self.cfg.analyzer.is_some() {
            let lower_field = node.children[0]
                .field_data()
                .map(field_str)
                .unwrap_or_else(|| "null".to_string());
            for bound in node.children.iter_mut() {
                if let Some(d) = bound.field_data_mut() {
                    let text = normalize(self.cfg, &lower_field, &d.text.as_string())?;
                    d.text = Text::plain(&text);
                }
            }
        }
        Ok(node)
    }
}

/// `AllowLeadingWildcardProcessor`, run when leading wildcards are not
/// allowed.
struct LeadingWildcardProc<'a> {
    cfg: &'a StandardQueryParser,
}

impl Processor for LeadingWildcardProc<'_> {
    fn post(&mut self, node: Node, _parent: Option<&Kind>) -> Result<Node> {
        if node.is_wildcard() {
            let d = node.field_data().expect("a wildcard node");
            if !d.text.is_empty()
                && !d.text.was_escaped(0)
                && matches!(d.text.char_at(0), 0x2a | 0x3f)
            {
                return Err(QueryParserError::runtime_class(
                    QUERY_NODE_EXCEPTION,
                    format!(
                        "Leading wildcard is not allowed: {}",
                        node.to_query_string(None, self.cfg.default_case_locale)?
                    ),
                ));
            }
        }
        Ok(node)
    }
}

// ------------------------------------------------------------- analysis

/// `AnalyzerQueryNodeProcessor`.
struct AnalyzerProc<'a> {
    cfg: &'a StandardQueryParser,
}

/// One token: its text and position increment.
struct Tok {
    term: String,
    pos_inc: i32,
}

impl AnalyzerProc<'_> {
    fn tokens(&self, field: &str, text: &str) -> Result<Vec<Tok>> {
        let analyzer = self.cfg.analyzer.as_ref().expect("an analyzer");
        let mut ts = analyzer
            .token_stream(field, text)
            .map_err(QueryParserError::from_analysis)?;
        ts.reset().map_err(QueryParserError::from_analysis)?;
        let mut out = Vec::new();
        while ts
            .increment_token()
            .map_err(QueryParserError::from_analysis)?
        {
            let atts = ts.attributes();
            out.push(Tok {
                term: atts.term().to_string(),
                pos_inc: atts.position_increment(),
            });
        }
        ts.end().map_err(QueryParserError::from_analysis)?;
        ts.close().map_err(QueryParserError::from_analysis)?;
        Ok(out)
    }
}

impl Processor for AnalyzerProc<'_> {
    fn post(&mut self, mut node: Node, parent: Option<&Kind>) -> Result<Node> {
        let analyzable = matches!(node.kind, Kind::Field(_) | Kind::Quoted(_))
            && !matches!(parent, Some(Kind::TermRange { .. }));
        if !analyzable {
            return Ok(node);
        }
        let quoted = matches!(node.kind, Kind::Quoted(_));
        let d = node.field_data().expect("a field node");
        let field = d.field.clone();
        let field_s = field_str(d);
        let tokens = self.tokens(&field_s, &d.text.as_string())?;
        let num_tokens = tokens.len();
        let mut position_count = 0;
        let mut several_at_same_position = false;
        for t in &tokens {
            if t.pos_inc != 0 {
                position_count += t.pos_inc;
            } else {
                several_at_same_position = true;
            }
        }
        let term = |t: &Tok| {
            Node::leaf(Kind::Field(FieldData::new(
                field.clone(),
                Text::plain(&t.term),
            )))
        };
        if num_tokens == 0 {
            return Ok(Node::leaf(Kind::NoTokenFound));
        }
        if num_tokens == 1 {
            node.field_data_mut().expect("a field node").text = Text::plain(&tokens[0].term);
            return Ok(node);
        }
        if several_at_same_position || !quoted {
            if position_count == 1 || !quoted {
                if position_count == 1 {
                    let children = tokens.iter().map(term).collect();
                    return Ok(Node::with(
                        Kind::Group,
                        vec![Node::with(Kind::Synonym, children)],
                    ));
                }
                let mut q: Vec<Node> = Vec::new();
                let mut current: Option<Node> = None;
                let and = self.cfg.default_operator == Operator::And;
                let wrap = |n: Node| {
                    if and {
                        Node::with(Kind::Modifier(Modifier::Req), vec![n])
                    } else {
                        n
                    }
                };
                for t in &tokens {
                    if t.pos_inc == 0 {
                        // A first token stacked on nothing: Lucene wraps the
                        // null `currentQuery` and `QueryNodeImpl.add(null)`
                        // refuses it.
                        let Some(cur) = current.take() else {
                            return Err(illegal("This node does not support this action."));
                        };
                        let mut syn = if cur.is_boolean() {
                            cur
                        } else {
                            Node::with(Kind::Synonym, vec![cur])
                        };
                        syn.children.push(term(t));
                        current = Some(syn);
                    } else {
                        if let Some(cur) = current.take() {
                            q.push(wrap(cur));
                        }
                        current = Some(term(t));
                    }
                }
                if let Some(cur) = current.take() {
                    q.push(wrap(cur));
                }
                return Ok(Node::with(Kind::Group, vec![Node::with(Kind::Boolean, q)]));
            }
            // several tokens at one position, in quotes: a multi-phrase
            let mut mpq = Vec::new();
            let mut multi_terms: Vec<Node> = Vec::new();
            let mut position = -1;
            let mut term_group_count = 0;
            let positions = self.cfg.enable_position_increments;
            let flush = |multi: &mut Vec<Node>, mpq: &mut Vec<Node>, pos: i32, group: i32| {
                for mut n in multi.drain(..) {
                    n.field_data_mut().expect("a term").position_increment =
                        if positions { pos } else { group };
                    mpq.push(n);
                }
            };
            for t in &tokens {
                if t.pos_inc > 0 && !multi_terms.is_empty() {
                    flush(&mut multi_terms, &mut mpq, position, term_group_count);
                    term_group_count += 1;
                }
                position += t.pos_inc;
                multi_terms.push(term(t));
            }
            flush(&mut multi_terms, &mut mpq, position, term_group_count);
            return Ok(Node::with(Kind::MultiPhrase, mpq));
        }
        // a phrase
        let mut pq = Vec::new();
        let mut position = -1;
        for (i, t) in tokens.iter().enumerate() {
            let mut n = term(t);
            let pi = if self.cfg.enable_position_increments {
                position += t.pos_inc;
                position
            } else {
                i as i32
            };
            n.field_data_mut().expect("a term").position_increment = pi;
            pq.push(n);
        }
        Ok(Node::with(Kind::TokenizedPhrase, pq))
    }
}

/// `PhraseSlopQueryNodeProcessor`: a slop on anything but a phrase is
/// dropped.
struct PhraseSlopProc;

impl Processor for PhraseSlopProc {
    fn post(&mut self, node: Node, _parent: Option<&Kind>) -> Result<Node> {
        if matches!(node.kind, Kind::Slop(_))
            && !matches!(
                node.children.first().map(|c| &c.kind),
                Some(Kind::TokenizedPhrase | Kind::MultiPhrase)
            )
        {
            return Ok(node.unwrap_child(0));
        }
        Ok(node)
    }
}

// ------------------------------------------------- boolean to modifiers

/// `BooleanQuery2ModifierNodeProcessor`: its own walk, which tags without
/// replacing children, then has each top boolean take its nested booleans'
/// children up, modifiers applied.
fn boolean_to_modifier(node: &mut Node, parent: Option<&Kind>, using_and: bool) {
    // preProcessNode
    if node.is_boolean() {
        if parent.is_some_and(Kind::is_boolean) {
            node.tags.remove = true;
        } else {
            node.tags.boolean_root = true;
        }
    } else if let Some(p) = parent.filter(|p| p.is_boolean()) {
        if *p == Kind::And || (using_and && *p == Kind::Boolean) {
            // `tagModifierButDoNotOverride(node, MOD_REQ)`
            match node.modifier() {
                Some(Modifier::None) => node.tags.modifier = Some(Modifier::Req),
                Some(_) => {}
                None => node.tags.modifier = Some(Modifier::Req),
            }
        }
    }
    let kind = node.kind.clone();
    for child in node.children.iter_mut() {
        boolean_to_modifier(child, Some(&kind), using_and);
    }
    // postProcessNode
    if node.tags.boolean_root {
        let mut buffer = Vec::new();
        fill_children(std::mem::take(&mut node.children), &mut buffer);
        node.children = buffer;
    }
}

fn fill_children(children: Vec<Node>, buffer: &mut Vec<Node>) {
    for child in children {
        if child.tags.remove {
            fill_children(child.children, buffer);
        } else if let Some(m) = child.tags.modifier {
            buffer.push(apply_modifier(child, m));
        } else {
            buffer.push(child);
        }
    }
}

/// `applyModifier(node, mod)`.
fn apply_modifier(mut node: Node, m: Modifier) -> Node {
    match node.modifier() {
        None => Node::with(Kind::BooleanModifier(m), vec![node]),
        Some(Modifier::None) => Node::with(Kind::Modifier(m), vec![node.children.remove(0)]),
        Some(_) => node,
    }
}

impl Kind {
    /// `instanceof BooleanQueryNode`.
    pub(crate) fn is_boolean(&self) -> bool {
        matches!(self, Kind::Boolean | Kind::And | Kind::Or | Kind::Synonym)
    }
}

// ------------------------------------------------------- optimisations

struct NoChildOptimizationProc;

impl Processor for NoChildOptimizationProc {
    fn post(&mut self, node: Node, _parent: Option<&Kind>) -> Result<Node> {
        let applies = node.is_boolean()
            || node.is_modifier()
            || matches!(node.kind, Kind::Boost(_) | Kind::TokenizedPhrase);
        if applies && node.children.iter().all(Node::is_deleted) {
            return Ok(Node::leaf(Kind::MatchNoDocs));
        }
        Ok(node)
    }
}

struct RemoveDeletedProc;

impl Processor for RemoveDeletedProc {
    fn post(&mut self, node: Node, _parent: Option<&Kind>) -> Result<Node> {
        if !node.is_leaf() && node.children.iter().all(Node::is_deleted) {
            return Ok(Node::leaf(Kind::Deleted));
        }
        Ok(node)
    }

    fn order(&mut self, children: Vec<Node>) -> Result<Vec<Node>> {
        Ok(children.into_iter().filter(|c| !c.is_deleted()).collect())
    }
}

struct RemoveEmptyNonLeafProc;

impl Processor for RemoveEmptyNonLeafProc {
    fn order(&mut self, children: Vec<Node>) -> Result<Vec<Node>> {
        Ok(children
            .into_iter()
            .filter(|c| c.is_leaf() || !c.children.is_empty())
            .collect())
    }
}

struct BooleanSingleChildProc;

impl Processor for BooleanSingleChildProc {
    fn post(&mut self, node: Node, _parent: Option<&Kind>) -> Result<Node> {
        if node.is_boolean() && node.children.len() == 1 {
            let child = &node.children[0];
            let unwrap = match child.kind {
                Kind::BooleanModifier(_) => true,
                Kind::Modifier(m) => m == Modifier::None,
                _ => true,
            };
            if unwrap {
                return Ok(node.unwrap_child(0));
            }
        }
        Ok(node)
    }
}

/// `DefaultPhraseSlopQueryNodeProcessor`: a phrase with no slop of its own
/// takes the default one.
struct DefaultPhraseSlopProc {
    slop: i32,
    skip: bool,
}

impl Processor for DefaultPhraseSlopProc {
    fn pre(&mut self, node: Node, _parent: Option<&Kind>) -> Result<Node> {
        if matches!(node.kind, Kind::Slop(_)) {
            self.skip = true;
        }
        Ok(node)
    }

    fn skip_children(&mut self) -> bool {
        std::mem::replace(&mut self.skip, false)
    }

    fn post(&mut self, node: Node, _parent: Option<&Kind>) -> Result<Node> {
        if matches!(node.kind, Kind::TokenizedPhrase | Kind::MultiPhrase) {
            return Ok(Node::with(Kind::Slop(self.slop), vec![node]));
        }
        Ok(node)
    }
}

/// `BoostQueryNodeProcessor`: a field's configured boost, on the outermost
/// fieldable node.
struct BoostProc<'a> {
    cfg: &'a StandardQueryParser,
}

impl Processor for BoostProc<'_> {
    fn post(&mut self, node: Node, parent: Option<&Kind>) -> Result<Node> {
        if !node.is_fieldable() || parent.is_some_and(Kind::is_fieldable) {
            return Ok(node);
        }
        let field = node.get_field();
        check_field_config(field.as_deref())?;
        let field = field.expect("checked");
        if let Some((_, boost)) = self.cfg.field_boosts.iter().find(|(f, _)| *f == field) {
            return Ok(Node::with(Kind::Boost(*boost), vec![node]));
        }
        Ok(node)
    }
}

impl Kind {
    /// `instanceof FieldableNode`.
    pub(crate) fn is_fieldable(&self) -> bool {
        matches!(
            self,
            Kind::Field(_)
                | Kind::Quoted(_)
                | Kind::Fuzzy { .. }
                | Kind::Wildcard(_)
                | Kind::PrefixWildcard(_)
                | Kind::Slop(_)
                | Kind::TokenizedPhrase
                | Kind::MultiPhrase
                | Kind::TermRange { .. }
                | Kind::Regexp { .. }
                | Kind::Interval { .. }
                | Kind::Point { .. }
                | Kind::PointRange { .. }
        )
    }
}

/// `IntervalQueryNodeProcessor`: an interval node needs the analyzer.
struct IntervalProc<'a> {
    cfg: &'a StandardQueryParser,
}

impl Processor for IntervalProc<'_> {
    fn pre(&mut self, node: Node, _parent: Option<&Kind>) -> Result<Node> {
        if let Kind::Interval { field, source } = &node.kind {
            if self.cfg.analyzer.is_none() {
                return Err(QueryParserError::runtime_class(
                    QUERY_NODE_EXCEPTION,
                    format!(
                        "An analyzer is required to parse interval sub-query \"{}\"",
                        super::builders::interval_node_string(field.as_deref(), source)
                    ),
                ));
            }
        }
        Ok(node)
    }
}
