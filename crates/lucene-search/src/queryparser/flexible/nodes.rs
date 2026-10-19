//! The query node tree of `org.apache.lucene.queryparser.flexible`
//! (`core.nodes`, `standard.nodes`, `standard.nodes.intervalfn`) as one
//! enum. Java's node classes inherit from each other and implement marker
//! interfaces, and the processors test both with `instanceof`; the `is_*`
//! methods answer those tests class by class, so a processor reads as its
//! Java original.

/// `UnescapedCharSequence`: text as UTF-16 units and, when the parser made
/// it from a query string, which units were escaped there. Text a processor
/// sets (`setText(String)`) has no escape record.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Text {
    pub units: Vec<u16>,
    pub escaped: Option<Vec<bool>>,
}

impl Text {
    pub(crate) fn plain(s: &str) -> Text {
        Text {
            units: s.encode_utf16().collect(),
            escaped: None,
        }
    }

    /// `toString()`.
    pub(crate) fn as_string(&self) -> String {
        String::from_utf16_lossy(&self.units)
    }

    pub(crate) fn len(&self) -> usize {
        self.units.len()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.units.is_empty()
    }

    pub(crate) fn char_at(&self, i: usize) -> u16 {
        self.units[i]
    }

    /// `UnescapedCharSequence.wasEscaped(text, i)`: `false` for text with no
    /// escape record.
    pub(crate) fn was_escaped(&self, i: usize) -> bool {
        self.escaped.as_ref().is_some_and(|e| e[i])
    }
}

/// `ModifierQueryNode.Modifier`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Modifier {
    None,
    Req,
    Not,
}

/// What `FieldQueryNode` carries (and every node that extends it).
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct FieldData {
    pub field: Option<String>,
    pub text: Text,
    /// `positionIncrement`: a phrase term's position.
    pub position_increment: i32,
}

impl FieldData {
    pub(crate) fn new(field: Option<String>, text: Text) -> FieldData {
        FieldData {
            field,
            text,
            position_increment: 0,
        }
    }
}

/// A node's class, with the fields that class adds.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Kind {
    /// `BooleanQueryNode`.
    Boolean,
    /// `AndQueryNode` (a `BooleanQueryNode`).
    And,
    /// `OrQueryNode` (a `BooleanQueryNode`).
    Or,
    /// `SynonymQueryNode` (a `BooleanQueryNode`).
    Synonym,
    /// `BoostQueryNode`.
    Boost(f32),
    /// `GroupQueryNode`.
    Group,
    /// `ModifierQueryNode`.
    Modifier(Modifier),
    /// `BooleanModifierNode` (a `ModifierQueryNode`).
    BooleanModifier(Modifier),
    /// `SlopQueryNode`.
    Slop(i32),
    /// `MinShouldMatchNode`: its child is the `GroupQueryNode`.
    MinShouldMatch(i32),
    /// `TokenizedPhraseQueryNode`.
    TokenizedPhrase,
    /// `MultiPhraseQueryNode`.
    MultiPhrase,
    /// `FieldQueryNode`.
    Field(FieldData),
    /// `QuotedFieldQueryNode` (a `FieldQueryNode`).
    Quoted(FieldData),
    /// `FuzzyQueryNode` (a `FieldQueryNode`).
    Fuzzy {
        data: FieldData,
        similarity: f32,
        prefix_length: i32,
    },
    /// `WildcardQueryNode` (a `FieldQueryNode`).
    Wildcard(FieldData),
    /// `PrefixWildcardQueryNode` (a `WildcardQueryNode`).
    PrefixWildcard(FieldData),
    /// `RegexpQueryNode`.
    Regexp { field: Option<String>, text: Text },
    /// `TermRangeQueryNode`: its two children are the bounds.
    TermRange {
        lower_inclusive: bool,
        upper_inclusive: bool,
    },
    /// `MatchAllDocsQueryNode`.
    MatchAll,
    /// `MatchNoDocsQueryNode` (a `DeletedQueryNode`).
    MatchNoDocs,
    /// `DeletedQueryNode`.
    Deleted,
    /// `NoTokenFoundQueryNode` (a `DeletedQueryNode`).
    NoTokenFound,
    /// `IntervalQueryNode`.
    Interval {
        field: Option<String>,
        source: IntervalFn,
    },
}

/// The tags the processors leave on a node: `BooleanQuery2ModifierNodeProcessor`'s
/// three (`MultiTermRewriteMethodProcessor`'s is folded into the builders,
/// which read the parser's method).
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct Tags {
    pub remove: bool,
    pub boolean_root: bool,
    pub modifier: Option<Modifier>,
}

/// One node: its class and data, its children (a leaf has none), its tags.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Node {
    pub kind: Kind,
    pub children: Vec<Node>,
    pub tags: Tags,
    /// The parent a node unwrapped out of the root still points at: Java's
    /// `QueryNodeImpl.parent` is only reset when a node is set into a new
    /// parent, which the root never is, so a processor that returns the
    /// root's child leaves that child pointing at the node it replaced.
    pub stale_parent: Option<Box<Kind>>,
}

impl Node {
    pub(crate) fn leaf(kind: Kind) -> Node {
        Node::with(kind, Vec::new())
    }

    pub(crate) fn with(kind: Kind, children: Vec<Node>) -> Node {
        Node {
            kind,
            children,
            tags: Tags::default(),
            stale_parent: None,
        }
    }

    /// The child a processor returns in place of this node, still pointing
    /// at this node as its parent.
    pub(crate) fn unwrap_child(mut self, i: usize) -> Node {
        let mut child = self.children.remove(i);
        child.stale_parent = Some(Box::new(self.kind));
        child
    }

    /// `isLeaf()`.
    pub(crate) fn is_leaf(&self) -> bool {
        matches!(
            self.kind,
            Kind::Field(_)
                | Kind::Quoted(_)
                | Kind::Fuzzy { .. }
                | Kind::Wildcard(_)
                | Kind::PrefixWildcard(_)
                | Kind::Regexp { .. }
                | Kind::MatchAll
                | Kind::MatchNoDocs
                | Kind::Deleted
                | Kind::NoTokenFound
                | Kind::Interval { .. }
        )
    }

    /// `instanceof BooleanQueryNode`.
    pub(crate) fn is_boolean(&self) -> bool {
        matches!(
            self.kind,
            Kind::Boolean | Kind::And | Kind::Or | Kind::Synonym
        )
    }

    /// `instanceof FieldQueryNode`.
    pub(crate) fn is_field(&self) -> bool {
        self.field_data().is_some()
    }

    /// `instanceof WildcardQueryNode`.
    pub(crate) fn is_wildcard(&self) -> bool {
        matches!(self.kind, Kind::Wildcard(_) | Kind::PrefixWildcard(_))
    }

    /// `instanceof ModifierQueryNode`.
    pub(crate) fn is_modifier(&self) -> bool {
        matches!(self.kind, Kind::Modifier(_) | Kind::BooleanModifier(_))
    }

    /// `instanceof DeletedQueryNode`.
    pub(crate) fn is_deleted(&self) -> bool {
        matches!(
            self.kind,
            Kind::Deleted | Kind::MatchNoDocs | Kind::NoTokenFound
        )
    }

    /// `instanceof FieldableNode`.
    pub(crate) fn is_fieldable(&self) -> bool {
        self.is_field()
            || matches!(
                self.kind,
                Kind::Slop(_)
                    | Kind::TokenizedPhrase
                    | Kind::MultiPhrase
                    | Kind::TermRange { .. }
                    | Kind::Regexp { .. }
                    | Kind::Interval { .. }
            )
    }

    /// The `FieldQueryNode` part of a node of that family.
    pub(crate) fn field_data(&self) -> Option<&FieldData> {
        match &self.kind {
            Kind::Field(d) | Kind::Quoted(d) | Kind::Wildcard(d) | Kind::PrefixWildcard(d) => {
                Some(d)
            }
            Kind::Fuzzy { data, .. } => Some(data),
            _ => None,
        }
    }

    pub(crate) fn field_data_mut(&mut self) -> Option<&mut FieldData> {
        match &mut self.kind {
            Kind::Field(d) | Kind::Quoted(d) | Kind::Wildcard(d) | Kind::PrefixWildcard(d) => {
                Some(d)
            }
            Kind::Fuzzy { data, .. } => Some(data),
            _ => None,
        }
    }

    /// `FieldableNode.getField()`.
    pub(crate) fn get_field(&self) -> Option<String> {
        if let Some(d) = self.field_data() {
            return d.field.clone();
        }
        match &self.kind {
            Kind::Regexp { field, .. } | Kind::Interval { field, .. } => field.clone(),
            // `SlopQueryNode`: its child's
            Kind::Slop(_) => self
                .children
                .first()
                .filter(|c| c.is_fieldable())
                .and_then(Node::get_field),
            // the first fieldable child's
            Kind::TokenizedPhrase => self
                .children
                .iter()
                .find(|c| c.is_fieldable())
                .and_then(Node::get_field),
            Kind::MultiPhrase => self.children.first().and_then(Node::get_field),
            // `AbstractRangeQueryNode`: the lower bound's, else the upper's
            Kind::TermRange { .. } => match self.children.first() {
                Some(lower) => lower.get_field(),
                None => self.children.get(1).and_then(Node::get_field),
            },
            _ => None,
        }
    }

    /// `FieldableNode.setField(field)`.
    pub(crate) fn set_field(&mut self, field: &str) {
        if let Some(d) = self.field_data_mut() {
            d.field = Some(field.to_string());
            return;
        }
        match &mut self.kind {
            Kind::Regexp { field: f, .. } | Kind::Interval { field: f, .. } => {
                *f = Some(field.to_string())
            }
            Kind::Slop(_) => {
                if let Some(c) = self.children.first_mut().filter(|c| c.is_fieldable()) {
                    c.set_field(field);
                }
            }
            Kind::TokenizedPhrase | Kind::MultiPhrase | Kind::TermRange { .. } => {
                for c in self.children.iter_mut().filter(|c| c.is_fieldable()) {
                    c.set_field(field);
                }
            }
            _ => {}
        }
    }

    /// `ModifierQueryNode.getModifier()`.
    pub(crate) fn modifier(&self) -> Option<Modifier> {
        match self.kind {
            Kind::Modifier(m) | Kind::BooleanModifier(m) => Some(m),
            _ => None,
        }
    }

    /// The node's class name (`getClass().getName()`), for the messages that
    /// print it.
    pub(crate) fn class_name(&self) -> &'static str {
        match self.kind {
            Kind::Boolean => "org.apache.lucene.queryparser.flexible.core.nodes.BooleanQueryNode",
            Kind::And => "org.apache.lucene.queryparser.flexible.core.nodes.AndQueryNode",
            Kind::Or => "org.apache.lucene.queryparser.flexible.core.nodes.OrQueryNode",
            Kind::Synonym => {
                "org.apache.lucene.queryparser.flexible.standard.nodes.SynonymQueryNode"
            }
            Kind::Boost(_) => "org.apache.lucene.queryparser.flexible.core.nodes.BoostQueryNode",
            Kind::Group => "org.apache.lucene.queryparser.flexible.core.nodes.GroupQueryNode",
            Kind::Modifier(_) => {
                "org.apache.lucene.queryparser.flexible.core.nodes.ModifierQueryNode"
            }
            Kind::BooleanModifier(_) => {
                "org.apache.lucene.queryparser.flexible.standard.nodes.BooleanModifierNode"
            }
            Kind::Slop(_) => "org.apache.lucene.queryparser.flexible.core.nodes.SlopQueryNode",
            Kind::MinShouldMatch(_) => {
                "org.apache.lucene.queryparser.flexible.standard.nodes.MinShouldMatchNode"
            }
            Kind::TokenizedPhrase => {
                "org.apache.lucene.queryparser.flexible.core.nodes.TokenizedPhraseQueryNode"
            }
            Kind::MultiPhrase => {
                "org.apache.lucene.queryparser.flexible.standard.nodes.MultiPhraseQueryNode"
            }
            Kind::Field(_) => "org.apache.lucene.queryparser.flexible.core.nodes.FieldQueryNode",
            Kind::Quoted(_) => {
                "org.apache.lucene.queryparser.flexible.core.nodes.QuotedFieldQueryNode"
            }
            Kind::Fuzzy { .. } => {
                "org.apache.lucene.queryparser.flexible.core.nodes.FuzzyQueryNode"
            }
            Kind::Wildcard(_) => {
                "org.apache.lucene.queryparser.flexible.standard.nodes.WildcardQueryNode"
            }
            Kind::PrefixWildcard(_) => {
                "org.apache.lucene.queryparser.flexible.standard.nodes.PrefixWildcardQueryNode"
            }
            Kind::Regexp { .. } => {
                "org.apache.lucene.queryparser.flexible.standard.nodes.RegexpQueryNode"
            }
            Kind::TermRange { .. } => {
                "org.apache.lucene.queryparser.flexible.standard.nodes.TermRangeQueryNode"
            }
            Kind::MatchAll => {
                "org.apache.lucene.queryparser.flexible.core.nodes.MatchAllDocsQueryNode"
            }
            Kind::MatchNoDocs => {
                "org.apache.lucene.queryparser.flexible.core.nodes.MatchNoDocsQueryNode"
            }
            Kind::Deleted => "org.apache.lucene.queryparser.flexible.core.nodes.DeletedQueryNode",
            Kind::NoTokenFound => {
                "org.apache.lucene.queryparser.flexible.core.nodes.NoTokenFoundQueryNode"
            }
            Kind::Interval { .. } => {
                "org.apache.lucene.queryparser.flexible.standard.nodes.IntervalQueryNode"
            }
        }
    }

    /// `toQueryString(new EscapeQuerySyntaxImpl())` of the nodes a message
    /// prints here: a wildcard (`LEADING_WILDCARD_NOT_ALLOWED`), its text as
    /// is behind its field unless that is the default; and the deleted nodes
    /// no builder takes (`LUCENE_QUERY_CONVERSION_ERROR`).
    pub(crate) fn to_query_string(&self) -> String {
        let with_field = |field: &Option<String>, text: String| match field {
            // `isDefaultField`: no field, or `_plain`
            Some(f) if f != "_plain" => format!("{f}:{text}"),
            _ => text,
        };
        match &self.kind {
            Kind::Wildcard(d) | Kind::PrefixWildcard(d) => with_field(&d.field, d.text.as_string()),
            Kind::Deleted => "[DELETEDCHILD]".to_string(),
            Kind::NoTokenFound => "[NTF]".to_string(),
            _ => String::new(),
        }
    }
}

/// `org.apache.lucene.queryparser.flexible.standard.nodes.intervalfn`: an
/// interval function and its arguments.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum IntervalFn {
    AnalyzedText(String),
    AtLeast(i32, Vec<IntervalFn>),
    MaxWidth(i32, Box<IntervalFn>),
    MaxGaps(i32, Box<IntervalFn>),
    Ordered(Vec<IntervalFn>),
    Unordered(Vec<IntervalFn>),
    UnorderedNoOverlaps(Box<IntervalFn>, Box<IntervalFn>),
    Or(Vec<IntervalFn>),
    Phrase(Vec<IntervalFn>),
    After(Box<IntervalFn>, Box<IntervalFn>),
    Before(Box<IntervalFn>, Box<IntervalFn>),
    Containing(Box<IntervalFn>, Box<IntervalFn>),
    NotContaining(Box<IntervalFn>, Box<IntervalFn>),
    ContainedBy(Box<IntervalFn>, Box<IntervalFn>),
    NotContainedBy(Box<IntervalFn>, Box<IntervalFn>),
    Within(Box<IntervalFn>, i32, Box<IntervalFn>),
    NotWithin(Box<IntervalFn>, i32, Box<IntervalFn>),
    Overlapping(Box<IntervalFn>, Box<IntervalFn>),
    NonOverlapping(Box<IntervalFn>, Box<IntervalFn>),
    Extend(Box<IntervalFn>, i32, i32),
    /// `Wildcard(wildcard, maxExpansions)`: 0 is the default.
    Wildcard(String, i32),
    /// `FuzzyTerm(term, maxEdits, maxExpansions)`, defaults applied.
    FuzzyTerm(String, i32, i32),
}
