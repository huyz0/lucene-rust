//! `FieldQuery`: a query flattened to the terms and phrases the term vector
//! is searched for, arranged as a tree per field (`QueryPhraseMap`).

use std::collections::{BTreeSet, HashMap, HashSet};

use super::super::extractor::expand_clause;
use super::super::{Error, Result};
use crate::directory_reader::DirectoryReader;
use crate::extended_query::ExtendedQuery;
use crate::query::Clause;

use super::field_term_stack::TermInfo;

/// `MAX_MTQ_TERMS`: the most terms one multi-term query is expanded to.
const MAX_MTQ_TERMS: usize = 1024;

/// A term or a phrase, as `flatten` leaves it.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum FlatKind {
    Term {
        field: String,
        text: String,
    },
    Phrase {
        field: String,
        terms: Vec<String>,
        positions: Vec<i32>,
        slop: i32,
    },
}

/// A flat query: [`FlatKind`], wrapped in one `BoostQuery` when `boost` is
/// set. Equal as `Query.equals` is (a `BoostQuery(q, 1)` is not `q`).
#[derive(Debug, Clone)]
struct Flat {
    kind: FlatKind,
    boost: Option<f32>,
}

impl PartialEq for Flat {
    fn eq(&self, other: &Self) -> bool {
        self.kind == other.kind && self.boost.map(f32::to_bits) == other.boost.map(f32::to_bits)
    }
}

impl Eq for Flat {}

impl std::hash::Hash for Flat {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.kind.hash(state);
        self.boost.map(f32::to_bits).hash(state);
    }
}

/// `LinkedHashSet<Query>`: the flat queries in the order they were first
/// added, each once.
#[derive(Debug, Default)]
struct FlatSet {
    list: Vec<Flat>,
    seen: HashSet<Flat>,
}

impl FlatSet {
    /// `add(query)`: `false` when an equal query is already in.
    fn add(&mut self, flat: Flat) -> bool {
        if self.seen.contains(&flat) {
            return false;
        }
        self.seen.insert(flat.clone());
        self.list.push(flat);
        true
    }
}

impl Flat {
    fn boost_value(&self) -> f32 {
        self.boost.unwrap_or(1.0)
    }
}

fn text_of(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

fn wrap(kind: FlatKind, boost: f32) -> Flat {
    Flat {
        kind,
        boost: (boost != 1.0).then_some(boost),
    }
}

/// `FieldQuery`.
#[derive(Debug)]
pub struct FieldQuery {
    field_match: bool,
    root_maps: HashMap<Option<String>, QueryPhraseMap>,
    term_set_map: HashMap<Option<String>, HashSet<String>>,
    /// `termOrPhraseNumber`: the next number a terminal gets.
    term_or_phrase_number: i32,
}

impl FieldQuery {
    /// `FieldQuery(query, reader, phraseHighlight, fieldMatch)`. Without a
    /// `reader` the multi-term queries are dropped, as Java drops them
    /// without a searcher.
    ///
    /// # Errors
    /// A multi-term query whose terms cannot be read, or a regular expression
    /// that does not parse.
    pub fn new(
        query: &Clause,
        reader: Option<&DirectoryReader>,
        phrase_highlight: bool,
        field_match: bool,
    ) -> Result<Self> {
        let mut flat = FlatSet::default();
        flatten(query, reader, &mut flat, 1.0)?;
        let flat = flat.list;
        let mut this = Self {
            field_match,
            root_maps: HashMap::new(),
            term_set_map: HashMap::new(),
            term_or_phrase_number: 0,
        };
        this.save_terms(&flat);
        let expanded = this.expand(flat);
        for q in &expanded {
            let key = this.key(&q.kind);
            let mut root = this.root_maps.remove(&key).unwrap_or_default();
            root.add(q, &mut this.term_or_phrase_number);
            if !phrase_highlight {
                // `pq.getTerms().length > 1`: a flat phrase always has more
                // than one term (`flatten` makes a single one a term)
                if let FlatKind::Phrase { terms, .. } = &q.kind {
                    for t in terms {
                        root.add_term(t, q.boost_value(), &mut this.term_or_phrase_number);
                    }
                }
            }
            this.root_maps.insert(key, root);
        }
        Ok(this)
    }

    /// `getKey(query)`.
    fn key(&self, kind: &FlatKind) -> Option<String> {
        if !self.field_match {
            return None;
        }
        Some(match kind {
            FlatKind::Term { field, .. } | FlatKind::Phrase { field, .. } => field.clone(),
        })
    }

    /// `saveTerms`.
    fn save_terms(&mut self, flat: &[Flat]) {
        for q in flat {
            let key = self.key(&q.kind);
            let set = self.term_set_map.entry(key).or_default();
            match &q.kind {
                FlatKind::Term { text, .. } => {
                    set.insert(text.clone());
                }
                FlatKind::Phrase { terms, .. } => {
                    set.extend(terms.iter().cloned());
                }
            }
        }
    }

    /// `expand`: each phrase a later one overlaps with is joined by the
    /// phrase the two make together.
    fn expand(&self, flat: Vec<Flat>) -> Vec<Flat> {
        let mut out = FlatSet::default();
        for (i, query) in flat.iter().enumerate() {
            // the iterator removes what it returns: the rest follow it
            out.add(query.clone());
            let FlatKind::Phrase { .. } = &query.kind else {
                continue;
            };
            for qj in &flat[i + 1..] {
                if let FlatKind::Phrase { .. } = &qj.kind {
                    self.check_overlap(&mut out, query, qj);
                }
            }
        }
        out.list
    }

    fn check_overlap(&self, out: &mut FlatSet, a: &Flat, b: &Flat) {
        let (
            FlatKind::Phrase {
                field: af,
                terms: ats,
                slop: aslop,
                ..
            },
            FlatKind::Phrase {
                field: bf,
                terms: bts,
                slop: bslop,
                ..
            },
        ) = (&a.kind, &b.kind)
        else {
            return;
        };
        if aslop != bslop {
            return;
        }
        if self.field_match && af != bf {
            return;
        }
        Self::overlap(out, af, ats, bts, *aslop, a.boost_value());
        Self::overlap(out, bf, bts, ats, *bslop, b.boost_value());
    }

    fn overlap(
        out: &mut FlatSet,
        field: &str,
        src: &[String],
        dest: &[String],
        slop: i32,
        boost: f32,
    ) {
        for i in 1..src.len() {
            let overlap = (i..src.len()).all(|j| j - i >= dest.len() || src[j] == dest[j - i]);
            if overlap && src.len() - i < dest.len() {
                let mut terms: Vec<String> = src.to_vec();
                terms.extend(dest[src.len() - i..].iter().cloned());
                let n = i32::try_from(terms.len()).unwrap_or(i32::MAX);
                let pq = wrap(
                    FlatKind::Phrase {
                        field: field.to_string(),
                        positions: (0..n).collect(),
                        terms,
                        slop,
                    },
                    1.0,
                );
                // `new BoostQuery(pq, 1f)` when the source was boosted
                let pq = if boost != 1.0 {
                    Flat {
                        kind: pq.kind,
                        boost: Some(1.0),
                    }
                } else {
                    pq
                };
                out.add(pq);
            }
        }
    }

    /// `getTermSet(fieldName)`: the terms the field's term vector is read
    /// for; `None` when the query has none for it.
    pub fn term_set(&self, field: &str) -> Option<&HashSet<String>> {
        self.term_set_map.get(&self.field_key(field))
    }

    fn field_key(&self, field: &str) -> Option<String> {
        self.field_match.then(|| field.to_string())
    }

    fn root_map(&self, field: &str) -> Option<&QueryPhraseMap> {
        self.root_maps.get(&self.field_key(field))
    }

    /// `getFieldTermMap(fieldName, term)`.
    pub fn field_term_map(&self, field: &str, term: &str) -> Option<&QueryPhraseMap> {
        self.root_map(field)?.sub_map.get(term)
    }

    /// `searchPhrase(fieldName, phraseCandidate)`.
    pub fn search_phrase(&self, field: &str, candidate: &[&TermInfo]) -> Option<&QueryPhraseMap> {
        self.root_map(field)?.search_phrase(candidate)
    }
}

/// `flatten`: terms and phrases, each with the boost it is under.
fn flatten(
    query: &Clause,
    reader: Option<&DirectoryReader>,
    out: &mut FlatSet,
    boost: f32,
) -> Result<()> {
    match query {
        Clause::Boost(b) => flatten(&b.inner, reader, out, boost * b.boost)?,
        Clause::Boolean(b) => {
            // `BooleanQuery` keeps its clauses in the order they were added;
            // this one keeps them by kind, so the order is theirs.
            for c in b.must.iter().chain(&b.filter).chain(&b.should) {
                flatten(c, reader, out, boost)?;
            }
        }
        Clause::DisjunctionMax(d) => {
            for c in &d.disjuncts {
                flatten(c, reader, out, boost)?;
            }
        }
        Clause::Term(t) => {
            let flat = wrap(
                FlatKind::Term {
                    field: t.field.clone(),
                    text: text_of(&t.term),
                },
                boost,
            );
            out.add(flat);
        }
        Clause::Extended(e) => {
            if let ExtendedQuery::Synonym(s) = &**e {
                for (term, _) in &s.terms {
                    let t =
                        Clause::Term(crate::query::TermQuery::new(s.field.clone(), term.clone()));
                    flatten(&t, reader, out, boost)?;
                }
            }
        }
        Clause::Phrase(p) => {
            let kind = if p.terms.len() == 1 {
                FlatKind::Term {
                    field: p.field.clone(),
                    text: text_of(&p.terms[0]),
                }
            } else {
                let n = i32::try_from(p.terms.len()).unwrap_or(i32::MAX);
                FlatKind::Phrase {
                    field: p.field.clone(),
                    terms: p.terms.iter().map(|t| text_of(t)).collect(),
                    positions: if p.positions.is_empty() {
                        (0..n).collect()
                    } else {
                        p.positions.clone()
                    },
                    slop: i32::try_from(p.slop).unwrap_or(i32::MAX),
                }
            };
            out.add(wrap(kind, boost));
        }
        Clause::ConstantScore(c) => flatten(&c.inner, reader, out, boost)?,
        Clause::MultiPhrase(m) if m.term_arrays.len() == 1 && reader.is_some() => {
            // `MultiPhraseQuery.rewrite` (which needs a reader): a single
            // position is a disjunction
            for t in &m.term_arrays[0] {
                let t = Clause::Term(crate::query::TermQuery::new(m.field.clone(), t.clone()));
                flatten(&t, reader, out, boost)?;
            }
        }
        _ => {
            let Some(reader) = reader else {
                return Ok(());
            };
            if let Some((field, terms)) = expand_multi_term(reader, query)? {
                for (term, term_boost) in terms {
                    let t = Clause::Term(crate::query::TermQuery::new(field.clone(), term));
                    flatten(&t, Some(reader), out, boost * term_boost)?;
                }
            }
        }
    }
    Ok(())
}

/// The field of a multi-term query and its expanded terms with their boosts.
type MtqTerms = (String, Vec<(Vec<u8>, f32)>);

/// `TopTermsScoringBooleanQueryRewrite(1024)` over a multi-term query: its
/// field and the (at most 1024) index terms it matches with their boosts, in
/// term order. `None` for any other query.
fn expand_multi_term(reader: &DirectoryReader, query: &Clause) -> Result<Option<MtqTerms>> {
    let field = match query {
        Clause::Prefix(q) => &q.field,
        Clause::Wildcard(q) => &q.field,
        Clause::Regexp(q) => &q.field,
        Clause::Fuzzy(q) => &q.field,
        Clause::TermInSet(q) => &q.field,
        _ => return Ok(None),
    };
    let mut all: BTreeSet<Vec<u8>> = BTreeSet::new();
    for segment in reader.segment_readers() {
        segment
            .with_open_segment(segment.doc_base, |open| {
                if let Some(ft) = open.fields.field(field) {
                    let mut terms = ft.iter();
                    while let Some((t, _)) = terms.next() {
                        all.insert(t.to_vec());
                    }
                }
                Ok(())
            })
            .map_err(Error::Search)?;
    }
    let terms: Vec<Vec<u8>> = all.into_iter().collect();
    let mut expanded = match query {
        Clause::TermInSet(q) => {
            let wanted: BTreeSet<&Vec<u8>> = q.terms.iter().collect();
            terms
                .into_iter()
                .filter(|t| wanted.contains(t))
                .map(|t| (t, 1.0))
                .collect()
        }
        _ => expand_clause(query, &terms)?.unwrap_or_default(),
    };
    if expanded.len() > MAX_MTQ_TERMS {
        // the queue keeps the highest boosts, the smaller terms among equals
        expanded.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        expanded.truncate(MAX_MTQ_TERMS);
        expanded.sort_by(|a, b| a.0.cmp(&b.0));
    }
    Ok(Some((field.clone(), expanded)))
}

/// `QueryPhraseMap`: a trie over the terms of the phrases of one field.
#[derive(Debug, Default)]
pub struct QueryPhraseMap {
    terminal: bool,
    slop: i32,
    boost: f32,
    term_or_phrase_number: i32,
    sub_map: HashMap<String, QueryPhraseMap>,
}

impl QueryPhraseMap {
    fn mark_terminal(&mut self, slop: i32, boost: f32, number: &mut i32) {
        self.terminal = true;
        self.slop = slop;
        self.boost = boost;
        self.term_or_phrase_number = *number;
        *number += 1;
    }

    fn add_term(&mut self, term: &str, boost: f32, number: &mut i32) {
        self.sub_map
            .entry(term.to_string())
            .or_default()
            .mark_terminal(0, boost, number);
    }

    fn add(&mut self, query: &Flat, number: &mut i32) {
        // the boost of the innermost `BoostQuery`, 1 for none
        let boost = query.boost_value();
        match &query.kind {
            FlatKind::Term { text, .. } => self.add_term(text, boost, number),
            FlatKind::Phrase { terms, slop, .. } => {
                let mut map = self;
                for t in terms {
                    map = map.sub_map.entry(t.clone()).or_default();
                }
                map.mark_terminal(*slop, boost, number);
            }
        }
    }

    /// `getTermMap(term)`.
    pub fn term_map(&self, term: &str) -> Option<&QueryPhraseMap> {
        self.sub_map.get(term)
    }

    /// `isTerminal()`.
    pub fn is_terminal(&self) -> bool {
        self.terminal
    }

    /// `getSlop()`.
    pub fn slop(&self) -> i32 {
        self.slop
    }

    /// `getBoost()`.
    pub fn boost(&self) -> f32 {
        self.boost
    }

    /// `getTermOrPhraseNumber()`.
    pub fn term_or_phrase_number(&self) -> i32 {
        self.term_or_phrase_number
    }

    /// `searchPhrase(phraseCandidate)`.
    pub fn search_phrase(&self, candidate: &[&TermInfo]) -> Option<&QueryPhraseMap> {
        let mut curr = self;
        for ti in candidate {
            curr = curr.sub_map.get(ti.text())?;
        }
        curr.is_valid_term_or_phrase(candidate).then_some(curr)
    }

    /// `isValidTermOrPhrase(phraseCandidate)`.
    pub fn is_valid_term_or_phrase(&self, candidate: &[&TermInfo]) -> bool {
        if !self.terminal {
            return false;
        }
        if candidate.len() == 1 {
            return true;
        }
        let mut pos = candidate[0].position();
        for ti in &candidate[1..] {
            let next = ti.position();
            if (i64::from(next) - i64::from(pos) - 1).abs() > i64::from(self.slop) {
                return false;
            }
            pos = next;
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn term(text: &str, boost: Option<f32>) -> Flat {
        Flat {
            kind: FlatKind::Term {
                field: "f".to_string(),
                text: text.to_string(),
            },
            boost,
        }
    }

    fn phrase(terms: &[&str]) -> Flat {
        Flat {
            kind: FlatKind::Phrase {
                field: "f".to_string(),
                terms: terms.iter().map(|t| (*t).to_string()).collect(),
                positions: (0..terms.len() as i32).collect(),
                slop: 0,
            },
            boost: None,
        }
    }

    fn field_query() -> FieldQuery {
        FieldQuery {
            field_match: true,
            root_maps: HashMap::new(),
            term_set_map: HashMap::new(),
            term_or_phrase_number: 0,
        }
    }

    /// `LinkedHashSet.add`: first insertion order, each query once, and a
    /// `BoostQuery(q, 1)` is not `q`.
    #[test]
    fn flat_set_keeps_first_insertion_order_and_boost_identity() {
        let mut set = FlatSet::default();
        assert!(set.add(term("a", None)));
        assert!(set.add(term("b", None)));
        assert!(!set.add(term("a", None)));
        assert!(set.add(term("a", Some(1.0))));
        assert!(!set.add(term("a", Some(1.0))));
        assert_eq!(
            set.list,
            vec![term("a", None), term("b", None), term("a", Some(1.0))]
        );
    }

    /// `FieldQuery.expand`'s javadoc: `a, "b c", "c d"` gains `"b c d"`, added
    /// as `"b c"` meets `"c d"` and so before it; `"a a a a"` with `"a a a"`
    /// gains five and six `a`s, before the second.
    #[test]
    fn expand_appends_overlapping_phrases_in_javas_order() {
        let fq = field_query();
        assert_eq!(
            fq.expand(vec![
                term("a", None),
                phrase(&["b", "c"]),
                phrase(&["c", "d"])
            ]),
            vec![
                term("a", None),
                phrase(&["b", "c"]),
                phrase(&["b", "c", "d"]),
                phrase(&["c", "d"]),
            ]
        );
        let got = fq.expand(vec![
            phrase(&["a", "a", "a", "a"]),
            phrase(&["a", "a", "a"]),
        ]);
        // both directions make the same two phrases; each is kept once
        assert_eq!(
            got,
            vec![
                phrase(&["a"; 4]),
                phrase(&["a"; 5]),
                phrase(&["a"; 6]),
                phrase(&["a"; 3]),
            ]
        );
        // an expansion that recreates a query already in is not added twice
        let again = fq.expand(vec![
            phrase(&["b", "c"]),
            phrase(&["c", "d"]),
            phrase(&["b", "c", "d"]),
        ]);
        assert_eq!(again.len(), 3);
    }
}
