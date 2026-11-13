//! The occur-tagged boolean clause arrays the query-tree decoder
//! (the query-blob decoder) builds, decoded into a [`BooleanQuery`], plus the clause
//! limits and error mapping shared by the JVM entry points.

use lucene_search::{BooleanQuery, BoostQuery, Clause, ConstantScoreQuery, PhraseQuery, TermQuery};

use crate::alloc::try_with_capacity;
use crate::status::{set_last_error, Status};

/// Maps a search error to [`Status::Search`], recording its message.
pub fn map_search_error(e: lucene_search::Error) -> Status {
    if let lucene_search::Error::Memory(m) = &e {
        return crate::memory::exceeded(m);
    }
    if let lucene_search::Error::TooManyBuckets(count, _) = e {
        set_last_error(count.to_string());
        return Status::TooManyBuckets;
    }
    set_last_error(format!("search failed: {e}"));
    Status::Search
}

/// Lucene's default `IndexSearcher.getMaxClauseCount()`. `BooleanQuery` here
/// has no builder to enforce it, so the boundary refuses larger clause arrays.
pub const MAX_CLAUSE_COUNT: usize = 1024;

/// Rejects a query with more than [`MAX_CLAUSE_COUNT`] clauses, counted over
/// the whole tree. Java counts per nesting level; one total is the bound a
/// denial-of-service guard needs.
pub fn check_clause_count(clause_count: usize) -> Result<(), Status> {
    if clause_count > MAX_CLAUSE_COUNT {
        set_last_error(format!(
            "maxClauseCount is set to {MAX_CLAUSE_COUNT}, but this query has {clause_count} clauses"
        ));
        return Err(Status::InvalidArgument);
    }
    Ok(())
}

/// `BooleanClause.Occur` ordinals, in Java's declaration order, so the plugin
/// sends `occur.ordinal()` as is.
pub const OCCUR_MUST: u8 = 0;
/// `BooleanClause.Occur.FILTER` -- "like `MUST` except that these clauses do
/// not participate in scoring".
pub const OCCUR_FILTER: u8 = 1;
/// `BooleanClause.Occur.SHOULD`.
pub const OCCUR_SHOULD: u8 = 2;
/// `BooleanClause.Occur.MUST_NOT`.
pub const OCCUR_MUST_NOT: u8 = 3;

/// A leaf `TermQuery` clause: `clause_fields[i]`/`clause_terms[i]` are its
/// field and its raw, already-analyzed term bytes.
pub const CLAUSE_KIND_TERM: u8 = 0;
/// A nested `BooleanQuery` clause: carries no field/term of its own (both may
/// be null), and every clause naming index `i` in `clause_parents` is one of
/// its children. `clause_params[i]` is its own `minimumNumberShouldMatch`.
pub const CLAUSE_KIND_BOOLEAN: u8 = 1;
/// A `ConstantScoreQuery` wrapping exactly one child clause, whose occur must
/// be `MUST`. `clause_params[i]` is the constant score as `f32::to_bits` --
/// finite and non-negative. OpenSearch builds one for every `term` query on a
/// `keyword` field.
pub const CLAUSE_KIND_CONSTANT_SCORE: u8 = 2;
/// A `BoostQuery` wrapping exactly one child clause, whose occur must be
/// `MUST`. `clause_params[i]` is the boost as `f32::to_bits` -- finite and
/// non-negative (Lucene's `BoostQuery` rejects a negative boost).
pub const CLAUSE_KIND_BOOST: u8 = 3;

/// True for the clause kinds that contain other clauses.
fn is_container(kind: u8) -> bool {
    kind != CLAUSE_KIND_TERM
}

/// True for the single-child wrapper kinds.
fn is_wrapper(kind: u8) -> bool {
    kind == CLAUSE_KIND_CONSTANT_SCORE || kind == CLAUSE_KIND_BOOST
}

/// A container clause under construction in [`read_boolean_clauses`]'s
/// reverse pass.
enum Node {
    Boolean(BooleanQuery),
    Wrapper {
        kind: u8,
        value: f32,
        child: Option<Clause>,
    },
}

/// The deepest `clause_parents` chain accepted. Not a Lucene limit: nested
/// clauses are evaluated and dropped recursively, so depth is stack depth, and
/// a stack overflow aborts past `catch_unwind`.
pub const MAX_CLAUSE_DEPTH: usize = 32;

/// One decoded clause, before it is attached to its parent.
struct PendingClause<'a> {
    occur: u8,
    kind: u8,
    field: &'a str,
    term: &'a [u8],
    parent: i32,
}

/// Appends `clause` to `parent`'s bucket for `occur`.
fn push_clause(parent: &mut BooleanQuery, occur: u8, clause: Clause) {
    match occur {
        OCCUR_MUST => parent.must.push(clause),
        OCCUR_FILTER => parent.filter.push(clause),
        OCCUR_SHOULD => parent.should.push(clause),
        // `read_boolean_query` rejects every other tag before this runs.
        _ => parent.must_not.push(clause),
    }
}

/// Decodes a `BooleanQuery` from occur-tagged parallel clause arrays (what
/// the query-blob decoder's boolean node expands into).
///
/// # Wire format
///
/// One flat array of `clause_count` clauses, each described by the same
/// index `i` across eight parallel arrays:
///
/// | array | type | meaning |
/// |---|---|---|
/// | `clause_occurs` | `u8` | [`OCCUR_MUST`]/[`OCCUR_FILTER`]/[`OCCUR_SHOULD`]/[`OCCUR_MUST_NOT`], i.e. Java's `Occur.ordinal()` |
/// | `clause_kinds` | `u8` | [`CLAUSE_KIND_TERM`], [`CLAUSE_KIND_BOOLEAN`], [`CLAUSE_KIND_CONSTANT_SCORE`] or [`CLAUSE_KIND_BOOST`] |
/// | `clause_fields`/`clause_field_lens` | `(*const c_char, usize)` | the field name, for a `TERM` clause |
/// | `clause_terms`/`clause_term_lens` | `(*const u8, usize)` | the raw term bytes, for a `TERM` clause |
/// | `clause_parents` | `i32` | index of the enclosing `BOOLEAN`, `CONSTANT_SCORE` or `BOOST` clause, or `-1` for a top-level clause. Must be `< i`. May be null, meaning "every clause is top-level" |
/// | `clause_params` | `i32` | a `BOOLEAN` clause's own `minimumNumberShouldMatch`; a `CONSTANT_SCORE`/`BOOST` clause's score/boost as `f32` bits; must be `0` for a `TERM` clause. May be null, meaning "all zero" |
///
/// A `CONSTANT_SCORE` or `BOOST` clause has exactly one child, with occur
/// `MUST` -- the wrapped query. Field and term bytes are ignored for every
/// container kind.
///
/// `minimum_should_match` is the *root* query's own
/// `minimumNumberShouldMatch` (Java's
/// `BooleanQuery.Builder.setMinimumNumberShouldMatch`).
///
/// A new `(field, term)`-shaped clause kind or a new `Occur` is a new tag
/// value; a clause needing more attributes (a phrase's term list and slop)
/// does not fit this format.
///
/// `term_at(i)` is the field and term bytes of clause `i`, read only for a
/// `TERM` clause: the boundary that owns the caller's raw arrays supplies
/// them.
///
/// `occurs`, `kinds`, `parents` and `params` have one element per clause;
/// `parents` and `params` are `None` for the flat, all-top-level,
/// all-default query that is by far the common case.
pub fn read_boolean_clauses<'a>(
    occurs: &[u8],
    kinds: &[u8],
    parents: Option<&[i32]>,
    params: Option<&[i32]>,
    term_at: &dyn Fn(usize) -> Result<(&'a str, &'a [u8]), Status>,
    minimum_should_match: i32,
) -> Result<BooleanQuery, Status> {
    if minimum_should_match < 0 {
        set_last_error(format!(
            "minimumNumberShouldMatch {minimum_should_match} is negative"
        ));
        return Err(Status::InvalidArgument);
    }
    let mut root = BooleanQuery::new().with_minimum_should_match(minimum_should_match as usize);
    let clause_count = occurs.len();
    if clause_count == 0 {
        return Ok(root);
    }
    debug_assert_eq!(kinds.len(), clause_count);

    let mut pending: Vec<PendingClause<'_>> = try_with_capacity(clause_count)?;
    let mut depth: Vec<usize> = try_with_capacity(clause_count)?;
    // Whether each wrapper clause has been given its one child yet.
    let mut wrapped: Vec<bool> = try_with_capacity(clause_count)?;
    wrapped.resize(clause_count, false);
    for i in 0..clause_count {
        let occur = occurs[i];
        if occur > OCCUR_MUST_NOT {
            set_last_error(format!(
                "clause {i}: unknown Occur tag {occur} (expected 0=MUST, 1=FILTER, 2=SHOULD, \
                 3=MUST_NOT)"
            ));
            return Err(Status::InvalidArgument);
        }
        let kind = kinds[i];
        if kind > CLAUSE_KIND_BOOST {
            set_last_error(format!(
                "clause {i}: unknown clause kind {kind} (expected 0=TERM, 1=BOOLEAN, \
                 2=CONSTANT_SCORE, 3=BOOST)"
            ));
            return Err(Status::InvalidArgument);
        }
        let parent = parents.map_or(-1, |p| p[i]);
        // A parent must already have been seen, which both rules out a cycle
        // (a clause can never be its own ancestor) and guarantees the
        // single reverse pass below sees every child before its parent.
        if parent < -1 || parent >= i as i32 {
            set_last_error(format!(
                "clause {i}: parent index {parent} must be -1 (top level) or an earlier clause's \
                 index"
            ));
            return Err(Status::InvalidArgument);
        }
        if parent >= 0 && !is_container(kinds[parent as usize]) {
            set_last_error(format!(
                "clause {i}: parent clause {parent} is not a BOOLEAN clause, so it cannot contain \
                 other clauses"
            ));
            return Err(Status::InvalidArgument);
        }
        if parent >= 0 && is_wrapper(kinds[parent as usize]) {
            if occur != OCCUR_MUST {
                set_last_error(format!(
                    "clause {i}: the child of wrapper clause {parent} must have occur MUST, got \
                     {occur}"
                ));
                return Err(Status::InvalidArgument);
            }
            if wrapped[parent as usize] {
                set_last_error(format!(
                    "clause {i}: wrapper clause {parent} already has a child"
                ));
                return Err(Status::InvalidArgument);
            }
            wrapped[parent as usize] = true;
        }
        let my_depth = if parent < 0 {
            0
        } else {
            depth[parent as usize] + 1
        };
        if my_depth >= MAX_CLAUSE_DEPTH {
            set_last_error(format!(
                "clause {i}: nesting depth {} exceeds the maximum of {MAX_CLAUSE_DEPTH}",
                my_depth + 1
            ));
            return Err(Status::InvalidArgument);
        }
        depth.push(my_depth);

        let param = params.map_or(0, |p| p[i]);
        let (field, term) = match kind {
            CLAUSE_KIND_TERM => {
                if param != 0 {
                    set_last_error(format!(
                        "clause {i}: clause_params must be 0 for a TERM clause (it is a BOOLEAN \
                         clause's minimumNumberShouldMatch), got {param}"
                    ));
                    return Err(Status::InvalidArgument);
                }
                term_at(i)?
            }
            CLAUSE_KIND_BOOLEAN => {
                if param < 0 {
                    set_last_error(format!(
                        "clause {i}: minimumNumberShouldMatch {param} is negative"
                    ));
                    return Err(Status::InvalidArgument);
                }
                ("", &[][..])
            }
            _ => {
                let value = f32::from_bits(param as u32);
                if !value.is_finite() || value < 0.0 {
                    set_last_error(format!(
                        "clause {i}: a CONSTANT_SCORE/BOOST value must be finite and \
                         non-negative, got {value}"
                    ));
                    return Err(Status::InvalidArgument);
                }
                ("", &[][..])
            }
        };
        pending.push(PendingClause {
            occur,
            kind,
            field,
            term,
            parent,
        });
    }

    if let Some(i) = (0..clause_count).find(|&i| is_wrapper(kinds[i]) && !wrapped[i]) {
        set_last_error(format!("clause {i}: wrapper clause has no child"));
        return Err(Status::InvalidArgument);
    }

    // Every container clause, pre-created -- a `BOOLEAN` with its own
    // `minimumNumberShouldMatch`, a wrapper with its value -- so the reverse
    // pass below can push each child straight into the parent it names.
    let mut nodes: Vec<Option<Node>> = try_with_capacity(clause_count)?;
    for (i, c) in pending.iter().enumerate() {
        let param = params.map_or(0, |p| p[i]);
        nodes.push(match c.kind {
            CLAUSE_KIND_BOOLEAN => Some(Node::Boolean(
                BooleanQuery::new().with_minimum_should_match(param as usize),
            )),
            CLAUSE_KIND_TERM => None,
            kind => Some(Node::Wrapper {
                kind,
                value: f32::from_bits(param as u32),
                child: None,
            }),
        });
    }

    // Reverse order: a parent's index is always smaller than its children's,
    // so by the time index `p` is reached every clause that named it has
    // already been pushed into it. No recursion -- see `MAX_CLAUSE_DEPTH` for
    // why recursion over caller-controlled depth is not acceptable here.
    for i in (0..clause_count).rev() {
        let c = &pending[i];
        let clause = match nodes[i].take() {
            None => Clause::Term(TermQuery::new(c.field, c.term.to_vec())),
            Some(Node::Boolean(mut nested)) => {
                // Restore caller order: the reverse walk appended children
                // back-to-front.
                nested.must.reverse();
                nested.filter.reverse();
                nested.should.reverse();
                nested.must_not.reverse();
                Clause::Boolean(Box::new(nested))
            }
            Some(Node::Wrapper { kind, value, child }) => {
                let child = child.expect("every wrapper was checked to have a child");
                if kind == CLAUSE_KIND_CONSTANT_SCORE {
                    Clause::ConstantScore(Box::new(ConstantScoreQuery::new(child, value)))
                } else {
                    Clause::Boost(Box::new(BoostQuery::new(child, value)))
                }
            }
        };
        match c.parent {
            -1 => push_clause(&mut root, c.occur, clause),
            p => match nodes[p as usize]
                .as_mut()
                .expect("a validated parent is a container clause with a node")
            {
                Node::Boolean(parent) => push_clause(parent, c.occur, clause),
                Node::Wrapper { child, .. } => *child = Some(clause),
            },
        }
    }
    root.must.reverse();
    root.filter.reverse();
    root.should.reverse();
    root.must_not.reverse();
    Ok(root)
}

/// Every distinct field name a decoded `BooleanQuery`'s *scoring* clauses
/// mention, at any nesting depth -- the set a scored search needs norms for.
/// A `FILTER` or `MUST_NOT` clause and a constant-score query's inner query
/// never score, so their fields are left out: building norms nobody reads
/// was most of a small query's setup.
pub fn clause_field_names(query: &BooleanQuery) -> Vec<&str> {
    // An explicit stack: the nesting depth is caller-controlled.
    let mut out: Vec<&str> = Vec::new();
    let mut stack: Vec<&Clause> = Vec::new();
    fn push_clauses<'q>(q: &'q BooleanQuery, stack: &mut Vec<&'q Clause>) {
        stack.extend(q.must.iter().chain(q.should.iter()));
    }
    push_clauses(query, &mut stack);
    while let Some(clause) = stack.pop() {
        match clause {
            Clause::Boolean(nested) => push_clauses(nested, &mut stack),
            Clause::Boost(b) => stack.push(&b.inner),
            // Dismax disjuncts score with their own fields' norms.
            Clause::DisjunctionMax(d) => stack.extend(d.disjuncts.iter()),
            Clause::Term(TermQuery { field, .. })
            | Clause::Phrase(PhraseQuery { field, .. })
            | Clause::Fuzzy(lucene_search::query::FuzzyQuery { field, .. })
                if !out.contains(&field.as_str()) =>
            {
                out.push(field.as_str())
            }
            // A block join's child and a function score's sub-query score
            // with their own fields' norms,
            // a span query with its (masked) field's, BM25F with each field's, a
            // covering query with its sub-queries'.
            Clause::Extended(e) => {
                use lucene_search::extended_query::ExtendedQuery;
                match e.as_ref() {
                    ExtendedQuery::ToParentBlockJoin(j) => stack.push(&j.child),
                    ExtendedQuery::FunctionScore(f) => stack.push(&f.in_query),
                    ExtendedQuery::Covering(c) => stack.extend(c.queries.iter()),
                    ExtendedQuery::Span(s) => {
                        if let Some(field) = s.field() {
                            if !out.contains(&field) {
                                out.push(field);
                            }
                        }
                    }
                    ExtendedQuery::Synonym(s) => {
                        if !out.contains(&s.field.as_str()) {
                            out.push(s.field.as_str());
                        }
                    }
                    ExtendedQuery::CombinedField(c) => {
                        for (field, _) in &c.fields {
                            if !out.contains(&field.as_str()) {
                                out.push(field.as_str());
                            }
                        }
                    }
                    _ => {}
                }
            }
            // Everything else scores without norms (a constant) or matches
            // nothing.
            _ => {}
        }
    }
    out
}
